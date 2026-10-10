//! Headless video pipeline: pipe raw RGB frames into `ffmpeg`, parallelise
//! per-clip encodes, concatenate with the concat demuxer.
//!
//! The plan is a list of `(generation, seconds)` tuples. For each entry we
//! render the best individual of that generation (looked up in the GA history)
//! for `seconds` at 60 fps, write the frames to a temporary `.rgb` buffer,
//! spawn an `ffmpeg` subprocess that reads them as `rawvideo rgb24` from
//! stdin, and produce a per-clip `.mp4`. Finally we concatenate all clips
//! (re-encoded for safety — `concat -c copy` can drop frames when input
//! timestamps are inconsistent) into the final mp4 at the requested path.
//!
//! The whole pipeline runs entirely offscreen: no X / Wayland / window.
//!
//! ## Why re-encode at the concat step?
//!
//! `concat -c copy` works when every input clip has identical codec /
//! pixel-format / GOP structure and frame-accurate PTS. In practice,
//! producing that guarantee from independent `ffmpeg` invocations is
//! brittle. A short re-encode (`-crf 18`) is cheap (a few seconds for our
//! clip sizes) and removes the class of failure entirely.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use rayon::prelude::*;

use crate::evolution::{GenerationRecord, History};
use crate::physics::{DT, SUBSTEPS_PER_FRAME, World};
use crate::render::{Camera, FRAME_H, FRAME_W, FrameBuffer, draw_robot_positions, draw_terrain};
use crate::tasks::terrain_for;
use crate::text::draw_overlay;

/// A single clip spec: render this generation for `seconds` seconds.
#[derive(Debug, Clone)]
pub struct ClipSpec {
    pub generation: usize,
    pub seconds: f32,
    /// Overlay this clip as "Final (Generation N)" instead of "Generation N".
    pub is_final: bool,
}

/// Result of running the video pipeline.
#[derive(Debug, Clone)]
pub struct VideoOutput {
    pub output_path: PathBuf,
    pub clips: Vec<PathBuf>,
}

/// Errors that can arise while running the pipeline.
#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("ffmpeg not found on PATH — install it to render video")]
    FfmpegMissing,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ffmpeg failed (status {status}): {stderr}")]
    FfmpegFailed { status: i32, stderr: String },
    #[error("no clip specs provided")]
    EmptyPlan,
    #[error("generation {0} not present in evolution history")]
    MissingGeneration(usize),
    #[error("invalid clip seconds ({0}); must be > 0")]
    InvalidSeconds(f32),
}

/// Verify `ffmpeg` is callable.
pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Encode a single raw-RGB byte stream to an h264 `.mp4` file by piping into
/// `ffmpeg` via stdin.
fn encode_clip(frames: &[u8], path: &Path) -> Result<(), VideoError> {
    use std::process::Stdio;
    let mut child = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{FRAME_W}x{FRAME_H}"),
            "-r",
            "60",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            "-loglevel",
            "error",
        ])
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                VideoError::FfmpegMissing
            } else {
                VideoError::Io(e)
            }
        })?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(frames)?;
    }
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(VideoError::FfmpegFailed {
            status: out.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        });
    }
    Ok(())
}

/// Concatenate per-clip mp4s into a final output.
fn concat_clips(clips: &[PathBuf], out: &Path) -> Result<(), VideoError> {
    if clips.is_empty() {
        return Err(VideoError::EmptyPlan);
    }
    if clips.len() == 1 {
        fs::copy(&clips[0], out)?;
        return Ok(());
    }
    let list_path = out.with_extension("clips.txt");
    {
        let mut f = File::create(&list_path)?;
        for c in clips {
            writeln!(f, "file '{}'", c.display())?;
        }
    }
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path.to_str().unwrap_or("clips.txt"),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            "-loglevel",
            "error",
        ])
        .arg(out)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                VideoError::FfmpegMissing
            } else {
                VideoError::Io(e)
            }
        })?;
    if !status.success() {
        return Err(VideoError::FfmpegFailed {
            status: status.code().unwrap_or(-1),
            stderr: String::new(),
        });
    }
    let _ = fs::remove_file(&list_path);
    Ok(())
}

/// Simulate one generation's best individual and render every frame to a
/// contiguous raw-RGB byte stream.
fn render_clip(
    record: &GenerationRecord,
    seconds: f32,
    task_kind: crate::tasks::TaskKind,
    clip_is_final: bool,
) -> Result<Vec<u8>, VideoError> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(VideoError::InvalidSeconds(seconds));
    }
    if record.genome.morph.w == 0 || record.genome.morph.h == 0 {
        return Err(VideoError::InvalidSeconds(seconds));
    }
    let (ground, obstacles) =
        terrain_for(task_kind, (record.genome.morph.w, record.genome.morph.h));
    let mut world = World::new(record.genome.morph.clone()).with_terrain(ground, obstacles);
    world.reset();
    let n_act = record.genome.morph.actuators().len();
    let mut offsets = vec![0.0_f32; n_act];
    let mut cam = Camera::for_robot(record.genome.morph.w, record.genome.morph.h);

    let total_frames = (seconds * 60.0).round() as usize;
    if total_frames == 0 {
        return Err(VideoError::InvalidSeconds(seconds));
    }

    let mut t = 0.0_f32;

    // Pass 1: simulate sequentially (camera smoothing is stateful), recording
    // a lightweight snapshot per frame.
    struct Snap {
        positions: Vec<crate::physics::Vec2>,
        cam: Camera,
        offsets: Vec<f32>,
    }
    let mut snaps: Vec<Snap> = Vec::with_capacity(total_frames);
    for _ in 0..total_frames {
        for (i, o) in offsets.iter_mut().enumerate() {
            *o = record.genome.controller.delta_l(i, t);
        }
        for _ in 0..SUBSTEPS_PER_FRAME {
            world.substep(&offsets);
            t += DT;
        }
        let com = world.com();
        cam.follow(com.x, com.y, 0.3);
        snaps.push(Snap {
            positions: world.masses.iter().map(|m| m.pos).collect(),
            cam,
            offsets: offsets.clone(),
        });
    }

    // Pass 2: rasterize frames in parallel, then join into one stream.
    let frame_bytes = FRAME_W * FRAME_H * 3;
    let frames: Vec<Vec<u8>> = snaps
        .par_iter()
        .map(|snap| {
            let mut buf = FrameBuffer::new(FRAME_W, FRAME_H);
            draw_terrain(&mut buf, &snap.cam, &world.ground, &world.obstacles);
            draw_robot_positions(&mut buf, &snap.cam, &world, &snap.positions, &snap.offsets);
            draw_overlay(&mut buf, record.generation, record.fitness, clip_is_final);
            buf.rgb
        })
        .collect();
    let mut all_bytes: Vec<u8> = Vec::with_capacity(total_frames * frame_bytes);
    for f in &frames {
        all_bytes.extend_from_slice(f);
    }
    Ok(all_bytes)
}

/// Full pipeline: render each plan entry in parallel, then concat.
///
/// `out_path` is the final `.mp4`. A temp dir is created alongside to hold
/// the per-clip files.
pub fn render_plan(
    plan: &[ClipSpec],
    history: &History,
    task_kind: crate::tasks::TaskKind,
    out_path: &Path,
) -> Result<VideoOutput, VideoError> {
    if plan.is_empty() {
        return Err(VideoError::EmptyPlan);
    }
    for c in plan {
        if !c.seconds.is_finite() || c.seconds <= 0.0 {
            return Err(VideoError::InvalidSeconds(c.seconds));
        }
    }
    if !ffmpeg_available() {
        return Err(VideoError::FfmpegMissing);
    }

    // Collect records up front so we can detect missing generations before
    // spawning any work.
    let records: Vec<&GenerationRecord> = plan
        .iter()
        .map(|spec| {
            history
                .best_of(spec.generation)
                .ok_or(VideoError::MissingGeneration(spec.generation))
        })
        .collect::<Result<_, _>>()?;

    // Render frames in parallel: each clip is independently built in memory.
    let clips_dir = out_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(
            ".{}.clips",
            out_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("evogym")
        ));
    fs::create_dir_all(&clips_dir)?;

    // Pre-allocate so the parallel closures don't fight over a single Vec.
    let rendered: Vec<(ClipSpec, Vec<u8>)> = plan
        .par_iter()
        .zip(records.par_iter().copied())
        .map(|(spec, rec)| {
            let frames = render_clip(rec, spec.seconds, task_kind, spec.is_final)?;
            Ok::<_, VideoError>((spec.clone(), frames))
        })
        .collect::<Result<_, _>>()?;

    // Encode each clip — also parallel; each writes to its own file via its
    // own ffmpeg process. ffmpeg is single-threaded per clip but we run them
    // concurrently.
    let clip_files: Vec<(ClipSpec, PathBuf)> = rendered
        .into_iter()
        .map(|(spec, frames)| {
            let path = clips_dir.join(format!("clip_gen{}.mp4", spec.generation));
            encode_clip(&frames, &path)?;
            Ok::<_, VideoError>((spec, path))
        })
        .collect::<Result<_, _>>()?;

    let ordered_files: Vec<PathBuf> = plan
        .iter()
        .filter_map(|spec| {
            clip_files
                .iter()
                .find(|(s, _)| s.generation == spec.generation)
                .map(|(_, p)| p.clone())
        })
        .collect();

    // Write final output.
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    concat_clips(&ordered_files, out_path)?;

    // Best-effort cleanup of temp clips.
    for (_, p) in &clip_files {
        let _ = fs::remove_file(p);
    }
    let _ = fs::remove_dir(&clips_dir);

    Ok(VideoOutput {
        output_path: out_path.to_path_buf(),
        clips: ordered_files,
    })
}

/// Quick sanity check: simulate one frame, write a single-frame mp4. Useful
/// for verifying that the encoding pipeline is wired correctly. The actual
/// integration test lives in `cli.rs` / the integration smoke test.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::Controller;
    use crate::evolution::Genome;
    use crate::tasks::TaskKind;
    use crate::voxel::{Cell, Morphology};

    fn tiny_history() -> History {
        let mut m = Morphology::new(2, 2);
        for cy in 0..2 {
            for cx in 0..2 {
                m.set(cx, cy, Cell::Soft);
            }
        }
        m.set(0, 0, Cell::ActH);
        let genome = Genome::new(m, Controller::zeros(1));
        let mut h = History::default();
        h.records.push(GenerationRecord {
            generation: 0,
            fitness: 0.5,
            genome,
        });
        h
    }

    #[test]
    fn ffmpeg_is_reachable() {
        // We expect ffmpeg to be installed in the verification env.
        assert!(
            ffmpeg_available(),
            "ffmpeg must be on PATH for tests to pass"
        );
    }

    #[test]
    fn render_clip_produces_nonempty_bytes() {
        let h = tiny_history();
        let rec = &h.records[0];
        let bytes = render_clip(rec, 1.0, TaskKind::Walker, false).unwrap();
        // 60 fps * 1 s = 60 frames * 1080 * 1080 * 3 bytes = 209_952_000.
        let expected = (1.0_f32 * 60.0).round() as usize * FRAME_W * FRAME_H * 3;
        assert_eq!(bytes.len(), expected);
    }

    #[test]
    fn invalid_seconds_is_rejected() {
        let h = tiny_history();
        let bytes = render_clip(&h.records[0], 0.0, TaskKind::Walker, false);
        assert!(matches!(bytes, Err(VideoError::InvalidSeconds(_))));
        // Plan with zero seconds also rejected.
        let r = render_plan(
            &[ClipSpec {
                generation: 0,
                seconds: 0.0,
                is_final: false,
            }],
            &h,
            TaskKind::Walker,
            Path::new("/tmp/x.mp4"),
        );
        assert!(matches!(r, Err(VideoError::InvalidSeconds(_))));
        // missing generation still surfaces as an error before encoding.
        let r = render_plan(
            &[ClipSpec {
                generation: 99,
                seconds: 1.0,
                is_final: false,
            }],
            &h,
            TaskKind::Walker,
            Path::new("/tmp/x.mp4"),
        );
        assert!(matches!(r, Err(VideoError::MissingGeneration(_))));
    }
}
