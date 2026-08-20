//! Integration smoke test: run the headless CLI end-to-end and verify the
//! output is a valid h264 mp4 at 1080×1080.
//!
//! Disabled by default — opt in with `cargo test --features integration`
//! because it takes a few seconds and depends on `ffmpeg` being installed.

#![cfg(feature = "integration")]

use std::path::Path;
use std::process::Command;

#[test]
fn headless_cli_produces_valid_video() {
    let out = Path::new("/tmp/opencode/evogym_integration_test.mp4");
    let _ = std::fs::remove_file(out);

    evogym_labs::cli::run_with(
        2, // gens
        4, // pop
        0.5, // eval_secs
        1.0, // clip_secs
        out,
        Some(42), // seed
        evogym_labs::tasks::TaskKind::Walker,
    )
    .expect("headless CLI should succeed");

    assert!(out.exists(), "output mp4 was not written");

    let ffprobe = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name,pix_fmt,r_frame_rate",
            "-of", "csv=s=x:p=0",
        ])
        .arg(out)
        .output()
        .expect("ffprobe must be available");
    assert!(ffprobe.status.success(), "ffprobe failed");
    let stdout = String::from_utf8_lossy(&ffprobe.stdout).into_owned();
    let parts: Vec<&str> = stdout.trim().split('x').collect();
    assert_eq!(parts.len(), 5, "unexpected stream line shape: {stdout}");
    assert_eq!(parts[0], "h264", "codec mismatch: {stdout}");
    assert_eq!(parts[1], "1080", "width mismatch: {stdout}");
    assert_eq!(parts[2], "1080", "height mismatch: {stdout}");
    assert_eq!(parts[3], "yuv420p", "pix_fmt mismatch: {stdout}");
    assert_eq!(parts[4], "60/1", "fps mismatch: {stdout}");
    let _ = std::fs::remove_file(out);
}