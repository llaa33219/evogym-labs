//! Headless CLI verification mode.
//!
//! Designed for CI / smoke tests where no display is available. Builds a
//! default 5×5 demo robot (a base of rigid cells with a vertical column of
//! actuators), evolves a few generations, renders short clips of the best
//! individuals of selected generations, and concatenates them into a single
//! `.mp4` — fully offscreen.
//!
//! ## Examples
//!
//! ```text
//! evogym-labs --headless \
//!     --gens 3 --pop 8 --eval-secs 2 \
//!     --clip-secs 1 --out /tmp/evogym_test.mp4
//! ```
//!
//! Exits 0 on success, non-zero on failure (with a one-line error on stderr).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;

use crate::evolution::{EvolutionConfig, Mode, Progress, run as evolve};
use crate::tasks::TaskKind;
use crate::video::{ClipSpec, VideoError, render_plan};
use crate::voxel::{Cell, Morphology};

/// Build a sensible 5×5 demo robot: rigid base + a few horizontal actuators
/// stacked in the centre. The horizontal actuators modulate the horizontal
/// edge springs, which is what produces forward locomotion; a vertical-only
/// design will only squish vertically and never move meaningfully.
pub fn default_demo_robot() -> Morphology {
    let mut m = Morphology::new(5, 5);
    // rigid base — two rows of rigid cells (legs).
    for cx in 0..5 {
        m.set(cx, 0, Cell::Rigid);
        m.set(cx, 1, Cell::Rigid);
    }
    // soft body in the middle.
    for cy in 2..5 {
        for cx in 0..5 {
            m.set(cx, cy, Cell::Soft);
        }
    }
    // three horizontal actuators stacked up the centre — these shorten and
    // lengthen horizontal springs and so drive forward locomotion.
    m.set(2, 2, Cell::ActH);
    m.set(2, 3, Cell::ActH);
    m.set(2, 4, Cell::ActH);
    m
}

#[derive(Parser, Debug)]
#[command(name = "evogym-labs", about = "2D soft-voxel robot evolution simulator", long_about = None)]
pub struct Cli {
    /// Run headless: evolve a default demo robot and render an .mp4. Never
    /// initialises eframe / winit / wgpu.
    #[arg(long)]
    pub headless: bool,

    /// Number of generations to evolve.
    #[arg(long, default_value_t = 3)]
    pub gens: usize,

    /// Population size.
    #[arg(long, default_value_t = 8)]
    pub pop: usize,

    /// Per-individual evaluation duration (seconds) used during evolution.
    #[arg(long, default_value_t = 2.0)]
    pub eval_secs: f32,

    /// Duration of each clip rendered into the final mp4 (seconds).
    #[arg(long, default_value_t = 1.0)]
    pub clip_secs: f32,

    /// Path of the output mp4.
    #[arg(long, default_value = "/tmp/evogym_test.mp4")]
    pub out: PathBuf,

    /// Optional seed for reproducibility.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Task / terrain to evolve and render on.
    #[arg(long, value_enum, default_value_t = CliTask::Walker)]
    pub task: CliTask,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum CliTask {
    Walker,
    Obstacle,
    Slope,
    Stairs,
    Obstacles,
    Hills,
}

impl From<CliTask> for TaskKind {
    fn from(t: CliTask) -> Self {
        match t {
            CliTask::Walker => TaskKind::Walker,
            CliTask::Obstacle => TaskKind::ObstacleTraverser,
            CliTask::Slope => TaskKind::Slope45,
            CliTask::Stairs => TaskKind::Stairs,
            CliTask::Obstacles => TaskKind::Obstacles,
            CliTask::Hills => TaskKind::Hills,
        }
    }
}

/// Run the headless CLI. Returns an exit code — never panics on user input.
pub fn run() -> ExitCode {
    let cli = Cli::parse();
    if !cli.headless {
        eprintln!("error: --headless is required for the CLI mode");
        return ExitCode::from(2);
    }
    match run_inner(&cli) {
        Ok(()) => ExitCode::from(0),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

fn run_inner(cli: &Cli) -> Result<(), VideoError> {
    let seed = default_demo_robot();
    let cfg = EvolutionConfig {
        mode: Mode::Full,
        task: cli.task.into(),
        population: cli.pop.max(2),
        eval_secs: cli.eval_secs.max(0.05),
        morph_size: (seed.w, seed.h),
        ..EvolutionConfig::defaults()
    };
    eprintln!(
        "[cli] headless evolution: gens={} pop={} eval_secs={:.2} task={:?}",
        cli.gens, cfg.population, cfg.eval_secs, cfg.task
    );

    let plan_gens: Vec<usize> = if cli.gens <= 1 {
        vec![0]
    } else {
        // First / last / midpoints, dedup'd & sorted.
        let mut plan = std::collections::BTreeSet::new();
        plan.insert(0);
        plan.insert(cli.gens - 1);
        if cli.gens >= 3 {
            plan.insert(cli.gens / 2);
        }
        plan.into_iter().collect()
    };

    let n_gens = cli.gens.max(1);
    let result = evolve(&cfg, seed, n_gens, cli.seed, |p| match p {
        Progress::Started { total_gens, pop } => {
            eprintln!("[cli] starting evolution: total_gens={total_gens} pop={pop}");
        }
        Progress::Generation {
            generation,
            total,
            best,
            mean,
        } => {
            eprintln!(
                "[cli]   gen {generation:>3}/{total}  best={best:>6.2} m  mean={mean:>6.2} m"
            );
        }
        Progress::Done {
            generations,
            best_overall,
        } => {
            eprintln!(
                "[cli] evolution done: generations={generations} best_overall={best_overall:.2} m"
            );
        }
        Progress::Error(e) => {
            eprintln!("[cli] GA error: {e}");
        }
    });

    let plan: Vec<ClipSpec> = plan_gens
        .iter()
        .enumerate()
        .map(|(i, &g)| ClipSpec {
            generation: g,
            seconds: cli.clip_secs.max(0.05),
            is_final: i + 1 == plan_gens.len(),
        })
        .collect();
    eprintln!(
        "[cli] rendering {} clip(s) into {}",
        plan.len(),
        cli.out.display()
    );
    for s in &plan {
        eprintln!(
            "[cli]   - generation {} for {:.2}s",
            s.generation, s.seconds
        );
    }

    let out = render_plan(&plan, &result.history, cfg.task, Path::new(&cli.out))?;
    eprintln!(
        "[cli] wrote {} ({} clip(s))",
        out.output_path.display(),
        out.clips.len()
    );
    Ok(())
}

/// Public helper for integration tests: run the same path as the CLI without
/// touching clap argument parsing.
pub fn run_with(
    gens: usize,
    pop: usize,
    eval_secs: f32,
    clip_secs: f32,
    out: &Path,
    seed: Option<u64>,
    task: TaskKind,
) -> Result<(), VideoError> {
    let cli = Cli {
        headless: true,
        gens,
        pop,
        eval_secs,
        clip_secs,
        out: out.to_path_buf(),
        seed,
        task: match task {
            TaskKind::Walker => CliTask::Walker,
            TaskKind::ObstacleTraverser => CliTask::Obstacle,
            TaskKind::Slope45 => CliTask::Slope,
            TaskKind::Stairs => CliTask::Stairs,
            TaskKind::Obstacles => CliTask::Obstacles,
            TaskKind::Hills => CliTask::Hills,
        },
    };
    run_inner(&cli)
}

/// Simple smoke integration test: build the demo robot, evolve 2 gens,
/// render a 1-second clip into /tmp. Skipped if ffmpeg is unavailable.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::Controller;

    #[test]
    fn demo_robot_is_valid() {
        let m = default_demo_robot();
        assert!(crate::voxel::validate(&m).is_ok());
    }

    #[test]
    fn controller_size_matches_demo_robot() {
        let m = default_demo_robot();
        let acts = m.actuators();
        let ctrl = Controller::random(&mut rand::rng(), acts.len());
        assert_eq!(ctrl.amps.len(), acts.len());
        assert_eq!(ctrl.phases.len(), acts.len());
    }
}
