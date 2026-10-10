//! Simple genetic algorithm for morphology + controller co-evolution.
//!
//! Genome:
//! * **Morphology** — a fixed-size grid of `Cell` (width × height chosen by the
//!   user / CLI and held constant across the run).
//! * **Controller** — `freq_hz` + per-actuator `(phase, amp)` (the number of
//!   actuators is determined by the morphology, so it changes when morphology
//!   mutation rewires the cell grid).
//!
//! Loop:
//! 1. Evaluate every individual in the population (roll out the physics for
//!    `eval_secs`, fitness = forward COM displacement).
//! 2. Tournament-select parents (size 3). Elitism: top 2 carry over verbatim.
//! 3. Mutate offspring:
//!    * **Morphology** — with probability `p_cell_flip`, flip a random
//!      non-empty cell to `Empty` or to a random solid type; with probability
//!      `p_cell_add`, switch a random `Empty` cell to a random solid type; with
//!      probability `p_cell_del`, flip a random solid cell to `Empty`. Reject
//!      offspring that fail `voxel::validate` (empty / no actuator /
//!      disconnected / wrong size) — fall back to a clone of the parent.
//!    * **Controller** — gaussian perturbation of `freq_hz` / `phases` / `amps`
//!      via `Controller::mutate`. If the offspring's actuator count changed
//!      (because morphology mutation grew or shrank the actuator list), the
//!      controller is resized to match.
//!
//! Evaluation is parallelised across the population with rayon.
//!
//! ## History
//!
//! Each generation's best individual is saved in `History::records` so any
//! generation can be replayed for video rendering.

use std::sync::Arc;

use rand::prelude::IndexedRandom;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;

use crate::controller::{Controller, MutationParams as CtrlMutation};
use crate::physics::World;
use crate::tasks::{TaskKind, fitness as task_fitness, terrain_for};
use crate::voxel::{Cell, Morphology, validate};

/// Evolution mode — morphology mutates too, or stays fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    Full,
    ControllerOnly,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Full => "Full (morph + ctrl)",
            Self::ControllerOnly => "Controller only",
        }
    }
}

/// Configuration for a single GA run. Cheap to copy / clone.
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    pub mode: Mode,
    pub task: TaskKind,
    pub population: usize,
    pub eval_secs: f32,
    pub morph_size: (u8, u8),
    pub elitism: usize,
    pub tournament_size: usize,
    pub p_cell_flip: f32,
    pub p_cell_add: f32,
    pub p_cell_del: f32,
    pub morph_mutations_per_child: usize,
    pub ctrl: CtrlMutation,
}

impl EvolutionConfig {
    pub fn defaults() -> Self {
        Self {
            mode: Mode::Full,
            task: TaskKind::Walker,
            population: 32,
            eval_secs: 10.0,
            morph_size: (5, 5),
            elitism: 2,
            tournament_size: 3,
            p_cell_flip: 0.6,
            p_cell_add: 0.3,
            p_cell_del: 0.1,
            morph_mutations_per_child: 2,
            ctrl: CtrlMutation::default(),
        }
    }
}

/// A genome — morphology + controller.
#[derive(Debug, Clone)]
pub struct Genome {
    pub morph: Morphology,
    pub controller: Controller,
}

impl Genome {
    pub fn new(morph: Morphology, controller: Controller) -> Self {
        Self { morph, controller }
    }

    /// Build a controller sized for this morphology.
    pub fn random_controller(&self, rng: &mut impl RngCore) -> Controller {
        Controller::random(rng, self.morph.actuators().len())
    }
}

/// One generation's best individual, saved for later replay.
#[derive(Debug, Clone)]
pub struct GenerationRecord {
    pub generation: usize,
    pub fitness: f32,
    pub genome: Genome,
}

/// Snapshot of the full GA history — used by the GUI / video planner.
#[derive(Debug, Clone, Default)]
pub struct History {
    pub records: Vec<GenerationRecord>,
}

impl History {
    /// Lookup the best genome of a particular generation.
    pub fn best_of(&self, generation: usize) -> Option<&GenerationRecord> {
        self.records.iter().find(|r| r.generation == generation)
    }
}

/// Progress events emitted by the GA, surfaced to the GUI via mpsc.
#[derive(Debug, Clone)]
pub enum Progress {
    Started {
        total_gens: usize,
        pop: usize,
    },
    Generation {
        generation: usize,
        total: usize,
        best: f32,
        mean: f32,
    },
    Done {
        generations: usize,
        best_overall: f32,
    },
    Error(String),
}

/// Outcome of an evolution run, used by both CLI and GUI paths.
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    pub history: History,
    pub best_fitness: f32,
}

/// Run evolution end-to-end. `seed_morph` is the user's drawn design — it
/// seeds generation 0; the rest of gen-0 is filled with mutations of it.
pub fn run(
    cfg: &EvolutionConfig,
    seed_morph: Morphology,
    n_generations: usize,
    seed_rng: Option<u64>,
    mut on_progress: impl FnMut(Progress) + Send,
) -> EvolutionResult {
    on_progress(Progress::Started {
        total_gens: n_generations,
        pop: cfg.population,
    });

    // Two RNG streams: master for GA operations (sequential), and one-shot
    // seeds for the parallel evaluation rayon pass.
    let mut master = match seed_rng {
        Some(s) => Box::new(rand::rngs::StdRng::seed_from_u64(s)) as Box<dyn RngCore + Send>,
        None => Box::new(rand::rngs::StdRng::from_os_rng()) as Box<dyn RngCore + Send>,
    };

    // Generation 0 population: seed + mutations of the seed.
    let mut population: Vec<Genome> = Vec::with_capacity(cfg.population);
    {
        let n_act = seed_morph.actuators().len();
        let base_ctrl = Controller::random(&mut *master, n_act);
        population.push(Genome::new(seed_morph.clone(), base_ctrl));
        while population.len() < cfg.population {
            let mut child =
                Genome::new(seed_morph.clone(), Controller::random(&mut *master, n_act));
            if cfg.mode == Mode::Full {
                child.morph = mutate_morphology(&child.morph, cfg, &mut *master);
            }
            child.controller =
                mutate_controller(child.controller.clone(), &child.morph, cfg, &mut *master);
            // validate (controller-only: morph is seed_morph so still valid)
            if validate(&child.morph).is_err() {
                child.morph = seed_morph.clone();
            }
            population.push(child);
        }
    }

    let (ground, obstacles) = terrain_for(cfg.task, cfg.morph_size);
    let obstacles = Arc::new(obstacles);
    let mut history = History::default();

    for g in 0..n_generations {
        // Evaluate in parallel.
        let fits: Vec<(usize, f32)> = population
            .par_iter()
            .enumerate()
            .map(|(idx, genome)| {
                let mut world = World::new(genome.morph.clone())
                    .with_terrain(ground, obstacles.as_ref().clone());
                world.reset();
                let start_x = world.com().x;
                let n_act = genome.morph.actuators().len();
                // 60 fps control rate
                let frames = (cfg.eval_secs * 60.0).round() as i32;
                let mut t = 0.0_f32;
                let mut offsets = vec![0.0_f32; n_act];
                let mut any_nan = false;
                for _ in 0..frames {
                    for (i, o) in offsets.iter_mut().enumerate() {
                        *o = genome.controller.delta_l(i, t);
                    }
                    for _ in 0..crate::physics::SUBSTEPS_PER_FRAME {
                        world.substep(&offsets);
                        t += crate::physics::DT;
                    }
                    if world.nan_seen {
                        any_nan = true;
                        break;
                    }
                }
                let final_com = world.com();
                let fit = task_fitness(start_x, final_com.x, any_nan || world.nan_seen);
                (idx, fit)
            })
            .collect();

        // Sort descending by fitness. (`fits[i].0` is the population index —
        // using it as the primary key silently ranks by index, not fitness.)
        let mut order: Vec<usize> = (0..population.len()).collect();
        order.sort_by(|&a, &b| {
            fits[b]
                .1
                .partial_cmp(&fits[a].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Record best
        let best_idx = order[0];
        let best_fit = fits[best_idx].1;
        let best_genome = population[best_idx].clone();
        history.records.push(GenerationRecord {
            generation: g,
            fitness: best_fit,
            genome: best_genome,
        });

        let sum: f32 = fits.iter().map(|(_, f)| *f).sum();
        let mean = sum / fits.len() as f32;
        on_progress(Progress::Generation {
            generation: g,
            total: n_generations,
            best: best_fit,
            mean,
        });

        if g + 1 == n_generations {
            break;
        }

        // Build next generation.
        let mut next: Vec<Genome> = Vec::with_capacity(cfg.population);
        // Elitism
        for &i in order.iter().take(cfg.elitism.min(population.len())) {
            next.push(population[i].clone());
        }
        // Fill by tournament + mutation
        while next.len() < cfg.population {
            let parent_a = tournament(&population, &fits, cfg.tournament_size, &mut *master);
            let parent_b = tournament(&population, &fits, cfg.tournament_size, &mut *master);
            let mut child = uniform_crossover(&parent_a, &parent_b, &mut *master);
            // mutate morphology (unless ControllerOnly)
            if cfg.mode == Mode::Full {
                child.morph = mutate_morphology(&child.morph, cfg, &mut *master);
            }
            child.controller =
                mutate_controller(child.controller.clone(), &child.morph, cfg, &mut *master);
            // validate morphology; if invalid, fall back to parent A
            if validate(&child.morph).is_err() {
                child.morph = parent_a.morph.clone();
                child.controller =
                    mutate_controller(parent_a.controller.clone(), &child.morph, cfg, &mut *master);
            }
            next.push(child);
        }
        population = next;
    }

    let best_overall = history
        .records
        .iter()
        .map(|r| r.fitness)
        .fold(f32::NEG_INFINITY, f32::max);
    let best_overall = if best_overall.is_finite() {
        best_overall
    } else {
        0.0
    };
    on_progress(Progress::Done {
        generations: n_generations,
        best_overall,
    });

    EvolutionResult {
        history,
        best_fitness: best_overall,
    }
}

fn tournament<R: Rng + ?Sized>(
    pop: &[Genome],
    fits: &[(usize, f32)],
    size: usize,
    rng: &mut R,
) -> Genome {
    let mut best_idx = rng.random_range(0..pop.len());
    let mut best_fit = fits[best_idx].1;
    for _ in 1..size {
        let i = rng.random_range(0..pop.len());
        if fits[i].1 > best_fit {
            best_fit = fits[i].1;
            best_idx = i;
        }
    }
    pop[best_idx].clone()
}

fn uniform_crossover<R: Rng + ?Sized>(a: &Genome, b: &Genome, rng: &mut R) -> Genome {
    let mut cells = a.morph.cells.clone();
    let mut colors = a.morph.colors.clone();
    for (i, bc) in b.morph.cells.iter().enumerate() {
        if rng.random::<f32>() < 0.5 {
            cells[i] = *bc;
            colors[i] = b.morph.colors[i];
        }
    }
    let morph = Morphology {
        w: a.morph.w,
        h: a.morph.h,
        cells,
        colors,
    };
    let n_act_a = a.morph.actuators().len();
    let n_act_b = b.morph.actuators().len();
    let n = n_act_a.max(n_act_b);
    let mut phases = a.controller.phases.clone();
    phases.resize(n, 0.0);
    let mut amps = a.controller.amps.clone();
    amps.resize(n, 0.0);
    for i in 0..n {
        if i < b.controller.phases.len() && rng.random::<f32>() < 0.5 {
            phases[i] = b.controller.phases[i];
        }
        if i < b.controller.amps.len() && rng.random::<f32>() < 0.5 {
            amps[i] = b.controller.amps[i];
        }
    }
    let freq = if rng.random::<f32>() < 0.5 {
        a.controller.freq_hz
    } else {
        b.controller.freq_hz
    };
    let controller = Controller {
        freq_hz: freq,
        phases,
        amps,
    };
    Genome::new(morph, controller)
}

fn mutate_morphology<R: Rng + ?Sized>(
    parent: &Morphology,
    cfg: &EvolutionConfig,
    rng: &mut R,
) -> Morphology {
    let mut m = parent.clone();
    for _ in 0..cfg.morph_mutations_per_child.max(1) {
        let r: f32 = rng.random();
        if r < cfg.p_cell_flip {
            // Flip a random cell (could become Empty or any type).
            let cx = rng.random_range(0..m.w);
            let cy = rng.random_range(0..m.h);
            let new_kind = match rng.random_range(0..5u8) {
                0 => Cell::Empty,
                1 => Cell::Rigid,
                2 => Cell::Soft,
                3 => Cell::ActH,
                _ => Cell::ActV,
            };
            m.set(cx, cy, new_kind);
        } else if r < cfg.p_cell_flip + cfg.p_cell_add {
            // Add a solid cell somewhere Empty.
            let empties: Vec<(u8, u8)> = (0..m.h)
                .flat_map(|cy| (0..m.w).map(move |cx| (cx, cy)))
                .filter(|&(cx, cy)| m.get(cx, cy) == Cell::Empty)
                .collect();
            if let Some(&(cx, cy)) = empties.choose(rng) {
                let new_kind = match rng.random_range(0..3u8) {
                    0 => Cell::Rigid,
                    1 => Cell::Soft,
                    _ => Cell::ActH,
                };
                m.set(cx, cy, new_kind);
            }
        } else {
            // Delete a random solid cell.
            let solids: Vec<(u8, u8)> = (0..m.h)
                .flat_map(|cy| (0..m.w).map(move |cx| (cx, cy)))
                .filter(|&(cx, cy)| m.get(cx, cy).is_solid())
                .collect();
            if let Some(&(cx, cy)) = solids.choose(rng) {
                m.set(cx, cy, Cell::Empty);
            }
        }
    }
    recolour_changed_cells(parent, &mut m, rng);
    m
}

/// After a morphology mutation, any cell whose type changed gets a fresh
/// colour: a slight variation (±20 per channel) of one of the creature's
/// existing same-type colour overrides, or the type default when no such
/// colour exists. Cells that became Empty drop their override.
fn recolour_changed_cells<R: Rng + ?Sized>(parent: &Morphology, m: &mut Morphology, rng: &mut R) {
    for i in 0..m.cells.len() {
        if m.cells[i] == parent.cells[i] {
            continue;
        }
        if !m.cells[i].is_solid() {
            m.colors[i] = None;
            continue;
        }
        let kind = m.cells[i];
        let pool: Vec<[u8; 3]> = (0..m.cells.len())
            .filter(|&j| j != i && m.cells[j] == kind)
            .filter_map(|j| m.colors[j])
            .collect();
        m.colors[i] = pool.choose(rng).map(|&base| {
            let mut vary = |c: u8| (c as i32 + rng.random_range(-20..=20)).clamp(0, 255) as u8;
            [vary(base[0]), vary(base[1]), vary(base[2])]
        });
    }
}

fn mutate_controller<R: Rng + ?Sized>(
    parent: Controller,
    morph: &Morphology,
    cfg: &EvolutionConfig,
    rng: &mut R,
) -> Controller {
    let mut c = parent;
    let n_act = morph.actuators().len();
    if c.phases.len() != n_act {
        c.resize(n_act, rng);
    }
    c.mutate(rng, cfg.ctrl);
    c
}

#[cfg(test)]
#[path = "evolution_tests.rs"]
mod tests;
