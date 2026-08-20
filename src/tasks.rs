//! Tasks / environments the robot must traverse.
//!
//! Both tasks share the world physics (gravity, ground) — only the obstacle
//! layout differs. The fitness in every task is **horizontal distance
//! travelled by the robot's centre of mass** (`com().x - start_x`) over the
//! evaluation window.
//!
//! Obstacles are world-fixed: a robot spawning at the origin walks forward
//! and hits the obstacle field. The `ObstacleTraverser` task adds three low
//! boxes that the robot must climb over.

use crate::physics::{Ground, Obstacle};

/// Task kinds selectable from the GUI / CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskKind {
    Walker,
    ObstacleTraverser,
    Slope45,
    Stairs,
    Obstacles,
    Hills,
}

impl TaskKind {
    pub const ALL: [TaskKind; 6] = [
        TaskKind::Walker,
        TaskKind::ObstacleTraverser,
        TaskKind::Slope45,
        TaskKind::Stairs,
        TaskKind::Obstacles,
        TaskKind::Hills,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Walker => "Walker (flat)",
            Self::ObstacleTraverser => "ObstacleTraverser (boxes)",
            Self::Slope45 => "Slope (45°)",
            Self::Stairs => "Stairs",
            Self::Obstacles => "Obstacles (random)",
            Self::Hills => "Hills (sine)",
        }
    }
}

/// Build the terrain (ground profile + obstacle boxes) for a task. The
/// feature sizes scale with the creature's grid size `(w, h)` per spec:
///
/// * Slope45 — 45° incline continuing through and behind the spawn point.
/// * Stairs — tread `w` m wide, riser `floor(h/2)` m tall.
/// * Obstacles — random boxes with sides in `1 ..= max(1, floor(h/2))` m,
///   placed ahead of the spawn on otherwise flat ground.
/// * Hills — sine of amplitude `ceil(h²)` m, wavelength `4 × amplitude`.
///
/// Ground profiles are continuous over the whole x-axis — including under
/// the spawn footprint; the world lifts the robot just above the terrain
/// under it and lets it drop. The random obstacle layout is derived from a
/// fixed seed so evolution and video replay see the *same* course (replay
/// determinism is what keeps best-of-generation fitness meaningful).
pub fn terrain_for(kind: TaskKind, morph_size: (u8, u8)) -> (Ground, Vec<Obstacle>) {
    let (w, h) = morph_size;
    match kind {
        TaskKind::Walker => (Ground::Flat, Vec::new()),
        TaskKind::ObstacleTraverser => (
            Ground::Flat,
            vec![
                Obstacle::new(2.5, 3.2, 0.0, 0.40),
                Obstacle::new(5.5, 6.2, 0.0, 0.40),
                Obstacle::new(8.5, 9.2, 0.0, 0.40),
            ],
        ),
        TaskKind::Slope45 => (Ground::Slope { rate: 1.0 }, Vec::new()),
        TaskKind::Stairs => (
            Ground::Stairs { run: w.max(1) as f32, rise: (h / 2).max(1) as f32 },
            Vec::new(),
        ),
        TaskKind::Obstacles => (Ground::Flat, random_obstacles(w as f32 + 1.0, h)),
        TaskKind::Hills => {
            // h²/4 amplitude with a wavelength of 8× the amplitude: rolling
            // hills the creature can actually see crest (max slope ≈ 38°).
            // A plain h² amplitude makes a mountain the camera never gets
            // past the toe of.
            let amp = (h as f32) * (h as f32) / 4.0;
            (Ground::Hills { amp, wavelength: 8.0 * amp }, Vec::new())
        }
    }
}

/// Deterministic random sculpture field: 8 boxes, sides in
/// `1 ..= max(1, floor(h/2))` m, gaps of 2–5 m, seeded by the grid height.
fn random_obstacles(x0: f32, h: u8) -> Vec<Obstacle> {
    use rand::{Rng, SeedableRng};
    let max_side = (h / 2).max(1) as f32;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5eed_5eed ^ h as u64);
    let mut out = Vec::new();
    let mut x = x0 + 1.0;
    for _ in 0..8 {
        let bw = rng.random_range(1.0..=max_side);
        let bh = rng.random_range(1.0..=max_side);
        out.push(Obstacle::new(x, x + bw, 0.0, bh));
        x += bw + rng.random_range(2.0..5.0);
    }
    out
}

/// Compute fitness for a single rollout.
///
/// `start_com_x` is the COM x-position at t=0 (we record it before stepping).
/// `final_com_x` is the COM x-position after the eval. Fitness is the signed
/// horizontal displacement: positive means forward progress, zero or negative
/// means the robot did not move (or moved backwards). A NaN-scrubbed eval
/// yields `fitness = 0`.
pub fn fitness(start_com_x: f32, final_com_x: f32, nan_seen: bool) -> f32 {
    if nan_seen || !start_com_x.is_finite() || !final_com_x.is_finite() {
        return 0.0;
    }
    final_com_x - start_com_x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walker_has_no_obstacles() {
        let (ground, obs) = terrain_for(TaskKind::Walker, (5, 5));
        assert_eq!(ground, Ground::Flat);
        assert!(obs.is_empty());
    }

    #[test]
    fn obstacle_task_has_three_boxes() {
        let (_, obs) = terrain_for(TaskKind::ObstacleTraverser, (5, 5));
        assert_eq!(obs.len(), 3);
        for o in &obs {
            assert!(o.y_max < 1.0, "boxes are designed to be climbable");
            assert!(o.x_min < o.x_max);
        }
    }

    #[test]
    fn ground_profiles_match_spec() {
        let (slope, _) = terrain_for(TaskKind::Slope45, (5, 5));
        assert!((slope.height(2.5) - 2.5).abs() < 1e-4, "45° everywhere, no flat shelf");
        assert!((slope.height(-4.0) - (-4.0)).abs() < 1e-4, "continues behind spawn");

        let (stairs, _) = terrain_for(TaskKind::Stairs, (5, 5));
        assert_eq!(stairs.height(2.5), 0.0, "first tread: run = w = 5");
        assert_eq!(stairs.height(5.5), 2.0, "riser = floor(h/2) = 2");
        assert_eq!(stairs.height(10.5), 4.0);
        assert_eq!(stairs.height(-0.5), -2.0, "descends behind spawn");

        let (hills, _) = terrain_for(TaskKind::Hills, (5, 5));
        let peak = 12.5; // quarter wavelength (amp = h²/4 = 6.25, λ = 50)
        assert!((hills.height(peak) - 6.25).abs() < 1e-3, "amp = h²/4 = 6.25");
        assert!(hills.height(-12.5).abs() > 0.5, "waves continue behind spawn");
        assert!(hills.height(2.5) > 0.5, "no flat shelf under the spawn");
    }

    #[test]
    fn random_obstacles_are_deterministic_and_spec_sized() {
        let (_, a) = terrain_for(TaskKind::Obstacles, (5, 5));
        let (_, b) = terrain_for(TaskKind::Obstacles, (5, 5));
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.x_min, y.x_min);
            assert_eq!(x.y_max, y.y_max);
        }
        for o in &a {
            assert!(o.x_min >= 6.0, "past spawn margin");
            assert!((1.0..=2.0).contains(&(o.x_max - o.x_min)), "width in 1..=floor(5/2)");
            assert!((1.0..=2.0).contains(&o.y_max), "height in 1..=floor(5/2)");
        }
    }

    #[test]
    fn fitness_measures_forward_distance() {
        assert!((fitness(0.0, 5.0, false) - 5.0).abs() < 1e-5);
        assert!((fitness(2.0, 2.0, false) - 0.0).abs() < 1e-5);
        assert!((fitness(5.0, 1.0, false) - (-4.0)).abs() < 1e-5);
    }

    #[test]
    fn nan_yields_zero_fitness() {
        assert_eq!(fitness(0.0, f32::NAN, false), 0.0);
        assert_eq!(fitness(0.0, 5.0, true), 0.0);
    }
}