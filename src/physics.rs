//! Hand-rolled 2-D point-mass / spring physics for voxel robots.
//!
//! # Model
//!
//! * One `PointMass` per grid vertex `(vx, vy)` (so a `w × h` cell grid has
//!   `(w+1) × (h+1)` masses, initialised at world position `(vx, vy)`).
//! * Each non-empty cell contributes six springs (4 edges + 2 diagonals) among
//!   its four corners. Adjacent cells share the same edge spring — when
//!   multiple cells share an edge, the stiffer one (Rigid > Soft) wins.
//! * An `ActH` cell modulates the rest length of its two horizontal edge
//!   springs; an `ActV` cell modulates its two vertical edge springs. Diagonals
//!   are never modulated by actuators (they remain pure structural links).
//!   Adjacent actuator cells add their signals.
//!
//! # Integration
//!
//! Semi-implicit Euler at `dt = 1/240 s` with **4 substeps per 60 Hz frame**.
//! Numerical stability for random high-K designs over hundreds of generations
//! rests on four guards:
//!   1. **Per-mass linear damping** (`v *= 0.995` per substep) — bleeds energy
//!      out faster than a stiff spring can pump it in.
//!   2. **Velocity clamp** (`|v| ≤ 50 m/s`) — caps the integrator's worst case
//!      step.
//!   3. **Position clamp** (`y ≥ -0.5`, `|x| ≤ 2000`) — keeps NaN-producing
//!      explosions bounded so the simulation can keep running.
//!   4. **Post-step NaN scrub**: any mass that ends up with NaN coords or
//!      infinite velocity is yanked to origin with zero velocity; the *next*
//!      eval reports fitness 0 to the GA.
//!
//! # Forces
//!
//! Each spring contributes `F = -k · (|d| - L_eff) · d̂ - c · (v_rel · d̂) · d̂`
//! where `L_eff = base + Σ actuator_ΔL`. Ground & obstacle contact applies a
//! stiff normal penalty spring and Coulomb-ish friction: the tangential force
//! is clamped to `μ · F_normal`.

use crate::voxel::Morphology;

/// Gravity (y points up).
pub const GRAVITY: (f32, f32) = (0.0, -9.81);

/// Per-vertex mass. One mass per grid vertex; a `w × h` cell grid therefore
/// has `(w+1)(h+1)` masses and each cell contributes `~0.25` of this mass on
/// average to its four corners. Tuned together with stiffnesses to stay below
/// the semi-implicit Euler stability limit at `dt = 1/240`.
pub const MASS: f32 = 1.0;

/// Substep size. 2 substeps per 60 fps frame — the worst-case spring
/// stiffness budget keeps the semi-implicit stability limit at ~1/77 s,
/// so 1/120 s is safely inside it and twice as fast as 1/240 s.
pub const DT: f32 = 1.0 / 120.0;
pub const SUBSTEPS_PER_FRAME: usize = 2;

/// Spring stiffness buckets.
pub(crate) const K_RIGID_EDGE: f32 = 4_000.0;
pub(crate) const K_RIGID_DIAG: f32 = 2_000.0;
pub(crate) const K_SOFT_EDGE: f32 = 800.0;
pub(crate) const K_SOFT_DIAG: f32 = 400.0;

/// Damping coefficient per spring (Rayleigh-style along the spring axis).
const SPRING_DAMP: f32 = 8.0;
/// Per-substep linear damping applied to every mass velocity (energy bleed).
const GLOBAL_DAMP: f32 = 0.005;
/// Maximum allowed |v| after integration (explosion guard).
const V_MAX: f32 = 50.0;
/// Horizontal position clamp — robot cannot tunnel beyond this.
const X_BOUND: f32 = 2_000.0;

/// Ground contact: normal stiffness, friction coefficient.
pub const GROUND_K: f32 = 4_000.0;
pub const GROUND_MU: f32 = 1.0;
/// Obstacle contact: same shape, slightly softer.
pub const OBSTACLE_K: f32 = 3_000.0;
pub const OBSTACLE_MU: f32 = 0.9;

/// Hard cap on any single force contribution (keeps stiff springs from
/// generating numerical noise when a mass tunnels).
const F_MAX: f32 = 20_000.0;
const N_MAX: f32 = 20_000.0;

/// Self-collision skin thickness. Body parts may press to within this gap;
/// blocking happens at the cell *surface* (edges and interior), not at
/// mass points, so parts can neither overlap nor pass through each other.
const SELF_COLLIDE_SKIN: f32 = 0.12;
/// Largest positional correction applied in a single substep — big enough
/// to resolve contact fast, small enough to never look (or act) like a
/// teleport.
const MAX_POS_CORRECT: f32 = 0.15;
/// Minimum spring length as a fraction of base length, enforced
/// *positionally* after integration. Cells squish softly through normal
/// spring forces, but a spring can never be crushed past half its length —
/// and because no giant compression force ever appears, a crumple can
/// never fling the body.
const MIN_SPRING_RATIO: f32 = 0.5;

#[derive(Debug, Clone, Copy, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PointMass {
    pub pos: Vec2,
    pub vel: Vec2,
    pub force: Vec2,
    pub mass: f32,
}

impl PointMass {
    fn fresh(pos: Vec2) -> Self {
        Self { pos, vel: Vec2::ZERO, force: Vec2::ZERO, mass: MASS }
    }
}

/// One spring in the world. `actuator_idx` lists controller indices whose
/// per-actuator phase / amp modulates this spring's effective rest length.
#[derive(Debug, Clone)]
pub struct Spring {
    pub a: usize,
    pub b: usize,
    pub base_len: f32,
    pub k: f32,
    pub is_diag: bool,
    pub actuator_idx: Vec<usize>,
}

/// Axis-aligned obstacle box. The robot must push past / climb these.
#[derive(Debug, Clone, Copy)]
pub struct Obstacle {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
}

impl Obstacle {
    pub fn new(x_min: f32, x_max: f32, y_min: f32, y_max: f32) -> Self {
        Self { x_min, x_max, y_min, y_max }
    }
}

/// Heightfield ground profile, continuous over the whole x-axis. The robot
/// spawns lifted just above the highest terrain point under its footprint
/// (so it never spawns wedged inside a wall) and then drops onto whatever
/// lies below — falling is part of the challenge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ground {
    Flat,
    /// Inclined plane with the given rate (tan of the incline angle).
    Slope { rate: f32 },
    /// Staircase: treads of `run` width, each `rise` taller than the last.
    Stairs { run: f32, rise: f32 },
    /// Sine-wave hills with the given amplitude / wavelength.
    Hills { amp: f32, wavelength: f32 },
}

impl Ground {
    pub fn height(&self, x: f32) -> f32 {
        match *self {
            Ground::Flat => 0.0,
            Ground::Slope { rate } => x * rate,
            Ground::Stairs { run, rise } => (x / run).floor() * rise,
            Ground::Hills { amp, wavelength } => {
                amp * (std::f32::consts::TAU * x / wavelength).sin()
            }
        }
    }

    /// Local surface slope dg/dx by central difference.
    fn gradient(&self, x: f32) -> f32 {
        const EPS: f32 = 0.02;
        (self.height(x + EPS) - self.height(x - EPS)) / (2.0 * EPS)
    }

    /// Contact resolution for a point mass: `Some((normal, penetration))`
    /// when the point is inside the ground, `None` when above the surface.
    /// `(vx, vy)` is the mass velocity, used only in the stair corner zone:
    /// when a point sits near a convex step corner (both the top face and a
    /// side face are close), the exit face is chosen by *approach direction*
    /// — a mass arriving horizontally is pushed back out of the riser, a
    /// mass falling onto the corner lands on the tread. Pure nearest-face
    /// resolution can't tell these apart and pops sideways-arriving feet
    /// up onto the step.
    pub fn contact(&self, x: f32, y: f32, vx: f32, vy: f32) -> Option<((f32, f32), f32)> {
        /// Below this distance to both faces we are "at the corner".
        const CORNER_ZONE: f32 = 0.3;
        let g = self.height(x);
        if y >= g { return None; }
        match *self {
            Ground::Stairs { run, .. } => {
                let i = (x / run).floor();
                let pen_up = g - y;
                let pen_left = x - i * run;
                let pen_right = (i + 1.0) * run - x;
                let side = if pen_left <= pen_right { (-1.0, pen_left) } else { (1.0, pen_right) };
                if pen_up <= side.1 {
                    let at_corner = side.1 < CORNER_ZONE && pen_up < CORNER_ZONE;
                    if at_corner && vx.abs() > vy.abs() {
                        Some(((side.0, 0.0), side.1))
                    } else {
                        Some(((0.0, 1.0), pen_up))
                    }
                } else {
                    Some(((side.0, 0.0), side.1))
                }
            }
            _ => {
                let gp = self.gradient(x);
                let inv_len = 1.0 / (1.0 + gp * gp).sqrt();
                Some(((-gp * inv_len, inv_len), g - y))
            }
        }
    }
}

/// A built world: points, springs, and per-spring actuator wiring.
#[derive(Debug)]
pub struct World {
    pub morph: Morphology,
    pub masses: Vec<PointMass>,
    pub springs: Vec<Spring>,
    /// For each cell, the index of its actuator in the controller's array —
    /// `None` for non-actuator cells. Order matches `Morphology::actuators()`.
    pub cell_actuator_idx: Vec<Option<usize>>,
    pub obstacles: Vec<Obstacle>,
    pub ground: Ground,
    /// Self-collision: solid-cell corners + non-adjacent cell pairs.
    cell_verts: Vec<[usize; 4]>,
    cell_pairs: Vec<(usize, usize)>,
    /// Which masses belong to at least one spring (i.e. to the robot body).
    /// Dead masses just fall and lie on the terrain — they must not count
    /// toward the COM, or the camera / fitness would track debris.
    active: Vec<bool>,
    /// Spawn height offset added to every vertex on `reset`.
    spawn_y: f32,
    /// Set true the first time `step` produces NaN; never cleared.
    pub nan_seen: bool,
}

#[path = "physics_build.rs"]
mod build;

use self::build::{build_masses, cell_actuator_indices, cell_collision_data, collect_cell_springs, dedupe_springs};

impl World {
    /// Build a world from a morphology. Vertices are placed at world
    /// `(vx, vy)`; actuators are wired to their modulated springs.
    pub fn new(morph: Morphology) -> Self {
        let spawn_y = 0.05;
        let masses = build_masses(morph.w, morph.h, spawn_y);
        let actuator_at = cell_actuator_indices(&morph.cells);
        let raw = collect_cell_springs(&morph, &actuator_at);
        let springs = dedupe_springs(raw);
        let (cell_verts, cell_pairs) = cell_collision_data(&morph);
        let mut active = vec![false; masses.len()];
        for s in &springs {
            active[s.a] = true;
            active[s.b] = true;
        }
        Self {
            morph,
            masses,
            springs,
            cell_actuator_idx: actuator_at,
            obstacles: Vec::new(),
            ground: Ground::Flat,
            cell_verts,
            cell_pairs,
            active,
            spawn_y,
            nan_seen: false,
        }
    }

    pub fn with_obstacles(mut self, obs: Vec<Obstacle>) -> Self {
        self.obstacles = obs;
        self
    }

    pub fn with_terrain(mut self, ground: Ground, obs: Vec<Obstacle>) -> Self {
        self.ground = ground;
        self.obstacles = obs;
        self
    }

    /// Reset all masses back to their initial positions / zero velocity. The
    /// robot is lifted so its lowest vertex clears the highest terrain or
    /// obstacle top under its footprint (sampled at quarter-metre resolution
    /// so a sine crest or box edge between vertices can't wedge the spawn).
    pub fn reset(&mut self) {
        let w = self.morph.w as usize;
        let samples = (w + 1) * 4 + 1;
        let lift = (0..samples)
            .map(|i| {
                let x = i as f32 * 0.25;
                let mut surf = self.ground.height(x);
                for o in &self.obstacles {
                    if x >= o.x_min && x <= o.x_max {
                        surf = surf.max(o.y_max);
                    }
                }
                surf
            })
            .fold(0.0_f32, f32::max)
            + self.spawn_y;
        for (i, m) in self.masses.iter_mut().enumerate() {
            let vx = (i % (w + 1)) as f32;
            let vy = (i / (w + 1)) as f32;
            m.pos = Vec2::new(vx, vy + lift);
            m.vel = Vec2::ZERO;
            m.force = Vec2::ZERO;
        }
        self.nan_seen = false;
    }

    /// Centre-of-mass position over the robot's *active* masses only.
    pub fn com(&self) -> Vec2 {
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut n = 0.0;
        for (i, m) in self.masses.iter().enumerate() {
            if self.active[i] {
                sx += m.pos.x;
                sy += m.pos.y;
                n += 1.0;
            }
        }
        if n > 0.0 { Vec2::new(sx / n, sy / n) } else { Vec2::ZERO }
    }
}

/// Zero every mass's force accumulator.
fn zero_forces(masses: &mut [PointMass]) {
    for m in masses.iter_mut() { m.force = Vec2::ZERO; }
}

/// Apply gravity (per-mass).
fn apply_gravity(masses: &mut [PointMass]) {
    for m in masses.iter_mut() {
        m.force.x += GRAVITY.0 * m.mass;
        m.force.y += GRAVITY.1 * m.mass;
    }
}

/// Apply Hookean spring forces + axial damping. `actuator_offsets[i]` is
/// summed into the rest length of every spring that actuator `i` modulates.
fn apply_springs(masses: &mut [PointMass], springs: &[Spring], actuator_offsets: &[f32]) {
    for s in springs {
        let (a, b) = (s.a, s.b);
        let pa = masses[a].pos;
        let pb = masses[b].pos;
        let va = masses[a].vel;
        let vb = masses[b].vel;
        let dx = pb.x - pa.x;
        let dy = pb.y - pa.y;
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        let dirx = dx / len;
        let diry = dy / len;

        let mut l_eff = s.base_len;
        for &i in &s.actuator_idx {
            if let Some(&d) = actuator_offsets.get(i) { l_eff += d; }
        }

        let stretch = len - l_eff;
        let mut fmag = s.k * stretch;
        fmag = fmag.clamp(-F_MAX, F_MAX);

        let vrel_along = (vb.x - va.x) * dirx + (vb.y - va.y) * diry;
        let fdamp = SPRING_DAMP * vrel_along;

        let total = fmag + fdamp;
        masses[a].force.x += total * dirx;
        masses[a].force.y += total * diry;
        masses[b].force.x -= total * dirx;
        masses[b].force.y -= total * diry;
    }
}

/// Positional minimum-length constraint: any spring shorter than
/// `MIN_SPRING_RATIO × base` is eased back apart (split between the two
/// masses) and its approach velocity along the axis is removed.
fn solve_min_length(masses: &mut [PointMass], springs: &[Spring]) {
    for s in springs {
        let (a, b) = (s.a, s.b);
        let dx = masses[b].pos.x - masses[a].pos.x;
        let dy = masses[b].pos.y - masses[a].pos.y;
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        let min_len = s.base_len * MIN_SPRING_RATIO;
        if len >= min_len { continue; }
        let ux = dx / len;
        let uy = dy / len;
        let push = ((min_len - len) * 0.5).min(MAX_POS_CORRECT);
        masses[a].pos.x -= push * ux;
        masses[a].pos.y -= push * uy;
        masses[b].pos.x += push * ux;
        masses[b].pos.y += push * uy;
        let vrel = (masses[b].vel.x - masses[a].vel.x) * ux
            + (masses[b].vel.y - masses[a].vel.y) * uy;
        if vrel < 0.0 {
            masses[a].vel.x += vrel * 0.5 * ux;
            masses[a].vel.y += vrel * 0.5 * uy;
            masses[b].vel.x -= vrel * 0.5 * ux;
            masses[b].vel.y -= vrel * 0.5 * uy;
        }
    }
}

/// Per-substep cap for the inversion-unfold correction.
const UNFOLD_STEP: f32 = 0.1;

/// Positional unfold for inverted (bowtie) cells. A corner whose interior
/// angle has passed 180° (a *reflex* corner — one that crossed the line
/// between its two neighbours) is eased back to the proper side a little
/// each substep. Spring forces alone can sit in equilibrium with a cell
/// flipped inside-out; this gentle persistent correction unwinds it without
/// any force spike.
fn solve_inversions(masses: &mut [PointMass], cells: &[[usize; 4]]) {
    for c in cells {
        let p = [
            masses[c[0]].pos,
            masses[c[1]].pos,
            masses[c[2]].pos,
            masses[c[3]].pos,
        ];
        for k in 0..4 {
            let prev = p[(k + 3) % 4];
            let next = p[(k + 1) % 4];
            let cur = p[k];
            let ax = next.x - prev.x;
            let ay = next.y - prev.y;
            let alen = (ax * ax + ay * ay).sqrt().max(1e-6);
            // For a properly-wound (CCW) corner this cross product is
            // negative; positive ⇒ the corner folded across prev→next.
            let cross = ax * (cur.y - prev.y) - ay * (cur.x - prev.x);
            if cross <= 0.0 { continue; }
            let depth = cross / alen;
            let step = ((depth + 0.02) * 0.5).min(UNFOLD_STEP);
            // Move against the cross-product gradient (direction (ay, -ax)),
            // with the reaction split across the two neighbours so the
            // correction stays momentum-neutral.
            let nx = ay / alen;
            let ny = -ax / alen;
            masses[c[k]].pos.x += nx * step;
            masses[c[k]].pos.y += ny * step;
            for &nb in &[c[(k + 3) % 4], c[(k + 1) % 4]] {
                masses[nb].pos.x -= nx * step * 0.5;
                masses[nb].pos.y -= ny * step * 0.5;
            }
        }
    }
}

/// If point `p` is strictly inside the CCW quad, return `(depth, inward_normal,
/// edge_index)` of the edge it is closest to — the minimal way back out.
pub(crate) fn point_quad_depth(p: Vec2, quad: &[Vec2; 4]) -> Option<(f32, (f32, f32), usize)> {
    let mut best = f32::MAX;
    let mut best_n = (0.0, 0.0);
    let mut best_e = 0;
    for e in 0..4 {
        let a = quad[e];
        let b = quad[(e + 1) % 4];
        let ex = b.x - a.x;
        let ey = b.y - a.y;
        let el = (ex * ex + ey * ey).sqrt().max(1e-9);
        // Interior of a CCW quad lies to the left of every edge.
        let (nx, ny) = (-ey / el, ex / el);
        let d = (p.x - a.x) * nx + (p.y - a.y) * ny;
        if d < 0.0 { return None; }
        if d < best { best = d; best_n = (nx, ny); best_e = e; }
    }
    Some((best, best_n, best_e))
}

/// Closest points between segments `p1→q1` and `p2→q2` (Ericson 5.1.9).
/// Returns parameters `s, t` and the two closest points.
fn closest_seg_seg(p1: Vec2, q1: Vec2, p2: Vec2, q2: Vec2) -> (f32, f32, Vec2, Vec2) {
    let d1 = Vec2::new(q1.x - p1.x, q1.y - p1.y);
    let d2 = Vec2::new(q2.x - p2.x, q2.y - p2.y);
    let r = Vec2::new(p1.x - p2.x, p1.y - p2.y);
    let a = d1.x * d1.x + d1.y * d1.y;
    let e = d2.x * d2.x + d2.y * d2.y;
    let f = d2.x * r.x + d2.y * r.y;
    let c = d1.x * r.x + d1.y * r.y;
    let b = d1.x * d2.x + d1.y * d2.y;
    let denom = a * e - b * b;
    let mut s = if denom > 1e-12 { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
    let mut t = if e > 1e-12 { (b * s + f) / e } else { 0.0 };
    if t < 0.0 {
        t = 0.0;
        s = if a > 1e-12 { (-c / a).clamp(0.0, 1.0) } else { 0.0 };
    } else if t > 1.0 {
        t = 1.0;
        s = if a > 1e-12 { ((b - c) / a).clamp(0.0, 1.0) } else { 0.0 };
    }
    let c1 = Vec2::new(p1.x + d1.x * s, p1.y + d1.y * s);
    let c2 = Vec2::new(p2.x + d2.x * s, p2.y + d2.y * s);
    (s, t, c1, c2)
}

/// Self-collision as surface blocking: two non-adjacent cells may press to
/// within `SELF_COLLIDE_SKIN` of each other but never interpenetrate —
/// vertices found inside the other quad are pushed out through the nearest
/// edge, and edge pairs closer than the skin are separated. All corrections
/// are positional (plus approach-velocity removal), so no energy is added.
fn solve_self_collision(masses: &mut [PointMass], cells: &[[usize; 4]], pairs: &[(usize, usize)]) {
    for _ in 0..2 {
        for &(ia, ib) in pairs {
            let qa = [masses[cells[ia][0]].pos, masses[cells[ia][1]].pos, masses[cells[ia][2]].pos, masses[cells[ia][3]].pos];
            let qb = [masses[cells[ib][0]].pos, masses[cells[ib][1]].pos, masses[cells[ib][2]].pos, masses[cells[ib][3]].pos];
            // AABB broadphase, inflated by the skin.
            let aabb = |q: &[Vec2; 4]| {
                let mut lo = Vec2::new(f32::MAX, f32::MAX);
                let mut hi = Vec2::new(f32::MIN, f32::MIN);
                for p in q {
                    lo.x = lo.x.min(p.x);
                    lo.y = lo.y.min(p.y);
                    hi.x = hi.x.max(p.x);
                    hi.y = hi.y.max(p.y);
                }
                (lo, hi)
            };
            let (alo, ahi) = aabb(&qa);
            let (blo, bhi) = aabb(&qb);
            let s = SELF_COLLIDE_SKIN;
            if alo.x > bhi.x + s || blo.x > ahi.x + s || alo.y > bhi.y + s || blo.y > ahi.y + s {
                continue;
            }
            // Vertices of each cell inside the other quad → push out through
            // the nearest edge. The correction is split between the vertex
            // and that edge's endpoints (barycentric, momentum-neutral):
            // a one-sided pushout would let a creature pump free momentum by
            // wedging a limb into its own body.
            for (cell_idx, quad) in [(ia, &qb), (ib, &qa)] {
                for &mi in &cells[cell_idx] {
                    let quad_idx = if cell_idx == ia { ib } else { ia };
                    let Some((depth, (nx, ny), e)) = point_quad_depth(masses[mi].pos, quad) else { continue; };
                    let (e1, e2) = (cells[quad_idx][e], cells[quad_idx][(e + 1) % 4]);
                    let a = masses[e1].pos;
                    let b = masses[e2].pos;
                    let ex = b.x - a.x;
                    let ey = b.y - a.y;
                    let el2 = (ex * ex + ey * ey).max(1e-9);
                    let t = (((masses[mi].pos.x - a.x) * ex + (masses[mi].pos.y - a.y) * ey) / el2).clamp(0.0, 1.0);
                    let w = [1.0, 1.0 - t, t]; // vertex, edge endpoints
                    let sumw = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
                    let corr = ((depth + SELF_COLLIDE_SKIN).min(MAX_POS_CORRECT)) / sumw;
                    masses[mi].pos.x -= nx * corr * w[0];
                    masses[mi].pos.y -= ny * corr * w[0];
                    masses[e1].pos.x += nx * corr * w[1];
                    masses[e1].pos.y += ny * corr * w[1];
                    masses[e2].pos.x += nx * corr * w[2];
                    masses[e2].pos.y += ny * corr * w[2];
                    let v_edge = Vec2::new(
                        masses[e1].vel.x * w[1] + masses[e2].vel.x * w[2],
                        masses[e1].vel.y * w[1] + masses[e2].vel.y * w[2],
                    );
                    // Relative approach along the inward normal → remove it,
                    // again split so total momentum is unchanged.
                    let vrel = (masses[mi].vel.x - v_edge.x) * nx + (masses[mi].vel.y - v_edge.y) * ny;
                    if vrel > 0.0 {
                        let j = vrel / sumw;
                        masses[mi].vel.x -= nx * j * w[0];
                        masses[mi].vel.y -= ny * j * w[0];
                        masses[e1].vel.x += nx * j * w[1];
                        masses[e1].vel.y += ny * j * w[1];
                        masses[e2].vel.x += nx * j * w[2];
                        masses[e2].vel.y += ny * j * w[2];
                    }
                }
            }
            // Edge pairs closer than the skin → separate along the gap axis.
            for e1 in 0..4 {
                for e2 in 0..4 {
                    let (a1, a2) = (cells[ia][e1], cells[ia][(e1 + 1) % 4]);
                    let (b1, b2) = (cells[ib][e2], cells[ib][(e2 + 1) % 4]);
                    let (sp, tp, c1, c2) = closest_seg_seg(
                        masses[a1].pos, masses[a2].pos, masses[b1].pos, masses[b2].pos,
                    );
                    let dx = c2.x - c1.x;
                    let dy = c2.y - c1.y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist >= SELF_COLLIDE_SKIN { continue; }
                    let (ux, uy) = if dist > 1e-6 {
                        (dx / dist, dy / dist)
                    } else {
                        let ex = masses[a2].pos.x - masses[a1].pos.x;
                        let ey = masses[a2].pos.y - masses[a1].pos.y;
                        let el = (ex * ex + ey * ey).sqrt().max(1e-9);
                        (ey / el, -ex / el)
                    };
                    let wa = [1.0 - sp, sp];
                    let wb = [1.0 - tp, tp];
                    let sumw = wa[0] * wa[0] + wa[1] * wa[1] + wb[0] * wb[0] + wb[1] * wb[1];
                    if sumw < 1e-9 { continue; }
                    let corr = ((SELF_COLLIDE_SKIN - dist).min(MAX_POS_CORRECT)) / sumw;
                    for (mi, w, sign) in [(a1, wa[0], -1.0), (a2, wa[1], -1.0), (b1, wb[0], 1.0), (b2, wb[1], 1.0)] {
                        masses[mi].pos.x += sign * ux * corr * w;
                        masses[mi].pos.y += sign * uy * corr * w;
                    }
                    let vpa = Vec2::new(
                        masses[a1].vel.x * wa[0] + masses[a2].vel.x * wa[1],
                        masses[a1].vel.y * wa[0] + masses[a2].vel.y * wa[1],
                    );
                    let vpb = Vec2::new(
                        masses[b1].vel.x * wb[0] + masses[b2].vel.x * wb[1],
                        masses[b1].vel.y * wb[0] + masses[b2].vel.y * wb[1],
                    );
                    let vrel = (vpb.x - vpa.x) * ux + (vpb.y - vpa.y) * uy;
                    if vrel < 0.0 {
                        let imp = vrel / sumw;
                        for (mi, w, sign) in [(a1, wa[0], 1.0), (a2, wa[1], 1.0), (b1, wb[0], -1.0), (b2, wb[1], -1.0)] {
                            masses[mi].vel.x += sign * imp * w * ux;
                            masses[mi].vel.y += sign * imp * w * uy;
                        }
                    }
                }
            }
        }
    }
}

/// Heightfield ground penalty + clamped Coulomb friction. Contact normals
/// come from `Ground::contact`; friction acts along the tangent. The normal
/// force uses a capped penetration so a deeply-tunnelled mass is steered
/// out without being launched.
fn apply_ground(masses: &mut [PointMass], ground: &Ground) {
    /// Max penetration depth converted into normal force (launch guard).
    const PEN_CAP: f32 = 0.5;
    for m in masses.iter_mut() {
        if let Some(((nx, ny), pen)) = ground.contact(m.pos.x, m.pos.y, m.vel.x, m.vel.y) {
            let n = (GROUND_K * pen.min(PEN_CAP)).min(N_MAX);
            let (tx, ty) = (ny, -nx);
            m.force.x += n * nx;
            m.force.y += n * ny;
            let vt = m.vel.x * tx + m.vel.y * ty;
            let max_t = GROUND_MU * n;
            let fric = (-vt * m.mass / DT).clamp(-max_t, max_t);
            m.force.x += fric * tx;
            m.force.y += fric * ty;
        }
    }
}

/// AABB penalty for every obstacle. A mass inside the box is pushed out
/// through its *nearest* face (minimal translation), with Coulomb friction
/// along that face.
fn apply_obstacles(masses: &mut [PointMass], obstacles: &[Obstacle]) {
    for o in obstacles {
        for m in masses.iter_mut() {
            // Bottom epsilon: a mass resting on the ground penetrates it by
            // a few millimetres, which would place it *below* the box base
            // and let it walk through the box's footprint at ground level.
            if m.pos.x <= o.x_min || m.pos.x >= o.x_max || m.pos.y <= o.y_min - 0.05 || m.pos.y >= o.y_max {
                continue;
            }
            let d_left = m.pos.x - o.x_min;
            let d_right = o.x_max - m.pos.x;
            let d_top = o.y_max - m.pos.y;
            // The bottom face is excluded: obstacle boxes sit on the ground,
            // so nothing can legitimately approach from below — and a mass
            // overlapping the box's buried base must not be pushed down
            // into the terrain.
            let min = d_left.min(d_right).min(d_top);
            let (nx, ny) = if min == d_top {
                (0.0, 1.0)
            } else if min == d_left {
                (-1.0, 0.0)
            } else {
                (1.0, 0.0)
            };
            let n = (OBSTACLE_K * min.min(0.5)).min(N_MAX);
            let (tx, ty) = (ny, -nx);
            m.force.x += n * nx;
            m.force.y += n * ny;
            let vt = m.vel.x * tx + m.vel.y * ty;
            let cap = OBSTACLE_MU * n;
            let fric = (-vt * m.mass / DT).clamp(-cap, cap);
            m.force.x += fric * tx;
            m.force.y += fric * ty;
        }
    }
}

/// Semi-implicit Euler with damping, velocity clamp, position clamp, and a
/// NaN scrub. Sets `nan_seen` if any mass ends up with non-finite values.
fn integrate_and_clamp(masses: &mut [PointMass], ground: &Ground, nan_seen: &mut bool) {
    for m in masses.iter_mut() {
        m.vel.x *= 1.0 - GLOBAL_DAMP;
        m.vel.y *= 1.0 - GLOBAL_DAMP;
        let ax = m.force.x / m.mass;
        let ay = m.force.y / m.mass;
        m.vel.x += ax * DT;
        m.vel.y += ay * DT;
        let vmag2 = m.vel.x * m.vel.x + m.vel.y * m.vel.y;
        if vmag2 > V_MAX * V_MAX {
            let s = V_MAX / vmag2.sqrt();
            m.vel.x *= s;
            m.vel.y *= s;
        }
        m.pos.x += m.vel.x * DT;
        m.pos.y += m.vel.y * DT;
        // Anti-tunnel floor clamp — only where the ground resolves to an
        // up-facing contact. Inside a stair column (side contact) clamping y
        // would teleport the mass up toward the tread.
        let up_facing = ground
            .contact(m.pos.x, m.pos.y, m.vel.x, m.vel.y)
            .map(|((_, ny), _)| ny > 0.5)
            .unwrap_or(false);
        if up_facing {
            let y_min = ground.height(m.pos.x) - 0.5;
            if m.pos.y < y_min { m.pos.y = y_min; if m.vel.y < 0.0 { m.vel.y = 0.0; } }
        }
        if m.pos.x > X_BOUND { m.pos.x = X_BOUND; if m.vel.x > 0.0 { m.vel.x = 0.0; } }
        if m.pos.x < -X_BOUND { m.pos.x = -X_BOUND; if m.vel.x < 0.0 { m.vel.x = 0.0; } }
        if !m.pos.x.is_finite() || !m.pos.y.is_finite() || !m.vel.x.is_finite() || !m.vel.y.is_finite() {
            m.pos = Vec2::ZERO;
            m.vel = Vec2::ZERO;
            *nan_seen = true;
        }
    }
}

impl World {
    /// Run one substep with the given per-actuator ΔL offsets. Each entry
    /// `offsets[i]` is the rest-length offset contributed by actuator `i` to
    /// each spring it modulates; the simulator sums contributions across all
    /// actuators attached to a given spring.
    pub fn substep(&mut self, actuator_offsets: &[f32]) {
        zero_forces(&mut self.masses);
        apply_gravity(&mut self.masses);
        apply_springs(&mut self.masses, &self.springs, actuator_offsets);
        apply_ground(&mut self.masses, &self.ground);
        apply_obstacles(&mut self.masses, &self.obstacles);
        integrate_and_clamp(&mut self.masses, &self.ground, &mut self.nan_seen);
        solve_min_length(&mut self.masses, &self.springs);
        solve_inversions(&mut self.masses, &self.cell_verts);
        solve_self_collision(&mut self.masses, &self.cell_verts, &self.cell_pairs);
    }

    /// Convenience: simulate for `secs` seconds with the given per-actuator
    /// ΔL offset callback. Returns the final COM x.
    pub fn simulate<F: FnMut(usize, f32) -> f32>(
        &mut self,
        secs: f32,
        mut offset_fn: F,
    ) -> Vec2 {
        let n_act = self.morph.actuators().len();
        let mut offsets = vec![0.0_f32; n_act];
        let frames = (secs * 60.0).round() as i32;
        let mut t = 0.0_f32;
        for _ in 0..frames {
            for (i, o) in offsets.iter_mut().enumerate() {
                *o = offset_fn(i, t);
            }
            for _ in 0..SUBSTEPS_PER_FRAME {
                self.substep(&offsets);
                t += DT;
            }
        }
        self.com()
    }
}

#[cfg(test)]
#[path = "physics_tests.rs"]
mod tests;