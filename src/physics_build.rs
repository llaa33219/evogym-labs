//! World-construction helpers extracted into a sibling file so the
//! integration step in `physics.rs` stays small. Public so `pub(crate)`
//! callers (i.e. `World::new` in the parent module) can use them; the
//! `#[allow(dead_code)]` on the parent module's `mod build;` keeps rustc
//! quiet for the test build.

use crate::physics::{K_RIGID_DIAG, K_RIGID_EDGE, K_SOFT_DIAG, K_SOFT_EDGE, MASS, PointMass, Spring, Vec2};
use crate::voxel::{Cell, Morphology, Stiffness};

#[derive(Debug)]
pub(crate) struct PendingSpring {
    pub a: usize,
    pub b: usize,
    pub k: f32,
    pub base_len: f32,
    pub is_diag: bool,
    pub actuator_wire: Vec<usize>,
}

/// Per-cell global actuator index, in the controller's address space.
pub(crate) fn cell_actuator_indices(cells: &[Cell]) -> Vec<Option<usize>> {
    let mut out = Vec::with_capacity(cells.len());
    let mut n = 0usize;
    for &c in cells {
        if c.is_actuator() {
            out.push(Some(n));
            n += 1;
        } else {
            out.push(None);
        }
    }
    out
}

/// Build the initial point-mass array. Vertices are placed at world
/// `(vx, vy)`; `spawn_y` is added to the y so the bottom row rests on the
/// ground by compression rather than by spawning inside it.
pub(crate) fn build_masses(w: u8, h: u8, spawn_y: f32) -> Vec<PointMass> {
    let n_v = ((w as usize) + 1) * ((h as usize) + 1);
    (0..n_v).map(|i| {
        let vx = (i % (w as usize + 1)) as f32;
        let vy = (i / (w as usize + 1)) as f32;
        let mut pm = PointMass::fresh(Vec2::new(vx, vy + spawn_y));
        pm.mass = MASS;
        pm
    }).collect()
}

/// For each cell, push its 6 springs (4 edges + 2 diagonals) and wire up
/// the actuator wiring if the cell is `ActH` / `ActV`. Adjacent cells share
/// the same edge spring — we defer dedupe to the next step.
pub(crate) fn collect_cell_springs(morph: &Morphology, actuator_at: &[Option<usize>]) -> Vec<PendingSpring> {
    let w = morph.w as usize;
    let h = morph.h as usize;
    let vert_idx = |vx: usize, vy: usize| vy * (w + 1) + vx;
    let mut raw: Vec<PendingSpring> = Vec::new();
    for cy in 0..h {
        for cx in 0..w {
            let c = morph.cells[cy * w + cx];
            let Some(stiff) = c.stiffness_kind() else { continue; };
            let (k_e, k_d) = match stiff {
                Stiffness::Rigid => (K_RIGID_EDGE, K_RIGID_DIAG),
                Stiffness::Soft => (K_SOFT_EDGE, K_SOFT_DIAG),
            };
            let bl = vert_idx(cx, cy);
            let br = vert_idx(cx + 1, cy);
            let tl = vert_idx(cx, cy + 1);
            let tr = vert_idx(cx + 1, cy + 1);
            raw.push(PendingSpring { a: bl, b: br, k: k_e, base_len: 1.0, is_diag: false, actuator_wire: Vec::new() });
            raw.push(PendingSpring { a: tl, b: tr, k: k_e, base_len: 1.0, is_diag: false, actuator_wire: Vec::new() });
            raw.push(PendingSpring { a: bl, b: tl, k: k_e, base_len: 1.0, is_diag: false, actuator_wire: Vec::new() });
            raw.push(PendingSpring { a: br, b: tr, k: k_e, base_len: 1.0, is_diag: false, actuator_wire: Vec::new() });
            raw.push(PendingSpring { a: bl, b: tr, k: k_d, base_len: std::f32::consts::SQRT_2, is_diag: true, actuator_wire: Vec::new() });
            raw.push(PendingSpring { a: br, b: tl, k: k_d, base_len: std::f32::consts::SQRT_2, is_diag: true, actuator_wire: Vec::new() });

            if let Some(act_idx) = actuator_at[cy * w + cx] {
                let horiz = matches!(c, Cell::ActH);
                let vert = matches!(c, Cell::ActV);
                if horiz || vert {
                    for ps in raw.iter_mut().rev().take(6) {
                        if ps.is_diag { continue; }
                        let k_match = if horiz {
                            (ps.a == bl && ps.b == br)
                                || (ps.a == tl && ps.b == tr)
                                || (ps.a == br && ps.b == bl)
                                || (ps.a == tr && ps.b == tl)
                        } else {
                            (ps.a == bl && ps.b == tl)
                                || (ps.a == br && ps.b == tr)
                                || (ps.a == tl && ps.b == bl)
                                || (ps.a == tr && ps.b == br)
                        };
                        if k_match { ps.actuator_wire.push(act_idx); }
                    }
                }
            }
        }
    }
    raw
}

/// Self-collision data: every solid cell's four corner mass indices
/// (bl, br, tr, tl — CCW), plus the list of cell pairs that share no vertex
/// (adjacent cells share corners and must not collide with each other).
pub(crate) fn cell_collision_data(morph: &Morphology) -> (Vec<[usize; 4]>, Vec<(usize, usize)>) {
    let w = morph.w as usize;
    let vert = |vx: usize, vy: usize| vy * (w + 1) + vx;
    let mut cells: Vec<[usize; 4]> = Vec::new();
    for cy in 0..morph.h as usize {
        for cx in 0..w {
            if morph.cells[cy * w + cx].is_solid() {
                cells.push([vert(cx, cy), vert(cx + 1, cy), vert(cx + 1, cy + 1), vert(cx, cy + 1)]);
            }
        }
    }
    let mut pairs = Vec::new();
    for a in 0..cells.len() {
        for b in (a + 1)..cells.len() {
            let shares = cells[a].iter().any(|v| cells[b].contains(v));
            if !shares { pairs.push((a, b)); }
        }
    }
    (cells, pairs)
}

/// Collapse shared edge springs (and shared diagonals separately) into one
/// entry per `(min_a, max_b, is_diag)` triple. Stiffness: Rigid wins.
pub(crate) fn dedupe_springs(raw: Vec<PendingSpring>) -> Vec<Spring> {
    let mut raw = raw;
    raw.sort_by_key(|p| (p.a.min(p.b), p.a.max(p.b), p.is_diag));
    let mut springs: Vec<Spring> = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let key_a = raw[i].a.min(raw[i].b);
        let key_b = raw[i].a.max(raw[i].b);
        let diag = raw[i].is_diag;
        let mut k_best = raw[i].k;
        let mut wire: Vec<usize> = raw[i].actuator_wire.clone();
        let mut j = i + 1;
        while j < raw.len()
            && raw[j].a.min(raw[j].b) == key_a
            && raw[j].a.max(raw[j].b) == key_b
            && raw[j].is_diag == diag
        {
            if raw[j].k > k_best { k_best = raw[j].k; }
            wire.extend_from_slice(&raw[j].actuator_wire);
            j += 1;
        }
        wire.sort_unstable();
        wire.dedup();
        springs.push(Spring {
            a: raw[i].a.min(raw[i].b),
            b: raw[i].a.max(raw[i].b),
            base_len: raw[i].base_len,
            k: k_best,
            is_diag: diag,
            actuator_idx: wire,
        });
        i = j;
    }
    springs
}
