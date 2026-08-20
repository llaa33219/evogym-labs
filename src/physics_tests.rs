//! Physics tests. Split into a sibling file so the production module stays
//! under the 300-LOC soft ceiling.

use super::*;
use crate::voxel::Cell;

fn soft_3x3() -> Morphology {
    let mut m = Morphology::new(3, 3);
    for cy in 0..3 { for cx in 0..3 { m.set(cx, cy, Cell::Soft); } }
    m.set(1, 1, Cell::ActV);
    m
}

#[test]
fn soft_block_drops_and_settles_without_nan() {
    let m = soft_3x3();
    let mut world = World::new(m);
    let com0 = world.com();
    let com1 = world.simulate(1.0, |_, _| 0.0);
    assert!(com1.x.is_finite() && com1.y.is_finite(), "NaN after drop");
    assert!(com1.y < com0.y, "robot did not fall");
    assert!(!world.nan_seen);
}

#[test]
fn actuator_offsets_modulate_spring_length() {
    let mut m = Morphology::new(2, 2);
    m.set(0, 0, Cell::Soft);
    m.set(1, 0, Cell::ActH);
    m.set(0, 1, Cell::Soft);
    m.set(1, 1, Cell::Soft);
    let w = World::new(m);
    assert_eq!(w.morph.actuators().len(), 1);
    let mut found = false;
    for s in &w.springs {
        if !s.is_diag && s.actuator_idx.contains(&0) { found = true; break; }
    }
    assert!(found, "no edge spring wired to actuator");
}

#[test]
fn obstacles_can_be_added() {
    let mut m = Morphology::new(2, 1);
    m.set(0, 0, Cell::Soft);
    m.set(1, 0, Cell::ActH);
    let world = World::new(m).with_obstacles(vec![Obstacle::new(0.5, 1.5, 0.0, 0.5)]);
    assert_eq!(world.obstacles.len(), 1);
}

#[test]
fn obstacle_blocks_from_above_and_from_the_side() {
    // Drop a soft block onto a tall box: it must land on top, not fall through.
    let mut m = Morphology::new(2, 2);
    for cy in 0..2 { for cx in 0..2 { m.set(cx, cy, Cell::Soft); } }
    let mut world = World::new(m).with_obstacles(vec![Obstacle::new(0.0, 2.0, 0.0, 1.5)]);
    world.reset();
    let offs = vec![0.0; world.morph.actuators().len()];
    for _ in 0..240 {
        for _ in 0..SUBSTEPS_PER_FRAME { world.substep(&offs); }
    }
    assert!(!world.nan_seen);
    assert!(world.com().y > 1.5, "block fell through the box (com y = {})", world.com().y);

    // Ram a block sideways into a box: climbing over is fine, but no mass
    // may ever be found deep inside the box interior (that is tunnelling).
    let mut m2 = Morphology::new(2, 2);
    for cy in 0..2 { for cx in 0..2 { m2.set(cx, cy, Cell::Soft); } }
    let mut world2 = World::new(m2).with_obstacles(vec![Obstacle::new(6.0, 7.0, 0.0, 2.0)]);
    world2.reset();
    let offs2 = vec![0.0; world2.morph.actuators().len()];
    for _ in 0..120 {
        for mass in &mut world2.masses { mass.vel.x = mass.vel.x.max(3.0); }
        for _ in 0..SUBSTEPS_PER_FRAME { world2.substep(&offs2); }
        for mass in &world2.masses {
            let deep_x = mass.pos.x > 6.3 && mass.pos.x < 6.7;
            let deep_y = mass.pos.y < 1.5;
            assert!(!(deep_x && deep_y), "mass tunnelled into the box at ({}, {})", mass.pos.x, mass.pos.y);
        }
    }
    assert!(!world2.nan_seen);
}

#[test]
fn velocity_clamp_prevents_unbounded_growth() {
    let mut m = Morphology::new(1, 8);
    for cy in 0..8 { m.set(0, cy, Cell::Soft); }
    m.set(0, 0, Cell::ActV);
    let mut world = World::new(m);
    let com = world.simulate(2.0, |_, _| 0.0);
    assert!(com.x.is_finite() && com.y.is_finite());
    assert!(!world.nan_seen);
}

#[test]
fn stair_riser_side_hit_does_not_launch() {
    // A point deep inside a step column must exit through the nearest face
    // (horizontal for a riser hit), never straight up.
    let g = Ground::Stairs { run: 5.0, rise: 2.0 };
    // Inside the second step column (x in 5..10, tread top at y=2), near the riser.
    let ((nx, _), _) = g.contact(5.1, 1.5, 5.0, 0.0).unwrap();
    assert_eq!(nx, -1.0, "riser hit must push horizontally");
    // On the tread surface: pushed up.
    let ((nx, ny), _) = g.contact(7.5, 1.9, 0.0, -1.0).unwrap();
    assert_eq!((nx, ny), (0.0, 1.0), "tread contact must push up");
    // Above the surface: no contact.
    assert!(g.contact(7.5, 2.5, 0.0, 0.0).is_none());
    // Corner zone: sliding in horizontally → pushed back out of the riser;
    // falling onto the corner → lands on the tread.
    let ((nx, _), _) = g.contact(5.05, 1.95, 3.0, 0.0).unwrap();
    assert_eq!(nx, -1.0, "horizontal arrival at the corner must not pop up");
    let ((nx, ny), _) = g.contact(5.05, 1.95, 0.0, -3.0).unwrap();
    assert_eq!((nx, ny), (0.0, 1.0), "falling onto the corner lands on the tread");

    // Dynamic check: ram a robot into the first riser at high speed; its COM
    // must stay below the tread top + a small bounce allowance.
    let mut m = Morphology::new(3, 2);
    for cy in 0..2 { for cx in 0..3 { m.set(cx, cy, Cell::Soft); } }
    let mut world = World::new(m).with_terrain(g, Vec::new());
    world.reset();
    for mass in &mut world.masses { mass.vel.x = 30.0; }
    let mut max_y = f32::NEG_INFINITY;
    let offs = vec![0.0; world.morph.actuators().len()];
    for _ in 0..240 {
        for _ in 0..SUBSTEPS_PER_FRAME {
            world.substep(&offs);
        }
        max_y = max_y.max(world.com().y);
    }
    assert!(!world.nan_seen);
    assert!(max_y < 6.0, "riser hit launched the robot (peak COM y = {max_y})");
}
fn cell_signed_area(world: &World, cx: usize, cy: usize) -> f32 {
    let w = world.morph.w as usize;
    let vert = |vx: usize, vy: usize| vy * (w + 1) + vx;
    let idx = [vert(cx, cy), vert(cx + 1, cy), vert(cx + 1, cy + 1), vert(cx, cy + 1)];
    let p: Vec<Vec2> = idx.iter().map(|&i| world.masses[i].pos).collect();
    let mut cross = 0.0;
    for e in 0..4 {
        let (a, b) = (p[e], p[(e + 1) % 4]);
        cross += a.x * b.y - b.x * a.y;
    }
    0.5 * cross
}

#[test]
fn crumpled_cells_reopen_and_overlaps_separate() {
    // Heavily compress a 3x2 soft block (rows squeezed to 40% height), then
    // let the sim run: cells must reopen and no two masses may stay
    // interpenetrating.
    let mut m = Morphology::new(3, 2);
    for cy in 0..2 { for cx in 0..3 { m.set(cx, cy, Cell::Soft); } }
    let mut world = World::new(m);
    world.reset();
    let w = world.morph.w as usize;
    for i in 0..world.masses.len() {
        let vy = (i / (w + 1)) as f32;
        world.masses[i].pos.y = 0.05 + vy * 0.4; // squeeze
    }
    let offs = vec![0.0; world.morph.actuators().len()];
    for _ in 0..90 {
        for _ in 0..SUBSTEPS_PER_FRAME { world.substep(&offs); }
    }
    assert!(!world.nan_seen);
    for cy in 0..2 {
        for cx in 0..3 {
            let a = cell_signed_area(&world, cx, cy);
            assert!(a > 0.35, "cell ({cx},{cy}) stayed crumpled (area {a})");
        }
    }
    // Self-collision acts as a near-contact surface: track the deepest
    // vertex-in-quad penetration among non-adjacent cells *during* the run —
    // parts may come almost into contact but never overlap meaningfully.
    let mut max_pen = 0.0f32;
    for _ in 0..60 {
        for _ in 0..SUBSTEPS_PER_FRAME { world.substep(&offs); }
        for &(ia, ib) in &world.cell_pairs {
            for &(from, into) in &[(ia, ib), (ib, ia)] {
                let quad = [
                    world.masses[world.cell_verts[into][0]].pos,
                    world.masses[world.cell_verts[into][1]].pos,
                    world.masses[world.cell_verts[into][2]].pos,
                    world.masses[world.cell_verts[into][3]].pos,
                ];
                for k in 0..4 {
                    let p = world.masses[world.cell_verts[from][k]].pos;
                    if let Some((depth, _, _)) = point_quad_depth(p, &quad) {
                        max_pen = max_pen.max(depth);
                    }
                }
            }
        }
    }
    assert!(max_pen < 0.3, "body parts interpenetrated (deepest vertex {max_pen} m inside another cell)");
}

#[test]
fn self_collision_cannot_propel_the_body() {
    // Fold the body onto itself so self-collision fires continuously; with
    // momentum-neutral corrections the COM must not drift away — a one-sided
    // pushout would pump the body sideways for free.
    let mut m = Morphology::new(3, 2);
    for cy in 0..2 { for cx in 0..3 { m.set(cx, cy, Cell::Soft); } }
    let mut world = World::new(m);
    world.reset();
    let w = world.morph.w as usize;
    for i in 0..world.masses.len() {
        let vy = (i / (w + 1)) as f32;
        world.masses[i].pos.y = 0.05 + vy * 0.4;
    }
    let start_x = world.com().x;
    let offs = vec![0.0; world.morph.actuators().len()];
    for _ in 0..150 {
        for _ in 0..SUBSTEPS_PER_FRAME { world.substep(&offs); }
    }
    assert!(!world.nan_seen);
    let drift = (world.com().x - start_x).abs();
    assert!(drift < 0.5, "self-collision propelled the body by {drift} m");
}

#[test]
fn large_32x32_grid_simulates() {
    let mut m = Morphology::new(32, 32);
    for cy in 10..14 {
        for cx in 10..14 { m.set(cx, cy, Cell::Soft); }
    }
    m.set(11, 11, Cell::ActH);
    let mut world = World::new(m);
    world.reset();
    let offs = vec![0.0; world.morph.actuators().len()];
    for _ in 0..30 {
        for _ in 0..SUBSTEPS_PER_FRAME { world.substep(&offs); }
    }
    assert!(!world.nan_seen);
    assert!(world.com().y.is_finite());
}

#[test]
fn inverted_cell_unwinds() {
    // Flip a cell inside-out by swapping two opposite corners, then run:
    // every cell must recover a healthy positive area.
    let mut m = Morphology::new(2, 2);
    for cy in 0..2 { for cx in 0..2 { m.set(cx, cy, Cell::Soft); } }
    let mut world = World::new(m);
    world.reset();
    // Cell (0,0) corners: bl=(0,0) index 0, tr=(1,1) index w+2 = 4.
    let (p0, p4) = (world.masses[0].pos, world.masses[4].pos);
    world.masses[0].pos = p4;
    world.masses[4].pos = p0;
    let offs = vec![0.0; world.morph.actuators().len()];
    for _ in 0..120 {
        for _ in 0..SUBSTEPS_PER_FRAME { world.substep(&offs); }
    }
    assert!(!world.nan_seen);
    for cy in 0..2 {
        for cx in 0..2 {
            let a = cell_signed_area(&world, cx, cy);
            assert!(a > 0.2, "cell ({cx},{cy}) stayed inverted (area {a})");
        }
    }
}

#[test]
fn violent_actuation_never_launches() {
    // Every cell is an actuator driven at max amplitude with mismatched
    // phases — the worst case for internal stress. No mass may ever reach
    // anywhere near the 50 m/s velocity clamp (i.e. no launch), and no
    // spring may stay crushed below its positional minimum.
    let mut m = Morphology::new(3, 3);
    for cy in 0..3 { for cx in 0..3 { m.set(cx, cy, Cell::ActH); } }
    let mut world = World::new(m);
    world.reset();
    let n = world.morph.actuators().len();
    let mut max_v = 0.0f32;
    let mut t = 0.0f32;
    for _ in 0..240 {
        let offs: Vec<f32> = (0..n)
            .map(|i| 0.3 * (t * 8.0 + i as f32 * 1.7).sin())
            .collect();
        for _ in 0..SUBSTEPS_PER_FRAME {
            world.substep(&offs);
            t += DT;
        }
        for mass in &world.masses {
            max_v = max_v.max(mass.vel.x.hypot(mass.vel.y));
        }
    }
    assert!(!world.nan_seen);
    assert!(max_v < 20.0, "launch detected: peak velocity {max_v} m/s");
    for s in &world.springs {
        let a = world.masses[s.a].pos;
        let b = world.masses[s.b].pos;
        let len = (b.x - a.x).hypot(b.y - a.y);
        assert!(len > s.base_len * 0.35, "spring stayed crushed ({len} of {})", s.base_len);
    }
}

#[test]
fn stair_corner_slide_does_not_pop_upward() {
    // A robot pressed sideways along the lower tread into the first riser:
    // no mass may jump upward by a large step within a single frame.
    let g = Ground::Stairs { run: 5.0, rise: 2.0 };
    let mut m = Morphology::new(2, 2);
    for cy in 0..2 { for cx in 0..2 { m.set(cx, cy, Cell::Soft); } }
    let mut world = World::new(m).with_terrain(g, Vec::new());
    world.reset();
    let offs = vec![0.0; world.morph.actuators().len()];
    let mut max_jump = 0.0f32;
    let mut prev_ys: Vec<f32> = world.masses.iter().map(|m| m.pos.y).collect();
    for frame in 0..480 {
        for mass in &mut world.masses { mass.vel.x = mass.vel.x.max(2.0); }
        for _ in 0..SUBSTEPS_PER_FRAME { world.substep(&offs); }
        for (i, mass) in world.masses.iter().enumerate() {
            let jump = mass.pos.y - prev_ys[i];
            if frame > 60 { max_jump = max_jump.max(jump); }
            prev_ys[i] = mass.pos.y;
        }
    }
    assert!(max_jump < 0.5, "corner popped a mass upward by {max_jump} m in one frame");
}
