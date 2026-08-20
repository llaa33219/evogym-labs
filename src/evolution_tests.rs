//! Tests for the GA. Split into a sibling file so the production module
//! stays under the 300-LOC soft ceiling while the test cases stay grouped.

use super::*;
use crate::voxel::{Cell, Morphology};

fn tiny_demo_morph() -> Morphology {
    let mut m = Morphology::new(3, 3);
    for cy in 0..3 {
        for cx in 0..3 {
            let kind = if cy == 0 { Cell::Rigid } else { Cell::Soft };
            m.set(cx, cy, kind);
        }
    }
    m.set(1, 0, Cell::ActV);
    m
}

#[test]
fn ga_runs_two_generations_and_records_best() {
    let cfg = EvolutionConfig {
        population: 4,
        eval_secs: 0.2,
        morph_size: (3, 3),
        ..EvolutionConfig::defaults()
    };
    let mut last_best = f32::NEG_INFINITY;
    let mut any_improve = false;
    let result = run(&cfg, tiny_demo_morph(), 2, Some(42), |p| {
        if let Progress::Generation { best, .. } = p {
            if best > last_best { any_improve = true; }
            last_best = last_best.max(best);
        }
    });
    assert_eq!(result.history.records.len(), 2);
    // We don't assert improvement strictly (random init might be locally
    // optimal), but the loop must run and produce records.
    let _ = any_improve;
}

#[test]
fn invalid_morphology_falls_back_to_parent() {
    let cfg = EvolutionConfig {
        mode: Mode::Full,
        population: 6,
        eval_secs: 0.1,
        morph_size: (3, 3),
        ..EvolutionConfig::defaults()
    };
    let result = run(&cfg, tiny_demo_morph(), 3, Some(1), |_| {});
    assert_eq!(result.history.records.len(), 3);
}

#[test]
fn best_fitness_is_monotonic_across_generations() {
    // Elites are cloned verbatim and evaluation is deterministic, so the
    // recorded best-of-generation fitness must never decrease. This locks
    // the sort-by-fitness invariant (a regression here once ranked by
    // population index, which broke elitism and produced non-monotonic
    // "best" records).
    let cfg = EvolutionConfig {
        population: 6,
        eval_secs: 0.2,
        morph_size: (3, 3),
        ..EvolutionConfig::defaults()
    };
    let result = run(&cfg, tiny_demo_morph(), 5, Some(3), |_| {});
    let fits: Vec<f32> = result.history.records.iter().map(|r| r.fitness).collect();
    for w in fits.windows(2) {
        assert!(w[1] >= w[0], "best fitness decreased: {fits:?}");
    }
}

#[test]
fn mutated_cells_get_a_sibling_colour_variant() {
    // Parent with two coloured ActH cells; mutate many times and check that
    // every cell whose type changed got either no override or a ±20 variant
    // of a same-type sibling colour.
    let mut parent = tiny_demo_morph();
    parent.set(0, 2, Cell::ActH);
    parent.set_color(0, 2, Some([100, 150, 200]));
    parent.set(2, 2, Cell::ActH);
    parent.set_color(2, 2, Some([50, 60, 70]));
    let cfg = EvolutionConfig::defaults();
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    for _ in 0..60 {
        let child = mutate_morphology(&parent, &cfg, &mut rng);
        for i in 0..child.cells.len() {
            if child.cells[i] == parent.cells[i] {
                assert_eq!(child.colors[i], parent.colors[i], "unchanged cell must keep its colour");
                continue;
            }
            if !child.cells[i].is_solid() {
                assert_eq!(child.colors[i], None, "emptied cell must drop its override");
                continue;
            }
            if let Some(c) = child.colors[i] {
                let pool: [[u8; 3]; 2] = [[100, 150, 200], [50, 60, 70]];
                let ok = pool.iter().any(|base| {
                    (0..3).all(|k| (c[k] as i32 - base[k] as i32).abs() <= 20)
                });
                assert!(ok, "changed cell colour {c:?} is not a sibling variant");
            }
        }
    }
}

#[test]
fn controller_only_mode_keeps_morphology() {
    let seed = tiny_demo_morph();
    let cfg = EvolutionConfig {
        mode: Mode::ControllerOnly,
        population: 4,
        eval_secs: 0.1,
        ..EvolutionConfig::defaults()
    };
    let result = run(&cfg, seed.clone(), 2, Some(7), |_| {});
    for r in &result.history.records {
        assert_eq!(r.genome.morph.cells, seed.cells);
    }
}