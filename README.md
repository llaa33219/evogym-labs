# EvoGym-Labs

A 2D soft-voxel robot evolution simulator inspired by EvoGym. Hand-rolled
point-mass / spring physics, an open-loop sinusoidal controller, a simple GA
with morphology + controller co-evolution, software-rasterized video
rendering, and both an eframe / egui GUI and a fully-headless CLI for
display-less verification.

```
cargo build --release
cargo test --release --all-features
./target/release/evogym-labs --headless --gens 3 --pop 8 --eval-secs 2 \
    --clip-secs 1 --out /tmp/opencode/evogym_test.mp4
ffprobe -show_streams /tmp/opencode/evogym_test.mp4
```

## Features

- **Voxel grid** — 5×5 by default, configurable up to 8×8. Five cell types:
  `Empty`, `Rigid`, `Soft`, `ActuatorHorizontal`, `ActuatorVertical`.
- **Hand-rolled 2-D physics** — point masses live on grid vertices shared
  between adjacent cells. Each non-empty cell adds 6 springs (4 edges + 2
  diagonals) among its corners. `ActH` / `ActV` modulate the rest length of
  their horizontal / vertical edge springs. Diagonals stay structural.
  Semi-implicit Euler at `dt = 1/240 s` with 4 substeps per 60 fps frame.
- **Open-loop sinusoidal controller** — global frequency (0.5..4 Hz) +
  per-actuator phase (0..2π) + per-actuator amplitude (0.05..0.30).
- **GA** — population ~32, tournament selection (size 3), elitism 2,
  morphology + controller co-evolution (or controller-only on a fixed
  design). Evaluation parallelised with `rayon`.
- **Tasks** — `Walker` (flat ground) and `ObstacleTraverser` (three low
  boxes to climb over). Fitness = horizontal COM displacement.
- **Headless video pipeline** — software rasterizer → 1080×1080 rawvideo
  rgb24 → `ffmpeg` per-clip `libx264 -crf 18` → `ffmpeg concat` re-encode
  into the final mp4. Renders run in parallel via `rayon`.
- **GUI** — `eframe` 0.32 / `egui` 0.32. Palette + paint grid + evolution
  config + editable video-plan table + RUN button. Worker thread +
  `std::sync::mpsc` progress channel.

## Headless mode

The binary runs without `eframe` / `winit` / `wgpu` when `--headless` is
passed. The GUI code is feature-gated (`gui`) and is *not* compiled when
`--no-default-features` is set.

```text
evogym-labs --headless --gens 3 --pop 8 --eval-secs 2 \
    --clip-secs 1 --out /tmp/opencode/evogym_test.mp4
```

Optional flags: `--seed <u64>` for reproducibility, `--task walker` or
`--task obstacle`.

Exit codes: `0` success, `1` runtime error, `2` usage error.

## GUI controls

- **Left panel** — voxel palette (click to select brush), grid size W/H
  (1..8), Apply Size, Clear.
- **Centre panel** — paint grid. Click and drag to paint the selected cell
  type onto a grid cell. Status line shows whether the design passes
  validation.
- **Right panel** — evolution config (population, eval seconds, mode, task),
  editable video plan (per-row `generation` and `seconds`, ✕ to remove,
  + Add row to insert), output path, RUN button. While running a progress
  bar replaces the RUN button; status lines show generation progress; on
  completion the output path is shown in green or the error in red.

The worker thread owns its own RNG; the GUI remains responsive (eg::Context
`request_repaint_after(100ms)`) and polls the channel each frame.

## Physics

### Setup

For a `w × h` cell grid, there are `(w + 1) × (h + 1)` point masses — one
per grid vertex. A cell `(cx, cy)` occupies world `[cx, cx + 1] × [cy,
cy + 1]`; its four corners are at world positions `(cx, cy)`, `(cx + 1,
cy)`, `(cx, cy + 1)`, `(cx + 1, cy + 1)`.

Each non-empty cell contributes 6 springs:

| Spring | Type | Rest length | Notes |
|--------|------|-------------|-------|
| BL ↔ BR (bottom edge) | edge | 1.0 | `ActH` modulates |
| TL ↔ TR (top edge)    | edge | 1.0 | `ActH` modulates |
| BL ↔ TL (left edge)   | edge | 1.0 | `ActV` modulates |
| BR ↔ TR (right edge)  | edge | 1.0 | `ActV` modulates |
| BL ↔ TR (diagonal)    | diag | √2 | structural |
| BR ↔ TL (diagonal)    | diag | √2 | structural |

Adjacent cells share the same edge spring — when multiple cells share an
edge, the stiffer one wins (`Rigid > Soft`). Multiple actuators
contributing to the same edge spring have their `ΔL` offsets summed.

### Forces

Each spring contributes `F = -k · (|d| - L_eff) · d̂ - c · (v_rel · d̂) ·
d̂` where `L_eff = base + Σ actuator_ΔL`. Ground (and obstacle) contact
applies a stiff normal penalty spring with friction: the tangential
friction force is clamped to `μ · F_normal` (Coulomb-ish).

### Integration

Semi-implicit Euler at `dt = 1/240 s` with **4 substeps per 60 fps frame**.
For each substep:

1. Zero forces, apply gravity.
2. Add spring + axial damping forces (per spring).
3. Apply ground penalty + clamped friction.
4. Apply obstacle penalty + clamped friction.
5. Velocity update with per-substep linear damping (`v *= 0.995`).
6. Velocity clamp to `|v| ≤ 50 m/s`.
7. Position update + position clamps (`y ≥ -0.5`, `|x| ≤ 2000`).
8. NaN scrub: any mass that ends up with non-finite coords / velocity is
   yanked to the origin with zero velocity, and `World::nan_seen` is set.
   The next eval reports fitness 0 to the GA.

### Numerical stability tricks

- **Per-substep linear damping** (`v *= 0.995`) bleeds energy faster than a
  stiff spring can pump it in.
- **Velocity clamp** (`|v| ≤ 50 m/s`) caps integrator overshoot.
- **Position clamp** keeps NaN-producing explosions bounded so the
  simulation can keep running instead of cascading into a full eval failure.
- **Per-spring force cap** (`±20 kN`) prevents stiff springs from injecting
  numerical noise when a vertex tunnels.
- **NaN scrub** as a last resort — fitness 0 is reported and the GA keeps
  going.

These four guards together let the GA survive hundreds of generations on
random designs without the simulation diverging.

### Controller

Open-loop sinusoidal per actuator. For actuator `i` at time `t`:

```
ΔL_i(t) = amp_i · sin(2π · freq_hz · t + phase_i)
```

`ΔL` is added to the rest length of each edge spring the actuator
modulates. Multiple actuators on the same spring sum their contributions.

## Module map

| Module | Purpose |
|--------|---------|
| `voxel.rs` | `Cell` enum, `Morphology`, validation (connectedness / actuator presence) |
| `physics.rs` | `World` struct + integration step (force application split into free fns) |
| `physics_build.rs` | `World`-construction helpers — split out to keep `physics.rs` under 300 LOC |
| `physics_tests.rs` | Unit tests for `physics.rs` (sibling file via `#[path]`) |
| `controller.rs` | Sinusoidal open-loop controller + mutation |
| `tasks.rs` | `TaskKind` enum + obstacle layouts + fitness function |
| `evolution.rs` | GA with `rayon`-parallel evaluation, morphology + controller mutation |
| `evolution_tests.rs` | Unit tests for `evolution.rs` |
| `editor.rs` | Paint / resize / clear helpers for the GUI |
| `render.rs` | Software rasterizer (1080×1080), calls into `text.rs` for overlays |
| `text.rs` | `ab_glyph` text renderer with a 5×7 bitmap-font fallback when no TTF is found |
| `video.rs` | `ffmpeg` piping per clip + parallel encode + concat |
| `cli.rs` | clap CLI for `--headless` mode (no `eframe`/`winit`/`wgpu` linked) |
| `app.rs` | eframe / egui GUI (feature-gated behind `gui`) |
| `main.rs` | Binary entry: dispatches to CLI or GUI based on `--headless` |

## Engineering choices

- **No physics engine** — point-mass spring model is hand-rolled. Keeps the
  code understandable and avoids a heavy dependency.
- **No video crate** — frames are produced as raw RGB bytes and piped
  through `ffmpeg` subprocesses. `ffmpeg` is the canonical video encoder
  and is already installed on Linux.
- **eframe 0.32 with `glow`** — OpenGL backend; smaller binary than `wgpu`,
  compiles faster. The GUI is feature-gated so headless builds do not pull
  any windowing / GPU dependencies.
- **rand 0.9** — `rand::rng()`, `random_range`. Boxed RNG for the master
  thread so it can be shared with the parallel evaluation pass.
- **Clippy `-D warnings` clean** — both `cargo clippy --all-features` and
  `cargo clippy --no-default-features` succeed with no warnings.
- **No `unsafe`** in the library path. `Box::leak` is used in the font
  loader to give `FontRef` a `'static` lifetime; this is the documented
  ab_glyph pattern and the leak is bounded by the program font file size.

## Build matrix

| Command | Notes |
|---------|-------|
| `cargo build --release` | GUI + CLI both linked |
| `cargo build --release --no-default-features` | CLI only, no `eframe` / `winit` / `wgpu` |
| `cargo test --release --all-features` | All 35 unit tests + integration test |
| `cargo clippy --release --all-features --tests --lib --bins -- -D warnings` | Clean |
| `cargo test --release --no-default-features` | 35 unit tests (no integration) |
| `cargo test --release --features integration --test integration` | Integration smoke test |

## Known limitations

- The demo robot (5×5 with three ActH stacked vertically) only travels a
  few centimetres per evolution run. Evolution was tested at small scale
  (3–8 individuals × 3 generations) which is not enough to discover
  effective gaits. The physics is correct; more compute and more
  generations produce more locomotion, but that is left to the user.
- File dialogs in the GUI are stubbed (`rfd` was deliberately not added to
  keep the dependency surface small). Type the output path directly.
- The `ab_glyph` font loader leaks one font file's worth of bytes per
  process — acceptable for a session-length tool, but not for a long-running
  server.
- `concat -c copy` is not used for stitching clips; we re-encode at the
  concat step. This guarantees frame-accurate PTS and avoids the class of
  failure where `-c copy` produces inconsistent output across clips.
## License

Apache License 2.0 — see [LICENSE](LICENSE).
