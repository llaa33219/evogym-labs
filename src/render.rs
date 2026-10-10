//! Software rasterizer: 1080 × 1080 RGB frames.
//!
//! Pipeline per frame:
//! 1. Clear to background colour (a soft sky blue → ground band → solid
//!    neutral floor).
//! 2. Draw the ground strip (horizontal band) and any obstacles.
//! 3. Camera-follow: the camera centres on the robot's COM, with a fixed
//!    world-to-pixel scale so the world never zooms.
//! 4. For every voxel cell, draw a filled quadrilateral using the four
//!    corner vertex positions (so the robot visibly *deforms* as it moves —
//!    springs / actuators actually change cell shape).
//! 5. Stroke each quad with a subtle dark edge so cells stay visually
//!    separable.
//! 6. Overlay the text label — see `text.rs` for the `ab_glyph` + bitmap
//!    fallback renderer.

use crate::physics::{Ground, Obstacle, Vec2, World};
#[cfg(test)]
use crate::text::find_noto_font;
use crate::voxel::Cell;

/// Frame dimensions — fixed by the spec.
pub const FRAME_W: usize = 1080;
pub const FRAME_H: usize = 1080;

/// Default pixels-per-metre zoom for small robots; larger robots zoom out
/// via `Camera::for_robot`.
const PX_PER_M: f32 = 80.0;
/// Half-canvas vertical offset: ground sits a bit below the centre so the
/// robot has headroom to extend upward.
const CAMERA_BASELINE_PX: f32 = FRAME_H as f32 * 0.55;

/// RGB pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    pub const BLACK: Rgb = Rgb(0, 0, 0);
    pub const WHITE: Rgb = Rgb(255, 255, 255);
}

/// Cell colour palette.
fn cell_colour(c: Cell) -> Rgb {
    match c {
        Cell::Empty => Rgb(0, 0, 0), // never drawn
        Cell::Rigid => Rgb(140, 140, 150),
        Cell::Soft => Rgb(80, 200, 110),
        Cell::ActH => Rgb(220, 70, 70),
        Cell::ActV => Rgb(70, 100, 220),
    }
}

/// Edge / shadow colour — drawn on top of the cell fill to give it definition.
fn cell_edge(c: Cell) -> Rgb {
    match c {
        Cell::Empty => Rgb::BLACK,
        Cell::Rigid => Rgb(40, 40, 45),
        Cell::Soft => Rgb(20, 70, 40),
        Cell::ActH => Rgb(80, 20, 20),
        Cell::ActV => Rgb(20, 30, 80),
    }
}

/// 2-D pixel buffer for one frame.
pub struct FrameBuffer {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>, // w * h * 3 bytes, RGB row-major top-to-bottom.
}

impl FrameBuffer {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            rgb: vec![0; w * h * 3],
        }
    }

    pub fn clear(&mut self, colour: Rgb) {
        for chunk in self.rgb.chunks_exact_mut(3) {
            chunk[0] = colour.0;
            chunk[1] = colour.1;
            chunk[2] = colour.2;
        }
    }

    pub fn set_pixel(&mut self, x: i32, y: i32, c: Rgb) {
        if x < 0 || y < 0 || (x as usize) >= self.w || (y as usize) >= self.h {
            return;
        }
        let i = (y as usize) * self.w * 3 + (x as usize) * 3;
        self.rgb[i] = c.0;
        self.rgb[i + 1] = c.1;
        self.rgb[i + 2] = c.2;
    }

    /// Solid filled horizontal band (used for ground). Ignores `x` clipping.
    pub fn fill_band(&mut self, y_top: i32, y_bot: i32, c: Rgb) {
        let y_top = y_top.max(0) as usize;
        let y_bot = (y_bot.min(self.h as i32)) as usize;
        for y in y_top..y_bot {
            let row_start = y * self.w * 3;
            for x in 0..self.w {
                let i = row_start + x * 3;
                self.rgb[i] = c.0;
                self.rgb[i + 1] = c.1;
                self.rgb[i + 2] = c.2;
            }
        }
    }

    /// Append a frame to an encoder sink — used by the headless CLI.
    pub fn as_bytes(&self) -> &[u8] {
        &self.rgb
    }
}

/// Camera state (smooth follow). `px_per_m` adapts to the robot's grid
/// size so a large creature still fits the frame instead of filling it.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub com_x: f32,
    pub com_y: f32,
    pub px_per_m: f32,
    leash_x: f32,
    leash_y: f32,
}

impl Camera {
    /// Build a camera for a `morph_w × morph_h` robot. Small robots keep the
    /// default zoom (80 px/m); anything whose largest dimension would exceed
    /// ~45 % of the frame gets zoomed out. Leash bounds are derived from the
    /// resulting visible window.
    pub fn for_robot(morph_w: u8, morph_h: u8) -> Self {
        let max_dim = morph_w.max(morph_h) as f32;
        let px_per_m = PX_PER_M.min(FRAME_H as f32 * 0.45 / max_dim.max(1.0));
        let half_w_m = FRAME_W as f32 * 0.5 / px_per_m;
        let down_m = CAMERA_BASELINE_PX / px_per_m;
        let up_m = (FRAME_H as f32 - CAMERA_BASELINE_PX) / px_per_m;
        Self {
            com_x: 0.0,
            com_y: 0.0,
            px_per_m,
            leash_x: (half_w_m - max_dim * 0.5 - 0.5).max(0.5),
            leash_y: (down_m.min(up_m) - max_dim * 0.5 - 0.5).max(0.5),
        }
    }

    pub fn world_to_screen(&self, x: f32, y: f32) -> (f32, f32) {
        let sx = (x - self.com_x) * self.px_per_m + FRAME_W as f32 * 0.5;
        let sy = CAMERA_BASELINE_PX - (y - self.com_y) * self.px_per_m;
        (sx, sy)
    }

    /// Snap camera target to new COM with a small smoothing factor (so the
    // camera does not jitter on every frame). `alpha` closer to 1.0 → more
    // responsive, closer to 0 → more sluggish. After smoothing, a hard leash
    // drags the camera along so a fast robot ( downhill slide, fall ) can
    // never leave the visible window.
    pub fn follow(&mut self, target_x: f32, target_y: f32, alpha: f32) {
        self.com_x += (target_x - self.com_x) * alpha;
        self.com_y += (target_y - self.com_y) * alpha;
        let dx = (target_x - self.com_x).clamp(-self.leash_x, self.leash_x);
        self.com_x = target_x - dx;
        let dy = (target_y - self.com_y).clamp(-self.leash_y, self.leash_y);
        self.com_y = target_y - dy;
    }
}

/// Edge colour for outlines.
const EDGE_COLOR: Rgb = Rgb(25, 25, 28);

/// Stroke a line between two screen-space points.
fn draw_line(fb: &mut FrameBuffer, x0: f32, y0: f32, x1: f32, y1: f32, colour: Rgb) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steps = dx.abs().max(dy.abs()).ceil() as i32;
    if steps <= 0 {
        return;
    }
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let px = (x0 + dx * t) as i32;
        let py = (y0 + dy * t) as i32;
        fb.set_pixel(px, py, colour);
        fb.set_pixel(px + 1, py, colour);
        fb.set_pixel(px, py + 1, colour);
    }
}

/// Fill a convex quadrilateral (4 corner screen positions) by scanline.
fn fill_quad(fb: &mut FrameBuffer, q: &[(f32, f32); 4], colour: Rgb) {
    let mut ys = [q[0].1, q[1].1, q[2].1, q[3].1];
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let y_min = ys[0].ceil() as i32;
    let y_max = ys[3].floor() as i32;
    // 4 edges, closing the loop (3→0) — `q.windows(2)` would miss the wrap.
    let edges = [(q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])];
    for y in y_min..=y_max {
        if y < 0 || (y as usize) >= fb.h {
            continue;
        }
        let mut xs: Vec<f32> = Vec::new();
        for &((x0, y0), (x1, y1)) in &edges {
            if (y0 <= y as f32 && y1 >= y as f32) || (y1 <= y as f32 && y0 >= y as f32) {
                if (y1 - y0).abs() < 1e-6 {
                    continue;
                }
                let t = (y as f32 - y0) / (y1 - y0);
                xs.push(x0 + t * (x1 - x0));
            }
        }
        if xs.len() < 2 {
            continue;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let xa = xs[0].ceil() as i32;
        let xb = xs[xs.len() - 1].floor() as i32;
        for x in xa..=xb {
            if x < 0 || (x as usize) >= fb.w {
                continue;
            }
            fb.set_pixel(x, y, colour);
        }
    }
}

/// Max |ΔL| an actuator can produce (matches the controller amplitude cap) —
/// used to normalise the brightness tint.
const ACTUATOR_DL_MAX: f32 = 0.3;

/// Brighten an actuator cell's colour toward white in proportion to its
/// current drive amplitude.
fn actuator_tint(base: Rgb, dl: f32) -> Rgb {
    let t = (dl.abs() / ACTUATOR_DL_MAX).clamp(0.0, 1.0);
    let mix = |c: u8| {
        (c as f32 + (255.0 - c as f32) * 0.35 * t)
            .round()
            .min(255.0) as u8
    };
    Rgb(mix(base.0), mix(base.1), mix(base.2))
}

/// Draw the robot by iterating cells, projecting their four corners from world
/// vertex positions, filling each cell with its colour and stroking the four
/// edges. Actuator cells brighten with their current drive `actuator_dl[i]`.
pub fn draw_robot(fb: &mut FrameBuffer, cam: &Camera, world: &World, actuator_dl: &[f32]) {
    let positions: Vec<Vec2> = world.masses.iter().map(|m| m.pos).collect();
    draw_robot_positions(fb, cam, world, &positions, actuator_dl);
}

/// Position-snapshot variant of `draw_robot` — used by the parallel video
/// renderer, where each frame is drawn from a recorded snapshot while the
/// simulation has long moved on.
pub fn draw_robot_positions(
    fb: &mut FrameBuffer,
    cam: &Camera,
    world: &World,
    positions: &[Vec2],
    actuator_dl: &[f32],
) {
    let w = world.morph.w as usize;
    let vert_idx = |vx: usize, vy: usize| vy * (w + 1) + vx;
    for cy in 0..world.morph.h as usize {
        for cx in 0..world.morph.w as usize {
            let cell_i = cy * w + cx;
            let c = world.morph.cells[cell_i];
            if !c.is_solid() {
                continue;
            }
            let base = world.morph.colors[cell_i]
                .map(|rgb| Rgb(rgb[0], rgb[1], rgb[2]))
                .unwrap_or(cell_colour(c));
            let fill = match world.cell_actuator_idx[cell_i] {
                Some(ai) => actuator_tint(base, actuator_dl.get(ai).copied().unwrap_or(0.0)),
                None => base,
            };
            let bl = positions[vert_idx(cx, cy)];
            let br = positions[vert_idx(cx + 1, cy)];
            let tl = positions[vert_idx(cx, cy + 1)];
            let tr = positions[vert_idx(cx + 1, cy + 1)];
            let sbl = cam.world_to_screen(bl.x, bl.y);
            let sbr = cam.world_to_screen(br.x, br.y);
            let stl = cam.world_to_screen(tl.x, tl.y);
            let str = cam.world_to_screen(tr.x, tr.y);
            let q = [sbl, sbr, str, stl];
            fill_quad(fb, &q, fill);
            draw_line(fb, sbl.0, sbl.1, sbr.0, sbr.1, cell_edge(c));
            draw_line(fb, sbr.0, sbr.1, str.0, str.1, cell_edge(c));
            draw_line(fb, str.0, str.1, stl.0, stl.1, cell_edge(c));
            draw_line(fb, stl.0, stl.1, sbl.0, sbl.1, cell_edge(c));
        }
    }
}

/// Block-noise shade for terrain and obstacle blocks, keyed by integer
/// world-grid coordinates so ground and obstacles share one seamless 1 m
/// block pattern.
fn ground_shade(col: i32, row: i32) -> Rgb {
    let base = (60i32, 60i32, 65i32);
    let mut h = (col as u32).wrapping_mul(2_654_435_761) ^ (row as u32).wrapping_mul(40_503);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    let shade = (h & 15) as i32 - 8;
    let stone = h % 23 < 3;
    let (r, g, b) = if stone {
        (base.0 - 20, base.1 - 20, base.2 - 18)
    } else {
        (base.0 + shade, base.1 + shade, base.2 + shade)
    };
    Rgb(r as u8, g as u8, b as u8)
}

/// Draw the heightfield ground and obstacle boxes. The ground is filled
/// column-by-column because its surface height varies with world x; the
/// block noise is anchored to *world* coordinates (not screen pixels) so the
/// pattern scrolls under the follow-camera — without that anchor a
/// camera-locked robot appears to stand still no matter how far it walks.
/// Blocks are 1 m × 1 m, matching the on-screen voxel size.
pub fn draw_terrain(fb: &mut FrameBuffer, cam: &Camera, ground: &Ground, obstacles: &[Obstacle]) {
    fb.fill_band(0, FRAME_H as i32, Rgb(180, 200, 220));
    for px in 0..FRAME_W as i32 {
        let world_x = (px as f32 - FRAME_W as f32 * 0.5) / cam.px_per_m + cam.com_x;
        let (_, sy_f) = cam.world_to_screen(world_x, ground.height(world_x));
        let sy = sy_f.round() as i32;
        if sy >= 0 && (sy as usize) < FRAME_H {
            fb.set_pixel(px, sy, Rgb(150, 150, 155));
            fb.set_pixel(px, sy + 1, Rgb(120, 120, 126));
        }
        let col = world_x.floor() as i32;
        // Fill in depth bands: the shade only changes every 1 m of depth, so
        // hash once per band instead of once per pixel.
        let mut y = (sy + 2).max(0);
        while y < FRAME_H as i32 {
            let row = ((y - sy) as f32 / cam.px_per_m).floor() as i32;
            let band_end = (sy + ((row + 1) as f32 * cam.px_per_m) as i32)
                .min(FRAME_H as i32)
                .max(y + 1);
            let c = ground_shade(col, row);
            for yy in y..band_end {
                fb.set_pixel(px, yy, c);
            }
            y = band_end;
        }
    }

    for ob in obstacles {
        let (x0, y1) = cam.world_to_screen(ob.x_min, ob.y_min);
        let (x1, y0) = cam.world_to_screen(ob.x_max, ob.y_max);
        // Same 1 m world-grid block texture as the ground, so obstacles read
        // as part of the terrain rather than foreign props.
        let ix0 = (x0.round() as i32).max(0);
        let ix1 = (x1.round() as i32).min(FRAME_W as i32 - 1);
        let iy0 = (y0.round() as i32).max(0);
        let iy1 = (y1.round() as i32).min(FRAME_H as i32 - 1);
        for py in iy0..=iy1 {
            let world_y = cam.com_y - (py as f32 - CAMERA_BASELINE_PX) / cam.px_per_m;
            let row = world_y.floor() as i32;
            for px in ix0..=ix1 {
                let world_x = (px as f32 - FRAME_W as f32 * 0.5) / cam.px_per_m + cam.com_x;
                fb.set_pixel(px, py, ground_shade(world_x.floor() as i32, row));
            }
        }
        for px in ix0..=ix1 {
            fb.set_pixel(px, iy0, Rgb(150, 150, 155));
            fb.set_pixel(px, iy0 + 1, Rgb(120, 120, 126));
        }
        draw_line(fb, x0, y0, x1, y0, EDGE_COLOR);
        draw_line(fb, x1, y0, x1, y1, EDGE_COLOR);
        draw_line(fb, x1, y1, x0, y1, EDGE_COLOR);
        draw_line(fb, x0, y1, x0, y0, EDGE_COLOR);
    }
}

/// Render one frame of the world into the given framebuffer, using the
/// supplied camera (which the caller should update to follow `world.com()`).
pub fn render_world(fb: &mut FrameBuffer, cam: &Camera, world: &World, actuator_dl: &[f32]) {
    draw_terrain(fb, cam, &world.ground, &world.obstacles);
    draw_robot(fb, cam, world, actuator_dl);
}

/// Debug helper — write a PPM (P6) file. Used by the unit tests; the format
/// is trivial so we avoid pulling in `image`.
#[allow(dead_code)]
pub fn save_ppm(path: &str, fb: &FrameBuffer) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    write!(f, "P6\n{} {}\n255\n", fb.w, fb.h)?;
    f.write_all(&fb.rgb)
}

// Re-exports so callers don't need to know the split.
pub use crate::text::{draw_overlay, draw_text};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{Cell, Morphology};

    fn dummy_morph() -> Morphology {
        let mut m = Morphology::new(2, 2);
        for cy in 0..2 {
            for cx in 0..2 {
                m.set(cx, cy, Cell::Soft);
            }
        }
        m.set(0, 0, Cell::ActH);
        m
    }

    #[test]
    fn framebuffer_clear_and_pixel() {
        let mut fb = FrameBuffer::new(10, 10);
        fb.clear(Rgb(10, 20, 30));
        assert_eq!(fb.rgb[0], 10);
        fb.set_pixel(5, 5, Rgb(99, 99, 99));
        let i = 5 * 10 * 3 + 5 * 3;
        assert_eq!(fb.rgb[i], 99);
    }

    #[test]
    fn render_world_draws_nonempty_framebuffer() {
        let mut fb = FrameBuffer::new(FRAME_W, FRAME_H);
        let morph = dummy_morph();
        let mut world = World::new(morph);
        world.reset();
        let com = world.com();
        let mut cam = Camera::for_robot(2, 2);
        cam.follow(com.x, com.y, 1.0);
        render_world(&mut fb, &cam, &world, &[]);
        let _ = save_ppm("/tmp/evogym_test_render.ppm", &fb);
        let mut has_robot = false;
        for chunk in fb.rgb.chunks_exact(3) {
            let r = chunk[0];
            let g = chunk[1];
            let b = chunk[2];
            if (r == 80 && g == 200 && b == 110)
                || (r == 140 && g == 140 && b == 150)
                || (r == 220 && g == 70 && b == 70)
                || (r == 70 && g == 100 && b == 220)
            {
                has_robot = true;
                break;
            }
        }
        assert!(has_robot, "robot did not paint any cell-coloured pixel");
    }

    #[test]
    fn text_overlay_writes_pixels() {
        let mut fb = FrameBuffer::new(FRAME_W, FRAME_H);
        fb.clear(Rgb(50, 50, 50));
        draw_overlay(&mut fb, 12, 0.42, false);
        let mut white_count = 0;
        for y in 0..200 {
            for x in 0..600 {
                let i = y * FRAME_W * 3 + x * 3;
                if fb.rgb[i] >= 200 && fb.rgb[i + 1] >= 200 && fb.rgb[i + 2] >= 200 {
                    white_count += 1;
                }
            }
        }
        assert!(white_count > 50, "overlay drew nothing");
    }

    /// Regression for the "white rounded rectangles behind each glyph" bug.
    ///
    /// The ab_glyph rasterizer's `for_each_pixel_2d` callback fires for
    /// **every pixel** in the rasterizer's buffer (size = `px_bounds`) and
    /// passes a *cumulative* running sum as the coverage value — never reset
    /// across the iteration. With a naive `if cov > 0.0` filter, every pixel
    /// from the first non-zero pixel onwards gets painted, which fills each
    /// glyph's bounding box with a solid block of the text colour. The fix
    /// uses the per-pixel delta (`acc - prev_acc`) as the real coverage.
    #[test]
    fn text_does_not_fill_glyph_bounding_boxes() {
        use crate::text::draw_text;

        let mut fb = FrameBuffer::new(800, 200);
        fb.clear(Rgb(50, 50, 50));
        draw_text(&mut fb, 20, 30, "Generation 4", 1, Rgb::WHITE);

        // Sample the region big enough to fully contain every glyph's bbox for
        // "Generation 4" at the default font scale (~40 px). With the bug the
        // entire region is ~100% white (every bbox painted solid); without the
        // bug only the glyph strokes and antialiased edges are white.
        let x0 = 0;
        let x1 = 800;
        let y0 = 20;
        let y1 = 130;
        let mut pure_white = 0usize;
        let mut total = 0usize;
        for y in y0..y1 {
            for x in x0..x1 {
                let i = y * fb.w * 3 + x * 3;
                let r = fb.rgb[i];
                let g = fb.rgb[i + 1];
                let b = fb.rgb[i + 2];
                total += 1;
                if r == 255 && g == 255 && b == 255 {
                    pure_white += 1;
                }
            }
        }
        let frac = pure_white as f32 / total as f32;
        assert!(
            frac < 0.30,
            "glyph bboxes are being filled (white fraction {frac:.2}); the \
         ab_glyph cumulative-acc fix did not take effect",
        );

        // Also assert no glyph-sized solid white stamp appears: a horizontal
        // run of pure white longer than any single stroke in the font indicates
        // a bbox fill. The thickest stroke in NotoSans-Bold at scale 36 is
        // ~25 px — anything significantly wider is the bug.
        let mut max_white_streak = 0usize;
        for y in y0..y1 {
            let mut run = 0usize;
            for x in x0..x1 {
                let i = y * fb.w * 3 + x * 3;
                if fb.rgb[i] == 255 && fb.rgb[i + 1] == 255 && fb.rgb[i + 2] == 255 {
                    run += 1;
                    max_white_streak = max_white_streak.max(run);
                } else {
                    run = 0;
                }
            }
        }
        assert!(
            max_white_streak < 80,
            "found a horizontal run of {max_white_streak} pure-white pixels — \
         likely a filled glyph bbox (the bug)",
        );
    }

    #[test]
    fn bitmap_fallback_renders_ascii() {
        let mut fb = FrameBuffer::new(200, 100);
        fb.clear(Rgb(0, 0, 0));
        draw_text(&mut fb, 10, 10, "HELLO 42:", 2, Rgb::WHITE);
    }

    #[test]
    fn camera_leash_keeps_fast_target_on_screen() {
        let mut cam = Camera::for_robot(5, 5);
        // A target teleporting far away (fall / slide) must never be more
        // than the leash distance from frame centre after follow().
        for i in 0..120 {
            cam.follow(i as f32 * 2.0, -(i as f32), 0.3);
            assert!((i as f32 * 2.0 - cam.com_x).abs() <= cam.leash_x + 1e-4);
            assert!((-(i as f32) - cam.com_y).abs() <= cam.leash_y + 1e-4);
        }
    }

    #[test]
    fn large_robots_zoom_out() {
        let small = Camera::for_robot(5, 5);
        let big = Camera::for_robot(32, 32);
        assert!(
            (small.px_per_m - 80.0).abs() < 1e-4,
            "small robots keep the default zoom"
        );
        assert!(
            big.px_per_m < 20.0,
            "32-wide robot must zoom out (got {})",
            big.px_per_m
        );
        // A 32 m robot spans under half the frame width at this zoom.
        assert!(32.0 * big.px_per_m < FRAME_W as f32 * 0.5);
    }

    #[test]
    fn custom_colour_overrides_type_colour() {
        let mut fb = FrameBuffer::new(FRAME_W, FRAME_H);
        let mut morph = dummy_morph();
        morph.set_color(0, 0, Some([12, 34, 200]));
        let mut world = World::new(morph);
        world.reset();
        let com = world.com();
        let mut cam = Camera::for_robot(2, 2);
        cam.follow(com.x, com.y, 1.0);
        render_world(&mut fb, &cam, &world, &[]);
        let mut found = false;
        for chunk in fb.rgb.chunks_exact(3) {
            if chunk[0] == 12 && chunk[1] == 34 && chunk[2] == 200 {
                found = true;
                break;
            }
        }
        assert!(found, "custom cell colour not rendered");
    }

    #[test]
    fn custom_colour_still_tints_when_actuator_driven() {
        // A driven actuator with a custom colour brightens from *its* base.
        let base = Rgb(12, 34, 200);
        let lit = actuator_tint(base, ACTUATOR_DL_MAX);
        assert!(lit.0 > base.0 && lit.1 > base.1 && lit.2 > base.2);
    }

    #[test]
    fn actuator_tint_brightens_with_drive() {
        let base = cell_colour(Cell::ActH);
        assert_eq!(actuator_tint(base, 0.0), base, "no drive → base colour");
        let lit = actuator_tint(base, ACTUATOR_DL_MAX);
        assert!(
            lit.0 > base.0 && lit.1 > base.1 && lit.2 > base.2,
            "full drive → brighter"
        );
        let half = actuator_tint(base, ACTUATOR_DL_MAX * 0.5);
        assert!(half.0 > base.0 && half.0 < lit.0, "monotonic in |dl|");
    }

    #[test]
    fn cell_colours_are_distinct() {
        let mut seen = std::collections::HashSet::new();
        for c in [Cell::Rigid, Cell::Soft, Cell::ActH, Cell::ActV] {
            seen.insert(cell_colour(c));
        }
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn noto_font_is_discoverable_on_this_system() {
        // Not strictly required (bitmap fallback exists), but in our CI box
        // we expect at least one Noto font to be present.
        assert!(
            find_noto_font().is_some(),
            "expected Noto font under /usr/share/fonts"
        );
    }
}
