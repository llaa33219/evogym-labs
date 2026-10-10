//! Text rendering — `ab_glyph` with a 5×7 bitmap-font fallback.
//!
//! The renderer first tries to load a Noto TTF from `/usr/share/fonts/noto*`.
//! If that fails (no font installed, no `std::fs::read` permission, the
//! font is malformed), it falls back to a hand-rolled 5×7 bitmap font that
//! only covers ASCII printable characters — enough for the "Generation N" /
//! "Fitness X m" overlay.
//!
//! ## ab_glyph coordinate-system gotcha
//!
//! `OutlinedGlyph::draw(callback)` yields pixel coordinates relative to
//! `px_bounds().min` — i.e. the top-left of the glyph's bounding box.
//! `px_bounds().min` is itself in the same coordinate space as
//! `glyph.position`, which we set to absolute pixel coordinates. So the
//! final on-screen pixel is `(gx + bx, gy + by)` where `(bx, by) =
//! px_bounds().min`. A future maintainer reading this code should *not*
//! strip that `+ bx / + by` offset — that was the bug behind the
//! placeholder bubbles you saw in the very first render of the headless
//! CLI.

use ab_glyph::{Font, FontRef, Glyph, Point, PxScale, ScaleFont};
use std::sync::OnceLock;

use crate::render::{FRAME_W, FrameBuffer, Rgb};

/// Locate a usable Noto TTF on the system, preferring Regular / Bold over
/// the CJK collection (smaller).
pub fn find_noto_font() -> Option<Vec<u8>> {
    let candidates = [
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/TTF/NotoSans-Bold.ttf",
        "/usr/share/fonts/TTF/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    ];
    for path in candidates {
        if let Ok(b) = std::fs::read(path) {
            return Some(b);
        }
    }
    if let Ok(rd) = std::fs::read_dir("/usr/share/fonts/noto") {
        for entry in rd.flatten() {
            let p = entry.path();
            if let Some(ext) = p.extension().and_then(|s| s.to_str())
                && ext.eq_ignore_ascii_case("ttf")
                && let Ok(b) = std::fs::read(&p)
            {
                return Some(b);
            }
        }
    }
    None
}

/// Process-wide cached font — loaded and parsed once. Loading per draw call
/// (a TTF read + parse + leak per frame of video) was the single biggest
/// cost in the whole render pipeline.
static FONT: OnceLock<Option<FontRef<'static>>> = OnceLock::new();

fn cached_font() -> Option<&'static FontRef<'static>> {
    FONT.get_or_init(|| {
        let bytes = find_noto_font()?;
        // Leak the bytes once so the FontRef borrow lives forever.
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        FontRef::try_from_slice(leaked).ok()
    })
    .as_ref()
}

/// A tiny 5×7 bitmap font used as a fallback when no TTF can be loaded.
/// Only ASCII printable characters are supported — that's enough for the
/// "Generation N" overlay.
const GLYPH_W: usize = 5;
const GLYPH_H: usize = 7;

const BITMAP: &[[u8; GLYPH_H]] = &[
    [0, 0, 0, 0, 0, 0, 0],                // ' '
    [0, 0x04, 0x04, 0x04, 0x04, 0, 0x04], // '!'
    [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
    [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
    [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F],
    [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
    [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
    [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
    [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
    [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
    [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
    [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
    [0, 0, 0x04, 0, 0, 0x04, 0], // ':'
    [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
    [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
    [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
    [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
    [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
    [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E],
    [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
    [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
    [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
    [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
    [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
    [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
    [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
    [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
    [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
    [0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E],
    [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
    [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
    [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
    [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
    [0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04],
    [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
];

/// Map ASCII byte → bitmap index in `BITMAP`. Returns `None` for unsupported
/// characters; we render those as blanks.
fn bitmap_index(b: u8) -> Option<usize> {
    match b {
        b' ' => Some(0),
        b'!' => Some(1),
        b'0'..=b'9' => Some(2 + (b - b'0') as usize),
        b':' => Some(12),
        b'A'..=b'Z' => Some(13 + (b - b'A') as usize),
        b'a'..=b'z' => Some(13 + (b - b'A') as usize),
        _ => None,
    }
}

/// Draw text at the given pixel position with a scale factor (1 = 5×7 px per
/// glyph, 2 = 10×14 px, …). `colour` is the foreground colour.
pub fn draw_text(fb: &mut FrameBuffer, x: i32, y: i32, text: &str, scale: usize, colour: Rgb) {
    if let Some(font) = cached_font() {
        draw_text_glyph(fb, font, x, y, text, colour);
        return;
    }
    let mut pen_x = x;
    for b in text.bytes() {
        if let Some(idx) = bitmap_index(b) {
            let g = &BITMAP[idx];
            for (row, &byte) in g.iter().enumerate() {
                for col in 0..GLYPH_W {
                    if (byte >> (GLYPH_W - 1 - col)) & 1 == 1 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                fb.set_pixel(
                                    pen_x + (col * scale + dx) as i32,
                                    y + (row * scale + dy) as i32,
                                    colour,
                                );
                            }
                        }
                    }
                }
            }
        }
        pen_x += (GLYPH_W + 1) as i32 * scale as i32;
    }
}

fn draw_text_glyph(
    fb: &mut FrameBuffer,
    font: &FontRef<'static>,
    x: i32,
    y: i32,
    text: &str,
    colour: Rgb,
) {
    let scale = PxScale::from(72.0);
    let scaled = font.as_scaled(scale);
    let mut pen: Point = Point {
        x: x as f32,
        y: y as f32 + scaled.ascent(),
    };
    for ch in text.chars() {
        let glyph_id = font.glyph_id(ch);
        let glyph: Glyph = glyph_id.with_scale_and_position(scale, pen);
        if let Some(outline) = font.outline_glyph(glyph) {
            let b = outline.px_bounds();
            let bx = b.min.x as i32;
            let by = b.min.y as i32;
            // cov is per-pixel coverage (see module docs for the px_bounds offset).
            outline.draw(|gx, gy, cov| {
                let a = cov.clamp(0.0, 1.0);
                if a > 0.0 {
                    blend_pixel(fb, gx as i32 + bx, gy as i32 + by, colour, a);
                }
            });
        }
        pen.x += scaled.h_advance(glyph_id);
    }
}

fn blend_pixel(fb: &mut FrameBuffer, x: i32, y: i32, c: Rgb, a: f32) {
    if x < 0 || y < 0 || (x as usize) >= fb.w || (y as usize) >= fb.h {
        return;
    }
    let i = (y as usize) * fb.w * 3 + (x as usize) * 3;
    for (ch, src) in [(c.0, 0usize), (c.1, 1), (c.2, 2)] {
        let dst = fb.rgb[i + src] as f32;
        fb.rgb[i + src] = (ch as f32 * a + dst * (1.0 - a)).round() as u8;
    }
}

/// Convenience: format `gen` / `fit` text and draw it at the top-left of the
/// framebuffer in white with a drop-shadow.
pub fn draw_overlay(fb: &mut FrameBuffer, generation: usize, fit: f32, is_final: bool) {
    let line1 = if is_final {
        format!("Final (Generation {})", generation)
    } else {
        format!("Generation {}", generation)
    };
    let line2 = format!("Fitness {:.2} m", fit);
    draw_text(fb, 27, 27, &line1, 8, Rgb::BLACK);
    draw_text(fb, 24, 24, &line1, 8, Rgb::WHITE);
    draw_text(fb, 27, 117, &line2, 8, Rgb::BLACK);
    draw_text(fb, 24, 114, &line2, 8, Rgb::WHITE);
}

// Keep `FRAME_W` referenced from this module so rustc does not strip the
// re-export when nothing else uses it; otherwise the integration test of
// the overlay would lose its pixel-budget assertion.
#[allow(dead_code)]
const _FRAME_W_REF: usize = FRAME_W;
