//! Editor glue: paint operations + grid resize.
//!
//! Pure data wrangling — no GUI / physics knowledge. The GUI calls into this
//! to apply a brush stroke and to resize the grid; the GA reuses the resize
//! helper when mutating morphology boundaries (it currently does not, since
//! morphology size is a config knob).

use crate::voxel::{Cell, Morphology};

/// Apply a paint action — sets the cell at `(cx, cy)` to `cell` and clears
/// any colour override (a freshly placed cell starts in its type's default
/// colour). No-op if out of bounds.
pub fn paint(m: &mut Morphology, cx: u8, cy: u8, cell: Cell) {
    m.set(cx, cy, cell);
    m.set_color(cx, cy, None);
}

/// Colour-only paint: overrides the cell's display colour without touching
/// its type. `None` removes the override (back to the type colour).
pub fn paint_color(m: &mut Morphology, cx: u8, cy: u8, color: Option<[u8; 3]>) {
    m.set_color(cx, cy, color);
}

/// Replace the entire grid, copying cells from `cells` row-major into a new
/// `w × h` morphology. Cells beyond the source are left `Empty`.
pub fn resize(m: &mut Morphology, w: u8, h: u8) {
    let mut next = Morphology::new(w, h);
    let copy_w = w.min(m.w);
    let copy_h = h.min(m.h);
    for cy in 0..copy_h {
        for cx in 0..copy_w {
            next.set(cx, cy, m.get(cx, cy));
            next.set_color(cx, cy, m.color_of(cx, cy));
        }
    }
    *m = next;
}

/// Clear the entire grid (cells → Empty, colours → none).
pub fn clear(m: &mut Morphology) {
    for c in m.cells.iter_mut() { *c = Cell::Empty; }
    for c in m.colors.iter_mut() { *c = None; }
}

/// Physics puts cell row 0 at the *bottom* (world y grows upward) while the
/// editor draws row 0 at the top of the screen. This maps a screen row to a
/// cell row so what the user draws matches what the simulation runs.
pub fn screen_to_cell_y(screen_row: u8, h: u8) -> u8 {
    h - 1 - screen_row
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screen_row_zero_is_physics_top() {
        assert_eq!(screen_to_cell_y(0, 5), 4);
        assert_eq!(screen_to_cell_y(4, 5), 0);
    }

    #[test]
    fn paint_sets_cell() {
        let mut m = Morphology::new(3, 3);
        paint(&mut m, 1, 1, Cell::Rigid);
        assert_eq!(m.get(1, 1), Cell::Rigid);
    }

    #[test]
    fn paint_ignores_out_of_bounds() {
        let mut m = Morphology::new(2, 2);
        paint(&mut m, 5, 5, Cell::Rigid);
        // No crash, nothing changes.
        assert!(m.cells.iter().all(|c| *c == Cell::Empty));
    }

    #[test]
    fn resize_keeps_overlap() {
        let mut m = Morphology::new(4, 4);
        m.set(0, 0, Cell::Rigid);
        m.set(3, 3, Cell::Rigid);
        resize(&mut m, 2, 2);
        assert_eq!(m.get(0, 0), Cell::Rigid);
        // Out-of-range cells are dropped.
        assert!(m.cells.iter().all(|c| *c == Cell::Rigid || *c == Cell::Empty));
    }

    #[test]
    fn paint_colour_keeps_type_and_survives_resize() {
        let mut m = Morphology::new(3, 3);
        paint(&mut m, 1, 1, Cell::ActH);
        paint_color(&mut m, 1, 1, Some([10, 20, 30]));
        assert_eq!(m.get(1, 1), Cell::ActH, "colour paint must not change the type");
        assert_eq!(m.color_of(1, 1), Some([10, 20, 30]));
        resize(&mut m, 4, 4);
        assert_eq!(m.get(1, 1), Cell::ActH);
        assert_eq!(m.color_of(1, 1), Some([10, 20, 30]), "resize must carry colours");
        paint_color(&mut m, 1, 1, None);
        assert_eq!(m.color_of(1, 1), None, "erase restores the type colour");
        // Re-painting a type onto a coloured cell resets it to the default colour.
        paint_color(&mut m, 0, 0, Some([200, 10, 10]));
        paint(&mut m, 0, 0, Cell::Soft);
        assert_eq!(m.color_of(0, 0), None, "freshly placed cell starts in the default colour");
    }

    #[test]
    fn clear_empties_grid() {
        let mut m = Morphology::new(3, 3);
        m.set(0, 0, Cell::Rigid);
        m.set(1, 1, Cell::ActH);
        clear(&mut m);
        assert!(m.cells.iter().all(|c| *c == Cell::Empty));
    }
}