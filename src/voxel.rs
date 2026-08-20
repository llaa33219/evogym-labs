//! Voxel grid & morphology.
//!
//! A morphology is a fixed-size grid of cells. The point-mass / spring physics
//! places a `PointMass` at every grid vertex and connects adjacent vertices
//! with springs whenever at least one of the cells sharing that edge / diagonal
//! is solid.
//!
//! Conventions:
//! * world axes: +x right, +y up. Ground sits at `y = 0`.
//! * cell `(cx, cy)` spans world x ∈ `[cx, cx+1]`, y ∈ `[cy, cy+1]`,
//!   `cy = 0` is the bottom row.
//! * vertex `(vx, vy)` is at world `(vx, vy)` and is shared by up to four cells.

use std::fmt;

/// Cell material.
///
/// `Empty` contributes no springs. `Rigid` and `Soft` add 6 springs (4 edges +
/// 2 diagonals) among their four corners. `ActH` / `ActV` do the same plus
/// modulate the rest length of their two horizontal / vertical edge springs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cell {
    Empty,
    Rigid,
    Soft,
    ActH,
    ActV,
}

impl Cell {
    pub const ALL: [Cell; 5] = [
        Cell::Empty,
        Cell::Rigid,
        Cell::Soft,
        Cell::ActH,
        Cell::ActV,
    ];

    pub fn is_solid(self) -> bool {
        matches!(
            self,
            Cell::Rigid | Cell::Soft | Cell::ActH | Cell::ActV
        )
    }

    pub fn is_actuator(self) -> bool {
        matches!(self, Cell::ActH | Cell::ActV)
    }

    /// Spring stiffness bucket the cell contributes to its edges/diagonals.
    pub fn stiffness_kind(self) -> Option<Stiffness> {
        match self {
            Cell::Empty => None,
            Cell::Rigid => Some(Stiffness::Rigid),
            Cell::Soft => Some(Stiffness::Soft),
            Cell::ActH | Cell::ActV => Some(Stiffness::Soft),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Cell::Empty => "Empty",
            Cell::Rigid => "Rigid",
            Cell::Soft => "Soft",
            Cell::ActH => "ActH",
            Cell::ActV => "ActV",
        }
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.label()) }
}

/// Edge / diagonal stiffness bucket.
///
/// `Rigid` (high-K) wins whenever a `Rigid` cell shares a spring. Otherwise the
/// spring is `Soft` (low-K). This keeps structural cells stiff while still
/// letting adjacent soft/actuator cells deform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stiffness {
    Soft,
    Rigid,
}

/// A rectangular voxel morphology.
#[derive(Debug, Clone)]
pub struct Morphology {
    pub w: u8,
    pub h: u8,
    pub cells: Vec<Cell>,
    /// Optional per-cell colour override (purely cosmetic — physics and
    /// evolution never read it). Parallel to `cells`.
    pub colors: Vec<Option<[u8; 3]>>,
}

impl Morphology {
    pub fn new(w: u8, h: u8) -> Self {
        let len = (w as usize) * (h as usize);
        Self { w, h, cells: vec![Cell::Empty; len], colors: vec![None; len] }
    }

    pub fn idx(&self, cx: u8, cy: u8) -> usize { (cy as usize) * (self.w as usize) + (cx as usize) }

    pub fn get(&self, cx: u8, cy: u8) -> Cell {
        if cx >= self.w || cy >= self.h { Cell::Empty } else { self.cells[self.idx(cx, cy)] }
    }

    pub fn set(&mut self, cx: u8, cy: u8, c: Cell) {
        if cx < self.w && cy < self.h {
            let i = self.idx(cx, cy);
            self.cells[i] = c;
        }
    }

    pub fn color_of(&self, cx: u8, cy: u8) -> Option<[u8; 3]> {
        if cx >= self.w || cy >= self.h { None } else { self.colors[self.idx(cx, cy)] }
    }

    pub fn set_color(&mut self, cx: u8, cy: u8, color: Option<[u8; 3]>) {
        if cx < self.w && cy < self.h {
            let i = self.idx(cx, cy);
            self.colors[i] = color;
        }
    }

    /// Count non-empty cells.
    pub fn n_solid(&self) -> usize {
        self.cells.iter().filter(|c| c.is_solid()).count()
    }

    /// Count actuators and list their (cell_x, cell_y, kind) tuples in a stable
    /// bottom-up, left-to-right order — the same order the controller uses to
    /// address per-actuator phase/amplitude genes.
    pub fn actuators(&self) -> Vec<(u8, u8, Cell)> {
        let mut out = Vec::new();
        for cy in 0..self.h {
            for cx in 0..self.w {
                let c = self.get(cx, cy);
                if c.is_actuator() { out.push((cx, cy, c)); }
            }
        }
        out
    }

    /// Iterator yielding (cell_x, cell_y, Cell).
    pub fn iter(&self) -> impl Iterator<Item = (u8, u8, Cell)> + '_ {
        let w = self.w;
        self.cells.iter().copied().enumerate().map(move |(i, c)| {
            let cx = (i % w as usize) as u8;
            let cy = (i / w as usize) as u8;
            (cx, cy, c)
        })
    }
}

/// Validation result for a morphology, used by the editor and the GA.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MorphologyError {
    Empty,
    NoActuator,
    Disconnected,
    TooLarge,
}

impl fmt::Display for MorphologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Empty => "design has no solid cells",
            Self::NoActuator => "design has no actuator",
            Self::Disconnected => "non-empty cells must be 4-connected to the actuator structure",
            Self::TooLarge => "design exceeds 32x32",
        };
        f.write_str(s)
    }
}

impl std::error::Error for MorphologyError {}

pub fn validate(m: &Morphology) -> Result<(), MorphologyError> {
    if m.w > 32 || m.h > 32 { return Err(MorphologyError::TooLarge); }
    if m.n_solid() == 0 { return Err(MorphologyError::Empty); }
    if m.actuators().is_empty() { return Err(MorphologyError::NoActuator); }

    // BFS from the first solid cell we encounter in bottom-up order and ensure
    // every solid cell is reachable.
    let mut seen = vec![false; m.cells.len()];
    let mut stack: Vec<(u8, u8)> = Vec::new();
    let mut started = false;
    'outer: for cy in 0..self_h(m) {
        for cx in 0..self_w(m) {
            if m.get(cx, cy).is_solid() {
                stack.push((cx, cy));
                started = true;
                break 'outer;
            }
        }
    }
    if !started { return Err(MorphologyError::Empty); }
    while let Some((cx, cy)) = stack.pop() {
        let i = m.idx(cx, cy);
        if seen[i] { continue; }
        seen[i] = true;
        let w = m.w;
        let h = m.h;
        let try_push = |stack: &mut Vec<(u8, u8)>, nx: i16, ny: i16| {
            if nx >= 0 && ny >= 0 && (nx as u8) < w && (ny as u8) < h {
                let (ux, uy) = (nx as u8, ny as u8);
                if !seen[m.idx(ux, uy)] && m.get(ux, uy).is_solid() {
                    stack.push((ux, uy));
                }
            }
        };
        try_push(&mut stack, cx as i16 - 1, cy as i16);
        try_push(&mut stack, cx as i16 + 1, cy as i16);
        try_push(&mut stack, cx as i16, cy as i16 - 1);
        try_push(&mut stack, cx as i16, cy as i16 + 1);
    }
    for (i, c) in m.cells.iter().enumerate() {
        if c.is_solid() && !seen[i] { return Err(MorphologyError::Disconnected); }
    }
    Ok(())
}

fn self_w(m: &Morphology) -> u8 { m.w }
fn self_h(m: &Morphology) -> u8 { m.h }

#[cfg(test)]
mod tests {
    use super::*;

    fn filled_with_actuator() -> Morphology {
        let mut m = Morphology::new(3, 3);
        for cy in 0..3 { for cx in 0..3 {
            if (cx, cy) == (1, 1) { m.set(cx, cy, Cell::ActV); } else { m.set(cx, cy, Cell::Rigid); }
        }}
        m
    }

    #[test]
    fn cell_methods_are_consistent() {
        assert!(!Cell::Empty.is_solid());
        assert!(Cell::Rigid.is_solid());
        assert!(!Cell::Rigid.is_actuator());
        assert!(Cell::ActH.is_actuator());
        assert!(Cell::ActV.is_actuator());
        assert_eq!(Cell::Rigid.stiffness_kind(), Some(Stiffness::Rigid));
        assert_eq!(Cell::Soft.stiffness_kind(), Some(Stiffness::Soft));
        assert_eq!(Cell::Empty.stiffness_kind(), None);
    }

    #[test]
    fn validate_ok_for_connected() {
        let m = filled_with_actuator();
        assert!(validate(&m).is_ok());
    }

    #[test]
    fn validate_rejects_empty() {
        let m = Morphology::new(3, 3);
        assert_eq!(validate(&m), Err(MorphologyError::Empty));
    }

    #[test]
    fn validate_rejects_no_actuator() {
        let mut m = Morphology::new(3, 3);
        m.set(0, 0, Cell::Rigid);
        m.set(1, 0, Cell::Rigid);
        assert_eq!(validate(&m), Err(MorphologyError::NoActuator));
    }

    #[test]
    fn validate_rejects_disconnected() {
        let mut m = Morphology::new(3, 3);
        m.set(0, 0, Cell::Rigid);
        m.set(1, 0, Cell::ActH); // connected to rigid at (0,0)
        m.set(2, 2, Cell::Rigid); // floating island
        assert_eq!(validate(&m), Err(MorphologyError::Disconnected));
    }

    #[test]
    fn actuators_in_stable_order() {
        let m = filled_with_actuator();
        let a = m.actuators();
        assert_eq!(a, vec![(1, 1, Cell::ActV)]);
    }

    #[test]
    fn rejects_oversize_grid() {
        let m = Morphology::new(33, 33);
        assert_eq!(validate(&m), Err(MorphologyError::TooLarge));
        let ok = Morphology::new(32, 32);
        assert_ne!(validate(&ok), Err(MorphologyError::TooLarge));
    }
}