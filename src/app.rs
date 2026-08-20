//! egui / eframe GUI for the voxel editor + evolution runner.
//!
//! Layout:
//! * **Left panel** — voxel palette (Empty/eraser + 4 cell types), grid-size
//!   selector, Clear button.
//! * **Centre** — paint area: click & drag to paint the selected cell type
//!   onto the grid. The whole morphology is shown at the centre.
//! * **Right panel** — evolution config (population, eval seconds, mode, task),
//!   editable video plan (add / remove rows), output path, RUN button. While
//!   the worker thread is running, a progress bar and status text replace
//!   the RUN button.
//!
//! The worker thread owns its own RNG and communicates progress to the GUI
//! via an `std::sync::mpsc::Sender<ProgressMsg>`. The GUI polls the channel
//! each frame in `update()`. On completion the GUI shows the final output
//! path and any error.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;

use eframe::egui;

use crate::cli::default_demo_robot;
use crate::editor::{clear as editor_clear, paint as editor_paint, paint_color as editor_paint_color, resize as editor_resize, screen_to_cell_y as editor_flip_y};
use crate::evolution::{run as evolve, EvolutionConfig, Mode, Progress as GaProgress};
use crate::tasks::TaskKind;
use crate::video::{render_plan, ClipSpec};
use crate::voxel::{validate, Cell, Morphology};

/// One entry the user types into the video plan table.
#[derive(Debug, Clone)]
struct PlanRow {
    generation: usize,
    seconds: f32,
    is_final: bool,
}

fn preset_default() -> Vec<PlanRow> {
    let gens = [0, 10, 30, 50, 100, 300, 500, 1000];
    gens.iter()
        .map(|&g| PlanRow { generation: g, seconds: if g == 1000 { 10.0 } else { 5.0 }, is_final: g == 1000 })
        .collect()
}

fn preset_long() -> Vec<PlanRow> {
    let gens = [0, 10, 100, 500, 1000, 3000, 5000, 10000];
    gens.iter()
        .map(|&g| PlanRow { generation: g, seconds: if g == 10000 { 10.0 } else { 5.0 }, is_final: g == 10000 })
        .collect()
}

fn preset_short() -> Vec<PlanRow> {
    let gens = [0, 3, 5, 10, 20, 30, 50, 100];
    gens.iter()
        .map(|&g| PlanRow { generation: g, seconds: if g == 100 { 10.0 } else { 5.0 }, is_final: g == 100 })
        .collect()
}

/// Progress / completion messages from the worker thread.
#[derive(Debug, Clone)]
enum ProgressMsg {
    Started,
    Generation { generation: usize, total: usize, best: f32, mean: f32 },
    Final(Result<PathBuf, String>),
}

/// Shared GUI state — everything the GUI reads + the channels it talks on.
pub struct App {
    morph: Morphology,
    brush: Cell,
    grid_w: u8,
    grid_h: u8,
    colour_mode: bool,
    brush_colour: [u8; 3],
    colour_erase: bool,

    population: usize,
    eval_secs: f32,
    mode: Mode,
    task: TaskKind,

    plan: Vec<PlanRow>,
    out_path: String,
    zoom: f32,

    progress: Vec<ProgressMsg>,
    running: bool,
    progress_total: usize,
    progress_current: usize,
    progress_best: f32,
    last_result: Option<Result<PathBuf, String>>,
    rx: Option<Receiver<ProgressMsg>>,
    tx: Sender<ProgressMsg>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let _ = cc;
        let mut s = Self::default();
        let (tx, rx) = std::sync::mpsc::channel();
        s.tx = tx;
        s.rx = Some(rx);
        s
    }

    fn start_run(&mut self) {
        if self.running { return; }
        // Validate plan
        for r in &self.plan {
            if r.generation == usize::MAX || r.seconds <= 0.0 {
                self.last_result = Some(Err("plan rows must have a valid generation and seconds > 0".into()));
                return;
            }
        }
        if self.plan.is_empty() {
            self.last_result = Some(Err("plan must contain at least one row".into()));
            return;
        }
        // Validate morphology
        if let Err(e) = validate(&self.morph) {
            self.last_result = Some(Err(format!("design invalid: {e}")));
            return;
        }

        let cfg = EvolutionConfig {
            mode: self.mode,
            task: self.task,
            population: self.population.max(2),
            eval_secs: self.eval_secs.max(0.05),
            morph_size: (self.morph.w, self.morph.h),
            ..EvolutionConfig::defaults()
        };
        let max_gen = self.plan.iter().map(|p| p.generation).max().unwrap_or(0);
        let seed_morph = self.morph.clone();
        let plan: Vec<ClipSpec> = self.plan.iter().map(|p| ClipSpec { generation: p.generation, seconds: p.seconds.max(0.05), is_final: p.is_final }).collect();
        let trimmed = self.out_path.trim();
        if trimmed.is_empty() {
            self.last_result = Some(Err("output path must not be empty".into()));
            return;
        }
        let out = PathBuf::from(trimmed);
        let task = self.task;
        let tx = self.tx.clone();

        self.progress.clear();
        self.progress.push(ProgressMsg::Started);
        self.running = true;
        self.progress_total = max_gen + 1;
        self.progress_current = 0;
        self.progress_best = 0.0;
        self.last_result = None;

        thread::spawn(move || {
            fn last_progress(tx: &Sender<ProgressMsg>, p: GaProgress) {
                match p {
                    GaProgress::Started { .. } => { let _ = tx.send(ProgressMsg::Started); }
                    GaProgress::Generation { generation, total, best, mean } => {
                        let _ = tx.send(ProgressMsg::Generation { generation, total, best, mean });
                    }
                    GaProgress::Done { .. } => { /* forwarded as Final */ }
                    GaProgress::Error(e) => { let _ = tx.send(ProgressMsg::Final(Err(e))); }
                }
            }
            let result = evolve(&cfg, seed_morph, max_gen + 1, None, |p| last_progress(&tx, p));
            match render_plan(&plan, &result.history, task, &out) {
                Ok(out_meta) => { let _ = tx.send(ProgressMsg::Final(Ok(out_meta.output_path))); }
                Err(e) => { let _ = tx.send(ProgressMsg::Final(Err(format!("video: {e}")))); }
            }
        });
    }

    fn drain_progress(&mut self) {
        let Some(rx) = self.rx.as_ref() else { return; };
        while let Ok(msg) = rx.try_recv() {
            match &msg {
                ProgressMsg::Generation { generation, total, best, .. } => {
                    self.progress_total = *total;
                    self.progress_current = *generation + 1;
                    self.progress_best = *best;
                }
                ProgressMsg::Final(result) => {
                    self.running = false;
                    self.last_result = Some(result.clone());
                }
                ProgressMsg::Started => {}
            }
            self.progress.push(msg);
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_progress();

        egui::SidePanel::left("palette").min_width(170.0).show(ctx, |ui| {
            ui.heading("Palette");
            ui.add_space(4.0);
            let colours: [(Cell, egui::Color32, &str); 5] = [
                (Cell::Empty, egui::Color32::from_rgb(30, 30, 30), "Empty (eraser)"),
                (Cell::Rigid, egui::Color32::from_rgb(140, 140, 150), "Rigid"),
                (Cell::Soft, egui::Color32::from_rgb(80, 200, 110), "Soft"),
                (Cell::ActH, egui::Color32::from_rgb(220, 70, 70), "Actuator H"),
                (Cell::ActV, egui::Color32::from_rgb(70, 100, 220), "Actuator V"),
            ];
            for (cell, col, label) in colours.iter() {
                let selected = self.brush == *cell;
                let btn = egui::Button::new(*label).fill(if selected { egui::Color32::from_rgb(60, 60, 80) } else { *col });
                if ui.add(btn).clicked() { self.brush = *cell; }
            }
            ui.add_space(8.0);
            ui.heading("Grid");
            ui.horizontal(|ui| {
                ui.label("W");
                ui.add(egui::DragValue::new(&mut self.grid_w).range(1..=32));
                ui.label("H");
                ui.add(egui::DragValue::new(&mut self.grid_h).range(1..=32));
            });
            if ui.button("Apply size").clicked() {
                editor_resize(&mut self.morph, self.grid_w, self.grid_h);
                self.grid_w = self.morph.w;
                self.grid_h = self.morph.h;
            }
            if ui.button("Clear").clicked() { editor_clear(&mut self.morph); }
            ui.add_space(8.0);
            ui.heading("Colour");
            ui.checkbox(&mut self.colour_mode, "Paint colour (type stays)");
            ui.horizontal(|ui| {
                ui.color_edit_button_srgb(&mut self.brush_colour);
                ui.checkbox(&mut self.colour_erase, "Erase colour");
            });
        });

        egui::SidePanel::right("config").min_width(260.0).show(ctx, |ui| {
            ui.heading("Evolution");
            ui.horizontal(|ui| { ui.label("Pop"); ui.add(egui::DragValue::new(&mut self.population).range(2..=128)); });
            ui.horizontal(|ui| { ui.label("Eval s"); ui.add(egui::DragValue::new(&mut self.eval_secs).range(0.05..=60.0)); });
            ui.horizontal(|ui| {
                egui::ComboBox::from_label("Mode").selected_text(self.mode.label()).show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.mode, Mode::Full, Mode::Full.label());
                    ui.selectable_value(&mut self.mode, Mode::ControllerOnly, Mode::ControllerOnly.label());
                });
            });
            ui.horizontal(|ui| {
                egui::ComboBox::from_label("Task").selected_text(self.task.label()).show_ui(ui, |ui| {
                    for t in TaskKind::ALL {
                        ui.selectable_value(&mut self.task, t, t.label());
                    }
                });
            });

            ui.add_space(8.0);
            ui.heading("Video plan");
            ui.horizontal(|ui| {
                ui.label("Presets:");
                if ui.button("Default (≤1k)").clicked() { self.plan = preset_default(); }
                if ui.button("Long (≤10k)").clicked() { self.plan = preset_long(); }
                if ui.button("Short (≤100)").clicked() { self.plan = preset_short(); }
            });
            let mut to_remove: Option<usize> = None;
            for (i, row) in self.plan.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label("Gen");
                    ui.add(egui::DragValue::new(&mut row.generation).range(0..=10_000));
                    ui.label("s");
                    ui.add(egui::DragValue::new(&mut row.seconds).range(0.05..=120.0));
                    ui.checkbox(&mut row.is_final, "Final");
                    if ui.button("✕").clicked() { to_remove = Some(i); }
                });
            }
            if let Some(i) = to_remove { self.plan.remove(i); }
            if ui.button("+ Add row").clicked() {
                let next_gen = self.plan.iter().map(|r| r.generation).max().map(|g| g + 1).unwrap_or(0);
                self.plan.push(PlanRow { generation: next_gen, seconds: 2.0, is_final: false });
            }

            ui.add_space(8.0);
            ui.label("Output");
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.out_path).desired_width(180.0));
                if ui.button("…").clicked()
                    && let Some(p) = rfd_open()
                {
                    self.out_path = p.display().to_string();
                }
            });

            ui.add_space(12.0);
            if self.running {
                let frac = if self.progress_total > 0 { self.progress_current as f32 / self.progress_total as f32 } else { 0.0 };
                ui.add(egui::ProgressBar::new(frac.clamp(0.0, 1.0)).show_percentage());
                ui.label(format!("gen {}/{}  best {:.2} m", self.progress_current, self.progress_total, self.progress_best));
            } else {
                if ui.button("RUN").clicked() { self.start_run(); }
            }

            ui.add_space(12.0);
            egui::ScrollArea::vertical().max_height(140.0).show(ui, |ui| {
                for msg in &self.progress {
                    match msg {
                        ProgressMsg::Started => { ui.label("started"); }
                        ProgressMsg::Generation { generation, total, best, mean } => {
                            ui.label(format!("gen {}/{}  best {:.2}  mean {:.2}", generation + 1, total, best, mean));
                        }
                        ProgressMsg::Final(Ok(p)) => { ui.colored_label(egui::Color32::GREEN, format!("done: {}", p.display())); }
                        ProgressMsg::Final(Err(e)) => { ui.colored_label(egui::Color32::RED, e); }
                    }
                }
            });

            if let Some(res) = &self.last_result {
                ui.add_space(8.0);
                match res {
                    Ok(p) => { ui.colored_label(egui::Color32::GREEN, format!("output: {}", p.display())); }
                    Err(e) => { ui.colored_label(egui::Color32::RED, e); }
                }
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Design");
                ui.separator();
                if ui.button("−").clicked() { self.zoom = (self.zoom / 1.25).clamp(0.2, 3.0); }
                ui.label(format!("{:.0}%", self.zoom * 100.0));
                if ui.button("+").clicked() { self.zoom = (self.zoom * 1.25).clamp(0.2, 3.0); }
                if ui.button("Reset").clicked() { self.zoom = 0.55; }
                ui.label("(Ctrl+wheel)");
            });
            let wheel = ui.input(|i| i.smooth_scroll_delta.y);
            let ctrl = ui.input(|i| i.modifiers.ctrl);
            if ctrl && wheel != 0.0 {
                self.zoom = (self.zoom * (1.0 + wheel * 0.001)).clamp(0.2, 3.0);
            }

            let avail = ui.available_size();
            let max_dim = self.morph.w.max(self.morph.h) as f32;
            let base = (avail.x.min(avail.y) / (max_dim + 1.0)).max(8.0);
            let cell_size = (base * self.zoom).max(4.0);
            let grid_px = egui::vec2(cell_size * self.morph.w as f32, cell_size * self.morph.h as f32);

            let (outer, _) = ui.allocate_exact_size(avail, egui::Sense::hover());
            let origin = egui::pos2(
                outer.min.x + (outer.width() - grid_px.x) * 0.5,
                outer.min.y + (outer.height() - grid_px.y) * 0.5,
            );
            let grid_rect = egui::Rect::from_min_size(origin, grid_px);
            let painter = ui.painter_at(outer);
            painter.rect_filled(grid_rect, 0.0, egui::Color32::from_gray(20));

            // Screen row 0 is the top of the grid = physics row h-1 (see
            // editor::screen_to_cell_y).
            for sy in 0..self.morph.h {
                let cy = editor_flip_y(sy, self.morph.h);
                for cx in 0..self.morph.w {
                    let c = self.morph.get(cx, cy);
                    let pos = egui::pos2(origin.x + cx as f32 * cell_size, origin.y + sy as f32 * cell_size);
                    let r = egui::Rect::from_min_size(pos, egui::vec2(cell_size, cell_size));
                    let colour = match self.morph.color_of(cx, cy) {
                        Some([r, g, b]) => egui::Color32::from_rgb(r, g, b),
                        None => match c {
                            Cell::Empty => egui::Color32::from_gray(25),
                            Cell::Rigid => egui::Color32::from_rgb(140, 140, 150),
                            Cell::Soft => egui::Color32::from_rgb(80, 200, 110),
                            Cell::ActH => egui::Color32::from_rgb(220, 70, 70),
                            Cell::ActV => egui::Color32::from_rgb(70, 100, 220),
                        },
                    };
                    painter.rect_filled(r.shrink(1.0), 0.0, colour);
                }
            }
            let line_stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_gray(60));
            for cx in 0..=(self.morph.w) {
                let x = origin.x + cx as f32 * cell_size;
                painter.line_segment([egui::pos2(x, origin.y), egui::pos2(x, origin.y + grid_px.y)], line_stroke);
            }
            for sy in 0..=(self.morph.h) {
                let y = origin.y + sy as f32 * cell_size;
                painter.line_segment([egui::pos2(origin.x, y), egui::pos2(origin.x + grid_px.x, y)], line_stroke);
            }

            let painting = ui.input(|i| i.pointer.primary_down());
            let pointer_pos = ui.input(|i| i.pointer.latest_pos());
            if painting && let Some(pos) = pointer_pos && grid_rect.contains(pos) {
                let local = pos - origin;
                let cx = (local.x / cell_size) as i32;
                let sy = (local.y / cell_size) as i32;
                if cx >= 0 && sy >= 0 && (cx as u8) < self.morph.w && (sy as u8) < self.morph.h {
                    let cy = editor_flip_y(sy as u8, self.morph.h);
                    if self.colour_mode {
                        let c = if self.colour_erase { None } else { Some(self.brush_colour) };
                        editor_paint_color(&mut self.morph, cx as u8, cy, c);
                    } else {
                        editor_paint(&mut self.morph, cx as u8, cy, self.brush);
                    }
                }
            }
            // status
            ui.add_space(8.0);
            match validate(&self.morph) {
                Ok(()) => { ui.colored_label(egui::Color32::GREEN, "design OK"); }
                Err(e) => { ui.colored_label(egui::Color32::YELLOW, format!("design: {e}")); }
            }
        });


        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
}

/// Stub file-dialog opener — returns None (no native dialog). The GUI still
/// accepts text input of the output path. Real file dialogs are not part of
/// the deliverable.
#[cfg(feature = "gui")]
fn rfd_open() -> Option<PathBuf> { None }

impl Default for App {
    fn default() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self {
            morph: default_demo_robot(),
            brush: Cell::ActV,
            colour_mode: false,
            brush_colour: [255, 200, 40],
            colour_erase: false,
            grid_w: 5,
            grid_h: 5,
            population: 32,
            eval_secs: 10.0,
            mode: Mode::Full,
            task: TaskKind::Walker,
            plan: preset_default(),
            out_path: String::from("/tmp/evogym_out.mp4"),
            zoom: 0.55,
            progress: Vec::new(),
            running: false,
            progress_total: 0,
            progress_current: 0,
            progress_best: 0.0,
            last_result: None,
            rx: Some(rx),
            tx,
        }
    }
}