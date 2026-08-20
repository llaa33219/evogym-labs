//! EvoGym-labs library — voxel physics, evolution, render, CLI.
//!
//! Modules are organised so that `--headless` builds (no default `gui` feature)
//! still link cleanly: the GUI crate `eframe`/`egui` are only pulled in when
//! the `gui` feature is enabled.

pub mod cli;
pub mod controller;
pub mod editor;
pub mod evolution;
pub mod physics;
pub mod render;
pub mod tasks;
pub mod text;
pub mod video;
pub mod voxel;

#[cfg(feature = "gui")]
pub mod app;