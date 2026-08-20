//! Binary entry point.
//!
//! Two modes:
//! * `--headless` — runs the headless CLI (clap-parsed). Never touches eframe.
//! * default (no flag) — launches the eframe / egui GUI.
//!
//! When the binary is built without the `gui` feature, only the CLI mode is
//! available. Running without `--headless` then prints an error and exits 2.

fn main() -> std::process::ExitCode {
    // Detect --headless by parsing args with clap before deciding the GUI path.
    // This keeps the GUI feature opt-in: building with `--no-default-features`
    // skips the eframe code entirely.
    let args: Vec<String> = std::env::args().collect();
    let headless = args.iter().any(|a| a == "--headless");

    if headless {
        return evogym_labs::cli::run();
    }

    // GUI path
    #[cfg(feature = "gui")]
    {
        let viewport = egui::ViewportBuilder::default().with_title("EvoGym-Labs").with_inner_size([1280.0, 800.0]);
        let options = eframe::NativeOptions { viewport, ..Default::default() };
        match eframe::run_native(
            "EvoGym-Labs",
            options,
            Box::new(|cc| Ok(Box::new(evogym_labs::app::App::new(cc)))),
        ) {
            Ok(()) => std::process::ExitCode::from(0),
            Err(e) => {
                eprintln!("eframe failed: {e}");
                std::process::ExitCode::from(1)
            }
        }
    }

    #[cfg(not(feature = "gui"))]
    {
        let _ = args;
        eprintln!("This build does not include the GUI; pass --headless to use the CLI.");
        std::process::ExitCode::from(2)
    }
}