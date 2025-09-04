use anyhow::{Result, anyhow};
use clap::Parser;
use std::sync::{Arc, Mutex};
use std::thread;
use visualizer::audio;
use visualizer::cli;
use visualizer::shared;
use visualizer::ui;

fn main() -> Result<()> {
    // initialize tracing subscriber for logging
    tracing_subscriber::fmt::init();

    let cli = cli::Cli::parse();

    if cli.list_devices {
        audio::list_input_devices()?;
        return Ok(());
    }

    if cli.list_configs {
        if cli.device.is_none() {
            return Err(anyhow!("--list-configs requires --device to be specified"));
        }
        audio::list_device_configs(cli.device.unwrap())?;
        return Ok(());
    }

    // shared state for audio (producer) and ui (consumer)
    let shared = Arc::new(Mutex::new(shared::SharedState::new()));

    // spawn audio thread
    let audio_shared = shared.clone();
    let cli_copy = cli.clone();
    thread::Builder::new().name("audio".into()).spawn(move || {
        if let Err(e) = audio::run_input_mode(cli_copy.device, cli_copy.config, audio_shared) {
            tracing::error!("audio thread error: {}", e);
        }
    })?;

    // run ui on main thread (wgpu/winit prefer main thread on macos)
    pollster::block_on(ui::run_ui(shared))?;
    Ok(())
}
