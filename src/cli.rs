use clap::Parser;

/// command-line interface for the audio visualizer
#[derive(Parser, Debug, Clone)]
#[command(name = "visualizer")]
#[command(about = "real-time audio onset detection and bpm analysis")]
pub struct Cli {
    /// input device index (use --list-devices to see options)
    #[arg(short, long)]
    pub device: Option<usize>,

    /// config index for the selected device (use --list-configs to see options)
    #[arg(short, long)]
    pub config: Option<usize>,

    /// list available input devices and exit
    #[arg(long)]
    pub list_devices: bool,

    /// list available configs for a device and exit (requires --device)
    #[arg(long)]
    pub list_configs: bool,
}
