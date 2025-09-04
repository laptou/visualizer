use clap::{Parser, Subcommand};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Configuration constants for audio processing
const SAMPLE_RATE: u32 = 44100;
const CHANNELS: u16 = 1; // Mono input
const BUFFER_SIZE: usize = 1024; // FFT window size
const HOP_SIZE: usize = 512; // Overlap between windows (50% overlap)
const ONSET_HISTORY_SIZE: usize = 100; // Number of onset strength values to keep for BPM calculation
const SILENCE_THRESHOLD: f32 = 0.001; // RMS threshold below which audio is considered silence
const BPM_SMOOTHING_WINDOW: usize = 5; // Number of BPM values to average for smoothing

/// Command-line interface for the audio visualizer
#[derive(Parser)]
#[command(name = "audio-visualizer")]
#[command(about = "Real-time audio onset detection and BPM analysis")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List available input devices and exit
    ListDevices,
    /// Use microphone or input device for analysis
    Input {
        /// Input device index (use list-devices to see options)
        #[arg(short, long)]
        device: Option<usize>,
    },
    /// Create virtual output device for audio routing (not yet implemented)
    Output,
}

/// Shared state for real-time audio processing
struct AudioProcessor {
    /// Ring buffer to store incoming audio samples
    sample_buffer: VecDeque<f32>,
    /// History of onset strength values for BPM calculation
    onset_history: VecDeque<f32>,
    /// Last time BPM was calculated and printed
    last_bpm_print: Instant,
    /// Current estimated BPM
    current_bpm: f32,
    /// History of BPM values for smoothing
    bpm_history: VecDeque<f32>,
    /// Previous magnitude spectrum for spectral flux calculation
    prev_magnitudes: Vec<f32>,
    /// Running RMS for silence detection
    rms_history: VecDeque<f32>,
}

impl AudioProcessor {
    fn new() -> Self {
        Self {
            sample_buffer: VecDeque::with_capacity(BUFFER_SIZE * 2),
            onset_history: VecDeque::with_capacity(ONSET_HISTORY_SIZE),
            last_bpm_print: Instant::now(),
            current_bpm: 0.0,
            bpm_history: VecDeque::with_capacity(BPM_SMOOTHING_WINDOW),
            prev_magnitudes: vec![0.0; BUFFER_SIZE / 2],
            rms_history: VecDeque::with_capacity(10),
        }
    }

    /// Add new audio samples to the buffer and process if we have enough data
    fn add_samples(&mut self, samples: &[f32]) {
        // Calculate RMS for this chunk of samples for silence detection
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        self.rms_history.push_back(rms);
        if self.rms_history.len() > 10 {
            self.rms_history.pop_front();
        }

        // Add new samples to the ring buffer
        for &sample in samples {
            self.sample_buffer.push_back(sample);
            
            // Keep buffer size manageable - remove old samples
            if self.sample_buffer.len() > BUFFER_SIZE * 2 {
                self.sample_buffer.pop_front();
            }
        }

        // Process audio if we have enough samples for analysis
        if self.sample_buffer.len() >= BUFFER_SIZE {
            self.process_audio_window();
        }

        // Print BPM once per second
        if self.last_bpm_print.elapsed() >= Duration::from_secs(1) {
            let avg_rms = self.rms_history.iter().sum::<f32>() / self.rms_history.len() as f32;
            
            if avg_rms < SILENCE_THRESHOLD {
                println!("Current BPM: N/A (silence detected)");
            } else if self.current_bpm > 0.0 {
                println!("Current BPM: {:.1}", self.current_bpm);
            } else {
                println!("Current BPM: N/A (analyzing...)");
            }
            self.last_bpm_print = Instant::now();
        }
    }

    /// Process a window of audio samples for onset detection
    fn process_audio_window(&mut self) {
        // Extract the most recent BUFFER_SIZE samples for analysis
        let window: Vec<f32> = self.sample_buffer
            .iter()
            .skip(self.sample_buffer.len().saturating_sub(BUFFER_SIZE))
            .copied()
            .collect();

        if window.len() < BUFFER_SIZE {
            return;
        }

        // Calculate onset strength for this window
        let onset_strength = self.calculate_onset_strength(&window);
        
        // Add to onset history for BPM calculation
        self.onset_history.push_back(onset_strength);
        if self.onset_history.len() > ONSET_HISTORY_SIZE {
            self.onset_history.pop_front();
        }

        // Calculate BPM from onset history if we have enough data
        if self.onset_history.len() >= 20 { // Need at least 20 frames for meaningful BPM
            self.current_bpm = self.calculate_bpm();
        }
    }

    /// Calculate onset strength using spectral flux method
    /// This measures the change in spectral energy between consecutive frames
    fn calculate_onset_strength(&mut self, window: &[f32]) -> f32 {
        // Apply Hanning window to reduce spectral leakage
        let windowed: Vec<f32> = window
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let hann_coeff = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (BUFFER_SIZE - 1) as f32).cos());
                sample * hann_coeff
            })
            .collect();

        // Compute FFT to get frequency domain representation
        let mut fft_input: Vec<rustfft::num_complex::Complex<f32>> = windowed
            .iter()
            .map(|&x| rustfft::num_complex::Complex::new(x, 0.0))
            .collect();

        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_forward(BUFFER_SIZE);
        fft.process(&mut fft_input);

        // Calculate magnitude spectrum
        let magnitudes: Vec<f32> = fft_input
            .iter()
            .take(BUFFER_SIZE / 2) // Only use positive frequencies
            .map(|complex| complex.norm())
            .collect();

        // Calculate spectral centroid (center of mass of the spectrum)
        // This gives us information about the "brightness" of the sound
        let total_magnitude: f32 = magnitudes.iter().sum();
        let weighted_sum: f32 = magnitudes
            .iter()
            .enumerate()
            .map(|(i, &mag)| i as f32 * mag)
            .sum();

        let spectral_centroid = if total_magnitude > 0.0 {
            weighted_sum / total_magnitude
        } else {
            0.0
        };

        // Calculate spectral flux (change in spectral energy between frames)
        // This is the proper way to detect onsets - compare current frame to previous
        let spectral_flux: f32 = magnitudes
            .iter()
            .zip(self.prev_magnitudes.iter())
            .map(|(&current, &previous)| (current - previous).max(0.0)) // Only positive changes
            .sum();

        // Update previous magnitudes for next frame
        self.prev_magnitudes.copy_from_slice(&magnitudes);

        // Combine spectral centroid and flux for onset strength
        // Spectral flux is the primary indicator, centroid adds brightness information
        spectral_flux + spectral_centroid * 0.001
    }

    /// Calculate BPM from the history of onset strength values
    fn calculate_bpm(&mut self) -> f32 {
        if self.onset_history.len() < 20 {
            return self.current_bpm;
        }

        // Convert onset history to Vec for FFT processing
        let onset_data: Vec<f32> = self.onset_history.iter().copied().collect();
        
        // Apply autocorrelation to find periodic patterns in onset strength
        let autocorr = self.autocorrelation(&onset_data);
        
        // Find the peak in autocorrelation (excluding the zero-lag peak)
        let mut max_peak = 0.0;
        let mut peak_lag = 1;
        
        // Search for peaks in a reasonable BPM range (60-180 BPM)
        // Each onset strength value represents HOP_SIZE samples at SAMPLE_RATE
        let samples_per_onset = HOP_SIZE as f32;
        let min_lag = ((60.0 / 180.0) * SAMPLE_RATE as f32 / samples_per_onset) as usize; // 180 BPM
        let max_lag = ((60.0 / 60.0) * SAMPLE_RATE as f32 / samples_per_onset) as usize;  // 60 BPM
        
        for lag in min_lag..max_lag.min(autocorr.len()) {
            if autocorr[lag] > max_peak {
                max_peak = autocorr[lag];
                peak_lag = lag;
            }
        }

        // Convert lag back to BPM
        // BPM = 60 * sample_rate / (lag * samples_per_onset)
        let raw_bpm = if peak_lag > 0 {
            60.0 * SAMPLE_RATE as f32 / (peak_lag as f32 * samples_per_onset)
        } else {
            return self.current_bpm; // Keep previous BPM if no peak found
        };

        // Add to BPM history for smoothing
        self.bpm_history.push_back(raw_bpm);
        if self.bpm_history.len() > BPM_SMOOTHING_WINDOW {
            self.bpm_history.pop_front();
        }

        // Return smoothed BPM (median filter to reduce outliers)
        if self.bpm_history.len() >= 3 {
            let mut sorted_bpm: Vec<f32> = self.bpm_history.iter().copied().collect();
            sorted_bpm.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_bpm[sorted_bpm.len() / 2] // Median
        } else {
            raw_bpm
        }
    }

    /// Calculate autocorrelation of the input signal
    /// This helps find periodic patterns in the onset strength
    fn autocorrelation(&self, signal: &[f32]) -> Vec<f32> {
        let n = signal.len();
        let mut result = vec![0.0; n];

        for lag in 0..n {
            let mut sum = 0.0;
            let mut count = 0;

            for i in 0..(n - lag) {
                sum += signal[i] * signal[i + lag];
                count += 1;
            }

            result[lag] = if count > 0 { sum / count as f32 } else { 0.0 };
        }

        result
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::ListDevices => {
            list_input_devices()?;
        }
        Commands::Input { device } => {
            run_input_mode(device)?;
        }
        Commands::Output => {
            println!("Output mode (virtual audio device) is not yet implemented.");
            println!("This would create a virtual audio device that applications could route audio to.");
            return Ok(());
        }
    }

    Ok(())
}

/// List all available input devices
fn list_input_devices() -> Result<(), Box<dyn std::error::Error>> {
    println!("Available input devices:");
    
    let host = cpal::default_host();
    let devices = host.input_devices()?;

    for (index, device) in devices.enumerate() {
        let device_name = device.name().unwrap_or_else(|_| "Unknown Device".to_string());
        
        // Try to get supported configurations
        match device.supported_input_configs() {
            Ok(mut configs) => {
                if let Some(config) = configs.next() {
                    println!("  {}: {} ({}Hz, {} channels)", 
                        index, 
                        device_name, 
                        config.max_sample_rate().0,
                        config.channels()
                    );
                } else {
                    println!("  {}: {} (no supported configs)", index, device_name);
                }
            }
            Err(_) => {
                println!("  {}: {} (config query failed)", index, device_name);
            }
        }
    }
    
    println!("\nUsage: audio-visualizer input --device <INDEX>");
    Ok(())
}

/// Run the input mode with the specified device
fn run_input_mode(device_index: Option<usize>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting real-time onset detection and BPM analysis...");
    
    let host = cpal::default_host();
    
    // Get the specified device or default
    let device = if let Some(index) = device_index {
        let devices: Vec<_> = host.input_devices()?.collect();
        devices.into_iter()
            .nth(index)
            .ok_or_else(|| format!("Input device index {} not found", index))?
    } else {
        host.default_input_device()
            .ok_or("No default input device available")?
    };

    println!("Using input device: {}", device.name()?);

    // Get supported input configurations
    let mut supported_configs = device.supported_input_configs()?;
    let supported_config = supported_configs
        .next()
        .ok_or("No supported input config")?
        .with_max_sample_rate();

    println!("Sample rate: {}, Channels: {}", supported_config.sample_rate().0, supported_config.channels());

    // Create stream configuration
    let config = StreamConfig {
        channels: CHANNELS,
        sample_rate: cpal::SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    // Create shared processor state
    let processor = Arc::new(Mutex::new(AudioProcessor::new()));

    // Build input stream with callback for processing audio data
    let stream = build_input_stream(&device, &config, &supported_config, processor.clone())?;

    // Start the audio stream
    stream.play()?;
    println!("Audio stream started. Listening for audio input...");
    println!("Press Ctrl+C to stop.");

    // Keep the main thread alive
    loop {
        thread::sleep(Duration::from_millis(100));
    }
}

/// Build an input stream that handles different sample formats
fn build_input_stream(
    device: &Device,
    config: &StreamConfig,
    supported_config: &cpal::SupportedStreamConfig,
    processor: Arc<Mutex<AudioProcessor>>,
) -> Result<Stream, Box<dyn std::error::Error>> {
    let stream = match supported_config.sample_format() {
        SampleFormat::F32 => {
            let processor_clone = processor.clone();
            device.build_input_stream(
                config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut proc) = processor_clone.lock() {
                        proc.add_samples(data);
                    }
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            )?
        },
        SampleFormat::I16 => {
            let processor_clone = processor.clone();
            device.build_input_stream(
                config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    // Convert i16 samples to f32
                    let float_data: Vec<f32> = data
                        .iter()
                        .map(|&sample| sample as f32 / i16::MAX as f32)
                        .collect();
                    
                    if let Ok(mut proc) = processor_clone.lock() {
                        proc.add_samples(&float_data);
                    }
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            )?
        },
        SampleFormat::U16 => {
            let processor_clone = processor.clone();
            device.build_input_stream(
                config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    // Convert u16 samples to f32
                    let float_data: Vec<f32> = data
                        .iter()
                        .map(|&sample| (sample as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0))
                        .collect();
                    
                    if let Ok(mut proc) = processor_clone.lock() {
                        proc.add_samples(&float_data);
                    }
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            )?
        },
        sample_format => {
            return Err(format!("Unsupported sample format: {}", sample_format).into());
        }
    };

    Ok(stream)
}
