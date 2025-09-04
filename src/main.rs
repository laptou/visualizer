use anyhow::Result;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{error, info};

/// configuration constants for audio processing
const SAMPLE_RATE: u32 = 44100;
const CHANNELS: u16 = 1; // mono input
const BUFFER_SIZE: usize = 1024; // fft window size
const HOP_SIZE: usize = 512; // overlap between windows (50% overlap)
const ONSET_HISTORY_SIZE: usize = 100; // number of onset strength values to keep for bpm calculation
const SILENCE_THRESHOLD: f32 = 0.001; // rms threshold below which audio is considered silence
const BPM_SMOOTHING_WINDOW: usize = 10; // number of bpm values to average for smoothing
const LOW_FREQ_WEIGHT: f32 = 2.0; // weight boost for low frequencies (bass/kick drums)
const HIGH_FREQ_CUTOFF: f32 = 0.3; // fraction of spectrum to consider (focus on lower frequencies)

/// command-line interface for the audio visualizer
#[derive(Parser)]
#[command(name = "audio-visualizer")]
#[command(about = "real-time audio onset detection and bpm analysis")]
struct Cli {
    /// input device index (use --list-devices to see options)
    #[arg(short, long)]
    device: Option<usize>,
    
    /// list available input devices and exit
    #[arg(long)]
    list_devices: bool,
}

/// shared state for real-time audio processing
struct AudioProcessor {
    /// ring buffer to store incoming audio samples
    sample_buffer: VecDeque<f32>,
    /// history of onset strength values for bpm calculation
    onset_history: VecDeque<f32>,
    /// last time bpm was calculated and printed
    last_bpm_print: Instant,
    /// current estimated bpm
    current_bpm: f32,
    /// history of bpm values for smoothing
    bpm_history: VecDeque<f32>,
    /// previous magnitude spectrum for spectral flux calculation
    prev_magnitudes: Vec<f32>,
    /// running rms for silence detection
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

    /// add new audio samples to the buffer and process if we have enough data
    fn add_samples(&mut self, samples: &[f32]) {
        // calculate rms for this chunk of samples for silence detection
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        self.rms_history.push_back(rms);
        if self.rms_history.len() > 10 {
            self.rms_history.pop_front();
        }

        // add new samples to the ring buffer
        for &sample in samples {
            self.sample_buffer.push_back(sample);

            // keep buffer size manageable - remove old samples
            if self.sample_buffer.len() > BUFFER_SIZE * 2 {
                self.sample_buffer.pop_front();
            }
        }

        // process audio if we have enough samples for analysis
        if self.sample_buffer.len() >= BUFFER_SIZE {
            self.process_audio_window();
        }

        // print bpm once per second
        if self.last_bpm_print.elapsed() >= Duration::from_secs(1) {
            let avg_rms = self.rms_history.iter().sum::<f32>() / self.rms_history.len() as f32;

            if avg_rms < SILENCE_THRESHOLD {
                info!("current bpm: n/a (silence detected)");
            } else if self.current_bpm > 0.0 {
                info!("current bpm: {:.1}", self.current_bpm);
            } else {
                info!("current bpm: n/a (analyzing...)");
            }
            self.last_bpm_print = Instant::now();
        }
    }

    /// process a window of audio samples for onset detection
    fn process_audio_window(&mut self) {
        // extract the most recent buffer_size samples for analysis
        let window: Vec<f32> = self
            .sample_buffer
            .iter()
            .skip(self.sample_buffer.len().saturating_sub(BUFFER_SIZE))
            .copied()
            .collect();

        if window.len() < BUFFER_SIZE {
            return;
        }

        // calculate onset strength for this window
        let onset_strength = self.calculate_onset_strength(&window);

        // add to onset history for bpm calculation
        self.onset_history.push_back(onset_strength);
        if self.onset_history.len() > ONSET_HISTORY_SIZE {
            self.onset_history.pop_front();
        }

        // calculate bpm from onset history if we have enough data
        if self.onset_history.len() >= 20 {
            // need at least 20 frames for meaningful bpm
            self.current_bpm = self.calculate_bpm();
        }
    }

    /// calculate onset strength using frequency-weighted spectral flux method
    /// this measures the change in spectral energy between consecutive frames,
    /// with emphasis on low frequencies where beat information is most reliable
    fn calculate_onset_strength(&mut self, window: &[f32]) -> f32 {
        // apply hanning window to reduce spectral leakage
        let windowed: Vec<f32> = window
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let hann_coeff = 0.5
                    * (1.0
                        - (2.0 * std::f32::consts::PI * i as f32 / (BUFFER_SIZE - 1) as f32).cos());
                sample * hann_coeff
            })
            .collect();

        // compute fft to get frequency domain representation
        let mut fft_input: Vec<rustfft::num_complex::Complex<f32>> = windowed
            .iter()
            .map(|&x| rustfft::num_complex::Complex::new(x, 0.0))
            .collect();

        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_forward(BUFFER_SIZE);
        fft.process(&mut fft_input);

        // calculate magnitude spectrum - focus on lower frequencies for beat detection
        let freq_bins_to_use = ((BUFFER_SIZE / 2) as f32 * HIGH_FREQ_CUTOFF) as usize;
        let magnitudes: Vec<f32> = fft_input
            .iter()
            .take(freq_bins_to_use) // only use lower frequencies
            .map(|complex| complex.norm())
            .collect();

        // calculate frequency-weighted spectral flux
        // give more weight to low frequencies (bass/kick drums) and less to high frequencies
        let weighted_spectral_flux: f32 = magnitudes
            .iter()
            .zip(self.prev_magnitudes.iter())
            .enumerate()
            .map(|(bin, (&current, &previous))| {
                let flux = (current - previous).max(0.0); // only positive changes
                
                // create frequency weighting: boost low frequencies, reduce high frequencies
                let freq_ratio = bin as f32 / magnitudes.len() as f32;
                let weight = if freq_ratio < 0.2 {
                    // boost bass frequencies (0-20% of spectrum)
                    LOW_FREQ_WEIGHT
                } else if freq_ratio < 0.5 {
                    // normal weight for mid frequencies (20-50% of spectrum)
                    1.0
                } else {
                    // reduce weight for higher frequencies (50%+ of spectrum)
                    0.5
                };
                
                flux * weight
            })
            .sum();

        // update previous magnitudes for next frame (resize if needed)
        if self.prev_magnitudes.len() != magnitudes.len() {
            self.prev_magnitudes.resize(magnitudes.len(), 0.0);
        }
        self.prev_magnitudes.copy_from_slice(&magnitudes);

        // return weighted spectral flux as onset strength
        // removed spectral centroid as it was adding noise from high frequencies
        weighted_spectral_flux
    }

    /// calculate bpm from the history of onset strength values
    fn calculate_bpm(&mut self) -> f32 {
        if self.onset_history.len() < 20 {
            return self.current_bpm;
        }

        // convert onset history to vec for fft processing
        let onset_data: Vec<f32> = self.onset_history.iter().copied().collect();

        // apply autocorrelation to find periodic patterns in onset strength
        let autocorr = self.autocorrelation(&onset_data);

        // find the peak in autocorrelation (excluding the zero-lag peak)
        let mut max_peak = 0.0;
        let mut peak_lag = 1;

        // search for peaks in a reasonable bpm range (60-180 bpm)
        // each onset strength value represents hop_size samples at sample_rate
        let samples_per_onset = HOP_SIZE as f32;
        let min_lag = ((60.0 / 180.0) * SAMPLE_RATE as f32 / samples_per_onset) as usize; // 180 bpm
        let max_lag = ((60.0 / 60.0) * SAMPLE_RATE as f32 / samples_per_onset) as usize; // 60 bpm

        for lag in min_lag..max_lag.min(autocorr.len()) {
            if autocorr[lag] > max_peak {
                max_peak = autocorr[lag];
                peak_lag = lag;
            }
        }

        // convert lag back to bpm
        // bpm = 60 * sample_rate / (lag * samples_per_onset)
        let raw_bpm = if peak_lag > 0 {
            60.0 * SAMPLE_RATE as f32 / (peak_lag as f32 * samples_per_onset)
        } else {
            return self.current_bpm; // keep previous bpm if no peak found
        };

        // add to bpm history for smoothing
        self.bpm_history.push_back(raw_bpm);
        if self.bpm_history.len() > BPM_SMOOTHING_WINDOW {
            self.bpm_history.pop_front();
        }

        // return smoothed bpm (median filter to reduce outliers)
        if self.bpm_history.len() >= 3 {
            let mut sorted_bpm: Vec<f32> = self.bpm_history.iter().copied().collect();
            sorted_bpm.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_bpm[sorted_bpm.len() / 2] // median
        } else {
            raw_bpm
        }
    }

    /// calculate autocorrelation of the input signal
    /// this helps find periodic patterns in the onset strength
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

fn main() -> Result<()> {
    // initialize tracing subscriber for logging
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();

    if cli.list_devices {
        list_input_devices()?;
    } else {
        run_input_mode(cli.device)?;
    }

    Ok(())
}

/// list all available input devices
fn list_input_devices() -> Result<()> {
    info!("available input devices:");

    let host = cpal::default_host();
    let devices = host.input_devices()?;

    for (index, device) in devices.enumerate() {
        let device_name = device
            .name()
            .unwrap_or_else(|_| "Unknown Device".to_string());

        // Try to get supported configurations
        match device.supported_input_configs() {
            Ok(mut configs) => {
                if let Some(config) = configs.next() {
                    info!(
                        "  {}: {} ({}hz, {} channels)",
                        index,
                        device_name,
                        config.max_sample_rate().0,
                        config.channels()
                    );
                } else {
                    info!("  {}: {} (no supported configs)", index, device_name);
                }
            }
            Err(_) => {
                info!("  {}: {} (config query failed)", index, device_name);
            }
        }
    }

    info!("\nusage: audio-visualizer --device <index>");
    Ok(())
}

/// run the input mode with the specified device
fn run_input_mode(device_index: Option<usize>) -> Result<()> {
    info!("starting real-time onset detection and bpm analysis...");

    let host = cpal::default_host();

    // Get the specified device or default
    let device = if let Some(index) = device_index {
        let devices: Vec<_> = host.input_devices()?.collect();
        devices
            .into_iter()
            .nth(index)
            .ok_or_else(|| anyhow::anyhow!("input device index {} not found", index))?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("no default input device available"))?
    };

    info!("using input device: {}", device.name()?);

    // get supported input configurations
    let mut supported_configs = device.supported_input_configs()?;
    let supported_config = supported_configs
        .next()
        .ok_or_else(|| anyhow::anyhow!("no supported input config"))?
        .with_max_sample_rate();

    info!(
        "sample rate: {}, channels: {}",
        supported_config.sample_rate().0,
        supported_config.channels()
    );

    // create stream configuration
    let config = StreamConfig {
        channels: CHANNELS,
        sample_rate: cpal::SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    // create shared processor state
    let processor = Arc::new(Mutex::new(AudioProcessor::new()));

    // build input stream with callback for processing audio data
    let stream = build_input_stream(&device, &config, &supported_config, processor.clone())?;

    // start the audio stream
    stream.play()?;
    info!("audio stream started. listening for audio input...");
    info!("press ctrl+c to stop.");

    // keep the main thread alive
    loop {
        thread::sleep(Duration::from_millis(100));
    }
}

/// build an input stream that handles different sample formats
fn build_input_stream(
    device: &Device,
    config: &StreamConfig,
    supported_config: &cpal::SupportedStreamConfig,
    processor: Arc<Mutex<AudioProcessor>>,
) -> Result<Stream> {
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
                |err| error!("audio stream error: {}", err),
                None,
            )?
        }
        SampleFormat::I16 => {
            let processor_clone = processor.clone();
            device.build_input_stream(
                config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    // convert i16 samples to f32
                    let float_data: Vec<f32> = data
                        .iter()
                        .map(|&sample| sample as f32 / i16::MAX as f32)
                        .collect();

                    if let Ok(mut proc) = processor_clone.lock() {
                        proc.add_samples(&float_data);
                    }
                },
                |err| error!("audio stream error: {}", err),
                None,
            )?
        }
        SampleFormat::U16 => {
            let processor_clone = processor.clone();
            device.build_input_stream(
                config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    // convert u16 samples to f32
                    let float_data: Vec<f32> = data
                        .iter()
                        .map(|&sample| {
                            (sample as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0)
                        })
                        .collect();

                    if let Ok(mut proc) = processor_clone.lock() {
                        proc.add_samples(&float_data);
                    }
                },
                |err| error!("audio stream error: {}", err),
                None,
            )?
        }
        sample_format => {
            return Err(anyhow::anyhow!("unsupported sample format: {}", sample_format));
        }
    };

    Ok(stream)
}

