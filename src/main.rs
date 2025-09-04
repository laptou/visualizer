use anyhow::Result;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{error, info, trace};

/// configuration constants for audio processing
const BUFFER_SIZE: usize = 1024; // fft window size
const HOP_SIZE: usize = 512; // analysis hop size (50% overlap)
const ONSET_HISTORY_SIZE: usize = 128; // number of onset strength values to keep for bpm calculation
const SILENCE_THRESHOLD: f32 = 0.001; // rms threshold below which audio is considered silence
const BPM_SMOOTHING_WINDOW: usize = 21; // median smoothing window for bpm
const BPM_EMA_ALPHA: f32 = 0.1; // additional EMA smoothing factor
const BPM_MIN: f32 = 80.0; // preferred bpm lower bound for folding
const BPM_MAX: f32 = 160.0; // preferred bpm upper bound for folding
const BPM_MIN_CONFIDENCE_RATIO: f32 = 1.8; // min peak/median ratio to accept bpm update

// frequency weighting for onset detection
const LOW_EMPH_FC: f32 = 300.0; // corner frequency for low boost
const HIGH_ALLOW_FC: f32 = 4000.0; // corner frequency to allow highs (hi-hats/snares)
const HIGH_ALLOW_GAIN: f32 = 0.3; // highs are allowed but with reduced weight vs lows
const FREQ_EPS: f32 = 1e-3; // avoid div-by-zero in weighting math

/// command-line interface for the audio visualizer
#[derive(Parser)]
#[command(name = "audio-visualizer")]
#[command(about = "real-time audio onset detection and bpm analysis")]
struct Cli {
    /// input device index (use --list-devices to see options)
    #[arg(short, long)]
    device: Option<usize>,

    /// config index for the selected device (use --list-configs to see options)
    #[arg(short, long)]
    config: Option<usize>,

    /// list available input devices and exit
    #[arg(long)]
    list_devices: bool,

    /// list available configs for a device and exit (requires --device)
    #[arg(long)]
    list_configs: bool,
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
    /// previous weighted magnitude spectrum for spectral flux calculation
    prev_weighted_magnitudes: Vec<f32>,
    /// running rms for silence detection
    rms_history: VecDeque<f32>,
    /// actual sample rate in Hz
    sample_rate: u32,
    /// analysis hop size in samples
    hop_size: usize,
}

impl AudioProcessor {
    fn new(sample_rate: u32, hop_size: usize) -> Self {
        Self {
            sample_buffer: VecDeque::with_capacity(BUFFER_SIZE * 2),
            onset_history: VecDeque::with_capacity(ONSET_HISTORY_SIZE),
            last_bpm_print: Instant::now(),
            current_bpm: 0.0,
            bpm_history: VecDeque::with_capacity(BPM_SMOOTHING_WINDOW),
            prev_weighted_magnitudes: vec![0.0; BUFFER_SIZE / 2],
            rms_history: VecDeque::with_capacity(10),
            sample_rate,
            hop_size,
        }
    }

    /// add new mono audio samples to the buffer and process if we have enough data
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
        }

        // process audio with a fixed hop if we have enough samples for analysis
        while self.sample_buffer.len() >= BUFFER_SIZE {
            self.process_audio_window();
            // advance by hop size
            for _ in 0..self.hop_size {
                if self.sample_buffer.pop_front().is_none() {
                    break;
                }
            }
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
            // trace summary once per second to limit spam
            trace!(
                "diag: avg_rms={:.6}, onset_frames={}, bpm_history={}, hop={}, sr={}",
                avg_rms,
                self.onset_history.len(),
                self.bpm_history.len(),
                self.hop_size,
                self.sample_rate
            );
            self.last_bpm_print = Instant::now();
        }
    }

    /// process a window of audio samples for onset detection
    /// overview: take a hop-sized frame, run an stft (hann + fft), weight
    /// frequencies (lows prioritized, highs allowed, mids suppressed), compute
    /// spectral flux, and update bpm via autocorr with confidence gating.
    fn process_audio_window(&mut self) {
        // extract the first BUFFER_SIZE samples for analysis
        let window: Vec<f32> = self
            .sample_buffer
            .iter()
            .take(BUFFER_SIZE)
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

    /// calculate onset strength using stft + spectral flux
    /// overview: we compute an stft frame (hann window + fft), apply a frequency weighting
    /// that prioritizes lows and allows highs while suppressing mids, then measure positive
    /// spectral changes (spectral flux). this stabilizes beats when kicks pause but hats continue.
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

        // compute fft to get frequency domain representation (this is one stft frame)
        let mut fft_input: Vec<Complex<f32>> =
            windowed.iter().map(|&x| Complex::new(x, 0.0)).collect();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(BUFFER_SIZE);
        fft.process(&mut fft_input);

        // calculate magnitude spectrum (positive frequencies only)
        let magnitudes: Vec<f32> = fft_input
            .iter()
            .take(BUFFER_SIZE / 2)
            .map(|complex| complex.norm())
            .collect();

        // frequency weighting: suppress mids, prioritize lows, allow highs with reduced gain
        // low emphasis: 1 / (1 + (f/fc)^2)
        // high allowance: gentle shelf that ramps up above HIGH_ALLOW_FC with capped gain
        let bin_hz = self.sample_rate as f32 / BUFFER_SIZE as f32;
        let weighted: Vec<f32> = magnitudes
            .iter()
            .enumerate()
            .map(|(k, &m)| {
                let f = k as f32 * bin_hz + FREQ_EPS;
                // low emphasis (strong near kick region)
                let low_w = 1.0 / (1.0 + (f / LOW_EMPH_FC).powi(2));
                // high allowance: start allowing above HIGH_ALLOW_FC but keep under low priority
                let high_ratio = (f / HIGH_ALLOW_FC).max(0.0);
                let high_w =
                    (HIGH_ALLOW_GAIN * high_ratio / (1.0 + high_ratio)).clamp(0.0, HIGH_ALLOW_GAIN);
                // mid suppression emerges naturally since low_w decays and high_w is capped small
                let w = (low_w + high_w).clamp(0.0, 1.0);
                m * w
            })
            .collect();

        // compute simple band energies to diagnose dominance of hats vs kick
        // bands: low [0, LOW_EMPH_FC), mid [LOW_EMPH_FC, HIGH_ALLOW_FC), high [HIGH_ALLOW_FC, nyquist)
        let low_idx_end = (LOW_EMPH_FC / bin_hz).floor().max(0.0) as usize;
        let high_idx_start = (HIGH_ALLOW_FC / bin_hz).floor().max(0.0) as usize;
        let n_bins = magnitudes.len();

        let clamp = |v: usize| v.min(n_bins);
        let li = 0usize;
        let le = clamp(low_idx_end);
        let mi = clamp(low_idx_end);
        let me = clamp(high_idx_start);
        let hi = clamp(high_idx_start);
        let he = n_bins;

        let sum_range = |v: &Vec<f32>, i: usize, e: usize| -> f32 { v[i..e].iter().copied().sum() };
        let raw_low = if le > li { sum_range(&magnitudes, li, le) } else { 0.0 };
        let raw_mid = if me > mi { sum_range(&magnitudes, mi, me) } else { 0.0 };
        let raw_high = if he > hi { sum_range(&magnitudes, hi, he) } else { 0.0 };
        let w_low = if le > li { sum_range(&weighted, li, le) } else { 0.0 };
        let w_mid = if me > mi { sum_range(&weighted, mi, me) } else { 0.0 };
        let w_high = if he > hi { sum_range(&weighted, hi, he) } else { 0.0 };

        // spectral flux (positive differences only) on weighted magnitudes
        let spectral_flux: f32 = weighted
            .iter()
            .zip(self.prev_weighted_magnitudes.iter())
            .map(|(&current, &previous)| (current - previous).max(0.0))
            .sum();

        // update previous weighted magnitudes for next frame
        self.prev_weighted_magnitudes.copy_from_slice(&weighted);

        // trace: report energies and spectral flux to understand onset behavior
        trace!(
            "diag: bands raw(l/m/h)={:.2}/{:.2}/{:.2}, weighted(l/m/h)={:.2}/{:.2}/{:.2}, flux={:.3}",
            raw_low,
            raw_mid,
            raw_high,
            w_low,
            w_mid,
            w_high,
            spectral_flux
        );

        spectral_flux
    }

    /// calculate bpm from the history of onset strength values
    fn calculate_bpm(&mut self) -> f32 {
        if self.onset_history.len() < 20 {
            // not enough data yet
            trace!(
                "diag: bpm skipped, onset_frames={} (<20)",
                self.onset_history.len()
            );
            return self.current_bpm;
        }

        // convert onset history to vec for fft processing
        let onset_data: Vec<f32> = self.onset_history.iter().copied().collect();

        // apply autocorrelation to find periodic patterns in onset strength
        let autocorr = self.autocorrelation(&onset_data);

        // find the peak in autocorrelation (excluding the zero-lag peak)
        let mut max_peak = 0.0;
        let mut peak_lag = 1;

        // search for peaks in preferred bpm range, translated to lags
        let hop_seconds = self.hop_size as f32 / self.sample_rate as f32;
        let min_lag = (60.0 / BPM_MAX / hop_seconds).round().max(1.0) as usize;
        let max_lag = (60.0 / BPM_MIN / hop_seconds).round() as usize;

        // guard the search bounds to avoid empty/invalid ranges early on
        let search_end = max_lag.min(autocorr.len());
        let search_start = min_lag.min(search_end);

        // trace: summarize search space
        trace!(
            "diag: bpm search lags={}..{}, hop_seconds={:.6}",
            search_start,
            search_end,
            hop_seconds
        );

        for lag in search_start..search_end {
            if autocorr[lag] > max_peak {
                max_peak = autocorr[lag];
                peak_lag = lag;
            }
        }

        // if we couldn't find a valid peak yet, keep previous bpm
        if max_peak == 0.0 || peak_lag == 0 {
            trace!("diag: no valid peak found, keeping bpm={:.1}", self.current_bpm);
            return self.current_bpm;
        }

        // convert lag back to bpm
        // bpm = 60 / (lag * hop_seconds)
        let raw_bpm = if peak_lag > 0 {
            60.0 / (peak_lag as f32 * hop_seconds)
        } else {
            return self.current_bpm; // keep previous bpm if no peak found
        };

        // fold to preferred band by ×2/÷2
        let mut folded = raw_bpm;
        while folded < BPM_MIN {
            folded *= 2.0;
        }
        while folded > BPM_MAX {
            folded /= 2.0;
        }

        // trace: show top-3 peaks in range to understand competing periodicities
        if search_end > search_start {
            let mut ranked: Vec<(usize, f32)> =
                (search_start..search_end).map(|lag| (lag, autocorr[lag])).collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let tops = ranked.iter().take(3).map(|(lag, val)| {
                let bpm = 60.0 / (*lag as f32 * hop_seconds);
                let mut fbpm = bpm;
                let mut k = 0;
                while fbpm < BPM_MIN && k < 4 { fbpm *= 2.0; k += 1; }
                while fbpm > BPM_MAX && k < 8 { fbpm /= 2.0; k += 1; }
                (*lag, *val, bpm, fbpm)
            }).collect::<Vec<_>>();
            if let Some((l0, v0, b0, f0)) = tops.get(0).copied() {
                let msg = if let Some((l1, v1, b1, f1)) = tops.get(1).copied() {
                    if let Some((l2, v2, b2, f2)) = tops.get(2).copied() {
                        format!(
                            "diag: peaks [lag,val,bpm,folded] = [{},{:.3},{:.1},{:.1}] | [{},{:.3},{:.1},{:.1}] | [{},{:.3},{:.1},{:.1}]",
                            l0, v0, b0, f0, l1, v1, b1, f1, l2, v2, b2, f2
                        )
                    } else {
                        format!(
                            "diag: peaks [lag,val,bpm,folded] = [{},{:.3},{:.1},{:.1}] | [{},{:.3},{:.1},{:.1}]",
                            l0, v0, b0, f0, l1, v1, b1, f1
                        )
                    }
                } else {
                    format!(
                        "diag: peak [lag,val,bpm,folded] = [{},{:.3},{:.1},{:.1}]",
                        l0, v0, b0, f0
                    )
                };
                trace!("{}", msg);
            }
        }

        // compute confidence from autocorrelation: peak prominence vs local baseline
        // we use median of the searched lag range as a robust baseline
        let hop_seconds = self.hop_size as f32 / self.sample_rate as f32;
        let min_lag = (60.0 / BPM_MAX / hop_seconds).round().max(1.0) as usize;
        let max_lag = (60.0 / BPM_MIN / hop_seconds).round() as usize;

        let end = max_lag.min(autocorr.len());
        let start = min_lag.min(end);

        let baseline_median = if end > start + 2 {
            let mut slice: Vec<f32> = autocorr[start..end].to_vec();
            slice.sort_by(|a, b| a.partial_cmp(b).unwrap());
            slice[slice.len() / 2]
        } else {
            0.0
        };

        let confidence_ratio = if baseline_median > 0.0 {
            max_peak / baseline_median
        } else {
            0.0
        };

        // only accept and smooth bpm when confidence is sufficient
        if confidence_ratio >= BPM_MIN_CONFIDENCE_RATIO {
            // add to bpm history for median smoothing
            self.bpm_history.push_back(folded);
            if self.bpm_history.len() > BPM_SMOOTHING_WINDOW {
                self.bpm_history.pop_front();
            }

            // median filter to reduce outliers
            let median = if self.bpm_history.len() >= 3 {
                let mut sorted_bpm: Vec<f32> = self.bpm_history.iter().copied().collect();
                sorted_bpm.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted_bpm[sorted_bpm.len() / 2]
            } else {
                folded
            };

            // apply EMA on top of median for stability
            let ema = if self.current_bpm > 0.0 {
                BPM_EMA_ALPHA * median + (1.0 - BPM_EMA_ALPHA) * self.current_bpm
            } else {
                median
            };

            // report acceptance
            if self.current_bpm > 0.0 {
                let delta = ema - self.current_bpm;
                trace!(
                    "diag: accept bpm: raw={:.1} folded={:.1}, baseline={:.3} conf={:.2} >= {:.2}, median={:.1}, ema={:.1} (prev {:.1}, delta {:.1})",
                    raw_bpm,
                    folded,
                    baseline_median,
                    confidence_ratio,
                    BPM_MIN_CONFIDENCE_RATIO,
                    median,
                    ema,
                    self.current_bpm,
                    delta
                );
            } else {
                trace!(
                    "diag: initial bpm: raw={:.1} folded={:.1}, baseline={:.3} conf={:.2} >= {:.2}, median={:.1}",
                    raw_bpm,
                    folded,
                    baseline_median,
                    confidence_ratio,
                    BPM_MIN_CONFIDENCE_RATIO,
                    ema
                );
            }

            return ema;
        } else {
            // keep previous bpm if confidence is low
            trace!(
                "diag: reject bpm: raw={:.1} folded={:.1}, baseline={:.3} conf={:.2} < {:.2}; keep {:.1}",
                raw_bpm,
                folded,
                baseline_median,
                confidence_ratio,
                BPM_MIN_CONFIDENCE_RATIO,
                self.current_bpm
            );
            return self.current_bpm;
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
    } else if cli.list_configs {
        if cli.device.is_none() {
            return Err(anyhow::anyhow!(
                "--list-configs requires --device to be specified"
            ));
        }
        list_device_configs(cli.device.unwrap())?;
    } else {
        run_input_mode(cli.device, cli.config)?;
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

    info!("\nusage: audio-visualizer --device <index> [--config <index>]");
    Ok(())
}

/// list all available configs for a specific device
fn list_device_configs(device_index: usize) -> Result<()> {
    let host = cpal::default_host();
    let devices: Vec<_> = host.input_devices()?.collect();

    let device = devices
        .into_iter()
        .nth(device_index)
        .ok_or_else(|| anyhow::anyhow!("input device index {} not found", device_index))?;

    let device_name = device
        .name()
        .unwrap_or_else(|_| "Unknown Device".to_string());

    info!(
        "available configs for device {}: {}",
        device_index, device_name
    );

    match device.supported_input_configs() {
        Ok(configs) => {
            for (config_index, config_range) in configs.enumerate() {
                // show the range of sample rates supported by this config
                let min_rate = config_range.min_sample_rate().0;
                let max_rate = config_range.max_sample_rate().0;
                let channels = config_range.channels();
                let sample_format = config_range.sample_format();
                let buffer_size = match config_range.buffer_size() {
                    cpal::SupportedBufferSize::Range { min, max } => format!("{}-{}", min, max),
                    cpal::SupportedBufferSize::Unknown => "unknown".to_string(),
                };

                if min_rate == max_rate {
                    info!(
                        "  {}: {}hz, {} channels, {:?}, buffer: {}",
                        config_index, min_rate, channels, sample_format, buffer_size
                    );
                } else {
                    info!(
                        "  {}: {}-{}hz, {} channels, {:?}, buffer: {}",
                        config_index, min_rate, max_rate, channels, sample_format, buffer_size
                    );
                }
            }
        }
        Err(e) => {
            return Err(anyhow::anyhow!("failed to get supported configs: {}", e));
        }
    }

    info!(
        "\nusage: audio-visualizer --device {} --config <index>",
        device_index
    );
    Ok(())
}

/// run the input mode with the specified device and config
fn run_input_mode(device_index: Option<usize>, config_index: Option<usize>) -> Result<()> {
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
    let supported_configs: Vec<_> = device.supported_input_configs()?.collect();

    // select the specified config or use the first one with max sample rate as default
    let supported_config = if let Some(config_idx) = config_index {
        supported_configs
            .get(config_idx)
            .ok_or_else(|| anyhow::anyhow!("config index {} not found for device", config_idx))?
            .clone()
            .with_max_sample_rate()
    } else {
        supported_configs
            .first()
            .ok_or_else(|| anyhow::anyhow!("no supported input config"))?
            .clone()
            .with_max_sample_rate()
    };

    let config_msg = if let Some(idx) = config_index {
        format!("using config {}: ", idx)
    } else {
        "using default config: ".to_string()
    };

    info!(
        "{}sample rate: {}, channels: {}, format: {:?}",
        config_msg,
        supported_config.sample_rate().0,
        supported_config.channels(),
        supported_config.sample_format()
    );

    // create stream configuration using the supported device config
    let config = StreamConfig {
        channels: supported_config.channels(),
        sample_rate: supported_config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    // create shared processor state with actual sample rate and hop size
    let processor = Arc::new(Mutex::new(AudioProcessor::new(
        config.sample_rate.0,
        HOP_SIZE,
    )));

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
    let num_channels = config.channels as usize;
    let stream = match supported_config.sample_format() {
        SampleFormat::F32 => {
            let processor_clone = processor.clone();
            device.build_input_stream(
                config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // downmix interleaved frames to mono by averaging channels
                    let mut mono: Vec<f32> = Vec::with_capacity(data.len() / num_channels);
                    for frame in data.chunks_exact(num_channels) {
                        let sum: f32 = frame.iter().copied().sum();
                        mono.push(sum / num_channels as f32);
                    }
                    if let Ok(mut proc) = processor_clone.lock() {
                        proc.add_samples(&mono);
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
                    // convert interleaved i16 frames to mono f32
                    let mut mono: Vec<f32> = Vec::with_capacity(data.len() / num_channels);
                    for frame in data.chunks_exact(num_channels) {
                        let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                        let avg = (sum as f32 / num_channels as f32) / i16::MAX as f32;
                        mono.push(avg);
                    }
                    if let Ok(mut proc) = processor_clone.lock() {
                        proc.add_samples(&mono);
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
                    // convert interleaved u16 frames to mono f32 in [-1,1]
                    let mut mono: Vec<f32> = Vec::with_capacity(data.len() / num_channels);
                    for frame in data.chunks_exact(num_channels) {
                        let sum: u32 = frame.iter().map(|&s| s as u32).sum();
                        let avg_u16 = (sum / num_channels as u32) as f32;
                        let centered = avg_u16 - (u16::MAX as f32 / 2.0);
                        let norm = centered / (u16::MAX as f32 / 2.0);
                        mono.push(norm);
                    }
                    if let Ok(mut proc) = processor_clone.lock() {
                        proc.add_samples(&mono);
                    }
                },
                |err| error!("audio stream error: {}", err),
                None,
            )?
        }
        sample_format => {
            return Err(anyhow::anyhow!(
                "unsupported sample format: {}",
                sample_format
            ));
        }
    };

    Ok(stream)
}
