use crate::shared::{SharedState, SPECTRO_BINS};
use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{error, info, trace};

// configuration constants for audio processing
const BUFFER_SIZE: usize = 1024; // fft window size
const HOP_SIZE: usize = 512; // analysis hop size (50% overlap)
const ONSET_HISTORY_SIZE: usize = 128; // number of onset strength values to keep for bpm calculation
const SILENCE_THRESHOLD: f32 = 0.001; // rms threshold below which audio is considered silence
const BPM_SMOOTHING_WINDOW: usize = 21; // median smoothing window for bpm
const BPM_EMA_ALPHA: f32 = 0.1; // additional EMA smoothing factor
const BPM_MIN: f32 = 80.0; // preferred bpm lower bound for folding
const BPM_MAX: f32 = 160.0; // preferred bpm upper bound for folding
const BPM_MIN_CONFIDENCE_RATIO: f32 = 1.8; // min peak/median ratio to accept bpm update

// onset detection thresholding
const ONSET_PEAK_RATIO: f32 = 2.5; // how strong onset must be vs median to count as beat
const KICK_PEAK_RATIO: f32 = 3.0; // how strong low-band onset must be vs median to count as kick
const KICK_BAND_MAX_HZ: f32 = 150.0; // low-frequency band upper bound for kick detection

// frequency weighting for onset detection
const LOW_EMPH_FC: f32 = 300.0; // corner frequency for low boost
const HIGH_ALLOW_FC: f32 = 4000.0; // corner frequency to allow highs (hi-hats/snares)
const HIGH_ALLOW_GAIN: f32 = 0.3; // highs are allowed but with reduced weight vs lows
const FREQ_EPS: f32 = 1e-3; // avoid div-by-zero in weighting math

/// shared state for real-time audio processing
struct AudioProcessor {
    sample_buffer: VecDeque<f32>,
    onset_history: VecDeque<f32>,
    last_bpm_print: Instant,
    current_bpm: f32,
    bpm_history: VecDeque<f32>,
    prev_weighted_magnitudes: Vec<f32>,
    rms_history: VecDeque<f32>,
    sample_rate: u32,
    hop_size: usize,
    shared: Arc<Mutex<SharedState>>,
    // last known beat instant for phase derivation in ui
    last_beat_at: Option<Instant>,
    // last strong kick timestamp
    last_kick_at: Option<Instant>,
}

impl AudioProcessor {
    fn new(sample_rate: u32, hop_size: usize, shared: Arc<Mutex<SharedState>>) -> Self {
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
            shared,
            last_beat_at: None,
            last_kick_at: None,
        }
    }

    /// add new mono audio samples to the buffer and process if we have enough data
    fn add_samples(&mut self, samples: &[f32]) {
        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        self.rms_history.push_back(rms);
        if self.rms_history.len() > 10 {
            self.rms_history.pop_front();
        }

        for &sample in samples {
            self.sample_buffer.push_back(sample);
        }

        while self.sample_buffer.len() >= BUFFER_SIZE {
            self.process_audio_window();
            
            for _ in 0..self.hop_size {
                if self.sample_buffer.pop_front().is_none() {
                    break;
                }
            }
        }

        if self.last_bpm_print.elapsed() >= Duration::from_millis(100) {
            let avg_rms = self.rms_history.iter().sum::<f32>() / self.rms_history.len() as f32;
            if let Ok(mut s) = self.shared.lock() {
                s.set_measurements(
                    self.current_bpm,
                    avg_rms,
                    self.last_beat_at,
                    self.last_kick_at,
                );
            }

            if avg_rms < SILENCE_THRESHOLD {
                info!("current bpm: n/a (silence detected)");
            } else if self.current_bpm > 0.0 {
                info!("current bpm: {:.1}", self.current_bpm);
            } else {
                info!("current bpm: n/a (analyzing...)");
            }
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

    fn process_audio_window(&mut self) {
        let window: Vec<f32> = self
            .sample_buffer
            .iter()
            .take(BUFFER_SIZE)
            .copied()
            .collect();
        if window.len() < BUFFER_SIZE {
            return;
        }

        let (onset_strength, spectro_slice, lowband_flux) =
            self.calculate_onset_strength_and_slice(&window);
        let prev_last = self.onset_history.back().copied().unwrap_or(0.0);
        self.onset_history.push_back(onset_strength);
        if self.onset_history.len() > ONSET_HISTORY_SIZE {
            self.onset_history.pop_front();
        }
        if self.onset_history.len() >= 20 {
            self.current_bpm = self.calculate_bpm();
        }

        // on strong onset: mark a beat timestamp
        if self.onset_history.len() >= 16 {
            // compute median baseline from recent history (excluding the just-pushed value would be similar)
            let mut slice: Vec<f32> = self.onset_history.iter().copied().collect();
            slice.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = slice[slice.len() / 2].max(1e-6);
            let is_peak = onset_strength > ONSET_PEAK_RATIO * median && onset_strength > prev_last;
            if is_peak {
                self.last_beat_at = Some(Instant::now());
                trace!("beat marked at strong onset");
            }
        }

        // push spectrogram slice and track kick from low-band flux
        if let Ok(mut s) = self.shared.lock() {
            // map spectro_slice linearly to SPECTRO_BINS (no log remap; UV shader will handle mapping)
            let bins = SPECTRO_BINS as usize;
            let src = &spectro_slice;
            let src_len = src.len();
            let mut tmp = vec![0.0f32; bins];
            if bins <= src_len {
                let block = src_len as f32 / bins as f32;
                for b in 0..bins {
                    let start = (b as f32 * block).floor() as usize;
                    let end = ((b as f32 + 1.0) * block).ceil() as usize;
                    let end = end.min(src_len);
                    let start = start.min(end);
                    let mut sum = 0.0;
                    let mut cnt = 0;
                    for i in start..end {
                        sum += src[i];
                        cnt += 1;
                    }
                    tmp[b] = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
                }
            } else {
                for b in 0..bins {
                    let i = ((b as f32 / bins as f32) * src_len as f32).floor() as usize;
                    tmp[b] = src[i.min(src_len - 1)];
                }
            }
            s.push_spectrogram_slice(&tmp);
            // push normalized onset for graph (normalize by dynamic baseline)
            let norm = {
                // compute a small baseline from recent history
                let mut slice: Vec<f32> = self.onset_history.iter().copied().collect();
                slice.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = if !slice.is_empty() { slice[slice.len() / 2] } else { 1e-6 };
                (onset_strength / med.max(1e-6)).min(4.0) / 4.0
            };
            s.push_onset(norm);
        }

        // detect kick using low-band flux with a simple median threshold
        // keep a small ring buffer of recent lowband fluxes using onset_history for simplicity
        // here we just re-use the same thresholding as above but on lowband_flux
        if self.onset_history.len() >= 16 {
            let mut slice: Vec<f32> = self.onset_history.iter().copied().collect();
            slice.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_full = slice[slice.len() / 2].max(1e-6);
            // derive a rough lowband baseline as some fraction of full median
            // this is a heuristic to avoid an extra buffer; if lowband flux is very high vs baseline, mark kick
            let baseline = 0.5 * median_full;
            if lowband_flux > KICK_PEAK_RATIO * baseline {
                self.last_kick_at = Some(Instant::now());
                trace!("kick marked at strong low-band onset");
            }
        }
    }

    fn calculate_onset_strength_and_slice(&mut self, window: &[f32]) -> (f32, Vec<f32>, f32) {
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

        let mut fft_input: Vec<Complex<f32>> =
            windowed.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(BUFFER_SIZE);
        fft.process(&mut fft_input);

        let magnitudes: Vec<f32> = fft_input
            .iter()
            .take(BUFFER_SIZE / 2)
            .map(|complex| complex.norm())
            .collect();

        let bin_hz = self.sample_rate as f32 / BUFFER_SIZE as f32;
        let weighted: Vec<f32> = magnitudes
            .iter()
            .enumerate()
            .map(|(k, &m)| {
                let f = k as f32 * bin_hz + FREQ_EPS;
                let low_w = 1.0 / (1.0 + (f / LOW_EMPH_FC).powi(2));
                let high_ratio = (f / HIGH_ALLOW_FC).max(0.0);
                let high_w =
                    (HIGH_ALLOW_GAIN * high_ratio / (1.0 + high_ratio)).clamp(0.0, HIGH_ALLOW_GAIN);
                let w = (low_w + high_w).clamp(0.0, 1.0);
                m * w
            })
            .collect();

        let spectral_flux: f32 = weighted
            .iter()
            .zip(self.prev_weighted_magnitudes.iter())
            .map(|(&current, &previous)| (current - previous).max(0.0))
            .sum();
        // low-band-only flux for kick detection
        let max_low_bin = ((KICK_BAND_MAX_HZ / bin_hz) as usize).min(weighted.len());
        let lowband_flux: f32 = weighted
            [0..max_low_bin]
            .iter()
            .zip(self.prev_weighted_magnitudes[0..max_low_bin].iter())
            .map(|(&current, &previous)| (current - previous).max(0.0))
            .sum();

        // build a spectrogram slice from weighted magnitudes with simple dynamic range mapping
        let slice: Vec<f32> = weighted
            .iter()
            .map(|&v| {
                let v = v.max(0.0);
                let v = (v / 50.0).min(1.0); // simple scale; tuned empirically
                v
            })
            .collect();

        self.prev_weighted_magnitudes.copy_from_slice(&weighted);
        (spectral_flux, slice, lowband_flux)
    }

    fn calculate_bpm(&mut self) -> f32 {
        if self.onset_history.len() < 20 {
            return self.current_bpm;
        }
        let onset_data: Vec<f32> = self.onset_history.iter().copied().collect();
        let autocorr = self.autocorrelation(&onset_data);

        let mut max_peak = 0.0;
        let mut peak_lag = 1;

        let hop_seconds = self.hop_size as f32 / self.sample_rate as f32;
        let min_lag = (60.0 / BPM_MAX / hop_seconds).round().max(1.0) as usize;
        let max_lag = (60.0 / BPM_MIN / hop_seconds).round() as usize;
        let search_end = max_lag.min(autocorr.len());
        let search_start = min_lag.min(search_end);
        trace!(
            "diag: bpm search lags={}..{}, hop_seconds={:.6}",
            search_start, search_end, hop_seconds
        );

        for lag in search_start..search_end {
            if autocorr[lag] > max_peak {
                max_peak = autocorr[lag];
                peak_lag = lag;
            }
        }
        if max_peak == 0.0 || peak_lag == 0 {
            return self.current_bpm;
        }

        let raw_bpm = 60.0 / (peak_lag as f32 * hop_seconds);
        let mut folded = raw_bpm;
        while folded < BPM_MIN {
            folded *= 2.0;
        }
        while folded > BPM_MAX {
            folded /= 2.0;
        }

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

        if confidence_ratio >= BPM_MIN_CONFIDENCE_RATIO {
            self.bpm_history.push_back(folded);
            if self.bpm_history.len() > BPM_SMOOTHING_WINDOW {
                self.bpm_history.pop_front();
            }
            let median = if self.bpm_history.len() >= 3 {
                let mut sorted_bpm: Vec<f32> = self.bpm_history.iter().copied().collect();
                sorted_bpm.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted_bpm[sorted_bpm.len() / 2]
            } else {
                folded
            };
            let ema = if self.current_bpm > 0.0 {
                BPM_EMA_ALPHA * median + (1.0 - BPM_EMA_ALPHA) * self.current_bpm
            } else {
                median
            };
            ema
        } else {
            self.current_bpm
        }
    }

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

    // no phase correction function needed; beats are marked by timestamp
}

pub fn list_input_devices() -> Result<()> {
    let host = cpal::default_host();
    let devices = host.input_devices()?;
    tracing::info!("available input devices:");
    for (index, device) in devices.enumerate() {
        let device_name = device
            .name()
            .unwrap_or_else(|_| "unknown device".to_string());
        match device.supported_input_configs() {
            Ok(mut configs) => {
                if let Some(config) = configs.next() {
                    tracing::info!(
                        "  {}: {} ({}hz, {} channels)",
                        index,
                        device_name,
                        config.max_sample_rate().0,
                        config.channels()
                    );
                } else {
                    tracing::info!("  {}: {} (no supported configs)", index, device_name);
                }
            }
            Err(_) => tracing::info!("  {}: {} (config query failed)", index, device_name),
        }
    }
    tracing::info!("\nusage: audio-visualizer --device <index> [--config <index>]");
    Ok(())
}

pub fn list_device_configs(device_index: usize) -> Result<()> {
    let host = cpal::default_host();
    let devices: Vec<_> = host.input_devices()?.collect();
    let device = devices
        .into_iter()
        .nth(device_index)
        .ok_or_else(|| anyhow::anyhow!("input device index {} not found", device_index))?;
    let device_name = device
        .name()
        .unwrap_or_else(|_| "unknown device".to_string());
    tracing::info!(
        "available configs for device {}: {}",
        device_index,
        device_name
    );
    match device.supported_input_configs() {
        Ok(configs) => {
            for (config_index, config_range) in configs.enumerate() {
                let min_rate = config_range.min_sample_rate().0;
                let max_rate = config_range.max_sample_rate().0;
                let channels = config_range.channels();
                let sample_format = config_range.sample_format();
                let buffer_size = match config_range.buffer_size() {
                    cpal::SupportedBufferSize::Range { min, max } => format!("{}-{}", min, max),
                    cpal::SupportedBufferSize::Unknown => "unknown".to_string(),
                };
                if min_rate == max_rate {
                    tracing::info!(
                        "  {}: {}hz, {} channels, {:?}, buffer: {}",
                        config_index,
                        min_rate,
                        channels,
                        sample_format,
                        buffer_size
                    );
                } else {
                    tracing::info!(
                        "  {}: {}-{}hz, {} channels, {:?}, buffer: {}",
                        config_index,
                        min_rate,
                        max_rate,
                        channels,
                        sample_format,
                        buffer_size
                    );
                }
            }
        }
        Err(e) => return Err(anyhow::anyhow!("failed to get supported configs: {}", e)),
    }
    tracing::info!(
        "\nusage: audio-visualizer --device {} --config <index>",
        device_index
    );
    Ok(())
}

pub fn run_input_mode(
    device_index: Option<usize>,
    config_index: Option<usize>,
    shared: Arc<Mutex<SharedState>>,
) -> Result<()> {
    info!("starting real-time onset detection and bpm analysis...");
    let host = cpal::default_host();
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

    let supported_configs: Vec<_> = device.supported_input_configs()?.collect();
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

    let config = StreamConfig {
        channels: supported_config.channels(),
        sample_rate: supported_config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let processor = Arc::new(Mutex::new(AudioProcessor::new(
        config.sample_rate.0,
        HOP_SIZE,
        shared.clone(),
    )));
    let stream = build_input_stream(&device, &config, &supported_config, processor.clone())?;
    stream.play()?;
    info!("audio stream started. listening for audio input...");
    info!("press ctrl+c to stop.");
    loop {
        thread::sleep(Duration::from_millis(100));
    }
}

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
