use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
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
}

impl AudioProcessor {
    fn new() -> Self {
        Self {
            sample_buffer: VecDeque::with_capacity(BUFFER_SIZE * 2),
            onset_history: VecDeque::with_capacity(ONSET_HISTORY_SIZE),
            last_bpm_print: Instant::now(),
            current_bpm: 0.0,
        }
    }

    /// Add new audio samples to the buffer and process if we have enough data
    fn add_samples(&mut self, samples: &[f32]) {
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
            println!("Current BPM: {:.1}", self.current_bpm);
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
    fn calculate_onset_strength(&self, window: &[f32]) -> f32 {
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

        // Calculate spectral flux (change in spectral energy)
        // For simplicity, we'll use the sum of magnitudes as a proxy for onset strength
        // In a more sophisticated implementation, you would compare with the previous frame
        let spectral_flux = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;

        // Combine spectral centroid and flux for onset strength
        // Higher values indicate potential onsets
        spectral_centroid * 0.001 + spectral_flux
    }

    /// Calculate BPM from the history of onset strength values
    fn calculate_bpm(&self) -> f32 {
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
        if peak_lag > 0 {
            60.0 * SAMPLE_RATE as f32 / (peak_lag as f32 * samples_per_onset)
        } else {
            self.current_bpm // Keep previous BPM if no peak found
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
    println!("Starting real-time onset detection and BPM analysis...");
    
    // Initialize CPAL host and get default input device
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;

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
    let processor_clone = processor.clone();

    // Build input stream with callback for processing audio data
    let stream = match supported_config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if let Ok(mut proc) = processor_clone.lock() {
                    proc.add_samples(data);
                }
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )?,
        SampleFormat::I16 => {
            let processor_clone = processor.clone();
            device.build_input_stream(
                &config,
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
                &config,
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

    // Start the audio stream
    stream.play()?;
    println!("Audio stream started. Listening for audio input...");
    println!("Press Ctrl+C to stop.");

    // Keep the main thread alive
    loop {
        thread::sleep(Duration::from_millis(100));
    }
}
