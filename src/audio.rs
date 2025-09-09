use crate::shared::{SPECTRO_BINS, SharedState};
use crate::utils::{downsample_average, median, mean_and_variance2};
use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use std::cmp::min;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, trace};

// configuration constants for audio processing
const BUFFER_SIZE: usize = 2048; // fft window size (larger window for better frequency resolution)
const HOP_SIZE: usize = 1024; // analysis hop size (50% overlap)
const ONSET_HISTORY_SIZE: usize = 128; // number of onset strength values to keep for bpm calculation
// (removed) silence threshold currently unused
const BPM_SMOOTHING_WINDOW: usize = 21; // median smoothing window for bpm
const BPM_EMA_ALPHA: f32 = 0.1; // additional EMA smoothing factor
const BPM_MIN: f32 = 80.0; // preferred bpm lower bound for folding
const BPM_MAX: f32 = 160.0; // preferred bpm upper bound for folding
const BPM_MIN_CONFIDENCE_RATIO: f32 = 1.8; // min peak/median ratio to accept bpm update
const BPM_UPDATE_EPSILON: f32 = 0.1; // min bpm change to log as an update

// onset fft gating for bpm estimation
const ONSET_FFT_MIN_PEAK_RATIO: f32 = 1.5; // min (peak/median) in onset spectrum to accept band
const ONSET_FFT_SPREAD_PCT: f32 = 0.15; // expand band around strong onset frequencies
const ONSET_FFT_INCLUDE_RELATIVE: f32 = 0.6; // include bins >= this fraction of peak
const ONSET_FFT_BPM_SEARCH_MIN: f32 = 60.0; // onset fft search floor
const ONSET_FFT_BPM_SEARCH_MAX: f32 = 180.0; // onset fft search ceiling

// onset detection thresholding
const ONSET_PEAK_RATIO: f32 = 2.5; // how strong onset must be vs median to count as beat
// (removed) old derivative-based kick threshold no longer used
const KICK_BAND_MAX_HZ: f32 = 200.0; // low-frequency band upper bound for kick detection

// frequency weighting for onset detection
const LOW_EMPH_FC: f32 = 300.0; // corner frequency for low boost
const HIGH_ALLOW_FC: f32 = 4000.0; // corner frequency to allow highs (hi-hats/snares)
const HIGH_ALLOW_GAIN: f32 = 0.3; // highs are allowed but with reduced weight vs lows
const FREQ_EPS: f32 = 1e-3; // avoid div-by-zero in weighting math

// spectrogram display cap (we only visualize up to this frequency)
const SPECTRO_MAX_HZ: f32 = 8000.0;

// band definitions for energy-based beat detection (from blog method)
const BASS_MIN_HZ: f32 = 60.0;
const BASS_MAX_HZ: f32 = 130.0;
const LOWMID_MIN_HZ: f32 = 301.0;
const LOWMID_MAX_HZ: f32 = 750.0;
// small offsets to reduce noise floor, adapted from blog post
const BASS_OFFSET: f32 = 0.05;
const LOWMID_OFFSET: f32 = 0.005;

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
    // previous frame's summed low-band flux (for temporal differencing)
    prev_lowband_flux_sum: f32,
    // previous frame's low-band onset (for derivative-based kick detection)
    prev_lowband_onset: f32,
    // energy-based beat detection state
    beat_history: VecDeque<[f32; 2]>, // [bass, lowmid]
    beat_history_max: usize,
    band_limits: [(usize, usize); 2], // [(bass_start, bass_end), (lowmid_start, lowmid_end)] exclusive end
}

impl AudioProcessor {
    fn new(sample_rate: u32, hop_size: usize, shared: Arc<Mutex<SharedState>>) -> Self {
        // compute band indices from sample rate and fft size
        let bin_hz = sample_rate as f32 / BUFFER_SIZE as f32;
        let to_idx = |hz: f32| -> usize { f32::floor(hz / bin_hz) as usize };
        let bass_start = to_idx(BASS_MIN_HZ);
        let bass_end = to_idx(BASS_MAX_HZ);
        let lowmid_start = to_idx(LOWMID_MIN_HZ);
        let lowmid_end = to_idx(LOWMID_MAX_HZ);
        // number of analysis hops per second ~ sample_rate / hop_size
        let hist_max = std::cmp::max((sample_rate as usize) / std::cmp::max(hop_size, 1), 1);
        Self {
            sample_buffer: VecDeque::with_capacity(BUFFER_SIZE * 2),
            onset_history: VecDeque::with_capacity(ONSET_HISTORY_SIZE),
            last_bpm_print: Instant::now(),
            current_bpm: 0.0,
            bpm_history: VecDeque::with_capacity(BPM_SMOOTHING_WINDOW),
            prev_weighted_magnitudes: vec![0.0; BUFFER_SIZE / 4],
            rms_history: VecDeque::with_capacity(10),
            sample_rate,
            hop_size,
            shared,
            last_beat_at: None,
            last_kick_at: None,
            prev_lowband_flux_sum: 0.0,
            prev_lowband_onset: 0.0,
            beat_history: VecDeque::with_capacity(hist_max + 1),
            beat_history_max: hist_max,
            band_limits: [(bass_start, bass_end), (lowmid_start, lowmid_end)],
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
                // also publish most recent beat debug metrics if available
                // use last history avg/var by recomputing from beat_history
                if !self.beat_history.is_empty() {
                    let (avg, variance) = mean_and_variance2(self.beat_history.as_slices().0);
                    let thrs = [
                        (-10.0 * variance[0] + 1.55) * avg[0],
                        (-15.0 * variance[1] + 1.55) * avg[1],
                    ];
                    let last = *self.beat_history.back().unwrap_or(&[0.0, 0.0]);
                    let det_bass = (last[0] - BASS_OFFSET) > thrs[0];
                    let det_low = (last[1] - LOWMID_OFFSET) > thrs[1];
                    s.set_beat_debug(last, avg, variance, thrs, det_bass, det_low);
                }
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

        let (onset_strength, spectro_slice, lowband_flux, band_energies) =
            self.calculate_onset_strength_and_slice(&window);
        // compute low-band onset as temporal diff of summed low-band flux
        let lowband_onset = f32::max(lowband_flux - self.prev_lowband_flux_sum, 0.0);
        self.prev_lowband_flux_sum = lowband_flux;
        // compute first derivative of the low-band onset for kick detection
        let _lowband_onset_deriv = f32::max(lowband_onset - self.prev_lowband_onset, 0.0);
        self.prev_lowband_onset = lowband_onset;
        let prev_last = self.onset_history.back().copied().unwrap_or(0.0);
        self.onset_history.push_back(onset_strength);
        if self.onset_history.len() > ONSET_HISTORY_SIZE {
            self.onset_history.pop_front();
        }
        if self.onset_history.len() >= 20 {
            let prev_bpm = self.current_bpm;
            let next_bpm = self.calculate_bpm();
            let changed = next_bpm > 0.0
                && (prev_bpm <= 0.0 || (next_bpm - prev_bpm).abs() >= BPM_UPDATE_EPSILON);
            self.current_bpm = next_bpm;
            if changed {
                info!("current bpm: {:.1}", self.current_bpm);
            }
        }

        // on strong onset: mark a beat timestamp
        if self.onset_history.len() >= 16 {
            // compute median baseline from recent history (excluding the just-pushed value would be similar)
            let slice: Vec<f32> = self.onset_history.iter().copied().collect();
            let median = f32::max(median(slice), 1e-6);
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
                    let start = f32::floor(b as f32 * block) as usize;
                    let end = f32::ceil((b as f32 + 1.0) * block) as usize;
                    let end = min(end, src_len);
                    let start = min(start, end);
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
                    let i = f32::floor((b as f32 / bins as f32) * src_len as f32) as usize;
                    tmp[b] = src[min(i, src_len - 1)];
                }
            }
            s.push_spectrogram_slice(&tmp);
            // compute a small baseline from recent history
            let slice: Vec<f32> = self.onset_history.iter().copied().collect();
            let med = if !slice.is_empty() {
                median(slice)
            } else {
                1e-6
            };
            // push normalized onset for graph (normalize by dynamic baseline)
            let norm = f32::clamp(onset_strength / f32::max(med, 1e-6), 0.0, 4.0) / 4.0;
            s.push_onset(norm);
            // derive low-frequency onset normalization using a fraction of median as baseline
            let low_baseline = 0.5 * f32::max(med, 1e-6);
            let norm_low = f32::clamp(lowband_onset / low_baseline, 0.0, 4.0) / 4.0;
            s.push_low_onset(norm_low);
        }

        // energy-based kick detection (bass and low-mid) using variance-adaptive threshold
        // compute threshold from 1s history (without current sample), then update history
        let mut is_bass = false;
        let mut _is_lowmid = false;
        if !self.beat_history.is_empty() {
            let (avg, variance) = mean_and_variance2(self.beat_history.as_slices().0);
            let thr = [
                (-15.0 * variance[0] + 1.55) * avg[0],
                (-15.0 * variance[1] + 1.55) * avg[1],
            ];
            is_bass = (band_energies[0] - BASS_OFFSET) > thr[0];
            _is_lowmid = (band_energies[1] - LOWMID_OFFSET) > thr[1];
        }

        // update history (keep ~1 second)
        if self.beat_history.len() >= self.beat_history_max {
            self.beat_history.pop_front();
        }
        self.beat_history.push_back(band_energies);

        // mark kick on bass detection with a refractory window tied to bpm
        if is_bass {
            let time_since_last_kick = self
                .last_kick_at
                .map(|t| Instant::now().duration_since(t).as_secs_f32())
                .unwrap_or(10.0);
            let refractory = if self.current_bpm > 0.0 { 15.0 / self.current_bpm } else { 0.1 };
            if time_since_last_kick > refractory {
                self.last_kick_at = Some(Instant::now());
                debug!("kick marked by energy threshold");
            }
        }
    }

    fn calculate_onset_strength_and_slice(&mut self, window: &[f32]) -> (f32, Vec<f32>, f32, [f32; 2]) {
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
                let high_ratio = f32::max(f / HIGH_ALLOW_FC, 0.0);
                let high_w = f32::clamp(
                    HIGH_ALLOW_GAIN * high_ratio / (1.0 + high_ratio),
                    0.0,
                    HIGH_ALLOW_GAIN,
                );
                let w = f32::clamp(low_w + high_w, 0.0, 1.0);
                m * w
            })
            .collect();

        // downsample frequency bins to reduce micro-shifts smearing onsets
        let ds_group = 4; // average every 2 bins -> halves resolution
        let weighted_ds = downsample_average(&weighted, ds_group);

        // ensure prev buffer length matches
        if self.prev_weighted_magnitudes.len() != weighted_ds.len() {
            self.prev_weighted_magnitudes = vec![0.0; weighted_ds.len()];
        }

        let spectral_flux: f32 = weighted_ds
            .iter()
            .zip(self.prev_weighted_magnitudes.iter())
            .map(|(&current, &previous)| f32::max(current - previous, 0.0))
            .sum();
        // low-band-only flux for kick detection (map cutoff to downsampled index)
        let max_low_bin_full = min((KICK_BAND_MAX_HZ / bin_hz) as usize, weighted.len());
        let max_low_bin_ds = min(
            (max_low_bin_full + ds_group - 1) / ds_group,
            weighted_ds.len(),
        );
        let lowband_flux: f32 = weighted_ds[0..max_low_bin_ds]
            .iter()
            .zip(self.prev_weighted_magnitudes[0..max_low_bin_ds].iter())
            .map(|(&current, &previous)| f32::max(current - previous, 0.0))
            .sum();

        // compute mean magnitudes in bass and low-mid bands, normalized to 0..1 per frame
        // blog assumes fft magnitudes are normalized; approximate by dividing by frame max
        let mut max_mag = 0.0f32;
        for &m in magnitudes.iter() {
            if m > max_mag {
                max_mag = m;
            }
        }
        let denom = f32::max(max_mag, 1e-6);
        let mut band_vals = [0.0f32; 2];
        for (band_idx, (start, end)) in self.band_limits.iter().enumerate() {
            let s = min(*start, magnitudes.len());
            let e = min(*end, magnitudes.len());
            if e > s {
                let mut acc = 0.0f32;
                let mut cnt = 0usize;
                for k in s..e {
                    let v01 = magnitudes[k] / denom;
                    acc += v01;
                    cnt += 1;
                }
                let avg = if cnt > 0 { acc / cnt as f32 } else { 0.0 };
                band_vals[band_idx] = f32::clamp(avg, 0.0, 1.0);
            }
        }

        // build a spectrogram slice from weighted magnitudes with simple dynamic range mapping
        // truncate to <= 8khz so we spend vertical resolution on lows/mids
        let max_bin_inclusive = {
            let max_bin = f32::floor(SPECTRO_MAX_HZ / bin_hz) as usize;
            // clamp within computed spectrum size
            min(max_bin, weighted.len().saturating_sub(1))
        };
        let slice: Vec<f32> = weighted
            .iter()
            .take(max_bin_inclusive + 1)
            .map(|&v| {
                let v = f32::max(v, 0.0);
                let v = f32::min(v / 50.0, 1.0); // simple scale; tuned empirically
                v
            })
            .collect();

        self.prev_weighted_magnitudes.copy_from_slice(&weighted_ds);
        (spectral_flux, slice, lowband_flux, band_vals)
    }

    fn calculate_bpm(&mut self) -> f32 {
        if self.onset_history.len() < 20 {
            return self.current_bpm;
        }
        let onset_data: Vec<f32> = self.onset_history.iter().copied().collect();
        let hop_seconds = self.hop_size as f32 / self.sample_rate as f32;
        let min_lag = f32::max(f32::round(60.0 / BPM_MAX / hop_seconds), 1.0) as usize;
        let max_lag = (60.0 / BPM_MIN / hop_seconds).round() as usize;

        // estimate a bpm band from the onset fft; if absent, keep current bpm
        let onset_band = self.detect_onset_fft_band(hop_seconds);
        if onset_band.is_none() {
            trace!("diag: onset-fft found no strong peaks; skipping bpm update");
            return self.current_bpm;
        }
        let (band_bpm_min, band_bpm_max) = onset_band.unwrap();

        // convert bpm band to lag band and intersect with global [min_lag, max_lag]
        let band_start_lag = {
            let v = 60.0 / (f32::max(band_bpm_max, 1e-6) * hop_seconds);
            f32::max(f32::floor(v), 1.0) as usize
        };
        let band_end_lag = {
            let v = 60.0 / (f32::max(band_bpm_min, 1e-6) * hop_seconds);
            f32::max(f32::ceil(v), 1.0) as usize
        };

        // compute autocorrelation only once we have a valid band
        let autocorr = self.autocorrelation(&onset_data);
        let mut search_end = min(max_lag, autocorr.len());
        let mut search_start = min(min_lag, search_end);
        // intersect with band [band_start_lag, band_end_lag]
        let lower = if search_start > band_start_lag {
            search_start
        } else {
            band_start_lag
        };
        let upper_pre = if search_end < band_end_lag {
            search_end
        } else {
            band_end_lag
        };
        let upper = min(upper_pre, autocorr.len());
        if lower >= upper {
            trace!("diag: onset-fft band yields empty lag window; skipping bpm update");
            return self.current_bpm;
        }
        search_start = lower;
        search_end = upper;

        trace!(
            "diag: bpm search lags={}..{}, hop_seconds={:.6}, band_bpm=[{:.1},{:.1}]",
            search_start, search_end, hop_seconds, band_bpm_min, band_bpm_max
        );

        let mut max_peak = 0.0;
        let mut peak_lag = 1;
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

        // compute confidence against local baseline within the constrained lag window
        let end = search_end;
        let start = search_start;
        let baseline_median = if end > start + 2 {
            let slice: Vec<f32> = autocorr[start..end].to_vec();
            median(slice)
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
                median(self.bpm_history.iter().copied().collect())
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

    /// analyze the onset envelope using an fft to find a strong tempo band
    /// returns (bpm_min, bpm_max) if strong peaks are present; otherwise none
    fn detect_onset_fft_band(&self, hop_seconds: f32) -> Option<(f32, f32)> {
        let n = self.onset_history.len();
        if n < 16 {
            return None;
        }
        let mut data: Vec<f32> = self.onset_history.iter().copied().collect();
        let med = median(data.clone());
        for v in data.iter_mut() {
            *v -= med;
        }
        // apply a hann window to reduce leakage
        for (i, v) in data.iter_mut().enumerate() {
            let w = 0.5
                * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (n.saturating_sub(1)) as f32).cos());
            *v *= w;
        }
        // zero-pad to next power of two for fft efficiency
        let mut n_fft: usize = 1;
        while n_fft < n {
            n_fft <<= 1;
        }
        let fs = 1.0 / hop_seconds;
        let mut fft_in: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];
        for i in 0..n {
            fft_in[i] = Complex::new(data[i], 0.0);
        }
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);
        fft.process(&mut fft_in);

        // examine magnitude spectrum excluding dc
        let mut mags_bpms: Vec<(f32, f32)> = Vec::new();
        for k in 1..(n_fft / 2) {
            let mag = fft_in[k].norm();
            let freq_hz = k as f32 * fs / n_fft as f32;
            let bpm = freq_hz * 60.0;
            if bpm >= ONSET_FFT_BPM_SEARCH_MIN && bpm <= ONSET_FFT_BPM_SEARCH_MAX {
                mags_bpms.push((bpm, mag));
            }
        }
        if mags_bpms.is_empty() {
            return None;
        }
        let mags_only: Vec<f32> = mags_bpms.iter().map(|(_, m)| *m).collect();
        let base = median(mags_only.clone());
        let base = f32::max(base, 1e-6);
        // find peak
        let mut peak_mag = 0.0;
        let mut peak_bpm = 0.0;
        for (bpm, mag) in mags_bpms.iter() {
            if *mag > peak_mag {
                peak_mag = *mag;
                peak_bpm = *bpm;
            }
        }
        let ratio = peak_mag / base;
        if ratio < ONSET_FFT_MIN_PEAK_RATIO {
            trace!(
                "diag: onset-fft weak spectrum: peak_bpm={:.1}, ratio={:.2}",
                peak_bpm, ratio
            );
            return None;
        }
        // collect bins close to the peak
        let include_thresh = ONSET_FFT_INCLUDE_RELATIVE * peak_mag;
        let mut min_bpm = peak_bpm;
        let mut max_bpm = peak_bpm;
        let mut included = 0usize;
        for (bpm, mag) in mags_bpms.iter() {
            if *mag >= include_thresh {
                included += 1;
                if *bpm < min_bpm {
                    min_bpm = *bpm;
                }
                if *bpm > max_bpm {
                    max_bpm = *bpm;
                }
            }
        }
        // expand by spread and clamp to preferred bounds to stabilize search
        let spread_low = min_bpm * (1.0 - ONSET_FFT_SPREAD_PCT);
        let spread_high = max_bpm * (1.0 + ONSET_FFT_SPREAD_PCT);
        let band_min = f32::max(spread_low, BPM_MIN / 2.0);
        let band_max = f32::min(spread_high, BPM_MAX * 2.0);
        trace!(
            "diag: onset-fft band peak_bpm={:.1}, ratio={:.2}, band=[{:.1},{:.1}], bins={}",
            peak_bpm, ratio, band_min, band_max, included
        );
        Some((band_min, band_max))
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
