use std::time::Instant;

/// shared application state between audio processing and ui rendering
/// overview: audio updates bpm and rms; ui reads bpm to show in title and
/// flashes screen red in time with the beat.
#[derive(Debug)]
pub struct SharedState {
    /// current estimated bpm (0.0 when unknown)
    pub current_bpm: f32,
    /// recent average rms (for basic silence detection)
    pub avg_rms: f32,
    /// last time a beat onset was detected; used to derive phase in ui
    pub last_beat_at: Option<Instant>,
    /// last time a strong low-frequency onset (kick) was detected
    pub last_kick_at: Option<Instant>,
    /// time of last update from audio thread
    pub last_update: Instant,

    /// spectrogram storage: bins x columns, newest column at the rightmost index
    /// values are normalized 0..1
    spectro_bins: usize,
    spectro_cols: usize,
    spectrogram: Vec<f32>,
    /// monotonically increasing counter incremented whenever a new slice is pushed
    pub spectrogram_version: u64,

    /// rolling onset intensity values (newest at the rightmost index)
    onset_len: usize,
    onset: Vec<f32>,
    /// version bump whenever a new onset value is pushed
    pub onset_version: u64,

    /// rolling low-frequency onset intensity values (newest at the rightmost index)
    low_onset_len: usize,
    low_onset: Vec<f32>,
    /// version bump whenever a new low onset value is pushed
    pub low_onset_version: u64,

    /// debug: beat detection metrics for diagnostics
    pub debug: BeatDebug,
}

impl SharedState {
    pub fn new() -> Self {
        // choose compact square spectrogram storage
        let spectro_bins = SPECTRO_BINS as usize;
        let spectro_cols = SPECTRO_COLS as usize;
        Self {
            current_bpm: 0.0,
            avg_rms: 0.0,
            last_beat_at: None,
            last_kick_at: None,
            last_update: Instant::now(),
            spectro_bins,
            spectro_cols,
            spectrogram: vec![0.0; spectro_bins * spectro_cols],
            spectrogram_version: 0,
            onset_len: spectro_cols,
            onset: vec![0.0; spectro_cols],
            onset_version: 0,
            low_onset_len: spectro_cols,
            low_onset: vec![0.0; spectro_cols],
            low_onset_version: 0,
            debug: BeatDebug::default(),
        }
    }

    /// set latest bpm and rms from audio thread
    pub fn set_measurements(
        &mut self,
        bpm: f32,
        avg_rms: f32,
        last_beat_at: Option<Instant>,
        last_kick_at: Option<Instant>,
    ) {
        self.current_bpm = bpm;
        self.avg_rms = avg_rms;
        self.last_beat_at = last_beat_at;
        self.last_kick_at = last_kick_at;
        self.last_update = Instant::now();
    }

    /// push a new spectrogram column; newest data ends up in the last (rightmost) column
    pub fn push_spectrogram_slice(&mut self, slice: &[f32]) {
        if slice.len() != self.spectro_bins {
            return;
        }
        // shift existing columns left by 1 for each bin row, and write new value to last col
        // layout: row-major [bin][col]
        for b in 0..self.spectro_bins {
            let row_start = b * self.spectro_cols;
            // move [1..cols) -> [0..cols-1)
            self.spectrogram
                .copy_within(row_start + 1..row_start + self.spectro_cols, row_start);
            // write newest at last column
            self.spectrogram[row_start + self.spectro_cols - 1] = slice[b].clamp(0.0, 1.0);
        }
        self.spectrogram_version = self.spectrogram_version.wrapping_add(1);
        self.last_update = Instant::now();
    }

    pub fn spectrogram_dims(&self) -> (usize, usize) {
        (self.spectro_bins, self.spectro_cols)
    }

    pub fn spectrogram_data(&self) -> &Vec<f32> {
        &self.spectrogram
    }

    /// push a new onset value (0..1); shifts existing values left
    pub fn push_onset(&mut self, v: f32) {
        if self.onset.is_empty() {
            return;
        }
        // shift left
        self.onset.copy_within(1..self.onset_len, 0);
        self.onset[self.onset_len - 1] = f32::clamp(v, 0.0, 1.0);
        self.onset_version = self.onset_version.wrapping_add(1);
        self.last_update = Instant::now();
    }

    pub fn onset_dims(&self) -> usize {
        self.onset_len
    }
    pub fn onset_data(&self) -> &Vec<f32> {
        &self.onset
    }

    /// push a new low-frequency onset value (0..1); shifts existing values left
    pub fn push_low_onset(&mut self, v: f32) {
        if self.low_onset.is_empty() {
            return;
        }
        // shift left
        self.low_onset.copy_within(1..self.low_onset_len, 0);
        self.low_onset[self.low_onset_len - 1] = f32::clamp(v, 0.0, 1.0);
        self.low_onset_version = self.low_onset_version.wrapping_add(1);
        self.last_update = Instant::now();
    }

    pub fn low_onset_dims(&self) -> usize {
        self.low_onset_len
    }
    pub fn low_onset_data(&self) -> &Vec<f32> {
        &self.low_onset
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BeatDebug {
    /// current frame band energies [bass, lowmid]
    pub energies: [f32; 2],
    /// rolling averages [bass, lowmid]
    pub avgs: [f32; 2],
    /// rolling variances [bass, lowmid]
    pub vars: [f32; 2],
    /// thresholds computed from variance relation [bass, lowmid]
    pub thrs: [f32; 2],
    /// detection flags
    pub detected_bass: bool,
    pub detected_lowmid: bool,
}

impl Default for BeatDebug {
    fn default() -> Self {
        Self {
            energies: [0.0, 0.0],
            avgs: [0.0, 0.0],
            vars: [0.0, 0.0],
            thrs: [0.0, 0.0],
            detected_bass: false,
            detected_lowmid: false,
        }
    }
}

impl SharedState {
    /// set beat debug values; called from audio thread
    pub fn set_beat_debug(
        &mut self,
        energies: [f32; 2],
        avgs: [f32; 2],
        vars: [f32; 2],
        thrs: [f32; 2],
        detected_bass: bool,
        detected_lowmid: bool,
    ) {
        self.debug.energies = energies;
        self.debug.avgs = avgs;
        self.debug.vars = vars;
        self.debug.thrs = thrs;
        self.debug.detected_bass = detected_bass;
        self.debug.detected_lowmid = detected_lowmid;
        self.last_update = Instant::now();
    }
}

/// spectrogram resolution used for cpu staging (bins x columns)
pub const SPECTRO_BINS: u32 = 256;
pub const SPECTRO_COLS: u32 = 256;
