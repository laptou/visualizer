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
    /// time of last update from audio thread
    pub last_update: Instant,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            current_bpm: 0.0,
            avg_rms: 0.0,
            last_beat_at: None,
            last_update: Instant::now(),
        }
    }

    /// set latest bpm and rms from audio thread
    pub fn set_measurements(&mut self, bpm: f32, avg_rms: f32, last_beat_at: Option<Instant>) {
        self.current_bpm = bpm;
        self.avg_rms = avg_rms;
        self.last_beat_at = last_beat_at;
        self.last_update = Instant::now();
    }
}


