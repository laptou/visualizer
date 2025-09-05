//! small math helpers used across the project
//! overview: keep common computations like median here to reduce cognitive load

/// median of a list of f32 values; returns 0.0 for empty input
/// note: input is taken by value and sorted in place for simplicity
pub fn median(mut values: Vec<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

/// mean (average) of a slice; returns 0.0 for empty slice
pub fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f32 = values.iter().copied().sum();
    sum / values.len() as f32
}

/// clamp to 0..1
pub fn clamp01(x: f32) -> f32 {
    f32::clamp(x, 0.0, 1.0)
}

/// linear interpolation between `a` and `b` by `t` in 0..1
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}


