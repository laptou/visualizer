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

/// downsample a spectrum by averaging groups of `group_size` bins
/// note: last partial group is averaged over its actual length
pub fn downsample_average(values: &[f32], group_size: usize) -> Vec<f32> {
    if values.is_empty() || group_size == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity((values.len() + group_size - 1) / group_size);
    let mut i = 0;
    while i < values.len() {
        let end = std::cmp::min(i + group_size, values.len());
        let sum: f32 = values[i..end].iter().copied().sum();
        let avg = sum / (end - i) as f32;
        out.push(avg);
        i = end;
    }
    out
}
