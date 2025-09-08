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

/// variance (population) of a slice; returns 0.0 for empty slice
/// note: uses population variance (divides by n), which matches our rolling history use
pub fn variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let m = mean(values);
    let mut acc = 0.0;
    for &v in values {
        let d = v - m;
        acc += d * d;
    }
    acc / values.len() as f32
}

/// compute mean and variance together for efficiency; returns (mean, variance)
pub fn mean_and_variance(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let m = mean(values);
    let mut acc = 0.0;
    for &v in values {
        let d = v - m;
        acc += d * d;
    }
    (m, acc / values.len() as f32)
}

/// compute per-component mean and variance for a slice of 2d samples
/// returns ([mean0, mean1], [var0, var1])
pub fn mean_and_variance2(values: &[[f32; 2]]) -> ([f32; 2], [f32; 2]) {
    if values.is_empty() {
        return ([0.0, 0.0], [0.0, 0.0]);
    }
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    for v in values {
        sum0 += v[0];
        sum1 += v[1];
    }
    let n = values.len() as f32;
    let m0 = sum0 / n;
    let m1 = sum1 / n;
    let mut acc0 = 0.0;
    let mut acc1 = 0.0;
    for v in values {
        let d0 = v[0] - m0;
        let d1 = v[1] - m1;
        acc0 += d0 * d0;
        acc1 += d1 * d1;
    }
    ([m0, m1], [acc0 / n, acc1 / n])
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
