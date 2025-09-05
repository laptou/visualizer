use crate::gfx::{Color, DrawContext};
use anyhow::{Context, Result, anyhow};

/// axis scaling function for mapping output coordinates (0..1) to source indices
#[derive(Copy, Clone, Debug)]
pub enum AxisScale {
    /// direct mapping: f(t) = t
    Linear,
    /// logarithmic-like mapping using log(1 + factor * t) normalized to 0..1
    /// factor controls the curvature; larger values increase compression near 0
    Log { factor: f32 },
}

impl Default for AxisScale {
    fn default() -> Self {
        AxisScale::Linear
    }
}

/// intensity scaling applied to input values before tinting
#[derive(Copy, Clone, Debug)]
pub enum IntensityScale {
    /// direct mapping: g(v) = v
    Linear,
    /// logarithmic compression: g(v) = log(1 + k * v) / log(1 + k)
    /// k controls the compression strength (default ~9.0 mimics prior behavior)
    Log { k: f32 },
}

impl Default for IntensityScale {
    fn default() -> Self {
        IntensityScale::Linear
    }
}

/// configuration for an intensity chart
#[derive(Copy, Clone, Debug)]
pub struct IntensityChartOptions {
    /// display texture resolution in columns (x) and rows (y)
    pub display_cols: u32,
    pub display_rows: u32,
    /// axis mapping from display space (0..1) to source data space
    pub x_scale: AxisScale,
    pub y_scale: AxisScale,
    /// intensity mapping before tinting
    pub intensity_scale: IntensityScale,
    /// rgb tint applied to scaled intensity (0..1); alpha is fixed to 1.0
    pub tint: Color,
}

impl Default for IntensityChartOptions {
    fn default() -> Self {
        Self {
            display_cols: 1,
            display_rows: 1,
            x_scale: AxisScale::Linear,
            y_scale: AxisScale::Linear,
            intensity_scale: IntensityScale::Linear,
            tint: Color::rgba(1.0, 1.0, 1.0, 1.0),
        }
    }
}

/// renders a 2d intensity field as a tinted rgba texture
/// overview: samples the provided source data using configurable axis scaling
/// into a display-sized rgba buffer, then uploads it to a gpu texture and
/// issues a textured quad draw into requested bounds.
pub struct IntensityChart {
    opts: IntensityChartOptions,
    tex_id: Option<usize>,
    cpu_rgba: Vec<u8>,
}

impl IntensityChart {
    pub fn new(opts: IntensityChartOptions) -> Self {
        Self {
            opts,
            tex_id: None,
            cpu_rgba: Vec::new(),
        }
    }

    /// change display resolution; recreates texture on the next update
    pub fn set_display_dims(&mut self, cols: u32, rows: u32) {
        self.opts.display_cols = cols.max(1);
        self.opts.display_rows = rows.max(1);
        // drop texture so it is recreated with new size on next update
        self.tex_id = None;
        self.cpu_rgba.clear();
    }

    /// ensure the backing texture exists with the correct size
    fn ensure_texture(&mut self, draw: &mut DrawContext) -> usize {
        if let Some(id) = self.tex_id {
            return id;
        }
        let id = draw.create_texture_rgba8(self.opts.display_cols, self.opts.display_rows);
        self.tex_id = Some(id);
        id
    }

    /// map 0..1 coordinate using axis scaling
    fn map_axis(t: f32, scale: AxisScale) -> f32 {
        let t = f32::clamp(t, 0.0, 1.0);
        match scale {
            AxisScale::Linear => t,
            AxisScale::Log { factor } => {
                let k = f32::max(factor, 0.0);
                if k == 0.0 {
                    t
                } else {
                    let num = f32::ln(1.0 + k * t);
                    let den = f32::ln(1.0 + k);
                    if den > 0.0 { num / den } else { t }
                }
            }
        }
    }

    /// map raw value 0..1 using intensity scaling
    fn map_intensity(v: f32, scale: IntensityScale) -> f32 {
        let v = f32::clamp(v, 0.0, 1.0);
        match scale {
            IntensityScale::Linear => v,
            IntensityScale::Log { k } => {
                let k = f32::max(k, 0.0);
                if k == 0.0 {
                    v
                } else {
                    let num = f32::ln(1.0 + k * v);
                    let den = f32::ln(1.0 + k);
                    if den > 0.0 { num / den } else { v }
                }
            }
        }
    }

    /// update the chart's texture from source data with given source dimensions
    /// data length must be src_cols * src_rows
    pub fn update(&mut self, draw: &mut DrawContext, src_cols: u32, src_rows: u32, data: &[f32]) -> Result<()> {
        let expected = (src_cols as usize) * (src_rows as usize);
        if data.len() != expected {
            return Err(anyhow!("intensity chart data size mismatch"));
        }

        // ensure texture of current display size exists
        let tex_id = self.ensure_texture(draw);
        let _ = tex_id; // silence unused in release when not needed directly

        let w = self.opts.display_cols as usize;
        let h = self.opts.display_rows as usize;
        self.cpu_rgba.resize(w * h * 4, 0);

        // resample from source into display grid using axis scalers
        for oy in 0..h {
            let ty = if h <= 1 { 0.0 } else { oy as f32 / (h as f32 - 1.0) };
            let sy01 = Self::map_axis(ty, self.opts.y_scale);
            let sy = sy01 * f32::max(src_rows as f32 - 1.0, 0.0);
            let syi = f32::clamp(f32::round(sy), 0.0, (src_rows as f32 - 1.0).max(0.0)) as u32;
            for ox in 0..w {
                let tx = if w <= 1 { 0.0 } else { ox as f32 / (w as f32 - 1.0) };
                let sx01 = Self::map_axis(tx, self.opts.x_scale);
                let sx = sx01 * f32::max(src_cols as f32 - 1.0, 0.0);
                let sxi = f32::clamp(f32::round(sx), 0.0, (src_cols as f32 - 1.0).max(0.0)) as u32;

                let s_idx = (syi * src_cols + sxi) as usize;
                let v = data[s_idx];
                let l = Self::map_intensity(v, self.opts.intensity_scale);

                // tint multiplication; convert to u8
                let to_u8 = |f: f32| -> u8 { f32::round(f32::clamp(f, 0.0, 1.0) * 255.0) as u8 };
                let r = to_u8(self.opts.tint.r * l);
                let g = to_u8(self.opts.tint.g * l);
                let b = to_u8(self.opts.tint.b * l);
                let o = (oy * w + ox) * 4;
                self.cpu_rgba[o + 0] = r;
                self.cpu_rgba[o + 1] = g;
                self.cpu_rgba[o + 2] = b;
                self.cpu_rgba[o + 3] = 255;
            }
        }

        if let Some(id) = self.tex_id {
            draw
                .update_texture_rgba8(id, &self.cpu_rgba)
                .context("intensity chart update texture")?;
        }
        Ok(())
    }

    /// draw the chart texture stretched into [x, y, w, h]
    pub fn render(&mut self, draw: &mut DrawContext, x: f32, y: f32, w: f32, h: f32) -> Result<()> {
        if let Some(id) = self.tex_id {
            draw.texture(x, y, w, h, id, Color::rgba(1.0, 1.0, 1.0, 1.0))?;
        }
        Ok(())
    }
}


