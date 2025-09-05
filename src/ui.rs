use crate::app::{GpuContext, run_windowed};
use crate::gfx::{Color, DrawContext};
use crate::shared::{SPECTRO_BINS, SPECTRO_COLS, SharedState};
use anyhow::{Context, Result};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::info;

/// simple wgpu surface renderer that clears the screen with a color each frame
/// overview: we derive a beat phase from bpm and wall time, and map that to a
/// red flash intensity. the color is stronger around beat instants.
pub async fn run_ui(shared: Arc<Mutex<SharedState>>) -> Result<()> {
    // app-local state holding draw context and timers
    struct UiState {
        draw: DrawContext,
        last_title_update: Instant,
        last_size: (u32, u32),
        spectro_tex_id: Option<usize>,
        spectro_version_seen: u64,
        cpu_rgba: Vec<u8>,
        onset_tex_id: Option<usize>,
        onset_version_seen: u64,
        onset_rgba: Vec<u8>,
        uv_tex_id: Option<usize>,
    }

    info!("ui started");

    run_windowed(
        "visualizer: analyzing...",
        640,
        360,
        |ctx: &mut GpuContext| {
            let draw = DrawContext::new(
                &ctx.device,
                &ctx.queue,
                ctx.config.format,
                ctx.config.width,
                ctx.config.height,
            )?;
            Ok(UiState {
                draw,
                last_title_update: Instant::now(),
                last_size: (ctx.config.width, ctx.config.height),
                spectro_tex_id: None,
                spectro_version_seen: 0,
                cpu_rgba: vec![0; (SPECTRO_BINS as usize) * (SPECTRO_COLS as usize) * 4],
                onset_tex_id: None,
                onset_version_seen: 0,
                onset_rgba: Vec::new(),
                uv_tex_id: None,
            })
        },
        |ctx: &mut GpuContext,
         state: &mut UiState,
         encoder: &mut wgpu::CommandEncoder,
         view: &wgpu::TextureView| {
            // handle resize
            let cur_size = (ctx.config.width, ctx.config.height);
            if cur_size != state.last_size {
                state
                    .draw
                    .resize(ctx.config.format, ctx.config.width, ctx.config.height);
                state.last_size = cur_size;
            }

            // read shared measurements
            let (
                bpm,
                _avg_rms,
                _last_beat_at,
                last_kick_at,
                spectro_ver,
                bins,
                cols,
                spectro_data,
                _onset_ver,
                onset_len,
                onset_data,
            ) = {
                if let Ok(s) = shared.lock() {
                    (
                        s.current_bpm,
                        s.avg_rms,
                        s.last_beat_at,
                        s.last_kick_at,
                        s.spectrogram_version,
                        s.spectrogram_dims().0 as u32,
                        s.spectrogram_dims().1 as u32,
                        s.spectrogram_data().clone(),
                        s.onset_version,
                        s.onset_dims(),
                        s.onset_data().clone(),
                    )
                } else {
                    (
                        0.0,
                        0.0,
                        None,
                        None,
                        0,
                        SPECTRO_BINS,
                        SPECTRO_COLS,
                        Vec::new(),
                        0,
                        0,
                        Vec::new(),
                    )
                }
            };
            let now = Instant::now();

            if state.last_title_update.elapsed() >= Duration::from_millis(250) {
                if bpm > 0.0 {
                    ctx.window.set_title(&format!("visualizer: {:.1} bpm", bpm));
                } else {
                    ctx.window.set_title("visualizer: analyzing...");
                }
                state.last_title_update = Instant::now();
            }

            // clear background
            {
                let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("clear pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.04,
                                g: 0.05,
                                b: 0.06,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
            }

            // draw ui
            state.draw.begin_frame();
            let w = ctx.config.width as f32;
            let h = ctx.config.height as f32;

            // ensure spectrogram texture exists
            if state.spectro_tex_id.is_none() {
                let tex_id = state.draw.create_texture_rgba8(cols, bins);
                state.spectro_tex_id = Some(tex_id);
            }

            // update spectrogram texture if there's new data
            if spectro_ver != state.spectro_version_seen && !spectro_data.is_empty() {
                // map floats -> rgba8; flip vertically so low freqs are at bottom
                let bins_usize = bins as usize;
                let cols_usize = cols as usize;
                state.cpu_rgba.resize(bins_usize * cols_usize * 4, 0);
                for b in 0..bins_usize {
                    let src_row = bins_usize - 1 - b;
                    let src_off = src_row * cols_usize;
                    let dst_off = b * cols_usize * 4;
                    for c in 0..cols_usize {
                        let v = spectro_data[src_off + c].clamp(0.0, 1.0);
                        let g = (v * 255.0) as u8;
                        let i = dst_off + c * 4;
                        state.cpu_rgba[i + 0] = g; // r
                        state.cpu_rgba[i + 1] = g; // g
                        state.cpu_rgba[i + 2] = g; // b
                        state.cpu_rgba[i + 3] = 255; // a
                    }
                }
                if let Some(id) = state.spectro_tex_id {
                    let _ = state.draw.update_texture_rgba8(id, &state.cpu_rgba);
                }
                state.spectro_version_seen = spectro_ver;
            }

            // layout: center bpm text; spectrogram square below it
            let center_x = w * 0.5;
            let center_y = h * 0.5;

            // kick-synced big square behind spectrogram
            let dt_kick = last_kick_at
                .map(|t| now.duration_since(t).as_secs_f32())
                .unwrap_or(10.0);
            let t_norm = (dt_kick / 0.6).clamp(0.0, 1.0);
            let ease = (1.0 - t_norm).powi(2);
            let max_square = 0.7 * w.min(h);
            let kick_size = (0.2 * max_square) + ease * (0.8 * max_square);
            let kick_x = center_x - kick_size * 0.5;
            let kick_y = center_y + h * 0.08 - kick_size * 0.5; // slightly below true center to make room for text
            let _ = state.draw.rect(
                kick_x,
                kick_y,
                kick_size,
                kick_size,
                Color::rgba(0.2, 0.5, 0.9, 0.25),
            );

            // spectrogram square size (bigger)
            let square = 0.55 * w.min(h);
            let spec_x = center_x - square * 0.5;
            let spec_y = center_y + h * 0.08 - square * 0.5;
            // ensure uv texture exists and encode log-y, linear-x mapping in RG
            if state.uv_tex_id.is_none() {
                let uv_w = cols.max(1);
                let uv_h = bins.max(1);
                let uv_id = state.draw.create_texture_rgba8(uv_w, uv_h);
                state.uv_tex_id = Some(uv_id);
                // build uv cpu buffer: for each output (x,y), write src uv in 0..1
                let mut buf = vec![0u8; (uv_w * uv_h * 4) as usize];
                for y in 0..uv_h {
                    let ty = y as f32 / (uv_h - 1).max(1) as f32;
                    for x in 0..uv_w {
                        let tx = x as f32 / (uv_w - 1).max(1) as f32;
                        let u = (tx.clamp(0.0, 1.0) * 255.0) as u8;
                        // compute logarithmic mapping: v = log10(1 + 9*ty)
                        let log_v = (1.0 + 9.0 * ty).log10();
                        let v = (log_v.clamp(0.0, 1.0) * 255.0) as u8;
                        let o = ((y * uv_w + x) * 4) as usize;
                        buf[o + 0] = u; // R = u
                        buf[o + 1] = v; // G = v
                        buf[o + 2] = 0;
                        buf[o + 3] = 255;
                    }
                }
                if let Some(id) = state.uv_tex_id {
                    let _ = state.draw.update_texture_rgba8(id, &buf);
                }
            }

            if let (Some(src), Some(uv)) = (state.spectro_tex_id, state.uv_tex_id) {
                let _ = state.draw.texture_with_uv(
                    spec_x,
                    spec_y,
                    square,
                    square,
                    src,
                    uv,
                    Color::rgba(1.0, 1.0, 1.0, 1.0),
                );

                let _ = state.draw.texture(
                    spec_x,
                    spec_y,
                    square,
                    square,
                    uv,
                    Color::rgba(1.0, 1.0, 1.0, 1.0),
                );
            }

            // onset graph below spectrogram: draw a filled polyline area
            if onset_len > 0 && !onset_data.is_empty() {
                // ensure onset texture exists (1px tall, stretched when drawn)
                if state.onset_tex_id.is_none() {
                    let tex_id = state.draw.create_texture_rgba8(onset_len as u32, 1);
                    state.onset_tex_id = Some(tex_id);
                    state.onset_rgba.resize(onset_len * 4, 0);
                    state.onset_version_seen = 0;
                }
                // update onset texture if new data arrived
                if state.onset_version_seen != _onset_ver {
                    state.onset_rgba.resize(onset_len * 4, 0);
                    for i in 0..onset_len {
                        let v = onset_data[i].clamp(0.0, 1.0);
                        // tint: teal-green scaled by intensity
                        let r = (0.20 * v * 255.0) as u8;
                        let g = (0.90 * v * 255.0) as u8;
                        let b = (0.60 * v * 255.0) as u8;
                        let o = i * 4;
                        state.onset_rgba[o + 0] = r;
                        state.onset_rgba[o + 1] = g;
                        state.onset_rgba[o + 2] = b;
                        state.onset_rgba[o + 3] = 255;
                    }
                    if let Some(id) = state.onset_tex_id {
                        let _ = state.draw.update_texture_rgba8(id, &state.onset_rgba);
                    }
                    state.onset_version_seen = _onset_ver;
                }
                // draw stretched intensity strip
                if let Some(id) = state.onset_tex_id {
                    let graph_w = square;
                    let graph_h = (square * 0.22).max(40.0);
                    let gx = spec_x;
                    let gy = spec_y + square + 10.0;
                    let _ = state.draw.texture(
                        gx,
                        gy,
                        graph_w,
                        graph_h,
                        id,
                        Color::rgba(1.0, 1.0, 1.0, 1.0),
                    );
                }
            }

            // bpm text in center
            let label = if bpm > 0.0 {
                format!("{:.1} bpm", bpm)
            } else {
                "analyzing...".to_string()
            };
            state.draw.text_with(
                center_x,
                center_y - square * 0.55,
                &label,
                crate::gfx::TextOptions {
                    px: (w.min(h) * 0.12).clamp(28.0, 128.0),
                    color: Color::rgba(1.0, 1.0, 1.0, 1.0),
                    halign: crate::gfx::TextHAlign::Center,
                    valign: crate::gfx::TextVAlign::Middle,
                    ..Default::default()
                },
            );

            state
                .draw
                .render(encoder, view)
                .context("drawcontext render")?;
            Ok(())
        },
    )
}
