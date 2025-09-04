use crate::app::{run_windowed, GpuContext};
use crate::gfx::{Color, DrawContext};
use crate::shared::SharedState;
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
            Ok(UiState { draw, last_title_update: Instant::now(), last_size: (ctx.config.width, ctx.config.height) })
        },
        |ctx: &mut GpuContext, state: &mut UiState, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView| {
            // handle resize
            let cur_size = (ctx.config.width, ctx.config.height);
            if cur_size != state.last_size {
                state.draw.resize(ctx.config.format, ctx.config.width, ctx.config.height);
                state.last_size = cur_size;
            }

            // compute beat phase and color
            let (bpm, avg_rms, last_beat_at) = {
                if let Ok(s) = shared.lock() {
                    (s.current_bpm, s.avg_rms, s.last_beat_at)
                } else {
                    (0.0, 0.0, None)
                }
            };
            let beats_per_second = if bpm > 0.0 { bpm / 60.0 } else { 0.0 };
            let now = Instant::now();
            let phase = match (last_beat_at, beats_per_second > 0.0) {
                (Some(t), true) => {
                    let dt = now.duration_since(t).as_secs_f32();
                    (dt * beats_per_second).fract()
                }
                _ => 0.0,
            };
            // red flash envelope: quick spike at beat, decay over the rest of the cycle
            let flash = if beats_per_second > 0.0 {
                let attack = 0.1; // first 10% of cycle bright
                if phase < attack { 1.0 } else { (1.0 - (phase - attack) / (1.0 - attack)).max(0.0) }
            } else { 0.05 };
            // dampen if essentially silent
            let silence = if avg_rms < 0.001 { 0.2 } else { 1.0 };
            let r = (flash * silence).clamp(0.05, 1.0);
            let g = 0.05f32;
            let b = 0.06f32;

            if state.last_title_update.elapsed() >= Duration::from_millis(250) {
                if bpm > 0.0 {
                    ctx.window.set_title(&format!("visualizer: {:.1} bpm", bpm));
                } else {
                    ctx.window.set_title("visualizer: analyzing...");
                }
                state.last_title_update = Instant::now();
            }

            // clear
            {
                let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("clear pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: r as f64, g: g as f64, b: b as f64, a: 1.0 }),
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
            // animated rectangle based on beat phase
            let rect_w = (w * 0.2) * (0.5 + 0.5 * flash);
            let rect_h = 40.0;
            let x = (w - rect_w) * phase;
            let y = h * 0.1;
            let _ = state.draw.rect(x, y, rect_w, rect_h, Color::rgba(0.2, 0.8, 0.6, 1.0));
            // a simple polygon bar responding to rms
            let bar_h = (avg_rms * 600.0).clamp(2.0, h * 0.5);
            let poly = vec![
                [w * 0.1, h * 0.9],
                [w * 0.1 + 10.0, h * 0.9],
                [w * 0.1 + 10.0, h * 0.9 - bar_h],
                [w * 0.1, h * 0.9 - bar_h],
            ];
            let _ = state.draw.polygon(&poly, Color::rgba(0.9, 0.9, 0.2, 1.0));
            // text showing bpm
            if bpm > 0.0 {
                state.draw.text(w * 0.05, h * 0.08, &format!("{:.1} bpm", bpm), 28.0, Color::rgba(1.0, 1.0, 1.0, 1.0));
            } else {
                state.draw.text(w * 0.05, h * 0.08, "analyzing...", 28.0, Color::rgba(1.0, 1.0, 1.0, 1.0));
            }

            state.draw.render(encoder, view).context("drawcontext render")?;
            Ok(())
        },
    )
}
