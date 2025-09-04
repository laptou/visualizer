use anyhow::Result;
use std::time::Instant;
use visualizer::app::{run_windowed, GpuContext};
use visualizer::gfx::{BlendMode, Color, DrawContext, LayerOptions};

fn main() -> Result<()> {
    pollster::block_on(async move {
        struct State {
            draw: DrawContext,
            t0: Instant,
        }
        run_windowed(
            "drawcontext showcase",
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
                Ok(State { draw, t0: Instant::now() })
            },
            |ctx: &mut GpuContext, s: &mut State, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView| {
                // clear
                {
                    let _r = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                }

                s.draw.begin_frame();
                let w = ctx.config.width as f32;
                let h = ctx.config.height as f32;
                let t = s.t0.elapsed().as_secs_f32();

                let _ = s.draw.rect(20.0, 20.0, 120.0, 60.0, Color::rgba(0.2, 0.6, 1.0, 1.0));
                let _ = s.draw.rect(80.0, 50.0, 140.0, 60.0, Color::rgba(1.0, 0.4, 0.4, 0.7));

                // star polygon
                let cx = w * 0.5;
                let cy = h * 0.5;
                let r1 = 80.0; let r2 = 40.0;
                let mut pts = Vec::new();
                for i in 0..10 {
                    let a = (i as f32) * std::f32::consts::TAU / 10.0;
                    let r = if i % 2 == 0 { r1 } else { r2 };
                    pts.push([cx + r * a.cos(), cy + r * a.sin()]);
                }
                let _ = s.draw.polygon(&pts, Color::rgba(0.9, 0.9, 0.2, 0.9));

                // additive layer
                s.draw.with_layer(
                    LayerOptions { blend: BlendMode::Additive, clip_polygon: None, z_index: 1 },
                    |d| {
                        let x = (w * 0.5) + (t.sin() * 0.3 + 0.3) * w * 0.2;
                        let _ = d.rect(x, cy - 20.0, 160.0, 40.0, Color::rgba(0.2, 1.0, 0.6, 0.6));
                    },
                );

                // clip layer with text
                let mut clip = Vec::new();
                let r = 90.0;
                for i in 0..24 { let a = (i as f32) * std::f32::consts::TAU / 24.0; clip.push([cx + r * a.cos(), cy + r * a.sin()]); }
                s.draw.with_layer(
                    LayerOptions { blend: BlendMode::Alpha, clip_polygon: Some(clip), z_index: 2 },
                    |d| { d.text(cx - 120.0, cy - 10.0, "masked text", 36.0, Color::rgba(1.0, 1.0, 1.0, 1.0)); },
                );

                s.draw.render(encoder, view)?;
                Ok(())
            },
        )
    })
}


