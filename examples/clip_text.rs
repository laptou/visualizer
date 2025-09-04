use anyhow::Result;
use visualizer::app::{run_windowed, GpuContext};
use visualizer::gfx::{BlendMode, Color, DrawContext, LayerOptions};

fn main() -> Result<()> {
    pollster::block_on(async move {
        struct State { draw: DrawContext }
        run_windowed(
            "clip text example",
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
                Ok(State { draw })
            },
            |ctx: &mut GpuContext, s: &mut State, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView| {
                // clear
                {
                    let _r = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.07, g: 0.07, b: 0.08, a: 1.0 }), store: wgpu::StoreOp::Store },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                }

                s.draw.begin_frame();
                let w = ctx.config.width as f32;
                let h = ctx.config.height as f32;
                let _ = s.draw.rect(w * 0.15, h * 0.25, w * 0.7, h * 0.5, Color::rgba(0.2, 0.3, 0.6, 1.0));

                let clip = vec![
                    [w * 0.2, h * 0.3],
                    [w * 0.8, h * 0.3],
                    [w * 0.8, h * 0.7],
                    [w * 0.2, h * 0.7],
                ];
                s.draw.with_layer(
                    LayerOptions { blend: BlendMode::Alpha, clip_polygon: Some(clip), z_index: 1 },
                    |d| {
                        d.text(w * 0.25, h * 0.48, "clipped text sample", 42.0, Color::rgba(1.0, 1.0, 1.0, 1.0));
                        let _ = d.rect(w * 0.22, h * 0.32, w * 0.56, h * 0.36, Color::rgba(1.0, 0.6, 0.2, 0.4));
                    },
                );

                s.draw.render(encoder, view)?;
                Ok(())
            },
        )
    })
}


