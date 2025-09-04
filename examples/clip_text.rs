use anyhow::Result;
use visualizer::app::{run_windowed, GpuContext};
use visualizer::gfx::{BlendMode, Color, DrawContext, LayerOptions, TextOptions, TextHAlign, TextVAlign};

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
                s.draw.begin_frame();
                // clear via draw context api
                s.draw.clear(Color::rgba(0.07, 0.07, 0.08, 1.0));

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
                        let opts = TextOptions { px: 42.0, color: Color::rgba(1.0, 1.0, 1.0, 1.0), halign: TextHAlign::Left, valign: TextVAlign::Baseline, ..Default::default() };
                        d.text_with(w * 0.25, h * 0.48, "clipped text sample", opts);
                        let _ = d.rect(w * 0.22, h * 0.32, w * 0.56, h * 0.36, Color::rgba(1.0, 0.6, 0.2, 0.4));
                    },
                );

                s.draw.render(encoder, view)?;
                Ok(())
            },
        )
    })
}


