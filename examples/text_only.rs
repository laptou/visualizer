use anyhow::Result;
use visualizer::app::{GpuContext, run_windowed};
use visualizer::gfx::{Color, DrawContext, TextHAlign, TextOptions, TextVAlign};

// minimal example: clear to black and render large white text, no clipping
// overview: initializes drawcontext, clears the surface, and draws centered text
fn main() -> Result<()> {
    // initialize tracing subscriber for logging
    tracing_subscriber::fmt::init();

    pollster::block_on(async move {
        struct State {
            draw: DrawContext,
        }

        run_windowed(
            "text only",
            800,
            450,
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
            |ctx: &mut GpuContext,
             s: &mut State,
             encoder: &mut wgpu::CommandEncoder,
             view: &wgpu::TextureView| {
                s.draw.begin_frame();
                // clear to black
                s.draw.clear(Color::rgba(0.0, 0.0, 0.0, 1.0));

                // centered, large, white text
                let w = ctx.config.width as f32;
                let h = ctx.config.height as f32;
                let opts = TextOptions {
                    px: 72.0,
                    color: Color::rgba(1.0, 1.0, 1.0, 1.0),
                    halign: TextHAlign::Center,
                    valign: TextVAlign::Middle,
                    ..Default::default()
                };
                s.draw.text_with(w * 0.5, h * 0.5, "hello, text", opts);

                s.draw.render(encoder, view)?;
                Ok(())
            },
        )
    })
}
