use anyhow::Result;
use visualizer::app::{GpuContext, run_windowed};
use visualizer::gfx::{Color, DrawContext};

// example: draw a simple polygon
// overview: initializes drawcontext, clears the surface, and draws a concave polygon
fn main() -> Result<()> {
    // initialize tracing subscriber for logging
    tracing_subscriber::fmt::init();

    pollster::block_on(async move {
        struct State {
            draw: DrawContext,
        }

        run_windowed(
            "polygon",
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
                // clear to dark gray
                s.draw.clear(Color::rgba(0.07, 0.07, 0.08, 1.0));

                let w = ctx.config.width as f32;
                let h = ctx.config.height as f32;
                let cx = w * 0.5;
                let cy = h * 0.5;

                // concave five-point polygon (a simple "house" shape)
                // note: polygon requires at least 3 points; we provide 6 for a concave example
                let pts = vec![
                    [cx - 120.0, cy + 60.0],
                    [cx + 120.0, cy + 60.0],
                    [cx + 120.0, cy - 20.0],
                    [cx, cy - 100.0],
                    [cx - 120.0, cy - 20.0],
                    [cx - 120.0, cy + 60.0],
                ];
                let _ = s.draw.polygon(&pts, Color::rgba(0.9, 0.6, 0.2, 0.95));

                s.draw.render(encoder, view)?;
                Ok(())
            },
        )
    })
}
