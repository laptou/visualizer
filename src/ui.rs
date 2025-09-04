use crate::shared::SharedState;
use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{error, info};
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use crate::gfx::Immediate2D;
use crate::gfx::Color;

/// simple wgpu surface renderer that clears the screen with a color each frame
/// overview: we derive a beat phase from bpm and wall time, and map that to a
/// red flash intensity. the color is stronger around beat instants.
pub async fn run_ui(shared: Arc<Mutex<SharedState>>) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("visualizer: bpm --")
        .with_inner_size(winit::dpi::LogicalSize::new(640.0, 360.0))
        .build(&event_loop)?;
    let window = Arc::new(window);

    // instance + surface
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        dx12_shader_compiler: Default::default(),
        flags: wgpu::InstanceFlags::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });
    let surface = instance.create_surface(window.as_ref())?;

    // adapter + device + queue
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| anyhow::anyhow!("no wgpu adapter found"))?;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
            },
            None,
        )
        .await?;

    // configure surface
    let size = window.inner_size();
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    let mut last_title_update = Instant::now();

    // immediate 2d renderer
    let device_arc = Arc::new(device);
    let queue_arc = Arc::new(queue);
    let mut im2d = Immediate2D::new(&device_arc, &queue_arc, config.format, config.width, config.height)?;

    info!("ui started");

    let window_for_loop = window.clone();
    let device_for_loop = device_arc.clone();
    let queue_for_loop = queue_arc.clone();
    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
            }
            Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
                if new_size.width > 0 && new_size.height > 0 {
                    config.width = new_size.width;
                    config.height = new_size.height;
                    surface.configure(&device_for_loop, &config);
                    im2d.resize(config.format, config.width, config.height);
                }
            }
            Event::AboutToWait => {
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
                    if phase < attack { 1.0 } else { ((1.0 - (phase - attack) / (1.0 - attack))).max(0.0) }
                } else { 0.05 };

                // dampen if essentially silent
                let silence = if avg_rms < 0.001 { 0.2 } else { 1.0 };
                let r = (flash * silence).clamp(0.05, 1.0);
                let g = 0.05f32;
                let b = 0.06f32;

                if last_title_update.elapsed() >= Duration::from_millis(250) {
                    if bpm > 0.0 { window_for_loop.set_title(&format!("visualizer: {:.1} bpm", bpm)); }
                    else { window_for_loop.set_title("visualizer: analyzing..."); }
                    last_title_update = Instant::now();
                }

                // render pass: clear to color
                let frame = match surface.get_current_texture() {
                    Ok(f) => f,
                    Err(e) => {
                        error!("surface error: {}", e);
                        surface.configure(&device_for_loop, &config);
                        return;
                    }
                };
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device_for_loop.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("encoder") });
                {
                    let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
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

                // build immediate ui for this frame
                im2d.begin_frame();
                let w = config.width as f32;
                let h = config.height as f32;
                // animated rectangle based on beat phase
                let rect_w = (w * 0.2) * (0.5 + 0.5 * flash);
                let rect_h = 40.0;
                let x = (w - rect_w) * phase;
                let y = h * 0.1;
                let _ = im2d.rect(x, y, rect_w, rect_h, Color::rgba(0.2, 0.8, 0.6, 1.0));
                // a simple polygon bar responding to rms
                let bar_h = (avg_rms * 600.0).clamp(2.0, h * 0.5);
                let poly = vec![[w*0.1, h*0.9], [w*0.1 + 10.0, h*0.9], [w*0.1 + 10.0, h*0.9 - bar_h], [w*0.1, h*0.9 - bar_h]];
                let _ = im2d.polygon(&poly, Color::rgba(0.9, 0.9, 0.2, 1.0));
                // text showing bpm
                if bpm > 0.0 {
                    im2d.text(w*0.05, h*0.08, &format!("{:.1} bpm", bpm), 28.0, Color::rgba(1.0, 1.0, 1.0, 1.0));
                } else {
                    im2d.text(w*0.05, h*0.08, "analyzing...", 28.0, Color::rgba(1.0, 1.0, 1.0, 1.0));
                }

                // render accumulated draws
                im2d.render(&mut encoder, &view);
                queue_for_loop.submit(std::iter::once(encoder.finish()));
                frame.present();
                window_for_loop.request_redraw();
            }
            _ => {}
        }
    })?;
    Ok(())
}


