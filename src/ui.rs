use crate::shared::SharedState;
use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{error, info};
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

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

    info!("ui started");

    let window_for_loop = window.clone();
    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
            }
            Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
                if new_size.width > 0 && new_size.height > 0 {
                    config.width = new_size.width;
                    config.height = new_size.height;
                    surface.configure(&device, &config);
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
                        surface.configure(&device, &config);
                        return;
                    }
                };
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("encoder") });
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
                queue.submit(std::iter::once(encoder.finish()));
                frame.present();
                window_for_loop.request_redraw();
            }
            _ => {}
        }
    })?;
    Ok(())
}


