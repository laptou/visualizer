use anyhow::{Context as AnyhowContext, Result};
use std::sync::Arc;
use tracing::{error, info};
use wgpu::{Device, Queue, TextureFormat};
use winit::window::Window;

/// shared gpu/window context provided to app init and frame callbacks
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub config: wgpu::SurfaceConfiguration,
    pub format: TextureFormat,
    pub window: Arc<Window>,
}

/// run a simple windowed wgpu app
/// init: called once with a mutable gpu context to build app state
/// frame: called each frame with the current view and encoder to record commands
pub fn run_windowed<T, Init, Frame>(
    title: &str,
    width: u32,
    height: u32,
    init: Init,
    mut frame: Frame,
) -> Result<()>
where
    Init: FnOnce(&mut GpuContext) -> Result<T>,
    Frame:
        FnMut(&mut GpuContext, &mut T, &mut wgpu::CommandEncoder, &wgpu::TextureView) -> Result<()>,
{
    let event_loop = winit::event_loop::EventLoop::new()?;
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .with_inner_size(winit::dpi::LogicalSize::new(width as f64, height as f64))
        .build(&event_loop)?;
    let window = Arc::new(window);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        backend_options: Default::default(),
        flags: wgpu::InstanceFlags::default(),
    });
    // safety: window outlives surface due to event loop ownership
    let surface = instance.create_surface(window.as_ref())?;
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .context("no wgpu adapter found")?;

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("device"),
        required_features: wgpu::Features::empty(),
        required_limits:
            wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
        ..Default::default()
    }))?;

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(caps.formats[0]);
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width,
        height: size.height,
        present_mode: caps.present_modes[0],
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    let mut ctx = GpuContext {
        instance,
        adapter,
        device: Arc::new(device),
        queue: Arc::new(queue),
        config,
        format,
        window: window.clone(),
    };

    let mut state = init(&mut ctx)?;

    let window_for_loop = window.clone();
    info!("event loop starting");

    event_loop.run(move |event, elwt| match event {
        winit::event::Event::WindowEvent { event, .. } => match event {
            winit::event::WindowEvent::CloseRequested => elwt.exit(),
            winit::event::WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    ctx.config.width = new_size.width;
                    ctx.config.height = new_size.height;
                    surface.configure(&ctx.device, &ctx.config);
                }
            }
            _ => {}
        },
        winit::event::Event::AboutToWait => {
            let frame_tex = match surface.get_current_texture() {
                Ok(f) => f,
                Err(e) => {
                    error!("surface error: {}", e);
                    surface.configure(&ctx.device, &ctx.config);
                    return;
                }
            };
            let view = frame_tex
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

            if let Err(e) = frame(&mut ctx, &mut state, &mut encoder, &view) {
                error!("frame error: {}", e);
            }

            ctx.queue.submit(std::iter::once(encoder.finish()));
            frame_tex.present();
            window_for_loop.request_redraw();
        }
        _ => {}
    })?;
    #[allow(unreachable_code)]
    Ok(())
}
