use anyhow::{Result, anyhow};
use bytemuck::{Pod, Zeroable};
use lyon_geom as geom;
use lyon_path as path;
use lyon_tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupLayout, BufferAddress, Device, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, TextureFormat, TextureView,
};

/// rgba color in linear space (0..1)
#[derive(Copy, Clone, Debug)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
    fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuVertex {
    pos: [f32; 2],
    color: [f32; 4],
}

/// immediate-mode 2d batcher for rectangles, polygons and text
/// overview: shapes are tessellated via lyon into triangles and drawn with a
/// simple colored pipeline. text is handled by glyphon (cosmic-text) in a
/// separate pass.
pub struct Immediate2D {
    device: Arc<Device>,
    queue: Arc<Queue>,
    surface_format: TextureFormat,
    width: u32,
    height: u32,

    // shape pipeline
    pipeline: wgpu::RenderPipeline,
    screen_bind_group_layout: BindGroupLayout,
    screen_bind_group: BindGroup,
    screen_uniform: wgpu::Buffer,

    // geometry accumulators per frame
    vertices: Vec<GpuVertex>,
    indices: Vec<u32>,
}

impl Immediate2D {
    pub fn new(
        device: &Arc<Device>,
        queue: &Arc<Queue>,
        format: TextureFormat,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        // create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("immediate2d shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(IMMEDIATE_WGSL)),
        });

        // uniform for screen size
        let screen_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("screen bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let screen_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screen uniform"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let screen_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("screen bg"),
            layout: &screen_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: screen_uniform.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("immediate2d pipeline layout"),
            bind_group_layouts: &[&screen_bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuVertex>() as BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    shader_location: 1,
                    offset: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("immediate2d pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let this = Self {
            device: device.clone(),
            queue: queue.clone(),
            surface_format: format,
            width,
            height,
            pipeline,
            screen_bind_group_layout,
            screen_bind_group,
            screen_uniform,
            vertices: Vec::new(),
            indices: Vec::new(),
        };
        this.update_screen_uniform();
        Ok(this)
    }

    fn update_screen_uniform(&self) {
        let data = [self.width as f32, self.height as f32];
        self.queue
            .write_buffer(&self.screen_uniform, 0, bytemuck::cast_slice(&data));
    }

    pub fn resize(&mut self, format: TextureFormat, width: u32, height: u32) {
        self.surface_format = format;
        self.width = width;
        self.height = height;
        self.update_screen_uniform();
    }

    pub fn begin_frame(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color) -> Result<()> {
        // build rectangle path manually in clockwise order
        let mut builder = path::Path::builder();
        builder.begin(geom::point(x, y));
        builder.line_to(geom::point(x + w, y));
        builder.line_to(geom::point(x + w, y + h));
        builder.line_to(geom::point(x, y + h));
        builder.end(true);
        let p = builder.build();
        self.tessellate_path(&p, color)
    }

    pub fn polygon(&mut self, points: &[[f32; 2]], color: Color) -> Result<()> {
        if points.len() < 3 {
            return Err(anyhow!("polygon needs at least 3 points"));
        }
        let mut builder = path::Path::builder();
        builder.begin(geom::point(points[0][0], points[0][1]));
        for p in &points[1..] {
            builder.line_to(geom::point(p[0], p[1]));
        }
        builder.end(true);
        let path = builder.build();
        self.tessellate_path(&path, color)
    }

    fn tessellate_path(&mut self, path: &path::Path, color: Color) -> Result<()> {
        let mut geometry: VertexBuffers<GpuVertex, u32> = VertexBuffers::new();
        let mut tess = FillTessellator::new();
        tess.tessellate_path(
            path,
            &FillOptions::default(),
            &mut BuffersBuilder::new(&mut geometry, |v: FillVertex| {
                let p = v.position();
                GpuVertex {
                    pos: [p.x, p.y],
                    color: color.to_array(),
                }
            }),
        )
        .map_err(|e| anyhow!("tessellation error: {:?}", e))?;

        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&geometry.vertices);
        self.indices
            .extend(geometry.indices.into_iter().map(|i| i + base));
        Ok(())
    }

    pub fn text(&mut self, _x: f32, _y: f32, _text: &str, _px: f32, _color: Color) {
        // text rendering will be wired up with cosmic-text in a future step
    }

    /// record draw commands into the encoder targeting the given view
    pub fn render(&mut self, encoder: &mut wgpu::CommandEncoder, view: &TextureView) {
        // shapes pass
        if !self.indices.is_empty() {
            let vsize = (self.vertices.len() * std::mem::size_of::<GpuVertex>()) as u64;
            let isize = (self.indices.len() * std::mem::size_of::<u32>()) as u64;
            let vbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("im2d vbuf"),
                size: vsize,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let ibuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("im2d ibuf"),
                size: isize,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&vbuf, 0, bytemuck::cast_slice(&self.vertices));
            self.queue
                .write_buffer(&ibuf, 0, bytemuck::cast_slice(&self.indices));

            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("im2d shapes"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.screen_bind_group, &[]);
            rpass.set_vertex_buffer(0, vbuf.slice(..));
            rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..self.indices.len() as u32, 0, 0..1);
        }

        // text pass omitted in this scaffolding stage
    }
}

const IMMEDIATE_WGSL: &str = r#"
struct Screen { size: vec2<f32> };
@group(0) @binding(0) var<uniform> screen: Screen;

struct VsIn {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let ndc_x = (in.pos.x / screen.size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (in.pos.y / screen.size.y) * 2.0;
    out.pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;
