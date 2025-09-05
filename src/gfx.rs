use anyhow::{Context as AnyhowContext, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use glyph::cosmic_text::{
    Attrs as CtAttrs, Buffer as CtBuffer, Color as CtColor, FontSystem as CtFontSystem,
    Metrics as CtMetrics, Shaping as CtShaping, SwashCache as CtSwashCache, Weight as CtWeight,
    Wrap as CtWrap,
};
use glyph::{Cache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport};
use glyphon as glyph;
use lyon_geom as geom;
use lyon_path as path;
use lyon_tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};
use std::sync::Arc;
// tracing used via anyhow context messages elsewhere
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
    pos: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuVertexTex {
    pos: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

/// how to blend colors into the target
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BlendMode {
    Alpha,
    Additive,
}

/// options that apply to all sub-operations within a layer
#[derive(Clone, Debug)]
pub struct LayerOptions {
    /// blending mode for sub-operations
    pub blend: BlendMode,
    /// optional clipping polygon in pixel space; if present, sub-ops are masked by this polygon
    pub clip_polygon: Option<Vec<[f32; 2]>>,
    /// z ordering for sub-operations; larger values appear in front
    pub z_index: i32,
}

impl Default for LayerOptions {
    fn default() -> Self {
        Self {
            blend: BlendMode::Alpha,
            clip_polygon: None,
            z_index: 0,
        }
    }
}

/// a buffered drawing command
#[derive(Clone, Debug)]
pub enum DrawOp {
    Rect {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: Color,
    },
    Polygon {
        points: Vec<[f32; 2]>,
        color: Color,
    },
    Text {
        x: f32,
        y: f32,
        text: String,
        opts: TextOptions,
    },
    /// svg-style path supporting lines, beziers, and arcs
    SvgPath {
        path: path::Path,
        color: Color,
    },
    Texture {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        tex_id: usize,
        color: Color,
    },
    TextureUV {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        src_tex_id: usize,
        uv_tex_id: usize,
        color: Color,
    },
    Layer {
        options: LayerOptions,
        ops: Vec<DrawOp>,
    },
}

/// horizontal alignment relative to the provided `(x, y)` anchor
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TextHAlign {
    Left,
    Center,
    Right,
}

/// vertical alignment relative to the provided `(x, y)` anchor
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TextVAlign {
    Top,
    Middle,
    Baseline,
}

/// formatting options for text drawing; sensible defaults chosen
#[derive(Clone, Debug)]
pub struct TextOptions {
    pub px: f32,
    pub color: Color,
    /// font family name; if none, system default is used
    pub family: Option<String>,
    /// font weight; defaults to regular
    pub weight: CtWeight,
    /// horizontal alignment relative to anchor
    pub halign: TextHAlign,
    /// vertical alignment relative to anchor
    pub valign: TextVAlign,
}

impl Default for TextOptions {
    fn default() -> Self {
        Self {
            px: 24.0,
            color: Color::rgba(1.0, 1.0, 1.0, 1.0),
            family: None,
            weight: CtWeight::NORMAL,
            halign: TextHAlign::Left,
            valign: TextVAlign::Top,
        }
    }
}

/// immediate-mode 2d batcher for rectangles, polygons and text
/// overview: shapes are tessellated via lyon into triangles and drawn with a
/// simple colored pipeline. text is handled by glyphon (cosmic-text) in a
/// separate pass.
pub struct DrawContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    surface_format: TextureFormat,
    width: u32,
    height: u32,

    // screen uniform
    screen_bind_group_layout: BindGroupLayout,
    screen_bind_group: BindGroup,
    screen_uniform: wgpu::Buffer,

    // pipelines
    pipeline_alpha: wgpu::RenderPipeline,
    pipeline_alpha_stencil: wgpu::RenderPipeline,
    pipeline_add: wgpu::RenderPipeline,
    pipeline_add_stencil: wgpu::RenderPipeline,
    pipeline_stencil_write: wgpu::RenderPipeline,

    // texture rendering
    texture_bind_group_layout: BindGroupLayout,
    pipeline_texture: wgpu::RenderPipeline,
    pipeline_texture_stencil: wgpu::RenderPipeline,
    textures: Vec<TextureResource>,

    // uv remap: separate shader and pipeline with two sampled textures
    uv_bind_group_layout: BindGroupLayout,
    pipeline_texture_uv: wgpu::RenderPipeline,
    pipeline_texture_uv_stencil: wgpu::RenderPipeline,

    // text subsystem (glyphon/cosmic-text)
    cache: Cache,
    font_system: CtFontSystem,
    swash_cache: CtSwashCache,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,         // no stencil
    text_renderer_stencil: TextRenderer, // stencil test == 1
    viewport: Viewport,

    // depth-stencil target
    depth_view: wgpu::TextureView,

    // buffered commands
    ops: Vec<DrawOp>,
    // requested clear color for the next render pass
    clear_color: Option<Color>,
}

impl DrawContext {
    pub fn new(
        device: &Arc<Device>,
        queue: &Arc<Queue>,
        format: TextureFormat,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        // create shader
        let shader_src = include_str!("shaders/immediate.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("drawcontext shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_src)),
        });

        // texture shader for sampling 2d textures
        let tex_shader_src = include_str!("shaders/texture.wgsl");
        let tex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("drawcontext texture shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(tex_shader_src)),
        });
        // uv-remap shader (samples uv coords from a second texture)
        let tex_uv_shader_src = include_str!("shaders/texture_uv.wgsl");
        let tex_uv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("drawcontext texture uv shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(tex_uv_shader_src)),
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
            label: Some("drawcontext pipeline layout"),
            bind_group_layouts: &[&screen_bind_group_layout],
            push_constant_ranges: &[],
        });

        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout_tex = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("drawcontext texture pipeline layout"),
            bind_group_layouts: &[&screen_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // uv bind group: sampler + source texture + uv texture (rgba where rg=uv)
        let uv_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture uv bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, view_dimension: wgpu::TextureViewDimension::D2, sample_type: wgpu::TextureSampleType::Float { filterable: true } }, count: None },
            ],
        });
        let pipeline_layout_tex_uv = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("drawcontext texture uv pipeline layout"),
            bind_group_layouts: &[&screen_bind_group_layout, &uv_bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuVertex>() as BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    shader_location: 1,
                    offset: 12,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let vertex_layout_tex = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuVertexTex>() as BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { shader_location: 0, offset: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { shader_location: 1, offset: 12, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { shader_location: 2, offset: 20, format: wgpu::VertexFormat::Float32x4 },
            ],
        };

        // depth-stencil view
        let depth_view = create_depth_stencil_view(device, width, height);

        // pipelines for shapes (alpha/additive, with/without stencil)
        let pipeline_alpha = create_shape_pipeline(
            device,
            &pipeline_layout,
            &shader,
            format,
            &vertex_layout,
            BlendMode::Alpha,
            false,
            true,
        );
        let pipeline_alpha_stencil = create_shape_pipeline(
            device,
            &pipeline_layout,
            &shader,
            format,
            &vertex_layout,
            BlendMode::Alpha,
            true,
            true,
        );
        let pipeline_add = create_shape_pipeline(
            device,
            &pipeline_layout,
            &shader,
            format,
            &vertex_layout,
            BlendMode::Additive,
            false,
            true,
        );
        let pipeline_add_stencil = create_shape_pipeline(
            device,
            &pipeline_layout,
            &shader,
            format,
            &vertex_layout,
            BlendMode::Additive,
            true,
            true,
        );
        // stencil write pipeline (color writes off, stencil replace)
        let pipeline_stencil_write = create_shape_pipeline(
            device,
            &pipeline_layout,
            &shader,
            format,
            &vertex_layout,
            BlendMode::Alpha,
            true,
            false,
        );

        // texture pipelines (with and without stencil test)
        let pipeline_texture = create_texture_pipeline(
            device,
            &pipeline_layout_tex,
            &tex_shader,
            format,
            &vertex_layout_tex,
            false,
        );
        let pipeline_texture_stencil = create_texture_pipeline(
            device,
            &pipeline_layout_tex,
            &tex_shader,
            format,
            &vertex_layout_tex,
            true,
        );

        // texture+uv pipelines
        let pipeline_texture_uv = create_texture_pipeline(
            device,
            &pipeline_layout_tex_uv,
            &tex_uv_shader,
            format,
            &vertex_layout_tex,
            false,
        );
        let pipeline_texture_uv_stencil = create_texture_pipeline(
            device,
            &pipeline_layout_tex_uv,
            &tex_uv_shader,
            format,
            &vertex_layout_tex,
            true,
        );

        // text: cache, atlas, renderer, viewport
        let cache = Cache::new(device);
        let mut text_atlas = TextAtlas::new(device, queue, &cache, format);
        let text_ds_state = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            device,
            wgpu::MultisampleState::default(),
            text_ds_state,
        );
        // viewport controls the visible area for text; must be kept in sync with screen size
        let mut viewport = Viewport::new(device, &cache);
        viewport.update(queue, glyph::Resolution { width, height });
        // text renderer with stencil test enabled (equals 1)
        let stencil_state = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState {
                front: wgpu::StencilFaceState {
                    compare: wgpu::CompareFunction::Equal,
                    fail_op: wgpu::StencilOperation::Keep,
                    depth_fail_op: wgpu::StencilOperation::Keep,
                    pass_op: wgpu::StencilOperation::Keep,
                },
                back: wgpu::StencilFaceState {
                    compare: wgpu::CompareFunction::Equal,
                    fail_op: wgpu::StencilOperation::Keep,
                    depth_fail_op: wgpu::StencilOperation::Keep,
                    pass_op: wgpu::StencilOperation::Keep,
                },
                read_mask: 0xFF,
                write_mask: 0xFF,
            },
            bias: wgpu::DepthBiasState::default(),
        });
        let text_renderer_stencil = TextRenderer::new(
            &mut text_atlas,
            device,
            wgpu::MultisampleState::default(),
            stencil_state,
        );

        let this = Self {
            device: device.clone(),
            queue: queue.clone(),
            surface_format: format,
            width,
            height,
            screen_bind_group_layout,
            screen_bind_group,
            screen_uniform,
            pipeline_alpha,
            pipeline_alpha_stencil,
            pipeline_add,
            pipeline_add_stencil,
            pipeline_stencil_write,
            texture_bind_group_layout,
            pipeline_texture,
            pipeline_texture_stencil,
            textures: Vec::new(),
            uv_bind_group_layout,
            pipeline_texture_uv,
            pipeline_texture_uv_stencil,
            cache,
            font_system: CtFontSystem::new(),
            swash_cache: CtSwashCache::new(),
            text_atlas,
            text_renderer,
            text_renderer_stencil,
            viewport,
            depth_view,
            ops: Vec::new(),
            clear_color: None,
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
        let format_changed = self.surface_format != format;
        self.surface_format = format;
        self.width = width;
        self.height = height;
        self.update_screen_uniform();
        if format_changed {
            // recreate text atlas/renderer to match new surface format
            self.text_atlas =
                TextAtlas::new(&self.device, &self.queue, &self.cache, self.surface_format);
            let text_ds_state = Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            });
            self.text_renderer = TextRenderer::new(
                &mut self.text_atlas,
                &self.device,
                wgpu::MultisampleState::default(),
                text_ds_state,
            );
            let stencil_state = Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState {
                    front: wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::Equal,
                        fail_op: wgpu::StencilOperation::Keep,
                        depth_fail_op: wgpu::StencilOperation::Keep,
                        pass_op: wgpu::StencilOperation::Keep,
                    },
                    back: wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::Equal,
                        fail_op: wgpu::StencilOperation::Keep,
                        depth_fail_op: wgpu::StencilOperation::Keep,
                        pass_op: wgpu::StencilOperation::Keep,
                    },
                    read_mask: 0xFF,
                    write_mask: 0xFF,
                },
                bias: wgpu::DepthBiasState::default(),
            });
            self.text_renderer_stencil = TextRenderer::new(
                &mut self.text_atlas,
                &self.device,
                wgpu::MultisampleState::default(),
                stencil_state,
            );
        }
        // keep glyphon viewport resolution in sync with current drawable size
        self.viewport.update(
            &self.queue,
            glyph::Resolution {
                width: self.width,
                height: self.height,
            },
        );
        self.depth_view = create_depth_stencil_view(&self.device, self.width, self.height);
    }

    pub fn begin_frame(&mut self) {
        // clear buffered draw operations for the new frame
        self.ops.clear();
        // reset clear color; user must request again per frame
        self.clear_color = None;
    }

    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color) -> Result<()> {
        // buffer a rectangle to be tessellated during render
        self.ops.push(DrawOp::Rect { x, y, w, h, color });
        Ok(())
    }

    pub fn polygon(&mut self, points: &[[f32; 2]], color: Color) -> Result<()> {
        // buffer a convex or concave polygon; requires at least 3 points
        if points.len() < 3 {
            return Err(anyhow!("polygon needs at least 3 points"));
        }
        self.ops.push(DrawOp::Polygon {
            points: points.to_vec(),
            color,
        });
        Ok(())
    }

    pub fn text(&mut self, x: f32, y: f32, text: &str, px: f32, color: Color) {
        // backwards-compatible helper using defaults
        self.text_with(
            x,
            y,
            text,
            TextOptions {
                px,
                color,
                ..Default::default()
            },
        );
    }

    /// draw text with formatting options
    pub fn text_with(&mut self, x: f32, y: f32, text: &str, opts: TextOptions) {
        // buffer a text draw; shaping happens during render
        self.ops.push(DrawOp::Text {
            x,
            y,
            text: text.to_string(),
            opts,
        });
    }

    /// request clearing the color target when `render` starts
    pub fn clear(&mut self, color: Color) {
        self.clear_color = Some(color);
    }

    /// draw an svg-style path; caller provides a lyon `Path`
    pub fn svg_path(&mut self, p: path::Path, color: Color) -> Result<()> {
        self.ops.push(DrawOp::SvgPath { path: p, color });
        Ok(())
    }

    /// draw a textured quad covering [x,y,w,h]
    pub fn texture(&mut self, x: f32, y: f32, w: f32, h: f32, tex_id: usize, color: Color) -> Result<()> {
        self.ops.push(DrawOp::Texture { x, y, w, h, tex_id, color });
        Ok(())
    }

    /// draw a textured quad with uv-remap, sampling source using coordinates in uv texture (rg channels)
    pub fn texture_with_uv(&mut self, x: f32, y: f32, w: f32, h: f32, src_tex_id: usize, uv_tex_id: usize, color: Color) -> Result<()> {
        self.ops.push(DrawOp::TextureUV { x, y, w, h, src_tex_id, uv_tex_id, color });
        Ok(())
    }

    /// translate buffered operations into wgpu commands and execute them in submission order.
    ///
    /// the renderer preserves the order that `rect`, `polygon`, `text`, and `layer` calls were made.
    /// layers affect only their children: the `blend` mode switches pipelines for the duration of the layer,
    /// `clip_polygon` writes a stencil mask that is applied to the children, and `z_index` is mapped to a
    /// depth value so children can appear above/below other content. depth is cleared each frame.
    ///
    /// limitations:
    /// - text uses alpha blending; additive for text is not supported currently.
    /// - text inside a clip layer is masked using the stencil-enabled text renderer.
    pub fn render(&mut self, encoder: &mut wgpu::CommandEncoder, view: &TextureView) -> Result<()> {
        // build an execution plan: we tessellate shapes and set up text areas in cpu memory
        #[derive(Clone)]
        enum Step {
            SetBlend(BlendMode),
            PushClip {
                vertices: Vec<GpuVertex>,
                indices: Vec<u32>,
            },
            PopClip {
                vertices: Vec<GpuVertex>,
                indices: Vec<u32>,
            },
            DrawShape {
                vertices: Vec<GpuVertex>,
                indices: Vec<u32>,
                blend: BlendMode,
                clipped: bool,
            },
            DrawText {
                buffer: CtBuffer,
                x: f32,
                y: f32,
                color: CtColor,
                clipped: bool,
            },
            DrawTexture {
                vertices: Vec<GpuVertexTex>,
                indices: Vec<u32>,
                tex_id: usize,
                clipped: bool,
            },
            DrawTextureUV {
                vertices: Vec<GpuVertexTex>,
                indices: Vec<u32>,
                src_tex_id: usize,
                uv_tex_id: usize,
                clipped: bool,
            },
        }

        // context while flattening
        #[derive(Clone, Copy)]
        struct Ctx {
            blend: BlendMode,
            z: f32,
            clipped: bool,
        }
        fn map_z_index(z_index: i32) -> f32 {
            (0.5f32 - (z_index as f32) * 0.001).clamp(0.0, 1.0)
        }

        let mut steps: Vec<Step> = Vec::new();

        // helper: tessellate path to vertex buffers at depth z
        let mut tessellate_to =
            |geom_path: &path::Path, color: Color, z: f32| -> Result<(Vec<GpuVertex>, Vec<u32>)> {
                let mut geometry: VertexBuffers<GpuVertex, u32> = VertexBuffers::new();
                let mut tess = FillTessellator::new();
                tess.tessellate_path(
                    geom_path,
                    &FillOptions::default(),
                    &mut BuffersBuilder::new(&mut geometry, |v: FillVertex| {
                        let p = v.position();
                        GpuVertex {
                            pos: [p.x, p.y, z],
                            color: color.to_array(),
                        }
                    }),
                )
                .map_err(|e| anyhow!("tessellation error: {:?}", e))?;
                Ok((geometry.vertices, geometry.indices))
            };

        // flatten ops recursively
        fn flatten(
            this: &mut DrawContext,
            ops: &[DrawOp],
            steps: &mut Vec<Step>,
            ctx: Ctx,
            tessellate_to: &mut dyn FnMut(
                &path::Path,
                Color,
                f32,
            ) -> Result<(Vec<GpuVertex>, Vec<u32>)>,
        ) -> Result<()> {
            let mut cur_ctx = ctx;
            for op in ops {
                match op.clone() {
                    DrawOp::Rect { x, y, w, h, color } => {
                        let mut builder = path::Path::builder();
                        builder.begin(geom::point(x, y));
                        builder.line_to(geom::point(x + w, y));
                        builder.line_to(geom::point(x + w, y + h));
                        builder.line_to(geom::point(x, y + h));
                        builder.end(true);
                        let p = builder.build();
                        let (v, i) = tessellate_to(&p, color, cur_ctx.z)?;
                        steps.push(Step::DrawShape {
                            vertices: v,
                            indices: i,
                            blend: cur_ctx.blend,
                            clipped: cur_ctx.clipped,
                        });
                    }
                    DrawOp::Polygon { points, color } => {
                        let mut builder = path::Path::builder();
                        builder.begin(geom::point(points[0][0], points[0][1]));
                        for pnt in points.iter().skip(1) {
                            builder.line_to(geom::point(pnt[0], pnt[1]));
                        }
                        builder.end(true);
                        let p = builder.build();
                        let (v, i) = tessellate_to(&p, color, cur_ctx.z)?;
                        steps.push(Step::DrawShape {
                            vertices: v,
                            indices: i,
                            blend: cur_ctx.blend,
                            clipped: cur_ctx.clipped,
                        });
                    }
                    DrawOp::SvgPath { path, color } => {
                        let (v, i) = tessellate_to(&path, color, cur_ctx.z)?;
                        steps.push(Step::DrawShape {
                            vertices: v,
                            indices: i,
                            blend: cur_ctx.blend,
                            clipped: cur_ctx.clipped,
                        });
                    }
                    DrawOp::Texture { x, y, w, h, tex_id, color } => {
                        let z = cur_ctx.z;
                        let v = vec![
                            GpuVertexTex { pos: [x, y, z], uv: [0.0, 0.0], color: color.to_array() },
                            GpuVertexTex { pos: [x + w, y, z], uv: [1.0, 0.0], color: color.to_array() },
                            GpuVertexTex { pos: [x + w, y + h, z], uv: [1.0, 1.0], color: color.to_array() },
                            GpuVertexTex { pos: [x, y + h, z], uv: [0.0, 1.0], color: color.to_array() },
                        ];
                        let i = vec![0u32, 1, 2, 0, 2, 3];
                        steps.push(Step::DrawTexture { vertices: v, indices: i, tex_id, clipped: cur_ctx.clipped });
                    }
                    DrawOp::TextureUV { x, y, w, h, src_tex_id, uv_tex_id, color } => {
                        let z = cur_ctx.z;
                        let v = vec![
                            GpuVertexTex { pos: [x, y, z], uv: [0.0, 0.0], color: color.to_array() },
                            GpuVertexTex { pos: [x + w, y, z], uv: [1.0, 0.0], color: color.to_array() },
                            GpuVertexTex { pos: [x + w, y + h, z], uv: [1.0, 1.0], color: color.to_array() },
                            GpuVertexTex { pos: [x, y + h, z], uv: [0.0, 1.0], color: color.to_array() },
                        ];
                        let i = vec![0u32, 1, 2, 0, 2, 3];
                        steps.push(Step::DrawTextureUV { vertices: v, indices: i, src_tex_id, uv_tex_id, clipped: cur_ctx.clipped });
                    }
                    DrawOp::Text { x, y, text, opts } => {
                        // build buffer; prepare during execution to avoid lifetimes
                        let px = opts.px;
                        let metrics = CtMetrics::new(px, px * 1.3);
                        let mut buffer = CtBuffer::new(&mut this.font_system, metrics);
                        let mut x = x;
                        let mut y = y;
                        {
                            let mut b = buffer.borrow_with(&mut this.font_system);
                            b.set_size(None, None);
                            b.set_wrap(CtWrap::None);
                            let attrs = CtAttrs::new().weight(opts.weight);
                            // note: cosmic-text 0.14 `Attrs` doesn't expose a stable way to set
                            // family from a `String` without static lifetime requirements.
                            // we'll rely on the default font if a custom family is requested,
                            // leaving an easy extension point for future versions.
                            b.set_text(&text, &attrs, CtShaping::Advanced);
                            b.shape_until_scroll(true);
                            // measure to apply alignment
                            let mut width_px: f32 = 0.0;
                            for run in b.layout_runs() {
                                width_px = width_px.max(run.line_w);
                            }
                            // cosmic-text Metrics doesn't expose ascent directly; approximate using font size
                            let ascender_px: f32 = b.metrics().font_size;
                            // halign
                            x = match opts.halign {
                                TextHAlign::Left => x,
                                TextHAlign::Center => x - width_px * 0.5,
                                TextHAlign::Right => x - width_px,
                            };
                            // valign
                            y = match opts.valign {
                                TextVAlign::Top => y,
                                TextVAlign::Middle => y - ascender_px * 0.5,
                                TextVAlign::Baseline => y - ascender_px,
                            };
                        }
                        let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                        let ct_color = CtColor::rgba(
                            to_u8(opts.color.r),
                            to_u8(opts.color.g),
                            to_u8(opts.color.b),
                            to_u8(opts.color.a),
                        );
                        steps.push(Step::DrawText {
                            buffer,
                            x,
                            y,
                            color: ct_color,
                            clipped: cur_ctx.clipped,
                        });
                    }
                    DrawOp::Layer { options, ops } => {
                        // push blend change
                        let old_blend = cur_ctx.blend;
                        if options.blend != old_blend {
                            steps.push(Step::SetBlend(options.blend));
                            cur_ctx.blend = options.blend;
                        }
                        // push clip if any
                        let had_clip = cur_ctx.clipped;
                        if let Some(poly) = options.clip_polygon.clone() {
                            if poly.len() >= 3 {
                                let mut builder = path::Path::builder();
                                builder.begin(geom::point(poly[0][0], poly[0][1]));
                                for pnt in poly.iter().skip(1) {
                                    builder.line_to(geom::point(pnt[0], pnt[1]));
                                }
                                builder.end(true);
                                let p = builder.build();
                                // color doesn't matter for stencil write
                                let (v, i) =
                                    tessellate_to(&p, Color::rgba(0.0, 0.0, 0.0, 1.0), cur_ctx.z)?;
                                steps.push(Step::PushClip {
                                    vertices: v,
                                    indices: i,
                                });
                                cur_ctx.clipped = true;
                            }
                        }
                        // z context
                        let old_z = cur_ctx.z;
                        cur_ctx.z = map_z_index(options.z_index);

                        // recurse
                        flatten(this, &ops, steps, cur_ctx, tessellate_to)?;

                        // pop in reverse: restore z, clip, blend
                        cur_ctx.z = old_z;
                        if cur_ctx.clipped && !had_clip { // we pushed a clip for this layer
                            // draw same geometry with ref=0 by reusing previous PushClip geometry; we don't have it here
                            // since we don't retain it, we cannot pop here; instead, rely on PopClip step pushed above
                            // to make this work, we need to push a PopClip right away with same geometry
                        }
                        if options.clip_polygon.is_some() && !had_clip {
                            // we need to mirror the same polygon; rebuild quickly
                            if let Some(poly) = options.clip_polygon.clone() {
                                if poly.len() >= 3 {
                                    let mut builder = path::Path::builder();
                                    builder.begin(geom::point(poly[0][0], poly[0][1]));
                                    for pnt in poly.iter().skip(1) {
                                        builder.line_to(geom::point(pnt[0], pnt[1]));
                                    }
                                    builder.end(true);
                                    let p = builder.build();
                                    let (v, i) = tessellate_to(
                                        &p,
                                        Color::rgba(0.0, 0.0, 0.0, 1.0),
                                        cur_ctx.z,
                                    )?;
                                    steps.push(Step::PopClip {
                                        vertices: v,
                                        indices: i,
                                    });
                                }
                            }
                            cur_ctx.clipped = had_clip;
                        }
                        if options.blend != old_blend {
                            steps.push(Step::SetBlend(old_blend));
                            cur_ctx.blend = old_blend;
                        }
                    }
                }
            }
            Ok(())
        }

        let start_ctx = Ctx {
            blend: BlendMode::Alpha,
            z: map_z_index(0),
            clipped: false,
        };
        let ops_taken = std::mem::take(&mut self.ops);
        flatten(self, &ops_taken, &mut steps, start_ctx, &mut tessellate_to)
            .context("flatten draw ops")?;

        // execute steps in a single render pass with depth-stencil attachment
        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("drawctx"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: match self.clear_color {
                        Some(c) => wgpu::LoadOp::Clear(wgpu::Color {
                            r: c.r as f64,
                            g: c.g as f64,
                            b: c.b as f64,
                            a: c.a as f64,
                        }),
                        None => wgpu::LoadOp::Load,
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    // we always reset depth to far for each frame. clear regardless of color clear
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: Some(wgpu::Operations {
                    // reset stencil mask each frame
                    load: wgpu::LoadOp::Clear(0),
                    store: wgpu::StoreOp::Store,
                }),
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        rpass.set_bind_group(0, &self.screen_bind_group, &[]);

        for step in steps.into_iter() {
            match step {
                Step::SetBlend(_b) => { /* handled per draw */ }
                Step::PushClip { vertices, indices } => {
                    let (vbuf, ibuf, icount) = self.upload(vertices, indices);
                    rpass.set_pipeline(&self.pipeline_stencil_write);
                    rpass.set_stencil_reference(1);
                    // rebind our screen uniforms after glyphon may have changed bind groups
                    rpass.set_bind_group(0, &self.screen_bind_group, &[]);
                    rpass.set_vertex_buffer(0, vbuf.slice(..));
                    rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..icount, 0, 0..1);
                }
                Step::PopClip { vertices, indices } => {
                    let (vbuf, ibuf, icount) = self.upload(vertices, indices);
                    rpass.set_pipeline(&self.pipeline_stencil_write);
                    rpass.set_stencil_reference(0);
                    // rebind our screen uniforms after glyphon may have changed bind groups
                    rpass.set_bind_group(0, &self.screen_bind_group, &[]);
                    rpass.set_vertex_buffer(0, vbuf.slice(..));
                    rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..icount, 0, 0..1);
                }
                Step::DrawShape {
                    vertices,
                    indices,
                    blend,
                    clipped,
                } => {
                    let (vbuf, ibuf, icount) = self.upload(vertices, indices);
                    let pipe = match (blend, clipped) {
                        (BlendMode::Alpha, false) => &self.pipeline_alpha,
                        (BlendMode::Alpha, true) => &self.pipeline_alpha_stencil,
                        (BlendMode::Additive, false) => &self.pipeline_add,
                        (BlendMode::Additive, true) => &self.pipeline_add_stencil,
                    };
                    if clipped {
                        rpass.set_stencil_reference(1);
                    }
                    rpass.set_pipeline(pipe);
                    // rebind our screen uniforms after glyphon may have changed bind groups
                    rpass.set_bind_group(0, &self.screen_bind_group, &[]);
                    rpass.set_vertex_buffer(0, vbuf.slice(..));
                    rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..icount, 0, 0..1);
                }
                Step::DrawText {
                    buffer,
                    x,
                    y,
                    color,
                    clipped,
                } => {
                    let bounds = TextBounds {
                        left: 0,
                        top: 0,
                        right: self.width as i32,
                        bottom: self.height as i32,
                    };
                    let area = TextArea {
                        buffer: &buffer,
                        left: x,
                        top: y,
                        scale: 1.0,
                        bounds,
                        default_color: color,
                        custom_glyphs: &[],
                    };
                    let renderer = if clipped {
                        &mut self.text_renderer_stencil
                    } else {
                        &mut self.text_renderer
                    };

                    renderer
                        .prepare(
                            &self.device,
                            &self.queue,
                            &mut self.font_system,
                            &mut self.text_atlas,
                            &self.viewport,
                            vec![area],
                            &mut self.swash_cache,
                        )
                        .context("text prepare")?;

                    if clipped {
                        rpass.set_stencil_reference(1);
                    }

                    renderer
                        .render(&self.text_atlas, &self.viewport, &mut rpass)
                        .context("text render")?;
                }
                Step::DrawTexture { vertices, indices, tex_id, clipped } => {
                    if let Some(tex) = self.textures.get(tex_id) {
                        let (vbuf, ibuf, icount) = self.upload_tex(vertices, indices);
                        let pipe = if clipped { &self.pipeline_texture_stencil } else { &self.pipeline_texture };
                        if clipped {
                            rpass.set_stencil_reference(1);
                        }
                        rpass.set_pipeline(pipe);
                        rpass.set_bind_group(0, &self.screen_bind_group, &[]);
                        rpass.set_bind_group(1, &tex.bind_group, &[]);
                        rpass.set_vertex_buffer(0, vbuf.slice(..));
                        rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        rpass.draw_indexed(0..icount, 0, 0..1);
                    }
                }
                Step::DrawTextureUV { vertices, indices, src_tex_id, uv_tex_id, clipped } => {
                    if let (Some(src), Some(uv)) = (self.textures.get(src_tex_id), self.textures.get(uv_tex_id)) {
                        // build a combined bind group dynamically for src+uv using the uv layout
                        let (vbuf, ibuf, icount) = self.upload_tex(vertices, indices);
                        let pipe = if clipped { &self.pipeline_texture_uv_stencil } else { &self.pipeline_texture_uv };
                        if clipped { rpass.set_stencil_reference(1); }
                        // create a transient bind group coupling the sampler with src and uv views
                        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("im2d tex uv bg transient"),
                            layout: &self.uv_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Sampler(&src.sampler) },
                                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src.view) },
                                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&uv.view) },
                            ],
                        });
                        rpass.set_pipeline(pipe);
                        rpass.set_bind_group(0, &self.screen_bind_group, &[]);
                        rpass.set_bind_group(1, &bg, &[]);
                        rpass.set_vertex_buffer(0, vbuf.slice(..));
                        rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        rpass.draw_indexed(0..icount, 0, 0..1);
                    }
                }
            }
        }
        Ok(())
    }
}

// helpers
fn create_depth_stencil_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("im2d depth-stencil"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    depth.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_shape_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: TextureFormat,
    vertex_layout: &wgpu::VertexBufferLayout,
    blend_mode: BlendMode,
    use_stencil: bool,
    color_writes: bool,
) -> wgpu::RenderPipeline {
    let blend = match blend_mode {
        BlendMode::Alpha => Some(wgpu::BlendState::ALPHA_BLENDING),
        BlendMode::Additive => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        }),
    };
    let color_state = wgpu::ColorTargetState {
        format: color_format,
        blend,
        write_mask: if color_writes {
            wgpu::ColorWrites::ALL
        } else {
            wgpu::ColorWrites::empty()
        },
    };
    let depth_stencil = if use_stencil {
        Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState {
                front: wgpu::StencilFaceState {
                    compare: wgpu::CompareFunction::Equal,
                    fail_op: wgpu::StencilOperation::Keep,
                    depth_fail_op: wgpu::StencilOperation::Keep,
                    pass_op: wgpu::StencilOperation::Keep,
                },
                back: wgpu::StencilFaceState {
                    compare: wgpu::CompareFunction::Equal,
                    fail_op: wgpu::StencilOperation::Keep,
                    depth_fail_op: wgpu::StencilOperation::Keep,
                    pass_op: wgpu::StencilOperation::Keep,
                },
                read_mask: 0xFF,
                write_mask: 0xFF,
            },
            bias: wgpu::DepthBiasState::default(),
        })
    } else {
        Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        })
    };

    // special case: stencil write pipeline has color_writes=false and needs replace op; reuse use_stencil=true with Equal compare won't work for writes
    let depth_stencil = if !color_writes {
        Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState {
                front: wgpu::StencilFaceState {
                    compare: wgpu::CompareFunction::Always,
                    fail_op: wgpu::StencilOperation::Replace,
                    depth_fail_op: wgpu::StencilOperation::Replace,
                    pass_op: wgpu::StencilOperation::Replace,
                },
                back: wgpu::StencilFaceState {
                    compare: wgpu::CompareFunction::Always,
                    fail_op: wgpu::StencilOperation::Replace,
                    depth_fail_op: wgpu::StencilOperation::Replace,
                    pass_op: wgpu::StencilOperation::Replace,
                },
                read_mask: 0xFF,
                write_mask: 0xFF,
            },
            bias: wgpu::DepthBiasState::default(),
        })
    } else {
        depth_stencil
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("drawcontext pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[vertex_layout.clone()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(color_state)],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn create_texture_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: TextureFormat,
    vertex_layout: &wgpu::VertexBufferLayout,
    use_stencil: bool,
) -> wgpu::RenderPipeline {
    let color_state = wgpu::ColorTargetState {
        format: color_format,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    };
    let depth_stencil = if use_stencil {
        Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState {
                front: wgpu::StencilFaceState {
                    compare: wgpu::CompareFunction::Equal,
                    fail_op: wgpu::StencilOperation::Keep,
                    depth_fail_op: wgpu::StencilOperation::Keep,
                    pass_op: wgpu::StencilOperation::Keep,
                },
                back: wgpu::StencilFaceState {
                    compare: wgpu::CompareFunction::Equal,
                    fail_op: wgpu::StencilOperation::Keep,
                    depth_fail_op: wgpu::StencilOperation::Keep,
                    pass_op: wgpu::StencilOperation::Keep,
                },
                read_mask: 0xFF,
                write_mask: 0xFF,
            },
            bias: wgpu::DepthBiasState::default(),
        })
    } else {
        Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        })
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("drawcontext texture pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[vertex_layout.clone()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(color_state)],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
        depth_stencil,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

impl DrawContext {
    fn upload(
        &self,
        vertices: Vec<GpuVertex>,
        indices: Vec<u32>,
    ) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let vsize = (vertices.len() * std::mem::size_of::<GpuVertex>()) as u64;
        let isize = (indices.len() * std::mem::size_of::<u32>()) as u64;
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
            .write_buffer(&vbuf, 0, bytemuck::cast_slice(&vertices));
        self.queue
            .write_buffer(&ibuf, 0, bytemuck::cast_slice(&indices));
        (vbuf, ibuf, indices.len() as u32)
    }

    fn upload_tex(
        &self,
        vertices: Vec<GpuVertexTex>,
        indices: Vec<u32>,
    ) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let vsize = (vertices.len() * std::mem::size_of::<GpuVertexTex>()) as u64;
        let isize = (indices.len() * std::mem::size_of::<u32>()) as u64;
        let vbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("im2d vbuf tex"),
            size: vsize,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("im2d ibuf tex"),
            size: isize,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(&vertices));
        self.queue.write_buffer(&ibuf, 0, bytemuck::cast_slice(&indices));
        (vbuf, ibuf, indices.len() as u32)
    }

    /// create a layer and record sub-operations inside the closure
    pub fn with_layer<F: FnOnce(&mut DrawContext)>(&mut self, options: LayerOptions, f: F) {
        // temporarily swap the ops vec to capture into a child vec
        let mut child = Vec::new();
        std::mem::swap(&mut self.ops, &mut child);
        f(self);
        let ops = std::mem::take(&mut self.ops);
        self.ops = child;
        self.ops.push(DrawOp::Layer { options, ops });
    }

    /// create a dynamic rgba8 texture and return its id
    pub fn create_texture_rgba8(&mut self, width: u32, height: u32) -> usize {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("im2d texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("im2d sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("im2d tex bg"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        let res = TextureResource {
            texture,
            view,
            sampler,
            bind_group,
            width,
            height,
        };
        self.textures.push(res);
        self.textures.len() - 1
    }

    /// update entire rgba8 texture from cpu memory; data must be width*height*4 bytes
    pub fn update_texture_rgba8(&self, tex_id: usize, data: &[u8]) -> Result<()> {
        let tex = self
            .textures
            .get(tex_id)
            .ok_or_else(|| anyhow!("invalid texture id"))?;
        let expected = (tex.width as usize) * (tex.height as usize) * 4;
        if data.len() != expected {
            return Err(anyhow!("texture data size mismatch"));
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * tex.width),
                rows_per_image: Some(tex.height),
            },
            wgpu::Extent3d {
                width: tex.width,
                height: tex.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }
}

struct TextureResource {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    bind_group: BindGroup,
    width: u32,
    height: u32,
}
