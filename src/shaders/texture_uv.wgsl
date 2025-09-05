struct Screen {
  size: vec2<f32>,
};

@group(0) @binding(0) var<uniform> screen: Screen;
@group(1) @binding(0) var tex_sampler: sampler;
@group(1) @binding(1) var src_tex: texture_2d<f32>;
@group(1) @binding(2) var uv_tex: texture_2d<f32>;

struct VsIn {
  @location(0) pos: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) color: vec4<f32>,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
  var out: VsOut;
  let sx = 2.0 * in.pos.x / screen.size.x - 1.0;
  let sy = 1.0 - 2.0 * in.pos.y / screen.size.y;
  out.pos = vec4<f32>(sx, sy, in.pos.z, 1.0);
  out.uv = in.uv;
  out.color = in.color;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  // sample uv coordinates from uv texture using the incoming uv
  let uv_rg = textureSample(uv_tex, tex_sampler, in.uv).rg;
  // sample the source texture at remapped coordinates
  let base = textureSample(src_tex, tex_sampler, uv_rg);
  return base * in.color;
}


