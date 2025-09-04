// screen uniform holds logical pixel size of the target surface
struct Screen { size: vec2<f32> };
@group(0) @binding(0) var<uniform> screen: Screen;

// vertex inputs/outputs
struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    // convert from pixel space to normalized device coordinates
    let ndc_x = (in.pos.x / screen.size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (in.pos.y / screen.size.y) * 2.0;
    out.pos = vec4<f32>(ndc_x, ndc_y, in.pos.z, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}


