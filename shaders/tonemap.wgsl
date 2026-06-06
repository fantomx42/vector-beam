// Tonemap pass: resolve the HDR (Rgba16Float) beam target down to the sRGB
// swapchain. A fullscreen triangle samples the HDR texture, applies an exposure
// scale and Reinhard tonemapping. The surface is an *_srgb format, so the GPU
// encodes the linear result to sRGB for us on write.

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Oversized triangle covering the whole clip rectangle in one primitive.
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    let xy = corners[vid];

    var out: VsOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    // Map clip space (y up) to texture UV (y down, origin top-left).
    out.uv = vec2<f32>(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_tex, hdr_samp, in.uv).rgb;

    let exposure = 1.1;
    let c = hdr * exposure;
    let mapped = c / (c + vec3<f32>(1.0)); // Reinhard

    return vec4<f32>(mapped, 1.0);
}
