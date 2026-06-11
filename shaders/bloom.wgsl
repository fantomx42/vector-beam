// Bloom: make the hottest parts of the HDR beam image bleed a wide, soft halo,
// the way a bright trace blooms on real CRT glass. Three half-resolution
// passes share this file: a bright-pass downsample (fs_downsample), then a
// separable Gaussian blur run twice (fs_blur, horizontal then vertical via
// u.direction). The blurred result is added back in the tonemap pass.

struct BloomUniforms {
    texel: vec2<f32>,     // 1 / source resolution for this pass
    direction: vec2<f32>, // blur axis: (1,0) or (0,1); unused by downsample
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
@group(0) @binding(2) var<uniform> u: BloomUniforms;

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
    out.uv = vec2<f32>(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
    return out;
}

// Only HDR values above THRESHOLD bloom; the quadratic knee fades the cutoff
// in over [THRESHOLD - KNEE, THRESHOLD + KNEE] so there is no hard edge where
// a stroke's brightness crosses the threshold.
const THRESHOLD: f32 = 1.0;
const KNEE: f32 = 0.6;

fn bright(c: vec3<f32>) -> vec3<f32> {
    let l = max(c.r, max(c.g, c.b));
    let soft = clamp(l - THRESHOLD + KNEE, 0.0, 2.0 * KNEE);
    let contrib = max(soft * soft / (4.0 * KNEE), l - THRESHOLD);
    return c * max(contrib, 0.0) / max(l, 1e-4);
}

// 2x downsample with a 4-tap box (each bilinear tap straddles a 2x2 source
// quad), then the bright-pass. Filtering before thresholding keeps single hot
// pixels from flickering as they alias in and out of the half-res grid.
@fragment
fn fs_downsample(in: VsOut) -> @location(0) vec4<f32> {
    let o = u.texel;
    var c = textureSample(src_tex, src_samp, in.uv + vec2<f32>(-o.x, -o.y)).rgb;
    c += textureSample(src_tex, src_samp, in.uv + vec2<f32>(o.x, -o.y)).rgb;
    c += textureSample(src_tex, src_samp, in.uv + vec2<f32>(-o.x, o.y)).rgb;
    c += textureSample(src_tex, src_samp, in.uv + vec2<f32>(o.x, o.y)).rgb;
    return vec4<f32>(bright(c * 0.25), 1.0);
}

// 9-tap Gaussian (sigma ~3 half-res texels) along u.direction. Run once per
// axis; separability makes the full 9x9 kernel cost 18 taps instead of 81.
@fragment
fn fs_blur(in: VsOut) -> @location(0) vec4<f32> {
    var w = array<f32, 5>(0.1532, 0.1450, 0.1226, 0.0929, 0.0630);
    let step = u.direction * u.texel;

    var c = textureSample(src_tex, src_samp, in.uv).rgb * w[0];
    for (var i = 1; i < 5; i++) {
        let o = step * f32(i);
        c += (textureSample(src_tex, src_samp, in.uv + o).rgb
            + textureSample(src_tex, src_samp, in.uv - o).rgb) * w[i];
    }
    return vec4<f32>(c, 1.0);
}
