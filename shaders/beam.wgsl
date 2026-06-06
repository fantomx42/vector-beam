// Beam pass: expand 3D line segments (instanced) into screen-space-width quads
// and draw them with a Gaussian "electron beam" profile into an HDR target.
//
// The vertex stage performs the perspective projection, the wide-line expansion,
// and the beam-speed model (a slow beam dwells -> bright & thick; a fast beam
// -> dim & thin). The fragment stage shapes the glowing cross-section.

struct BeamUniforms {
    mvp: mat4x4<f32>,
    resolution: vec2<f32>,
    base_width: f32,
    brightness: f32,
};

@group(0) @binding(0) var<uniform> u: BeamUniforms;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) across: f32,    // signed distance across the line, -1..1
    @location(1) along: f32,     // position along the segment, 0..1
    @location(2) color: vec3<f32>,
    @location(3) intensity: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @location(0) p0: vec3<f32>,
    @location(1) p1: vec3<f32>,
    @location(2) color: vec3<f32>,
) -> VsOut {
    // Two triangles forming the quad. (end, side): end picks p0/p1, side picks
    // the perpendicular offset direction.
    var ends = array<f32, 6>(0.0, 1.0, 1.0, 0.0, 1.0, 0.0);
    var sides = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let end = ends[vid];
    let side = sides[vid];

    let clip0 = u.mvp * vec4<f32>(p0, 1.0);
    let clip1 = u.mvp * vec4<f32>(p1, 1.0);

    // Endpoints in pixel space (for a constant screen-space line width and the
    // beam-speed estimate).
    let half_res = u.resolution * 0.5;
    let screen0 = (clip0.xy / clip0.w) * half_res;
    let screen1 = (clip1.xy / clip1.w) * half_res;

    let seg = screen1 - screen0;
    let seg_len = max(length(seg), 1e-4);
    let dir = seg / seg_len;
    let normal = vec2<f32>(-dir.y, dir.x);

    // Beam-speed model: a longer screen segment is "drawn faster" in a fixed
    // unit of beam time, so it is dimmer and thinner.
    let dwell = clamp(160.0 / (seg_len + 60.0), 0.4, 2.5);
    let width = u.base_width * mix(0.7, 1.7, clamp(dwell - 0.3, 0.0, 1.0));
    let intensity = u.brightness * dwell;

    var clip = clip0;
    if (end > 0.5) {
        clip = clip1;
    }

    // Offset perpendicular by a constant pixel width (scaled by clip.w so the
    // width stays constant in screen space regardless of perspective depth).
    let offset_px = normal * side * (width * 0.5);
    let offset_ndc = offset_px / half_res;
    clip = vec4<f32>(clip.xy + offset_ndc * clip.w, clip.z, clip.w);

    var out: VsOut;
    out.clip = clip;
    out.across = side;
    out.along = end;
    out.color = color;
    out.intensity = intensity;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Gaussian beam cross-section: a tight bright core plus a wide soft glow.
    let d = in.across;
    let core = exp(-d * d * 7.0);
    let glow = exp(-abs(d) * 2.2) * 0.35;

    // Vertices dwell slightly longer where strokes meet -> brighter dots.
    let endpoint = max(1.0 - in.along, in.along);
    let vertex_glow = 1.0 + 0.6 * smoothstep(0.7, 1.0, endpoint);

    let profile = (core + glow) * vertex_glow;
    let rgb = in.color * in.intensity * profile;
    return vec4<f32>(rgb, profile);
}
