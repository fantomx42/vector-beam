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

    var c0 = u.mvp * vec4<f32>(p0, 1.0);
    var c1 = u.mvp * vec4<f32>(p1, 1.0);

    // Near-plane clipping. The screen-space expansion below divides by w, which
    // is meaningless for a point at or behind the camera plane (w <= 0). Clip the
    // segment against w = near_w *before* the perspective divide so both
    // endpoints sit safely in front of the camera; cull the whole primitive if
    // neither does. Without this, a segment crossing the near plane explodes into
    // garbage geometry.
    let near_w = 1e-4;
    if (c0.w <= near_w && c1.w <= near_w) {
        var culled: VsOut;
        culled.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0); // outside the clip volume
        culled.across = 0.0;
        culled.along = 0.0;
        culled.color = vec3<f32>(0.0);
        culled.intensity = 0.0;
        return culled;
    }
    if (c0.w <= near_w) {
        c0 = mix(c0, c1, (near_w - c0.w) / (c1.w - c0.w));
    } else if (c1.w <= near_w) {
        c1 = mix(c1, c0, (near_w - c1.w) / (c0.w - c1.w));
    }

    // Endpoints in pixel space (for a constant screen-space line width and the
    // beam-speed estimate).
    let half_res = u.resolution * 0.5;
    let screen0 = (c0.xy / c0.w) * half_res;
    let screen1 = (c1.xy / c1.w) * half_res;

    let seg = screen1 - screen0;
    let seg_len = max(length(seg), 1e-4);
    let dir = seg / seg_len;
    let normal = vec2<f32>(-dir.y, dir.x);

    // Beam-speed model: a longer screen segment is "drawn faster" in a fixed
    // unit of beam time, so it is dimmer and thinner. Intensity is divided by the
    // width factor so a slower (thicker) beam spreads its energy across the wider
    // line instead of also multiplying peak brightness -- without this, short or
    // slow segments over-blow on the HDR target.
    let dwell = clamp(160.0 / (seg_len + 60.0), 0.4, 2.5);
    let width_scale = mix(0.7, 1.7, clamp(dwell - 0.3, 0.0, 1.0));
    let width = u.base_width * width_scale;
    let intensity = u.brightness * dwell / width_scale;

    var clip = c0;
    if (end > 0.5) {
        clip = c1;
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
