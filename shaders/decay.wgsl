// Decay pass: fade the persistent HDR target toward black, simulating phosphor
// persistence. The fade is done entirely by fixed-function blending — the
// pipeline uses (src_factor: Zero, dst_factor: Constant), so the result is
// `dst * blend_constant` and this fragment's output color is irrelevant. The
// host loads exp(-dt / persistence) into the blend constant each frame, which
// makes the fade framerate-independent.

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    // Oversized triangle covering the whole clip rectangle in one primitive.
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return vec4<f32>(corners[vid], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
