# vector-beam

A tiny **vector-display / "electron beam" line renderer** in [wgpu](https://wgpu.rs/).
3D line segments are expanded into screen-space ribbons and drawn with a Gaussian
beam cross-section into an HDR target, so they glow like phosphor strokes on an
old oscilloscope or vector arcade monitor.

![A glowing wireframe cube rendered as vector-display beams, its edges smearing into fading phosphor trails](docs/screenshot.png)

## Run it

```sh
cargo run --release
```

A window opens showing a slowly tumbling wireframe cube drawn as glowing beams,
leaving fading phosphor trails as it turns.
Needs a GPU with a Vulkan / Metal / DX12 / GL backend (anything wgpu supports).

The trail length is tunable with `--persistence <seconds>` (the phosphor decay
time constant, default `0.1`); `--persistence 0` disables trails entirely.

### Regenerate the screenshot

The same renderer can capture a single frame to a PNG headlessly (no window),
which is how `docs/screenshot.png` is produced:

```sh
cargo run --release -- --screenshot docs/screenshot.png 1280x960
```

## How it works

The interesting part is the shader, [`shaders/beam.wgsl`](shaders/beam.wgsl).

- **Instanced wide-line expansion.** Each line segment is one *instance*; the
  vertex shader synthesizes a 6-vertex quad from `@builtin(vertex_index)` (no
  vertex buffer) and offsets the two endpoints perpendicular to the segment in
  *screen space*. The offset is pre-multiplied by `clip.w`
  (`clip.xy + offset_ndc * clip.w`) so it survives the GPU's perspective divide
  and the line keeps a **constant pixel width** at any depth.
- **Gaussian beam profile.** The fragment shader shapes the cross-section as a
  tight bright core (`exp(-d²·7)`) plus a wide soft glow, and brightens the
  endpoints slightly so stroke junctions read as brighter dots — the way a real
  beam dwells where it changes direction.
- **Beam-speed model.** A segment that covers more screen distance is treated as
  being "drawn faster," so it comes out **dimmer and thinner**; short segments
  dwell and come out **bright and thick**. As the cube spins, edges sweeping
  quickly across the screen visibly dim — that is this model at work.
- **HDR + additive blending.** The beam pass renders into an `Rgba16Float`
  target with additive blending so overlapping strokes *accumulate* light, then
  a fullscreen [`shaders/tonemap.wgsl`](shaders/tonemap.wgsl) pass applies
  exposure + Reinhard tonemapping and resolves to the sRGB swapchain.
- **Phosphor persistence.** The HDR target is never cleared; instead, each frame
  starts by fading it with a fullscreen [`shaders/decay.wgsl`](shaders/decay.wgsl)
  draw before the new beams are added on top, so strokes linger and fade like
  excited phosphor. The fade needs no extra textures or shader reads — the
  pipeline blends with `(src: Zero, dst: Constant)`, computing
  `dst * blend_constant` in fixed-function hardware, and the host loads
  `exp(-dt / persistence)` into the blend constant each frame so the decay is
  framerate-independent.

The host code in [`src/main.rs`](src/main.rs) is a minimal winit + wgpu setup;
the swappable scene generators live in [`src/geometry.rs`](src/geometry.rs)
(a wireframe cube by default, plus a Lissajous "oscilloscope" curve).

## Implementation notes

- **Near-plane clipping is handled.** A segment with an endpoint at or behind the
  camera plane (`w <= 0`) is clipped against `w = ε` *in clip space, before* the
  perspective divide, by interpolating the crossing point; segments fully behind
  the camera are culled. Without this, a segment crossing the near plane would
  explode into garbage geometry. (Resolved [#1](../../issues/1).)
- **Energy-normalized dwell.** The beam-speed model makes slow beams thicker, but
  intensity is divided by the width factor so a thicker beam *spreads* its energy
  across the wider line rather than also multiplying peak brightness — otherwise
  short/slow segments over-blow on the HDR target (`intensity ∝ dwell / width`).
- **Screenshots replay history.** A persistence trail is, by definition, light
  from *previous* frames, so a one-shot headless render would show none. The
  screenshot path instead simulates the preceding ~5 time constants of frames at
  60 Hz into the persistent HDR target and captures the last one.

## License

[CC BY-NC 4.0](LICENSE) (Creative Commons Attribution-NonCommercial 4.0
International) — share and adapt with attribution, non-commercial use only.
