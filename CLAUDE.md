# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A vector-display / "electron beam" line renderer in Rust + wgpu: 3D line segments
are expanded into screen-space ribbons with a Gaussian beam cross-section and
drawn additively into an HDR target, imitating phosphor strokes on an
oscilloscope or vector arcade monitor. Single binary crate, no test suite —
verification is visual (run it) plus `cargo check`/`clippy` and the built-in
telemetry line.

## Commands

```sh
cargo run --release                                   # tumbling wireframe cube, scan mode on
cargo run --release -- --scene ship                   # Asteroids-style ship (arrows/WASD) — the way to feel latency
cargo run --release -- --scene ufo                    # motion-clarity test pattern; toggle scan with S
cargo run --release -- --scene lissajous              # morphing 3D Lissajous curve
cargo run --release -- --scene draw                   # interactive storage-scope drawing
cargo run --release -- --scene text --text "HELLO"    # stroke-font text (src/font.rs)
cargo run --release -- --screenshot docs/screenshot.png 1280x960   # headless PNG capture (rejects draw)
```

Key flags (parsed in `src/cli.rs`): `--persistence <s>` (phosphor decay; default
3 ms in scan mode, 100 ms with `--no-scan`, 1 s in the draw scene; 0 disables),
`--scan-hz` / `--hw-hz`
(logical scan rate vs. panel rate), `--beams <B>` and `--beam-gain <x>`
(multi-beam simulation), `--present-mode immediate|mailbox|fifo` (defaults to
lowest-latency supported), `--fullscreen` (needed for direct scanout), `--no-scan`
(legacy draw-everything-every-frame mode; `S` toggles live), `--text <message>`
(the text scene's message; only valid with `--scene text`).

Needs a GPU with any wgpu backend (Vulkan / Metal / DX12 / GL). Debug builds
work but the beam pass is heavy; prefer `--release`.

## Branch state

Everything lives on `main`. The draw scene always forces scan mode off
(draw-once storage-scope strokes can't survive scan slicing — see the
`Scene::Draw` doc comment).

## Architecture

Two loops, deliberately decoupled (this is the core v0.3.0 idea):

- **Hardware loop** presents at the panel rate (e.g. 240 Hz) with at most one
  frame of queued latency.
- **Logical scan** refreshes the scene at `--scan-hz` (default 60). Each
  hardware frame draws only the *slice* of the stroke list the beam would have
  covered in that subframe window. The scheduler in `src/scan.rs` slices the
  stroke list by cumulative arc length — stroke list order *is* the beam path
  order. Phosphor decay fills the time between slices; with `--beams B` the
  list is split into B arcs and each subframe draws one bucket from every arc,
  with brightness compensated by `--beam-gain` (each stroke is lit only 1/N of
  the cycle).

Render pipeline, in frame order (passes set up in `src/main.rs::GpuState`):

1. **Decay** (`shaders/decay.wgsl`) — the HDR target is *never cleared*; a
   fullscreen draw fades it via fixed-function blending `(src: Zero, dst:
   Constant)` with the host loading `exp(-dt / persistence)` into the blend
   constant. Stroke history lives in the HDR texture, not in any CPU-side list.
2. **Beam** (`shaders/beam.wgsl`) — the heart of the project. Each segment is
   one *instance*; the vertex shader synthesizes a 6-vertex quad from
   `@builtin(vertex_index)` (no vertex buffer) and offsets endpoints in screen
   space, pre-multiplied by `clip.w` so lines keep constant pixel width after
   the perspective divide. Fragment shader: Gaussian core + soft glow.
   Additive blending into an `Rgba16Float` HDR target.
3. **Bloom** (`shaders/bloom.wgsl`, host side `src/bloom.rs`) — bright-pass
   downsample with a soft threshold knee, then separable Gaussian blur at
   quarter resolution.
4. **Tonemap** (`shaders/tonemap.wgsl`) — exposure + Reinhard, adds bloom back,
   resolves to the sRGB swapchain.

Scenes live in `src/geometry.rs` behind the `Scene` enum; a scene owns its
segments, model matrix, and a fixed instance-buffer capacity (`max_segments`).
The `ship` scene is input-driven (closed-loop steering exists specifically to
make latency perceptible); `ufo` is a steady scroller for eye-tracked
motion-clarity comparison; `lissajous` regenerates segments every frame;
`cube` is rigid. `text` renders a `--text` message through the stroke font in
`src/font.rs` (glyphs are polylines on a 4x6 grid, laid out in beam path
order; the `Text(String)` variant owning its message is why `Scene` is not
`Copy`). `draw` is the storage scope: `segments()` returns nothing,
`main.rs` builds segments per frame from cursor input (queued on `InputState`,
drained at render time), only *new* movement is drawn, and the decayed HDR
buffer is the stroke memory — it always runs in no-scan mode with a 1 s
persistence default.

`src/telemetry.rs` prints a 5-second line of input-to-submit percentiles and
GPU frame time (timestamp queries where supported) — use it to verify latency
claims instead of guessing.

`src/screenshot.rs` is a headless twin of the windowed path: since a
persistence trail is light from previous frames, it simulates several decay
time constants of frames into the HDR target before capturing.

## Physical-model invariants

These are deliberate and interlocking — don't "fix" one in isolation:

- **Beam-speed model**: longer screen-space segments are dimmer/thinner (drawn
  "faster"); intensity is divided by the width factor so energy spreads rather
  than peak brightness multiplying (`intensity ∝ dwell / width`).
- **Wall-clock energy normalization**: with persistence on, per-frame
  brightness scales with `dt` (clamped) so emission is light per unit time,
  not per frame, regardless of frame rate.
- **Near-plane clipping** happens in clip space against `w = ε` *before* the
  perspective divide; segments fully behind the camera are culled. Removing
  this makes near-plane-crossing segments explode into garbage geometry.
- **Scan cadence assumes fixed-rate scanout** — VRR/adaptive sync re-times
  scanout and jitters the subframe windows; the README tells users to disable
  it rather than the code compensating.

## Conventions

- README.md's "How it works" section is the design document — keep it in sync
  when changing shaders or the pass structure.
- License is CC BY-NC 4.0 (non-commercial), set deliberately; don't change it.
