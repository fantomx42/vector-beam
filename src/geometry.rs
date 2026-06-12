//! Scene geometry: line segments fed to the beam shader as per-instance data.
//!
//! `Segment` is the per-instance vertex-buffer record. Its memory layout is the
//! host side of the shader contract — `p0`, `p1`, `color` map to `@location(0)`,
//! `@location(1)`, `@location(2)` with a tight 36-byte stride (no padding, since
//! every field is `f32`-aligned and vec3 is packed in *vertex* buffers).

use bytemuck::{Pod, Zeroable};

/// Which demo scene to render, parsed from `--scene` on the command line.
///
/// A scene owns both its line segments and its model matrix. The cube is rigid
/// — its segments never change and all motion lives in the model matrix. The
/// Lissajous curve *morphs* (its phase drifts over time), so its segments are
/// regenerated every frame and the host must upload them with
/// `queue.write_buffer` instead of a one-time buffer init.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Scene {
    #[default]
    Cube,
    Lissajous,
    /// Turn/thrust Asteroids-style ship, flown with the keyboard — the
    /// canonical input-latency feel test. The glyph is static; all motion
    /// lives in a host-supplied model matrix driven by the flight keys.
    Ship,
    /// A saucer glyph scrolling steadily across the screen (UFO-test-style
    /// motion-clarity pattern): track it with your eyes and toggle scan mode
    /// with S — the difference in edge clarity is the verification.
    Ufo,
    /// Interactive storage scope: segments come from mouse input (built per
    /// frame in `main`, so `segments` returns nothing) and stroke history
    /// lives in the phosphor buffer, not in any list. Draw-once semantics are
    /// the opposite of the scan scheduler's redraw-every-cycle contract, so
    /// this scene always runs with scan mode off.
    Draw,
}

impl Scene {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "cube" => Some(Scene::Cube),
            "lissajous" => Some(Scene::Lissajous),
            "ship" => Some(Scene::Ship),
            "ufo" => Some(Scene::Ufo),
            "draw" => Some(Scene::Draw),
            _ => None,
        }
    }

    /// The scene's line segments at `time`. Both scenes emit a fixed segment
    /// count, so the instance buffer never needs to grow.
    ///
    /// Draw-order contract: the order of this list IS the beam path order.
    /// The scan scheduler (`scan.rs`) slices the list by cumulative arc
    /// length into per-subframe windows, so segments must be laid out as the
    /// beam would traverse them — consecutive where the stroke is continuous,
    /// with discontinuities ("pen lifts") only where unavoidable.
    pub fn segments(self, time: f32) -> Vec<Segment> {
        match self {
            Scene::Cube => wireframe_cube(0.7),
            Scene::Lissajous => lissajous(600, time),
            Scene::Ship => ship(),
            Scene::Ufo => ufo(),
            Scene::Draw => Vec::new(),
        }
    }

    /// Instance-buffer capacity in segments. Fixed per scene so the buffer
    /// never needs to grow; the draw scene's headroom covers a crosshair plus
    /// one frame's worth of cursor movement (history lives in the phosphor).
    pub fn max_segments(self) -> usize {
        match self {
            Scene::Cube => 12,
            Scene::Lissajous => 600,
            Scene::Ship => 5,
            Scene::Ufo => 18,
            Scene::Draw => 256,
        }
    }

    /// Whether `segments(time)` changes between frames (=> the instance buffer
    /// must be rewritten per frame).
    pub fn animated(self) -> bool {
        matches!(self, Scene::Lissajous)
    }

    /// Model matrix at `time`: tumble the cube around two axes; turn the
    /// Lissajous slowly around Y so its depth reads without overpowering the
    /// curve's own morphing. The ship has no autonomous motion — the host
    /// overrides its model matrix from the flight controls.
    pub fn model(self, time: f32) -> glam::Mat4 {
        match self {
            Scene::Cube => {
                glam::Mat4::from_rotation_y(time * 0.7) * glam::Mat4::from_rotation_x(time * 0.4)
            }
            Scene::Lissajous => glam::Mat4::from_rotation_y(time * 0.25),
            Scene::Ship => glam::Mat4::IDENTITY,
            // The draw scene is a flat board at z = 0 facing the camera.
            Scene::Draw => glam::Mat4::IDENTITY,
            // Constant-velocity horizontal scroll, wrapping at +-2 world
            // units. Steady speed matters: the eye must be able to smoothly
            // pursue the glyph for the motion-clarity comparison to read.
            Scene::Ufo => {
                glam::Mat4::from_translation(glam::Vec3::new(
                    (time * 1.2).rem_euclid(4.0) - 2.0,
                    0.0,
                    0.0,
                ))
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Segment {
    pub p0: [f32; 3],
    pub p1: [f32; 3],
    pub color: [f32; 3],
}

impl Segment {
    pub fn new(a: [f32; 3], b: [f32; 3], color: [f32; 3]) -> Self {
        Self { p0: a, p1: b, color }
    }
}

/// The 12 edges of an axis-aligned cube, half-extent `s`.
///
/// This is the default demo scene: as the cube spins, each edge's *screen-space*
/// length changes, which is exactly what the shader's beam-speed model keys on —
/// fast-sweeping edges go dim and thin, slow ones bright and thick.
pub fn wireframe_cube(s: f32) -> Vec<Segment> {
    // 8 corners.
    let v = [
        [-s, -s, -s], // 0
        [s, -s, -s],  // 1
        [s, s, -s],   // 2
        [-s, s, -s],  // 3
        [-s, -s, s],  // 4
        [s, -s, s],   // 5
        [s, s, s],    // 6
        [-s, s, s],   // 7
    ];
    // Edge list as corner-index pairs, in beam path order (see the draw-order
    // contract on `Scene::segments`). No Eulerian path exists — every corner
    // has odd degree 3 — so the traversal is one continuous 9-edge stroke
    // followed by three pen-lift strokes for the remaining connectors.
    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), // back face loop
        (0, 4),                          // drop to the front face
        (4, 5), (5, 6), (6, 7), (7, 4), // front face loop
        (1, 5), (2, 6), (3, 7),          // remaining connectors (pen lifts)
    ];

    // Phosphor green, with a faint cyan tint on the depth-connecting edges so
    // the color channel is visibly exercised.
    let green = [0.35, 1.0, 0.55];
    let cyan = [0.30, 0.95, 0.95];

    edges
        .iter()
        .map(|&(a, b)| {
            // The four z-connectors pair corner i with corner i+4.
            let color = if b == a + 4 { cyan } else { green };
            Segment::new(v[a], v[b], color)
        })
        .collect()
}

/// The classic Asteroids dart at z = 0, nose up (+Y), in beam path order: one
/// continuous stroke nose -> left fin -> left notch -> right notch -> right
/// fin -> nose. About 0.24 world units tall against a view frustum ~3.5 units
/// high at z = 0 — small enough that flying it across the screen is a real
/// deflection test.
pub fn ship() -> Vec<Segment> {
    let green = [0.35, 1.0, 0.55];
    let nose = [0.0, 0.12, 0.0];
    let left_fin = [-0.08, -0.12, 0.0];
    let left_notch = [-0.04, -0.07, 0.0];
    let right_notch = [0.04, -0.07, 0.0];
    let right_fin = [0.08, -0.12, 0.0];
    let path = [nose, left_fin, left_notch, right_notch, right_fin, nose];
    path.windows(2)
        .map(|w| Segment::new(w[0], w[1], green))
        .collect()
}

/// A flying-saucer glyph at z = 0 in beam path order: a 12-segment hull
/// ellipse loop, then (one pen lift) a 6-segment dome arc on top.
pub fn ufo() -> Vec<Segment> {
    use std::f32::consts::{PI, TAU};
    let green = [0.35, 1.0, 0.55];
    let mut segs = Vec::with_capacity(18);

    // Hull: full ellipse, rx 0.35 / ry 0.10.
    let hull = |t: f32| [0.35 * t.cos(), 0.10 * t.sin(), 0.0];
    for i in 0..12 {
        let t0 = TAU * i as f32 / 12.0;
        let t1 = TAU * (i + 1) as f32 / 12.0;
        segs.push(Segment::new(hull(t0), hull(t1), green));
    }
    // Dome: upper half-ellipse, rx 0.16 / ry 0.12, sitting on the hull.
    let dome = |t: f32| [0.16 * t.cos(), 0.06 + 0.12 * t.sin(), 0.0];
    for i in 0..6 {
        let t0 = PI * i as f32 / 6.0;
        let t1 = PI * (i + 1) as f32 / 6.0;
        segs.push(Segment::new(dome(t0), dome(t1), green));
    }
    segs
}

/// A 3D Lissajous curve sampled into `n` connected segments — an "oscilloscope"
/// scene (`--scene lissajous`) showing the beam-speed model over a dense
/// continuous stroke. The X and Z phases drift with `time` so the figure
/// continuously morphs; under phosphor persistence the morph is what leaves
/// trails (and keeps any one pixel from accumulating to a hot static glow).
pub fn lissajous(n: usize, time: f32) -> Vec<Segment> {
    use std::f32::consts::TAU;
    let color = [0.4, 1.0, 0.7];
    let phase = 0.5 * time;
    let point = |t: f32| -> [f32; 3] {
        [
            (3.0 * t + phase).sin() * 0.8,
            (2.0 * t + 0.5).sin() * 0.8,
            (5.0 * t - 0.3 * phase).cos() * 0.4,
        ]
    };
    (0..n)
        .map(|i| {
            let t0 = TAU * i as f32 / n as f32;
            let t1 = TAU * (i + 1) as f32 / n as f32;
            Segment::new(point(t0), point(t1), color)
        })
        .collect()
}
