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
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum Scene {
    #[default]
    Cube,
    Lissajous,
}

impl Scene {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "cube" => Some(Scene::Cube),
            "lissajous" => Some(Scene::Lissajous),
            _ => None,
        }
    }

    /// The scene's line segments at `time`. Both scenes emit a fixed segment
    /// count, so the instance buffer never needs to grow.
    pub fn segments(self, time: f32) -> Vec<Segment> {
        match self {
            Scene::Cube => wireframe_cube(0.7),
            Scene::Lissajous => lissajous(600, time),
        }
    }

    /// Whether `segments(time)` changes between frames (=> the instance buffer
    /// must be rewritten per frame).
    pub fn animated(self) -> bool {
        matches!(self, Scene::Lissajous)
    }

    /// Model matrix at `time`: tumble the cube around two axes; turn the
    /// Lissajous slowly around Y so its depth reads without overpowering the
    /// curve's own morphing.
    pub fn model(self, time: f32) -> glam::Mat4 {
        match self {
            Scene::Cube => {
                glam::Mat4::from_rotation_y(time * 0.7) * glam::Mat4::from_rotation_x(time * 0.4)
            }
            Scene::Lissajous => glam::Mat4::from_rotation_y(time * 0.25),
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
    fn new(a: [f32; 3], b: [f32; 3], color: [f32; 3]) -> Self {
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
    // Edge list as corner-index pairs.
    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), // back face
        (4, 5), (5, 6), (6, 7), (7, 4), // front face
        (0, 4), (1, 5), (2, 6), (3, 7), // connecting edges
    ];

    // Phosphor green, with a faint cyan tint on the depth-connecting edges so
    // the color channel is visibly exercised.
    let green = [0.35, 1.0, 0.55];
    let cyan = [0.30, 0.95, 0.95];

    edges
        .iter()
        .enumerate()
        .map(|(i, &(a, b))| {
            let color = if i >= 8 { cyan } else { green };
            Segment::new(v[a], v[b], color)
        })
        .collect()
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
