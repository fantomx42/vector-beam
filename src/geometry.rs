//! Scene geometry: line segments fed to the beam shader as per-instance data.
//!
//! `Segment` is the per-instance vertex-buffer record. Its memory layout is the
//! host side of the shader contract — `p0`, `p1`, `color` map to `@location(0)`,
//! `@location(1)`, `@location(2)` with a tight 36-byte stride (no padding, since
//! every field is `f32`-aligned and vec3 is packed in *vertex* buffers).

use bytemuck::{Pod, Zeroable};

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
/// scene. Not used by default; kept as a drop-in alternative for `main` to show
/// the beam-speed model over a dense continuous stroke.
#[allow(dead_code)]
pub fn lissajous(n: usize) -> Vec<Segment> {
    use std::f32::consts::TAU;
    let color = [0.4, 1.0, 0.7];
    let point = |t: f32| -> [f32; 3] {
        [
            (3.0 * t).sin() * 0.8,
            (2.0 * t + 0.5).sin() * 0.8,
            (5.0 * t).cos() * 0.4,
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
