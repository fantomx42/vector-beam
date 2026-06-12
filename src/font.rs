//! Stroke font: glyphs as polylines, laid out into beam segments.
//!
//! Glyphs are defined on a 4-wide x 6-tall integer grid (baseline at y = 0)
//! in the blocky, mostly axis-aligned style of vector arcade fonts. Each
//! glyph is a list of polylines ordered and oriented as the beam would draw
//! them — `layout` emits segments polyline-by-polyline, glyph-by-glyph, left
//! to right, so the output obeys the draw-order contract on
//! `Scene::segments` (stroke list order IS the beam path order).

use crate::geometry::Segment;

/// A glyph: polylines on the grid, each at least two points.
type Glyph = &'static [&'static [(f32, f32)]];

/// Glyph cell width in grid units (advance adds tracking on top).
const GRID_WIDTH: f32 = 4.0;
/// Cap height in grid units.
const GRID_HEIGHT: f32 = 6.0;
/// Pen advance per character (cell width + tracking).
const ADVANCE: f32 = 6.0;
/// Baseline-to-baseline distance for multi-line text.
const LINE_HEIGHT: f32 = 9.0;

/// Fit the text block within this box (world units). The frustum half-height
/// at z = 0 is ~1.73 (`CAMERA_Z * tan(FOV_Y / 2)` in main.rs), so this leaves
/// a comfortable margin even on a narrow window.
const MAX_BLOCK_WIDTH: f32 = 3.4;
const MAX_BLOCK_HEIGHT: f32 = 2.6;
/// Cap glyph height (world units) so short strings don't become giant.
const MAX_GLYPH_HEIGHT: f32 = 0.5;

/// The strokes for `c`, or `None` for unsupported characters (which still
/// advance the pen, like a space). Lowercase folds to uppercase.
fn glyph(c: char) -> Option<Glyph> {
    let g: Glyph = match c.to_ascii_uppercase() {
        'A' => &[&[(0., 0.), (0., 4.), (2., 6.), (4., 4.), (4., 0.)], &[(0., 2.), (4., 2.)]],
        'B' => &[
            &[(0., 0.), (0., 6.), (3., 6.), (4., 5.), (4., 4.), (3., 3.), (4., 2.), (4., 1.), (3., 0.), (0., 0.)],
            &[(0., 3.), (3., 3.)],
        ],
        'C' => &[&[(4., 6.), (0., 6.), (0., 0.), (4., 0.)]],
        'D' => &[&[(0., 0.), (0., 6.), (2., 6.), (4., 4.), (4., 2.), (2., 0.), (0., 0.)]],
        'E' => &[&[(4., 6.), (0., 6.), (0., 0.), (4., 0.)], &[(0., 3.), (3., 3.)]],
        'F' => &[&[(4., 6.), (0., 6.), (0., 0.)], &[(0., 3.), (3., 3.)]],
        'G' => &[&[(4., 6.), (0., 6.), (0., 0.), (4., 0.), (4., 3.), (2., 3.)]],
        'H' => &[&[(0., 6.), (0., 0.)], &[(0., 3.), (4., 3.)], &[(4., 0.), (4., 6.)]],
        'I' => &[&[(1., 6.), (3., 6.)], &[(2., 6.), (2., 0.)], &[(1., 0.), (3., 0.)]],
        'J' => &[&[(4., 6.), (4., 1.), (3., 0.), (1., 0.), (0., 1.)]],
        'K' => &[&[(0., 6.), (0., 0.)], &[(4., 6.), (0., 3.), (4., 0.)]],
        'L' => &[&[(0., 6.), (0., 0.), (4., 0.)]],
        'M' => &[&[(0., 0.), (0., 6.), (2., 3.), (4., 6.), (4., 0.)]],
        'N' => &[&[(0., 0.), (0., 6.), (4., 0.), (4., 6.)]],
        'O' => &[&[(0., 0.), (0., 6.), (4., 6.), (4., 0.), (0., 0.)]],
        'P' => &[&[(0., 0.), (0., 6.), (4., 6.), (4., 3.), (0., 3.)]],
        'Q' => &[&[(0., 0.), (0., 6.), (4., 6.), (4., 0.), (0., 0.)], &[(2., 2.), (4., 0.)]],
        'R' => &[&[(0., 0.), (0., 6.), (4., 6.), (4., 3.), (0., 3.)], &[(2., 3.), (4., 0.)]],
        'S' => &[&[(4., 6.), (0., 6.), (0., 3.), (4., 3.), (4., 0.), (0., 0.)]],
        'T' => &[&[(0., 6.), (4., 6.)], &[(2., 6.), (2., 0.)]],
        'U' => &[&[(0., 6.), (0., 0.), (4., 0.), (4., 6.)]],
        'V' => &[&[(0., 6.), (2., 0.), (4., 6.)]],
        'W' => &[&[(0., 6.), (0., 0.), (2., 3.), (4., 0.), (4., 6.)]],
        'X' => &[&[(0., 0.), (4., 6.)], &[(0., 6.), (4., 0.)]],
        'Y' => &[&[(0., 6.), (2., 3.), (4., 6.)], &[(2., 3.), (2., 0.)]],
        'Z' => &[&[(0., 6.), (4., 6.), (0., 0.), (4., 0.)]],
        '0' => &[&[(0., 0.), (0., 6.), (4., 6.), (4., 0.), (0., 0.)], &[(0., 1.), (4., 5.)]],
        '1' => &[&[(1., 5.), (2., 6.), (2., 0.)], &[(1., 0.), (3., 0.)]],
        '2' => &[&[(0., 6.), (4., 6.), (4., 3.), (0., 3.), (0., 0.), (4., 0.)]],
        '3' => &[&[(0., 6.), (4., 6.), (4., 0.), (0., 0.)], &[(1., 3.), (4., 3.)]],
        '4' => &[&[(0., 6.), (0., 3.), (4., 3.)], &[(4., 6.), (4., 0.)]],
        '5' => &[&[(4., 6.), (0., 6.), (0., 3.), (4., 3.), (4., 0.), (0., 0.)]],
        '6' => &[&[(4., 6.), (0., 6.), (0., 0.), (4., 0.), (4., 3.), (0., 3.)]],
        '7' => &[&[(0., 6.), (4., 6.), (2., 0.)]],
        '8' => &[&[(0., 0.), (0., 6.), (4., 6.), (4., 0.), (0., 0.)], &[(0., 3.), (4., 3.)]],
        '9' => &[&[(0., 0.), (4., 0.), (4., 6.), (0., 6.), (0., 3.), (4., 3.)]],
        '.' => &[&[(2., 0.), (2., 0.5)]],
        ',' => &[&[(2., 1.), (1., -1.)]],
        '-' => &[&[(1., 3.), (3., 3.)]],
        ':' => &[&[(2., 4.), (2., 5.)], &[(2., 1.), (2., 2.)]],
        '!' => &[&[(2., 6.), (2., 2.)], &[(2., 0.), (2., 0.5)]],
        '?' => &[&[(0., 5.), (1., 6.), (3., 6.), (4., 5.), (4., 4.), (2., 3.), (2., 2.)], &[(2., 0.), (2., 0.5)]],
        '\'' => &[&[(2., 6.), (2., 5.)]],
        _ => return None,
    };
    Some(g)
}

/// Width of an `n`-character line in grid units: every char advances the pen,
/// but the last one only occupies its cell.
fn line_width(n: usize) -> f32 {
    if n == 0 {
        0.0
    } else {
        (n - 1) as f32 * ADVANCE + GRID_WIDTH
    }
}

/// Lay `text` out as beam segments centered on the origin at z = 0.
///
/// `\n` splits lines; each line is centered horizontally and lines stack top
/// to bottom. The block is uniformly scaled to fit `MAX_BLOCK_WIDTH` x
/// `MAX_BLOCK_HEIGHT` world units, capped so glyphs never exceed
/// `MAX_GLYPH_HEIGHT` tall. Output order is the beam path: lines top to
/// bottom, glyphs left to right, polylines in glyph definition order.
pub fn layout(text: &str, color: [f32; 3]) -> Vec<Segment> {
    let lines: Vec<&str> = text.split('\n').collect();
    let widest = lines
        .iter()
        .map(|l| line_width(l.chars().count()))
        .fold(0.0, f32::max);
    if widest <= 0.0 {
        return Vec::new();
    }
    let block_height = (lines.len() - 1) as f32 * LINE_HEIGHT + GRID_HEIGHT;

    let scale = (MAX_BLOCK_WIDTH / widest)
        .min(MAX_BLOCK_HEIGHT / block_height)
        .min(MAX_GLYPH_HEIGHT / GRID_HEIGHT);

    let mut segments = Vec::new();
    for (row, line) in lines.iter().enumerate() {
        // Baseline of this row: the block spans [-h/2, h/2], the first row's
        // cap sits at the top.
        let base_y = block_height * 0.5 - GRID_HEIGHT - row as f32 * LINE_HEIGHT;
        let mut pen_x = -line_width(line.chars().count()) * 0.5;
        for c in line.chars() {
            if let Some(polylines) = glyph(c) {
                for polyline in polylines {
                    for w in polyline.windows(2) {
                        let p = |(x, y): (f32, f32)| {
                            [(pen_x + x) * scale, (base_y + y) * scale, 0.0]
                        };
                        segments.push(Segment::new(p(w[0]), p(w[1]), color));
                    }
                }
            }
            pen_x += ADVANCE;
        }
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHITE: [f32; 3] = [1.0, 1.0, 1.0];

    fn bounds(segs: &[Segment]) -> ([f32; 2], [f32; 2]) {
        let mut min = [f32::INFINITY; 2];
        let mut max = [f32::NEG_INFINITY; 2];
        for s in segs {
            for p in [s.p0, s.p1] {
                for a in 0..2 {
                    min[a] = min[a].min(p[a]);
                    max[a] = max[a].max(p[a]);
                }
            }
        }
        (min, max)
    }

    #[test]
    fn glyph_polylines_become_connected_segments() {
        // 'A' is a 5-point outline (4 segments) plus a 2-point bar (1).
        let segs = layout("A", WHITE);
        assert_eq!(segs.len(), 5);
        // Within the outline polyline, each segment starts where the
        // previous one ended (beam path continuity).
        for w in segs[..4].windows(2) {
            assert_eq!(w[0].p1, w[1].p0);
        }
    }

    #[test]
    fn every_glyph_is_continuous_within_polylines() {
        for c in ('A'..='Z').chain('0'..='9').chain(".,-:!?'".chars()) {
            let polylines = glyph(c).unwrap_or_else(|| panic!("missing glyph {c:?}"));
            assert!(!polylines.is_empty(), "empty glyph {c:?}");
            for polyline in polylines {
                assert!(polyline.len() >= 2, "degenerate polyline in {c:?}");
            }
        }
    }

    #[test]
    fn layout_is_centered() {
        // H's ink fills its whole cell, so the ink box equals the metric box
        // and centering is exact.
        let (min, max) = bounds(&layout("HH", WHITE));
        assert!((min[0] + max[0]).abs() < 1e-5, "x-centered: {min:?} {max:?}");
        assert!((min[1] + max[1]).abs() < 1e-5, "y-centered: {min:?} {max:?}");
    }

    #[test]
    fn short_text_respects_glyph_height_cap() {
        let (min, max) = bounds(&layout("A", WHITE));
        assert!(max[1] - min[1] <= MAX_GLYPH_HEIGHT + 1e-5);
    }

    #[test]
    fn long_text_fits_block_width() {
        let (min, max) = bounds(&layout("0123456789 ABCDEFGHIJ", WHITE));
        assert!(max[0] - min[0] <= MAX_BLOCK_WIDTH + 1e-5);
    }

    #[test]
    fn multiline_stacks_and_fits() {
        let one = bounds(&layout("AB", WHITE));
        let two = bounds(&layout("AB\nCD", WHITE));
        assert!(two.1[1] - two.0[1] > one.1[1] - one.0[1]);
        assert!(two.1[1] - two.0[1] <= MAX_BLOCK_HEIGHT + 1e-5);
    }

    #[test]
    fn lowercase_folds_and_unknown_chars_advance() {
        assert_eq!(layout("abc", WHITE).len(), layout("ABC", WHITE).len());
        // Unsupported glyphs draw nothing but still take a cell.
        assert!(layout("~", WHITE).is_empty());
        let (min, max) = bounds(&layout("A~B", WHITE));
        let (min2, max2) = bounds(&layout("AB", WHITE));
        assert!(max[0] - min[0] > max2[0] - min2[0]);
    }
}
