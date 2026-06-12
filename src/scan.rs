//! Scan scheduler: slices the stroke list into per-subframe windows.
//!
//! The hardware loop presents at the panel rate (e.g. 240 Hz) while the scene
//! refreshes at a logical scan rate (60 Hz default). Each hardware frame draws
//! only the slice of the stroke list the beam would have covered in that
//! subframe window; phosphor decay fills the gaps between slices. Segments are
//! contiguous in the instance buffer, so a subframe is just an instance range
//! `draw(0..6, start..end)` — zero buffer churn for static scenes.
//!
//! Beam time is proportional to deflection distance, so buckets are split by
//! cumulative segment length. Model-space length is used for now; screen-space
//! (projected on the host) is the authentic metric and a possible upgrade.
//!
//! The subframe index is driven by elapsed time, not a frame counter: under
//! Immediate/Mailbox the frame rate is uncapped, and a counter would scan as
//! fast as the GPU can present rather than at the configured scan rate.

use std::ops::Range;

use crate::geometry::Segment;

#[derive(Clone, Copy, Debug)]
pub struct ScanConfig {
    /// Logical scene refresh rate (full beam passes per second).
    pub scan_hz: f32,
    /// Number of simultaneous beams; the stroke list is split into this many
    /// arcs and each subframe draws one bucket from every arc.
    pub beams: u32,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self { scan_hz: 60.0, beams: 1 }
    }
}

/// Subframes per full scan: how many hardware refreshes one beam pass spans.
pub fn subframes_per_scan(hw_hz: f32, scan_hz: f32) -> u32 {
    (hw_hz / scan_hz.max(f32::EPSILON)).round().max(1.0) as u32
}

fn segment_len(s: &Segment) -> f32 {
    let d = [s.p1[0] - s.p0[0], s.p1[1] - s.p0[1], s.p1[2] - s.p0[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Split `segments[range]` into `n` buckets of roughly equal cumulative
/// model-space length. Returns `n + 1` monotone boundaries with
/// `bounds[0] == range.start` and `bounds[n] == range.end`; bucket `k` is
/// `bounds[k]..bounds[k + 1]`. Buckets may be empty when `n` exceeds the
/// segment count. Falls back to an index-uniform split when the total length
/// is zero (degenerate geometry).
pub fn partition_by_arc_length(segments: &[Segment], range: Range<u32>, n: u32) -> Vec<u32> {
    assert!(range.end as usize <= segments.len());
    let slice = &segments[range.start as usize..range.end as usize];
    let count = slice.len() as u32;
    let total: f32 = slice.iter().map(segment_len).sum();

    let mut bounds = Vec::with_capacity(n as usize + 1);
    bounds.push(range.start);
    if total <= 0.0 {
        // Index-uniform fallback: every segment "costs" the same.
        for k in 1..n {
            bounds.push(range.start + (count as u64 * k as u64 / n as u64) as u32);
        }
    } else {
        // A segment occupies the beam-time interval [cum, cum + len); it goes
        // in the bucket where that interval's midpoint falls. Boundary k is
        // the first segment whose midpoint reaches k/n of the total. Walking
        // the targets and the prefix sum together keeps the boundaries
        // monotone even with zero-length segments in the mix.
        let mut cum = 0.0;
        let mut idx = 0u32;
        for k in 1..n {
            let target = total * k as f32 / n as f32;
            while idx < count && cum + segment_len(&slice[idx as usize]) * 0.5 < target {
                cum += segment_len(&slice[idx as usize]);
                idx += 1;
            }
            bounds.push(range.start + idx);
        }
    }
    bounds.push(range.end);
    bounds
}

/// Owns the bucket boundaries and the scan clock. Pure math — no GPU types —
/// so the slicing is testable off-GPU.
pub struct ScanScheduler {
    /// Subframes per full scan.
    pub n: u32,
    pub scan_hz: f32,
    pub beams: u32,
    /// `beams + 1` boundaries splitting the stroke list into arcs.
    arc_bounds: Vec<u32>,
    /// Per arc: `n + 1` bucket boundaries.
    buckets: Vec<Vec<u32>>,
    /// Last subframe window drawn (absolute index since t=0).
    last_window: Option<u64>,
}

impl ScanScheduler {
    pub fn new(segments: &[Segment], cfg: ScanConfig, hw_hz: f32) -> Self {
        let mut s = Self {
            n: subframes_per_scan(hw_hz, cfg.scan_hz),
            scan_hz: cfg.scan_hz,
            beams: cfg.beams.max(1),
            arc_bounds: Vec::new(),
            buckets: Vec::new(),
            last_window: None,
        };
        s.rebuild(segments);
        s
    }

    /// Recompute bucket boundaries; called once for static scenes and per
    /// frame for morphing ones (whose instance buffer is rewritten anyway).
    pub fn rebuild(&mut self, segments: &[Segment]) {
        let len = segments.len() as u32;
        self.arc_bounds = partition_by_arc_length(segments, 0..len, self.beams);
        self.buckets = (0..self.beams as usize)
            .map(|b| {
                let arc = self.arc_bounds[b]..self.arc_bounds[b + 1];
                partition_by_arc_length(segments, arc, self.n)
            })
            .collect();
    }

    /// Forget scan progress; called when scan mode is toggled at runtime so
    /// re-enabling doesn't "catch up" over the windows spent in full-redraw.
    pub fn reset(&mut self) {
        self.last_window = None;
    }

    /// Absolute subframe window index at `elapsed` seconds.
    fn window(&self, elapsed: f32) -> u64 {
        (elapsed.max(0.0) as f64 * self.scan_hz as f64 * self.n as f64) as u64
    }

    /// Append the instance ranges to draw this hardware frame. Empty when the
    /// frame lands in an already-drawn window (decay still runs); when frames
    /// span multiple windows the beam catches up on the missed buckets, capped
    /// at one full scan. Advances the scan clock.
    pub fn ranges(&mut self, elapsed: f32, out: &mut Vec<Range<u32>>) {
        let cur = self.window(elapsed);
        let first = match self.last_window {
            None => cur,
            Some(last) if cur <= last => return,
            // Cap the catch-up at one full scan: missing more than n windows
            // means whole scans were dropped, and redrawing them would just
            // draw everything brighter than intended.
            Some(last) => (last + 1).max(cur.saturating_sub(self.n as u64 - 1)),
        };
        for w in first..=cur {
            let k = (w % self.n as u64) as usize;
            for bucket in &self.buckets {
                let (start, end) = (bucket[k], bucket[k + 1]);
                if start < end {
                    out.push(start..end);
                }
            }
        }
        self.last_window = Some(cur);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `count` collinear segments of the given lengths laid end to end on X.
    fn chain(lengths: &[f32]) -> Vec<Segment> {
        let mut x = 0.0;
        lengths
            .iter()
            .map(|&l| {
                let s = Segment {
                    p0: [x, 0.0, 0.0],
                    p1: [x + l, 0.0, 0.0],
                    color: [1.0, 1.0, 1.0],
                };
                x += l;
                s
            })
            .collect()
    }

    fn assert_valid_bounds(bounds: &[u32], range: Range<u32>, n: u32) {
        assert_eq!(bounds.len(), n as usize + 1);
        assert_eq!(bounds[0], range.start);
        assert_eq!(*bounds.last().unwrap(), range.end);
        assert!(bounds.windows(2).all(|w| w[0] <= w[1]), "monotone: {bounds:?}");
    }

    #[test]
    fn subframe_counts() {
        assert_eq!(subframes_per_scan(240.0, 60.0), 4);
        assert_eq!(subframes_per_scan(144.0, 60.0), 2);
        assert_eq!(subframes_per_scan(60.0, 60.0), 1);
        assert_eq!(subframes_per_scan(30.0, 60.0), 1); // never below 1
        assert_eq!(subframes_per_scan(240.0, 120.0), 2);
    }

    #[test]
    fn equal_lengths_split_evenly() {
        let segs = chain(&[1.0; 12]);
        let bounds = partition_by_arc_length(&segs, 0..12, 4);
        assert_eq!(bounds, vec![0, 3, 6, 9, 12]);
    }

    #[test]
    fn skewed_length_isolates_long_segment() {
        // One segment 100x longer than the other eleven: it dominates the
        // total, so it must not share a bucket with many short ones.
        let mut lengths = vec![1.0; 12];
        lengths[5] = 100.0;
        let segs = chain(&lengths);
        let bounds = partition_by_arc_length(&segs, 0..12, 4);
        assert_valid_bounds(&bounds, 0..12, 4);
        // The long segment (index 5) sits alone in its bucket.
        let bucket_of_5 = (0..4).find(|&k| bounds[k] <= 5 && 5 < bounds[k + 1]).unwrap();
        assert_eq!(bounds[bucket_of_5 + 1] - bounds[bucket_of_5], 1);
    }

    #[test]
    fn n_one_is_full_range() {
        let segs = chain(&[1.0; 5]);
        assert_eq!(partition_by_arc_length(&segs, 0..5, 1), vec![0, 5]);
    }

    #[test]
    fn n_exceeding_count_allows_empty_buckets() {
        let segs = chain(&[1.0; 3]);
        let bounds = partition_by_arc_length(&segs, 0..3, 8);
        assert_valid_bounds(&bounds, 0..3, 8);
    }

    #[test]
    fn zero_length_segments_fall_back_to_uniform() {
        let segs = chain(&[0.0; 8]);
        let bounds = partition_by_arc_length(&segs, 0..8, 4);
        assert_eq!(bounds, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn sub_range_partition_respects_offsets() {
        let segs = chain(&[1.0; 12]);
        let bounds = partition_by_arc_length(&segs, 4..12, 4);
        assert_eq!(bounds, vec![4, 6, 8, 10, 12]);
    }

    fn collect(s: &mut ScanScheduler, t: f32) -> Vec<Range<u32>> {
        let mut out = Vec::new();
        s.ranges(t, &mut out);
        out
    }

    #[test]
    fn exact_cadence_visits_each_bucket_once_per_scan() {
        let segs = chain(&[1.0; 12]);
        let (hw_hz, scan_hz) = (240.0, 60.0);
        let mut s = ScanScheduler::new(&segs, ScanConfig { scan_hz, beams: 1 }, hw_hz);
        assert_eq!(s.n, 4);
        // Drive at exactly hw cadence over two full scans; sample mid-window
        // to dodge float edges.
        let mut drawn = Vec::new();
        for i in 0..8 {
            let t = (i as f32 + 0.5) / hw_hz;
            drawn.extend(collect(&mut s, t));
        }
        assert_eq!(
            drawn,
            vec![0..3, 3..6, 6..9, 9..12, 0..3, 3..6, 6..9, 9..12]
        );
    }

    #[test]
    fn same_window_repeat_draws_nothing() {
        let segs = chain(&[1.0; 12]);
        let mut s = ScanScheduler::new(&segs, ScanConfig::default(), 240.0);
        assert_eq!(collect(&mut s, 0.002), vec![0..3]);
        assert!(collect(&mut s, 0.003).is_empty()); // still window 0
    }

    #[test]
    fn skipped_windows_catch_up() {
        let segs = chain(&[1.0; 12]);
        let mut s = ScanScheduler::new(&segs, ScanConfig::default(), 240.0);
        assert_eq!(collect(&mut s, 0.002), vec![0..3]); // window 0
        // Jump straight to window 2: windows 1 and 2 are both drawn.
        assert_eq!(collect(&mut s, 0.0105), vec![3..6, 6..9]);
    }

    #[test]
    fn huge_gap_caps_at_one_scan() {
        let segs = chain(&[1.0; 12]);
        let mut s = ScanScheduler::new(&segs, ScanConfig::default(), 240.0);
        assert_eq!(collect(&mut s, 0.002), vec![0..3]);
        // A one-second stall skips ~240 windows; only the last n are drawn.
        let out = collect(&mut s, 1.002);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn multi_beam_ranges_cover_list_disjointly() {
        let segs = chain(&[1.0; 24]);
        let mut s = ScanScheduler::new(&segs, ScanConfig { scan_hz: 60.0, beams: 3 }, 240.0);
        assert_eq!(s.n, 4);
        let mut drawn = Vec::new();
        for i in 0..4 {
            let t = (i as f32 + 0.5) / 240.0;
            let ranges = collect(&mut s, t);
            assert_eq!(ranges.len(), 3, "3 beams -> 3 ranges per subframe");
            drawn.extend(ranges);
        }
        // One full scan: 12 ranges, disjoint, covering 0..24 exactly.
        let mut indices: Vec<u32> = drawn.iter().flat_map(|r| r.clone()).collect();
        indices.sort_unstable();
        assert_eq!(indices, (0..24).collect::<Vec<_>>());
    }
}
