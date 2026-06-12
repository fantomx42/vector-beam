//! Verify, don't vibe: input-to-submit latency histogram (CPU clocks) and a
//! GPU frame timer (wgpu timestamp queries). Together they cover the whole
//! pipeline except scanout, which is bounded analytically at one refresh.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Instant;

/// Fixed-bucket histogram of input-to-submit latency. 0.25 ms buckets cover
/// 0–32 ms; anything slower lands in the last bucket (and `max` keeps the
/// true worst case).
pub struct LatencyHistogram {
    buckets: [u32; 128],
    count: u64,
    max_secs: f32,
    last_gpu_ms: Option<f32>,
    last_report: Instant,
}

impl LatencyHistogram {
    const BUCKET_SECS: f32 = 0.000_25;
    const REPORT_EVERY_SECS: f32 = 5.0;

    pub fn new() -> Self {
        Self {
            buckets: [0; 128],
            count: 0,
            max_secs: 0.0,
            last_gpu_ms: None,
            last_report: Instant::now(),
        }
    }

    pub fn record(&mut self, secs: f32) {
        let idx = ((secs / Self::BUCKET_SECS) as usize).min(self.buckets.len() - 1);
        self.buckets[idx] += 1;
        self.count += 1;
        self.max_secs = self.max_secs.max(secs);
    }

    /// Upper edge of the bucket where the cumulative count reaches `p`
    /// (0..=1) of the total, in seconds. 0 when empty.
    pub fn percentile(&self, p: f32) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let threshold = (p.clamp(0.0, 1.0) as f64 * self.count as f64).ceil() as u64;
        let mut cum = 0u64;
        for (idx, &n) in self.buckets.iter().enumerate() {
            cum += n as u64;
            if cum >= threshold {
                return (idx + 1) as f32 * Self::BUCKET_SECS;
            }
        }
        self.max_secs
    }

    /// Print percentiles every few seconds and reset. `gpu_ms` is the latest
    /// GPU frame time, if the timer produced one this frame.
    pub fn maybe_report(&mut self, gpu_ms: Option<f32>) {
        if gpu_ms.is_some() {
            self.last_gpu_ms = gpu_ms;
        }
        if self.last_report.elapsed().as_secs_f32() < Self::REPORT_EVERY_SECS {
            return;
        }
        self.last_report = Instant::now();
        if self.count == 0 && self.last_gpu_ms.is_none() {
            return;
        }
        let ms = |s: f32| s * 1e3;
        let input = if self.count > 0 {
            format!(
                "input->submit p50 {:.2} p90 {:.2} p99 {:.2} max {:.2} ms (n={}, scanout adds <=1 refresh)",
                ms(self.percentile(0.5)),
                ms(self.percentile(0.9)),
                ms(self.percentile(0.99)),
                ms(self.max_secs),
                self.count,
            )
        } else {
            "input->submit: no input events".to_string()
        };
        let gpu = match self.last_gpu_ms {
            Some(g) => format!("gpu {g:.2} ms"),
            None => "gpu: timestamps unsupported".to_string(),
        };
        eprintln!("latency: {input} | {gpu}");
        self.buckets = [0; 128];
        self.count = 0;
        self.max_secs = 0.0;
    }
}

const RING: usize = 3;
const BYTES: u64 = 16; // two u64 timestamps

/// GPU frame timer: timestamp at the start of the decay+beam pass and the end
/// of the tonemap pass, resolved into a small ring of staging buffers and
/// read back without ever stalling — when all slots are in flight the frame
/// simply goes unmeasured.
pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    staging: [wgpu::Buffer; RING],
    in_flight: [bool; RING],
    tx: Sender<(usize, bool)>,
    rx: Receiver<(usize, bool)>,
    /// Nanoseconds per timestamp tick.
    period: f32,
}

impl GpuTimer {
    /// `None` when the device lacks TIMESTAMP_QUERY; callers degrade to
    /// CPU-only logging.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp resolve"),
            size: BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("timestamp staging {i}")),
                size: BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let (tx, rx) = channel();
        Some(Self {
            query_set,
            resolve,
            staging,
            in_flight: [false; RING],
            tx,
            rx,
            period: queue.get_timestamp_period(),
        })
    }

    /// Timestamp write for the first pass of the frame (start of GPU work).
    pub fn begin_writes(&self) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: None,
        }
    }

    /// Timestamp write for the last pass of the frame (end of GPU work).
    pub fn end_writes(&self) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: Some(1),
        }
    }

    /// Encode the resolve + copy into a free staging slot. Returns the slot to
    /// pass to [`Self::map_slot`] after submit, or `None` when the ring is
    /// saturated (skip measuring this frame rather than wait).
    pub fn resolve_and_read(&mut self, encoder: &mut wgpu::CommandEncoder) -> Option<usize> {
        let slot = self.in_flight.iter().position(|f| !f)?;
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.staging[slot], 0, BYTES);
        Some(slot)
    }

    /// Kick off the async map of `slot`; must be called after the encoder
    /// from [`Self::resolve_and_read`] was submitted.
    pub fn map_slot(&mut self, slot: usize) {
        self.in_flight[slot] = true;
        let tx = self.tx.clone();
        self.staging[slot]
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send((slot, r.is_ok()));
            });
    }

    /// Non-blocking poll: drives map callbacks and returns the most recent
    /// completed GPU frame time in milliseconds, if any landed this call.
    pub fn poll_ms(&mut self, device: &wgpu::Device) -> Option<f32> {
        let _ = device.poll(wgpu::PollType::Poll);
        let mut latest = None;
        while let Ok((slot, ok)) = self.rx.try_recv() {
            if ok {
                let data = self.staging[slot].slice(..).get_mapped_range();
                let ticks: [u64; 2] = [
                    u64::from_le_bytes(data[0..8].try_into().unwrap()),
                    u64::from_le_bytes(data[8..16].try_into().unwrap()),
                ];
                drop(data);
                self.staging[slot].unmap();
                latest = Some(ticks[1].wrapping_sub(ticks[0]) as f32 * self.period * 1e-6);
            }
            self.in_flight[slot] = false;
        }
        latest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_on_known_distribution() {
        let mut h = LatencyHistogram::new();
        // 50 samples at ~1 ms, 40 at ~2 ms, 10 at ~10 ms.
        for _ in 0..50 {
            h.record(0.001);
        }
        for _ in 0..40 {
            h.record(0.002);
        }
        for _ in 0..10 {
            h.record(0.010);
        }
        // Bucket width is 0.25 ms; the percentile reports the bucket's upper
        // edge, so allow that much slack.
        let close = |a: f32, b: f32| (a - b).abs() <= 0.000_26;
        assert!(close(h.percentile(0.5), 0.001), "p50={}", h.percentile(0.5));
        assert!(close(h.percentile(0.9), 0.002), "p90={}", h.percentile(0.9));
        assert!(close(h.percentile(0.99), 0.010), "p99={}", h.percentile(0.99));
        assert_eq!(h.max_secs, 0.010);
    }

    #[test]
    fn percentile_empty_is_zero() {
        assert_eq!(LatencyHistogram::new().percentile(0.5), 0.0);
    }

    #[test]
    fn oversized_samples_clamp_to_last_bucket() {
        let mut h = LatencyHistogram::new();
        h.record(1.0); // 1 s >> 32 ms range
        assert!(h.percentile(1.0) <= 0.032 + f32::EPSILON);
        assert_eq!(h.max_secs, 1.0);
    }
}
