//! vector-beam: a native wgpu host that drives `shaders/beam.wgsl`.
//!
//! Pipeline: decay pass (fade the persistent HDR target -> phosphor trails) ->
//! beam pass (instanced line segments, additive blend into the same Rgba16Float
//! HDR target) -> tonemap pass (Reinhard, resolve to the sRGB swapchain). The
//! only animated input is the MVP matrix, updated per frame.

mod bloom;
mod cli;
mod geometry;
mod screenshot;

use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Mirror of the WGSL `BeamUniforms` block. 80 bytes; the field order and the
/// natural `#[repr(C)]` layout already satisfy WGSL std-uniform alignment, so no
/// explicit padding is required.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct BeamUniforms {
    pub mvp: [f32; 16],
    pub resolution: [f32; 2],
    pub base_width: f32,
    pub brightness: f32,
}

/// Per-instance vertex attributes: three vec3s at offsets 0, 12, 24 (stride 36).
pub(crate) const SEGMENT_ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3];

pub(crate) const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Default phosphor persistence time constant (seconds). The HDR target fades
/// by `exp(-dt / persistence)` each frame, so strokes are ~5% bright after
/// 3 time constants. 0 disables persistence (every frame starts black).
pub(crate) const DEFAULT_PERSISTENCE: f32 = 0.1;

/// Build the render pipeline for `shaders/decay.wgsl`: a fullscreen triangle
/// whose only effect is the fixed-function blend `dst * blend_constant`, fading
/// the persistent HDR target. Shared by the live window and the screenshot path.
pub(crate) fn make_decay_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/decay.wgsl"));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("decay pl layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    // result = src * 0 + dst * constant — the fragment output never matters.
    let fade = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::Constant,
        operation: wgpu::BlendOperation::Add,
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("decay pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState { color: fade, alpha: fade }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Model-view-projection for a scene at a given time, viewed from +Z (the
/// scene supplies its own model motion). Shared by the live window and the
/// headless screenshot so both frame the scene identically.
pub(crate) fn beam_mvp(scene: geometry::Scene, aspect: f32, time: f32) -> glam::Mat4 {
    let proj = glam::Mat4::perspective_rh(60f32.to_radians(), aspect, 0.1, 100.0);
    let view = glam::Mat4::look_at_rh(
        glam::Vec3::new(0.0, 0.0, 3.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    proj * view * scene.model(time)
}

/// Everything the live window needs to construct a `GpuState`, gathered from
/// the CLI before the event loop starts.
#[derive(Default)]
struct RenderOptions {
    /// `None` = use the default persistence time constant.
    persistence: Option<f32>,
    scene: geometry::Scene,
    present_mode: Option<cli::PresentModeArg>,
}

/// Pick the swapchain present mode: honor an explicit `--present-mode` when
/// the surface supports it, otherwise prefer the lowest-latency mode
/// available (Immediate beats Mailbox beats Fifo).
fn choose_present_mode(
    supported: &[wgpu::PresentMode],
    requested: Option<cli::PresentModeArg>,
) -> wgpu::PresentMode {
    if let Some(req) = requested {
        let mode = req.to_wgpu();
        if supported.contains(&mode) {
            return mode;
        }
        eprintln!("requested present mode {mode:?} not supported (surface offers {supported:?}); auto-selecting");
    }
    [
        wgpu::PresentMode::Immediate,
        wgpu::PresentMode::Mailbox,
        wgpu::PresentMode::Fifo,
    ]
    .into_iter()
    .find(|m| supported.contains(m))
    .unwrap_or(wgpu::PresentMode::Fifo)
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    decay_pipeline: wgpu::RenderPipeline,
    beam_pipeline: wgpu::RenderPipeline,
    tonemap_pipeline: wgpu::RenderPipeline,

    // Phosphor persistence time constant (seconds); 0 disables. `last_frame`
    // feeds the framerate-independent decay factor exp(-dt / persistence).
    persistence: f32,
    last_frame: Instant,

    scene: geometry::Scene,

    uniform_buffer: wgpu::Buffer,
    beam_bind_group: wgpu::BindGroup,

    instance_buffer: wgpu::Buffer,
    instance_count: u32,

    // HDR target, bloom chain, and the tonemap bind group that samples both.
    // All rebuilt on resize because the bind groups hold views of the
    // (resized) textures.
    tonemap_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    hdr_view: wgpu::TextureView,
    bloom: bloom::Bloom,
    tonemap_bind_group: wgpu::BindGroup,
}

impl GpuState {
    async fn new(window: Arc<Window>, opts: &RenderOptions) -> Self {
        let persistence = opts.persistence.unwrap_or(DEFAULT_PERSISTENCE);
        let scene = opts.scene;
        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("request device");

        // Prefer an sRGB swapchain so the tonemap pass can write linear values
        // and let the GPU handle the sRGB encode.
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        // Latency substrate: at most one frame queued ahead of the display,
        // and the lowest-latency present mode the surface offers (overridable
        // with --present-mode). Immediate tears but never waits; Mailbox
        // replaces the queued frame; Fifo is the vsync fallback.
        let present_mode = choose_present_mode(&caps.present_modes, opts.present_mode);
        eprintln!("present mode: {present_mode:?}");
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            desired_maximum_frame_latency: 1,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // --- Geometry ---
        // COPY_DST because animated scenes (Lissajous) rewrite the segments
        // every frame; the segment count is fixed, so the buffer never grows.
        let segments = scene.segments(0.0);
        let instance_count = segments.len() as u32;
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("segments"),
            contents: bytemuck::cast_slice(&segments),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // --- Uniforms ---
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("beam uniforms"),
            size: std::mem::size_of::<BeamUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let beam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("beam bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let beam_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("beam bg"),
            layout: &beam_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // --- Beam pipeline (-> HDR target, additive blend) ---
        let beam_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/beam.wgsl"));
        let beam_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("beam pl layout"),
            bind_group_layouts: &[Some(&beam_layout)],
            immediate_size: 0,
        });
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let beam_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("beam pipeline"),
            layout: Some(&beam_pl_layout),
            vertex: wgpu::VertexState {
                module: &beam_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<geometry::Segment>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &SEGMENT_ATTRS,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &beam_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(additive),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // --- Decay pipeline (phosphor persistence: fades the HDR target) ---
        let decay_pipeline = make_decay_pipeline(&device);

        // --- Tonemap pipeline (HDR + bloom textures -> swapchain) ---
        let tonemap_layout = make_tonemap_layout(&device);
        let tonemap_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/tonemap.wgsl"));
        let tonemap_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tonemap pl layout"),
            bind_group_layouts: &[Some(&tonemap_layout)],
            immediate_size: 0,
        });
        let tonemap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap pipeline"),
            layout: Some(&tonemap_pl_layout),
            vertex: wgpu::VertexState {
                module: &tonemap_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &tonemap_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hdr sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let hdr_view = make_hdr_target(&device, width, height);
        let bloom = bloom::Bloom::new(&device, &sampler, &hdr_view, width, height);
        let tonemap_bind_group = make_tonemap_bind_group(
            &device,
            &tonemap_layout,
            &sampler,
            &hdr_view,
            bloom.output_view(),
        );

        Self {
            surface,
            device,
            queue,
            config,
            decay_pipeline,
            beam_pipeline,
            tonemap_pipeline,
            persistence,
            last_frame: Instant::now(),
            scene,
            uniform_buffer,
            beam_bind_group,
            instance_buffer,
            instance_count,
            tonemap_layout,
            sampler,
            hdr_view,
            bloom,
            tonemap_bind_group,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.hdr_view = make_hdr_target(&self.device, width, height);
        self.bloom
            .resize(&self.device, &self.sampler, &self.hdr_view, width, height);
        self.tonemap_bind_group = make_tonemap_bind_group(
            &self.device,
            &self.tonemap_layout,
            &self.sampler,
            &self.hdr_view,
            self.bloom.output_view(),
        );
    }

    fn render(&mut self, start: Instant) {
        // Acquire the swapchain frame FIRST: under Fifo this is where the loop
        // blocks for vsync, so everything sampled after it (timing, animation
        // state, uniforms) is as fresh as possible when the frame is submitted.
        // Nothing latency-sensitive may run before this point.
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            // Swapchain went stale (resize/minimize) — reconfigure and skip this frame.
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            // Timeout / Occluded / Validation — skip this frame and try again next tick.
            _ => return,
        };
        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Sample time only now, after the block above, so the MVP reflects the
        // moment of submission rather than the moment the redraw was requested.
        let time = start.elapsed().as_secs_f32();

        // Framerate-independent phosphor fade for this frame. With persistence
        // disabled the factor is 0, which multiplies the old frame away entirely
        // (equivalent to the pre-persistence clear).
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        let decay = if self.persistence > 0.0 {
            (-dt / self.persistence).exp() as f64
        } else {
            0.0
        };

        // Animate: the scene's model matrix always moves; a morphing scene
        // (Lissajous) additionally regenerates its segments each frame.
        if self.scene.animated() {
            let segments = self.scene.segments(time);
            self.queue
                .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&segments));
        }
        let aspect = self.config.width as f32 / self.config.height as f32;
        let mvp = beam_mvp(self.scene, aspect, time);

        let uniforms = BeamUniforms {
            mvp: mvp.to_cols_array(),
            resolution: [self.config.width as f32, self.config.height as f32],
            base_width: 6.0,
            brightness: 1.0,
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        // Pass 1: fade last frame's light (phosphor persistence), then add this
        // frame's beams on top — both into the persistent HDR target, which is
        // loaded rather than cleared so trails survive across frames.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("decay+beam pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.decay_pipeline);
            pass.set_blend_constant(wgpu::Color {
                r: decay,
                g: decay,
                b: decay,
                a: decay,
            });
            pass.draw(0..3, 0..1);

            pass.set_pipeline(&self.beam_pipeline);
            pass.set_bind_group(0, &self.beam_bind_group, &[]);
            pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
            pass.draw(0..6, 0..self.instance_count);
        }

        // Pass 2: bloom chain (bright-pass downsample + separable blur).
        self.bloom.encode(&mut encoder);

        // Pass 3: tonemap HDR + bloom -> swapchain.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tonemap pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.tonemap_pipeline);
            pass.set_bind_group(0, &self.tonemap_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

/// Build the HDR render target at `width`x`height`. Called on init and on
/// every resize.
pub(crate) fn make_hdr_target(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdr target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Bind group layout for `shaders/tonemap.wgsl`: HDR texture, sampler, bloom
/// texture. Shared by the live window and the headless screenshot.
pub(crate) fn make_tonemap_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tonemap bgl"),
        entries: &[
            texture_entry(0),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            texture_entry(2),
        ],
    })
}

/// Tonemap bind group over the (resize-dependent) HDR and bloom views.
pub(crate) fn make_tonemap_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    hdr_view: &wgpu::TextureView,
    bloom_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tonemap bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(hdr_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(bloom_view),
            },
        ],
    })
}

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    start: Option<Instant>,
    opts: RenderOptions,
    fullscreen: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        // Borderless fullscreen is a latency feature, not cosmetics: a
        // fullscreen surface can be scanned out directly, while a windowed one
        // always goes through a compositor copy.
        let mut attrs = Window::default_attributes()
            .with_title("vector-beam")
            .with_inner_size(winit::dpi::LogicalSize::new(960.0, 720.0));
        if self.fullscreen {
            attrs = attrs.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        }
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let gpu = pollster::block_on(GpuState::new(window.clone(), &self.opts));
        self.window = Some(window);
        self.gpu = Some(gpu);
        self.start = Some(Instant::now());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => gpu.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                let start = self.start.unwrap_or_else(Instant::now);
                gpu.render(start);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    // Headless mode: `vector-beam --screenshot [path] [WxH]` renders one frame to
    // a PNG and exits, with no window. Everything else opens the live window.
    // See cli.rs for the full flag set.
    let args: Vec<String> = std::env::args().collect();
    let cli = match cli::parse(&args) {
        Ok(cli) => cli,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };
    if let Some(shot) = &cli.screenshot {
        // The capture always renders the full scene per simulated frame with
        // the legacy persistence default; a few seconds in, the cube sits at a
        // pleasant three-quarter angle.
        let persistence = cli.persistence.unwrap_or(DEFAULT_PERSISTENCE);
        screenshot::capture(&shot.path, shot.width, shot.height, 2.6, persistence, cli.scene);
        println!("wrote {} ({}x{})", shot.path, shot.width, shot.height);
        return;
    }

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        opts: RenderOptions {
            persistence: cli.persistence,
            scene: cli.scene,
            present_mode: cli.present_mode,
        },
        fullscreen: cli.fullscreen,
        ..App::default()
    };
    event_loop.run_app(&mut app).expect("run app");
}
