//! vector-beam: a native wgpu host that drives `shaders/beam.wgsl`.
//!
//! Pipeline: decay pass (fade the persistent HDR target -> phosphor trails) ->
//! beam pass (instanced line segments, additive blend into the same Rgba16Float
//! HDR target) -> tonemap pass (Reinhard, resolve to the sRGB swapchain). The
//! only animated input is the MVP matrix, updated per frame.

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

/// Model-view-projection for the demo scene at a given time: a cube viewed from
/// +Z, tumbling around two axes. Shared by the live window and the headless
/// screenshot so both frame the scene identically.
pub(crate) fn beam_mvp(aspect: f32, time: f32) -> glam::Mat4 {
    let proj = glam::Mat4::perspective_rh(60f32.to_radians(), aspect, 0.1, 100.0);
    let view = glam::Mat4::look_at_rh(
        glam::Vec3::new(0.0, 0.0, 3.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    let model =
        glam::Mat4::from_rotation_y(time * 0.7) * glam::Mat4::from_rotation_x(time * 0.4);
    proj * view * model
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

    uniform_buffer: wgpu::Buffer,
    beam_bind_group: wgpu::BindGroup,

    instance_buffer: wgpu::Buffer,
    instance_count: u32,

    // HDR target + the tonemap bind group that samples it. Both are rebuilt on
    // resize because the bind group holds a view of the (resized) HDR texture.
    tonemap_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    hdr_view: wgpu::TextureView,
    tonemap_bind_group: wgpu::BindGroup,
}

impl GpuState {
    async fn new(window: Arc<Window>, persistence: f32) -> Self {
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

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // --- Geometry ---
        let segments = geometry::wireframe_cube(0.7);
        let instance_count = segments.len() as u32;
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("segments"),
            contents: bytemuck::cast_slice(&segments),
            usage: wgpu::BufferUsages::VERTEX,
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

        // --- Tonemap pipeline (HDR texture -> swapchain) ---
        let tonemap_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
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

        let (hdr_view, tonemap_bind_group) =
            make_hdr_target(&device, &tonemap_layout, &sampler, width, height);

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
            uniform_buffer,
            beam_bind_group,
            instance_buffer,
            instance_count,
            tonemap_layout,
            sampler,
            hdr_view,
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
        let (hdr_view, tonemap_bind_group) = make_hdr_target(
            &self.device,
            &self.tonemap_layout,
            &self.sampler,
            width,
            height,
        );
        self.hdr_view = hdr_view;
        self.tonemap_bind_group = tonemap_bind_group;
    }

    fn render(&mut self, time: f32) {
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

        // Animate: spin the cube around two axes, look at it from +Z.
        let aspect = self.config.width as f32 / self.config.height as f32;
        let mvp = beam_mvp(aspect, time);

        let uniforms = BeamUniforms {
            mvp: mvp.to_cols_array(),
            resolution: [self.config.width as f32, self.config.height as f32],
            base_width: 6.0,
            brightness: 1.0,
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

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

        // Pass 2: tonemap HDR -> swapchain.
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

/// Build the HDR render target at `width`x`height` and a tonemap bind group that
/// samples it. Called on init and on every resize.
fn make_hdr_target(
    device: &wgpu::Device,
    tonemap_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    width: u32,
    height: u32,
) -> (wgpu::TextureView, wgpu::BindGroup) {
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
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tonemap bg"),
        layout: tonemap_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    (view, bind_group)
}

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    start: Option<Instant>,
    persistence: f32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("vector-beam")
                        .with_inner_size(winit::dpi::LogicalSize::new(960.0, 720.0)),
                )
                .expect("create window"),
        );
        let gpu = pollster::block_on(GpuState::new(window.clone(), self.persistence));
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
                let t = self.start.map(|s| s.elapsed().as_secs_f32()).unwrap_or(0.0);
                gpu.render(t);
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
    // `--persistence <seconds>` sets the phosphor fade time constant in either
    // mode (0 disables trails).
    let args: Vec<String> = std::env::args().collect();
    let persistence = args
        .iter()
        .position(|a| a == "--persistence")
        .and_then(|p| args.get(p + 1))
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(DEFAULT_PERSISTENCE)
        .max(0.0);
    if let Some(pos) = args.iter().position(|a| a == "--screenshot") {
        let path = args
            .get(pos + 1)
            .filter(|a| !a.starts_with("--"))
            .cloned()
            .unwrap_or_else(|| "docs/screenshot.png".to_string());
        let (width, height) = args
            .get(pos + 2)
            .and_then(|s| s.split_once('x'))
            .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
            .unwrap_or((1280, 960));
        // A few seconds in, the cube sits at a pleasant three-quarter angle.
        screenshot::capture(&path, width, height, 2.6, persistence);
        println!("wrote {path} ({width}x{height})");
        return;
    }

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        persistence,
        ..App::default()
    };
    event_loop.run_app(&mut app).expect("run app");
}
