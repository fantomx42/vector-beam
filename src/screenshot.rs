//! Headless one-frame capture: render the beam scene into an offscreen sRGB
//! texture (no window, no compositor), read it back, and write a PNG. Used by
//! `vector-beam --screenshot`. Reuses the same shaders, geometry, and MVP as the
//! live renderer so the docs image matches what the window shows.

use wgpu::util::DeviceExt;

use crate::{beam_mvp, geometry, BeamUniforms, HDR_FORMAT, SEGMENT_ATTRS};

/// The swapchain is sRGB at runtime; match that here so the read-back bytes are
/// already sRGB-encoded and drop straight into a PNG.
const OUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub fn capture(path: &str, width: u32, height: u32, time: f32) {
    pollster::block_on(capture_async(path, width, height, time));
}

async fn capture_async(path: &str, width: u32, height: u32, time: f32) {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None, // headless: no surface to be compatible with
            force_fallback_adapter: false,
        })
        .await
        .expect("no suitable GPU adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("screenshot device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("request device");

    // --- Geometry + uniforms ---
    let segments = geometry::wireframe_cube(0.7);
    let instance_count = segments.len() as u32;
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("segments"),
        contents: bytemuck::cast_slice(&segments),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let uniforms = BeamUniforms {
        mvp: beam_mvp(width as f32 / height as f32, time).to_cols_array(),
        resolution: [width as f32, height as f32],
        // Slightly wider than the interactive default so beams read well at the
        // higher still-image resolution.
        base_width: 8.0,
        brightness: 1.0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("beam uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // --- Beam pipeline (-> HDR target, additive) ---
    let beam_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/beam.wgsl"));
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
    let beam_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("beam pl layout"),
        bind_group_layouts: &[Some(&beam_layout)],
        immediate_size: 0,
    });
    let additive = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
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
                blend: Some(wgpu::BlendState {
                    color: additive,
                    alpha: additive,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // --- Tonemap pipeline (HDR texture -> sRGB output) ---
    let tonemap_shader =
        device.create_shader_module(wgpu::include_wgsl!("../shaders/tonemap.wgsl"));
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
                format: OUT_FORMAT,
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

    // --- Targets ---
    let hdr = device.create_texture(&wgpu::TextureDescriptor {
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
    let hdr_view = hdr.create_view(&wgpu::TextureViewDescriptor::default());

    let out = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OUT_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_view = out.create_view(&wgpu::TextureViewDescriptor::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("hdr sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let tonemap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tonemap bg"),
        layout: &tonemap_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&hdr_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    // Readback buffer. Rows in a texture-to-buffer copy must be aligned to
    // COPY_BYTES_PER_ROW_ALIGNMENT (256 bytes), so pad and strip afterwards.
    let unpadded_bpr = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bpr = unpadded_bpr.div_ceil(align) * align;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded_bpr * height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- Encode + submit ---
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("capture") });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("beam pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &hdr_view,
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
        pass.set_pipeline(&beam_pipeline);
        pass.set_bind_group(0, &beam_bind_group, &[]);
        pass.set_vertex_buffer(0, instance_buffer.slice(..));
        pass.draw(0..6, 0..instance_count);
    }
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tonemap pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &out_view,
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
        pass.set_pipeline(&tonemap_pipeline);
        pass.set_bind_group(0, &tonemap_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &out,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    // --- Map + strip row padding ---
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("map channel").expect("map readback buffer");

    let mapped = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((unpadded_bpr * height) as usize);
    for row in 0..height {
        let start = (row * padded_bpr) as usize;
        let end = start + unpadded_bpr as usize;
        pixels.extend_from_slice(&mapped[start..end]);
    }
    drop(mapped);
    readback.unmap();

    // --- Encode PNG ---
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = std::fs::File::create(path).expect("create png file");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .expect("png header")
        .write_image_data(&pixels)
        .expect("png data");
}
