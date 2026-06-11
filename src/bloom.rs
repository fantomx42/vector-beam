//! Bloom: the host side of `shaders/bloom.wgsl`. Owns the two half-resolution
//! ping-pong textures and the three passes (bright-pass downsample, horizontal
//! blur, vertical blur). Shared by the live window and the headless screenshot;
//! the tonemap pass samples [`Bloom::output_view`] and adds it to the HDR image.

use wgpu::util::DeviceExt;

use crate::HDR_FORMAT;

/// Mirror of the WGSL `BloomUniforms` block: the source texel size for this
/// pass, and the blur axis (unused by the downsample pass).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomUniforms {
    texel: [f32; 2],
    direction: [f32; 2],
}

pub struct Bloom {
    downsample_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sized: SizedResources,
}

/// Everything that depends on the render size (and on the HDR view, which is
/// itself recreated on resize). Rebuilt wholesale by [`Bloom::resize`].
struct SizedResources {
    view_a: wgpu::TextureView,
    view_b: wgpu::TextureView,
    bg_downsample: wgpu::BindGroup, // HDR -> a
    bg_blur_h: wgpu::BindGroup,     // a -> b
    bg_blur_v: wgpu::BindGroup,     // b -> a
}

impl Bloom {
    pub fn new(
        device: &wgpu::Device,
        sampler: &wgpu::Sampler,
        hdr_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/bloom.wgsl"));
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom pl layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make_pipeline = |label: &str, entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let downsample_pipeline = make_pipeline("bloom downsample pipeline", "fs_downsample");
        let blur_pipeline = make_pipeline("bloom blur pipeline", "fs_blur");

        let sized = build_sized(device, &layout, sampler, hdr_view, width, height);
        Self {
            downsample_pipeline,
            blur_pipeline,
            layout,
            sized,
        }
    }

    /// Rebuild the size-dependent resources. Call after the HDR target was
    /// recreated, with the *new* HDR view.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        sampler: &wgpu::Sampler,
        hdr_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        self.sized = build_sized(device, &self.layout, sampler, hdr_view, width, height);
    }

    /// The blurred bloom texture the tonemap pass should sample.
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.sized.view_a
    }

    /// Encode the three bloom passes: HDR -> a (bright downsample),
    /// a -> b (horizontal blur), b -> a (vertical blur).
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let passes = [
            (&self.downsample_pipeline, &self.sized.bg_downsample, &self.sized.view_a),
            (&self.blur_pipeline, &self.sized.bg_blur_h, &self.sized.view_b),
            (&self.blur_pipeline, &self.sized.bg_blur_v, &self.sized.view_a),
        ];
        for (pipeline, bind_group, target) in passes {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
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
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

fn build_sized(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    hdr_view: &wgpu::TextureView,
    width: u32,
    height: u32,
) -> SizedResources {
    // Quarter resolution: with the fixed 9-tap kernel, the halo's reach scales
    // with the texel size, and at quarter res it clearly outranges the beam
    // shader's own analytic glow (which would otherwise mask it).
    let (bw, bh) = ((width / 4).max(1), (height / 4).max(1));
    let make_target = |label: &str| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: bw,
                    height: bh,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    };
    let view_a = make_target("bloom a");
    let view_b = make_target("bloom b");

    // The uniform buffers are tiny and immutable per size; recreate with the
    // bind groups instead of tracking COPY_DST writes.
    let make_bind_group = |label: &str, src: &wgpu::TextureView, u: BloomUniforms| {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        })
    };

    let full_texel = [1.0 / width as f32, 1.0 / height as f32];
    let half_texel = [1.0 / bw as f32, 1.0 / bh as f32];
    let bg_downsample = make_bind_group(
        "bloom downsample bg",
        hdr_view,
        BloomUniforms {
            texel: full_texel,
            direction: [0.0, 0.0],
        },
    );
    let bg_blur_h = make_bind_group(
        "bloom blur h bg",
        &view_a,
        BloomUniforms {
            texel: half_texel,
            direction: [1.0, 0.0],
        },
    );
    let bg_blur_v = make_bind_group(
        "bloom blur v bg",
        &view_b,
        BloomUniforms {
            texel: half_texel,
            direction: [0.0, 1.0],
        },
    );

    SizedResources {
        view_a,
        view_b,
        bg_downsample,
        bg_blur_h,
        bg_blur_v,
    }
}
