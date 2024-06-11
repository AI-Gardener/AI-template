use super::*;

mod buffers;
pub use buffers::*;

// Always-supported uniform range: 16384 bytes. [From here](https://docs.vulkan.org/spec/latest/chapters/limits.html)
// 1 Mat4: 64 bytes
// 8 images for the swapchain should be enough, leaving 32 matrices per uniform.
// 64 bytes per matrix * 32 matrices per uniform * 8 uniforms total = 16384 bytes
pub(crate) const MAX_MATRICES: usize = 32;
pub(crate) const MAX_UNIFORM_BUFFERS: usize = 8;

pub(crate) struct Simulation {
    device: Arc<Device>,


    drawing_vs: EntryPoint,
    drawing_fs: EntryPoint,
    
    drawing_pipeline: Arc<GraphicsPipeline>,
    drawing_descriptor_sets: Vec<Arc<PersistentDescriptorSet>>,

    vertices: Vec<InputVertex>,
    vertex_buffer: Subbuffer<[InputVertex]>,
    indexed: bool,
    indices: Vec<u32>,
    index_buffer: Subbuffer<[u32]>,
    transformations: [Mat4; MAX_MATRICES],
    staging_transform_uniform: Subbuffer<[Mat4]>,
    transform_uniforms: Vec<Subbuffer<[Mat4]>>,
}

impl Painter for Simulation {
    fn init<State: Reinforcement + Clone + Send + Sync>(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        ds_allocator: StandardDescriptorSetAllocator,
        render_pass: Arc<RenderPass>,
        image_extent: [f32; 3],
    ) -> Self {
        let device = device.clone();

        let drawing_vs = simulation_vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let drawing_fs = simulation_fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let drawing_pipeline = {
            let vertex_input_state = InputVertex::per_vertex()
                .definition(&drawing_vs.info().input_interface)
                .unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(drawing_vs.clone()),
                PipelineShaderStageCreateInfo::new(drawing_fs.clone()),
            ];
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
    
            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [Viewport {
                            offset: [0.0, 0.0],
                            extent: [image_extent[0] as f32, image_extent[1] as f32], // images[0].extent()
                            depth_range: 0.0..=1.0,
                        }]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    }),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState {
                        rasterization_samples: subpass.num_samples().unwrap(),
                        ..Default::default()
                    }),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    subpass: Some(subpass.into()),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };
        
        let (vertices, indices) = State::draw_vertices();
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices.clone(),
        )
        .unwrap();

        let indexed = indices.is_some();
        let indices = match indices {
            Some(ind) => ind,
            None => (0..10).into_iter().collect(),
        };
        let index_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices.clone(),
        )
        .unwrap();

        let transformations = [Mat4::new(); MAX_MATRICES];
        let staging_transform_uniform: Subbuffer<[Mat4]> = Buffer::new_slice::<Mat4>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            // transformations.len() as u64,
            MAX_MATRICES as u64,
        )
        .unwrap();
        let transform_uniforms: Vec<Subbuffer<[Mat4]>> = (0..MAX_UNIFORM_BUFFERS).map(|_| Buffer::new_slice::<Mat4>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            // transformations.len() as u64,
            MAX_MATRICES as u64,
        )
        .unwrap()).collect();

        let drawing_descriptor_sets = (0..MAX_UNIFORM_BUFFERS).map(|n| PersistentDescriptorSet::new(
            &ds_allocator,
            drawing_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, transform_uniforms[n].clone()),
            ],
            [],
        )
        .unwrap()).collect();

        Self {
            device,


            drawing_vs,
            drawing_fs,
            
            drawing_pipeline,
            drawing_descriptor_sets,
        
            vertices,
            vertex_buffer,
            indexed,
            indices,
            index_buffer,
            transformations,
            staging_transform_uniform,
            transform_uniforms,
        }
    }

    fn recreate_pipeline(
            &mut self,
            render_pass: Arc<RenderPass>,
            image_extent: [f32; 3]
        ) {

            self.drawing_pipeline = {
                let vertex_input_state = InputVertex::per_vertex()
                    .definition(&self.drawing_vs.info().input_interface)
                    .unwrap();
                let stages = [
                    PipelineShaderStageCreateInfo::new(self.drawing_vs.clone()),
                    PipelineShaderStageCreateInfo::new(self.drawing_fs.clone()),
                ];
                let layout = PipelineLayout::new(
                    self.device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                        .into_pipeline_layout_create_info(self.device.clone())
                        .unwrap(),
                )
                .unwrap();
                let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        
                GraphicsPipeline::new(
                    self.device.clone(),
                    None,
                    GraphicsPipelineCreateInfo {
                        stages: stages.into_iter().collect(),
                        vertex_input_state: Some(vertex_input_state),
                        input_assembly_state: Some(InputAssemblyState::default()),
                        viewport_state: Some(ViewportState {
                            viewports: [Viewport {
                                offset: [0.0, 0.0],
                                extent: [image_extent[0] as f32, image_extent[1] as f32], // images[0].extent()
                                depth_range: 0.0..=1.0,
                            }]
                            .into_iter()
                            .collect(),
                            ..Default::default()
                        }),
                        rasterization_state: Some(RasterizationState::default()),
                        multisample_state: Some(MultisampleState {
                            rasterization_samples: subpass.num_samples().unwrap(),
                            ..Default::default()
                        }),
                        color_blend_state: Some(ColorBlendState::with_attachment_states(
                            subpass.num_color_attachments(),
                            ColorBlendAttachmentState::default(),
                        )),
                        subpass: Some(subpass.into()),
                        depth_stencil_state: Some(DepthStencilState {
                            depth: Some(DepthState::simple()),
                            ..Default::default()
                        }),
                        ..GraphicsPipelineCreateInfo::layout(layout)
                    },
                )
                .unwrap()
            };
    }

    fn prepare_draw(&mut self) {
        
    }

    fn draw(&mut self, cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        cb_builder
            .bind_pipeline_graphics(self.drawing_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.drawing_pipeline.layout().clone(),
                0,
                self.drawing_descriptor_sets[image_index as usize].clone(),
            )
            .unwrap()
            .push_constants(
                self.drawing_pipeline.layout().clone(),
                0,
                push
            )
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap();
        if self.indexed {
            cb_builder
            .bind_index_buffer(self.index_buffer.clone())
            .unwrap()
            .draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0)
            .unwrap()
        } else {
            cb_builder
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap()
        };
    }
}