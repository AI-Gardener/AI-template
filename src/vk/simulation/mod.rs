use super::*;

mod buffers;
pub use buffers::*;

// Always-supported uniform range: 16384 bytes. [From here](https://docs.vulkan.org/spec/latest/chapters/limits.html), maxUniformBufferRange entry.
// 1 Mat4: 64 bytes
// 8 images for the swapchain should be enough, leaving a maximum of 32 matrices per uniform. 8 matrices should be enough.
// 64 bytes per matrix * 8 matrices per uniform * 8 uniforms total = 4096 bytes, a fourth of the limit.
pub(crate) const MAX_MATRICES: usize = 32;
pub(crate) const MAX_UNIFORM_BUFFERS: usize = 8;

pub(crate) struct Simulation {
    device: Arc<Device>,


    drawing_vs: EntryPoint,
    drawing_fs: EntryPoint,

    drawing_pipeline: Arc<GraphicsPipeline>,
    drawing_descriptor_sets: Vec<Arc<PersistentDescriptorSet>>,

    // vertices: Vec<InputVertex>,
    vertex_buffer: Subbuffer<[InputVertex]>,

    indexed: bool,
    // indices: Vec<u32>,
    index_buffer: Subbuffer<[u32]>,

    transformations: [Mat4; MAX_MATRICES],
    staging_transform_uniform: Subbuffer<[Mat4]>,
    transform_uniforms: Vec<Subbuffer<[Mat4]>>,

    push: Push,
}

impl Simulation {
    pub(crate) fn init<State: Reinforcement + Clone + Send + Sync>(
        vkinner: &VkInner,
        // device: Arc<Device>,
        // memory_allocator: Arc<StandardMemoryAllocator>,
        // ds_allocator: &StandardDescriptorSetAllocator,
        // render_pass: Arc<RenderPass>,
        // image_extent: [u32; 3],
    ) -> Self {
        let device = vkinner.device.clone();

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
            let subpass = Subpass::from(vkinner.render_pass.clone(), 0).unwrap();

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
                            extent: [vkinner.image_extent[0] as f32, vkinner.image_extent[1] as f32],
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
            vkinner.memory_allocator.clone(),
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
            vkinner.memory_allocator.clone(),
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
            vkinner.memory_allocator.clone(),
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
        let transform_uniforms: Vec<Subbuffer<[Mat4]>> = (0..MAX_UNIFORM_BUFFERS)
            .map(|_| {
                Buffer::new_slice::<Mat4>(
                    vkinner.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
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
                .unwrap()
            })
            .collect();

        let drawing_descriptor_sets = (0..MAX_UNIFORM_BUFFERS)
            .map(|n| {
                PersistentDescriptorSet::new(
                    &vkinner.ds_allocator,
                    drawing_pipeline
                        .layout()
                        .set_layouts()
                        .get(0)
                        .unwrap()
                        .clone(),
                    [WriteDescriptorSet::buffer(0, transform_uniforms[n].clone())],
                    [],
                )
                .unwrap()
            })
            .collect();

        let push = Push {
            resolution: [vkinner.image_extent[0], vkinner.image_extent[1]],
            time: 0.0,
            hw_ratio: vkinner.image_extent[1] as f32 / vkinner.image_extent[0] as f32,
            swap_redblue: 0,
        };

        Self {
            device,


            drawing_vs,
            drawing_fs,

            drawing_pipeline,
            drawing_descriptor_sets,

            // vertices,
            vertex_buffer,

            indexed,
            // indices,
            index_buffer,

            transformations,
            staging_transform_uniform,
            transform_uniforms,

            push,
        }
    }
}

impl<State: Reinforcement + Clone + Send + Sync> Painter<State> for Simulation {
    fn recreate_pipeline(&mut self, render_pass: Arc<RenderPass>, image_extent: [u32; 2]) {
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

    fn prepare_draw(
        &mut self,
        vkinner: &VkInner,
        agent: &mut Agent<State>,
        image_index: u32,
    ) {
        // TODO Less buffers? Less desciptors? Less descriptor updates?
        // let transform_uniform_offset = MAX_MATRICES as u32 * image_index;
        // ----- While GPU is executing
        agent.state.draw_transformations(&mut self.transformations);
        {
            let mut write_guard = self.staging_transform_uniform.write().unwrap();

            for (o, i) in write_guard.iter_mut().zip(self.transformations.iter()) {
                *o = *i;
            }
        }

        let mut uniform_copy_cb_builder = AutoCommandBufferBuilder::primary(
            &vkinner.cb_allocator,
            vkinner.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        uniform_copy_cb_builder
            .copy_buffer(CopyBufferInfo::buffers(
                    self.staging_transform_uniform.clone().into_bytes(),
                    self.transform_uniforms[image_index as usize].clone().into_bytes(),
                )
            )
            .unwrap();
        let uniform_copy_command_buffer = uniform_copy_cb_builder.build().unwrap();

        sync::now(vkinner.device.clone()).boxed()
            .then_execute(vkinner.queue.clone(), uniform_copy_command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        self.push = Push {
            resolution: [vkinner.image_extent[0], vkinner.image_extent[1]],
            time: agent.instant,
            hw_ratio: vkinner.image_extent[1] as f32 / vkinner.image_extent[0] as f32,
            swap_redblue: vkinner.swap_redblue as u32,
        };
    }

    fn draw(
        &mut self,
        cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: u32
    ) {
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
            .push_constants(self.drawing_pipeline.layout().clone(), 0, self.push.clone())
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
