use super::*;

mod matrix;
pub use matrix::*;

use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, RenderPassBeginInfo
    }, descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet}, device::{
        physical::{PhysicalDevice, PhysicalDeviceType}, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags
    }, format::Format, image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount}, instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        }, layout::PipelineDescriptorSetLayoutCreateInfo, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo
    }, render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass}, shader::EntryPoint, swapchain::{
        acquire_next_image, CompositeAlpha, PresentMode, Surface, SurfaceCapabilities, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo
    }, sync::{self, GpuFuture}, Validated, Version, VulkanError, VulkanLibrary
};
use winit::{
    dpi::{LogicalSize, Size}, event::{Event, WindowEvent}, event_loop::{ControlFlow, EventLoop}, platform::run_return::EventLoopExtRunReturn, window::{Window, WindowBuilder}
};
const MAX_MATRICES: usize = 128;
const MAX_IMAGES: usize = 16;

pub struct Vk<State: Reinforcement + Clone + Send + Sync>  {
    event_loop: EventLoop<()>,
    inner: VkInner<State>,
}
#[allow(unused)]
struct VkInner<State: Reinforcement + Clone + Send + Sync>  {
    library: Arc<vulkano::VulkanLibrary>,
    instance_extensions: InstanceExtensions,
    instance: Arc<Instance>,

    window: Arc<Window>,
    surface: Arc<Surface>,

    device_extensions: DeviceExtensions,
    physical_device: Arc<PhysicalDevice>,
    queue_family_index: u32,
    device: Arc<Device>,
    queue: Arc<Queue>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    cb_allocator: StandardCommandBufferAllocator,
    ds_allocator: StandardDescriptorSetAllocator,
    
    drawing_vs: EntryPoint,
    drawing_fs: EntryPoint,

    surface_capabilities: SurfaceCapabilities,
    min_image_count: u32,
    image_format: Format,
    composite_alpha: CompositeAlpha,
    present_mode: PresentMode,

    msaa_image: Arc<ImageView>,
    images: Vec<Arc<Image>>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    drawing_pipeline: Arc<GraphicsPipeline>,
    drawing_descriptor_set: Arc<PersistentDescriptorSet>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,

    vertices: Vec<InputVertex>,
    vertex_buffer: Subbuffer<[InputVertex]>,
    transformations: [Mat4; MAX_MATRICES],
    staging_transform_uniform: Subbuffer<[Mat4]>,
    transform_uniform: Subbuffer<[Mat4]>,

    state: State,
}

/// A single vertex, part of a triangle that will be drawn
#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
pub struct InputVertex {
    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32_SINT)]
    transform_id: i32,
}

impl InputVertex {
    pub fn new(color: [f32; 4], position: [f32; 3], transform_id: i32) -> Self {
        Self {
            color,
            position,
            transform_id,
        }
    }
}

/// Push constants
#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub(crate) struct Push {
    /// The offset of the transformations data in the dynamic transform_uniform buffer.
    pub(crate) transform_uniform_offset: u32,
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(set = 0, binding = 0) uniform ModelTransformations {
                mat4 data[128];
            } tf;
            layout(push_constant) uniform GeneralInfo {
                int offset;
            } gen;

            layout(location = 0) in vec4 color;
            layout(location = 1) in vec3 position;
            layout(location = 2) in int transform_id;

            layout(location = 0) out vec4 v_color;

            void main() {
                gl_Position = vec4(position, 1.0);
                if (transform_id > -1) {
                    gl_Position *= tf.data[gen.offset + transform_id];
                }
                v_color = color;
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) in vec4 v_color;

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = v_color;
            }
        ",
    }
}

impl<State: Reinforcement + Clone + Send + Sync> Vk<State> {
    pub fn init() -> Self {
        let event_loop = EventLoop::new();

        // --------------------------------------------------

        let library = VulkanLibrary::new().unwrap();
        let instance_extensions = Surface::required_extensions(&event_loop);
        let instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: instance_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        // --------------------------------------------------

        let window = Arc::new(WindowBuilder::new()
            .with_decorations(true)
            .with_inner_size(Size::Logical(LogicalSize::from([1280, 720])))
            .with_resizable(true)
            .with_title("RENAME ME")
            .with_transparent(true)
            .build(&event_loop)
            .unwrap()
        );
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        // --------------------------------------------------

        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
            })
            .filter(|p| {
                p.supported_extensions().contains(&device_extensions)
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| {
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("no suitable physical device found");
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );
        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }
        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: Features {
                    dynamic_rendering: true,
                    ..Features::empty()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let queue = queues.next().unwrap();

        // --------------------------------------------------

        let memory_allocator = Arc::new(
            StandardMemoryAllocator::new_default(device.clone()));
        let cb_allocator = 
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let ds_allocator = 
            StandardDescriptorSetAllocator::new(device.clone(), Default::default());

        // --------------------------------------------------

        let drawing_vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let drawing_fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        // --------------------------------------------------
    
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let min_image_count = surface_capabilities.min_image_count.max(2);
        println!("Max image count is {:?}", surface_capabilities.max_image_count);
        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;
        let composite_alpha = surface_capabilities
            .supported_composite_alpha
            .into_iter()
            .next()
            .unwrap();
        let present_mode = State::draw_view_present_mode();

        // --------------------------------------------------

    
        let (msaa_image, (swapchain, images)) = (
            ImageView::new_default(
                Image::new(
                    memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: image_format,
                        extent: [window.inner_size().width, window.inner_size().width, 1],
                        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                        samples: SampleCount::Sample8,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            ).unwrap(),
            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count,
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::TRANSFER_SRC
                                | ImageUsage::TRANSFER_DST
                                | ImageUsage::COLOR_ATTACHMENT
                                | ImageUsage::STORAGE,
                    composite_alpha,
                    present_mode,
    
                    ..Default::default()
                },
            ).unwrap()
        );
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                msaa_color: {
                    format: image_format,
                    samples: 8,
                    load_op: Clear,
                    store_op: DontCare,
                },
                color: {
                    format: image_format,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [msaa_color],
                color_resolve: [color],
                depth_stencil: {},
            },
        )
        .unwrap();
        let framebuffers = images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![msaa_image.clone(), view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
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
            let extent = images[0].extent();
    
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
                            extent: [extent[0] as f32, extent[1] as f32],
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
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };
        let recreate_swapchain = false;
        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        // --------------------------------------------------
    
        let vertices = State::draw_vertices().0;
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

        let transformations = [Mat4::new(); MAX_MATRICES];
        let staging_transform_uniform: Subbuffer<[Mat4]> = Buffer::new_slice::<Mat4>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
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
        .unwrap();
        let transform_uniform: Subbuffer<[Mat4]> = Buffer::new_slice::<Mat4>(
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
            (MAX_MATRICES * MAX_IMAGES) as u64,
        )
        .unwrap();

        let drawing_descriptor_set = PersistentDescriptorSet::new(
            &ds_allocator,
            drawing_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, transform_uniform.clone()),
            ],
            [],
        )
        .unwrap();

        let state = State::init();

        Vk {
            event_loop,
            inner: VkInner {
                library,
                instance_extensions,
                instance,

                window,
                surface,

                device_extensions,
                physical_device,
                queue_family_index,
                device,
                queue,

                memory_allocator,
                cb_allocator,
                ds_allocator,

                drawing_vs,
                drawing_fs,

                surface_capabilities,
                min_image_count,
                image_format,
                composite_alpha,
                present_mode,

                msaa_image,
                images,
                swapchain,
                render_pass,
                framebuffers,
                drawing_pipeline,
                drawing_descriptor_set,
                recreate_swapchain,
                previous_frame_end,

                vertices,
                vertex_buffer,
                transformations,
                staging_transform_uniform,
                transform_uniform,

                state,
            },
        }
    }

    pub fn view(&mut self) {
        self.inner.view(&mut self.event_loop);
    }

    pub fn save(&mut self) {
        // TODO self.inner.save(&mut self.event_loop);
    }
}

impl<State: Reinforcement + Clone + Send + Sync> VkInner<State> {
    fn view(&mut self, event_loop: &mut EventLoop<()>) {
        let mut count = 0;
        event_loop.run_return(move |event, _, control_flow| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    *control_flow = ControlFlow::Exit;
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    self.recreate_swapchain = true;
                }
                Event::RedrawEventsCleared => {
                    let image_extent: [u32; 2] = self.window.inner_size().into();
                    if image_extent.contains(&0) {
                        return;
                    }
                    self.recreate_swapchain(image_extent);
                    self.previous_frame_end.as_mut().unwrap().cleanup_finished();
                    
                    self.state.draw_transformations(&mut self.transformations);

                    let (image_index, suboptimal, acquire_future) =
                        match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                            Ok(r) => r,
                            Err(VulkanError::OutOfDate) => {
                                self.recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("failed to acquire next image: {e}"),
                        };
                    if suboptimal {
                        self.recreate_swapchain = true;
                    }

                    let transform_uniform_offset = MAX_MATRICES as u32 * image_index;
                    {
                        let mut write_guard = self.staging_transform_uniform.write().unwrap();
            
                        for (o, i) in write_guard.iter_mut().skip(transform_uniform_offset as usize).take(MAX_MATRICES).zip(self.transformations.iter()) {
                            *o = *i;
                        }
                    }
                    let push = Push { transform_uniform_offset };

                    let mut uniform_copy_cb_builder = AutoCommandBufferBuilder::primary(
                        &self.cb_allocator,
                        self.queue.queue_family_index(),
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap();
                    uniform_copy_cb_builder
                        .copy_buffer({
                            let mut cbi = CopyBufferInfo::buffers(
                                self.staging_transform_uniform.clone().into_bytes(),
                                self.transform_uniform.clone().into_bytes(),
                            );
                            cbi.regions[0].dst_offset = (MAX_MATRICES * image_index as usize) as u64;
                            cbi
                        })
                        .unwrap();
                    let uniform_copy_command_buffer = uniform_copy_cb_builder.build().unwrap();

                    sync::now(self.device.clone()).boxed()
                        .then_execute(self.queue.clone(), uniform_copy_command_buffer)
                        .unwrap()
                        .then_signal_fence_and_flush()
                        .unwrap()
                        .wait(None)
                        .unwrap();

                    // TODO Reuse command buffer
                    let mut cb_builder = AutoCommandBufferBuilder::primary(
                        &self.cb_allocator,
                        self.queue.queue_family_index(),
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap();
                    cb_builder
                        .begin_render_pass(
                            RenderPassBeginInfo {
                                clear_values: vec![
                                    Some([0.5, 0.5, 0.5, 0.5].into()),
                                    None,
                                ],
                                ..RenderPassBeginInfo::framebuffer(self.framebuffers[image_index as usize].clone())
                            },
                            Default::default(),
                        )
                        .unwrap()
                        .bind_pipeline_graphics(self.drawing_pipeline.clone())
                        .unwrap()
                        .bind_vertex_buffers(0, self.vertex_buffer.clone())
                        .unwrap()
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            self.drawing_pipeline.layout().clone(),
                            0,
                            self.drawing_descriptor_set.clone(),
                        )
                        .unwrap()
                        .push_constants(
                            self.drawing_pipeline.layout().clone(),
                            0,
                            push
                        )
                        .unwrap()
                        .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
                        .unwrap()
                        .end_render_pass(Default::default())
                        .unwrap();
                    let command_buffer = cb_builder.build().unwrap();
                        
                    let future = self.previous_frame_end
                        .take()
                        .unwrap()
                        .join(acquire_future)
                        .then_execute(self.queue.clone(), command_buffer)
                        .unwrap()
                        .then_swapchain_present(
                            self.queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
                        )
                        .then_signal_fence_and_flush();
                    count+=1;
                    println!("Drawing frame {}", count);
                    match future.map_err(Validated::unwrap) {
                        Ok(future) => {
                            self.previous_frame_end = Some(future.boxed());
                        }
                        Err(VulkanError::OutOfDate) => {
                            self.recreate_swapchain = true;
                            self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                        }
                        Err(e) => {
                            println!("failed to flush future: {e}");
                            self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                        }
                    }
                }
                _ => (),
            }
        });
    }

    fn recreate_swapchain(&mut self, image_extent: [u32; 2]) {
        if self.recreate_swapchain {
            self.recreate_swapchain = false;

            (self.msaa_image, (self.swapchain, self.images)) = {
                (
                    ImageView::new_default(
                        Image::new(
                            self.memory_allocator.clone(),
                            ImageCreateInfo {
                                image_type: ImageType::Dim2d,
                                format: self.image_format,
                                extent: [self.window.inner_size().width, self.window.inner_size().width, 1],
                                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                                samples: SampleCount::Sample8,
                                ..Default::default()
                            },
                            AllocationCreateInfo::default(),
                        )
                        .unwrap(),
                    ).unwrap(),
                    self.swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..self.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain")
                )
            };

            self.framebuffers = self.images
                .iter()
                .map(|image| {
                    let view = ImageView::new_default(image.clone()).unwrap();
                    Framebuffer::new(
                        self.render_pass.clone(),
                        FramebufferCreateInfo {
                            attachments: vec![self.msaa_image.clone(), view],
                            ..Default::default()
                        },
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();

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
                let subpass = Subpass::from(self.render_pass.clone(), 0).unwrap();
                let extent = self.images[0].extent();

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
                                extent: [extent[0] as f32, extent[1] as f32],
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
                        ..GraphicsPipelineCreateInfo::layout(layout)
                    },
                )
                .unwrap()
            };
        }
    }
}