use super::*;

mod matrix;
use matrix::*;

use std::sync::Arc;

use vulkano::{
    buffer::{BufferContents, Subbuffer},
    command_buffer::{allocator::StandardCommandBufferAllocator, CommandBufferExecFuture},
    descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, StandardMemoryAllocator},
    padded::Padded,
    pipeline::{Pipeline, graphics::{color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState}, depth_stencil::{DepthState, DepthStencilState}, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::RasterizationState, vertex_input::{Vertex, VertexDefinition}, viewport::ViewportState, GraphicsPipelineCreateInfo}, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::EntryPoint,
    swapchain::{PresentFuture, PresentMode, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo},
    sync::{
        event::Event,
        future::{FenceSignalFuture, JoinFuture},
        GpuFuture,
    },
    Version, VulkanLibrary,
};
use winit::{
    dpi::LogicalSize,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

pub enum VkMode {
    View,
    Save(String), // And encoding params.
}

/// The struct for operating with Vulkan, to visualize the AI learning.
pub struct Vk<State: Reinforcement + Clone + Send + Sync> {
    event_loop: EventLoop<()>,
    inner: VkInner<State>,
}
struct VkInner<State: Reinforcement + Clone + Send + Sync> {
    state: State,

    // Vulkan boilerplate

    _library: Arc<VulkanLibrary>,
    _instance: Arc<Instance>,
    _surface: Arc<Surface>,
    window: Arc<Window>,

    _physical_device: Arc<PhysicalDevice>,
    _queue_family_index: u32,
    device: Arc<Device>,
    queue: Arc<Queue>,

    memory_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    cb_allocator: StandardCommandBufferAllocator,
    ds_allocator: StandardDescriptorSetAllocator,

    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,

    recreate_swapchain: bool,
    show_count: u32,
    time: f32,

    draw_pipeline: Arc<GraphicsPipeline>,
    draw_descriptor_set: Arc<PersistentDescriptorSet>,

    // vertex_compute_pipeline: Arc<ComputePipeline>,
    // vertex_compute_descriptor_set: Arc<PersistentDescriptorSet>,

    drawing_vs: EntryPoint,
    drawing_fs: EntryPoint,
    drawing_fences: Vec<
        Option<
            FenceSignalFuture<
                PresentFuture<
                    CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>>,
                >,
            >,
        >,
    >,
    previous_drawing_fence_i: u32,

    // -----
    background_color: [f32; 4],
    start_time: f32,
    end_time: f32,
    // vertex_dispatch_len: u32,
    // model_dispatch_len: u32,
}

/// A vertex with some position and color, given as input to the graphics pipeline.
#[derive(BufferContents, Vertex, Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct SingleVertex {
    /// The (x, y, z, w) coordinates of this vertex.
    #[format(R32G32B32A32_SFLOAT)]
    pub(crate) position: [f32; 4],
    /// The rgba color of this vertex.
    #[format(R32G32B32A32_SFLOAT)]
    pub(crate) color: [f32; 4],
}
impl Default for SingleVertex {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0, 1.0],
            color: [0.5, 1.0, 0.8, 1.0],
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec4 position;
            layout(location = 1) in vec4 color;

            layout(location = 0) out vec4 vert_color;

            void main() {
                gl_Position = position;
                vert_color = color;
            }
        ",
    }
}
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) in vec4 vert_color;

            layout(location = 0) out vec4 frag_color;

            void main() {
                frag_color = vert_color;
            }
        ",
    }
}
impl<State: Reinforcement + Clone + Send + Sync> Vk<State> {
    pub fn init() -> Self {
        // --------------------------------------------------     Vulkan boilerplate

        let event_loop = EventLoop::new();
        let _library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let required_extensions = Surface::required_extensions(&event_loop);
        let _instance = Instance::new(
            _library.clone(),
            InstanceCreateInfo {
                application_name: Some("AI tester".to_owned()),
                application_version: Version::major_minor(0, 1),
                enabled_extensions: required_extensions,
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let window = Arc::new(
            WindowBuilder::new()
                .with_inner_size(LogicalSize {
                    width: 1280,
                    height: 720,
                })
                .with_resizable(true)
                .with_decorations(false) // TODOFEATURES
                .with_title("AI tester")
                .with_transparent(true)
                .with_maximized(true)
                .build(&event_loop)
                .unwrap(),
        );
        let _surface = Surface::from_window(_instance.clone(), window.clone()).unwrap();
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (_physical_device, _queue_family_index) = _instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.contains(QueueFlags::GRAPHICS)
                            && q.queue_flags.contains(QueueFlags::COMPUTE)
                            && q.queue_flags.contains(QueueFlags::TRANSFER)
                            && p.surface_support(i as u32, &_surface).unwrap_or(false)
                    })
                    .map(|q| (p, q as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("no device available");

        println!(
            "Using device: {} (type: {:?})",
            _physical_device.properties().device_name,
            _physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            _physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: _queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create device");
        let queue = queues.next().unwrap();

        // Allocators
        // ----------
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let cb_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let ds_allocator = StandardDescriptorSetAllocator::new(device.clone(), Default::default());

        // --------

        let caps = _physical_device
            .surface_capabilities(&_surface, Default::default())
            .expect("failed to get surface capabilities");
        let dimensions = window.inner_size();
        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = _physical_device
            .clone()
            .surface_formats(&_surface, Default::default())
            .unwrap()[0]
            .0;

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            _surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count.max(2),
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::TRANSFER_SRC
                    | ImageUsage::TRANSFER_DST
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::STORAGE,
                composite_alpha,
                present_mode: PresentMode::Fifo,
                ..Default::default()
            },
        )
        .unwrap();

        let extent: [u32; 3] = images[0].extent();

        let msaa_image: Arc<ImageView> = ImageView::new_default(
            Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: swapchain.image_format(),
                    extent,
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                    samples: SampleCount::Sample16,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                msaa_color: {
                    format: swapchain.image_format(),
                    samples: 8,
                    load_op: Clear,
                    store_op: DontCare,
                },
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [msaa_color],
                color_resolve: [color],
                depth_stencil: {  },
            },
        )
        .unwrap();

        let framebuffers: Vec<Arc<Framebuffer>> = images
            .iter()
            .map(|image| {
                let final_view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![
                            msaa_image.clone(),
                            final_view,
                        ],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<Arc<Framebuffer>>>();

        

        // Graphics pipeline & Drawing command buffer
        // ------------------------------------------
        let drawing_vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .expect("failed to create vertex shader module");
        let drawing_fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .expect("failed to create fragment shader module");
        // let vertex_cs = vertex_cs::load(device.clone())
        //     .unwrap()
        //     .entry_point("main")
        //     .expect("failed to create vertex compute shader module");
        // let model_cs = model_cs::load(device.clone())
        //     .unwrap()
        //     .entry_point("main")
        //     .expect("failed to create model compute shader module");

        let draw_pipeline: Arc<GraphicsPipeline> = {
            let vertex_input_state = SingleVertex::per_vertex()
                .definition(&drawing_vs.info().input_interface)
                .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(drawing_vs.clone()),
                PipelineShaderStageCreateInfo::new(drawing_fs.clone()),
            ];

            let pipeline_layout = PipelineLayout::new(
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
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState {
                        rasterization_samples: subpass.num_samples().unwrap(),
                        ..Default::default()
                    }),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend::alpha()),
                            ..Default::default()
                        },
                    )),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
                },
            )
            .unwrap()
        };
        let draw_descriptor_set: Arc<PersistentDescriptorSet> = PersistentDescriptorSet::new(
            &ds_allocator,
            draw_pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [

            ],
            [],
        )
        .unwrap();

        // let vertex_compute_pipeline: Arc<ComputePipeline> = {
        //     let stage = PipelineShaderStageCreateInfo::new(vertex_cs);
        //     let layout = PipelineLayout::new(
        //         device.clone(),
        //         PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
        //             .into_pipeline_layout_create_info(device.clone())
        //             .unwrap(),
        //     )
        //     .unwrap();
        //     ComputePipeline::new(
        //         device.clone(),
        //         None,
        //         ComputePipelineCreateInfo::stage_layout(stage, layout),
        //     )
        //     .unwrap()
        // };
        // let vertex_compute_descriptor_set: Arc<PersistentDescriptorSet> =
        //     PersistentDescriptorSet::new(
        //         &ds_allocator,
        //         vertex_compute_pipeline
        //             .layout()
        //             .set_layouts()
        //             .get(0)
        //             .unwrap()
        //             .clone(),
        //         [
        //             WriteDescriptorSet::buffer(0, vsinput_buffer.clone()),
        //             WriteDescriptorSet::buffer(1, basevertex_buffer.clone()),
        //             WriteDescriptorSet::buffer(2, vertex_matrixtransformer_buffer.clone()),
        //             WriteDescriptorSet::buffer(3, vertex_matrixtransformation_buffer.clone()),
        //             WriteDescriptorSet::buffer(4, vertex_colortransformer_buffer.clone()),
        //             WriteDescriptorSet::buffer(5, vertex_colortransformation_buffer.clone()),
        //             // WriteDescriptorSet::buffer(6, entity_buffer.clone()),
        //             // WriteDescriptorSet::buffer(7, modelt_buffer.clone()),
        //             // WriteDescriptorSet::buffer(8, model_matrixtransformer_buffer.clone()),
        //             // WriteDescriptorSet::buffer(9, model_matrixtransformation_buffer.clone()),
        //         ],
        //         [],
        //     )
        //     .unwrap();

        let drawing_fences: Vec<Option<FenceSignalFuture<_>>> =
            (0..framebuffers.len()).map(|_| None).collect();
        let previous_drawing_fence_i: u32 = 0;

        // ------------------------------------------

        // Window-related updates
        // ----------------------
        let recreate_swapchain: bool = false;
        let show_count: u32 = 0;
        let time: f32 = 0.0;

        let background_color = [0.5, 0.5, 0.5, 0.5];
        let start_time = 0.0;
        let end_time = 0.0;

        // ----------------------

        Vk {
            event_loop,
            inner: VkInner {
                state: State::init(),

                _library,
                _instance,
                _surface,
                window,

                _physical_device,
                _queue_family_index,
                device,
                queue,

                swapchain,
                images,
                render_pass,
                framebuffers,

                memory_allocator,
                cb_allocator,
                ds_allocator,

                recreate_swapchain,
                show_count,
                time,

                draw_pipeline,
                draw_descriptor_set,

                // vertex_compute_pipeline,
                // vertex_compute_descriptor_set,

                drawing_vs,
                drawing_fs,
                drawing_fences,
                previous_drawing_fence_i,

                background_color,
                start_time,
                end_time,
                // vertex_dispatch_len,
                // model_dispatch_len,
            },
        }
    }

    /// Runs the performance for this graph network.
    pub fn evolution(&mut self, mode: VkMode, dac: &DAC) {

    }

    /// Runs the performance for this graph network.
    pub fn dac(&mut self, mode: VkMode, dac: &DAC) {

    }
}
