use super::*;

mod utils;
pub use utils::*;
mod simulation;
pub use simulation::*;
mod network;
// use network::*;
mod shaders;
use shaders::*;

use image::ImageBuffer;

use std::{sync::Arc, time::Instant};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyImageToBufferInfo, PrimaryAutoCommandBuffer, RenderPassBeginInfo
    }, descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet}, device::{
        physical::{PhysicalDevice, PhysicalDeviceType}, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags
    }, format::Format, image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount}, instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState}, depth_stencil::{DepthState, DepthStencilState}, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::RasterizationState, vertex_input::{Vertex, VertexDefinition}, viewport::{Viewport, ViewportState}, GraphicsPipelineCreateInfo
        }, layout::PipelineDescriptorSetLayoutCreateInfo, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo
    }, render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass}, shader::EntryPoint, swapchain::{
        acquire_next_image, CompositeAlpha, PresentMode, Surface, SurfaceCapabilities, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo
    }, sync::{self, GpuFuture}, Validated, Version, VulkanError, VulkanLibrary
};
use winit::{
    dpi::{LogicalSize, Size}, event::{Event, WindowEvent}, event_loop::{ControlFlow, EventLoop}, platform::run_return::EventLoopExtRunReturn, window::{Window, WindowBuilder}
};


// -------------------------------------------------- Structs


pub struct Vk<State: Reinforcement + Clone + Send + Sync> {
    event_loop: EventLoop<()>,
    inner: VkInner,
    painters: Vec<Box<dyn Painter<State>>>,
}
#[allow(unused)]
pub(crate) struct VkInner {
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

    surface_capabilities: SurfaceCapabilities,
    min_image_count: u32,
    image_format: Format,
    composite_alpha: CompositeAlpha,
    present_mode: PresentMode,

    msaa_image: Arc<ImageView>,
    depth_image: Arc<ImageView>,
    images: Vec<Arc<Image>>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,

    image_extent: [u32; 2],
    swap_redblue: bool,

    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

pub(crate) trait Painter<State: Reinforcement + Clone + Send + Sync> {
    /// On swapchain recreation, recreates the pipeline of this painter.
    fn recreate_pipeline(
        &mut self,
        render_pass: Arc<RenderPass>,
        image_extent: [u32; 2]
    );
    /// Executes arbitrary CPU operations before drawing.
    fn prepare_draw(
        &mut self,
        vkinner: &VkInner,
        agent: &mut Agent<State>,
        image_index: u32,
    );
    /// Informs this command buffer builder of the steps to take to paint.
    fn draw(
        &mut self,
        cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: u32
    );
}


// -------------------------------------------------- Vk


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
            .with_title("AI Agent 007 or so")
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
    
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let min_image_count = surface_capabilities.min_image_count.max(8);
        println!("Max image count is {:?}", surface_capabilities.max_image_count);
        println!("max_uniform_buffer_range is {:?}", device.physical_device().properties().max_uniform_buffer_range);
        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;
        println!("Image format: {:?}", image_format);
        let composite_alpha = surface_capabilities
            .supported_composite_alpha
            .into_iter()
            .next()
            .unwrap();
        let present_mode = State::draw_view_present_mode();

        // --------------------------------------------------

    
        let ((swapchain, images), msaa_image, depth_image) = (
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
            ).unwrap(),
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
            ImageView::new_default(
                Image::new(
                    memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::D16_UNORM,
                        extent: [window.inner_size().width, window.inner_size().width, 1],
                        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                        samples: SampleCount::Sample8,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            ).unwrap(),
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
                msaa_depth_stencil: {
                    format: Format::D16_UNORM,
                    samples: 8,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            pass: {
                color: [msaa_color],
                color_resolve: [color],
                depth_stencil: {msaa_depth_stencil},
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
                        attachments: vec![msaa_image.clone(), view, depth_image.clone()],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let image_extent = [images[0].extent()[0], images[0].extent()[1]];
        let swap_redblue = false;

        let recreate_swapchain = false;
        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        let vkinner = VkInner {
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

            surface_capabilities,
            min_image_count,
            image_format,
            composite_alpha,
            present_mode,

            msaa_image,
            depth_image,
            images,
            swapchain,
            render_pass,
            framebuffers,

            image_extent,
            swap_redblue,

            recreate_swapchain,
            previous_frame_end,
        };

        let painters: Vec<Box<dyn Painter<State>>> = vec![
            Box::new(
                Simulation::init::<State>(
                    &vkinner
                    // device.clone(),
                    // memory_allocator.clone(),
                    // &ds_allocator,
                    // render_pass.clone(),
                    // images[0].extent()
                )
            ),
        ];

        Vk {
            event_loop,
            inner: vkinner,
            painters,
        }
    }

    pub fn view_agent(&mut self, agent: &mut Agent<State>) {
        self.inner.view(&mut self.painters, &mut self.event_loop, agent);
    }

    pub fn save_agent(&mut self, agent: &mut Agent<State>, width: u32, height: u32, frames: u32, delta_t: f32) {
        self.inner.save(&mut self.painters, agent, width, height, frames, delta_t);
    }
}

impl VkInner {
    fn view<State: Reinforcement + Clone + Send + Sync>(&mut self, painters: &mut Vec<Box<dyn Painter<State>>>, event_loop: &mut EventLoop<()>, agent: &mut Agent<State>) {
        self.swap_redblue = false;

        agent.dac.reordered();
        let mut count = 0;
        let mut last_time = Instant::now();
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
                    self.recreate_swapchain(painters, image_extent);
                    self.previous_frame_end.as_mut().unwrap().cleanup_finished();

                    let delta_t = last_time.elapsed();
                    agent.evaluate_step(delta_t.as_secs_f32());
                    last_time += delta_t;

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
                    if image_index > MAX_UNIFORM_BUFFERS as u32 {
                        println!("\nTHERE ARE MORE SWAPCHAIN IMAGES THAN UNIFORM DESCRIPTORS\nimages_index: {}\n MAX_UNIFORM_BUFFERS: {}", image_index, MAX_UNIFORM_BUFFERS);
                    }

                    painters.iter_mut().for_each(|painter| {
                        painter.prepare_draw(self, agent, image_index);
                    });

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
                                    Some(1.0.into()),
                                ],
                                ..RenderPassBeginInfo::framebuffer(self.framebuffers[image_index as usize].clone())
                            },
                            Default::default(),
                        )
                        .unwrap();
                    painters.iter_mut().for_each(|painter| {
                        painter.draw(&mut cb_builder, image_index);
                    });
                    cb_builder
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
                            println!("failed to flush future: OUT OF DATE");
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

    fn save<State: Reinforcement + Clone + Send + Sync>(&mut self, painters: &mut Vec<Box<dyn Painter<State>>>, agent: &mut Agent<State>, width: u32, height: u32, frames: u32, delta_t: f32) {
        self.swap_redblue = true;

        self.recreate_swapchain = true;
        self.recreate_swapchain(painters, [width, height]);

        agent.dac.reordered();

        let save_buffer: Subbuffer<[u8]> = Buffer::new_slice(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (width * height * 4) as u64
        ).unwrap();

        for i in 0..frames {
            self.previous_frame_end.as_mut().unwrap().cleanup_finished();

            painters.iter_mut().for_each(|painter| {
                painter.prepare_draw(&self, agent, 0);
            });

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
                            Some(1.0.into()),
                        ],
                        ..RenderPassBeginInfo::framebuffer(self.framebuffers[0].clone())
                    },
                    Default::default(),
                )
                .unwrap();
            painters.iter_mut().for_each(|painter| {
                painter.draw(&mut cb_builder, 0);
            });
            cb_builder
                .end_render_pass(Default::default())
                .unwrap()

                .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                    self.images[0].clone(), 
                    save_buffer.clone()
                ))
                .unwrap();
            let command_buffer = cb_builder.build().unwrap();
                
            let future = self.previous_frame_end
                .take()
                .unwrap()
                .join(sync::now(self.device.clone()).boxed())
                .then_execute(self.queue.clone(), command_buffer)
                .unwrap()
                .then_signal_fence_and_flush();
                
            match future.map_err(Validated::unwrap) {
                Ok(future) => {
                    agent.evaluate_step(delta_t); // while GPU is executing
                    
                    future
                        .wait(None)
                        .unwrap();

                    // Export frame
                    {
                        let filename = format!("output/frame_{:010}.png", i);
        
                        let read_buffer_guard = save_buffer.read().unwrap();
                        let img: ImageBuffer<image::Rgba<u8>, &[u8]> = ImageBuffer::from_raw(width, height, read_buffer_guard.iter().as_slice()).unwrap();
                        img.save(filename).unwrap();
                    };
                    println!("Exported frame {}", i);

                    self.previous_frame_end = Some(future.boxed());
                }
                Err(VulkanError::OutOfDate) => {
                    println!("\n\n\n\n\nfailed to flush future: OUT OF DATE\n\n\n\n\n");
                    self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                }
                Err(e) => {
                    println!("\n\n\n\n\nfailed to flush future: {e}\n\n\n\n\n");
                    self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                }
            }
        }
    }

    fn recreate_swapchain<State: Reinforcement + Clone + Send + Sync>(&mut self, painters: &mut Vec<Box<dyn Painter<State>>>, image_extent: [u32; 2]) {
        if self.recreate_swapchain {
            self.recreate_swapchain = false;
            self.image_extent = image_extent;

            ((self.swapchain, self.images), self.msaa_image, self.depth_image) = {
                (
                    self.swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..self.swapchain.create_info()
                        })
                    .expect("failed to recreate swapchain"),
                    ImageView::new_default(
                        Image::new(
                            self.memory_allocator.clone(),
                            ImageCreateInfo {
                                image_type: ImageType::Dim2d,
                                format: self.image_format,
                                extent: [image_extent[0], image_extent[1], 1],
                                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                                samples: SampleCount::Sample8,
                                ..Default::default()
                            },
                            AllocationCreateInfo::default(),
                        )
                        .unwrap(),
                    ).unwrap(),
                    ImageView::new_default(
                        Image::new(
                            self.memory_allocator.clone(),
                            ImageCreateInfo {
                                image_type: ImageType::Dim2d,
                                format: Format::D16_UNORM,
                                extent: [image_extent[0], image_extent[1], 1],
                                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                                samples: SampleCount::Sample8,
                                ..Default::default()
                            },
                            AllocationCreateInfo::default(),
                        )
                        .unwrap(),
                    ).unwrap(),
                )
            };

            self.framebuffers = self.images
                .iter()
                .map(|image| {
                    let view = ImageView::new_default(image.clone()).unwrap();
                    Framebuffer::new(
                        self.render_pass.clone(),
                        FramebufferCreateInfo {
                            attachments: vec![self.msaa_image.clone(), view, self.depth_image.clone()],
                            ..Default::default()
                        },
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();

            painters.iter_mut().for_each(|painter| {
                painter.recreate_pipeline(self.render_pass.clone(), image_extent);
            });
        }
    }
}