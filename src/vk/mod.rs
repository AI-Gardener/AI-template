use super::*;

mod matrix;
pub use matrix::*;

use image::ImageBuffer;

use std::{hash::{Hash, Hasher}, sync::Arc, time::Instant};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyImageToBufferInfo, RenderPassBeginInfo
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

// Always-supported uniform range: 16384 bytes. [From here](https://docs.vulkan.org/spec/latest/chapters/limits.html)
// 1 Mat4: 64 bytes
// 8 images for the swapchain should be enough, leaving 32 matrices per uniform.
// 64 bytes per matrix * 32 matrices per uniform * 8 uniforms total = 16384 bytes
const MAX_MATRICES: usize = 32;
const MAX_UNIFORM_BUFFERS: usize = 8;


// -------------------------------------------------- Structs


pub struct Vk  {
    event_loop: EventLoop<()>,
    inner: VkInner,
}
#[allow(unused)]
struct VkInner  {
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
    depth_image: Arc<ImageView>,
    images: Vec<Arc<Image>>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    drawing_pipeline: Arc<GraphicsPipeline>,
    drawing_descriptor_sets: Vec<Arc<PersistentDescriptorSet>>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,

    vertices: Vec<InputVertex>,
    vertex_buffer: Subbuffer<[InputVertex]>,
    indexed: bool,
    indices: Vec<u32>,
    index_buffer: Subbuffer<[u32]>,
    transformations: [Mat4; MAX_MATRICES],
    staging_transform_uniform: Subbuffer<[Mat4]>,
    transform_uniforms: Vec<Subbuffer<[Mat4]>>,
}

/// A single vertex, part of a triangle that will be drawn
#[derive(BufferContents, Vertex, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct InputVertex {
    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32_SINT)]
    transform_id: i32,
}

impl Eq for InputVertex {}

impl Hash for InputVertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.color[3].to_bits().hash(state);
        self.position[0].to_bits().hash(state);
        self.position[1].to_bits().hash(state);
        self.position[2].to_bits().hash(state);
        self.transform_id.hash(state);
    }
}

impl InputVertex {
    /// `transform_id`: if this is negative, the vertex won't be transformed.
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
    pub(crate) resolution: [u32; 2],
    pub(crate) time: f32,
    pub(crate) hw_ratio: f32,
    /// 0 is window rendering, asking not to swap channels, and 1 is video saving, asking to convert bgra to rgba.
    pub(crate) swap_redblue: u32,
}


// -------------------------------------------------- SHADERS


mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(set = 0, binding = 0) uniform ModelTransformations {
                mat4 data[32];
            } tf;
            layout(push_constant) uniform GeneralInfo {
                uvec2 resolution;
                float time;
                float hw_ratio;
                uint swap_redblue;
            } gen;

            layout(location = 0) in vec4 color;
            layout(location = 1) in vec3 position;
            layout(location = 2) in int transform_id;

            layout(location = 0) out vec4 v_color;
            layout(location = 1) out uint v_transform_id;

            void main() {
                // Vertex
                gl_Position = vec4(position, 1.0);
                // Model
                if (transform_id > -1) {
                    gl_Position *= tf.data[transform_id];
                }
                // View
                gl_Position.x *= gen.hw_ratio;

                v_color = color;
                v_transform_id = transform_id;
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(push_constant) uniform GeneralInfo {
                uvec2 resolution;
                float time;
                float hw_ratio;
                uint swap_redblue;
            } gen;

            layout(location = 0) in vec4 v_color;
            layout(location = 1) in flat uint v_transform_id;

            layout(location = 0) out vec4 f_color;


            ////////////////////////////////////////////////////////
            // BACKGROUND -- Cyber Fuji 2020 Shader art by kaiware007 on Shadertoy: https://www.shadertoy.com/view/Wt33Wf
            ////////////////////////////////////////////////////////
            float sun(vec2 uv, float battery)
            {
                 float val = smoothstep(0.3, 0.29, length(uv));
                 float bloom = smoothstep(0.7, 0.0, length(uv));
                float cut = 3.0 * sin((uv.y + gen.time * 0.2 * (battery + 0.02)) * 100.0) 
                            + clamp(uv.y * 14.0 + 1.0, -6.0, 6.0);
                cut = clamp(cut, 0.0, 1.0);
                return clamp(val * cut, 0.0, 1.0) + bloom * 0.6;
            }
            
            float grid(vec2 uv, float battery)
            {
                vec2 size = vec2(uv.y, uv.y * uv.y * 0.2) * 0.01;
                uv += vec2(0.0, gen.time * 4.0 * (battery + 0.05));
                uv = abs(fract(uv) - 0.5);
                 vec2 lines = smoothstep(size, vec2(0.0), uv);
                 lines += smoothstep(size * 5.0, vec2(0.0), uv) * 0.4 * battery;
                return clamp(lines.x + lines.y, 0.0, 3.0);
            }
            
            float dot2(in vec2 v ) { return dot(v,v); }
            
            float sdTrapezoid( in vec2 p, in float r1, float r2, float he )
            {
                vec2 k1 = vec2(r2,he);
                vec2 k2 = vec2(r2-r1,2.0*he);
                p.x = abs(p.x);
                vec2 ca = vec2(p.x-min(p.x,(p.y<0.0)?r1:r2), abs(p.y)-he);
                vec2 cb = p - k1 + k2*clamp( dot(k1-p,k2)/dot2(k2), 0.0, 1.0 );
                float s = (cb.x<0.0 && ca.y<0.0) ? -1.0 : 1.0;
                return s*sqrt( min(dot2(ca),dot2(cb)) );
            }
            
            float sdLine( in vec2 p, in vec2 a, in vec2 b )
            {
                vec2 pa = p-a, ba = b-a;
                float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
                return length( pa - ba*h );
            }
            
            float sdBox( in vec2 p, in vec2 b )
            {
                vec2 d = abs(p)-b;
                return length(max(d,vec2(0))) + min(max(d.x,d.y),0.0);
            }
            
            float opSmoothUnion(float d1, float d2, float k){
                float h = clamp(0.5 + 0.5 * (d2 - d1) /k,0.0,1.0);
                return mix(d2, d1 , h) - k * h * ( 1.0 - h);
            }
            
            float sdCloud(in vec2 p, in vec2 a1, in vec2 b1, in vec2 a2, in vec2 b2, float w)
            {
                //float lineVal1 = smoothstep(w - 0.0001, w, sdLine(p, a1, b1));
                float lineVal1 = sdLine(p, a1, b1);
                float lineVal2 = sdLine(p, a2, b2);
                vec2 ww = vec2(w*1.5, 0.0);
                vec2 left = max(a1 + ww, a2 + ww);
                vec2 right = min(b1 - ww, b2 - ww);
                vec2 boxCenter = (left + right) * 0.5;
                //float boxW = right.x - left.x;
                float boxH = abs(a2.y - a1.y) * 0.5;
                //float boxVal = sdBox(p - boxCenter, vec2(boxW, boxH)) + w;
                float boxVal = sdBox(p - boxCenter, vec2(0.04, boxH)) + w;
                
                float uniVal1 = opSmoothUnion(lineVal1, boxVal, 0.05);
                float uniVal2 = opSmoothUnion(lineVal2, boxVal, 0.05);
                
                return min(uniVal1, uniVal2);
            }
            
            vec4 background()
            {
                vec4 fragCoord = gl_FragCoord;
                vec2 uv = -(2.0 * fragCoord.xy - gen.resolution.xy)/gen.resolution.y;
                float battery = 1.0;
                
                {
                    // Grid
                    float fog = smoothstep(0.1, -0.02, abs(uv.y + 0.2));
                    vec3 col = vec3(0.0, 0.1, 0.2);
                    if (uv.y < -0.2)
                    {
                        uv.y = 3.0 / (abs(uv.y + 0.2) + 0.05);
                        uv.x *= uv.y * 1.0;
                        float gridVal = grid(uv, battery);
                        col = mix(col, vec3(1.0, 0.5, 1.0), gridVal);
                    }
                    else
                    {
                        float fujiD = min(uv.y * 4.5 - 0.5, 1.0);
                        uv.y -= battery * 1.1 - 0.51;
                        
                        vec2 sunUV = uv;
                        vec2 fujiUV = uv;
                        
                        // Sun
                        sunUV += vec2(0.75, 0.2);
                        //uv.y -= 1.1 - 0.51;
                        col = vec3(1.0, 0.2, 1.0);
                        float sunVal = sun(sunUV, battery);
                        
                        col = mix(col, vec3(1.0, 0.4, 0.1), sunUV.y * 2.0 + 0.2);
                        col = mix(vec3(0.0, 0.0, 0.0), col, sunVal);
                        
                        // fuji
                        float fujiVal = sdTrapezoid( uv  + vec2(-0.75+sunUV.y * 0.0, 0.5), 1.75 + pow(uv.y * uv.y, 2.1), 0.2, 0.5);
                        float waveVal = uv.y + sin(uv.x * 20.0 + gen.time * 2.0) * 0.05 + 0.2;
                        float wave_width = smoothstep(0.0,0.01,(waveVal));
                        
                        // fuji color
                        col = mix( col, mix(vec3(0.0, 0.0, 0.25), vec3(1.0, 0.0, 0.5), fujiD), step(fujiVal, 0.0));
                        // fuji top snow
                        col = mix( col, vec3(1.0, 0.5, 1.0), wave_width * step(fujiVal, 0.0));
                        // fuji outline
                        col = mix( col, vec3(1.0, 0.5, 1.0), 1.0-smoothstep(0.0,0.01,abs(fujiVal)) );
                        //col = mix( col, vec3(1.0, 1.0, 1.0), 1.0-smoothstep(0.03,0.04,abs(fujiVal)) );
                        //col = vec3(1.0, 1.0, 1.0) *(1.0-smoothstep(0.03,0.04,abs(fujiVal)));
                        
                        // horizon color
                        col += mix( col, mix(vec3(1.0, 0.12, 0.8), vec3(0.0, 0.0, 0.2), clamp(uv.y * 3.5 + 3.0, 0.0, 1.0)), step(0.0, fujiVal) );
                        
                        // cloud
                        vec2 cloudUV = uv;
                        cloudUV.x = mod(cloudUV.x + gen.time * 0.1, 4.0) - 2.0;
                        float cloudTime = gen.time * 0.5;
                        float cloudY = -0.5;
                        float cloudVal1 = sdCloud(cloudUV, 
                                                 vec2(0.1 + sin(cloudTime + 140.5)*0.1,cloudY), 
                                                 vec2(1.05 + cos(cloudTime * 0.9 - 36.56) * 0.1, cloudY), 
                                                 vec2(0.2 + cos(cloudTime * 0.867 + 387.165) * 0.1,0.25+cloudY), 
                                                 vec2(0.5 + cos(cloudTime * 0.9675 - 15.162) * 0.09, 0.25+cloudY), 0.075);
                        cloudY = -0.6;
                        float cloudVal2 = sdCloud(cloudUV, 
                                                 vec2(-0.9 + cos(cloudTime * 1.02 + 541.75) * 0.1,cloudY), 
                                                 vec2(-0.5 + sin(cloudTime * 0.9 - 316.56) * 0.1, cloudY), 
                                                 vec2(-1.5 + cos(cloudTime * 0.867 + 37.165) * 0.1,0.25+cloudY), 
                                                 vec2(-0.6 + sin(cloudTime * 0.9675 + 665.162) * 0.09, 0.25+cloudY), 0.075);
                        
                        float cloudVal = min(cloudVal1, cloudVal2);
                        
                        //col = mix(col, vec3(1.0,1.0,0.0), smoothstep(0.0751, 0.075, cloudVal));
                        col = mix(col, vec3(0.0, 0.0, 0.2), 1.0 - smoothstep(0.075 - 0.0001, 0.075, cloudVal));
                        col += vec3(1.0, 1.0, 1.0)*(1.0 - smoothstep(0.0,0.01,abs(cloudVal - 0.075)));
                    }
            
                    col += fog * fog * fog;
                    col = mix(vec3(col.r, col.r, col.r) * 0.5, col, battery * 0.7);
            
                    return vec4(col,1.0);
                }
            }
            ////////////////////////////////////////////////////////
            // BACKGROUND -- End of this Shader Artwork
            ////////////////////////////////////////////////////////



            void main() {
                f_color = v_color;
                if (gen.swap_redblue == 0 && v_transform_id == 0) {
                    f_color = background();
                }
                
                if (gen.swap_redblue==1) {
                    float blue = f_color.z;
                    f_color.z = f_color.x;
                    f_color.x = blue;
                }
            }
        ",
    }
}


// -------------------------------------------------- Vk


impl Vk {
    pub fn init<State: Reinforcement + Clone + Send + Sync>() -> Self {
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
        let min_image_count = surface_capabilities.min_image_count.max(8);
        println!("Max image count is {:?}", surface_capabilities.max_image_count);
        println!("max_uniform_buffer_range is {:?}", device.physical_device().properties().max_uniform_buffer_range);
        println!("max_descriptor_set_uniform_buffers is {:?}", device.physical_device().properties().max_descriptor_set_uniform_buffers);
        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;
        println!("{:?}", image_format);
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
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };
        let recreate_swapchain = false;
        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        // --------------------------------------------------
    
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
                depth_image,
                images,
                swapchain,
                render_pass,
                framebuffers,
                drawing_pipeline,
                drawing_descriptor_sets,
                recreate_swapchain,
                previous_frame_end,

                vertices,
                vertex_buffer,
                indexed,
                indices,
                index_buffer,
                transformations,
                staging_transform_uniform,
                transform_uniforms,
            },
        }
    }

    pub fn view_agent<State: Reinforcement + Clone + Send + Sync>(&mut self, agent: Agent<State>) {
        self.inner.view(&mut self.event_loop, agent);
    }

    pub fn save_agent<State: Reinforcement + Clone + Send + Sync>(&mut self, agent: Agent<State>, width: u32, height: u32, frames: u32, delta_t: f32) {
        self.inner.save(agent, width, height, frames, delta_t);
    }
}

impl VkInner {
    fn view<State: Reinforcement + Clone + Send + Sync>(&mut self, event_loop: &mut EventLoop<()>, mut agent: Agent<State>) {
        agent.dac.reordered();
        agent.state.draw_transformations(&mut self.transformations);
        let mut count = 0;
        // let start_time = Instant::now();
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
                    self.recreate_swapchain(image_extent);
                    self.previous_frame_end.as_mut().unwrap().cleanup_finished();

                    let delta_t = last_time.elapsed();
                    agent.evaluate_step(delta_t.as_secs_f32());
                    last_time += delta_t;
                    agent.state.draw_transformations(&mut self.transformations);

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

                    // TODO Less buffers? Less desciptors? Less descriptor updates?
                    // let transform_uniform_offset = MAX_MATRICES as u32 * image_index;
                    {
                        let mut write_guard = self.staging_transform_uniform.write().unwrap();
            
                        for (o, i) in write_guard.iter_mut().zip(self.transformations.iter()) {
                            *o = *i;
                        }
                    }
                    let push = Push {
                        resolution: [self.window.inner_size().width, self.window.inner_size().height],
                        time: agent.instant,
                        hw_ratio: self.window.inner_size().height as f32 / self.window.inner_size().width as f32,
                        swap_redblue: 0,
                    };

                    let mut uniform_copy_cb_builder = AutoCommandBufferBuilder::primary(
                        &self.cb_allocator,
                        self.queue.queue_family_index(),
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
                                    Some(1.0.into()),
                                ],
                                ..RenderPassBeginInfo::framebuffer(self.framebuffers[image_index as usize].clone())
                            },
                            Default::default(),
                        )
                        .unwrap()
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
                    }
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

    fn save<State: Reinforcement + Clone + Send + Sync>(&mut self, mut agent: Agent<State>, width: u32, height: u32, frames: u32, delta_t: f32) {
        self.recreate_swapchain = true;
        self.recreate_swapchain([width, height]);

        agent.dac.reordered();
        agent.state.draw_transformations(&mut self.transformations);


        {
            let mut write_guard = self.staging_transform_uniform.write().unwrap();

            for (o, i) in write_guard.iter_mut().zip(self.transformations.iter()) {
                *o = *i;
            }
        }

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
        let push = Push {
            // Unused for saving
            resolution: [self.window.inner_size().width, self.window.inner_size().height],
            // Unused for saving
            time: agent.instant,
            hw_ratio: height as f32 / width as f32,
            swap_redblue: 1,
        };

        for i in 0..frames {
            // ----- Draw frame

            let mut cb_builder = AutoCommandBufferBuilder::primary(
                &self.cb_allocator,
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            cb_builder
                .copy_buffer(CopyBufferInfo::buffers(
                        self.staging_transform_uniform.clone().into_bytes(),
                        self.transform_uniforms[0].clone().into_bytes(),
                    )
                )
                .unwrap()

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
                .unwrap()
                .bind_pipeline_graphics(self.drawing_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.drawing_pipeline.layout().clone(),
                    0,
                    self.drawing_descriptor_sets[0].clone(),
                )
                .unwrap()
                .push_constants(
                    self.drawing_pipeline.layout().clone(),
                    0,
                    push.clone()
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
            }
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


                    // ----- Update for next frame

                    agent.evaluate_step(delta_t);
                    agent.state.draw_transformations(&mut self.transformations);

                    // ----- Finish GPU execution
                    
                    future
                        .wait(None)
                        .unwrap();

                    // ----- Post-Execution updates

                    {
                        let mut write_guard = self.staging_transform_uniform.write().unwrap();
            
                        for (o, i) in write_guard.iter_mut().zip(self.transformations.iter()) {
                            *o = *i;
                        }
                    }
                    
                    // ----- Export frame

                    {
                        let filename = format!("output/frame_{:010}.png", i);
        
                        let read_buffer_guard = save_buffer.read().unwrap();
                        let img: ImageBuffer<image::Rgba<u8>, &[u8]> = ImageBuffer::from_raw(width, height, read_buffer_guard.iter().as_slice()).unwrap();
                        img.save(filename).unwrap();
                    };
                    println!("Exported frame {}", i);

                    // ----- ---

                    self.previous_frame_end = Some(future.boxed());

                    self.previous_frame_end.as_mut().unwrap().cleanup_finished();


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

    fn recreate_swapchain(&mut self, image_extent: [u32; 2]) {
        if self.recreate_swapchain {
            self.recreate_swapchain = false;

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
    }
}