use std::hash::{Hash, Hasher};
use vulkano_macros::{BufferContents, Vertex};

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