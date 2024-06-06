use super::*;

mod functions;
pub use functions::*;
mod serialize;
pub use serialize::*;

/// A neuron
#[derive(Clone, Debug)]
pub struct Node {
    pub val: f32,
    pub(crate) bias: f32,
    pub(crate) activation_f: fn(f32) -> f32,
    pub(crate) activation_function: ActivationFunction,
    /// Parents indices
    pub(crate) parents: Vec<usize>,
    /// Children (indices, connections weights)
    pub(crate) children: Vec<(usize, f32)>,
    /// The layer of this node
    pub(crate) layer: u32,
}

impl Node {
    /// A new input node.
    pub fn new(
        bias: f32,
        activation_function: ActivationFunction,
        parents: Vec<usize>,
        children: Vec<(usize, f32)>,
        layer: u32,
    ) -> Self {
        Self {
            val: 0.0,
            bias,
            activation_f: activation_function.function(),
            activation_function,
            parents,
            children,
            layer,
        }
    }

    /// Updates `val` from `bias` and `activation_f`.
    pub(crate) fn update(&mut self) {
        self.val = (self.activation_f)(self.val + self.bias);
        // println!("{}", self.val);
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ActivationFunction {
    Custom1 = 1,
    Custom2 = 2,
    Custom3 = 3,
    Custom4 = 4,
    Custom5 = 5,
    Identity = 6,
    Tanh = 7,
    Relu = 8,
}

impl ActivationFunction {
    pub(crate) fn from_id(id: u32) -> Self {
        #[allow(unreachable_patterns)]
        match id {
            1 => Self::Custom1,
            2 => Self::Custom2,
            3 => Self::Custom3,
            4 => Self::Custom4,
            5 => Self::Custom5,
            6 => Self::Identity,
            7 => Self::Tanh,
            8 => Self::Relu,
            _ => {
                println!("Unknown function, using identity.");
                Self::Identity
            },
        }
    }

    pub(crate) fn function(&self) -> fn(f32) -> f32 {
        #[allow(unreachable_patterns)]
        match self {
            Self::Custom1 => *CUSTOM_ACTIVATION_F1.get().unwrap(),
            Self::Custom2 => *CUSTOM_ACTIVATION_F2.get().unwrap(),
            Self::Custom3 => *CUSTOM_ACTIVATION_F3.get().unwrap(),
            Self::Custom4 => *CUSTOM_ACTIVATION_F4.get().unwrap(),
            Self::Custom5 => *CUSTOM_ACTIVATION_F5.get().unwrap(),
            Self::Identity => functions::identity,
            Self::Tanh => functions::tanh,
            Self::Relu => functions::relu,
            _ => {
                println!("Unknown function, using identity.");
                functions::identity
            },
        }
    }
}