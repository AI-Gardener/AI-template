/// A neuron
#[derive(Clone, Debug)]
pub struct Node {
    pub val: f32,
    pub(crate) bias: f32,
    pub(crate) activation_f: fn(f32) -> f32,
    /// Parents indices
    pub(crate) parents: Vec<usize>,
    /// Children (indices, connections weights)
    pub(crate) children: Vec<(usize, f32)>,
    /// The layer of this node
    pub(crate) layer: u32,
}

impl Node {
    /// A new input node.
    pub fn new(val: f32, bias: f32, activation_f: fn(f32) -> f32, parents: Vec<usize>, children: Vec<(usize, f32)>, layer: u32) -> Self {
        Self {
            val,
            bias,
            activation_f,
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