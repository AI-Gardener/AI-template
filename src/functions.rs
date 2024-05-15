/// The identity function
pub fn identity(input: f32) -> f32 {
    input
}

/// The hyperbolic tangent function
pub fn tanh(input: f32) -> f32 {
    input.tanh()
}

/// The REctified Linear Unit function
pub fn relu(input: f32) -> f32 {
    input.max(0.0)
}