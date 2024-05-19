//! Internally, from bigger to smaller:
//! pub AI -> Agent -> pub DAC -> pub Node
//! 
//! This top-level file contains the crate interface, imports and reexports.

// TODO everything private, like Agent. Except maybe AI.

mod ai;
pub use ai::*;
mod agent;
pub(crate) use agent::*;
mod dac;
pub use dac::*;
mod node;
pub use node::*;
mod functions;
pub use functions::*;

use once_cell::sync::OnceCell;
use opengl_graphics::GlGraphics;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use piston::RenderArgs;


pub(crate) static AGENTS_NUM: OnceCell<usize> = OnceCell::new();
pub(crate) static NUM_GENERATIONS: OnceCell<usize> = OnceCell::new();
pub(crate) static TICKS_PER_EVALUATION: OnceCell<usize> = OnceCell::new(); // 10 seconds
pub(crate) static TICK_DURATION: OnceCell<f32> = OnceCell::new(); // 60 modifications per second
pub(crate) static START_NODES: OnceCell<Vec<Box<Node>>> = OnceCell::new();
pub(crate) static NEWNODE_ACTIVATION_F: OnceCell<fn(f32) -> f32> = OnceCell::new();

/// This trait must be implemented for your state struct (a struct with fields used for the training).  
/// While training, the implemented methods will (repeatedly) be called in the order:  
/// ```
/// state.set_inputs(&mut dac);
/// // Here, run network.
/// state.get_outputs(&dac);
/// state.update_physics();
/// state.update_score();
/// ```
pub trait Reinforcement {
    /// The number of agents per generation.
    /// 
    /// This variable is read once when initializing AI. Modifying it after that will have no inner effect.
    fn num_agents() -> usize { 300 }
    /// The number of generations trained.
    /// 
    /// This variable is read once when initializing AI. Modifying it after that will have no inner effect.
    fn num_generations() -> usize { 600 }
    /// The number of ticks a simulation lasts.
    /// 
    /// This variable is read once when initializing AI. Modifying it after that will have no inner effect.
    fn ticks_per_evaluation() -> usize { 60 * 10 }
    /// The duration of a tick, and the difference in time between 2 consecutive updates, Delta time.
    /// 
    /// This variable is read once when initializing AI. Modifying it after that will have no inner effect.
    fn tick_duration() -> f32 { 1.0 / 60.0 }
    /// The starting nodes of the graph network. Most likely contains all the input and outputs.
    /// 
    /// This variable is read once when initializing AI. Modifying it after that will have no inner effect.
    fn start_nodes() -> Vec<Box<Node>>;
    /// The activation function used by new hidden nodes.
    fn newnode_activation_f() -> fn(f32) -> f32;
    /// Default values when creating an inctance of this.
    fn init() -> Self;
    /// Reflects state values to the network inputs.
    fn set_inputs(&self, dac: &mut DAC);
    /// Reflects the network outputs to state values.
    fn get_outputs(&mut self, dac: &DAC);
    /// Updates the state values after getting the output.
    fn update_physics(&mut self);
    /// Updates the score, reflecting how well the AI is doing at that instant.  
    /// The score should not end up negative.
    fn update_score(&mut self, score: &mut f32);
    /// Get a mutated version of this network.  
    /// 
    /// Default implementation:  
    /// ```
    /// let f = fastrand::f32();
    /// let has_linked = dac.has_linked();
    /// 
    /// if f < 0.10 && has_linked {
    ///     dac.newnode_mutated()
    /// } else if f < 0.35 && has_linked {
    ///     dac.weight_mutated()
    /// } else if f < 0.60 && has_linked {
    ///     dac.bias_mutated()
    /// } else if f < 0.80 && dac.has_unlinked() {
    ///     dac.newconnection_mutated()
    /// } else {
    ///     dac.unmutated()
    /// }
    /// ```
    fn mutated(dac: &DAC) -> DAC {
        let f = fastrand::f32();
        let has_linked = dac.has_linked();

        if f < 0.10 && has_linked {
            dac.newnode_mutated()
        } else if f < 0.35 && has_linked {
            dac.weight_mutated()
        } else if f < 0.60 && has_linked {
            dac.bias_mutated()
        } else if f < 0.80 && dac.has_unlinked() {
            dac.newconnection_mutated()
        } else {
            dac.unmutated()
        }
    }
    /// Draws the current scene
    fn render(&self, gl: &mut GlGraphics, args: &RenderArgs);
}