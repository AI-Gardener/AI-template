//! No docs :D (yet)

mod dac;
use dac::*;
mod functions;
use functions::*;
mod node;
use node::*;

use once_cell::sync::OnceCell;
use opengl_graphics::GlGraphics;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use piston::RenderArgs;


pub static AGENTS_NUM: OnceCell<usize> = OnceCell::new();
pub static NUM_GENERATIONS: OnceCell<usize> = OnceCell::new();
pub static TICKS_PER_EVALUATION: OnceCell<usize> = OnceCell::new(); // 10 seconds
pub static TICK_DURATION: OnceCell<f32> = OnceCell::new(); // 60 modifications per second
pub static START_NODES: OnceCell<Vec<Box<Node>>> = OnceCell::new();

/// While training, the implemented methods will (repeatedly) be called in the order:
/// ```
/// state.set_inputs(&mut dac);
/// // Here, run network.
/// state.get_outputs(&dac);
/// state.update_physics();
/// state.update_score();
/// ```
pub trait Reinforcement {
    /// Default values when creating an inctance of this.
    fn init() -> Self;
    /// Reflects state values to the network inputs.
    fn set_inputs(&self, dac: &mut DAC);
    /// Reflects the network outputs to state values.
    fn get_outputs(&mut self, dac: &DAC);
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
    /// Updates the state values after getting the output.
    fn update_physics(&mut self);
    /// Updates the score, reflecting how well the AI is doing at that instant.
    fn update_score(&mut self);
    /// Draws the current scene
    fn render(&self, gl: &mut GlGraphics, args: &RenderArgs);
}

/// Top-level of the training process 
pub struct AI<State: Reinforcement + Clone + Send + Sync> {
    agents: Vec<Agent<State>>,
    past_agents: Vec<Vec<Agent<State>>>,
}

#[derive(Clone)]
struct Agent<State: Reinforcement + Clone + Send + Sync> {
    dac: DAC,
    score: f32,
    instant: f32,

    state: State
}

impl<State: Reinforcement + Clone + Send + Sync> AI<State> {
    pub fn init(
        agents_num: usize,
        num_generations: usize,
        ticks_per_evaluation: usize,
        tick_duration: f32,
        start_nodes: Vec<Box<Node>>,
    ) -> Self {
        AGENTS_NUM.set(agents_num).unwrap();
        NUM_GENERATIONS.set(num_generations).unwrap();
        TICKS_PER_EVALUATION.set(ticks_per_evaluation).unwrap();
        TICK_DURATION.set(tick_duration).unwrap();
        START_NODES.set(start_nodes).unwrap();
        Self {
            agents: (0..*AGENTS_NUM.get().unwrap()).map(|_| Agent::new()).collect(),
            past_agents: Vec::new(),
        }
    }

    pub fn train(&mut self) {
        (0..*NUM_GENERATIONS.get().unwrap()).into_iter().for_each(|gen| {
            println!("Testing generation {}...", gen);
            self.evaluate();
            self.sort();
            self.next_gen();
        });
        self.evaluate();
        self.sort();
    }

    pub fn render_best(&self) {
        extern crate glutin_window;
        extern crate graphics;
        extern crate opengl_graphics;
        extern crate piston;
        
        use glutin_window::GlutinWindow as Window;
        use opengl_graphics::{GlGraphics, OpenGL};
        use piston::event_loop::{EventSettings, Events};
        use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
        use piston::window::WindowSettings;
        
        pub struct View {
            gl: GlGraphics, // OpenGL drawing backend.
        }
        
        impl View {
            fn render<InnerState: Reinforcement + Clone + Send + Sync>(&mut self, args: &RenderArgs, agent: &Agent<InnerState>) {
                agent.state.render(&mut self.gl, args);
            }
        
            fn update(&mut self, _args: &UpdateArgs) {
        
            }
        }


        let mut best_agent = self.agents
            .iter().nth(0).unwrap()
            .clone();
        println!("Best DAC score: {}.\nDisplaying the evaluation...", best_agent.score);
        println!("Best DAC network: {:?}", best_agent.dac);
        best_agent.score = 0.1;
        best_agent.instant = 0.0;


        // Running the visual simulation.
        let opengl = OpenGL::V3_2;
        let mut window: Window = WindowSettings::new("Pendulum Simulation", [1440, 900])
            .graphics_api(opengl)
            .fullscreen(false)
            .exit_on_esc(true)
            .samples(8)
            .build()
            .unwrap();
        let mut view = View {
            gl: GlGraphics::new(opengl),
        };
        let mut events = Events::new(EventSettings {
                max_fps: 60,
                ups: 60,
                swap_buffers: true,
                bench_mode: false,
                lazy: false,
                ups_reset: 2,
            });
        let mut frame_count: usize = 0;
        while let Some(e) = events.next(&mut window) {
            if frame_count > *TICKS_PER_EVALUATION.get().unwrap() {
                break;
            }
            if let Some(args) = e.render_args() {
                best_agent.evaluate_step();
                view.render(&args, &best_agent);
            }
    
            if let Some(args) = e.update_args() {
                frame_count += 1;
                view.update(&args);
            }
        }
    }

    fn evaluate(&mut self) {
        (&mut self.agents).par_iter_mut().for_each(|agent| {
            agent.evaluate();
        })
    }

    fn sort(&mut self) {
        self.agents.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap()); // Bigger scores come before
    }

    fn next_gen(&mut self) {
        let mut cumulative_sum: Vec<f32> = Vec::with_capacity(*AGENTS_NUM.get().unwrap());
        let mut total_score = 0.0;
        self.agents.iter()
            .map(|agent| agent.score)
            .for_each(|score| {
                total_score+=score;
                cumulative_sum.push(total_score);
            });
        if total_score == 0.0 { total_score = 0.001 }
        cumulative_sum.iter_mut()
            .for_each(|score| {
                *score/=total_score;
            });

        let mut new_agents = self.agents[0..*AGENTS_NUM.get().unwrap()/3].to_vec();
        while new_agents.len() < *AGENTS_NUM.get().unwrap() {
            let random = fastrand::f32();
            new_agents.push(self.agents[
                cumulative_sum.iter().position(|&cumulative_score| cumulative_score > random).unwrap()
            ].mutated());
        }

        std::mem::swap(&mut self.agents, &mut new_agents);
        self.past_agents.push(new_agents); // `new_agents` are the old ones, as we just swapped them.
    }
}

impl<State: Reinforcement + Clone + Send + Sync> Agent<State> {
    fn new() -> Self {
        Self {
            dac: DAC::new((*START_NODES.get().unwrap()).clone()),
            score: 0.1,
            instant: 0.0,

            state: State::init(),
        }
    }

    fn mutated(&self) -> Self {
        Self {
            dac: State::mutated(&self.dac),
            score: 0.1,
            instant: 0.0,

            state: State::init(),
        }
    }

    fn evaluate(&mut self) {
        // Run network, Update physics values (cart speed, cart pos, pendulum physics data from new position), then run again, for a number of ticks and a given tick duration.
        self.score = 0.1;
        self.dac.reordered();
        for _ in 0..*TICKS_PER_EVALUATION.get().unwrap() {
            self.evaluate_step();
        }
    }

    fn evaluate_step(&mut self) {
        self.state.set_inputs(&mut self.dac);

        self.dac.run();

        self.state.get_outputs(&self.dac);
        self.state.update_physics();
        self.state.update_score();

        self.instant += TICK_DURATION.get().unwrap();
    }
}