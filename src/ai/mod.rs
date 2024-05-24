use super::*;

mod agent;
#[allow(unused_imports)]
pub use agent::*;
mod dac;
pub use dac::*;
mod node;
pub use node::*;
mod functions;
pub use functions::*;

/// Top-level of the training process
pub struct AI<State: Reinforcement + Clone + Send + Sync> {
    agents: Vec<Agent<State>>,
    past_agents: Vec<Vec<Agent<State>>>,
}

impl<State: Reinforcement + Clone + Send + Sync> AI<State> {
    /// Initializes the AI.
    pub fn init() -> Self {
        AGENTS_NUM.set(State::num_agents()).unwrap();
        NUM_GENERATIONS.set(State::num_generations()).unwrap();
        TICKS_PER_EVALUATION
            .set(State::ticks_per_evaluation())
            .unwrap();
        TICK_DURATION.set(State::tick_duration()).unwrap();
        START_NODES.set(State::start_nodes()).unwrap();
        NEWNODE_ACTIVATION_F
            .set(State::newnode_activation_f())
            .unwrap();
        Self {
            agents: (0..*AGENTS_NUM.get().unwrap())
                .map(|_| Agent::new())
                .collect(),
            past_agents: Vec::new(),
        }
    }

    /// Trains the AI
    pub fn train(&mut self) {
        (1..*NUM_GENERATIONS.get().unwrap())
            .into_iter()
            .for_each(|gen| {
                println!("Testing generation {}...", gen);
                self.evaluate();
                self.sort();
                self.next_gen();
            });
        println!("Testing generation {}...", *NUM_GENERATIONS.get().unwrap());
        self.evaluate();
        self.sort();
    }

    // pub fn render_best(&self) {
    //     extern crate glutin_window;
    //     extern crate graphics;
    //     extern crate opengl_graphics;
    //     extern crate piston;

    //     use glutin_window::GlutinWindow as Window;
    //     use opengl_graphics::{GlGraphics, OpenGL};
    //     use piston::event_loop::{EventSettings, Events};
    //     use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
    //     use piston::window::WindowSettings;

    //     pub struct View {
    //         gl: GlGraphics, // OpenGL drawing backend.
    //     }

    //     impl View {
    //         fn render<InnerState: Reinforcement + Clone + Send + Sync>(&mut self, args: &RenderArgs, agent: &Agent<InnerState>) {
    //             agent.state.render(&mut self.gl, args);
    //         }

    //         fn update(&mut self, _args: &UpdateArgs) {

    //         }
    //     }

    //     let best_agent = self.agents
    //         .iter().nth(0).unwrap()
    //         .clone();
    //     println!("Best DAC score: {}.\nDisplaying the evaluation...", best_agent.score);
    //     println!("Best DAC network: {:?}", best_agent.dac);
    //     let mut best_agent_clone: Agent<State> = Agent {
    //         dac: best_agent.dac,
    //         ..Agent::new()
    //     };

    //     // Running the visual simulation.
    //     let opengl = OpenGL::V3_2;
    //     let mut window: Window = WindowSettings::new("AI Agent Test", [1920, 1080])
    //         .graphics_api(opengl)
    //         .exit_on_esc(true)
    //         .samples(8)
    //         .build()
    //         .unwrap();
    //     let mut view = View {
    //         gl: GlGraphics::new(opengl),
    //     };
    //     let mut events = Events::new(EventSettings {
    //             max_fps: (1.0 / State::tick_duration()) as u64,
    //             ups: (1.0 / State::tick_duration()) as u64,
    //             swap_buffers: true,
    //             bench_mode: false,
    //             lazy: false,
    //             ups_reset: 2,
    //         });
    //     let mut frame_count: usize = 0;
    //     while let Some(e) = events.next(&mut window) {
    //         if frame_count > *TICKS_PER_EVALUATION.get().unwrap() {
    //             break;
    //         }
    //         if let Some(args) = e.render_args() {
    //             view.render(&args, &best_agent_clone);
    //         }

    //         if let Some(args) = e.update_args() {
    //             frame_count += 1;
    //             best_agent_clone.evaluate_step();
    //             view.update(&args);
    //         }
    //     }
    //     println!("Best score: {}", best_agent_clone.score);
    // }

    fn evaluate(&mut self) {
        (&mut self.agents).par_iter_mut().for_each(|agent| {
            agent.evaluate();
        })
    }

    fn sort(&mut self) {
        self.agents
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap()); // Bigger scores come before
    }

    fn next_gen(&mut self) {
        let mut cumulative_sum: Vec<f32> = Vec::with_capacity(*AGENTS_NUM.get().unwrap());
        let mut total_score = 0.0;
        self.agents
            .iter()
            .map(|agent| agent.score)
            .for_each(|score| {
                total_score += score;
                cumulative_sum.push(total_score);
            });
        if total_score == 0.0 {
            total_score = 0.001
        }
        cumulative_sum.iter_mut().for_each(|score| {
            *score /= total_score;
        });

        let mut new_agents: Vec<Agent<State>> =
            self.agents[0..*AGENTS_NUM.get().unwrap() / 3].to_vec();
        new_agents
            .iter_mut()
            .for_each(|agent| agent.state = State::init());
        while new_agents.len() < *AGENTS_NUM.get().unwrap() {
            let random = fastrand::f32();
            new_agents.push(
                self.agents[cumulative_sum
                    .iter()
                    .position(|&cumulative_score| cumulative_score > random)
                    .unwrap()]
                .mutated(),
            );
        }

        std::mem::swap(&mut self.agents, &mut new_agents);
        self.past_agents.push(new_agents); // `new_agents` are the old ones, as we just swapped them.
    }
}
