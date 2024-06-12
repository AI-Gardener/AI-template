use super::*;

mod agent;
#[allow(unused_imports)]
pub use agent::*;
mod dac;
pub use dac::*;
mod node;
pub use node::*;

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
        TICKS_PER_EVALUATION.set(State::ticks_per_evaluation()).unwrap();
        TICK_DURATION.set(State::tick_duration()).unwrap();
        START_NODES.set(State::start_nodes()).unwrap();
        NEWNODE_ACTIVATION_FUNCTION.set(State::newnode_activation_function()).unwrap();
        CUSTOM_ACTIVATION_F1.set(State::custom_activation_f1()).unwrap();
        CUSTOM_ACTIVATION_F2.set(State::custom_activation_f2()).unwrap();
        CUSTOM_ACTIVATION_F3.set(State::custom_activation_f3()).unwrap();
        CUSTOM_ACTIVATION_F4.set(State::custom_activation_f4()).unwrap();
        CUSTOM_ACTIVATION_F5.set(State::custom_activation_f5()).unwrap();

        Self {
            agents: (0..*AGENTS_NUM.get().unwrap())
                .map(|_| Agent::new())
                .collect(),
            past_agents: Vec::new(),
        }
    }

    /// Trains the AI
    pub fn train(&mut self) {
        (0..*NUM_GENERATIONS.get().unwrap())
            .into_iter()
            .for_each(|gen| {
                println!("Testing generation {}...", gen);
                self.evaluate();
                self.sort();
                self.next_gen();
            });
    }

    /// Checks the latest score is the best score.
    pub fn check(&self) {
        assert!(
            self.past_agents
                .iter()
                .map(|generation_agents| generation_agents[0].score)
                .all(|score| score <= self.agents[0].score)
        );
    }
    
    // TODO Interface only in `Vk`.
    pub fn best_agent_latest(&self) -> Agent<State> {
        let best_agent = self.past_agents.last().unwrap()[0].clone();
        println!("Best DAC network: {:?}\nBest DAC network score: {}.\nDisplaying the evaluation...", best_agent.dac, best_agent.score);
        Agent::from_nodes(best_agent.dac.nodes)
    }

    pub fn best_agent_at(&self, generation: usize) -> Agent<State> {
        let agent = self.past_agents[generation][0].clone();
        println!("Best DAC network: {:?}\nBest DAC network score: {}.\nDisplaying the evaluation...", agent.dac, agent.score);
        Agent {
            dac: agent.dac,
            ..Agent::new()
        }
    }

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
