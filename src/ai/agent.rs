use std::fs::File;
use std::io::{Read, Write};

use super::*;

#[derive(Clone)]
pub struct Agent<State: Reinforcement + Clone + Send + Sync> {
    pub(crate) dac: DAC,
    pub(crate) score: f32,
    pub(crate) instant: f32,

    pub(crate) state: State,
}

impl<State: Reinforcement + Clone + Send + Sync> Agent<State> {
    pub(crate) fn new() -> Self {
        Self {
            dac: DAC::new((*START_NODES.get().unwrap()).clone()),
            score: 0.1,
            instant: 0.0,

            state: State::init(),
        }
    }

    pub(crate) fn mutated(&self) -> Self {
        Self {
            dac: State::mutated(&self.dac),
            score: 0.1,
            instant: 0.0,

            state: State::init(),
        }
    }

    pub(crate) fn from_nodes(nodes: Vec<Box<Node>>) -> Self {
        Self {
            dac: DAC::new(nodes),
            score: 0.1,
            instant: 0.0,

            state: State::init(),
        }
    }

    pub(crate) fn evaluate(&mut self) {
        // Run network, Update physics values (cart speed, cart pos, pendulum physics data from new position), then run again, for a number of ticks and a given tick duration.
        self.score = 0.1;
        self.dac.reordered();
        for _ in 0..*TICKS_PER_EVALUATION.get().unwrap() {
            self.evaluate_step(*TICK_DURATION.get().unwrap());
        }
    }

    pub(crate) fn evaluate_step(&mut self, delta_t: f32) {
        self.dac.zeroed();

        self.state.set_inputs(&mut self.dac);
        self.dac.run();
        self.state.get_outputs(&self.dac);

        self.state.update_physics(delta_t);
        self.state.update_score(&mut self.score, delta_t);

        self.instant += delta_t;
    }

    pub fn export_network(&self, filename: &str) {
        let network = Network::from_nodes(&self.dac.nodes);
        let serialized = serde_json::to_string_pretty(&network).unwrap();
        let mut file = File::create(filename).unwrap();
        file.write_all(serialized.into_bytes().as_slice()).unwrap();
    }

    pub fn import_network(filename: &str) -> Self {
        let mut file = File::open(filename).unwrap();
        let mut serialized: Vec<u8> = Vec::new();
        file.read_to_end(&mut serialized).unwrap();
        let network: Network = serde_json::from_slice(&serialized).unwrap();

        Self::from_nodes(network.to_nodes())
    }
}
