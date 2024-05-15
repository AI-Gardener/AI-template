use super::*;

#[derive(Clone)]
pub(crate) struct Agent<State: Reinforcement + Clone + Send + Sync> {
    pub(crate) dac: DAC,
    pub(crate) score: f32,
    pub(crate) instant: f32,

    pub(crate) state: State
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

    pub(crate) fn evaluate(&mut self) {
        // Run network, Update physics values (cart speed, cart pos, pendulum physics data from new position), then run again, for a number of ticks and a given tick duration.
        self.score = 0.1;
        self.dac.reordered();
        for _ in 0..*TICKS_PER_EVALUATION.get().unwrap() {
            self.evaluate_step();
        }
    }

    pub(crate) fn evaluate_step(&mut self) {
        self.state.set_inputs(&mut self.dac);

        self.dac.run();

        self.state.get_outputs(&self.dac);
        self.state.update_physics();
        self.state.update_score();

        self.instant += TICK_DURATION.get().unwrap();
    }
}