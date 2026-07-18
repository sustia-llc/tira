pub use aif::{
    Agent, CopyAgent, CommunicatingAgent, CommunicatingPOMDPAgent, CommunicationChannel,
    AgentMessage, Message, MessageContent,
    GroupAgent, GroupAgentBuilder, VotingAgent, VotingMode,
    AifError, POMDPAgent,
};

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Bernoulli, Distribution};

mod plotter;
mod simulation;

pub use simulation::{
    RecoveryResult, TrialData, env_seed, experiment_certainty_weighted, experiment_deterministic,
    experiment_identical, experiment_varying_alpha, experiment_varying_preferences, group_seed,
    heterogeneity_seed, log_likelihood, log_likelihood_learning, parameter_recovery_single,
    recover_alpha, run_group_simulation, run_single_simulation, substream,
};

pub use plotter::ScatterPoint;

#[allow(clippy::missing_errors_doc)]
pub trait Environment {
    fn step(&mut self, action: usize) -> Result<usize, AifError>;
}

#[allow(clippy::missing_errors_doc)]
pub trait MultiAgentEnvironment {
    fn step(
        &mut self,
        agent_id: usize,
        action: usize,
    ) -> Result<(usize, Option<StateChange>), AifError>;
    fn reset(&mut self);
    fn num_agents(&self) -> usize;
    fn num_actions(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct StateChange {
    pub bandit_selected: Option<usize>,
    pub reward_obtained: bool,
    pub agent_id: usize,
}

/// Validate an arm-probability vector: every entry must lie in `[0, 1]`. The single
/// source of the 0..=1 check shared by both environment constructors.
fn validate_probabilities(probabilities: &[f64]) -> Result<(), AifError> {
    for p in probabilities {
        if !(0.0..=1.0).contains(p) {
            return Err(AifError::InvalidProbability(*p));
        }
    }
    Ok(())
}

// `Clone` is intentionally not derived: `StdRng` is not `Clone` in rand 0.10
// (cloning an RNG duplicates its stream, which is rarely intended), and no caller
// clones an environment.
#[derive(Debug)]
pub struct BanditEnvironment {
    probabilities: Vec<f64>,
    rng: StdRng,
}

impl BanditEnvironment {
    /// Validate the arm probabilities and assemble around a supplied RNG. Shared by
    /// [`new`](Self::new) and [`with_seed`](Self::with_seed) (same shape as
    /// [`SharedBanditEnvironment::build`]).
    fn build(probabilities: Vec<f64>, rng: StdRng) -> Result<Self, AifError> {
        validate_probabilities(&probabilities)?;
        Ok(Self { probabilities, rng })
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn new(probabilities: Vec<f64>) -> Result<Self, AifError> {
        Self::build(probabilities, StdRng::from_rng(&mut rand::rng()))
    }

    /// Construct with a fixed RNG seed so the reward draws are reproducible (issue #2).
    #[allow(clippy::missing_errors_doc)]
    pub fn with_seed(probabilities: Vec<f64>, seed: u64) -> Result<Self, AifError> {
        Self::build(probabilities, StdRng::seed_from_u64(seed))
    }
}

impl Environment for BanditEnvironment {
    fn step(&mut self, action: usize) -> Result<usize, AifError> {
        if action >= self.probabilities.len() {
            return Err(AifError::InvalidAction(action));
        }
        let prob = self.probabilities[action];
        let dist = Bernoulli::new(prob).map_err(AifError::Distribution)?;
        let won = dist.sample(&mut self.rng);
        // Observation index follows the agent's generative model: index 0 = preferred
        // (high-probability) outcome. Bernoulli(prob) is true iff that outcome occurred.
        Ok(usize::from(!won))
    }
}

// `Clone` intentionally not derived — see [`BanditEnvironment`].
#[derive(Debug)]
pub struct SharedBanditEnvironment {
    base_probabilities: Vec<f64>,
    current_probabilities: Vec<f64>,
    bandit_selection: Vec<Option<usize>>,
    agents_acted: Vec<bool>,
    n_agents: usize,
    competitive: bool,
    rng: StdRng,
    step_counter: usize,
}

impl SharedBanditEnvironment {
    /// Validate inputs and assemble the environment around a supplied RNG.
    /// Shared by [`new`](Self::new) and [`with_seed`](Self::with_seed) so the
    /// validation lives in exactly one place.
    fn build(probabilities: Vec<f64>, n_agents: usize, rng: StdRng) -> Result<Self, AifError> {
        validate_probabilities(&probabilities)?;
        if n_agents == 0 {
            return Err(AifError::InvalidAgentId(0));
        }
        let n_bandits = probabilities.len();
        Ok(Self {
            base_probabilities: probabilities.clone(),
            current_probabilities: probabilities,
            bandit_selection: vec![None; n_bandits],
            agents_acted: vec![false; n_agents],
            n_agents,
            competitive: true,
            rng,
            step_counter: 0,
        })
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn new(probabilities: Vec<f64>, n_agents: usize) -> Result<Self, AifError> {
        Self::build(probabilities, n_agents, StdRng::from_rng(&mut rand::rng()))
    }

    /// Construct with a fixed RNG seed so the reward draws are reproducible (issue #2).
    #[allow(clippy::missing_errors_doc)]
    pub fn with_seed(
        probabilities: Vec<f64>,
        n_agents: usize,
        seed: u64,
    ) -> Result<Self, AifError> {
        Self::build(probabilities, n_agents, StdRng::seed_from_u64(seed))
    }

    pub fn set_competitive(&mut self, competitive: bool) {
        self.competitive = competitive;
    }

    #[must_use]
    pub fn is_bandit_available(&self, bandit: usize) -> bool {
        self.bandit_selection
            .get(bandit)
            .is_some_and(Option::is_none)
    }

    fn next_round(&mut self) {
        self.bandit_selection = vec![None; self.base_probabilities.len()];
        self.agents_acted = vec![false; self.n_agents];
        self.step_counter += 1;
        self.current_probabilities.clone_from(&self.base_probabilities);
    }

    #[must_use]
    pub fn rounds(&self) -> usize {
        self.step_counter
    }
}

impl MultiAgentEnvironment for SharedBanditEnvironment {
    fn step(
        &mut self,
        agent_id: usize,
        action: usize,
    ) -> Result<(usize, Option<StateChange>), AifError> {
        if agent_id >= self.n_agents {
            return Err(AifError::InvalidAgentId(agent_id));
        }
        if action >= self.current_probabilities.len() {
            return Err(AifError::InvalidAction(action));
        }
        if self.competitive && self.bandit_selection[action].is_some() {
            return Err(AifError::ResourceConflict(action));
        }

        self.bandit_selection[action] = Some(agent_id);
        self.agents_acted[agent_id] = true;

        let prob = self.current_probabilities[action];
        let dist = Bernoulli::new(prob).map_err(AifError::Distribution)?;
        let won = dist.sample(&mut self.rng);
        // Observation index 0 = preferred outcome (agent's generative-model convention);
        // reward_obtained keeps reward semantics (true == win).
        let observation = usize::from(!won);

        let state_change = StateChange {
            bandit_selected: Some(action),
            reward_obtained: won,
            agent_id,
        };

        if self.agents_acted.iter().all(|&acted| acted) {
            self.next_round();
        }

        Ok((observation, Some(state_change)))
    }

    fn reset(&mut self) {
        self.bandit_selection = vec![None; self.base_probabilities.len()];
        self.agents_acted = vec![false; self.n_agents];
        self.current_probabilities.clone_from(&self.base_probabilities);
        self.step_counter = 0;
    }

    fn num_agents(&self) -> usize {
        self.n_agents
    }

    fn num_actions(&self) -> usize {
        self.current_probabilities.len()
    }
}

impl Environment for SharedBanditEnvironment {
    fn step(&mut self, action: usize) -> Result<usize, AifError> {
        let (observation, _) =
            <Self as MultiAgentEnvironment>::step(self, 0, action)?;
        Ok(observation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_seed_reproduces_observation_sequence() -> Result<(), AifError> {
        // Same seed + same action sequence ⇒ bit-identical Bernoulli reward draws.
        let actions = [0usize, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 1, 0, 1, 2];
        let mut a = BanditEnvironment::with_seed(vec![0.8, 0.2, 0.2], 777)?;
        let mut b = BanditEnvironment::with_seed(vec![0.8, 0.2, 0.2], 777)?;
        let obs_a: Vec<usize> = actions.iter().map(|&x| a.step(x)).collect::<Result<_, _>>()?;
        let obs_b: Vec<usize> = actions.iter().map(|&x| b.step(x)).collect::<Result<_, _>>()?;
        assert_eq!(obs_a, obs_b, "same seed must reproduce the observation sequence");

        // A different seed should diverge on this sequence (sanity, not a guarantee
        // for every seed pair, but overwhelmingly likely across 15 draws).
        let mut c = BanditEnvironment::with_seed(vec![0.8, 0.2, 0.2], 778)?;
        let obs_c: Vec<usize> = actions.iter().map(|&x| c.step(x)).collect::<Result<_, _>>()?;
        assert_ne!(obs_a, obs_c, "distinct seeds should diverge on the reward stream");
        Ok(())
    }
}
