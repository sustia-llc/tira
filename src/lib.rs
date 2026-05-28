use rand::prelude::*;
use rand::seq::WeightError;
use rand_distr::{Bernoulli, BernoulliError, Distribution};
use thiserror::Error;
mod agent;
mod communication;
mod group;
mod plotter;
mod simulation;

pub use agent::{Agent, CopyAgent, POMDPAgent};
pub use communication::{
    AgentMessage, CommunicatingAgent, CommunicatingPOMDPAgent, CommunicationChannel, Message,
    MessageContent,
};
pub use group::{GroupAgent, GroupAgentBuilder, VotingAgent, VotingMode};
pub use simulation::{
    RecoveryResult, TrialData, experiment_certainty_weighted, experiment_deterministic,
    experiment_identical, experiment_varying_alpha, experiment_varying_preferences, log_likelihood,
    parameter_recovery_single, recover_alpha, run_group_simulation, run_single_simulation,
};

#[derive(Error, Debug)]
pub enum OneManyError {
    #[error("Invalid probability value: {0}")]
    InvalidProbability(f64),
    #[error("Invalid action: {0}")]
    InvalidAction(usize),
    #[error("Distribution error: {0}")]
    Distribution(#[from] BernoulliError),
    #[error("Weight error: {0}")]
    Weight(#[from] WeightError),
    #[error("Invalid agent ID: {0}")]
    InvalidAgentId(usize),
    #[error("Resource conflict: Bandit {0} already selected")]
    ResourceConflict(usize),
    #[error("Communication error: {0}")]
    Communication(String),
}

#[allow(clippy::missing_errors_doc)]
pub trait Environment {
    fn step(&mut self, action: usize) -> Result<usize, OneManyError>;
}

#[allow(clippy::missing_errors_doc)]
pub trait MultiAgentEnvironment {
    fn step(
        &mut self,
        agent_id: usize,
        action: usize,
    ) -> Result<(usize, Option<StateChange>), OneManyError>;
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

#[derive(Debug, Clone)]
pub struct BanditEnvironment {
    probabilities: Vec<f64>,
    rng: ThreadRng,
}

impl BanditEnvironment {
    #[allow(clippy::missing_errors_doc)]
    pub fn new(probabilities: Vec<f64>) -> Result<Self, OneManyError> {
        for p in &probabilities {
            if !(0.0..=1.0).contains(p) {
                return Err(OneManyError::InvalidProbability(*p));
            }
        }
        Ok(Self {
            probabilities,
            rng: rand::rng(),
        })
    }
}

impl Environment for BanditEnvironment {
    fn step(&mut self, action: usize) -> Result<usize, OneManyError> {
        if action >= self.probabilities.len() {
            return Err(OneManyError::InvalidAction(action));
        }
        let prob = self.probabilities[action];
        let dist = Bernoulli::new(prob).map_err(OneManyError::Distribution)?;
        Ok(usize::from(dist.sample(&mut self.rng)))
    }
}

#[derive(Debug, Clone)]
pub struct SharedBanditEnvironment {
    base_probabilities: Vec<f64>,
    current_probabilities: Vec<f64>,
    bandit_selection: Vec<Option<usize>>,
    agents_acted: Vec<bool>,
    n_agents: usize,
    competitive: bool,
    rng: ThreadRng,
    step_counter: usize,
}

impl SharedBanditEnvironment {
    #[allow(clippy::missing_errors_doc)]
    pub fn new(probabilities: Vec<f64>, n_agents: usize) -> Result<Self, OneManyError> {
        for p in &probabilities {
            if !(0.0..=1.0).contains(p) {
                return Err(OneManyError::InvalidProbability(*p));
            }
        }
        if n_agents == 0 {
            return Err(OneManyError::InvalidAgentId(0));
        }
        let n_bandits = probabilities.len();
        Ok(Self {
            base_probabilities: probabilities.clone(),
            current_probabilities: probabilities,
            bandit_selection: vec![None; n_bandits],
            agents_acted: vec![false; n_agents],
            n_agents,
            competitive: true,
            rng: rand::rng(),
            step_counter: 0,
        })
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
    ) -> Result<(usize, Option<StateChange>), OneManyError> {
        if agent_id >= self.n_agents {
            return Err(OneManyError::InvalidAgentId(agent_id));
        }
        if action >= self.current_probabilities.len() {
            return Err(OneManyError::InvalidAction(action));
        }
        if self.competitive && self.bandit_selection[action].is_some() {
            return Err(OneManyError::ResourceConflict(action));
        }

        self.bandit_selection[action] = Some(agent_id);
        self.agents_acted[agent_id] = true;

        let prob = self.current_probabilities[action];
        let dist = Bernoulli::new(prob).map_err(OneManyError::Distribution)?;
        let reward = usize::from(dist.sample(&mut self.rng));

        let state_change = StateChange {
            bandit_selected: Some(action),
            reward_obtained: reward == 1,
            agent_id,
        };

        if self.agents_acted.iter().all(|&acted| acted) {
            self.next_round();
        }

        Ok((reward, Some(state_change)))
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
    fn step(&mut self, action: usize) -> Result<usize, OneManyError> {
        let (reward, _) =
            <Self as MultiAgentEnvironment>::step(self, 0, action)?;
        Ok(reward)
    }
}
