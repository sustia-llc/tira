use nalgebra::{DMatrix, DVector, Matrix, Matrix3};
use rand::prelude::*;
use rand::seq::WeightError;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::weighted::WeightedIndex;
use rand_distr::{Bernoulli, BernoulliError, Distribution};
use serde::{Deserialize, Serialize};
use std::ops::AddAssign;
use thiserror::Error;
mod simulation;
mod plotter;
mod agent;

// Re-export the Agent trait and POMDPAgent for tests and external use
pub use agent::{Agent, POMDPAgent, CopyAgent};

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
}

// Environment trait for different bandit environments
pub trait Environment {
    fn step(&mut self, action: usize) -> Result<usize, OneManyError>;
}

// Multi-agent environment trait with agent identification
pub trait MultiAgentEnvironment {
    fn step(&mut self, agent_id: usize, action: usize) -> Result<(usize, Option<StateChange>), OneManyError>;
    fn reset(&mut self);
    fn num_agents(&self) -> usize;
    fn num_actions(&self) -> usize;
}

// Struct to represent changes in environment state that might be relevant to other agents
#[derive(Debug, Clone)]
pub struct StateChange {
    // Changes in resource availability, agent positions, etc.
    pub bandit_selected: Option<usize>,
    pub reward_obtained: bool,
    pub agent_id: usize,
}

// Basic multi-armed bandit environment
#[derive(Debug, Clone)]
pub struct BanditEnvironment {
    probabilities: Vec<f64>,
    rng: ThreadRng,
}

impl BanditEnvironment {
    pub fn new(probabilities: Vec<f64>) -> Result<Self, OneManyError> {
        // Validate probabilities are between 0 and 1
        for p in &probabilities {
            if *p < 0.0 || *p > 1.0 {
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

        Ok(if dist.sample(&mut self.rng) { 1 } else { 0 })
    }
}

// Shared multi-armed bandit environment for multi-agent scenarios
// This environment implements competition for resources and adaptive probabilities
#[derive(Debug, Clone)]
pub struct SharedBanditEnvironment {
    // Base probabilities for each bandit
    base_probabilities: Vec<f64>,
    // Current probabilities (may be affected by other agents' actions)
    current_probabilities: Vec<f64>,
    // Which agent selected which bandit in current round
    bandit_selection: Vec<Option<usize>>, // None = not selected, Some(agent_id) = selected by agent_id
    // Number of agents in the environment
    n_agents: usize,
    // Are we using competitive mode? (agents can't select same bandit in one round)
    competitive: bool,
    // Random number generator
    rng: ThreadRng,
    // Current round/step
    step_counter: usize,
}

impl SharedBanditEnvironment {
    pub fn new(probabilities: Vec<f64>, n_agents: usize) -> Result<Self, OneManyError> {
        // Validate probabilities
        for p in &probabilities {
            if *p < 0.0 || *p > 1.0 {
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
            n_agents,
            competitive: true, // Default to competitive mode
            rng: rand::rng(),
            step_counter: 0,
        })
    }

    /// Set whether the environment is competitive (agents compete for bandits)
    pub fn set_competitive(&mut self, competitive: bool) {
        self.competitive = competitive;
    }

    /// Check if a bandit is available (not selected by any agent in this round)
    pub fn is_bandit_available(&self, bandit: usize) -> bool {
        self.bandit_selection.get(bandit).map_or(false, |agent| agent.is_none())
    }

    /// Reset the environment for a new round
    fn next_round(&mut self) {
        self.bandit_selection = vec![None; self.base_probabilities.len()];
        self.step_counter += 1;

        // Reset probabilities to base values (could implement more complex dynamics here)
        self.current_probabilities = self.base_probabilities.clone();
    }

    /// Get the number of rounds that have passed
    pub fn rounds(&self) -> usize {
        self.step_counter
    }
}

impl MultiAgentEnvironment for SharedBanditEnvironment {
    fn step(&mut self, agent_id: usize, action: usize) -> Result<(usize, Option<StateChange>), OneManyError> {
        // Validate agent_id
        if agent_id >= self.n_agents {
            return Err(OneManyError::InvalidAgentId(agent_id));
        }

        // Validate action
        if action >= self.current_probabilities.len() {
            return Err(OneManyError::InvalidAction(action));
        }

        // In competitive mode, check if bandit is already selected
        if self.competitive && self.bandit_selection[action].is_some() {
            return Err(OneManyError::ResourceConflict(action));
        }

        // Mark bandit as selected by this agent
        self.bandit_selection[action] = Some(agent_id);

        // Sample reward from probability distribution
        let prob = self.current_probabilities[action];
        let dist = Bernoulli::new(prob).map_err(OneManyError::Distribution)?;
        let reward = if dist.sample(&mut self.rng) { 1 } else { 0 };

        // Create state change notification for other agents
        let state_change = StateChange {
            bandit_selected: Some(action),
            reward_obtained: reward == 1,
            agent_id,
        };

        // Check if all agents have acted
        let all_acted = (0..self.n_agents).all(|id| {
            self.bandit_selection.iter().any(|&selection| selection == Some(id))
        });

        // If all agents have acted, prepare for next round
        if all_acted {
            self.next_round();
        }

        Ok((reward, Some(state_change)))
    }

    fn reset(&mut self) {
        self.bandit_selection = vec![None; self.base_probabilities.len()];
        self.current_probabilities = self.base_probabilities.clone();
        self.step_counter = 0;
    }

    fn num_agents(&self) -> usize {
        self.n_agents
    }

    fn num_actions(&self) -> usize {
        self.current_probabilities.len()
    }
}

// For standard Environment trait compatibility 
// (allows using it with single-agent code)
impl Environment for SharedBanditEnvironment {
    fn step(&mut self, action: usize) -> Result<usize, OneManyError> {
        // Use agent_id 0 when called through the single-agent interface
        let (reward, _) = self.step_as_agent(0, action)?;
        Ok(reward)
    }
}

// Implement with a different method name to avoid conflict
impl SharedBanditEnvironment {
    fn step_as_agent(&mut self, agent_id: usize, action: usize) -> Result<(usize, Option<StateChange>), OneManyError> {
        // Delegate to the MultiAgentEnvironment implementation
        <Self as MultiAgentEnvironment>::step(self, agent_id, action)
    }
}

