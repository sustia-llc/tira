// `clippy::cast_precision_loss` is allowed crate-wide (issue #11 pedantic burn-down).
// Every occurrence is a `usize as f64` on a dimension or a count — state/observation/
// action counts, window lengths, Dirichlet denominators, uniform-distribution
// reciprocals. These are all far below 2^53, so no precision is actually lost, and the
// engine's arithmetic is bit-pinned by seeded tests: rewriting the casts (e.g. via
// `f64::from(u32::try_from(..)?)`) would add fallible plumbing to hot numeric loops
// without changing a single computed value.
#![allow(clippy::cast_precision_loss)]

use rand::seq::WeightError;
use rand_distr::BernoulliError;
use thiserror::Error;

mod agent;
mod coalition;
mod communication;
mod group;
mod special;

pub use agent::{
    Agent, AgentParams, CopyAgent, GenerativeModel, InternalAgent, POMDPAgent,
    ParameterFreeEnergies, PrecisionDynamics, StateInference,
};
pub use coalition::{
    AgentId, CoalitionHistory, CompatibilityBeliefs, ObsPrecisionParams, TrustBeliefs,
    belief_weighted_preference, competence_efe,
};
pub use communication::{
    AgentMessage, CommunicatingAgent, CommunicatingPOMDPAgent, CommunicationChannel, Message,
    MessageContent,
};
pub use group::{Aggregator, GroupAgent, GroupAgentBuilder, VotingAgent, VotingMode};

#[derive(Error, Debug)]
pub enum AifError {
    #[error("Invalid probability value: {0}")]
    InvalidProbability(f64),
    #[error("Invalid distribution: {0}")]
    InvalidDistribution(String),
    #[error("Invalid action: {0}")]
    InvalidAction(usize),
    #[error("Invalid length: expected {expected}, got {got}")]
    InvalidLength { expected: usize, got: usize },
    #[error("Distribution error: {0}")]
    Distribution(#[from] BernoulliError),
    #[error("Weight error: {0}")]
    Weight(#[from] WeightError),
    #[error("Invalid agent ID: {0}")]
    InvalidAgentId(usize),
    #[error("Resource conflict: option {0} already claimed")]
    ResourceConflict(usize),
    #[error("Communication error: {0}")]
    Communication(String),
}
