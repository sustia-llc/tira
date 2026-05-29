use rand::seq::WeightError;
use rand_distr::BernoulliError;
use thiserror::Error;

mod agent;
mod communication;
mod group;

pub use agent::{Agent, CopyAgent, POMDPAgent};
pub use communication::{
    AgentMessage, CommunicatingAgent, CommunicatingPOMDPAgent, CommunicationChannel, Message,
    MessageContent,
};
pub use group::{GroupAgent, GroupAgentBuilder, VotingAgent, VotingMode};

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
