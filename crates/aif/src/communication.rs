//! Reserved infrastructure for the paper's §4.1 network-communication extension
//! (Extension #6): network topologies where only some internal agents communicate
//! directly with the active agent, with influence routed through intermediate
//! connections. This module is currently exercised only by tests — it is not yet
//! wired into the group-agent pipeline.

use crate::{Agent, AifError};
use flume::{Receiver, Sender};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MessageContent {
    Text(String),
    Beliefs(Vec<f64>),
    Action(usize),
    Reward(f64),
    RequestInfo(InfoRequestType),
    Custom(String),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InfoRequestType {
    Beliefs,
    PlannedAction,
    RecentRewards,
    Custom(String),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Message {
    pub sender_id: usize,
    pub recipient_id: Option<usize>,
    pub content: MessageContent,
    pub priority: Option<u8>,
    pub timestamp: usize,
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let recipient = match self.recipient_id {
            Some(id) => format!("Agent {id}"),
            None => "Everyone".to_string(),
        };
        let content_str = match &self.content {
            MessageContent::Text(text) => format!("Text: {text}"),
            MessageContent::Beliefs(beliefs) => format!("Beliefs: {beliefs:?}"),
            MessageContent::Action(action) => format!("Action: {action}"),
            MessageContent::Reward(reward) => format!("Reward: {reward}"),
            MessageContent::RequestInfo(req) => format!("Request: {req:?}"),
            MessageContent::Custom(data) => format!("Custom: {data}"),
        };
        write!(
            f,
            "From Agent {} to {}: {} (t={})",
            self.sender_id, recipient, content_str, self.timestamp
        )
    }
}

pub struct CommunicationChannel {
    senders: HashMap<usize, Sender<Message>>,
    receivers: HashMap<usize, Receiver<Message>>,
    current_step: usize,
}

impl CommunicationChannel {
    pub fn new(n_agents: usize) -> Self {
        let mut senders = HashMap::new();
        let mut receivers = HashMap::new();

        for i in 0..n_agents {
            let (tx, rx) = flume::unbounded();
            senders.insert(i, tx);
            receivers.insert(i, rx);
        }

        Self {
            senders,
            receivers,
            current_step: 0,
        }
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn send(
        &self,
        sender_id: usize,
        recipient_id: usize,
        content: MessageContent,
    ) -> Result<(), AifError> {
        if let Some(tx) = self.senders.get(&recipient_id) {
            let message = Message {
                sender_id,
                recipient_id: Some(recipient_id),
                content,
                priority: None,
                timestamp: self.current_step,
            };
            tx.send(message)
                .map_err(|e| AifError::Communication(e.to_string()))
        } else {
            Err(AifError::InvalidAgentId(recipient_id))
        }
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn has_messages(&self, agent_id: usize) -> Result<bool, AifError> {
        if let Some(rx) = self.receivers.get(&agent_id) {
            Ok(!rx.is_empty())
        } else {
            Err(AifError::InvalidAgentId(agent_id))
        }
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn receive_all(&self, agent_id: usize) -> Result<Vec<Message>, AifError> {
        let mut messages = Vec::new();

        if let Some(rx) = self.receivers.get(&agent_id) {
            while let Ok(msg) = rx.try_recv() {
                messages.push(msg);
            }
        } else {
            return Err(AifError::InvalidAgentId(agent_id));
        }

        messages.sort_by(|a, b| {
            b.priority
                .unwrap_or(0)
                .cmp(&a.priority.unwrap_or(0))
                .then_with(|| a.timestamp.cmp(&b.timestamp))
        });

        Ok(messages)
    }

    pub fn advance_step(&mut self) {
        self.current_step += 1;
    }
}

pub struct AgentMessage {
    pub sender_id: usize,
    pub content: MessageContent,
}

#[allow(clippy::missing_errors_doc)]
pub trait CommunicatingAgent: Agent {
    fn process_messages(&mut self, messages: Vec<Message>) -> Result<(), AifError>;
    fn generate_messages(&self) -> Vec<AgentMessage>;
    fn act_with_communication(
        &mut self,
        observation: usize,
        messages: Vec<Message>,
    ) -> Result<usize, AifError>;
}

pub struct CommunicatingPOMDPAgent {
    pub agent: crate::POMDPAgent,
    pub id: usize,
    pub share_beliefs: bool,
    pub share_actions: bool,
    pub share_rewards: bool,
    pub message_history: Vec<Message>,
    /// `agent_id` → per-action belief weights
    pub agent_action_beliefs: HashMap<usize, Vec<f64>>,
    pub communication_frequency: usize,
    pub steps_since_communication: usize,
    pub n_actions: usize,
    pub last_selected_action: Option<usize>,
    pub current_beliefs: Option<Vec<f64>>,
}

impl CommunicatingPOMDPAgent {
    #[must_use]
    pub fn new(
        agent: crate::POMDPAgent,
        id: usize,
        n_actions: usize,
        share_beliefs: bool,
        share_actions: bool,
        share_rewards: bool,
        communication_frequency: usize,
    ) -> Self {
        Self {
            agent,
            id,
            share_beliefs,
            share_actions,
            share_rewards,
            message_history: Vec::new(),
            agent_action_beliefs: HashMap::new(),
            communication_frequency,
            steps_since_communication: 0,
            n_actions,
            last_selected_action: None,
            current_beliefs: None,
        }
    }

    fn update_agent_beliefs(&mut self, sender_id: usize, action: usize) {
        let n = self.n_actions;
        let beliefs = self
            .agent_action_beliefs
            .entry(sender_id)
            .or_insert_with(|| vec![1.0 / n as f64; n]);

        for (i, belief) in beliefs.iter_mut().enumerate() {
            if i == action {
                *belief += 0.1;
            } else {
                *belief *= 0.9;
            }
        }

        let sum: f64 = beliefs.iter().sum();
        if sum > 0.0 {
            for belief in beliefs.iter_mut() {
                *belief /= sum;
            }
        }
    }

    fn should_communicate(&self) -> bool {
        self.steps_since_communication >= self.communication_frequency
    }

    fn update_communication_counter(&mut self) {
        self.steps_since_communication += 1;
        if self.steps_since_communication >= self.communication_frequency {
            self.steps_since_communication = 0;
        }
    }
}

impl Agent for CommunicatingPOMDPAgent {
    fn act(&mut self, observation: usize) -> Result<usize, AifError> {
        let action = self.agent.act(observation)?;
        self.last_selected_action = Some(action);
        Ok(action)
    }
}

impl CommunicatingAgent for CommunicatingPOMDPAgent {
    fn process_messages(&mut self, messages: Vec<Message>) -> Result<(), AifError> {
        self.message_history.extend(messages.iter().cloned());
        if self.message_history.len() > 100 {
            self.message_history.drain(0..50);
        }
        for message in &messages {
            if let MessageContent::Action(action) = &message.content {
                self.update_agent_beliefs(message.sender_id, *action);
            }
        }
        Ok(())
    }

    fn generate_messages(&self) -> Vec<AgentMessage> {
        let mut messages = Vec::new();
        if !self.should_communicate() {
            return messages;
        }
        if self.share_beliefs
            && let Some(ref beliefs) = self.current_beliefs
        {
            messages.push(AgentMessage {
                sender_id: self.id,
                content: MessageContent::Beliefs(beliefs.clone()),
            });
        }
        if self.share_actions
            && let Some(action) = self.last_selected_action
        {
            messages.push(AgentMessage {
                sender_id: self.id,
                content: MessageContent::Action(action),
            });
        }
        messages
    }

    fn act_with_communication(
        &mut self,
        observation: usize,
        messages: Vec<Message>,
    ) -> Result<usize, AifError> {
        self.process_messages(messages)?;
        let action = self.agent.act(observation)?;
        self.last_selected_action = Some(action);
        self.update_communication_counter();
        Ok(action)
    }
}
