//! Reserved infrastructure for the paper's §4.1 network-communication extension
//! (extension 6, tracked as issue #46): network topologies where only some internal
//! agents communicate directly with the active agent, with influence routed through
//! intermediate connections. This module is currently exercised only by tests — it is
//! not wired into the group-agent pipeline.
//!
//! Behind the default-off `communication` feature since issue #5, so that downstream
//! consumers of the engine do not carry a channel dependency for a module they never
//! reach.
//!
//! **Scope, deliberately narrow (issue #5).** The module demonstrates message
//! *transport* and emission *cadence*, and nothing more: received messages have **no
//! effect on any agent's decision**. The belief-sharing and reward-sharing surfaces
//! that previously suggested otherwise were unreachable — `current_beliefs` was never
//! populated, `share_rewards` was stored and never read, and the received-action
//! belief map was a write-only sink — so they were removed rather than left as an
//! implied contract. Wiring messages into inference is issue #46's subject.

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

/// A message in transit.
///
/// `recipient_id` is `None` for a broadcast (see
/// [`CommunicationChannel::broadcast`]); the same message is then delivered into every
/// other agent's queue. `priority` orders delivery within a step — see
/// [`CommunicationChannel::receive_all`].
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
    /// Build a channel with one unbounded sender/receiver pair per agent
    /// (`0..n_agents`).
    #[must_use]
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

    /// Number of agents this channel was built for.
    #[must_use]
    pub fn n_agents(&self) -> usize {
        self.senders.len()
    }

    /// Push `message` into `recipient_id`'s queue.
    fn deliver(&self, recipient_id: usize, message: Message) -> Result<(), AifError> {
        let Some(tx) = self.senders.get(&recipient_id) else {
            return Err(AifError::InvalidAgentId(recipient_id));
        };
        tx.send(message)
            .map_err(|e| AifError::Communication(e.to_string()))
    }

    /// Send `content` to a single recipient at the default (lowest) priority.
    ///
    /// # Errors
    ///
    /// [`AifError::InvalidAgentId`] if `recipient_id` is not one of this channel's
    /// agents; [`AifError::Communication`] if the recipient's queue is disconnected.
    pub fn send(
        &self,
        sender_id: usize,
        recipient_id: usize,
        content: MessageContent,
    ) -> Result<(), AifError> {
        self.deliver(
            recipient_id,
            Message {
                sender_id,
                recipient_id: Some(recipient_id),
                content,
                priority: None,
                timestamp: self.current_step,
            },
        )
    }

    /// Send `content` to a single recipient at an explicit `priority`.
    ///
    /// Higher values are delivered first by [`receive_all`](Self::receive_all); an
    /// absent priority sorts as `0`.
    ///
    /// # Errors
    ///
    /// As [`send`](Self::send).
    pub fn send_with_priority(
        &self,
        sender_id: usize,
        recipient_id: usize,
        content: MessageContent,
        priority: u8,
    ) -> Result<(), AifError> {
        self.deliver(
            recipient_id,
            Message {
                sender_id,
                recipient_id: Some(recipient_id),
                content,
                priority: Some(priority),
                timestamp: self.current_step,
            },
        )
    }

    /// Deliver `content` to every agent except `sender_id`, as a broadcast
    /// (`recipient_id: None`).
    ///
    /// Returns the number of recipients the message was delivered to.
    ///
    /// # Errors
    ///
    /// [`AifError::InvalidAgentId`] if `sender_id` is not one of this channel's
    /// agents; [`AifError::Communication`] if any recipient's queue is disconnected.
    /// Delivery is not atomic — an error partway through leaves earlier recipients
    /// holding the message.
    pub fn broadcast(
        &self,
        sender_id: usize,
        content: &MessageContent,
        priority: Option<u8>,
    ) -> Result<usize, AifError> {
        if !self.senders.contains_key(&sender_id) {
            return Err(AifError::InvalidAgentId(sender_id));
        }

        // Deterministic recipient order: `senders` is a HashMap, so iterate the index
        // range rather than its arbitrary iteration order.
        let mut delivered = 0;
        for recipient_id in 0..self.senders.len() {
            if recipient_id == sender_id {
                continue;
            }
            self.deliver(
                recipient_id,
                Message {
                    sender_id,
                    recipient_id: None,
                    content: content.clone(),
                    priority,
                    timestamp: self.current_step,
                },
            )?;
            delivered += 1;
        }
        Ok(delivered)
    }

    /// Whether `agent_id` has any undelivered messages.
    ///
    /// # Errors
    ///
    /// [`AifError::InvalidAgentId`] if `agent_id` is not one of this channel's agents.
    pub fn has_messages(&self, agent_id: usize) -> Result<bool, AifError> {
        if let Some(rx) = self.receivers.get(&agent_id) {
            Ok(!rx.is_empty())
        } else {
            Err(AifError::InvalidAgentId(agent_id))
        }
    }

    /// Drain `agent_id`'s queue, highest priority first and oldest first within a
    /// priority (an absent priority sorts as `0`).
    ///
    /// # Errors
    ///
    /// [`AifError::InvalidAgentId`] if `agent_id` is not one of this channel's agents.
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

    /// Emit this step's outgoing messages, if the emission cadence is due.
    ///
    /// Takes `&mut self` because emitting resets the cadence counter — the check and
    /// the reset must happen together (issue #5).
    fn generate_messages(&mut self) -> Vec<AgentMessage>;

    fn act_with_communication(
        &mut self,
        observation: usize,
        messages: Vec<Message>,
    ) -> Result<usize, AifError>;
}

/// A [`POMDPAgent`](crate::POMDPAgent) wrapped with message transport.
///
/// Received messages are recorded in `message_history` and **do not influence the
/// wrapped agent's decisions** — see the module docs. The one live behaviour is
/// cadence: the agent emits its last selected action every `communication_frequency`
/// steps when `share_actions` is set.
pub struct CommunicatingPOMDPAgent {
    pub agent: crate::POMDPAgent,
    pub id: usize,
    pub share_actions: bool,
    pub message_history: Vec<Message>,
    /// Emit every `n`th step. `0` and `1` both emit every step.
    pub communication_frequency: usize,
    pub steps_since_communication: usize,
    pub last_selected_action: Option<usize>,
}

impl CommunicatingPOMDPAgent {
    #[must_use]
    pub fn new(
        agent: crate::POMDPAgent,
        id: usize,
        share_actions: bool,
        communication_frequency: usize,
    ) -> Self {
        Self {
            agent,
            id,
            share_actions,
            message_history: Vec::new(),
            communication_frequency,
            steps_since_communication: 0,
            last_selected_action: None,
        }
    }

    /// Whether an emission is due.
    ///
    /// The counter is advanced by [`act_with_communication`](CommunicatingAgent::act_with_communication)
    /// and reset by [`generate_messages`](CommunicatingAgent::generate_messages) — and
    /// only when it actually emits. Resetting anywhere else is the issue-#5 bug: the
    /// old code incremented and reset inside the same call, so the counter's observable
    /// values were `0..frequency-1` and the `>= frequency` test could never be true.
    fn should_communicate(&self) -> bool {
        self.steps_since_communication >= self.communication_frequency
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
        self.message_history.extend(messages);
        if self.message_history.len() > 100 {
            self.message_history.drain(0..50);
        }
        Ok(())
    }

    fn generate_messages(&mut self) -> Vec<AgentMessage> {
        if !self.should_communicate() {
            return Vec::new();
        }
        self.steps_since_communication = 0;

        let mut messages = Vec::new();
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
        self.steps_since_communication += 1;
        Ok(action)
    }
}
