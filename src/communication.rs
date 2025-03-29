use flume::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use crate::{OneManyError, Agent};

/// Message content types that agents can send to each other
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageContent {
    /// A simple text message
    Text(String),
    /// A vector of values (could represent beliefs, states, etc.)
    Beliefs(Vec<f64>),
    /// A specific action the agent took or plans to take
    Action(usize),
    /// A reward the agent received
    Reward(f64),
    /// A request for information from another agent
    RequestInfo(InfoRequestType),
    /// Custom message with JSON-serializable content
    Custom(String),
}

/// Types of information an agent can request from another
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InfoRequestType {
    /// Request the other agent's current beliefs
    Beliefs,
    /// Request the other agent's planned action
    PlannedAction,
    /// Request the other agent's recent rewards
    RecentRewards,
    /// Custom request with specific details
    Custom(String),
}

/// A message sent between agents containing metadata and content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// ID of the sender agent
    pub sender_id: usize,
    /// ID of the recipient agent (None means broadcast to all)
    pub recipient_id: Option<usize>,
    /// Message content
    pub content: MessageContent,
    /// Optional message priority (higher means more important)
    pub priority: Option<u8>,
    /// Message timestamp (simulation step when it was sent)
    pub timestamp: usize,
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let recipient = match self.recipient_id {
            Some(id) => format!("Agent {}", id),
            None => "Everyone".to_string(),
        };
        
        let content_str = match &self.content {
            MessageContent::Text(text) => format!("Text: {}", text),
            MessageContent::Beliefs(beliefs) => format!("Beliefs: {:?}", beliefs),
            MessageContent::Action(action) => format!("Action: {}", action),
            MessageContent::Reward(reward) => format!("Reward: {}", reward),
            MessageContent::RequestInfo(req) => format!("Request: {:?}", req),
            MessageContent::Custom(data) => format!("Custom: {}", data),
        };
        
        write!(f, "From Agent {} to {}: {} (t={})", 
            self.sender_id, recipient, content_str, self.timestamp)
    }
}

/// Communication channel for two-way message passing between agents
pub struct CommunicationChannel {
    /// Map of sender channels keyed by agent ID
    senders: HashMap<usize, Sender<Message>>,
    /// Map of receiver channels keyed by agent ID
    receivers: HashMap<usize, Receiver<Message>>,
    /// Broadcast channel for messages to all agents
    broadcast_tx: Sender<Message>,
    /// Receivers for broadcast messages, keyed by agent ID
    broadcast_rx: HashMap<usize, Receiver<Message>>,
    /// Current simulation step (for message timestamps)
    current_step: usize,
}

impl CommunicationChannel {
    /// Create a new communication channel for a given number of agents
    pub fn new(n_agents: usize) -> Self {
        let mut senders = HashMap::new();
        let mut receivers = HashMap::new();
        let mut broadcast_rx = HashMap::new();
        
        // Create direct communication channels for each agent pair
        for i in 0..n_agents {
            for j in 0..n_agents {
                if i != j {
                    let (tx, rx) = flume::unbounded();
                    senders.entry(i).or_insert_with(HashMap::new).insert(j, tx);
                    receivers.entry(j).or_insert_with(HashMap::new).insert(i, rx);
                }
            }
        }
        
        // Create broadcast channel
        let (broadcast_tx, _) = flume::unbounded();
        
        // Create individual broadcast receivers for each agent
        for i in 0..n_agents {
            let (tx, rx) = flume::unbounded();
            broadcast_rx.insert(i, rx);
        }
        
        Self {
            senders: senders.into_iter()
                .map(|(k, v)| (k, v.into_iter().next().unwrap().1))
                .collect(),
            receivers: receivers.into_iter()
                .map(|(k, v)| (k, v.into_iter().next().unwrap().1))
                .collect(),
            broadcast_tx,
            broadcast_rx,
            current_step: 0,
        }
    }
    
    /// Send a message from one agent to another
    pub fn send(&self, sender_id: usize, recipient_id: usize, content: MessageContent) 
        -> Result<(), OneManyError> 
    {
        if let Some(tx) = self.senders.get(&sender_id) {
            let message = Message {
                sender_id,
                recipient_id: Some(recipient_id),
                content,
                priority: None,
                timestamp: self.current_step,
            };
            
            tx.send(message).map_err(|e| OneManyError::Communication(e.to_string()))
        } else {
            Err(OneManyError::InvalidAgentId(sender_id))
        }
    }
    
    /// Broadcast a message from one agent to all other agents
    pub fn broadcast(&self, sender_id: usize, content: MessageContent) 
        -> Result<(), OneManyError> 
    {
        let message = Message {
            sender_id,
            recipient_id: None,  // None indicates broadcast
            content,
            priority: None,
            timestamp: self.current_step,
        };
        
        self.broadcast_tx.send(message)
            .map_err(|e| OneManyError::Communication(e.to_string()))
    }
    
    /// Check if an agent has any messages waiting
    pub fn has_messages(&self, agent_id: usize) -> Result<bool, OneManyError> {
        if let Some(rx) = self.receivers.get(&agent_id) {
            Ok(!rx.is_empty())
        } else {
            Err(OneManyError::InvalidAgentId(agent_id))
        }
    }
    
    /// Receive messages for a specific agent (both direct and broadcast)
    pub fn receive_all(&self, agent_id: usize) -> Result<Vec<Message>, OneManyError> {
        let mut messages = Vec::new();
        
        // Get direct messages
        if let Some(rx) = self.receivers.get(&agent_id) {
            while let Ok(msg) = rx.try_recv() {
                messages.push(msg);
            }
        } else {
            return Err(OneManyError::InvalidAgentId(agent_id));
        }
        
        // Get broadcast messages
        if let Some(rx) = self.broadcast_rx.get(&agent_id) {
            while let Ok(msg) = rx.try_recv() {
                // Don't receive our own broadcast messages
                if msg.sender_id != agent_id {
                    messages.push(msg);
                }
            }
        }
        
        // Sort by priority (if specified) and then timestamp
        messages.sort_by(|a, b| {
            b.priority.unwrap_or(0).cmp(&a.priority.unwrap_or(0))
                .then_with(|| a.timestamp.cmp(&b.timestamp))
        });
        
        Ok(messages)
    }
    
    /// Advance the simulation step counter
    pub fn advance_step(&mut self) {
        self.current_step += 1;
    }
}

/// A wrapper for an agent message
pub struct AgentMessage {
    pub sender_id: usize,
    pub content: MessageContent,
}

/// Trait for agents that can communicate with each other
pub trait CommunicatingAgent: crate::Agent {
    /// Process incoming messages and update internal state
    fn process_messages(&mut self, messages: Vec<Message>) -> Result<(), OneManyError>;
    
    /// Generate outgoing messages based on internal state
    fn generate_messages(&self) -> Vec<AgentMessage>;
    
    /// Act based on observations and communication
    fn act_with_communication(
        &mut self, 
        observation: usize, 
        messages: Vec<Message>
    ) -> Result<usize, OneManyError>;
}

/// A POMDP agent that can communicate with other agents
pub struct CommunicatingPOMDPAgent {
    /// The base POMDP agent for decision making
    pub agent: crate::POMDPAgent,
    /// Agent's unique identifier
    pub id: usize,
    /// Whether this agent should communicate its beliefs
    pub share_beliefs: bool,
    /// Whether this agent should communicate its actions
    pub share_actions: bool,
    /// Whether this agent should communicate its rewards
    pub share_rewards: bool,
    /// The agent's memory of recent messages
    pub message_history: Vec<Message>,
    /// Beliefs about other agents' preferred actions (agent_id -> action -> belief)
    pub agent_action_beliefs: HashMap<usize, Vec<f64>>,
    /// Communication strategy (e.g., how often to communicate)
    pub communication_frequency: usize,
    /// Steps since last communication
    pub steps_since_communication: usize,
    /// Number of bandits/actions in the environment
    pub n_actions: usize,
    /// Last action the agent selected
    pub last_selected_action: Option<usize>,
    /// Current beliefs about environment states
    pub current_beliefs: Option<Vec<f64>>,
}

impl CommunicatingPOMDPAgent {
    /// Create a new communicating POMDP agent
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
    
    /// Update beliefs about other agents based on their messages
    fn update_agent_beliefs(&mut self, sender_id: usize, action: usize) {
        // Initialize beliefs for this agent if not already present
        if !self.agent_action_beliefs.contains_key(&sender_id) {
            let uniform_beliefs = vec![1.0 / self.n_actions as f64; self.n_actions];
            self.agent_action_beliefs.insert(sender_id, uniform_beliefs);
        }
        
        // Update beliefs
        if let Some(beliefs) = self.agent_action_beliefs.get_mut(&sender_id) {
            // Increase weight for the observed action
            for (i, belief) in beliefs.iter_mut().enumerate() {
                if i == action {
                    *belief += 0.1;  // Increase confidence in this action
                } else {
                    *belief *= 0.9;  // Decrease others proportionally
                }
            }
            
            // Normalize
            let sum: f64 = beliefs.iter().sum();
            if sum > 0.0 {
                for belief in beliefs.iter_mut() {
                    *belief /= sum;
                }
            }
        }
    }
    
    /// Check if we should communicate this step
    fn should_communicate(&self) -> bool {
        self.steps_since_communication >= self.communication_frequency
    }
    
    /// Update the communication counter
    fn update_communication_counter(&mut self) {
        self.steps_since_communication += 1;
        if self.steps_since_communication >= self.communication_frequency {
            self.steps_since_communication = 0;
        }
    }
}

impl crate::Agent for CommunicatingPOMDPAgent {
    fn act(&mut self, observation: usize) -> Result<usize, OneManyError> {
        // Delegate to the base agent
        let action = self.agent.act(observation)?;
        
        // Store action and update beliefs
        self.last_selected_action = Some(action);
        
        Ok(action)
    }
}

impl CommunicatingAgent for CommunicatingPOMDPAgent {
    fn process_messages(&mut self, messages: Vec<Message>) -> Result<(), OneManyError> {
        // Store messages for history
        self.message_history.extend(messages.clone());
        
        // Trim history if it gets too long
        if self.message_history.len() > 100 {
            self.message_history.drain(0..50);
        }
        
        // Process each message
        for message in messages {
            match message.content {
                MessageContent::Action(action) => {
                    // Update beliefs about this agent's action tendencies
                    self.update_agent_beliefs(message.sender_id, action);
                },
                MessageContent::Beliefs(beliefs) => {
                    // Could use other agents' beliefs to inform our own
                    // Would need custom logic here
                },
                MessageContent::RequestInfo(info_type) => {
                    // Could respond to requests
                    // Would need custom logic based on the request type
                },
                _ => {}  // Handle other message types as needed
            }
        }
        
        Ok(())
    }
    
    fn generate_messages(&self) -> Vec<AgentMessage> {
        let mut messages = Vec::new();
        
        // Only communicate on the schedule
        if !self.should_communicate() {
            return messages;
        }
        
        // Share beliefs if enabled
        if self.share_beliefs && self.current_beliefs.is_some() {
            messages.push(AgentMessage {
                sender_id: self.id,
                content: MessageContent::Beliefs(self.current_beliefs.clone().unwrap()),
            });
        }
        
        // Share last action if enabled and available
        if self.share_actions && self.last_selected_action.is_some() {
            messages.push(AgentMessage {
                sender_id: self.id,
                content: MessageContent::Action(self.last_selected_action.unwrap()),
            });
        }
        
        messages
    }
    
    fn act_with_communication(
        &mut self, 
        observation: usize, 
        messages: Vec<Message>
    ) -> Result<usize, OneManyError> {
        // Process incoming messages
        self.process_messages(messages)?;
        
        // Adjust our action based on our beliefs about others
        let action = self.agent.act(observation)?;
        
        // Update our internal tracking
        self.last_selected_action = Some(action);
        self.update_communication_counter();
        
        Ok(action)
    }
} 