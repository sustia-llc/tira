use one_many_rs::{
    Agent, POMDPAgent, Environment, MultiAgentEnvironment, SharedBanditEnvironment, 
    OneManyError, StateChange, CommunicatingAgent, CommunicationChannel, Message, 
    AgentMessage, MessageContent, CommunicatingPOMDPAgent
};
use std::collections::HashMap;

/// A simple test environment to track agent actions and communications
struct TestEnvironment {
    n_agents: usize,
    n_bandits: usize,
    actions: Vec<Vec<usize>>,
    rewards: Vec<Vec<usize>>,
    comm_channel: CommunicationChannel,
    messages_sent: Vec<Vec<String>>,
}

impl TestEnvironment {
    fn new(n_agents: usize, n_bandits: usize) -> Self {
        Self {
            n_agents,
            n_bandits,
            actions: vec![Vec::new(); n_agents],
            rewards: vec![Vec::new(); n_agents],
            comm_channel: CommunicationChannel::new(n_agents),
            messages_sent: vec![Vec::new(); n_agents],
        }
    }
    
    fn record_action(&mut self, agent_id: usize, action: usize, reward: usize) {
        self.actions[agent_id].push(action);
        self.rewards[agent_id].push(reward);
    }
    
    fn record_message(&mut self, message: &Message) {
        let message_str = format!("{}", message);
        self.messages_sent[message.sender_id].push(message_str);
    }
    
    fn advance_step(&mut self) {
        self.comm_channel.advance_step();
    }
}

/// This test demonstrates two communicating POMDP agents that share information
/// to coordinate their actions in a competitive environment
#[test]
fn test_communicating_agents() -> Result<(), OneManyError> {
    // Setup environment
    let n_agents = 2;
    let n_bandits = 3;
    let mut test_env = TestEnvironment::new(n_agents, n_bandits);
    let mut bandit_env = SharedBanditEnvironment::new(vec![0.8, 0.4, 0.6], n_agents)?;
    
    // Create two communicating agents
    let base_agent1 = POMDPAgent::new(
        n_bandits,
        Some(vec![0.5, 0.5, 0.5]),
        Some(vec![1.0, 1.0, 1.0]),
        vec![1.0, 0.5, 0.7],  // Prefers bandit 0
        None,
        8.0,
        true
    )?;
    
    let base_agent2 = POMDPAgent::new(
        n_bandits,
        Some(vec![0.5, 0.5, 0.5]),
        Some(vec![1.0, 1.0, 1.0]),
        vec![0.6, 0.5, 1.0],  // Prefers bandit 2
        None,
        8.0,
        true
    )?;
    
    // Wrap them as communicating agents
    let mut agent1 = CommunicatingPOMDPAgent::new(
        base_agent1,
        0,              // Agent ID
        n_bandits,
        true,           // Share beliefs
        true,           // Share actions
        false,          // Don't share rewards
        3               // Communicate every 3 steps
    );
    
    let mut agent2 = CommunicatingPOMDPAgent::new(
        base_agent2,
        1,              // Agent ID
        n_bandits,
        false,          // Don't share beliefs
        true,           // Share actions
        false,          // Don't share rewards
        2               // Communicate every 2 steps
    );
    
    // Run simulation for 30 rounds
    let mut prev_obs1 = 0;
    let mut prev_obs2 = 0;
    
    for step in 0..30 {
        // Advance environment step
        test_env.advance_step();
        
        // ----- Agent 1's turn -----
        
        // Get messages from other agents
        let messages_for_agent1 = test_env.comm_channel.receive_all(0)?;
        
        // Agent 1 acts with communication
        let action1 = agent1.act_with_communication(prev_obs1, messages_for_agent1)?;
        
        // Environment processes action
        let (reward1, _) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut bandit_env, 0, action1)?;
        prev_obs1 = reward1;
        
        // Record action and reward
        test_env.record_action(0, action1, reward1);
        
        // Agent 1 generates outgoing messages
        let outgoing_messages1 = agent1.generate_messages();
        
        // Send any messages through the communication channel
        for msg in outgoing_messages1 {
            let content = msg.content.clone();
            test_env.comm_channel.send(msg.sender_id, 1, content)?;
            
            // For debugging, record the message that was sent
            if let Ok(messages) = test_env.comm_channel.receive_all(1) {
                for m in messages {
                    test_env.record_message(&m);
                }
            }
        }
        
        // ----- Agent 2's turn -----
        
        // Get messages from other agents
        let messages_for_agent2 = test_env.comm_channel.receive_all(1)?;
        
        // Agent 2 acts with communication
        let action2 = agent2.act_with_communication(prev_obs2, messages_for_agent2)?;
        
        // Try to take action, and handle conflicts
        match <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut bandit_env, 1, action2) {
            Ok((reward2, _)) => {
                prev_obs2 = reward2;
                test_env.record_action(1, action2, reward2);
            },
            Err(OneManyError::ResourceConflict(_)) => {
                // Try alternative actions
                let mut found_valid_action = false;
                
                for alt_action in (0..n_bandits).filter(|&a| a != action2) {
                    match <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut bandit_env, 1, alt_action) {
                        Ok((reward, _)) => {
                            prev_obs2 = reward;
                            test_env.record_action(1, alt_action, reward);
                            found_valid_action = true;
                            break;
                        },
                        Err(OneManyError::ResourceConflict(_)) => continue,
                        Err(e) => return Err(e),
                    }
                }
                
                if !found_valid_action {
                    // If no valid action found, record a failed attempt
                    test_env.record_action(1, action2, 0);
                    prev_obs2 = 0;
                }
            },
            Err(e) => return Err(e),
        }
        
        // Agent 2 generates outgoing messages
        let outgoing_messages2 = agent2.generate_messages();
        
        // Send any messages through the communication channel
        for msg in outgoing_messages2 {
            let content = msg.content.clone();
            test_env.comm_channel.send(msg.sender_id, 0, content)?;
            
            // For debugging, record the message that was sent
            if let Ok(messages) = test_env.comm_channel.receive_all(0) {
                for m in messages {
                    test_env.record_message(&m);
                }
            }
        }
    }
    
    // Print communications statistics
    println!("Agent 1 sent {} messages", test_env.messages_sent[0].len());
    println!("Agent 2 sent {} messages", test_env.messages_sent[1].len());
    
    if !test_env.messages_sent[0].is_empty() {
        println!("First message from Agent 1: {}", test_env.messages_sent[0][0]);
    }
    
    if !test_env.messages_sent[1].is_empty() {
        println!("First message from Agent 2: {}", test_env.messages_sent[1][0]);
    }
    
    // Count actions for each agent
    let count_actions = |actions: &Vec<usize>| {
        let mut counts = vec![0; n_bandits];
        for &a in actions {
            counts[a] += 1;
        }
        counts
    };
    
    let agent1_actions = count_actions(&test_env.actions[0]);
    let agent2_actions = count_actions(&test_env.actions[1]);
    
    println!("Agent 1 action distribution: {:?}", agent1_actions);
    println!("Agent 2 action distribution: {:?}", agent2_actions);
    
    // VALIDATION: Communication should help agents develop complementary strategies
    
    // Check if agents learned to avoid conflicts
    // By the end, they should select different bandits
    let agent1_preferred = agent1_actions.iter().enumerate()
        .max_by_key(|&(_, count)| count)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    
    let agent2_preferred = agent2_actions.iter().enumerate()
        .max_by_key(|&(_, count)| count)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    
    println!("Agent 1 preferred bandit: {}", agent1_preferred);
    println!("Agent 2 preferred bandit: {}", agent2_preferred);
    
    // Agents should develop different preferences to avoid conflicts
    assert_ne!(agent1_preferred, agent2_preferred, 
        "Agents should develop different bandit preferences through communication");
    
    Ok(())
}

/// This test demonstrates communication facilitating cooperative behavior
#[test]
fn test_cooperative_communication() -> Result<(), OneManyError> {
    // Setup environment with equal reward probabilities
    let n_agents = 2;
    let n_bandits = 3;
    let mut test_env = TestEnvironment::new(n_agents, n_bandits);
    
    // Non-competitive environment so agents can share bandits
    let mut bandit_env = SharedBanditEnvironment::new(vec![0.6, 0.6, 0.6], n_agents)?;
    bandit_env.set_competitive(false);
    
    // Create two agents with complementary preferences
    let base_agent1 = POMDPAgent::new(
        n_bandits,
        Some(vec![0.5, 0.5, 0.5]),
        Some(vec![1.0, 1.0, 1.0]),
        vec![1.0, 0.3, 0.3], // Strong preference for bandit 0
        None,
        12.0,
        true
    )?;
    
    let base_agent2 = POMDPAgent::new(
        n_bandits,
        Some(vec![0.5, 0.5, 0.5]),
        Some(vec![1.0, 1.0, 1.0]),
        vec![0.3, 0.3, 1.0], // Strong preference for bandit 2
        None,
        12.0,
        true
    )?;
    
    // Wrap as communicating agents with frequent communication
    let mut agent1 = CommunicatingPOMDPAgent::new(
        base_agent1,
        0,              // Agent ID
        n_bandits,
        false,          // Don't share beliefs (simplifies test)
        true,           // Share actions
        true,           // Share rewards
        1               // Communicate every step
    );
    
    let mut agent2 = CommunicatingPOMDPAgent::new(
        base_agent2,
        1,              // Agent ID
        n_bandits,
        false,          // Don't share beliefs
        true,           // Share actions
        true,           // Share rewards
        1               // Communicate every step
    );
    
    // Run simulation with more explicit cooperation
    let mut prev_obs1 = 0;
    let mut prev_obs2 = 0;
    
    for step in 0..30 {
        // Advance environment step
        test_env.advance_step();
        
        // ----- Agent 1's turn -----
        let messages_for_agent1 = test_env.comm_channel.receive_all(0)?;
        let action1 = agent1.act_with_communication(prev_obs1, messages_for_agent1)?;
        let (reward1, _) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut bandit_env, 0, action1)?;
        prev_obs1 = reward1;
        test_env.record_action(0, action1, reward1);

        // Agent 1 sends its action and reward
        test_env.comm_channel.send(0, 1, MessageContent::Action(action1))?;
        test_env.comm_channel.send(0, 1, MessageContent::Reward(reward1 as f64))?;
        
        // ----- Agent 2's turn -----
        let messages_for_agent2 = test_env.comm_channel.receive_all(1)?;
        let action2 = agent2.act_with_communication(prev_obs2, messages_for_agent2)?;
        let (reward2, _) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut bandit_env, 1, action2)?;
        prev_obs2 = reward2;
        test_env.record_action(1, action2, reward2);
        
        // Agent 2 sends its action and reward
        test_env.comm_channel.send(1, 0, MessageContent::Action(action2))?;
        test_env.comm_channel.send(1, 0, MessageContent::Reward(reward2 as f64))?;
    }
    
    // Count actions for each agent
    let count_actions = |actions: &Vec<usize>| {
        let mut counts = vec![0; n_bandits];
        for &a in actions {
            counts[a] += 1;
        }
        counts
    };
    
    let agent1_actions = count_actions(&test_env.actions[0]);
    let agent2_actions = count_actions(&test_env.actions[1]);
    
    println!("Agent 1 action distribution: {:?}", agent1_actions);
    println!("Agent 2 action distribution: {:?}", agent2_actions);
    
    // VALIDATION: In cooperative mode, agents should follow their preferences since there's no conflict
    
    // Agent 1 should prefer bandit 0 (its preference)
    assert!(agent1_actions[0] >= agent1_actions[1] && agent1_actions[0] >= agent1_actions[2],
        "Agent 1 should prefer bandit 0");
    
    // Agent 2 should prefer bandit 2 (its preference)
    assert!(agent2_actions[2] >= agent2_actions[0] && agent2_actions[2] >= agent2_actions[1],
        "Agent 2 should prefer bandit 2");
    
    Ok(())
} 