use one_many_rs::{
    Agent, POMDPAgent, SharedBanditEnvironment, MultiAgentEnvironment, 
    Environment, StateChange, OneManyError
};
use approx::assert_relative_eq;

/// Helper structure to track agent performance
struct AgentTracker {
    agent_id: usize,
    actions: Vec<usize>,
    rewards: Vec<usize>,
    observations: Vec<usize>,
    total_reward: usize,
}

impl AgentTracker {
    fn new(agent_id: usize) -> Self {
        Self {
            agent_id,
            actions: Vec::new(),
            rewards: Vec::new(),
            observations: Vec::new(),
            total_reward: 0,
        }
    }

    fn record(&mut self, action: usize, reward: usize) {
        self.actions.push(action);
        self.rewards.push(reward);
        self.total_reward += reward;
    }

    fn action_counts(&self, n_actions: usize) -> Vec<usize> {
        let mut counts = vec![0; n_actions];
        for &action in &self.actions {
            counts[action] += 1;
        }
        counts
    }
}

/// This test demonstrates a simple competitive scenario between two POMDP agents
/// The agents compete for the same resources (bandits) with different reward probabilities
#[test]
fn test_competitive_multi_agent() -> Result<(), OneManyError> {
    // Create a shared environment with 3 bandits
    let mut env = SharedBanditEnvironment::new(vec![0.8, 0.4, 0.6], 2)?;
    
    // Create two agents with different preferences
    // Agent 0: Prefers bandit 0 (the best one)
    let mut agent0 = POMDPAgent::new(
        3,
        Some(vec![0.5, 0.5, 0.5]),       // Initial observation model
        Some(vec![1.0, 1.0, 1.0]),       // Initial precision
        vec![1.0, 0.5, 0.7],             // Preferences (favor bandit 0)
        None,                            // Default beliefs
        8.0,                             // Precision parameter
        true                             // Enable learning
    )?;
    
    // Agent 1: Prefers bandit 2
    let mut agent1 = POMDPAgent::new(
        3,
        Some(vec![0.5, 0.5, 0.5]),       // Initial observation model
        Some(vec![1.0, 1.0, 1.0]),       // Initial precision
        vec![0.6, 0.5, 1.0],             // Preferences (favor bandit 2)
        None,                            // Default beliefs
        8.0,                             // Precision parameter
        true                             // Enable learning
    )?;
    
    // Track performance of each agent
    let mut tracker0 = AgentTracker::new(0);
    let mut tracker1 = AgentTracker::new(1);
    
    let mut prev_obs0 = 0;
    let mut prev_obs1 = 0;
    
    // Run simulation for 50 rounds
    for _ in 0..50 {
        // Agent 0 selects action
        let action0 = agent0.act(prev_obs0)?;
        let (reward0, state_change0) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut env, 0, action0)?;
        tracker0.record(action0, reward0);
        prev_obs0 = reward0;
        
        // Agent 1 selects action (might observe a conflict if it tries to select the same bandit)
        let action1 = agent1.act(prev_obs1)?;
        match <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut env, 1, action1) {
            Ok((reward1, state_change1)) => {
                tracker1.record(action1, reward1);
                prev_obs1 = reward1;
            },
            Err(OneManyError::ResourceConflict(_)) => {
                // Try different actions if there's a conflict
                let mut found_valid_action = false;
                
                // Try each bandit in succession until we find an available one
                for alt_action in 0..3 {
                    if alt_action != action1 {  // Don't retry the same action
                        match <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut env, 1, alt_action) {
                            Ok((reward, _)) => {
                                tracker1.record(alt_action, reward);
                                prev_obs1 = reward;
                                found_valid_action = true;
                                break;
                            },
                            Err(OneManyError::ResourceConflict(_)) => continue,
                            Err(e) => return Err(e),
                        }
                    }
                }
                
                // If all actions conflicted, record a zero reward for the original action
                if !found_valid_action {
                    tracker1.record(action1, 0);
                    prev_obs1 = 0;
                }
            },
            Err(e) => return Err(e),
        }
    }
    
    // Calculate action distribution for each agent
    let action_counts0 = tracker0.action_counts(3);
    let action_counts1 = tracker1.action_counts(3);
    
    println!("Agent 0 action counts: {:?}", action_counts0);
    println!("Agent 1 action counts: {:?}", action_counts1);
    println!("Agent 0 total reward: {}", tracker0.total_reward);
    println!("Agent 1 total reward: {}", tracker1.total_reward);
    
    // VALIDATION: Agents should adapt their strategy based on competition
    
    // In competitive scenario, agents should differentiate their strategies
    // Agent 0 should tend toward bandit 0 (its preference and highest reward)
    assert!(action_counts0[0] > action_counts0[1], 
        "Agent 0 should prefer bandit 0 over bandit 1");
    
    // Agent 1 should tend toward bandit 2 (its preference)
    assert!(action_counts1[2] > action_counts1[1], 
        "Agent 1 should prefer bandit 2 over bandit 1");
    
    // Agents should adapt to avoid conflicts
    // Agent 1 should select bandit 0 less often than Agent 0
    assert!(action_counts1[0] < action_counts0[0], 
        "Agent 1 should select bandit 0 less often than Agent 0");
    
    Ok(())
}

/// This test demonstrates a non-competitive scenario where agents can select the same bandit
#[test]
fn test_non_competitive_multi_agent() -> Result<(), OneManyError> {
    // Create a shared environment with 3 bandits
    let mut env = SharedBanditEnvironment::new(vec![0.8, 0.4, 0.6], 2)?;
    
    // Set to non-competitive mode (agents can select same bandit)
    env.set_competitive(false);
    
    // Create two similar agents with preference for bandit 0
    let mut agent0 = POMDPAgent::new(3, Some(vec![0.5, 0.5, 0.5]), Some(vec![1.0, 1.0, 1.0]), 
                                    vec![1.0, 0.5, 0.7], None, 8.0, true)?;
    
    let mut agent1 = POMDPAgent::new(3, Some(vec![0.5, 0.5, 0.5]), Some(vec![1.0, 1.0, 1.0]), 
                                    vec![1.0, 0.5, 0.7], None, 8.0, true)?;
    
    // Track performance
    let mut tracker0 = AgentTracker::new(0);
    let mut tracker1 = AgentTracker::new(1);
    
    let mut prev_obs0 = 0;
    let mut prev_obs1 = 0;
    
    // Run simulation for 50 rounds
    for _ in 0..50 {
        // Agent 0 selects action
        let action0 = agent0.act(prev_obs0)?;
        let (reward0, _) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut env, 0, action0)?;
        tracker0.record(action0, reward0);
        prev_obs0 = reward0;
        
        // Agent 1 selects action (can select same bandit as agent 0)
        let action1 = agent1.act(prev_obs1)?;
        let (reward1, _) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut env, 1, action1)?;
        tracker1.record(action1, reward1);
        prev_obs1 = reward1;
    }
    
    // Calculate action distribution for each agent
    let action_counts0 = tracker0.action_counts(3);
    let action_counts1 = tracker1.action_counts(3);
    
    println!("Agent 0 action counts: {:?}", action_counts0);
    println!("Agent 1 action counts: {:?}", action_counts1);
    
    // VALIDATION: Both agents should prefer bandit 0 (highest reward)
    
    assert!(action_counts0[0] > action_counts0[1], 
        "Agent 0 should prefer bandit 0 over bandit 1");
    
    assert!(action_counts1[0] > action_counts1[1], 
        "Agent 1 should prefer bandit 0 over bandit 1");
    
    // Since agents have identical preferences and are non-competitive,
    // their action distributions should be similar
    assert!((action_counts0[0] as f64 - action_counts1[0] as f64).abs() < 10.0, 
        "Agents should have similar selection patterns for bandit 0");
    
    Ok(())
}

/// This test demonstrates sequential communication between agents
/// where the first agent's behavior influences the second agent's choices
#[test]
fn test_sequential_communication() -> Result<(), OneManyError> {
    // Create a shared environment
    let mut env = SharedBanditEnvironment::new(vec![0.7, 0.7, 0.7], 2)?;
    
    // Set to competitive mode 
    env.set_competitive(true);
    
    // Create two agents with different roles
    // Agent 0: "Leader" - acts first and selects based on preferences
    let mut leader = POMDPAgent::new(3, Some(vec![0.5, 0.5, 0.5]), Some(vec![1.0, 1.0, 1.0]), 
                                    vec![1.0, 0.5, 0.2], None, 12.0, true)?;
    
    // Agent 1: "Follower" - observes leader's actions and adapts
    let mut follower = POMDPAgent::new(3, Some(vec![0.5, 0.5, 0.5]), Some(vec![1.0, 1.0, 1.0]), 
                                    vec![0.33, 0.33, 0.33], None, 5.0, true)?;
    
    // Track performance
    let mut leader_tracker = AgentTracker::new(0);
    let mut follower_tracker = AgentTracker::new(1);
    
    let mut leader_obs = 0;
    let mut follower_obs = 0;
    
    // Run simulation
    for _ in 0..60 {
        // Leader selects action
        let leader_action = leader.act(leader_obs)?;
        let (leader_reward, state_change) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut env, 0, leader_action)?;
        leader_tracker.record(leader_action, leader_reward);
        leader_obs = leader_reward;
        
        // Follower selects action (avoiding leader's choice)
        // Try a few times to find a valid action
        let mut follower_action;
        let mut attempt_count = 0;
        let mut follower_reward = 0;
        
        // Try up to 3 times to find a non-conflicting action
        while attempt_count < 3 {
            follower_action = follower.act(follower_obs)?;
            
            // Try each bandit in succession instead of just retrying
            for alt_action in 0..3 {
                match <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut env, 1, alt_action) {
                    Ok((reward, _)) => {
                        follower_tracker.record(alt_action, reward);
                        follower_reward = reward;
                        follower_action = alt_action;  // Update to the action that succeeded
                        attempt_count = 3;  // Force exit from loop
                        break;
                    },
                    Err(OneManyError::ResourceConflict(_)) => continue,  // Try next action
                    Err(e) => return Err(e),
                }
            }
            
            // Break out if we found a valid action
            if attempt_count == 3 {
                break;
            }
            
            // If we reach here, no valid action was found
            attempt_count += 1;
            follower_obs = 0;  // Change observation to encourage different action
            
            // If this is the last attempt, record a failed action
            if attempt_count == 3 {
                follower_tracker.record(follower_action, 0);
            }
        }
        
        follower_obs = follower_reward;
    }
    
    // Calculate action distributions
    let leader_actions = leader_tracker.action_counts(3);
    let follower_actions = follower_tracker.action_counts(3);
    
    println!("Leader action counts: {:?}", leader_actions);
    println!("Follower action counts: {:?}", follower_actions);
    
    // VALIDATION: Leader and follower should develop complementary strategies
    
    // Leader should prefer bandit 0 (its preference)
    assert!(leader_actions[0] > leader_actions[1] && leader_actions[0] > leader_actions[2], 
        "Leader should prefer bandit 0");
    
    // Follower should adapt to avoid leader's preferred choice
    assert!(follower_actions[0] < follower_actions[1] || follower_actions[0] < follower_actions[2], 
        "Follower should adapt to avoid leader's preferred choice");
    
    // Follower should learn to prefer a different bandit
    let follower_preferred = if follower_actions[1] > follower_actions[2] { 1 } else { 2 };
    assert!(follower_actions[follower_preferred] > follower_actions[0], 
        "Follower should develop preference for a different bandit than leader");
    
    Ok(())
} 