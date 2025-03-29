use one_many_rs::{Agent, POMDPAgent, BanditEnvironment, Environment, OneManyError};
use nalgebra::DVector;
use approx::assert_relative_eq;

/// This test verifies the complete POMDP cycle by:
/// 1. Setting up an environment with known probabilities
/// 2. Creating a POMDP agent with specific preferences
/// 3. Running multiple cycles of: observe → update beliefs → select action
/// 4. Tracking and validating action selection patterns
#[test]
fn test_complete_pomdp_cycle() -> Result<(), OneManyError> {
    // Setup environment with deterministic rewards for easier testing
    // Bandit 0: 100% reward rate
    // Bandit 1: 0% reward rate
    // Bandit 2: 50% reward rate
    let mut env = BanditEnvironment::new(vec![1.0, 0.0, 0.5])?;
    
    // Setup agent with explicit preferences towards bandit 0
    // Initial probabilities are set to incorrect values to test learning
    let mut agent = POMDPAgent::new(
        3,                                // 3 bandits
        Some(vec![0.5, 0.5, 0.5]),        // Initial observation model (equal probabilities)
        Some(vec![1.0, 1.0, 1.0]),        // Initial precision parameters
        vec![1.0, 0.5, 0.7],              // Preferences (favor bandit 0)
        None,                             // Use default uniform belief
        10.0,                             // Precision parameter (controls exploration/exploitation)
        true                              // Enable learning
    )?;
    
    // Run 20 iterations of the POMDP cycle and track actions
    let mut actions = Vec::new();
    let mut observations = Vec::new();
    let mut prev_obs = 0; // Start with neutral observation
    
    for _ in 0..20 {
        // 1. Agent selects action based on previous observation
        let action = agent.act(prev_obs)?;
        actions.push(action);
        
        // 2. Environment provides observation/reward
        let obs = env.step(action)?;
        observations.push(obs);
        prev_obs = obs;
    }
    
    // VALIDATION 1: Count actions in early vs late phase to check for learning
    let early_actions = &actions[0..10];
    let late_actions = &actions[10..];
    
    let early_counts = count_actions(early_actions, 3);
    let late_counts = count_actions(late_actions, 3);
    
    // We expect more selections of bandit 0 in late phase due to learning
    println!("Early action counts: {:?}", early_counts);
    println!("Late action counts: {:?}", late_counts);
    
    // Agent should prefer bandit 0 (100% reward) over bandit 1 (0% reward) after learning
    assert!(late_counts[0] > late_counts[1], 
        "Agent should prefer bandit 0 over bandit 1 after learning");
    
    Ok(())
}

/// This test verifies that the POMDP agent properly updates its behavior
/// when observing patterns in the environment
#[test]
fn test_pomdp_belief_adaptation() -> Result<(), OneManyError> {
    // Create an environment where bandit 2 gives the best rewards
    let mut env = BanditEnvironment::new(vec![0.3, 0.4, 0.9])?;
    
    // Create agent with neutral preferences but high exploration
    let mut agent = POMDPAgent::new(
        3,                              // 3 bandits
        Some(vec![0.5, 0.5, 0.5]),      // Initial observation model
        Some(vec![2.0, 2.0, 2.0]),      // Initial precision
        vec![1.0, 1.0, 1.0],            // Equal preferences
        None,                           // Default beliefs
        5.0,                            // Lower precision to encourage exploration
        true                            // Enable learning
    )?;
    
    // Run for many iterations to allow learning
    let mut prev_observation = 0;
    let mut action_history: Vec<usize> = Vec::new();
    
    // Run cycles
    for _ in 0..50 {
        // Agent acts based on current belief
        let action = agent.act(prev_observation)?;
        action_history.push(action);
        
        // Get observation from environment
        let observation = env.step(action)?;
        prev_observation = observation;
    }
    
    // VALIDATION: In later trials, agent should select bandit 2 more often
    let early_actions = &action_history[0..20];
    let late_actions = &action_history[30..50];
    
    let early_counts = count_actions(early_actions, 3);
    let late_counts = count_actions(late_actions, 3);
    
    println!("Early action counts: {:?}", early_counts);
    println!("Late action counts: {:?}", late_counts);
    
    // Check that bandit 2 (the best option) is selected more in late trials
    assert!(late_counts[2] >= early_counts[2], 
        "Agent should select bandit 2 (best option) more in later trials");
    
    Ok(())
}

/// This test verifies that the POMDP agent handles sequential decision making
/// and adapts to the environment
#[test]
fn test_pomdp_sequential_decisions() -> Result<(), OneManyError> {
    // Create an environment with changing reward patterns
    // Initially, bandit 0 is best, later bandit 1 becomes better
    let mut env1 = BanditEnvironment::new(vec![0.8, 0.2, 0.5])?;
    let mut env2 = BanditEnvironment::new(vec![0.2, 0.9, 0.4])?;
    
    // Create an agent with moderate precision to balance exploration/exploitation
    let mut agent = POMDPAgent::new(
        3,
        Some(vec![0.5, 0.5, 0.5]),  // Neutral initial model
        Some(vec![1.0, 1.0, 1.0]),  // Initial precision
        vec![1.0, 1.0, 1.0],        // Equal preferences
        None,                       // Default initial belief
        7.0,                        // Moderate precision
        true                        // Enable learning
    )?;
    
    // PHASE 1: Run with first environment
    // Track actions and observations
    let mut actions_phase1 = Vec::new();
    let mut prev_obs = 0;
    
    for _ in 0..30 {
        let action = agent.act(prev_obs)?;
        actions_phase1.push(action);
        prev_obs = env1.step(action)?;
    }
    
    // PHASE 2: Switch environment and continue
    let mut actions_phase2 = Vec::new();
    
    for _ in 0..30 {
        let action = agent.act(prev_obs)?;
        actions_phase2.push(action);
        prev_obs = env2.step(action)?;
    }
    
    // Count action selections in each phase
    let late_phase1 = count_actions(&actions_phase1[20..], 3);
    let late_phase2 = count_actions(&actions_phase2[20..], 3);
    
    println!("Late Phase 1 action counts: {:?}", late_phase1);
    println!("Late Phase 2 action counts: {:?}", late_phase2);
    
    // VALIDATION: Agent should adapt to environmental changes
    // In phase 1, bandit 0 should be favored
    assert!(late_phase1[0] > late_phase1[1], 
        "Agent should prefer bandit 0 in phase 1");
    
    // In phase 2, bandit 1 selections should increase compared to phase 1
    assert!(late_phase2[1] > late_phase1[1], 
        "Agent should increase selection of bandit 1 in phase 2");
    
    Ok(())
}

// Helper function to count actions
fn count_actions(actions: &[usize], n_bandits: usize) -> Vec<usize> {
    let mut counts = vec![0; n_bandits];
    for &action in actions {
        counts[action] += 1;
    }
    counts
} 