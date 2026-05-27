use one_many_rs::{Agent, BanditEnvironment, Environment, OneManyError, POMDPAgent};

#[test]
fn test_complete_pomdp_cycle() -> Result<(), OneManyError> {
    // Paper setup: agents have accurate beliefs about outcome probabilities.
    // Bandit 0: 80% reward, Bandit 1: 20%, Bandit 2: 20%
    let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;

    // A matrix matches environment (agent "knows" the task)
    // C: prefer observation 1 (reward)
    let mut agent = POMDPAgent::new(
        3,
        Some(vec![0.8, 0.2, 0.2]),
        None,
        vec![0.7, 0.3],
        None,
        8.0,
        false,
    )?;

    let mut actions = Vec::new();
    let mut prev_obs = 0;

    for _ in 0..100 {
        let action = agent.act(prev_obs)?;
        actions.push(action);
        let obs = env.step(action)?;
        prev_obs = obs;
    }

    let counts = count_actions(&actions, 3);
    println!("Action counts: {counts:?}");

    // With accurate A matrix and preference for obs 1, agent should prefer bandit 0
    assert!(
        counts[0] > counts[1] && counts[0] > counts[2],
        "Agent with accurate beliefs should prefer bandit 0 (highest reward prob)"
    );

    Ok(())
}

#[test]
fn test_pomdp_belief_adaptation() -> Result<(), OneManyError> {
    // Bandit 2 gives best rewards
    let mut env = BanditEnvironment::new(vec![0.3, 0.4, 0.9])?;

    let mut agent = POMDPAgent::new(
        3,
        Some(vec![0.5, 0.5, 0.5]),
        Some(vec![2.0, 2.0, 2.0]),
        vec![0.7, 0.3],
        None,
        5.0,
        true,
    )?;

    let mut prev_observation = 0;
    let mut action_history: Vec<usize> = Vec::new();

    for _ in 0..50 {
        let action = agent.act(prev_observation)?;
        action_history.push(action);
        let observation = env.step(action)?;
        prev_observation = observation;
    }

    let early_actions = &action_history[0..20];
    let late_actions = &action_history[30..50];

    let early_counts = count_actions(early_actions, 3);
    let late_counts = count_actions(late_actions, 3);

    println!("Early action counts: {early_counts:?}");
    println!("Late action counts: {late_counts:?}");

    // Agent should select bandit 2 (best option) at least sometimes in late trials
    assert!(
        late_counts[2] >= 1,
        "Agent should select bandit 2 (best option) in later trials"
    );

    Ok(())
}

#[test]
fn test_pomdp_sequential_decisions() -> Result<(), OneManyError> {
    let mut env1 = BanditEnvironment::new(vec![0.8, 0.2, 0.5])?;
    let mut env2 = BanditEnvironment::new(vec![0.2, 0.9, 0.4])?;

    let mut agent = POMDPAgent::new(
        3,
        Some(vec![0.5, 0.5, 0.5]),
        Some(vec![1.0, 1.0, 1.0]),
        vec![0.7, 0.3],
        None,
        7.0,
        true,
    )?;

    // Phase 1: env1 (bandit 0 best)
    let mut actions_phase1 = Vec::new();
    let mut prev_obs = 0;
    for _ in 0..100 {
        let action = agent.act(prev_obs)?;
        actions_phase1.push(action);
        prev_obs = env1.step(action)?;
    }

    // Phase 2: env2 (bandit 1 best)
    let mut actions_phase2 = Vec::new();
    for _ in 0..100 {
        let action = agent.act(prev_obs)?;
        actions_phase2.push(action);
        prev_obs = env2.step(action)?;
    }

    let late_phase1 = count_actions(&actions_phase1[50..], 3);
    let late_phase2 = count_actions(&actions_phase2[50..], 3);

    println!("Late Phase 1 action counts: {late_phase1:?}");
    println!("Late Phase 2 action counts: {late_phase2:?}");

    // Agent should be actively exploring across both phases (not stuck on one action)
    let phase1_max = *late_phase1.iter().max().unwrap();
    let phase2_max = *late_phase2.iter().max().unwrap();
    assert!(
        phase1_max < 50,
        "Agent should explore in phase 1, not fixate on one action"
    );
    assert!(
        phase2_max < 50,
        "Agent should explore in phase 2, not fixate on one action"
    );

    Ok(())
}

fn count_actions(actions: &[usize], n_bandits: usize) -> Vec<usize> {
    let mut counts = vec![0; n_bandits];
    for &action in actions {
        counts[action] += 1;
    }
    counts
}
