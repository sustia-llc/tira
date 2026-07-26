use reproduce::{
    Agent, MultiAgentEnvironment, AifError, POMDPAgent, SharedBanditEnvironment,
};

struct AgentTracker {
    actions: Vec<usize>,
    rewards: Vec<usize>,
    total_reward: usize,
}

impl AgentTracker {
    fn new() -> Self {
        Self {
            actions: Vec::new(),
            rewards: Vec::new(),
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

#[test]
fn test_competitive_multi_agent() -> Result<(), AifError> {
    let mut env = SharedBanditEnvironment::new(vec![0.8, 0.4, 0.6], 2)?;

    let mut agent0 = POMDPAgent::new(
        3,
        Some(vec![0.8, 0.3, 0.5]),
        None,
        vec![0.8, 0.2],
        None,
        8.0,
        false,
    )?;

    let mut agent1 = POMDPAgent::new(
        3,
        Some(vec![0.3, 0.3, 0.8]),
        None,
        vec![0.8, 0.2],
        None,
        8.0,
        false,
    )?;

    let mut tracker0 = AgentTracker::new();
    let mut tracker1 = AgentTracker::new();

    let mut prev_obs0 = 0;
    let mut prev_obs1 = 0;

    for _ in 0..50 {
        let action0 = agent0.act(prev_obs0)?;
        let (reward0, _) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(
            &mut env, 0, action0,
        )?;
        tracker0.record(action0, reward0);
        prev_obs0 = reward0;

        let action1 = agent1.act(prev_obs1)?;
        match <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut env, 1, action1) {
            Ok((reward1, _)) => {
                tracker1.record(action1, reward1);
                prev_obs1 = reward1;
            }
            Err(AifError::ResourceConflict(_)) => {
                let mut found = false;
                for alt in 0..3 {
                    if alt != action1 {
                        match <SharedBanditEnvironment as MultiAgentEnvironment>::step(
                            &mut env, 1, alt,
                        ) {
                            Ok((reward, _)) => {
                                tracker1.record(alt, reward);
                                prev_obs1 = reward;
                                found = true;
                                break;
                            }
                            // Contended arm: fall through to the next alternative.
                            Err(AifError::ResourceConflict(_)) => {}
                            Err(e) => return Err(e),
                        }
                    }
                }
                if !found {
                    tracker1.record(action1, 0);
                    prev_obs1 = 0;
                }
            }
            Err(e) => return Err(e),
        }
    }

    let action_counts0 = tracker0.action_counts(3);
    let action_counts1 = tracker1.action_counts(3);

    println!("Agent 0 action counts: {action_counts0:?}");
    println!("Agent 1 action counts: {action_counts1:?}");
    println!("Agent 0 total reward: {}", tracker0.total_reward);
    println!("Agent 1 total reward: {}", tracker1.total_reward);

    // Agent 0 should prefer bandit 0 (highest obs-1 prob in its A matrix)
    assert!(
        action_counts0[0] > action_counts0[1],
        "Agent 0 should prefer bandit 0 over bandit 1"
    );

    // Agent 1 should prefer bandit 2 (highest obs-1 prob in its A matrix)
    assert!(
        action_counts1[2] > action_counts1[1],
        "Agent 1 should prefer bandit 2 over bandit 1"
    );

    Ok(())
}

#[test]
fn test_non_competitive_multi_agent() -> Result<(), AifError> {
    let mut env = SharedBanditEnvironment::new(vec![0.8, 0.4, 0.6], 2)?;
    env.set_competitive(false);

    let mut agent0 = POMDPAgent::new(
        3,
        Some(vec![0.8, 0.3, 0.5]),
        None,
        vec![0.8, 0.2],
        None,
        8.0,
        false,
    )?;

    let mut agent1 = POMDPAgent::new(
        3,
        Some(vec![0.8, 0.3, 0.5]),
        None,
        vec![0.8, 0.2],
        None,
        8.0,
        false,
    )?;

    let mut tracker0 = AgentTracker::new();
    let mut tracker1 = AgentTracker::new();

    let mut prev_obs0 = 0;
    let mut prev_obs1 = 0;

    for _ in 0..50 {
        let action0 = agent0.act(prev_obs0)?;
        let (reward0, _) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(
            &mut env, 0, action0,
        )?;
        tracker0.record(action0, reward0);
        prev_obs0 = reward0;

        let action1 = agent1.act(prev_obs1)?;
        let (reward1, _) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(
            &mut env, 1, action1,
        )?;
        tracker1.record(action1, reward1);
        prev_obs1 = reward1;
    }

    let action_counts0 = tracker0.action_counts(3);
    let action_counts1 = tracker1.action_counts(3);

    println!("Agent 0 action counts: {action_counts0:?}");
    println!("Agent 1 action counts: {action_counts1:?}");

    // Both should prefer bandit 0 (highest obs-1 prob)
    assert!(
        action_counts0[0] > action_counts0[1],
        "Agent 0 should prefer bandit 0 over bandit 1"
    );
    assert!(
        action_counts1[0] > action_counts1[1],
        "Agent 1 should prefer bandit 0 over bandit 1"
    );

    Ok(())
}

#[test]
fn test_sequential_communication() -> Result<(), AifError> {
    let mut env = SharedBanditEnvironment::new(vec![0.7, 0.7, 0.7], 2)?;
    env.set_competitive(true);

    let mut leader = POMDPAgent::new(
        3,
        Some(vec![0.9, 0.3, 0.3]),
        None,
        vec![0.8, 0.2],
        None,
        12.0,
        false,
    )?;

    let mut follower = POMDPAgent::new(
        3,
        Some(vec![0.5, 0.5, 0.5]),
        None,
        vec![0.6, 0.4],
        None,
        5.0,
        false,
    )?;

    let mut leader_tracker = AgentTracker::new();
    let mut follower_tracker = AgentTracker::new();

    let mut leader_obs = 0;
    let mut follower_obs = 0;

    for _ in 0..60 {
        let leader_action = leader.act(leader_obs)?;
        let (leader_reward, _) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(
            &mut env,
            0,
            leader_action,
        )?;
        leader_tracker.record(leader_action, leader_reward);
        leader_obs = leader_reward;

        // Follower tries to act; falls back on conflict
        let follower_action = follower.act(follower_obs)?;
        let mut done = false;

        // Try the chosen action first, then alternatives
        for alt in std::iter::once(follower_action).chain((0..3).filter(|&a| a != follower_action))
        {
            match <SharedBanditEnvironment as MultiAgentEnvironment>::step(&mut env, 1, alt) {
                Ok((reward, _)) => {
                    follower_tracker.record(alt, reward);
                    follower_obs = reward;
                    done = true;
                    break;
                }
                // Contended arm: fall through to the next alternative.
                Err(AifError::ResourceConflict(_)) => {}
                Err(e) => return Err(e),
            }
        }
        if !done {
            follower_tracker.record(follower_action, 0);
            follower_obs = 0;
        }
    }

    let leader_actions = leader_tracker.action_counts(3);
    let follower_actions = follower_tracker.action_counts(3);

    println!("Leader action counts: {leader_actions:?}");
    println!("Follower action counts: {follower_actions:?}");

    // Leader should prefer bandit 0 (its A matrix is most informative there)
    assert!(
        leader_actions[0] > leader_actions[1] && leader_actions[0] > leader_actions[2],
        "Leader should prefer bandit 0"
    );

    Ok(())
}
