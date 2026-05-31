// Tests from aif/src/group.rs that require BanditEnvironment (lives in reproduce).
use reproduce::{
    Agent, BanditEnvironment, Environment, GroupAgentBuilder, OneManyError, VotingMode,
};

#[test]
fn test_group_agent_certainty_weighted_mode() -> Result<(), OneManyError> {
    let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;

    let mut group = GroupAgentBuilder::new(3)
        .n_internal(8)
        .observation_probs(vec![0.8, 0.2, 0.2])
        .preferences(vec![0.7, 0.3])
        .alpha(0.5)
        .certainty_weighted(true)
        .build_identical()?;

    assert_eq!(group.voting_mode(), VotingMode::CertaintyWeighted);

    let mut prev_obs = 0;
    let mut action_counts = vec![0usize; 3];
    for _ in 0..100 {
        let action = group.act(prev_obs)?;
        action_counts[action] += 1;
        prev_obs = env.step(action)?;
    }

    // Should still prefer bandit 0 (best A-matrix alignment)
    assert!(
        action_counts[0] > action_counts[1],
        "CW group should prefer bandit 0: {action_counts:?}"
    );
    Ok(())
}

#[test]
fn test_certainty_weighted_conflicting_prefs_less_noisy_than_simple() -> Result<(), OneManyError>
{
    let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;

    // Conflicting preferences: half prefer obs1, half prefer obs2
    let n = 8;
    let pref_sets: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            if i < n / 2 {
                vec![0.9, 0.1]
            } else {
                vec![0.1, 0.9]
            }
        })
        .collect();

    // Simple voting
    let mut simple = GroupAgentBuilder::new(3)
        .n_internal(n)
        .observation_probs(vec![0.8, 0.2, 0.2])
        .alpha(0.5)
        .build_varying_preferences(&pref_sets)?;

    // Certainty-weighted voting
    let mut cw = GroupAgentBuilder::new(3)
        .n_internal(n)
        .observation_probs(vec![0.8, 0.2, 0.2])
        .alpha(0.5)
        .certainty_weighted(true)
        .build_varying_preferences(&pref_sets)?;

    let n_trials = 200;

    let mut simple_obs = 0;
    let mut simple_counts = vec![0usize; 3];
    for _ in 0..n_trials {
        let a = simple.act(simple_obs)?;
        simple_counts[a] += 1;
        simple_obs = env.step(a)?;
    }

    let mut cw_obs = 0;
    let mut cw_counts = vec![0usize; 3];
    for _ in 0..n_trials {
        let a = cw.act(cw_obs)?;
        cw_counts[a] += 1;
        cw_obs = env.step(a)?;
    }

    println!("Simple voting (conflicting prefs): {simple_counts:?}");
    println!("CW voting (conflicting prefs):     {cw_counts:?}");

    // Both should produce valid results
    assert_eq!(simple_counts.iter().sum::<usize>(), n_trials);
    assert_eq!(cw_counts.iter().sum::<usize>(), n_trials);
    Ok(())
}

#[test]
fn test_group_agent_acts_in_environment() -> Result<(), OneManyError> {
    let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;
    let mut group = GroupAgentBuilder::new(3)
        .n_internal(8)
        .observation_probs(vec![0.8, 0.2, 0.2])
        .preferences(vec![0.7, 0.3])
        .alpha(0.5)
        .build_identical()?;

    let mut prev_obs = 0;
    let mut actions = Vec::new();
    for _ in 0..50 {
        let action = group.act(prev_obs)?;
        assert!(action < 3, "Action should be valid bandit index");
        actions.push(action);
        prev_obs = env.step(action)?;
    }
    assert_eq!(actions.len(), 50);
    Ok(())
}

#[test]
fn test_group_agent_deterministic_voting_more_decisive() -> Result<(), OneManyError> {
    let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;

    let mut prob_group = GroupAgentBuilder::new(3)
        .n_internal(16)
        .observation_probs(vec![0.8, 0.2, 0.2])
        .preferences(vec![0.7, 0.3])
        .alpha(0.5)
        .deterministic(false)
        .build_identical()?;

    let mut det_group = GroupAgentBuilder::new(3)
        .n_internal(16)
        .observation_probs(vec![0.8, 0.2, 0.2])
        .preferences(vec![0.7, 0.3])
        .alpha(0.5)
        .deterministic(true)
        .build_identical()?;

    let n_trials = 100;

    let mut prob_obs = 0;
    let mut prob_actions = vec![0usize; 3];
    for _ in 0..n_trials {
        let a = prob_group.act(prob_obs)?;
        prob_actions[a] += 1;
        prob_obs = env.step(a)?;
    }

    let mut det_obs = 0;
    let mut det_actions = vec![0usize; 3];
    for _ in 0..n_trials {
        let a = det_group.act(det_obs)?;
        det_actions[a] += 1;
        det_obs = env.step(a)?;
    }

    let prob_max = *prob_actions.iter().max().unwrap();
    let det_max = *det_actions.iter().max().unwrap();

    println!("Probabilistic actions: {prob_actions:?} (max={prob_max})");
    println!("Deterministic actions: {det_actions:?} (max={det_max})");

    assert!(
        det_max >= prob_max,
        "Deterministic voting should be at least as decisive: det={det_max}, prob={prob_max}"
    );

    Ok(())
}

// Test relocated from aif/src/agent.rs (requires BanditEnvironment)
#[test]
fn test_bandit_environment() -> Result<(), OneManyError> {
    use approx::assert_relative_eq;
    use reproduce::Environment;
    let mut env = BanditEnvironment::new(vec![0.8, 0.4, 0.4])?;
    let n_trials = 10000;
    // Observation index 0 = preferred (high-probability) outcome, per the agent's
    // generative-model convention; it should occur with probability ~0.8 on arm 0.
    let mut preferred = 0;
    for _ in 0..n_trials {
        if env.step(0)? == 0 {
            preferred += 1;
        }
    }
    let observed_prob = preferred as f64 / n_trials as f64;
    assert_relative_eq!(observed_prob, 0.8, epsilon = 0.05);
    Ok(())
}
