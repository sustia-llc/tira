use reproduce::{
    Agent, BanditEnvironment, Environment, GroupAgentBuilder, AifError, env_seed, group_seed,
};

/// Experiment 1 setup: identical agents, verify group behaves coherently.
#[test]
fn test_experiment1_identical_agents() -> Result<(), AifError> {
    let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;

    for &n_internal in &[4, 8, 16] {
        let mut group = GroupAgentBuilder::new(3)
            .n_internal(n_internal)
            .observation_probs(vec![0.8, 0.2, 0.2])
            .preferences(vec![0.7, 0.3])
            .alpha(0.5)
            .build_identical()?;

        let mut prev_obs = 0;
        let mut action_counts = vec![0usize; 3];
        for _ in 0..200 {
            let action = group.act(prev_obs)?;
            action_counts[action] += 1;
            prev_obs = env.step(action)?;
        }

        println!("n_internal={n_internal}: actions={action_counts:?}");

        // Group should prefer bandit 0 (highest obs-1 probability in A matrix)
        assert!(
            action_counts[0] > action_counts[1],
            "n={n_internal}: group should prefer bandit 0 over 1"
        );
        assert!(
            action_counts[0] > action_counts[2],
            "n={n_internal}: group should prefer bandit 0 over 2"
        );
    }
    Ok(())
}

/// Experiment 3 setup: deterministic voting should produce more decisive behavior.
#[test]
fn test_experiment3_deterministic_vs_probabilistic() -> Result<(), AifError> {
    let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;
    let n_trials = 200;

    // Probabilistic voting
    let mut prob_group = GroupAgentBuilder::new(3)
        .n_internal(16)
        .observation_probs(vec![0.8, 0.2, 0.2])
        .preferences(vec![0.7, 0.3])
        .alpha(0.5)
        .deterministic(false)
        .build_identical()?;

    let mut prob_obs = 0;
    let mut prob_best_count = 0;
    for _ in 0..n_trials {
        let a = prob_group.act(prob_obs)?;
        if a == 0 {
            prob_best_count += 1;
        }
        prob_obs = env.step(a)?;
    }

    // Deterministic voting
    let mut det_group = GroupAgentBuilder::new(3)
        .n_internal(16)
        .observation_probs(vec![0.8, 0.2, 0.2])
        .preferences(vec![0.7, 0.3])
        .alpha(0.5)
        .deterministic(true)
        .build_identical()?;

    let mut det_obs = 0;
    let mut det_best_count = 0;
    for _ in 0..n_trials {
        let a = det_group.act(det_obs)?;
        if a == 0 {
            det_best_count += 1;
        }
        det_obs = env.step(a)?;
    }

    println!("Probabilistic best-action count: {prob_best_count}/{n_trials}");
    println!("Deterministic best-action count: {det_best_count}/{n_trials}");

    // Deterministic should pick the preferred action at least as often
    assert!(
        det_best_count >= prob_best_count,
        "Deterministic voting should be at least as decisive"
    );
    Ok(())
}

/// Experiment 4 setup: conflicting preferences should produce noisy group behavior.
#[test]
fn test_experiment4_conflicting_preferences() -> Result<(), AifError> {
    let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;

    // Create agents where half prefer obs 1 and half prefer obs 2
    let n_internal = 8;
    let pref_sets: Vec<Vec<f64>> = (0..n_internal)
        .map(|i| {
            if i < n_internal / 2 {
                vec![0.9, 0.1] // prefer obs 1
            } else {
                vec![0.1, 0.9] // prefer obs 2
            }
        })
        .collect();

    let mut group = GroupAgentBuilder::new(3)
        .n_internal(n_internal)
        .observation_probs(vec![0.8, 0.2, 0.2])
        .alpha(0.5)
        .build_varying_preferences(&pref_sets)?;

    let mut prev_obs = 0;
    let mut action_counts = vec![0usize; 3];
    for _ in 0..200 {
        let a = group.act(prev_obs)?;
        action_counts[a] += 1;
        prev_obs = env.step(a)?;
    }

    println!("Conflicting preferences: {action_counts:?}");

    // With conflicting preferences, no single action should dominate overwhelmingly
    let max_count = *action_counts.iter().max().unwrap();
    assert!(
        max_count < 180,
        "Conflicting preferences should not produce fully deterministic behavior"
    );
    Ok(())
}

/// Experiment 2 setup: heterogeneous internal α (§2.4). Despite the precision spread,
/// the group still resolves to the best arm — α-heterogeneity changes *how sharply* the
/// group acts, not *what* it prefers.
///
/// Seeded (issue #8) so the counts are reproducible; the previous version asserted only
/// that the histogram summed to 200, which is true of any run whatsoever. Seeds checked:
/// 2026 → [170, 13, 17], 815 → [175, 18, 7], 4242 → [169, 16, 15] — arm 0 dominates in
/// every case.
#[test]
fn test_experiment2_varying_alpha() -> Result<(), AifError> {
    const SEED: u64 = 2026;
    let mut env = BanditEnvironment::with_seed(vec![0.8, 0.2, 0.2], env_seed(SEED))?;

    let alphas = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6];
    let mut group = GroupAgentBuilder::new(3)
        .n_internal(8)
        .observation_probs(vec![0.8, 0.2, 0.2])
        .preferences(vec![0.7, 0.3])
        .seed(group_seed(SEED))
        .build_varying_alpha(&alphas)?;

    // The αs really are heterogeneous (the premise of Experiment 2), and the builder
    // threads each one through to its own internal agent.
    let min_alpha = alphas.iter().copied().fold(f64::INFINITY, f64::min);
    let max_alpha = alphas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        min_alpha < max_alpha,
        "Experiment 2 requires heterogeneous α: min={min_alpha} max={max_alpha}"
    );
    for (i, agent) in group.internal_agents().iter().enumerate() {
        assert!(
            (agent.alpha() - alphas[i]).abs() < 1e-12,
            "internal agent {i} must carry α={}, got {}",
            alphas[i],
            agent.alpha()
        );
    }

    let mut prev_obs = 0;
    let mut action_counts = vec![0usize; 3];
    for _ in 0..200 {
        let a = group.act(prev_obs)?;
        assert!(a < 3, "action must be a valid bandit index, got {a}");
        action_counts[a] += 1;
        prev_obs = env.step(a)?;
    }

    println!("Varying alpha (seed {SEED}): {action_counts:?}");

    assert_eq!(
        action_counts.iter().sum::<usize>(),
        200,
        "Should have exactly 200 total actions"
    );
    // The real claim: the group still prefers the best arm, strictly over both others.
    assert!(
        action_counts[0] > action_counts[1] && action_counts[0] > action_counts[2],
        "α-heterogeneous group should still prefer bandit 0: {action_counts:?}"
    );
    Ok(())
}
