// Tests from aif/src/group.rs that require BanditEnvironment (lives in reproduce).
// See the crate-level note in `reproduce/src/lib.rs`: every cast is a `usize as f64` on a
// trial/agent count, all far below 2^53 (issue #11 pedantic burn-down).
#![allow(clippy::cast_precision_loss)]

use reproduce::{
    Agent, BanditEnvironment, Environment, GroupAgentBuilder, AifError, VotingMode, env_seed,
    group_seed,
};

/// Shannon entropy (nats) of an action-count histogram — the concentration measure used
/// by the certainty-weighting test below. Lower = more concentrated.
fn count_entropy(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    assert!(total > 0, "entropy of an empty histogram is undefined");
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total as f64;
            -p * p.ln()
        })
        .sum()
}

#[test]
fn test_group_agent_certainty_weighted_mode() -> Result<(), AifError> {
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

/// Under CONFLICTING preferences, certainty-weighted voting is less noisy than simple
/// probabilistic voting: confident members dominate the mixture instead of each casting
/// one equal discrete vote, so the group's action histogram is more concentrated.
///
/// Matched-pair design (issue #8, mirroring `simulation.rs`'s Figure-6 assertion): both
/// arms are built from the SAME seed and each gets its OWN identically seeded environment,
/// so the two runs differ only in voting mode. (The previous version shared one entropy-
/// seeded environment sequentially — the CW arm continued the simple arm's reward stream,
/// so the two were neither matched nor reproducible, and the computed histograms were
/// never actually compared.)
///
/// Seeds checked while writing this test — CW was strictly more concentrated at all of
/// them, by a wide margin:
///   2026     → simple max  98, CW max 125   (chosen)
///   7        → simple max 108, CW max 134
///   815      → simple max  96, CW max 133
///   4242     → simple max  85, CW max 123
///   20260211 → simple max  98, CW max 136
#[test]
fn test_certainty_weighted_conflicting_prefs_less_noisy_than_simple() -> Result<(), AifError>
{
    const SEED: u64 = 2026;
    const N_TRIALS: usize = 200;

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

    let run = |certainty_weighted: bool| -> Result<Vec<usize>, AifError> {
        let mut builder = GroupAgentBuilder::new(3)
            .n_internal(n)
            .observation_probs(vec![0.8, 0.2, 0.2])
            .alpha(0.5)
            .seed(group_seed(SEED));
        if certainty_weighted {
            builder = builder.certainty_weighted(true);
        }
        let mut group = builder.build_varying_preferences(&pref_sets)?;
        let mut env = BanditEnvironment::with_seed(vec![0.8, 0.2, 0.2], env_seed(SEED))?;
        let mut obs = 0;
        let mut counts = vec![0usize; 3];
        for _ in 0..N_TRIALS {
            let a = group.act(obs)?;
            counts[a] += 1;
            obs = env.step(a)?;
        }
        Ok(counts)
    };

    let simple_counts = run(false)?;
    let cw_counts = run(true)?;

    println!("Simple voting (conflicting prefs): {simple_counts:?}");
    println!("CW voting (conflicting prefs):     {cw_counts:?}");

    assert_eq!(simple_counts.iter().sum::<usize>(), N_TRIALS);
    assert_eq!(cw_counts.iter().sum::<usize>(), N_TRIALS);

    // The named property, two ways: modal mass and histogram entropy.
    let simple_max = *simple_counts.iter().max().expect("3 actions ⇒ non-empty");
    let cw_max = *cw_counts.iter().max().expect("3 actions ⇒ non-empty");
    assert!(
        cw_max >= simple_max,
        "CW voting must be at least as concentrated as simple voting (seed {SEED}): \
         CW max {cw_max} {cw_counts:?} vs simple max {simple_max} {simple_counts:?}"
    );

    let simple_h = count_entropy(&simple_counts);
    let cw_h = count_entropy(&cw_counts);
    println!("action-count entropy: simple={simple_h:.4} nats, CW={cw_h:.4} nats");
    assert!(
        cw_h <= simple_h,
        "CW voting must not be noisier than simple voting (seed {SEED}): \
         CW H={cw_h:.4} vs simple H={simple_h:.4}"
    );
    Ok(())
}

#[test]
fn test_group_agent_acts_in_environment() -> Result<(), AifError> {
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
fn test_group_agent_deterministic_voting_more_decisive() -> Result<(), AifError> {
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
fn test_bandit_environment() -> Result<(), AifError> {
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
    let observed_prob = f64::from(preferred) / f64::from(n_trials);
    assert_relative_eq!(observed_prob, 0.8, epsilon = 0.05);
    Ok(())
}
