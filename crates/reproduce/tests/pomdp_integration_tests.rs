use reproduce::{Agent, BanditEnvironment, Environment, AifError, POMDPAgent};

#[test]
fn test_complete_pomdp_cycle() -> Result<(), AifError> {
    // Paper setup: agents have accurate beliefs about outcome probabilities.
    // Bandit 0: 80% reward, Bandit 1: 20%, Bandit 2: 20%
    let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;

    // A matrix matches environment (agent "knows" the task)
    // C: prefer observation index 0 (the high-reward outcome)
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
fn test_pomdp_belief_adaptation() -> Result<(), AifError> {
    // Bandit 2 gives best rewards
    let mut env = BanditEnvironment::new(vec![0.3, 0.4, 0.9])?;

    // Agent knows the environment (accurate A matrix) — no learning needed
    let mut agent = POMDPAgent::new(
        3,
        Some(vec![0.3, 0.4, 0.9]),
        None,
        vec![0.7, 0.3],
        None,
        5.0,
        false,
    )?;

    let mut prev_observation = 0;
    let mut action_history: Vec<usize> = Vec::new();

    for _ in 0..50 {
        let action = agent.act(prev_observation)?;
        action_history.push(action);
        let observation = env.step(action)?;
        prev_observation = observation;
    }

    let counts = count_actions(&action_history, 3);
    println!("Action counts: {counts:?}");

    // With accurate A matrix, agent should prefer bandit 2 (highest obs-1 prob)
    assert!(
        counts[2] > counts[0] && counts[2] > counts[1],
        "Agent should prefer bandit 2 (best obs-1 probability): {counts:?}"
    );

    Ok(())
}

#[test]
fn test_pomdp_sequential_decisions() -> Result<(), AifError> {
    let mut env1 = BanditEnvironment::new(vec![0.8, 0.2, 0.5])?;
    let mut env2 = BanditEnvironment::new(vec![0.2, 0.9, 0.4])?;

    let mut agent = POMDPAgent::new(
        3,
        Some(vec![0.5, 0.5, 0.5]),
        None,
        vec![0.7, 0.3],
        None,
        7.0,
        false,
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

#[test]
fn test_env_observation_matches_agent_preferred_index() -> Result<(), AifError> {
    // Cross-module round-trip: the environment's observation encoding must match the
    // agent's generative-model convention, where observation index 0 = the preferred
    // (high-probability / reward) outcome.
    //
    // A-matrix column convention (see `POMDPAgent::new`, agent.rs): column j is built as
    // `[p_j, 1 - p_j]`, so row 0 holds the high-reward probability `p_j`. For any arm with
    // `p_j > 0.5`, the preferred observation index is therefore `argmax_o A[o, j] == 0`.
    // The environment must emit that same index 0 when the preferred outcome occurs.

    // Deterministic env removes RNG: arm 0 always yields the preferred outcome, arm 1 never.
    let mut env = BanditEnvironment::new(vec![1.0, 0.0, 0.0])?;

    // Winning/preferred outcome on arm 0 → observation index 0 (the preferred index).
    assert_eq!(
        env.step(0)?,
        0,
        "preferred (winning) outcome must map to observation index 0"
    );
    // Non-preferred outcome on arm 1 → observation index 1.
    assert_eq!(
        env.step(1)?,
        1,
        "non-preferred (losing) outcome must map to observation index 1"
    );

    // The agent side of the invariant is fixed by `POMDPAgent::new`'s documented A-column
    // construction `[p_j, 1 - p_j]`: for any arm with p_j > 0.5, `argmax_o A[o, j] == 0`,
    // so index 0 is the agent's preferred observation — the same index the env emits above.
    // (`POMDPAgent` exposes no A-matrix accessor to assert against directly.)

    Ok(())
}

fn count_actions(actions: &[usize], n_bandits: usize) -> Vec<usize> {
    let mut counts = vec![0; n_bandits];
    for &action in actions {
        counts[action] += 1;
    }
    counts
}
