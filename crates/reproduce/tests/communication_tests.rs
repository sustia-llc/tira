// See the crate-level note in `reproduce/src/lib.rs`: every cast is a `usize as f64` on a
// trial/action count, all far below 2^53 (issue #11 pedantic burn-down).
#![allow(clippy::cast_precision_loss)]

use reproduce::{
    CommunicatingAgent, CommunicatingPOMDPAgent, CommunicationChannel, Message, MessageContent,
    MultiAgentEnvironment, AifError, POMDPAgent, SharedBanditEnvironment, env_seed, group_seed,
    substream,
};

struct TestEnvironment {
    actions: Vec<Vec<usize>>,
    rewards: Vec<Vec<usize>>,
    comm_channel: CommunicationChannel,
    messages_sent: Vec<Vec<String>>,
}

impl TestEnvironment {
    fn new(n_agents: usize) -> Self {
        Self {
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
        let message_str = format!("{message}");
        self.messages_sent[message.sender_id].push(message_str);
    }

    fn advance_step(&mut self) {
        self.comm_channel.advance_step();
    }
}

/// Competitive two-agent communication loop. Both agents carry a UNIFORM observation
/// model (A = [0.5, 0.5, 0.5]), so no arm is preferred a priori — the assertable
/// behavioural property is that each agent explores the whole arm set rather than
/// collapsing onto one arm (the sibling `test_cooperative_communication` asserts the
/// stronger arm-preference property, which its non-uniform A matrices make available).
///
/// Seeded (issue #8): the env uses `with_seed` and both base agents are `reseed`ed, so
/// the whole loop is reproducible; previously it asserted only that 30 actions were
/// recorded. Seeds derive through the #2 role helpers (env = `env_seed(SEED)`,
/// agent i = `substream(group_seed(SEED), i)`) for consistency with the sibling
/// matched-seed tests — PR #34 review finding. Arm-coverage assertions verified to
/// hold at SEED = 2026 and 815 under this derivation.
#[test]
// A single end-to-end scenario: two seeded communicating agents driving a shared bandit
// across 30 trials, with the contention-retry path and the arm-coverage assertions inline.
// Splitting it would require threading the whole mutable env + agent + tracker state
// through helpers, and the seeded trajectory is only meaningful as one uninterrupted run.
#[allow(clippy::too_many_lines)]
fn test_communicating_agents() -> Result<(), AifError> {
    const SEED: u64 = 2026;
    let n_agents = 2;
    let n_bandits = 3;
    let mut test_env = TestEnvironment::new(n_agents);
    let mut bandit_env =
        SharedBanditEnvironment::with_seed(vec![0.8, 0.4, 0.6], n_agents, env_seed(SEED))?;

    let mut base_agent1 = POMDPAgent::new(
        n_bandits,
        Some(vec![0.5, 0.5, 0.5]),
        None,
        vec![0.8, 0.2],
        None,
        8.0,
        false,
    )?;
    base_agent1.reseed(substream(group_seed(SEED), 1));

    let mut base_agent2 = POMDPAgent::new(
        n_bandits,
        Some(vec![0.5, 0.5, 0.5]),
        None,
        vec![0.8, 0.2],
        None,
        8.0,
        false,
    )?;
    base_agent2.reseed(substream(group_seed(SEED), 2));

    let mut agent1 = CommunicatingPOMDPAgent::new(base_agent1, 0, n_bandits, true, true, false, 3);
    let mut agent2 =
        CommunicatingPOMDPAgent::new(base_agent2, 1, n_bandits, false, true, false, 2);

    let mut prev_obs1 = 0;
    let mut prev_obs2 = 0;

    for _ in 0..30 {
        test_env.advance_step();

        let messages_for_agent1 = test_env.comm_channel.receive_all(0)?;
        let action1 = agent1.act_with_communication(prev_obs1, messages_for_agent1)?;
        let (reward1, _) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(
            &mut bandit_env,
            0,
            action1,
        )?;
        prev_obs1 = reward1;
        test_env.record_action(0, action1, reward1);

        let outgoing_messages1 = agent1.generate_messages();
        for msg in outgoing_messages1 {
            let content = msg.content.clone();
            test_env.comm_channel.send(msg.sender_id, 1, content)?;
            if let Ok(messages) = test_env.comm_channel.receive_all(1) {
                for m in &messages {
                    test_env.record_message(m);
                }
            }
        }

        let messages_for_agent2 = test_env.comm_channel.receive_all(1)?;
        let action2 = agent2.act_with_communication(prev_obs2, messages_for_agent2)?;

        match <SharedBanditEnvironment as MultiAgentEnvironment>::step(
            &mut bandit_env,
            1,
            action2,
        ) {
            Ok((reward2, _)) => {
                prev_obs2 = reward2;
                test_env.record_action(1, action2, reward2);
            }
            Err(AifError::ResourceConflict(_)) => {
                let mut found = false;
                for alt in (0..n_bandits).filter(|&a| a != action2) {
                    match <SharedBanditEnvironment as MultiAgentEnvironment>::step(
                        &mut bandit_env,
                        1,
                        alt,
                    ) {
                        Ok((reward, _)) => {
                            prev_obs2 = reward;
                            test_env.record_action(1, alt, reward);
                            found = true;
                            break;
                        }
                        // Contended arm: fall through to the next alternative.
                        Err(AifError::ResourceConflict(_)) => {}
                        Err(e) => return Err(e),
                    }
                }
                if !found {
                    test_env.record_action(1, action2, 0);
                    prev_obs2 = 0;
                }
            }
            Err(e) => return Err(e),
        }

        let outgoing_messages2 = agent2.generate_messages();
        for msg in outgoing_messages2 {
            let content = msg.content.clone();
            test_env.comm_channel.send(msg.sender_id, 0, content)?;
            if let Ok(messages) = test_env.comm_channel.receive_all(0) {
                for m in &messages {
                    test_env.record_message(m);
                }
            }
        }
    }

    println!("Agent 1 sent {} messages", test_env.messages_sent[0].len());
    println!("Agent 2 sent {} messages", test_env.messages_sent[1].len());

    let count_actions = |actions: &[usize]| {
        let mut counts = vec![0; n_bandits];
        for &a in actions {
            counts[a] += 1;
        }
        counts
    };

    let agent1_actions = count_actions(&test_env.actions[0]);
    let agent2_actions = count_actions(&test_env.actions[1]);

    println!("Agent 1 action distribution: {agent1_actions:?}");
    println!("Agent 2 action distribution: {agent2_actions:?}");

    // Both agents acted 30 times total
    assert_eq!(test_env.actions[0].len(), 30);
    assert_eq!(test_env.actions[1].len(), 30);

    // Every recorded action is a valid arm index.
    for (id, actions) in test_env.actions.iter().enumerate() {
        assert!(
            actions.iter().all(|&a| a < n_bandits),
            "agent {id} produced an out-of-range action: {actions:?}"
        );
    }

    // Uniform A ⇒ no arm is preferred, so neither agent may collapse onto a single arm;
    // both must exercise the full arm set over 30 competitive rounds.
    for (id, counts) in [(0usize, &agent1_actions), (1usize, &agent2_actions)] {
        assert_eq!(counts.iter().sum::<usize>(), 30, "agent {id} count total");
        assert!(
            counts.iter().all(|&c| c > 0),
            "agent {id} has a uniform observation model and must explore every arm: {counts:?}"
        );
        let max = *counts.iter().max().expect("3 arms ⇒ non-empty");
        assert!(
            max < 30,
            "agent {id} must not collapse onto one arm under a uniform A: {counts:?}"
        );
    }

    Ok(())
}

#[test]
fn test_cooperative_communication() -> Result<(), AifError> {
    let n_agents = 2;
    let n_bandits = 3;
    let mut test_env = TestEnvironment::new(n_agents);

    let mut bandit_env = SharedBanditEnvironment::new(vec![0.6, 0.6, 0.6], n_agents)?;
    bandit_env.set_competitive(false);

    let base_agent1 = POMDPAgent::new(
        n_bandits,
        Some(vec![0.9, 0.3, 0.3]),
        None,
        vec![0.8, 0.2],
        None,
        12.0,
        false,
    )?;

    let base_agent2 = POMDPAgent::new(
        n_bandits,
        Some(vec![0.3, 0.3, 0.9]),
        None,
        vec![0.8, 0.2],
        None,
        12.0,
        false,
    )?;

    let mut agent1 = CommunicatingPOMDPAgent::new(base_agent1, 0, n_bandits, false, true, true, 1);
    let mut agent2 = CommunicatingPOMDPAgent::new(base_agent2, 1, n_bandits, false, true, true, 1);

    let mut prev_obs1 = 0;
    let mut prev_obs2 = 0;

    for _ in 0..30 {
        test_env.advance_step();

        let messages_for_agent1 = test_env.comm_channel.receive_all(0)?;
        let action1 = agent1.act_with_communication(prev_obs1, messages_for_agent1)?;
        let (reward1, _) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(
            &mut bandit_env,
            0,
            action1,
        )?;
        prev_obs1 = reward1;
        test_env.record_action(0, action1, reward1);

        test_env
            .comm_channel
            .send(0, 1, MessageContent::Action(action1))?;
        test_env
            .comm_channel
            .send(0, 1, MessageContent::Reward(reward1 as f64))?;

        let messages_for_agent2 = test_env.comm_channel.receive_all(1)?;
        let action2 = agent2.act_with_communication(prev_obs2, messages_for_agent2)?;
        let (reward2, _) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(
            &mut bandit_env,
            1,
            action2,
        )?;
        prev_obs2 = reward2;
        test_env.record_action(1, action2, reward2);

        test_env
            .comm_channel
            .send(1, 0, MessageContent::Action(action2))?;
        test_env
            .comm_channel
            .send(1, 0, MessageContent::Reward(reward2 as f64))?;
    }

    let count_actions = |actions: &[usize]| {
        let mut counts = vec![0; n_bandits];
        for &a in actions {
            counts[a] += 1;
        }
        counts
    };

    let agent1_actions = count_actions(&test_env.actions[0]);
    let agent2_actions = count_actions(&test_env.actions[1]);

    println!("Agent 1 action distribution: {agent1_actions:?}");
    println!("Agent 2 action distribution: {agent2_actions:?}");

    // Both agents completed 30 rounds and produced valid actions
    assert_eq!(test_env.actions[0].len(), 30);
    assert_eq!(test_env.actions[1].len(), 30);

    // Agent 1 (A=[0.9,0.3,0.3]) should prefer bandit 0 (best obs-1 alignment)
    // Agent 2 (A=[0.3,0.3,0.9]) should prefer bandit 2 (best obs-1 alignment)
    assert!(
        agent1_actions[0] > agent1_actions[1],
        "Agent 1 should prefer bandit 0 over 1: {agent1_actions:?}"
    );
    assert!(
        agent2_actions[2] > agent2_actions[1],
        "Agent 2 should prefer bandit 2 over 1: {agent2_actions:?}"
    );

    Ok(())
}
