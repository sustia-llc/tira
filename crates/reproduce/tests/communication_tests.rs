// See the crate-level note in `reproduce/src/lib.rs`: every cast is a `usize as f64` on a
// trial/action count, all far below 2^53 (issue #11 pedantic burn-down).
#![allow(clippy::cast_precision_loss)]

use reproduce::{
    Agent, AifError, CommunicatingAgent, CommunicatingPOMDPAgent, CommunicationChannel,
    MessageContent, MultiAgentEnvironment, POMDPAgent, SharedBanditEnvironment, env_seed,
    group_seed, substream,
};

/// A minimal wrapped agent for the cadence tests — the emission schedule is independent
/// of what the wrapped agent decides.
fn cadence_agent(frequency: usize) -> Result<CommunicatingPOMDPAgent, AifError> {
    let base = POMDPAgent::new(3, Some(vec![0.8, 0.2, 0.2]), None, vec![0.7, 0.3], None, 8.0, false)?;
    Ok(CommunicatingPOMDPAgent::new(base, 0, true, frequency))
}

/// Issue #5.1: for any `communication_frequency >= 1`, emission could never fire — the
/// counter was incremented and reset inside the same call, so its observable values were
/// `0..frequency-1` and the `>= frequency` test was unreachable. These pin the fixed
/// cadence at the two frequencies named in the issue, plus the emission-resets-the-counter
/// invariant that makes the schedule periodic rather than one-shot.
#[test]
fn test_emission_cadence_every_step() -> Result<(), AifError> {
    let mut agent = cadence_agent(1)?;
    let mut emissions = Vec::new();
    for _ in 0..6 {
        agent.act_with_communication(0, Vec::new())?;
        emissions.push(agent.generate_messages().len());
    }
    assert_eq!(
        emissions,
        vec![1, 1, 1, 1, 1, 1],
        "f = 1 must emit on every step"
    );
    Ok(())
}

#[test]
fn test_emission_cadence_every_third_step() -> Result<(), AifError> {
    let mut agent = cadence_agent(3)?;
    let mut emissions = Vec::new();
    for _ in 0..9 {
        agent.act_with_communication(0, Vec::new())?;
        emissions.push(agent.generate_messages().len());
    }
    assert_eq!(
        emissions,
        vec![0, 0, 1, 0, 0, 1, 0, 0, 1],
        "f = 3 must emit on every third step, periodically"
    );
    Ok(())
}

/// `share_actions = false` silences emission at every cadence — the flag is read, not
/// merely stored.
#[test]
fn test_share_actions_false_never_emits() -> Result<(), AifError> {
    let base = POMDPAgent::new(3, Some(vec![0.8, 0.2, 0.2]), None, vec![0.7, 0.3], None, 8.0, false)?;
    let mut agent = CommunicatingPOMDPAgent::new(base, 0, false, 1);
    for _ in 0..4 {
        agent.act_with_communication(0, Vec::new())?;
        assert!(agent.generate_messages().is_empty());
    }
    Ok(())
}

/// A due cadence slot is consumed even when there is nothing to send, so the schedule
/// stays periodic instead of latching true. Flipping `share_actions` back on must not
/// produce a backlog — it resumes on the next due step, not immediately.
#[test]
fn test_silent_steps_still_consume_the_cadence_slot() -> Result<(), AifError> {
    let base = POMDPAgent::new(3, Some(vec![0.8, 0.2, 0.2]), None, vec![0.7, 0.3], None, 8.0, false)?;
    let mut agent = CommunicatingPOMDPAgent::new(base, 0, false, 3);

    for _ in 0..6 {
        agent.act_with_communication(0, Vec::new())?;
        assert!(agent.generate_messages().is_empty(), "share_actions is off");
    }
    // Two slots came due and were consumed while silent; the counter is not latched.
    assert_eq!(agent.steps_since_communication, 0);

    agent.share_actions = true;
    let resumed: Vec<usize> = (0..3)
        .map(|_| {
            agent
                .act_with_communication(0, Vec::new())
                .map(|_| agent.generate_messages().len())
        })
        .collect::<Result<_, _>>()?;
    assert_eq!(
        resumed,
        vec![0, 0, 1],
        "emission resumes on the next due step, with no backlog"
    );
    Ok(())
}

/// `CommunicatingAgent: Agent`, and the harness runners take `&mut impl Agent`, so an
/// agent can be driven entirely through `Agent::act`. That path must advance the cadence
/// too, or such an agent would never emit at any frequency.
#[test]
fn test_plain_agent_act_advances_the_cadence() -> Result<(), AifError> {
    let mut agent = cadence_agent(2)?;
    let mut emissions = Vec::new();
    for _ in 0..4 {
        Agent::act(&mut agent, 0)?;
        emissions.push(agent.generate_messages().len());
    }
    assert_eq!(
        emissions,
        vec![0, 1, 0, 1],
        "the plain Agent path must drive the same cadence as act_with_communication"
    );
    Ok(())
}

/// Issue #5.3: broadcast (`recipient_id: None`) and non-default priorities were both
/// unconstructible through the public API. Both are now reachable, and `receive_all`'s
/// priority sort — previously sorting a field that was always `None` — has something to
/// order.
#[test]
fn test_broadcast_reaches_every_other_agent() -> Result<(), AifError> {
    let channel = CommunicationChannel::new(3);
    let delivered = channel.broadcast(1, &MessageContent::Action(2), None)?;
    assert_eq!(delivered, 2, "broadcast skips the sender");

    for recipient in [0, 2] {
        let messages = channel.receive_all(recipient)?;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].sender_id, 1);
        assert!(
            messages[0].recipient_id.is_none(),
            "a broadcast carries no single recipient"
        );
        assert_eq!(format!("{}", messages[0]), "From Agent 1 to Everyone: Action: 2 (t=0)");
    }
    assert!(!channel.has_messages(1)?, "the sender receives nothing");
    Ok(())
}

#[test]
fn test_receive_all_orders_by_priority_then_timestamp() -> Result<(), AifError> {
    let mut channel = CommunicationChannel::new(2);
    channel.send(1, 0, MessageContent::Text("no-priority".into()))?;
    channel.send_with_priority(1, 0, MessageContent::Text("low".into()), 1)?;
    channel.advance_step();
    channel.send_with_priority(1, 0, MessageContent::Text("high-late".into()), 9)?;
    channel.send_with_priority(1, 0, MessageContent::Text("low-late".into()), 1)?;

    let ordered: Vec<String> = channel
        .receive_all(0)?
        .iter()
        .map(|m| match &m.content {
            MessageContent::Text(t) => t.clone(),
            other => panic!("unexpected content: {other:?}"),
        })
        .collect();

    assert_eq!(
        ordered,
        vec!["high-late", "low", "low-late", "no-priority"],
        "highest priority first; oldest first within a priority; absent priority sorts as 0"
    );
    Ok(())
}

#[test]
fn test_channel_rejects_unknown_agent_ids() {
    let channel = CommunicationChannel::new(2);
    assert!(matches!(
        channel.send(0, 7, MessageContent::Action(0)),
        Err(AifError::InvalidAgentId(7))
    ));
    assert!(matches!(
        channel.broadcast(7, &MessageContent::Action(0), None),
        Err(AifError::InvalidAgentId(7))
    ));
    assert!(matches!(
        channel.receive_all(7),
        Err(AifError::InvalidAgentId(7))
    ));
    assert!(matches!(
        channel.has_messages(7),
        Err(AifError::InvalidAgentId(7))
    ));
    assert_eq!(channel.n_agents(), 2);
}

struct TestEnvironment {
    actions: Vec<Vec<usize>>,
    rewards: Vec<Vec<usize>>,
    comm_channel: CommunicationChannel,
    messages_sent: Vec<Vec<String>>,
    /// Messages each agent actually *received* — counted separately from sends so a
    /// fixture that never delivers cannot masquerade as one where delivery has no effect.
    messages_received: Vec<usize>,
}

impl TestEnvironment {
    fn new(n_agents: usize) -> Self {
        Self {
            actions: vec![Vec::new(); n_agents],
            rewards: vec![Vec::new(); n_agents],
            comm_channel: CommunicationChannel::new(n_agents),
            messages_sent: vec![Vec::new(); n_agents],
            messages_received: vec![0; n_agents],
        }
    }

    fn record_action(&mut self, agent_id: usize, action: usize, reward: usize) {
        self.actions[agent_id].push(action);
        self.rewards[agent_id].push(reward);
    }

    fn record_sent(&mut self, sender_id: usize, content: &MessageContent) {
        self.messages_sent[sender_id].push(format!("{content:?}"));
    }

    fn record_received(&mut self, agent_id: usize, count: usize) {
        self.messages_received[agent_id] += count;
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

    let mut agent1 = CommunicatingPOMDPAgent::new(base_agent1, 0, true, 3);
    let mut agent2 = CommunicatingPOMDPAgent::new(base_agent2, 1, true, 2);

    let mut prev_obs1 = 0;
    let mut prev_obs2 = 0;

    for _ in 0..30 {
        test_env.advance_step();

        let messages_for_agent1 = test_env.comm_channel.receive_all(0)?;
        test_env.record_received(0, messages_for_agent1.len());
        let action1 = agent1.act_with_communication(prev_obs1, messages_for_agent1)?;
        let (reward1, _) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(
            &mut bandit_env,
            0,
            action1,
        )?;
        prev_obs1 = reward1;
        test_env.record_action(0, action1, reward1);

        // Record at SEND time. Draining the recipient's queue here to record it (the
        // pre-#5 shape) stole the mail: the `receive_all(1)` below then always saw an
        // empty queue, so no agent in this fixture ever actually received anything —
        // which made the "messages have no decision effect" claim untestable here.
        for msg in agent1.generate_messages() {
            test_env.record_sent(msg.sender_id, &msg.content);
            test_env.comm_channel.send(msg.sender_id, 1, msg.content)?;
        }

        let messages_for_agent2 = test_env.comm_channel.receive_all(1)?;
        test_env.record_received(1, messages_for_agent2.len());
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

        for msg in agent2.generate_messages() {
            test_env.record_sent(msg.sender_id, &msg.content);
            test_env.comm_channel.send(msg.sender_id, 0, msg.content)?;
        }
    }

    // Issue #5.1 regression pin: before the cadence fix BOTH counts were 0 — emission
    // could never fire. Agent 1 (f = 3) emits on steps 3,6,…,30; agent 2 (f = 2) on
    // steps 2,4,…,30.
    assert_eq!(
        test_env.messages_sent[0].len(),
        10,
        "agent 1 at f = 3 over 30 steps"
    );
    assert_eq!(
        test_env.messages_sent[1].len(),
        15,
        "agent 2 at f = 2 over 30 steps"
    );

    // …and they were actually DELIVERED. Without this, the action-distribution pin
    // below would be vacuous: an undelivered message trivially has no effect. Agent 2
    // reads agent 1's mail in the same iteration (10 of 10); agent 1 reads agent 2's on
    // the following one, so the step-30 emission is still queued when the loop ends
    // (14 of 15).
    assert_eq!(test_env.messages_received[0], 14, "agent 1 receipts");
    assert_eq!(test_env.messages_received[1], 10, "agent 2 receipts");

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

    // Issue #5.4, pinned: delivered messages have NO decision effect. These are the
    // distributions this seeded fixture produced BEFORE the cadence fix, when zero
    // messages were emitted; they are unchanged now that 24 are emitted and delivered.
    // Wiring messages into inference (#46) must move these — that is the point of the pin.
    assert_eq!(
        agent1_actions,
        vec![13, 10, 7],
        "delivered messages must not move the seeded trajectory"
    );
    assert_eq!(
        agent2_actions,
        vec![13, 11, 6],
        "delivered messages must not move the seeded trajectory"
    );

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

    let mut agent1 = CommunicatingPOMDPAgent::new(base_agent1, 0, true, 1);
    let mut agent2 = CommunicatingPOMDPAgent::new(base_agent2, 1, true, 1);

    let mut prev_obs1 = 0;
    let mut prev_obs2 = 0;

    for _ in 0..30 {
        test_env.advance_step();

        let messages_for_agent1 = test_env.comm_channel.receive_all(0)?;
        test_env.record_received(0, messages_for_agent1.len());
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
        test_env.record_received(1, messages_for_agent2.len());
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
