use one_many_rs::{
    CommunicatingAgent, CommunicatingPOMDPAgent, CommunicationChannel, Message, MessageContent,
    MultiAgentEnvironment, OneManyError, POMDPAgent, SharedBanditEnvironment,
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

#[test]
fn test_communicating_agents() -> Result<(), OneManyError> {
    let n_agents = 2;
    let n_bandits = 3;
    let mut test_env = TestEnvironment::new(n_agents);
    let mut bandit_env = SharedBanditEnvironment::new(vec![0.8, 0.4, 0.6], n_agents)?;

    // preferences is now [p(obs1), p(obs2)] — 2 elements for binary observations
    let base_agent1 = POMDPAgent::new(
        n_bandits,
        Some(vec![0.5, 0.5, 0.5]),
        Some(vec![1.0, 1.0, 1.0]),
        vec![0.8, 0.2],
        None,
        8.0,
        true,
    )?;

    let base_agent2 = POMDPAgent::new(
        n_bandits,
        Some(vec![0.5, 0.5, 0.5]),
        Some(vec![1.0, 1.0, 1.0]),
        vec![0.8, 0.2],
        None,
        8.0,
        true,
    )?;

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
            Err(OneManyError::ResourceConflict(_)) => {
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
                        Err(OneManyError::ResourceConflict(_)) => continue,
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

    Ok(())
}

#[test]
fn test_cooperative_communication() -> Result<(), OneManyError> {
    let n_agents = 2;
    let n_bandits = 3;
    let mut test_env = TestEnvironment::new(n_agents);

    let mut bandit_env = SharedBanditEnvironment::new(vec![0.6, 0.6, 0.6], n_agents)?;
    bandit_env.set_competitive(false);

    let base_agent1 = POMDPAgent::new(
        n_bandits,
        Some(vec![0.9, 0.3, 0.3]),
        Some(vec![1.0, 1.0, 1.0]),
        vec![0.8, 0.2],
        None,
        12.0,
        true,
    )?;

    let base_agent2 = POMDPAgent::new(
        n_bandits,
        Some(vec![0.3, 0.3, 0.9]),
        Some(vec![1.0, 1.0, 1.0]),
        vec![0.8, 0.2],
        None,
        12.0,
        true,
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

    // Agent 1's A matrix says bandit 0 has highest obs-1 prob → should prefer it
    assert!(
        agent1_actions[0] >= agent1_actions[1] && agent1_actions[0] >= agent1_actions[2],
        "Agent 1 should prefer bandit 0"
    );

    // Agent 2's A matrix says bandit 2 has highest obs-1 prob → should prefer it
    assert!(
        agent2_actions[2] >= agent2_actions[0] && agent2_actions[2] >= agent2_actions[1],
        "Agent 2 should prefer bandit 2"
    );

    Ok(())
}
