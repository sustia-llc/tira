# As One and Many: Relating Individual and Emergent Group-Level Generative Models in Active Inference

**Waade, P.T.; Olesen, C.L.; Laursen, J.E.; Nehrer, S.W.; Heins, C.; Friston, K.; Mathys, C.**

*Entropy* 2025, 27, 143. https://doi.org/10.3390/e27020143

Published: 1 February 2025 (Received: 21 October 2024; Revised: 15 January 2025; Accepted: 16 January 2025)

## Abstract

Active inference under the Free Energy Principle has been proposed as an across-scales compatible
framework for understanding and modelling behaviour and self-maintenance. Crucially, a collective
of active inference agents can, if they maintain a group-level Markov blanket, constitute a larger
group-level active inference agent with a generative model of its own. This potential for computational
scale-free structures speaks to the application of active inference to self-organizing systems across
spatiotemporal scales, from cells to human collectives. Due to the difficulty of reconstructing the
generative model that explains the behaviour of emergent group-level agents, there has been little
research on this kind of multi-scale active inference. Here, we propose a data-driven methodology for
characterising the relation between the generative model of a group-level agent and the dynamics of
its constituent individual agents. We apply methods from computational cognitive modelling and
computational psychiatry, applicable for active inference as well as other types of modelling approaches.
Using a simple Multi-Armed Bandit task as an example, we employ the new ActiveInference.jl library
for Julia to simulate a collective of agents who are equipped with a Markov blanket. We use
sampling-based parameter estimation to make inferences about the generative model of the group-level
agent, and we show that there is a non-trivial relationship between the generative models of individual
agents and the group-level agent they constitute, even in this simple setting. Finally, we point to a
number of ways in which this methodology might be applied to better understand the relations between
nested active inference agents across scales.

**Keywords:** active inference; free energy principle; Markov blanket; predictive processing;
cognitive modelling; multi-scale; collective intelligence; emergence

## Paper Structure

### 1. Introduction
- Active inference under FEP: perception, learning, action as approximate Bayesian inference
- Markov blankets: sensory states, active states, internal states, external states
- Nested Markov blanket structures: cells → organs → organisms → collectives
- Three scales of multi-agent active inference:
  1. **Within-agent**: generative models for interacting with environments containing other agents
  2. **Between-agent**: how interactions shape behavioural and belief dynamics over time
  3. **Group-as-agent**: collective forms emergent Markov blanket, becomes agent in its own right
- Gap: no prior work reconstructs emergent group-level generative model or compares it to constituent agents

### 2. Materials and Methods

#### 2.1 Active Inference and Multi-Armed Bandits
- POMDP generative model with matrices A-E:
  - **A** (observation model): P(o_t | s_t) — 2×3 matrix, outcome likelihoods per bandit
  - **B** (transition model): P(s_t | s_{t-1}, π) — deterministic action-state mapping
  - **C** (preference prior): P(o | C) — preference for observation 1 (reward)
  - **D** (state prior): P(s_1) — uniform over 3 bandits
  - **E** (policy prior): P(π) — uniform over 3 actions
- Variational free energy F minimized for perception (Eq. 1)
- Expected free energy G minimized for action selection (Eq. 2): information gain + pragmatic value
- Key parameters: α (action precision), γ (policy posterior precision = 16)
- 3 bandits, binary outcomes, probabilities [0.8, 0.2, 0.2], policy length = 2

#### 2.2 Computational Cognitive Modelling
- MCMC-based approximate Bayesian inference for parameter estimation
- Bayesian model comparison for model selection
- Parameter recovery to validate inference reliability

#### 2.3 Cognitive Modelling for Collective Agents
- Group agent Markov blanket structure:
  - **Sensory agent** (1): copies environment observations → passes to internal agents
  - **Internal agents** (n): POMDP active inference agents, observe sensory agent, act
  - **Active agent** (1): aggregates internal agent actions via probabilistic voting
- Group blanket states = sensory agent observations + active agent actions
- Same POMDP generative model used at both individual and group levels

#### 2.4 Simulation Experiments
- Focus on action precision α (softmax temperature over action probabilities)
- Parameter recovery: α recoverable in [0, 1], degenerates above 1
- Half-normal prior (mean=0, SD=4, truncated non-negative)
- **Experiment 1**: Identical α across all internal agents → group α = individual α
- **Experiment 2**: Varying α (Dirichlet-constructed) → group α ≈ sub-linear function of mean
- **Experiment 3**: Deterministic voting aggregation → super-linear group α scaling
- **Experiment 4**: Varying preferences (Beta(0.8, 0.8)) → strongly reduced group α
- Each experiment: 4, 8, 16, 100 internal agents

### 3. Results
- α recovery: good in [0, 1], clusters at ~4 for true α > 1
- Exp 1: group α identical to shared individual α (expected — aggregation preserves)
- Exp 2: sub-linear scaling, reduced variance with more agents (Bayesian model average)
- Exp 3: deterministic voting → super-linear, quickly saturates, slope ↑ with n_agents
- Exp 4: conflicting preferences → low group α (stochastic active agent)

### 4. Discussion
- Method enables relating all three scales simultaneously
- Limitations: parameter identifiability, model space, MCMC convergence
- Key extensions proposed:
  - Parameter learning (temporal dynamics)
  - Infer other parameters (γ, matrix contents, learning rates)
  - Sensory/active agents as active inference agents proper
  - Certainty-weighted actions → certainty-weighted Bayesian model average
  - Network communication structures (not just simple voting)
  - Game-theoretic inter-group competitions
  - Greater-than-two-scale nesting
  - Dynamically emerging Markov blankets
  - Evolutionary algorithms at different scales
  - Free energy extensivity question
  - Renormalization group for slow-timescale blankets
  - Application to dishbrains and organoids

## Implementation Status (one_many_rs)

### Core POMDP Agent (src/agent.rs)
- [x] POMDP agent with A-E matrices following paper specification
- [x] Expected free energy G (Eq. 2): information gain (observation entropy) + pragmatic value
- [x] Extracted `efe_step()` helper — single source of truth for per-step G computation
- [x] C vector as log-preference prior (ln P(o|C))
- [x] Separate α (action precision) and γ (policy posterior precision, default 16)
- [x] E vector (policy prior) participates in policy posterior
- [x] Multi-step policy evaluation (configurable policy_depth)
- [x] State inference (Bayesian belief updating via B × prior × likelihood)
- [x] A-matrix learning (pA concentration updates propagated back to A via column normalization)
- [x] action_probabilities() / record_action() / reset() for parameter recovery replay
- [x] Input validation: observation_probs.len() and initial_belief.len() must match n_states

### Environments (src/lib.rs)
- [x] BanditEnvironment (single-agent)
- [x] SharedBanditEnvironment (multi-agent, competitive/non-competitive modes)
- [x] Non-competitive round tracking via agents_acted vec (not bandit_selection scan)

### Group Agent (src/group.rs)
- [x] VotingMode enum: Probabilistic, Deterministic, CertaintyWeighted
- [x] VotingAgent: discrete vote aggregation + confidence-weighted distribution mixing
- [x] GroupAgent: Markov blanket composition (CopyAgent → Vec<POMDPAgent> → VotingAgent)
- [x] CertaintyWeighted mode: agents report full action distributions, active agent forms
      confidence-weighted mixture P_group(a) = Σ w_i P_i(a) / Σ w_i where w_i = exp(-H(P_i))
- [x] GroupAgentBuilder with factory methods for all experiments + .certainty_weighted(true)

### Simulation & Parameter Recovery (src/simulation.rs)
- [x] run_group_simulation() / run_single_simulation()
- [x] log_likelihood() — replay-based log-likelihood for candidate α
- [x] recover_alpha() — grid search with half-normal prior (MAP estimate)
- [x] Experiment factories: experiment_identical, experiment_varying_alpha,
      experiment_deterministic, experiment_varying_preferences, experiment_certainty_weighted
- [x] parameter_recovery_single() for Figure 4 validation
- [x] Dirichlet-constructed α distributions (with n<2 guard), Beta(0.8, 0.8) preferences

### Paper Reproduction (src/bin/reproduce.rs)
- [x] Full sweep: parameter recovery (Fig 4) + 4 paper experiments (Fig 5) + CW extension (Fig 6)
- [x] Rayon-parallelized, ~16s in release mode
- [x] Generates plots/figure4_recovery.png, figure5_experiments.png, figure6_certainty_weighted.png
- [x] All qualitative findings from paper reproduced:
  - Exp 1: group α ≈ individual α (identity line)
  - Exp 2: sub-linear scaling (below identity)
  - Exp 3: super-linear scaling, slope ↑ with n_agents
  - Exp 4: group α crushed near 0 (conflicting preferences)
  - Exp 5 (extension): CW voting tracks closer to identity than simple voting

### Communication Framework (src/communication.rs)
- [x] Flume-based message channels, CommunicatingPOMDPAgent

### Tests
- [x] 46 tests total (34 unit + 12 integration), all passing, 0 clippy warnings

### Possible Extensions
- [x] ~~Certainty-weighted voting (agents signal confidence)~~ — implemented as VotingMode::CertaintyWeighted
- [ ] MCMC parameter estimation (Metropolis-Hastings instead of grid search)
- [ ] Network communication structures (beyond simple voting)
- [ ] Greater-than-two-scale nesting (group of groups)
- [ ] Game-theoretic inter-group competition
- [ ] Evolutionary selection pressure on individual vs group level
