//! Extension 2b (#33) Phase 0 — risk items, verified before any feature work
//! (plan `2026-07-30-ext2b-stochastic-b.md`).
//!
//! **D1 spike outcome (2026-07-30): the plan's original encoding FAILED its
//! gate; owner ratified the POSITIONAL encoding (C2) after exploration.**
//! Pinned findings:
//! 1. Deterministic controlled factor + stochastic UNCONTROLLED hazard factor
//!    ⇒ `F_π` exactly policy-constant ⇒ γ/β loop inert (finding 1).
//! 2. Uniform action-slip doesn't fix it: from-state-independent noise keeps
//!    the controlled B rank-1 — the inert result generalizes from
//!    "deterministic MAB B" to "any rank-1 controlled B" (finding 2).
//! 3. Sticky slip (switch fails w.p. ε, previous arm pulled) is
//!    from-state-dependent and the loop is live (finding 3) — but weaker than:
//! 4. **Positional (foraging) encoding — the ENCODING OF RECORD (owner,
//!    2026-07-30)**: factor 0 = position, controls {left, stay, right} with
//!    deterministic clamped moves; factor 1 = good-arm hazard chain. `to`
//!    depends on `from` structurally, so B is column-varying while fully
//!    deterministic — largest `F_π` spread of every candidate (~3× sticky),
//!    zero noise knobs. Corollary: the docs' "F varies only with stochastic B"
//!    phrasing is imprecise — column-varying DETERMINISTIC controlled B
//!    suffices (the inert theorem is about rank-1 B) — doc-correction pass due
//!    at PR time (CLAUDE.md + aif module docs).
//!
//! **Dynamics-replay fidelity** (the issue's do-first item): sampled
//! generation (`Agent::act`) and forced replay (`action_probabilities` +
//! `record_action`) produce bit-identical β/γ/F trajectories under MMP +
//! `PrecisionDynamics` — including `reset_window()` trial-boundary β semantics
//! and the combined learning+dynamics path — on the LIVE encoding, so the β/γ
//! comparisons are non-vacuous.
//!
//! Lives in `reproduce` (aif untouched, per the issue-#33 scope); drives only
//! aif's public surface, exactly as the Phase-2 `ModelParams` path will.
#![allow(clippy::float_cmp)] // bit-identity pins: exact f64 equality is the assertion

use aif::{
    Agent, AgentParams, AifError, GenerativeModel, POMDPAgent, PrecisionDynamics, StateInference,
};
use nalgebra::DMatrix;

const GOOD_P: f64 = 0.8;
const BAD_P: f64 = 0.2;

/// Controlled-factor (chosen arm) transition shape — the Phase-0 spike axis.
#[derive(Clone, Copy)]
enum Slip {
    /// Paper-faithful: control u sets arm u deterministically (rank-1).
    None,
    /// Uniform slip: to = u w.p. 1−ε, elsewhere ε/(n−1) — from-state-INDEPENDENT,
    /// so the columns are identical and B stays rank-1 despite being stochastic.
    Uniform(f64),
    /// Sticky slip: switching to u fails w.p. ε, leaving the PREVIOUS arm pulled
    /// — from-state-DEPENDENT (column-varying), the property the γ/β loop needs.
    Sticky(f64),
}

/// Two-factor restless-bandit generative model (plan D1/D1′). Joint states are
/// little-endian with factor 0 (chosen arm) fastest: `joint = chosen + n·good`.
/// Factor 1 carries exactly ONE control (uncontrolled), so `n_actions = n`.
#[allow(clippy::cast_precision_loss)] // n ≤ 3 here
fn switching_model(n: usize, hazard: f64, slip: Slip) -> GenerativeModel {
    let mut a = DMatrix::zeros(2, n * n);
    for good in 0..n {
        for chosen in 0..n {
            let p = if chosen == good { GOOD_P } else { BAD_P };
            a[(0, chosen + n * good)] = p;
            a[(1, chosen + n * good)] = 1.0 - p;
        }
    }
    // Factor 0 (B[to][from], column-stochastic).
    let b_chosen: Vec<DMatrix<f64>> = (0..n)
        .map(|u| {
            let mut m = DMatrix::zeros(n, n);
            for from in 0..n {
                match slip {
                    Slip::None => m[(u, from)] = 1.0,
                    Slip::Uniform(eps) => {
                        for to in 0..n {
                            m[(to, from)] =
                                if to == u { 1.0 - eps } else { eps / (n as f64 - 1.0) };
                        }
                    }
                    Slip::Sticky(eps) => {
                        if from == u {
                            m[(u, from)] = 1.0;
                        } else {
                            m[(u, from)] = 1.0 - eps;
                            m[(from, from)] = eps;
                        }
                    }
                }
            }
            m
        })
        .collect();
    // Factor 1: single control — stay w.p. 1−h, jump uniformly otherwise.
    let mut h = DMatrix::zeros(n, n);
    for from in 0..n {
        for to in 0..n {
            h[(to, from)] = if to == from { 1.0 - hazard } else { hazard / (n as f64 - 1.0) };
        }
    }
    GenerativeModel {
        a: vec![a],
        b: vec![b_chosen, vec![h]],
        c: vec![vec![0.7, 0.3]],
        d: vec![vec![1.0 / n as f64; n], vec![1.0 / n as f64; n]],
    }
}

fn dynamics_params(seed: u64) -> AgentParams {
    AgentParams {
        alpha: 1.0,
        policy_depth: 2,
        state_inference: StateInference::MarginalMessagePassing { horizon: 3, iters: 16 },
        precision_dynamics: Some(PrecisionDynamics::default()),
        seed: Some(seed),
        ..Default::default()
    }
}

/// Per-step internal-state capture: (β, γ, F) after each step, exact values.
#[derive(Debug, Default, PartialEq)]
struct Trace {
    steps: Vec<(f64, f64, f64)>,
}

impl Trace {
    fn push(&mut self, agent: &POMDPAgent) {
        self.steps.push((
            agent.beta().expect("invariant: dynamics on in every Phase-0 agent"),
            agent.gamma(),
            agent.variational_free_energy(),
        ));
    }
}

/// Generation side: seeded `act` loop mirroring `run_single_simulation`'s
/// feed-previous-observation convention with a scripted observation stream.
fn drive_generation(
    agent: &mut POMDPAgent,
    obs_script: &[usize],
) -> Result<(Vec<usize>, Vec<usize>, Trace), AifError> {
    let (mut fed, mut actions, mut trace) = (Vec::new(), Vec::new(), Trace::default());
    let mut prev = 0usize;
    for &next_obs in obs_script {
        let action = agent.act(prev)?;
        fed.push(prev);
        actions.push(action);
        trace.push(agent);
        prev = next_obs;
    }
    Ok((fed, actions, trace))
}

/// Replay side: the `score_replay` path — `action_probabilities` + forced
/// `record_action` over the recorded (input-obs, action) stream.
fn drive_replay(agent: &mut POMDPAgent, fed: &[usize], actions: &[usize]) -> Trace {
    let mut trace = Trace::default();
    for (i, &obs) in fed.iter().enumerate() {
        let probs = agent.action_probabilities(obs);
        assert!(probs.iter().all(|p| p.is_finite()), "replay probs must be finite");
        agent.record_action(actions[i]);
        trace.push(agent);
    }
    trace
}

const OBS_SCRIPT: [usize; 8] = [0, 1, 1, 0, 1, 0, 0, 1];

// ---------------------------------------------------------------------------
// D1 spike gate
// ---------------------------------------------------------------------------

/// Drive a scripted 5-step stream and return (`F_π` spread, |γ − 1|).
fn spike_drive(model: GenerativeModel) -> Result<(f64, f64), AifError> {
    let mut agent = POMDPAgent::from_model(model, dynamics_params(7))?;
    // The 1-control good-arm factor must not multiply the action space.
    assert_eq!(agent.n_actions(), 2);
    let obs = [0usize, 1, 0, 1, 0];
    let acts = [0usize, 1, 0, 1];
    agent.action_probabilities(obs[0]);
    for i in 1..obs.len() {
        agent.record_action(acts[i - 1]);
        agent.action_probabilities(obs[i]);
    }
    let fpi = agent.policy_free_energies().expect("invariant: MMP surfaces F_π");
    assert_eq!(fpi.len(), agent.n_actions().pow(2));
    assert!(fpi.iter().all(|x| x.is_finite()));
    let max = fpi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = fpi.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(!agent.gamma_trajectory().is_empty());
    Ok((max - min, (agent.gamma() - 1.0).abs()))
}

#[test]
fn spike_uncontrolled_hazard_alone_is_policy_constant() -> Result<(), AifError> {
    // D1 FINDING 1 (2026-07-30): the plan's original encoding — deterministic
    // controlled factor + stochastic UNCONTROLLED hazard factor — does NOT make
    // the loop live: factor 0 is rank-1 (B† uniform ⇒ policy-independent) and
    // factor 1 is shared across policies (one control ⇒ policy-independent),
    // so F_π is exactly constant and β never moves.
    let (spread, gamma_dev) = spike_drive(switching_model(2, 0.2, Slip::None))?;
    assert!(spread < 1e-12, "finding 1 violated: F_π spread {spread}");
    assert!(gamma_dev < 1e-12, "finding 1 violated: γ moved {gamma_dev}");
    Ok(())
}

#[test]
fn spike_uniform_slip_is_still_rank1_still_inert() -> Result<(), AifError> {
    // D1 FINDING 2: uniform slip (the coalition.rs `transition_noise` shape) is
    // stochastic but from-state-INDEPENDENT — identical columns, still rank-1,
    // still B† uniform, still inert. The inert theorem generalizes from
    // "deterministic MAB B" to "any rank-1 controlled B".
    let (spread, gamma_dev) = spike_drive(switching_model(2, 0.2, Slip::Uniform(0.1)))?;
    assert!(spread < 1e-12, "finding 2 violated: F_π spread {spread}");
    assert!(gamma_dev < 1e-12, "finding 2 violated: γ moved {gamma_dev}");
    Ok(())
}

#[test]
fn spike_sticky_slip_makes_precision_loop_live() -> Result<(), AifError> {
    // D1′ GATE (the redesign): sticky slip — switch-to-u fails w.p. ε, leaving
    // the previous arm pulled — is from-state-DEPENDENT (column-varying, rank
    // ≥ 2), which is the property per-policy F actually requires. With the
    // hazard factor present this is the ext-2b encoding of record.
    let (spread, gamma_dev) = spike_drive(switching_model(2, 0.2, Slip::Sticky(0.1)))?;
    assert!(spread > 1e-6, "D1′ GATE FAIL: F_π policy-constant (spread {spread})");
    assert!(gamma_dev > 1e-9, "D1′ GATE FAIL: γ pinned at its prior");
    Ok(())
}

#[test]
fn spike_sticky_only_no_hazard_characterization() -> Result<(), AifError> {
    // Sticky slip without hazard: characterizes whether the hazard factor
    // contributes to the precision loop or only to the environment's dynamics.
    let (spread, gamma_dev) = spike_drive(switching_model(2, 0.0, Slip::Sticky(0.1)))?;
    assert!(spread > 1e-6, "sticky alone should already vary F_π (spread {spread})");
    assert!(gamma_dev > 1e-9, "sticky alone should drive the loop");
    Ok(())
}

/// Positional ("foraging") candidate: factor 0 = position on a line of n
/// arms; controls = {left, stay, right} with edge-clamping; pulling happens at
/// the current position. `to` depends on `from` even with DETERMINISTIC
/// movement — column-varying without any noise knob. `slip` (fail-to-move,
/// stay in place) optional on top.
#[allow(clippy::cast_precision_loss)] // n ≤ 3 here
fn positional_model(n: usize, hazard: f64, slip: f64) -> GenerativeModel {
    let mut a = DMatrix::zeros(2, n * n);
    for good in 0..n {
        for pos in 0..n {
            let p = if pos == good { GOOD_P } else { BAD_P };
            a[(0, pos + n * good)] = p;
            a[(1, pos + n * good)] = 1.0 - p;
        }
    }
    // Controls 0/1/2 = left/stay/right, clamped at the edges (B[to][from]).
    let b_pos: Vec<DMatrix<f64>> = [-1i64, 0, 1]
        .iter()
        .map(|&mv| {
            let mut m = DMatrix::zeros(n, n);
            for from in 0..n {
                let from_i = i64::try_from(from).expect("invariant: n ≤ 3");
                let hi = i64::try_from(n - 1).expect("invariant: n ≤ 3");
                let to = usize::try_from((from_i + mv).clamp(0, hi))
                    .expect("invariant: clamped to 0..n");
                m[(to, from)] += 1.0 - slip;
                m[(from, from)] += slip;
            }
            m
        })
        .collect();
    let mut h = DMatrix::zeros(n, n);
    for from in 0..n {
        for to in 0..n {
            h[(to, from)] = if to == from { 1.0 - hazard } else { hazard / (n as f64 - 1.0) };
        }
    }
    GenerativeModel {
        a: vec![a],
        b: vec![b_pos, vec![h]],
        c: vec![vec![0.7, 0.3]],
        d: vec![vec![1.0 / n as f64; n], vec![1.0 / n as f64; n]],
    }
}

/// Encoding-candidate comparison table (exploration that led to the C2
/// ratification, kept as cheap documentation of the choice): drives each
/// candidate identically and prints (`F_π` spread, |γ−1|). Non-pinning; the
/// ratified candidate's gate is `spike_positional_deterministic_loop_live`.
#[test]
fn explore_encoding_candidates() -> Result<(), AifError> {
    let drive = |model: GenerativeModel, n_actions: usize| -> Result<(f64, f64), AifError> {
        let mut agent = POMDPAgent::from_model(model, dynamics_params(7))?;
        assert_eq!(agent.n_actions(), n_actions);
        let obs = [0usize, 1, 0, 1, 0];
        let acts = [0usize, 1, 2, 1];
        agent.action_probabilities(obs[0]);
        for i in 1..obs.len() {
            agent.record_action(acts[i - 1] % n_actions);
            agent.action_probabilities(obs[i]);
        }
        let fpi = agent.policy_free_energies().expect("invariant: MMP surfaces F_π");
        assert!(fpi.iter().all(|x| x.is_finite()));
        let max = fpi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = fpi.iter().copied().fold(f64::INFINITY, f64::min);
        Ok((max - min, (agent.gamma() - 1.0).abs()))
    };

    let candidates: [(&str, GenerativeModel, usize); 5] = [
        ("C1 sticky n=3 (eps=0.1, h=0.2)   ", switching_model(3, 0.2, Slip::Sticky(0.1)), 3),
        ("C2 positional DETERMINISTIC h=0.2", positional_model(3, 0.2, 0.0), 3),
        ("C2b positional determ.     h=0   ", positional_model(3, 0.0, 0.0), 3),
        ("C3 positional slip=0.1     h=0.2 ", positional_model(3, 0.2, 0.1), 3),
        ("C1b sticky n=2 (eps=0.1, h=0.2)  ", switching_model(2, 0.2, Slip::Sticky(0.1)), 2),
    ];
    for (name, model, n_actions) in candidates {
        let (spread, gamma_dev) = drive(model, n_actions)?;
        println!("{name}  F_pi spread = {spread:.6e}   |gamma-1| = {gamma_dev:.6e}");
    }
    Ok(())
}

#[test]
fn spike_positional_deterministic_loop_live() -> Result<(), AifError> {
    // THE GATE for the ratified encoding (C2, owner 2026-07-30): deterministic
    // clamped movement + hazard chain must keep the γ/β loop live. Thresholds
    // are deliberately loose liveness bounds (measured ~4.5e-1 / ~1.2e-2 on
    // this drive), not value pins.
    let mut agent = POMDPAgent::from_model(positional_model(3, 0.2, 0.0), dynamics_params(7))?;
    assert_eq!(agent.n_actions(), 3); // {left, stay, right}
    let obs = [0usize, 1, 0, 1, 0];
    let acts = [0usize, 2, 1, 2];
    agent.action_probabilities(obs[0]);
    for i in 1..obs.len() {
        agent.record_action(acts[i - 1]);
        agent.action_probabilities(obs[i]);
    }
    let fpi = agent.policy_free_energies().expect("invariant: MMP surfaces F_π");
    assert_eq!(fpi.len(), agent.n_actions().pow(2));
    let max = fpi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = fpi.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(max - min > 1e-2, "C2 GATE FAIL: F_π spread collapsed ({})", max - min);
    assert!(
        (agent.gamma() - 1.0).abs() > 1e-4,
        "C2 GATE FAIL: γ pinned at its prior (γ = {})",
        agent.gamma()
    );
    Ok(())
}

#[test]
fn spike_fully_deterministic_is_inert() -> Result<(), AifError> {
    // h = 0, slip = None: fully deterministic joint B — the inert regime.
    // (Factor-1 identity B is deterministic but not rank-1, so this is pinned
    // empirically, not assumed from the theorem.)
    let (spread, gamma_dev) = spike_drive(switching_model(2, 0.0, Slip::None))?;
    assert!(spread < 1e-9, "h = 0, slip = None: expected F_π constant, got {spread}");
    assert!(gamma_dev < 1e-12, "h = 0, slip = None: expected inert γ");
    Ok(())
}

// ---------------------------------------------------------------------------
// Dynamics-replay fidelity
// ---------------------------------------------------------------------------

#[test]
fn dynamics_replay_matches_generation_bit_identically() -> Result<(), AifError> {
    // The RATIFIED encoding (live loop) so the β/γ fidelity assertions are
    // non-vacuous — under an inert encoding β ≡ β₀ and they would pin nothing.
    let model = positional_model(3, 0.2, 0.0);
    let params = dynamics_params(42);

    let mut generator = POMDPAgent::from_model(model.clone(), params.clone())?;
    let (fed, actions, gen_trace) = drive_generation(&mut generator, &OBS_SCRIPT)?;

    let mut replayer = POMDPAgent::from_model(model, params)?;
    let replay_trace = drive_replay(&mut replayer, &fed, &actions);

    assert_eq!(gen_trace, replay_trace, "β/γ/F must be bit-identical per step");
    assert_eq!(
        generator.gamma_trajectory(),
        replayer.gamma_trajectory(),
        "final γ trajectories must be bit-identical"
    );
    assert_eq!(
        generator.policy_free_energies(),
        replayer.policy_free_energies(),
        "final F_π must be bit-identical"
    );
    Ok(())
}

#[test]
fn dynamics_replay_reset_window_beta_semantics_match() -> Result<(), AifError> {
    // Trial boundary mid-stream: reset must restore β₀/γ₀ and empty the
    // trajectory on BOTH sides, and the continuation must stay bit-identical.
    let model = positional_model(3, 0.2, 0.0);
    let params = dynamics_params(11);
    let (first, second) = (&OBS_SCRIPT[..4], &OBS_SCRIPT[4..]);

    let mut generator = POMDPAgent::from_model(model.clone(), params.clone())?;
    let (fed_a, acts_a, gen_a) = drive_generation(&mut generator, first)?;
    generator.reset_window();
    assert_eq!(generator.beta().expect("dynamics on"), 1.0, "β must reset to β₀");
    assert_eq!(generator.gamma(), 1.0, "γ must reset to 1/β₀");
    assert!(generator.gamma_trajectory().is_empty());
    let (fed_b, acts_b, gen_b) = drive_generation(&mut generator, second)?;

    let mut replayer = POMDPAgent::from_model(model, params)?;
    let rep_a = drive_replay(&mut replayer, &fed_a, &acts_a);
    replayer.reset_window();
    assert_eq!(replayer.beta().expect("dynamics on"), 1.0);
    assert_eq!(replayer.gamma(), 1.0);
    assert!(replayer.gamma_trajectory().is_empty());
    let rep_b = drive_replay(&mut replayer, &fed_b, &acts_b);

    assert_eq!(gen_a, rep_a, "pre-boundary trajectories must be bit-identical");
    assert_eq!(gen_b, rep_b, "post-boundary trajectories must be bit-identical");
    Ok(())
}

#[test]
fn dynamics_with_learning_replay_matches_generation() -> Result<(), AifError> {
    // The combined path (learn_a + MMP + dynamics) — the 0.9.0 reorder ran the
    // γ/β loop post-Dirichlet-updates; replay must walk the same sequence.
    let model = positional_model(3, 0.2, 0.0);
    let params = AgentParams {
        learn_a: true,
        initial_precision: Some(vec![1.0; 9]), // n_joint = 3 positions × 3 good-arms
        ..dynamics_params(23)
    };

    let mut generator = POMDPAgent::from_model(model.clone(), params.clone())?;
    let (fed, actions, gen_trace) = drive_generation(&mut generator, &OBS_SCRIPT)?;

    let mut replayer = POMDPAgent::from_model(model, params)?;
    let replay_trace = drive_replay(&mut replayer, &fed, &actions);

    assert_eq!(gen_trace, replay_trace, "β/γ/F must be bit-identical per step");
    let (gen_pa, rep_pa) = (
        generator.pa().expect("learn_a on"),
        replayer.pa().expect("learn_a on"),
    );
    assert_eq!(gen_pa, rep_pa, "learned Dirichlet counts must be bit-identical");
    Ok(())
}
