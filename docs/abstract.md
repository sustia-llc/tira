# As One and Many: Relating Individual and Emergent Group-Level Generative Models in Active Inference

**Waade, P.T.; Olesen, C.L.; Laursen, J.E.; Nehrer, S.W.; Heins, C.; Friston, K.; Mathys, C.**

*Entropy* 2025, 27, 143. <https://doi.org/10.3390/e27020143>
Published 1 February 2025. Full text: [doi.org/10.3390/e27020143](https://doi.org/10.3390/e27020143)
(open access, CC BY).

*This is a summary of the paper. For the section-by-section mapping of paper → tira code — and
for what tira deliberately does differently — see [aif-coverage.md](aif-coverage.md).*

## Abstract (verbatim)

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

## What the paper does

**§1 Introduction.** Active inference describes systems that maintain a Markov blanket as minimizing
variational free energy — and blankets nest: collectives of AIF agents can form a group-level blanket
and thereby a group-level AIF agent. Prior multi-agent work operates at the *within-agent* and
*between-agent* scales; almost nothing reconstructs the generative model of the emergent
*group-as-agent* (the exception being spin-glass collectives, in a restricted setting). The obstacle:
the group's generative model is emergent and a priori unknown, so it must be inferred from behaviour.

**§2 Methods.** The proposal is cognitive modelling applied one level up: fit a generative model to
the group's *blanket states* (its observations and actions), exactly as computational psychiatry fits
models to human behaviour.

- **§2.1** — the individual model: a POMDP for a 3-armed Multi-Armed Bandit. Matrices A (observation
  model; outcome-1 probabilities 0.8/0.2/0.2 across arms), B (deterministic arm-selection
  transitions), C (preference prior, 0.7/0.3), D and E (uniform priors). Perception minimizes
  variational free energy F (Eq. 1); action selection minimizes expected free energy G (Eq. 2 —
  information gain + pragmatic value), with two precisions: γ (policy posterior, fixed at 16) and
  **α** (action selection — the paper's target parameter). Policy length 2, no parameter learning:
  agents get accurate beliefs, so behaviour is driven by pragmatic value.
- **§2.2** — fitting and parameter recovery: simulate behaviour at known α, re-estimate it
  (half-normal(0, 4) prior, MCMC posterior median via Turing.jl/ActionModels.jl), and check the
  estimates track the truth.
- **§2.3** — the group agent: a fixed Markov blanket of *sensory* agents (copy observations in),
  *internal* agents (POMDP AIF agents), and an *active* agent (probabilistic vote aggregator).
  Group observations and group actions are the blanket states used for fitting; the same MAB-POMDP
  serves as the group-level model.
- **§2.4** — four experiments at group sizes 4/8/16/100, internal α ∈ [0, 1]: (1) identical α;
  (2) Dirichlet-varying α (sufficient statistic 1.5) with controlled mean; (3) deterministic
  (majority) voting; (4) Beta(0.8, 0.8)-varying preferences.

**§3 Results.**

- α recovers well in [0, 1]; above ~1 behaviour saturates (deterministic) and estimates cluster
  high (~4, set by the prior) regardless of truth — an identifiability ceiling (Fig 4).
- Experiment 1: identical internal α → group α equals it (identity line, Fig 5A).
- Experiment 2: varying α → group α tracks the internal **mean sub-linearly**; more agents, less
  variance (Fig 5B) — an unweighted Bayesian model average.
- Experiment 3: deterministic voting → **super-linear** group α inflation, steeper with group size
  (law of large numbers), quickly hitting the identifiability ceiling (Fig 5C).
- Experiment 4: conflicting preferences → votes cancel, the active agent looks stochastic, and
  group α is **crushed** toward 0 (Fig 5D).

The headline: the group-level generative model relates non-trivially to its constituents — group
parameters are not simply the members' parameters, and non-α properties (voting rule, preference
heterogeneity) masquerade as group-level α.

**§4 Discussion.** Limitations (behaviour must be informative; model space must be specified;
inference cost) and a rich extension list: parameter learning, inferring other parameters,
sensory/active agents as proper AIF agents, certainty-weighted vote aggregation, network
communication topologies, replacing the environment with a competing group (game theory), >2-scale
nesting, dynamically emerging blankets, evolutionary selection at different scales, free-energy
extensivity, continuous state-space models, renormalization-group detection of blankets at slower
timescales, and application to systems with unknown generative models (animals, artificial systems,
organoids/"dishbrains"). tira's numbered extension tracking (1–12) for this list lives in
[aif-coverage.md](aif-coverage.md) §4.

## Key results reproduced by tira

Figures 4 and 5 (all four experiments) are reproduced in
`plots/figure4_recovery.png` / `plots/figure5_experiments.png`, and the paper's proposed
certainty-weighted aggregation (an extension the paper suggests but does not simulate) is
implemented and evaluated in `plots/figure6_certainty_weighted.png`. Methodological choices
(grid-MAP recovery as the fast default, with MCMC posterior-median recovery also available
since #25; policy depth 1 with a proven-equivalent action marginal) are documented in
[aif-coverage.md](aif-coverage.md).
