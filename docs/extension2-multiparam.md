# Extension 2 — multi-parameter (joint) MCMC recovery

_CLAUDE.md extension 2 / issues #29 + #30 (paper §4.1): joint MCMC recovery of parameters
beyond α, run under **two matched sampler arms**. Reproduce-side study
(`crates/reproduce/src/bin/extension2.rs`); the AIF engine is unchanged (the #30 kernel
extension lives in `reproduce`). Fully deterministic (master seed `0xE2_2026`, per-rep
substreams via `run_sweep`; MCMC chains seed from the dedicated MCMC role, disjoint from
generation): each cell is median · IQR over 5 seeded reps and re-running reproduces every
number exactly._

Run: `cargo run --release -p reproduce --bin extension2` (~3 min; the Q2 4× probe is most
of the added cost over #29's ~57 s).

## Headline

**#30 settles what #29 deliberately left open — identifiability, per joint.** The #29
sampler-scoped negative stands, embodied as the JointScale arm below (same seeds, same
budget: its numbers reproduce the #29 report exactly). The new Covariance arm (a
Haario-style adaptive-covariance proposal in log/logit-transformed space) converts it into
posterior-level answers:

- **(α, γ): partially identifiable — CLOSED.** The covariance sampler mixes (worst max
  R-hat 16.4 → 1.46, converged reps 5% → 60%) and the pooled-draw median of the **product
  α·γ lands within 5% of truth in all four cells**, even the two still failing the R-hat
  gate — unmixed chains still sit *on* the ridge. The factors separately stay
  prior-shaped: the behavioral stream constrains one temperature, not two.
- **(α, p): not marginally identifiable — CLOSED (negative).** A 4× budget probe
  near-converges (worst R-hat 1.081) onto **tight-but-wrong** marginals (rec p ≈
  0.36/0.50 vs true 0.8); more budget sharpens the wrong answer. The degeneracy is
  genuine and non-multiplicative (α·p is +32/+61% off).
- **(η, ω): weakly identifiable — not sampler-limited.** The covariance proposal does not
  help (worst R-hat 46); the pathology is likelihood structure (ω → 1 boundary regime),
  unfixed by either proposal geometry tested (within-Gibbs/tempered RW variants untested).

β₀/ψ remain analytically excluded (inert γ/β loop on deterministic B — see below).

## Protocol

- Single-agent generation at known params (`generate_params_data`); 300 trials/run; 5 reps/cell; matched seeds (generation + MCMC share a seed via disjoint substream roles).
- Joint MCMC (`recover_mcmc_vec`): 4 chains × (1000 burn-in + 2000 samples), per question, **per arm**.
- **Two arms, matched.** Both see the same seeds (hence the same generated data), the same budget, dims, bounds and objective (`log_likelihood_params + Σ per-dim prior`); the only difference is the proposal geometry.
  - **JointScale** (`ProposalMode::JointScale`, the #29 sampler): a joint **diagonal-Gaussian** random walk in θ space (uncorrelated across dimensions, each dimension reflected into its bounds) with **jointly-scaled** adaptation — a single Robbins-Monro increment adapts the global scale during burn-in (then freezes) while the per-dim σ RATIOS stay frozen at their initial values.
  - **Covariance** (`ProposalMode::Covariance`, the #30 sampler): a Haario-style **adaptive-covariance** random walk with global scaling, sampled in **log/logit-transformed** space (so the bounds are handled by the transform, not by reflection) with the transform's log-Jacobian added **in-kernel**; the covariance and scale freeze at burn-in end.
- Priors: α half-normal(0, 4) (the paper's), γ half-normal(0, 32) (scale-appropriate for the default 16), p uniform on [0.01, 0.99], η/ω uniform on [0.01, 1.0]. Convergence: per-dimension Gelman-Rubin R-hat, gate 1.05.

## Q1 — joint (α, γ): the temperature confound — JointScale (#29 sampler)

| true α | true γ | rec α | rec γ | corr(α,γ) | max R-hat | converged |
|--:|--:|--:|--:|--:|--:|--:|
| 0.30 | 4.00 | 0.445 · 0.445 | 15.990 · 19.795 | -0.732 · 0.156 | 16.440 · 25.610 | 0% |
| 0.30 | 16.00 | 1.067 · 1.011 | 19.198 · 18.944 | -0.792 · 0.141 | 11.690 · 4.280 | 0% |
| 0.70 | 4.00 | 0.967 · 0.680 | 2.701 · 2.084 | -0.659 · 0.054 | 7.899 · 4.538 | 0% |
| 0.70 | 16.00 | 0.968 · 1.227 | 13.857 · 14.537 | -0.712 · 0.048 | 3.351 · 2.718 | 20% |

## Q1 — joint (α, γ): the temperature confound — Covariance (#30 sampler)

| true α | true γ | rec α | rec γ | corr(α,γ) | max R-hat | converged |
|--:|--:|--:|--:|--:|--:|--:|
| 0.30 | 4.00 | 0.412 · 0.122 | 2.950 · 0.783 | -0.409 · 0.026 | 1.007 · 0.029 | 100% |
| 0.30 | 16.00 | 1.034 · 0.747 | 5.029 · 2.345 | -0.583 · 0.098 | 1.462 · 0.369 | 20% |
| 0.70 | 4.00 | 1.439 · 1.037 | 2.201 · 2.748 | -0.478 · 0.061 | 1.174 · 0.287 | 40% |
| 0.70 | 16.00 | 1.221 · 0.164 | 9.627 · 1.500 | -0.598 · 0.018 | 1.002 · 0.003 | 80% |

## Q2 — joint (α, p): A-matrix contents — JointScale (#29 sampler)

| true α | true p | rec α | rec p | corr(α,p) | max R-hat | converged |
|--:|--:|--:|--:|--:|--:|--:|
| 0.30 | 0.80 | 1.117 · 1.539 | 0.357 · 0.186 | -0.835 · 0.085 | 2.868 · 1.937 | 0% |
| 0.70 | 0.80 | 2.483 · 1.341 | 0.373 · 0.198 | -0.801 · 0.098 | 1.269 · 1.472 | 40% |

## Q2 — joint (α, p): A-matrix contents — Covariance (#30 sampler)

| true α | true p | rec α | rec p | corr(α,p) | max R-hat | converged |
|--:|--:|--:|--:|--:|--:|--:|
| 0.30 | 0.80 | 1.679 · 0.931 | 0.297 · 0.070 | -0.745 · 0.047 | 1.598 · 0.099 | 0% |
| 0.70 | 0.80 | 1.774 · 0.488 | 0.485 · 0.025 | -0.776 · 0.017 | 1.067 · 0.202 | 40% |

## Q3 — joint (η, ω): learning rates — JointScale (#29 sampler)

| true η | true ω | rec η | rec ω | corr(η,ω) | max R-hat | converged |
|--:|--:|--:|--:|--:|--:|--:|
| 0.50 | 0.90 | 0.512 · 0.161 | 0.875 · 0.464 | +0.660 · 0.496 | 11.512 · 15.584 | 40% |
| 0.50 | 1.00 | 0.389 · 0.103 | 0.591 · 0.568 | +0.267 · 0.246 | 1.596 · 100.573 | 0% |
| 1.00 | 0.90 | 0.713 · 0.098 | 0.872 · 0.023 | +0.152 · 0.053 | 1.004 · 0.007 | 100% |
| 1.00 | 1.00 | 0.646 · 0.110 | 0.674 · 0.480 | +0.304 · 0.281 | 19.051 · 50.724 | 20% |

## Q3 — joint (η, ω): learning rates — Covariance (#30 sampler)

| true η | true ω | rec η | rec ω | corr(η,ω) | max R-hat | converged |
|--:|--:|--:|--:|--:|--:|--:|
| 0.50 | 0.90 | 0.489 · 0.372 | 0.859 · 0.435 | +0.261 · 0.262 | 11.475 · 6.711 | 0% |
| 0.50 | 1.00 | 0.078 · 0.284 | 0.424 · 0.682 | +0.301 · 0.553 | 46.162 · 79.609 | 20% |
| 1.00 | 0.90 | 0.674 · 0.038 | 0.855 · 0.037 | +0.217 · 0.090 | 1.004 · 0.003 | 80% |
| 1.00 | 1.00 | 0.541 · 0.137 | 0.681 · 0.305 | +0.395 · 0.128 | 8.576 · 91.528 | 40% |

## #30 comparison (computed)

| question | arm | worst max R-hat | converged | pooled corr |
|:--|:--|--:|--:|--:|
| Q1 (α, γ) | JointScale | 16.440 | 5% | -0.724 |
| Q1 (α, γ) | Covariance | 1.462 | 60% | -0.517 |
| Q2 (α, p) | JointScale | 2.868 | 20% | -0.818 |
| Q2 (α, p) | Covariance | 1.598 | 20% | -0.761 |
| Q3 (η, ω) | JointScale | 19.051 | 40% | +0.346 |
| Q3 (η, ω) | Covariance | 46.162 | 35% | +0.293 |

_Worst max R-hat = the largest per-cell median of `max(R-hat over dims)`; converged = the fraction of reps passing the 1.05 gate, averaged over cells; pooled corr = the mean of the per-cell median pooled-sample Pearson correlations._

### Covariance arm — recovered vs true, Q1 (α, γ)

| true α | rec α | true γ | rec γ | true α·γ | rec α·γ |
|--:|--:|--:|--:|--:|--:|
| 0.30 | 0.412 · 0.122 | 4.00 | 2.950 · 0.783 | 1.200 | 1.179 · 0.112 |
| 0.30 | 1.034 · 0.747 | 16.00 | 5.029 · 2.345 | 4.800 | 4.741 · 0.291 |
| 0.70 | 1.439 · 1.037 | 4.00 | 2.201 · 2.748 | 2.800 | 2.709 · 0.238 |
| 0.70 | 1.221 · 0.164 | 16.00 | 9.627 · 1.500 | 11.200 | 10.668 · 3.229 |

### Covariance arm — recovered vs true, Q2 (α, p)

| true α | rec α | true p | rec p | true α·p | rec α·p |
|--:|--:|--:|--:|--:|--:|
| 0.30 | 1.679 · 0.931 | 0.80 | 0.297 · 0.070 | 0.240 | 0.498 · 0.172 |
| 0.70 | 1.774 · 0.488 | 0.80 | 0.485 · 0.025 | 0.560 | 0.848 · 0.179 |

### Covariance arm — recovered vs true, Q3 (η, ω)

| true η | rec η | true ω | rec ω | true η·ω | rec η·ω |
|--:|--:|--:|--:|--:|--:|
| 0.50 | 0.489 · 0.372 | 0.90 | 0.859 · 0.435 | 0.450 | 0.445 · 0.499 |
| 0.50 | 0.078 · 0.284 | 1.00 | 0.424 · 0.682 | 0.500 | 0.014 · 0.327 |
| 1.00 | 0.674 · 0.038 | 0.90 | 0.855 · 0.037 | 0.900 | 0.583 · 0.187 |
| 1.00 | 0.541 · 0.137 | 1.00 | 0.681 · 0.305 | 1.000 | 0.415 · 0.182 |

## Q2 extended-budget probe — Covariance, 4× budget

_The probe re-runs the Q2 Covariance arm on the **same data** (same cells, same base seed, so each (cell, rep) regenerates identically) at 4× the budget — 4000 burn-in + 8000 samples per chain, 4 chains; nothing else differs._

| question | arm | worst max R-hat | converged | pooled corr |
|:--|:--|--:|--:|--:|
| Q2 (α, p) probe | Covariance 4× | 1.081 | 60% | -0.718 |

### Probe — recovered vs true, Q2 (α, p)

| true α | rec α | true p | rec p | true α·p | rec α·p |
|--:|--:|--:|--:|--:|--:|
| 0.30 | 1.096 · 0.216 | 0.80 | 0.363 · 0.041 | 0.240 | 0.387 · 0.051 |
| 0.70 | 1.704 · 0.255 | 0.80 | 0.497 · 0.028 | 0.560 | 0.738 · 0.146 |

_Numbers only; interpretation follows below._

## Interpretation

**Headline — the #29 non-convergence was the sampler for (α, γ), and fixing the sampler shows the confound is a property of the posterior: the identified combination is the product α·γ.** Under the covariance-adapted transformed-space proposal, Q1 mixing largely recovers (worst max R-hat 16.4 → 1.46; converged reps 5% → 60%) and the pooled-draw median of α·γ lands within 5% of truth in **all four** cells — including the two still failing the R-hat gate: chains that have not mixed *along* the ridge still sit *on* it, so the ridge-aligned combination is pinned while the factor marginals stay prior-shaped (recovered α up to ~2× truth, γ down to ~⅓). **(α, γ) on the MAB: CLOSED as partially identifiable — α·γ is recoverable, the factors separately are not.** The paper's α/γ-conflation warning becomes a measured statement: the behavioral stream constrains one temperature, not two.

**Q1 mechanics.** In log space the α·γ ridge is the straight line ln α + ln γ = const — exactly what a frozen full-covariance Gaussian proposal can traverse, and what the #29 diagonal frozen-ratio proposal stepped off (rejected off-ridge, hence crawling chains). The Q1 pooled correlation (−0.52) is a near-posterior quantity in the converged cells, unlike #29's sampler-path −0.72.

**Q2 — genuine degeneracy, not budget.** At the shared budget the Covariance arm improves R-hat (2.87 → 1.60 worst) without converging (20%); the 4× probe near-converges (worst R-hat 1.081, 60% past the gate) onto **tight-but-wrong** marginals — rec p 0.363/0.497 (IQRs ≤ 0.04) vs true 0.8, α inflated to 1.10/1.70 — and the product α·p is *not* the identified combination (+61%/+32% off at 4×). More budget sharpens the wrong answer rather than finding the right one: the (α, p) posterior is a curved ridge whose mass sits away from the truth marginals under these priors. **(α, p): CLOSED as not marginally identifiable on this fixture** — the identified functional is some non-product curve (plausibly the good-arm choice probability that α and p jointly determine); characterizing it is future work.

**Q3 — (η, ω) is not sampler-limited.** Covariance mode does not help (worst R-hat 46 vs 19; converged 35% vs 40%) — itself informative: the pathology is likelihood structure (near-flat directions with an ω → 1 boundary regime; the ω = 1.0 rows mix worst, R-hat IQRs to ~90), not proposal geometry, and no product-like invariant appears (η·ω errors −1% to −97%). Weak identifiability stands, now with evidence it is not fixed by either proposal geometry tested here (diagonal and Haario-adaptive-covariance); within-Gibbs or tempered RW variants remain untested.

**Excluded — β₀/ψ (precision dynamics).** These are *unidentifiable* on the paper's MAB: deterministic B ⇒ transpose-normalized B† is uniform ⇒ the variational free energy `F_π` is policy-**constant** ⇒ the Smith Table-2 γ/β update is provably inert (test-pinned in aif). No amount of data recovers a parameter that does not move the likelihood; recovery would need a stochastic-B environment. Out of scope, noted for future work.

_Caveats: 2-D slices (pairwise joints), not the full joint over all parameters; two adaptive random-walk MH samplers (diagonal jointly-scaled, and Haario-style covariance-adapted in transformed space) — neither is NUTS; a single MAB fixture; single-agent generation. Recovered-γ medians are prior-sensitive under a near-flat ridge likelihood — the half-normal(0, 32) mass shapes them. Point estimates are posterior medians; correlations are pooled-sample Pearson, and over unconverged chains they are a sampler-path statistic rather than a posterior quantity. A gradient sampler (NUTS, needing a differentiable likelihood) or a problem-specific reparameterization could characterize the Q2 ridge curve or sharpen Q3; the question of what a random-walk MH sampler can extract from this fixture is settled here._

## History — #29 → #30

- **#29 (PR #31)** shipped the vector MH kernel (`recover_mcmc_vec`; the #25 scalar
  sampler is its bit-identical dim-1 case) and ran this study under the joint
  diagonal-Gaussian, jointly-scaled proposal only. Finding: confound-dominated
  (anti-correlated ridges, R-hat ≫ gate) — a **sampler-scoped** negative, identifiability
  left open and tracked as #30.
- **#30** added `ProposalMode::Covariance` to the kernel — Haario-style adaptive
  covariance with global scaling, sampled in log/logit-transformed space with the
  transform's log-Jacobian applied in-kernel (per-coordinate reflection is only symmetric
  for diagonal proposals, so the correlated mode must not reflect); frozen at burn-in end,
  plain MH thereafter. `JointScale` remains the default; the scalar/extension-1 path is
  untouched (draw-order byte-identity test-pinned). This revisit reruns the study with
  both arms matched and adds the Q2 4× probe.

## Guards (deterministic, assert-before-print)

The binary pins both generations of findings and aborts before printing a half-report if
a rerun disagrees: the #29 confound (JointScale mean |corr| > 0.4 on Q1/Q2), the #30
partial-identifiability result (Q1 Covariance α·γ within ±15% of truth in every cell —
measured devs ≤ 4.8%; arm contrast: Covariance conv ≥ 0.4 vs JointScale ≤ 0.1, worst
R-hat < 2), and the #30 degeneracy result (Q2 probe worst R-hat < 1.3 with recovered
p < 0.6 in every cell — the tight-but-wrong pin).
