# Extension 1 — MCMC parameter recovery for α

_CLAUDE.md extension 1 / issue #25 (paper §4.1): recover α by Metropolis-Hastings and
report the posterior **median** (the paper's estimator), alongside tira's fast grid
point-MAP. Reproduce-side study (`crates/reproduce/src/bin/extension1.rs`); the AIF engine
is unchanged. Fully deterministic (master seed `0xE1_2026`, distinct per-rep substreams via
`run_sweep`; the MCMC chains seed from a **dedicated** role so they never collide with the
data-generation streams; no shared root with the reproduce (2026), extension11 (0xE11_2026),
or extension3 (0xE3_2026) binaries): each cell is median · IQR over 5 seeded reps and
re-running reproduces every number below exactly._

Run: `cargo run --release -p reproduce --bin extension1` (~35 s).

## The claim under test

The paper reports α by the posterior median (via MCMC). tira's default `recover_alpha` is a
grid point-MAP over α ∈ [0, 5] step 0.01 under the paper's half-normal(0, 4) prior. It
reproduces the paper in the **identifiable** region (α < 1) but — as CLAUDE.md's design note
records — cannot reproduce the paper's Figure-4 **degenerate**-region behaviour: once
behaviour saturates the likelihood flattens and a single highest-node point-MAP has nothing
to grip, whereas the posterior median is pulled toward the prior-dominated region. We recover
α **both** ways on the *same* single-agent data (matched seeds) across both regions and report
what we measure. Nothing is forced toward a target: the half-normal(0, 4) truncated prior
alone has median ≈ 2.7, and where the posterior medians land IS the result.

## Protocol

- Single-agent generation (`single_agent_data`, standard MAB obs probs `[0.8, 0.2, 0.2]`,
  prefs `[0.7, 0.3]`); 300 trials/run; 5 reps/cell; true α ∈ {0.1, 0.3, 0.5, 0.7, 1.0, 1.5,
  2.0, 3.0}. Grid MAP and MCMC score the **same** trajectory per rep (matched seed).
- **MCMC** (`recover_alpha_mcmc`): Gaussian random-walk Metropolis-Hastings on α, **reflected
  at 0** (`α' = |α + N(0, σ)|` — reflection keeps the proposal symmetric, so plain MH
  acceptance is exact). 4 chains × (800 burn-in + 1500 samples), overdispersed `|N(0, 4)|`
  inits per chain (seeded from a dedicated MCMC role, `substream(mcmc_base_seed(seed), k)`).
- **Adaptive proposal**: the proposal SD starts at 0.5 and **adapts** by Robbins-Monro
  (diminishing gain `1/(i+1)^0.6` on `log σ`) toward ~0.35 acceptance during burn-in, then
  **freezes** — so the sampling phase is plain MH with detailed balance intact. One config
  thus mixes in both the narrow identifiable posteriors and the broad degenerate ones, with
  no per-region hand-tuning.
- **Objective**: `log_likelihood(α) + half_normal_log_prior(α)` — the *same* target the grid
  MAP maximizes (shared `half_normal_log_prior`/`recover_alpha_with` seam from extension 3),
  so the two estimators differ only in summary (argmax node vs posterior median).
- Convergence: classic Gelman-Rubin R-hat (`R_HAT_THRESHOLD` = 1.05) + acceptance rate.

## Results (median · IQR over 5 reps)

| true α | grid-MAP α | MCMC median | R-hat | acceptance | adapted SD |
|-------:|-----------:|------------:|------:|-----------:|-----------:|
| 0.1 | 0.100 · 0.010 | 0.097 · 0.010 | 1.002 · 0.001 | 0.357 · 0.012 | 0.046 · 0.001 |
| 0.3 | 0.310 · 0.010 | 0.310 · 0.010 | 1.001 · 0.001 | 0.364 · 0.015 | 0.067 · 0.005 |
| 0.5 | 0.490 · 0.010 | 0.493 · 0.010 | 1.001 · 0.001 | 0.347 · 0.018 | 0.121 · 0.011 |
| 0.7 | 0.700 · 0.090 | 0.725 · 0.108 | 1.001 · 0.002 | 0.341 · 0.004 | 0.312 · 0.141 |
| 1.0 | 1.350 · 0.000 | 3.216 · 0.056 | 1.001 · 0.001 | 0.338 · 0.008 | 12.120 · 0.204 |
| 1.5 | 1.350 · 0.000 | 3.247 · 0.053 | 1.002 · 0.006 | 0.338 · 0.025 | 12.216 · 0.752 |
| 2.0 | 1.350 · 0.000 | 3.224 · 0.142 | 1.003 · 0.001 | 0.357 · 0.013 | 11.748 · 0.389 |
| 3.0 | 1.350 · 0.000 | 3.176 · 0.106 | 1.002 · 0.002 | 0.337 · 0.023 | 12.034 · 0.866 |

## Region summary (means of the cell medians)

| region | mean grid-MAP | mean MCMC median | mean acceptance | mean adapted SD |
|:-------|--------------:|-----------------:|----------------:|----------------:|
| identifiable (α < 1) | 0.400 | 0.406 | 0.352 | 0.136 |
| degenerate (α ≥ 1) | 1.350 | 3.216 | 0.342 | 12.030 |

## Interpretation

**Identifiable region (α < 1): the two estimators agree and track the truth.** For α ∈
{0.1, 0.3, 0.5, 0.7} the grid MAP and the MCMC median coincide and follow the true α (region
means 0.400 vs 0.406). The likelihood is informative, so the prior and the choice of summary
barely matter. Every cell's median R-hat is ≈ 1.00 (worst across the *entire* sweep 1.003 <
the 1.05 threshold), i.e. the chains mixed and the medians are trustworthy.

**Degenerate region (α ≥ 1): MCMC reproduces the Figure-4 clustering the point-MAP cannot.**
The saturation onset lies in (0.7, 1.0] — α = 0.7 still recovers, α = 1.0 is already
degenerate. From there the likelihood flattens and the two estimators diverge sharply: the
grid MAP pins to a single saturated node (1.350 for every degenerate cell — it literally
cannot express a spread), while the MCMC posterior median clusters at ≈ 3.2 (region mean
3.216), pulled up toward the prior-dominated region. This is exactly the paper's Figure-4
pattern — high, prior-driven medians in the degenerate regime — and it is precisely what the
point-MAP is structurally unable to produce. The medians are governed by the half-normal(0, 4)
prior (own median ≈ 2.7), **not** by the data, so this is a statement about *identifiability*,
not a recovered "true" value. The measured cluster (~3.2) sits between the prior median (2.7)
and the paper's quoted ~4 — consistent with the paper's qualitative claim without forcing the
number.

**The adaptation works in both regimes.** The frozen proposal SD tells the story: it settles
at ≈ 0.05–0.31 in the identifiable region (narrow posteriors) and ≈ 12 in the degenerate
region (broad, prior-width posteriors), landing acceptance near the 0.35 target throughout
(region means 0.35 and 0.34). A single fixed proposal could not have done this — a small SD
stalls in the degenerate regime (0.9 acceptance, poor mixing) and a large one stalls in the
identifiable regime — which is why the earlier fixed-SD sampler needed 3000 samples and still
touched R-hat 1.2. Adaptation gets R-hat to ~1.00 everywhere with fewer samples.

**Bottom line for #25.** MCMC delivers what the grid MAP cannot: a full posterior whose
median matches the grid MAP in the identifiable region and reproduces the paper's
degenerate-region clustering (~3.2, well above the identifiable band and above the prior-only
median) where the point-MAP saturates. This unblocks Extension 2 (multi-parameter recovery),
where a grid is intractable.

## Caveats

- Plain random-walk MH, not NUTS/HMC — adequate for this 1-D target but not a gradient sampler.
  Proposal adaptation is confined to burn-in and frozen before sampling (detailed balance intact).
- α-only. Generalizing the likelihood to a parameter vector (γ, A-matrix contents, η/ω, β₀/ψ)
  is Extension 2, explicitly out of #25's scope; this ships the sampler Extension 2 needs.
- Single-agent generation (the grid-MAP baseline's Figure-4 comparison is single-agent).
- Convergence is carried by classic Gelman-Rubin R-hat + acceptance. ESS is **deliberately not
  implemented** (a correct multi-chain ESS is more than a single-lag autocorrelation sum);
  revisit alongside Extension 2. Point estimate is the posterior median, per the paper.
