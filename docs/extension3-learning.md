# Extension 3 — individual A-learning and group-α recovery

_Waade et al. 2025 §2.1 explicitly omits parameter learning ("we do not include
parameter learning"). This study turns it on: every internal agent learns its
observation model `A` online (Dirichlet pA prior `[1,1,1]`, weak ⇒ fast learning), and
we measure how that reshapes the **recovered group α**. Reproduce-side study
(`crates/reproduce/src/bin/extension3.rs`); the AIF engine is unchanged. Fully
deterministic since issue #2 (master seed `0xE3_2026`, distinct per-rep substreams; no
shared root with the `reproduce` (2026) or `extension11` (0xE11_2026) binaries): each
cell is median · IQR over 5 seeded reps, and re-running reproduces every number below
exactly._

Run: `cargo run --release -p reproduce --bin extension3` (~10 s).

## Two questions

1. **Does individual-level A-learning shift the recovered GROUP α** relative to the
   fixed-A baseline?
2. **Does mis-specified (fixed-A) recovery of learning-group data bias α** — i.e. is the
   learning-aware replay load-bearing for unbiased point recovery?

## Protocol

- Experiment-1 identical group (`GroupAgentBuilder::build_identical`), the paper's
  standard MAB (obs probs `[0.8, 0.2, 0.2]`, prefs `[0.7, 0.3]`), `BanditEnvironment`.
- 300 trials per run; 5 reps per cell; sweep `n ∈ {4, 8, 16}` × true `α ∈ {0.1, 0.3,
  0.5, 0.7, 0.9}`.
- **Matched seeds**: within a rep every arm shares one per-rep seed, so the fixed-A and
  learning groups draw identical internal-agent streams and identical environments —
  they differ ONLY in whether `learn_a` is on (the #2 paired design).
- **fixed-A** (baseline): learning off → its `recover_alpha` fit.
- **misspec**: learning on → `recover_alpha` (fixed-A recovery of learning data).
- **aware**: the same learning-group blanket stream → `recover_alpha_learning` with the
  same `initial_precision` (relearns `A` during the replay — the well-specified fit).
- `gap = aware − misspec` (the mis-specification bias in the recovered α).

## Results (median · IQR over 5 reps)

| n | true α | fixed-A | misspec | aware | gap (aware−misspec) |
|--:|-------:|--------:|--------:|------:|--------------------:|
| 4 | 0.1 | 0.100 · 0.010 | 0.030 · 0.030 | 0.030 · 0.030 | +0.000 · 0.000 |
| 4 | 0.3 | 0.300 · 0.010 | 0.060 · 0.010 | 0.060 · 0.020 | +0.000 · 0.000 |
| 4 | 0.5 | 0.500 · 0.050 | 0.100 · 0.030 | 0.110 · 0.050 | +0.010 · 0.010 |
| 4 | 0.7 | 0.700 · 0.140 | 0.180 · 0.090 | 0.170 · 0.080 | +0.030 · 0.020 |
| 4 | 0.9 | 1.350 · 0.000 | 0.240 · 0.030 | 0.300 · 0.020 | +0.050 · 0.020 |
| 8 | 0.1 | 0.110 · 0.010 | 0.020 · 0.000 | 0.020 · 0.000 | +0.000 · 0.000 |
| 8 | 0.3 | 0.300 · 0.010 | 0.060 · 0.020 | 0.060 · 0.000 | +0.000 · 0.010 |
| 8 | 0.5 | 0.510 · 0.010 | 0.060 · 0.020 | 0.070 · 0.030 | +0.010 · 0.010 |
| 8 | 0.7 | 0.790 · 0.090 | 0.090 · 0.050 | 0.080 · 0.040 | +0.010 · 0.010 |
| 8 | 0.9 | 1.350 · 0.000 | 0.110 · 0.090 | 0.130 · 0.090 | +0.010 · 0.020 |
| 16 | 0.1 | 0.100 · 0.000 | 0.010 · 0.010 | 0.010 · 0.010 | +0.000 · 0.000 |
| 16 | 0.3 | 0.310 · 0.030 | 0.020 · 0.010 | 0.020 · 0.020 | +0.010 · 0.010 |
| 16 | 0.5 | 0.490 · 0.030 | 0.050 · 0.010 | 0.070 · 0.010 | +0.010 · 0.010 |
| 16 | 0.7 | 0.700 · 0.090 | 0.040 · 0.010 | 0.040 · 0.010 | +0.000 · 0.000 |
| 16 | 0.9 | 1.350 · 0.560 | 0.080 · 0.090 | 0.080 · 0.070 | +0.010 · 0.010 |

## Summary (means of the cell medians)

| mean true α | mean fixed-A | mean misspec | mean aware | mean gap |
|------------:|-------------:|-------------:|-----------:|---------:|
| 0.500 | 0.597 | 0.077 | 0.083 | +0.010 |

## Interpretation

**Q1 — individual A-learning shifts the recovered group α sharply DOWNWARD, and that
shift dominates every other effect in the sweep.** The fixed-A baseline recovers the
true α faithfully in the identifiable region (`0.100, 0.300, 0.500, 0.700` at the
matching true αs) and saturates at `≈ 1.35` for true α = 0.9 — exactly the degenerate
high-α behaviour the paper's Figure 4 shows, so the baseline is behaving correctly. Turn
learning on and the *same* group, at the *same* seed, recovers a **much lower** α: the
learning-arm estimates sit at `0.01–0.30` across the whole sweep (mean aware 0.083 vs
mean fixed-A 0.597), and they *fall further with n* (e.g. at true α = 0.7: 0.17 at n=4,
0.08 at n=8, 0.04 at n=16). Mechanism: with a weak pA prior the learned `A` is diffuse
early in each run, which flattens each member's action distribution; the Markov-blanket
aggregation of many such under-confident members yields a group action stream that the
recovery pipeline reads as a low-precision, exploratory agent — i.e. small α. So the
answer is an emphatic **yes**: online A-learning at the individual level makes the group
look far less precise at the blanket level, and more members amplify the effect.

**Q2 — mis-specified (fixed-A) recovery barely biases the POINT estimate, even though
the learning-aware model is a strictly better fit.** The `gap = aware − misspec` is
essentially zero everywhere (mean +0.010, max +0.050 at the n=4/α=0.9 corner), so for
recovering a single α on this MAB you get almost the same number whether or not the
replay relearns `A`. This is *not* because the two models are equivalent: on the same
data the learning-aware model attains a strictly higher maximum log-posterior. (Both the
aware and fixed-A recoveries are blanket-level *approximations* of the group stream — the
aware replay is the well-specified single-agent surrogate, not the literal generative
model for n>1 — so the claim is only that it "fits strictly better than fixed-A", pinned
by the unit test `test_learning_aware_recovery_fits_better_than_misspecified`.) The
takeaway is narrower:
the mis-specification shows up as *fit quality / likelihood*, not as a *point-α* bias,
because both models are driven to the same low-α corner by the flattened action stream.
The learning-aware replay is therefore load-bearing for likelihood-based claims (model
comparison, and the interval/posterior work now available via `recover_alpha_mcmc_learning`,
#25) but not for the α point estimate here.

**Takeaway.** Individual-level A-learning is not a second-order correction at the group
scale — it moves the recovered group α from "tracks the truth" to "looks near-uniform,"
and the shift grows with group size. Whether that recovered α is *biased* depends on what
you ask: the fixed-A point estimate matches the learning-aware one (small gap), so
point-α recovery is robust to the mis-specification, while the *fit* is not (aware always
wins on log-posterior).

_Caveats: one pA prior studied (`[1,1,1]`, weak/fast); learning is A-only (pB/pD/pE not
swept); η/ω at engine defaults (1.0). Recovery here is grid-search MAP over α ∈ [0, 5] step
0.01 under the paper's half-normal(0, 4) prior — a point estimate. Posterior-level (interval)
claims now have a home: MCMC shipped in #25 (`recover_alpha_mcmc[_learning]`,
`docs/extension1-mcmc.md`), where the aware-vs-misspec fit difference is expected to matter
more than it does for the point estimate. The low recovered group α is a property of this
blanket-level recovery pipeline, not a claim about the members' own α._
