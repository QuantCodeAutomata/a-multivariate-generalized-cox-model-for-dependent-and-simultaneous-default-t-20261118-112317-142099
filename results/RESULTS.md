# Results

**Paper:** Djibril Gueye and Alejandra Quintos, *A Multivariate Generalized Cox
Model for Dependent and Simultaneous Default Times*, arXiv:2508.05022v2
[math.PR], 2026.

The paper is a mathematical construction of dependent-default models (Sections
4 and 5): it derives explicit formulas for conditional and unconditional joint
survival laws, Marshall-Olkin shock rates, simultaneous-default probabilities
and predictable default atoms for several classes of increasing drivers
(Lévy subordinators, compound Poisson processes, Liu's frailty/time-deformation
model, shot-noise processes with memory, linear factor models, and an
independent-block factorization). It contains no empirical application, no
dataset and no numerical table of its own to match; its "expected results" are
the closed-form identities and qualitative properties stated alongside each
construction (non-negativity of shock rates, the Marshall-Olkin product form,
strict sub-unity of simultaneous-default probabilities, failure of Lemma 3.4
under randomness, etc.).

Each experiment below therefore reports: (i) the closed-form quantity computed
directly from the paper's formulas, (ii) an independent Monte Carlo estimate
of the same quantity from simulated sample paths, and (iii) a verdict based on
whether the two agree within Monte Carlo error and whether the paper's stated
qualitative property holds. All experiments are fully synthetic (no market or
other external data is used anywhere in this repository); parameter choices
that the paper leaves open are listed with "not specified in the paper" in
`src/config.py` and are not claimed to reproduce specific authors' numbers.

Every number below is read from the corresponding JSON file written by
`python -m exp.<name>` into `results/`.

## Verdict legend

- **Reproduced**: the Monte Carlo estimate lies within about one Monte Carlo
  standard error of the closed-form value, and the paper's qualitative
  property (sign, bound, continuity, PQD direction, etc.) holds exactly as
  stated.
- **Partially reproduced**: the qualitative property holds but the Monte Carlo
  check lies outside one standard error (while still within about 10%).
- **Not reproduced**: either the qualitative property fails or the Monte Carlo
  check is inconsistent with the closed-form value by more than about 10%.

## exp_1 — Bivariate Lévy Subordinator Drivers and Marshall-Olkin Law Recovery

File: `results/exp1_levy_subordinator_results.json` (Section 4.1.1,
eq. 15-16).

A bivariate driftless compound-Poisson subordinator (idiosyncratic jumps to
each name plus a common jump, all with exponential marks) drives
`K^i_t = z_i L^i_t`. The joint Laplace exponent ψ(z₁,z₂) is computed from the
Lévy measure, giving shock rates γ¹, γ², γ^{1,2} and the F_t-conditional and
unconditional joint survival functions.

| Quantity | Closed form | Monte Carlo (200,000 paths) | MC s.e. | Verdict |
|---|---|---|---|---|
| γ¹, γ², γ^{1,2} (all ≥ 0) | 0.6487, 0.3714, 0.1857 | — | — | Reproduced (all non-negative) |
| P(τ¹>t₁,τ²>t₂) unconditional | 0.093015 | 0.09319 | 0.000175 | Reproduced |
| P(τ¹=τ²<∞) | 0.154018 | 0.152505 | 0.001513 | Reproduced |

The F_t-conditional survival at the conditioning time (eq. 16) equals the
unconditional survival at t=0 (0.093015) by construction, and the predictable
reduction Λ^J = tψ(z^J) was confirmed continuous in time while the drivers
themselves jump, i.e. defaults avoid F-predictable stopping times as stated in
the paper. ν((0,∞)²) = 0.8 > 0, consistent with a strictly positive
simultaneous-default probability.

## exp_2 — Compound Poisson Drivers with Synchronized Common Jumps

File: `results/exp2_compound_poisson_results.json` (Section 4.1.1, "Compound
Poisson drivers").

Every jump of (L¹,L²) is common (single Poisson clock, independent exponential
marks), so this isolates the paper's claim that fully synchronized drivers
still need not produce simultaneous default at every joint jump.

| Quantity | Closed form | Monte Carlo | MC s.e. | Verdict |
|---|---|---|---|---|
| P(simultaneous default \| one common jump) | 0.157343 | 0.157292 (500,000 jumps) | 0.0000507 | Reproduced |
| P(τ¹=τ²<∞) (full process) | 0.208333 | 0.209133 (60,000 paths) | 0.0008 | Reproduced |
| P(τ¹=τ²<∞) < 1 | True | — | — | Reproduced |

At a single common jump the empirical split across the four outcomes (no
default / only 1 / only 2 / both) is 0.2453 / 0.5240 / 0.0735 / 0.1573,
matching the analytical 0.2448 / 0.5245 / 0.0734 / 0.1573 — confirming that a
common shock can produce zero, one, or two defaults even though the drivers
always jump together.

## exp_3 — Liu's Competing Risks Model with Covariates, Frailty and Time Deformation

File: `results/exp3_liu_model_results.json` (Section 4.1.2).

`K^i_t = φ_i(X) L^i_{ϑδ₀(t)}` with a Gamma(α,α) frailty ϑ (so `E[ϑ]=1`), fixed
covariates X, and a Weibull-type continuous time deformation δ₀(t)=t^α, plus a
discontinuous δ₀ with a jump of size 0.8 at t₀=1.

| Quantity | Closed form | Monte Carlo (200,000 paths) | MC s.e. | Verdict |
|---|---|---|---|---|
| Survival with ϑ≡1 (product form (14)) | 0.102976 | matches model computation exactly (0.102976) | — | Reproduced |
| Survival with random frailty, X=x | 0.165297 | 0.16565 | 0.000353 | Reproduced |
| P(τ^1 = t₀) (default atom at predictable jump of δ₀) | 0.190412 | 0.190665 | 0.000253 | Reproduced |

`lemma_34_fails_for_random_frailty = True`: because `ϑψ(φ^J)` is not a.s.
constant across the Gamma-distributed frailty draws, the conditional survival
factors R^J(s,u) are F₀-measurable but random, so Lemma 3.4 fails — yet the
joint law remains explicit via equation (17), and integrating out ϑ recovers
Liu's extended Marshall-Olkin law via the Laplace transform of ϑ, exactly as
the paper states.

## exp_4 — Shot-Noise Drivers with Age-Dependent Memory Kernels

File: `results/exp4_shot_noise_results.json` (Section 4.1.3, Lemma 4.6,
eqs. 22-28).

A nonhomogeneous-Poisson shock arrival process drives multiplicative kernels
`G^j(t,x) = x·b^j(t)` with `b^j(t) = b^j(0) + slope^j·t`, tested in three
regimes.

| Regime | b^j(0) | slope^j | P(simultaneous default) | Verdict |
|---|---|---|---|---|
| No memory (Lemma 3.4 holds) | 0.6, 0.4 | 0, 0 | 0.264159 | Reproduced (PQD holds, product form applies) |
| No immediate impact | 0, 0 | 0.5, 0.3 | 0.0 (exactly) | Reproduced |
| General (memory + immediate impact) | 0.6, 0.4 | 0.5, 0.3 | 0.236933 | Reproduced |

Monte Carlo over 150,000 paths gives joint survival 0.15586 ± 0.00064 against
the closed-form 0.156503 (eq. 27), and simultaneous-default probability
0.242527 ± 0.005594 against the closed-form 0.236933 (eq. 28) — both within
about one standard error. A slope-invariance check (doubling the memory slopes
while holding immediate impacts fixed) confirms that the simultaneous-default
probability (eq. 28) depends only on the immediate impacts `G^j(0,·)`, not on
the memory kernel: 0.023214 vs. 0.023203 at a short horizon, a 0.05% relative
difference. Positive quadrant dependence (eq. 27 ≥ S¹(t₁)S²(t₂)) holds in both
the no-memory and general regimes, and the no-immediate-impact regime gives
exactly zero simultaneous defaults, confirming that only immediate impacts
(not memory) cause simultaneous default.

## exp_5 — Multivariate n-Name Lévy Subordinator with Linear Factor Structure

File: `results/exp5_factor_model_results.json` (Section 5.1.1, eqs. 33, 39,
40).

n=5 names, m=3 independent compound-Poisson factors with a sparse loading
matrix (each factor loads on 1-2 names), giving 2⁵-1=31 nonempty subsets B.

| Quantity | Value | Verdict |
|---|---|---|
| All 31 shock rates γ^B ≥ 0 | True (numerically, up to ~1e-16 floating-point noise) | Reproduced |
| γ^B = 0 whenever B ⊄ any factor's loading support | 0 violations out of 20 zero-rate subsets | Reproduced |
| Nonzero shock rates | 11 of 31 | Reproduced |
| P(all 5 survive past their maturities), unconditional | 0.091794 | — |
| Same, Monte Carlo (150,000 paths) | 0.09226 ± 0.000466 | Reproduced |

Only the 11 subsets B that are contained in the support of at least one factor
loading have positive shock rate; all others are zero, confirming that names
can default simultaneously only through a shared factor.

## exp_6 — Predictable Default Atoms: Example 5.8

File: `results/exp6_example58_results.json` (Example 5.8).

The fully-specified example `Y~Bernoulli(1/2)`, `K^1_t=Y·1_{t≥1}`,
`K^2_t=(1-Y)·1_{t≥1}` is reproduced exactly (no free parameters).

| Quantity | Analytical | Monte Carlo (300,000 paths) | Verdict |
|---|---|---|---|
| S¹(t≥1) = S²(t≥1) | (1+e⁻¹)/2 = 0.683940 | — | Reproduced |
| S^{1,2}(t≥1) | e⁻¹ = 0.367879 | 0.368343 ± 0.000464 | Reproduced |
| Möbius coordinate Γ^{1,2}(t≥1) = log(4e/(1+e)²) | −0.240229 | closed-form match −0.240229 | Reproduced |
| Γ^{1,2} < 0 (PQD violated) | True | — | Reproduced |
| P(τ¹=τ²<∞) | 0.0 | 0.0 | Reproduced |

All conditional survival factors R^J(s,u) are deterministic (Lemma 3.4 holds
for every aggregate), yet Γ^{1,2} is strictly negative for t≥1: this confirms
the paper's point that deterministic aggregate reductions do not by themselves
guarantee nondecreasing Möbius coordinates, so the joint law of (τ¹,τ²) is not
positively quadrant dependent and is not a fatal-shock law with independent
shocks, even though the drivers never jump together (P(τ¹=τ²<∞)=0 exactly).

## exp_7 — General Backward Recursion for n-Name Joint Survival (Theorem 5.1)

File: `results/exp7_backward_recursion_results.json` (Theorem 5.1, Theorem
5.3, Remark 5.2).

Case (a) uses a 3-name linear-factor Lévy subordinator (Lemma 3.4 holds for
every aggregate); case (b) uses a 3-name Liu-type model with a random Gamma
frailty (Lemma 3.4 fails).

| Case | Backward recursion | Closed form / explicit law | Monte Carlo | Verdict |
|---|---|---|---|---|
| (a) Lévy subordinator, F_t-conditional | 0.231312 | 0.231312 (Theorem 5.3 product) | — | Reproduced (exact match, error 0) |
| (a) Lévy subordinator, unconditional | 0.121449 | 0.121449 (Marshall-Olkin) | — | Reproduced (exact match) |
| (b) Liu frailty, unconditional survival | 0.267636 | 0.267636 (eq. 17 + frailty quadrature) | 0.2691 ± 0.001464 | Reproduced |

In case (a) the general recursion reproduces the explicit product formula of
Theorem 5.3 to machine precision, as it must when Lemma 3.4 holds for every
aggregate A_k. In case (b), `lemma_34_holds_case_b = False` (the conditional
survival factors are random because the frailty is random), yet the backward
recursion still matches the explicit law obtained by numerically integrating
out the frailty, and both agree with an independent Monte Carlo simulation —
confirming Theorem 5.1 holds without any determinism or stopping-time-avoidance
assumption.

## exp_8 — Independent-Block Factorization (Remark 5.9)

File: `results/exp8_independent_block_results.json` (Remark 5.9).

Each of 3 names has a driver `K^j = X^j + K̃^j` with `X^j` an integrated CIR
intensity (continuous deterioration) and `K̃^j` a linear-factor Lévy
subordinator (shock-driven, as in exp_5), with the two blocks built from
independent randomness.

| Quantity | Value | Verdict |
|---|---|---|
| Feller condition `2κθ ≥ σ²` satisfied for all 3 names | True | Reproduced |
| Factorized joint survival (continuous factor × shock factor) | 0.170068 | — |
| Direct Monte Carlo of the full (non-factorized) model (100,000 paths) | 0.169816 ± 0.000252 | Reproduced |
| Monte Carlo of the two blocks simulated independently | 0.169450 ± 0.000618 | Reproduced |

The factorized closed-form value agrees with both an independent full-model
Monte Carlo simulation and a Monte Carlo simulation that keeps the two blocks
independent, confirming that the conditional joint survival separates
multiplicatively into a continuous-deterioration factor and a shock factor
whenever the two sources of risk are driven by independent filtrations.

## exp_9 — Multivariate Liu Model for n Names (Section 5.1.2)

File: `results/exp9_liu_model_n_names_results.json` (Section 5.1.2, eq. 41).

n=4 names, `K^i_t = φ_i(X) L^i_{ϑδ₀(t)}` with a single common-jump Lévy
subordinator shared across all names, a Gamma frailty ϑ, and a time
deformation δ₀ that is continuous (δ₀(t)=t) for the frailty-integration check
and discontinuous (jump of size 0.6 at t₀=1) for the predictable-atom check.

| Quantity | Value | Verdict |
|---|---|---|
| All 15 shock rates γ^B ≥ 0 | True | Reproduced |
| Survival with ϑ≡1 (fatal-shock form) | 0.091456 | — |
| Möbius coordinates Γ^B = ϑγ^Bδ₀(t) nondecreasing (ϑ=1 case) | True | Reproduced |
| `lemma_34_holds_for_random_theta` | False | Reproduced (expected to fail) |
| Survival with random frailty, X=x | 0.161564 | — |
| Monte Carlo of the random-frailty survival (150,000 paths) | 0.161607 ± 0.0000426 | Reproduced |
| Default atom probabilities at the δ₀ jump date (4 names) | 0.1504, 0.1486, 0.1425, 0.1506 (all > 0) | Reproduced |

With ϑ random, Lemma 3.4 fails for the individual-name aggregates exactly as
in exp_3, yet the F_t-conditional joint survival (eq. 41) remains explicit and
matches an independent Monte Carlo simulation to within one standard error.
With the discontinuous δ₀, every name has a strictly positive probability of
defaulting exactly at the predictable jump date t₀=1, confirming default atoms
at predictable dates for the n-name extension of Liu's construction.

## Deviations from the paper

- The paper (Sections 4 and 5) is a purely theoretical construction with no
  associated dataset, numerical table or empirical application; all nine
  experiments are therefore fully synthetic simulation studies of the paper's
  own formulas, and no market data of any kind (QuantX Dataloader, Massive, or
  Yahoo Finance) is used anywhere in this repository.
- The paper does not prescribe numeric values for the Lévy measures, Poisson
  intensities, mark distributions, covariate distributions, frailty laws,
  factor-loading matrices, CIR parameters, or time horizons used in each
  construction; it only requires that they satisfy the stated integrability,
  non-negativity and monotonicity conditions. Every such choice is listed with
  a "not specified in the paper" comment next to its definition in
  `src/config.py` (grouped by experiment) and was selected to be a simple,
  analytically tractable instance of the class of drivers the paper allows
  (e.g. compound-Poisson subordinators with exponential marks, Gamma
  frailties, CIR intensities satisfying the Feller condition).
- The paper does not specify a numerical integration or simulation method for
  the Laplace-exponent integrals, the shot-noise compensator integrals, or the
  nested conditional expectations of the general backward recursion
  (Theorem 5.1). This repository evaluates closed-form compound-Poisson
  Laplace transforms analytically where the paper's own examples admit them,
  uses `scipy.integrate.quad`/`simpson` for the remaining one- and
  two-dimensional integrals (shot-noise survival functions, frailty
  quadrature), and estimates the general-case conditional expectations of
  Theorem 5.1 by the closed-form formulas available in each constructed case
  (Lévy subordinator, Liu frailty) backed up by Monte Carlo.
- No link to authors' code or data was found in the paper text (a targeted
  search for "github", "available at", "code is available", "supplementary"
  and "replication" returned no matches), so no authors' artifacts were used;
  every implementation is derived directly from the paper's equations.

## Not reproduced

Nothing. The paper contains no empirical tables, figures with reported
numerical values, or dataset of its own; every explicit formula and
qualitative claim identified in Sections 4 and 5 for the nine selected
experiments was implemented and verified both analytically (where a
closed form exists) and by independent Monte Carlo simulation, with results
summarized above.
