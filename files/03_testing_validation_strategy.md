# 03 — Testing & Validation Strategy

## Philosophy

Stage 4 revision (2026-07-20): tests for `softmax_c1_v1` separately verify raw W identities,
stable S-share normalization, final V-weight normalization, finite negative raw `W_g`, NumPy
broadcasting, seeded truncated-normal sampling, exact V-weighted shelter composition, and
unchanged `panic_rate` at the Stage 5 file contract. Manhattan tests are software validation,
not Toulouse or scientific validation.

Three tiers, each with a distinct purpose:

- **Unit tests**: pure-function correctness for formulas in Section 1 (e.g., `W_s(x)` boundary
  values, softmax normalization, survival probability computation). Fast, deterministic, run on
  every change.
- **Integration tests**: correctness across module boundaries using small synthetic fixtures
  (e.g., a toy 3-zone, 2-shelter instance with a hand-computable p-median optimum). Verify data
  contracts hold end-to-end on a scale small enough to reason about by hand.
- **Validation**: research-quality checks against the real Toulouse instance, mostly visual or
  statistical, requiring research-lead review, not just a pass/fail assertion (e.g., does the
  synthetic population's spatial distribution look like Toulouse; does the fire front visually
  track through the red zone sensibly).

## Test-to-KPI mapping

This strategy deliberately does not duplicate the per-stage tables in
`02_implementation_plan.md`; each stage section there lists its own KPIs, Tests, and Validation
Gate. This document defines the *shared conventions* used across all of them:

| Convention | Rule |
|---|---|
| Numerical tolerance | Default `1e-9` for exact algebraic identities (e.g., boundary conditions); `1e-6` for solver outputs (MIP/softmax) unless stated otherwise. |
| Statistical tests | KS test for distributional claims (e.g., departure time, panic rate); report p-value, do not just threshold silently. |
| Reproducibility | Every stochastic test fixes `random_seed` from `configs/base.yaml`; no test may rely on an unseeded RNG. |
| Spatial validation | Any claim of "point within polygon" or "within bounding box" is checked via GeoPandas containment predicates, not approximate distance checks. |

## Project-specific validation checks (called out explicitly per the brief)

1. **Stage 4 score/weight correctness**: the analytical C1 curve satisfies `W_s(0)=ε`,
   `W_s(c)=1-q`, `W_s(1)=0`, and zero one-sided derivatives at `c`; raw-score, S-share,
   and final V-weight identities are checked to `probability_tolerance` — see Stage 4.
2. **p-median correctness**: capacity constraints respected exactly; optimality gap reported
   and, for exact solves, must be 0% — see Stage 3.
3. **Softmax sensitivity**: `P_s` responds monotonically and sensibly to `beta_t`/`beta_a`
   parameter sweeps — see Stage 4.
4. **Fire/SUMO spatial alignment**: 100% of fire-front coordinates, once transformed, fall
   within the SUMO network bounding box — see Stage 6.

## What is explicitly out of scope for automated testing

- Whether the *chosen* default parameter values (`beta_t`, `alpha_t`, `delta_t`, etc. marked
  "assumption — needs validation" in the per-stage config tables) are *realistic* — that is a
  research calibration question, not a software-correctness question, and is tracked instead
  in `05_risk_register.md`.
