# 04 — Glossary

## Stage 4 terminology revision (2026-07-20)

- `W_s` / `W_g`: raw dimensionless selfish/compliance scores; finite negative `W_g` is valid.
- `S_s` / `S_g`: stable two-score softmax shares of non-panic behaviour at fixed `T=1`.
- `V_p` / `V_s` / `V_g`: final mixture probabilities; only these compose shelter choice.
- `softmax_c1_v1`: analytical C1 raw-score curve, softmax non-panic split, seeded truncated-normal
  panic sampling, and versioned outputs.

| Term | Definition |
|---|---|
| **p-median problem** | A facility-location optimization: choose an assignment of demand points to a fixed set of facilities (here, shelters) minimizing total weighted travel cost, subject to constraints (here, per-facility capacity). |
| **Softmax** | A function mapping a vector of real-valued scores to a probability distribution: `softmax(a)_j = exp(a_j) / Σ_k exp(a_k)`. Used here to convert shelter attractiveness scores into selfish choice probabilities. |
| **Logit choice model** | A discrete-choice model where the probability of choosing option `k` is proportional to `exp(U_k)` for a linear utility function `U_k`; mathematically a softmax over utilities. |
| **Raw selfish score `W_s(x)`** | A dimensionless analytical C1 score. It is an input to the two-score softmax and is never used directly as a mixture probability. |
| **Raw compliance score `W_g(x)`** | The dimensionless score `1-x-W_s(x)`. A finite negative value is valid because this is a softmax input, not a probability. |
| **Softmax shares `S_s(x)`, `S_g(x)`** | Stable two-score softmax shares at fixed temperature `T=1`; they sum to one and split the non-panic mass. |
| **Final Stage 4 weights `V_p(x)`, `V_s(x)`, `V_g(x)`** | The only shelter-mixture probabilities: `V_p=x`, `V_s=(1-x)S_s`, and `V_g=(1-x)S_g`. |
| **Attractiveness score `A_ij`** | A per-shelter score for evacuee `i` combining normalized travel-time advantage and normalized angular deviation away from the disaster direction. |
| **Panic rate `x_i`** | A scalar in `[0,1]` sampled once per evacuee from the configured seeded truncated normal, persisted unchanged, and shared by the Stage 4 shelter-choice and Stage 5 file contracts. |
| **Departure curve** | The distribution (log-normal, here) governing when evacuees depart, per origin-zone/shelter pair. |
| **SUMO edge** | A directed road segment in SUMO's network representation; the atomic unit onto which trip origins/destinations and routes must be mapped (as opposed to raw lat/lon coordinates). |
| **SUMO POI** | "Point of Interest" — a SUMO additional-file element used here to represent the fire front as a time-stamped marker in the simulation. |
| **TraCI** | SUMO's "Traffic Control Interface" — a Python API for querying and controlling a running SUMO simulation step-by-step, used here for dynamic route re-assignment. |
| **Survival probability `S(k)`** | A route-level measure of hazard exposure, computed as the product (via log-sum-exp) of per-edge survival probabilities along route `k`. |
| **Red zone / blue zone** | Red = disaster-affected/origin side of the study area; blue = safe/destination side, each blue zone hosting exactly one centroid shelter (simplification). |
| **`delta_t`** | The interval, in seconds, at which each evacuee's route choice is re-evaluated during the simulation. |
