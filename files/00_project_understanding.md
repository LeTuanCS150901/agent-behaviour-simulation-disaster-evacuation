# 00 — Project Understanding

*Status: finalized from Step 0 checkpoint, approved by research lead.*

## Stage 4 revision (2026-07-20)

The active Stage 4 model is `softmax_c1_v1`. Panic rate is sampled once from a seeded
truncated normal and persisted for the Stage 5 file contract. `W_s/W_g` are raw dimensionless
scores, `S_s/S_g` are normalized softmax shares, and only final `V_p/V_s/V_g` values compose
the shelter distribution. This revision supersedes older direct-W and free-curvature statements
below without rewriting their historical context.

## What is being simulated

A car-based mass evacuation of a synthetic population in Toulouse, triggered by a spreading
fire, with the aim of quantifying how individual behavioral deviation (panic, self-interest)
degrades outcomes relative to a system-optimal, centrally planned shelter assignment. There is
**no learned/trained component** (no LLM, no MARL) — all behavior is governed by explicit,
closed-form, configurable formulas.

## How the six stages fit together

1. **Study area & population** — Toulouse network (OpenStreetMap) and disaster/safe zone
   polygons (Toulouse Métropole open data) define the spatial universe. A synthetic population
   is generated directly from Toulouse census distributions (no `eqasim-france` — see Risk
   Register R-01).
2. **Demand generation** — raw travel demand: departure times (log-normal) per origin
   zone → shelter pair, origins sampled within zones and snapped to SUMO edges.
3. **Shelter allocation (planning layer)** — a p-median optimization computes the
   system-optimal, capacity-constrained shelter assignment. This is what a fully rational,
   fully compliant population *should* do.
4. **Behavioral modeling (individual layer)** — each evacuee's actual shelter choice is a
   panic/selfish/compliant mixture that may deviate from the Stage 3 assignment.
5. **Dynamic route choice** — every `delta_t`, each evacuee re-evaluates their route via a
   panic-weighted mixture of a logit choice model (travel time + hazard exposure) and a
   uniform "panic" distribution over routes.
6. **Fire simulation & hazard injection** — `simfire` produces a fire-front coordinate time
   series independently; this is converted into a SUMO `.add.xml` POI hazard using SUMO's
   default network coordinate system, and feeds the hazard-exposure term used in Stage 5's
   utility function.

## The behavioral model

Each evacuee `i` has a scalar panic rate `x_i ∈ [0,1]` (sampled once, shared identically
between Stage 4 and Stage 5 — confirmed, not resampled per stage). It drives a 3-way mixture
over shelter-choice distributions in Stage 4 (`V_p, V_s, V_g`), and a simpler 2-term mixture
(logit vs. uniform) over route choice in Stage 5, using the same `x_i` as the panic weight
directly.

**Stage 3 vs. Stage 4 distinction (confirmed):**
- Stage 3 is the **planner's problem** — one static, population-level optimal assignment.
- Stage 4 is the **individual's problem** — a probabilistic deviation from that assignment,
  parameterized by personal panic rate. The gap between the two is the quantity of research
  interest.

## Open items resolved during Step 0

| # | Question | Resolution |
|---|---|---|
| 1 | `W_s(x)` under-determination | Superseded for `softmax_c1_v1`: the C1 conditions analytically fix both quadratic pieces; free-curvature configurations are legacy-only. |
| 2 | Logit route-choice utility | `U_k = -α_t · normalized_travel_time_k − α_h · hazard_exposure_k`, with hazard exposure as a log-survival-probability term over edges (see `04_glossary.md`). |
| 3 | Panic rate sharing between Stage 4/5 | Same `x_i`, sampled once per evacuee. |
| 4 | `ω_i`, `β_t`, `β_a` | Global constants shared across the population (not per-agent). |
| 5 | LLM/MARL scope | Confirmed out of scope entirely. |
| 6 | Existing codebase | Greenfield. |
| 7 | Configurability | Population size, shelter count, duration all configurable (Section 5 YAML). |
| 8 | CRS | Use SUMO's default network CRS as the common reference frame; simfire output is transformed into it. |
| 9 | Validation scale | Percentage-of-population is a configurable parameter; start with a small percentage for initial validation runs. |
| 10 | Population source | Toulouse census distributions directly; `eqasim-france` dropped from the pipeline. |

## Engineering risk summary

The most fragile point in the pipeline is the SUMO ↔ simfire coupling: two independent
simulators, no shared clock, one-way offline hand-off (fire time series generated first, then
baked into a static `.add.xml`). CRS misalignment or timing drift would silently invalidate
results without crashing anything. See `05_risk_register.md` R-03/R-04 for mitigations.

## Stage breakdown (final)

1. Study area & population
2. Demand generation
3. Shelter allocation (p-median)
4. Behavioral modeling (panic/selfish/compliant mixture)
5. Dynamic route choice (logit + panic mixture)
6. Fire simulation & SUMO hazard integration
