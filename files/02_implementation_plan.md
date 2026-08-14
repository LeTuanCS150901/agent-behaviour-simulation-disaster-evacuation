# 02 — Implementation Plan

Master phased plan. Stages are implemented sequentially and gated — each stage requires an
approved `reports/stageN_report.md` before the next begins (see `06` workflow rules in the
project brief).

---

## Stage 1: Study Area & Population

### Problem Description
Defines the spatial universe (Toulouse road network, red/blue zone polygons) and generates a
synthetic population with configurable size, distributed across zones consistent with Toulouse
census data. Everything downstream depends on this stage's outputs.

### Scope & Assumptions
- In scope: OSM network extraction for Toulouse, ingestion of Toulouse Métropole zone/quartier
  boundaries, red/blue zone tagging, population synthesis from census marginals.
- Out of scope: demand generation, routing, behavior (later stages).
- Assumes: a bounding box / administrative boundary for the study area has been agreed with
  the research lead before extraction begins.
- `eqasim-france` is **not** used (confirmed out of scope) — population is synthesized directly
  from Toulouse census distributions (e.g., INSEE zone-level population counts and available
  demographic marginals).

### Tasks
1. Define and freeze the Toulouse study-area bounding polygon.
2. Extract SUMO network via OSM (netconvert / OSMWebWizard or equivalent).
3. Ingest Toulouse Métropole zone/quartier boundaries as GeoJSON; tag each zone `red` or `blue`
   per the disaster-affected/safe-side definition agreed with the research lead.
4. Ingest INSEE (or equivalent) census counts at zone level for Toulouse.
5. Implement population synthesis: sample `N` synthetic agents (configurable) distributed
   across zones proportional to census population counts.
6. Validate zone-level synthetic counts against census targets.
7. Produce a diagnostic map (zones colored red/blue, population density overlay).

### Proposed Solution / Approach
Population synthesis is a proportional allocation problem: for a configured total `N`, allocate
`n_z = round(N * pop_z / pop_total)` agents to zone `z`, then sample home locations within
`z`'s polygon using `geopandas.sample_points`. This is simpler than a full IPF/synthetic-population
pipeline (no `eqasim-france`) and is explicitly documented as a simplification — see
`05_risk_register.md` R-01.

### Tools / Frameworks / Algorithms
- OSMnx / SUMO's OSM import tools (`netconvert`) — standard for SUMO network generation.
- GeoPandas — zone polygon handling, point sampling.
- INSEE open data (or equivalent Toulouse Métropole source) — census counts, justified as the
  most granular public source for zone-level population.

### Example Code
```python
def synthesize_population(zones_gdf, total_population, rng):
    zones_gdf["n_agents"] = (
        zones_gdf["census_pop"] / zones_gdf["census_pop"].sum() * total_population
    ).round().astype(int)
    agents = []
    for _, zone in zones_gdf.iterrows():
        pts = zone.geometry.sample_points(zone.n_agents, rng=rng)
        agents.extend([{"zone_id": zone.zone_id, "geometry": p} for p in pts.geoms])
    return gpd.GeoDataFrame(agents, crs=zones_gdf.crs)
```

### Configuration Parameters
| Name | Meaning | Unit | Range/Type | Default | Source |
|---|---|---|---|---|---|
| `total_population` | Number of synthetic evacuees | count | int > 0 | 5000 | assumption — needs validation |
| `study_area_bbox` | Study area bounding polygon | GeoJSON | polygon | — | agreed with research lead |
| `crs` | Common coordinate reference system | — | EPSG code | SUMO network default | Step 0 answer (8) |
| `random_seed` | RNG seed for reproducibility | — | int | 42 | engineering default |

### KPIs / Success Metrics
- Synthetic zone-level population counts within a configurable tolerance (default 5%) of
  census targets.
- 100% of sampled agent points fall within their assigned zone polygon.
- Network extraction covers 100% of the defined bounding polygon with no disconnected
  components larger than a configurable threshold.

### Tests
- Unit: `synthesize_population` allocates agents summing exactly to `total_population`.
- Unit: sampled points are within polygon bounds (geometry containment check).
- Integration: zone-level counts vs. census counts within tolerance — mapped to KPI 1.
- Validation: visual diagnostic map reviewed against known Toulouse geography.

### Validation Gate
- [ ] Network extracted and loads cleanly in SUMO (`netconvert` succeeds, no critical warnings).
- [ ] Zone counts within tolerance of census data.
- [ ] Red/blue zone tagging reviewed and confirmed by research lead.
- [ ] Diagnostic map produced and reviewed.

### Disciplines / Expertise Required
GIS/geospatial specialist; familiarity with SUMO network tooling.

### Risks Specific to This Stage
See `05_risk_register.md` R-01 (population source adaptation).

### Dependencies
None — first stage.

---

## Stage 2: Demand Generation

### Problem Description
Converts the static synthetic population into a time-resolved trip table: who leaves, from
where, when, headed toward which side of the city (shelter assignment itself is Stage 3's job
— this stage only establishes zone-level demand and SUMO-edge-mapped origins).

### Scope & Assumptions
- In scope: departure time sampling (log-normal), origin point sampling within zones, SUMO
  edge-mapping of origins.
- Out of scope: shelter assignment (Stage 3), behavioral deviation (Stage 4).
- Assumes Stage 1's population and network outputs are final.
- Simplification carried from the spec: each blue (safe-side) zone has exactly one centroid
  shelter.

### Tasks
1. Compute one centroid shelter location per blue zone.
2. Sample a departure time per agent from a log-normal distribution parameterized per
   origin-zone → shelter-destination pair.
3. Map each agent's sampled origin point to the nearest valid SUMO edge (not raw coordinates).
4. Assemble the trip table: `agent_id, origin_edge, dep_time, candidate_shelter_zone`.
5. Validate: no trips assigned to non-routable edges (e.g., pedestrian-only, disconnected).

### Proposed Solution / Approach
Origin-to-edge mapping uses SUMO's network object to find the nearest edge allowing the
vehicle class in use, with a maximum snap-distance sanity check to catch pathological cases
(e.g., a sampled point far from any road).

### Tools / Frameworks / Algorithms
- `sumolib` — network querying and edge snapping (standard SUMO Python tooling).
- `numpy`/`scipy.stats.lognorm` — departure time sampling.

### Example Code
```python
def map_to_edge(net, point, vclass="passenger", max_snap_dist=200):
    edge, dist = net.getNeighboringEdges(point.x, point.y, r=max_snap_dist)[0]
    if dist > max_snap_dist:
        raise ValueError(f"No routable edge within {max_snap_dist}m of {point}")
    return edge.getID()
```

### Configuration Parameters
| Name | Meaning | Unit | Range/Type | Default | Source |
|---|---|---|---|---|---|
| `departure_lognorm_mu` | Log-normal location parameter for departure time | — | float | assumption | needs validation |
| `departure_lognorm_sigma` | Log-normal scale parameter | — | float > 0 | assumption | needs validation |
| `max_snap_distance` | Max distance for origin→edge snapping | m | float > 0 | 200 | engineering default |

### KPIs / Success Metrics
- 100% of trips map to a valid, routable SUMO edge.
- Departure time distribution's empirical mean/variance matches configured log-normal
  parameters within sampling tolerance.

### Tests
- Unit: edge-mapping raises on out-of-range points.
- Unit: sampled departure times follow the configured log-normal distribution (KS test).
- Integration: full trip table has no null/invalid edges.

### Validation Gate
- [ ] Trip table fully populated, zero invalid edges.
- [ ] Departure time distribution validated statistically.
- [ ] One centroid shelter computed per blue zone, reviewed.

### Disciplines / Expertise Required
Transportation/GIS engineer.

### Risks Specific to This Stage
Edge-snapping failures at zone boundaries near network gaps — see `05_risk_register.md` R-05.

### Dependencies
Stage 1 complete and approved.

---

## Stage 3: Shelter Allocation (p-median)

### Problem Description
Computes the planner's system-optimal shelter assignment: minimize total population travel
time subject to per-shelter capacity constraints. This is the baseline against which Stage 4's
individual behavioral deviation is measured.

### Scope & Assumptions
- In scope: p-median formulation, capacity constraints, solving at zone or agent granularity
  (to be decided based on solve-time — see Open Question resolved via KPI below).
- Out of scope: individual deviation from this assignment (Stage 4).
- Assumes total shelter capacity ≥ total demand (with configurable slack), per the original
  spec.

### Tasks
1. Compute a travel-time matrix between origin zones (or origin edges) and candidate shelters
   using SUMO network shortest paths (not Euclidean distance).
2. Formulate the p-median problem: minimize Σ (demand_i * t_ij * assignment_ij) subject to
   Σ_i assignment_ij ≤ capacity_j and Σ_j assignment_ij = 1 for all i.
3. Solve using an exact MIP solver for validation-scale runs; benchmark against a heuristic
   for full-scale runs if solve time is prohibitive.
4. Validate capacity constraints are respected and record the optimality gap.
5. Output the assignment table (zone/agent → shelter).

### Proposed Solution / Approach
Start with an exact MIP formulation (small validation-scale population percentage, per Step 0
answer 9) to establish a ground-truth optimal baseline; only fall back to a heuristic
(e.g., greedy or Lagrangian relaxation) if solve time at full scale is empirically shown to be
prohibitive — this order (exact-first) is deliberate so KPIs can be benchmarked against a known
optimum before any heuristic is trusted.

### Tools / Frameworks / Algorithms
- **PuLP** or **OR-Tools** (CP-SAT/MIP) — chosen over a hand-rolled heuristic for the initial
  implementation because the problem is a well-studied MIP with mature open-source solvers,
  and an exact/near-exact solution is needed as the behavioral-deviation baseline.
- SUMO network shortest-path queries (via `sumolib` / `duarouter`) for the travel-time matrix.

### Example Code
```python
import pulp

def solve_p_median(demand, capacity, travel_time):
    zones, shelters = demand.index, capacity.index
    prob = pulp.LpProblem("p_median", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("assign", (zones, shelters), 0, 1, cat="Continuous")
    prob += pulp.lpSum(demand[i] * travel_time[i][j] * x[i][j] for i in zones for j in shelters)
    for i in zones:
        prob += pulp.lpSum(x[i][j] for j in shelters) == 1
    for j in shelters:
        prob += pulp.lpSum(demand[i] * x[i][j] for i in zones) <= capacity[j]
    prob.solve()
    return x
```

### Configuration Parameters
| Name | Meaning | Unit | Range/Type | Default | Source |
|---|---|---|---|---|---|
| `shelter_capacity_slack` | Extra capacity beyond total demand | fraction | float ≥ 0 | 0.1 | original spec (qualitative: "slightly larger") |
| `solver` | MIP solver backend | — | enum {pulp, ortools} | pulp | engineering default |
| `solve_granularity` | Assignment granularity | — | enum {zone, agent} | zone | assumption — needs validation |

### KPIs / Success Metrics
- Zero capacity violations in the solution.
- Reported optimality gap (0% for exact solve; explicit % for heuristic fallback).
- Solve time recorded and reported against population-percentage parameter.

### Tests
- Unit: solution respects `Σ assignment_ij == 1` and capacity constraints exactly.
- Integration: solve on a small synthetic instance with known optimal cost, verify match.
- Validation: solve time vs. population-percentage curve produced as a diagnostic.

### Validation Gate
- [ ] Capacity constraints respected on all validation runs.
- [ ] Optimality gap reported and within acceptable bound (0% exact, or documented % for heuristic).
- [ ] Solve-time-vs-scale curve reviewed by research lead before scaling up.

### Disciplines / Expertise Required
Operations research / optimization specialist.

### Risks Specific to This Stage
Solve time at city scale; capacity feasibility — see `05_risk_register.md` R-06, R-07.

### Dependencies
Stage 2 complete and approved (needs trip table and travel-time matrix inputs).

---

## Stage 4: Behavioral Modeling — active revision 2026-07-20

The active model is `softmax_c1_v1`. It uses analytical C1 coefficients, treats `W_s/W_g` as
raw scores, applies stable two-score softmax to obtain `S_s/S_g`, and defines final probabilities
`V_p=x`, `V_s=(1-x)S_s`, and `V_g=(1-x)S_g`. Shelter choice is composed only as
`V_g P_g + V_s P_s + V_p P_p`. Panic rates use seeded `scipy.stats.truncnorm` sampling and are
persisted unchanged for Stage 5. `theta_2` and `beta_2` are incompatible with the active model;
historical configurations are separately versioned. Post-behaviour shelter loads and overflow
are mandatory outputs. All parameter calibration and Toulouse validation remain provisional.

### Superseded historical Stage 4 plan

The remaining Stage 4 material in this section documents the former direct-W/free-curvature
design and is retained only as historical development context. It must not be implemented or
used to interpret `softmax_c1_v1` outputs.

### Problem Description
Models each evacuee's actual shelter-choice behavior as a mixture of panic, selfish, and
compliant modes, producing a probability distribution over shelters per agent, from which an
actual chosen shelter is sampled. This is where the individual's behavior may deviate from the
Stage 3 planner assignment.

### Scope & Assumptions
- In scope: panic rate sampling, `W_p/W_s/W_g(x)` weight computation, `A_ij` attractiveness
  scoring, softmax selfish choice, mixture sampling of final shelter choice.
- Out of scope: en-route route choice (Stage 5) — this stage only decides *which shelter*, not
  *which path*.
- Assumes Stage 3's assignment table is available as `P_g`'s deterministic target.
- Assumes `ω_i`, `β_t`, `β_a` are global constants (Step 0 answer 4).

### Tasks
1. Sample panic rate `x_i` per agent from a truncated/clipped Gaussian on `[0,1]` (initial
   choice per spec; to be revisited against literature distributions — Beta, truncated
   log-normal — as a sensitivity check, per prior literature review).
2. Implement `W_p(x) = x`.
3. Implement piecewise `W_s(x)` per the resolved boundary conditions (Section 1.1), with
   `θ₂`, `β₂` as configurable curvature parameters (default `0`), solving `θ₀,θ₁,β₀,β₁`
   algebraically at runtime from `ε, c, q` and the chosen curvature.
4. Implement `W_g(x) = 1 - W_p(x) - W_s(x)`; assert non-negativity for all sampled `x`.
5. Compute `A_ij` attractiveness scores using travel time (`t_ij`) and disaster-relative angle
   (`a_ij`) per the Section 1.2 formula.
6. Compute `P_s(j) = softmax_j(A_ij)`.
7. Compute the final mixture `P_ij = W_g·P_g + W_s·P_s + W_p·P_p` and sample the agent's actual
   chosen shelter from it.
8. Produce the `W_s(x)` diagnostic plot over `[0,1]` (mandatory per spec).

### Proposed Solution / Approach
The curve-fitting for `W_s(x)` is solved as a small linear system per piece (2 equations, 2
remaining unknowns once curvature is fixed), not via general-purpose numerical curve fitting —
the system is exactly determined once `θ₂`/`β₂` are chosen, so an exact linear solve is more
transparent and reproducible than a fitted approximation.

### Tools / Frameworks / Algorithms
- `numpy.linalg.solve` — exact linear system solve for the piecewise coefficients (chosen over
  `scipy.optimize.curve_fit` because the system is exactly determined, not a fitting problem).
- `scipy.stats.truncnorm` — panic rate sampling.
- `scipy.special.softmax` — selfish shelter-choice probabilities.
- `matplotlib` — `W_s(x)` diagnostic plot.

### Example Code
```python
def solve_w_s_coefficients(epsilon, c, q, theta_2=0.0, beta_2=0.0):
    # Left piece: theta_0 = epsilon; theta_0 + theta_1*c + theta_2*c^2 = 1-q
    theta_0 = epsilon
    theta_1 = ((1 - q) - theta_0 - theta_2 * c**2) / c
    # Right piece: beta_0 + beta_1*c + beta_2*c^2 = 1-q; beta_0 + beta_1 + beta_2 = 0
    A = np.array([[1, c], [1, 1]])
    b = np.array([1 - q - beta_2 * c**2, -beta_2])
    beta_0, beta_1 = np.linalg.solve(A, b)
    return (theta_0, theta_1, theta_2), (beta_0, beta_1, beta_2)

def w_s(x, coeffs_left, coeffs_right, c):
    t0, t1, t2 = coeffs_left
    b0, b1, b2 = coeffs_right
    return np.where(x < c, t0 + t1*x + t2*x**2, b0 + b1*x + b2*x**2)
```

### Configuration Parameters
| Name | Meaning | Unit | Range/Type | Default | Source |
|---|---|---|---|---|---|
| `epsilon` | `W_s(0)` boundary value | — | 0 < ε < 1-q | 0.01 | original spec |
| `c` | Panic-rate threshold splitting the two pieces | — | 0 < c < 1 | 0.30 | original spec |
| `q` | Boundary parameter, `W_s(c) = 1-q` | — | 0 < q < 1 | 0.30 | original spec |
| `theta_2` | Left-piece curvature (free parameter) | — | float | 0.0 | assumption — needs validation (Step 0 answer 1) |
| `beta_2` | Right-piece curvature (free parameter) | — | float | 0.0 | assumption — needs validation (Step 0 answer 1) |
| `omega` | Global weight: travel-time vs. angle importance | — | 0 ≤ ω ≤ 1 | assumption | needs validation |
| `beta_t` | Sensitivity: travel time term in `A_ij` | — | float > 0 | assumption | needs validation |
| `beta_a` | Sensitivity: angle term in `A_ij` | — | float > 0 | assumption | needs validation |
| `panic_rate_distribution` | Distribution family for `x_i` | — | enum {gaussian_clipped, beta, trunc_lognormal} | gaussian_clipped | original spec, literature alternatives noted |

### KPIs / Success Metrics
- `W_s(x)` boundary conditions satisfied to numerical tolerance (`1e-9`) at `x=0, c, 1`.
- `W_p + W_s + W_g = 1` for all sampled `x`, and all three weights remain in `[0,1]`.
- Softmax output sums to 1 per agent; monotonic response to `beta_t`/`beta_a` sweeps (higher
  `beta_t` sharpens preference toward minimum-travel-time shelter).

### Tests
- Unit: `solve_w_s_coefficients` boundary conditions — mapped to KPI 1.
- Unit: weight non-negativity and sum-to-1 — mapped to KPI 2.
- Unit: softmax sums to 1 — mapped to KPI 3.
- Integration: parameter sweep of `beta_t`/`beta_a` produces monotonic, sensible shifts in
  `P_s` — mapped to KPI 3.
- Validation: `W_s(x)` diagnostic plot visually reviewed by research lead.

### Validation Gate
- [ ] `W_s(x)` boundary conditions verified numerically and plot produced.
- [ ] Weight sum-to-1 and range checks pass for a large sample of `x`.
- [ ] Softmax parameter sweep reviewed and sensible.
- [ ] Curvature parameter defaults (`theta_2=0, beta_2=0`) explicitly confirmed or overridden
      by research lead.

### Disciplines / Expertise Required
Behavioral modeling / statistics specialist.

### Risks Specific to This Stage
`W_s(x)` under-determination — see `05_risk_register.md` R-02.

### Dependencies
Stage 3 complete and approved (needs assignment table for `P_g`).

---

## Stage 5: Dynamic Route Choice

### Problem Description
Recomputes each evacuee's route-choice distribution every `delta_t`, mixing a logit model
(driven by travel time and hazard exposure) with a uniform panic distribution over available
routes, using the same panic rate `x_i` sampled in Stage 4.

### Scope & Assumptions
- In scope: logit utility computation (`U_k`), hazard-exposure/survival-probability term,
  panic-weighted mixture, periodic re-evaluation via TraCI.
- Out of scope: shelter choice (already fixed by Stage 4's output for this evacuation).
- Assumes live or pre-computed hazard state (edge-level distance-to-disaster) is available at
  each `delta_t` tick — supplied by the Stage 6 fire-front time series.

### Tasks
1. Implement `normalized_travel_time_k` for each candidate route `k`.
2. Implement per-edge survival probability `P_i = d_i / d_max`, where `d_i` is Euclidean
   distance from edge `i` to the nearest disaster location and `d_max` is the configured safe
   distance parameter.
3. Implement route-level survival `S(k) = exp(Σ log(P_i))` over edges in route `k`.
4. Implement `U_k = -alpha_t * normalized_travel_time_k - alpha_h * (1 - S(k))` as the hazard
   cost (using `1 - S(k)` so that lower survival probability increases cost — flagged as an
   explicit interpretation choice, see note below).
5. Implement `P(route_k)` as the softmax/logit over `U_k`.
6. Implement the final mixture `P_route_k = (1 - x_i) * P(route_k) + x_i * P(panic)`.
7. Wire into TraCI: re-evaluate and reassign route every `delta_t` per agent.

> **Note on hazard-exposure sign convention:** the spec defines `S(k)` as a survival
> probability (higher = safer) but `U_k` is defined with a negative coefficient on
> "`hazard_exposure_k`" as if higher hazard_exposure = lower utility. We interpret
> `hazard_exposure_k = 1 - S(k)` (i.e., "risk", not "survival") so the signs are consistent;
> this is flagged as an explicit modeling choice for research-lead confirmation, not silently
> assumed.

### Proposed Solution / Approach
Route candidates are generated once per agent per re-evaluation window using k-shortest-paths
(or SUMO's alternative-route generation), then scored via the utility function above — this
avoids an intractable search over all possible paths.

### Tools / Frameworks / Algorithms
- TraCI — periodic route re-assignment.
- `sumolib`/`duarouter` — k-shortest-path candidate route generation.
- `scipy.special.softmax` — logit route-choice probabilities.

### Example Code
```python
def route_survival(route_edges, edge_dist_to_disaster, d_max):
    p = np.clip(np.array(edge_dist_to_disaster) / d_max, 1e-9, 1.0)
    return np.exp(np.sum(np.log(p)))

def route_utility(norm_travel_time, survival, alpha_t, alpha_h):
    hazard_exposure = 1 - survival
    return -alpha_t * norm_travel_time - alpha_h * hazard_exposure

def route_choice_mixture(utilities, panic_rate):
    p_logit = softmax(utilities)
    p_panic = np.full_like(p_logit, 1 / len(p_logit))
    return (1 - panic_rate) * p_logit + panic_rate * p_panic
```

### Configuration Parameters
| Name | Meaning | Unit | Range/Type | Default | Source |
|---|---|---|---|---|---|
| `delta_t` | Route re-evaluation interval | s | float > 0 | 60 | assumption — needs validation |
| `alpha_t` | Utility weight on travel time | — | float > 0 | assumption | needs validation |
| `alpha_h` | Utility weight on hazard exposure | — | float > 0 | assumption | needs validation |
| `d_max` | Max safe distance from disaster (survival normalization) | m | float > 0 | assumption | needs validation |
| `k_alternative_routes` | Number of candidate routes per re-evaluation | count | int > 0 | 3 | engineering default |

### KPIs / Success Metrics
- Route-choice mixture probabilities sum to 1 per agent per tick.
- Higher `x_i` (panic) empirically correlates with more uniform (less optimal) route
  selection, verified via entropy of `P_route`.
- Routes closer to the disaster front receive systematically lower utility as fire spreads
  (sanity check against Stage 6 hazard state).

### Tests
- Unit: `route_survival`/`route_utility` numeric correctness on hand-computed small examples.
- Unit: mixture sums to 1 — mapped to KPI 1.
- Integration: entropy-vs-panic-rate correlation — mapped to KPI 2.
- Integration: utility vs. distance-to-fire-front sanity check — mapped to KPI 3.

### Validation Gate
- [ ] Hazard-exposure sign convention confirmed by research lead.
- [ ] Mixture and survival computations pass unit tests.
- [ ] Entropy/panic correlation validated on a test run.

### Disciplines / Expertise Required
Transportation route-choice modeling specialist.

### Risks Specific to This Stage
Dependency on Stage 6 hazard state being available and correctly aligned in time and space —
see `05_risk_register.md` R-03.

### Dependencies
Stage 4 complete and approved; Stage 6 fire-front time series available (at least in
preliminary form) for hazard-exposure computation.

---

## Stage 6: Fire Simulation & SUMO Hazard Integration

### Problem Description
Simulates fire spread independently using `simfire`, seeded from the top and bottom edges of
the red/disaster zone, then converts the resulting fire-front coordinate time series into a
SUMO `.add.xml` hazard file, aligned to SUMO's coordinate system.

### Scope & Assumptions
- In scope: simfire configuration and run, coordinate transform to SUMO CRS, `.add.xml`
  generation with time-stamped POIs.
- Out of scope: any live feedback loop between SUMO state and fire spread (explicitly a
  one-way, offline hand-off per the spec and Step 0 discussion).
- Assumes Stage 1's red-zone polygon and study-area CRS are final.

### Tasks
1. Configure `simfire` ignition at the top and bottom edges of the red zone polygon.
2. Run `simfire` independently to produce a discretized fire-front coordinate time series.
3. Transform simfire's native output coordinates into SUMO's default network CRS (Step 0
   answer 8); validate alignment by overlaying fire-front points on the SUMO network early
   (per engineering recommendation — do not defer this check).
4. Generate `.add.xml` with time-stamped `<poi>` elements per the example in Section 1.4.
5. Load `.add.xml` into a test SUMO run and visually confirm the fire front tracks
   sensibly across the red zone over time.
6. Expose fire-front edge-distance data for Stage 5's hazard-exposure computation.

### Proposed Solution / Approach
The CRS alignment check is elevated to an explicit, early validation task (a small
proof-of-concept overlay of a handful of fire-front points on the SUMO network) rather than
being deferred to full integration, per the engineering risk flagged in Step 0.

### Tools / Frameworks / Algorithms
- `simfire` (mitrefireline/simfire) — per spec, open-source wildfire simulator.
- `pyproj` — coordinate transformation between simfire's native CRS and SUMO's network CRS.
- `sumolib` — network bounding-box queries for alignment validation.

### Example Code
```python
def transform_to_sumo_crs(fire_points, src_crs, sumo_crs):
    transformer = pyproj.Transformer.from_crs(src_crs, sumo_crs, always_xy=True)
    return [transformer.transform(x, y) for x, y in fire_points]

def write_fire_poi_xml(fire_time_series, path):
    root = ET.Element("additional")
    for i, (t, x, y) in enumerate(fire_time_series):
        ET.SubElement(root, "poi", id=f"poi_{i}", time=str(t), x=f"{x:.2f}", y=f"{y:.2f}",
                       color="255,0,0", type="large_geographic_marker", width="50.00", height="50.00")
    ET.ElementTree(root).write(path)
```

### Configuration Parameters
| Name | Meaning | Unit | Range/Type | Default | Source |
|---|---|---|---|---|---|
| `simfire_config` | Path to simfire scenario config | — | file path | — | simfire docs |
| `ignition_edges` | Top/bottom edges of red zone for ignition | — | GeoJSON | derived from Stage 1 | original spec |
| `src_crs` | simfire's native output CRS | — | EPSG code | simfire default | simfire docs — to confirm |
| `sumo_crs` | Target CRS (SUMO network default) | — | EPSG code | SUMO network default | Step 0 answer 8 |
| `poi_size` | Fire marker width/height in `.add.xml` | m | float > 0 | 50.0 | original spec example |

### KPIs / Success Metrics
- 100% of transformed fire-front coordinates fall within the SUMO network bounding box.
- Fire-front progression is visually and quantitatively consistent with ignition at the
  top/bottom red-zone edges (spreads inward/outward as expected).
- `.add.xml` loads into SUMO without errors.

### Tests
- Unit: coordinate transform round-trips correctly on known reference points.
- Integration: all transformed fire points within network bounding box — mapped to KPI 1.
- Validation: visual overlay of fire front on SUMO network reviewed by research lead — mapped
  to KPI 2.

### Validation Gate
- [ ] CRS transform validated against known reference points.
- [ ] Fire-front-on-network overlay reviewed and approved.
- [ ] `.add.xml` loads cleanly in a test SUMO run.

### Disciplines / Expertise Required
Fire/hazard simulation specialist; GIS/coordinate systems expertise.

### Risks Specific to This Stage
CRS/timing misalignment between simfire and SUMO — see `05_risk_register.md` R-03, R-04.

### Dependencies
Stage 1 complete and approved (red zone polygon, study area CRS).
