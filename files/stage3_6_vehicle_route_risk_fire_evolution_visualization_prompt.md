# Implement a Combined Vehicle Route-Risk and Fire-Evolution Figure

You are the senior engineer working in the current Stage 3–6 Manhattan fire-evacuation integration codebase.

## Goal

Add one deterministic scientific figure with two vertically aligned subplots:

1. **Top subplot:** Mean route-risk level of all vehicles currently running in the SUMO network.
2. **Bottom subplot:** Number of currently burning cells and number of burned cells.

Both subplots must share the same simulation-time x-axis. The figure should make it possible to compare how the risk faced by vehicles evolves as the fire spreads.

Before implementation, inspect:

```text
07_stage6_cell_to_edge_implementation_plan.md
```

Also inspect the existing Stage 3–6 integration code, runtime configuration, scientific tables, route-risk implementation, visualization pipeline, manifests, tests, and reports.

Reuse the accepted route-risk formula and existing validated fire-state data. Do not introduce a second risk formula or change the Stage 3–6 scientific behavior.

## 1. Top Subplot: Mean Vehicle Route Risk

At each recorded SUMO time step $t$, define the active-vehicle set as:

$$
V_{\mathrm{active}}(t)
=
\{\text{vehicles that have departed and are currently present in the SUMO network at time }t\}.
$$

Use the authoritative SUMO active-vehicle list. Exclude:

- vehicles that have not yet departed;
- vehicles that have already arrived;
- vehicles removed from the simulation according to the existing documented removal policy.

For each active vehicle $i$, obtain its currently committed route after any route reconsideration scheduled at time $t$.

Construct the remaining route from the vehicle's current route index to its assigned shelter. Do not include edges already traversed. Handle SUMO internal edges using the existing Stage 5 edge-handling rules.

Use the existing Stage 6 route-risk computation:

$$
S_k(t)
=
\exp\left(
\sum_{e\in k_{\mathrm{remaining}}(t)}
\log\bigl(\operatorname{clip}(S_e(t),\varepsilon,1)\bigr)
\right),
$$

$$
R_k(t)=1-S_k(t).
$$

Therefore:

$$
0\le R_k(t)\le 1.
$$

Do not replace this calculation with:

- the arithmetic mean of edge risks;
- the maximum edge risk;
- the sum of edge risks;
- the risk of the complete original route, including traversed edges;
- a newly invented or duplicated route-risk implementation.

For every time step containing active vehicles, calculate:

$$
\operatorname{mean\_active\_route\_risk}(t)
=
\frac{1}{N_{\mathrm{active}}(t)}
\sum_{i\in V_{\mathrm{active}}(t)}R_i(t).
$$

Also record:

- `active_vehicle_count`;
- `valid_route_risk_vehicle_count`;
- `minimum_active_route_risk`;
- `maximum_active_route_risk`.

The main plotted series is `mean_active_route_risk`.

All vehicle risks entering the mean must be finite and lie in $[0,1]$ within the configured numerical tolerance.

If an active vehicle's route risk cannot be calculated because of missing hazard data, an invalid route, an unresolved edge, or another scientific-data problem, fail with a clear diagnostic. Do not silently exclude the vehicle or substitute zero risk.

If a valid active vehicle has no remaining normal edge, apply the existing documented route-risk convention. Do not invent a special convention only for this plot.

When no vehicles are active, record:

```text
active_vehicle_count = 0
valid_route_risk_vehicle_count = 0
mean_active_route_risk = null
minimum_active_route_risk = null
maximum_active_route_risk = null
```

Do not record zero for the risk statistics, because zero would incorrectly imply that active vehicles were present and faced no risk. The top plot should show a gap or use the configured missing-value representation.

## 2. Measurement Ordering

Use one consistent measurement point at every simulation time step.

At a fire-update boundary, the required order is:

1. Activate the scientific fire snapshot.
2. Activate the corresponding edge hazards.
3. Perform the Stage 5 route-reconsideration actions scheduled for that time.
4. Read every active vehicle's resulting committed route.
5. Calculate and record the route-risk metrics.
6. Advance vehicle movement to the next SUMO step.

This ensures that the upper subplot measures the risk of routes vehicles actually hold after responding to the newly activated fire state.

At times without a fire update, use the currently active scientific fire snapshot. Never interpolate between fire snapshots.

Apply the same ordering and definitions in headless and GUI execution. The derived scientific table and figure must be identical between modes, or covered by the existing parity policy.

If the current codebase already has a formally accepted measurement order that differs from the sequence above, do not silently change runtime semantics. Stop and report the difference, its effect on the metric, and the smallest scientifically consistent integration option for approval.

## 3. Bottom Subplot: Fire-State Counts

At every recorded time step, calculate or join:

```text
burning_cell_count(t)
    = number of cells currently in the canonical BURNING state

burned_cell_count(t)
    = number of cells currently in the canonical BURNED state
```

Use the validated `fire_cell_time_series` data and the existing canonical state mapping. Do not infer these counts from GUI polygons, edge-risk values, colors, or screenshots.

Between two scientific fire boundaries, retain the most recently activated snapshot counts. This is a left-closed, right-open step function. Do not linearly interpolate fire counts.

Plot the two series as step lines:

- currently burning cells;
- burned cells.

Do not confuse:

```text
burned cells
```

with:

```text
affected cells = burning cells + burned cells
```

Only burning and burned counts are required in the lower subplot.

## 4. Required Derived Table

Create a machine-readable table, with its exact filename configured, containing at least:

```text
time_seconds
time_step
active_fire_snapshot_time_seconds
active_vehicle_count
valid_route_risk_vehicle_count
mean_active_route_risk
minimum_active_route_risk
maximum_active_route_risk
burning_cell_count
burned_cell_count
```

If the existing architecture already has separate vehicle-risk and fire-evolution tables, it is acceptable to produce this result as a validated deterministic join. However, the final figure must be generated solely from persisted scientific tables, not from a separate hidden calculation.

Use stable sorting and unique time keys. Record the source-table hashes and resolved-configuration hash.

Define and document precisely how `time_step` maps to `time_seconds`, including the SUMO step length and whether the initial state is step 0. Do not assume that one time step equals one second.

## 5. Required Figure

Generate one figure with two vertically stacked subplots and a shared x-axis.

### Top subplot

- x-axis: simulation time steps, or the configured display-time unit;
- y-axis: mean remaining-route risk of active vehicles;
- fixed y-range: $[0,1]$;
- plotted series: `mean_active_route_risk`.

Include a clear legend or label. The active-vehicle count may appear as annotations only when this is enabled by configuration. Do not add an unlabeled secondary axis.

### Bottom subplot

- x-axis: the same simulation-time coordinates as the upper subplot;
- y-axis: number of cells;
- first step line: `burning_cell_count`;
- second step line: `burned_cell_count`.

Use configuration-defined colors, line styles, widths, labels, figure size, DPI, fonts, titles, and output filename.

Suggested title:

```text
Vehicle Route Risk and Fire Evolution Over Time
```

Suggested subplot labels:

```text
Top: Mean remaining-route risk of active vehicles
Bottom: Number of fire cells
```

The two subplots must have perfectly aligned time coordinates. Do not normalize the two time axes independently.

Render headlessly and deterministically. Do not mutate any source table while sorting, joining, or plotting.

## 6. Configuration Requirements

Add all new settings to the existing Stage 3–6 runtime YAML and its validated schema, following the current architecture.

Configure at least:

- whether this visualization is enabled;
- output table and figure filenames;
- output format;
- recorded time resolution;
- display-time unit;
- figure dimensions and DPI;
- subplot height ratio and spacing;
- colors, line styles, widths, labels, and title;
- route-risk axis limits;
- fire-count axis limits or normalization policy;
- missing-value display policy;
- numerical tolerances;
- deterministic rendering metadata.

Do not place Manhattan-specific paths, timestamps, counts, IDs, colors, limits, or filenames in reusable Python modules.

Do not modify:

```text
configs/test/common_manhattan_test.yaml
```

unless I explicitly authorize that change later.

## 7. Validation

Before publishing, verify:

- every scientific route risk is finite and in $[0,1]$;
- the arithmetic mean is computed over the declared active-vehicle population;
- `valid_route_risk_vehicle_count == active_vehicle_count` whenever vehicles are active;
- the recorded denominator matches the number of vehicle-level risk records;
- active vehicles are counted at the documented measurement point;
- no pre-departure or arrived vehicle enters the mean;
- no traversed route edge is included;
- the current remaining route is the vehicle's post-reconsideration committed route;
- route risk uses the active fire snapshot and edge hazards for the same time;
- fire counts agree with an independent aggregation of the authoritative cell table;
- burning and burned counts are nonnegative integers;
- `burning_cell_count + burned_cell_count` does not exceed the total grid-cell count;
- burned-cell count is nondecreasing under the accepted fire-state model;
- all time keys are unique, sorted, and within the configured simulation horizon;
- `time_step` and `time_seconds` are mutually consistent with the configured SUMO step length;
- the two subplots use identical time coordinates;
- no configured axis clips a value;
- headless and GUI scientific outputs satisfy the existing parity rules;
- source tables and the sealed handoff remain unchanged.

Do not silently repair, interpolate, omit, or replace invalid scientific data.

## 8. Tests

Add tests covering:

- exact route-risk values for manually constructed routes;
- mean aggregation across multiple active vehicles;
- distinction between remaining and already traversed edges;
- post-reconsideration route measurement at a fire boundary;
- vehicles departing and arriving at different times;
- no-active-vehicle time steps producing null rather than zero;
- route risks equal to exactly 0 and 1;
- invalid or missing edge-hazard data;
- internal-edge handling;
- burning and burned counts at known snapshots;
- stepwise fire counts between snapshot boundaries;
- mapping between `time_step` and `time_seconds` for a non-unit SUMO step length;
- common time-axis alignment;
- axis-clipping detection;
- deterministic table and figure generation;
- headless/GUI parity;
- source-table immutability;
- behavior on the existing non-Manhattan fixture using configuration changes only.

Include at least one manual example in a test or report showing:

- individual vehicle remaining-route risks;
- the arithmetic mean of those vehicle risks;
- the active-vehicle denominator;
- the corresponding burning- and burned-cell counts;
- the active fire-snapshot time;
- all values at the same recorded simulation time step.

## 9. Integration with Manifests and Reports

Register the new table and figure in:

```text
visualization_manifest.json
```

Record:

- source hashes;
- resolved-configuration hash;
- simulation horizon and time resolution;
- mapping between simulation steps and seconds;
- vehicle-population definition;
- route segment included in risk;
- measurement ordering;
- risk formula;
- fire-count definitions;
- plotting encoding and axis ranges;
- output hashes.

Update the Stage 3–6 run report and reproduction guide to explain how to generate and interpret this figure.

The report must state that the figure is descriptive. A visual correlation between fire evolution and mean route risk does not, by itself, prove causal avoidance or the effectiveness of the routing policy.

## 10. Scope Restrictions

Do not:

- change Stage 3 allocation;
- change Stage 4 behavior;
- change Stage 5 routing decisions;
- change the fire simulation or snapshots;
- tune hazard coefficients to obtain a preferred curve;
- alter vehicle demand, departures, or shelter assignments;
- use GUI state as scientific input;
- recompute results differently only for plotting.

This task adds observation, persisted metrics, visualization, tests, and documentation only.

## 11. Completion Evidence

In your final response, provide:

1. Files added or modified.
2. The exact execution command.
3. Exact test pass/fail/skip totals and runtime.
4. The derived time-series table path.
5. The combined figure path.
6. The `visualization_manifest.json` entry.
7. Headless/GUI parity result.
8. Manual calculation evidence.
9. Before/after sealed-handoff verification.
10. Confirmation that Stage 3–5 scientific behavior was unchanged.
11. Confirmation that no scenario-specific reusable-code hard-coding was added.
12. Confirmation that the configured axes passed no-clipping validation.

Do not stop merely because the observed mean route risk is flat, zero, increasing, or otherwise different from expectations. Report the scientifically produced result without tuning it.

## Authorization Boundary

This prompt authorizes implementation of the visualization, its persisted metrics, tests, validation, manifest registration, and documentation within the restrictions above. It is not final acceptance of the completed work. Final acceptance will follow inspection of the code, scientific outputs, exact test evidence, parity evidence, reports, and checksum results.
