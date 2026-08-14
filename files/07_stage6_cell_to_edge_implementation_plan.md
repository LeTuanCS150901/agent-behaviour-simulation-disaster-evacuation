# Stage 6 Cell-to-Edge Fire Hazard Integration Plan

## 1. Objective

Redesign the Stage 6 fire-to-SUMO integration using a **cell-to-edge hazard approach**.

The new pipeline is:

```text
simfire cell states over time
→ fire-cell coordinates in the SUMO coordinate system
→ static cell-to-edge intersection mapping
→ time-dependent edge hazard
→ edge survival
→ route survival and route risk
→ Stage 5 route-choice utility
```

Keep the fire-front coordinate outputs for visualization and validation, but use the edge-hazard time series as the main Stage 6-to-Stage 5 interface.

Do not include travel time, traversal time, or edge length as an exposure multiplier in the edge survival formula.

---

## 2. Final Mathematical Model

### 2.1 Cell hazard

For each fire-grid cell $c$ at time $t$, define:

$$
h_c(t)\in[0,1].
$$

For the first implementation, use a configurable state-to-hazard mapping.

Recommended default for the Manhattan software test:

```text
unburned → 0
burning  → 1
burned   → 0
```

Do not hardcode this mapping inside the calculation. Expose it in the Stage 6 config.

The implementation must support any normalized cell hazard value in $[0,1]$.

### 2.2 Static edge-cell overlap

For each SUMO edge $e$ and fire-grid cell $c$, define:

$$
\ell_{e,c}
=
\text{length of the edge geometry lying inside cell }c.
$$

Let:

$$
L_e
=
\text{total geometric length of edge }e.
$$

Let $C_e$ be the set of fire-grid cells intersecting edge $e$.

### 2.3 Edge hazard

At time $t$, compute the edge hazard as the length-weighted mean of the cell hazards:

$$
H_e(t)
=
\frac{
\sum_{c\in C_e} h_c(t)\ell_{e,c}
}{
L_e
}.
$$

Important rules:

- Do not compute an ordinary mean of $h_c(t)\ell_{e,c}$.
- Do not divide by the number of intersected cells.
- Do not multiply the final hazard by travel time.
- Do not multiply it by traversal time.
- Do not silently ignore uncovered edge portions.

Define the edge-cell coverage ratio:

$$
\rho_e
=
\frac{
\sum_{c\in C_e}\ell_{e,c}
}{
L_e
}.
$$

The fire grid should cover the complete network, so $\rho_e$ should be approximately $1$.

If $\rho_e$ is below a configurable tolerance, report the edge and fail by default.

### 2.4 Edge survival

Define edge survival as:

$$
S_e(t)
=
\exp\left(-\lambda H_e(t)\right),
$$

where:

$$
\lambda>0.
$$

Interpretation:

- $H_e(t)=0$ gives $S_e(t)=1$;
- increasing hazard decreases edge survival;
- larger $\lambda$ makes survival decrease more strongly.

Recommended config key:

```yaml
edge_survival:
  lambda: 1.0
```

The default value is acceptable for software testing but must be reported as uncalibrated.

Define edge risk, when needed, as:

$$
R_e(t)
=
1-S_e(t).
$$

### 2.5 Route survival

For route $k$ containing edges $e_1,\ldots,e_m$, compute:

$$
S_k(t)
=
\prod_{r=1}^{m}S_{e_r}(t).
$$

For numerical stability, implement:

$$
\log S_k(t)
=
\sum_{r=1}^{m}\log S_{e_r}(t),
$$

then:

$$
S_k(t)
=
\exp\left(\log S_k(t)\right).
$$

Equivalently:

$$
S_k(t)
=
\exp\left(
\sum_{e\in k}\log S_e(t)
\right).
$$

If an edge appears multiple times in a route, include it once per traversal.

Before taking logarithms, clip the edge survival:

```text
S_e_clipped = clip(S_e, numerical_epsilon, 1)
```

### 2.6 Route risk

Define:

$$
R_k(t)
=
1-S_k(t).
$$

Pass this route risk to Stage 5 as the hazard-exposure term:

$$
U_k
=
-\alpha_t\,\text{normalized travel time}_k
-\alpha_h R_k(t).
$$

Do not change the Stage 5 utility formula in this task.

### 2.7 Time lookup

For the first implementation, evaluate every edge in a route using the fire state at the current route-decision time.

When Stage 5 asks for hazard at time $t$:

- use the snapshot at exactly $t$ if available;
- otherwise use the most recent snapshot at or before $t$.

Recommended config:

```yaml
hazard_time_lookup: "previous_snapshot"
```

Do not implement predicted edge-arrival-time exposure in this task.

---

## 3. Isolated Simfire Environment

Create a separate reproducible environment for simfire to avoid dependency conflicts with the main SUMO environment.

Recommended environment name:

```text
evac-simfire
```

Create one of:

```text
environments/environment-simfire.yml
```

or:

```text
environments/requirements-simfire.txt
```

Record or pin:

- Python version;
- simfire version;
- NumPy version;
- raster/geospatial dependencies;
- visualization dependencies required by simfire.

Do not import simfire directly from the main `evac-sumo` environment.

The two environments must communicate through files.

Recommended workflow:

```text
main/SUMO environment:
    create grid metadata and simfire input manifest

simfire environment:
    run fire simulation
    write cell-state time series and run metadata

main/SUMO environment:
    read simfire outputs
    compute edge hazard and route risk
```

Add a script such as:

```text
scripts/run_simfire_cells.py
```

to execute inside the simfire environment.

Add a main orchestration script such as:

```text
scripts/run_stage6_cell_to_edge.py
```

Record exact commands, exit codes, standard output, and standard error.

---

## 4. Fire-Grid Construction

Construct a rectangular grid that covers the SUMO network.

Use:

- the SUMO network bounding box;
- configurable padding;
- configurable cell width and height.

Recommended config:

```yaml
fire_grid:
  cell_width: 20.0
  cell_height: 20.0
  padding: 50.0
  row_origin: "top"
```

Do not hardcode these values.

Use deterministic cell IDs, for example:

```text
r000_c000
r000_c001
r001_c000
```

For each cell, store:

```text
cell_id
row
column
xmin
ymin
xmax
ymax
center_x
center_y
geometry
```

Required outputs:

```text
outputs/test/manhattan/stage6/grid/fire_grid.geojson
outputs/test/manhattan/stage6/grid/fire_grid_metadata.json
```

---

## 5. Coordinate Mapping

Define an explicit affine mapping between simfire row/column coordinates and SUMO coordinates.

For a grid with top-left row origin, a typical mapping is:

$$
x_c
=
x_{\min}
+
\left(\text{column}+\frac12\right)\Delta x,
$$

$$
y_c
=
y_{\max}
-
\left(\text{row}+\frac12\right)\Delta y.
$$

Do not assume the $y$ direction without verifying simfire’s indexing convention.

The grid metadata must include:

```text
network_bbox
grid_bbox
cell_width
cell_height
number_of_rows
number_of_columns
row_origin
x_direction
y_direction
affine_transform
SUMO projection metadata
```

Produce:

```text
outputs/test/manhattan/stage6/grid/fire_grid_network_overlay.png
```

The overlay must show:

- the SUMO road network;
- the fire grid;
- ignition cells;
- the network bounding box.

Do not continue if the grid is shifted, flipped, or scaled incorrectly.

---

## 6. Static Edge-to-Cell Mapping

Create a preprocessing module such as:

```text
src/evacuation_sim/fire_sim/edge_cell_mapping.py
```

For each routeable passenger edge:

1. load the SUMO edge geometry;
2. convert it to a line geometry;
3. identify intersecting fire-grid cells;
4. compute the exact overlap length $\ell_{e,c}$;
5. compute the edge length $L_e$;
6. compute the coverage ratio $\rho_e$.

Required output:

```text
outputs/test/manhattan/stage6/edge_cell_intersections.parquet
```

Required columns:

```text
edge_id
cell_id
overlap_length
edge_length
overlap_fraction
edge_coverage_ratio
```

Also produce:

```text
outputs/test/manhattan/stage6/edge_cell_coverage_summary.csv
outputs/test/manhattan/stage6/edge_cell_mapping_summary.json
```

The summary must report:

```text
total edges processed
edges with zero intersecting cells
edges below coverage tolerance
minimum coverage ratio
mean coverage ratio
maximum coverage ratio
total intersection records
```

Recommended config:

```yaml
edge_cell_mapping:
  minimum_coverage_ratio: 0.999
  incomplete_coverage_policy: "error"
```

Do not recompute this mapping at every fire time step.

---

## 7. Simfire Cell-State Time Series

Run simfire in the isolated environment and write:

```text
outputs/test/manhattan/stage6/simfire/fire_cell_time_series.parquet
```

Required columns:

```text
time
cell_id
row
column
fire_state
hazard_value
```

Validate:

$$
0\leq h_c(t)\leq1.
$$

Also produce:

```text
outputs/test/manhattan/stage6/simfire/simfire_run_metadata.json
outputs/test/manhattan/stage6/simfire/simfire_stdout.log
outputs/test/manhattan/stage6/simfire/simfire_stderr.log
```

The metadata must include:

```text
simfire version
Python version
random seed
grid dimensions
simulation start/end time
fire time step
ignition cells
state-to-hazard mapping
number of output snapshots
real simfire or mock fixture
```

If simfire cannot be installed or run, a deterministic mock cell-fire fixture may be used for software tests.

The report must then clearly state:

```text
mock cell-fire pipeline passed
real simfire run not performed
```

Do not claim real simfire validation if a mock was used.

---

## 8. Fire-Front Visualization Output

Keep the fire-front coordinate output for compatibility and visualization.

Derive it from the boundaries or centers of currently burning cells.

Produce:

```text
outputs/test/manhattan/stage6/fire_front_time_series.geojson
```

or a partitioned equivalent if the output is too large.

This is not the primary input for edge-risk calculations.

---

## 9. Dynamic Edge Hazard

Create a module such as:

```text
src/evacuation_sim/fire_sim/edge_hazard.py
```

At each time step:

1. read the cell hazards $h_c(t)$;
2. join them with the static edge-cell mapping;
3. compute:

$$
H_e(t)
=
\frac{
\sum_c h_c(t)\ell_{e,c}
}{
L_e
};
$$

4. validate:

$$
0\leq H_e(t)\leq1;
$$

5. compute:

$$
S_e(t)
=
\exp(-\lambda H_e(t));
$$

6. compute:

$$
R_e(t)
=
1-S_e(t).
$$

Required output:

```text
outputs/test/manhattan/stage6/edge_hazard_time_series.parquet
```

Required columns:

```text
time
edge_id
edge_hazard
edge_survival
edge_risk
edge_length
coverage_ratio
lambda
```

Also produce:

```text
outputs/test/manhattan/stage6/edge_hazard_summary.json
outputs/test/manhattan/stage6/edge_hazard_diagnostic.png
```

The diagnostic should show representative edge hazard and survival values over time.

---

## 10. Missing-Data Policies

Do not silently treat missing data as safe.

Recommended config:

```yaml
hazard_missing_data:
  missing_cell_policy: "error"
  missing_edge_policy: "error"
```

If simfire produces sparse output containing only nonzero-hazard cells, declare that explicitly:

```yaml
simfire_output:
  storage_mode: "sparse_nonzero"
  absent_cell_hazard: 0.0
```

The default full-output mode should contain every grid cell at every recorded snapshot.

---

## 11. Stage 5 Hazard Provider

Create a reusable Stage 5-facing module such as:

```text
src/evacuation_sim/route_choice/hazard_provider.py
```

Expose:

```python
get_edge_hazard(edge_id, time)
get_edge_survival(edge_id, time)
compute_route_survival(route_edge_ids, time)
compute_route_risk(route_edge_ids, time)
```

Use:

$$
S_k(t)
=
\exp\left(
\sum_{e\in k}
\log\left(
\operatorname{clip}(S_e(t),\varepsilon_{\text{num}},1)
\right)
\right),
$$

and:

$$
R_k(t)
=
1-S_k(t).
$$

Required diagnostic:

```text
outputs/test/manhattan/stage6/route_hazard_samples.parquet
```

Required columns:

```text
time
route_id
edge_count
route_survival
route_risk
sum_log_edge_survival
```

Do not include travel time inside the survival formula.

---

## 12. Edge-Segmentation Limitation

The approved route-survival formula multiplies one survival value per SUMO edge.

Therefore, route survival may depend on how the road network is segmented into edges.

Do not change the approved formula in this task.

Instead:

1. document this limitation;
2. report route edge counts;
3. add a controlled diagnostic comparing:
   - one edge with hazard $H$;
   - two consecutive edges with the same hazard $H$;
4. explain the difference in the report.

Do not introduce length weighting or travel-time weighting without approval.

---

## 13. SUMO Visualization Artifact

Generate:

```text
outputs/test/manhattan/stage6/fire_hazard.add.xml
```

Use cell centers, cell polygons, or edge annotations depending on the existing integration.

The file must be valid XML and load in SUMO.

Also produce:

```text
outputs/test/manhattan/stage6/fire_network_overlay.png
outputs/test/manhattan/stage6/sumo_additional_file_check.json
```

The `.add.xml` file is only a visualization artifact.

Stage 5 must read:

```text
edge_hazard_time_series.parquet
```

instead of parsing POIs or visual annotations.

---

## 14. Config and Schema Updates

Update `stage6.yaml` and `stage6.schema.json` with sections such as:

```yaml
hazard_representation: "cell_to_edge"

fire_grid:
  cell_width: 20.0
  cell_height: 20.0
  padding: 50.0
  row_origin: "top"

cell_hazard_mapping:
  unburned: 0.0
  burning: 1.0
  burned: 0.0

edge_cell_mapping:
  minimum_coverage_ratio: 0.999
  incomplete_coverage_policy: "error"

edge_survival:
  lambda: 1.0
  numerical_epsilon: 1.0e-12

hazard_time_lookup: "previous_snapshot"

hazard_missing_data:
  missing_cell_policy: "error"
  missing_edge_policy: "error"
```

Validate:

```text
cell_width > 0
cell_height > 0
padding >= 0
all cell hazard values in [0,1]
lambda > 0
minimum coverage ratio in (0,1]
numerical epsilon in (0,1)
```

The defaults are acceptable for Manhattan software testing but must be marked as uncalibrated.

---

## 15. Required Tests

Add:

```text
tests/unit/test_fire_grid_transform.py
tests/unit/test_edge_cell_mapping.py
tests/unit/test_edge_hazard.py
tests/unit/test_edge_survival.py
tests/unit/test_route_survival.py
tests/integration/test_simfire_cell_output_contract.py
tests/integration/test_stage6_cell_to_edge_pipeline.py
tests/integration/test_stage5_hazard_provider.py
```

Required cases:

### Test 1 — weighted edge hazard

An edge of length $100$ crosses:

```text
cell A: overlap 20, hazard 0
cell B: overlap 30, hazard 1
cell C: overlap 50, hazard 1
```

Verify:

$$
H_e
=
\frac{0(20)+1(30)+1(50)}{100}
=
0.8.
$$

### Test 2 — no ordinary mean

Verify that the code does not divide by the number of intersected cells.

### Test 3 — edge survival

For $H_e=0.8$ and $\lambda=2$:

$$
S_e
=
\exp(-1.6).
$$

### Test 4 — safe edge

For $H_e=0$:

$$
S_e=1,
\qquad
R_e=0.
$$

### Test 5 — route survival

For known $S_1,S_2,S_3$:

$$
S_k
=
\exp(\log S_1+\log S_2+\log S_3).
$$

Verify:

$$
R_k=1-S_k.
$$

### Test 6 — coverage failure

Verify that incomplete edge coverage fails clearly under the default policy.

### Test 7 — coordinate transform

Verify known row/column values against expected SUMO coordinates.

### Test 8 — time lookup

Verify that `previous_snapshot` returns the most recent snapshot at or before the query time.

### Test 9 — missing edge

Verify that an edge absent from the hazard table fails clearly.

### Test 10 — real versus mock metadata

Verify that the metadata cannot mark real simfire validation as true when a mock was used.

### Test 11 — route segmentation diagnostic

Demonstrate the difference between one edge and two consecutive edges with the same hazard.

---

## 16. Required Manhattan Outputs

Produce:

```text
outputs/test/manhattan/stage6/grid/fire_grid.geojson
outputs/test/manhattan/stage6/grid/fire_grid_metadata.json
outputs/test/manhattan/stage6/grid/fire_grid_network_overlay.png

outputs/test/manhattan/stage6/edge_cell_intersections.parquet
outputs/test/manhattan/stage6/edge_cell_coverage_summary.csv
outputs/test/manhattan/stage6/edge_cell_mapping_summary.json

outputs/test/manhattan/stage6/simfire/fire_cell_time_series.parquet
outputs/test/manhattan/stage6/simfire/simfire_run_metadata.json
outputs/test/manhattan/stage6/simfire/simfire_stdout.log
outputs/test/manhattan/stage6/simfire/simfire_stderr.log

outputs/test/manhattan/stage6/fire_front_time_series.geojson
outputs/test/manhattan/stage6/edge_hazard_time_series.parquet
outputs/test/manhattan/stage6/edge_hazard_summary.json
outputs/test/manhattan/stage6/edge_hazard_diagnostic.png

outputs/test/manhattan/stage6/route_hazard_samples.parquet
outputs/test/manhattan/stage6/fire_hazard.add.xml
outputs/test/manhattan/stage6/fire_network_overlay.png
outputs/test/manhattan/stage6/sumo_additional_file_check.json
```

Keep all Manhattan outputs separate from Toulouse outputs.

---

## 17. Required Commands

Create the simfire environment:

```bash
conda env create -f environments/environment-simfire.yml
```

Run simfire:

```bash
conda run -n evac-simfire python scripts/run_simfire_cells.py \
  --config configs/test/manhattan_test.yaml \
  --stage6-config configs/stage6.yaml
```

Run preprocessing and integration:

```bash
conda run -n evac-sumo python scripts/create_fire_grid.py \
  --config configs/test/manhattan_test.yaml \
  --stage6-config configs/stage6.yaml

conda run -n evac-sumo python scripts/build_edge_cell_mapping.py \
  --config configs/test/manhattan_test.yaml \
  --stage6-config configs/stage6.yaml

conda run -n evac-sumo python scripts/compute_edge_hazard.py \
  --config configs/test/manhattan_test.yaml \
  --stage6-config configs/stage6.yaml

conda run -n evac-sumo python scripts/run_stage6_cell_to_edge.py \
  --config configs/test/manhattan_test.yaml \
  --stage6-config configs/stage6.yaml
```

Run tests:

```bash
conda run -n evac-sumo python -m pytest \
  tests/unit/test_fire_grid_transform.py \
  tests/unit/test_edge_cell_mapping.py \
  tests/unit/test_edge_hazard.py \
  tests/unit/test_edge_survival.py \
  tests/unit/test_route_survival.py \
  tests/integration/test_simfire_cell_output_contract.py \
  tests/integration/test_stage6_cell_to_edge_pipeline.py \
  tests/integration/test_stage5_hazard_provider.py
```

Record only commands that were actually executed.

---

## 18. Required Reports

Create:

```text
reports/stage6_cell_to_edge_implementation.md
```

Update:

```text
reports/manhattan_test_report.md
```

Add:

```markdown
## Stage 6 Cell-to-Edge Fire Hazard Integration
```

The reports must include:

1. limitation of the previous fire-front-only approach;
2. new grid and cell-state architecture;
3. cell-hazard definition;
4. edge-hazard formula;
5. confirmation that travel time is excluded from edge survival;
6. edge-survival formula;
7. route-survival and route-risk formulas;
8. grid-to-SUMO coordinate mapping;
9. simfire environment details;
10. real simfire versus mock status;
11. files created or modified;
12. config and schema changes;
13. commands run;
14. test results;
15. edge-cell coverage statistics;
16. hazard and survival ranges;
17. output artifact paths;
18. SUMO `.add.xml` load result;
19. edge-segmentation limitation;
20. problems encountered;
21. fixes completed;
22. failures;
23. remaining unvalidated items;
24. recommendation before applying the method to Toulouse.

Do not claim real simfire validation if only a mock fixture was used.

---

## 19. Success Criteria

The Stage 6 redesign is complete only if:

1. the fire grid covers the SUMO network;
2. coordinate mapping is verified;
3. static edge-cell intersections are generated;
4. coverage ratios satisfy the configured tolerance;
5. real simfire or an explicitly identified mock produces valid cell-state time series;
6. all cell hazards are in $[0,1]$;
7. edge hazards are length weighted correctly;
8. edge survival follows $S_e=\exp(-\lambda H_e)$;
9. route survival is computed in log space;
10. route risk is $1-S_k$;
11. no travel-time term is included in edge survival;
12. Stage 5 can query route risk through the hazard provider;
13. all required files and plots are produced;
14. all relevant tests pass;
15. reports clearly separate real and mock validation.

After completing the work, stop and ask for review of the Stage 6 report and visual overlays before changing the Toulouse workflow or claiming final Manhattan validation.

---

## 20. Current Assumptions

The first implementation assumes:

- binary cell hazard mapping;
- $\lambda=1$ as an uncalibrated default;
- current-time hazard with previous-snapshot lookup;
- centerline edge-to-cell intersection without road-width buffering;
- no additional proximity hazard outside intersecting cells;
- no travel-time or traversal-time factor inside edge survival.
