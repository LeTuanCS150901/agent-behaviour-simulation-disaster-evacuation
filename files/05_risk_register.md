# 05 — Risk Register

## Stage 4 revision (2026-07-20)

- R-02 is superseded for `softmax_c1_v1`: C1 conditions determine both quadratic pieces;
  `theta_2` and `beta_2` are not active parameters.
- Historical direct-W, clipped-Gaussian, q=0.30/q=0.70 reports and artifacts may be confused
  with new results. Mitigation: explicit model version, truncated-normal mode, config hash,
  and segregated output paths.
- Passing Manhattan/toy software tests does not validate q, panic/attractiveness parameters,
  Toulouse realism, or scientific conclusions. Research-lead review remains mandatory.

| ID | Risk | Likelihood | Impact | Mitigation | Owner/Discipline |
|---|---|---|---|---|---|
| R-01 | Population synthesized directly from census marginals (no `eqasim-france`) may not capture within-zone demographic heterogeneity that a full synthetic-population pipeline would. | Medium | Medium | Document as an explicit simplification; validate zone-level counts against census within tolerance (Stage 1 KPI); revisit if downstream results are sensitive to within-zone heterogeneity. | GIS/geospatial |
| R-02 | Historical only: the former direct-W model left `W_s(x)` curvature under-determined. `softmax_c1_v1` supersedes it with analytically fixed C1 coefficients, but legacy configurations or outputs could still be misidentified as active. | Medium | High | Require `model_version`, reject free `theta_2`/`beta_2` fields in the active schema, and segregate outputs by version/config hash. | Behavioral modeling/statistics |
| R-03 | SUMO and `simfire` are two independent simulators with no shared clock and a one-way, offline data hand-off; CRS or timing misalignment would silently invalidate results without crashing. | Medium | High | Early CRS-alignment proof-of-concept (Stage 6, before full integration); explicit test that all fire-front points fall within the SUMO network bounding box; visual review gate. | Fire/hazard simulation, GIS |
| R-04 | `simfire`'s native output CRS may not be documented precisely or may vary by simfire version. | Medium | Medium | Verify empirically against known reference points before trusting the transform; pin `simfire` version in dependency lock file. | Fire/hazard simulation |
| R-05 | Origin-to-SUMO-edge snapping may fail or produce implausible results near zone boundaries adjacent to network gaps (e.g., pedestrian-only areas). | Medium | Low | `max_snap_distance` sanity check with explicit failure rather than silent snap to a distant edge; log all snap distances for review. | Transportation/GIS |
| R-06 | p-median solve time may be prohibitive at full city scale with an exact MIP solver. | Medium | Medium | Start at a small validation-scale population percentage (configurable); benchmark solve-time-vs-scale curve before committing to exact solving at full scale; heuristic fallback plan documented. | Operations research |
| R-07 | Shelter capacity constraints may be infeasible if configured capacity slack is too small relative to actual demand distribution. | Low | High | `shelter_capacity_slack` configurable and validated against total demand before solve; explicit infeasibility check with actionable error message. | Operations research |
| R-08 | Global (not per-agent) `ω`, `β_t`, `β_a` may be an oversimplification if evacuee heterogeneity in these preferences matters to the research question. | Low | Medium | Documented as an explicit, confirmed simplification (Step 0 answer 4); revisit only if requested by research lead. | Behavioral modeling |
| R-09 | Hazard-exposure sign convention in the Stage 5 utility function (`U_k`) required an interpretive choice (`hazard_exposure_k = 1 - S(k)`) not fully specified in the original formula. | Low | Medium | Explicitly flagged in Stage 5 documentation; requires research-lead confirmation before Stage 5 sign-off. | Transportation route-choice modeling |
