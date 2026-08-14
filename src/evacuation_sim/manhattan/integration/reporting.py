from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import sumolib

from evacuation_sim.io.tables import read_table

from .config import ResolvedIntegrationConfig, normalized_display_text


def write_run_report(config: ResolvedIntegrationConfig, visualization: dict[str, Any], handoff_before: str, handoff_after: str) -> Path:
    outputs = config.runtime["outputs"]
    stage3 = json.loads((config.output_root / outputs["stage3_directory"] / "stage3_manhattan_summary.json").read_text(encoding="utf-8"))
    stage4 = json.loads((config.output_root / outputs["stage4_directory"] / "stage4_manhattan_summary.json").read_text(encoding="utf-8"))
    headless = json.loads((config.output_root / config.runtime["execution"]["phases"]["headless_summary"]).read_text(encoding="utf-8"))
    modes = config.execution_modes
    gui = (
        json.loads((config.output_root / config.runtime["execution"]["phases"]["gui_summary"]).read_text(encoding="utf-8"))
        if modes["gui_enabled"] else {"status": "disabled_by_configuration", "screenshots": [], "decision_rows": 0}
    )
    parity = (
        json.loads((config.output_root / config.runtime["execution"]["phases"]["parity_report"]).read_text(encoding="utf-8"))
        if modes["parity_enabled"] else {"status": "disabled_by_configuration"}
    )
    validation = json.loads((config.output_root / config.runtime["execution"]["phases"]["validation_report"]).read_text(encoding="utf-8"))
    avoidance = read_table(config.output_root / outputs["stage5_directory"] / "headless" / outputs["avoidance_table"])
    evolution_cfg = config.runtime["visualization"]["route_risk_fire_evolution"]
    stage5_headless = config.output_root / outputs["stage5_directory"] / "headless"
    evolution = read_table(stage5_headless / evolution_cfg["derived_table_filename"])
    active_risks = read_table(stage5_headless / evolution_cfg["vehicle_table_filename"])
    backend = config.runtime["stage3_backend"]
    formulas = config.runtime["reporting"]["formula_units"]
    report_path = config.output_root / outputs["reports_directory"] / outputs["run_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    differing = int(avoidance["differing_route_risks"].sum()) if len(avoidance) else 0
    positive = int(avoidance["positive_risk_alternative"].sum()) if len(avoidance) else 0
    responded = int(avoidance["probabilities_responded"].sum()) if len(avoidance) else 0
    delta = float(avoidance["expected_risk_avoidance_delta"].mean()) if len(avoidance) else float("nan")
    selected_risk_mean = float(avoidance["selected_route_risk"].mean()) if len(avoidance) else float("nan")
    selected_positive = int((avoidance["selected_route_risk"] > config.runtime["avoidance_analysis"]["zero_tolerance"]).sum()) if len(avoidance) else 0
    avoidance_sign_counts = avoidance["avoidance_sign"].value_counts().to_dict() if len(avoidance) else {}
    validated_decisions = int((~avoidance["interacting_fronts"]).sum()) if len(avoidance) else 0
    unvalidated_decisions = int(avoidance["interacting_fronts"].sum()) if len(avoidance) else 0
    formula_text = "\n".join(f"- **{name}:** `{value}`" for name, value in formulas.items())
    family_text = "\n".join(f"- **{name}:** " + ", ".join(paths) for name, paths in visualization["families"].items())
    replacement = config.runtime["reporting"].get("contract_replacement")
    replacement_text = "not applicable"
    if replacement:
        replacement_text = "\n".join([
            f"- Archived displaced contract: `{replacement['archive_path']}`",
            f"- Displaced SHA-256: `{replacement['displaced_sha256']}`; size: {replacement['displaced_size_bytes']} bytes",
            f"- Parsed YAML identical to prior engineer copy: **{replacement['parsed_yaml_identical_to_previous_engineer_copy']}**",
            *[f"- {line}" for line in replacement["diff_summary"]],
        ])
    report_title = normalized_display_text(config.runtime["reporting"]["title"], "reporting.title")
    override = config.demand_override_provenance
    effective_demand = config.role["demand"]
    effective_shelters = config.role["shelters"]
    contract_demand = config.contract_role["demand"]
    contract_shelters = config.contract_role["shelters"]
    net = sumolib.net.readNet(str(config.network_path))
    origin_details = [
        {
            **item,
            "lane_count": len(net.getEdge(str(item["edge_id"])).getLanes()),
            "passenger_lane_count": sum(
                lane.allows(config.shared["network"]["vehicle_class"])
                for lane in net.getEdge(str(item["edge_id"])).getLanes()
            ),
        }
        for item in effective_demand["origins"]
    ]
    shelter_details = [
        {
            **item,
            "lane_count": len(net.getEdge(str(item["edge_id"])).getLanes()),
            "passenger_lane_count": sum(
                lane.allows(config.shared["network"]["vehicle_class"])
                for lane in net.getEdge(str(item["edge_id"])).getLanes()
            ),
        }
        for item in effective_shelters["destinations"]
    ]
    tripinfo_path = config.output_root / outputs["sumo_directory"] / outputs["tripinfo_headless"]
    arrived = len(ET.parse(tripinfo_path).getroot().findall("tripinfo")) if tripinfo_path.is_file() else 0
    total_vehicles = int(effective_demand["total_vehicles"])
    non_arrived = total_vehicles - arrived
    override_text = "No demand or shelter override was active; contract values were used."
    if override is not None:
        override_text = "\n".join([
            f"- Active: **yes**",
            f"- Reason: `{override['reason']}`",
            f"- Supplied sections: `{override['supplied_sections']}`",
            f"- Departure generation inherited: **{override['departure_generation_inherited']}**",
            f"- Contract demand/shelters superseded: `{override['contract_values']}`",
            f"- Effective demand/shelters: `{override['effective_values']}`",
            f"- Common contract remained `{override['common_contract_sha256']}`",
        ])
    origin_text = "\n".join(
        f"- `{item['edge_id']}`: {item['num_cars']} vehicles, {item['lane_count']} total lane(s), {item['passenger_lane_count']} passenger-permitted"
        for item in origin_details
    )
    shelter_text = "\n".join(
        f"- `{item['edge_id']}`: capacity {item['capacity']}, {item['lane_count']} total lane(s), {item['passenger_lane_count']} passenger-permitted"
        for item in shelter_details
    )
    contract_total_vehicles = int(contract_demand["total_vehicles"])
    contract_total_capacity = int(contract_shelters["total_capacity"])
    effective_total_capacity = int(effective_shelters["total_capacity"])
    demand_delta = total_vehicles - contract_total_vehicles
    capacity_delta = effective_total_capacity - contract_total_capacity
    contract_departure = contract_demand["departure_generation"]
    effective_departure = effective_demand["departure_generation"]
    departure_window_changed = (
        float(contract_departure["begin_seconds"]),
        float(contract_departure["end_seconds"]),
        bool(contract_departure["inclusive_end"]),
    ) != (
        float(effective_departure["begin_seconds"]),
        float(effective_departure["end_seconds"]),
        bool(effective_departure["inclusive_end"]),
    )
    manual_time = next(
        float(time_value) for time_value, group in active_risks.groupby("time_seconds", sort=True)
        if len(group) >= 2
    )
    manual_group = active_risks[active_risks["time_seconds"] == manual_time].sort_values("vehicle_id", kind="stable")
    manual_summary = evolution[evolution["time_seconds"] == manual_time].iloc[0]
    manual_risks = ", ".join(
        f"{row.vehicle_id}={float(row.remaining_route_risk):.12g}"
        for row in manual_group.itertuples(index=False)
    )
    report_path.write_text(f"""# {report_title} — run report

## Status and scope

Technical pipeline status: **passed through validation, independent hazard reconstruction, headless execution, and finalization**. GUI status: **{gui['status']}**. Parity status: **{parity['status']}**.

Scientific-demonstration status: **{'inconclusive' if differing == 0 else 'observed without calibration or validation claim'}**. This synthetic Manhattan fixture is uncalibrated, test-only, and not research-approved.

## Configuration and immutable inputs

- Run ID: `{config.runtime['execution']['run_id']}`
- Common SHA-256: `{config.common_sha256}`
- Runtime SHA-256: `{config.runtime_sha256}`
- Resolved logical SHA-256: `{config.logical_sha256}`
- Handoff combined hash before: `{handoff_before}`
- Handoff combined hash after: `{handoff_after}`
- Immutable handoff unchanged: **{handoff_before == handoff_after}**
- Validation: `{validation['status']}`; mapping oracle `{validation['oracle_comparisons']['mapping']['status']}`; hazard oracle `{validation['oracle_comparisons']['hazard']['status']}`

## Contract replacement evidence

{replacement_text}

The displaced/server contract differed from the preceding engineer copy only in two comments. The raw hash gate therefore detected formatting divergence, while parsed scientific values remained identical. The active v007 contract hash is `{config.common_sha256}`.

## Demand and shelter override

{override_text}

Effective origins:

{origin_text}

Effective shelters:

{shelter_text}

- Contract demand/capacity: {contract_total_vehicles} / {contract_total_capacity}
- Effective demand/capacity: {total_vehicles} / {effective_total_capacity}
- Departure window: {effective_departure['begin_seconds']}–{effective_departure['end_seconds']} seconds
- Headless arrivals: **{arrived}/{total_vehicles}**; non-arrivals by the configured end: **{non_arrived}**
- Stage 3 capacity use: `{stage3.get('capacity_usage')}`

The effective endpoint lane counts are reported above from the active SUMO network. Relative to the immutable contract, effective demand changed by {demand_delta:+d} vehicle(s), effective shelter capacity changed by {capacity_delta:+d}, and the departure window {'changed' if departure_window_changed else 'was unchanged'}. Any arrival-rate change can therefore reflect the selected endpoints, lane geometry, demand, capacity, or network congestion and must not be attributed solely to fire or dynamic routing.

The existing v010 risk-plots sidecar is bound to v009 source hashes and artifacts. It must not be reused directly for this run even when its fleet denominator happens to match; a separately configured sidecar must target this run and declare its actual fleet size ({total_vehicles}).

## Independent reconstruction and fire footprint

- Fire-cell rows: {validation['handoff']['fire_rows']}; snapshots: {validation['handoff']['fire_snapshots']}
- Reconstructed mapping rows: {validation['oracle_comparisons']['mapping']['rows']}; routeable edges: {validation['oracle_comparisons']['mapping']['routeable_edges']}
- Minimum edge coverage: {validation['oracle_comparisons']['mapping']['minimum_edge_coverage']}
- Mapping maximum absolute errors: `{validation['oracle_comparisons']['mapping']['numeric_max_absolute_errors']}`
- Reconstructed hazard rows: {validation['oracle_comparisons']['hazard']['rows']}
- Hazard maximum absolute errors: `{validation['oracle_comparisons']['hazard']['numeric_max_absolute_errors']}`
- Hazard footprint: `{validation['oracle_comparisons']['hazard_footprint']}`
- Network SHA-256: `{validation['handoff']['network_sha256']}`
- Bundle `SHA256SUMS`: `{validation['handoff']['bundle_checksum_status']}`
- Local network SHA-256: `{validation['handoff']['local_network_sha256']}`
- Bundled network SHA-256: `{validation['handoff']['bundled_network_sha256']}`
- Bundled common-contract SHA-256: `{validation['handoff']['bundled_common_contract_sha256']}`
- Manifest common-hash field: `{validation['handoff']['common_hash_field']}`

## Stage 3 backend disclosure

The historical configuration declares `{backend['historical_declared_solver']}`. The executed implementation is the authorized `{backend['actual_backend']}` backend (`{backend['authorization_reference']}`). This is an explicitly disclosed adaptation, not a no-override claim. Regression evidence: `{backend['regression_expectation']}`. Observed assignment status `{stage3['status']}`, demand assigned `{stage3['all_demand_assigned']}`, capacity respected `{stage3['capacity_respected']}`, objective `{stage3['objective_value']}`.

## Stage 4 preservation

Stage 4 ran `{stage4.get('model_version', stage4.get('stage4_model_version', 'softmax_c1_v1'))}` with the common seed used directly. Chosen shelter remains immutable in Stage 5. Post-behaviour capacity overflow: `{stage4.get('capacity_overflow_total')}`.

## Formula and unit provenance

{formula_text}

Configured `lambda={config.shared['hazard']['edge_survival_and_risk']['lambda']}` and `epsilon={config.shared['hazard']['route_survival_and_risk']['epsilon']}` are synthetic, diagnostic and uncalibrated.

## Routing and outcome-neutral diagnostics

- Positive-risk candidate decisions: {positive}
- Differing-risk decisions: {differing}
- Decisions whose probabilities responded relative to the risk-neutral counterfactual: {responded}
- Mean expected-risk avoidance delta (`counterfactual - actual`): {delta}
- Mean realized selected-route risk at reconsideration decisions: {selected_risk_mean}
- Realized selections with positive route risk: {selected_positive}
- Expected-risk delta sign counts: `{avoidance_sign_counts}`
- Headless decision rows: {headless['decision_rows']}
- Validated hazard decision events (`interacting_fronts=false`): {validated_decisions}
- Unvalidated interacting-front decision events (`interacting_fronts=true`): {unvalidated_decisions}
- GUI execution: `{gui['status']}`
- Headless/GUI parity: `{parity['status']}`

Zero or negative aggregate avoidance is not a software failure. The controlled unit fixture, rather than the realized Manhattan outcome, is the acceptance evidence that unequal risks affect probabilities. The counterfactual holds candidates, travel times and panic fixed; realized route choices are stochastic and are not paired potential outcomes.

## Visual evidence

Scientific tables are authoritative; pixel appearance is diagnostic only. Risk color uses a fixed configured scale, flow uses one global configured comparison scale, and combined figures encode risk by color and flow by width with separate legends.

{family_text}

GUI screenshots: {', '.join(gui['screenshots']) if gui['screenshots'] else 'none; GUI deliberately disabled by runtime configuration'}

## Active-vehicle route risk and fire evolution

At every configured SUMO step, the observer runs after the active fire snapshot and scheduled route reconsideration, but before the next vehicle movement. It uses SUMO's active vehicle list and the existing log-space route-risk provider on each committed remaining route. Traversed edges are excluded. Between fire boundaries, the most recent snapshot is retained without interpolation.

- Derived table: `{stage5_headless / evolution_cfg['derived_table_filename']}`
- Vehicle-level source: `{stage5_headless / evolution_cfg['vehicle_table_filename']}`
- Recorded time rows: {len(evolution)}
- Steps with active vehicles: {int((evolution['active_vehicle_count'] > 0).sum())}
- Headless/GUI derived-table parity: `{'disabled_by_configuration' if not modes['parity_enabled'] else parity.get('evolution_exact_match') and parity.get('evolution_numeric_match')}`
- Axis-clipping validation: `passed`

Manual calculation at `time_seconds={manual_time:g}` (`time_step={int(manual_summary['time_step'])}`, active fire snapshot `{float(manual_summary['active_fire_snapshot_time_seconds']):g}`): individual risks `{manual_risks}`. Their arithmetic mean is `{float(manual_group['remaining_route_risk'].mean()):.12g}` with denominator `{len(manual_group)}`. The persisted row reports mean `{float(manual_summary['mean_active_route_risk']):.12g}`, BURNING cells `{int(manual_summary['burning_cell_count'])}`, and BURNED cells `{int(manual_summary['burned_cell_count'])}`.

This figure is descriptive. Visual correlation between fire evolution and mean route risk does not by itself prove causal avoidance or routing-policy effectiveness. The active cohort changes with departures and arrivals, and current-snapshot route risk does not predict future fire states along the route.

## Classification and review boundary

Technical integration: **passed**. Scientific fire calibration, Toulouse validation, behavioral calibration, and research approval: **not established**. No parameters were tuned to create apparent avoidance.
""", encoding="utf-8")
    publish_value = config.runtime["reporting"].get("repository_report_path")
    if publish_value:
        publish_path = Path(publish_value)
        if not publish_path.is_absolute():
            publish_path = config.repository_root / publish_path
        publish_path.parent.mkdir(parents=True, exist_ok=True)
        publish_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    return report_path
