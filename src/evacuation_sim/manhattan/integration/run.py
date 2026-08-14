from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sumolib

from evacuation_sim.io.tables import read_table, write_table

from .config import ResolvedIntegrationConfig, sha256_file
from .handoff import hash_tree, validate_handoff
from .pipeline import (
    _assert_prepared,
    _choose_route,
    _json,
    _phase_path,
    _provider_and_engine,
    _tree_logical_hash,
    build_initial_stage5,
    prepare_run,
)
from .risk_fire_evolution import active_vehicle_risk_rows, build_evolution_table


def _rgba(value: str) -> tuple[int, int, int, int]:
    text = value.lstrip("#")
    if len(text) not in (6, 8):
        raise ValueError(f"Configured color must be #RRGGBB or #RRGGBBAA: {value}")
    if len(text) == 6:
        text += "ff"
    return tuple(int(text[index:index + 2], 16) for index in range(0, 8, 2))


def _update_gui(config: ResolvedIntegrationConfig, traci, cells: pd.DataFrame, query_time: float, previous: dict[str, str]) -> dict[str, str]:
    snapshot = cells[cells["time_seconds"] == query_time]
    colors = config.runtime["visualization"]["fire_spread"]["state_colors"]
    current = dict(zip(snapshot["cell_id"], snapshot["canonical_state_label"]))
    available = set(traci.polygon.getIDList())
    for cell_id, state in current.items():
        if previous.get(cell_id) == state:
            continue
        polygon_id = f"fire_{cell_id}"
        if polygon_id not in available:
            raise KeyError(f"GUI polygon missing for configured cell {cell_id!r}: {polygon_id!r}")
        traci.polygon.setColor(polygon_id, _rgba(colors[state]))
    return current


def _flow_table(config: ResolvedIntegrationConfig, entries: list[dict[str, Any]], edge_ids: list[str]) -> pd.DataFrame:
    clock = config.shared["clock"]
    flow = config.runtime["traffic_flow"]
    start = float(clock["simulation_start_seconds"])
    end = float(clock["simulation_end_seconds"])
    window = float(flow["window_seconds"])
    starts = np.arange(start, end, window)
    observed = pd.DataFrame(entries)
    if observed.empty:
        grouped = pd.DataFrame(columns=["interval_start", "edge_id", "vehicle_edge_entries", "unique_vehicle_count"])
    else:
        observed["interval_start"] = start + np.floor((observed["time"] - start) / window) * window
        observed = observed[(observed["interval_start"] >= start) & (observed["interval_start"] < end)]
        grouped = observed.groupby(["interval_start", "edge_id"], as_index=False).agg(
            vehicle_edge_entries=("vehicle_id", "size"), unique_vehicle_count=("vehicle_id", "nunique")
        )
    complete = pd.MultiIndex.from_product([starts, edge_ids], names=["interval_start", "edge_id"]).to_frame(index=False)
    complete["interval_end"] = complete["interval_start"] + window
    complete = complete.merge(grouped, on=["interval_start", "edge_id"], how="left")
    complete[["vehicle_edge_entries", "unique_vehicle_count"]] = complete[["vehicle_edge_entries", "unique_vehicle_count"]].fillna(0).astype(int)
    complete["metric"] = flow["metric"]
    complete["interval_closure"] = flow["interval_closure"]
    return complete.sort_values(["interval_start", "edge_id"], kind="stable").reset_index(drop=True)


def _avoidance_table(config: ResolvedIntegrationConfig, decisions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    tolerance = float(config.runtime["avoidance_analysis"]["zero_tolerance"])
    rows = []
    for (time_value, vehicle_id), group in decisions.groupby(["time", "vehicle_id"], sort=True):
        if group["active_fire_snapshot_time_seconds"].nunique() != 1 or group["interacting_fronts"].nunique() != 1:
            raise ValueError("Candidate rows for one route decision disagree on active fire metadata")
        expected_actual = float((group["probability"] * group["route_risk"]).sum())
        expected_no_risk = float((group["risk_neutral_probability"] * group["route_risk"]).sum())
        selected_risk = float(group.loc[group["selected"], "route_risk"].iloc[0])
        delta = expected_no_risk - expected_actual
        rows.append({
            "time": time_value, "vehicle_id": vehicle_id,
            "active_fire_snapshot_time_seconds": float(group["active_fire_snapshot_time_seconds"].iloc[0]),
            "interacting_fronts": bool(group["interacting_fronts"].iloc[0]),
            "positive_risk_alternative": bool((group["route_risk"] > tolerance).any()),
            "differing_route_risks": bool(group["route_risk"].max() - group["route_risk"].min() > tolerance),
            "probabilities_responded": bool((group["probability"] - group["risk_neutral_probability"]).abs().max() > tolerance),
            "expected_risk_actual": expected_actual, "expected_risk_counterfactual": expected_no_risk,
            "expected_risk_avoidance_delta": delta, "selected_route_risk": selected_risk,
            "avoidance_sign": "positive" if delta > tolerance else "negative" if delta < -tolerance else "zero",
        })
    table = pd.DataFrame(rows)
    differing = table[table["differing_route_risks"]]
    technical_failure = bool((differing["probabilities_responded"] == False).any()) if not differing.empty else False
    summary = {
        "technical_probability_response": "failed" if technical_failure else "passed",
        "positive_risk_decisions": int(table["positive_risk_alternative"].sum()),
        "differing_risk_decisions": int(table["differing_route_risks"].sum()),
        "responding_decisions": int(table["probabilities_responded"].sum()),
        "mean_expected_risk_avoidance_delta": float(table["expected_risk_avoidance_delta"].mean()) if len(table) else None,
        "avoidance_sign_counts": table["avoidance_sign"].value_counts().to_dict(),
        "scientific_demonstration": "inconclusive" if differing.empty else "observed_without_calibration_claim",
        "counterfactual_limitation": "The no-risk probabilities hold candidate routes, travel times, panic rates and random setup fixed; realized route selections are stochastic and not paired potential outcomes.",
        "validated_decision_events": int((~table["interacting_fronts"]).sum()),
        "unvalidated_interacting_decision_events": int(table["interacting_fronts"].sum()),
    }
    if technical_failure:
        raise ValueError("Differing route risks did not change probabilities despite nonzero configured hazard weight")
    return table, summary


def run_sumo_mode(config: ResolvedIntegrationConfig, mode: str) -> dict[str, Any]:
    if mode not in {"headless", "gui"}:
        raise ValueError(f"Unsupported SUMO mode: {mode}")
    if mode == "gui" and not config.execution_modes["gui_enabled"]:
        raise ValueError("GUI execution is disabled by the validated runtime configuration")
    if mode == "headless" and not config.execution_modes["headless_enabled"]:
        raise ValueError("Headless execution is disabled by the validated runtime configuration")
    _assert_prepared(config)
    outputs = config.runtime["outputs"]
    mode_dir = config.output_root / outputs["stage5_directory"] / mode
    if mode_dir.exists():
        raise FileExistsError(f"SUMO phase output already exists: {mode_dir}")
    mode_dir.mkdir(parents=True)
    provider, engine = _provider_and_engine(config)
    net = sumolib.net.readNet(str(config.network_path))
    stage4_dir = config.output_root / outputs["stage4_directory"]
    profiles = read_table(stage4_dir / "behavioral_profiles.parquet").set_index("vehicle_id")
    chosen = read_table(stage4_dir / "chosen_shelters.parquet").set_index("vehicle_id")
    rng = np.random.default_rng()
    rng_state = json.loads((config.output_root / outputs["stage5_directory"] / "stage5_rng_state_after_initial.json").read_text(encoding="utf-8"))
    rng.bit_generator.state = rng_state
    binary = config.runtime["sumo"][f"{mode}_binary"]
    cfg_path = config.output_root / outputs["sumo_directory"] / outputs[f"{mode}_config"]
    command = [binary, "-c", str(cfg_path), "--seed", str(config.rng_seeds[config.runtime["rng"]["sumo_stream"]["name"]])]
    if config.runtime["sumo"]["no_step_log"]:
        command += ["--no-step-log", "true"]
    if config.runtime["sumo"]["start"]:
        command += ["--start"]
    import traci

    log_path = config.output_root / outputs["logs_directory"] / f"{mode}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    decision_rows: list[dict[str, Any]] = []
    active_risk_rows: list[dict[str, Any]] = []
    flow_entries: list[dict[str, Any]] = []
    previous_edges: dict[str, str] = {}
    previous_fire: dict[str, str] = {}
    screenshots: list[str] = []
    cells = pd.read_parquet(config.handoff_directory / config.runtime["handoff"]["fire_cells"])
    clock = config.shared["clock"]
    route_interval = float(clock["route_update_seconds"])
    current_time = float(clock["simulation_start_seconds"])
    end_time = float(clock["simulation_end_seconds"])
    step = float(clock["sumo_step_seconds"])
    boundary_times = set(float(v) for v in provider.times)
    evolution_cfg = config.runtime["visualization"].get("route_risk_fire_evolution")
    screenshot_times = set(float(v) for v in config.runtime["visualization"]["gui_screenshot_times_seconds"])
    try:
        traci.start(command)
        while current_time <= end_time + 1e-9:
            active_vehicle_ids = sorted(traci.vehicle.getIDList())
            rerouted_selected_risks: dict[str, float] = {}
            if current_time in boundary_times:
                if mode == "gui":
                    previous_fire = _update_gui(config, traci, cells, current_time, previous_fire)
                    if current_time in screenshot_times:
                        screenshot = config.output_root / outputs["figures_directory"] / "gui" / config.runtime["sumo"]["screenshot_filename_template"].format(time_seconds=int(current_time))
                        screenshot.parent.mkdir(parents=True, exist_ok=True)
                        traci.gui.screenshot(config.runtime["sumo"]["screenshot_view_id"], str(screenshot))
                        screenshots.append(str(screenshot))
                for vehicle_id in active_vehicle_ids:
                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    if not current_edge or current_edge.startswith(":") or vehicle_id not in chosen.index:
                        continue
                    destination = str(chosen.loc[vehicle_id, "chosen_destination_edge_id"])
                    if current_edge == destination:
                        continue
                    route, rows, _ = _choose_route(config, provider, engine, net, rng, vehicle_id, current_edge, destination, float(profiles.loc[vehicle_id, "panic_rate"]), current_time)
                    if route[-1] != destination:
                        raise ValueError(f"Route for {vehicle_id} no longer terminates at immutable shelter {destination}")
                    traci.vehicle.setRoute(vehicle_id, route)
                    decision_rows.extend(rows)
                    selected_risks = [float(row["route_risk"]) for row in rows if row["selected"]]
                    if len(selected_risks) != 1:
                        raise ValueError(f"Route decision for {vehicle_id!r} must have exactly one selected route")
                    rerouted_selected_risks[vehicle_id] = selected_risks[0]
            if evolution_cfg and evolution_cfg["enabled"]:
                time_step = int(round((current_time - float(clock["simulation_start_seconds"])) / step))
                active_risk_rows.extend(active_vehicle_risk_rows(
                    traci, provider, chosen, active_vehicle_ids, rerouted_selected_risks,
                    current_time, time_step, float(evolution_cfg["numerical_tolerance"]),
                ))
            if current_time >= end_time:
                break
            traci.simulationStep(current_time + step)
            current_time = float(traci.simulation.getTime())
            active = set(traci.vehicle.getIDList())
            for vehicle_id in sorted(active):
                road = traci.vehicle.getRoadID(vehicle_id)
                if not road or road.startswith(":"):
                    continue
                if previous_edges.get(vehicle_id) != road:
                    flow_entries.append({"time": current_time, "vehicle_id": vehicle_id, "edge_id": road})
                previous_edges[vehicle_id] = road
            previous_edges = {vehicle_id: edge for vehicle_id, edge in previous_edges.items() if vehicle_id in active}
        traci.close(True)
    except Exception:
        try:
            traci.close(True)
        except Exception:
            pass
        raise
    decisions = pd.DataFrame(decision_rows)
    active_risks = pd.DataFrame(active_risk_rows, columns=[
        "time_seconds", "time_step", "active_fire_snapshot_time_seconds", "vehicle_id",
        "current_road_id", "position_class", "rerouted_at_boundary", "remaining_route_edges",
        "remaining_edge_count", "remaining_route_risk", "measurement_phase",
        "interacting_fronts",
    ])
    if active_risks.duplicated(["time_seconds", "vehicle_id"]).any():
        raise ValueError("Active-vehicle risk table contains duplicate (time_seconds,vehicle_id) rows")
    if decisions.empty:
        decisions = pd.DataFrame(columns=["time", "vehicle_id", "candidate_index", "route_edges", "travel_time_seconds", "route_risk", "normalized_travel_time", "utility", "risk_neutral_utility", "probability", "risk_neutral_probability", "selected", "active_fire_snapshot_time_seconds", "interacting_fronts"])
    edge_ids = sorted(provider.table["edge_id"].unique())
    flows = _flow_table(config, flow_entries, edge_ids)
    flow_risk = provider.table[["time", "edge_id", "edge_hazard", "edge_survival", "edge_risk", "interacting_fronts"]].rename(columns={"time": "interval_start"})
    flows = flows.merge(flow_risk, on=["interval_start", "edge_id"], how="left", validate="one_to_one")
    if flows[["edge_hazard", "edge_survival", "edge_risk"]].isna().any().any():
        raise ValueError("Traffic-flow table could not be joined to a complete active hazard snapshot")
    avoidance, avoidance_summary = _avoidance_table(config, decisions) if not decisions.empty else (pd.DataFrame(), {"scientific_demonstration": "inconclusive", "reason": "no active vehicles at reconsideration boundaries"})
    write_table(decisions, mode_dir / outputs["decision_table"])
    route_risk_columns = ["time", "active_fire_snapshot_time_seconds", "interacting_fronts", "vehicle_id", "candidate_index", "route_edges", "route_risk", "selected"]
    write_table(decisions[route_risk_columns], mode_dir / outputs["route_risk_table"])
    write_table(flows, mode_dir / outputs["flow_table"])
    write_table(avoidance, mode_dir / outputs["avoidance_table"])
    if evolution_cfg and evolution_cfg["enabled"]:
        evolution = build_evolution_table(
            active_risks, cells,
            start_seconds=float(clock["simulation_start_seconds"]),
            end_seconds=end_time,
            sumo_step_seconds=step,
            recording_interval_seconds=float(evolution_cfg["recording_interval_seconds"]),
            burning_label=evolution_cfg["burning_state_label"],
            burned_label=evolution_cfg["burned_state_label"],
            state_column=evolution_cfg["fire_state_column"],
            expected_grid_cells=int(config.shared["fire"]["grid"]["expected_cell_count"]),
            tolerance=float(evolution_cfg["numerical_tolerance"]),
        )
        write_table(active_risks, mode_dir / evolution_cfg["vehicle_table_filename"])
        write_table(evolution, mode_dir / evolution_cfg["derived_table_filename"])
    summary = {
        "status": "passed", "mode": mode, "elapsed_seconds": time.perf_counter() - started,
        "command": command, "decision_rows": len(decisions), "flow_rows": len(flows),
        "active_vehicle_route_risk_rows": len(active_risks),
        "route_risk_fire_evolution_rows": len(evolution) if evolution_cfg and evolution_cfg["enabled"] else 0,
        "screenshots": screenshots, "avoidance": avoidance_summary,
        "final_scientific_timestamp_activated": end_time,
        "sumo_seed": config.rng_seeds[config.runtime["rng"]["sumo_stream"]["name"]],
        "stage5_rng_seed": config.rng_seeds[config.runtime["rng"]["stage5_stream"]["name"]],
    }
    _json(_phase_path(config, f"{mode}_summary"), summary)
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parity_run(config: ResolvedIntegrationConfig) -> dict[str, Any]:
    if not config.execution_modes["parity_enabled"]:
        raise ValueError("Headless/GUI parity is disabled by the validated runtime configuration")
    _assert_prepared(config)
    outputs = config.runtime["outputs"]
    stage5 = config.output_root / outputs["stage5_directory"]
    headless = read_table(stage5 / "headless" / outputs["decision_table"])
    gui = read_table(stage5 / "gui" / outputs["decision_table"])
    columns = ["time", "vehicle_id", "candidate_index", "route_edges", "route_risk", "utility", "probability", "selected"]
    headless = headless[columns].sort_values(columns[:3], kind="stable").reset_index(drop=True)
    gui = gui[columns].sort_values(columns[:3], kind="stable").reset_index(drop=True)
    exact = headless[["time", "vehicle_id", "candidate_index", "route_edges", "selected"]].equals(gui[["time", "vehicle_id", "candidate_index", "route_edges", "selected"]])
    numeric = all(np.allclose(headless[column], gui[column], atol=config.runtime["validation"]["probability_tolerance"], rtol=0) for column in ("route_risk", "utility", "probability")) if len(headless) == len(gui) else False
    evolution_cfg = config.runtime["visualization"].get("route_risk_fire_evolution")
    active_exact = active_numeric = evolution_exact = evolution_numeric = True
    active_rows_headless = active_rows_gui = evolution_rows_headless = evolution_rows_gui = 0
    if evolution_cfg and evolution_cfg["enabled"]:
        active_columns = [
            "time_seconds", "time_step", "active_fire_snapshot_time_seconds", "vehicle_id",
            "current_road_id", "position_class", "rerouted_at_boundary", "remaining_route_edges",
            "remaining_edge_count", "remaining_route_risk", "measurement_phase",
        ]
        active_headless = read_table(stage5 / "headless" / evolution_cfg["vehicle_table_filename"])
        active_gui = read_table(stage5 / "gui" / evolution_cfg["vehicle_table_filename"])
        active_headless = active_headless[active_columns].sort_values(["time_seconds", "vehicle_id"], kind="stable").reset_index(drop=True)
        active_gui = active_gui[active_columns].sort_values(["time_seconds", "vehicle_id"], kind="stable").reset_index(drop=True)
        active_rows_headless, active_rows_gui = len(active_headless), len(active_gui)
        active_text = [column for column in active_columns if column != "remaining_route_risk"]
        active_exact = active_headless[active_text].equals(active_gui[active_text])
        active_numeric = len(active_headless) == len(active_gui) and np.allclose(
            active_headless["remaining_route_risk"], active_gui["remaining_route_risk"],
            atol=config.runtime["validation"]["probability_tolerance"], rtol=0.0,
        )
        evolution_headless = read_table(stage5 / "headless" / evolution_cfg["derived_table_filename"])
        evolution_gui = read_table(stage5 / "gui" / evolution_cfg["derived_table_filename"])
        evolution_headless = evolution_headless.sort_values("time_step", kind="stable").reset_index(drop=True)
        evolution_gui = evolution_gui.sort_values("time_step", kind="stable").reset_index(drop=True)
        evolution_rows_headless, evolution_rows_gui = len(evolution_headless), len(evolution_gui)
        exact_columns = [
            "time_seconds", "time_step", "active_fire_snapshot_time_seconds", "active_vehicle_count",
            "valid_route_risk_vehicle_count", "burning_cell_count", "burned_cell_count",
        ]
        numeric_columns = ["mean_active_route_risk", "minimum_active_route_risk", "maximum_active_route_risk"]
        evolution_exact = evolution_headless[exact_columns].equals(evolution_gui[exact_columns])
        evolution_numeric = len(evolution_headless) == len(evolution_gui) and all(
            np.allclose(evolution_headless[column], evolution_gui[column],
                        atol=config.runtime["validation"]["probability_tolerance"], rtol=0.0, equal_nan=True)
            for column in numeric_columns
        )
    passed = exact and numeric and active_exact and active_numeric and evolution_exact and evolution_numeric
    result = {
        "status": "passed" if passed else "failed", "headless_rows": len(headless), "gui_rows": len(gui),
        "exact_route_and_selection_match": exact, "numeric_match": numeric,
        "active_vehicle_headless_rows": active_rows_headless, "active_vehicle_gui_rows": active_rows_gui,
        "active_vehicle_exact_match": active_exact, "active_vehicle_numeric_match": active_numeric,
        "evolution_headless_rows": evolution_rows_headless, "evolution_gui_rows": evolution_rows_gui,
        "evolution_exact_match": evolution_exact, "evolution_numeric_match": evolution_numeric,
    }
    if result["status"] != "passed":
        raise ValueError(f"Headless/GUI scientific parity failed: {result}")
    _json(_phase_path(config, "parity_report"), result)
    return result


def finalize_run(config: ResolvedIntegrationConfig) -> dict[str, Any]:
    _assert_prepared(config)
    from .visualization import generate_visualizations
    from .reporting import write_run_report

    handoff = validate_handoff(config)
    source_manifest_path = config.output_root / config.runtime["outputs"]["provenance_directory"] / "config_source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    before = source_manifest["handoff_hashes_before"]
    if before != handoff.hashes:
        raise ValueError("Immutable handoff content changed during execution")
    visualization = generate_visualizations(config)
    report = write_run_report(config, visualization, _tree_logical_hash(before), _tree_logical_hash(handoff.hashes))
    output_manifest_path = _phase_path(config, "final_manifest")
    excluded = {str(output_manifest_path.relative_to(config.output_root)).replace("\\", "/"), config.runtime["execution"]["phases"]["checksums"]}
    output_hashes = hash_tree(config.output_root, excluded=excluded)
    manifest = {
        "run_id": config.runtime["execution"]["run_id"],
        "resolved_config_hash": config.logical_sha256,
        "files": output_hashes,
        "immutable_handoff_before": before,
        "immutable_handoff_after": handoff.hashes,
    }
    if config.demand_override_provenance is not None:
        manifest["demand_override"] = config.demand_override_provenance
    _json(output_manifest_path, manifest)
    checksum_path = config.output_root / config.runtime["execution"]["phases"]["checksums"]
    checksum_path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(hash_tree(config.output_root, excluded={config.runtime['execution']['phases']['checksums']}).items())), encoding="utf-8")
    return {"status": "passed", "visualizations": visualization, "report": str(report), "output_manifest": str(output_manifest_path), "checksums": str(checksum_path), "handoff_before_after_equal": True}


def run_all(config: ResolvedIntegrationConfig) -> dict[str, Any]:
    result: dict[str, Any] = {"prepare": prepare_run(config)}
    result["stage5_initialization"] = build_initial_stage5(config)
    result["headless"] = run_sumo_mode(config, "headless")
    if config.execution_modes["gui_enabled"]:
        result["gui"] = run_sumo_mode(config, "gui")
    else:
        result["gui"] = {"status": "disabled_by_configuration"}
    if config.execution_modes["parity_enabled"]:
        result["parity"] = parity_run(config)
    else:
        result["parity"] = {"status": "disabled_by_configuration"}
    result["finalize"] = finalize_run(config)
    return result
