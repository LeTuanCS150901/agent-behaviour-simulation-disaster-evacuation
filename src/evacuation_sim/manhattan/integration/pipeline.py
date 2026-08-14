from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sumolib
import yaml

from evacuation_sim.behavioral_model import run_stage4_manhattan
from evacuation_sim.io.tables import read_table, write_table
from evacuation_sim.route_choice.dynamic import candidate_routes, route_travel_time
from evacuation_sim.shelter_allocation import run_stage3_manhattan

from .config import ResolvedIntegrationConfig, sha256_file
from .decision import FireSnapshotProvider, Stage5DecisionEngine
from .handoff import ValidatedHandoff, hash_tree, validate_handoff
from .hazard import ReconstructedHazard, reconstruct_and_validate_hazard


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _phase_path(config: ResolvedIntegrationConfig, key: str) -> Path:
    return config.output_root / config.runtime["execution"]["phases"][key]


def validation_only(config: ResolvedIntegrationConfig) -> tuple[ValidatedHandoff, ReconstructedHazard, dict[str, Any]]:
    started = time.perf_counter()
    handoff = validate_handoff(config)
    reconstructed = reconstruct_and_validate_hazard(config, handoff)
    result = {
        "status": "passed",
        "elapsed_seconds": time.perf_counter() - started,
        "configuration": {
            "common_sha256": config.common_sha256,
            "runtime_sha256": config.runtime_sha256,
            "resolved_logical_sha256": config.logical_sha256,
        },
        "handoff": handoff.validation,
        "oracle_comparisons": reconstructed.comparison,
        "immutable_handoff_combined_sha256": _tree_logical_hash(handoff.hashes),
    }
    return handoff, reconstructed, result


def _tree_logical_hash(hashes: dict[str, str]) -> str:
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prepare_inputs(config: ResolvedIntegrationConfig, input_dir: Path) -> dict[str, Path]:
    role = config.role
    demand = role["demand"]
    shelters = role["shelters"]
    input_dir.mkdir(parents=True, exist_ok=False)
    origins = pd.DataFrame([
        {"origin_id": f"origin_{index:03d}", "edge_id": item["edge_id"], "num_cars": int(item["num_cars"])}
        for index, item in enumerate(demand["origins"])
    ])
    destinations = pd.DataFrame([
        {"destination_id": f"shelter_{index:03d}", "edge_id": item["edge_id"], "capacity": int(item["capacity"])}
        for index, item in enumerate(shelters["destinations"])
    ])
    departure = demand["departure_generation"]
    times = np.linspace(float(departure["begin_seconds"]), float(departure["end_seconds"]), int(demand["total_vehicles"]), endpoint=bool(departure["inclusive_end"]))
    vehicles: list[dict[str, Any]] = []
    vehicle_index = 0
    for origin in origins.itertuples(index=False):
        for _ in range(int(origin.num_cars)):
            vehicles.append({"vehicle_id": f"vehicle_{vehicle_index:05d}", "origin_id": origin.origin_id, "origin_edge_id": origin.edge_id, "depart_time": float(times[vehicle_index])})
            vehicle_index += 1
    vehicle_table = pd.DataFrame(vehicles)
    paths = {
        "origins": Path(write_table(origins, input_dir / "origins.parquet")),
        "destinations": Path(write_table(destinations, input_dir / "destinations.parquet")),
        "vehicles": Path(write_table(vehicle_table, input_dir / "vehicles.parquet")),
    }
    ignition = {"ignition_edges": [item["edge_id"] for item in config.shared["fire"]["ignition"]["sources"]]}
    paths["ignition"] = input_dir / "ignition_edges.json"
    _json(paths["ignition"], ignition)
    return paths


def _write_provenance(config: ResolvedIntegrationConfig, handoff: ValidatedHandoff, validation: dict[str, Any]) -> None:
    provenance = config.output_root / config.runtime["outputs"]["provenance_directory"]
    _yaml(provenance / "resolved_common_config.yaml", config.common)
    _yaml(provenance / "resolved_runtime_config.yaml", config.runtime)
    override = config.demand_override_provenance
    if override is not None:
        _yaml(provenance / "resolved_effective_role_config.yaml", {
            "demand_override": override,
            "effective_current_codebase_engineer_contract": config.role,
        })
    stage3 = config.runtime["stage3_backend"]
    source_manifest = {
        "common_config": {"path": str(config.common_path), "sha256": config.common_sha256},
        "runtime_config": {"path": str(config.runtime_path), "sha256": config.runtime_sha256},
        "resolved_logical_sha256": config.logical_sha256,
        "allowed_overrides": [
            "authorized Stage 3 solver-backend adaptation declared in runtime configuration",
            "optional configuration-driven demand and shelter override declared in runtime configuration",
        ],
        "applied_overrides": [{
            "field": "stage3.solver_backend",
            "historical_declared_value": stage3["historical_declared_solver"],
            "actual_implementation": stage3["actual_backend"],
            "authorized": stage3["authorized_substitution"],
            "authorization_reference": stage3["authorization_reference"],
            "regression_evidence": stage3["regression_expectation"],
        }],
        "rng_streams": config.rng_seeds,
        "handoff_hashes_before": handoff.hashes,
        "handoff_combined_sha256_before": _tree_logical_hash(handoff.hashes),
        "formula_units": config.runtime["reporting"]["formula_units"],
        "execution_modes": config.execution_modes,
    }
    if override is not None:
        source_manifest["applied_overrides"].append({
            "field": "current_codebase_engineer_contract.demand_and_shelters",
            "authorized": True,
            **override,
        })
        source_manifest["demand_override"] = override
    _json(provenance / "config_source_manifest.json", source_manifest)
    _json(_phase_path(config, "validation_report"), validation)


def prepare_run(config: ResolvedIntegrationConfig) -> dict[str, Any]:
    if config.output_root.exists():
        if config.runtime["execution"]["fail_if_output_exists"]:
            raise FileExistsError(f"Configured output_root already exists and fail_if_output_exists=true: {config.output_root}")
        raise FileExistsError(f"Refusing to reuse unverified output_root: {config.output_root}")
    handoff, reconstructed, validation = validation_only(config)
    config.output_root.mkdir(parents=True, exist_ok=False)
    _write_provenance(config, handoff, validation)
    outputs = config.runtime["outputs"]
    input_dir = config.output_root / outputs["input_directory"]
    input_paths = _prepare_inputs(config, input_dir)
    fire_dir = config.output_root / outputs["fire_directory"]
    fire_dir.mkdir(parents=True)
    write_table(reconstructed.mapping, fire_dir / outputs["mapping_table"])
    write_table(reconstructed.coverage, fire_dir / outputs["mapping_coverage_table"])
    write_table(reconstructed.edge_hazard, fire_dir / outputs["hazard_table"])
    _json(fire_dir / "oracle_comparison.json", reconstructed.comparison)
    stage3_dir = config.output_root / outputs["stage3_directory"]
    stage3_summary = run_stage3_manhattan(str(config.network_path), config.shared["network"]["vehicle_class"], input_dir, stage3_dir)
    expected_backend = config.runtime["stage3_backend"]["actual_backend"]
    if expected_backend != "scipy_exact_capacity_slot" or not stage3_summary["all_demand_assigned"] or not stage3_summary["capacity_respected"]:
        raise ValueError("Authorized Stage 3 capacity-slot regression expectations failed")
    stage4_cfg = yaml.safe_load(config.stage_configs["stage4"].read_text(encoding="utf-8"))
    stage4_dir = config.output_root / outputs["stage4_directory"]
    stage4_summary = run_stage4_manhattan(
        str(config.network_path), stage4_cfg, config.shared["reproducibility"]["random_seed"], input_dir, stage3_dir, stage4_dir,
        [item["edge_id"] for item in config.shared["fire"]["ignition"]["sources"]],
        run_identifier=config.runtime["execution"]["run_id"], config_hash=config.logical_sha256,
        result_classification=",".join(config.runtime["execution"]["result_classification"]),
    )
    prepared = {
        "status": "prepared",
        "run_id": config.runtime["execution"]["run_id"],
        "resolved_logical_sha256": config.logical_sha256,
        "input_paths": {name: str(path) for name, path in input_paths.items()},
        "stage3": stage3_summary,
        "stage4": stage4_summary,
        "hazard": reconstructed.comparison,
        "immutable_handoff_combined_sha256_before": _tree_logical_hash(handoff.hashes),
    }
    _json(_phase_path(config, "prepared_marker"), prepared)
    return prepared


def _assert_prepared(config: ResolvedIntegrationConfig) -> dict[str, Any]:
    marker = _phase_path(config, "prepared_marker")
    if not marker.is_file():
        raise FileNotFoundError("Run has not been prepared with the matching configuration")
    prepared = json.loads(marker.read_text(encoding="utf-8"))
    if prepared["resolved_logical_sha256"] != config.logical_sha256:
        raise ValueError("Prepared-run configuration hash differs from the active configuration")
    return prepared


def _provider_and_engine(config: ResolvedIntegrationConfig) -> tuple[FireSnapshotProvider, Stage5DecisionEngine]:
    outputs = config.runtime["outputs"]
    hazard = read_table(config.output_root / outputs["fire_directory"] / outputs["hazard_table"])
    shared_hazard = config.shared["hazard"]
    lookup = config.shared["clock"]["fire_time_lookup"]
    provider = FireSnapshotProvider(
        table=hazard,
        numerical_epsilon=float(shared_hazard["route_survival_and_risk"]["epsilon"]),
        time_lookup=lookup["method"], missing_edge_policy=shared_hazard["cell_to_edge"]["missing_edge_policy"],
        before_first_policy=lookup["before_first_snapshot"], after_last_policy=lookup["after_last_snapshot"],
    )
    stage5 = yaml.safe_load(config.stage_configs["stage5"].read_text(encoding="utf-8"))
    return provider, Stage5DecisionEngine(float(stage5["alpha_t"]), float(stage5["alpha_h"]), float(config.runtime["validation"]["probability_tolerance"]))


def _choose_route(config: ResolvedIntegrationConfig, provider: FireSnapshotProvider, engine: Stage5DecisionEngine, net, rng: np.random.Generator, vehicle_id: str, origin_edge: str, destination_edge: str, panic_rate: float, query_time: float) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    stage5 = yaml.safe_load(config.stage_configs["stage5"].read_text(encoding="utf-8"))
    routes = candidate_routes(net, origin_edge, destination_edge, config.shared["network"]["vehicle_class"], int(stage5["k_alternative_routes"]))
    if not routes:
        raise ValueError(f"No valid route for {vehicle_id}: {origin_edge}->{destination_edge}")
    travel = np.array([route_travel_time(net, route) for route in routes], dtype=float)
    risks = np.array([provider.route_risk(route, query_time)[1] for route in routes], dtype=float)
    probabilities = engine.probabilities(travel, risks, panic_rate)
    selected = int(rng.choice(len(routes), p=probabilities["final_probability"]))
    snapshot_metadata = provider.snapshot_metadata(query_time)
    candidate_rows = []
    for index, route in enumerate(routes):
        candidate_rows.append({
            "time": float(query_time), "vehicle_id": vehicle_id, "candidate_index": index,
            "route_edges": " ".join(route), "travel_time_seconds": travel[index], "route_risk": risks[index],
            "normalized_travel_time": probabilities["normalized_travel_time"][index], "utility": probabilities["utility"][index],
            "risk_neutral_utility": probabilities["risk_neutral_utility"][index],
            "probability": probabilities["final_probability"][index], "risk_neutral_probability": probabilities["risk_neutral_probability"][index],
            "selected": index == selected,
            "active_fire_snapshot_time_seconds": snapshot_metadata["snapshot_time"],
            "interacting_fronts": snapshot_metadata["interacting_fronts"],
        })
    summary = {
        "selected_index": selected,
        "positive_risk_candidate": bool(np.any(risks > config.runtime["avoidance_analysis"]["zero_tolerance"])),
        "differing_risks": bool(np.ptp(risks) > config.runtime["avoidance_analysis"]["zero_tolerance"]),
        "probabilities_responded": bool(np.max(np.abs(probabilities["final_probability"] - probabilities["risk_neutral_probability"])) > config.runtime["avoidance_analysis"]["zero_tolerance"]),
        "active_fire_snapshot_time_seconds": snapshot_metadata["snapshot_time"],
        "interacting_fronts": snapshot_metadata["interacting_fronts"],
    }
    return routes[selected], candidate_rows, summary


def _write_sumo_inputs(config: ResolvedIntegrationConfig, selected: pd.DataFrame) -> dict[str, Path]:
    outputs = config.runtime["outputs"]
    sumo_dir = config.output_root / outputs["sumo_directory"]
    sumo_dir.mkdir(parents=True, exist_ok=True)
    route_path = sumo_dir / outputs["route_file"]
    root = ET.Element("routes")
    ET.SubElement(root, "vType", {"id": config.runtime["sumo"]["route_vehicle_type_id"], "vClass": config.shared["network"]["vehicle_class"]})
    for row in selected.sort_values(["depart_time", "vehicle_id"], kind="stable").itertuples(index=False):
        route_id = f"initial_{row.vehicle_id}"
        ET.SubElement(root, "route", {"id": route_id, "edges": row.route_edges})
        ET.SubElement(root, "vehicle", {"id": row.vehicle_id, "depart": f"{float(row.depart_time):.6f}", "route": route_id, "type": config.runtime["sumo"]["route_vehicle_type_id"]})
    ET.ElementTree(root).write(route_path, encoding="utf-8", xml_declaration=True)
    result = {"routes": route_path}
    enabled_modes = [("headless", "headless_config", "tripinfo_headless")]
    if config.execution_modes["gui_enabled"]:
        enabled_modes.append(("gui", "gui_config", "tripinfo_gui"))
    for mode, config_key, trip_key in enabled_modes:
        cfg_path = sumo_dir / outputs[config_key]
        cfg_root = ET.Element("configuration")
        inputs = ET.SubElement(cfg_root, "input")
        ET.SubElement(inputs, "net-file", {"value": str(config.network_path)})
        ET.SubElement(inputs, "route-files", {"value": str(route_path)})
        if mode == "gui":
            add_path = config.handoff_directory / config.runtime["handoff"]["visualization_additional"]
            ET.SubElement(inputs, "additional-files", {"value": str(add_path)})
        times = ET.SubElement(cfg_root, "time")
        ET.SubElement(times, "begin", {"value": str(config.shared["clock"]["simulation_start_seconds"])})
        ET.SubElement(times, "end", {"value": str(config.shared["clock"]["simulation_end_seconds"])})
        processing = ET.SubElement(cfg_root, "processing")
        ET.SubElement(processing, "seed", {"value": str(config.rng_seeds[config.runtime["rng"]["sumo_stream"]["name"]])})
        ET.SubElement(processing, "time-to-teleport", {"value": str(config.runtime["sumo"]["teleport_seconds"])})
        output = ET.SubElement(cfg_root, "output")
        ET.SubElement(output, "tripinfo-output", {"value": str(sumo_dir / outputs[trip_key])})
        ET.ElementTree(cfg_root).write(cfg_path, encoding="utf-8", xml_declaration=True)
        result[mode] = cfg_path
    return result


def build_initial_stage5(config: ResolvedIntegrationConfig) -> dict[str, Any]:
    _assert_prepared(config)
    outputs = config.runtime["outputs"]
    input_dir = config.output_root / outputs["input_directory"]
    stage4_dir = config.output_root / outputs["stage4_directory"]
    vehicles = read_table(input_dir / "vehicles.parquet")
    profiles = read_table(stage4_dir / "behavioral_profiles.parquet")
    chosen = read_table(stage4_dir / "chosen_shelters.parquet")
    merged = vehicles.merge(profiles[["vehicle_id", "panic_rate"]], on="vehicle_id").merge(chosen[["vehicle_id", "chosen_destination_edge_id"]], on="vehicle_id")
    provider, engine = _provider_and_engine(config)
    net = sumolib.net.readNet(str(config.network_path))
    rng = np.random.default_rng(config.rng_seeds[config.runtime["rng"]["stage5_stream"]["name"]])
    candidate_rows, selected_rows = [], []
    for row in merged.sort_values("vehicle_id", kind="stable").itertuples(index=False):
        route, rows, summary = _choose_route(config, provider, engine, net, rng, row.vehicle_id, row.origin_edge_id, row.chosen_destination_edge_id, float(row.panic_rate), float(row.depart_time))
        candidate_rows.extend(rows)
        selected_rows.append({"vehicle_id": row.vehicle_id, "depart_time": row.depart_time, "panic_rate": row.panic_rate, "chosen_destination_edge_id": row.chosen_destination_edge_id, "route_edges": " ".join(route), **summary})
    stage5_dir = config.output_root / outputs["stage5_directory"]
    stage5_dir.mkdir(parents=True, exist_ok=True)
    write_table(pd.DataFrame(candidate_rows), stage5_dir / "initial_route_candidates.parquet")
    selected = pd.DataFrame(selected_rows)
    write_table(selected, stage5_dir / "initial_route_choices.parquet")
    _json(stage5_dir / "stage5_rng_state_after_initial.json", rng.bit_generator.state)
    sumo_paths = _write_sumo_inputs(config, selected)
    return {"vehicles": len(selected), "candidate_rows": len(candidate_rows), "sumo_paths": {key: str(path) for key, path in sumo_paths.items()}}
