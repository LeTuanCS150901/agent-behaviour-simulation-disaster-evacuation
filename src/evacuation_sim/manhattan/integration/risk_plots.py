from __future__ import annotations

import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import jsonschema
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import sumolib
import yaml

from evacuation_sim.io.tables import write_table

from .config import sha256_file
from .spatial_fire_route_map import (
    compare_png,
    derive_cell_ignition,
    derive_driven_trace,
    derive_origin_destination_points,
    derive_road_grid_cells,
    derive_route_edge_activity,
    render_spatial_fire_route_map,
    rendering_library_versions,
)


@dataclass(frozen=True)
class RiskPlotsConfig:
    repository_root: Path
    runtime_path: Path
    common_path: Path
    runtime: dict[str, Any]
    common: dict[str, Any]
    runtime_sha256: str
    common_sha256: str
    source_root: Path
    handoff_root: Path
    output_root: Path
    source_paths: dict[str, Path]
    handoff_paths: dict[str, Path]
    lambda_value: float
    epsilon: float


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _resolve(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite; received {value!r}")
    return number


def _hash_tree(root: Path) -> tuple[dict[str, str], int, str]:
    records: list[tuple[str, str, int]] = []
    hashes: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        hashes[relative] = digest
        records.append((relative, digest, size))
    payload = json.dumps(records, separators=(",", ":")).encode("utf-8")
    return hashes, sum(record[2] for record in records), hashlib.sha256(payload).hexdigest()


def _source_path(source_root: Path, configured: str, label: str) -> Path:
    candidate = (source_root / configured).resolve()
    if not candidate.is_relative_to(source_root):
        raise ValueError(f"{label} escapes the configured source run: {configured!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Configured {label} does not exist: {candidate}")
    return candidate


def _verify_sha256sums(root: Path, checksum_path: Path) -> dict[str, str]:
    declared: dict[str, str] = {}
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed SHA256SUMS line {line_number}: {line!r}") from exc
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"SHA256SUMS path is missing or escapes its root: {relative!r}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise ValueError(
                f"SHA256SUMS mismatch for {candidate}: expected {expected}, got {actual}"
            )
        declared[relative] = expected
    return declared


def load_risk_plots_config(common_config: str | Path, runtime_config: str | Path) -> RiskPlotsConfig:
    root = Path.cwd().resolve()
    runtime_path = Path(runtime_config).resolve()
    common_path = Path(common_config).resolve()
    runtime = _load_mapping(runtime_path)
    schema_path = _resolve(root, runtime.get("runtime_schema", ""))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(runtime)

    configured_common = _resolve(root, runtime["common_contract"]["path"])
    if configured_common != common_path:
        raise ValueError(
            f"CLI common config {common_path} differs from the runtime reference {configured_common}"
        )
    common = _load_mapping(common_path)
    common_schema_path = _resolve(root, runtime["common_contract"]["schema"])
    common_schema = json.loads(common_schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(common_schema).validate(common)
    common_sha256 = sha256_file(common_path)
    expected_common = runtime["common_contract"]["expected_sha256"]
    if common_sha256 != expected_common:
        raise ValueError(
            f"Common-contract SHA-256 mismatch: expected {expected_common}, got {common_sha256}"
        )

    source = runtime["source_run"]
    source_root = _resolve(root, source["output_root"])
    if not source_root.is_dir():
        raise FileNotFoundError(f"Configured source run does not exist: {source_root}")
    output_root = _resolve(root, runtime["execution"]["output_root"])
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("The risk-plots sidecar must not be written inside the immutable source run")

    source_paths = {
        "config_source_manifest": _source_path(
            source_root, source["config_source_manifest"], "config-source manifest"
        ),
        "active_vehicle_table": _source_path(
            source_root, source["active_vehicle_table"], "active-vehicle table"
        ),
        "hazard_table": _source_path(source_root, source["hazard_table"], "hazard table"),
        "tripinfo": _source_path(source_root, source["tripinfo"], "tripinfo XML"),
        "route_choice_table": _source_path(
            source_root, source["route_choice_table"], "route-choice table"
        ),
        "edge_cell_mapping_table": _source_path(
            source_root, source["edge_cell_mapping_table"], "edge-cell mapping table"
        ),
    }
    optional_endpoint_sources = (
        ("origin_table", "origin table"),
        ("destination_table", "destination table"),
    )
    configured_endpoint_keys = [key for key, _ in optional_endpoint_sources if key in source]
    if configured_endpoint_keys and len(configured_endpoint_keys) != len(optional_endpoint_sources):
        raise ValueError("Origin and destination source tables must be configured together")
    for key, label in optional_endpoint_sources:
        if key in source:
            hash_key = f"{key}_sha256"
            if hash_key not in source:
                raise ValueError(f"Configured {key} requires {hash_key}")
            source_paths[key] = _source_path(source_root, source[key], label)
    for name, path in source_paths.items():
        expected = source[f"{name}_sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Source SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    manifest = json.loads(source_paths["config_source_manifest"].read_text(encoding="utf-8"))
    observed_logical = manifest.get("resolved_logical_sha256")
    if observed_logical != source["expected_resolved_config_sha256"]:
        raise ValueError(
            "Source resolved configuration identity differs from the risk-plots contract: "
            f"expected {source['expected_resolved_config_sha256']}, got {observed_logical}"
        )
    _, _, source_tree_sha256 = _hash_tree(source_root)
    if source_tree_sha256 != source["expected_tree_sha256"]:
        raise ValueError(
            f"Source-run tree SHA-256 mismatch: expected {source['expected_tree_sha256']}, "
            f"got {source_tree_sha256}"
        )

    spatial = runtime["spatial_sources"]
    handoff_root = _resolve(root, spatial["handoff_root"])
    if not handoff_root.is_dir():
        raise FileNotFoundError(f"Configured sealed handoff does not exist: {handoff_root}")
    handoff_paths = {
        "checksums": _source_path(handoff_root, spatial["checksums"], "handoff SHA256SUMS"),
        "manifest": _source_path(handoff_root, spatial["manifest"], "handoff manifest"),
        "fire_state_table": _source_path(
            handoff_root, spatial["fire_state_table"], "fire-state table"
        ),
        "fire_grid_table": _source_path(
            handoff_root, spatial["fire_grid_table"], "fire-grid table"
        ),
        "network": _source_path(handoff_root, spatial["network"], "sealed SUMO network"),
    }
    for name, path in handoff_paths.items():
        expected = spatial[f"{name}_sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Sealed-handoff SHA-256 mismatch for {path}: expected {expected}, got {actual}"
            )
    _verify_sha256sums(handoff_root, handoff_paths["checksums"])
    _, _, handoff_tree_sha256 = _hash_tree(handoff_root)
    if handoff_tree_sha256 != spatial["expected_tree_sha256"]:
        raise ValueError(
            f"Sealed-handoff tree SHA-256 mismatch: expected {spatial['expected_tree_sha256']}, "
            f"got {handoff_tree_sha256}"
        )
    handoff_manifest = json.loads(handoff_paths["manifest"].read_text(encoding="utf-8"))
    manifest_network_hash = handoff_manifest.get("network", {}).get("sha256")
    if manifest_network_hash != spatial["network_sha256"]:
        raise ValueError(
            f"Handoff manifest network hash differs: manifest={manifest_network_hash}, "
            f"configured={spatial['network_sha256']}"
        )

    shared = common["shared_contract"]
    common_fleet_size = int(common["current_codebase_engineer_contract"]["demand"]["total_vehicles"])
    configured_fleet_size = int(runtime["derivation"]["fleet_size"])
    if configured_fleet_size != common_fleet_size:
        raise ValueError(
            f"Configured fleet denominator {configured_fleet_size} differs from common contract "
            f"total_vehicles {common_fleet_size}"
        )
    clock = shared["clock"]
    start = _finite(clock["simulation_start_seconds"], "simulation_start_seconds")
    end = _finite(clock["simulation_end_seconds"], "simulation_end_seconds")
    risk_interval = _finite(runtime["derivation"]["risk_interval_seconds"], "risk_interval_seconds")
    arrival_interval = _finite(
        runtime["derivation"]["arrival_interval_seconds"], "arrival_interval_seconds"
    )
    tolerance = _finite(runtime["derivation"]["numerical_tolerance"], "numerical_tolerance")
    for interval, label in (
        (risk_interval, "risk_interval_seconds"),
        (arrival_interval, "arrival_interval_seconds"),
    ):
        quotient = (end - start) / interval
        if not math.isclose(quotient, round(quotient), abs_tol=tolerance, rel_tol=0.0):
            raise ValueError(f"{label} must divide the configured simulation horizon exactly")
    if int(runtime["derivation"]["sample_size"]) > configured_fleet_size:
        raise ValueError("sample_size cannot exceed fleet_size")
    if int(runtime["acceptance"]["minimum_nonzero_cells"]) > int(
        runtime["acceptance"]["maximum_nonzero_cells"]
    ):
        raise ValueError("The advisory non-zero-cell acceptance range is reversed")
    spatial_map = runtime["visualization"]["spatial_fire_route_map"]
    if set(spatial_map["draw_order"]) != {
        "network", "fire", "traces", "interaction_marking"
    }:
        raise ValueError("spatial draw_order must contain each configured layer exactly once")
    time_colour = spatial_map["time_colour"]
    if float(time_colour["vmax_seconds"]) <= float(time_colour["vmin_seconds"]):
        raise ValueError("Spatial time_colour vmax_seconds must exceed vmin_seconds")
    route_lines = spatial_map["route_lines"]
    if float(route_lines["maximum_width_points"]) < float(
        route_lines["minimum_width_points"]
    ):
        raise ValueError("Route-line maximum width must be at least its minimum width")
    legend_values = [int(value) for value in route_lines["width_legend_values"]]
    if legend_values != sorted(legend_values):
        raise ValueError("Route-line width legend values must be sorted")
    endpoint_overlay = spatial_map.get("origin_destination_overlay")
    if endpoint_overlay is not None:
        if not {"origin_table", "destination_table"}.issubset(source_paths):
            raise ValueError(
                "origin_destination_overlay requires hashed origin_table and destination_table sources"
            )
        if endpoint_overlay["filename"] == endpoint_overlay["base_figure_filename"]:
            raise ValueError("Endpoint-overlay filename must not overwrite the base spatial figure")
        if endpoint_overlay["base_figure_filename"] != spatial_map["filename"]:
            raise ValueError("Endpoint-overlay base filename must match the configured spatial figure")
    preserve_paths = runtime["execution"]["preserve_byte_identical"]
    if len(preserve_paths) != 9:
        raise ValueError(
            "Exactly five existing scientific tables and four existing figures must be preserved"
        )

    survival = shared["hazard"]["edge_survival_and_risk"]
    route_survival = shared["hazard"]["route_survival_and_risk"]
    lambda_value = _finite(survival["lambda"], "edge_survival_and_risk.lambda")
    epsilon = _finite(route_survival["epsilon"], "route_survival_and_risk.epsilon")
    if lambda_value <= 0 or not 0 < epsilon <= 1:
        raise ValueError("Contract lambda must be positive and epsilon must lie in (0,1]")

    return RiskPlotsConfig(
        repository_root=root,
        runtime_path=runtime_path,
        common_path=common_path,
        runtime=runtime,
        common=common,
        runtime_sha256=sha256_file(runtime_path),
        common_sha256=common_sha256,
        source_root=source_root,
        handoff_root=handoff_root,
        output_root=output_root,
        source_paths=source_paths,
        handoff_paths=handoff_paths,
        lambda_value=lambda_value,
        epsilon=epsilon,
    )


class _HazardLookup:
    def __init__(self, table: pd.DataFrame, epsilon: float):
        required = {"time", "edge_id", "edge_survival", "edge_hazard", "interacting_fronts"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"Hazard table is missing columns: {sorted(missing)}")
        if table.duplicated(["time", "edge_id"]).any():
            raise ValueError("Hazard table contains duplicate (time,edge_id) rows")
        numeric = table[["time", "edge_survival", "edge_hazard"]].to_numpy(float)
        if not np.isfinite(numeric).all():
            raise ValueError("Hazard table contains non-finite values")
        survivals = table["edge_survival"].to_numpy(float)
        if (survivals < 0).any() or (survivals > 1).any():
            raise ValueError("Edge survival must lie in [0,1]")
        if table["interacting_fronts"].isna().any():
            raise ValueError("Hazard interacting_fronts values may not be null")
        flags = table.groupby("time")["interacting_fronts"].nunique()
        if (flags != 1).any():
            raise ValueError("Each hazard snapshot must have one interacting_fronts value")
        self.epsilon = float(epsilon)
        self.times = np.sort(table["time"].astype(float).unique())
        self.snapshots = {
            float(time_value): group.set_index("edge_id", drop=False)
            for time_value, group in table.groupby("time", sort=True)
        }

    def snapshot_time(self, query_time: float) -> float:
        position = int(np.searchsorted(self.times, float(query_time), side="right") - 1)
        if position < 0:
            raise ValueError(f"No fire snapshot exists at or before time {query_time}")
        return float(self.times[position])

    def metadata(self, query_time: float) -> tuple[float, bool]:
        snapshot_time = self.snapshot_time(query_time)
        snapshot = self.snapshots[snapshot_time]
        return snapshot_time, bool(snapshot["interacting_fronts"].iloc[0])

    def survival(self, edge_id: str, query_time: float, vehicle_id: str | None = None) -> float:
        snapshot_time = self.snapshot_time(query_time)
        snapshot = self.snapshots[snapshot_time]
        if edge_id not in snapshot.index:
            subject = f" for vehicle {vehicle_id!r}" if vehicle_id is not None else ""
            raise KeyError(
                f"Edge {edge_id!r}{subject} is absent from hazard snapshot {snapshot_time:g} "
                f"for observation time {query_time:g}"
            )
        return float(snapshot.loc[edge_id, "edge_survival"])


def _inclusive_times(start: float, end: float, interval: float, tolerance: float) -> np.ndarray:
    count = int(round((end - start) / interval)) + 1
    values = start + np.arange(count, dtype=float) * interval
    if not math.isclose(float(values[-1]), end, abs_tol=tolerance, rel_tol=0.0):
        raise ValueError("Configured timeline does not end at simulation_end_seconds")
    return values


def _validate_active_table(
    active: pd.DataFrame, fleet_size: int, sumo_step: float, tolerance: float
) -> list[str]:
    required = {
        "time_seconds", "vehicle_id", "current_road_id", "position_class",
        "active_fire_snapshot_time_seconds", "interacting_fronts",
    }
    missing = required - set(active.columns)
    if missing:
        raise ValueError(f"Active-vehicle table is missing columns: {sorted(missing)}")
    if active.duplicated(["time_seconds", "vehicle_id"]).any():
        raise ValueError("Active-vehicle table contains duplicate (time_seconds,vehicle_id) rows")
    if active[["time_seconds", "active_fire_snapshot_time_seconds"]].isna().any().any():
        raise ValueError("Active-vehicle timestamps may not be null")
    if not np.isfinite(active[["time_seconds", "active_fire_snapshot_time_seconds"]].to_numpy(float)).all():
        raise ValueError("Active-vehicle timestamps must be finite")
    if active[["vehicle_id", "current_road_id"]].isna().any().any():
        raise ValueError("Active vehicle and road identifiers may not be null")
    fleet = sorted(active["vehicle_id"].astype(str).unique())
    if len(fleet) != fleet_size:
        raise ValueError(
            f"Active-vehicle table contains {len(fleet)} unique vehicles; expected fleet_size={fleet_size}"
        )
    for vehicle_id, group in active.groupby("vehicle_id", sort=False):
        times = np.sort(group["time_seconds"].to_numpy(float))
        if len(times) > 1 and not np.allclose(np.diff(times), sumo_step, atol=tolerance, rtol=0.0):
            raise ValueError(f"Active observations for {vehicle_id!r} are not contiguous SUMO steps")
    return fleet


def derive_per_agent_risk(
    active: pd.DataFrame,
    hazard: pd.DataFrame,
    *,
    fleet_size: int,
    start_seconds: float,
    end_seconds: float,
    sumo_step_seconds: float,
    interval_seconds: float,
    epsilon: float,
    internal_edge_prefix: str,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fleet = _validate_active_table(active, fleet_size, sumo_step_seconds, tolerance)
    lookup = _HazardLookup(hazard, epsilon)
    sample_times = _inclusive_times(start_seconds, end_seconds, interval_seconds, tolerance)
    sample_index = {
        int(round((time_value - start_seconds) / interval_seconds)): float(time_value)
        for time_value in sample_times
    }
    sampled = active.copy()
    quotient = (sampled["time_seconds"].to_numpy(float) - start_seconds) / interval_seconds
    rounded = np.rint(quotient).astype(int)
    aligned = np.isclose(quotient, rounded, atol=tolerance, rtol=0.0)
    sampled = sampled.loc[aligned].copy()
    sampled["sample_index"] = rounded[aligned]
    sampled = sampled[sampled["sample_index"].isin(sample_index)]
    sampled_by_vehicle = {
        str(vehicle_id): group.set_index("sample_index", drop=False)
        for vehicle_id, group in sampled.groupby("vehicle_id", sort=False)
    }

    rows: list[dict[str, Any]] = []
    for vehicle_id in fleet:
        observations = sampled_by_vehicle.get(vehicle_id)
        frozen_log_sum = 0.0
        current_normal_edge: str | None = None
        current_log_survival: float | None = None
        was_active = False
        for index, time_value in sample_index.items():
            snapshot_time, interacting = lookup.metadata(time_value)
            row = observations.loc[index] if observations is not None and index in observations.index else None
            if isinstance(row, pd.DataFrame):
                raise ValueError(
                    f"Vehicle {vehicle_id!r} has multiple observations at time {time_value:g}"
                )
            is_active = row is not None
            current_edge: str | None = None
            position_class: str | None = None
            current_survival: float | None = None
            if is_active:
                current_edge = str(row["current_road_id"])
                position_class = str(row["position_class"])
                if not math.isclose(
                    float(row["active_fire_snapshot_time_seconds"]), snapshot_time,
                    abs_tol=tolerance, rel_tol=0.0,
                ):
                    raise ValueError(
                        f"Vehicle {vehicle_id!r} at {time_value:g} uses fire snapshot "
                        f"{row['active_fire_snapshot_time_seconds']}, expected {snapshot_time}"
                    )
                if bool(row["interacting_fronts"]) != interacting:
                    raise ValueError(
                        f"Vehicle {vehicle_id!r} at {time_value:g} has an inconsistent "
                        "interacting_fronts flag"
                    )
                if current_edge.startswith(internal_edge_prefix):
                    if current_normal_edge is not None:
                        if current_log_survival is None:
                            raise RuntimeError("Normal edge state is missing its last survival")
                        frozen_log_sum += current_log_survival
                        current_normal_edge = None
                        current_log_survival = None
                    risk = 1.0 - math.exp(frozen_log_sum)
                else:
                    current_survival = lookup.survival(current_edge, time_value, vehicle_id)
                    log_survival = math.log(float(np.clip(current_survival, epsilon, 1.0)))
                    if current_normal_edge is not None and current_normal_edge != current_edge:
                        if current_log_survival is None:
                            raise RuntimeError("Normal edge state is missing its last survival")
                        frozen_log_sum += current_log_survival
                    current_normal_edge = current_edge
                    current_log_survival = log_survival
                    risk = 1.0 - math.exp(frozen_log_sum + current_log_survival)
            else:
                if was_active and current_normal_edge is not None:
                    if current_log_survival is None:
                        raise RuntimeError("Normal edge state is missing its last survival")
                    frozen_log_sum += current_log_survival
                    current_normal_edge = None
                    current_log_survival = None
                risk = 0.0
            if not math.isfinite(risk) or risk < -tolerance or risk > 1.0 + tolerance:
                raise ValueError(f"Invalid R_i for vehicle {vehicle_id!r} at {time_value:g}: {risk}")
            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "time_seconds": float(time_value),
                    "current_edge_id": current_edge,
                    "position_class": position_class,
                    "S_e_current": current_survival,
                    "frozen_log_sum": float(frozen_log_sum),
                    "R_i": float(risk),
                    "is_active": bool(is_active),
                    "active_fire_snapshot_time_seconds": float(snapshot_time),
                    "interacting_fronts": bool(interacting),
                }
            )
            was_active = is_active
    result = pd.DataFrame(rows).sort_values(["time_seconds", "vehicle_id"], kind="stable").reset_index(drop=True)
    expected_rows = fleet_size * len(sample_times)
    if len(result) != expected_rows or result.duplicated(["time_seconds", "vehicle_id"]).any():
        raise ValueError(f"Per-agent risk matrix must contain exactly {expected_rows} unique rows")

    risk_zero = tolerance
    risk_agents = set(result.loc[result["R_i"] > risk_zero, "vehicle_id"])
    sampled_hazard_agents: set[str] = set()
    for row in sampled.itertuples(index=False):
        edge_id = str(row.current_road_id)
        if edge_id.startswith(internal_edge_prefix):
            continue
        if lookup.survival(edge_id, float(row.time_seconds), str(row.vehicle_id)) < 1.0 - tolerance:
            sampled_hazard_agents.add(str(row.vehicle_id))
    if risk_agents != sampled_hazard_agents:
        raise ValueError(
            "Sampled risk-agent set differs from sampled concurrent hazardous-edge occupancy; "
            f"risk_only={sorted(risk_agents - sampled_hazard_agents)}, "
            f"occupancy_only={sorted(sampled_hazard_agents - risk_agents)}"
        )
    return result, {
        "fleet": fleet,
        "sample_times": sample_times.tolist(),
        "nonzero_agents": sorted(risk_agents),
        "sampled_hazard_agents": sorted(sampled_hazard_agents),
    }


def derive_model_risk(per_agent: pd.DataFrame) -> pd.DataFrame:
    grouped = per_agent.groupby("time_seconds", sort=True)
    for time_value, group in grouped:
        if group["active_fire_snapshot_time_seconds"].nunique() != 1:
            raise ValueError(f"Per-agent rows at {time_value:g} disagree on the active fire snapshot")
        if group["interacting_fronts"].nunique() != 1:
            raise ValueError(f"Per-agent rows at {time_value:g} disagree on interacting_fronts")
    return grouped.agg(
        R_model=("R_i", "sum"),
        active_agent_count=("is_active", "sum"),
        active_fire_snapshot_time_seconds=("active_fire_snapshot_time_seconds", "first"),
        interacting_fronts=("interacting_fronts", "first"),
    ).reset_index()


def derive_positive_risk_vehicle_count(
    per_agent: pd.DataFrame,
    *,
    risk_zero_tolerance: float,
) -> pd.DataFrame:
    """Count unique vehicles with positive sampled ``R_i(t)`` at each timestamp.

    The result is point-in-time, not cumulative.  The source must contain one
    row per vehicle and timestamp and consistent fire-snapshot metadata within
    each timestamp.
    """
    required = {
        "vehicle_id",
        "time_seconds",
        "R_i",
        "active_fire_snapshot_time_seconds",
        "interacting_fronts",
    }
    missing = required.difference(per_agent.columns)
    if missing:
        raise ValueError(f"Per-agent risk table is missing columns: {sorted(missing)}")
    if per_agent.empty:
        raise ValueError("Per-agent risk table contains no rows")
    if per_agent.duplicated(["vehicle_id", "time_seconds"]).any():
        raise ValueError("Per-agent risk table contains duplicate (vehicle_id, time_seconds) rows")

    tolerance = float(risk_zero_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("risk_zero_tolerance must be finite and non-negative")

    risks = pd.to_numeric(per_agent["R_i"], errors="raise").to_numpy(float)
    times = pd.to_numeric(per_agent["time_seconds"], errors="raise").to_numpy(float)
    snapshots = pd.to_numeric(
        per_agent["active_fire_snapshot_time_seconds"], errors="raise"
    ).to_numpy(float)
    if not np.isfinite(risks).all() or not np.isfinite(times).all() or not np.isfinite(snapshots).all():
        raise ValueError("Risk, observation time, and fire-snapshot time must be finite")
    if (risks < -tolerance).any():
        raise ValueError("R_i values must be non-negative within the configured tolerance")

    rows: list[dict[str, Any]] = []
    expected_fleet_size: int | None = None
    for time_step, (time_seconds, group) in enumerate(
        per_agent.groupby("time_seconds", sort=True)
    ):
        if group["active_fire_snapshot_time_seconds"].nunique(dropna=False) != 1:
            raise ValueError(
                f"Rows at {float(time_seconds):g} s disagree on active_fire_snapshot_time_seconds"
            )
        if group["interacting_fronts"].nunique(dropna=False) != 1:
            raise ValueError(f"Rows at {float(time_seconds):g} s disagree on interacting_fronts")
        fleet_size = int(group["vehicle_id"].astype(str).nunique())
        if expected_fleet_size is None:
            expected_fleet_size = fleet_size
        elif fleet_size != expected_fleet_size:
            raise ValueError(
                "Every timestamp must contain the same complete fleet; "
                f"expected {expected_fleet_size}, got {fleet_size} at {float(time_seconds):g} s"
            )
        rows.append(
            {
                "time_step": int(time_step),
                "time_seconds": float(time_seconds),
                "active_fire_snapshot_time_seconds": float(
                    group["active_fire_snapshot_time_seconds"].iloc[0]
                ),
                "interacting_fronts": bool(group["interacting_fronts"].iloc[0]),
                "positive_risk_vehicle_count": int(
                    group.loc[
                        pd.to_numeric(group["R_i"], errors="raise") > tolerance,
                        "vehicle_id",
                    ]
                    .astype(str)
                    .nunique()
                ),
                "fleet_size": fleet_size,
                "risk_zero_tolerance": tolerance,
            }
        )
    return pd.DataFrame(rows)


def _parse_tripinfo(path: Path, fleet: Iterable[str]) -> pd.DataFrame:
    fleet_set = set(fleet)
    records: list[dict[str, Any]] = []
    for element in ET.parse(path).getroot().iter("tripinfo"):
        vehicle_id = str(element.attrib.get("id", ""))
        if not vehicle_id:
            raise ValueError("tripinfo contains a row without an id")
        if vehicle_id not in fleet_set:
            raise ValueError(f"tripinfo contains unknown vehicle {vehicle_id!r}")
        depart = _finite(element.attrib.get("depart"), f"tripinfo[{vehicle_id}].depart")
        arrival = _finite(element.attrib.get("arrival"), f"tripinfo[{vehicle_id}].arrival")
        if arrival < depart:
            raise ValueError(f"tripinfo arrival precedes departure for {vehicle_id!r}")
        records.append({"vehicle_id": vehicle_id, "depart_seconds": depart, "arrival_seconds": arrival})
    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError("tripinfo contains no completed trips")
    if table["vehicle_id"].duplicated().any():
        duplicates = sorted(table.loc[table["vehicle_id"].duplicated(False), "vehicle_id"].unique())
        raise ValueError(f"tripinfo contains duplicate vehicle rows: {duplicates}")
    return table.sort_values(["arrival_seconds", "vehicle_id"], kind="stable").reset_index(drop=True)


def derive_arrivals(
    tripinfo: pd.DataFrame,
    lookup: _HazardLookup,
    *,
    start_seconds: float,
    end_seconds: float,
    interval_seconds: float,
    fleet_size: int,
    tolerance: float,
) -> pd.DataFrame:
    times = _inclusive_times(start_seconds, end_seconds, interval_seconds, tolerance)
    arrivals = np.sort(tripinfo["arrival_seconds"].to_numpy(float))
    rows = []
    for time_value in times:
        arrived = int(np.searchsorted(arrivals, time_value, side="right"))
        snapshot_time, interacting = lookup.metadata(float(time_value))
        rows.append(
            {
                "time_seconds": float(time_value),
                "arrived_count": arrived,
                "arrived_fraction": arrived / fleet_size,
                "active_fire_snapshot_time_seconds": float(snapshot_time),
                "interacting_fronts": bool(interacting),
            }
        )
    result = pd.DataFrame(rows)
    if not result["arrived_count"].is_monotonic_increasing:
        raise ValueError("Cumulative arrival count must be monotonically non-decreasing")
    if (result["arrived_fraction"] < 0).any() or (result["arrived_fraction"] > 1).any():
        raise ValueError("Cumulative arrival fraction must lie in [0,1]")
    return result


def exposure_evidence(
    active: pd.DataFrame,
    lookup: _HazardLookup,
    *,
    start_seconds: float,
    internal_edge_prefix: str,
    intervals: Iterable[float],
    tolerance: float,
) -> dict[str, Any]:
    normal = active[~active["current_road_id"].astype(str).str.startswith(internal_edge_prefix)].copy()
    concurrently_exposed: set[str] = set()
    for row in normal.itertuples(index=False):
        if lookup.survival(str(row.current_road_id), float(row.time_seconds), str(row.vehicle_id)) < 1.0 - tolerance:
            concurrently_exposed.add(str(row.vehicle_id))
    ever_hazardous_edges = {
        str(edge_id)
        for snapshot in lookup.snapshots.values()
        for edge_id, survival in snapshot["edge_survival"].items()
        if float(survival) < 1.0 - tolerance
    }
    ever_hazardous_agents = set(
        normal.loc[normal["current_road_id"].astype(str).isin(ever_hazardous_edges), "vehicle_id"].astype(str)
    )
    sensitivity: dict[str, int] = {}
    for interval in intervals:
        interval = float(interval)
        quotient = (normal["time_seconds"].to_numpy(float) - start_seconds) / interval
        aligned = np.isclose(quotient, np.rint(quotient), atol=tolerance, rtol=0.0)
        found: set[str] = set()
        for row in normal.loc[aligned].itertuples(index=False):
            if lookup.survival(str(row.current_road_id), float(row.time_seconds), str(row.vehicle_id)) < 1.0 - tolerance:
                found.add(str(row.vehicle_id))
        sensitivity[f"{interval:g}"] = len(found)
    return {
        "one_second_concurrent_exposure_agents": sorted(concurrently_exposed),
        "ever_hazardous_edge_agents": sorted(ever_hazardous_agents),
        "interval_sensitivity_agent_counts": sensitivity,
    }


def _annotation(template: str, context: dict[str, Any]) -> str:
    try:
        return template.format_map(context)
    except KeyError as exc:
        raise ValueError(f"Annotation template references an unavailable field: {exc.args[0]}") from exc


def _save_figure(fig, path: Path, settings: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=int(settings["dpi"]),
        bbox_inches="tight",
        metadata={key: str(value) for key, value in settings["rendering_metadata"].items()},
    )
    plt.close(fig)


def _display_end(config: RiskPlotsConfig, tripinfo: pd.DataFrame) -> float:
    derivation = config.runtime["derivation"]
    clock = config.common["shared_contract"]["clock"]
    if derivation["x_axis_anchor"] == "last_arrival":
        anchor = float(tripinfo["arrival_seconds"].max())
    elif derivation["x_axis_anchor"] == "simulation_end":
        anchor = float(clock["simulation_end_seconds"])
    else:
        raise ValueError(f"Unsupported x-axis anchor: {derivation['x_axis_anchor']}")
    return anchor + float(derivation["x_axis_offset_seconds"])


def _plot_arrivals(
    arrival: pd.DataFrame, path: Path, config: RiskPlotsConfig, context: dict[str, Any]
) -> None:
    settings = config.runtime["visualization"]
    plot = settings["arrival"]
    fig, ax = plt.subplots(figsize=tuple(plot["figure_size_inches"]))
    ax.step(
        arrival["time_seconds"].to_numpy(float) / 60.0,
        arrival["arrived_fraction"].to_numpy(float) * 100.0,
        where="post", color=plot["line_color"], linewidth=float(plot["line_width"]),
    )
    ax.set_xlim(float(arrival["time_seconds"].min()) / 60.0, float(arrival["time_seconds"].max()) / 60.0)
    ax.set_ylim(*[float(value) for value in plot["y_limits"]])
    ax.set_xlabel(plot["x_label"])
    ax.set_ylabel(plot["y_label"])
    ax.set_title(plot["title"])
    ax.grid(True, color=settings["grid_color"], alpha=float(settings["grid_alpha"]))
    ax.text(
        0.02, 0.96, _annotation(plot["annotation_template"], context),
        transform=ax.transAxes, ha="left", va="top",
        bbox={"facecolor": "white", "edgecolor": plot["line_color"], "alpha": 0.92},
    )
    _save_figure(fig, path, settings)


def _risk_plot_series(table: pd.DataFrame, value_column: str, display_end: float) -> tuple[np.ndarray, np.ndarray]:
    visible = table[table["time_seconds"] <= display_end].sort_values("time_seconds", kind="stable")
    x = visible["time_seconds"].to_numpy(float)
    y = visible[value_column].to_numpy(float)
    if not len(x):
        raise ValueError("Risk plot has no data at or before the configured display end")
    if display_end > x[-1]:
        x = np.append(x, display_end)
        y = np.append(y, 0.0)
    elif math.isclose(display_end, x[-1]):
        y[-1] = 0.0
    return x, y


def _plot_agent_sample(
    per_agent: pd.DataFrame,
    sample: list[str],
    population: str,
    display_end: float,
    path: Path,
    config: RiskPlotsConfig,
    context: dict[str, Any],
) -> None:
    settings = config.runtime["visualization"]
    plot = settings["per_agent"]
    fig, ax = plt.subplots(figsize=tuple(plot["figure_size_inches"]))
    color_map = matplotlib.colormaps[plot["color_map"]]
    colors = color_map(np.linspace(0.0, 1.0, len(sample)))
    for vehicle_id, color in zip(sample, colors):
        group = per_agent[per_agent["vehicle_id"] == vehicle_id]
        x, y = _risk_plot_series(group, "R_i", display_end)
        ax.step(
            x, y, where="post", label=vehicle_id, color=color,
            linewidth=float(plot["line_width"]),
        )
        ax.plot(
            [display_end], [0.0], linestyle="none", marker=plot["terminal_marker"],
            markersize=float(plot["terminal_marker_size"]), color=color,
        )
    ax.set_xlim(float(per_agent["time_seconds"].min()), display_end)
    ax.set_ylim(*[float(value) for value in plot["y_limits"]])
    ax.set_xlabel(plot["x_label"])
    ax.set_ylabel(plot["y_label"])
    ax.set_title(plot["titles"][population])
    ax.grid(True, color=settings["grid_color"], alpha=float(settings["grid_alpha"]))
    ax.legend(ncol=int(plot["legend_columns"]), loc="upper left", fontsize="small")
    ax.text(
        0.01, -0.18, _annotation(plot["annotation_template"], context),
        transform=ax.transAxes, ha="left", va="top", wrap=True,
        bbox={"facecolor": "white", "edgecolor": settings["grid_color"], "alpha": 0.95},
    )
    _save_figure(fig, path, settings)


def _plot_model(
    model: pd.DataFrame,
    display_end: float,
    path: Path,
    config: RiskPlotsConfig,
    context: dict[str, Any],
) -> None:
    settings = config.runtime["visualization"]
    plot = settings["model"]
    x, y = _risk_plot_series(model, "R_model", display_end)
    fig, ax = plt.subplots(figsize=tuple(plot["figure_size_inches"]))
    ax.step(x, y, where="post", color=plot["line_color"], linewidth=float(plot["line_width"]))
    ax.plot(
        [display_end], [0.0], linestyle="none", marker=plot["terminal_marker"],
        markersize=float(plot["terminal_marker_size"]), color=plot["line_color"],
    )
    maximum = float(model["R_model"].max())
    upper = max(1.0, maximum * (1.0 + float(plot["headroom_fraction"])))
    ax.set_xlim(float(model["time_seconds"].min()), display_end)
    ax.set_ylim(0.0, upper)
    ax.set_xlabel(plot["x_label"])
    ax.set_ylabel(plot["y_label"])
    ax.set_title(plot["title"])
    ax.grid(True, color=settings["grid_color"], alpha=float(settings["grid_alpha"]))
    ax.text(
        0.01, -0.18, _annotation(plot["annotation_template"], context),
        transform=ax.transAxes, ha="left", va="top", wrap=True,
        bbox={"facecolor": "white", "edgecolor": plot["line_color"], "alpha": 0.95},
    )
    _save_figure(fig, path, settings)


def _plot_positive_risk_vehicle_count(
    table: pd.DataFrame,
    path: Path,
    config: RiskPlotsConfig,
) -> None:
    settings = config.runtime["visualization"]
    plot = settings["positive_risk_vehicle_count"]
    fig, ax = plt.subplots(figsize=tuple(plot["figure_size_inches"]))
    ax.step(
        table["time_seconds"].to_numpy(float),
        table["positive_risk_vehicle_count"].to_numpy(int),
        where="post",
        color=plot["line_color"],
        linewidth=float(plot["line_width"]),
        linestyle=plot["line_style"],
        marker=plot["marker"],
        markersize=float(plot["marker_size"]),
    )
    peak = int(table["positive_risk_vehicle_count"].max())
    upper = max(1.0, math.ceil(peak * (1.0 + float(plot["headroom_fraction"]))))
    ax.set_xlim(float(table["time_seconds"].min()), float(table["time_seconds"].max()))
    ax.set_ylim(0.0, upper)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(plot["x_label"])
    ax.set_ylabel(plot["y_label"])
    ax.set_title(plot["title"])
    ax.grid(True, color=settings["grid_color"], alpha=float(settings["grid_alpha"]))
    ax.text(
        0.01,
        -0.18,
        plot["annotation_template"].format(
            interval_seconds=float(config.runtime["derivation"]["risk_interval_seconds"]),
            fleet_size=int(table["fleet_size"].max()),
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        wrap=True,
        bbox={"facecolor": "white", "edgecolor": plot["line_color"], "alpha": 0.95},
    )
    _save_figure(fig, path, settings)


def _acceptance_metrics(
    per_agent: pd.DataFrame,
    model: pd.DataFrame,
    arrival: pd.DataFrame,
    tripinfo: pd.DataFrame,
    evidence: dict[str, Any],
    config: RiskPlotsConfig,
) -> dict[str, Any]:
    settings = config.runtime
    acceptance = settings["acceptance"]
    tolerance = float(settings["derivation"]["numerical_tolerance"])
    zero = float(settings["derivation"]["risk_zero_tolerance"])
    peaks = per_agent.groupby("vehicle_id")["R_i"].max()
    nonzero_agents = int((peaks > zero).sum())
    nonzero_cells = int((per_agent["R_i"] > zero).sum())
    maximum_agent_risk = float(per_agent["R_i"].max())
    positive_model_times = int((model["R_model"] > zero).sum())
    peak_row = model.loc[model["R_model"].idxmax()]
    one_second_count = len(evidence["one_second_concurrent_exposure_agents"])
    ever_hazardous_count = len(evidence["ever_hazardous_edge_agents"])
    arrived = int(arrival["arrived_count"].iloc[-1])
    first_arrival = float(tripinfo["arrival_seconds"].min())
    last_arrival = float(tripinfo["arrival_seconds"].max())
    exact_checks = {
        "nonzero_agents": nonzero_agents == int(acceptance["nonzero_agents"]),
        "maximum_agent_risk": math.isclose(
            maximum_agent_risk, float(acceptance["maximum_agent_risk"]),
            abs_tol=tolerance, rel_tol=0.0,
        ),
        "positive_model_times": positive_model_times == int(acceptance["positive_model_times"]),
        "peak_model_risk": math.isclose(
            float(peak_row["R_model"]), float(acceptance["peak_model_risk"]),
            abs_tol=tolerance, rel_tol=0.0,
        ),
        "peak_model_time": math.isclose(
            float(peak_row["time_seconds"]), float(acceptance["peak_model_time_seconds"]),
            abs_tol=tolerance, rel_tol=0.0,
        ),
        "one_second_exposed_agents": one_second_count == int(acceptance["one_second_exposed_agents"]),
        "ever_hazardous_edge_agents": ever_hazardous_count == int(acceptance["ever_hazardous_edge_agents"]),
        "arrived_agents": arrived == int(acceptance["arrived_agents"]),
        "first_arrival": math.isclose(first_arrival, float(acceptance["first_arrival_seconds"]), abs_tol=tolerance),
        "last_arrival": math.isclose(last_arrival, float(acceptance["last_arrival_seconds"]), abs_tol=tolerance),
    }
    if not all(exact_checks.values()):
        raise ValueError(f"Risk-plots exact acceptance checks failed: {exact_checks}")
    minimum = int(acceptance["minimum_nonzero_cells"])
    maximum = int(acceptance["maximum_nonzero_cells"])
    if not minimum <= nonzero_cells <= maximum:
        raise ValueError(
            f"Non-zero agent/time cells {nonzero_cells} fall outside advisory safety range [{minimum},{maximum}]; "
            "the implementation will not adjust scientific semantics to fit the advisory target"
        )
    return {
        "exact_checks": exact_checks,
        "nonzero_agents": nonzero_agents,
        "nonzero_cells": nonzero_cells,
        "advisory_nonzero_cells": int(acceptance["advisory_nonzero_cells"]),
        "advisory_range": [minimum, maximum],
        "maximum_agent_risk": maximum_agent_risk,
        "positive_model_times": positive_model_times,
        "peak_model_risk": float(peak_row["R_model"]),
        "peak_model_time_seconds": float(peak_row["time_seconds"]),
        "one_second_concurrent_exposure_agents": one_second_count,
        "ever_hazardous_edge_agents": ever_hazardous_count,
        "arrived_agents": arrived,
        "arrival_percentage": 100.0 * arrived / int(settings["derivation"]["fleet_size"]),
        "first_arrival_seconds": first_arrival,
        "last_arrival_seconds": last_arrival,
    }


def _sample_agents(
    per_agent: pd.DataFrame, populations: Iterable[str], sample_size: int, seed: int, zero: float
) -> dict[str, list[str]]:
    all_agents = sorted(per_agent["vehicle_id"].astype(str).unique())
    peaks = per_agent.groupby("vehicle_id")["R_i"].max()
    exposed = sorted(str(vehicle_id) for vehicle_id, value in peaks.items() if float(value) > zero)
    pools = {"all_agents": all_agents, "risk_exposed_agents": exposed}
    result: dict[str, list[str]] = {}
    for population in populations:
        pool = pools[population]
        if sample_size > len(pool):
            raise ValueError(f"sample_size={sample_size} exceeds {population} population size {len(pool)}")
        rng = np.random.default_rng(seed)
        result[population] = sorted(rng.choice(pool, size=sample_size, replace=False).tolist())
    return result


def _report_text(
    config: RiskPlotsConfig,
    metrics: dict[str, Any],
    samples: dict[str, list[str]],
    sample_peaks: dict[str, dict[str, float]],
    evidence: dict[str, Any],
    source_before: dict[str, Any],
    source_after: dict[str, Any],
    paths: dict[str, str],
    spatial: dict[str, Any],
    preservation: dict[str, Any],
) -> str:
    classifications = ", ".join(config.runtime["execution"]["result_classification"])
    sample_sections = []
    for population, agents in samples.items():
        values = ", ".join(f"`{agent}` ({sample_peaks[population][agent]:.10g})" for agent in agents)
        sample_sections.append(f"- **{population}:** {values}")
    sensitivity = ", ".join(
        f"{interval} s={count} agents"
        for interval, count in evidence["interval_sensitivity_agent_counts"].items()
    )
    positive_risk_outputs = ""
    if "positive_risk_count" in paths:
        positive_risk_outputs = f"""
- Positive-risk vehicle-count table: `{paths['positive_risk_count']}`
- Positive-risk vehicle-count figure: `{paths['positive_risk_count_figure']}`

The positive-risk count is the number of unique vehicles satisfying `R_i(t)>risk_zero_tolerance` at each configured 60-second sample. It is a point-in-time count, not cumulative. It is also distinct from the one-second remaining-route-risk evidence: the latter asks whether a vehicle currently has hazard ahead on its committed route, while this figure uses accumulated sampled `R_i(t)`.
"""
    endpoint_output = ""
    endpoint_summary = spatial.get("origin_destination_overlay")
    if "spatial_origin_destination_figure" in paths and endpoint_summary is not None:
        point_lines = "\n".join(
            f"- `{point['role']}` `{point['location_id']}` on `{point['edge_id']}` at "
            f"({float(point['x']):.6f}, {float(point['y']):.6f})."
            for point in endpoint_summary["points"]
        )
        endpoint_output = f"""

## Origin and shelter destination overlay

- Additional figure: `{paths['spatial_origin_destination_figure']}`
- Coordinate definition: {endpoint_summary['coordinate_definition']}.
- Origin markers: configured cyan circles; destination markers: configured bright-green stars.
- Marker colours are categorical. They do not encode time, risk, fire state, demand, or capacity.
- Endpoint source hashes: `{json.dumps(endpoint_summary['source_hashes'], sort_keys=True)}`.

{point_lines}
"""
    return f"""# {config.runtime['reporting']['title']}

## Status and provenance

- Status: technically verified derived sidecar; no SUMO or fire simulation was run.
- Classification: {classifications}.
- Source v009 tree before: `{source_before['tree_sha256']}` ({source_before['file_count']} files, {source_before['total_bytes']} bytes).
- Source v009 tree after: `{source_after['tree_sha256']}` ({source_after['file_count']} files, {source_after['total_bytes']} bytes).
- Source byte identity: **{source_before == source_after}**.
- Contract lambda: `{config.lambda_value}` from `shared_contract.hazard.edge_survival_and_risk.lambda`.
- Contract epsilon: `{config.epsilon}` from `shared_contract.hazard.route_survival_and_risk.epsilon`.

## Semantics

`R_i(t)` is evaluated on the configured 60-second observation grid using previous-snapshot edge survival and log-space accumulation. A repeated current normal edge replaces its current factor with the newly observed survival. When a vehicle leaves a normal edge, its last sampled log-survival is frozen. Internal SUMO junction edges have no rows in the hazard table: the departed normal edge is frozen, the internal identifier is retained, `S_e_current` is null, and only frozen normal-edge exposure contributes. No junction hazard is invented.

Activity follows the authoritative pre-movement active-vehicle table. A vehicle present at its tripinfo arrival timestamp remains active for that observation and is inactive afterward. Arrival counts use completed tripinfo events at or before each timestamp.

## Acceptance results

- Agents with non-zero sampled `R_i`: **{metrics['nonzero_agents']}**.
- Non-zero agent/time cells: **{metrics['nonzero_cells']}**; advisory target {metrics['advisory_nonzero_cells']}, safety range {metrics['advisory_range']}.
- Maximum `R_i`: **{metrics['maximum_agent_risk']:.12f}**.
- Positive `R_model` timestamps: **{metrics['positive_model_times']} of 31**.
- Peak `R_model`: **{metrics['peak_model_risk']:.12f} at {metrics['peak_model_time_seconds']:.0f} s**.
- Arrivals: **{metrics['arrived_agents']}/100 ({metrics['arrival_percentage']:.1f}%)**; first {metrics['first_arrival_seconds']:.0f} s, last {metrics['last_arrival_seconds']:.0f} s.
- All exact gates passed: **{all(metrics['exact_checks'].values())}**.

The 40-versus-40 sampled occupancy comparison is an internal consistency check only. It is not validation of complete exposure capture. At one-second resolution, **{metrics['one_second_concurrent_exposure_agents']} agents** occupy an edge hazardous under the active previous-snapshot state; **{metrics['ever_hazardous_edge_agents']} agents** traverse an edge that is hazardous at some time in the run. The configured interval sensitivity is: {sensitivity}. The sampled count first converges to 66 at 3 seconds and remains 66 at 2 and 1 seconds for the inspected intervals.

## Reproducible samples

Values in parentheses are each sampled agent's peak `R_i`.

{chr(10).join(sample_sections)}

## Outputs

- Per-agent table: `{paths['per_agent']}`
- Model table: `{paths['model']}`
- Arrival table: `{paths['arrival']}`
- Arrival figure: `{paths['arrival_figure']}`
- All-agent sample figure: `{paths['all_agents_figure']}`
- Risk-exposed sample figure: `{paths['risk_exposed_agents_figure']}`
- Model figure: `{paths['model_figure']}`
{positive_risk_outputs}

The derived risk, arrival, and positive-risk count tables include `active_fire_snapshot_time_seconds` and `interacting_fronts`. The sampled-agent and model-risk figures use a plot-only zero endpoint at the configured display end; scientific tables retain all 31 timestamps through 1800 seconds. Twenty-six vehicles remain active after the last arrival, so the default 1181-second display is intentionally truncated and must not be interpreted as the end of traffic or exposure.

## Spatial fire-and-route diagnostic

The spatial figure uses filled 50 m road cells as its neutral network background. Burning cells and driven road links use one `{spatial['shared_colormap']}` colormap, one normalization `{spatial['shared_normalization']}`, and one seconds colorbar. Observed fire ignition times are `{spatial['time_ranges_seconds']['fire_ignition']}` seconds; raw one-second route-presence times are `{spatial['time_ranges_seconds']['route_presence']}` seconds; per-link median vehicle-presence times are `{spatial['time_ranges_seconds']['route_median_occupancy']}` seconds.

- Road-intersecting 50 m cells drawn: **{spatial['road_grid_cell_count']}** from {spatial['road_cell_metadata']['mapped_routeable_edges']} passenger-routeable normal edges and {spatial['road_cell_metadata']['mapping_rows']} validated edge-cell intersections.
- The mapping records the handoff fire-grid artifact logical hash `{spatial['road_cell_metadata']['mapping_grid_artifact_logical_sha256']}`; grid rows record the distinct grid-geometry logical hash `{spatial['road_cell_metadata']['fire_grid_logical_sha256']}`. Both identities were verified against their named handoff-manifest fields before joining by `cell_id`.
- Burning cells drawn: **{spatial['burning_cells']}** of {spatial['total_grid_cells']}.
- Cells whose first burn has `interacting_fronts=true`: **{spatial['interacting_ignition_cells']}**.
- Vehicle traces: **{spatial['vehicle_traces']}**; normal-edge runs: **{spatial['trace_runs']}**; distinct normal edges: **{spatial['distinct_normal_edges']}**.
- Internal observations excluded: **{spatial['excluded_internal_observations']}**.
- Network hash comparison: **{spatial['network_hash_match']}** (`{spatial['network_hash']}`).
- Route links: **{spatial['route_edge_count']}**; distinct-vehicle overlap range: **{spatial['distinct_vehicle_count_range']}**; configured line-width range: **{spatial['route_line_width_range_points']} points**.
- The line colour is the median of all one-second vehicle-presence observations on that link: half of those observations occur earlier and half later. Line width is linear in the number of distinct vehicles using the link.
- No endpoint bridges are drawn or invented. Every rendered route line is the real polyline of one normal edge from the sealed SUMO network.
- Styles: road-cell alpha/edge width `{spatial['styles']['road_cells']['alpha']}`/`{spatial['styles']['road_cells']['edge_linewidth']}`, fire alpha/edge width `{spatial['styles']['fire_cells']['alpha']}`/`{spatial['styles']['fire_cells']['edge_linewidth']}`, route alpha/width range `{spatial['styles']['route_lines']['alpha']}`/`[{spatial['styles']['route_lines']['minimum_width_points']}, {spatial['styles']['route_lines']['maximum_width_points']}]` points.
- Existing v010 preservation gate: **{preservation['passed']}**. PNG comparisons: `{json.dumps(preservation['png_comparisons'], sort_keys=True)}`.

All 100 vehicles contribute without sampling. Grey squares have no artificial time value. Coloured squares report first fire ignition, coloured link lines report typical physical occupancy time, and both time layers use the same scale. Similar colours at the same location support a timing comparison but do not, by themselves, establish exposure or causal avoidance.

Spatial artifacts:

- Ignition table: `{paths['cell_ignition']}`
- Driven trace table: `{paths['driven_trace']}`
- Route-edge activity table: `{paths['route_edge_activity']}`
- Spatial figure: `{paths['spatial_figure']}`
{endpoint_output}
"""


def generate_risk_plots_sidecar(config: RiskPlotsConfig) -> dict[str, Any]:
    runtime = config.runtime
    existing_output = config.output_root.exists()
    if existing_output and not runtime["execution"]["transactional_rebuild_existing"]:
        raise FileExistsError(f"Risk-plots output already exists: {config.output_root}")
    existing_before: dict[str, str] = {}
    if existing_output:
        checksum_path = config.output_root / runtime["reporting"]["checksums"]
        if not checksum_path.is_file():
            raise FileNotFoundError(f"Existing v010 checksum manifest is missing: {checksum_path}")
        _verify_sha256sums(config.output_root, checksum_path)
        existing_before, _, _ = _hash_tree(config.output_root)

    source_hashes_before, source_bytes_before, source_tree_before = _hash_tree(config.source_root)
    source_before = {
        "file_count": len(source_hashes_before),
        "total_bytes": source_bytes_before,
        "tree_sha256": source_tree_before,
    }
    expected_tree = config.runtime["source_run"]["expected_tree_sha256"]
    if source_tree_before != expected_tree:
        raise ValueError(f"Source tree changed before generation: expected {expected_tree}, got {source_tree_before}")
    handoff_hashes_before, handoff_bytes_before, handoff_tree_before = _hash_tree(config.handoff_root)
    handoff_before = {
        "file_count": len(handoff_hashes_before),
        "total_bytes": handoff_bytes_before,
        "tree_sha256": handoff_tree_before,
    }
    if handoff_tree_before != runtime["spatial_sources"]["expected_tree_sha256"]:
        raise ValueError("Sealed handoff changed before spatial-map generation")

    active = pd.read_parquet(config.source_paths["active_vehicle_table"])
    hazard = pd.read_parquet(config.source_paths["hazard_table"])
    if "lambda" in hazard.columns:
        observed_lambda = hazard["lambda"].astype(float).unique()
        if len(observed_lambda) != 1 or not math.isclose(
            float(observed_lambda[0]), config.lambda_value,
            abs_tol=float(config.runtime["derivation"]["numerical_tolerance"]), rel_tol=0.0,
        ):
            raise ValueError("Hazard-table lambda differs from the common contract")

    derivation = runtime["derivation"]
    clock = config.common["shared_contract"]["clock"]
    start = float(clock["simulation_start_seconds"])
    end = float(clock["simulation_end_seconds"])
    sumo_step = float(clock["sumo_step_seconds"])
    tolerance = float(derivation["numerical_tolerance"])
    fleet_size = int(derivation["fleet_size"])
    per_agent, per_agent_evidence = derive_per_agent_risk(
        active,
        hazard,
        fleet_size=fleet_size,
        start_seconds=start,
        end_seconds=end,
        sumo_step_seconds=sumo_step,
        interval_seconds=float(derivation["risk_interval_seconds"]),
        epsilon=config.epsilon,
        internal_edge_prefix=derivation["internal_edge_prefix"],
        tolerance=tolerance,
    )
    lookup = _HazardLookup(hazard, config.epsilon)
    tripinfo = _parse_tripinfo(config.source_paths["tripinfo"], per_agent_evidence["fleet"])
    model = derive_model_risk(per_agent)
    positive_risk_count: pd.DataFrame | None = None
    positive_risk_plot = runtime["visualization"].get("positive_risk_vehicle_count")
    if positive_risk_plot is not None:
        if "positive_risk_count" not in runtime["tables"]:
            raise ValueError(
                "visualization.positive_risk_vehicle_count requires tables.positive_risk_count"
            )
        positive_risk_count = derive_positive_risk_vehicle_count(
            per_agent,
            risk_zero_tolerance=float(derivation["risk_zero_tolerance"]),
        )
    arrival = derive_arrivals(
        tripinfo,
        lookup,
        start_seconds=start,
        end_seconds=end,
        interval_seconds=float(derivation["arrival_interval_seconds"]),
        fleet_size=fleet_size,
        tolerance=tolerance,
    )
    evidence = exposure_evidence(
        active,
        lookup,
        start_seconds=start,
        internal_edge_prefix=derivation["internal_edge_prefix"],
        intervals=derivation["interval_sensitivity_seconds"],
        tolerance=tolerance,
    )
    metrics = _acceptance_metrics(per_agent, model, arrival, tripinfo, evidence, config)
    observed_interacting = sorted(
        float(time_value)
        for time_value, snapshot in lookup.snapshots.items()
        if bool(snapshot["interacting_fronts"].iloc[0])
    )
    expected_interacting = [
        float(value) for value in runtime["acceptance"]["interacting_fronts_snapshot_times_seconds"]
    ]
    if observed_interacting != expected_interacting:
        raise ValueError(
            f"Interacting-front snapshot times differ: expected={expected_interacting}, "
            f"observed={observed_interacting}"
        )

    samples = _sample_agents(
        per_agent,
        derivation["sampling_populations"],
        int(derivation["sample_size"]),
        int(derivation["sampling_seed"]),
        float(derivation["risk_zero_tolerance"]),
    )
    peaks = per_agent.groupby("vehicle_id")["R_i"].max().to_dict()
    sample_peaks = {
        population: {vehicle_id: float(peaks[vehicle_id]) for vehicle_id in agents}
        for population, agents in samples.items()
    }
    display_end = _display_end(config, tripinfo)
    context = {
        "arrival_percentage": metrics["arrival_percentage"],
        "arrived_count": metrics["arrived_agents"],
        "fleet_size": fleet_size,
        "non_arrived_count": fleet_size - metrics["arrived_agents"],
        "simulation_end_seconds": end,
        "last_arrival_seconds": metrics["last_arrival_seconds"],
        "display_end_seconds": display_end,
        "sampled_exposed_count": metrics["nonzero_agents"],
        "one_second_exposed_count": metrics["one_second_concurrent_exposure_agents"],
    }

    spatial_settings = runtime["visualization"]["spatial_fire_route_map"]
    spatial_acceptance = runtime["acceptance"]["spatial_map"]
    fire_states = pd.read_parquet(config.handoff_paths["fire_state_table"])
    fire_grid = pd.read_parquet(config.handoff_paths["fire_grid_table"])
    ignition, ignition_render, ignition_metadata = derive_cell_ignition(
        fire_states,
        fire_grid,
        expected_network_hash=runtime["spatial_sources"]["network_sha256"],
        expected_coordinate_frame=spatial_acceptance["coordinate_frame"],
        expected_extent=[float(value) for value in spatial_acceptance["grid_extent"]],
        tolerance=tolerance,
    )
    trace, trace_metadata = derive_driven_trace(
        active,
        internal_edge_prefix=derivation["internal_edge_prefix"],
        include_internal_edges=bool(spatial_settings["include_internal_edges"]),
    )
    network = sumolib.net.readNet(str(config.handoff_paths["network"]), withInternal=True)
    endpoint_overlay = spatial_settings.get("origin_destination_overlay")
    endpoint_points: pd.DataFrame | None = None
    endpoint_metadata: dict[str, Any] | None = None
    if endpoint_overlay is not None:
        origins = pd.read_parquet(config.source_paths["origin_table"])
        destinations = pd.read_parquet(config.source_paths["destination_table"])
        endpoint_points, endpoint_metadata = derive_origin_destination_points(
            origins,
            destinations,
            network,
        )
    route_activity, route_activity_metadata = derive_route_edge_activity(
        active,
        trace,
        network,
        internal_edge_prefix=derivation["internal_edge_prefix"],
    )
    edge_cell_mapping = pd.read_parquet(config.source_paths["edge_cell_mapping_table"])
    handoff_manifest = json.loads(config.handoff_paths["manifest"].read_text(encoding="utf-8"))
    fire_grid_artifacts = [
        item for item in handoff_manifest.get("artifacts", [])
        if item.get("path") == runtime["spatial_sources"]["fire_grid_table"]
    ]
    if len(fire_grid_artifacts) != 1:
        raise ValueError("Handoff manifest must identify exactly one configured fire-grid artifact")
    expected_mapping_grid_hash = str(fire_grid_artifacts[0]["logical_content_sha256"])
    expected_fire_grid_hash = str(handoff_manifest["grid_logical_sha256"])
    road_cells, road_cell_metadata = derive_road_grid_cells(
        edge_cell_mapping,
        fire_grid,
        expected_network_hash=runtime["spatial_sources"]["network_sha256"],
        expected_mapping_grid_hash=expected_mapping_grid_hash,
        expected_fire_grid_hash=expected_fire_grid_hash,
        expected_cell_size_m=float(spatial_settings["road_cells"]["cell_size_m"]),
        tolerance=tolerance,
    )

    spatial_observed = {
        "total_grid_cells": ignition_metadata["total_grid_cells"],
        "burning_cells": ignition_metadata["burning_cells"],
        "interacting_ignition_cells": ignition_metadata["interacting_ignition_cells"],
        "vehicle_traces": trace_metadata["vehicle_traces"],
        "distinct_normal_edges": trace_metadata["distinct_edges"],
        "excluded_internal_observations": trace_metadata["excluded_internal_observations"],
        "trace_runs": trace_metadata["trace_runs"],
        "fire_ignition_time_range_seconds": ignition_metadata["ignition_time_range_seconds"],
        "trace_entry_time_range_seconds": trace_metadata["entry_time_range_seconds"],
        "route_presence_time_range_seconds": route_activity_metadata["presence_time_range_seconds"],
        "route_median_occupancy_time_range_seconds": route_activity_metadata[
            "median_occupancy_time_range_seconds"
        ],
        "road_grid_cells": road_cell_metadata["road_grid_cells"],
        "maximum_distinct_vehicles_per_edge": route_activity_metadata[
            "distinct_vehicle_count_range"
        ][1],
    }
    for name, observed in spatial_observed.items():
        expected = spatial_acceptance[name]
        if isinstance(expected, list):
            if not np.allclose(observed, expected, atol=tolerance, rtol=0.0):
                raise ValueError(f"Spatial acceptance mismatch for {name}: expected={expected}, observed={observed}")
        elif observed != expected:
            raise ValueError(f"Spatial acceptance mismatch for {name}: expected={expected}, observed={observed}")

    config.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = config.output_root.with_name(
        f"{config.output_root.name}-staging-{uuid.uuid4().hex}"
    )
    temporary_root.mkdir(parents=False, exist_ok=False)
    committed = False
    output_backup: Path | None = None
    report_backup: Path | None = None
    repository_report_temp: Path | None = None
    try:
        table_directory = temporary_root / runtime["tables"]["directory"]
        figure_directory = temporary_root / runtime["visualization"]["directory"]
        report_directory = temporary_root / runtime["reporting"]["reports_directory"]
        paths = {
            "per_agent": table_directory / runtime["tables"]["per_agent"],
            "model": table_directory / runtime["tables"]["model"],
            "arrival": table_directory / runtime["tables"]["arrival"],
            "arrival_figure": figure_directory / runtime["visualization"]["arrival"]["filename"],
            "all_agents_figure": figure_directory / runtime["visualization"]["per_agent"]["filenames"]["all_agents"],
            "risk_exposed_agents_figure": figure_directory / runtime["visualization"]["per_agent"]["filenames"]["risk_exposed_agents"],
            "model_figure": figure_directory / runtime["visualization"]["model"]["filename"],
            "cell_ignition": table_directory / runtime["tables"]["cell_ignition"],
            "driven_trace": table_directory / runtime["tables"]["driven_trace"],
            "route_edge_activity": table_directory / runtime["tables"]["route_edge_activity"],
            "spatial_figure": figure_directory / spatial_settings["filename"],
        }
        if endpoint_overlay is not None:
            paths["spatial_origin_destination_figure"] = (
                figure_directory / endpoint_overlay["filename"]
            )
        if positive_risk_count is not None:
            paths["positive_risk_count"] = (
                table_directory / runtime["tables"]["positive_risk_count"]
            )
            paths["positive_risk_count_figure"] = (
                figure_directory / positive_risk_plot["filename"]
            )
        if len({path.resolve() for path in paths.values()}) != len(paths):
            raise ValueError("Configured risk-plots output filenames are not unique")
        write_table(per_agent, paths["per_agent"])
        write_table(model, paths["model"])
        write_table(arrival, paths["arrival"])
        write_table(ignition, paths["cell_ignition"])
        write_table(trace, paths["driven_trace"])
        write_table(route_activity, paths["route_edge_activity"])
        if positive_risk_count is not None:
            write_table(positive_risk_count, paths["positive_risk_count"])
        plt.rcParams["font.family"] = runtime["visualization"]["font_family"]
        _plot_arrivals(arrival, paths["arrival_figure"], config, context)
        for population in derivation["sampling_populations"]:
            _plot_agent_sample(
                per_agent,
                samples[population],
                population,
                display_end,
                paths[f"{population}_figure"],
                config,
                context,
            )
        _plot_model(model, display_end, paths["model_figure"], config, context)
        if positive_risk_count is not None:
            _plot_positive_risk_vehicle_count(
                positive_risk_count,
                paths["positive_risk_count_figure"],
                config,
            )
        spatial_render = render_spatial_fire_route_map(
            ignition_render,
            road_cells,
            route_activity,
            network,
            paths["spatial_figure"],
            spatial_settings,
            runtime["visualization"]["rendering_metadata"],
            int(runtime["visualization"]["dpi"]),
        )
        endpoint_render: dict[str, Any] | None = None
        if endpoint_overlay is not None and endpoint_points is not None:
            endpoint_render = render_spatial_fire_route_map(
                ignition_render,
                road_cells,
                route_activity,
                network,
                paths["spatial_origin_destination_figure"],
                spatial_settings,
                runtime["visualization"]["rendering_metadata"],
                int(runtime["visualization"]["dpi"]),
                endpoint_points=endpoint_points,
                endpoint_settings=endpoint_overlay,
            )
        preservation = {"passed": True, "png_comparisons": {}, "table_comparisons": {}}
        if existing_output:
            for relative in runtime["execution"]["preserve_byte_identical"]:
                old_path = config.output_root / relative
                new_path = temporary_root / relative
                if not old_path.is_file() or not new_path.is_file():
                    raise FileNotFoundError(f"Preservation artifact missing: old={old_path}, new={new_path}")
                if old_path.suffix.lower() == ".png":
                    comparison = compare_png(old_path, new_path)
                    preservation["png_comparisons"][relative] = comparison
                    if not comparison["idat_identical"]:
                        raise ValueError(
                            f"Existing PNG image data changed for {relative}; comparison={comparison}"
                        )
                    if comparison["metadata_only"]:
                        shutil.copyfile(old_path, new_path)
                else:
                    identical = old_path.read_bytes() == new_path.read_bytes()
                    preservation["table_comparisons"][relative] = {"byte_identical": identical}
                    if not identical:
                        raise ValueError(f"Existing scientific table changed during rebuild: {relative}")

        source_hashes_after, source_bytes_after, source_tree_after = _hash_tree(config.source_root)
        source_after = {
            "file_count": len(source_hashes_after),
            "total_bytes": source_bytes_after,
            "tree_sha256": source_tree_after,
        }
        if source_hashes_before != source_hashes_after or source_before != source_after:
            raise ValueError("Immutable v009 source run changed during risk-plots generation")
        handoff_hashes_after, handoff_bytes_after, handoff_tree_after = _hash_tree(config.handoff_root)
        handoff_after = {
            "file_count": len(handoff_hashes_after),
            "total_bytes": handoff_bytes_after,
            "tree_sha256": handoff_tree_after,
        }
        if handoff_hashes_before != handoff_hashes_after or handoff_before != handoff_after:
            raise ValueError("Immutable sealed handoff changed during spatial-map generation")

        spatial_summary = {
            **spatial_render,
            **spatial_observed,
            "total_grid_cells": ignition_metadata["total_grid_cells"],
            "burning_cells": ignition_metadata["burning_cells"],
            "interacting_ignition_cells": ignition_metadata["interacting_ignition_cells"],
            "distinct_normal_edges": trace_metadata["distinct_edges"],
            "network_hash": ignition_metadata["network_hash"],
            "network_hash_match": ignition_metadata["network_hash"] == runtime["spatial_sources"]["network_sha256"],
            "road_cell_source": "immutable_v009_independently_reconstructed_edge_cell_mapping",
            "road_cell_metadata": road_cell_metadata,
            "route_activity_metadata": route_activity_metadata,
            "connector_rendering": "none_aggregated_normal_edges_use_only_sealed_network_geometry",
            "styles": {
                "road_cells": spatial_settings["road_cells"],
                "fire_cells": spatial_settings["fire_cells"],
                "route_lines": spatial_settings["route_lines"],
            },
            "trace_source": spatial_settings["trace_source"],
            "trace_time_semantics": route_activity_metadata["time_semantics"],
        }
        if endpoint_metadata is not None and endpoint_render is not None:
            spatial_summary["origin_destination_overlay"] = {
                **endpoint_metadata,
                "render": endpoint_render["endpoint_overlay"],
                "source_hashes": {
                    "origins": sha256_file(config.source_paths["origin_table"]),
                    "destinations": sha256_file(config.source_paths["destination_table"]),
                },
                "styles": {
                    "origins": endpoint_overlay["origins"],
                    "destinations": endpoint_overlay["destinations"],
                },
                "categorical_colours_not_time_encoded": True,
            }

        final_paths = {
            name: str(config.output_root / path.relative_to(temporary_root))
            for name, path in paths.items()
        }
        report_text = _report_text(
            config, metrics, samples, sample_peaks, evidence,
            source_before, source_after, final_paths, spatial_summary, preservation,
        )
        run_report = report_directory / runtime["reporting"]["run_report"]
        run_report.parent.mkdir(parents=True, exist_ok=True)
        run_report.write_text(report_text, encoding="utf-8")

        artifact_paths = {name: path for name, path in paths.items()}
        artifact_paths["run_report"] = run_report
        artifact_hashes = {
            name: {
                "path": str(config.output_root / path.relative_to(temporary_root)),
                "sha256": sha256_file(path),
            }
            for name, path in artifact_paths.items()
        }
        manifest = {
            "run_id": runtime["execution"]["run_id"],
            "runtime_contract_version": runtime["runtime_contract_version"],
            "runtime_config": {"path": str(config.runtime_path), "sha256": config.runtime_sha256},
            "common_contract": {"path": str(config.common_path), "sha256": config.common_sha256},
            "source_run": {
                "path": str(config.source_root),
                "resolved_config_sha256": runtime["source_run"]["expected_resolved_config_sha256"],
                "before": source_before,
                "after": source_after,
                "byte_identical": True,
                "tables": {
                    name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in config.source_paths.items()
                },
            },
            "sealed_handoff": {
                "path": str(config.handoff_root),
                "before": handoff_before,
                "after": handoff_after,
                "byte_identical": True,
                "files": {
                    name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in config.handoff_paths.items()
                },
            },
            "rendering_environment": rendering_library_versions(),
            "preservation": preservation,
            "spatial_map": spatial_summary,
            "formula": {
                "agent_risk": "R_i(t)=1-exp(frozen_log_sum+log(clip(S_e_current(t),epsilon,1))) for an active normal edge; inactive R_i=0",
                "model_risk": "R_model(t)=sum_i R_i(t)",
                "positive_risk_vehicle_count": "N_positive(t)=count of unique vehicles with R_i(t)>risk_zero_tolerance at each risk-sampling timestamp; not cumulative",
                "hazard_time_lookup": "previous_snapshot",
                "interpolation": "none",
                "lambda": config.lambda_value,
                "lambda_source": "shared_contract.hazard.edge_survival_and_risk.lambda",
                "epsilon": config.epsilon,
                "epsilon_source": "shared_contract.hazard.route_survival_and_risk.epsilon",
                "internal_junction": "departed normal edge frozen; current internal S_e_current null; frozen edges only",
                "activity_boundary": derivation["activity_boundary"],
            },
            "acceptance": metrics,
            "exposure_evidence": {
                "sampled_internal_consistency_agents": len(per_agent_evidence["sampled_hazard_agents"]),
                "one_second_concurrent_exposure_agents": len(evidence["one_second_concurrent_exposure_agents"]),
                "ever_hazardous_edge_agents": len(evidence["ever_hazardous_edge_agents"]),
                "interval_sensitivity_agent_counts": evidence["interval_sensitivity_agent_counts"],
            },
            "samples": {
                population: {
                    "population": population,
                    "sample_size": int(derivation["sample_size"]),
                    "seed": int(derivation["sampling_seed"]),
                    "agents": agents,
                    "peak_R_i": sample_peaks[population],
                }
                for population, agents in samples.items()
            },
            "interacting_fronts_snapshot_times_seconds": observed_interacting,
            "classifications": runtime["execution"]["result_classification"],
            "display": {
                "x_axis_anchor": derivation["x_axis_anchor"],
                "x_axis_offset_seconds": derivation["x_axis_offset_seconds"],
                "display_end_seconds": display_end,
                "plot_only_zero_endpoint": True,
            },
            "artifacts": artifact_hashes,
        }
        manifest_path = temporary_root / runtime["reporting"]["manifest"]
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        checksum_path = temporary_root / runtime["reporting"]["checksums"]
        checksum_lines = []
        for path in sorted(item for item in temporary_root.rglob("*") if item.is_file() and item != checksum_path):
            checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(temporary_root).as_posix()}\n")
        checksum_path.write_text("".join(checksum_lines), encoding="utf-8")
        for line in checksum_lines:
            expected, relative = line.rstrip("\n").split("  ", 1)
            actual = sha256_file(temporary_root / relative)
            if actual != expected:
                raise RuntimeError(f"Generated checksum verification failed for {relative}")

        repository_report = _resolve(config.repository_root, runtime["reporting"]["repository_report"])
        repository_report.parent.mkdir(parents=True, exist_ok=True)
        repository_report_temp = repository_report.with_name(
            f"{repository_report.name}.staging-{uuid.uuid4().hex}"
        )
        repository_report_temp.write_text(report_text, encoding="utf-8")

        output_backup = config.output_root.with_name(
            f"{config.output_root.name}-backup-{uuid.uuid4().hex}"
        )
        if config.output_root.exists():
            config.output_root.replace(output_backup)
        try:
            temporary_root.replace(config.output_root)
            if repository_report.exists():
                report_backup = repository_report.with_name(
                    f"{repository_report.name}.backup-{uuid.uuid4().hex}"
                )
                repository_report.replace(report_backup)
            repository_report_temp.replace(repository_report)
            repository_report_temp = None
            _verify_sha256sums(
                config.output_root,
                config.output_root / runtime["reporting"]["checksums"],
            )
            committed = True
        except Exception:
            if config.output_root.exists():
                shutil.rmtree(config.output_root)
            if output_backup.exists():
                output_backup.replace(config.output_root)
            if repository_report.exists() and report_backup is not None:
                repository_report.unlink()
            if report_backup is not None and report_backup.exists():
                report_backup.replace(repository_report)
            raise
        if output_backup.exists():
            shutil.rmtree(output_backup)
        if report_backup is not None and report_backup.exists():
            report_backup.unlink()
    finally:
        if not committed and temporary_root.exists():
            shutil.rmtree(temporary_root)
        if repository_report_temp is not None and repository_report_temp.exists():
            repository_report_temp.unlink()

    final_hashes, final_bytes, final_tree = _hash_tree(config.output_root)
    return {
        "status": "passed",
        "output_root": str(config.output_root),
        "report": str(repository_report),
        "manifest": str(config.output_root / runtime["reporting"]["manifest"]),
        "checksums": str(config.output_root / runtime["reporting"]["checksums"]),
        "source_v009_byte_identical": True,
        "sealed_handoff_byte_identical": True,
        "v010_tree": {"file_count": len(final_hashes), "total_bytes": final_bytes, "tree_sha256": final_tree},
        "spatial_map": spatial_summary,
        "preservation": preservation,
        "acceptance": metrics,
        "samples": samples,
        "sample_peaks": sample_peaks,
        "exposure_evidence": {
            "one_second_concurrent_exposure_agents": len(evidence["one_second_concurrent_exposure_agents"]),
            "ever_hazardous_edge_agents": len(evidence["ever_hazardous_edge_agents"]),
            "interval_sensitivity_agent_counts": evidence["interval_sensitivity_agent_counts"],
        },
    }
