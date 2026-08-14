from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sumolib

from .config import ResolvedIntegrationConfig, sha256_file


@dataclass(frozen=True)
class ValidatedHandoff:
    directory: Path
    hashes: dict[str, str]
    handoff_manifest: dict[str, Any]
    state_mapping_manifest: dict[str, Any]
    grid: pd.DataFrame
    fire_cells: pd.DataFrame
    reference_mapping: pd.DataFrame
    reference_hazard: pd.DataFrame
    validation: dict[str, Any]


def hash_tree(directory: Path, excluded: set[str] | None = None) -> dict[str, str]:
    excluded = excluded or set()
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.relative_to(directory).as_posix() not in excluded
    }


def _parse_checksums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ").replace("\\", "/")
        if relative in values:
            raise ValueError(f"Duplicate checksum entry: {relative}")
        values[relative] = digest.lower()
    return values


def _routeable_counts(net, vehicle_class: str) -> tuple[int, int, int]:
    normal = [edge for edge in net.getEdges(withInternal=False)]
    internal = [edge for edge in net.getEdges(withInternal=True) if edge.getFunction() == "internal"]
    allowed = [edge for edge in normal if any(lane.allows(vehicle_class) for lane in edge.getLanes())]
    return len(normal), len(internal), len(allowed)


def manifest_contract_key(manifest: dict[str, Any]) -> str:
    """Select the common-contract identity strictly from the manifest version."""
    version = str(manifest.get("schema_version", ""))
    key = (
        "common_contract_raw_sha256"
        if version == "manhattan-fire-handoff-interacting-fronts-v007"
        else "common_config_raw_sha256"
    )
    if key not in manifest:
        raise ValueError(
            f"Handoff manifest {version!r} requires {key!r}; "
            f"available common-hash keys={sorted(name for name in manifest if 'common' in name and 'sha256' in name)}"
        )
    return key


def validate_handoff(config: ResolvedIntegrationConfig) -> ValidatedHandoff:
    runtime_handoff = config.runtime["handoff"]
    directory = config.handoff_directory
    checksum_path = directory / runtime_handoff["checksum_manifest"]
    declared = _parse_checksums(checksum_path)
    actual = hash_tree(directory, excluded={runtime_handoff["checksum_manifest"]})
    missing = sorted(set(declared) - set(actual))
    extra = sorted(set(actual) - set(declared))
    mismatched = sorted(name for name in declared.keys() & actual.keys() if declared[name] != actual[name])
    if missing or extra or mismatched:
        raise ValueError(f"Sealed handoff checksum failure: missing={missing}, extra={extra}, mismatched={mismatched}")
    manifest = json.loads((directory / runtime_handoff["handoff_manifest"]).read_text(encoding="utf-8"))
    state_manifest = json.loads((directory / runtime_handoff["state_mapping_manifest"]).read_text(encoding="utf-8"))
    manifest_version = str(manifest.get("schema_version", ""))
    contract_key = manifest_contract_key(manifest)
    if manifest[contract_key] != config.common_sha256:
        raise ValueError("Handoff manifest common-config hash does not match the active common contract")
    if state_manifest.get("common_contract_raw_sha256", state_manifest.get("common_config_raw_sha256")) != config.common_sha256:
        raise ValueError("State-mapping manifest common-contract hash does not match the active contract")
    shared = config.shared
    network = shared["network"]
    if sha256_file(config.network_path) != network["sha256"] or config.network_path.stat().st_size != int(network["size_bytes"]):
        raise ValueError("SUMO network hash or size differs from the common contract")
    if manifest["network"]["sha256"] != network["sha256"] or int(manifest["network"]["size_bytes"]) != int(network["size_bytes"]):
        raise ValueError("Handoff network provenance differs from the common contract")
    bundled_network = directory / runtime_handoff["provenance_network"]
    if sha256_file(bundled_network) != network["sha256"] or bundled_network.stat().st_size != int(network["size_bytes"]):
        raise ValueError("Bundled provenance network differs from the local authoritative test network")
    bundled_contract = directory / runtime_handoff["provenance_common_contract"]
    if sha256_file(bundled_contract) != config.common_sha256:
        raise ValueError("Bundled provenance common contract differs from the active canonical contract")
    artifact_manifest = {item["path"]: item for item in manifest["artifacts"]}
    for name, item in artifact_manifest.items():
        if name not in actual or actual[name] != item["sha256"] or (directory / name).stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"Handoff artifact metadata mismatch for {name}")
    grid = pd.read_parquet(directory / runtime_handoff["fire_grid"])
    cells = pd.read_parquet(directory / runtime_handoff["fire_cells"])
    reference_mapping = pd.read_parquet(directory / runtime_handoff["reference_mapping"])
    reference_hazard = pd.read_parquet(directory / runtime_handoff["reference_hazard"])
    targets = shared["validation_targets"]
    clock = shared["clock"]
    grid_cfg = shared["fire"]["grid"]
    if len(grid) != int(targets["cells_per_snapshot"]):
        raise ValueError("Fire-grid row count differs from the configured target")
    if len(cells) != int(targets["fire_cell_time_series_rows"]):
        raise ValueError("Fire-cell row count differs from the configured target")
    times = np.sort(cells["time_seconds"].unique())
    expected_times = np.arange(clock["simulation_start_seconds"], clock["simulation_end_seconds"] + clock["fire_update_seconds"], clock["fire_update_seconds"])
    if not np.array_equal(times, expected_times):
        raise ValueError("Fire-cell timestamps do not match the configured inclusive clock")
    if cells.duplicated(["time_seconds", "cell_id"]).any():
        raise ValueError("Fire-cell table contains duplicate (time_seconds,cell_id) keys")
    counts = cells.groupby("time_seconds")["cell_id"].nunique().to_numpy()
    if not np.all(counts == int(grid_cfg["expected_cell_count"])):
        raise ValueError("Every fire snapshot must contain every configured grid cell exactly once")
    expected_ids = cells.apply(lambda r: grid_cfg["cell_id_format"].format(row=int(r["row"]), column=int(r["column"])), axis=1)
    if not (expected_ids == cells["cell_id"]).all():
        raise ValueError("Fire-cell identifiers disagree with row/column fields")
    if set(cells["canonical_state_label"].unique()) != set(shared["fire"]["allowed_runtime_states"]):
        raise ValueError("Observed canonical fire states differ from the configured allowed states")
    mapping = shared["hazard"]["state_to_cell_hazard"]
    expected_hazard = cells["canonical_state_label"].map(mapping).astype(float)
    if not np.array_equal(expected_hazard.to_numpy(), cells["h_c"].to_numpy(dtype=float)):
        raise ValueError("Fire-cell h_c values disagree with the configured state mapping")
    if not np.array_equal(cells["h_c"].to_numpy(), cells["hazard_value"].to_numpy()):
        raise ValueError("Fire-cell h_c and hazard_value aliases disagree")
    interacting_expected: dict[int, bool] | None = None
    if manifest_version == "manhattan-fire-handoff-interacting-fronts-v007":
        if "interacting_fronts" not in cells or not pd.api.types.is_bool_dtype(cells["interacting_fronts"]):
            raise ValueError("v007 fire cells require a non-null boolean interacting_fronts column")
        if cells["interacting_fronts"].isna().any():
            raise ValueError("interacting_fronts may not contain null values")
        per_snapshot = cells.groupby("time_seconds")["interacting_fronts"].agg(["nunique", "first"])
        if (per_snapshot["nunique"] != 1).any():
            raise ValueError("Every fire snapshot must have exactly one interacting_fronts value")
        limitation = shared["fire"]["known_limitation"]
        first_interacting = int(limitation["first_interacting_export_seconds"])
        interacting_expected = {int(value): bool(value >= first_interacting) for value in times}
        observed = {int(index): bool(row["first"]) for index, row in per_snapshot.iterrows()}
        if observed != interacting_expected:
            raise ValueError(f"interacting_fronts boundary mismatch: expected={interacting_expected}, observed={observed}")
    ignition = cells[(cells["time_seconds"] == shared["fire"]["ignition"]["ignition_time_seconds"]) & cells["ignition_flag"]]
    expected_ignition = {item["expected_grid_cell_id"] for item in shared["fire"]["ignition"]["sources"]}
    if set(ignition["cell_id"]) != expected_ignition or len(ignition) != len(expected_ignition):
        raise ValueError("Initial ignition rows differ from the configured sources")
    if len(reference_mapping) != int(targets["positive_edge_cell_intersections"]):
        raise ValueError("Reference edge-cell row count differs from the configured target")
    if len(reference_hazard) != int(targets["edge_hazard_time_series_rows"]):
        raise ValueError("Reference edge-hazard row count differs from the configured target")
    if interacting_expected is not None:
        if "interacting_fronts" not in reference_hazard or not pd.api.types.is_bool_dtype(reference_hazard["interacting_fronts"]):
            raise ValueError("v007 reference hazards require a boolean interacting_fronts column")
        observed_reference = reference_hazard.groupby("time_seconds")["interacting_fronts"].agg(["nunique", "first"])
        if (observed_reference["nunique"] != 1).any() or {
            int(index): bool(row["first"]) for index, row in observed_reference.iterrows()
        } != interacting_expected:
            raise ValueError("Reference hazard interacting_fronts values disagree with fire snapshots")
    expected_classification = set(config.common["file_contract"]["classification"])
    if set(manifest.get("classification", [])) != expected_classification:
        raise ValueError("Handoff classification differs from the active common contract")
    net = sumolib.net.readNet(str(config.network_path), withInternal=True)
    normal_count, internal_count, passenger_count = _routeable_counts(net, network["vehicle_class"])
    expected_counts = network["expected_counts"]
    observed_counts = {"normal_edges": normal_count, "internal_edges": internal_count, "passenger_routeable_normal_edges": passenger_count}
    if any(observed_counts[name] != int(expected_counts[name]) for name in observed_counts):
        raise ValueError(f"SUMO network counts differ from contract: observed={observed_counts}")
    validation = {
        "status": "passed",
        "payload_count": len(actual),
        "common_sha256": config.common_sha256,
        "network_sha256": network["sha256"],
        "network_counts": observed_counts,
        "grid_rows": len(grid),
        "fire_rows": len(cells),
        "fire_snapshots": len(times),
        "reference_mapping_rows": len(reference_mapping),
        "reference_hazard_rows": len(reference_hazard),
        "ignition_cells": sorted(expected_ignition),
        "manifest_schema_version": manifest_version,
        "common_hash_field": contract_key,
        "bundle_checksum_status": "passed",
        "local_network_sha256": sha256_file(config.network_path),
        "bundled_network_sha256": sha256_file(bundled_network),
        "bundled_common_contract_sha256": sha256_file(bundled_contract),
    }
    if interacting_expected is not None:
        validation["interacting_fronts"] = {
            "false_snapshot_count": sum(not value for value in interacting_expected.values()),
            "true_snapshot_count": sum(interacting_expected.values()),
            "first_true_snapshot_seconds": min(time for time, value in interacting_expected.items() if value),
        }
    return ValidatedHandoff(directory, actual, manifest, state_manifest, grid, cells, reference_mapping, reference_hazard, validation)
