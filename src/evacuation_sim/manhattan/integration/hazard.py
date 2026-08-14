from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import sumolib
from scipy.sparse import csr_matrix
from shapely.geometry import LineString, box

from .config import ResolvedIntegrationConfig
from .handoff import ValidatedHandoff


@dataclass(frozen=True)
class ReconstructedHazard:
    mapping: pd.DataFrame
    coverage: pd.DataFrame
    edge_hazard: pd.DataFrame
    comparison: dict[str, Any]


def _routeable_edges(net, vehicle_class: str):
    return [
        edge
        for edge in net.getEdges(withInternal=False)
        if not edge.getFunction() and any(lane.allows(vehicle_class) for lane in edge.getLanes())
    ]


def reconstruct_mapping(config: ResolvedIntegrationConfig, grid_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = config.shared
    grid_cfg = shared["fire"]["grid"]
    geometry_cfg = shared["network"]["geometry_contract"]
    hazard_cfg = shared["hazard"]["cell_to_edge"]
    tolerance = float(geometry_cfg["positive_intersection_tolerance_m"])
    minimum_coverage = float(hazard_cfg["minimum_coverage_fraction"])
    geometries = [box(r.x_min, r.y_min, r.x_max, r.y_max) for r in grid_table.itertuples(index=False)]
    grid = gpd.GeoDataFrame(grid_table.copy(), geometry=geometries, crs=None)
    spatial_index = grid.sindex
    net = sumolib.net.readNet(str(config.network_path))
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for edge in _routeable_edges(net, shared["network"]["vehicle_class"]):
        edge_id = edge.getID()
        line = LineString(edge.getShape())
        length = float(line.length)
        if not np.isfinite(length) or length <= 0:
            raise ValueError(f"Routeable edge {edge_id!r} has invalid geometric length {length}")
        candidates = sorted(int(index) for index in spatial_index.query(line, predicate="intersects"))
        edge_rows: list[dict[str, Any]] = []
        for index in candidates:
            cell = grid.iloc[index]
            overlap = float(line.intersection(cell.geometry).length)
            if overlap <= tolerance:
                continue
            edge_rows.append({
                "edge_id": edge_id,
                "cell_id": str(cell.cell_id),
                "edge_length": length,
                "overlap_length": overlap,
                "overlap_fraction": overlap / length,
                "sumo_reported_edge_length": float(edge.getLength()),
            })
        if not edge_rows:
            raise ValueError(f"Routeable edge {edge_id!r} has no positive grid intersection")
        aggregated = pd.DataFrame(edge_rows).groupby(["edge_id", "cell_id"], as_index=False).agg(
            edge_length=("edge_length", "first"),
            overlap_length=("overlap_length", "sum"),
            sumo_reported_edge_length=("sumo_reported_edge_length", "first"),
        )
        aggregated["overlap_fraction"] = aggregated["overlap_length"] / aggregated["edge_length"]
        total = float(aggregated["overlap_length"].sum())
        coverage = total / length
        if total > length + tolerance:
            raise ValueError(f"Boundary double counting for edge {edge_id!r}: overlap={total}, length={length}")
        if coverage < minimum_coverage - tolerance:
            raise ValueError(f"Incomplete grid coverage for edge {edge_id!r}: {coverage} < {minimum_coverage}")
        aggregated["edge_coverage_ratio"] = coverage
        rows.extend(aggregated.to_dict("records"))
        coverage_rows.append({"edge_id": edge_id, "edge_length": length, "overlap_length": total, "coverage_ratio": coverage, "intersection_count": len(aggregated)})
    mapping = pd.DataFrame(rows).sort_values(["edge_id", "cell_id"], kind="stable").reset_index(drop=True)
    if mapping.duplicated(["edge_id", "cell_id"]).any():
        raise ValueError("Reconstructed mapping contains duplicate (edge_id,cell_id) keys")
    mapping["network_hash"] = shared["network"]["sha256"]
    mapping["crs"] = shared["network"]["coordinate_frame"]["projection"]
    mapping["edge_length_m"] = mapping["edge_length"]
    mapping["intersection_length_m"] = mapping["overlap_length"]
    mapping["length_fraction"] = mapping["overlap_fraction"]
    return mapping, pd.DataFrame(coverage_rows).sort_values("edge_id", kind="stable").reset_index(drop=True)


def compare_mapping(local: pd.DataFrame, reference: pd.DataFrame, absolute_tolerance: float, relative_tolerance: float) -> dict[str, Any]:
    columns = ["edge_id", "cell_id"]
    left = local.sort_values(columns, kind="stable").reset_index(drop=True)
    right = reference.sort_values(columns, kind="stable").reset_index(drop=True)
    if not left[columns].equals(right[columns]):
        merged = left[columns].merge(right[columns], how="outer", indicator=True)
        raise ValueError(f"Local/reference mapping key mismatch: {merged[merged['_merge'] != 'both'].head().to_dict('records')}")
    errors: dict[str, float] = {}
    aliases = {"edge_length": "edge_length", "overlap_length": "overlap_length", "overlap_fraction": "overlap_fraction", "edge_coverage_ratio": "edge_coverage_ratio"}
    for local_name, reference_name in aliases.items():
        a = left[local_name].to_numpy(float)
        b = right[reference_name].to_numpy(float)
        errors[local_name] = float(np.max(np.abs(a - b)))
        if not np.allclose(a, b, atol=absolute_tolerance, rtol=relative_tolerance):
            raise ValueError(f"Local/reference mapping differs for {local_name}; max_abs_error={errors[local_name]}")
    return {
        "status": "passed", "rows": len(left), "numeric_max_absolute_errors": errors,
        "routeable_edges": int(left["edge_id"].nunique()),
        "minimum_edge_coverage": float(left.groupby("edge_id")["edge_coverage_ratio"].first().min()),
    }


def compute_sparse_edge_hazard(config: ResolvedIntegrationConfig, mapping: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    shared = config.shared
    edge_ids = sorted(mapping["edge_id"].unique())
    cell_ids = sorted(cells["cell_id"].unique())
    edge_index = {value: index for index, value in enumerate(edge_ids)}
    cell_index = {value: index for index, value in enumerate(cell_ids)}
    rows = mapping["edge_id"].map(edge_index).to_numpy(int)
    columns = mapping["cell_id"].map(cell_index).to_numpy(int)
    weights = mapping["overlap_fraction"].to_numpy(float)
    matrix = csr_matrix((weights, (rows, columns)), shape=(len(edge_ids), len(cell_ids)))
    coverage = mapping.groupby("edge_id")["overlap_fraction"].sum().reindex(edge_ids).to_numpy(float)
    edge_lengths = mapping.groupby("edge_id")["edge_length"].first().reindex(edge_ids).to_numpy(float)
    lambda_value = float(shared["hazard"]["edge_survival_and_risk"]["lambda"])
    bound_tolerance = float(shared["hazard"]["cell_to_edge"]["numerical_bound_tolerance"])
    output: list[pd.DataFrame] = []
    for time_step, time_value in enumerate(sorted(cells["time_seconds"].unique())):
        snapshot = cells[cells["time_seconds"] == time_value].set_index("cell_id")
        interaction_values = snapshot["interacting_fronts"].unique() if "interacting_fronts" in snapshot else np.array([False])
        if len(interaction_values) != 1:
            raise ValueError(f"Snapshot {time_value} has inconsistent interacting_fronts values")
        interacting_fronts = bool(interaction_values[0])
        vector = snapshot.loc[cell_ids, "h_c"].to_numpy(float)
        hazard = np.asarray(matrix @ vector).reshape(-1)
        if np.any(hazard < -bound_tolerance) or np.any(hazard > 1.0 + bound_tolerance):
            raise ValueError(f"Computed edge hazards exceed configured numerical bounds at time {time_value}")
        survival = np.exp(-lambda_value * hazard)
        output.append(pd.DataFrame({
            "time_step": time_step,
            "time_seconds": time_value,
            "time": time_value,
            "edge_id": edge_ids,
            "edge_hazard": hazard,
            "edge_survival": survival,
            "edge_risk": 1.0 - survival,
            "edge_length": edge_lengths,
            "coverage_ratio": coverage,
            "lambda": lambda_value,
            "interacting_fronts": interacting_fronts,
        }))
    return pd.concat(output, ignore_index=True)


def compare_hazard(local: pd.DataFrame, reference: pd.DataFrame, absolute_tolerance: float, relative_tolerance: float) -> dict[str, Any]:
    keys = ["time_seconds", "edge_id"]
    left = local.sort_values(keys, kind="stable").reset_index(drop=True)
    right = reference.sort_values(keys, kind="stable").reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError("Local/reference edge-hazard keys differ")
    if "interacting_fronts" in right:
        if "interacting_fronts" not in left or not left["interacting_fronts"].equals(right["interacting_fronts"]):
            raise ValueError("Local/reference interacting_fronts values differ")
    errors: dict[str, float] = {}
    for column in ("edge_hazard", "edge_survival", "edge_risk", "edge_length", "coverage_ratio"):
        a = left[column].to_numpy(float)
        b = right[column].to_numpy(float)
        errors[column] = float(np.max(np.abs(a - b)))
        if not np.allclose(a, b, atol=absolute_tolerance, rtol=relative_tolerance):
            raise ValueError(f"Local/reference edge hazard differs for {column}; max_abs_error={errors[column]}")
    return {"status": "passed", "rows": len(left), "numeric_max_absolute_errors": errors}


def reconstruct_and_validate_hazard(config: ResolvedIntegrationConfig, handoff: ValidatedHandoff) -> ReconstructedHazard:
    mapping, coverage = reconstruct_mapping(config, handoff.grid)
    grid_name = config.runtime["handoff"]["fire_grid"]
    artifacts = {item["path"]: item for item in handoff.handoff_manifest["artifacts"]}
    if grid_name not in artifacts or not artifacts[grid_name].get("logical_content_sha256"):
        raise ValueError("Handoff manifest does not provide the configured fire-grid logical hash")
    mapping["grid_hash"] = artifacts[grid_name]["logical_content_sha256"]
    tolerance = config.runtime["validation"]
    mapping_comparison = compare_mapping(mapping, handoff.reference_mapping, float(tolerance["mapping_absolute_tolerance_m"]), float(tolerance["mapping_relative_tolerance"]))
    edge_hazard = compute_sparse_edge_hazard(config, mapping, handoff.fire_cells)
    edge_hazard["run_id"] = config.runtime["execution"]["run_id"]
    edge_hazard["hazard_model_version"] = config.shared["hazard"]["model_id"]
    edge_hazard["source_fire_model"] = f"{config.shared['fire']['engine']['name']} {config.shared['fire']['engine']['version']}"
    edge_hazard["configuration_hash"] = config.logical_sha256
    edge_hazard["classification"] = ",".join(config.runtime["execution"]["result_classification"])
    hazard_comparison = compare_hazard(edge_hazard, handoff.reference_hazard, float(tolerance["hazard_absolute_tolerance"]), float(tolerance["hazard_relative_tolerance"]))
    hazardous = edge_hazard[edge_hazard["edge_hazard"] > 0.0]
    counts = hazardous.groupby("time_seconds")["edge_id"].nunique()
    footprint = {
        "peak_simultaneous_hazardous_edges": int(counts.max()),
        "peak_time_seconds": int(counts.idxmax()),
        "distinct_edges_ever_hazardous": int(hazardous["edge_id"].nunique()),
        "hazardous_edges_at_initial_time": int(
            hazardous[hazardous["time_seconds"] == config.shared["clock"]["simulation_start_seconds"]]["edge_id"].nunique()
        ),
    }
    expected_footprint = tolerance.get("hazard_footprint")
    if expected_footprint is not None:
        mismatches = {
            key: {"expected": int(value), "actual": footprint[key]}
            for key, value in expected_footprint.items() if footprint[key] != int(value)
        }
        if mismatches:
            raise ValueError(f"Reconstructed hazard footprint differs from configured acceptance data: {mismatches}")
    return ReconstructedHazard(mapping, coverage, edge_hazard, {
        "mapping": mapping_comparison, "hazard": hazard_comparison, "hazard_footprint": footprint,
    })
