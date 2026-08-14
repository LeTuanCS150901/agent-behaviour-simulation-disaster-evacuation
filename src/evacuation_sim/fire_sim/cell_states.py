from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import sumolib

from evacuation_sim.fire_sim.fire_grid import edge_coordinate, sha256_file
from evacuation_sim.io.tables import read_table, write_table


def ignition_cells_for_edges(grid: gpd.GeoDataFrame, net, ignition_edges: list[str]) -> list[str]:
    cells = []
    sindex = grid.sindex
    for edge_id in ignition_edges:
        x, y = edge_coordinate(net.getEdge(edge_id))
        point = gpd.points_from_xy([x], [y])[0]
        candidate_idx = list(sindex.query(point, predicate="intersects"))
        if not candidate_idx:
            raise ValueError(f"Ignition edge {edge_id} at ({x}, {y}) did not fall inside any fire-grid cell.")
        cells.append(str(grid.iloc[int(candidate_idx[0])].cell_id))
    return sorted(set(cells))


def write_simfire_input_manifest(
    grid: gpd.GeoDataFrame,
    grid_metadata_path: str | Path,
    network_file: str,
    net,
    stage6_cfg: dict[str, Any],
    ignition_edges: list[str],
    random_seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir) / "simfire"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(grid_metadata_path).read_text(encoding="utf-8"))
    ignition_cells = ignition_cells_for_edges(grid, net, ignition_edges)
    expected_path = output_dir / "fire_cell_time_series.parquet"
    manifest = {
        "grid_metadata_path": str(Path(grid_metadata_path).resolve()),
        "grid_version_or_hash": metadata["grid_geojson_sha256"],
        "network_file": str(Path(network_file).resolve()),
        "network_bbox": metadata["network_bbox"],
        "grid_bbox": metadata["grid_bbox"],
        "number_of_rows": metadata["number_of_rows"],
        "number_of_columns": metadata["number_of_columns"],
        "cell_width": metadata["cell_width"],
        "cell_height": metadata["cell_height"],
        "row_origin": metadata["row_origin"],
        "valid_cell_identifiers": sorted(grid["cell_id"].astype(str).tolist()),
        "ignition_cells": ignition_cells,
        "ignition_edges": ignition_edges,
        "random_seed": int(random_seed),
        "simulation_duration": float(stage6_cfg["mock_fire"]["simulation_duration"]),
        "snapshot_interval": float(stage6_cfg["mock_fire"]["snapshot_interval"]),
        "cell_state_to_hazard_mapping": stage6_cfg["cell_hazard_mapping"],
        "expected_simfire_output_path": str(expected_path.resolve()),
    }
    manifest_path = output_dir / "simfire_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _cell_distance_to_ignition(row, ignition_lookup: dict[str, tuple[float, float]]) -> float:
    return min(math.hypot(float(row.center_x) - x, float(row.center_y) - y) for x, y in ignition_lookup.values())


def generate_mock_cell_time_series(
    grid: gpd.GeoDataFrame,
    manifest: dict[str, Any],
    stage6_cfg: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir) / "simfire"
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = float(manifest["simulation_duration"])
    interval = float(manifest["snapshot_interval"])
    spread_rate = float(stage6_cfg["mock_fire"]["spread_rate"])
    mapping = stage6_cfg["cell_hazard_mapping"]
    times = np.arange(0.0, duration + interval * 0.5, interval)
    ignition_grid = grid[grid["cell_id"].isin(manifest["ignition_cells"])]
    ignition_lookup = {
        row.cell_id: (float(row.center_x), float(row.center_y))
        for row in ignition_grid.itertuples(index=False)
    }
    if not ignition_lookup:
        raise ValueError("Mock fire generation requires at least one ignition cell.")
    band = max(float(manifest["cell_width"]), float(manifest["cell_height"]))
    base = grid[["cell_id", "row", "column", "center_x", "center_y"]].copy()
    base["distance_to_ignition"] = base.apply(lambda r: _cell_distance_to_ignition(r, ignition_lookup), axis=1)
    rows = []
    for time in times:
        radius = spread_rate * float(time)
        for cell in base.itertuples(index=False):
            if cell.cell_id in manifest["ignition_cells"] and time == 0:
                state = "burning"
            elif cell.distance_to_ignition < max(0.0, radius - band):
                state = "burned"
            elif cell.distance_to_ignition <= radius + band:
                state = "burning"
            else:
                state = "unburned"
            rows.append(
                {
                    "time": float(time),
                    "cell_id": cell.cell_id,
                    "row": int(cell.row),
                    "column": int(cell.column),
                    "fire_state": state,
                    "hazard_value": float(mapping[state]),
                }
            )
    df = pd.DataFrame(rows)
    path = output_dir / "fire_cell_time_series.parquet"
    write_table(df, path)
    stdout_path = output_dir / "simfire_stdout.log"
    stderr_path = output_dir / "simfire_stderr.log"
    stdout_path.write_text("Deterministic TEST_LOCAL mock cell-fire fixture generated.\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    metadata = {
        "real_simfire": False,
        "mock_fixture": True,
        "real_simfire_validation": "not_run_simfire_missing_or_unavailable",
        "python_version": sys.version,
        "platform": platform.platform(),
        "random_seed": manifest["random_seed"],
        "grid_dimensions": [manifest["number_of_rows"], manifest["number_of_columns"]],
        "simulation_start_time": 0.0,
        "simulation_end_time": duration,
        "fire_time_step": interval,
        "ignition_cells": manifest["ignition_cells"],
        "state_to_hazard_mapping": mapping,
        "number_of_output_snapshots": int(len(times)),
        "output_rows": int(len(df)),
        "output_path": str(path.resolve()),
        "simfire_version": None,
    }
    metadata_path = output_dir / "simfire_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def validate_simfire_output_contract(
    manifest_path: str | Path,
    time_series_path: str | Path,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    time_series_path = Path(time_series_path)
    if not time_series_path.exists():
        raise FileNotFoundError(
            f"Missing simfire cell-state output: {time_series_path}. "
            "Run real simfire or the deterministic mock cell generator before edge-hazard computation."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    df = read_table(time_series_path)
    required = {"time", "cell_id", "row", "column", "fire_state", "hazard_value"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"Simfire cell-state output is missing columns: {missing_cols}")
    valid_cells = set(manifest["valid_cell_identifiers"])
    bad_cells = sorted(set(df["cell_id"].astype(str)) - valid_cells)
    if bad_cells:
        raise ValueError(f"Simfire output contains cell IDs not present in manifest: {bad_cells[:5]}")
    if not df["hazard_value"].between(0.0, 1.0).all():
        raise ValueError("Simfire output hazard_value must be in [0,1].")
    expected_cells = len(valid_cells)
    times = sorted(df["time"].unique())
    counts = df.groupby("time")["cell_id"].nunique()
    full_output = bool((counts == expected_cells).all())
    if not full_output:
        raise ValueError(
            "Simfire output is not full grid output for every snapshot. "
            "Sparse output requires explicit sparse_nonzero handling and is not the default."
        )
    metadata = {}
    if metadata_path and Path(metadata_path).exists():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if metadata.get("mock_fixture") and metadata.get("real_simfire") is True:
            raise ValueError("Invalid simfire metadata: mock fixture cannot be marked as real simfire.")
    return {
        "time_series_rows": int(len(df)),
        "snapshot_count": int(len(times)),
        "cell_count": int(expected_cells),
        "full_output": full_output,
        "hazard_min": float(df["hazard_value"].min()),
        "hazard_max": float(df["hazard_value"].max()),
        "metadata_real_simfire": bool(metadata.get("real_simfire", False)),
        "metadata_mock_fixture": bool(metadata.get("mock_fixture", False)),
    }

