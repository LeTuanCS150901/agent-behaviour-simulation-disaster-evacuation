from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
import struct
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkt


def _require_columns(table: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def line_interpolated_midpoint(shape: list[tuple[float, float]]) -> tuple[float, float]:
    """Return the point at 50% of a polyline's cumulative geometric length."""
    if len(shape) < 2:
        raise ValueError("Edge shape must contain at least two points")
    points = [(float(x), float(y)) for x, y in shape]
    if not np.isfinite(np.asarray(points, dtype=float)).all():
        raise ValueError("Edge shape coordinates must be finite")
    segments: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    total = 0.0
    for start, end in zip(points, points[1:]):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        segments.append((start, end, length))
        total += length
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Edge shape must have positive finite geometric length")
    target = total / 2.0
    covered = 0.0
    for start, end, length in segments:
        if length > 0.0 and covered + length >= target:
            fraction = (target - covered) / length
            return (
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            )
        covered += length
    return points[-1]


def derive_origin_destination_points(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    network: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate configured endpoints and locate them on sealed SUMO geometry."""
    _require_columns(origins, {"origin_id", "edge_id", "num_cars"}, "Origin table")
    _require_columns(
        destinations,
        {"destination_id", "edge_id", "capacity"},
        "Destination table",
    )
    if origins.empty or destinations.empty:
        raise ValueError("Origin and destination tables must both be non-empty")
    for table, id_column, label in (
        (origins, "origin_id", "Origin"),
        (destinations, "destination_id", "Destination"),
    ):
        if table.isna().any().any():
            raise ValueError(f"{label} table may not contain missing values")
        if table[id_column].astype(str).duplicated().any():
            raise ValueError(f"{label} identifiers must be unique")
        if table["edge_id"].astype(str).duplicated().any():
            raise ValueError(f"{label} edge IDs must be unique")
    if (pd.to_numeric(origins["num_cars"], errors="raise") <= 0).any():
        raise ValueError("Origin num_cars values must be positive")
    if (pd.to_numeric(destinations["capacity"], errors="raise") <= 0).any():
        raise ValueError("Destination capacity values must be positive")

    rows: list[dict[str, Any]] = []
    definitions = (
        (origins, "origin", "origin_id", "num_cars"),
        (destinations, "destination", "destination_id", "capacity"),
    )
    for table, role, id_column, value_column in definitions:
        for row in table.sort_values(id_column, kind="stable").itertuples(index=False):
            edge_id = str(getattr(row, "edge_id"))
            try:
                edge = network.getEdge(edge_id)
            except Exception as exc:
                raise KeyError(f"{role} edge {edge_id!r} is absent from the sealed network") from exc
            if str(edge.getFunction()) not in {"", "normal"} or edge_id.startswith(":"):
                raise ValueError(f"{role} edge {edge_id!r} is not a normal SUMO edge")
            if not bool(edge.allows("passenger")):
                raise ValueError(f"{role} edge {edge_id!r} does not allow passenger vehicles")
            x, y = line_interpolated_midpoint(list(edge.getShape()))
            value = int(getattr(row, value_column))
            rows.append(
                {
                    "role": role,
                    "location_id": str(getattr(row, id_column)),
                    "edge_id": edge_id,
                    "x": x,
                    "y": y,
                    "vehicle_count": value if role == "origin" else None,
                    "capacity": value if role == "destination" else None,
                }
            )
    result = pd.DataFrame(rows)
    return result, {
        "coordinate_definition": "line-interpolated midpoint at 50% of SUMO edge centreline length",
        "origin_count": int((result["role"] == "origin").sum()),
        "destination_count": int((result["role"] == "destination").sum()),
        "points": result.to_dict("records"),
    }


def derive_cell_ignition(
    fire_states: pd.DataFrame,
    fire_grid: pd.DataFrame,
    *,
    expected_network_hash: str,
    expected_coordinate_frame: str,
    expected_extent: list[float],
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _require_columns(
        fire_states,
        {"cell_id", "time_seconds", "burn_status_label", "interacting_fronts", "network_hash", "grid_hash"},
        "Fire-state table",
    )
    _require_columns(
        fire_grid,
        {
            "cell_id", "geometry_wkt", "x_min", "y_min", "x_max", "y_max", "crs",
            "coordinate_frame", "network_hash", "grid_hash",
        },
        "Fire-grid table",
    )
    if fire_grid["cell_id"].duplicated().any():
        duplicate = str(fire_grid.loc[fire_grid["cell_id"].duplicated(), "cell_id"].iloc[0])
        raise ValueError(f"Fire-grid cell_id is not unique; duplicate={duplicate}")
    state_cells = set(fire_states["cell_id"].astype(str))
    grid_cells = set(fire_grid["cell_id"].astype(str))
    if state_cells != grid_cells:
        missing = sorted(state_cells - grid_cells)
        extra = sorted(grid_cells - state_cells)
        raise ValueError(
            f"Fire-state/grid cell_id mismatch: missing_grid={missing[:5]}, extra_grid={extra[:5]}"
        )
    if fire_states.duplicated(["cell_id", "time_seconds"]).any():
        row = fire_states.loc[fire_states.duplicated(["cell_id", "time_seconds"])].iloc[0]
        raise ValueError(
            f"Duplicate fire-state key cell_id={row.cell_id!r}, time_seconds={row.time_seconds!r}"
        )
    for label, table in (("fire-state", fire_states), ("fire-grid", fire_grid)):
        hashes = sorted(table["network_hash"].dropna().astype(str).unique())
        if hashes != [expected_network_hash]:
            raise ValueError(
                f"{label} network_hash mismatch: expected {expected_network_hash}, observed {hashes}"
            )
    coordinate_frames = sorted(fire_grid["coordinate_frame"].dropna().astype(str).unique())
    if coordinate_frames != [expected_coordinate_frame]:
        raise ValueError(
            f"Fire-grid coordinate frame mismatch: expected {expected_coordinate_frame!r}, "
            f"observed {coordinate_frames}"
        )
    grid_hashes = sorted(fire_grid["grid_hash"].dropna().astype(str).unique())
    state_grid_hashes = sorted(fire_states["grid_hash"].dropna().astype(str).unique())
    if len(grid_hashes) != 1 or grid_hashes != state_grid_hashes:
        raise ValueError(
            f"Fire-state/grid grid_hash mismatch: states={state_grid_hashes}, grid={grid_hashes}"
        )

    observed_extent = [
        float(fire_grid["x_min"].min()), float(fire_grid["y_min"].min()),
        float(fire_grid["x_max"].max()), float(fire_grid["y_max"].max()),
    ]
    if not np.allclose(observed_extent, expected_extent, atol=tolerance, rtol=0.0):
        raise ValueError(
            f"Fire-grid extent mismatch: expected={expected_extent}, observed={observed_extent}"
        )

    geometries: list[Any] = []
    for row in fire_grid.itertuples(index=False):
        try:
            geometry = wkt.loads(str(row.geometry_wkt))
        except Exception as exc:
            raise ValueError(f"Invalid geometry_wkt for cell {row.cell_id!r}: {exc}") from exc
        if geometry.geom_type != "Polygon" or geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Cell {row.cell_id!r} must have one valid non-empty Polygon geometry")
        if not np.allclose(
            geometry.bounds,
            [float(row.x_min), float(row.y_min), float(row.x_max), float(row.y_max)],
            atol=tolerance,
            rtol=0.0,
        ):
            raise ValueError(
                f"WKT/bounds mismatch for cell {row.cell_id!r}: "
                f"wkt_bounds={geometry.bounds}, table_bounds={[row.x_min, row.y_min, row.x_max, row.y_max]}"
            )
        geometries.append(geometry)
    geometry_table = fire_grid.copy()
    geometry_table["_geometry"] = geometries

    burning = fire_states.loc[fire_states["burn_status_label"].astype(str) != "UNBURNED"].copy()
    burning["time_seconds"] = pd.to_numeric(burning["time_seconds"], errors="raise")
    if not np.isfinite(burning["time_seconds"].to_numpy(float)).all():
        raise ValueError("Fire ignition times must be finite")
    first = (
        burning.sort_values(["cell_id", "time_seconds"], kind="mergesort")
        .drop_duplicates("cell_id", keep="first")
        .loc[:, ["cell_id", "time_seconds", "interacting_fronts"]]
        .rename(columns={"time_seconds": "ignition_time_seconds"})
        .sort_values("cell_id", kind="mergesort")
        .reset_index(drop=True)
    )
    first["interacting_fronts"] = first["interacting_fronts"].astype(bool)
    render = first.merge(
        geometry_table[["cell_id", "geometry_wkt", "_geometry"]],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    if render["_geometry"].isna().any():
        missing_cell = str(render.loc[render["_geometry"].isna(), "cell_id"].iloc[0])
        raise ValueError(f"Burning cell {missing_cell!r} has no canonical grid geometry")
    audit = first[["cell_id", "ignition_time_seconds", "interacting_fronts"]].copy()
    metadata = {
        "total_grid_cells": int(len(fire_grid)),
        "burning_cells": int(len(audit)),
        "interacting_ignition_cells": int(audit["interacting_fronts"].sum()),
        "ignition_time_range_seconds": [
            float(audit["ignition_time_seconds"].min()),
            float(audit["ignition_time_seconds"].max()),
        ],
        "grid_extent": observed_extent,
        "grid_hash": grid_hashes[0],
        "network_hash": expected_network_hash,
        "coordinate_frame": coordinate_frames[0],
        "crs_values": sorted(fire_grid["crs"].dropna().astype(str).unique()),
    }
    return audit, render, metadata


def derive_driven_trace(
    active: pd.DataFrame,
    *,
    internal_edge_prefix: str,
    include_internal_edges: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_columns(
        active,
        {"vehicle_id", "time_seconds", "current_road_id", "position_class"},
        "Active-vehicle table",
    )
    if active[["vehicle_id", "time_seconds", "current_road_id", "position_class"]].isna().any().any():
        raise ValueError("Active-vehicle trace fields may not be null")
    if active.duplicated(["vehicle_id", "time_seconds"]).any():
        row = active.loc[active.duplicated(["vehicle_id", "time_seconds"])].iloc[0]
        raise ValueError(
            f"Duplicate active observation vehicle={row.vehicle_id!r}, time={row.time_seconds!r}"
        )
    ordered = active.sort_values(["vehicle_id", "time_seconds"], kind="mergesort").copy()
    ordered["_run_start"] = ordered.groupby("vehicle_id", sort=False)["current_road_id"].transform(
        lambda values: values.ne(values.shift())
    )
    ordered["_run_id"] = ordered.groupby("vehicle_id", sort=False)["_run_start"].cumsum()
    runs = (
        ordered.groupby(["vehicle_id", "_run_id"], sort=False)
        .agg(
            edge_id=("current_road_id", "first"),
            entry_time_seconds=("time_seconds", "min"),
            exit_time_seconds=("time_seconds", "max"),
            position_class=("position_class", "first"),
        )
        .reset_index()
    )
    internal_mask = runs["edge_id"].astype(str).str.startswith(internal_edge_prefix)
    selected = runs if include_internal_edges else runs.loc[~internal_mask].copy()
    selected["sequence_index"] = selected.groupby("vehicle_id", sort=False).cumcount()
    trace = selected[
        ["vehicle_id", "sequence_index", "edge_id", "entry_time_seconds", "exit_time_seconds"]
    ].reset_index(drop=True)
    metadata = {
        "vehicle_traces": int(trace["vehicle_id"].nunique()),
        "trace_runs": int(len(trace)),
        "distinct_edges": int(trace["edge_id"].nunique()),
        "excluded_internal_observations": int(
            ordered["current_road_id"].astype(str).str.startswith(internal_edge_prefix).sum()
        ),
        "excluded_internal_runs": int(internal_mask.sum()) if not include_internal_edges else 0,
        "entry_time_range_seconds": [
            float(trace["entry_time_seconds"].min()), float(trace["entry_time_seconds"].max())
        ],
    }
    return trace, metadata


def derive_selected_route_trace(
    decisions: pd.DataFrame,
    *,
    internal_edge_prefix: str,
    include_internal_edges: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_columns(decisions, {"time", "vehicle_id", "route_edges", "selected"}, "Route-choice table")
    chosen = decisions.loc[decisions["selected"].astype(bool)].sort_values(
        ["vehicle_id", "time"], kind="mergesort"
    )
    rows: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    for row in chosen.itertuples(index=False):
        vehicle = str(row.vehicle_id)
        for edge_id in str(row.route_edges).split():
            if not include_internal_edges and edge_id.startswith(internal_edge_prefix):
                continue
            index = counters.get(vehicle, 0)
            rows.append(
                {
                    "vehicle_id": vehicle,
                    "sequence_index": index,
                    "edge_id": edge_id,
                    "entry_time_seconds": float(row.time),
                    "exit_time_seconds": float(row.time),
                }
            )
            counters[vehicle] = index + 1
    trace = pd.DataFrame(rows)
    if trace.empty:
        raise ValueError("Selected-route mode found no selected route edges")
    return trace, {
        "vehicle_traces": int(trace["vehicle_id"].nunique()),
        "trace_runs": int(len(trace)),
        "distinct_edges": int(trace["edge_id"].nunique()),
        "excluded_internal_observations": None,
        "excluded_internal_runs": None,
        "entry_time_range_seconds": [
            float(trace["entry_time_seconds"].min()), float(trace["entry_time_seconds"].max())
        ],
        "time_semantics": "route_decision_time_not_edge_occupancy_time",
    }


def _shape_segments(shape: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    return [[shape[index], shape[index + 1]] for index in range(len(shape) - 1)]


def derive_route_edge_activity(
    active: pd.DataFrame,
    trace: pd.DataFrame,
    network: Any,
    *,
    internal_edge_prefix: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_columns(
        active,
        {"vehicle_id", "time_seconds", "current_road_id", "position_class"},
        "Active-vehicle table",
    )
    _require_columns(
        trace,
        {"vehicle_id", "edge_id", "entry_time_seconds", "exit_time_seconds"},
        "Driven-route trace table",
    )
    if active.duplicated(["vehicle_id", "time_seconds"]).any():
        row = active.loc[active.duplicated(["vehicle_id", "time_seconds"])].iloc[0]
        raise ValueError(
            f"Duplicate active observation vehicle={row.vehicle_id!r}, time={row.time_seconds!r}"
        )
    normal = active.loc[
        ~active["current_road_id"].astype(str).str.startswith(internal_edge_prefix)
        & active["position_class"].astype(str).ne("internal")
    ].copy()
    normal["time_seconds"] = pd.to_numeric(normal["time_seconds"], errors="raise")
    if normal.empty or not np.isfinite(normal["time_seconds"].to_numpy(float)).all():
        raise ValueError("Normal-edge vehicle-presence times must be non-empty and finite")
    grouped = (
        normal.groupby("current_road_id", sort=True)
        .agg(
            distinct_vehicle_count=("vehicle_id", "nunique"),
            vehicle_presence_observation_count=("vehicle_id", "size"),
            first_active_time_seconds=("time_seconds", "min"),
            median_vehicle_presence_time_seconds=("time_seconds", "median"),
            last_active_time_seconds=("time_seconds", "max"),
        )
        .rename_axis("edge_id")
        .reset_index()
    )
    run_counts = trace.groupby("edge_id", sort=True).size().rename("route_run_count")
    grouped = grouped.merge(run_counts, on="edge_id", how="left", validate="one_to_one")
    if grouped["route_run_count"].isna().any():
        edge_id = str(grouped.loc[grouped["route_run_count"].isna(), "edge_id"].iloc[0])
        raise ValueError(f"Active normal edge {edge_id!r} is absent from the driven trace table")
    trace_edges = set(trace["edge_id"].astype(str))
    activity_edges = set(grouped["edge_id"].astype(str))
    if trace_edges != activity_edges:
        raise ValueError(
            "Driven-trace and active normal-edge sets differ: "
            f"trace_only={sorted(trace_edges - activity_edges)[:5]}, "
            f"activity_only={sorted(activity_edges - trace_edges)[:5]}"
        )
    for row in grouped.itertuples(index=False):
        edge_id = str(row.edge_id)
        try:
            edge = network.getEdge(edge_id)
            shape = [(float(x), float(y)) for x, y in edge.getShape()]
        except Exception as exc:
            raise KeyError(f"Network geometry missing route-activity edge {edge_id!r}") from exc
        if len(shape) < 2:
            raise ValueError(f"Route-activity edge {edge_id!r} has fewer than two shape points")
    result = grouped[
        [
            "edge_id", "distinct_vehicle_count", "route_run_count",
            "vehicle_presence_observation_count", "first_active_time_seconds",
            "median_vehicle_presence_time_seconds", "last_active_time_seconds",
        ]
    ].sort_values("edge_id", kind="mergesort").reset_index(drop=True)
    metadata = {
        "route_edge_count": int(len(result)),
        "vehicle_count": int(normal["vehicle_id"].nunique()),
        "normal_presence_observations": int(len(normal)),
        "presence_time_range_seconds": [
            float(normal["time_seconds"].min()), float(normal["time_seconds"].max())
        ],
        "median_occupancy_time_range_seconds": [
            float(result["median_vehicle_presence_time_seconds"].min()),
            float(result["median_vehicle_presence_time_seconds"].max()),
        ],
        "distinct_vehicle_count_range": [
            int(result["distinct_vehicle_count"].min()),
            int(result["distinct_vehicle_count"].max()),
        ],
        "time_semantics": "median_of_one_second_vehicle_presence_observations_per_edge",
    }
    return result, metadata


def linear_line_widths(counts: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values < 1).any():
        raise ValueError("Route-line overlap counts must be finite positive values")
    if minimum <= 0 or maximum < minimum:
        raise ValueError(f"Invalid route-line width range [{minimum}, {maximum}]")
    low, high = float(values.min()), float(values.max())
    if math.isclose(low, high):
        return np.full(values.shape, (minimum + maximum) / 2.0)
    return minimum + (values - low) / (high - low) * (maximum - minimum)


def derive_road_grid_cells(
    mapping: pd.DataFrame,
    fire_grid: pd.DataFrame,
    *,
    expected_network_hash: str,
    expected_mapping_grid_hash: str,
    expected_fire_grid_hash: str,
    expected_cell_size_m: float,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_columns(
        mapping,
        {"edge_id", "cell_id", "intersection_length_m", "network_hash", "grid_hash"},
        "Edge-cell mapping table",
    )
    _require_columns(
        fire_grid,
        {
            "cell_id", "geometry_wkt", "x_min", "y_min", "x_max", "y_max",
            "width_m", "height_m", "network_hash", "grid_hash",
        },
        "Fire-grid table",
    )
    if mapping.duplicated(["edge_id", "cell_id"]).any():
        row = mapping.loc[mapping.duplicated(["edge_id", "cell_id"])].iloc[0]
        raise ValueError(
            f"Duplicate edge-cell mapping edge={row.edge_id!r}, cell={row.cell_id!r}"
        )
    lengths = pd.to_numeric(mapping["intersection_length_m"], errors="raise").to_numpy(float)
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("Road-cell mapping intersection lengths must be finite and positive")
    mapping_network_hashes = sorted(mapping["network_hash"].dropna().astype(str).unique())
    grid_network_hashes = sorted(fire_grid["network_hash"].dropna().astype(str).unique())
    if mapping_network_hashes != [expected_network_hash] or grid_network_hashes != [expected_network_hash]:
        raise ValueError(
            "Road-cell network_hash mismatch: "
            f"expected={expected_network_hash}, mapping={mapping_network_hashes}, grid={grid_network_hashes}"
        )
    mapping_grid_hashes = sorted(mapping["grid_hash"].dropna().astype(str).unique())
    grid_hashes = sorted(fire_grid["grid_hash"].dropna().astype(str).unique())
    if mapping_grid_hashes != [expected_mapping_grid_hash] or grid_hashes != [expected_fire_grid_hash]:
        raise ValueError(
            "Road-cell grid_hash identity mismatch: "
            f"expected_mapping={expected_mapping_grid_hash}, observed_mapping={mapping_grid_hashes}, "
            f"expected_fire_grid={expected_fire_grid_hash}, observed_fire_grid={grid_hashes}"
        )
    if fire_grid["cell_id"].duplicated().any():
        cell = str(fire_grid.loc[fire_grid["cell_id"].duplicated(), "cell_id"].iloc[0])
        raise ValueError(f"Fire-grid cell_id is not unique; duplicate={cell}")
    for dimension in ("width_m", "height_m"):
        values = pd.to_numeric(fire_grid[dimension], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or not np.allclose(
            values, expected_cell_size_m, atol=tolerance, rtol=0.0
        ):
            raise ValueError(
                f"Fire-grid {dimension} must equal configured cell size {expected_cell_size_m} m"
            )
    cell_ids = sorted(mapping["cell_id"].astype(str).unique())
    selected = fire_grid.loc[fire_grid["cell_id"].astype(str).isin(cell_ids)].copy()
    if len(selected) != len(cell_ids):
        missing = sorted(set(cell_ids) - set(selected["cell_id"].astype(str)))
        raise ValueError(f"Mapped road cells are absent from the fire grid: {missing[:5]}")
    geometries: list[Any] = []
    for row in selected.itertuples(index=False):
        try:
            geometry = wkt.loads(str(row.geometry_wkt))
        except Exception as exc:
            raise ValueError(f"Invalid road-cell geometry_wkt for {row.cell_id!r}: {exc}") from exc
        if geometry.geom_type != "Polygon" or geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Road cell {row.cell_id!r} must have one valid non-empty Polygon")
        if not np.allclose(
            geometry.bounds,
            [float(row.x_min), float(row.y_min), float(row.x_max), float(row.y_max)],
            atol=tolerance,
            rtol=0.0,
        ):
            raise ValueError(f"Road-cell WKT/bounds mismatch for cell {row.cell_id!r}")
        geometries.append(geometry)
    selected["_geometry"] = geometries
    selected = selected.sort_values("cell_id", kind="mergesort").reset_index(drop=True)
    return selected, {
        "road_grid_cells": int(len(selected)),
        "mapped_routeable_edges": int(mapping["edge_id"].nunique()),
        "mapping_rows": int(len(mapping)),
        "mapping_grid_artifact_logical_sha256": mapping_grid_hashes[0],
        "fire_grid_logical_sha256": grid_hashes[0],
        "network_hash": expected_network_hash,
        "cell_size_m": float(expected_cell_size_m),
    }


def build_trace_geometry(
    trace: pd.DataFrame,
    network: Any,
    *,
    maximum_bridge_length_m: float,
    maximum_bridge_gap_seconds: float,
    include_internal_edges: bool,
) -> tuple[list[Any], list[float], list[Any], list[float], pd.DataFrame]:
    edge_segments: list[Any] = []
    edge_times: list[float] = []
    bridge_segments: list[Any] = []
    bridge_times: list[float] = []
    bridge_records: list[dict[str, Any]] = []
    shape_cache: dict[str, list[tuple[float, float]]] = {}

    for row in trace.itertuples(index=False):
        edge_id = str(row.edge_id)
        try:
            edge = network.getEdge(edge_id)
        except Exception as exc:
            raise KeyError(
                f"Network geometry missing edge {edge_id!r} for vehicle {row.vehicle_id!r} "
                f"at {row.entry_time_seconds!r} s"
            ) from exc
        shape = [(float(x), float(y)) for x, y in edge.getShape()]
        if len(shape) < 2:
            raise ValueError(
                f"Edge {edge_id!r} for vehicle {row.vehicle_id!r} has fewer than two shape points"
            )
        shape_cache[edge_id] = shape
        segments = _shape_segments(shape)
        edge_segments.extend(segments)
        edge_times.extend([float(row.entry_time_seconds)] * len(segments))

    if not include_internal_edges:
        for vehicle, group in trace.groupby("vehicle_id", sort=False):
            ordered = group.sort_values("sequence_index", kind="mergesort").reset_index(drop=True)
            for index in range(len(ordered) - 1):
                previous = ordered.iloc[index]
                following = ordered.iloc[index + 1]
                start = shape_cache[str(previous.edge_id)][-1]
                end = shape_cache[str(following.edge_id)][0]
                length = float(math.dist(start, end))
                gap = float(following.entry_time_seconds - previous.exit_time_seconds)
                spatial_exceeded = length > maximum_bridge_length_m
                temporal_exceeded = gap > maximum_bridge_gap_seconds
                reason = []
                if spatial_exceeded:
                    reason.append("length_cap")
                if temporal_exceeded:
                    reason.append("time_cap")
                drawn = not reason
                bridge_records.append(
                    {
                        "vehicle_id": str(vehicle),
                        "from_edge_id": str(previous.edge_id),
                        "to_edge_id": str(following.edge_id),
                        "bridge_length_m": length,
                        "bridge_gap_seconds": gap,
                        "drawn": drawn,
                        "suppression_reason": "+".join(reason) if reason else None,
                    }
                )
                if drawn:
                    bridge_segments.append([start, end])
                    bridge_times.append(float(following.entry_time_seconds))
    bridge_table = pd.DataFrame(bridge_records)
    return edge_segments, edge_times, bridge_segments, bridge_times, bridge_table


def _srgb_to_lab_lightness(rgb: np.ndarray) -> np.ndarray:
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = 0.2126729 * linear[:, 0] + 0.7151522 * linear[:, 1] + 0.0721750 * linear[:, 2]
    delta = 6.0 / 29.0
    f = np.where(y > delta**3, np.cbrt(y), y / (3 * delta**2) + 4.0 / 29.0)
    return 116.0 * f - 16.0


def validate_monotonic_colormap(name: str, samples: int, tolerance: float) -> str:
    try:
        colormap = matplotlib.colormaps[name]
    except KeyError as exc:
        raise ValueError(f"Unknown configured colormap {name!r}") from exc
    rgb = np.asarray(colormap(np.linspace(0.0, 1.0, samples)))[:, :3]
    differences = np.diff(_srgb_to_lab_lightness(rgb))
    increasing = bool(np.all(differences >= -tolerance))
    decreasing = bool(np.all(differences <= tolerance))
    if not increasing and not decreasing:
        raise ValueError(
            f"Colormap {name!r} has non-monotonic CIELAB lightness and is unsuitable for time"
        )
    return "increasing" if increasing else "decreasing"


def validate_shared_time_scale(
    ignition: pd.DataFrame,
    route_activity: pd.DataFrame,
    *,
    vmin: float,
    vmax: float,
) -> dict[str, list[float]]:
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Invalid shared time scale [{vmin}, {vmax}]")
    ranges = {
        "fire_ignition": [
            float(ignition["ignition_time_seconds"].min()),
            float(ignition["ignition_time_seconds"].max()),
        ],
        "route_presence": [
            float(route_activity["first_active_time_seconds"].min()),
            float(route_activity["last_active_time_seconds"].max()),
        ],
        "route_median_occupancy": [
            float(route_activity["median_vehicle_presence_time_seconds"].min()),
            float(route_activity["median_vehicle_presence_time_seconds"].max()),
        ],
    }
    for layer, observed in ranges.items():
        if observed[0] < vmin or observed[1] > vmax:
            raise ValueError(
                f"{layer} observed time range {observed} lies outside configured shared scale "
                f"[{vmin}, {vmax}]"
            )
    return ranges


def render_spatial_fire_route_map(
    ignition_render: pd.DataFrame,
    road_cells: pd.DataFrame,
    route_activity: pd.DataFrame,
    network: Any,
    path: Path,
    settings: dict[str, Any],
    rendering_metadata: dict[str, str],
    dpi: int,
    endpoint_points: pd.DataFrame | None = None,
    endpoint_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (endpoint_points is None) != (endpoint_settings is None):
        raise ValueError("endpoint_points and endpoint_settings must be supplied together")
    time_colour = settings["time_colour"]
    lightness_direction = validate_monotonic_colormap(
        time_colour["colormap"], int(time_colour["lightness_samples"]),
        float(time_colour["lightness_tolerance"]),
    )
    vmin = float(time_colour["vmin_seconds"])
    vmax = float(time_colour["vmax_seconds"])
    time_ranges = validate_shared_time_scale(
        ignition_render, route_activity, vmin=vmin, vmax=vmax
    )
    shared_norm = Normalize(vmin, vmax)
    shared_cmap = matplotlib.colormaps[time_colour["colormap"]]
    line_settings = settings["route_lines"]
    counts = route_activity["distinct_vehicle_count"].to_numpy(float)
    edge_widths = linear_line_widths(
        counts,
        float(line_settings["minimum_width_points"]),
        float(line_settings["maximum_width_points"]),
    )
    route_segments: list[Any] = []
    route_times: list[float] = []
    route_widths: list[float] = []
    for row, width in zip(route_activity.itertuples(index=False), edge_widths, strict=True):
        edge_id = str(row.edge_id)
        try:
            shape = [(float(x), float(y)) for x, y in network.getEdge(edge_id).getShape()]
        except Exception as exc:
            raise KeyError(f"Network geometry missing rendered route edge {edge_id!r}") from exc
        if len(shape) < 2:
            raise ValueError(f"Rendered route edge {edge_id!r} has fewer than two shape points")
        segments = _shape_segments(shape)
        route_segments.extend(segments)
        route_times.extend([float(row.median_vehicle_presence_time_seconds)] * len(segments))
        route_widths.extend([float(width)] * len(segments))

    road_polygons = [
        MplPolygon(np.asarray(geometry.exterior.coords), closed=True)
        for geometry in road_cells["_geometry"]
    ]
    fire_polygons = [
        MplPolygon(np.asarray(geometry.exterior.coords), closed=True)
        for geometry in ignition_render["_geometry"]
    ]
    fire_colours = shared_cmap(
        shared_norm(ignition_render["ignition_time_seconds"].to_numpy(float))
    )
    interaction_polygons = [
        MplPolygon(np.asarray(geometry.exterior.coords), closed=True)
        for geometry in ignition_render.loc[ignition_render["interacting_fronts"], "_geometry"]
    ]

    fig, ax = plt.subplots(figsize=tuple(settings["figure_size_inches"]))
    fig.patch.set_facecolor(settings["background_color"])
    ax.set_facecolor(settings["background_color"])
    zorders = {layer: index + 1 for index, layer in enumerate(settings["draw_order"])}
    ax.add_collection(
        PatchCollection(
            road_polygons,
            facecolors=settings["road_cells"]["fill_color"],
            edgecolors=settings["road_cells"]["edge_color"],
            linewidths=float(settings["road_cells"]["edge_linewidth"]),
            alpha=float(settings["road_cells"]["alpha"]),
            zorder=zorders["network"],
        )
    )
    ax.add_collection(
        PatchCollection(
            fire_polygons,
            facecolors=fire_colours,
            edgecolors=settings["fire_cells"]["edge_color"],
            linewidths=float(settings["fire_cells"]["edge_linewidth"]),
            alpha=float(settings["fire_cells"]["alpha"]),
            zorder=zorders["fire"],
        )
    )
    ax.add_collection(
        LineCollection(
            route_segments,
            colors=line_settings["casing_color"],
            linewidths=np.asarray(route_widths) + float(line_settings["casing_extra_width_points"]),
            alpha=float(line_settings["casing_alpha"]),
            zorder=zorders["traces"],
        )
    )
    route_collection = LineCollection(
        route_segments,
        cmap=shared_cmap,
        norm=shared_norm,
        linewidths=route_widths,
        alpha=float(line_settings["alpha"]),
        zorder=zorders["traces"] + 0.01,
    )
    route_collection.set_array(np.asarray(route_times, dtype=float))
    ax.add_collection(route_collection)
    if interaction_polygons:
        ax.add_collection(
            PatchCollection(
                interaction_polygons, facecolors="none", edgecolors=settings["interaction_edge_color"],
                linewidths=float(settings["interaction_linewidth"]), hatch=settings["interaction_hatch"],
                zorder=zorders["interaction_marking"],
            )
        )
    endpoint_handles: list[Line2D] = []
    if endpoint_points is not None and endpoint_settings is not None:
        _require_columns(
            endpoint_points,
            {"role", "location_id", "edge_id", "x", "y", "vehicle_count", "capacity"},
            "Endpoint marker table",
        )
        if set(endpoint_points["role"].astype(str)) != {"origin", "destination"}:
            raise ValueError("Endpoint marker roles must contain origin and destination")
        endpoint_zorder = max(zorders.values()) + float(endpoint_settings["zorder_offset"])
        for role, style_key in (("origin", "origins"), ("destination", "destinations")):
            style = endpoint_settings[style_key]
            subset = endpoint_points[endpoint_points["role"] == role]
            ax.scatter(
                subset["x"].to_numpy(float),
                subset["y"].to_numpy(float),
                s=float(style["marker_size_points_squared"]),
                marker=style["marker"],
                facecolor=style["face_color"],
                edgecolor=style["edge_color"],
                linewidth=float(style["edge_linewidth"]),
                zorder=endpoint_zorder,
            )
            if bool(endpoint_settings.get("show_labels", True)):
                for row in subset.itertuples(index=False):
                    label = style["label_template"].format(
                        location_id=str(row.location_id),
                        edge_id=str(row.edge_id),
                        vehicle_count="" if pd.isna(row.vehicle_count) else int(row.vehicle_count),
                        capacity="" if pd.isna(row.capacity) else int(row.capacity),
                    )
                    ax.annotate(
                        label,
                        xy=(float(row.x), float(row.y)),
                        xytext=tuple(style["label_offset_points"]),
                        textcoords="offset points",
                        fontsize=float(endpoint_settings["label_font_size"]),
                        fontweight=endpoint_settings["label_font_weight"],
                        color=endpoint_settings["label_text_color"],
                        zorder=endpoint_zorder + 0.1,
                        bbox={
                            "boxstyle": "round,pad=0.22",
                            "facecolor": endpoint_settings["label_box_color"],
                            "edgecolor": style["edge_color"],
                            "alpha": float(endpoint_settings["label_box_alpha"]),
                            "linewidth": 0.8,
                        },
                    )
            endpoint_handles.append(
                Line2D(
                    [0], [0],
                    marker=style["marker"],
                    color="none",
                    markerfacecolor=style["face_color"],
                    markeredgecolor=style["edge_color"],
                    markersize=float(style["legend_marker_size_points"]),
                    label=style["legend_label"],
                )
            )
    scalar = matplotlib.cm.ScalarMappable(norm=shared_norm, cmap=shared_cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.032, pad=0.02)
    colorbar.set_label(time_colour["colorbar_label"])
    layer_legend = ax.legend(
        handles=[
            Patch(
                facecolor=settings["road_cells"]["fill_color"],
                edgecolor=settings["road_cells"]["edge_color"],
                label=settings["road_cells"]["legend_label"],
            ),
            Patch(
                facecolor=shared_cmap(shared_norm((vmin + vmax) / 2.0)),
                edgecolor=settings["fire_cells"]["edge_color"],
                label=settings["fire_cells"]["legend_label"],
            ),
            Line2D(
                [0], [0], color=shared_cmap(shared_norm((vmin + vmax) / 2.0)),
                linewidth=2.5, label=line_settings["legend_label"],
            ),
            Patch(
                facecolor="none", edgecolor=settings["interaction_edge_color"],
                hatch=settings["interaction_hatch"], label=settings["interaction_legend_label"],
            )
        ],
        loc="lower right",
        framealpha=0.9,
    )
    ax.add_artist(layer_legend)
    count_min, count_max = float(counts.min()), float(counts.max())
    width_min = float(line_settings["minimum_width_points"])
    width_max = float(line_settings["maximum_width_points"])
    width_handles = []
    for value in line_settings["width_legend_values"]:
        if math.isclose(count_min, count_max):
            width = (width_min + width_max) / 2.0
        else:
            width = width_min + (float(value) - count_min) / (count_max - count_min) * (
                width_max - width_min
            )
        width_handles.append(
            Line2D(
                [0], [0], color="#30343b", linewidth=max(width_min, min(width_max, width)),
                label=str(value),
            )
        )
    width_legend = ax.legend(
        handles=width_handles,
        title=line_settings["width_legend_title"],
        loc="lower left",
        framealpha=0.9,
    )
    if endpoint_handles and endpoint_settings is not None:
        ax.add_artist(width_legend)
        ax.legend(
            handles=endpoint_handles,
            title=endpoint_settings["legend_title"],
            loc=endpoint_settings["legend_location"],
            framealpha=float(endpoint_settings["legend_frame_alpha"]),
        )
    ax.set_title(
        endpoint_settings["title"]
        if endpoint_settings is not None
        else settings["title"]
    )
    ax.set_xlabel(settings["x_label"])
    ax.set_ylabel(settings["y_label"])
    if settings["axis_equal"]:
        ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    bottom_margin = float(settings["caption_bottom_margin_fraction"])
    fig.text(
        0.5,
        bottom_margin / 3.0,
        settings["caption"],
        ha="center",
        va="bottom",
        fontsize=float(settings["caption_font_size"]),
    )
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, metadata=rendering_metadata, facecolor=fig.get_facecolor())
    plt.close(fig)
    result = {
        "time_ranges_seconds": time_ranges,
        "shared_normalization": [shared_norm.vmin, shared_norm.vmax],
        "shared_colormap": time_colour["colormap"],
        "colormap_lightness_direction": lightness_direction,
        "colorbar_count": 1,
        "road_grid_cell_count": int(len(road_cells)),
        "route_edge_count": int(len(route_activity)),
        "route_segment_count": int(len(route_segments)),
        "distinct_vehicle_count_range": [int(counts.min()), int(counts.max())],
        "route_line_width_range_points": [float(edge_widths.min()), float(edge_widths.max())],
    }
    if endpoint_points is not None and endpoint_settings is not None:
        result["endpoint_overlay"] = {
            "origin_count": int((endpoint_points["role"] == "origin").sum()),
            "destination_count": int((endpoint_points["role"] == "destination").sum()),
            "categorical_markers_not_time_encoded": True,
            "labels_visible": bool(endpoint_settings.get("show_labels", True)),
            "legend_count": 3,
        }
    return result


def parse_png_chunks(path: Path) -> list[tuple[str, bytes]]:
    payload = path.read_bytes()
    signature = b"\x89PNG\r\n\x1a\n"
    if not payload.startswith(signature):
        raise ValueError(f"Not a PNG file: {path}")
    position = len(signature)
    chunks: list[tuple[str, bytes]] = []
    while position < len(payload):
        if position + 12 > len(payload):
            raise ValueError(f"Truncated PNG chunk header: {path}")
        length = struct.unpack(">I", payload[position : position + 4])[0]
        kind = payload[position + 4 : position + 8].decode("ascii")
        end = position + 12 + length
        if end > len(payload):
            raise ValueError(f"Truncated PNG {kind} chunk: {path}")
        data = payload[position + 8 : position + 8 + length]
        chunks.append((kind, data))
        position = end
        if kind == "IEND":
            break
    return chunks


def compare_png(old: Path, new: Path) -> dict[str, Any]:
    if old.read_bytes() == new.read_bytes():
        return {"byte_identical": True, "idat_identical": True, "metadata_only": False, "differing_chunks": []}
    old_chunks = parse_png_chunks(old)
    new_chunks = parse_png_chunks(new)
    old_idat = b"".join(data for kind, data in old_chunks if kind == "IDAT")
    new_idat = b"".join(data for kind, data in new_chunks if kind == "IDAT")
    old_non_idat = [(kind, hashlib.sha256(data).hexdigest()) for kind, data in old_chunks if kind != "IDAT"]
    new_non_idat = [(kind, hashlib.sha256(data).hexdigest()) for kind, data in new_chunks if kind != "IDAT"]
    differing = sorted(
        set(kind for kind, _ in old_non_idat) ^ set(kind for kind, _ in new_non_idat)
        | {
            kind for kind in set(k for k, _ in old_non_idat) & set(k for k, _ in new_non_idat)
            if [digest for k, digest in old_non_idat if k == kind]
            != [digest for k, digest in new_non_idat if k == kind]
        }
    )
    idat_identical = old_idat == new_idat
    return {
        "byte_identical": False,
        "idat_identical": idat_identical,
        "metadata_only": idat_identical,
        "differing_chunks": differing,
    }


def rendering_library_versions() -> dict[str, str]:
    packages = {
        "matplotlib": "matplotlib", "numpy": "numpy", "pandas": "pandas",
        "pyarrow": "pyarrow", "shapely": "shapely", "pillow": "Pillow", "sumolib": "sumolib",
    }
    versions = {"python": platform.python_version()}
    for label, distribution in packages.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "unknown"
    return versions
