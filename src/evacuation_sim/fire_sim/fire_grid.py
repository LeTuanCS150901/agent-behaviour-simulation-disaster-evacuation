from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import sumolib
from shapely.geometry import LineString, box


def routeable_edges(net, vehicle_class: str = "passenger"):
    for edge in net.getEdges():
        edge_id = edge.getID()
        if edge.getFunction() or edge_id.startswith(":"):
            continue
        if any(lane.allows(vehicle_class) for lane in edge.getLanes()):
            yield edge


def edge_linestring(edge) -> LineString:
    shape = edge.getShape()
    if not shape or len(shape) < 2:
        raise ValueError(f"SUMO edge {edge.getID()} has no usable geometry.")
    line = LineString([(float(x), float(y)) for x, y in shape])
    if line.length <= 0:
        raise ValueError(f"SUMO edge {edge.getID()} has zero-length geometry.")
    return line


def edge_coordinate(edge) -> tuple[float, float]:
    line = edge_linestring(edge)
    point = line.interpolate(0.5, normalized=True)
    return float(point.x), float(point.y)


def build_fire_grid(net, stage6_cfg: dict[str, Any]) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    grid_cfg = stage6_cfg["fire_grid"]
    cell_width = float(grid_cfg["cell_width"])
    cell_height = float(grid_cfg["cell_height"])
    padding = float(grid_cfg["padding"])
    row_origin = grid_cfg["row_origin"]
    if row_origin != "top":
        raise ValueError(f"Unsupported row_origin={row_origin!r}; only 'top' is implemented.")

    xmin, ymin, xmax, ymax = map(float, net.getBoundary())
    grid_xmin = xmin - padding
    grid_ymin = ymin - padding
    grid_xmax = xmax + padding
    grid_ymax = ymax + padding
    n_cols = int(math.ceil((grid_xmax - grid_xmin) / cell_width))
    n_rows = int(math.ceil((grid_ymax - grid_ymin) / cell_height))
    grid_xmax = grid_xmin + n_cols * cell_width
    grid_ymin = grid_ymax - n_rows * cell_height

    rows = []
    for row in range(n_rows):
        y_top = grid_ymax - row * cell_height
        y_bottom = y_top - cell_height
        for col in range(n_cols):
            x_left = grid_xmin + col * cell_width
            x_right = x_left + cell_width
            rows.append(
                {
                    "cell_id": f"r{row:03d}_c{col:03d}",
                    "row": row,
                    "column": col,
                    "xmin": x_left,
                    "ymin": y_bottom,
                    "xmax": x_right,
                    "ymax": y_top,
                    "center_x": x_left + cell_width / 2.0,
                    "center_y": y_top - cell_height / 2.0,
                    "geometry": box(x_left, y_bottom, x_right, y_top),
                }
            )
    grid = gpd.GeoDataFrame(rows, geometry="geometry", crs=None)
    metadata = {
        "network_bbox": [xmin, ymin, xmax, ymax],
        "grid_bbox": [grid_xmin, grid_ymin, grid_xmax, grid_ymax],
        "cell_width": cell_width,
        "cell_height": cell_height,
        "number_of_rows": n_rows,
        "number_of_columns": n_cols,
        "row_origin": row_origin,
        "x_direction": "increasing_column",
        "y_direction": "decreasing_row",
        "affine_transform": {
            "center_x": "xmin + (column + 0.5) * cell_width",
            "center_y": "ymax - (row + 0.5) * cell_height",
        },
        "sumo_projection": str(net.getGeoProj()),
        "sumo_location_offset": [float(v) for v in net.getLocationOffset()],
    }
    return grid, metadata


def cell_center_from_row_col(metadata: dict[str, Any], row: int, column: int) -> tuple[float, float]:
    xmin, _ymin, _xmax, ymax = metadata["grid_bbox"]
    return (
        float(xmin) + (column + 0.5) * float(metadata["cell_width"]),
        float(ymax) - (row + 0.5) * float(metadata["cell_height"]),
    )


def write_grid_outputs(
    grid: gpd.GeoDataFrame,
    metadata: dict[str, Any],
    net,
    ignition_edges: list[str],
    output_dir: str | Path,
    vehicle_class: str = "passenger",
) -> dict[str, str]:
    output_dir = Path(output_dir)
    grid_dir = output_dir / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    grid_path = grid_dir / "fire_grid.geojson"
    metadata_path = grid_dir / "fire_grid_metadata.json"
    overlay_path = grid_dir / "fire_grid_network_overlay.png"

    grid.to_file(grid_path, driver="GeoJSON")
    metadata = dict(metadata)
    metadata["grid_geojson_sha256"] = sha256_file(grid_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_grid_network_overlay(grid, net, ignition_edges, overlay_path, vehicle_class)
    return {"grid": str(grid_path), "metadata": str(metadata_path), "overlay": str(overlay_path)}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_grid_network_overlay(grid, net, ignition_edges: list[str], path: str | Path, vehicle_class: str = "passenger") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    grid.boundary.plot(ax=ax, color="#dddddd", linewidth=0.2)
    xmin, ymin, xmax, ymax = net.getBoundary()
    ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color="black", linewidth=1.0)
    for edge in routeable_edges(net, vehicle_class):
        line = edge_linestring(edge)
        xs, ys = line.xy
        ax.plot(xs, ys, color="#555555", linewidth=0.35)
    ignition_points = []
    for edge_id in ignition_edges:
        x, y = edge_coordinate(net.getEdge(edge_id))
        ignition_points.append((x, y))
    if ignition_points:
        xs, ys = zip(*ignition_points)
        ax.scatter(xs, ys, color="red", s=40, label="ignition edges", zorder=4)
        ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SUMO x")
    ax.set_ylabel("SUMO y")
    ax.set_title("Fire grid and SUMO network overlay")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def grid_file_sizes(output_dir: str | Path) -> list[dict[str, Any]]:
    base = Path(output_dir)
    rows = []
    for path in sorted(base.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path), "bytes": path.stat().st_size})
    return rows
