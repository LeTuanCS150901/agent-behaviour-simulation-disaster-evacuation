from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import sumolib
from matplotlib.patches import Polygon as MplPolygon

from evacuation_sim.fire_sim.cell_states import validate_simfire_output_contract
from evacuation_sim.fire_sim.fire_grid import edge_linestring, routeable_edges
from evacuation_sim.io.tables import read_table, write_table


def compute_edge_hazard(
    mapping: pd.DataFrame,
    cell_states: pd.DataFrame,
    lambda_value: float,
) -> pd.DataFrame:
    merged = mapping.merge(cell_states[["time", "cell_id", "hazard_value"]], on="cell_id", how="left")
    if merged["hazard_value"].isna().any():
        missing = merged.loc[merged["hazard_value"].isna(), "cell_id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Missing cell hazards for mapped cells: {missing}")
    weighted = merged["hazard_value"] * merged["overlap_length"]
    merged = merged.assign(weighted_hazard=weighted)
    grouped = (
        merged.groupby(["time", "edge_id"], as_index=False)
        .agg(
            weighted_hazard_sum=("weighted_hazard", "sum"),
            edge_length=("edge_length", "first"),
            coverage_ratio=("edge_coverage_ratio", "first"),
        )
    )
    grouped["edge_hazard"] = grouped["weighted_hazard_sum"] / grouped["edge_length"]
    if not grouped["edge_hazard"].between(0.0, 1.0).all():
        bad = grouped.loc[~grouped["edge_hazard"].between(0.0, 1.0)].head(5).to_dict("records")
        raise ValueError(f"Computed edge_hazard outside [0,1]: {bad}")
    grouped["lambda"] = float(lambda_value)
    grouped["edge_survival"] = grouped["edge_hazard"].apply(lambda h: math.exp(-float(lambda_value) * float(h)))
    grouped["edge_risk"] = 1.0 - grouped["edge_survival"]
    return grouped[
        ["time", "edge_id", "edge_hazard", "edge_survival", "edge_risk", "edge_length", "coverage_ratio", "lambda"]
    ]


def write_edge_hazard_outputs(edge_hazard: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "edge_hazard_time_series.parquet"
    summary_path = output_dir / "edge_hazard_summary.json"
    diagnostic_path = output_dir / "edge_hazard_diagnostic.png"
    write_table(edge_hazard, path)
    summary = {
        "rows": int(len(edge_hazard)),
        "file_bytes": path.stat().st_size,
        "time_min": float(edge_hazard["time"].min()),
        "time_max": float(edge_hazard["time"].max()),
        "edge_count": int(edge_hazard["edge_id"].nunique()),
        "edge_hazard_min": float(edge_hazard["edge_hazard"].min()),
        "edge_hazard_max": float(edge_hazard["edge_hazard"].max()),
        "edge_survival_min": float(edge_hazard["edge_survival"].min()),
        "edge_survival_max": float(edge_hazard["edge_survival"].max()),
        "edge_risk_min": float(edge_hazard["edge_risk"].min()),
        "edge_risk_max": float(edge_hazard["edge_risk"].max()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_edge_hazard_diagnostic(edge_hazard, diagnostic_path)
    summary["paths"] = {
        "edge_hazard_time_series": str(path),
        "edge_hazard_summary": str(summary_path),
        "edge_hazard_diagnostic": str(diagnostic_path),
    }
    return summary


def write_edge_hazard_diagnostic(edge_hazard: pd.DataFrame, path: str | Path, max_edges: int = 8) -> None:
    path = Path(path)
    selected_edges = (
        edge_hazard.groupby("edge_id")["edge_hazard"].max().sort_values(ascending=False).head(max_edges).index.tolist()
    )
    fig, ax1 = plt.subplots(figsize=(8, 4))
    for edge_id in selected_edges:
        subset = edge_hazard[edge_hazard["edge_id"] == edge_id]
        ax1.plot(subset["time"], subset["edge_hazard"], label=edge_id, alpha=0.8)
    ax1.set_xlabel("time")
    ax1.set_ylabel("edge hazard H_e(t)")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_fire_front_from_cells(cell_states: pd.DataFrame, grid: gpd.GeoDataFrame, path: str | Path) -> None:
    burning = cell_states[cell_states["fire_state"] == "burning"][["time", "cell_id"]]
    joined = burning.merge(grid[["cell_id", "geometry", "center_x", "center_y"]], on="cell_id", how="left")
    gdf = gpd.GeoDataFrame(joined, geometry="geometry", crs=None)
    gdf.to_file(path, driver="GeoJSON")


def write_fire_addxml_from_cells(cell_states: pd.DataFrame, grid: gpd.GeoDataFrame, path: str | Path, poi_size: float) -> None:
    burning = cell_states[cell_states["fire_state"] == "burning"][["time", "cell_id"]]
    centers = burning.merge(grid[["cell_id", "center_x", "center_y"]], on="cell_id", how="left")
    root = ET.Element("additional")
    for idx, row in enumerate(centers.itertuples(index=False)):
        ET.SubElement(
            root,
            "poi",
            {
                "id": f"fire_cell_{idx}",
                "type": "fire",
                "color": "255,0,0",
                "layer": "100",
                "x": f"{float(row.center_x):.3f}",
                "y": f"{float(row.center_y):.3f}",
                "width": f"{float(poi_size):.3f}",
                "height": f"{float(poi_size):.3f}",
            },
        )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_fire_network_overlay_from_cells(cell_states, grid, net, path: str | Path, vehicle_class: str = "passenger") -> None:
    path = Path(path)
    latest_time = cell_states["time"].max()
    latest = cell_states[(cell_states["time"] == latest_time) & (cell_states["fire_state"] == "burning")]
    burning_grid = grid[grid["cell_id"].isin(latest["cell_id"])]
    fig, ax = plt.subplots(figsize=(9, 9))
    for edge in routeable_edges(net, vehicle_class):
        line = edge_linestring(edge)
        xs, ys = line.xy
        ax.plot(xs, ys, color="#bbbbbb", linewidth=0.35)
    if not burning_grid.empty:
        for geom in burning_grid.geometry:
            if geom.is_empty:
                continue
            polygons = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            for poly in polygons:
                coords = list(poly.exterior.coords)
                ax.add_patch(MplPolygon(coords, closed=True, facecolor="red", edgecolor="red", alpha=0.35))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SUMO x")
    ax.set_ylabel("SUMO y")
    ax.set_title(f"Fire cells and SUMO network at t={latest_time}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def addxml_loads_in_sumo(network_file: str, addxml: Path, output_dir: Path) -> dict[str, Any]:
    import subprocess

    log_path = output_dir / "sumo_addxml_load.log"
    cmd = ["sumo", "-n", network_file, "-a", str(addxml), "--begin", "0", "--end", "1", "--no-step-log", "true"]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    log_path.write_text(
        "COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    check = {
        "command": cmd,
        "returncode": int(result.returncode),
        "loads_in_sumo": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "log": str(log_path),
    }
    (output_dir / "sumo_additional_file_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    return check


def run_compute_edge_hazard(
    stage6_cfg: dict[str, Any],
    stage6_dir: str | Path,
) -> dict[str, Any]:
    stage6_dir = Path(stage6_dir)
    manifest = stage6_dir / "simfire" / "simfire_input_manifest.json"
    cell_path = stage6_dir / "simfire" / "fire_cell_time_series.parquet"
    metadata_path = stage6_dir / "simfire" / "simfire_run_metadata.json"
    contract = validate_simfire_output_contract(manifest, cell_path, metadata_path)
    mapping = read_table(stage6_dir / "edge_cell_intersections.parquet")
    cells = read_table(cell_path)
    edge_hazard = compute_edge_hazard(mapping, cells, float(stage6_cfg["edge_survival"]["lambda"]))
    summary = write_edge_hazard_outputs(edge_hazard, stage6_dir)
    summary["cell_output_contract"] = contract
    return summary
