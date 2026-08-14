from __future__ import annotations

import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sumolib

from evacuation_sim.io.tables import write_table


def edge_midpoint(edge):
    shape = edge.getShape()
    if not shape:
        return (0.0, 0.0)
    return shape[len(shape) // 2]


def generate_mock_fire(net, ignition_edges: list[str], times=(0, 60, 120, 180)) -> pd.DataFrame:
    rows = []
    for t in times:
        radius = 0.8 * t
        for edge_id in ignition_edges:
            edge = net.getEdge(edge_id)
            x, y = edge_midpoint(edge)
            rows.append({"time": float(t), "edge_id": edge_id, "x": float(x), "y": float(y), "radius": float(radius)})
    return pd.DataFrame(rows)


def write_geojson(points: pd.DataFrame, path: Path) -> None:
    features = []
    for row in points.itertuples(index=False):
        features.append(
            {
                "type": "Feature",
                "properties": {"time": row.time, "edge_id": row.edge_id, "radius": row.radius},
                "geometry": {"type": "Point", "coordinates": [row.x, row.y]},
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def write_fire_addxml(points: pd.DataFrame, path: Path) -> None:
    root = ET.Element("additional")
    for idx, row in enumerate(points.itertuples(index=False)):
        ET.SubElement(
            root,
            "poi",
            {
                "id": f"fire_{idx}",
                "type": "fire",
                "color": "255,0,0",
                "layer": "100",
                "x": f"{row.x:.3f}",
                "y": f"{row.y:.3f}",
                "width": f"{max(row.radius, 5.0):.3f}",
                "height": f"{max(row.radius, 5.0):.3f}",
            },
        )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def compute_edge_distance_to_fire(net, points: pd.DataFrame) -> pd.DataFrame:
    latest_time = points["time"].max()
    latest = points[points["time"] == latest_time]
    fire_points = [(row.x, row.y) for row in latest.itertuples(index=False)]
    rows = []
    for edge in net.getEdges():
        if edge.getFunction() or edge.getID().startswith(":"):
            continue
        x, y = edge_midpoint(edge)
        distance = min(math.hypot(x - fx, y - fy) for fx, fy in fire_points)
        rows.append({"edge_id": edge.getID(), "time": float(latest_time), "distance_to_fire": float(distance)})
    return pd.DataFrame(rows)


def write_overlay(net, points: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 8))
    for edge in net.getEdges():
        if edge.getFunction() or edge.getID().startswith(":"):
            continue
        shape = edge.getShape()
        if len(shape) >= 2:
            xs, ys = zip(*shape)
            plt.plot(xs, ys, color="#cccccc", linewidth=0.4)
    plt.scatter(points["x"], points["y"], c=points["time"], cmap="autumn", s=30)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def addxml_loads_in_sumo(network_file: str, addxml: Path, output_dir: Path) -> tuple[bool, str]:
    log_path = output_dir / "sumo_addxml_load.log"
    cmd = ["sumo", "-n", network_file, "-a", str(addxml), "--begin", "0", "--end", "1", "--no-step-log", "true"]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    log_path.write_text("COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr, encoding="utf-8")
    return result.returncode == 0, str(log_path)


def run_stage6_manhattan(network_file: str, ignition_edges: list[str], output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net = sumolib.net.readNet(network_file)
    edge_ids = {edge.getID() for edge in net.getEdges()}
    missing = [edge for edge in ignition_edges if edge not in edge_ids]
    if missing:
        raise ValueError(f"Ignition edge IDs do not exist in Manhattan network: {missing}")

    points = generate_mock_fire(net, ignition_edges)
    xmin, ymin, xmax, ymax = net.getBoundary()
    near_bbox = points["x"].between(xmin - 100, xmax + 100).all() and points["y"].between(ymin - 100, ymax + 100).all()
    write_table(points, output_dir / "fire_time_series.parquet")
    write_geojson(points, output_dir / "fire_front_test_local.geojson")
    addxml = output_dir / "fire_hazard.add.xml"
    write_fire_addxml(points, addxml)
    ET.parse(addxml)
    distances = compute_edge_distance_to_fire(net, points)
    write_table(distances, output_dir / "edge_distance_to_fire.parquet")
    write_overlay(net, points, output_dir / "fire_network_overlay.png")
    sumo_loaded, log_path = addxml_loads_in_sumo(network_file, addxml, output_dir)

    summary = {
        "stage": "stage6_manhattan",
        "fire_mode": "TEST_LOCAL deterministic mock-fire",
        "real_simfire_validation": "not_run_simfire_missing",
        "ignition_edges": ignition_edges,
        "hazard_points": len(points),
        "hazard_points_inside_or_near_bbox": bool(near_bbox),
        "addxml_valid_xml": True,
        "addxml_loads_in_sumo": bool(sumo_loaded),
        "sumo_addxml_log": log_path,
    }
    (output_dir / "stage6_manhattan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
