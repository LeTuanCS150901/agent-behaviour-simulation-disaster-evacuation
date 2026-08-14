from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import sumolib

from evacuation_sim.fire_sim.fire_grid import edge_linestring, routeable_edges
from evacuation_sim.io.tables import write_table


def build_edge_cell_mapping(
    net,
    grid: gpd.GeoDataFrame,
    vehicle_class: str,
    minimum_coverage_ratio: float,
    boundary_tolerance: float,
    incomplete_coverage_policy: str = "error",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if incomplete_coverage_policy != "error":
        raise ValueError("Only incomplete_coverage_policy='error' is supported.")
    sindex = grid.sindex
    rows = []
    summary_rows = []
    zero_intersection = []
    below_tolerance = []
    double_counted = []
    for edge in routeable_edges(net, vehicle_class):
        edge_id = edge.getID()
        line = edge_linestring(edge)
        edge_length = float(line.length)
        candidate_idx = list(sindex.query(line, predicate="intersects"))
        total_overlap = 0.0
        edge_rows = []
        for idx in candidate_idx:
            cell = grid.iloc[int(idx)]
            inter = line.intersection(cell.geometry)
            overlap = float(inter.length)
            if overlap <= boundary_tolerance:
                continue
            total_overlap += overlap
            edge_rows.append(
                {
                    "edge_id": edge_id,
                    "cell_id": cell.cell_id,
                    "overlap_length": overlap,
                    "edge_length": edge_length,
                    "sumo_reported_edge_length": float(edge.getLength()),
                    "overlap_fraction": overlap / edge_length,
                }
            )
        if total_overlap > edge_length + boundary_tolerance:
            double_counted.append(edge_id)
        coverage = total_overlap / edge_length if edge_length > 0 else 0.0
        if not edge_rows:
            zero_intersection.append(edge_id)
        if coverage + boundary_tolerance < minimum_coverage_ratio:
            below_tolerance.append(edge_id)
        for row in edge_rows:
            row["edge_coverage_ratio"] = coverage
            rows.append(row)
        summary_rows.append(
            {
                "edge_id": edge_id,
                "edge_length": edge_length,
                "sum_overlap_length": total_overlap,
                "edge_coverage_ratio": coverage,
                "intersection_records": len(edge_rows),
                "sumo_reported_edge_length": float(edge.getLength()),
            }
        )
    if double_counted:
        raise ValueError(
            "Probable edge-cell boundary double counting: summed overlap exceeds "
            f"edge length for {len(double_counted)} edges, examples={double_counted[:5]}"
        )
    if zero_intersection or below_tolerance:
        raise ValueError(
            "Incomplete fire-grid edge coverage: "
            f"zero_intersection={len(zero_intersection)}, "
            f"below_tolerance={len(below_tolerance)}, "
            f"examples={(zero_intersection + below_tolerance)[:5]}"
        )
    mapping = pd.DataFrame(rows)
    coverage = pd.DataFrame(summary_rows)
    summary = {
        "total_edges_processed": int(len(coverage)),
        "edges_with_zero_intersecting_cells": int(len(zero_intersection)),
        "edges_below_coverage_tolerance": int(len(below_tolerance)),
        "minimum_coverage_ratio": float(coverage["edge_coverage_ratio"].min()) if not coverage.empty else None,
        "mean_coverage_ratio": float(coverage["edge_coverage_ratio"].mean()) if not coverage.empty else None,
        "maximum_coverage_ratio": float(coverage["edge_coverage_ratio"].max()) if not coverage.empty else None,
        "total_intersection_records": int(len(mapping)),
        "boundary_double_counting_edges": 0,
    }
    return mapping, coverage, summary


def write_edge_cell_mapping_outputs(
    mapping: pd.DataFrame,
    coverage: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "edge_cell_intersections.parquet"
    coverage_path = output_dir / "edge_cell_coverage_summary.csv"
    summary_path = output_dir / "edge_cell_mapping_summary.json"
    write_table(mapping, mapping_path)
    coverage.to_csv(coverage_path, index=False)
    summary = dict(summary)
    summary["edge_cell_intersections_rows"] = int(len(mapping))
    summary["edge_cell_intersections_bytes"] = mapping_path.stat().st_size
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"mapping": str(mapping_path), "coverage": str(coverage_path), "summary": str(summary_path)}


def load_grid(path: str | Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def run_edge_cell_mapping(
    network_file: str,
    grid_path: str | Path,
    stage6_cfg: dict[str, Any],
    vehicle_class: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    net = sumolib.net.readNet(network_file)
    grid = load_grid(grid_path)
    cfg = stage6_cfg["edge_cell_mapping"]
    mapping, coverage, summary = build_edge_cell_mapping(
        net,
        grid,
        vehicle_class,
        float(cfg["minimum_coverage_ratio"]),
        float(cfg["boundary_tolerance"]),
        cfg["incomplete_coverage_policy"],
    )
    paths = write_edge_cell_mapping_outputs(mapping, coverage, summary, output_dir)
    summary["paths"] = paths
    return summary

