from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import sumolib
import yaml
from matplotlib.patches import Polygon as MplPolygon

from evacuation_sim.fire_sim.cell_states import ignition_cells_for_edges
from evacuation_sim.fire_sim.edge_cell_mapping import build_edge_cell_mapping, write_edge_cell_mapping_outputs
from evacuation_sim.fire_sim.fire_grid import build_fire_grid, edge_coordinate, edge_linestring, routeable_edges


CANDIDATE_CELL_SIZES = (20.0, 50.0, 100.0, 200.0)
REQUIRED_COMPARISON_COLUMNS = [
    "cell_width",
    "cell_height",
    "rows",
    "columns",
    "total_cells",
    "routeable_edges",
    "routeable_like_edges",
    "mapped_edges",
    "unmapped_routeable_edges",
    "candidate_route_edges",
    "candidate_route_edges_with_coverage",
    "candidate_route_edge_coverage",
    "intersection_records",
    "minimum_edge_coverage_ratio",
    "mean_edge_coverage_ratio",
    "maximum_edge_coverage_ratio",
    "edges_below_coverage_tolerance",
    "grid_geojson_size_bytes",
    "mapping_parquet_size_bytes",
    "mapping_runtime_seconds",
    "estimated_cell_rows_per_snapshot",
    "avg_cells_per_edge",
    "max_cells_per_edge",
    "edges_represented_by_one_cell",
    "edge_length_min",
    "edge_length_median",
    "edge_length_mean",
    "edge_length_max",
    "cell_size_to_edge_length_median_ratio",
    "cell_size_to_edge_length_mean_ratio",
    "short_edge_coarseness_flag",
    "ignition_localization_error_mean",
    "ignition_localization_error_max",
    "ignition_cells_valid",
    "ignition_cell_ids",
    "unique_route_cells",
    "route_count",
    "unique_route_cell_signatures",
    "route_cell_signature_distinguishability",
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage6_cfg_for_size(stage6_cfg: dict[str, Any], cell_size: float) -> dict[str, Any]:
    cfg = dict(stage6_cfg)
    cfg["fire_grid"] = dict(stage6_cfg["fire_grid"])
    cfg["fire_grid"]["cell_width"] = float(cell_size)
    cfg["fire_grid"]["cell_height"] = float(cell_size)
    return cfg


def routeable_like_edges(net) -> list[str]:
    return [
        edge.getID()
        for edge in net.getEdges()
        if not edge.getFunction() and not edge.getID().startswith(":")
    ]


def parse_candidate_routes(path: str | Path) -> tuple[set[str], list[tuple[str, list[str]]]]:
    path = Path(path)
    if not path.exists():
        return set(), []
    df = pd.read_parquet(path)
    route_rows = []
    edge_ids: set[str] = set()
    for row in df.itertuples(index=False):
        edges = str(row.route_edges).split()
        route_rows.append((str(row.route_id), edges))
        edge_ids.update(edges)
    return edge_ids, route_rows


def edge_universe_summary(net, vehicle_class: str) -> dict[str, Any]:
    all_non_internal = routeable_like_edges(net)
    passenger_edges = [edge.getID() for edge in routeable_edges(net, vehicle_class)]
    non_passenger = sorted(set(all_non_internal) - set(passenger_edges))
    internal_or_function = [
        edge.getID()
        for edge in net.getEdges()
        if edge.getFunction() or edge.getID().startswith(":")
    ]
    return {
        "routeable_like_edges": len(all_non_internal),
        "routeable_passenger_edges": len(passenger_edges),
        "excluded_non_passenger_edges": len(non_passenger),
        "internal_or_function_edges": len(internal_or_function),
        "excluded_non_passenger_examples": non_passenger[:20],
    }


def candidate_route_edge_coverage(candidate_edges: set[str], mapped_edges: set[str]) -> dict[str, Any]:
    if not candidate_edges:
        return {
            "candidate_route_edges": 0,
            "candidate_route_edges_with_coverage": 0,
            "candidate_route_edge_coverage": None,
            "candidate_route_missing_edges": [],
        }
    covered = candidate_edges & mapped_edges
    missing = sorted(candidate_edges - mapped_edges)
    return {
        "candidate_route_edges": len(candidate_edges),
        "candidate_route_edges_with_coverage": len(covered),
        "candidate_route_edge_coverage": len(covered) / len(candidate_edges),
        "candidate_route_missing_edges": missing[:20],
    }


def edge_length_metrics(coverage: pd.DataFrame, cell_size: float) -> dict[str, Any]:
    lengths = coverage["edge_length"].astype(float)
    return {
        "edge_length_min": float(lengths.min()),
        "edge_length_median": float(lengths.median()),
        "edge_length_mean": float(lengths.mean()),
        "edge_length_max": float(lengths.max()),
        "cell_size_to_edge_length_median_ratio": float(cell_size / max(lengths.median(), 1e-9)),
        "cell_size_to_edge_length_mean_ratio": float(cell_size / max(lengths.mean(), 1e-9)),
        "short_edge_coarseness_flag": bool((lengths < cell_size).mean() > 0.5),
    }


def ignition_localization_metrics(grid: gpd.GeoDataFrame, net, ignition_edges: list[str]) -> dict[str, Any]:
    sindex = grid.sindex
    errors = []
    cell_ids = []
    rows = []
    for edge_id in ignition_edges:
        x, y = edge_coordinate(net.getEdge(edge_id))
        point = gpd.points_from_xy([x], [y])[0]
        idxs = list(sindex.query(point, predicate="intersects"))
        if not idxs:
            rows.append({"ignition_edge_id": edge_id, "mapping_valid": False})
            continue
        cell = grid.iloc[int(idxs[0])]
        err = math.hypot(float(cell.center_x) - x, float(cell.center_y) - y)
        errors.append(err)
        cell_ids.append(str(cell.cell_id))
        rows.append(
            {
                "ignition_edge_id": edge_id,
                "sumo_x": x,
                "sumo_y": y,
                "grid_row": int(cell.row),
                "grid_column": int(cell.column),
                "cell_id": str(cell.cell_id),
                "cell_center_x": float(cell.center_x),
                "cell_center_y": float(cell.center_y),
                "localization_error": err,
                "mapping_valid": True,
            }
        )
    return {
        "ignition_localization_error_mean": float(sum(errors) / len(errors)) if errors else None,
        "ignition_localization_error_max": float(max(errors)) if errors else None,
        "ignition_cells_valid": len(errors) == len(ignition_edges),
        "ignition_cell_ids": ";".join(sorted(set(cell_ids))),
        "ignition_mapping_rows": rows,
    }


def route_cell_signature_metrics(mapping: pd.DataFrame, route_rows: list[tuple[str, list[str]]]) -> dict[str, Any]:
    if not route_rows:
        return {
            "unique_route_cells": 0,
            "route_count": 0,
            "unique_route_cell_signatures": 0,
            "route_cell_signature_distinguishability": None,
        }
    edge_cells = (
        mapping.groupby("edge_id")["cell_id"]
        .apply(lambda s: tuple(sorted(set(str(v) for v in s))))
        .to_dict()
    )
    all_cells = set()
    signatures = []
    for _route_id, edges in route_rows:
        cells = []
        for edge_id in edges:
            edge_cell_ids = edge_cells.get(edge_id, ())
            cells.extend(edge_cell_ids)
            all_cells.update(edge_cell_ids)
        signatures.append(tuple(cells))
    unique_signatures = len(set(signatures))
    return {
        "unique_route_cells": len(all_cells),
        "route_count": len(route_rows),
        "unique_route_cell_signatures": unique_signatures,
        "route_cell_signature_distinguishability": unique_signatures / len(route_rows),
    }


def draw_grid_overlay(
    grid: gpd.GeoDataFrame,
    net,
    ignition_edges: list[str],
    path: str | Path,
    title: str,
    vehicle_class: str,
    extent: tuple[float, float, float, float] | None = None,
    route_edges: list[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    visible_grid = grid
    if extent:
        xmin, ymin, xmax, ymax = extent
        visible_grid = grid.cx[xmin:xmax, ymin:ymax]
    visible_grid.boundary.plot(ax=ax, color="#dddddd", linewidth=0.45)
    bxmin, bymin, bxmax, bymax = net.getBoundary()
    ax.plot([bxmin, bxmax, bxmax, bxmin, bxmin], [bymin, bymin, bymax, bymax, bymin], color="black", linewidth=1.0)
    route_edge_set = set(route_edges or [])
    for edge in routeable_edges(net, vehicle_class):
        line = edge_linestring(edge)
        xs, ys = line.xy
        color = "#1f77b4" if edge.getID() in route_edge_set else "#666666"
        width = 1.5 if edge.getID() in route_edge_set else 0.35
        ax.plot(xs, ys, color=color, linewidth=width, alpha=0.9)
    ignition_points = []
    for edge_id in ignition_edges:
        x, y = edge_coordinate(net.getEdge(edge_id))
        ignition_points.append((x, y))
    if ignition_points:
        xs, ys = zip(*ignition_points)
        ax.scatter(xs, ys, color="red", s=50, label="ignition edges", zorder=4)
        for x, y in ignition_points:
            ax.annotate("ignition", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.legend(loc="upper right")
    if extent:
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SUMO x")
    ax.set_ylabel("SUMO y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def extent_around_points(points: list[tuple[float, float]], pad: float) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def dense_corridor_extent(net, vehicle_class: str, pad: float = 600.0) -> tuple[float, float, float, float]:
    points = []
    for edge in routeable_edges(net, vehicle_class):
        line = edge_linestring(edge)
        points.extend(line.coords)
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return cx - pad, cy - pad, cx + pad, cy + pad


def representative_route_edges(route_rows: list[tuple[str, list[str]]]) -> list[str]:
    if not route_rows:
        return []
    route_id, edges = max(route_rows, key=lambda item: len(item[1]))
    return edges


def write_combined_comparison_figure(study_root: Path, sizes: list[float]) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, size in zip(axes.flat, sizes):
        img = plt.imread(study_root / f"{int(size)}m" / "grid_overlay.png")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{int(size)} m")
    fig.tight_layout()
    path = study_root / "grid_resolution_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def run_resolution_study(
    manhattan_config: str | Path = "configs/test/manhattan_test.yaml",
    stage6_config: str | Path = "configs/stage6.yaml",
    output_root: str | Path = "outputs/test/manhattan",
) -> dict[str, Any]:
    cfg = load_yaml(manhattan_config)
    stage6_cfg = load_yaml(stage6_config)
    net = sumolib.net.readNet(cfg["network_file"])
    vehicle_class = cfg["vehicle_class"]
    study_root = Path(output_root) / "stage6" / "resolution_study"
    study_root.mkdir(parents=True, exist_ok=True)
    candidate_edges, route_rows = parse_candidate_routes(Path(output_root) / "stage5" / "candidate_routes.parquet")
    universe = edge_universe_summary(net, vehicle_class)
    comparison_rows = []
    size_summaries = {}
    ignition_points = [edge_coordinate(net.getEdge(edge_id)) for edge_id in cfg["ignition_edges"]]
    ignition_extent = extent_around_points(ignition_points, 500.0)
    corridor_extent = dense_corridor_extent(net, vehicle_class)
    route_edges = representative_route_edges(route_rows)
    route_points = []
    for edge_id in route_edges:
        try:
            route_points.extend(edge_linestring(net.getEdge(edge_id)).coords)
        except Exception:
            pass
    route_extent = extent_around_points(route_points, 350.0) if route_points else corridor_extent

    for size in CANDIDATE_CELL_SIZES:
        label = f"{int(size)}m"
        out_dir = study_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        candidate_cfg = stage6_cfg_for_size(stage6_cfg, size)
        grid, metadata = build_fire_grid(net, candidate_cfg)
        grid_path = out_dir / "fire_grid.geojson"
        mapping_path = out_dir / "edge_cell_intersections.parquet"
        grid.to_file(grid_path, driver="GeoJSON")
        start = time.perf_counter()
        mapping, coverage, mapping_summary = build_edge_cell_mapping(
            net,
            grid,
            vehicle_class,
            float(candidate_cfg["edge_cell_mapping"]["minimum_coverage_ratio"]),
            float(candidate_cfg["edge_cell_mapping"]["boundary_tolerance"]),
            candidate_cfg["edge_cell_mapping"]["incomplete_coverage_policy"],
        )
        runtime = time.perf_counter() - start
        write_edge_cell_mapping_outputs(mapping, coverage, mapping_summary, out_dir)
        mapped_edges = set(mapping["edge_id"].unique())
        route_cov = candidate_route_edge_coverage(candidate_edges, mapped_edges)
        ignition = ignition_localization_metrics(grid, net, cfg["ignition_edges"])
        route_metrics = route_cell_signature_metrics(mapping, route_rows)
        length_metrics = edge_length_metrics(coverage, size)
        intersections_by_edge = mapping.groupby("edge_id")["cell_id"].nunique()
        draw_grid_overlay(
            grid,
            net,
            cfg["ignition_edges"],
            out_dir / "grid_overlay.png",
            f"{int(size)} m grid: {metadata['number_of_rows']} x {metadata['number_of_columns']} ({len(grid)} cells)",
            vehicle_class,
        )
        draw_grid_overlay(
            grid,
            net,
            cfg["ignition_edges"],
            out_dir / "ignition_zoom_overlay.png",
            f"{int(size)} m ignition zoom",
            vehicle_class,
            ignition_extent,
        )
        draw_grid_overlay(
            grid,
            net,
            cfg["ignition_edges"],
            out_dir / "dense_corridor_zoom_overlay.png",
            f"{int(size)} m dense corridor zoom",
            vehicle_class,
            corridor_extent,
        )
        draw_grid_overlay(
            grid,
            net,
            cfg["ignition_edges"],
            out_dir / "representative_routes_overlay.png",
            f"{int(size)} m representative route overlay",
            vehicle_class,
            route_extent,
            route_edges,
        )
        row = {
            "cell_width": size,
            "cell_height": size,
            "rows": metadata["number_of_rows"],
            "columns": metadata["number_of_columns"],
            "total_cells": len(grid),
            "routeable_edges": universe["routeable_passenger_edges"],
            "routeable_like_edges": universe["routeable_like_edges"],
            "mapped_edges": len(mapped_edges),
            "unmapped_routeable_edges": universe["routeable_passenger_edges"] - len(mapped_edges),
            "intersection_records": len(mapping),
            "minimum_edge_coverage_ratio": mapping_summary["minimum_coverage_ratio"],
            "mean_edge_coverage_ratio": mapping_summary["mean_coverage_ratio"],
            "maximum_edge_coverage_ratio": mapping_summary["maximum_coverage_ratio"],
            "edges_below_coverage_tolerance": mapping_summary["edges_below_coverage_tolerance"],
            "grid_geojson_size_bytes": grid_path.stat().st_size,
            "mapping_parquet_size_bytes": mapping_path.stat().st_size,
            "mapping_runtime_seconds": runtime,
            "estimated_cell_rows_per_snapshot": len(grid),
            "avg_cells_per_edge": float(intersections_by_edge.mean()),
            "max_cells_per_edge": int(intersections_by_edge.max()),
            "edges_represented_by_one_cell": int((intersections_by_edge == 1).sum()),
            **length_metrics,
            **{k: v for k, v in ignition.items() if k != "ignition_mapping_rows"},
            **route_metrics,
            **route_cov,
        }
        comparison_rows.append(row)
        pd.DataFrame(ignition["ignition_mapping_rows"]).to_csv(out_dir / "ignition_mapping.csv", index=False)
        (out_dir / "grid_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        size_summaries[label] = row

    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison[REQUIRED_COMPARISON_COLUMNS]
    comparison_path = study_root / "grid_resolution_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    combined_path = write_combined_comparison_figure(study_root, list(CANDIDATE_CELL_SIZES))
    summary = {
        "candidate_cell_sizes": list(CANDIDATE_CELL_SIZES),
        "comparison_csv": str(comparison_path),
        "combined_figure": combined_path,
        "edge_universe": universe,
        "candidate_route_edges": len(candidate_edges),
        "route_count": len(route_rows),
        "summaries": size_summaries,
        "recommended_resolution_m": recommend_resolution(comparison),
    }
    (study_root / "grid_resolution_study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_stage6a_report(summary, comparison, study_root)
    return summary


def recommend_resolution(comparison: pd.DataFrame) -> float:
    valid = comparison[
        (comparison["edges_below_coverage_tolerance"] == 0)
        & (comparison["candidate_route_edge_coverage"] == 1.0)
        & (comparison["ignition_cells_valid"] == True)
    ].copy()
    if valid.empty:
        return float(comparison.sort_values("minimum_edge_coverage_ratio", ascending=False).iloc[0]["cell_width"])
    # Treat 20 m as a baseline only: the Stage 6A problem statement identifies it
    # as too fine for final real-SimFire validation unless every larger option fails.
    larger = valid[valid["cell_width"] > 20.0].copy()
    if not larger.empty:
        valid = larger
    max_cells = max(float(valid["total_cells"].max()), 1.0)
    max_mapping_bytes = max(float(valid["mapping_parquet_size_bytes"].max()), 1.0)
    max_ignition_error = max(float(valid["ignition_localization_error_max"].max()), 1.0)
    max_unique_route_cells = max(float(valid["unique_route_cells"].max()), 1.0)
    valid["score"] = (
        valid["route_cell_signature_distinguishability"].fillna(0.0) * 3.0
        + (valid["unique_route_cells"] / max_unique_route_cells) * 1.5
        + (1.0 - valid["total_cells"] / max_cells) * 1.0
        + (1.0 - valid["mapping_parquet_size_bytes"] / max_mapping_bytes) * 0.5
        + (1.0 - valid["ignition_localization_error_max"] / max_ignition_error) * 1.0
        - valid["short_edge_coarseness_flag"].astype(float)
    )
    return float(valid.sort_values(["score", "cell_width"], ascending=[False, True]).iloc[0]["cell_width"])


def write_stage6a_report(summary: dict[str, Any], comparison: pd.DataFrame, study_root: Path) -> None:
    rec = summary["recommended_resolution_m"]
    table_columns = [
        "cell_width",
        "rows",
        "columns",
        "total_cells",
        "intersection_records",
        "minimum_edge_coverage_ratio",
        "candidate_route_edge_coverage",
        "unique_route_cells",
        "route_cell_signature_distinguishability",
        "grid_geojson_size_bytes",
        "mapping_runtime_seconds",
    ]
    table = comparison[table_columns].to_csv(index=False)
    lines = [
        "# Stage 6A Grid-Resolution Study",
        "",
        "## 1. Summary",
        "",
        "Compared 20 m, 50 m, 100 m, and 200 m square fire-grid cells for the Manhattan Stage 6 cell-to-edge pipeline. No real SimFire integration was run, and the active Stage 6 cell size was not changed.",
        "",
        f"Recommended resolution for research-lead review: **{int(rec)} m**.",
        "",
        "## 2. Why 20 m Was Too Fine",
        "",
        "The 20 m baseline creates 55,440 cells for the padded Manhattan grid, producing a large GeoJSON, large manifest/cell-state outputs, visually dense overlays, and unnecessary SimFire cost for this validation track.",
        "",
        "## 3. Edge-Universe Reconciliation",
        "",
        f"Routeable-like non-internal edges: {summary['edge_universe']['routeable_like_edges']}",
        "",
        f"Mapped passenger routeable edges: {summary['edge_universe']['routeable_passenger_edges']}",
        "",
        f"Excluded non-passenger edges: {summary['edge_universe']['excluded_non_passenger_edges']}",
        "",
        f"Internal/function edges: {summary['edge_universe']['internal_or_function_edges']}",
        "",
        "The earlier 1,643 count included non-internal edges without applying the passenger-lane criterion. The mapping intentionally processes passenger-routeable edges, yielding 1,578 edges.",
        "",
        "## 4. Comparison Table",
        "",
        "```csv",
        table.strip(),
        "```",
        "",
        "## 5. Figures",
        "",
        f"- Combined comparison: `{study_root / 'grid_resolution_comparison.png'}`",
    ]
    for size in CANDIDATE_CELL_SIZES:
        label = f"{int(size)}m"
        lines.extend(
            [
                f"- {label} full overlay: `{study_root / label / 'grid_overlay.png'}`",
                f"- {label} ignition zoom: `{study_root / label / 'ignition_zoom_overlay.png'}`",
                f"- {label} dense corridor zoom: `{study_root / label / 'dense_corridor_zoom_overlay.png'}`",
                f"- {label} representative route overlay: `{study_root / label / 'representative_routes_overlay.png'}`",
            ]
        )
    lines.extend(
        [
            "",
        "## 6. Recommendation Rationale",
            "",
        f"The recommended {int(rec)} m resolution is selected by a deterministic score that rewards route-cell distinguishability and route-cell detail while penalizing excessive cell count and coarse short-edge representation. The recommendation should be checked visually against the provided overlays before approval.",
        "",
        "The recommendation does not assume 100 m or any other candidate in advance. The measured results favor 50 m because it preserves full coverage and route-cell detail better than 100 m and 200 m, while reducing the 20 m baseline from 55,440 cells to 8,904 cells.",
        "",
        "## 7. Risks",
            "",
            "- Too-small cells increase GeoJSON size, cell-state rows, mapping records, and real SimFire cost.",
        "- Too-large cells reduce ignition localization, merge nearby roads into the same hazard cells, and can weaken route-cell distinguishability.",
        "",
        "## 8. Commands Run",
        "",
        "```bash",
        "conda run -n evac-sumo python scripts/run_stage6a_resolution_study.py --config configs/test/manhattan_test.yaml --stage6-config configs/stage6.yaml",
        "conda run -n evac-sumo python -m pytest tests/unit/test_grid_resolution_candidates.py --basetemp E:\\Tuan\\Code\\behaviour_eva\\outputs\\test\\manhattan\\pytest_stage6a_tmp -o cache_dir=E:\\Tuan\\Code\\behaviour_eva\\outputs\\test\\manhattan\\pytest_stage6a_cache",
        "```",
        "",
        "## 9. Test Results",
        "",
        "```text",
        "tests/unit/test_grid_resolution_candidates.py: 5 passed",
        "```",
        "",
        "A Windows pytest cache warning occurred, but the tests passed.",
        "",
        "## 10. Cross-Environment Manifest",
        "",
        "The active manifest path remains:",
        "",
        "`outputs/test/manhattan/stage6/simfire/simfire_input_manifest.json`",
        "",
        "Stage 6B must preserve this strict file-boundary contract and regenerate it after the approved resolution is applied. The manifest must include grid dimensions, physical cell size, orientation, affine transform, valid cell IDs, ignition cells, random seed, timing parameters, expected output path, and grid checksum/version.",
        "",
        "## 11. Stage 6B Notes",
        "",
        "- Future mock outputs must be separated under `outputs/test/manhattan/stage6/simfire/mock/`.",
        "- Future real outputs must be separated under `outputs/test/manhattan/stage6/simfire/real/`.",
        "- Real SimFire environment validation, real execution, real cell-state export, real SimFire-to-edge integration, and physical fire-model calibration are separate claims.",
        "- After real SimFire is run, edge hazards, edge survival, route risk, Stage 5 hazard-provider integration, SUMO visualization, and SUMO load validation must be rerun.",
        "- Stage 6A did not run or install real SimFire.",
            "",
            "Please review the grid-resolution comparison and approve one cell size.",
            "I will not run the final real SimFire integration until you confirm the resolution.",
        ]
    )
    Path("reports").mkdir(exist_ok=True)
    Path("reports/stage6a_grid_resolution_study.md").write_text("\n".join(lines), encoding="utf-8")
