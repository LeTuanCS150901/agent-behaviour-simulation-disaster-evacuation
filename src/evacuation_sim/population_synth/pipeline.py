from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .synthesis import (
    allocate_agents_largest_remainder,
    build_population_validation,
    validate_zone_types,
)


def _require_geopandas():
    try:
        import geopandas as gpd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Stage 1A requires geopandas and shapely. Install geospatial dependencies "
            "before running real Toulouse population synthesis."
        ) from exc
    return gpd


def ensure_crs(gdf, target_crs: str, label: str):
    if gdf.crs is None:
        raise ValueError(f"{label} has no CRS. Refusing to assume CRS silently.")
    if str(gdf.crs) != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


def load_and_join_inputs(base_cfg: dict, stage_cfg: dict):
    gpd = _require_geopandas()
    target_crs = base_cfg["crs"]

    study_area = gpd.read_file(stage_cfg["study_area_bbox_path"])
    zones = gpd.read_file(stage_cfg["zones_path"])
    census = pd.read_csv(stage_cfg["census_data_path"])

    study_area = ensure_crs(study_area, target_crs, "study area")
    zones = ensure_crs(zones, target_crs, "zones")

    zone_id = stage_cfg["zone_id_column"]
    zone_type = stage_cfg["zone_type_column"]
    census_zone_id = stage_cfg["census_zone_id_column"]
    census_population = stage_cfg["census_population_column"]

    for column, label, frame in [
        (zone_id, "zone id", zones),
        (zone_type, "zone type", zones),
        (census_zone_id, "census zone id", census),
        (census_population, "census population", census),
    ]:
        if column not in frame.columns:
            raise ValueError(f"Missing configured {label} column '{column}'.")

    validate_zone_types(zones, zone_type)
    if census[census_zone_id].duplicated().any():
        duplicated = census.loc[census[census_zone_id].duplicated(), census_zone_id].tolist()
        raise ValueError(f"Census zone ids must be unique; duplicates: {duplicated}")

    joined = zones.merge(
        census[[census_zone_id, census_population]],
        left_on=zone_id,
        right_on=census_zone_id,
        how="left",
        validate="one_to_one",
    )
    missing = joined.loc[joined[census_population].isna(), zone_id].tolist()
    if missing:
        raise ValueError(f"Zones missing census population matches: {missing}")
    return study_area, joined


def sample_points_in_polygon(geometry, n_points: int, rng: np.random.Generator):
    if n_points == 0:
        return []
    minx, miny, maxx, maxy = geometry.bounds
    points = []
    attempts = 0
    max_attempts = max(1000, n_points * 10000)
    from shapely.geometry import Point  # type: ignore

    while len(points) < n_points and attempts < max_attempts:
        attempts += 1
        candidate = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if geometry.contains(candidate) or geometry.touches(candidate):
            points.append(candidate)
    if len(points) != n_points:
        raise RuntimeError(
            f"Could only sample {len(points)} of {n_points} requested points inside polygon."
        )
    return points


def synthesize_agents(zones, base_cfg: dict, stage_cfg: dict):
    gpd = _require_geopandas()
    zone_id = stage_cfg["zone_id_column"]
    zone_type = stage_cfg["zone_type_column"]
    census_population = stage_cfg["census_population_column"]
    rng = np.random.default_rng(int(base_cfg["random_seed"]))

    zones = zones.copy()
    zones["synthetic_count"] = allocate_agents_largest_remainder(
        zones, census_population, int(stage_cfg["total_population"])
    )

    agent_rows = []
    geometries = []
    agent_idx = 0
    for _, row in zones.iterrows():
        points = sample_points_in_polygon(row.geometry, int(row["synthetic_count"]), rng)
        for point in points:
            agent_rows.append(
                {
                    "agent_id": f"agent_{agent_idx:08d}",
                    "zone_id": row[zone_id],
                    "zone_type": str(row[zone_type]).lower(),
                    "home_x": float(point.x),
                    "home_y": float(point.y),
                }
            )
            geometries.append(point)
            agent_idx += 1
    agents = gpd.GeoDataFrame(agent_rows, geometry=geometries, crs=zones.crs)
    return zones, agents


def write_population_maps(study_area, zones, agents, output_dir: Path, zone_type_column: str) -> None:
    import matplotlib.pyplot as plt

    color_map = {"red": "#d73027", "blue": "#4575b4"}
    zone_colors = zones[zone_type_column].astype(str).str.lower().map(color_map).fillna("#999999")

    fig, ax = plt.subplots(figsize=(9, 9))
    study_area.boundary.plot(ax=ax, color="black", linewidth=1)
    zones.plot(ax=ax, color=zone_colors, edgecolor="white", linewidth=0.6, alpha=0.75)
    ax.set_title("Stage 1 Red/Blue Zone Split")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_dir / "red_blue_zone_map.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9))
    zones.plot(ax=ax, color=zone_colors, edgecolor="white", linewidth=0.6, alpha=0.45)
    agents.plot(ax=ax, markersize=1, color="black", alpha=0.45)
    ax.set_title("Stage 1 Synthetic Population")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_dir / "diagnostic_population_map.png", dpi=160)
    plt.close(fig)


def write_agents(agents, path: Path) -> str:
    try:
        agents.to_parquet(path, index=False)
        return str(path)
    except Exception as exc:
        fallback = path.with_suffix(".geojson")
        agents.to_file(fallback, driver="GeoJSON")
        raise RuntimeError(
            f"Could not write {path}; parquet engine may be missing. Wrote fallback {fallback}."
        ) from exc


def run_stage1a(base_cfg: dict, stage_cfg: dict, output_dir: str | Path = "outputs/stage1") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study_area, joined = load_and_join_inputs(base_cfg, stage_cfg)
    zones, agents = synthesize_agents(joined, base_cfg, stage_cfg)
    validation = build_population_validation(
        zones,
        stage_cfg["zone_id_column"],
        stage_cfg["zone_type_column"],
        stage_cfg["census_population_column"],
        float(stage_cfg["zone_count_tolerance_pct"]),
    )
    containment = agents.within(
        zones.set_index(stage_cfg["zone_id_column"]).loc[agents["zone_id"]].geometry.reset_index(drop=True)
    )

    zones_path = output_dir / "zones_processed.geojson"
    validation_path = output_dir / "population_validation.csv"
    agents_path = output_dir / "agents.parquet"

    zones.to_file(zones_path, driver="GeoJSON")
    validation.to_csv(validation_path, index=False)
    write_population_maps(study_area, zones, agents, output_dir, stage_cfg["zone_type_column"])
    written_agents_path = write_agents(agents, agents_path)

    summary = {
        "stage": "stage1a",
        "status": "complete",
        "crs": base_cfg["crs"],
        "total_agents": int(len(agents)),
        "total_population_config": int(stage_cfg["total_population"]),
        "all_points_within_assigned_zone": bool(containment.all()),
        "all_zone_counts_within_tolerance": bool(validation["within_tolerance"].all()),
        "configured_columns": {
            "zone_id_column": stage_cfg["zone_id_column"],
            "zone_type_column": stage_cfg["zone_type_column"],
            "census_zone_id_column": stage_cfg["census_zone_id_column"],
            "census_population_column": stage_cfg["census_population_column"],
        },
        "outputs": {
            "agents": written_agents_path,
            "zones_processed": str(zones_path),
            "population_validation": str(validation_path),
            "red_blue_zone_map": str(output_dir / "red_blue_zone_map.png"),
            "diagnostic_population_map": str(output_dir / "diagnostic_population_map.png"),
        },
    }
    (output_dir / "stage1a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
