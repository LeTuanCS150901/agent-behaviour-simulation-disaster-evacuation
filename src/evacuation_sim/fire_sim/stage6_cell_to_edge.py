from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import sumolib
import yaml

from evacuation_sim.fire_sim.cell_states import (
    generate_mock_cell_time_series,
    validate_simfire_output_contract,
    write_simfire_input_manifest,
)
from evacuation_sim.fire_sim.edge_cell_mapping import build_edge_cell_mapping, write_edge_cell_mapping_outputs
from evacuation_sim.fire_sim.edge_hazard import (
    addxml_loads_in_sumo,
    run_compute_edge_hazard,
    write_fire_addxml_from_cells,
    write_fire_front_from_cells,
    write_fire_network_overlay_from_cells,
)
from evacuation_sim.fire_sim.fire_grid import build_fire_grid, grid_file_sizes, routeable_edges, write_grid_outputs
from evacuation_sim.io.tables import read_table
from evacuation_sim.route_choice.hazard_provider import (
    EdgeHazardProvider,
    route_segmentation_sensitivity,
    write_route_hazard_samples,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_fire_grid(
    manhattan_config: str | Path,
    stage6_config: str | Path,
    output_root: str | Path = "outputs/test/manhattan",
) -> dict[str, Any]:
    cfg = load_yaml(manhattan_config)
    stage6 = load_yaml(stage6_config)
    stage6_dir = Path(output_root) / "stage6"
    net = sumolib.net.readNet(cfg["network_file"])
    grid, metadata = build_fire_grid(net, stage6)
    paths = write_grid_outputs(grid, metadata, net, cfg["ignition_edges"], stage6_dir, cfg["vehicle_class"])
    manifest = write_simfire_input_manifest(
        grid,
        paths["metadata"],
        cfg["network_file"],
        net,
        stage6,
        cfg["ignition_edges"],
        random_seed=int(load_yaml("configs/base.yaml")["random_seed"]),
        output_dir=stage6_dir,
    )
    summary = {"grid_paths": paths, "manifest_path": str(Path(stage6_dir) / "simfire" / "simfire_input_manifest.json"), "manifest": {k: v for k, v in manifest.items() if k != "valid_cell_identifiers"}}
    (stage6_dir / "grid" / "fire_grid_creation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_mapping(
    manhattan_config: str | Path,
    stage6_config: str | Path,
    output_root: str | Path = "outputs/test/manhattan",
) -> dict[str, Any]:
    cfg = load_yaml(manhattan_config)
    stage6 = load_yaml(stage6_config)
    stage6_dir = Path(output_root) / "stage6"
    net = sumolib.net.readNet(cfg["network_file"])
    grid = gpd.read_file(stage6_dir / "grid" / "fire_grid.geojson")
    map_cfg = stage6["edge_cell_mapping"]
    mapping, coverage, summary = build_edge_cell_mapping(
        net,
        grid,
        cfg["vehicle_class"],
        float(map_cfg["minimum_coverage_ratio"]),
        float(map_cfg["boundary_tolerance"]),
        map_cfg["incomplete_coverage_policy"],
    )
    paths = write_edge_cell_mapping_outputs(mapping, coverage, summary, stage6_dir)
    summary["paths"] = paths
    return summary


def run_mock_or_real_simfire(
    manhattan_config: str | Path,
    stage6_config: str | Path,
    output_root: str | Path = "outputs/test/manhattan",
) -> dict[str, Any]:
    cfg = load_yaml(manhattan_config)
    stage6 = load_yaml(stage6_config)
    stage6_dir = Path(output_root) / "stage6"
    grid = gpd.read_file(stage6_dir / "grid" / "fire_grid.geojson")
    manifest_path = stage6_dir / "simfire" / "simfire_input_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing simfire input manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stage6["mock_fire"]["enabled"]:
        return generate_mock_cell_time_series(grid, manifest, stage6, stage6_dir)
    raise RuntimeError("Real simfire execution is not wired in this environment and mock_fire.enabled is false.")


def compute_hazard_and_visuals(
    manhattan_config: str | Path,
    stage6_config: str | Path,
    output_root: str | Path = "outputs/test/manhattan",
) -> dict[str, Any]:
    cfg = load_yaml(manhattan_config)
    stage6 = load_yaml(stage6_config)
    stage6_dir = Path(output_root) / "stage6"
    net = sumolib.net.readNet(cfg["network_file"])
    grid = gpd.read_file(stage6_dir / "grid" / "fire_grid.geojson")
    edge_summary = run_compute_edge_hazard(stage6, stage6_dir)
    cell_states = read_table(stage6_dir / "simfire" / "fire_cell_time_series.parquet")
    write_fire_front_from_cells(cell_states, grid, stage6_dir / "fire_front_time_series.geojson")
    write_fire_addxml_from_cells(cell_states, grid, stage6_dir / "fire_hazard.add.xml", float(stage6["poi_size"]))
    write_fire_network_overlay_from_cells(cell_states, grid, net, stage6_dir / "fire_network_overlay.png", cfg["vehicle_class"])
    addxml_check = addxml_loads_in_sumo(cfg["network_file"], stage6_dir / "fire_hazard.add.xml", stage6_dir)
    return {"edge_hazard": edge_summary, "sumo_additional_file_check": addxml_check}


def compute_route_risk_outputs(
    manhattan_config: str | Path,
    stage6_config: str | Path,
    output_root: str | Path = "outputs/test/manhattan",
) -> dict[str, Any]:
    cfg = load_yaml(manhattan_config)
    stage6 = load_yaml(stage6_config)
    stage6_dir = Path(output_root) / "stage6"
    provider = EdgeHazardProvider(
        stage6_dir / "edge_hazard_time_series.parquet",
        numerical_epsilon=float(stage6["edge_survival"]["numerical_epsilon"]),
        time_lookup=stage6["hazard_time_lookup"],
        missing_edge_policy=stage6["hazard_missing_data"]["missing_edge_policy"],
    )
    sample_edges = [edge.getID() for edge in routeable_edges(sumolib.net.readNet(cfg["network_file"]), cfg["vehicle_class"])]
    routes = [
        ("sample_one_edge", [sample_edges[0]], 0.0),
        ("sample_two_edges", sample_edges[:2], 60.0),
        ("sample_three_edges", sample_edges[:3], 120.0),
    ]
    samples = write_route_hazard_samples(provider, routes, stage6_dir / "route_hazard_samples.parquet")
    seg = route_segmentation_sensitivity(
        edge_survival=float(pd.read_parquet(stage6_dir / "edge_hazard_time_series.parquet")["edge_survival"].min()),
        output_path=stage6_dir / "route_segmentation_sensitivity.json",
    )
    return {
        "route_hazard_samples_rows": int(len(samples)),
        "route_hazard_samples_path": str(stage6_dir / "route_hazard_samples.parquet"),
        "route_segmentation_sensitivity": seg,
    }


def run_stage6_cell_to_edge(
    manhattan_config: str | Path = "configs/test/manhattan_test.yaml",
    stage6_config: str | Path = "configs/stage6.yaml",
    output_root: str | Path = "outputs/test/manhattan",
) -> dict[str, Any]:
    stage6_dir = Path(output_root) / "stage6"
    stage6_dir.mkdir(parents=True, exist_ok=True)
    grid_summary = create_fire_grid(manhattan_config, stage6_config, output_root)
    mapping_summary = build_mapping(manhattan_config, stage6_config, output_root)
    simfire_summary = run_mock_or_real_simfire(manhattan_config, stage6_config, output_root)
    contract = validate_simfire_output_contract(
        stage6_dir / "simfire" / "simfire_input_manifest.json",
        stage6_dir / "simfire" / "fire_cell_time_series.parquet",
        stage6_dir / "simfire" / "simfire_run_metadata.json",
    )
    hazard_summary = compute_hazard_and_visuals(manhattan_config, stage6_config, output_root)
    route_summary = compute_route_risk_outputs(manhattan_config, stage6_config, output_root)
    summary = {
        "stage": "stage6_cell_to_edge_manhattan",
        "hazard_representation": "cell_to_edge",
        "grid": grid_summary,
        "edge_cell_mapping": mapping_summary,
        "simfire": simfire_summary,
        "cell_output_contract": contract,
        "edge_hazard": hazard_summary["edge_hazard"],
        "sumo_additional_file_check": hazard_summary["sumo_additional_file_check"],
        "route_risk": route_summary,
        "file_sizes": grid_file_sizes(stage6_dir),
    }
    (stage6_dir / "stage6_manhattan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

