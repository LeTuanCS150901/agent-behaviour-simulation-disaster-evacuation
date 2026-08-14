from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sumolib
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from evacuation_sim.io.tables import read_table

from .config import ResolvedIntegrationConfig, normalized_display_text, sha256_file
from .risk_fire_evolution import MEASUREMENT_PHASE, plot_evolution, validate_axis_coverage


def _edge_geometry(config: ResolvedIntegrationConfig) -> tuple[list[str], list[np.ndarray]]:
    net = sumolib.net.readNet(str(config.network_path))
    vehicle_class = config.shared["network"]["vehicle_class"]
    pairs = sorted(
        ((edge.getID(), np.asarray(edge.getShape(), dtype=float)) for edge in net.getEdges(withInternal=False) if not edge.getFunction() and any(lane.allows(vehicle_class) for lane in edge.getLanes())),
        key=lambda item: item[0],
    )
    return [item[0] for item in pairs], [item[1] for item in pairs]


def _base_axes(config: ResolvedIntegrationConfig):
    visual = config.runtime["visualization"]
    plt.rcParams["font.family"] = visual["font_family"]
    fig, ax = plt.subplots(figsize=tuple(visual["figure_size_inches"]))
    if visual["equal_axis"]:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"x ({config.shared['network']['coordinate_frame']['unit']})")
    ax.set_ylabel(f"y ({config.shared['network']['coordinate_frame']['unit']})")
    return fig, ax


def _save(fig, path: Path, config: ResolvedIntegrationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=int(config.runtime["visualization"]["dpi"]), metadata={"Creator": "behaviour-eva configuration-driven integration", "Software": "matplotlib"})
    plt.close(fig)


def generate_visualizations(config: ResolvedIntegrationConfig) -> dict[str, Any]:
    visual = config.runtime["visualization"]
    outputs = config.runtime["outputs"]
    figures = config.output_root / outputs["figures_directory"]
    fire_cells = pd.read_parquet(config.handoff_directory / config.runtime["handoff"]["fire_cells"])
    hazard = read_table(config.output_root / outputs["fire_directory"] / outputs["hazard_table"])
    flow = read_table(config.output_root / outputs["stage5_directory"] / "headless" / outputs["flow_table"])
    edge_ids, segments = _edge_geometry(config)
    edge_position = {edge_id: index for index, edge_id in enumerate(edge_ids)}
    created: dict[str, list[str]] = {"fire_state": [], "edge_risk": [], "traffic_flow": [], "combined_flow_risk": []}
    metadata_rows: list[dict[str, Any]] = []
    report_title = normalized_display_text(config.runtime["reporting"]["title"], "reporting.title")
    times = [float(value) for value in visual["evidence_times_seconds"]]
    fire_cfg = visual["fire_spread"]
    if fire_cfg["enabled"]:
        for time_value in times:
            snapshot = fire_cells[fire_cells["time_seconds"] == time_value].sort_values("cell_id", kind="stable")
            fig, ax = _base_axes(config)
            ax.add_collection(LineCollection(segments, colors=visual["network_color"], linewidths=0.3, zorder=1))
            patches_by_state: dict[str, list[Polygon]] = {state: [] for state in fire_cfg["state_colors"]}
            for row in snapshot.itertuples(index=False):
                patches_by_state[row.canonical_state_label].append(Polygon([(row.x_min, row.y_min), (row.x_max, row.y_min), (row.x_max, row.y_max), (row.x_min, row.y_max)], closed=True))
            handles = []
            for state, patches in patches_by_state.items():
                if patches:
                    ax.add_collection(PatchCollection(patches, facecolor=fire_cfg["state_colors"][state], edgecolor="none", zorder=2))
                handles.append(Patch(facecolor=fire_cfg["state_colors"][state], label=state))
            ax.autoscale()
            ax.legend(handles=handles, title="Fire state")
            ax.set_title(f"{report_title} — fire state at t={time_value:g} s")
            path = figures / fire_cfg["filename_template"].format(time_seconds=int(time_value))
            _save(fig, path, config)
            created["fire_state"].append(str(path))
            metadata_rows.append({"family": "fire_state", "time_seconds": time_value, "path": str(path), "source": config.runtime["handoff"]["fire_cells"]})
    risk_cfg = visual["edge_risk"]
    risk_normalizer = Normalize(vmin=float(risk_cfg["value_range"][0]), vmax=float(risk_cfg["value_range"][1]))
    risk_cmap = matplotlib.colormaps[risk_cfg["color_map"]]
    if risk_cfg["enabled"]:
        for time_value in times:
            snapshot = hazard[hazard["time"] == time_value].set_index("edge_id")
            values = np.array([snapshot.loc[edge_id, risk_cfg["metric"]] for edge_id in edge_ids], dtype=float)
            fig, ax = _base_axes(config)
            ax.add_collection(LineCollection(segments, colors=risk_cmap(risk_normalizer(values)), linewidths=float(risk_cfg["line_width"])))
            ax.autoscale()
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=risk_normalizer, cmap=risk_cmap), ax=ax, label=risk_cfg["metric"])
            ax.set_title(f"{report_title} — edge risk at t={time_value:g} s")
            path = figures / risk_cfg["filename_template"].format(time_seconds=int(time_value))
            _save(fig, path, config)
            created["edge_risk"].append(str(path))
            metadata_rows.append({"family": "edge_risk", "time_seconds": time_value, "path": str(path), "source": outputs["hazard_table"], "normalization": risk_cfg["value_range"]})
    flow_cfg = visual["traffic_flow"]
    interval_starts = [float(value) for value in flow_cfg["interval_starts_seconds"]]
    selected_flow = flow[flow["interval_start"].isin(interval_starts)]
    flow_max = max(float(selected_flow["vehicle_edge_entries"].max()), 1.0)
    flow_norm = Normalize(vmin=0.0, vmax=flow_max)
    flow_cmap = matplotlib.colormaps[flow_cfg["color_map"]]
    if flow_cfg["enabled"]:
        for interval_start in interval_starts:
            snapshot = flow[flow["interval_start"] == interval_start].set_index("edge_id")
            values = np.array([snapshot.loc[edge_id, "vehicle_edge_entries"] for edge_id in edge_ids], dtype=float)
            interval_end = float(snapshot["interval_end"].iloc[0])
            fig, ax = _base_axes(config)
            ax.add_collection(LineCollection(segments, colors=flow_cmap(flow_norm(values)), linewidths=float(flow_cfg["line_width"])))
            ax.autoscale()
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=flow_norm, cmap=flow_cmap), ax=ax, label="vehicle edge entries")
            ax.set_title(f"{report_title} — flow [{interval_start:g},{interval_end:g}) s")
            path = figures / flow_cfg["filename_template"].format(interval_start_seconds=int(interval_start), interval_end_seconds=int(interval_end))
            _save(fig, path, config)
            created["traffic_flow"].append(str(path))
            metadata_rows.append({"family": "traffic_flow", "interval_start": interval_start, "interval_end": interval_end, "path": str(path), "source": outputs["flow_table"], "global_max": flow_max})
    combined_cfg = visual["combined_flow_risk"]
    if combined_cfg["enabled"]:
        for interval_start in interval_starts:
            flow_snapshot = flow[flow["interval_start"] == interval_start].set_index("edge_id")
            risk_snapshot = hazard[hazard["time"] == interval_start].set_index("edge_id")
            flow_values = np.array([flow_snapshot.loc[edge_id, "vehicle_edge_entries"] for edge_id in edge_ids], dtype=float)
            risk_values = np.array([risk_snapshot.loc[edge_id, "edge_risk"] for edge_id in edge_ids], dtype=float)
            min_width = float(combined_cfg["min_edge_width"])
            max_width = float(combined_cfg["max_edge_width"])
            widths = min_width + (max_width - min_width) * flow_values / flow_max
            interval_end = float(flow_snapshot["interval_end"].iloc[0])
            fig, ax = _base_axes(config)
            ax.add_collection(LineCollection(segments, colors=risk_cmap(risk_normalizer(risk_values)), linewidths=widths))
            ax.autoscale()
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=risk_normalizer, cmap=risk_cmap), ax=ax, label="edge risk (color)")
            reference_flows = np.array([0.0, flow_max / 2.0, flow_max])
            reference_widths = min_width + (max_width - min_width) * reference_flows / flow_max
            handles = [Line2D([0], [0], color="black", linewidth=width, label=f"{value:g} entries") for value, width in zip(reference_flows, reference_widths)]
            ax.legend(handles=handles, title="Traffic flow (width)", loc="lower left")
            ax.set_title(f"{report_title} — risk and flow [{interval_start:g},{interval_end:g}) s")
            path = figures / combined_cfg["filename_template"].format(interval_start_seconds=int(interval_start), interval_end_seconds=int(interval_end))
            _save(fig, path, config)
            created["combined_flow_risk"].append(str(path))
            metadata_rows.append({"family": "combined_flow_risk", "interval_start": interval_start, "interval_end": interval_end, "path": str(path), "risk_source": outputs["hazard_table"], "flow_source": outputs["flow_table"], "risk_range": risk_cfg["value_range"], "flow_global_max": flow_max})
    evolution_cfg = visual.get("route_risk_fire_evolution")
    evolution_table_path = active_vehicle_table_path = None
    if evolution_cfg and evolution_cfg["enabled"]:
        stage5_headless = config.output_root / outputs["stage5_directory"] / "headless"
        evolution_table_path = stage5_headless / evolution_cfg["derived_table_filename"]
        active_vehicle_table_path = stage5_headless / evolution_cfg["vehicle_table_filename"]
        evolution = read_table(evolution_table_path)
        required = {
            "time_seconds", "time_step", "active_fire_snapshot_time_seconds", "active_vehicle_count",
            "valid_route_risk_vehicle_count", "mean_active_route_risk", "minimum_active_route_risk",
            "maximum_active_route_risk", "burning_cell_count", "burned_cell_count",
            "interacting_fronts",
        }
        if required - set(evolution.columns):
            raise ValueError(f"Evolution table is missing columns: {sorted(required - set(evolution.columns))}")
        if evolution.duplicated("time_seconds").any() or not evolution["time_seconds"].is_monotonic_increasing:
            raise ValueError("Evolution-table time keys must be unique and sorted")
        if not (evolution["active_vehicle_count"] == evolution["valid_route_risk_vehicle_count"]).all():
            raise ValueError("Evolution-table active and valid route-risk counts differ")
        risk_limits, fire_limits = validate_axis_coverage(evolution, evolution_cfg)
        path = figures / evolution_cfg["figure_filename"]
        encoding = plot_evolution(evolution, evolution_cfg, path)
        created["route_risk_fire_evolution"] = [str(path)]
        metadata_rows.append({
            "family": "route_risk_fire_evolution", "path": str(path),
            "derived_table": str(evolution_table_path), "vehicle_table": str(active_vehicle_table_path),
            "vehicle_population": "SUMO getIDList after departures/removals at each recorded step",
            "remaining_route_definition": "committed normal-edge route suffix after scheduled reconsideration; traversed edges excluded",
            "measurement_order": MEASUREMENT_PHASE,
            "route_risk_formula": config.runtime["reporting"]["formula_units"]["route_risk"],
            "fire_count_definitions": {
                "burning_cell_count": f"count({evolution_cfg['fire_state_column']} == {evolution_cfg['burning_state_label']})",
                "burned_cell_count": f"count({evolution_cfg['fire_state_column']} == {evolution_cfg['burned_state_label']})",
            },
            "simulation_horizon_seconds": [float(evolution["time_seconds"].min()), float(evolution["time_seconds"].max())],
            "recording_interval_seconds": float(evolution_cfg["recording_interval_seconds"]),
            "time_step_mapping": "time_seconds = simulation_start_seconds + time_step * sumo_step_seconds",
            "interacting_fronts_semantics": "inherited from the latest fire snapshot at or before the observation time",
            "sumo_step_seconds": float(config.shared["clock"]["sumo_step_seconds"]),
            "display_time_unit": evolution_cfg["display_time_unit"],
            "risk_axis_limits": list(risk_limits), "fire_count_axis_limits": list(fire_limits),
            "axis_clipping_validation": "passed", "plot_encoding": encoding,
        })
    manifest_path = figures / outputs["visualization_manifest"]
    manifest = {
        "resolved_config_sha256": config.logical_sha256,
        "network_sha256": config.shared["network"]["sha256"],
        "scientific_table_hashes": {
            "fire_cells": sha256_file(config.handoff_directory / config.runtime["handoff"]["fire_cells"]),
            "edge_hazard": sha256_file(config.output_root / outputs["fire_directory"] / outputs["hazard_table"]),
            "traffic_flow": sha256_file(config.output_root / outputs["stage5_directory"] / "headless" / outputs["flow_table"]),
        },
        "settings": visual,
        "figures": [{**row, "sha256": sha256_file(Path(row["path"]))} for row in metadata_rows],
    }
    if evolution_table_path is not None and active_vehicle_table_path is not None:
        manifest["scientific_table_hashes"]["active_vehicle_route_risk"] = sha256_file(active_vehicle_table_path)
        manifest["scientific_table_hashes"]["route_risk_fire_evolution"] = sha256_file(evolution_table_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"families": created, "manifest": str(manifest_path), "figure_count": sum(len(value) for value in created.values())}
