from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from .config import ResolvedIntegrationConfig, sha256_file
from .visualization import _edge_geometry


def _load_visualization_config(config: ResolvedIntegrationConfig, path: str | Path) -> tuple[dict[str, Any], Path, str]:
    source_path = Path(path).resolve()
    value = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Flow visualization configuration must be a YAML mapping: {source_path}")
    schema_path = (config.repository_root / value.get("schema", "")).resolve()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(value)
    digest = sha256_file(source_path)
    source = value["source_run"]
    configured_root = (config.repository_root / source["output_root"]).resolve()
    if configured_root != config.output_root:
        raise ValueError(f"Visualization source_run.output_root {configured_root} differs from active run {config.output_root}")
    if source["expected_resolved_config_sha256"] != config.logical_sha256:
        raise ValueError("Visualization configuration targets a different resolved scientific run")
    for path_key, hash_key in (("flow_table", "flow_table_sha256"), ("hazard_table", "hazard_table_sha256")):
        artifact = config.output_root / source[path_key]
        actual = sha256_file(artifact)
        if actual != source[hash_key]:
            raise ValueError(f"Visualization source hash mismatch for {artifact}: expected {source[hash_key]}, got {actual}")
    fire_artifact = (config.repository_root / source["fire_cell_table"]).resolve()
    fire_actual = sha256_file(fire_artifact)
    if fire_actual != source["fire_cell_table_sha256"]:
        raise ValueError(
            f"Visualization source hash mismatch for {fire_artifact}: "
            f"expected {source['fire_cell_table_sha256']}, got {fire_actual}"
        )
    traffic = value["traffic_flow"]
    combined = value["combined_flow_risk"]
    if traffic["maximum_active_width"] <= traffic["minimum_active_width"]:
        raise ValueError("traffic_flow.maximum_active_width must exceed minimum_active_width")
    if traffic["maximum_marker_area"] <= traffic["minimum_marker_area"]:
        raise ValueError("traffic_flow.maximum_marker_area must exceed minimum_marker_area")
    if combined["maximum_active_width"] <= combined["minimum_active_width"]:
        raise ValueError("combined_flow_risk.maximum_active_width must exceed minimum_active_width")
    if combined["risk_value_range"][1] <= combined["risk_value_range"][0]:
        raise ValueError("combined_flow_risk.risk_value_range must be increasing")
    cumulative = value["cumulative_flow_fire"]
    if cumulative["maximum_active_width"] <= cumulative["minimum_active_width"]:
        raise ValueError("cumulative_flow_fire.maximum_active_width must exceed minimum_active_width")
    if cumulative["maximum_marker_area"] <= cumulative["minimum_marker_area"]:
        raise ValueError("cumulative_flow_fire.maximum_marker_area must exceed minimum_marker_area")
    return value, source_path, digest


def _axes(settings: dict[str, Any], coordinate_unit: str):
    output = settings["output"]
    fig, ax = plt.subplots(figsize=tuple(output["figure_size_inches"]))
    if settings["network"]["equal_axis"]:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"x ({coordinate_unit})")
    ax.set_ylabel(f"y ({coordinate_unit})")
    return fig, ax


def _background(ax, segments: list[np.ndarray], settings: dict[str, Any]) -> None:
    network = settings["network"]
    ax.add_collection(LineCollection(segments, colors=network["background_color"], linewidths=network["background_width"], alpha=network["background_alpha"], zorder=1))


def _scaled(values: np.ndarray, minimum: float, maximum: float, global_maximum: float, gamma: float) -> np.ndarray:
    if global_maximum <= 0:
        return np.full(len(values), minimum)
    return minimum + (maximum - minimum) * np.power(np.clip(values / global_maximum, 0.0, 1.0), gamma)


def _reference_values(global_maximum: float) -> list[float]:
    if global_maximum <= 1:
        return [1.0]
    return sorted(set([1.0, float(max(1, round(global_maximum / 2))), float(global_maximum)]))


def _risk_exposure_table(flow: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Return interval exposure bounds without pretending per-edge aggregates contain vehicle IDs."""
    required = {
        "interval_start", "interval_end", "edge_id", "vehicle_edge_entries",
        "unique_vehicle_count", settings["risk_metric"],
    }
    if required - set(flow.columns):
        raise ValueError(f"Flow table is missing risk-exposure columns: {sorted(required - set(flow.columns))}")
    threshold = float(settings["high_risk_threshold"])
    intervals = flow[["interval_start", "interval_end"]].drop_duplicates().sort_values("interval_start", kind="stable")
    maximum_risk = (
        flow.groupby(["interval_start", "interval_end"], sort=False)[settings["risk_metric"]]
        .max().rename("maximum_observed_edge_risk").reset_index()
    )
    qualifying = flow[flow[settings["risk_metric"]] > threshold].copy()
    if qualifying.empty:
        grouped = pd.DataFrame(columns=[
            "interval_start", "interval_end", "high_risk_edge_count",
            "high_risk_edges_with_vehicle_entries", "vehicle_edge_exposure_events",
            "exposed_vehicle_lower_bound", "exposed_vehicle_upper_bound",
        ])
    else:
        qualifying["edge_has_vehicle_entry"] = qualifying["unique_vehicle_count"] > 0
        grouped = (
            qualifying.groupby(["interval_start", "interval_end"], sort=False)
            .agg(
                high_risk_edge_count=("edge_id", "nunique"),
                high_risk_edges_with_vehicle_entries=("edge_has_vehicle_entry", "sum"),
                vehicle_edge_exposure_events=("vehicle_edge_entries", "sum"),
                exposed_vehicle_lower_bound=("unique_vehicle_count", "max"),
                exposed_vehicle_upper_bound=("unique_vehicle_count", "sum"),
            )
            .reset_index()
        )
    result = intervals.merge(grouped, on=["interval_start", "interval_end"], how="left").merge(
        maximum_risk, on=["interval_start", "interval_end"], how="left", validate="one_to_one"
    )
    count_columns = [
        "high_risk_edge_count", "high_risk_edges_with_vehicle_entries",
        "vehicle_edge_exposure_events", "exposed_vehicle_lower_bound",
        "exposed_vehicle_upper_bound",
    ]
    for column in count_columns:
        result[column] = result[column].map(lambda value: 0 if pd.isna(value) else int(value)).astype(int)
    result["exact_unique_vehicle_count_available"] = (
        result["high_risk_edges_with_vehicle_entries"] <= 1
    )
    result["high_risk_threshold"] = threshold
    result["comparison"] = settings["comparison"]
    result["plotted_metric"] = settings["plot_metric"]
    return result


def _save(fig, path: Path, settings: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=int(settings["output"]["dpi"]), metadata={"Creator": "behaviour-eva flow_clarity_v2", "Description": "Configuration-driven diagnostic; scientific tables are authoritative"})
    plt.close(fig)


def generate_clear_flow_visualizations(config: ResolvedIntegrationConfig, visualization_config: str | Path) -> dict[str, Any]:
    settings, settings_path, settings_hash = _load_visualization_config(config, visualization_config)
    source = settings["source_run"]
    flow = pd.read_parquet(config.output_root / source["flow_table"])
    hazard = pd.read_parquet(config.output_root / source["hazard_table"])
    fire_cells = pd.read_parquet(config.repository_root / source["fire_cell_table"])
    required_flow = {"interval_start", "interval_end", "edge_id", settings["traffic_flow"]["metric"]}
    if required_flow - set(flow.columns):
        raise ValueError(f"Flow table is missing configured columns: {sorted(required_flow - set(flow.columns))}")
    if flow.duplicated(["interval_start", "edge_id"]).any():
        raise ValueError("Flow table contains duplicate (interval_start,edge_id) rows")
    if hazard.duplicated(["time", "edge_id"]).any():
        raise ValueError("Hazard table contains duplicate (time,edge_id) rows")
    edge_ids, segments = _edge_geometry(config)
    if set(edge_ids) != set(flow["edge_id"].unique()):
        raise ValueError("Flow-table edge set differs from the configured routeable network")
    segment_by_edge = dict(zip(edge_ids, segments))
    midpoint_by_edge = {edge_id: np.asarray(segment[len(segment) // 2], dtype=float) for edge_id, segment in segment_by_edge.items()}
    output_dir = config.output_root / settings["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_cfg = settings["traffic_flow"]
    combined_cfg = settings["combined_flow_risk"]
    requested = sorted(set(float(value) for value in flow_cfg["interval_starts_seconds"] + combined_cfg["interval_starts_seconds"]))
    available = set(flow["interval_start"].astype(float).unique())
    missing_times = sorted(set(requested) - available)
    if missing_times:
        raise ValueError(f"Requested flow intervals are absent from the scientific table: {missing_times}")
    metric = flow_cfg["metric"]
    global_maximum = float(flow[metric].max())
    if global_maximum <= 0:
        raise ValueError("The complete flow table contains no positive link flow")
    flow_norm = PowerNorm(gamma=float(flow_cfg["power_gamma"]), vmin=0.0, vmax=global_maximum)
    flow_cmap = matplotlib.colormaps[flow_cfg["color_map"]]
    risk_norm = Normalize(vmin=float(combined_cfg["risk_value_range"][0]), vmax=float(combined_cfg["risk_value_range"][1]))
    risk_cmap = matplotlib.colormaps[combined_cfg["risk_color_map"]]
    created: dict[str, list[str]] = {
        "traffic_flow_clear": [],
        "combined_flow_risk_clear": [],
        "cumulative_flow_fire": [],
    }
    records: list[dict[str, Any]] = []
    unit = config.shared["network"]["coordinate_frame"]["unit"]
    for interval_start in [float(value) for value in flow_cfg["interval_starts_seconds"]]:
        snapshot = flow[flow["interval_start"] == interval_start].set_index("edge_id")
        interval_end = float(snapshot["interval_end"].iloc[0])
        values = snapshot.loc[edge_ids, metric].to_numpy(float)
        active_mask = values > 0
        active_ids = np.asarray(edge_ids, dtype=object)[active_mask]
        active_values = values[active_mask]
        fig, ax = _axes(settings, unit)
        _background(ax, segments, settings)
        if len(active_ids):
            active_segments = [segment_by_edge[str(edge_id)] for edge_id in active_ids]
            widths = _scaled(active_values, float(flow_cfg["minimum_active_width"]), float(flow_cfg["maximum_active_width"]), global_maximum, float(flow_cfg["power_gamma"]))
            ax.add_collection(LineCollection(active_segments, colors=flow_cmap(flow_norm(active_values)), linewidths=widths, alpha=0.98, zorder=3))
            if flow_cfg["show_edge_entry_markers"]:
                centres = np.vstack([midpoint_by_edge[str(edge_id)] for edge_id in active_ids])
                marker_areas = _scaled(active_values, float(flow_cfg["minimum_marker_area"]), float(flow_cfg["maximum_marker_area"]), global_maximum, float(flow_cfg["power_gamma"]))
                ax.scatter(centres[:, 0], centres[:, 1], c=active_values, cmap=flow_cmap, norm=flow_norm, s=marker_areas, edgecolors=flow_cfg["marker_edge_color"], linewidths=float(flow_cfg["marker_edge_width"]), alpha=float(flow_cfg["marker_alpha"]), zorder=4)
        else:
            ax.text(0.5, 0.03, "No recorded vehicle edge entries in this interval", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": settings["network"]["background_color"]})
        ax.autoscale()
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=flow_norm, cmap=flow_cmap), ax=ax, label=f"{metric} (global power scale, γ={flow_cfg['power_gamma']})")
        references = _reference_values(global_maximum)
        reference_widths = _scaled(np.asarray(references), float(flow_cfg["minimum_active_width"]), float(flow_cfg["maximum_active_width"]), global_maximum, float(flow_cfg["power_gamma"]))
        handles = [Line2D([0], [0], color=flow_cmap(flow_norm(value)), linewidth=width, marker="o" if flow_cfg["show_edge_entry_markers"] else None, label=f"{value:g} entries") for value, width in zip(references, reference_widths)]
        handles.insert(0, Line2D([0], [0], color=settings["network"]["background_color"], linewidth=settings["network"]["background_width"], label="0 entries / network"))
        ax.legend(handles=handles, title="Link flow", loc="lower left")
        ax.set_title(f"{settings['output']['title']} — [{interval_start:g},{interval_end:g}) s\n{int(active_mask.sum())} active links; {int(values.sum())} edge entries")
        path = output_dir / flow_cfg["filename_template"].format(interval_start_seconds=int(interval_start), interval_end_seconds=int(interval_end))
        _save(fig, path, settings)
        created["traffic_flow_clear"].append(str(path))
        records.append({"family": "traffic_flow_clear", "interval_start": interval_start, "interval_end": interval_end, "active_edges": int(active_mask.sum()), "edge_entries": int(values.sum()), "path": str(path), "sha256": sha256_file(path)})
    for interval_start in [float(value) for value in combined_cfg["interval_starts_seconds"]]:
        flow_snapshot = flow[flow["interval_start"] == interval_start].set_index("edge_id")
        risk_snapshot = hazard[hazard["time"] == interval_start].set_index("edge_id")
        if set(edge_ids) - set(risk_snapshot.index):
            raise ValueError(f"Hazard snapshot at {interval_start} is missing routeable edges")
        interval_end = float(flow_snapshot["interval_end"].iloc[0])
        flow_values = flow_snapshot.loc[edge_ids, metric].to_numpy(float)
        risk_values = risk_snapshot.loc[edge_ids, combined_cfg["risk_metric"]].to_numpy(float)
        active_mask = flow_values > 0
        risky_inactive_mask = (~active_mask) & (risk_values > float(combined_cfg["risk_zero_tolerance"]))
        fig, ax = _axes(settings, unit)
        ax.add_collection(LineCollection(segments, colors=combined_cfg["zero_flow_safe_color"], linewidths=float(combined_cfg["zero_flow_safe_width"]), alpha=0.85, zorder=1))
        if combined_cfg["show_zero_flow_risky_edges"] and risky_inactive_mask.any():
            ax.add_collection(LineCollection([segments[index] for index in np.flatnonzero(risky_inactive_mask)], colors=risk_cmap(risk_norm(risk_values[risky_inactive_mask])), linewidths=float(combined_cfg["zero_flow_risky_width"]), alpha=0.9, zorder=2))
        if active_mask.any():
            widths = _scaled(flow_values[active_mask], float(combined_cfg["minimum_active_width"]), float(combined_cfg["maximum_active_width"]), global_maximum, float(combined_cfg["flow_power_gamma"]))
            ax.add_collection(LineCollection([segments[index] for index in np.flatnonzero(active_mask)], colors=risk_cmap(risk_norm(risk_values[active_mask])), linewidths=widths, alpha=0.98, zorder=3))
        ax.autoscale()
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=risk_norm, cmap=risk_cmap), ax=ax, label=f"{combined_cfg['risk_metric']} (link color)")
        references = _reference_values(global_maximum)
        reference_widths = _scaled(np.asarray(references), float(combined_cfg["minimum_active_width"]), float(combined_cfg["maximum_active_width"]), global_maximum, float(combined_cfg["flow_power_gamma"]))
        handles = [Line2D([0], [0], color="#333333", linewidth=width, label=f"{value:g} entries") for value, width in zip(references, reference_widths)]
        handles.insert(0, Line2D([0], [0], color=combined_cfg["zero_flow_safe_color"], linewidth=combined_cfg["zero_flow_safe_width"], label="0 entries, zero risk"))
        ax.legend(handles=handles, title="Traffic flow (link width)", loc="lower left")
        ax.set_title(f"{settings['output']['title']} and edge risk — [{interval_start:g},{interval_end:g}) s\n{int(active_mask.sum())} active links; {int(flow_values.sum())} edge entries")
        path = output_dir / combined_cfg["filename_template"].format(interval_start_seconds=int(interval_start), interval_end_seconds=int(interval_end))
        _save(fig, path, settings)
        created["combined_flow_risk_clear"].append(str(path))
        records.append({"family": "combined_flow_risk_clear", "interval_start": interval_start, "interval_end": interval_end, "active_edges": int(active_mask.sum()), "edge_entries": int(flow_values.sum()), "positive_risk_edges": int((risk_values > combined_cfg["risk_zero_tolerance"]).sum()), "path": str(path), "sha256": sha256_file(path)})

    cumulative_cfg = settings["cumulative_flow_fire"]
    cumulative_metric = cumulative_cfg["metric"]
    if cumulative_metric not in flow.columns:
        raise ValueError(f"Flow table is missing configured cumulative metric: {cumulative_metric}")
    required_fire = {
        "time_seconds", "cell_id", "x_min", "y_min", "x_max", "y_max",
        cumulative_cfg["fire_state_column"], cumulative_cfg["ignition_flag_column"],
    }
    if required_fire - set(fire_cells.columns):
        raise ValueError(f"Fire-cell table is missing configured columns: {sorted(required_fire - set(fire_cells.columns))}")
    if fire_cells.duplicated(["time_seconds", "cell_id"]).any():
        raise ValueError("Fire-cell table contains duplicate (time_seconds,cell_id) rows")
    geometry_columns = ["cell_id", "x_min", "y_min", "x_max", "y_max"]
    if fire_cells[geometry_columns].drop_duplicates().duplicated("cell_id").any():
        raise ValueError("Fire-cell geometry changes between snapshots")

    cumulative = flow.groupby("edge_id", sort=False)[cumulative_metric].sum()
    if set(cumulative.index) != set(edge_ids):
        raise ValueError("Cumulative-flow edge set differs from the configured routeable network")
    cumulative_values = cumulative.loc[edge_ids].to_numpy(float)
    cumulative_maximum = float(cumulative_values.max())
    cumulative_total = float(cumulative_values.sum())
    if cumulative_maximum <= 0:
        raise ValueError("The complete flow table contains no positive cumulative link flow")
    cumulative_norm = PowerNorm(
        gamma=float(cumulative_cfg["power_gamma"]), vmin=0.0, vmax=cumulative_maximum
    )
    cumulative_cmap = matplotlib.colormaps[cumulative_cfg["color_map"]]
    active_mask = cumulative_values > 0
    burning_rows = fire_cells[
        fire_cells[cumulative_cfg["fire_state_column"]] == cumulative_cfg["fire_state_label"]
    ]
    burning_cells = burning_rows[geometry_columns].drop_duplicates("cell_id").sort_values("cell_id", kind="stable")
    ignition_rows = fire_cells[fire_cells[cumulative_cfg["ignition_flag_column"]].astype(bool)]
    ignition_cells = ignition_rows[geometry_columns].drop_duplicates("cell_id").sort_values("cell_id", kind="stable")
    if burning_cells.empty:
        raise ValueError(
            f"No fire cells match {cumulative_cfg['fire_state_column']}={cumulative_cfg['fire_state_label']}"
        )

    fig, ax = _axes(settings, unit)
    fire_patches = [
        Polygon(
            [(row.x_min, row.y_min), (row.x_max, row.y_min),
             (row.x_max, row.y_max), (row.x_min, row.y_max)],
            closed=True,
        )
        for row in burning_cells.itertuples(index=False)
    ]
    ax.add_collection(
        PatchCollection(
            fire_patches,
            facecolor=cumulative_cfg["fire_fill_color"],
            edgecolor=cumulative_cfg["fire_edge_color"],
            linewidth=float(cumulative_cfg["fire_edge_width"]),
            alpha=float(cumulative_cfg["fire_alpha"]),
            zorder=0,
        )
    )
    _background(ax, segments, settings)
    active_values = cumulative_values[active_mask]
    active_segments = [segments[index] for index in np.flatnonzero(active_mask)]
    widths = _scaled(
        active_values,
        float(cumulative_cfg["minimum_active_width"]),
        float(cumulative_cfg["maximum_active_width"]),
        cumulative_maximum,
        float(cumulative_cfg["power_gamma"]),
    )
    ax.add_collection(
        LineCollection(
            active_segments,
            colors=cumulative_cmap(cumulative_norm(active_values)),
            linewidths=widths,
            alpha=0.98,
            zorder=3,
        )
    )
    if cumulative_cfg["show_edge_entry_markers"]:
        active_ids = np.asarray(edge_ids, dtype=object)[active_mask]
        centres = np.vstack([midpoint_by_edge[str(edge_id)] for edge_id in active_ids])
        marker_areas = _scaled(
            active_values,
            float(cumulative_cfg["minimum_marker_area"]),
            float(cumulative_cfg["maximum_marker_area"]),
            cumulative_maximum,
            float(cumulative_cfg["power_gamma"]),
        )
        ax.scatter(
            centres[:, 0], centres[:, 1], c=active_values, cmap=cumulative_cmap,
            norm=cumulative_norm, s=marker_areas,
            edgecolors=cumulative_cfg["marker_edge_color"],
            linewidths=float(cumulative_cfg["marker_edge_width"]),
            alpha=float(cumulative_cfg["marker_alpha"]), zorder=4,
        )
    if not ignition_cells.empty:
        ax.scatter(
            ((ignition_cells["x_min"] + ignition_cells["x_max"]) / 2).to_numpy(float),
            ((ignition_cells["y_min"] + ignition_cells["y_max"]) / 2).to_numpy(float),
            marker=cumulative_cfg["ignition_marker"],
            c=cumulative_cfg["ignition_marker_color"],
            edgecolors=cumulative_cfg["ignition_marker_edge_color"],
            s=float(cumulative_cfg["ignition_marker_size"]),
            linewidths=0.8, zorder=5,
        )
    ax.autoscale()
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=cumulative_norm, cmap=cumulative_cmap),
        ax=ax,
        label=f"Cumulative {cumulative_metric} (power scale, γ={cumulative_cfg['power_gamma']})",
    )
    references = _reference_values(cumulative_maximum)
    reference_widths = _scaled(
        np.asarray(references), float(cumulative_cfg["minimum_active_width"]),
        float(cumulative_cfg["maximum_active_width"]), cumulative_maximum,
        float(cumulative_cfg["power_gamma"]),
    )
    flow_handles = [
        Line2D([0], [0], color=cumulative_cmap(cumulative_norm(value)), linewidth=width,
               label=f"{value:g} cumulative entries")
        for value, width in zip(references, reference_widths)
    ]
    flow_handles.insert(
        0, Line2D([0], [0], color=cumulative_cfg["zero_flow_color"],
                  linewidth=settings["network"]["background_width"], label="0 entries / network")
    )
    flow_legend = ax.legend(handles=flow_handles, title="Cumulative link flow", loc="lower left")
    ax.add_artist(flow_legend)
    fire_handles = [
        Patch(facecolor=cumulative_cfg["fire_fill_color"], edgecolor=cumulative_cfg["fire_edge_color"],
              alpha=float(cumulative_cfg["fire_alpha"]),
              label=f"Ever {cumulative_cfg['fire_state_label']} during recorded snapshots"),
        Line2D([0], [0], marker=cumulative_cfg["ignition_marker"], color="none",
               markerfacecolor=cumulative_cfg["ignition_marker_color"],
               markeredgecolor=cumulative_cfg["ignition_marker_edge_color"],
               markersize=10, label="Ignition cell"),
    ]
    ax.legend(handles=fire_handles, title="Fire footprint", loc="upper right")
    ax.set_title(
        f"{settings['output']['title']} — cumulative flow with fire footprint\n"
        f"{int(active_mask.sum())} used links; {int(cumulative_total)} edge entries; "
        f"{len(burning_cells)} ever-burning cells"
    )
    cumulative_path = output_dir / cumulative_cfg["filename"]
    _save(fig, cumulative_path, settings)
    created["cumulative_flow_fire"].append(str(cumulative_path))
    records.append(
        {
            "family": "cumulative_flow_fire",
            "aggregation": cumulative_cfg["aggregation"],
            "fire_area_semantics": cumulative_cfg["fire_area_semantics"],
            "active_edges": int(active_mask.sum()),
            "edge_entries": int(cumulative_total),
            "maximum_cumulative_edge_entries": cumulative_maximum,
            "ever_burning_cells": int(len(burning_cells)),
            "ignition_cells": int(len(ignition_cells)),
            "path": str(cumulative_path),
            "sha256": sha256_file(cumulative_path),
        }
    )

    exposure_cfg = settings["risk_exposure_time_series"]
    exposure = _risk_exposure_table(flow, exposure_cfg)
    exposure_table_path = output_dir / exposure_cfg["table_filename"]
    exposure.to_csv(exposure_table_path, index=False)
    threshold = float(exposure_cfg["high_risk_threshold"])
    lower = exposure["exposed_vehicle_lower_bound"].to_numpy(float)
    upper = exposure["exposed_vehicle_upper_bound"].to_numpy(float)
    plotted = exposure[exposure_cfg["plot_metric"]].to_numpy(float)
    times = exposure["interval_start"].to_numpy(float)
    all_exact = bool(exposure["exact_unique_vehicle_count_available"].all())
    fig, ax = plt.subplots(figsize=tuple(exposure_cfg["figure_size_inches"]))
    ax.step(
        times, plotted, where="post", color=exposure_cfg["line_color"],
        linewidth=float(exposure_cfg["line_width"]), marker=exposure_cfg["marker"],
        label=exposure_cfg["line_label"],
    )
    observed_maximum = float(exposure["maximum_observed_edge_risk"].max())
    if int(plotted.max()) == 0:
        ax.set_ylim(-0.05, 1.0)
        ax.text(
            0.5, 0.55,
            f"No edge satisfies {exposure_cfg['risk_metric']} > {threshold:g}\n"
            f"Maximum observed edge risk = {observed_maximum:.6f}",
            transform=ax.transAxes, ha="center", va="center", fontsize=12,
            bbox={"facecolor": "white", "edgecolor": exposure_cfg["line_color"], "alpha": 0.95},
        )
    ax.set_xlabel("Interval start time (seconds)")
    ax.set_ylabel(exposure_cfg["y_axis_label"])
    ax.set_title(
        f"High-risk vehicle exposure over time\n"
        f"high risk: {exposure_cfg['risk_metric']} > {threshold:g}; intervals are left-closed/right-open"
    )
    ax.grid(True, color="#d7dce2", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper right")
    exposure_path = output_dir / exposure_cfg["filename"]
    _save(fig, exposure_path, settings)
    created["risk_exposure_time_series"] = [str(exposure_path)]
    records.append(
        {
            "family": "risk_exposure_time_series",
            "high_risk_threshold": threshold,
            "comparison": exposure_cfg["comparison"],
            "maximum_observed_edge_risk": observed_maximum,
            "total_vehicle_edge_exposure_events": int(exposure["vehicle_edge_exposure_events"].sum()),
            "peak_exposed_vehicle_lower_bound": int(lower.max()),
            "peak_exposed_vehicle_upper_bound": int(upper.max()),
            "exact_for_every_interval": all_exact,
            "path": str(exposure_path),
            "sha256": sha256_file(exposure_path),
            "table_path": str(exposure_table_path),
            "table_sha256": sha256_file(exposure_table_path),
        }
    )
    manifest_path = output_dir / settings["output"]["manifest_filename"]
    manifest = {
        "visualization_contract_version": settings["visualization_contract_version"],
        "visualization_config_path": str(settings_path), "visualization_config_sha256": settings_hash,
        "source_resolved_config_sha256": config.logical_sha256,
        "source_tables": {"flow": {"path": source["flow_table"], "sha256": source["flow_table_sha256"]}, "hazard": {"path": source["hazard_table"], "sha256": source["hazard_table_sha256"]}, "fire_cells": {"path": source["fire_cell_table"], "sha256": source["fire_cell_table_sha256"]}},
        "traffic_evidence": {"total_edge_entries": int(flow[metric].sum()), "positive_edge_interval_rows": int((flow[metric] > 0).sum()), "global_maximum_edge_interval_flow": global_maximum, "maximum_cumulative_edge_flow": cumulative_maximum, "cumulative_active_edges": int(active_mask.sum()), "interval_count": int(flow["interval_start"].nunique()), "edge_count": int(flow["edge_id"].nunique()), "risk_exposure_threshold": threshold, "maximum_observed_edge_risk": observed_maximum, "total_high_risk_vehicle_edge_exposure_events": int(exposure["vehicle_edge_exposure_events"].sum())},
        "settings": settings, "figures": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"status": "passed", "output_directory": str(output_dir), "manifest": str(manifest_path), "traffic_evidence": manifest["traffic_evidence"], "families": created, "figure_count": len(records)}
