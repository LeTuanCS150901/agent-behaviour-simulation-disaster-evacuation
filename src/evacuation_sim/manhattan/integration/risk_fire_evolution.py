from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MEASUREMENT_PHASE = "post_fire_post_reroute_pre_movement"


def remaining_route(
    route: tuple[str, ...] | list[str],
    route_index: int,
    current_road: str,
    next_normal_edge: str | None = None,
) -> tuple[list[str], str]:
    edges = list(route)
    if not edges:
        raise ValueError("Active vehicle has an empty SUMO route")
    if route_index < 0 or route_index >= len(edges):
        raise ValueError(f"Route index {route_index} is outside route length {len(edges)}")
    if current_road.startswith(":"):
        if not next_normal_edge or next_normal_edge.startswith(":"):
            raise ValueError(f"Internal road {current_road!r} has no verifiable next normal edge")
        matches = [index for index in range(route_index, len(edges)) if edges[index] == next_normal_edge]
        if not matches:
            raise ValueError(
                f"Upcoming normal edge {next_normal_edge!r} is absent from the committed route "
                f"at or after index {route_index}"
            )
        start = matches[0]
        position_class = "internal"
    else:
        if edges[route_index] != current_road:
            raise ValueError(
                f"Current normal road {current_road!r} differs from route[{route_index}]={edges[route_index]!r}"
            )
        start = route_index
        position_class = "normal"
    remaining = edges[start:]
    if not remaining or any(edge.startswith(":") for edge in remaining):
        raise ValueError("Remaining route must contain at least one normal edge and no internal edges")
    return remaining, position_class


def next_normal_edge(traci, vehicle_id: str, route: tuple[str, ...], route_index: int) -> str:
    for link in traci.vehicle.getNextLinks(vehicle_id):
        edge_id = str(traci.lane.getEdgeID(str(link[0])))
        if edge_id and not edge_id.startswith(":"):
            return edge_id
    if route_index == len(route) - 2:
        return route[-1]
    raise ValueError(
        f"Internal vehicle {vehicle_id!r} has no upcoming normal edge; "
        f"route_index={route_index}, route_suffix={list(route[max(route_index, 0):])}"
    )


def active_vehicle_risk_rows(
    traci,
    provider,
    chosen: pd.DataFrame,
    active_vehicle_ids: list[str],
    rerouted_selected_risks: dict[str, float],
    query_time: float,
    time_step: int,
    tolerance: float,
) -> list[dict[str, Any]]:
    snapshot_metadata = provider.snapshot_metadata(query_time)
    snapshot_time = float(snapshot_metadata["snapshot_time"])
    rows: list[dict[str, Any]] = []
    for vehicle_id in active_vehicle_ids:
        if vehicle_id not in chosen.index:
            raise KeyError(f"Active SUMO vehicle has no assigned shelter: {vehicle_id!r}")
        destination = str(chosen.loc[vehicle_id, "chosen_destination_edge_id"])
        current_road = str(traci.vehicle.getRoadID(vehicle_id))
        if not current_road:
            raise ValueError(f"Active SUMO vehicle has no current road: {vehicle_id!r}")
        route = tuple(str(edge) for edge in traci.vehicle.getRoute(vehicle_id))
        route_index = int(traci.vehicle.getRouteIndex(vehicle_id))
        upcoming = next_normal_edge(traci, vehicle_id, route, route_index) if current_road.startswith(":") else None
        remaining, position_class = remaining_route(route, route_index, current_road, upcoming)
        if remaining[-1] != destination:
            raise ValueError(
                f"Remaining route for {vehicle_id!r} ends at {remaining[-1]!r}, not shelter {destination!r}"
            )
        route_risk = float(provider.route_risk(remaining, query_time)[1])
        if not math.isfinite(route_risk) or route_risk < -tolerance or route_risk > 1.0 + tolerance:
            raise ValueError(f"Invalid remaining-route risk for {vehicle_id!r}: {route_risk}")
        rerouted = vehicle_id in rerouted_selected_risks
        if rerouted and not np.isclose(
            route_risk, rerouted_selected_risks[vehicle_id], atol=tolerance, rtol=0.0
        ):
            raise ValueError(f"Recorded risk differs from the selected route risk for {vehicle_id!r}")
        rows.append({
            "time_seconds": float(query_time), "time_step": int(time_step),
            "active_fire_snapshot_time_seconds": float(snapshot_time),
            "vehicle_id": vehicle_id, "current_road_id": current_road,
            "position_class": "destination" if current_road == destination else position_class,
            "rerouted_at_boundary": rerouted,
            "remaining_route_edges": " ".join(remaining),
            "remaining_edge_count": len(remaining), "remaining_route_risk": route_risk,
            "measurement_phase": MEASUREMENT_PHASE,
            "interacting_fronts": bool(snapshot_metadata["interacting_fronts"]),
        })
    return rows


def build_evolution_table(
    vehicle_risks: pd.DataFrame,
    fire_cells: pd.DataFrame,
    *,
    start_seconds: float,
    end_seconds: float,
    sumo_step_seconds: float,
    recording_interval_seconds: float,
    burning_label: str,
    burned_label: str,
    state_column: str,
    expected_grid_cells: int,
    tolerance: float,
) -> pd.DataFrame:
    if not math.isclose(recording_interval_seconds, sumo_step_seconds, abs_tol=tolerance, rel_tol=0.0):
        raise ValueError("Recording interval must equal the configured SUMO step")
    count = int(round((end_seconds - start_seconds) / recording_interval_seconds)) + 1
    times = start_seconds + np.arange(count, dtype=float) * recording_interval_seconds
    if not np.isclose(times[-1], end_seconds, atol=tolerance, rtol=0.0):
        raise ValueError("Simulation horizon is not an integer number of recorded SUMO steps")
    timeline = pd.DataFrame({"time_seconds": times, "time_step": np.arange(count, dtype=int)})

    required_vehicle = {
        "time_seconds", "time_step", "active_fire_snapshot_time_seconds", "vehicle_id",
        "remaining_route_risk", "measurement_phase", "interacting_fronts",
    }
    if required_vehicle - set(vehicle_risks.columns):
        raise ValueError(f"Vehicle-risk table is missing columns: {sorted(required_vehicle - set(vehicle_risks.columns))}")
    if vehicle_risks.duplicated(["time_seconds", "vehicle_id"]).any():
        raise ValueError("Vehicle-risk table contains duplicate (time_seconds,vehicle_id) rows")
    if len(vehicle_risks):
        values = vehicle_risks["remaining_route_risk"].to_numpy(float)
        if not np.isfinite(values).all() or (values < -tolerance).any() or (values > 1.0 + tolerance).any():
            raise ValueError("Every vehicle route risk must be finite and lie in [0,1]")
        if set(vehicle_risks["measurement_phase"].astype(str)) != {MEASUREMENT_PHASE}:
            raise ValueError("Vehicle-risk table contains an unexpected measurement phase")
        expected_steps = np.rint((vehicle_risks["time_seconds"] - start_seconds) / sumo_step_seconds).astype(int)
        if not np.array_equal(expected_steps, vehicle_risks["time_step"].to_numpy(int)):
            raise ValueError("Vehicle time_step/time_seconds mapping is inconsistent with SUMO step length")

    required_fire = {"time_seconds", "cell_id", state_column, "interacting_fronts"}
    if required_fire - set(fire_cells.columns):
        raise ValueError(f"Fire-cell table is missing columns: {sorted(required_fire - set(fire_cells.columns))}")
    if fire_cells.duplicated(["time_seconds", "cell_id"]).any():
        raise ValueError("Fire-cell table contains duplicate (time_seconds,cell_id) rows")
    snapshot_sizes = fire_cells.groupby("time_seconds")["cell_id"].nunique()
    if not (snapshot_sizes == int(expected_grid_cells)).all():
        raise ValueError("Every fire snapshot must contain the configured number of grid cells")
    interaction = fire_cells.groupby("time_seconds")["interacting_fronts"].agg(["nunique", "first"])
    if (interaction["nunique"] != 1).any():
        raise ValueError("Each fire snapshot must have one interacting_fronts value")
    counts = (
        fire_cells.groupby(["time_seconds", state_column])["cell_id"].nunique()
        .unstack(state_column, fill_value=0).reset_index()
        .rename(columns={"time_seconds": "active_fire_snapshot_time_seconds"})
    )
    counts["burning_cell_count"] = counts[burning_label].astype(int) if burning_label in counts else 0
    counts["burned_cell_count"] = counts[burned_label].astype(int) if burned_label in counts else 0
    counts["interacting_fronts"] = counts["active_fire_snapshot_time_seconds"].map(interaction["first"]).astype(bool)
    counts = counts[["active_fire_snapshot_time_seconds", "burning_cell_count", "burned_cell_count", "interacting_fronts"]]
    # Parquet may preserve integral fire-boundary timestamps as int64 while
    # SUMO observation times are represented as float seconds.  They share
    # the same physical unit; normalize the join key explicitly for a stable
    # previous-snapshot lookup across storage dtypes.
    counts["active_fire_snapshot_time_seconds"] = counts["active_fire_snapshot_time_seconds"].astype(float)
    counts = counts.sort_values("active_fire_snapshot_time_seconds", kind="stable")
    if not counts["burned_cell_count"].is_monotonic_increasing:
        raise ValueError("BURNED-cell count must be nondecreasing under the configured fire-state model")
    timeline = pd.merge_asof(
        timeline.sort_values("time_seconds"), counts,
        left_on="time_seconds", right_on="active_fire_snapshot_time_seconds", direction="backward",
    )
    if timeline[["active_fire_snapshot_time_seconds", "burning_cell_count", "burned_cell_count"]].isna().any().any():
        raise ValueError("A recorded SUMO time has no active fire snapshot")

    if len(vehicle_risks):
        summary = vehicle_risks.groupby(["time_seconds", "time_step"], sort=True).agg(
            active_fire_snapshot_time_seconds=("active_fire_snapshot_time_seconds", "first"),
            interacting_fronts=("interacting_fronts", "first"),
            active_vehicle_count=("vehicle_id", "nunique"),
            valid_route_risk_vehicle_count=("remaining_route_risk", "count"),
            mean_active_route_risk=("remaining_route_risk", "mean"),
            minimum_active_route_risk=("remaining_route_risk", "min"),
            maximum_active_route_risk=("remaining_route_risk", "max"),
        ).reset_index()
    else:
        summary = pd.DataFrame(columns=[
            "time_seconds", "time_step", "active_fire_snapshot_time_seconds", "active_vehicle_count",
            "valid_route_risk_vehicle_count", "mean_active_route_risk", "minimum_active_route_risk",
            "maximum_active_route_risk", "interacting_fronts",
        ])
    # The timeline owns the canonical active-snapshot column.  Remove the
    # observation copy even for a zero-row summary so pandas cannot create
    # ambiguous ``_x``/``_y`` columns for an entirely empty population.
    vehicle_snapshot = summary.pop("active_fire_snapshot_time_seconds")
    vehicle_interacting = summary.pop("interacting_fronts")
    result = timeline.merge(summary, on=["time_seconds", "time_step"], how="left", validate="one_to_one")
    if len(summary):
        expected_snapshot = timeline.set_index(["time_seconds", "time_step"]).loc[
            list(zip(summary["time_seconds"], summary["time_step"])), "active_fire_snapshot_time_seconds"
        ].to_numpy(float)
        if not np.allclose(vehicle_snapshot.to_numpy(float), expected_snapshot, atol=tolerance, rtol=0.0):
            raise ValueError("Vehicle risks do not use the fire snapshot active at their recorded time")
        expected_interacting = timeline.set_index(["time_seconds", "time_step"]).loc[
            list(zip(summary["time_seconds"], summary["time_step"])), "interacting_fronts"
        ].to_numpy(bool)
        if not np.array_equal(vehicle_interacting.to_numpy(bool), expected_interacting):
            raise ValueError("Vehicle interacting_fronts values differ from the active fire snapshot")
    for column in ("active_vehicle_count", "valid_route_risk_vehicle_count"):
        result[column] = result[column].fillna(0).astype(int)
    risk_columns = ["mean_active_route_risk", "minimum_active_route_risk", "maximum_active_route_risk"]
    empty = result["active_vehicle_count"] == 0
    if result.loc[empty, risk_columns].notna().any().any():
        raise ValueError("Empty-population risk statistics must be null")
    active = ~empty
    if not (result.loc[active, "valid_route_risk_vehicle_count"] == result.loc[active, "active_vehicle_count"]).all():
        raise ValueError("Valid route-risk count must equal the active-vehicle count")
    if (result["burning_cell_count"] < 0).any() or (result["burned_cell_count"] < 0).any():
        raise ValueError("Fire-state counts must be nonnegative")
    if ((result["burning_cell_count"] + result["burned_cell_count"]) > expected_grid_cells).any():
        raise ValueError("Burning plus burned cells exceeds the configured grid-cell count")
    return result[[
        "time_seconds", "time_step", "active_fire_snapshot_time_seconds", "active_vehicle_count",
        "valid_route_risk_vehicle_count", "mean_active_route_risk", "minimum_active_route_risk",
        "maximum_active_route_risk", "burning_cell_count", "burned_cell_count",
        "interacting_fronts",
    ]]


def validate_axis_coverage(table: pd.DataFrame, settings: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    risk_limits = tuple(float(value) for value in settings["route_risk_axis_limits"])
    risks = table["mean_active_route_risk"].dropna().to_numpy(float)
    if len(risks) and (risks.min() < risk_limits[0] or risks.max() > risk_limits[1]):
        raise ValueError("Configured route-risk axis clips scientific values")
    fire_values = table[["burning_cell_count", "burned_cell_count"]].to_numpy(float)
    if settings["fire_count_axis_policy"] != "auto_zero_to_data_max_with_headroom":
        raise ValueError("Unsupported fire-count axis policy")
    data_max = float(fire_values.max()) if fire_values.size else 0.0
    upper = max(1.0, data_max * (1.0 + float(settings["fire_count_headroom_fraction"])))
    fire_limits = (0.0, upper)
    if fire_values.size and (fire_values.min() < fire_limits[0] or fire_values.max() > fire_limits[1]):
        raise ValueError("Configured fire-count axis clips scientific values")
    return risk_limits, fire_limits


def plot_evolution(table: pd.DataFrame, settings: dict[str, Any], path: Path) -> dict[str, Any]:
    risk_limits, fire_limits = validate_axis_coverage(table, settings)
    x_column = "time_seconds" if settings["display_time_unit"] == "seconds" else "time_step"
    x = table[x_column].to_numpy(float)
    fig, (risk_ax, fire_ax) = plt.subplots(
        2, 1, sharex=True, figsize=tuple(settings["figure_size_inches"]),
        gridspec_kw={"height_ratios": settings["subplot_height_ratios"], "hspace": settings["subplot_spacing"]},
    )
    risk_ax.plot(
        x, table["mean_active_route_risk"], color=settings["mean_risk_color"],
        linestyle=settings["mean_risk_line_style"], linewidth=float(settings["mean_risk_line_width"]),
        label=settings["mean_risk_label"],
    )
    risk_ax.set_ylim(*risk_limits)
    risk_ax.set_ylabel(settings["mean_risk_y_label"])
    risk_ax.legend(loc=settings["mean_risk_legend_location"])
    risk_ax.grid(True, color=settings["grid_color"], alpha=float(settings["grid_alpha"]))
    fire_ax.step(
        x, table["burning_cell_count"], where="post", color=settings["burning_color"],
        linestyle=settings["burning_line_style"], linewidth=float(settings["fire_line_width"]),
        label=settings["burning_label"],
    )
    fire_ax.step(
        x, table["burned_cell_count"], where="post", color=settings["burned_color"],
        linestyle=settings["burned_line_style"], linewidth=float(settings["fire_line_width"]),
        label=settings["burned_label"],
    )
    fire_ax.set_ylim(*fire_limits)
    fire_ax.set_ylabel(settings["fire_count_y_label"])
    fire_ax.set_xlabel(settings["x_axis_label"])
    fire_ax.legend(loc=settings["fire_legend_location"])
    fire_ax.grid(True, color=settings["grid_color"], alpha=float(settings["grid_alpha"]))
    fig.suptitle(settings["title"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, dpi=int(settings["dpi"]), bbox_inches="tight",
        metadata={key: str(value) for key, value in settings["rendering_metadata"].items()},
    )
    plt.close(fig)
    return {"x_column": x_column, "risk_axis_limits": risk_limits, "fire_count_axis_limits": fire_limits}
