from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc


REQUIRED_COLUMNS = {
    "vehicle_id",
    "origin_edge_id",
    "destination_id",
    "destination_edge_id",
    "origin_x",
    "origin_y",
    "shelter_x",
    "shelter_y",
    "disaster_x",
    "disaster_y",
    "theta_ij",
    "angle_score",
}


def select_representative_angle_rows(
    attractiveness: pd.DataFrame,
    random_seed: int = 42,
    n_random: int = 2,
    min_rows: int = 5,
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(attractiveness.columns)
    if missing:
        raise ValueError(f"Cannot build Stage 4 geometry debug artifacts; missing columns: {sorted(missing)}")
    if attractiveness.empty:
        raise ValueError("Cannot build Stage 4 geometry debug artifacts from an empty attractiveness table.")

    selected_indices: list[tuple[int, str]] = []
    theta = attractiveness["theta_ij"]
    selected_indices.append((int(theta.idxmin()), "smallest_theta"))
    selected_indices.append((int(theta.idxmax()), "largest_theta"))
    median_theta = float(theta.median())
    selected_indices.append((int((theta - median_theta).abs().idxmin()), "median_theta"))

    rng = np.random.default_rng(random_seed)
    random_indices = rng.choice(attractiveness.index.to_numpy(), size=min(n_random, len(attractiveness)), replace=False)
    selected_indices.extend((int(idx), f"random_seed_{random_seed}") for idx in random_indices)

    if len({idx for idx, _ in selected_indices}) < min_rows:
        ranked = attractiveness.assign(_distance=(theta - median_theta).abs()).sort_values(
            ["_distance", "vehicle_id", "destination_id"]
        )
        for idx in ranked.index:
            if len({i for i, _ in selected_indices}) >= min_rows:
                break
            selected_indices.append((int(idx), "fill_to_min_rows"))

    reason_by_idx: dict[int, list[str]] = {}
    for idx, reason in selected_indices:
        reason_by_idx.setdefault(idx, []).append(reason)

    selected = attractiveness.loc[list(reason_by_idx)].copy()
    selected["selection_reason"] = [";".join(reason_by_idx[int(idx)]) for idx in selected.index]
    return selected.reset_index(drop=True)


def validate_theta_range(table: pd.DataFrame, tolerance: float = 1e-9) -> int:
    invalid = (table["theta_ij"] < -tolerance) | (table["theta_ij"] > math.pi + tolerance)
    return int(invalid.sum())


def validate_theta_angle_monotonicity(table: pd.DataFrame, tolerance: float = 1e-9) -> int:
    """Count pairwise violations of larger theta => smaller/equal angle_score per vehicle."""
    violations = 0
    for _, group in table.groupby("vehicle_id"):
        rows = group[["theta_ij", "angle_score"]].to_numpy(dtype=float)
        for i in range(len(rows)):
            for j in range(len(rows)):
                if rows[i, 0] > rows[j, 0] + tolerance and rows[i, 1] > rows[j, 1] + tolerance:
                    violations += 1
    return violations


def _angle_deg(vector: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def _draw_network(ax, net) -> None:
    for edge in net.getEdges():
        if edge.getFunction() or edge.getID().startswith(":"):
            continue
        shape = edge.getShape()
        if len(shape) >= 2:
            xs, ys = zip(*shape)
            ax.plot(xs, ys, color="#d0d0d0", linewidth=0.35, zorder=0)


def _draw_debug_rows(ax, selected: pd.DataFrame, title: str) -> None:
    ax.set_title(title)
    ax.scatter(selected["origin_x"], selected["origin_y"], marker="o", c="#1f77b4", label="origin s_o", zorder=5)
    ax.scatter(selected["shelter_x"], selected["shelter_y"], marker="^", c="#2ca02c", label="shelter s_j", zorder=5)
    ax.scatter(selected["disaster_x"], selected["disaster_y"], marker="*", c="#d62728", s=120, label="disaster s_f", zorder=6)

    for row in selected.itertuples(index=False):
        origin = np.array([row.origin_x, row.origin_y], dtype=float)
        shelter = np.array([row.shelter_x, row.shelter_y], dtype=float)
        disaster = np.array([row.disaster_x, row.disaster_y], dtype=float)
        shelter_vec = shelter - origin
        disaster_vec = disaster - origin
        ax.arrow(origin[0], origin[1], shelter_vec[0], shelter_vec[1], length_includes_head=True, head_width=30, alpha=0.65, color="#2ca02c")
        ax.arrow(origin[0], origin[1], disaster_vec[0], disaster_vec[1], length_includes_head=True, head_width=30, alpha=0.55, color="#d62728")

        radius = max(40.0, min(np.linalg.norm(shelter_vec), np.linalg.norm(disaster_vec)) * 0.25)
        a1 = _angle_deg(disaster_vec)
        a2 = _angle_deg(shelter_vec)
        delta = (a2 - a1 + 360.0) % 360.0
        if delta > 180.0:
            theta1, theta2 = a2, a1
        else:
            theta1, theta2 = a1, a2
        arc = Arc(origin, radius * 2, radius * 2, angle=0, theta1=theta1, theta2=theta2, color="#9467bd", linewidth=1.2)
        ax.add_patch(arc)
        label_xy = origin + 0.55 * shelter_vec
        ax.annotate(
            f"{row.theta_ij_rad:.2f} rad\n{row.theta_ij_deg:.1f} deg",
            xy=(label_xy[0], label_xy[1]),
            fontsize=7,
            color="#4b0082",
        )
    ax.set_xlabel("SUMO local x")
    ax.set_ylabel("SUMO local y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=7)


def write_stage4_geometry_debug(
    attractiveness_path: str | Path,
    net,
    output_dir: str | Path,
    random_seed: int = 42,
) -> dict:
    attractiveness_path = Path(attractiveness_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attractiveness = pd.read_parquet(attractiveness_path)
    selected = select_representative_angle_rows(attractiveness, random_seed=random_seed)

    debug = selected.rename(
        columns={
            "origin_x": "s_o_x",
            "origin_y": "s_o_y",
            "shelter_x": "s_j_x",
            "shelter_y": "s_j_y",
            "disaster_x": "s_f_x",
            "disaster_y": "s_f_y",
            "theta_ij": "theta_ij_rad",
        }
    ).copy()
    debug["theta_ij_deg"] = np.degrees(debug["theta_ij_rad"])
    columns = [
        "vehicle_id",
        "origin_edge_id",
        "destination_id",
        "destination_edge_id",
        "s_o_x",
        "s_o_y",
        "s_j_x",
        "s_j_y",
        "s_f_x",
        "s_f_y",
        "theta_ij_rad",
        "theta_ij_deg",
        "angle_score",
        "selection_reason",
    ]
    debug = debug[columns]

    theta_range_violations = validate_theta_range(selected)
    monotonicity_violations = validate_theta_angle_monotonicity(attractiveness)

    csv_path = output_dir / "stage4_geometry_angle_debug.csv"
    png_path = output_dir / "stage4_geometry_angle_debug.png"
    debug.to_csv(csv_path, index=False)

    plot_selected = selected.copy()
    plot_selected["theta_ij_rad"] = plot_selected["theta_ij"]
    plot_selected["theta_ij_deg"] = np.degrees(plot_selected["theta_ij"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    _draw_network(axes[0], net)
    _draw_debug_rows(axes[0], plot_selected, "Panel A: Full Manhattan Network")

    _draw_network(axes[1], net)
    _draw_debug_rows(axes[1], plot_selected, "Panel B: Zoomed Geometry Debug")
    xs = np.concatenate([debug["s_o_x"].to_numpy(), debug["s_j_x"].to_numpy(), debug["s_f_x"].to_numpy()])
    ys = np.concatenate([debug["s_o_y"].to_numpy(), debug["s_j_y"].to_numpy(), debug["s_f_y"].to_numpy()])
    margin = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 100.0) * 0.15
    axes[1].set_xlim(float(xs.min() - margin), float(xs.max() + margin))
    axes[1].set_ylim(float(ys.min() - margin), float(ys.max() + margin))
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    summary = {
        "csv": str(csv_path),
        "figure": str(png_path),
        "selected_pair_count": int(len(debug)),
        "selection_strategy": "smallest theta, largest theta, nearest median theta, two fixed-seed random rows, deterministic fill if needed",
        "theta_range_violation_count": theta_range_violations,
        "theta_angle_score_monotonicity_violation_count": monotonicity_violations,
        "manual_review_note": "Numeric checks do not replace visual review; inspect the PNG to validate geometry end-to-end.",
    }
    (output_dir / "stage4_geometry_angle_debug_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "Stage 4 geometry debug: "
        f"theta_range_violations={theta_range_violations}, "
        f"theta_angle_score_monotonicity_violations={monotonicity_violations}"
    )
    return summary
