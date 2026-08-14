from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sumolib

from evacuation_sim.behavioral_model.attractiveness import (
    DEFAULT_EPSILON,
    compute_attractiveness_scores,
    compute_vehicle_attractiveness_table,
    diagnostic_ranges,
)
from evacuation_sim.behavioral_model.stage4_core import (
    MODEL_VERSION,
    SOFTMAX_TEMPERATURE,
    analytical_c1_coefficients,
    compose_shelter_probabilities,
    compute_mixture_weights,
    raw_selfish_score,
    sample_truncated_normal,
    validate_softmax_c1_config,
)
from evacuation_sim.io.tables import read_table, write_table


def solve_ws_coefficients_c1(epsilon: float, c: float, q: float) -> dict:
    """Unambiguous three-parameter alias for the production C1 coefficients."""
    return analytical_c1_coefficients(epsilon, c, q)


def solve_ws_coefficients(epsilon: float, c: float, q: float, *legacy_args, **legacy_kwargs):
    """Three-parameter alias; reject legacy free-curvature arguments explicitly."""
    if legacy_args or legacy_kwargs:
        raise TypeError(
            "softmax_c1_v1 analytically fixes theta_2 and beta_2; call "
            "solve_ws_coefficients(epsilon, c, q) with exactly three parameters"
        )
    return solve_ws_coefficients_c1(epsilon, c, q)


def ws_value_c1(x, coeffs):
    return np.asarray(raw_selfish_score(x, coeffs["epsilon"], coeffs["c"], coeffs["q"]))


def ws_value(x, coeffs):
    return ws_value_c1(x, coeffs)


def ws_derivative_left_at_c(coeffs: dict) -> float:
    return coeffs["theta_1"] + 2 * coeffs["theta_2"] * coeffs["c"]


def ws_derivative_right_at_c(coeffs: dict) -> float:
    return coeffs["beta_1"] + 2 * coeffs["beta_2"] * coeffs["c"]


def validate_mixture_weights(coeffs: dict, grid_size: int = 1001, tolerance: float = 1e-9) -> None:
    x = np.linspace(0.0, 1.0, grid_size)
    weights = compute_mixture_weights(x, coeffs["epsilon"], coeffs["c"], coeffs["q"])
    share_sum = weights["selfish_softmax_share"] + weights["government_softmax_share"]
    non_panic = weights["selfish_weight"] + weights["government_weight"]
    total = weights["panic_weight"] + non_panic
    if not np.allclose(share_sum, 1.0, atol=tolerance):
        raise ValueError("S_s+S_g does not sum to one")
    if not np.allclose(non_panic, 1.0 - x, atol=tolerance):
        raise ValueError("V_s+V_g does not equal 1-x")
    if not np.allclose(total, 1.0, atol=tolerance):
        raise ValueError("V_p+V_s+V_g does not sum to one")
    for name in ("panic_weight", "selfish_weight", "government_weight"):
        values = np.asarray(weights[name])
        if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
            raise ValueError(f"{name} left [0,1]")


def mixture_weight_diagnostics(coeffs: dict, grid_size: int = 1001) -> dict:
    x = np.linspace(0.0, 1.0, grid_size)
    weights = compute_mixture_weights(x, coeffs["epsilon"], coeffs["c"], coeffs["q"])
    ws = np.asarray(weights["raw_selfish_score"])
    wg = np.asarray(weights["raw_government_score"])
    return {
        "raw_W_s_min": float(ws.min()),
        "raw_W_s_max": float(ws.max()),
        "raw_W_g_min": float(wg.min()),
        "raw_W_g_max": float(wg.max()),
        "raw_W_g_negative_count": int(np.sum(wg < 0.0)),
        "raw_score_identity_max_abs_error": float(np.max(np.abs(ws + wg - (1.0 - x)))),
        "share_sum_max_abs_error": float(np.max(np.abs(weights["selfish_softmax_share"] + weights["government_softmax_share"] - 1.0))),
        "final_weight_sum_max_abs_error": float(np.max(np.abs(weights["panic_weight"] + weights["selfish_weight"] + weights["government_weight"] - 1.0))),
    }


def write_ws_curve_plots(coeffs: dict, output_dir: str | Path, metadata_label: str = "") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(0, 1, 501)
    weights = compute_mixture_weights(xs, coeffs["epsilon"], coeffs["c"], coeffs["q"])
    ws_curve = weights["raw_selfish_score"]
    wg_curve = weights["raw_government_score"]

    plt.figure(figsize=(7, 4))
    plt.plot(xs, ws_curve)
    plt.axvline(coeffs["c"], color="#999999", linestyle="--", linewidth=1)
    plt.scatter([0, coeffs["c"], 1], [coeffs["epsilon"], coeffs["M"], 0], color="#d62728", zorder=3)
    plt.xlabel("panic rate x")
    plt.ylabel("W_s(x)")
    plt.tight_layout()
    if metadata_label:
        plt.figtext(0.01, 0.01, metadata_label, fontsize=6)
    ws_path = output_dir / "raw_selfish_score_curve.png"
    plt.savefig(ws_path, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(xs, ws_curve, label="raw W_s")
    plt.plot(xs, wg_curve, label="raw W_g")
    plt.xlabel("panic rate x")
    plt.ylabel("raw dimensionless score")
    plt.legend()
    plt.tight_layout()
    if metadata_label:
        plt.figtext(0.01, 0.01, metadata_label, fontsize=6)
    raw_path = output_dir / "raw_scores_curve.png"
    plt.savefig(raw_path, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(xs, weights["selfish_softmax_share"], label="S_s")
    plt.plot(xs, weights["government_softmax_share"], label="S_g")
    plt.xlabel("panic rate x")
    plt.ylabel("non-panic softmax share")
    plt.legend()
    plt.tight_layout()
    if metadata_label:
        plt.figtext(0.01, 0.01, metadata_label, fontsize=6)
    shares_path = output_dir / "softmax_shares_curve.png"
    plt.savefig(shares_path, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(xs, weights["panic_weight"], label="V_p")
    plt.plot(xs, weights["selfish_weight"], label="V_s")
    plt.plot(xs, weights["government_weight"], label="V_g")
    plt.xlabel("panic rate x")
    plt.ylabel("final mixture probability")
    plt.legend()
    plt.tight_layout()
    if metadata_label:
        plt.figtext(0.01, 0.01, metadata_label, fontsize=6)
    mixture_path = output_dir / "final_mixture_weights_curve.png"
    plt.savefig(mixture_path, dpi=160)
    plt.close()
    diagnostics = mixture_weight_diagnostics(coeffs)
    diagnostics["raw_selfish_score_curve"] = str(ws_path)
    diagnostics["raw_scores_curve"] = str(raw_path)
    diagnostics["softmax_shares_curve"] = str(shares_path)
    diagnostics["final_mixture_weights_curve"] = str(mixture_path)
    return diagnostics


def sample_panic(n: int, mean: float, std: float, rng: np.random.Generator) -> np.ndarray:
    """Compatibility name for authoritative seeded truncated-normal sampling."""
    return sample_truncated_normal(n, mean, std, rng)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def run_stage4_manhattan(
    network_file: str,
    stage4_cfg: dict,
    random_seed: int,
    input_dir: str | Path,
    stage3_dir: str | Path,
    output_dir: str | Path,
    ignition_edges: list[str] | None = None,
    run_identifier: str | None = None,
    config_hash: str | None = None,
    source_tree_hash: str | None = None,
    result_classification: str = "Manhattan",
) -> dict:
    validate_softmax_c1_config(stage4_cfg)
    run_identifier = run_identifier or MODEL_VERSION
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net = sumolib.net.readNet(network_file)
    vehicles = read_table(Path(input_dir) / "vehicles.parquet")
    destinations = read_table(Path(input_dir) / "destinations.parquet")
    assignment = read_table(Path(stage3_dir) / "per_vehicle_planner_assignment.parquet")
    travel_costs = read_table(Path(stage3_dir) / "travel_time_matrix.parquet")
    if ignition_edges is None:
        ignition_path = Path(input_dir) / "ignition_edges.json"
        if not ignition_path.exists():
            raise ValueError("Stage 4 attractiveness requires ignition_edges for the disaster reference point.")
        ignition_edges = json.loads(ignition_path.read_text(encoding="utf-8"))["ignition_edges"]

    omega = float(stage4_cfg["omega"])
    beta_t = float(stage4_cfg["beta_t"])
    beta_angle = float(stage4_cfg["beta_a"])
    tolerance = float(stage4_cfg["probability_tolerance"])
    metadata_label = (
        f"model={MODEL_VERSION}; q={float(stage4_cfg['q']):g}; M={1-float(stage4_cfg['q']):g}; "
        f"epsilon={float(stage4_cfg['epsilon']):g}; c={float(stage4_cfg['c']):g}; T={SOFTMAX_TEMPERATURE:g}; "
        f"panic=truncated_normal; seed={random_seed}; cfg={config_hash or 'not-recorded'}; "
        f"source={source_tree_hash or 'not-recorded'}; class={result_classification}; run={run_identifier}"
    )

    diagnostic_metadata = {
        "stage4_model_version": MODEL_VERSION,
        "run_identifier": run_identifier,
        "q": float(stage4_cfg["q"]),
        "M": 1.0 - float(stage4_cfg["q"]),
        "epsilon": float(stage4_cfg["epsilon"]),
        "c": float(stage4_cfg["c"]),
        "softmax_temperature": SOFTMAX_TEMPERATURE,
        "panic_rate_distribution": stage4_cfg["panic_rate_distribution"],
        "random_seed": int(random_seed),
        "config_hash": config_hash,
        "source_tree_manifest_hash": source_tree_hash,
        "result_classification": result_classification,
    }

    attractiveness, attr_diag = compute_vehicle_attractiveness_table(
        net,
        travel_costs,
        ignition_edges,
        omega=omega,
        beta_t=beta_t,
        beta_angle=beta_angle,
        epsilon=DEFAULT_EPSILON,
    )
    attr_ranges = diagnostic_ranges(attractiveness)
    for table in (attractiveness, attr_ranges):
        for name, value in diagnostic_metadata.items():
            table[name] = value
    previous_summary_path = output_dir / "stage4_manhattan_summary.json"
    previous_simplified_summary = None
    if previous_summary_path.exists():
        try:
            previous = json.loads(previous_summary_path.read_text(encoding="utf-8"))
            if not previous.get("attractiveness_formula_used", False):
                previous_simplified_summary = previous
                (output_dir / "stage4_previous_simplified_summary.json").write_text(
                    json.dumps(previous, indent=2), encoding="utf-8"
                )
        except json.JSONDecodeError:
            previous_simplified_summary = None

    rng = np.random.default_rng(random_seed)
    coeffs = solve_ws_coefficients_c1(
        float(stage4_cfg["epsilon"]),
        float(stage4_cfg["c"]),
        float(stage4_cfg["q"]),
    )
    panic = sample_panic(len(vehicles), float(stage4_cfg["panic_rate_mean"]), float(stage4_cfg["panic_rate_std"]), rng)
    weights = compute_mixture_weights(
        panic, float(stage4_cfg["epsilon"]), float(stage4_cfg["c"]), float(stage4_cfg["q"])
    )
    raw_ws = np.asarray(weights["raw_selfish_score"])
    raw_wg = np.asarray(weights["raw_government_score"])
    share_s = np.asarray(weights["selfish_softmax_share"])
    share_g = np.asarray(weights["government_softmax_share"])
    v_p = np.asarray(weights["panic_weight"])
    v_s = np.asarray(weights["selfish_weight"])
    v_g = np.asarray(weights["government_weight"])
    dest_ids = destinations["destination_id"].tolist()
    if not dest_ids or len(set(dest_ids)) != len(dest_ids):
        raise ValueError("Eligible destination_id values must be non-empty and unique")
    n_dest = len(dest_ids)
    selfish_lookup = {
        vehicle_id: group.set_index("destination_id").loc[dest_ids, "selfish_probability"].to_numpy(dtype=float)
        for vehicle_id, group in attractiveness.groupby("vehicle_id")
    }

    profile_rows = []
    prob_rows = []
    chosen_rows = []
    assignment_by_vehicle = assignment.set_index("vehicle_id")
    for idx, vehicle in enumerate(vehicles.itertuples(index=False)):
        assigned_dest = assignment_by_vehicle.loc[vehicle.vehicle_id, "destination_id"]
        if assigned_dest not in dest_ids:
            raise ValueError(
                f"Stage 3 planner destination {assigned_dest!r} for {vehicle.vehicle_id} is not eligible"
            )
        selfish = selfish_lookup[vehicle.vehicle_id]
        government = np.array([1.0 if d == assigned_dest else 0.0 for d in dest_ids])
        panic_uniform = np.ones(n_dest) / n_dest
        mixture = compose_shelter_probabilities(
            government,
            selfish,
            panic_uniform,
            government_mixture_weight=v_g[idx],
            selfish_mixture_weight=v_s[idx],
            panic_mixture_weight=v_p[idx],
            probability_tolerance=tolerance,
        )
        chosen = rng.choice(dest_ids, p=mixture)
        profile_rows.append(
            {
                "vehicle_id": vehicle.vehicle_id,
                "panic_rate": panic[idx],
                "raw_selfish_score": raw_ws[idx],
                "raw_government_score": raw_wg[idx],
                "selfish_softmax_share": share_s[idx],
                "government_softmax_share": share_g[idx],
                "panic_weight": v_p[idx],
                "selfish_weight": v_s[idx],
                "government_weight": v_g[idx],
                "W_s": raw_ws[idx],
                "W_g": raw_wg[idx],
                "S_s": share_s[idx],
                "S_g": share_g[idx],
                "V_p": v_p[idx],
                "V_s": v_s[idx],
                "V_g": v_g[idx],
                "planner_destination_id": assigned_dest,
                "final_probability_sum": float(mixture.sum()),
                "stage4_model_version": MODEL_VERSION,
                "run_identifier": run_identifier,
                "config_hash": config_hash,
                "source_tree_manifest_hash": source_tree_hash,
                "result_classification": result_classification,
            }
        )
        for dest_idx, (dest_id, prob) in enumerate(zip(dest_ids, mixture)):
            prob_rows.append(
                {
                    "vehicle_id": vehicle.vehicle_id,
                    "destination_id": dest_id,
                    "P_g": government[dest_idx],
                    "P_s": selfish[dest_idx],
                    "P_p": panic_uniform[dest_idx],
                    "W_s": raw_ws[idx],
                    "W_g": raw_wg[idx],
                    "S_s": share_s[idx],
                    "S_g": share_g[idx],
                    "V_p": v_p[idx],
                    "V_s": v_s[idx],
                    "V_g": v_g[idx],
                    "probability": prob,
                    "stage4_model_version": MODEL_VERSION,
                    "run_identifier": run_identifier,
                    "config_hash": config_hash,
                    "source_tree_manifest_hash": source_tree_hash,
                    "result_classification": result_classification,
                }
            )
        chosen_rows.append(
            {
                "vehicle_id": vehicle.vehicle_id,
                "chosen_destination_id": chosen,
                "stage4_model_version": MODEL_VERSION,
                "run_identifier": run_identifier,
                "config_hash": config_hash,
                "source_tree_manifest_hash": source_tree_hash,
                "result_classification": result_classification,
            }
        )

    profiles = pd.DataFrame(profile_rows)
    probabilities = pd.DataFrame(prob_rows)
    chosen = pd.DataFrame(chosen_rows).merge(destinations, left_on="chosen_destination_id", right_on="destination_id")
    chosen = chosen.rename(columns={"edge_id": "chosen_destination_edge_id"})

    planned_counts = assignment["destination_id"].value_counts().rename("planned_count")
    chosen_counts = chosen["chosen_destination_id"].value_counts().rename("chosen_count")
    capacity_check = destinations.set_index("destination_id")[["edge_id", "capacity"]].join(planned_counts).join(chosen_counts).fillna(0)
    capacity_check["planned_count"] = capacity_check["planned_count"].astype(int)
    capacity_check["chosen_count"] = capacity_check["chosen_count"].astype(int)
    capacity_check["overflow"] = (capacity_check["chosen_count"] - capacity_check["capacity"]).clip(lower=0)
    capacity_check["overflow_rate"] = capacity_check["overflow"] / capacity_check["capacity"]
    for name, value in diagnostic_metadata.items():
        capacity_check[name] = value
    capacity_check.reset_index().to_csv(output_dir / "chosen_shelter_capacity_check.csv", index=False)

    weight_plot_diagnostics = write_ws_curve_plots(coeffs, output_dir, metadata_label)

    plt.figure(figsize=(7, 4))
    plt.hist(panic, bins=20, edgecolor="black")
    plt.xlabel("panic rate")
    plt.ylabel("evacuee count")
    plt.title("Truncated-normal panic-rate sample")
    plt.tight_layout()
    plt.figtext(0.01, 0.01, metadata_label, fontsize=6)
    plt.savefig(output_dir / "truncated_normal_panic_rate_diagnostic.png", dpi=160)
    plt.close()

    beta_grid = [0.5, 1.0, 2.0, 4.0]
    entropies = []
    first_vehicle_id = assignment.iloc[0]["vehicle_id"]
    first_attr = attractiveness[attractiveness["vehicle_id"] == first_vehicle_id].set_index("destination_id").loc[dest_ids]
    for beta in beta_grid:
        a, _, _, _ = compute_attractiveness_scores(
            first_attr["travel_time_score"].to_numpy(),
            first_attr["angle_score"].to_numpy(),
            omega=omega,
            beta_t=beta,
            beta_angle=beta_angle,
        )
        p = softmax(a)
        entropies.append(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
    plt.figure(figsize=(7, 4))
    plt.plot(beta_grid, entropies, marker="o")
    plt.xlabel("softmax sensitivity multiplier")
    plt.ylabel("entropy")
    plt.tight_layout()
    plt.figtext(0.01, 0.01, metadata_label, fontsize=6)
    plt.savefig(output_dir / "softmax_sensitivity_diagnostic.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(attractiveness["theta_ij"], attractiveness["angle_score"], s=8)
    plt.xlabel("theta_ij physical angle")
    plt.ylabel("angle_score penalty")
    plt.tight_layout()
    plt.figtext(0.01, 0.01, metadata_label, fontsize=6)
    plt.savefig(output_dir / "angle_to_disaster_diagnostic.png", dpi=160)
    plt.close()

    omega_grid = np.linspace(0, 1, 11)
    omega_probs = []
    for omega_value in omega_grid:
        a, _, _, _ = compute_attractiveness_scores(
            first_attr["travel_time_score"].to_numpy(),
            first_attr["angle_score"].to_numpy(),
            omega=float(omega_value),
            beta_t=beta_t,
            beta_angle=beta_angle,
        )
        omega_probs.append(softmax(a)[0])
    plt.figure(figsize=(7, 4))
    plt.plot(omega_grid, omega_probs, marker="o")
    plt.xlabel("omega")
    plt.ylabel(f"P_s({dest_ids[0]}) for sample vehicle")
    plt.tight_layout()
    plt.figtext(0.01, 0.01, metadata_label, fontsize=6)
    plt.savefig(output_dir / "omega_sensitivity_diagnostic.png", dpi=160)
    plt.close()

    beta_grid_2d = [0.5, 1.0, 2.0, 4.0]
    records = []
    for beta_t_value in beta_grid_2d:
        for beta_angle_value in beta_grid_2d:
            a, _, _, _ = compute_attractiveness_scores(
                first_attr["travel_time_score"].to_numpy(),
                first_attr["angle_score"].to_numpy(),
                omega=omega,
                beta_t=beta_t_value,
                beta_angle=beta_angle_value,
            )
            p = softmax(a)
            records.append({"beta_t": beta_t_value, "beta_angle": beta_angle_value, "entropy": -(p * np.log(np.clip(p, 1e-12, 1))).sum()})
    beta_diag = pd.DataFrame(records)
    plt.figure(figsize=(7, 4))
    for beta_angle_value, group in beta_diag.groupby("beta_angle"):
        plt.plot(group["beta_t"], group["entropy"], marker="o", label=f"beta_angle={beta_angle_value}")
    plt.xlabel("beta_t")
    plt.ylabel("sample P_s entropy")
    plt.legend()
    plt.tight_layout()
    plt.figtext(0.01, 0.01, metadata_label, fontsize=6)
    plt.savefig(output_dir / "beta_t_beta_angle_sensitivity_diagnostic.png", dpi=160)
    plt.close()

    write_table(attractiveness, output_dir / "attractiveness_scores.parquet")
    attr_ranges.to_csv(output_dir / "attractiveness_diagnostic.csv", index=False)
    write_table(profiles, output_dir / "behavioral_profiles.parquet")
    write_table(probabilities, output_dir / "shelter_choice_probabilities.parquet")
    write_table(chosen, output_dir / "chosen_shelters.parquet")
    changed_vs_previous = None
    previous_chosen_counts = None
    if previous_simplified_summary is not None:
        previous_chosen_counts = previous_simplified_summary.get("chosen_shelter_counts")
        changed_vs_previous = previous_chosen_counts != chosen["chosen_destination_id"].value_counts().to_dict()
    summary = {
        "stage": "stage4_manhattan",
        "stage4_model_version": MODEL_VERSION,
        "run_identifier": run_identifier,
        "config_hash": config_hash,
        "source_tree_manifest_hash": source_tree_hash,
        "result_classification": result_classification,
        "q": float(stage4_cfg["q"]),
        "M": 1.0 - float(stage4_cfg["q"]),
        "epsilon": float(stage4_cfg["epsilon"]),
        "c": float(stage4_cfg["c"]),
        "softmax_temperature": SOFTMAX_TEMPERATURE,
        "probability_tolerance": tolerance,
        "panic_rate_distribution": stage4_cfg["panic_rate_distribution"],
        "panic_rate_mean_config": float(stage4_cfg["panic_rate_mean"]),
        "panic_rate_std_config": float(stage4_cfg["panic_rate_std"]),
        "random_seed": int(random_seed),
        "vehicle_count": len(vehicles),
        "panic_min": float(profiles["panic_rate"].min()),
        "panic_mean": float(profiles["panic_rate"].mean()),
        "panic_max": float(profiles["panic_rate"].max()),
        "weights_sum_to_one": bool(np.allclose((profiles[["V_p", "V_s", "V_g"]].sum(axis=1)).to_numpy(), 1.0, atol=tolerance)),
        "probabilities_sum_to_one": bool(np.allclose(probabilities.groupby("vehicle_id")["probability"].sum().to_numpy(), 1.0, atol=tolerance)),
        "chosen_shelter_counts": chosen["chosen_destination_id"].value_counts().to_dict(),
        "capacity_overflow_total": int(capacity_check["overflow"].sum()),
        "mock_or_real": "Manhattan",
        "ws_curve_mode": MODEL_VERSION,
        "ws_formula": "left: epsilon + 2(M-epsilon)x/c + (epsilon-M)x^2/c^2; right: M - M(x-c)^2/(1-c)^2",
        "ws_coefficients": coeffs,
        "ws_left_derivative_at_c": float(ws_derivative_left_at_c(coeffs)),
        "ws_right_derivative_at_c": float(ws_derivative_right_at_c(coeffs)),
        "mixture_weight_diagnostics": weight_plot_diagnostics,
        "legacy_curvature_fields_present": False,
        "final_composition": "P = V_g*P_g + V_s*P_s + V_p*P_p",
        "only_final_V_weights_used_for_sampling": True,
        "attractiveness_formula_used": True,
        "formula": "A_ij = omega * exp(-beta_t * travel_time_score_ij) + (1-omega) * exp(-beta_angle * angle_score_ij)",
        "phi_definition": "exactly 1 - omega; not independently configurable",
        "uses_omega": True,
        "uses_one_minus_omega": True,
        "uses_beta_t": True,
        "uses_beta_angle": True,
        "beta_angle_config_key": "beta_a",
        "travel_time_score_definition": "(t_ij - T_i) / T_i",
        "travel_time_score_semantics": "penalty/cost score; 0 is fastest shelter and larger is worse",
        "T_i_definition": "minimum free-flow travel time from origin i to any shelter",
        "angle_score_definition": "(theta_i - theta_ij) / theta_i",
        "angle_score_semantics": "penalty/cost score; larger physical theta_ij gives smaller angle_score and larger angle attractiveness",
        "theta_i_definition": "maximum theta_ij over candidate shelters for origin i",
        "theta_ij_interpretation": {
            "0": "toward disaster",
            "pi/2": "perpendicular",
            "pi": "away from disaster",
        },
        "epsilon_denominator": DEFAULT_EPSILON,
        "travel_time_denominator_fallback_count": attr_diag.travel_time_denominator_fallback_count,
        "theta_denominator_fallback_count": attr_diag.theta_denominator_fallback_count,
        "attractiveness_diagnostic_ranges": attr_ranges.to_dict(orient="records"),
        "previous_simplified_summary_preserved": previous_simplified_summary is not None,
        "previous_simplified_chosen_shelter_counts": previous_chosen_counts,
        "chosen_shelter_counts_changed_vs_previous_simplified": changed_vs_previous,
    }
    (output_dir / "stage4_manhattan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
