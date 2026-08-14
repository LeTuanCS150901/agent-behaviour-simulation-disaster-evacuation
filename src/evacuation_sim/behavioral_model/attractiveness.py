from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_EPSILON = 1e-9


@dataclass(frozen=True)
class AttractivenessDiagnostics:
    epsilon: float
    travel_time_denominator_fallback_count: int
    theta_denominator_fallback_count: int


def compute_edge_coordinate(net, edge_id: str) -> tuple[float, float]:
    """Return an edge coordinate from SUMO geometry.

    The edge midpoint is used when geometry exists. If shape geometry is absent,
    the coordinate is the average of from-node and to-node coordinates. The
    function fails clearly instead of silently returning an artificial point.
    """
    try:
        edge = net.getEdge(edge_id)
    except Exception as exc:
        raise ValueError(f"Cannot compute coordinate: edge does not exist: {edge_id}") from exc

    shape = edge.getShape()
    if shape:
        x, y = shape[len(shape) // 2]
        return float(x), float(y)

    from_node = edge.getFromNode()
    to_node = edge.getToNode()
    if from_node is None or to_node is None:
        raise ValueError(f"Cannot compute coordinate for edge {edge_id}: missing geometry and nodes.")
    x1, y1 = from_node.getCoord()
    x2, y2 = to_node.getCoord()
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0


def compute_disaster_reference_coordinate(net, ignition_edges: list[str]) -> tuple[float, float]:
    if not ignition_edges:
        raise ValueError("At least one ignition edge is required to compute the disaster reference coordinate.")
    coords = np.array([compute_edge_coordinate(net, edge_id) for edge_id in ignition_edges], dtype=float)
    centroid = coords.mean(axis=0)
    return float(centroid[0]), float(centroid[1])


def compute_theta_to_disaster(
    origin: tuple[float, float],
    shelter: tuple[float, float],
    disaster: tuple[float, float],
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    shelter_vec = np.asarray(shelter, dtype=float) - np.asarray(origin, dtype=float)
    disaster_vec = np.asarray(disaster, dtype=float) - np.asarray(origin, dtype=float)
    denom = float(np.linalg.norm(shelter_vec) * np.linalg.norm(disaster_vec))
    if denom <= epsilon:
        return 0.0
    cosine = float(np.dot(shelter_vec, disaster_vec) / denom)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def compute_travel_time_scores(
    travel_times: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[np.ndarray, float, bool]:
    """Compute travel-time penalty scores.

    `travel_time_score = 0` identifies the fastest shelter for an origin/vehicle.
    Larger scores are worse and therefore reduce
    `exp(-beta_t * travel_time_score)`.
    """
    times = np.asarray(travel_times, dtype=float)
    t_min = float(np.min(times))
    used_fallback = t_min <= epsilon
    denom = epsilon if used_fallback else t_min
    return (times - t_min) / denom, t_min, used_fallback


def compute_angle_scores(
    thetas: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[np.ndarray, float, bool]:
    """Compute angle penalty scores.

    A larger physical angle points farther away from the disaster. The largest
    available `theta_ij` receives `angle_score = 0`; smaller/less safe angles
    receive larger penalties and therefore lower
    `exp(-beta_angle * angle_score)`.
    """
    theta_values = np.asarray(thetas, dtype=float)
    theta_max = float(np.max(theta_values))
    used_fallback = theta_max <= epsilon
    if used_fallback:
        return np.zeros_like(theta_values), theta_max, True
    return (theta_max - theta_values) / theta_max, theta_max, False


def compute_attractiveness_scores(
    travel_time_scores: np.ndarray,
    angle_scores: np.ndarray,
    omega: float,
    beta_t: float,
    beta_angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not 0.0 <= omega <= 1.0:
        raise ValueError(f"omega must be in [0,1] because phi=1-omega; got {omega}")
    if beta_t <= 0:
        raise ValueError(f"beta_t must be positive; got {beta_t}")
    if beta_angle <= 0:
        raise ValueError(f"beta_angle must be positive; got {beta_angle}")
    # Stage 4 fixes the angle coefficient analytically as 1 - omega. It is not
    # an independently configurable parameter.
    phi_value = 1.0 - omega
    travel_attr = omega * np.exp(-beta_t * np.asarray(travel_time_scores, dtype=float))
    angle_attr = phi_value * np.exp(-beta_angle * np.asarray(angle_scores, dtype=float))
    return travel_attr + angle_attr, travel_attr, angle_attr, phi_value


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def compute_selfish_probabilities(attractiveness: np.ndarray) -> np.ndarray:
    return softmax(np.asarray(attractiveness, dtype=float))


def compute_vehicle_attractiveness_table(
    net,
    travel_costs: pd.DataFrame,
    ignition_edges: list[str],
    omega: float,
    beta_t: float,
    beta_angle: float,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[pd.DataFrame, AttractivenessDiagnostics]:
    disaster = compute_disaster_reference_coordinate(net, ignition_edges)
    rows: list[dict] = []
    t_fallbacks = 0
    theta_fallbacks = 0

    for vehicle_id, group in travel_costs.groupby("vehicle_id", sort=False):
        ordered = group.reset_index(drop=True)
        origin_edge_id = str(ordered.iloc[0]["origin_edge_id"])
        origin = compute_edge_coordinate(net, origin_edge_id)
        shelter_coords = [compute_edge_coordinate(net, str(edge_id)) for edge_id in ordered["destination_edge_id"]]
        travel_scores, t_min, t_fallback = compute_travel_time_scores(ordered["travel_time"].to_numpy(), epsilon)
        thetas = np.array(
            [compute_theta_to_disaster(origin, shelter, disaster, epsilon) for shelter in shelter_coords],
            dtype=float,
        )
        angle_scores, theta_max, theta_fallback = compute_angle_scores(thetas, epsilon)
        attractiveness, travel_attr, angle_attr, phi = compute_attractiveness_scores(
            travel_scores, angle_scores, omega, beta_t, beta_angle
        )
        selfish = compute_selfish_probabilities(attractiveness)
        t_fallbacks += int(t_fallback)
        theta_fallbacks += int(theta_fallback)
        for idx, cost_row in ordered.iterrows():
            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "origin_edge_id": origin_edge_id,
                    "destination_id": cost_row["destination_id"],
                    "destination_edge_id": cost_row["destination_edge_id"],
                    "travel_time": float(cost_row["travel_time"]),
                    "T_i": t_min,
                    "travel_time_score": float(travel_scores[idx]),
                    "theta_ij": float(thetas[idx]),
                    "theta_i": theta_max,
                    "angle_score": float(angle_scores[idx]),
                    "omega": float(omega),
                    "phi": float(phi),
                    "beta_t": float(beta_t),
                    "beta_angle": float(beta_angle),
                    "travel_attractiveness": float(travel_attr[idx]),
                    "angle_attractiveness": float(angle_attr[idx]),
                    "A_ij": float(attractiveness[idx]),
                    "selfish_probability": float(selfish[idx]),
                    "origin_x": origin[0],
                    "origin_y": origin[1],
                    "shelter_x": shelter_coords[idx][0],
                    "shelter_y": shelter_coords[idx][1],
                    "disaster_x": disaster[0],
                    "disaster_y": disaster[1],
                    "travel_time_denominator_fallback": bool(t_fallback),
                    "theta_denominator_fallback": bool(theta_fallback),
                }
            )
    return pd.DataFrame(rows), AttractivenessDiagnostics(epsilon, t_fallbacks, theta_fallbacks)


def diagnostic_ranges(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "travel_time_score",
        "theta_ij",
        "theta_i",
        "angle_score",
        "travel_attractiveness",
        "angle_attractiveness",
        "A_ij",
    ]
    rows = []
    for column in columns:
        values = table[column]
        rows.append(
            {
                "metric": column,
                "min": float(values.min()),
                "max": float(values.max()),
                "range": float(values.max() - values.min()),
            }
        )
    return pd.DataFrame(rows)
