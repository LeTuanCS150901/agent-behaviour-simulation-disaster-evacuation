from __future__ import annotations

import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sumolib

from evacuation_sim.io.tables import read_table, write_table
from evacuation_sim.route_choice.hazard_provider import EdgeHazardProvider


def route_travel_time(net, edge_ids: list[str]) -> float:
    return float(sum(net.getEdge(edge_id).getLength() / max(net.getEdge(edge_id).getSpeed(), 1e-9) for edge_id in edge_ids))


def route_survival(distances: list[float], d_max: float) -> float:
    probs = np.clip(np.asarray(distances, dtype=float) / d_max, 1e-12, 1.0)
    return float(np.exp(np.log(probs).sum()))


def route_utility(normalized_travel_time: float, hazard_exposure: float, alpha_t: float, alpha_h: float) -> float:
    return -alpha_t * normalized_travel_time - alpha_h * hazard_exposure


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def dijkstra_route(net, from_edge_id: str, to_edge_id: str, vehicle_class: str, banned: set[str] | None = None):
    banned = banned or set()
    if from_edge_id in banned or to_edge_id in banned:
        return None
    import heapq

    target = to_edge_id
    queue = [(0.0, from_edge_id, [])]
    best = {}
    while queue:
        cost, edge_id, path = heapq.heappop(queue)
        if edge_id in best and best[edge_id] <= cost:
            continue
        best[edge_id] = cost
        new_path = path + [edge_id]
        if edge_id == target:
            return new_path
        edge = net.getEdge(edge_id)
        for out_edge in edge.getOutgoing().keys():
            out_id = out_edge.getID()
            if out_id in banned or out_edge.getFunction() or out_id.startswith(":"):
                continue
            if not any(lane.allows(vehicle_class) for lane in out_edge.getLanes()):
                continue
            step = out_edge.getLength() / max(out_edge.getSpeed(), 1e-9)
            heapq.heappush(queue, (cost + step, out_id, new_path))
    return None


def candidate_routes(net, origin_edge: str, dest_edge: str, vehicle_class: str, k: int = 3) -> list[list[str]]:
    base = dijkstra_route(net, origin_edge, dest_edge, vehicle_class)
    if base is None:
        return []
    routes = [base]
    for banned_edge in base[1:-1]:
        alt = dijkstra_route(net, origin_edge, dest_edge, vehicle_class, banned={banned_edge})
        if alt and alt not in routes:
            routes.append(alt)
        if len(routes) >= k:
            break
    return routes


def write_sumo_files(network_file: str, routes: pd.DataFrame, output_dir: Path, addxml: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    route_file = output_dir / "manhattan_validation.rou.xml"
    add_copy = output_dir / "manhattan_validation.add.xml"
    cfg_file = output_dir / "manhattan_validation.sumocfg"
    add_copy.write_text(addxml.read_text(encoding="utf-8"), encoding="utf-8")

    root = ET.Element("routes")
    for row in routes.itertuples(index=False):
        ET.SubElement(root, "route", {"id": f"route_{row.vehicle_id}", "edges": row.route_edges})
        ET.SubElement(
            root,
            "vehicle",
            {"id": row.vehicle_id, "depart": f"{row.depart_time:.1f}", "route": f"route_{row.vehicle_id}", "type": "passenger"},
        )
    vtype = ET.Element("vType", {"id": "passenger", "vClass": "passenger"})
    root.insert(0, vtype)
    ET.ElementTree(root).write(route_file, encoding="utf-8", xml_declaration=True)

    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file", {"value": str(Path(network_file).resolve())})
    ET.SubElement(inp, "route-files", {"value": str(route_file.resolve())})
    ET.SubElement(inp, "additional-files", {"value": str(add_copy.resolve())})
    time = ET.SubElement(cfg, "time")
    ET.SubElement(time, "begin", {"value": "0"})
    ET.SubElement(time, "end", {"value": "180"})
    ET.ElementTree(cfg).write(cfg_file, encoding="utf-8", xml_declaration=True)
    return {"sumocfg": str(cfg_file), "routes": str(route_file), "additional": str(add_copy)}


def run_traci_smoke(
    sumocfg: str,
    network_file: str,
    destination_by_vehicle: dict[str, str],
    vehicle_class: str,
    delta_t: float,
    output_dir: Path,
) -> dict:
    log_path = output_dir / "traci_smoke.log"
    try:
        import traci

        net = sumolib.net.readNet(network_file)
        traci.start(["sumo", "-c", sumocfg, "--no-step-log", "true", "--quit-on-end", "true"])
        steps = 0
        reevaluations = 0
        while traci.simulation.getMinExpectedNumber() > 0 and steps < int(max(5, delta_t + 5)):
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            if abs(sim_time - delta_t) < 1e-6 or (steps == int(delta_t)):
                for veh_id in traci.vehicle.getIDList():
                    if veh_id in destination_by_vehicle:
                        current_edge = traci.vehicle.getRoadID(veh_id)
                        if current_edge.startswith(":"):
                            continue
                        new_route = dijkstra_route(
                            net,
                            current_edge,
                            destination_by_vehicle[veh_id],
                            vehicle_class,
                        )
                        if not new_route:
                            continue
                        try:
                            traci.vehicle.setRoute(veh_id, new_route)
                            reevaluations += 1
                        except Exception as exc:  # keep smoke alive, log below
                            log_path.write_text(f"Route reassignment failed for {veh_id}: {exc}", encoding="utf-8")
                            raise
            steps += 1
        traci.close(False)
        log_path.write_text(f"TraCI smoke completed: steps={steps}, reevaluations={reevaluations}\n", encoding="utf-8")
        return {"traci_ran": True, "traci_error": None, "steps": steps, "reevaluations": reevaluations, "log": str(log_path)}
    except Exception as exc:
        try:
            import traci

            traci.close(False)
        except Exception:
            pass
        log_path.write_text(f"TraCI smoke failed: {exc}\n", encoding="utf-8")
        return {"traci_ran": False, "traci_error": str(exc), "steps": 0, "reevaluations": 0, "log": str(log_path)}


def run_stage5_manhattan(
    network_file: str,
    vehicle_class: str,
    stage5_cfg: dict,
    input_dir: str | Path,
    stage4_dir: str | Path,
    stage6_dir: str | Path,
    output_dir: str | Path,
    sumo_dir: str | Path,
    stage6_cfg: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net = sumolib.net.readNet(network_file)
    vehicles = read_table(Path(input_dir) / "vehicles.parquet")
    profiles = read_table(Path(stage4_dir) / "behavioral_profiles.parquet")
    chosen = read_table(Path(stage4_dir) / "chosen_shelters.parquet")
    hazard_representation = (stage6_cfg or {}).get("hazard_representation")
    provider = None
    distances = None
    if hazard_representation == "cell_to_edge":
        provider = EdgeHazardProvider(
            Path(stage6_dir) / "edge_hazard_time_series.parquet",
            numerical_epsilon=float(stage6_cfg["edge_survival"]["numerical_epsilon"]),
            time_lookup=stage6_cfg["hazard_time_lookup"],
            missing_edge_policy=stage6_cfg["hazard_missing_data"]["missing_edge_policy"],
        )
    else:
        distances = read_table(Path(stage6_dir) / "edge_distance_to_fire.parquet").set_index("edge_id")
    chosen = chosen.merge(vehicles, on="vehicle_id").merge(profiles[["vehicle_id", "panic_rate"]], on="vehicle_id")

    route_rows = []
    prob_rows = []
    chosen_route_rows = []
    d_max = float(stage5_cfg["d_max"])
    rng = np.random.default_rng(42)
    for row in chosen.itertuples(index=False):
        routes = candidate_routes(net, row.origin_edge_id, row.chosen_destination_edge_id, vehicle_class, int(stage5_cfg["k_alternative_routes"]))
        if not routes:
            raise ValueError(f"No route found for {row.vehicle_id}: {row.origin_edge_id}->{row.chosen_destination_edge_id}")
        travel_times = np.array([route_travel_time(net, route) for route in routes], dtype=float)
        max_tt = max(float(travel_times.max()), 1e-9)
        utilities = []
        survivals = []
        exposures = []
        for route, tt in zip(routes, travel_times):
            if provider is not None:
                survival, exposure, _log_sum = provider.compute_route_risk(route, float(row.depart_time))
            else:
                edge_distances = [float(distances.loc[eid, "distance_to_fire"]) if eid in distances.index else d_max for eid in route]
                survival = route_survival(edge_distances, d_max)
                exposure = 1.0 - survival
            utilities.append(route_utility(tt / max_tt, exposure, float(stage5_cfg["alpha_t"]), float(stage5_cfg["alpha_h"])))
            survivals.append(survival)
            exposures.append(exposure)
        logit = softmax(np.asarray(utilities))
        uniform = np.ones(len(routes)) / len(routes)
        probs = (1 - row.panic_rate) * logit + row.panic_rate * uniform
        probs = probs / probs.sum()
        selected_idx = int(rng.choice(np.arange(len(routes)), p=probs))
        for idx, route in enumerate(routes):
            route_id = f"{row.vehicle_id}_route_{idx}"
            route_rows.append(
                {
                    "vehicle_id": row.vehicle_id,
                    "route_id": route_id,
                    "route_index": idx,
                    "route_edges": " ".join(route),
                    "travel_time": travel_times[idx],
                    "survival_probability": survivals[idx],
                    "hazard_exposure": exposures[idx],
                    "utility": utilities[idx],
                }
            )
            prob_rows.append(
                {
                    "vehicle_id": row.vehicle_id,
                    "route_id": route_id,
                    "panic_rate": row.panic_rate,
                    "probability": probs[idx],
                }
            )
        chosen_route_rows.append(
            {
                "vehicle_id": row.vehicle_id,
                "depart_time": row.depart_time,
                "chosen_destination_edge_id": row.chosen_destination_edge_id,
                "route_edges": " ".join(routes[selected_idx]),
                "selected_route_index": selected_idx,
            }
        )

    route_df = pd.DataFrame(route_rows)
    prob_df = pd.DataFrame(prob_rows)
    selected_df = pd.DataFrame(chosen_route_rows)
    write_table(route_df, output_dir / "candidate_routes.parquet")
    write_table(prob_df, output_dir / "route_probability_samples.parquet")
    write_table(selected_df, output_dir / "route_choice_logs.parquet")

    entropy = prob_df.groupby("vehicle_id")["probability"].apply(
        lambda s: float(-(s * np.log(np.clip(s, 1e-12, 1))).sum())
    )
    entropy_plot = profiles.set_index("vehicle_id")[["panic_rate"]].join(entropy.rename("entropy"))
    plt.figure(figsize=(7, 4))
    plt.scatter(entropy_plot["panic_rate"], entropy_plot["entropy"], s=10)
    plt.xlabel("panic rate")
    plt.ylabel("route probability entropy")
    plt.tight_layout()
    plt.savefig(output_dir / "entropy_vs_panic_diagnostic.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(1 - route_df["hazard_exposure"], route_df["utility"], s=8)
    plt.xlabel("survival probability")
    plt.ylabel("utility")
    plt.tight_layout()
    plt.savefig(output_dir / "utility_vs_fire_distance_diagnostic.png", dpi=160)
    plt.close()

    sumo_files = write_sumo_files(network_file, selected_df, Path(sumo_dir), Path(stage6_dir) / "fire_hazard.add.xml")
    destination_by_vehicle = {
        row.vehicle_id: row.chosen_destination_edge_id
        for row in selected_df.itertuples(index=False)
    }
    traci_summary = run_traci_smoke(
        sumo_files["sumocfg"],
        network_file,
        destination_by_vehicle,
        vehicle_class,
        float(stage5_cfg["delta_t"]),
        output_dir,
    )
    summary = {
        "stage": "stage5_manhattan",
        "vehicle_count": int(len(selected_df)),
        "candidate_route_rows": int(len(route_df)),
        "route_probabilities_sum_to_one": bool(np.allclose(prob_df.groupby("vehicle_id")["probability"].sum().to_numpy(), 1.0)),
        "delta_t": float(stage5_cfg["delta_t"]),
        "hazard_representation": hazard_representation or "distance_to_fire",
        "used_edge_hazard_time_series": provider is not None,
        "used_edge_distance_to_fire": provider is None,
        "sumo_files": sumo_files,
        "traci": traci_summary,
    }
    (output_dir / "stage5_manhattan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
