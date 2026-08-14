from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import sumolib
from scipy.optimize import linear_sum_assignment

from evacuation_sim.io.tables import read_table, write_table


def route_travel_time(route_edges) -> float:
    return float(sum(edge.getLength() / max(edge.getSpeed(), 1e-9) for edge in route_edges))


def shortest_route(net, from_edge_id: str, to_edge_id: str, vehicle_class: str):
    path, _ = net.getShortestPath(
        net.getEdge(from_edge_id),
        net.getEdge(to_edge_id),
        vClass=vehicle_class,
    )
    if path is None:
        return None, None
    edge_ids = [edge.getID() for edge in path]
    return edge_ids, route_travel_time(path)


def compute_vehicle_costs(vehicles: pd.DataFrame, destinations: pd.DataFrame, net, vehicle_class: str):
    rows = []
    unreachable = []
    for vehicle in vehicles.itertuples(index=False):
        for dest in destinations.itertuples(index=False):
            route, travel_time = shortest_route(net, vehicle.origin_edge_id, dest.edge_id, vehicle_class)
            if route is None:
                unreachable.append(
                    {
                        "vehicle_id": vehicle.vehicle_id,
                        "origin_edge_id": vehicle.origin_edge_id,
                        "destination_id": dest.destination_id,
                        "destination_edge_id": dest.edge_id,
                    }
                )
                continue
            rows.append(
                {
                    "vehicle_id": vehicle.vehicle_id,
                    "origin_id": vehicle.origin_id,
                    "origin_edge_id": vehicle.origin_edge_id,
                    "destination_id": dest.destination_id,
                    "destination_edge_id": dest.edge_id,
                    "travel_time": travel_time,
                    "route_edges": " ".join(route),
                }
            )
    if unreachable:
        raise ValueError(f"Unreachable origin-destination pairs found: {unreachable[:10]}")
    return pd.DataFrame(rows)


def solve_per_vehicle_assignment(costs: pd.DataFrame, destinations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    vehicles = sorted(costs["vehicle_id"].unique())
    dests = destinations["destination_id"].tolist()
    cost_lookup = {
        (row.vehicle_id, row.destination_id): float(row.travel_time)
        for row in costs.itertuples(index=False)
    }
    slots: list[str] = []
    for row in destinations.itertuples(index=False):
        slots.extend([row.destination_id] * int(row.capacity))
    if len(slots) < len(vehicles):
        raise ValueError("Total destination slot count is below vehicle count.")

    matrix = []
    for vehicle_id in vehicles:
        matrix.append([cost_lookup[vehicle_id, destination_id] for destination_id in slots])
    row_ind, col_ind = linear_sum_assignment(matrix)
    assigned_rows = []
    objective = 0.0
    for row_idx, col_idx in zip(row_ind, col_ind):
        vehicle_id = vehicles[row_idx]
        destination_id = slots[col_idx]
        objective += cost_lookup[vehicle_id, destination_id]
        match = costs[(costs["vehicle_id"] == vehicle_id) & (costs["destination_id"] == destination_id)].iloc[0]
        assigned_rows.append(match.to_dict())
    assignments = pd.DataFrame(assigned_rows)
    summary = {
        "solver": "scipy_linear_sum_assignment_capacity_slots",
        "status": "OPTIMAL",
        "objective_value": objective,
        "best_objective_bound": objective,
        "optimality_gap": 0.0,
        "total_demand": len(vehicles),
        "total_capacity": int(destinations["capacity"].sum()),
        "capacity_usage": assignments["destination_id"].value_counts().to_dict(),
    }
    return assignments, summary


def run_stage3_manhattan(
    network_file: str,
    vehicle_class: str,
    input_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net = sumolib.net.readNet(network_file)
    vehicles = read_table(input_dir / "vehicles.parquet")
    origins = read_table(input_dir / "origins.parquet")
    destinations = read_table(input_dir / "destinations.parquet")
    if int(destinations["capacity"].sum()) < len(vehicles):
        raise ValueError("Total shelter capacity is less than vehicle demand.")

    costs = compute_vehicle_costs(vehicles, destinations, net, vehicle_class)
    assignment, solver_summary = solve_per_vehicle_assignment(costs, destinations)

    planner = (
        assignment.groupby(["origin_id", "origin_edge_id", "destination_id", "destination_edge_id"], as_index=False)
        .agg(assigned_count=("vehicle_id", "count"), mean_travel_time=("travel_time", "mean"))
    )
    summary = solver_summary | {
        "stage": "stage3_manhattan",
        "all_demand_assigned": bool(len(assignment) == len(vehicles)),
        "capacity_respected": bool(
            all(
                assignment["destination_id"].value_counts().get(row.destination_id, 0) <= int(row.capacity)
                for row in destinations.itertuples(index=False)
            )
        ),
    }
    write_table(costs, output_dir / "travel_time_matrix.parquet")
    write_table(planner, output_dir / "planner_assignment.parquet")
    write_table(assignment, output_dir / "per_vehicle_planner_assignment.parquet")
    (output_dir / "solver_summary.json").write_text(json.dumps(solver_summary, indent=2), encoding="utf-8")
    (output_dir / "stage3_manhattan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
