from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evacuation_sim.io.tables import write_table

PLACEHOLDERS = ("EDGE_ORIGIN", "EDGE_SHELTER", "EDGE_IGNITION")


def load_manhattan_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return cfg


def has_placeholder_edges(cfg: dict[str, Any]) -> bool:
    text_ids = [o["edge_id"] for o in cfg["origins"]]
    text_ids += [d["edge_id"] for d in cfg["destinations"]]
    text_ids += list(cfg["ignition_edges"])
    return any(any(token in edge_id for token in PLACEHOLDERS) for edge_id in text_ids)


def validate_configured_edges(cfg: dict[str, Any], net) -> dict[str, Any]:
    edge_map = {edge.getID(): edge for edge in net.getEdges()}
    ids = [o["edge_id"] for o in cfg["origins"]]
    ids += [d["edge_id"] for d in cfg["destinations"]]
    ids += list(cfg["ignition_edges"])
    missing = [edge_id for edge_id in ids if edge_id not in edge_map]
    if missing:
        raise ValueError(f"Configured Manhattan edge IDs do not exist in network: {missing}")
    vehicle_class = cfg["vehicle_class"]
    disallowed = [
        edge_id
        for edge_id in ids
        if not any(lane.allows(vehicle_class) for lane in edge_map[edge_id].getLanes())
    ]
    if disallowed:
        raise ValueError(
            f"Configured Manhattan edge IDs do not allow vehicle_class={vehicle_class}: {disallowed}"
        )
    return {
        edge_id: {
            "length": edge_map[edge_id].getLength(),
            "speed": edge_map[edge_id].getSpeed(),
            "passenger_allowed": True,
        }
        for edge_id in ids
    }


def write_selected_edges(cfg: dict[str, Any], edge_validation: dict[str, Any], output_dir: Path) -> None:
    selected = {
        "selection_mode": "configured",
        "selection_logic": "Supplementary config already contains real edge IDs; all were validated against the SUMO network.",
        "origins": cfg["origins"],
        "destinations": cfg["destinations"],
        "ignition_edges": cfg["ignition_edges"],
        "edge_validation": edge_validation,
    }
    (output_dir / "selected_edges.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")


def prepare_inputs(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    origins = pd.DataFrame(
        [
            {
                "origin_id": f"origin_{idx}",
                "edge_id": row["edge_id"],
                "num_cars": int(row["num_cars"]),
            }
            for idx, row in enumerate(cfg["origins"])
        ]
    )
    destinations = pd.DataFrame(
        [
            {
                "destination_id": f"shelter_{idx}",
                "edge_id": row["edge_id"],
                "capacity": int(row["capacity"]),
            }
            for idx, row in enumerate(cfg["destinations"])
        ]
    )
    vehicle_rows = []
    vehicle_idx = 0
    for _, origin in origins.iterrows():
        for _ in range(int(origin["num_cars"])):
            vehicle_rows.append(
                {
                    "vehicle_id": f"veh_{vehicle_idx:05d}",
                    "origin_id": origin["origin_id"],
                    "origin_edge_id": origin["edge_id"],
                    "depart_time": 0.0,
                }
            )
            vehicle_idx += 1
    vehicles = pd.DataFrame(vehicle_rows)

    ignition_path = input_dir / "ignition_edges.json"
    ignition_path.write_text(json.dumps({"ignition_edges": cfg["ignition_edges"]}, indent=2), encoding="utf-8")
    return {
        "origins": write_table(origins, input_dir / "origins.parquet"),
        "destinations": write_table(destinations, input_dir / "destinations.parquet"),
        "vehicles": write_table(vehicles, input_dir / "vehicles.parquet"),
        "ignition_edges": str(ignition_path),
    }
