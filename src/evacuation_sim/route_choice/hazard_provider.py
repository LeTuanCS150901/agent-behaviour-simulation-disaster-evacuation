from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evacuation_sim.io.tables import read_table, write_table


class EdgeHazardProvider:
    def __init__(
        self,
        edge_hazard_time_series: str | Path,
        numerical_epsilon: float = 1e-12,
        time_lookup: str = "previous_snapshot",
        missing_edge_policy: str = "error",
    ) -> None:
        if time_lookup != "previous_snapshot":
            raise ValueError(f"Unsupported hazard time lookup: {time_lookup}")
        self.path = Path(edge_hazard_time_series)
        if not self.path.exists():
            raise FileNotFoundError(f"Missing edge hazard time series: {self.path}")
        self.df = read_table(self.path)
        required = {"time", "edge_id", "edge_hazard", "edge_survival", "edge_risk"}
        missing = sorted(required - set(self.df.columns))
        if missing:
            raise ValueError(f"Edge hazard table is missing columns: {missing}")
        if not self.df["edge_hazard"].between(0.0, 1.0).all():
            raise ValueError("edge_hazard values must be in [0,1].")
        if not self.df["edge_survival"].between(0.0, 1.0).all():
            raise ValueError("edge_survival values must be in [0,1].")
        self.numerical_epsilon = float(numerical_epsilon)
        self.missing_edge_policy = missing_edge_policy
        self.times = np.array(sorted(self.df["time"].unique()), dtype=float)
        self.by_time = {
            float(time): frame.set_index("edge_id")
            for time, frame in self.df.groupby("time")
        }

    def _snapshot_time(self, time: float) -> float:
        eligible = self.times[self.times <= float(time)]
        if len(eligible) == 0:
            raise ValueError(f"No hazard snapshot exists at or before time {time}.")
        return float(eligible[-1])

    def _row(self, edge_id: str, time: float):
        snapshot = self._snapshot_time(time)
        frame = self.by_time[snapshot]
        if edge_id not in frame.index:
            if self.missing_edge_policy == "error":
                raise KeyError(f"Edge {edge_id!r} is missing from hazard snapshot {snapshot}.")
            raise ValueError(f"Unsupported missing edge policy: {self.missing_edge_policy}")
        return frame.loc[edge_id]

    def get_edge_hazard(self, edge_id: str, time: float) -> float:
        return float(self._row(edge_id, time)["edge_hazard"])

    def get_edge_survival(self, edge_id: str, time: float) -> float:
        return float(self._row(edge_id, time)["edge_survival"])

    def compute_route_survival(self, route_edge_ids: Iterable[str], time: float) -> tuple[float, float]:
        log_sum = 0.0
        count = 0
        for edge_id in route_edge_ids:
            survival = self.get_edge_survival(edge_id, time)
            clipped = min(1.0, max(self.numerical_epsilon, survival))
            log_sum += math.log(clipped)
            count += 1
        if count == 0:
            raise ValueError("Route survival requires at least one edge.")
        return float(math.exp(log_sum)), float(log_sum)

    def compute_route_risk(self, route_edge_ids: Iterable[str], time: float) -> tuple[float, float, float]:
        survival, log_sum = self.compute_route_survival(route_edge_ids, time)
        return survival, 1.0 - survival, log_sum


def route_segmentation_sensitivity(edge_survival: float, output_path: str | Path | None = None) -> dict:
    s_one = float(edge_survival)
    s_two = float(math.exp(math.log(np.clip(edge_survival, 1e-12, 1.0)) * 2.0))
    result = {
        "edge_survival_input": s_one,
        "one_edge_route_survival": s_one,
        "one_edge_route_risk": 1.0 - s_one,
        "two_edge_same_hazard_route_survival": s_two,
        "two_edge_same_hazard_route_risk": 1.0 - s_two,
        "interpretation": (
            "The approved route-survival product includes one survival factor per edge traversal, "
            "so splitting one hazardous segment into two edges changes route survival."
        ),
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_route_hazard_samples(
    provider: EdgeHazardProvider,
    routes: list[tuple[str, list[str], float]],
    output_path: str | Path,
) -> pd.DataFrame:
    rows = []
    for route_id, edge_ids, time in routes:
        survival, risk, log_sum = provider.compute_route_risk(edge_ids, time)
        rows.append(
            {
                "time": float(time),
                "route_id": route_id,
                "edge_count": int(len(edge_ids)),
                "route_survival": survival,
                "route_risk": risk,
                "sum_log_edge_survival": log_sum,
            }
        )
    df = pd.DataFrame(rows)
    write_table(df, output_path)
    return df

