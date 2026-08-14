from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FireSnapshotProvider:
    table: pd.DataFrame
    numerical_epsilon: float
    time_lookup: str
    missing_edge_policy: str
    before_first_policy: str
    after_last_policy: str
    _snapshots: dict[float, pd.DataFrame] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        required = {"time", "edge_id", "edge_hazard", "edge_survival", "edge_risk"}
        missing = required - set(self.table.columns)
        if missing:
            raise ValueError(f"Edge hazard table is missing columns: {sorted(missing)}")
        if self.table.duplicated(["time", "edge_id"]).any():
            raise ValueError("Edge hazard table contains duplicate (time,edge_id) keys")
        if self.time_lookup != "previous_snapshot" or self.missing_edge_policy != "error":
            raise ValueError("Only previous_snapshot lookup and missing-edge error policy are supported")
        if self.before_first_policy != "error" or self.after_last_policy != "hold_last":
            raise ValueError("Unsupported before-first or after-last snapshot policy")
        if not np.isfinite(self.table[["time", "edge_hazard", "edge_survival", "edge_risk"]].to_numpy(float)).all():
            raise ValueError("Edge hazard table contains non-finite values")
        if self.numerical_epsilon <= 0 or self.numerical_epsilon > 1:
            raise ValueError("numerical_epsilon must lie in (0,1]")
        if "interacting_fronts" in self.table:
            if self.table["interacting_fronts"].isna().any() or not pd.api.types.is_bool_dtype(self.table["interacting_fronts"]):
                raise ValueError("interacting_fronts must be a non-null boolean column")
            values_per_time = self.table.groupby("time")["interacting_fronts"].nunique()
            if (values_per_time != 1).any():
                raise ValueError("Every hazard snapshot must have one interacting_fronts value")
        object.__setattr__(self, "_snapshots", {
            float(time_value): group.set_index("edge_id", drop=False)
            for time_value, group in self.table.groupby("time", sort=True)
        })

    @property
    def times(self) -> np.ndarray:
        return np.sort(self.table["time"].unique().astype(float))

    def snapshot_time(self, query_time: float) -> float:
        eligible = self.times[self.times <= float(query_time)]
        if not len(eligible):
            raise ValueError(f"No hazard snapshot exists at or before time {query_time}")
        return float(eligible[-1])

    def snapshot(self, query_time: float) -> pd.DataFrame:
        return self._snapshots[self.snapshot_time(query_time)]

    def snapshot_metadata(self, query_time: float) -> dict[str, float | bool]:
        snapshot_time = self.snapshot_time(query_time)
        snapshot = self._snapshots[snapshot_time]
        interacting = bool(snapshot["interacting_fronts"].iloc[0]) if "interacting_fronts" in snapshot else False
        return {"snapshot_time": snapshot_time, "interacting_fronts": interacting}

    def route_risk(self, edge_ids: Iterable[str], query_time: float) -> tuple[float, float, float]:
        ordered = list(edge_ids)
        if not ordered:
            raise ValueError("Route risk requires at least one ordered edge traversal")
        snapshot = self.snapshot(query_time)
        missing = [edge_id for edge_id in ordered if edge_id not in snapshot.index]
        if missing:
            raise KeyError(f"Edges missing from hazard snapshot {self.snapshot_time(query_time)}: {missing[:5]}")
        survivals = snapshot.loc[ordered, "edge_survival"].to_numpy(float)
        log_sum = float(np.log(np.clip(survivals, self.numerical_epsilon, 1.0)).sum())
        survival = float(math.exp(log_sum))
        return survival, 1.0 - survival, log_sum


@dataclass(frozen=True)
class Stage5DecisionEngine:
    alpha_t: float
    alpha_h: float
    probability_tolerance: float

    def __post_init__(self) -> None:
        values = np.array([self.alpha_t, self.alpha_h, self.probability_tolerance], dtype=float)
        if not np.isfinite(values).all() or self.alpha_t < 0 or self.alpha_h < 0 or self.probability_tolerance <= 0:
            raise ValueError("Stage 5 weights must be finite/non-negative and probability_tolerance positive")

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum()

    def probabilities(self, travel_times: np.ndarray, route_risks: np.ndarray, panic_rate: float) -> dict[str, np.ndarray]:
        travel_times = np.asarray(travel_times, dtype=float)
        route_risks = np.asarray(route_risks, dtype=float)
        if travel_times.ndim != 1 or route_risks.shape != travel_times.shape or not len(travel_times):
            raise ValueError("travel_times and route_risks must be non-empty one-dimensional arrays of equal shape")
        if not np.isfinite(travel_times).all() or np.any(travel_times < 0) or not np.isfinite(route_risks).all() or np.any(route_risks < 0) or np.any(route_risks > 1):
            raise ValueError("Travel times and route risks must be finite and within their valid ranges")
        if not np.isfinite(panic_rate) or not 0 <= panic_rate <= 1:
            raise ValueError("panic_rate must be finite and lie in [0,1]")
        maximum = float(travel_times.max())
        if maximum <= 0:
            raise ValueError("At least one candidate travel time must be positive")
        normalized = travel_times / maximum
        utility = -self.alpha_t * normalized - self.alpha_h * route_risks
        risk_neutral_utility = -self.alpha_t * normalized
        logit = self._softmax(utility)
        no_risk_logit = self._softmax(risk_neutral_utility)
        uniform = np.full(len(travel_times), 1.0 / len(travel_times))
        final = (1.0 - panic_rate) * logit + panic_rate * uniform
        no_risk_final = (1.0 - panic_rate) * no_risk_logit + panic_rate * uniform
        if not np.isclose(final.sum(), 1.0, atol=self.probability_tolerance) or np.any(final < 0):
            raise ValueError("Final route-choice probability vector is invalid")
        return {
            "normalized_travel_time": normalized,
            "utility": utility,
            "risk_neutral_utility": risk_neutral_utility,
            "behavioral_probability": logit,
            "risk_neutral_behavioral_probability": no_risk_logit,
            "final_probability": final,
            "risk_neutral_probability": no_risk_final,
        }
