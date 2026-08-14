from __future__ import annotations

import math

import numpy as np
import pandas as pd


VALID_ZONE_TYPES = {"red", "blue"}


def allocate_agents_largest_remainder(
    census: pd.DataFrame,
    population_column: str,
    total_population: int,
) -> pd.Series:
    """Allocate exactly total_population agents using Hamilton largest remainder."""
    if total_population <= 0:
        raise ValueError("total_population must be positive.")
    if population_column not in census.columns:
        raise ValueError(f"Missing census population column: {population_column}")

    populations = pd.to_numeric(census[population_column], errors="raise")
    if populations.isna().any():
        raise ValueError("Census population contains null values.")
    if (populations < 0).any():
        raise ValueError("Census population must be non-negative.")
    total_census = populations.sum()
    if total_census <= 0:
        raise ValueError("Total census population must be positive.")

    quotas = populations / total_census * total_population
    floors = np.floor(quotas).astype(int)
    remainder = int(total_population - floors.sum())
    order = np.argsort(-(quotas - floors).to_numpy(), kind="mergesort")
    allocation = floors.to_numpy(copy=True)
    if remainder > 0:
        allocation[order[:remainder]] += 1
    return pd.Series(allocation, index=census.index, name="synthetic_count")


def validate_zone_types(zones: pd.DataFrame, zone_type_column: str) -> None:
    if zone_type_column not in zones.columns:
        raise ValueError(f"Missing zone type column: {zone_type_column}")
    values = set(zones[zone_type_column].dropna().astype(str).str.lower())
    invalid = sorted(values - VALID_ZONE_TYPES)
    if invalid:
        raise ValueError(
            f"Invalid zone_type values {invalid}; expected only {sorted(VALID_ZONE_TYPES)}."
        )


def build_population_validation(
    zones,
    zone_id_column: str,
    zone_type_column: str,
    census_population_column: str,
    tolerance_pct: float,
) -> pd.DataFrame:
    rows = []
    total_census = float(zones[census_population_column].sum())
    total_synth = int(zones["synthetic_count"].sum())
    for _, row in zones.iterrows():
        census_pop = float(row[census_population_column])
        synthetic_count = int(row["synthetic_count"])
        target = census_pop / total_census if total_census else math.nan
        observed = synthetic_count / total_synth if total_synth else math.nan
        relative_error = abs(observed - target) / target if target > 0 else 0.0
        rows.append(
            {
                "zone_id": row[zone_id_column],
                "zone_type": row[zone_type_column],
                "census_population": census_pop,
                "target_share": target,
                "synthetic_count": synthetic_count,
                "synthetic_share": observed,
                "relative_error": relative_error,
                "within_tolerance": bool(relative_error <= tolerance_pct),
            }
        )
    return pd.DataFrame(rows)
