from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, path: str | Path) -> str:
    """Write a table to Parquet, failing clearly if no engine is installed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not write {path}: install pyarrow or fastparquet for Parquet outputs."
        ) from exc
    return str(path)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not read {path}: install pyarrow or fastparquet for Parquet inputs."
        ) from exc
