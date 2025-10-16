"""Schema drift utilities."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable

import pandas as pd

from ..config import Drift


def apply(
    frames: Dict[str, pd.DataFrame],
    spec: Iterable[Drift],
) -> Dict[str, pd.DataFrame]:
    """Apply schema drift events to the provided *frames*."""

    result = {name: df.copy() for name, df in frames.items()}
    for drift in spec:
        table = drift.table
        if table not in result:
            continue
        df = result[table]
        if "ds" in df.columns:
            ds_series = pd.to_datetime(df["ds"]).dt.floor("D")
        elif "event_time" in df.columns:
            ds_series = pd.to_datetime(df["event_time"]).dt.floor("D")
        else:
            continue
        mask = ds_series >= datetime.combine(drift.at, datetime.min.time())

        if drift.rename:
            for old, new in drift.rename.items():
                if old not in df.columns:
                    continue
                if new not in df.columns:
                    df[new] = pd.NA
                df.loc[mask, new] = df.loc[mask, old]

        if drift.add:
            for column, dtype_str in drift.add.items():
                if column not in df.columns:
                    df[column] = pd.Series(pd.NA, index=df.index, dtype="object")
                fill_value = _default_for_dtype(dtype_str)
                df.loc[mask, column] = fill_value
    return result


def _default_for_dtype(dtype: str) -> object:
    dtype = dtype.lower()
    if dtype in {"string", "str"}:
        return "unknown"
    if dtype in {"int", "integer"}:
        return 0
    if dtype in {"float", "number"}:
        return 0.0
    return None
