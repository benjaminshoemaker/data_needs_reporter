"""Duplicate row corruptor."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def add(
    df: pd.DataFrame,
    rate: float,
    jitter_s: Tuple[int, int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Append duplicate rows with small time jitter."""

    if rate <= 0 or df.empty:
        return df
    dup_count = int(round(len(df) * rate))
    if dup_count <= 0:
        return df
    indices = rng.choice(df.index.to_numpy(), size=dup_count, replace=True)
    duplicates = df.loc[indices].copy()
    if "event_time" in duplicates.columns:
        event_time = pd.to_datetime(duplicates["event_time"], utc=False)
        jitter = rng.integers(jitter_s[0], jitter_s[1] + 1, size=dup_count)
        delta = pd.to_timedelta(jitter, unit="s")
        duplicates["event_time"] = event_time + delta
        if "ingested_at" in duplicates.columns:
            ingested = pd.to_datetime(duplicates["ingested_at"], utc=False)
            duplicates["ingested_at"] = ingested + delta
        if "ds" in duplicates.columns:
            duplicates["ds"] = pd.to_datetime(duplicates["event_time"]).dt.date.astype(str)
    duplicates["is_duplicate"] = True
    # Ensure no NaNs by adding the flag to the original frame before concat
    if "is_duplicate" not in df.columns:
        df = df.copy()
        df["is_duplicate"] = False
    result = pd.concat([df, duplicates], ignore_index=True)
    return result
