"""Ingestion lateness corruptor."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_lag(df: pd.DataFrame, mean: float, sd: float, rng: np.random.Generator) -> pd.DataFrame:
    """Adjust ingested_at timestamps with Gaussian lag."""

    if "event_time" not in df.columns:
        return df
    result = df.copy()
    event_time = pd.to_datetime(result["event_time"], utc=False)
    lag_minutes = np.clip(rng.normal(loc=mean, scale=max(sd, 1e-6), size=len(result)), a_min=0, a_max=None)
    lag_delta = pd.to_timedelta(lag_minutes, unit="m")
    if "ingested_at" not in result.columns:
        result["ingested_at"] = event_time
    result["ingested_at"] = event_time + lag_delta
    result["ingest_lag_min"] = lag_minutes
    return result
