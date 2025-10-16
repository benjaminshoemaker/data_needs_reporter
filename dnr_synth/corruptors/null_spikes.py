"""Null spike corruptor."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import NullSpike


def apply(df: pd.DataFrame, spec: Iterable[NullSpike], rng: np.random.Generator) -> pd.DataFrame:
    """Apply conditional null spikes to *df*."""

    result = df.copy()
    if df.empty:
        return result
    ds_series = None
    if "ds" in result.columns:
        ds_series = pd.to_datetime(result["ds"])
    elif "event_time" in result.columns:
        ds_series = pd.to_datetime(result["event_time"]).dt.floor("D")

    for spike in spec:
        if spike.field not in result.columns:
            continue
        start_str, end_str = spike.when.split("/")
        start = datetime.fromisoformat(start_str)
        end = datetime.fromisoformat(end_str)
        mask = pd.Series(True, index=result.index)
        if ds_series is not None:
            mask &= (ds_series >= start) & (ds_series <= end)
        if spike.where:
            try:
                where_mask = result.eval(spike.where)
                mask &= where_mask.astype(bool)
            except Exception:
                continue
        indices = result.index[mask]
        if not len(indices):
            continue
        drop_mask = rng.random(len(indices)) < spike.p
        to_null = indices[drop_mask]
        result.loc[to_null, spike.field] = pd.NA
    return result
