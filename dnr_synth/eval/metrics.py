"""Evaluation metrics for generated datasets."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def health_metrics(frames: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Compute health metrics for a dict of DataFrames."""

    metrics: Dict[str, object] = {}
    metrics["key_null_rate"] = _key_null_rates(frames)
    fk_success = _fk_success_rates(frames)
    metrics["fk_success_rate"] = fk_success
    metrics["orphan_rate"] = {
        table: {dim: float(max(0.0, 1.0 - rate)) for dim, rate in dims.items()}
        for table, dims in fk_success.items()
    }
    metrics["duplicate_rate"] = _duplicate_rates(frames)
    metrics["ingest_lag"] = _ingest_lag_stats(frames)
    metrics["null_spikes"] = _null_spikes(frames)
    return metrics


def _key_null_rates(frames: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    rates: Dict[str, Dict[str, float]] = {}
    for table, df in frames.items():
        cols = [c for c in df.columns if c.endswith("_id") or c == "id"]
        if not cols:
            continue
        rates[table] = {col: float(df[col].isna().mean()) for col in cols}
    return rates


def _fk_success_rates(frames: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    dims = {
        name: _dimension_keys(df)
        for name, df in frames.items()
        if name.startswith("dim_")
    }
    results: Dict[str, Dict[str, float]] = {}
    for table, df in frames.items():
        if table.startswith("dim_"):
            continue
        for col in [c for c in df.columns if c.endswith("_id")]:
            dim_name = f"dim_{col[:-3]}"
            if dim_name not in dims:
                continue
            values = df[col].dropna()
            if values.empty:
                continue
            dim_keys = dims[dim_name]
            success = float(values.isin(dim_keys).mean())
            results.setdefault(table, {})[dim_name] = success
    return results


def _dimension_keys(df: pd.DataFrame) -> pd.Series:
    preferred = [c for c in df.columns if c.endswith("_id") or c.endswith("_sk") or c == "id"]
    column = preferred[0] if preferred else df.columns[0]
    return df[column].dropna().drop_duplicates()


def _duplicate_rates(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for table, df in frames.items():
        if df.empty:
            rates[table] = 0.0
            continue
        key_cols = [c for c in df.columns if c.endswith("_id") or c.endswith("_uuid") or c == "id"]
        if not key_cols:
            key_cols = [df.columns[0]]
        unique = df[key_cols].dropna().drop_duplicates()
        rate = 1.0 - (len(unique) / max(1, len(df)))
        rates[table] = float(max(0.0, min(1.0, rate)))
    return rates


def _ingest_lag_stats(frames: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for table, df in frames.items():
        if {"event_time", "ingested_at"}.issubset(df.columns):
            event = pd.to_datetime(df["event_time"])
            ingest = pd.to_datetime(df["ingested_at"])
            lag = (ingest - event).dt.total_seconds() / 60.0
            if lag.empty:
                continue
            stats[table] = {
                "avg_min": float(lag.mean()),
                "p95_min": float(np.percentile(lag, 95)),
            }
    return stats


def _null_spikes(frames: Dict[str, pd.DataFrame]) -> list[Dict[str, object]]:
    spikes: list[Dict[str, object]] = []
    for table, df in frames.items():
        if "ds" not in df.columns:
            continue
        grouped = df.groupby("ds")
        for column in df.columns:
            if df[column].isna().sum() == 0:
                continue
            daily = grouped[column].apply(lambda s: float(s.isna().mean()))
            std = daily.std(ddof=0)
            if std == 0:
                continue
            zscores = (daily - daily.mean()) / std
            idx = zscores.idxmax()
            if zscores.loc[idx] >= 1.5:
                spikes.append(
                    {
                        "table": table,
                        "column": column,
                        "ds": idx,
                        "zscore": float(zscores.loc[idx]),
                        "null_rate": float(daily.loc[idx]),
                    }
                )
    return spikes
