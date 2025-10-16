"""Data context builder for grounded sampling."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from pydantic import BaseModel, Field


class MetricSpec(BaseModel):
    name: str
    table: str
    formula: str
    grain: str
    cols_used: List[str]


class SignalAnomaly(BaseModel):
    metric: str
    scope: str
    when: Dict[str, str]
    delta: float
    direction: str


class SignalNullSpike(BaseModel):
    table: str
    col: str
    ds: str
    null_rate: float
    z: float


class SignalLagSpike(BaseModel):
    table: str
    ds: str
    p95_minutes: float


class Signals(BaseModel):
    anomalies: List[SignalAnomaly] = Field(default_factory=list)
    null_spikes: List[SignalNullSpike] = Field(default_factory=list)
    lag_spikes: List[SignalLagSpike] = Field(default_factory=list)


class TableProfile(BaseModel):
    row_count: int
    columns: List[str]
    dtypes: Dict[str, str]


class DataContext(BaseModel):
    domain: str
    calendar: Dict[str, str]
    tables: Dict[str, TableProfile]
    enums: Dict[str, List[str]]
    numerics: Dict[str, Dict[str, float]]
    facts: Dict[str, List[str]]
    metrics: List[MetricSpec]
    signals: Signals


def build_context(domain: str, data_dir: str, dbt_dir: str | None, seed: int | None) -> DataContext:
    base_path = Path(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    rng = np.random.default_rng(seed or 7)
    tables: Dict[str, TableProfile] = {}
    enums: Dict[str, List[str]] = {}
    numerics: Dict[str, Dict[str, float]] = {}
    facts: Dict[str, List[str]] = defaultdict(list)
    metrics: List[MetricSpec] = []
    signals = Signals()

    ds_min: datetime | None = None
    ds_max: datetime | None = None

    for table_dir in sorted(p for p in base_path.iterdir() if p.is_dir()):
        table_name = table_dir.name
        dataset = ds.dataset(str(table_dir), format="parquet", partitioning="hive")
        table = dataset.to_table()
        df = table.to_pandas()
        if df.empty:
            continue

        df_columns = list(df.columns)
        dtypes = {c: str(df[c].dtype) for c in df_columns}
        tables[table_name] = TableProfile(row_count=len(df), columns=df_columns, dtypes=dtypes)

        ds_series = _detect_ds_series(df)
        if ds_series is not None and not ds_series.empty:
            start = ds_series.min()
            end = ds_series.max()
            ds_min = start if ds_min is None else min(ds_min, start)
            ds_max = end if ds_max is None else max(ds_max, end)

        _collect_enums(df, table_name, enums)
        _collect_numerics(df, table_name, numerics)
        _collect_facts(domain, df, table_name, facts)
        metrics.extend(_derive_metrics(df, table_name))

        _detect_anomalies(df, table_name, signals)
        _detect_null_spikes(df, table_name, signals)
        _detect_lag_spikes(df, table_name, signals)

    if ds_min is None or ds_max is None:
        # default calendar
        now = datetime.utcnow()
        ds_min = now - timedelta(days=30)
        ds_max = now

    calendar = {
        "start": ds_min.isoformat(),
        "end": ds_max.isoformat(),
        "tz": "UTC",
    }

    # deduplicate facts values and trim
    facts_clean = {k: sorted(set(v))[:100] for k, v in facts.items() if v}

    return DataContext(
        domain=domain,
        calendar=calendar,
        tables=tables,
        enums=enums,
        numerics=numerics,
        facts=facts_clean,
        metrics=_unique_metrics(metrics),
        signals=signals,
    )


def _detect_ds_series(df: pd.DataFrame) -> pd.Series | None:
    if "ds" in df.columns:
        return pd.to_datetime(df["ds"]).dropna()
    if "event_time" in df.columns:
        return pd.to_datetime(df["event_time"]).dt.floor("D").dropna()
    return None


def _collect_enums(df: pd.DataFrame, table: str, enums: Dict[str, List[str]]) -> None:
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        values = (
            df[col]
            .dropna()
            .astype(str)
            .value_counts()
            .head(100)
            .index
            .tolist()
        )
        if values:
            enums[f"{table}.{col}"] = values


def _collect_numerics(df: pd.DataFrame, table: str, numerics: Dict[str, Dict[str, float]]) -> None:
    numeric_cols = df.select_dtypes(include=["number", "datetime64[ns]"]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        stats = {
            "min": float(series.min()),
            "p25": float(np.percentile(series, 25)),
            "p50": float(np.percentile(series, 50)),
            "p75": float(np.percentile(series, 75)),
            "p95": float(np.percentile(series, 95)),
            "max": float(series.max()),
        }
        numerics[f"{table}.{col}"] = stats


def _collect_facts(domain: str, df: pd.DataFrame, table: str, facts: Dict[str, List[str]]) -> None:
    domain_fields = {
        "fintech": [
            ("bin", "bins"),
            ("region", "regions"),
            ("merchant_category", "merchants"),
            ("merchant_id", "merchants"),
            ("card_network", "bins"),
        ],
        "ecom": [
            ("product_id", "products"),
            ("category", "categories"),
            ("channel", "channels"),
            ("region", "regions"),
        ],
        "saas": [
            ("plan", "plans"),
            ("feature_area", "features"),
            ("account_id", "accounts"),
            ("region", "regions"),
        ],
    }
    mappings = domain_fields.get(domain, [])
    for col, bucket in mappings:
        if col in df.columns:
            facts[bucket].extend(
                df[col]
                .dropna()
                .astype(str)
                .value_counts()
                .head(100)
                .index
                .tolist()
            )


def _derive_metrics(df: pd.DataFrame, table: str) -> List[MetricSpec]:
    specs: List[MetricSpec] = []
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        name = f"avg_{col}"
        cols_used = [f"{table}.{col}"]
        if "ds" in df.columns:
            cols_used.append(f"{table}.ds")
        elif "event_time" in df.columns:
            cols_used.append(f"{table}.event_time")
        specs.append(
            MetricSpec(
                name=name,
                table=table,
                formula=f"AVG({col})",
                grain="daily",
                cols_used=cols_used,
            )
        )
    return specs


def _detect_anomalies(df: pd.DataFrame, table: str, signals: Signals) -> None:
    if "approved" not in df.columns or "attempts" not in df.columns:
        return
    if "ds" in df.columns:
        ds_col = pd.to_datetime(df["ds"])
    elif "event_time" in df.columns:
        ds_col = pd.to_datetime(df["event_time"]).dt.floor("D")
    else:
        return
    daily = df.assign(__ds=ds_col).groupby("__ds").agg({"approved": "sum", "attempts": "sum"}).reset_index()
    daily = daily.rename(columns={"__ds": "ds"})
    daily = daily[daily["attempts"] > 0]
    if len(daily) < 8:
        return
    daily["rate"] = daily["approved"] / daily["attempts"]
    daily["median7"] = daily["rate"].rolling(window=7, min_periods=3).median()
    daily["delta"] = daily["rate"] - daily["median7"]
    drops = daily[daily["delta"] <= -0.05]
    for _, row in drops.iterrows():
        ds_str = pd.to_datetime(row["ds"]).date().isoformat()
        signals.anomalies.append(
            SignalAnomaly(
                metric=f"{table}.approved",
                scope=f"{table}",
                when={"start": ds_str, "end": ds_str},
                delta=float(row["delta"]),
                direction="down",
            )
        )


def _detect_null_spikes(df: pd.DataFrame, table: str, signals: Signals) -> None:
    if "ds" not in df.columns:
        return
    grouping = df.groupby("ds")
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        null_rate = grouping[col].apply(lambda s: s.isna().mean())
        if len(null_rate) < 14:
            continue
        null_rate.index = pd.to_datetime(null_rate.index)
        rolling_mean = null_rate.rolling(window=14, min_periods=7).mean()
        rolling_std = null_rate.rolling(window=14, min_periods=7).std(ddof=0)
        zscores = (null_rate - rolling_mean) / (rolling_std.replace(0, np.nan))
        spikes = zscores[zscores >= 3].dropna()
        for ds_value, z in spikes.items():
            signals.null_spikes.append(
                SignalNullSpike(
                    table=table,
                    col=col,
                    ds=pd.to_datetime(ds_value).date().isoformat(),
                    null_rate=float(null_rate.loc[ds_value]),
                    z=float(z),
                )
            )


def _detect_lag_spikes(df: pd.DataFrame, table: str, signals: Signals) -> None:
    if "event_time" not in df.columns or "ingested_at" not in df.columns:
        return
    event = pd.to_datetime(df["event_time"])
    ingested = pd.to_datetime(df["ingested_at"])
    lag_minutes = (ingested - event).dt.total_seconds() / 60.0
    df_lag = pd.DataFrame({"ds": event.dt.floor("D"), "lag": lag_minutes})
    daily = df_lag.groupby("ds")["lag"].quantile(0.95)
    spikes = daily[daily > 30]
    for ds_value, p95 in spikes.items():
        ds_str = pd.to_datetime(ds_value).date().isoformat()
        signals.lag_spikes.append(
            SignalLagSpike(table=table, ds=ds_str, p95_minutes=float(p95))
        )


def _unique_metrics(metrics: List[MetricSpec]) -> List[MetricSpec]:
    seen = {}
    for spec in metrics:
        seen[f"{spec.table}.{spec.name}"] = spec
    return list(seen.values())
