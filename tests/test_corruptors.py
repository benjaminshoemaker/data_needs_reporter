from __future__ import annotations

import numpy as np
import pandas as pd

from dnr_synth.corruptors import duplicates, keys, lateness, null_spikes
from dnr_synth.config import NullSpike


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def test_key_presence_applies_probabilities() -> None:
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(1, 1001)],
            "platform": ["web"] * 1000,
        }
    )
    result = keys.apply_key_presence(
        df,
        {"customer_id": {"base_p": 0.99, "web_p": 0.95}},
        lambda row: {"platform": row["platform"]},
        _rng(),
    )
    null_fraction = result["customer_id"].isna().mean()
    assert 0.04 <= null_fraction <= 0.07


def test_enforce_joinability_targets_rate() -> None:
    df = pd.DataFrame({"customer_id": [f"C{i:03d}" for i in range(200)]})
    result = keys.enforce_joinability(
        df,
        {"dim_customer": 0.9},
        {"dim_customer": "customer_id"},
        _rng(),
    )
    present = result["customer_id"].notna().mean()
    assert 0.85 <= present <= 0.92


def test_duplicates_add_rows() -> None:
    df = pd.DataFrame(
        {
            "id": [f"row_{i}" for i in range(50)],
            "event_time": pd.date_range("2025-01-01", periods=50, freq="h"),
            "ingested_at": pd.date_range("2025-01-01", periods=50, freq="h"),
            "ds": ["2025-01-01"] * 50,
        }
    )
    result = duplicates.add(df, 0.2, (10, 20), _rng())
    assert len(result) > len(df)
    assert "is_duplicate" in result.columns


def test_lateness_adds_ingest_lag() -> None:
    df = pd.DataFrame(
        {
            "event_time": pd.date_range("2025-01-01", periods=10, freq="h"),
            "ingested_at": pd.date_range("2025-01-01", periods=10, freq="h"),
        }
    )
    result = lateness.add_lag(df, mean=5, sd=2, rng=_rng())
    lag = (pd.to_datetime(result["ingested_at"]) - pd.to_datetime(result["event_time"])).dt.total_seconds() / 60
    assert lag.mean() > 0


def test_null_spikes_apply_window() -> None:
    df = pd.DataFrame(
        {
            "ds": ["2025-08-10", "2025-08-11", "2025-08-12", "2025-08-13"] * 10,
            "device_type": ["mobile"] * 40,
            "platform": ["web"] * 40,
        }
    )
    spec = [
        NullSpike(
            table="fct_transaction",
            field="device_type",
            p=0.5,
            when="2025-08-10/2025-08-12",
            where="platform == 'web'",
        )
    ]
    result = null_spikes.apply(df, spec, _rng())
    spike_days = result[result["ds"].isin(["2025-08-10", "2025-08-11", "2025-08-12"])]
    assert spike_days["device_type"].isna().mean() > 0.2
