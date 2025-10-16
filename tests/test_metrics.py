from __future__ import annotations

import numpy as np
import pandas as pd

from dnr_synth.eval.metrics import health_metrics


def test_health_metrics_detects_anomalies() -> None:
    dims = pd.DataFrame({"customer_id": ["C1", "C2", "C3"]})
    facts = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", None, "C9"],
            "event_time": pd.date_range("2025-08-10", periods=4, freq="D"),
            "ingested_at": pd.date_range("2025-08-10", periods=4, freq="D") + pd.to_timedelta([0, 30, 60, 90], unit="m"),
            "ds": ["2025-08-10", "2025-08-11", "2025-08-12", "2025-08-13"],
        }
    )

    metrics = health_metrics({"dim_customer": dims, "fct_transaction": facts})

    assert metrics["key_null_rate"]["fct_transaction"]["customer_id"] > 0
    assert metrics["fk_success_rate"]["fct_transaction"]["dim_customer"] < 1
    assert metrics["ingest_lag"]["fct_transaction"]["avg_min"] > 0
    assert metrics["null_spikes"]
