from __future__ import annotations

import numpy as np
import pandas as pd

from dnr_synth.corruptors import duplicates


def _rng(seed: int = 17) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def test_duplicates_sets_flags_consistently() -> None:
    df = pd.DataFrame(
        {
            "id": [f"row_{i}" for i in range(20)],
            "event_time": pd.date_range("2025-01-01", periods=20, freq="h"),
            "ingested_at": pd.date_range("2025-01-01", periods=20, freq="h"),
        }
    )
    out = duplicates.add(df, rate=0.25, jitter_s=(5, 10), rng=_rng())
    assert "is_duplicate" in out.columns
    # Original rows should be marked False
    assert out["is_duplicate"].head(len(df)).eq(False).all()
    # At least one duplicate appended and marked True
    assert out["is_duplicate"].sum() >= 1
