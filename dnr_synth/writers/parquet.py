"""Writers for parquet datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..utils import ensure_dir


def write_partitioned(
    frames: Dict[str, pd.DataFrame],
    base_path: Path,
    partition_col: str = "ds",
) -> None:
    """Write DataFrames to *base_path* partitioned by *partition_col*."""

    ensure_dir(base_path)
    for table, df in frames.items():
        table_dir = base_path / table
        ensure_dir(table_dir)
        frame = df.copy()
        if partition_col not in frame.columns and "event_time" in frame.columns:
            frame[partition_col] = pd.to_datetime(frame["event_time"]).dt.date.astype(str)
        if partition_col not in frame.columns:
            raise ValueError(f"Table {table} missing partition column '{partition_col}'")

        ordered_cols = list(frame.columns)
        frame = frame[ordered_cols]
        table_pa = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_to_dataset(
            table_pa,
            root_path=str(table_dir),
            partition_cols=[partition_col],
            existing_data_behavior="overwrite_or_ignore",
        )
