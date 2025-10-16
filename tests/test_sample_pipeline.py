from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnr_synth.cli import app
from dnr_synth.sample import (
    build_context,
    generate_queries,
    generate_threads,
    validate_queries,
    validate_threads,
)


def _write_table(path: Path, df: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / "part.parquet")


def _make_fintech_tables(base: Path) -> None:
    dates = pd.date_range("2025-07-01", periods=20, freq="D")
    df = pd.DataFrame(
        {
            "ds": dates.strftime("%Y-%m-%d"),
            "approved": [80 + (i % 5) for i in range(len(dates))],
            "attempts": [100 + (i % 7) for i in range(len(dates))],
            "bin": ["1234", "5678", "9012", "3456", "7890"] * 4,
            "region": ["NA", "EU", "APAC", "LATAM", "NA"] * 4,
            "merchant_id": [f"M{i:04d}" for i in range(len(dates))],
            "amount": [150.0 + i for i in range(len(dates))],
            "event_time": pd.date_range("2025-07-01", periods=len(dates), freq="D"),
            "ingested_at": pd.date_range("2025-07-01", periods=len(dates), freq="D")
            + pd.to_timedelta([10] * len(dates), unit="m"),
        }
    )
    _write_table(base / "fct_transaction", df)

    dim_customer = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(10)],
            "region": ["NA", "EU", "APAC", "LATAM", "NA", "EU", "APAC", "LATAM", "NA", "EU"],
            "segment": ["enterprise", "smb", "consumer", "consumer", "smb", "consumer", "enterprise", "consumer", "smb", "consumer"],
        }
    )
    _write_table(base / "dim_customer", dim_customer)


def test_sample_modules_generate(tmp_path):
    data_dir = tmp_path / "data"
    _make_fintech_tables(data_dir)
    dbt_dir = tmp_path / "dbt"
    dbt_dir.mkdir()

    ctx = build_context("fintech", str(data_dir), str(dbt_dir), seed=7)
    queries = validate_queries(generate_queries(ctx, 20, seed=7), ctx)
    assert queries
    threads = validate_threads(generate_threads(ctx, 3, seed=7), ctx)
    assert threads


def test_cli_sample_command(tmp_path):
    data_dir = tmp_path / "data"
    _make_fintech_tables(data_dir)
    dbt_dir = tmp_path / "dbt"
    dbt_dir.mkdir()

    out_dir = tmp_path / "artifacts"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "sample",
            "--domain",
            "fintech",
            "--data",
            str(data_dir),
            "--dbt",
            str(dbt_dir),
            "--out",
            str(out_dir),
            "--seed",
            "4242",
        ],
    )
    assert result.exit_code == 0, result.output
    nl_path = out_dir / "nl_queries.json"
    slack_path = out_dir / "slack_threads.json"
    assert nl_path.exists()
    assert slack_path.exists()
    queries = json.loads(nl_path.read_text())
    threads = json.loads(slack_path.read_text())
    assert len(queries) >= 10
    assert len(threads) >= 1
