from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner
import pandas as pd

from dnr_synth.cli import app


def _setup_artifacts(tmp: Path) -> tuple[Path, Path]:
    data_dir = tmp / "data" / "demo"
    tdir = data_dir / "fct_demo"
    tdir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "ds": ["2025-01-01", "2025-01-02"],
            "value": [10.0, 20.0],
            "event_time": pd.date_range("2025-01-01", periods=2, freq="D"),
            "ingested_at": pd.date_range("2025-01-01", periods=2, freq="D"),
        }
    )
    df.to_parquet(tdir / "part.parquet")

    art_dir = tmp / "artifacts" / "demo"
    art_dir.mkdir(parents=True)
    (art_dir / "health_profile.json").write_text(json.dumps({"tables": {"fct_demo": {"row_count": 2}}}))
    (art_dir / "nl_queries.json").write_text(
        json.dumps([
            {
                "query": "avg value by ds",
                "role": "Product Manager",
                "intent": "descriptive",
                "time_range": "2025-01-01 to 2025-01-02",
                "references": ["fct_demo.value", "fct_demo.ds"],
            }
        ])
    )
    (art_dir / "slack_threads.json").write_text(
        json.dumps(
            [
                {
                    "thread_id": "s1",
                    "channel": "#demo",
                    "messages": [
                        {"user": "pm_riley", "text": "Check fct_demo.value", "ts": "1700000000.0"}
                    ],
                }
            ]
        )
    )
    return data_dir, art_dir


def test_gaps_report_emits_prompt(tmp_path: Path) -> None:
    data_dir, art_dir = _setup_artifacts(tmp_path)
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gaps-report",
            "--domain",
            "demo",
            "--data",
            str(data_dir),
            "--artifacts",
            str(art_dir),
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    prompt_path = out_dir / "PROMPT_GAPS.md"
    assert prompt_path.exists()
    text = prompt_path.read_text()
    assert "You are an expert data quality analyst" in text
    assert "DATASET SCHEMA" in text
    assert "NL QUERIES" in text
    assert "SLACK THREADS" in text
