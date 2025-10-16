from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnr_synth.cli import app


def test_preview_command(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "demo" / "fct_demo"
    data_dir.mkdir(parents=True)
    df = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.5, 30.1]})
    df.to_parquet(data_dir / "part.parquet")

    artifacts_dir = tmp_path / "artifacts" / "demo"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "nl_queries.json").write_text(
        json.dumps([
            {
                "query": "How did approval rate trend for bin 1234?",
                "role": "Product Manager",
                "intent": "descriptive",
                "time_range": "2025-07-01 to 2025-07-07",
                "references": ["fct_demo.approved", "fct_demo.attempts"],
            }
        ])
    )
    (artifacts_dir / "slack_threads.json").write_text(
        json.dumps([
            {
                "thread_id": "s1",
                "channel": "#demo",
                "messages": [
                    {"user": "pm_riley", "text": "Investigating spike in nulls. Source: `fct_demo.value`", "ts": "1700000000.0"},
                    {"user": "da_jules", "text": "Sharing breakdown. Source: `fct_demo.id`", "ts": "1700000060.0"},
                    {"user": "eng_kai", "text": "Mitigating." , "ts": "1700000120.0"},
                ],
            }
        ])
    )

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["preview", "demo", "--limit", "5"])
    assert result.exit_code == 0
    assert "fct_demo" in result.output
    assert "NL Queries" in result.output
    assert "Slack s1" in result.output
