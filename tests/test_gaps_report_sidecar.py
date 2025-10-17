from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnr_synth.cli import app


def test_gaps_report_writes_sidecar_and_report(monkeypatch, tmp_path: Path):
    # Prepare minimal data and artifacts
    data_dir = tmp_path / "data" / "demo" / "fct_demo"
    data_dir.mkdir(parents=True)
    # minimal parquet
    import pandas as pd
    df = pd.DataFrame({"id": [1, 2], "event_time": pd.date_range("2025-01-01", periods=2)})
    df.to_parquet(data_dir / "part.parquet")

    art_dir = tmp_path / "artifacts" / "demo"
    art_dir.mkdir(parents=True)
    (art_dir / "health_profile.json").write_text("{}")
    (art_dir / "nl_queries.json").write_text(json.dumps([]))
    (art_dir / "slack_threads.json").write_text(json.dumps([]))

    # Mock LLM output to include markdown + JSON code fence with gaps
    payload = (
        "# Exec Report\n\nSection...\n\n"
        "```json\n{\n  \"gaps\": [ {\n    \"gap_id\": \"G1\", \"type\": \"JOIN\", \"title\": \"A\", \"priority_score\": 90\n  } ]\n}\n```"
    )

    captured = {}

    def fake_run_llm(prompt: str, model: str, api_base: str, api_key: str, timeout_s: float = 0) -> str:  # noqa
        captured["timeout_s"] = timeout_s
        return payload
    # Patch the symbol used in CLI (imported binding)
    import dnr_synth.cli as cli_mod
    monkeypatch.setattr(cli_mod, "run_llm", fake_run_llm)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gaps-report",
            "--domain",
            "demo",
            "--data",
            str(tmp_path / "data" / "demo"),
            "--artifacts",
            str(tmp_path / "artifacts" / "demo"),
            "--run",
        ],
    )
    assert result.exit_code == 0, result.output
    out_dir = tmp_path / "artifacts" / "demo" / "gaps"
    assert (out_dir / "report.md").exists(), "report.md should be written"
    sidecar = out_dir / "data_gaps_report.json"
    assert sidecar.exists(), "data_gaps_report.json should be written"
    data = json.loads(sidecar.read_text())
    assert isinstance(data, list) and data and data[0]["gap_id"] == "G1"
    assert captured["timeout_s"] == 120.0

    report_text = out_dir / "report_text.md"
    assert report_text.exists(), "report_text.md should be written"
    assert "Exec Report" in report_text.read_text()
