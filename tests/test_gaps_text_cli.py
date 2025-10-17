from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnr_synth.cli import app


def test_gaps_text_renders_markdown(tmp_path: Path) -> None:
    payload = [
        {
            "role": "assistant",
            "type": "message",
            "content": [
                {"type": "output_text", "text": "# A) Executive Report\n\n## Top 3\n- Item 1"}
            ],
        }
    ]
    p = tmp_path / "llm_output.txt"
    p.write_text(json.dumps(payload))

    runner = CliRunner()
    res = runner.invoke(app, ["gaps-text", "--path", str(p)])
    assert res.exit_code == 0, res.output
    # Should include a Markdown heading rendered to text output
    assert "Executive Report" in res.output
    assert "Item 1" in res.output


def test_gaps_text_falls_back_to_recursive_search(tmp_path: Path) -> None:
    payload = {
        "message": {
            "parts": [
                {"meta": {"kind": "X"}},
                {"nested": {"text": "# A) Executive Report for Data Quality Gaps\n\nBody..."}},
            ]
        }
    }
    p = tmp_path / "llm_output.txt"
    p.write_text(json.dumps(payload))
    runner = CliRunner()
    res = runner.invoke(app, ["gaps-text", "--path", str(p)])
    assert res.exit_code == 0, res.output
    assert "Executive Report" in res.output
