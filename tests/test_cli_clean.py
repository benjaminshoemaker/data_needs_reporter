from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnr_synth.cli import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_clean_domain_removes_data_and_artifacts(tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = Path("data/fintech")
    art_dir = Path("artifacts/fintech")
    report_md = Path("report_previous.md")
    (data_dir).mkdir(parents=True)
    (art_dir / "gaps").mkdir(parents=True)
    (data_dir / "part.parquet").write_text("dummy")
    (art_dir / "gaps" / "report.md").write_text("dummy")
    prompt_nl = art_dir / "PROMPT_NL_QUERIES.md"
    prompt_slack = art_dir / "PROMPT_SLACK_THREADS.md"
    prompt_nl.write_text("manual prompt")
    prompt_slack.write_text("manual prompt")
    nl_queries = art_dir / "nl_queries.json"
    slack_threads = art_dir / "slack_threads.json"
    nl_queries.write_text("[]")
    slack_threads.write_text("[]")
    report_md.write_text("old report")

    result = runner.invoke(app, ["clean", "--domain", "fintech", "--yes"], catch_exceptions=False)

    assert result.exit_code == 0
    assert not data_dir.exists()
    assert not art_dir.joinpath("gaps").exists()
    assert prompt_nl.exists()
    assert prompt_slack.exists()
    assert art_dir.exists()
    remaining = sorted(p.name for p in art_dir.iterdir())
    assert remaining == [
        "PROMPT_NL_QUERIES.md",
        "PROMPT_SLACK_THREADS.md",
        "nl_queries.json",
        "slack_threads.json",
    ]
    assert not report_md.exists()


def test_clean_all_removes_all_domains(tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for domain in ("fintech", "ecom"):
        (Path("data") / domain).mkdir(parents=True)
        art_dir = (Path("artifacts") / domain)
        art_dir.mkdir(parents=True)
        (art_dir / "PROMPT_NL_QUERIES.md").write_text("manual")
        (art_dir / "PROMPT_SLACK_THREADS.md").write_text("manual")
        (art_dir / "nl_queries.json").write_text("[]")
        (art_dir / "slack_threads.json").write_text("[]")

    result = runner.invoke(app, ["clean", "--all", "--yes"], catch_exceptions=False)

    assert result.exit_code == 0
    data_root = Path("data")
    art_root = Path("artifacts")
    assert not any(data_root.glob("*")) if data_root.exists() else True
    if art_root.exists():
        for domain in ("fintech", "ecom"):
            domain_dir = art_root / domain
            if domain_dir.exists():
                remaining = sorted(p.name for p in domain_dir.iterdir())
                assert remaining == [
                    "PROMPT_NL_QUERIES.md",
                    "PROMPT_SLACK_THREADS.md",
                    "nl_queries.json",
                    "slack_threads.json",
                ]
