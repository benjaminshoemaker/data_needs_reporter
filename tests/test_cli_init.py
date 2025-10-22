from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnr_synth.cli import app
from dnr_synth.utils import load_yaml


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.parametrize("domain", ["ecom", "saas"])
def test_init_writes_domain_config(tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, domain: str) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init", domain], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    config_path = tmp_path / "domains" / domain / "config.yaml"
    dbt_path = tmp_path / f"dbt_{domain}"
    artifacts_path = tmp_path / "artifacts" / domain

    assert config_path.exists()
    assert dbt_path.exists()
    assert artifacts_path.exists()

    cfg = load_yaml(config_path)
    assert cfg["domain"] == domain
    assert cfg["outputs"]["warehouse"].endswith(f"data/{domain}")
    assert cfg["outputs"]["dbt_project_dir"].endswith(f"dbt_{domain}")
