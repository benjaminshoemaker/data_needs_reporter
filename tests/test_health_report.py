from __future__ import annotations

from pathlib import Path

from dnr_synth.eval.report import write_reports


def test_write_reports_creates_executive_summary(tmp_path: Path) -> None:
    metrics = {
        "key_null_rate": {"fct_transaction": {"customer_id": 0.12}},
        "fk_success_rate": {"fct_transaction": {"dim_customer": 0.94}},
        "duplicate_rate": {"fct_transaction": 0.08},
        "ingest_lag": {"fct_transaction": {"avg_min": 12.3, "p95_min": 45.6}},
        "null_spikes": [
            {
                "table": "fct_transaction",
                "column": "amount",
                "ds": "2025-07-14",
                "null_rate": 0.22,
                "zscore": 2.1,
            }
        ],
    }

    write_reports(metrics, tmp_path)

    md_path = tmp_path / "health_profile.md"
    assert md_path.exists()
    text = md_path.read_text()

    # Executive summary should flag the high risks
    assert "Missing IDs" in text
    assert "Join failures" in text
    assert "Duplicate rows" in text
    assert "Freshness risk" in text
    assert "Null spike" in text

    # Sections are present
    for heading in (
        "## Key Findings",
        "## Recommended Actions",
        "## Glossary",
        "## Appendix",
    ):
        assert heading in text

    # Appendix tables should contain traffic-light statuses
    assert "| fct_transaction | customer_id | 12.0% | High |" in text
    assert "| fct_transaction | dim_customer | 94.0% | 6.0% | High |" in text
    assert "| fct_transaction | 8.0% | High |" in text
    assert "| fct_transaction | 12.3 | 45.6 | High |" in text
    assert "| fct_transaction | amount | 2025-07-14 | 22.0% | 2.10 | High |" in text


def test_write_reports_handles_missing_metrics(tmp_path: Path) -> None:
    write_reports({}, tmp_path)
    text = (tmp_path / "health_profile.md").read_text()

    assert "Overall health" in text
    # Each category should note absence
    assert "Completeness" in text and "Not reported" in text
