"""Evaluation report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from ..utils import ensure_dir


def write_reports(metrics: Dict[str, object], out_dir: str | Path) -> None:
    """Persist metrics to JSON and Markdown summaries."""

    out_path = Path(out_dir)
    ensure_dir(out_path)
    json_path = out_path / "health_profile.json"
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    md_path = out_path / "health_profile.md"
    lines = ["# Data Health Summary", ""]
    key_null = metrics.get("key_null_rate", {})
    if key_null:
        lines.append("## Key Null Rates")
        for table, cols in key_null.items():
            lines.append(f"- **{table}**: " + ", ".join(f"{col}={rate:.3f}" for col, rate in cols.items()))
        lines.append("")

    fk_success = metrics.get("fk_success_rate", {})
    if fk_success:
        lines.append("## Join Success")
        for table, dims in fk_success.items():
            lines.append(f"- **{table}**: " + ", ".join(f"{dim}={rate:.3f}" for dim, rate in dims.items()))
        lines.append("")

    ingest_lag = metrics.get("ingest_lag", {})
    if ingest_lag:
        lines.append("## Ingestion Lag (minutes)")
        for table, stats in ingest_lag.items():
            lines.append(f"- **{table}**: avg={stats['avg_min']:.2f}, p95={stats['p95_min']:.2f}")
        lines.append("")

    spikes = metrics.get("null_spikes", [])
    if spikes:
        lines.append("## Null Spikes")
        for spike in spikes[:10]:
            lines.append(
                f"- {spike['table']}.{spike['column']} ds={spike['ds']} null_rate={spike['null_rate']:.3f} z={spike['zscore']:.2f}"
            )
        lines.append("")

    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
