"""Evaluation report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ..utils import ensure_dir


HIGH_KEY_NULL = 0.05
HIGH_FK_SUCCESS = 0.98
HIGH_DUPLICATE = 0.05
HIGH_LAG_P95 = 30.0
WATCH_LAG_P95 = 15.0


def write_reports(metrics: Dict[str, object], out_dir: str | Path) -> None:
    """Persist metrics to JSON and Markdown summaries."""

    out_path = Path(out_dir)
    ensure_dir(out_path)
    json_path = out_path / "health_profile.json"
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    md_path = out_path / "health_profile.md"
    md_path.write_text(_build_markdown(metrics), encoding="utf-8")


def _build_markdown(metrics: Dict[str, object]) -> str:
    key_null_rows = _flatten_key_null(metrics.get("key_null_rate", {}))
    fk_rows = _flatten_fk_success(metrics.get("fk_success_rate", {}))
    duplicate_rows = _flatten_duplicates(metrics.get("duplicate_rate", {}))
    lag_rows = _flatten_lag(metrics.get("ingest_lag", {}))
    spikes = metrics.get("null_spikes", []) or []

    summary = _build_executive_summary(key_null_rows, fk_rows, duplicate_rows, lag_rows, spikes)

    lines: List[str] = []
    lines.append("# Data Health – Executive Summary")
    lines.extend(summary)
    lines.append("")

    lines.append("## Key Findings")
    lines.extend(_section_completeness(key_null_rows))
    lines.extend(_section_join_integrity(fk_rows))
    lines.extend(_section_duplicates(duplicate_rows))
    lines.extend(_section_freshness(lag_rows))
    lines.extend(_section_anomalies(spikes))
    lines.append("")

    lines.append("## Recommended Actions")
    near, medium = _recommended_actions(key_null_rows, fk_rows, duplicate_rows, lag_rows, spikes)
    lines.append("### Near-term (0–14d)")
    lines.extend(near or ["- Monitor metrics weekly; no high-risk items reported."])
    lines.append("")
    lines.append("### Medium-term (15–45d)")
    lines.extend(medium or ["- Review data contracts and alerts monthly."])
    lines.append("")

    lines.append("## Glossary")
    lines.extend(
        [
            "- **Key null rate:** Share of rows where a key column (e.g., customer_id) is missing. High rates block attribution and joins.",
            "- **FK success rate:** Fraction of fact rows that successfully join to their dimension table. Low success undercounts KPIs.",
            "- **Orphan rate:** Rows that fail the join (1 − FK success). Orphans remove revenue or user activity from dashboards.",
            "- **Duplicate rate:** Share of duplicate keys; inflates counts and skews KPIs.",
            "- **p95 ingest lag:** Minutes between event time and data availability for the slowest 5% of rows; high lag delays decisions.",
            "- **Null spike:** A day where nulls surge for a column, signalling a pipeline or source outage.",
        ]
    )
    lines.append("")

    lines.append("## Appendix")
    lines.extend(_appendix_tables(key_null_rows, fk_rows, duplicate_rows, lag_rows, spikes))

    return "\n".join(lines).strip() + "\n"


def _flatten_key_null(data: Dict[str, Dict[str, float]]) -> List[Tuple[str, str, float]]:
    rows: List[Tuple[str, str, float]] = []
    for table, cols in data.items():
        for col, rate in cols.items():
            rows.append((table, col, float(rate)))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows


def _flatten_fk_success(data: Dict[str, Dict[str, float]]) -> List[Tuple[str, str, float, float]]:
    rows: List[Tuple[str, str, float, float]] = []
    for table, dims in data.items():
        for dim, rate in dims.items():
            success = float(rate)
            orphan = max(0.0, 1.0 - success)
            rows.append((table, dim, success, orphan))
    rows.sort(key=lambda x: x[2])  # ascending success, worst first
    return rows


def _flatten_duplicates(data: Dict[str, float]) -> List[Tuple[str, float]]:
    rows = [(table, float(rate)) for table, rate in data.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _flatten_lag(data: Dict[str, Dict[str, float]]) -> List[Tuple[str, float, float]]:
    rows: List[Tuple[str, float, float]] = []
    for table, stats in data.items():
        avg = float(stats.get("avg_min", 0.0))
        p95 = float(stats.get("p95_min", 0.0))
        rows.append((table, avg, p95))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows


def _build_executive_summary(
    key_null: List[Tuple[str, str, float]],
    fk_rows: List[Tuple[str, str, float, float]],
    duplicates: List[Tuple[str, float]],
    lag_rows: List[Tuple[str, float, float]],
    spikes: List[Dict[str, object]],
) -> List[str]:
    bullets: List[str] = []

    high_key_null = [row for row in key_null if row[2] > HIGH_KEY_NULL]
    high_fk = [row for row in fk_rows if row[2] < HIGH_FK_SUCCESS]
    high_duplicates = [row for row in duplicates if row[1] > HIGH_DUPLICATE]
    high_lag = [row for row in lag_rows if row[2] > HIGH_LAG_P95]

    if high_key_null:
        table, col, rate = high_key_null[0]
        bullets.append(
            f"- Missing IDs: {table}.{col} has {rate*100:.1f}% nulls, preventing attribution and cohort KPIs."
        )
    if high_fk:
        table, dim, success, _ = high_fk[0]
        bullets.append(
            f"- Join failures: {table} → {dim} succeeds at {success*100:.1f}%, undercounting revenue by orphaning transactions."
        )
    if high_duplicates:
        table, rate = high_duplicates[0]
        bullets.append(
            f"- Duplicate rows: {table} shows {rate*100:.1f}% duplicates, inflating totals and KPIs."
        )
    if high_lag:
        table, _avg, p95 = high_lag[0]
        bullets.append(
            f"- Freshness risk: {table} p95 ingest lag is {p95:.1f} min, delaying real-time decisions."  # noqa: E741
        )
    if spikes:
        spike = spikes[0]
        bullets.append(
            f"- Null spike: {spike['table']}.{spike['column']} on {spike['ds']} (null {float(spike['null_rate'])*100:.1f}%, z={float(spike['zscore']):.2f}) needs pipeline triage."
        )

    if len(bullets) < 3:
        watch_items = []
        watch_key = [row for row in key_null if 0 < row[2] <= HIGH_KEY_NULL]
        watch_fk = [row for row in fk_rows if HIGH_FK_SUCCESS <= row[2] < 1.0]
        watch_lag = [row for row in lag_rows if WATCH_LAG_P95 < row[2] <= HIGH_LAG_P95]
        if watch_key:
            t, c, r = watch_key[0]
            watch_items.append(f"{t}.{c} nulls {r*100:.1f}%")
        if watch_fk:
            t, d, s, _ = watch_fk[0]
            watch_items.append(f"{t}→{d} joins {s*100:.1f}%")
        if watch_lag:
            t, _a, p = watch_lag[0]
            watch_items.append(f"{t} p95 lag {p:.1f} min")
        if watch_items:
            bullets.append("- Watch list: " + "; ".join(watch_items) + ".")

    if not bullets:
        bullets.append("- Overall health within thresholds; continue weekly monitoring and automated alerts.")

    # Ensure 3–5 bullets
    if len(bullets) < 3:
        bullets.append("- No additional high-risk metrics reported; maintain current monitoring cadence.")
    return bullets[:5]


def _section_completeness(rows: List[Tuple[str, str, float]]) -> List[str]:
    lines = ["### Completeness (Missing IDs)"]
    lines.append("- **What this means:** Missing IDs stop us from tying activity to customers and revenue.")
    if not rows:
        lines.append("- **Issues:** Not reported.")
        lines.append("")
        return lines

    issues = []
    for table, col, rate in rows[:5]:
        status = _status_key_null(rate)
        issues.append(
            f"  - {table}.{col}: {rate*100:.1f}% nulls ({status}) – Missing IDs prevent attribution and cohort KPIs."
        )
    lines.append("- **Issues:**")
    lines.extend(issues)
    lines.append("")
    return lines


def _section_join_integrity(rows: List[Tuple[str, str, float, float]]) -> List[str]:
    lines = ["### Join Integrity (FK Success / Orphans)"]
    lines.append("- **What this means:** Joins failing between facts and dimensions drop rows from dashboards.")
    if not rows:
        lines.append("- **Issues:** Not reported.")
        lines.append("")
        return lines

    lines.append("- **Issues:**")
    for table, dim, success, orphan in rows[:5]:
        status = _status_fk(success)
        lines.append(
            f"  - {table} → {dim}: success {success*100:.1f}% / orphan {orphan*100:.1f}% ({status}) – Low joins undercount revenue and engagement."
        )
    lines.append("")
    return lines


def _section_duplicates(rows: List[Tuple[str, float]]) -> List[str]:
    lines = ["### Duplicates"]
    lines.append("- **What this means:** Duplicate keys inflate totals and can double-count revenue or users.")
    if not rows:
        lines.append("- **Issues:** Not reported.")
        lines.append("")
        return lines

    lines.append("- **Issues:**")
    for table, rate in rows[:5]:
        status = _status_duplicate(rate)
        lines.append(
            f"  - {table}: {rate*100:.1f}% duplicates ({status}) – Duplicates skew performance metrics."  # noqa: E741
        )
    lines.append("")
    return lines


def _section_freshness(rows: List[Tuple[str, float, float]]) -> List[str]:
    lines = ["### Freshness (Ingest Lag)"]
    lines.append("- **What this means:** Ingest lag measures how long data takes to show up for analysis.")
    if not rows:
        lines.append("- **Issues:** Not reported.")
        lines.append("")
        return lines

    lines.append("- **Issues:**")
    for table, avg, p95 in rows[:5]:
        status = _status_lag(p95)
        lines.append(
            f"  - {table}: avg {avg:.1f} min / p95 {p95:.1f} min ({status}) – High lag delays operational decisions."
        )
    lines.append("")
    return lines


def _section_anomalies(spikes: List[Dict[str, object]]) -> List[str]:
    lines = ["### Anomalies (Null Spikes)"]
    lines.append("- **What this means:** Spikes in nulls highlight pipeline or upstream outages.")
    if not spikes:
        lines.append("- **Issues:** Not reported.")
        lines.append("")
        return lines

    lines.append("- **Issues:**")
    for spike in spikes[:10]:
        lines.append(
            "  - {table}.{column} on {ds}: null {null:.1f}% (z={z:.2f}) – Investigate source and backfill.".format(
                table=spike["table"],
                column=spike["column"],
                ds=spike["ds"],
                null=float(spike["null_rate"]) * 100,
                z=float(spike["zscore"]),
            )
        )
    lines.append("")
    return lines


def _recommended_actions(
    key_null: List[Tuple[str, str, float]],
    fk_rows: List[Tuple[str, str, float, float]],
    duplicates: List[Tuple[str, float]],
    lag_rows: List[Tuple[str, float, float]],
    spikes: List[Dict[str, object]],
) -> Tuple[List[str], List[str]]:
    near: List[str] = []
    medium: List[str] = []

    for table, col, rate in key_null:
        if rate > HIGH_KEY_NULL:
            near.append(
                f"- Backfill {table}.{col} (nulls {rate*100:.1f}%) and add ingestion validation to keep missing IDs ≤5%."
            )
        elif rate > 0:
            medium.append(
                f"- Monitor {table}.{col} nulls ({rate*100:.1f}%) and include in weekly data quality checks."
            )

    for table, dim, success, orphan in fk_rows:
        if success < HIGH_FK_SUCCESS:
            near.append(
                f"- Add FK validation for {table}.{dim.split('dim_')[-1]} (join {success*100:.1f}%, orphan {orphan*100:.1f}%); alert at orphan >2%."
            )
        elif success < 1.0:
            medium.append(
                f"- Tune {table}→{dim} load/lookup process to reach ≥99% join success."  # noqa: E741
            )

    for table, rate in duplicates:
        if rate > HIGH_DUPLICATE:
            near.append(f"- Deduplicate {table} ({rate*100:.1f}% duplicates) and add uniqueness checks in ETL.")
        elif rate > 0:
            medium.append(f"- Track {table} duplicate rate ({rate*100:.1f}%) through automated QA.")

    for table, _avg, p95 in lag_rows:
        if p95 > HIGH_LAG_P95:
            near.append(
                f"- Increase load frequency for {table} (p95 ingest lag {p95:.1f} min) to keep freshness ≤30 min."  # noqa: E741
            )
        elif p95 > WATCH_LAG_P95:
            medium.append(
                f"- Review {table} pipeline capacity (p95 lag {p95:.1f} min) and plan scaling within 45 days."
            )

    for spike in spikes:
        near.append(
            "- Investigate null spike {table}.{column} on {ds}; backfill affected day and add monitoring.".format(
                table=spike["table"], column=spike["column"], ds=spike["ds"]
            )
        )

    return (near, medium)


def _appendix_tables(
    key_null: List[Tuple[str, str, float]],
    fk_rows: List[Tuple[str, str, float, float]],
    duplicates: List[Tuple[str, float]],
    lag_rows: List[Tuple[str, float, float]],
    spikes: List[Dict[str, object]],
) -> List[str]:
    lines: List[str] = []

    lines.append("### Completeness")
    if key_null:
        lines.extend(_table_header(["Table", "Column", "Null Rate", "Status"]))
        for table, col, rate in key_null:
            lines.append(f"| {table} | {col} | {rate*100:.1f}% | {_status_key_null(rate)} |")
    else:
        lines.append("_Not reported_")
    lines.append("")

    lines.append("### Join Integrity")
    if fk_rows:
        lines.extend(_table_header(["Fact Table", "Dimension", "FK Success", "Orphan Rate", "Status"]))
        for table, dim, success, orphan in fk_rows:
            lines.append(
                f"| {table} | {dim} | {success*100:.1f}% | {orphan*100:.1f}% | {_status_fk(success)} |"
            )
    else:
        lines.append("_Not reported_")
    lines.append("")

    lines.append("### Duplicates")
    if duplicates:
        lines.extend(_table_header(["Table", "Duplicate Rate", "Status"]))
        for table, rate in duplicates:
            lines.append(f"| {table} | {rate*100:.1f}% | {_status_duplicate(rate)} |")
    else:
        lines.append("_Not reported_")
    lines.append("")

    lines.append("### Freshness")
    if lag_rows:
        lines.extend(_table_header(["Table", "Avg Lag (min)", "p95 Lag (min)", "Status"]))
        for table, avg, p95 in lag_rows:
            lines.append(f"| {table} | {avg:.1f} | {p95:.1f} | {_status_lag(p95)} |")
    else:
        lines.append("_Not reported_")
    lines.append("")

    lines.append("### Null Spikes")
    if spikes:
        lines.extend(_table_header(["Table", "Column", "Date", "Null Rate", "Z-score", "Status"]))
        for spike in spikes:
            lines.append(
                "| {table} | {column} | {ds} | {null:.1f}% | {z:.2f} | High |".format(
                    table=spike["table"],
                    column=spike["column"],
                    ds=spike["ds"],
                    null=float(spike["null_rate"]) * 100,
                    z=float(spike["zscore"]),
                )
            )
    else:
        lines.append("_Not reported_")
    lines.append("")

    return lines


def _table_header(columns: Iterable[str]) -> List[str]:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(tuple(columns))) + " |"
    cols = list(columns)
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    return [header, separator]


def _status_key_null(rate: float) -> str:
    if rate > HIGH_KEY_NULL:
        return "High"
    if rate > 0:
        return "Watch"
    return "OK"


def _status_fk(success: float) -> str:
    if success < HIGH_FK_SUCCESS:
        return "High"
    if success < 1.0:
        return "Watch"
    return "OK"


def _status_duplicate(rate: float) -> str:
    if rate > HIGH_DUPLICATE:
        return "High"
    if rate > 0:
        return "Watch"
    return "OK"


def _status_lag(p95: float) -> str:
    if p95 > HIGH_LAG_P95:
        return "High"
    if p95 > WATCH_LAG_P95:
        return "Watch"
    return "OK"
