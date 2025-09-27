from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from jsonschema import Draft202012Validator, FormatChecker


def _load_schema(name: str) -> Dict:
    # Try local schemas/ first; fallback to embedded
    local = Path.cwd() / "schemas" / name
    if local.exists():
        return json.loads(local.read_text())
    from importlib import resources

    with resources.files("datagap_synth.schemas").joinpath(name).open("rb") as f:
        return json.loads(f.read())


def _load_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def validate_pack(pack_dir: Path) -> Tuple[bool, str]:
    errors: List[str] = []
    # Load files
    nlq_path = pack_dir / "nl_queries.jsonl"
    slack_path = pack_dir / "slack.jsonl"
    email_path = pack_dir / "email.jsonl"
    datasets_path = pack_dir / "catalog" / "datasets.json"
    freshness_path = pack_dir / "catalog" / "freshness.csv"

    if not nlq_path.exists():
        errors.append("Missing nl_queries.jsonl")
    if not slack_path.exists():
        errors.append("Missing slack.jsonl")
    if not email_path.exists():
        errors.append("Missing email.jsonl")
    if not datasets_path.exists():
        errors.append("Missing catalog/datasets.json")
    if not freshness_path.exists():
        errors.append("Missing catalog/freshness.csv")
    if errors:
        return False, "; ".join(errors)

    nlq_schema = _load_schema("nl_query.schema.json")
    slack_schema = _load_schema("slack.schema.json")
    email_schema = _load_schema("email.schema.json")
    v_nlq = Draft202012Validator(nlq_schema, format_checker=FormatChecker())
    v_slack = Draft202012Validator(slack_schema, format_checker=FormatChecker())
    v_email = Draft202012Validator(email_schema, format_checker=FormatChecker())

    nlq = _load_jsonl(nlq_path)
    slack = _load_jsonl(slack_path)
    email = _load_jsonl(email_path)

    # Schema validation
    for i, r in enumerate(nlq, 1):
        for err in sorted(v_nlq.iter_errors(r), key=str):
            errors.append(f"nl_queries line {i}: {err.message}")
    for i, r in enumerate(slack, 1):
        for err in sorted(v_slack.iter_errors(r), key=str):
            errors.append(f"slack line {i}: {err.message}")
    for i, r in enumerate(email, 1):
        for err in sorted(v_email.iter_errors(r), key=str):
            errors.append(f"email line {i}: {err.message}")

    # Cross-file checks
    nlq_by_id: Dict[str, Dict] = {r["id"]: r for r in nlq}
    # referenced_query_id exists
    for i, r in enumerate(slack, 1):
        ref = r.get("referenced_query_id")
        if ref and ref not in nlq_by_id:
            errors.append(f"slack line {i}: referenced_query_id {ref} not found in nl_queries")
    for i, r in enumerate(email, 1):
        ref = r.get("referenced_query_id")
        if ref and ref not in nlq_by_id:
            errors.append(f"email line {i}: referenced_query_id {ref} not found in nl_queries")

    # answered NLQs tables exist in catalog
    datasets = json.loads(datasets_path.read_text())
    dataset_names = {d["name"] for d in datasets}
    for r in nlq:
        if r.get("outcome") == "answered":
            for t in r.get("tables", []):
                if t not in dataset_names:
                    errors.append(f"nlq {r['id']}: table {t} not in catalog/datasets.json")

    # freshness CSV rows match datasets
    with freshness_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fresh_rows = list(reader)
    fresh_names = {row["dataset"] for row in fresh_rows}
    if len(fresh_rows) != len(datasets):
        errors.append(
            f"freshness.csv rows ({len(fresh_rows)}) != datasets.json count ({len(datasets)})"
        )
    missing = dataset_names - fresh_names
    if missing:
        errors.append(f"freshness.csv missing datasets: {sorted(missing)[:5]}{'...' if len(missing)>5 else ''}")

    # Additional checks: language distribution and gap triggers
    # language coverage
    langs = [r.get("lang", "en") for r in nlq]
    non_en = sum(1 for x in langs if x != "en")
    if nlq:
        pct_non_en = 100.0 * non_en / len(nlq)
        if pct_non_en < 5.0:  # expect small multilingual presence
            errors.append(f"multilingual coverage low: {pct_non_en:.1f}% < 5%")
    # gap triggers presence (ensure missing asset/column present somewhere)
    gap_counts: Dict[str, int] = {}
    for r in nlq:
        for g in r.get("gap_types", []) or []:
            gap_counts[g] = gap_counts.get(g, 0) + 1
    missing_ref = gap_counts.get("missing_column", 0) + gap_counts.get("missing_asset", 0)
    if nlq and missing_ref < max(1, int(0.1 * len(nlq))):
        errors.append("insufficient missing_column/missing_asset coverage (<10%)")

    ok = not errors
    summary = (
        f"OK: {len(nlq)} nlq, {len(slack)} slack, {len(email)} email, {len(datasets)} datasets"
        if ok
        else "Validation failed:\n- " + "\n- ".join(errors)
    )
    return ok, summary
