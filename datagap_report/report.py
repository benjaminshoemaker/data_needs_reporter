from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from jsonschema import Draft202012Validator

from .llm import LLMClient, LLMClientError
from .schemas import INTENT_SCHEMA, GAPTYPES_SCHEMA, BACKLOG_SUMMARY_SCHEMA


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _load_pack(pack_dir: Path) -> Dict[str, Any]:
    m = json.loads((pack_dir / "manifest.json").read_text())
    nlq = _read_jsonl(pack_dir / "nl_queries.jsonl")
    slack = _read_jsonl(pack_dir / "slack.jsonl")
    email = _read_jsonl(pack_dir / "email.jsonl")
    datasets = json.loads((pack_dir / "catalog" / "datasets.json").read_text())
    fresh_rows: List[Dict[str, str]]
    with (pack_dir / "catalog" / "freshness.csv").open("r", newline="", encoding="utf-8") as f:
        fresh_rows = list(csv.DictReader(f))
    return {
        "manifest": m,
        "nlq": nlq,
        "slack": slack,
        "email": email,
        "datasets": datasets,
        "freshness": fresh_rows,
    }


def _utc_parse(s: str) -> datetime:
    # handle Z or +00:00
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def _redact(s: str) -> str:
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", "[redacted_email]", s)
    s = re.sub(r"@([A-Za-z0-9_\-\.]+)", "@[redacted]", s)
    s = re.sub(r"\bu_[A-Za-z0-9_]+\b", "u_[redacted]", s)
    return s


def _catalog_tokens(datasets: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    tables = [d["name"] for d in datasets]
    cols = []
    for d in datasets:
        cols.extend(c["name"] for c in d.get("columns", []))
    # dedup preserve order
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]
    return {"tables": tables, "cols": cols}


def _normalize_events(pack: Dict[str, Any], horizon_days: int = 7) -> List[Dict[str, Any]]:
    now = _utc_parse(pack["manifest"]["generated_at"]) if "generated_at" in pack["manifest"] else datetime.now(UTC)
    start = now - timedelta(days=horizon_days)
    rows: List[Dict[str, Any]] = []
    # NLQ
    for r in pack["nlq"]:
        t = _utc_parse(r["when"]) if "when" in r else now
        if t < start or t > now:
            continue
        rows.append(
            {
                "id": r["id"],
                "when": t,
                "channel": "nlq",
                "actor": r.get("actor"),
                "text": _redact(r.get("nl_text", "")),
                "parsed_sql": r.get("parsed_sql"),
                "outcome": r.get("outcome"),
                "gap_types": r.get("gap_types", []),
                "tables": r.get("tables", []),
            }
        )
    # Slack
    for r in pack["slack"]:
        t = _utc_parse(r["when"]) if "when" in r else now
        if t < start or t > now:
            continue
        rows.append(
            {
                "id": r["id"],
                "when": t,
                "channel": "slack",
                "actor": r.get("actor"),
                "text": _redact(r.get("text", "")),
                "parsed_sql": None,
                "outcome": None,
                "gap_types": [],
                "tables": [],
            }
        )
    # Email
    for r in pack["email"]:
        t = _utc_parse(r["when"]) if "when" in r else now
        if t < start or t > now:
            continue
        subject = r.get("subject", "")
        body = r.get("body", "")
        rows.append(
            {
                "id": r["id"],
                "when": t,
                "channel": "email",
                "actor": r.get("actor"),
                "text": _redact((subject + ": " + body).strip()),
                "parsed_sql": None,
                "outcome": None,
                "gap_types": [],
                "tables": [],
            }
        )
    rows.sort(key=lambda r: (r["when"], r["id"]))
    return rows


def _rule_intent(text: str, parsed_sql: Optional[str], tokens: Dict[str, List[str]]) -> Dict[str, Any]:
    metric: List[str] = []
    dimensions: List[str] = []
    filters: List[str] = []
    timeframe: Optional[str] = None
    tables_ref: List[str] = []
    cols_ref: List[str] = []

    hay = (parsed_sql or "") + "\n" + text
    hay_l = hay.lower()
    # timeframe simple patterns
    for tf in ["7 days", "30 days", "quarter", "year", "week", "month"]:
        if tf in hay_l:
            timeframe = tf
            break
    # tables
    for t in tokens["tables"]:
        if t in hay:
            tables_ref.append(t)
    # columns
    for c in tokens["cols"]:
        if c in hay:
            cols_ref.append(c)
    # heuristics: numeric-like columns are metrics (endswith s or known names)
    for c in cols_ref:
        if re.search(r"revenue|orders|units|visits|customers|amount|count|qty", c, re.I):
            if c not in metric:
                metric.append(c)
        else:
            if c not in dimensions:
                dimensions.append(c)
    # crude filters: words after 'where' or 'for '
    m = re.search(r"where\s+([^\n]+)", hay_l)
    if m:
        filters.append(m.group(1)[:80])
    m = re.search(r"for\s+([a-z0-9_\s]+)\b", hay_l)
    if m:
        f = m.group(1).strip()
        if f and f not in filters:
            filters.append(f[:80])
    return {
        "metric": metric,
        "dimensions": dimensions,
        "filters": filters,
        "timeframe": timeframe,
        "tables_ref": tables_ref,
        "cols_ref": cols_ref,
    }


def _post_filter_vocab(intent: Dict[str, Any], tokens: Dict[str, List[str]]) -> Dict[str, Any]:
    intent["tables_ref"] = [t for t in intent.get("tables_ref", []) if t in tokens["tables"]]
    intent["cols_ref"] = [c for c in intent.get("cols_ref", []) if c in tokens["cols"]]
    # metric/dimensions limited to known cols
    intent["metric"] = [c for c in intent.get("metric", []) if c in tokens["cols"]]
    intent["dimensions"] = [c for c in intent.get("dimensions", []) if c in tokens["cols"]]
    # filters left as given text snippets
    return intent


def _map_gap_types(raw: List[str], outcome: Optional[str]) -> List[str]:
    out: set[str] = set()
    for g in raw:
        if g in {"missing_asset", "missing_column"}:
            out.add("missing_dataset_or_column")
        elif g in {"type_mismatch", "grain_mismatch"}:
            out.add("grain_or_type_mismatch")
        elif g == "freshness_breach":
            out.add("freshness_breach")
        elif g == "joinability_issue":
            out.add("joinability_not_defined")
        elif g == "access_denied":
            out.add("access_denied")
        elif g == "docs_missing":
            out.add("semantics_unclear")
        elif g == "performance_limit":
            out.add("performance_limit")
    if outcome == "slow":
        out.add("performance_limit")
    if outcome == "uncertain":
        out.add("semantics_unclear")
    return sorted(out)


def _cosine(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def generate_report(
    pack_dir: Path,
    out_dir: Path,
    owners_yaml: Optional[Path],
    p95_ms: int,
    llm_on: bool,
    model: str,
    embed_model: str,
    api_base: str,
    api_key: Optional[str],
    llm_limit: int = 0,
) -> Dict[str, Any]:
    log = logging.getLogger("datagap_report")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "artifacts" / "llm_cache"
    client = LLMClient(api_base=api_base, api_key=api_key, cache_dir=cache_dir)

    pack = _load_pack(pack_dir)
    log.info(
        "Loaded pack: nlq=%d slack=%d email=%d datasets=%d",
        len(pack["nlq"]),
        len(pack["slack"]),
        len(pack["email"]),
        len(pack["datasets"]),
    )
    tokens = _catalog_tokens(pack["datasets"])
    events = _normalize_events(pack)
    log.info("Normalized events: %d (last 7 days)", len(events))

    owners: Dict[str, Any] = {}
    if owners_yaml and owners_yaml.exists():
        owners = yaml.safe_load(owners_yaml.read_text()) or {}

    # Intent extraction
    intents: Dict[str, Dict[str, Any]] = {}
    v_intent = Draft202012Validator(INTENT_SCHEMA["schema"])  # local validation guardrail
    # First pass: rules only, decide which need LLM
    need_ids: List[str] = []
    rules_only: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        rid = ev["id"]
        it = _rule_intent(ev["text"], ev.get("parsed_sql"), tokens)
        rules_only[rid] = _post_filter_vocab(it, tokens)
        if llm_on and (len(it["metric"]) == 0 or len(it["dimensions"]) == 0 or len(it["tables_ref"]) == 0):
            need_ids.append(rid)
    log.info("Intent: rules completed; %d need LLM out of %d", len(need_ids), len(events))
    # Second pass: call LLM for those that need it
    if llm_limit and llm_limit > 0 and len(need_ids) > llm_limit:
        log.info("Intent: applying LLM cap to first %d of %d", llm_limit, len(need_ids))
        need_ids = need_ids[:llm_limit]
    llm_disabled = False
    for idx, rid in enumerate(need_ids, start=1):
        ev = next(e for e in events if e["id"] == rid)
        if llm_on and not llm_disabled:
            system = (
                "Extract analytics intent from short text or SQL. "
                "Use only tokens present in the input or in the provided catalog token list. "
                "Do not invent fields. Return JSON that matches the schema exactly."
            )
            user = {
                "text": ev["text"],
                "parsed_sql": ev.get("parsed_sql"),
                "catalog_tokens": tokens,
            }
            if idx % 25 == 1 or idx == len(need_ids):
                log.info("Intent LLM progress: %d/%d", idx, len(need_ids))
            try:
                llm_out = client.structured_json(
                    model=model,
                    system=system,
                    user=user,
                    response_format=INTENT_SCHEMA,
                    temperature=0.0,
                    max_output_tokens=512,
                )
                # Validate + post-filter
                v_intent.validate(llm_out)
                intents[rid] = _post_filter_vocab(llm_out, tokens)
            except LLMClientError as e:
                log.error("Intent LLM client error; disabling LLM for this run: %s", e)
                llm_disabled = True
                intents[rid] = rules_only[rid]
            except Exception:
                # keep rules-only result on failure
                intents[rid] = rules_only[rid]
    # For items that didn't need LLM, fill from rules
    for rid, it in rules_only.items():
        if rid not in intents:
            intents[rid] = it
    log.info("Intent extraction complete")

    # Gap detection
    gaps: Dict[str, List[str]] = {}
    v_gap = Draft202012Validator(GAPTYPES_SCHEMA["schema"])  # guardrail
    # Compute which need gap LLM
    need_gap_ids: List[str] = []
    for ev in events:
        rid = ev["id"]
        rule = _map_gap_types(ev.get("gap_types", []), ev.get("outcome"))
        if not rule and llm_on:
            need_gap_ids.append(rid)
        gaps[rid] = rule
    log.info("Gap types: rules completed; %d need LLM out of %d", len(need_gap_ids), len(events))
    if llm_limit and llm_limit > 0 and len(need_gap_ids) > llm_limit:
        log.info("Gap: applying LLM cap to first %d of %d", llm_limit, len(need_gap_ids))
        need_gap_ids = need_gap_ids[:llm_limit]
    for idx, rid in enumerate(need_gap_ids, start=1):
        ev = next(e for e in events if e["id"] == rid)
        if llm_on and not llm_disabled:
            system = (
                "Classify applicable gap types from the allowed set based only on the provided "
                "intent, text, outcome, errors, and catalog snapshot."
            )
            user = {
                "intent": intents[rid],
                "text": ev["text"],
                "outcome": ev.get("outcome"),
                "errors": None,
                "catalog_view": {"tables": tokens["tables"]},
            }
            if idx % 25 == 1 or idx == len(need_gap_ids):
                log.info("Gap LLM progress: %d/%d", idx, len(need_gap_ids))
            try:
                llm_out = client.structured_json(
                    model=model,
                    system=system,
                    user=user,
                    response_format=GAPTYPES_SCHEMA,
                    temperature=0.0,
                    max_output_tokens=256,
                )
                v_gap.validate(llm_out)
                rule = list(dict.fromkeys(llm_out.get("gap_types", [])))
            except LLMClientError as e:
                log.error("Gap LLM client error; disabling LLM for this run: %s", e)
                llm_disabled = True
                rule = gaps.get(rid, [])
            except Exception:
                rule = gaps.get(rid, [])
        gaps[rid] = rule
    log.info("Gap detection complete")

    # Clustering: key-based
    clusters: Dict[str, Dict[str, Any]] = {}
    by_key: Dict[str, List[str]] = defaultdict(list)
    for ev in events:
        rid = ev["id"]
        it = intents[rid]
        key = json.dumps(
            {
                "metric": sorted(it.get("metric", [])),
                "dimensions": sorted(it.get("dimensions", [])),
                "timeframe": it.get("timeframe"),
                "tables": sorted(it.get("tables_ref", [])),
            },
            sort_keys=True,
        )
        by_key[key].append(rid)

    # Optional embedding merge
    merged_groups: List[List[str]] = [list(v) for v in by_key.values()]
    if llm_on and embed_model:
        texts = [next((e["text"] for e in events if e["id"] == rid), "") for group in merged_groups for rid in group]
        try:
            embs = client.embeddings(embed_model, texts)
            log.info("Embeddings computed: vectors=%d", len(embs))
            # Compute centroid per group and merge groups with cosine >= 0.85
            group_vecs: List[List[float]] = []
            idx = 0
            for group in merged_groups:
                vecs = embs[idx : idx + len(group)]
                idx += len(group)
                centroid = [sum(col) / len(vecs) for col in zip(*vecs)] if vecs else []
                group_vecs.append(centroid)
            keep: List[List[str]] = []
            used = [False] * len(merged_groups)
            for i in range(len(merged_groups)):
                if used[i]:
                    continue
                bucket = list(merged_groups[i])
                used[i] = True
                for j in range(i + 1, len(merged_groups)):
                    if used[j]:
                        continue
                    if _cosine(group_vecs[i], group_vecs[j]) >= 0.85:
                        bucket.extend(merged_groups[j])
                        used[j] = True
                keep.append(bucket)
            merged_groups = keep
        except Exception as e:
            log.warning(f"Embeddings failed; skipping merge: {e}")

    # Prepare cluster objects
    for idx, group in enumerate(merged_groups, start=1):
        gid = f"c_{idx:04d}"
        ex_ids = group[: min(5, len(group))]
        ex_texts = [next(e["text"] for e in events if e["id"] == rid) for rid in ex_ids]
        all_tables = sorted(
            list(
                {t for rid in group for t in intents[rid].get("tables_ref", [])}
            )
        )
        all_gaps = sorted(list({g for rid in group for g in gaps[rid]}))
        clusters[gid] = {
            "ids": group,
            "examples": ex_texts,
            "gap_types": all_gaps,
            "datasets": all_tables,
        }

    # Titles/summaries via LLM
    summaries: Dict[str, Dict[str, str]] = {}
    v_summary = Draft202012Validator(BACKLOG_SUMMARY_SCHEMA["schema"])
    for gid, cl in clusters.items():
        title = None
        why = None
        if llm_on:
            system = (
                "Write a concise backlog item title and one-sentence rationale. "
                "Use only supplied facts."
            )
            user = {
                "cluster_examples": cl["examples"],
                "gap_types": cl["gap_types"],
                "datasets": cl["datasets"],
                "severity": "M",
                "effort": "M",
            }
            try:
                out = client.structured_json(
                    model=model,
                    system=system,
                    user=user,
                    response_format=BACKLOG_SUMMARY_SCHEMA,
                    temperature=0.0,
                    max_output_tokens=200,
                )
                v_summary.validate(out)
                title = out["title"]
                why = out["why"]
            except Exception as e:
                log.warning(f"Backlog summary LLM failed for {gid}: {e}")
        if not title:
            gap = ", ".join(cl["gap_types"]) or "data gap"
            ds = ", ".join(cl["datasets"]) or "unspecified dataset"
            title = f"Address {gap} for {ds}"[:90]
        if not why:
            why = "Clustered user questions indicate a recurring data gap to address."
        summaries[gid] = {"title": title, "why": why}

    # Write outputs
    artifacts = out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    # intents
    with (artifacts / "intents.jsonl").open("w", encoding="utf-8") as f:
        for ev in events:
            rid = ev["id"]
            f.write(json.dumps({"id": rid, **intents[rid]}, sort_keys=True) + "\n")
    # clusters
    (artifacts / "clusters.json").write_text(json.dumps(clusters, indent=2, sort_keys=True))

    # backlog.csv
    bl_path = out_dir / "backlog.csv"
    with bl_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "title", "why", "gap_types", "datasets", "severity", "effort"])
        for gid, cl in clusters.items():
            row = [
                gid,
                summaries[gid]["title"],
                summaries[gid]["why"],
                ";".join(cl["gap_types"]),
                ";".join(cl["datasets"]),
                "M",
                "M",
            ]
            w.writerow(row)

    # freshness_table.csv (copy from pack, limited to a week already by source)
    fresh_out = out_dir / "freshness_table.csv"
    with fresh_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "max_loaded_at", "sla_hours"])
        w.writeheader()
        for row in pack["freshness"]:
            w.writerow({k: row[k] for k in ["dataset", "max_loaded_at", "sla_hours"]})

    # hotspots.dot
    dot = ["digraph hotspots {", "rankdir=LR;"]
    # rank datasets by frequency across events (via intents tables_ref)
    freq: Dict[str, int] = defaultdict(int)
    for it in intents.values():
        for t in it.get("tables_ref", []):
            freq[t] += 1
    for t, n in sorted(freq.items(), key=lambda x: -x[1])[:50]:
        dot.append(f'  "{t}" [label="{t}\n{n} refs"];')
    dot.append("}")
    (out_dir / "hotspots.dot").write_text("\n".join(dot))

    # report.md and report.html
    report_md = out_dir / "report.md"
    report_md.write_text(
        "\n".join(
            [
                "# Data Gaps Report (1 week)",
                "",
                f"Clusters: {len(clusters)}",
                "",
                "## Top Backlog Items",
                "",
            ]
            + [f"- [{gid}] {summaries[gid]['title']}" for gid in sorted(clusters.keys())[:50]]
        )
    )
    report_html = out_dir / "report.html"
    report_html.write_text(
        """
<!doctype html>
<meta charset="utf-8"/>
<title>Data Gaps Report</title>
<style>body{font-family:system-ui,Arial;margin:2rem;max-width:900px}li{margin:.25rem 0}</style>
<h1>Data Gaps Report (1 week)</h1>
<p>See backlog.csv for details.</p>
<h2>Top Backlog Items</h2>
<ul>
"""
        + "\n".join(
            [
                f"<li><code>{gid}</code> {summaries[gid]['title']}</li>"
                for gid in sorted(clusters.keys())[:50]
            ]
        )
        + "\n</ul>\n"
    )

    log.info("Report done: clusters=%d", len(clusters))
    return {
        "clusters": len(clusters),
        "backlog_items": len(clusters),
        "outputs": [
            str(bl_path),
            str(fresh_out),
            str(out_dir / "hotspots.dot"),
            str(report_md),
            str(report_html),
        ],
    }


def validate_outputs(out_dir: Path) -> tuple[bool, str]:
    missing = []
    for name in ["backlog.csv", "freshness_table.csv", "hotspots.dot", "report.md", "report.html"]:
        if not (out_dir / name).exists():
            missing.append(name)
    if missing:
        return False, f"Missing outputs: {', '.join(missing)}"
    # quick CSV read
    try:
        with (out_dir / "backlog.csv").open("r", encoding="utf-8") as f:
            next(csv.reader(f))
    except Exception as e:
        return False, f"backlog.csv not readable: {e}"
    return True, "OK"


def print_sample(out_dir: Path) -> None:
    path = out_dir / "backlog.csv"
    if not path.exists():
        print("No backlog.csv found")
        return
    with path.open("r", encoding="utf-8") as f:
        r = list(csv.reader(f))
    header, rows = r[0], r[1:6]
    for row in rows:
        print(" | ".join(row[:3]))
