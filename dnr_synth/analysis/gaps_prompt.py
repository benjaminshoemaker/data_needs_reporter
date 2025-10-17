from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROMPT_HEADER = """You are an expert data quality analyst and product analytics PM.
Task: read the provided {DOMAIN} domain artifacts and produce a concise, executive-ready report of the highest-priority data gaps that block analysis and decision making.

Artifacts (attached or pasted below):
1) health_profile.json and/or health_profile.md  ← data health metrics
2) Partitioned Parquet datasets under {DATA_PATH}  ← fact + dim tables
3) slack_threads.json  ← multi-role threads; messages cite table.column
4) nl_queries.json  ← natural-language questions with "references": ["table.col", ...]

Objectives
- Identify and rank data gaps that most hinder analytics and KPI reporting.
- Quantify impact with evidence from metrics, queries, and conversations.
- Recommend fixes with owners, effort, and time to value.

Method
1) Parse health_profile.*. Extract by table/column: null_rate, fk_join_success, duplicate_rate, schema_drift, ingest_lag_p50/p95, null_spikes_by_day.
2) Scan nl_queries.json. Map each query’s “references” to required fields and grains. Detect unmet needs: missing columns, missing dimensions, insufficient granularity, inadequate history, timeliness gaps, semantic ambiguity.
3) Scan slack_threads.json. Pull pain points, blocked analyses, and workarounds. Count mentions by table.column. Keep message ids (thread_id, ts) as evidence.
4) Corroborate across sources. For each suspected gap, link: {{metric evidence}} + {{queries affected}} + {{threads complaining}}.
5) Prioritize with a score 0–100:
   Priority = 35*Impact_on_key_decisions + 25*Blocker_Severity + 20*Frequency_across_queries/threads
              + 10*Revenue_or_Risk_Exposure + 10*Solvability_short_term
   Notes: normalize each 0–1. If Time_to_fix > 30 days, multiply score by 0.9. If privacy/compliance risk present, add 5.
6) Propose fixes: data contract or schema change, source ownership, pipeline work, backfill plan, monitoring/SLA. Include interim workaround if fix >30 days.

Gap taxonomy (use codes)
- COMPL: completeness/missingness
- TIMEL: freshness/ingest lag
- JOIN: fk integrity/joinability
- CONSIS: type/range/semantic conflicts
- SCHEMA: drift/rename/add/drop
- DEDUP: duplicates/keys
- SEMANTIC: unclear definitions or grain
- ACCESS: permissions/PII constraints

Output
Return TWO artifacts:

A) Markdown report for executives
## Executive Summary
- Top 3 gaps and why they matter in 3 bullets.
- Expected business impact in 30/60/90 days.

## Priority Table
| Rank | Gap (code) | Impacted KPIs / decisions | Evidence (metrics, queries, threads) | Priority | ETA | Owner | Fix summary | Monitoring/SLA |
|---|---|---|---|---:|---|---|---|---|

## Deep Dives (Top 5)
For each: Problem, Evidence (quote ids + metrics), Root-cause hypothesis, Fix plan (steps, owner, effort S/M/L, dependencies), Risks, Interim workaround, Acceptance criteria.

## 30-60-90 Plan
30: quick wins (<2 weeks) • 60: core fixes • 90: hardening + SLAs.

## Appendix
- Evidence Map: gap_id → [query_ids], [thread_ids(ts)], [table.column]
- Metric Snapshots: null rates, fk success, p95 lag by day
- Glossary of terms and grains

B) JSON sidecar for backlog tooling (name: data_gaps_report.json)
"""


def _safe_read(path: Path, max_chars: int) -> str:
    try:
        txt = path.read_text(encoding="utf-8")
        return txt[:max_chars]
    except Exception:
        return ""


def _summarize_datasets(data_dir: Path, per_table_col_limit: int = 16, max_tables: int = 40) -> str:
    try:
        import pyarrow.dataset as ds
    except Exception:
        ds = None  # type: ignore
    lines: List[str] = []
    tables = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    for table_dir in tables[:max_tables]:
        cols: List[str] = []
        if ds is not None:
            try:
                dataset = ds.dataset(str(table_dir), format="parquet", partitioning="hive")
                schema = dataset.schema
                cols = [f.name for f in schema][:per_table_col_limit]
            except Exception:
                pass
        if not cols:
            # fallback: peek first Parquet file via pandas
            try:
                first = next(table_dir.glob("*.parquet"), None)
                if first:
                    df = pd.read_parquet(first)
                    cols = df.columns.tolist()[:per_table_col_limit]
            except Exception:
                cols = []
        col_str = ", ".join(cols)
        lines.append(f"- {table_dir.name}: {col_str}")
    return "\n".join(lines)


def _summarize_nl_queries(nl_path: Path, limit: int = 100) -> str:
    if not nl_path.exists():
        return ""
    try:
        items = json.loads(nl_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    lines: List[str] = []
    for idx, it in enumerate(items[:limit]):
        refs = ", ".join(it.get("references", [])[:6])
        q = str(it.get("query", "")).replace("\n", " ")
        lines.append(f"- Q{idx}: {q} | refs: [{refs}]")
    return "\n".join(lines)


def _summarize_slack(slack_path: Path, limit_threads: int = 10, limit_msgs: int = 10) -> str:
    if not slack_path.exists():
        return ""
    try:
        threads = json.loads(slack_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    out: List[str] = []
    for th in threads[:limit_threads]:
        tid = th.get("thread_id", "")
        ch = th.get("channel", "")
        out.append(f"- Thread {tid} {ch}")
        for msg in th.get("messages", [])[:limit_msgs]:
            user = msg.get("user", "u")
            ts = msg.get("ts", "")
            text = str(msg.get("text", "")).replace("\n", " ")
            out.append(f"  [{user} {ts}] {text}")
    return "\n".join(out)


def build_prompt(domain: str, data_dir: Path, artifacts_dir: Path, max_attach_chars: int = 8000) -> str:
    header = PROMPT_HEADER.replace("{DOMAIN}", domain.upper()).replace("{DATA_PATH}", str(data_dir))

    # Attach artifacts (summaries and bounded raw content)
    health_json = artifacts_dir / "health_profile.json"
    health_md = artifacts_dir / "health_profile.md"
    nl_path = artifacts_dir / "nl_queries.json"
    slack_path = artifacts_dir / "slack_threads.json"

    sections: List[str] = [header]
    # Datasets summary
    sections.append("\n=== DATASET SCHEMA (summary) ===\n" + _summarize_datasets(data_dir))
    # NL Queries summary
    sections.append("\n=== NL QUERIES (sample) ===\n" + _summarize_nl_queries(nl_path))
    # Slack summary
    sections.append("\n=== SLACK THREADS (sample) ===\n" + _summarize_slack(slack_path))
    # Health profiles raw (bounded)
    if health_json.exists():
        sections.append("\n=== HEALTH PROFILE JSON (truncated) ===\n" + _safe_read(health_json, max_attach_chars))
    elif health_md.exists():
        sections.append("\n=== HEALTH PROFILE MD (truncated) ===\n" + _safe_read(health_md, max_attach_chars))
    else:
        sections.append("\n=== HEALTH PROFILE ===\n<not found>")

    # Return combined prompt
    return "\n\n".join(filter(None, sections))


def run_llm(
    prompt: str,
    model: str,
    api_base: str,
    api_key: str,
    timeout_s: float = 60.0,
) -> str:
    """Optional: call OpenAI Responses API. Returns raw text output.
    Import httpx lazily to avoid mandatory dependency at import time.
    """
    try:
        import httpx  # type: ignore
    except Exception as e:
        raise RuntimeError("httpx not installed; cannot run LLM. Install httpx or use --emit-prompt.") from e
    headers = {"Authorization": f"Bearer {api_key}"}
    payload: Dict[str, Any] = {"model": model, "input": prompt}
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(api_base.rstrip("/") + "/v1/responses", headers=headers, json=payload)
        resp.raise_for_status()
    data = resp.json()
    # Try common fields (Responses API variants)
    if isinstance(data, dict):
        if "output_text" in data and isinstance(data["output_text"], str):
            return data["output_text"]
        if isinstance(data.get("content"), list):
            for ch in data["content"]:
                if ch.get("type") == "output_text" and isinstance(ch.get("text"), str):
                    return ch.get("text", "")
        # Some SDKs wrap under data["response"]["output_text"]
        if isinstance(data.get("response"), dict):
            inner = data["response"]
            if isinstance(inner.get("output_text"), str):
                return inner["output_text"]
            if isinstance(inner.get("content"), list):
                for ch in inner["content"]:
                    if ch.get("type") == "output_text" and isinstance(ch.get("text"), str):
                        return ch.get("text", "")
    # Some endpoints return a list of messages
    if isinstance(data, list):
        for msg in data:
            if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                for ch in msg["content"]:
                    if ch.get("type") == "output_text" and isinstance(ch.get("text"), str):
                        return ch.get("text", "")
                    # Sometimes key is just 'text'
                    if isinstance(ch.get("text"), str):
                        return ch.get("text", "")
    # Fallback: return text if present anywhere
    try:
        s = json.dumps(data)
        return s
    except Exception:
        return str(data)


def split_outputs(raw: str) -> tuple[str, List[Dict[str, Any]]]:
    """Heuristically split combined output into markdown and JSON array.
    Returns (markdown, json_array). If not found, JSON list is empty.
    """
    # If raw looks like JSON (from Responses), try to unwrap to text first
    try:
        if raw.lstrip().startswith("{") or raw.lstrip().startswith("["):
            obj = json.loads(raw)
            # try to extract text content similar to run_llm
            text_blob: str | None = None
            if isinstance(obj, dict):
                if isinstance(obj.get("output_text"), str):
                    text_blob = obj["output_text"]
                elif isinstance(obj.get("content"), list):
                    for ch in obj["content"]:
                        if ch.get("type") == "output_text" and isinstance(ch.get("text"), str):
                            text_blob = ch["text"]; break
                elif isinstance(obj.get("response"), dict):
                    inner = obj["response"]
                    if isinstance(inner.get("output_text"), str):
                        text_blob = inner["output_text"]
                    elif isinstance(inner.get("content"), list):
                        for ch in inner["content"]:
                            if ch.get("type") == "output_text" and isinstance(ch.get("text"), str):
                                text_blob = ch["text"]; break
            elif isinstance(obj, list):
                for msg in obj:
                    if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                        for ch in msg["content"]:
                            if isinstance(ch.get("text"), str):
                                text_blob = ch["text"]; break
                    if text_blob:
                        break
            if isinstance(text_blob, str) and text_blob:
                raw = text_blob
    except Exception:
        pass
    # Attempt to find JSON array or object with "gaps" inside code fences first
    try:
        fences = list(re.finditer(r"```json\s*(.*?)\s*```", raw, flags=re.S | re.I))
        for m in fences:
            snippet = m.group(1)
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict) and (
                    isinstance(obj.get("gaps"), list) or isinstance(obj.get("data_gaps"), list)
                ):
                    md = (raw[: m.start()] + raw[m.end() :]).strip()
                    arr = obj.get("gaps") or obj.get("data_gaps")
                    return md, arr  # type: ignore
                if isinstance(obj, list):
                    md = (raw[: m.start()] + raw[m.end() :]).strip()
                    return md, obj
            except Exception:
                continue
    except Exception:
        pass
    # Find first JSON array in the output (non-fenced)
    try:
        # crude heuristic: first '[' to matching ']' block that parses
        start = raw.find("[")
        if start == -1:
            return raw.strip(), []
        for end in range(len(raw), start, -1):
            chunk = raw[start:end]
            try:
                arr = json.loads(chunk)
                if isinstance(arr, list):
                    md = (raw[:start] + raw[end:]).strip()
                    return md, arr  # type: ignore
            except Exception:
                continue
        return raw.strip(), []
    except Exception:
        return raw.strip(), []
