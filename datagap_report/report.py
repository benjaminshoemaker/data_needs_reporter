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
from .normalize import filter_events
from .gap_miner import mine_missing_contexts
from .score import score_missing_context
from . import __version__ as REPORT_VERSION
import subprocess


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
    s = re.sub(r"\b\+?\d[\d\-\s]{7,}\b", "[redacted_phone]", s)
    s = re.sub(r"\b\d{9,}\b", "[redacted_id]", s)
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


def _normalize_events_by_channel(pack: Dict[str, Any], horizon_days: int = 7) -> Dict[str, List[Dict[str, Any]]]:
    now = _utc_parse(pack["manifest"]["generated_at"]) if "generated_at" in pack["manifest"] else datetime.now(UTC)
    start = now - timedelta(days=horizon_days)
    per: Dict[str, List[Dict[str, Any]]] = {"nlq": [], "slack": [], "email": []}
    # NLQ
    for r in pack["nlq"]:
        t = _utc_parse(r["when"]) if "when" in r else now
        if t < start or t > now:
            continue
        per["nlq"].append(
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
        per["slack"].append(
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
        per["email"].append(
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
    for ch in per:
        per[ch].sort(key=lambda r: (r["when"], r["id"]))
    return per


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
    llm_budget_tokens: int = 0,
    sample_per_channel: int = 25,
    sample_random: bool = False,
    source: str = "all",
    focus_gaps: List[str] | None = None,
    min_frequency: int = 3,
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
    by_ch = _normalize_events_by_channel(pack)
    # sampling per channel (default 25; 0 means all)
    if sample_per_channel and sample_per_channel > 0:
        if sample_random:
            import random as _rnd
            n = sample_per_channel
            nlq_ev = _rnd.sample(by_ch["nlq"], k=min(n, len(by_ch["nlq"])) )
            slack_ev = _rnd.sample(by_ch["slack"], k=min(n, len(by_ch["slack"])) )
            email_ev = _rnd.sample(by_ch["email"], k=min(n, len(by_ch["email"])) )
        else:
            nlq_ev = by_ch["nlq"][: sample_per_channel]
            slack_ev = by_ch["slack"][: sample_per_channel]
            email_ev = by_ch["email"][: sample_per_channel]
    else:
        nlq_ev, slack_ev, email_ev = by_ch["nlq"], by_ch["slack"], by_ch["email"]
    events_all = sorted(nlq_ev + slack_ev + email_ev, key=lambda r: (r["when"], r["id"]))
    if source != "all" or (focus_gaps):
        events = filter_events(events_all, source=source, focus_gaps=(focus_gaps or []))
    else:
        events = events_all
    log.info(
        "Normalized events: %d (nlq=%d slack=%d email=%d; last 7 days) filtered=%d",
        len(events_all), len(nlq_ev), len(slack_ev), len(email_ev), len(events)
    )

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
        if llm_on and not llm_disabled and (llm_budget_tokens <= 0 or (client.cost_summary()["responses"]["prompt_tokens"] + client.cost_summary()["responses"]["completion_tokens"]) < llm_budget_tokens):
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
                    max_output_tokens=256,
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
        if llm_on and not llm_disabled and (llm_budget_tokens <= 0 or (client.cost_summary()["responses"]["prompt_tokens"] + client.cost_summary()["responses"]["completion_tokens"]) < llm_budget_tokens):
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
    # Determine top audience from config (highest weight key if present)
    cfg = pack.get("manifest", {}).get("config_echo", {})
    aud_weights = cfg.get("audience_weights") or {"Exec": 3, "Ops": 2, "Eng": 1}
    audience_top = None
    if isinstance(aud_weights, dict) and aud_weights:
        audience_top = sorted(aud_weights.items(), key=lambda kv: -kv[1])[0][0]
    for gid, cl in clusters.items():
        title = None
        why = None
        if llm_on and not llm_disabled and (llm_budget_tokens <= 0 or (client.cost_summary()["responses"]["prompt_tokens"] + client.cost_summary()["responses"]["completion_tokens"]) < llm_budget_tokens):
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
                "audience_top": audience_top,
            }
            try:
                out = client.structured_json(
                    model=model,
                    system=system,
                    user=user,
                    response_format=BACKLOG_SUMMARY_SCHEMA,
                    max_output_tokens=192,
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
    # persist run config (with version, timestamp, seed, git commit)
    try:
        git_sha = None
        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            git_sha = None
        run_cfg = {
            "version": REPORT_VERSION,
            "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "seed": pack.get("manifest", {}).get("seed"),
            "model": model if llm_on else "off",
            "embed_model": embed_model,
            "cli_flags": {
                "llm_on": llm_on,
                "limit_llm": llm_limit,
                "llm_budget_tokens": llm_budget_tokens,
                "sample_per_channel": sample_per_channel,
                "sample_random": sample_random,
                "source": source,
                "focus_gaps": focus_gaps or [],
                "min_frequency": min_frequency,
            },
            "git_commit": git_sha,
        }
        (artifacts / "run_config.json").write_text(json.dumps(run_cfg, indent=2, sort_keys=True))
    except Exception as e:
        raise RuntimeError(f"Failed to write artifacts/run_config.json: {e}")
    # intents
    with (artifacts / "intents.jsonl").open("w", encoding="utf-8") as f:
        for ev in events:
            rid = ev["id"]
            f.write(json.dumps({"id": rid, **intents[rid]}, sort_keys=True) + "\n")
    # clusters
    (artifacts / "clusters.json").write_text(json.dumps(clusters, indent=2, sort_keys=True))

    # scoring and ranking (by frequency/size)
    rank = sorted(clusters.keys(), key=lambda c: len(clusters[c]["ids"]), reverse=True)

    # backlog.csv
    bl_path = out_dir / "backlog.csv"
    with bl_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "title", "why", "gap_types", "datasets", "severity", "effort"])
        for gid in rank:
            cl = clusters[gid]
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

    # Optional: NLQ missing data gap mining section data
    nlq_missing_section_md: List[str] = []
    nlq_missing_section_html: List[str] = []
    if source == "nlq" and (focus_gaps or []) and any(g in {"missing_column", "missing_asset"} for g in (focus_gaps or [])):
        nlq_only = [e for e in events if e.get("channel") == "nlq"]
        log.info("NLQ mining: nlq_total=%d (filtered)", len(nlq_only))
        mined = mine_missing_contexts(nlq_only, intents, pack["datasets"], min_freq=min_frequency)
        log.info("NLQ mining: contexts=%d (min_freq=%d)", len(mined), min_frequency)
        for ctx in mined:
            ctx["score"] = score_missing_context(ctx)
        mined.sort(key=lambda x: x["score"], reverse=True)
        # write CSV artifact
        with (artifacts / "nlq_missing_summary.csv").open("w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["missing_type", "missing_target", "metrics", "dims", "timeframe", "count", "score", "datasets", "owners", "examples_json"])
            for ctx in mined:
                w.writerow([
                    ctx["missing_type"],
                    ctx["missing_target"],
                    ";".join(ctx["metric_set"]),
                    ";".join(ctx["dim_set"]),
                    ctx.get("timeframe") or "",
                    ctx["count"],
                    ctx["score"],
                    ";".join(ctx["datasets_ref"]),
                    ";".join(ctx["owners"]),
                    json.dumps(ctx["examples"], ensure_ascii=False),
                ])
        # Build sections (top 10)
        if mined:
            nlq_missing_section_md.append("## Top NLQ Missing Data Gaps")
            nlq_missing_section_html.append("<h2>Top NLQ Missing Data Gaps</h2>")
            topk = mined[:10]
            fresh_by_ds = {row["dataset"]: row for row in pack["freshness"]}
            now_dt = _utc_parse(pack["manifest"].get("generated_at", datetime.now(UTC).isoformat()))
            for ctx in topk:
                metrics = ", ".join(ctx["metric_set"]) or "metrics"
                dims = ", ".join(ctx["dim_set"]) or "dimensions"
                tf = f", timeframe {ctx['timeframe']}" if ctx.get("timeframe") else ""
                title = (
                    f"Add column {ctx['missing_target']} to enable {metrics} by {dims}"
                    if ctx["missing_type"] == "missing_column"
                    else f"Create dataset {ctx['missing_target']} to support {metrics} by {dims}"
                )
                why = (
                    f"In the last 7 days we saw {ctx['count']} NL queries asking for {metrics} by {dims}{tf}, "
                    f"but {ctx['missing_type'].replace('_',' ')} {ctx['missing_target']} is not available in the catalog. "
                    f"Affected datasets: {', '.join(ctx['datasets_ref']) or 'N/A'}. Owners: {', '.join(ctx['owners']) or 'unassigned'}."
                )
                ex_lines = [f"[{ch}] {_redact(txt)}" for (_id, ch, txt) in ctx["examples"]]
                fresh_lines: List[str] = []
                for ds in ctx["datasets_ref"][:8]:
                    row = fresh_by_ds.get(ds)
                    if not row:
                        fresh_lines.append(f"{ds} (freshness: unknown)")
                        continue
                    try:
                        dt = _utc_parse(row["max_loaded_at"]) if isinstance(row["max_loaded_at"], str) else now_dt
                        sla_h = int(row["sla_hours"]) if isinstance(row["sla_hours"], str) else int(row["sla_hours"])
                        breach = (now_dt - dt).total_seconds() > sla_h * 3600
                        fresh_lines.append(f"{ds} (loaded: {row['max_loaded_at']}, SLA: {sla_h}h, breach: {str(breach).lower()})")
                    except Exception:
                        fresh_lines.append(f"{ds} (freshness: unreadable)")
                # Append md
                nlq_missing_section_md += [f"### {title}", f"- Why: {why}", "- Examples:"]
                for ex in ex_lines:
                    nlq_missing_section_md.append(f"  - {ex}")
                nlq_missing_section_md.append("- Datasets:")
                for line in fresh_lines:
                    nlq_missing_section_md.append(f"  - {line}")
                nlq_missing_section_md.append("")
                # Append html
                nlq_missing_section_html += [f"<h3>{title}</h3>", f"<p><strong>Why:</strong> {why}</p>", "<p><strong>Examples:</strong></p><ul>"]
                for ex in ex_lines:
                    nlq_missing_section_html.append(f"<li>{ex}</li>")
                nlq_missing_section_html.append("</ul>")
                nlq_missing_section_html.append("<p><strong>Datasets:</strong></p><ul>")
                for line in fresh_lines:
                    nlq_missing_section_html.append(f"<li>{line}</li>")
                nlq_missing_section_html.append("</ul>")

    # report.md and report.html with Top 3 Data Gaps
    # helpers for plain-language section
    def _human_gap(label: str) -> str:
        mapping = {
            "missing_dataset_or_column": "Missing dataset or column",
            "grain_or_type_mismatch": "Grain or type mismatch",
            "freshness_breach": "Freshness breach",
            "joinability_not_defined": "Joinability not defined",
            "access_denied": "Access denied",
            "semantics_unclear": "Semantics unclear",
            "performance_limit": "Performance limit",
        }
        return mapping.get(label, label)

    def _pick_examples(cluster_id: str) -> List[str]:
        pool = [e for e in events if e["id"] in clusters[cluster_id]["ids"]]
        pool.sort(key=lambda r: len(r.get("text", "")))
        seen = set()
        out: List[str] = []
        for r in pool:
            key = (r.get("actor"), r.get("channel"))
            if key in seen:
                continue
            seen.add(key)
            out.append(f"[{r.get('channel')}] {_redact(r.get('text',''))}")
            if len(out) >= 3:
                break
        return out

    fresh_rows = pack["freshness"]
    fresh_by_ds = {row["dataset"]: row for row in fresh_rows}
    now_dt = _utc_parse(pack["manifest"].get("generated_at", datetime.now(UTC).isoformat()))

    def _dataset_fresh_lines(cluster_id: str) -> List[str]:
        out: List[str] = []
        for ds in clusters[cluster_id]["datasets"][:8]:
            row = fresh_by_ds.get(ds)
            if not row:
                out.append(f"{ds} (freshness: unknown)")
                continue
            try:
                dt = _utc_parse(row["max_loaded_at"])
                sla_h = int(row["sla_hours"]) if isinstance(row["sla_hours"], str) else int(row["sla_hours"])
                breach = (now_dt - dt).total_seconds() > sla_h * 3600
                out.append(f"{ds} (loaded: {row['max_loaded_at']}, SLA: {sla_h}h, breach: {str(breach).lower()})")
            except Exception:
                out.append(f"{ds} (freshness: unreadable)")
        return out

    top3 = rank[:3]

    report_md = out_dir / "report.md"
    md_lines: List[str] = []
    md_lines += ["# Data Gaps Report (1 week)", "", f"Clusters: {len(clusters)}", ""]
    md_lines += ["## Top 3 Data Gaps (Plain Language)"]
    for gid in top3:
        cl = clusters[gid]
        md_lines += [
            f"### {summaries[gid]['title']}",
            f"- Why: {summaries[gid]['why']}",
            f"- Root cause: {', '.join(_human_gap(g) for g in cl['gap_types']) or 'N/A'}",
            f"- Impact: {len(cl['ids'])} related items; top audience: {audience_top or 'N/A'}",
            "- Examples:",
        ]
        for ex in _pick_examples(gid):
            md_lines.append(f"  - {ex}")
        md_lines.append("- Datasets:")
        for line in _dataset_fresh_lines(gid):
            md_lines.append(f"  - {line}")
        md_lines.append("")
    # Optional NLQ missing section
    if nlq_missing_section_md:
        md_lines += nlq_missing_section_md
    md_lines += ["## Top Backlog Items", ""]
    md_lines += [f"- [{gid}] {summaries[gid]['title']}" for gid in rank[:50]]
    report_md.write_text("\n".join(md_lines))

    report_html = out_dir / "report.html"
    html_lines: List[str] = []
    html_lines.append(
        """
<!doctype html>
<meta charset=\"utf-8\"/>
<title>Data Gaps Report</title>
<style>body{font-family:system-ui,Arial;margin:2rem;max-width:900px}li{margin:.25rem 0}</style>
<h1>Data Gaps Report (1 week)</h1>
<p>See backlog.csv for details.</p>
"""
    )
    html_lines.append("<h2>Top 3 Data Gaps (Plain Language)</h2>")
    for gid in top3:
        cl = clusters[gid]
        html_lines.append(f"<h3>{summaries[gid]['title']}</h3>")
        html_lines.append(f"<p><strong>Why:</strong> {summaries[gid]['why']}</p>")
        html_lines.append(f"<p><strong>Root cause:</strong> {', '.join(_human_gap(g) for g in cl['gap_types']) or 'N/A'}</p>")
        html_lines.append(f"<p><strong>Impact:</strong> {len(cl['ids'])} related items; top audience: {audience_top or 'N/A'}</p>")
        html_lines.append("<p><strong>Examples:</strong></p><ul>")
        for ex in _pick_examples(gid):
            html_lines.append(f"<li>{ex}</li>")
        html_lines.append("</ul>")
        html_lines.append("<p><strong>Datasets:</strong></p><ul>")
        for line in _dataset_fresh_lines(gid):
            html_lines.append(f"<li>{line}</li>")
        html_lines.append("</ul>")
    # Optional NLQ mined section
    report_has_nlq_section = False
    try:
        # If we wrote the md section, the html list will be non-empty
        report_has_nlq_section = bool('nlq_missing_section_html' in locals() and nlq_missing_section_html)
    except Exception:
        report_has_nlq_section = False
    if report_has_nlq_section:
        html_lines.append("<h2>Top NLQ Missing Data Gaps</h2>")
        html_lines.extend(nlq_missing_section_html)
    if nlq_missing_section_html:
        html_lines.append("<h2>Top NLQ Missing Data Gaps</h2>")
        html_lines.extend(nlq_missing_section_html)
    html_lines.append("<h2>Top Backlog Items</h2><ul>")
    for gid in rank[:50]:
        html_lines.append(f"<li><code>{gid}</code> {summaries[gid]['title']}</li>")
    html_lines.append("</ul>")
    report_html.write_text("\n".join(html_lines))

    log.info("Report done: clusters=%d", len(clusters))
    # write cost summary
    cost = client.cost_summary()
    (artifacts / "cost.json").write_text(json.dumps(cost, indent=2, sort_keys=True))

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
