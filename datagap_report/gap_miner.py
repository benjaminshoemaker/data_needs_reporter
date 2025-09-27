from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple, Any


_re_missing_col = re.compile(r"column [\"'`]?([A-Za-z0-9_]+)[\"'`]? does not exist", re.I)
_re_missing_tab = re.compile(r"(relation|table) [\"'`]?([A-Za-z0-9_.]+)[\"'`]? does not exist", re.I)


def _norm_token(tok: str) -> str:
    t = tok.strip().strip("`'\"")
    t = t.replace(" ", "_")
    return t.lower()


def _infer_target_type(token: str, tables: set[str], cols: set[str], parsed_sql: str | None) -> str:
    t = _norm_token(token)
    if t in tables:
        return "table"
    if t in cols:
        return "column"
    if parsed_sql:
        up = parsed_sql.upper()
        if re.search(rf"\bFROM\s+{re.escape(token)}\b|\bJOIN\s+{re.escape(token)}\b", up):
            return "table"
        if re.search(rf"\bSELECT\s+.*\b{re.escape(token)}\b|\bWHERE\s+.*\b{re.escape(token)}\b|\bGROUP BY\s+.*\b{re.escape(token)}\b", up):
            return "column"
    return "column"


def mine_missing_contexts(
    nlq_events: List[Dict[str, Any]],
    intents: Dict[str, Dict[str, Any]],
    catalog_datasets: List[Dict[str, Any]],
    min_freq: int = 3,
) -> List[Dict[str, Any]]:
    # Catalog refs
    tables = {d["name"].lower() for d in catalog_datasets}
    cols = {c["name"].lower() for d in catalog_datasets for c in d.get("columns", [])}
    owner_by_ds = {d["name"].lower(): d.get("owner") for d in catalog_datasets}

    buckets: Dict[tuple, Dict[str, Any]] = {}

    for ev in nlq_events:
        rid = ev["id"]
        intent = intents.get(rid, {})
        text = ev.get("text", "")
        parsed_sql = ev.get("parsed_sql")
        gaps = ev.get("gap_types") or []
        # extract targets via regex
        targets: List[tuple[str, str]] = []  # (type, token)
        m1 = _re_missing_col.search(text)
        if m1:
            targets.append(("missing_column", _norm_token(m1.group(1))))
        m2 = _re_missing_tab.search(text)
        if m2:
            targets.append(("missing_asset", _norm_token(m2.group(2))))
        # fallback: from intent tokens not in catalog
        for token in intent.get("metrics", []) + intent.get("dimensions", []):
            t = _norm_token(token)
            if len(t) < 3:
                continue
            if t not in cols:
                targets.append(("missing_column", t))
        # dedup targets
        seen_t = set()
        dedup_targets: List[tuple[str, str]] = []
        for t in targets:
            if t in seen_t:
                continue
            seen_t.add(t)
            dedup_targets.append(t)
        if not dedup_targets:
            continue
        # build context per target
        metric_set = sorted(list({m for m in intent.get("metrics", [])}))
        dim_set = sorted(list({d for d in intent.get("dimensions", [])}))
        timeframe = intent.get("timeframe")
        ds_ref = [t.lower() for t in intent.get("tables_ref", [])]
        owners = sorted(list({owner_by_ds.get(x) for x in ds_ref if owner_by_ds.get(x)}))
        example = (rid, ev.get("channel", "nlq"), text[:160])
        blocked = ev.get("outcome") == "blocked"
        uncertain = ev.get("outcome") in {"slow", "uncertain"}

        for missing_type, target in dedup_targets:
            # If generic label present, infer type
            if "missing_dataset_or_column" in gaps:
                tt = _infer_target_type(target, tables, cols, parsed_sql)
                missing_type = "missing_asset" if tt == "table" else "missing_column"
            key = (missing_type, target, tuple(metric_set), tuple(dim_set), timeframe)
            rec = buckets.get(key)
            if not rec:
                rec = {
                    "metric_set": metric_set,
                    "dim_set": dim_set,
                    "timeframe": timeframe,
                    "missing_type": missing_type,
                    "missing_target": target,
                    "datasets_ref": sorted(list(set(ds_ref))),
                    "owners": owners,
                    "examples": [example],
                    "count": 1,
                    "_signals": 0,
                    "_blocked": blocked,
                    "_uncertain": uncertain,
                }
                buckets[key] = rec
            else:
                rec["count"] += 1
                rec["datasets_ref"] = sorted(list(set(rec["datasets_ref"] + ds_ref)))
                rec["owners"] = sorted(list(set(rec["owners"] + owners)))
                # keep 3 shortest examples
                exs = rec["examples"] + [example]
                exs.sort(key=lambda e: len(e[2]))
                rec["examples"] = exs[:3]
                rec["_blocked"] = rec["_blocked"] or blocked
                rec["_uncertain"] = rec["_uncertain"] or uncertain
            # signals heuristics
            if m1 or m2:
                rec["_signals"] += 1
            if target in tables or target in cols:
                rec["_signals"] += 1

    # filter by min frequency
    results = [v for v in buckets.values() if v["count"] >= max(1, int(min_freq))]
    # sort by count desc, then blocked severity
    results.sort(key=lambda r: (r["count"], r["_blocked"], r["_uncertain"]), reverse=True)
    return results
