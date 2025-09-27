from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from .util import bounded_time, isoformat_z, sorted_by_when_id, write_jsonl
from .scenarios import load_scenario
import re


def _id_maker(prefix: str, width: int, start: int = 1):
    i = start
    while True:
        yield f"{prefix}_{i:0{width}d}"
        i += 1


def _choose_gap_types(rng: random.Random, gaps_dist: Dict[str, float]) -> List[str]:
    weights = list(gaps_dist.values())
    keys = list(gaps_dist.keys())
    total = sum(weights)
    if total <= 0:
        return []
    if rng.random() < total:
        pick = rng.random() * total
        acc = 0.0
        for k, w in zip(keys, weights):
            acc += w
            if pick <= acc:
                return [k]
    return []


def _nl_text_from_template(metric: str, dim: str, segment: str, window: str) -> str:
    return f"daily {metric} by {dim} for {segment} last {window}"


def _sample_dims_metrics_from_catalog(datasets: List[Dict], rng: random.Random) -> Tuple[List[str], List[str]]:
    dims: List[str] = []
    metrics: List[str] = []
    for d in datasets:
        for c in d.get("columns", []):
            if c.get("type") in {"string"}:
                dims.append(c.get("name"))
            if c.get("type") in {"integer", "number"}:
                metrics.append(c.get("name"))
    # Dedup preserve order
    seen = set()
    dims = [x for x in dims if not (x in seen or seen.add(x))]
    seen = set()
    metrics = [x for x in metrics if not (x in seen or seen.add(x))]
    return dims or ["channel"], metrics or ["orders"]


def generate_nlq(
    out_dir: Path,
    cfg: Dict,
    rng: random.Random,
    now,
    time_window_days: int,
    datasets: List[Dict],
) -> int:
    scenario_name = cfg.get("scenario", "enterprise")
    scenario = load_scenario(scenario_name)
    nl_count = int(cfg["nlq"]["count"])
    success_rate = float(cfg["nlq"]["success_rate"])
    slow_rate = float(cfg["nlq"]["slow_rate"])
    uncertain_rate = float(cfg["nlq"]["uncertain_rate"])
    gaps_dist = dict(cfg["nlq"]["gaps_distribution"])  # type: ignore

    all_dims, all_metrics = _sample_dims_metrics_from_catalog(datasets, rng)
    # incorporate scenario metrics/dimensions
    scen_dims = list(scenario.get("dimensions", []))
    if scen_dims:
        all_dims = list(dict.fromkeys(all_dims + scen_dims))
    scen_metrics = [m for v in scenario.get("metrics", {}).values() for m in v]
    if scen_metrics:
        all_metrics = list(dict.fromkeys(all_metrics + scen_metrics))
    q_ids = _id_maker("q", 6)
    records: List[Dict] = []
    langs = cfg["nlq"].get("languages") or scenario.get("languages", [{"lang": "en", "pct": 1.0}])
    noise = cfg["nlq"].get("noise") or scenario.get("noise", {})
    templates = scenario.get("nlq_templates", ["what is {metric} by {dimension} last {timeframe}"])
    synonyms = scenario.get("synonyms", {})
    timeframes = scenario.get("timeframes", ["last week"])
    filters = scenario.get("filters", ["all"])

    def pick_lang() -> str:
        r = rng.random()
        acc = 0.0
        for ent in langs:
            acc += float(ent.get("pct", 0))
            if r <= acc:
                return ent.get("lang", "en")
        return langs[-1].get("lang", "en")

    def maybe_typo(s: str) -> str:
        if rng.random() >= float(noise.get("typos_pct", 0)) or len(s) < 5:
            return s
        i = rng.randint(1, len(s)-2)
        return s[:i] + s[i+1] + s[i] + s[i+2:]

    def maybe_abbrev(token: str) -> str:
        if rng.random() >= float(noise.get("abbreviations_pct", 0)):
            return token
        return token[:3]

    def maybe_vague_time(tf: str) -> str:
        if rng.random() >= float(noise.get("vague_time_pct", 0)):
            return tf
        return rng.choice(["recently", "these days", "lately"])

    def maybe_oos_term(text: str) -> str:
        if rng.random() >= float(noise.get("oos_term_pct", 0)):
            return text
        return text + " for cohort xyz"  # out-of-schema phrase

    for _ in range(nl_count):
        qid = next(q_ids)
        when = isoformat_z(bounded_time(rng, now, time_window_days))
        actor = rng.choice(list(cfg["actors"]))
        metric = rng.choice(all_metrics)
        dim = rng.choice(all_dims)
        filter_token = rng.choice(filters)
        window = rng.choice(timeframes)
        # apply synonyms sometimes
        if rng.random() < 0.3 and metric in synonyms:
            metric = rng.choice(synonyms[metric])
        if rng.random() < 0.3 and dim in synonyms:
            dim = rng.choice(synonyms[dim])
        # template expand
        tmpl = rng.choice(templates)
        nl_text = tmpl.format(metric=metric, dimension=dim, filter=filter_token, timeframe=window)
        nl_text = maybe_oos_term(maybe_typo(nl_text))
        lang = pick_lang()

        gap_types = _choose_gap_types(rng, gaps_dist)

        if gap_types:
            hard = {"missing_asset", "missing_column", "access_denied"}
            outcome = "blocked" if any(g in hard for g in gap_types) else "degraded"
        else:
            r = rng.random()
            if r < success_rate:
                outcome = "answered"
            elif r < success_rate + slow_rate:
                outcome = "slow"
            elif r < success_rate + slow_rate + uncertain_rate:
                outcome = "uncertain"
            else:
                outcome = "answered"

        tables = [rng.choice(datasets)["name"]]
        if rng.random() < 0.2:
            tables.append(rng.choice(datasets)["name"])

        parsed_sql = None
        if outcome == "answered":
            base = tables[0]
            if len(tables) > 1:
                join = tables[1]
                key = rng.choice(["customer_id", "order_id", "sku", "id"])
                parsed_sql = (
                    f"select t1.{dim}, sum(t1.{metric}) as {metric} "
                    f"from {base} t1 join {join} t2 on t1.{key}=t2.{key} "
                    f"where t1.dt >= current_date - interval '7 day' group by 1"
                )
            else:
                parsed_sql = f"select {metric}, {dim} from {base} where dt >= current_date - interval '7 day'"

        records.append(
            {
                "id": qid,
                "when": when,
                "actor": actor,
                "channel": "nlq",
                "nl_text": nl_text,
                "parsed_sql": parsed_sql,
                "tables": tables,
                "dims": [dim],
                "metrics": [metric],
                "timeframe": maybe_vague_time(window),
                "outcome": outcome,
                "gap_types": gap_types,
                "lang": lang,
            }
        )

    records = sorted_by_when_id(records)
    return write_jsonl(out_dir / "nl_queries.jsonl", records)
