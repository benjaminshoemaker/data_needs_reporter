from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from .util import bounded_time, isoformat_z, sorted_by_when_id, write_jsonl


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
    nl_count = int(cfg["nlq"]["count"])
    success_rate = float(cfg["nlq"]["success_rate"])
    slow_rate = float(cfg["nlq"]["slow_rate"])
    uncertain_rate = float(cfg["nlq"]["uncertain_rate"])
    gaps_dist = dict(cfg["nlq"]["gaps_distribution"])  # type: ignore

    all_dims, all_metrics = _sample_dims_metrics_from_catalog(datasets, rng)
    q_ids = _id_maker("q", 6)
    records: List[Dict] = []
    for _ in range(nl_count):
        qid = next(q_ids)
        when = isoformat_z(bounded_time(rng, now, time_window_days))
        actor = rng.choice(list(cfg["actors"]))
        metric = rng.choice(all_metrics)
        dim = rng.choice(all_dims)
        segment = rng.choice(["all", "vip", "new", "returning", "promo"])
        window = rng.choice(["7 days", "30 days", "quarter", "year"])
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
            tbl = tables[0]
            parsed_sql = f"select {metric}, {dim} from {tbl} where dt >= current_date - interval '7 day'"

        records.append(
            {
                "id": qid,
                "when": when,
                "actor": actor,
                "channel": "nlq",
                "nl_text": _nl_text_from_template(metric, dim, segment, window),
                "parsed_sql": parsed_sql,
                "tables": tables,
                "dims": [dim],
                "metrics": [metric],
                "timeframe": window,
                "outcome": outcome,
                "gap_types": gap_types,
            }
        )

    records = sorted_by_when_id(records)
    return write_jsonl(out_dir / "nl_queries.jsonl", records)

