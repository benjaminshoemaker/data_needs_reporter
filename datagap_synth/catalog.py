from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from .util import ensure_dir, isoformat_z, write_csv, write_json


@dataclass
class CatalogItem:
    name: str
    domain: str
    layer: str  # staging|mart
    owner: str
    columns: List[Dict[str, str]]


def _random_columns(rng: random.Random) -> List[Dict[str, str]]:
    dim_names = [
        "country",
        "state",
        "channel",
        "device",
        "segment",
        "sku",
        "category",
    ]
    metric_names = ["revenue", "orders", "units", "visits", "customers"]
    cols: List[Dict[str, str]] = []
    # some dims
    for _ in range(rng.randint(2, 5)):
        cols.append({"name": rng.choice(dim_names), "type": "string"})
    # some metrics
    for _ in range(rng.randint(2, 5)):
        cols.append({"name": rng.choice(metric_names), "type": rng.choice(["integer", "number"])})
    # ts
    cols.append({"name": "dt", "type": "date"})
    return cols


def generate_catalog(
    out_dir: Path,
    cfg: Dict,
    rng: random.Random,
    now: datetime,
    time_window_days: int,
) -> Dict[str, int]:
    ensure_dir(out_dir)
    cat_dir = out_dir / "catalog"
    ensure_dir(cat_dir)

    domains: List[str] = list(cfg["catalog"]["domains"])
    staging_n = int(cfg["catalog"]["staging_tables"])
    mart_n = int(cfg["catalog"]["mart_tables"])
    actors: List[str] = list(cfg["actors"])

    datasets: List[CatalogItem] = []
    # staging
    for i in range(staging_n):
        dom = rng.choice(domains)
        name = f"stg_{dom}_{i:03d}"
        datasets.append(
            CatalogItem(
                name=name,
                domain=dom,
                layer="staging",
                owner=rng.choice(actors),
                columns=_random_columns(rng),
            )
        )
    # mart
    for i in range(mart_n):
        dom = rng.choice(domains)
        name = f"mart_{dom}_{i:03d}"
        datasets.append(
            CatalogItem(
                name=name,
                domain=dom,
                layer="mart",
                owner=rng.choice(actors),
                columns=_random_columns(rng),
            )
        )

    datasets_json = [
        {
            "name": d.name,
            "domain": d.domain,
            "layer": d.layer,
            "owner": d.owner,
            "columns": d.columns,
        }
        for d in datasets
    ]
    write_json(cat_dir / "datasets.json", datasets_json)

    # lineage: link staging to mart; and include a notional source
    edges = []
    for d in datasets:
        if d.layer == "staging":
            src = f"src_{d.domain}"
            edges.append({"from": src, "to": d.name})
    for d in datasets:
        if d.layer == "mart":
            # pick any staging in same domain
            stg = rng.choice([s for s in datasets if s.layer == "staging" and s.domain == d.domain])
            edges.append({"from": stg.name, "to": d.name})
    write_json(cat_dir / "lineage.json", edges)

    # freshness CSV for all datasets
    sla_choices: List[int] = list(cfg["catalog"]["freshness_sla_hours"])
    breach_rate: float = float(cfg["catalog"]["inject_freshness_breaches"])
    headers = ["dataset", "max_loaded_at", "sla_hours"]
    rows: List[Tuple[str, str, int]] = []
    for d in datasets:
        sla = rng.choice(sla_choices)
        is_breach = rng.random() < breach_rate
        hours_back = sla + rng.randint(1, 72) if is_breach else rng.randint(0, max(1, sla - 1))
        loaded_at = now - timedelta(hours=hours_back)
        rows.append((d.name, isoformat_z(loaded_at), sla))
    write_csv(cat_dir / "freshness.csv", headers, rows)

    return {
        "datasets": len(datasets),
        "lineage_edges": len(edges),
        "freshness_rows": len(rows),
    }
