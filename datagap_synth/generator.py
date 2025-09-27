from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from . import __version__
from .config import load_config
from .catalog import generate_catalog
from .gen_nlq import generate_nlq
from .gen_slack import generate_slack
from .gen_email import generate_email
from .util import (
    bounded_time,
    collect_file_hashes,
    deterministic_now,
    ensure_dir,
    isoformat_z,
    make_rng,
    sorted_by_when_id,
    write_json,
    write_jsonl,
)


def _id_maker(prefix: str, width: int, start: int = 1):
    i = start
    while True:
        yield f"{prefix}_{i:0{width}d}"
        i += 1


def generate_pack(cfg_path: Path, out_dir: Path, seed: int | None) -> Dict:
    cfg = load_config(cfg_path)
    rng = make_rng(seed)
    now = deterministic_now(seed)
    time_window_days = int(cfg.get("time_window_days", 7))

    ensure_dir(out_dir)

    # Catalog first
    counts_catalog = generate_catalog(out_dir, cfg, rng, now, time_window_days)
    datasets = json.loads((out_dir / "catalog" / "datasets.json").read_text())
    # NL Queries
    nlq_count = generate_nlq(out_dir, cfg, rng, now, time_window_days, datasets)
    # Load queries to feed references to slack/email
    queries = [
        json.loads(line)
        for line in (out_dir / "nl_queries.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # Slack
    slack_count = generate_slack(out_dir, cfg, rng, now, time_window_days, queries)
    # Email
    email_count = generate_email(out_dir, cfg, rng, now, time_window_days, queries)

    # Manifest
    manifest = {
        "pack_id": out_dir.name,
        "generated_at": isoformat_z(now),
        "generator_version": __version__,
        "seed": seed,
        "counts": {
            "nl_queries": nlq_count,
            "slack": slack_count,
            "email": email_count,
            **{f"catalog_{k}": v for k, v in counts_catalog.items()},
        },
        "config_echo": cfg,
    }
    write_json(out_dir / "manifest.json", manifest)

    # Hashes
    hashes = collect_file_hashes(out_dir)
    write_json(out_dir / "hashes.json", hashes)

    # Summary metrics
    # Load files for summary
    def _load_jsonl(path: Path):
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    queries = _load_jsonl(out_dir / "nl_queries.jsonl")
    slack = _load_jsonl(out_dir / "slack.jsonl")
    email = _load_jsonl(out_dir / "email.jsonl")
    total_events = len(queries) + len(slack) + len(email)
    multilingual = sum(1 for r in queries if r.get("lang") and r.get("lang") != "en")
    ref_q = sum(1 for r in slack if r.get("referenced_query_id")) + sum(1 for r in email if r.get("referenced_query_id"))
    # top metrics/dims
    from collections import Counter
    top_metrics = [m for r in queries for m in r.get("metrics", [])]
    top_dims = [d for r in queries for d in r.get("dims", [])]
    top10_metrics = [k for k,_ in Counter(top_metrics).most_common(10)]
    top10_dims = [k for k,_ in Counter(top_dims).most_common(10)]

    result = {
        "pack_dir": str(out_dir),
        "counts": manifest["counts"],
        "summary": {
            "pct_multilingual": round(100.0 * multilingual / max(1, len(queries)), 2),
            "pct_with_references": round(100.0 * ref_q / max(1, total_events), 2),
            "top_metrics": top10_metrics,
            "top_dims": top10_dims,
        },
    }
    return result
