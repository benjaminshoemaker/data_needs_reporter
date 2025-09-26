from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

from .util import bounded_time, isoformat_z, sorted_by_when_id, write_jsonl


def _id_maker(prefix: str, width: int, start: int = 1):
    i = start
    while True:
        yield f"{prefix}_{i:0{width}d}"
        i += 1


def generate_slack(
    out_dir: Path,
    cfg: Dict,
    rng: random.Random,
    now,
    time_window_days: int,
    queries: List[Dict],
) -> int:
    slack_threads = int(cfg["slack"]["threads"])
    replies_mean = int(cfg["slack"]["replies_mean"])
    s_ids = _id_maker("s", 4)
    events: List[Dict] = []
    for _ in range(slack_threads):
        thread_id = next(s_ids)
        actor = rng.choice(list(cfg["actors"]))
        when = isoformat_z(bounded_time(rng, now, time_window_days))
        referenced = rng.choice([None] * 2 + [rng.choice(queries)["id"]]) if queries else None
        text = rng.choice([
            "Anyone know if this table has column X?",
            "Is the revenue metric delayed today?",
            "Access denied on dataset, who can help?",
            "Docs missing on customers table",
            "Query seems slow this morning",
        ])
        events.append(
            {
                "id": thread_id,
                "when": when,
                "actor": actor,
                "channel": "slack",
                "thread_id": None,
                "text": text,
                "labels": [],
                "referenced_query_id": referenced,
            }
        )

        n_replies = max(0, int(rng.gauss(mu=replies_mean, sigma=max(1, replies_mean / 2))))
        for _r in range(n_replies):
            events.append(
                {
                    "id": next(s_ids),
                    "when": isoformat_z(bounded_time(rng, now, time_window_days)),
                    "actor": rng.choice(list(cfg["actors"])) if rng.random() < 0.8 else actor,
                    "channel": "slack",
                    "thread_id": thread_id,
                    "text": rng.choice([
                        "I'll check the docs",
                        "Looks like a freshness breach",
                        "Try joining on customer_id",
                        "We need a mart table for this",
                        "Grant added. Please retry",
                    ]),
                    "labels": [],
                    "referenced_query_id": None,
                }
            )

    events = sorted_by_when_id(events)
    return write_jsonl(out_dir / "slack.jsonl", events)

