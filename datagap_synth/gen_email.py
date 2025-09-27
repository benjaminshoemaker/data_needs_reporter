from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

from .util import bounded_time, isoformat_z, sorted_by_when_id, write_jsonl
from .scenarios import load_scenario


def _id_maker(prefix: str, width: int, start: int = 1):
    i = start
    while True:
        yield f"{prefix}_{i:0{width}d}"
        i += 1


def generate_email(
    out_dir: Path,
    cfg: Dict,
    rng: random.Random,
    now,
    time_window_days: int,
    queries: List[Dict],
) -> int:
    scenario = load_scenario(cfg.get("scenario", "enterprise"))
    threads = int(cfg["email"]["threads"])
    replies_mean = int(cfg["email"]["replies_mean"])
    e_ids = _id_maker("e", 4)
    events: List[Dict] = []
    subjects = scenario.get("email_subjects", ["Data request"])
    error_phrases = scenario.get("error_phrases", [])
    for _ in range(threads):
        thread = next(e_ids)
        actor = rng.choice(list(cfg["actors"]))
        when = isoformat_z(bounded_time(rng, now, time_window_days))
        referenced = (rng.choice(queries)["id"] if queries and rng.random() < 0.25 else None)
        events.append(
            {
                "id": thread,
                "when": when,
                "actor": actor,
                "channel": "email",
                "thread_id": None,
                "subject": rng.choice(subjects).format(
                    metric=rng.choice(["revenue", "orders", "active_users"]),
                    dimension=rng.choice(["region", "device", "plan"]),
                    table=rng.choice((queries[0].get("tables") if queries else ["orders"])) if queries else "orders",
                    dataset=rng.choice((queries[0].get("tables") if queries else ["customers"])) if queries else "customers",
                ),
                "body": rng.choice([
                    "Could you share a quick pull by segment for the last month?",
                    "Seeing delays on the mart table, is freshness within SLA?",
                    "Access request for the marketing schema, permission denied errors.",
                    "Is the join key customer_id or user_id?",
                ]) + (f" ({rng.choice(error_phrases)})" if rng.random() < 0.2 and error_phrases else ""),
                "referenced_query_id": referenced,
            }
        )
        n_replies = max(0, int(rng.gauss(mu=replies_mean, sigma=max(1, replies_mean / 2))))
        for _r in range(n_replies):
            events.append(
                {
                    "id": next(e_ids),
                    "when": isoformat_z(bounded_time(rng, now, time_window_days)),
                    "actor": rng.choice(list(cfg["actors"])) if rng.random() < 0.8 else actor,
                    "channel": "email",
                    "thread_id": thread,
                    "subject": "Re: ",
                    "body": rng.choice([
                        "Added notes to the doc.",
                        "Refreshing the dataset now.",
                        "Approved access.",
                        "We lack the requested column.",
                    ]),
                    "referenced_query_id": None,
                }
            )

    events = sorted_by_when_id(events)
    return write_jsonl(out_dir / "email.jsonl", events)
