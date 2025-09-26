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


def generate_email(
    out_dir: Path,
    cfg: Dict,
    rng: random.Random,
    now,
    time_window_days: int,
    queries: List[Dict],
) -> int:
    threads = int(cfg["email"]["threads"])
    replies_mean = int(cfg["email"]["replies_mean"])
    e_ids = _id_maker("e", 4)
    events: List[Dict] = []
    for _ in range(threads):
        thread = next(e_ids)
        actor = rng.choice(list(cfg["actors"]))
        when = isoformat_z(bounded_time(rng, now, time_window_days))
        referenced = rng.choice([None] * 3 + [rng.choice(queries)["id"]]) if queries else None
        events.append(
            {
                "id": thread,
                "when": when,
                "actor": actor,
                "channel": "email",
                "thread_id": None,
                "subject": rng.choice([
                    "Weekly KPI question",
                    "Data issue on orders mart",
                    "Access request",
                    "Help with join keys",
                ]),
                "body": rng.choice([
                    "Can we segment by device and country?",
                    "Seeing a mismatch vs finance reports.",
                    "Requesting access to marketing datasets.",
                    "Which column is the join key?",
                ]),
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
                        "Added notes to the doc",
                        "Refreshing the dataset now",
                        "Approved access",
                        "We lack column requested",
                    ]),
                    "referenced_query_id": None,
                }
            )

    events = sorted_by_when_id(events)
    return write_jsonl(out_dir / "email.jsonl", events)

