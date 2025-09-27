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


def generate_slack(
    out_dir: Path,
    cfg: Dict,
    rng: random.Random,
    now,
    time_window_days: int,
    queries: List[Dict],
) -> int:
    scenario = load_scenario(cfg.get("scenario", "enterprise"))
    slack_threads = int(cfg["slack"]["threads"])
    replies_mean = int(cfg["slack"]["replies_mean"])
    s_ids = _id_maker("s", 4)
    events: List[Dict] = []
    open_templates = scenario.get("slack_templates", {}).get("opening", [
        "Anyone know if this table has column X?",
    ])
    reply_templates = scenario.get("slack_templates", {}).get("replies", [
        "I'll check the docs",
    ])
    code_templates = scenario.get("slack_templates", {}).get("code_snippets", [])
    error_phrases = scenario.get("error_phrases", [])
    for _ in range(slack_threads):
        thread_id = next(s_ids)
        actor = rng.choice(list(cfg["actors"]))
        when = isoformat_z(bounded_time(rng, now, time_window_days))
        ref_prob = rng.random()
        referenced = (rng.choice(queries)["id"] if queries and ref_prob < 0.2 else None)
        # Fill opening with placeholders
        sample_query = rng.choice(queries) if queries else {"tables": ["orders"], "metrics": ["orders"], "dims": ["date"]}
        tmpl = rng.choice(open_templates)
        filter_token = rng.choice(scenario.get("filters", ["all"]))
        text = tmpl.format(
            table=rng.choice(sample_query.get("tables", ["orders"])),
            column=rng.choice(sample_query.get("dims", ["date"])),
            metric=rng.choice(sample_query.get("metrics", ["orders"])),
            dimension=rng.choice(sample_query.get("dims", ["date"])),
            timeframe=rng.choice(["last week", "last month"]),
            segment=rng.choice(["vip", "all"]),
            filter=filter_token,
            key=rng.choice(["customer_id", "order_id", "sku"]),
        )
        # Occasionally include error phrase
        if rng.random() < 0.15 and error_phrases:
            text += f" — {rng.choice(error_phrases)}"
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

        n_replies = max(3, min(8, int(rng.gauss(mu=replies_mean, sigma=max(1, replies_mean / 2)))))
        for r_i in range(n_replies):
            # occasional code block
            reply_text = rng.choice(reply_templates).format(
                table=rng.choice(sample_query.get("tables", ["orders"])),
                column=rng.choice(sample_query.get("dims", ["date"])),
                key=rng.choice(["customer_id", "order_id", "sku"]),
                metric=rng.choice(sample_query.get("metrics", ["orders"]))
            )
            if rng.random() < 0.15 and code_templates:
                code = rng.choice(code_templates).format(
                    table=rng.choice(sample_query.get("tables", ["orders"])),
                    dimension=rng.choice(sample_query.get("dims", ["date"])),
                    metric=rng.choice(sample_query.get("metrics", ["orders"]))
                )
                reply_text += f"\n```sql\n{code}\n```"
            events.append(
                {
                    "id": next(s_ids),
                    "when": isoformat_z(bounded_time(rng, now, time_window_days)),
                    "actor": rng.choice(list(cfg["actors"])) if rng.random() < 0.8 else actor,
                    "channel": "slack",
                    "thread_id": thread_id,
                    "text": reply_text,
                    "labels": [],
                    "referenced_query_id": None,
                }
            )

    events = sorted_by_when_id(events)
    return write_jsonl(out_dir / "slack.jsonl", events)
