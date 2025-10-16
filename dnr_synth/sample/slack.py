"""Generate grounded Slack threads."""

from __future__ import annotations

import itertools
from datetime import datetime
from typing import Dict, List

import numpy as np

from .context import DataContext, SignalAnomaly, SignalLagSpike, SignalNullSpike

ROLE_SEQUENCE = ["Product Manager", "Data Analyst", "Engineer", "Data Engineer", "UX"]
ROLE_ALIASES = {
    "Product Manager": "pm_riley",
    "Data Analyst": "da_jules",
    "Engineer": "eng_kai",
    "Data Engineer": "de_amelia",
    "UX": "ux_sol",
}


def generate_threads(ctx: DataContext, k: int, seed: int | None) -> List[List[Dict]]:
    rng = np.random.default_rng(seed or 7)
    base_epoch = int(rng.integers(1_700_000_000, 1_800_000_000))
    threads: List[List[Dict]] = []
    for idx in range(k):
        scenario = _pick_scenario(ctx, rng)
        message_count = int(rng.integers(12, 21))
        roles_cycle = list(itertools.islice(itertools.cycle(ROLE_SEQUENCE), message_count))
        # ensure each role appears at least once
        for i, role in enumerate(ROLE_SEQUENCE):
            roles_cycle[i] = role

        ts_base = float(base_epoch + idx * 1000)
        references = list(scenario["references"])
        if not references:
            references = ["quality.issue"]
        ref_cycle = itertools.cycle(references)
        thread_messages: List[Dict] = []
        for m_idx in range(message_count):
            ts = ts_base + m_idx * 60.0
            role = roles_cycle[m_idx]
            alias = ROLE_ALIASES[role]
            text = _compose_message(role, m_idx, scenario, ctx, rng)
            if m_idx < len(references):
                ref = references[m_idx]
                text += f"\nSource: `{ref}`"
            elif m_idx % 3 == 2:
                ref = next(ref_cycle)
                text += f"\nData source: `{ref}`"
            message = {
                "ts": f"{ts:.1f}",
                "user": alias,
                "text": text.strip(),
                "thread_ts": f"{ts_base:.1f}",
            }
            thread_messages.append(message)
        # first message thread_ts should match its ts
        thread_messages[0]["thread_ts"] = thread_messages[0]["ts"]
        threads.append(thread_messages)
    return threads


def _pick_scenario(ctx: DataContext, rng: np.random.Generator) -> Dict:
    if ctx.signals.anomalies:
        idx = int(rng.integers(0, len(ctx.signals.anomalies)))
        signal: SignalAnomaly = ctx.signals.anomalies[idx]
        references = _ensure_references(signal, ctx)
        return {
            "type": "anomaly",
            "signal": signal,
            "references": references,
        }
    if ctx.signals.null_spikes:
        idx = int(rng.integers(0, len(ctx.signals.null_spikes)))
        signal: SignalNullSpike = ctx.signals.null_spikes[idx]
        references = {f"{signal.table}.{signal.col}"}
        return {
            "type": "null",
            "signal": signal,
            "references": references,
        }
    if ctx.signals.lag_spikes:
        idx = int(rng.integers(0, len(ctx.signals.lag_spikes)))
        signal: SignalLagSpike = ctx.signals.lag_spikes[idx]
        references = {f"{signal.table}.ingested_at", f"{signal.table}.event_time"}
        return {
            "type": "lag",
            "signal": signal,
            "references": references,
        }
    # fallback quality thread
    if ctx.tables:
        ref = next(iter(ctx.tables.keys()))
        profile = ctx.tables[ref]
        references = {f"{ref}.{col}" for col in profile.columns[:3]}
        if len(references) < 3:
            for table, prof in ctx.tables.items():
                references.update(f"{table}.{col}" for col in prof.columns[:2])
                if len(references) >= 3:
                    break
    else:
        references = {"quality.issue"}
    return {
        "type": "quality",
        "references": references,
    }


def _ensure_references(signal: SignalAnomaly, ctx: DataContext) -> set[str]:
    table = signal.metric.split(".")[0]
    profile = ctx.tables.get(table)
    references = set()
    if profile:
        if "approved" in profile.columns:
            references.add(f"{table}.approved")
        if "attempts" in profile.columns:
            references.add(f"{table}.attempts")
    if profile:
        for col in profile.columns[:3]:
            references.add(f"{table}.{col}")
    return {ref for ref in references if ref.split(".")[0] in ctx.tables}


def _compose_message(role: str, idx: int, scenario: Dict, ctx: DataContext, rng: np.random.Generator) -> str:
    if scenario["type"] == "anomaly":
        signal: SignalAnomaly = scenario["signal"]
        delta_pct = abs(signal.delta) * 100
        ds = signal.when["start"]
        if role == "Product Manager":
            return (
                f"Heads up team, {signal.metric} dropped by {delta_pct:.1f} pts on {ds}."
                f" Need to understand impact on conversion."
            )
        if role == "Data Analyst":
            return (
                f"Pulling daily cuts from `{signal.metric}` around {ds}; will break down by bin and region."
            )
        if role == "Engineer":
            return "Assessing service latency and auth retries; will flag if we see API degradation."
        if role == "Data Engineer":
            ticket = f"ENG-{rng.integers(100, 999)}"
            return (
                f"Opening {ticket} to backfill `{signal.metric}` and confirm ingestion pacing."
            )
        if role == "UX":
            return "I'll draft comms for CX once we confirm scope; focusing on checkout cohorts."

    if scenario["type"] == "null":
        signal: SignalNullSpike = scenario["signal"]
        rate = signal.null_rate * 100
        if role == "Product Manager":
            return (
                f"Null spike detected on `{signal.table}.{signal.col}` reaching {rate:.1f}% on {signal.ds}."
            )
        if role == "Data Analyst":
            return "I'll quantify impact on dashboard filters and share before standup."
        if role == "Engineer":
            return "Checking recent deploys touching the form payload; suspect validation drift."
        if role == "Data Engineer":
            ticket = f"DATA-{rng.integers(400, 799)}"
            return f"Reprocessing affected partitions now; tracking in {ticket}."
        if role == "UX":
            return "Reviewing the flows to see if alternate paths create missing fields."

    if scenario["type"] == "lag":
        signal: SignalLagSpike = scenario["signal"]
        if role == "Product Manager":
            return (
                f"Ingestion lag for `{signal.table}` hit {signal.p95_minutes:.0f}m on {signal.ds}."
            )
        if role == "Data Analyst":
            return "Publishing disclaimer on freshness in dashboards until lag clears."
        if role == "Engineer":
            return "Investigating queue depth and scaling workers; expecting catch-up in 30m."
        if role == "Data Engineer":
            return "Restarted the loader with higher batch size; monitoring S3 landing times."
        if role == "UX":
            return "Will prepare status update copy for in-product alerts."

    # quality fallback
    if role == "Product Manager":
        return "Need a quick readout on pipeline quality; last week's review flagged gaps."
    if role == "Data Analyst":
        return "I'll summarize the metric drift and outline affected dashboards."
    if role == "Engineer":
        return "Checking application logs to ensure no 500 spikes correlate."
    if role == "Data Engineer":
        return "Syncing with infra to confirm storage latency; will update with plan."
    return "Documenting UX considerations so we can align messaging."
