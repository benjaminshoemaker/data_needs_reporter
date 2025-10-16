"""Generate grounded NL queries."""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

from .context import DataContext, MetricSpec
from .provider import LLMProvider, TemplateProvider


ROLES = [
    "Product Manager",
    "Engineer",
    "UX",
    "Data Engineer",
    "Data Analyst",
]

INTENT_WEIGHTS = {
    "descriptive": 0.40,
    "comparative": 0.25,
    "diagnostic": 0.20,
    "anomaly": 0.10,
    "forecast": 0.05,
}


def generate_queries(
    ctx: DataContext,
    n: int,
    seed: int | None,
    role_mix: Dict[str, int] | None = None,
    provider: LLMProvider | None = None,
) -> List[Dict]:
    rng = np.random.default_rng(seed or 7)
    provider = provider or TemplateProvider(ctx)
    schema = SchemaIndex(ctx)

    intents = list(INTENT_WEIGHTS.keys())
    weights = np.array(list(INTENT_WEIGHTS.values()))
    weights = weights / weights.sum()

    role_sequence = _build_roles(n, role_mix)

    start_dt = datetime.fromisoformat(ctx.calendar["start"])
    end_dt = datetime.fromisoformat(ctx.calendar["end"])
    span_days = max(1, (end_dt - start_dt).days)

    candidates: List[Dict] = []
    for i in range(max(n * 2, 2)):
        role = role_sequence[i % len(role_sequence)]
        intent = rng.choice(intents, p=weights)
        metric = schema.sample_metric(rng)
        dims = schema.sample_dimensions(metric.table, rng)
        entities = {}
        references = set(metric.cols_used)

        for dim in dims:
            value = schema.sample_value(dim, rng)
            if value is None:
                continue
            entities[dim.split(".")[-1]] = value
            references.add(dim)

        time_length = int(rng.integers(7, 15))
        start_offset = int(rng.integers(0, max(1, span_days - time_length + 1)))
        window_start = (start_dt + timedelta(days=int(start_offset))).date()
        window_end = (window_start + timedelta(days=int(time_length))).isoformat()
        window_start_iso = window_start.isoformat()
        time_range = f"{window_start_iso} to {window_end}"

        prompt = _compose_prompt(intent, metric, entities, time_range, ctx.domain)
        text = provider.complete("You write grounded data questions.", prompt, seed=(seed or 7) + i)

        query = {
            "query": text.strip(),
            "role": role,
            "intent": intent,
            "entities": entities,
            "time_range": time_range,
            "references": sorted(references),
        }
        candidates.append(query)
        if len(candidates) >= n * 2:
            break

    return candidates[:n]


def _build_roles(n: int, role_mix: Dict[str, int] | None) -> List[str]:
    if role_mix:
        seq = list(itertools.chain.from_iterable([[role] * count for role, count in role_mix.items()]))
    else:
        repeat = (n // len(ROLES)) + 1
        seq = list(itertools.islice(itertools.cycle(ROLES), repeat * len(ROLES)))
    if len(seq) < n:
        seq.extend(seq[: n - len(seq)])
    return seq[:n]


def _compose_prompt(intent: str, metric: MetricSpec, entities: Dict[str, str], time_range: str, domain: str) -> str:
    entity_text = ", ".join(f"{k}={v}" for k, v in entities.items()) or "overall"
    base = f"How {intent} is {metric.name} from {metric.table} ({entity_text}) between {time_range}?"
    if intent == "comparative":
        base = f"Compare {metric.name} across {entity_text} in {metric.table} for {time_range}."
    elif intent == "diagnostic":
        base = f"Investigate drivers for {metric.name} in {metric.table} for {entity_text} during {time_range}."
    elif intent == "anomaly":
        base = f"Confirm anomaly in {metric.name} for {entity_text} ({metric.table}) circa {time_range}."
    elif intent == "forecast":
        base = f"Project trend of {metric.name} ({metric.table}) for {entity_text} beyond {time_range}."
    if domain == "fintech":
        base += " Include approval and fraud context if relevant."
    return base


class SchemaIndex:
    def __init__(self, ctx: DataContext):
        self.ctx = ctx
        self.metrics = ctx.metrics or self._fallback_metrics()
        self.enums = ctx.enums
        self.table_columns = {t: prof.columns for t, prof in ctx.tables.items()}
        self.numeric_cols = [key for key in ctx.numerics.keys()]

    def _fallback_metrics(self) -> List[MetricSpec]:
        metrics: List[MetricSpec] = []
        for key in self.ctx.numerics.keys():
            table, col = key.split(".", 1)
            metrics.append(
                MetricSpec(
                    name=f"avg_{col}",
                    table=table,
                    formula=f"AVG({col})",
                    grain="daily",
                    cols_used=[key],
                )
            )
        return metrics

    def sample_metric(self, rng: np.random.Generator) -> MetricSpec:
        index = int(rng.integers(0, len(self.metrics)))
        return self.metrics[index]

    def sample_dimensions(self, table: str, rng: np.random.Generator, max_dims: int = 2) -> List[str]:
        candidates = [key for key in self.enums.keys() if key.startswith(f"{table}.")]
        if not candidates:
            # use other table enums as fallbacks
            candidates = list(self.enums.keys())
        if not candidates:
            return []
        order = rng.permutation(len(candidates))
        return [candidates[i] for i in order[:max_dims]]

    def sample_value(self, column: str, rng: np.random.Generator) -> str | None:
        values = self.enums.get(column)
        if not values:
            return None
        return rng.choice(values)
