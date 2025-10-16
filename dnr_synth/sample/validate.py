"""Validators for generated outputs."""

from __future__ import annotations

from datetime import datetime
import itertools
from typing import Dict, Iterable, List

from .context import DataContext


VALID_ROLES = {
    "Product Manager",
    "Engineer",
    "UX",
    "Data Engineer",
    "Data Analyst",
}

VALID_USERS = {
    "pm_riley",
    "eng_kai",
    "ux_sol",
    "de_amelia",
    "da_jules",
}


def validate_queries(queries: List[Dict], ctx: DataContext) -> List[Dict]:
    valid_refs = _valid_references(ctx)
    enums = ctx.enums
    cal_start = datetime.fromisoformat(ctx.calendar["start"]).date()
    cal_end = datetime.fromisoformat(ctx.calendar["end"]).date()

    seen_text: List[str] = []
    passed: List[Dict] = []
    for entry in queries:
        if not set(entry.get("references", [])) <= valid_refs:
            continue
        time_range = entry.get("time_range", "")
        if not _within_calendar(time_range, cal_start, cal_end):
            continue
        if entry.get("role") not in VALID_ROLES:
            continue
        if not _entities_exist(entry.get("entities", {}), enums):
            continue
        if _is_duplicate(entry.get("query", ""), seen_text):
            continue
        passed.append(entry)
        seen_text.append(entry.get("query", ""))

    if not passed or len(passed) < max(1, int(0.8 * len(queries))):
        raise ValueError("Insufficient validated NL queries")
    return passed


def validate_threads(threads: List[List[Dict]], ctx: DataContext) -> List[List[Dict]]:
    valid_refs = _valid_references(ctx)
    filtered: List[List[Dict]] = []
    for thread in threads:
        if not thread:
            continue
        ref_hits = set()
        ok = True
        for message in thread:
            text = message.get("text", "")
            if any(token in text for token in ["{", "lorem", "TODO"]):
                ok = False
                break
            if message.get("user") not in VALID_USERS:
                ok = False
                break
            ref_hits.update(ref for ref in valid_refs if f"`{ref}`" in text or ref in text)
        if ok and len(ref_hits) >= 3:
            filtered.append(thread)

    if not filtered or len(filtered) < max(1, int(0.8 * len(threads))):
        raise ValueError("Insufficient validated Slack threads")
    return filtered


def _valid_references(ctx: DataContext) -> set[str]:
    refs = set()
    for table, profile in ctx.tables.items():
        for col in profile.columns:
            refs.add(f"{table}.{col}")
    return refs


def _within_calendar(time_range: str, start: datetime.date, end: datetime.date) -> bool:
    if "to" not in time_range:
        return True
    try:
        start_str, end_str = [part.strip() for part in time_range.split("to", 1)]
        start_dt = datetime.fromisoformat(start_str).date()
        end_dt = datetime.fromisoformat(end_str).date()
    except ValueError:
        return False
    return start <= start_dt <= end and start <= end_dt <= end


def _entities_exist(entities: Dict[str, str], enums: Dict[str, List[str]]) -> bool:
    if not entities:
        return True
    enum_values = set(itertools.chain.from_iterable(enums.values()))
    return all(value in enum_values for value in entities.values())


def _is_duplicate(text: str, seen: Iterable[str]) -> bool:
    text_tokens = set(text.lower().split())
    for other in seen:
        other_tokens = set(other.lower().split())
        overlap = len(text_tokens & other_tokens)
        if not text_tokens or not other_tokens:
            continue
        if overlap / min(len(text_tokens), len(other_tokens)) >= 0.9:
            return True
    return False
