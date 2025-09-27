from __future__ import annotations

from typing import Dict, List, Tuple


def filter_events(events: List[Dict], source: str, focus_gaps: List[str]) -> List[Dict]:
    out: List[Dict] = []
    focus = set(focus_gaps or [])
    for ev in events:
        ch = ev.get("channel")
        if ch not in {"nlq", "slack", "email"}:
            continue
        if source != "all" and ch != source:
            continue
        if focus:
            # Only keep events with outcome != answered and matching focus gaps
            if ev.get("outcome") == "answered":
                continue
            gaps = set(ev.get("gap_types") or [])
            # Treat generic label as matching either missing_column or missing_asset
            if "missing_dataset_or_column" in gaps and ("missing_column" in focus or "missing_asset" in focus):
                pass
            elif not (gaps & focus):
                continue
        out.append(ev)
    return out
