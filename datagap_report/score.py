from __future__ import annotations

import math
from typing import Dict


def score_missing_context(ctx: Dict) -> float:
    count = int(ctx.get("count", 1))
    blocked = bool(ctx.get("_blocked"))
    uncertain = bool(ctx.get("_uncertain"))
    signals = int(ctx.get("_signals", 0))
    owners = ctx.get("owners") or []
    missing_type = ctx.get("missing_type")
    # effort
    effort = 1.0 + (1 if missing_type == "missing_column" else 0) + (2 if missing_type == "missing_asset" else 0)
    # audience weight (proxy by owners present)
    audience_weight = 1.2 if owners else 1.0
    # severity
    severity = 3.0 if blocked else (1.5 if uncertain else 1.0)
    # confidence
    confidence = min(1.0, 0.5 + 0.1 * max(0, signals))
    score = math.log1p(max(1, count)) * severity * audience_weight * confidence / max(0.5, effort)
    return round(score, 4)
