from __future__ import annotations

from datagap_report.normalize import filter_events
from datagap_report.score import score_missing_context


def test_filter_events_source_and_focus():
    events = [
        {"id": "a", "channel": "nlq", "outcome": "blocked", "gap_types": ["missing_dataset_or_column"]},
        {"id": "b", "channel": "slack", "outcome": None, "gap_types": []},
        {"id": "c", "channel": "nlq", "outcome": "answered", "gap_types": ["missing_dataset_or_column"]},
    ]
    out = filter_events(events, source="nlq", focus_gaps=["missing_column"])
    # Only the first should remain (nlq, not answered, gap matches)
    assert len(out) == 1 and out[0]["id"] == "a"


def test_score_missing_context_blocked_higher():
    low = {
        "count": 3,
        "_blocked": False,
        "_uncertain": True,
        "_signals": 1,
        "owners": [],
        "missing_type": "missing_column",
    }
    high = {
        "count": 3,
        "_blocked": True,
        "_uncertain": False,
        "_signals": 2,
        "owners": ["u_owner"],
        "missing_type": "missing_column",
    }
    assert score_missing_context(high) > score_missing_context(low)
