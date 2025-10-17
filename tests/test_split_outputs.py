from __future__ import annotations

from dnr_synth.analysis.gaps_prompt import split_outputs


def test_split_outputs_parses_gaps_key():
    raw = (
        "# Report\n\nDetails...\n\n"
        "```json\n{\n  \"gaps\": [ {\n    \"gap_id\": \"G1\", \"type\": \"JOIN\", \"title\": \"A\", \"priority_score\": 90\n  } ]\n}\n```"
    )
    md, gaps = split_outputs(raw)
    assert "# Report" in md
    assert isinstance(gaps, list) and gaps and gaps[0]["gap_id"] == "G1"


def test_split_outputs_parses_data_gaps_key():
    raw = (
        "some md...\n\n"
        "```json\n{\n  \"data_gaps\": [ {\n    \"gap_code\": \"TIMEL\", \"description\": \"Lag\", \"priority\": 75\n  } ]\n}\n```"
    )
    md, gaps = split_outputs(raw)
    assert "some md" in md
    assert isinstance(gaps, list) and gaps and gaps[0]["gap_code"] == "TIMEL"
