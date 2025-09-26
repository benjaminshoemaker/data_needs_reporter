INTENT_SCHEMA = {
    "name": "Intent",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "metric": {"type": "array", "items": {"type": "string"}},
            "dimensions": {"type": "array", "items": {"type": "string"}},
            "filters": {"type": "array", "items": {"type": "string"}},
            "timeframe": {"type": ["string", "null"]},
            "tables_ref": {"type": "array", "items": {"type": "string"}},
            "cols_ref": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "metric",
            "dimensions",
            "filters",
            "timeframe",
            "tables_ref",
            "cols_ref",
        ],
    },
    "strict": True,
}

GAPTYPES_SCHEMA = {
    "name": "GapTypes",
    "schema": {
        "type": "object",
        "properties": {
            "gap_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "missing_dataset_or_column",
                        "grain_or_type_mismatch",
                        "freshness_breach",
                        "joinability_not_defined",
                        "access_denied",
                        "semantics_unclear",
                        "performance_limit",
                    ],
                },
            }
        },
        "required": ["gap_types"],
        "additionalProperties": False,
    },
    "strict": True,
}

BACKLOG_SUMMARY_SCHEMA = {
    "name": "BacklogSummary",
    "schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "maxLength": 90},
            "why": {"type": "string"},
        },
        "required": ["title", "why"],
        "additionalProperties": False,
    },
    "strict": True,
}

