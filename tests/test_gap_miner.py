from __future__ import annotations

from datagap_report.gap_miner import mine_missing_contexts


def _catalog():
    return [
        {"name": "mart_orders_001", "owner": "u_owner1", "columns": [{"name": "orders", "type": "integer"}, {"name": "date", "type": "date"}]},
        {"name": "stg_customers_001", "owner": "u_owner2", "columns": [{"name": "customer_id", "type": "string"}, {"name": "region", "type": "string"}]},
    ]


def test_regex_extraction_missing_column_and_table_grouping():
    # two NLQ events about same missing column and same grouping context
    events = [
        {
            "id": "q_1",
            "channel": "nlq",
            "text": 'Error: column "foo_bar" does not exist',
            "outcome": "blocked",
            "parsed_sql": "select foo_bar from mart_orders_001",
        },
        {
            "id": "q_2",
            "channel": "nlq",
            "text": 'Error: column `foo_bar` does not exist',
            "outcome": "blocked",
            "parsed_sql": "select foo_bar from mart_orders_001",
        },
        {
            "id": "q_3",
            "channel": "nlq",
            "text": 'relation "tbl_missing" does not exist',
            "outcome": "blocked",
            "parsed_sql": "select orders from tbl_missing",
        },
    ]
    intents = {
        "q_1": {"metrics": ["orders"], "dimensions": ["date"], "timeframe": "last week", "tables_ref": ["mart_orders_001"]},
        "q_2": {"metrics": ["orders"], "dimensions": ["date"], "timeframe": "last week", "tables_ref": ["mart_orders_001"]},
        "q_3": {"metrics": ["orders"], "dimensions": ["region"], "timeframe": None, "tables_ref": []},
    }
    res = mine_missing_contexts(events, intents, _catalog(), min_freq=1)
    # expect a grouped missing_column foo_bar with count 2
    foo = [r for r in res if r["missing_type"] == "missing_column" and r["missing_target"] == "foo_bar"]
    assert foo, "expected missing_column foo_bar"
    assert foo[0]["count"] == 2
    # expect a missing_asset tbl_missing
    tab = [r for r in res if r["missing_type"] == "missing_asset" and r["missing_target"] == "tbl_missing"]
    assert tab, "expected missing_asset tbl_missing"
