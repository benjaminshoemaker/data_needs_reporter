from __future__ import annotations

import pytest

from dnr_synth.config import NullSpike


def test_nullspike_when_validator_accepts_iso_range() -> None:
    ns = NullSpike(table="t", field="c", p=0.5, when="2025-01-01/2025-01-31")
    assert ns.when == "2025-01-01/2025-01-31"


@pytest.mark.parametrize("bad", ["2025-01-01", "2025/01/01-2025/01/31", "", "2025-13-01/2025-01-01"])
def test_nullspike_when_validator_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        NullSpike(table="t", field="c", p=0.5, when=bad)
