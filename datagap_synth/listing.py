from __future__ import annotations

import json
from pathlib import Path
from typing import List


def list_packs(root: Path) -> List[str]:
    lines: List[str] = []
    if not root.exists():
        return lines
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        m = p / "manifest.json"
        if not m.exists():
            continue
        try:
            mf = json.loads(m.read_text())
            counts = mf.get("counts", {})
            lines.append(
                f"{p.name}\tgenerated_at={mf.get('generated_at')}\tnlq={counts.get('nl_queries', 0)}\tslack={counts.get('slack', 0)}\temail={counts.get('email', 0)}\tseed={mf.get('seed')}"
            )
        except Exception:
            lines.append(f"{p.name}\tINVALID manifest.json")
    return lines

