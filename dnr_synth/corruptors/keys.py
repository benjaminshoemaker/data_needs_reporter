"""Key related corruptors."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd


ContextFn = Callable[[pd.Series], Dict[str, str]]


def apply_key_presence(
    df: pd.DataFrame,
    rules: Dict[str, Dict[str, float]],
    context_fn: ContextFn,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Null out key columns according to presence probabilities."""

    if not rules:
        return df
    result = df.copy()
    for column, spec in rules.items():
        if column not in result.columns:
            continue
        col_values = result[column].to_numpy(copy=False)
        mask = np.zeros(len(result), dtype=bool)
        base_p = float(spec.get("base_p", 1.0))
        for idx, (_, row) in enumerate(result.iterrows()):
            ctx = context_fn(row)
            prob = base_p
            for value in ctx.values():
                key = f"{value}_p"
                if key in spec:
                    prob = float(spec[key])
                    break
            if rng.random() > prob:
                mask[idx] = True
        if mask.any():
            col_values[mask] = pd.NA
            result[column] = col_values
    return result


def enforce_joinability(
    df: pd.DataFrame,
    targets: Dict[str, float],
    fk_map: Dict[str, str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Null foreign keys so join success approximates *targets*."""

    if not targets:
        return df
    result = df.copy()
    for dim_name, target in targets.items():
        col = fk_map.get(dim_name)
        if col is None or col not in result.columns:
            continue
        present_idx = result.index[result[col].notna()]
        if not len(present_idx):
            continue
        target = float(np.clip(target, 0.0, 1.0))
        keep = int(round(len(present_idx) * target))
        drop = len(present_idx) - keep
        if drop <= 0:
            continue
        drop_idx = rng.choice(present_idx.to_numpy(), size=drop, replace=False)
        result.loc[drop_idx, col] = pd.NA
    return result
