"""Walk-forward, day-granular splits with a 1-day embargo.

Ported discipline from research/lob/train.py::temporal_split, adapted to
calendar days per §5.2.1: train <= T-5, validate T-4..T-1, embargo 1 day.
The leakage test asserts no validation timestamp <= any train timestamp
plus the embargo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class WalkForwardSplit:
    train_idx: list[int]
    val_idx: list[int]


def walk_forward_day_split(
    timestamps: list[datetime], val_days: int = 4, embargo_days: int = 1
) -> WalkForwardSplit:
    """Split bar indices by UTC date: last `val_days` distinct days are
    validation; everything before (last val day start - embargo) is train."""
    days: list[date] = sorted({t.date() for t in timestamps})
    if len(days) < val_days + embargo_days + 2:
        raise ValueError(f"need more distinct days: have {len(days)}")
    val_set = set(days[-val_days:])
    embargo_set = set(days[-(val_days + embargo_days): -val_days])
    train_idx = [i for i, t in enumerate(timestamps)
                 if t.date() not in val_set and t.date() not in embargo_set]
    val_idx = [i for i, t in enumerate(timestamps) if t.date() in val_set]
    return WalkForwardSplit(train_idx=train_idx, val_idx=val_idx)
