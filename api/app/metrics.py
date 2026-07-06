"""Performance metrics computed from trades + equity snapshots.

Kept as pure functions over plain sequences so the golden-file test can
feed fixtures without a DB.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_metrics(equity: list[tuple[Any, float]],
                    trade_pnls: list[float]) -> dict[str, Any]:
    """equity: [(ts, value)] ascending; trade_pnls: closed round-trip P&Ls."""
    out: dict[str, Any] = {
        "total_return_pct": 0.0, "sharpe_daily_annualized": 0.0,
        "max_drawdown_pct": 0.0, "hit_rate": 0.0,
        "avg_win": 0.0, "avg_loss": 0.0, "turnover": len(trade_pnls),
        "n_trades": len(trade_pnls),
    }
    if len(equity) >= 2:
        vals = np.array([v for _, v in equity], dtype=np.float64)
        out["total_return_pct"] = round(float(vals[-1] / vals[0] - 1.0) * 100, 4)
        # daily-ish returns: resample by taking one point per date
        by_day: dict[Any, float] = {}
        for ts, v in equity:
            key = ts.date() if hasattr(ts, "date") else ts
            by_day[key] = v
        dvals = np.array(list(by_day.values()), dtype=np.float64)
        if len(dvals) >= 3:
            rets = np.diff(dvals) / np.maximum(dvals[:-1], 1e-9)
            if rets.std() > 1e-12:
                out["sharpe_daily_annualized"] = round(
                    float(rets.mean() / rets.std() * np.sqrt(252)), 4)
        peak = np.maximum.accumulate(vals)
        out["max_drawdown_pct"] = round(float(np.max((peak - vals) / peak)) * 100, 4)
    if trade_pnls:
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        out["hit_rate"] = round(len(wins) / len(trade_pnls), 4)
        out["avg_win"] = round(float(np.mean(wins)), 4) if wins else 0.0
        out["avg_loss"] = round(float(np.mean(losses)), 4) if losses else 0.0
    return out


def downsample(points: list[dict[str, Any]], max_points: int = 2000) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return [points[i] for i in idx]
