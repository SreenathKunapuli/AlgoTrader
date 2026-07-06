"""`lobctl train-signal` — walk-forward training of lob_flow (§5.2.1).

Backfills 1-min bars, aggregates to 5-min, builds features/labels per
symbol, pools windows across the MEDIUM universe, trains via the ported
loop, and hot-swaps models/lob_flow.pt only when validation macro-F1
beats the incumbent.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import structlog

from .config.settings import get_settings
from .config.tiers import TIERS, Tier
from .data.bar_builder import Bar, aggregate
from .data.history import fetch_minute_bars
from .signals.nn.features import build_features
from .signals.nn.labels import HORIZON_K, calibrate_alpha, make_labels
from .signals.nn.splits import walk_forward_day_split
from .signals.nn.training import export, train

log = structlog.get_logger()


def to_5min(bars_1m: list[Bar]) -> list[Bar]:
    out: list[Bar] = []
    group: list[Bar] = []
    for b in bars_1m:
        group.append(b)
        if len(group) == 5:
            agg = aggregate(group, 300)
            if agg:
                out.append(agg)
            group = []
    return out


def run_training(days: int = 30) -> None:
    s = get_settings()
    universe = TIERS[Tier.MEDIUM].universe
    log.info("train.backfill", days=days, symbols=len(universe))
    history = fetch_minute_bars(s.alpaca_api_key, s.alpaca_secret_key, universe, days)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts_all: list[tuple[list[datetime], int]] = []
    for _sym, bars_1m in history.items():
        bars = to_5min(bars_1m)
        if len(bars) < 200:
            continue
        x = build_features(bars)
        closes = np.array([b.close for b in bars])
        timestamps = [b.ts for b in bars]
        # alpha calibrated on the TRAIN portion only (first 80% of days)
        split = walk_forward_day_split(timestamps)
        train_close = closes[split.train_idx]
        alpha = calibrate_alpha(train_close, HORIZON_K)
        y = make_labels(closes, HORIZON_K, alpha)
        # offset indices into the pooled arrays
        base = sum(len(a) for a in xs)
        xs.append(x)
        ys.append(y)
        ts_all.append((timestamps, base))
    if not xs:
        print("No data — check API keys / market availability.")
        return
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    # pooled walk-forward split: recompute per-symbol, offset, merge
    train_idx: list[int] = []
    val_idx: list[int] = []
    for timestamps, base in ts_all:
        sp = walk_forward_day_split(timestamps)
        train_idx += [i + base for i in sp.train_idx]
        val_idx += [i + base for i in sp.val_idx]

    result = train(x, y, train_idx, val_idx, arch=s.lob_flow_arch)
    log.info("train.done", **result.metrics)

    incumbent = Path(s.models_dir) / "lob_flow_metrics.json"
    if incumbent.exists():
        old = json.loads(incumbent.read_text()).get("val_macro_f1", 0.0)
        if result.best_val_f1 <= old:
            print(f"New model val F1 {result.best_val_f1:.4f} <= incumbent {old:.4f}; "
                  "keeping incumbent.")
            return
    pt = export(result, s.models_dir, n_features=x.shape[1])
    print(f"Exported {pt} — metrics: {json.dumps(result.metrics)}")
