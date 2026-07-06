"""Cost-aware labeling experiment.

Question: is there ANY horizon at which a spread-crossing strategy on this
stock is profitable, even with perfect foresight?

Method: instead of labeling every sub-spread wiggle, only label up/down when
the smoothed future move exceeds the FULL round-trip cost (one spread + two
sides of fees+slippage). Then run the perfect-foresight oracle through the
real cost-aware backtest. If the oracle can't profit at any horizon, no model
can. Where the oracle DOES profit and enough labels exist, we then train the
model to see if it retains skill.

Stage 1 here is the fast oracle sweep (pure numpy + backtest, no training).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob.backtest import BacktestConfig, run_backtest, signal_from_labels
from lob.data import clean_snapshots, load_lobster
from lob.labels import DOWN, INVALID, UP, LabelConfig, class_distribution, make_labels


def frictionless_oracle_pnl(labels: np.ndarray, mid: np.ndarray, px: float) -> float:
    """Directional PnL marking every fill at MID (no spread, no fees).

    Long when label==UP, short when DOWN, flat otherwise; earn the mid move
    to the next step. This isolates whether the DIRECTION is right — i.e.
    what a perfect passive (market-maker) execution could capture. If this is
    strongly positive while spread-crossing is negative, the spread is the
    sole killer and passive execution is the real path to profit.
    """
    pos = np.zeros(len(labels))
    pos[labels == UP] = 1.0
    pos[labels == DOWN] = -1.0
    dmid = np.zeros(len(mid))
    dmid[:-1] = np.diff(mid)
    return float(np.sum(pos * dmid) * px)


FEE_BPS = 0.5
SLIP_BPS = 0.3


def round_trip_cost_dollars(mid_px: float, spread_px: float) -> float:
    """One spread crossed + fees/slippage on both legs."""
    fee_frac = (FEE_BPS + SLIP_BPS) * 1e-4
    return spread_px + 2.0 * fee_frac * mid_px


def cost_aware_alpha(mid: np.ndarray, spread: np.ndarray, px: float,
                     cost_mult: float) -> float:
    """Relative threshold so labeled moves exceed cost_mult * round-trip cost."""
    mid_px = float(np.median(mid)) * px
    spread_px = float(np.median(spread)) * px
    cost = round_trip_cost_dollars(mid_px, spread_px)
    return cost_mult * cost / mid_px


def sweep(name: str, ob: str, msg: str, horizons, cost_mult: float = 1.0):
    r, _ = clean_snapshots(load_lobster(ob, msg))
    mid, spread, px = r.mid, r.spread, r.tick_size
    mid_px = float(np.median(mid)) * px
    spread_px = float(np.median(spread)) * px
    cost = round_trip_cost_dollars(mid_px, spread_px)

    print(f"\n=== {name} ===")
    print(f"price ~${mid_px:.2f}  median spread ${spread_px:.4f}  "
          f"round-trip cost ${cost:.4f}  (label move must exceed "
          f"{cost_mult:.0f}x = ${cost_mult*cost:.4f})")
    print(f"{'horizon':>7} {'%up':>6} {'%flat':>7} {'%down':>6} {'n_trd':>6} "
          f"{'cross_net$':>11} {'frictionless$':>14} {'passive_works?':>15}")

    bid = r.snapshots[:, 2] * px
    ask = r.snapshots[:, 0] * px
    for k in horizons:
        alpha = cost_aware_alpha(mid, spread, px, cost_mult)
        labels = make_labels(mid, LabelConfig(horizon=k, alpha=alpha))
        valid = labels != INVALID
        dist = class_distribution(labels)
        probs = signal_from_labels(labels)
        bt = run_backtest(
            probs[valid], bid[valid], ask[valid],
            r.buy_flow[valid], r.sell_flow[valid],
            BacktestConfig(prob_threshold=0.55, use_toxicity_filter=False),
        )
        net = bt.summary["final_net_pnl"]
        fl = frictionless_oracle_pnl(labels, mid, px)
        passive = "YES" if fl > 0 else "no"
        print(f"{k:>7} {dist['up']*100:>5.1f}% {dist['flat']*100:>6.1f}% "
              f"{dist['down']*100:>5.1f}% {bt.summary['n_trades']:>6} "
              f"{net:>11.2f} {fl:>14.2f} {passive:>15}")


def main():
    horizons = [20, 50, 100, 300, 600, 1000, 2000]
    stocks = [
        ("AAPL", "data/aapl_orderbook_10.csv", "data/aapl_message_10.csv"),
        ("INTC", "data/intc_orderbook_10.csv", "data/intc_message_10.csv"),
        ("AMZN", "data/amzn_orderbook_10.csv", "data/amzn_message_10.csv"),
        ("GOOG", "data/goog_orderbook_10.csv", "data/goog_message_10.csv"),
    ]
    for name, ob, msg in stocks:
        sweep(name, ob, msg, horizons)


if __name__ == "__main__":
    main()
