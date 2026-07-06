"""Cost-aware backtest for snapshot-level directional signals.

Execution model (deliberately pessimistic — marketable orders only):
- enter/exit by crossing the spread: buy at best ask, sell at best bid
- additional slippage + fees in basis points of notional
- position in {-1, 0, +1} units, re-evaluated every snapshot
- a signal fires only when the model's class probability clears
  `prob_threshold`; otherwise the position is closed

Adverse-selection filter:
- VPIN-style toxicity = |rolling buy flow - sell flow| / total flow over a
  trailing window. High toxicity means flow is one-sided: an informed trader
  is likely active and a passive counterparty is being run over. Entries are
  suppressed when toxicity > `toxicity_threshold` (existing positions may
  still exit).

This measures whether the classifier's edge survives microstructure
frictions — accuracy alone does not answer that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .labels import DOWN, UP


@dataclass
class BacktestConfig:
    prob_threshold: float = 0.55      # min class prob to open a position
    fee_bps: float = 0.5              # per-side fees, bps of notional
    slippage_bps: float = 0.3         # extra slippage beyond touch, bps
    toxicity_window: int = 50         # snapshots for VPIN-style toxicity
    toxicity_threshold: float = 0.8   # suppress entries above this
    use_toxicity_filter: bool = True
    snapshots_per_year: float = 252 * 6.5 * 3600  # ~1 snapshot/sec assumption


@dataclass
class BacktestResult:
    net_pnl: np.ndarray               # cumulative, per snapshot
    gross_pnl: np.ndarray
    positions: np.ndarray
    n_trades: int
    hit_rate: float
    total_costs: float
    sharpe: float
    max_drawdown: float
    suppressed_entries: int
    summary: dict = field(default_factory=dict)


def vpin_toxicity(buy_flow: np.ndarray, sell_flow: np.ndarray, window: int) -> np.ndarray:
    """|net signed flow| / total flow over a trailing window, in [0, 1]."""
    net = _rolling_sum(buy_flow - sell_flow, window)
    tot = _rolling_sum(buy_flow + sell_flow, window)
    return np.abs(net) / np.maximum(tot, 1e-9)


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([[0.0], x]))
    out = np.empty_like(x, dtype=np.float64)
    out[:w] = c[1 : w + 1]
    out[w:] = c[w + 1 :] - c[1:-w]
    return out


def run_backtest(
    probs: np.ndarray,        # [N, 3] model probabilities at each snapshot
    best_bid: np.ndarray,     # [N] price units (e.g. ticks)
    best_ask: np.ndarray,     # [N]
    buy_flow: np.ndarray,     # [N] market buy volume per snapshot
    sell_flow: np.ndarray,    # [N]
    cfg: BacktestConfig | None = None,
) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    n = len(probs)
    assert best_bid.shape == best_ask.shape == (n,)

    mid = (best_bid + best_ask) / 2.0
    tox = vpin_toxicity(buy_flow, sell_flow, cfg.toxicity_window)

    pos = np.zeros(n, dtype=np.int64)
    cash_gross = np.zeros(n)
    cash_net = np.zeros(n)
    n_trades = 0
    suppressed = 0
    trade_pnls: list[float] = []
    entry_px = 0.0

    cur = 0
    g_cash = 0.0
    n_cash = 0.0

    for t in range(n):
        p_down, _, p_up = probs[t]
        want = cur
        if p_up > cfg.prob_threshold:
            want = 1
        elif p_down > cfg.prob_threshold:
            want = -1
        else:
            want = 0

        opening = (want != 0) and (want != cur)
        if (
            opening
            and cfg.use_toxicity_filter
            and tox[t] > cfg.toxicity_threshold
        ):
            # block the new entry; an open position may still close
            suppressed += 1
            want = 0

        if want != cur:
            # close existing position
            if cur != 0:
                exit_px = best_bid[t] if cur > 0 else best_ask[t]
                fric = exit_px * (cfg.fee_bps + cfg.slippage_bps) * 1e-4
                g_cash += cur * exit_px
                n_cash += cur * exit_px - fric
                trade_pnls.append(cur * (exit_px - entry_px))
                n_trades += 1
            # open new position
            if want != 0:
                entry_px_t = best_ask[t] if want > 0 else best_bid[t]
                fric = entry_px_t * (cfg.fee_bps + cfg.slippage_bps) * 1e-4
                g_cash -= want * entry_px_t
                n_cash -= want * entry_px_t + fric
                entry_px = entry_px_t
                n_trades += 1
            cur = want

        pos[t] = cur
        cash_gross[t] = g_cash + cur * mid[t]   # mark-to-mid
        cash_net[t] = n_cash + cur * mid[t]

    # close any open position at the end (mark-to-mid already reflects it)
    rets = np.diff(cash_net, prepend=0.0)
    denom = rets.std()
    sharpe = (
        float(rets.mean() / denom * np.sqrt(cfg.snapshots_per_year))
        if denom > 1e-12 else 0.0
    )
    peak = np.maximum.accumulate(cash_net)
    max_dd = float(np.max(peak - cash_net))
    hit = float(np.mean([p > 0 for p in trade_pnls])) if trade_pnls else 0.0
    total_costs = float(cash_gross[-1] - cash_net[-1])

    return BacktestResult(
        net_pnl=cash_net,
        gross_pnl=cash_gross,
        positions=pos,
        n_trades=n_trades,
        hit_rate=hit,
        total_costs=total_costs,
        sharpe=sharpe,
        max_drawdown=max_dd,
        suppressed_entries=suppressed,
        summary={
            "final_net_pnl": float(cash_net[-1]),
            "final_gross_pnl": float(cash_gross[-1]),
            "n_trades": n_trades,
            "round_trips": len(trade_pnls),
            "hit_rate": round(hit, 4),
            "total_costs": round(total_costs, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
            "suppressed_entries": suppressed,
            "time_in_market": round(float((pos != 0).mean()), 4),
        },
    )


def signal_from_labels(labels: np.ndarray) -> np.ndarray:
    """Oracle probabilities from true labels — upper bound for the strategy."""
    n = len(labels)
    probs = np.full((n, 3), 1.0 / 3.0)
    probs[labels == UP] = [0.0, 0.0, 1.0]
    probs[labels == DOWN] = [1.0, 0.0, 0.0]
    return probs
