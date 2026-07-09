"""Walk-forward ranking tournament with costs, vs SPY and EW-universe.

Design decisions:
- Expanding-window retrain once per calendar year: train on every sample
  whose forward-return window closed before the test year starts (cutoff
  Nov-30 of the prior year, so no label window crosses the boundary).
- Any ranker with fit(X, y)/predict(X) competes; a plain mom_12_1 ranking
  runs alongside as the no-ML baseline. Models are judged ONLY by the same
  cost-aware portfolio and by rank IC — no per-model scoreboards.
- Portfolio: equal-weight top-N, buy-and-hold within the month (weights
  drift), long-only (matches live platform LOW/MEDIUM tiers).
- Costs: cost_bps per one-way traded notional at each rebalance, turnover
  measured against drifted (not initial) prior weights.
- Vol overlay (Barroso-Santa-Clara): scale next-day exposure by
  target_vol / trailing realized vol, capped at 1 (long-only, remainder in
  cash). Uses only PAST strategy returns — shift(1) enforced. Extra
  scaling turnover is not charged (changes are monthly-smooth and small
  next to the 10bps rebalance charge; noted, not hidden).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .features import FEATURES
from .models import GBTRanker

COST_BPS = 10.0  # one-way, on traded notional; liquid names, patient limit entry
TOP_N = 50


class Ranker(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class BacktestResult:
    daily_returns: pd.Series          # strategy, net of costs
    gross_returns: pd.Series          # strategy, before costs
    holdings: dict[pd.Timestamp, list[str]] = field(default_factory=dict)
    turnover: pd.Series | None = None  # one-way, per rebalance


def train_cutoff(year: int) -> pd.Timestamp:
    """Latest sample date whose 21d label window closed before Jan 1 of `year`."""
    return pd.Timestamp(f"{year - 1}-11-30")


def yearly_models(ds: pd.DataFrame, oos_years: list[int],
                  make_model: Any = None,
                  features: list[str] | None = None) -> dict[int, Ranker]:
    """Train one ranker per OOS year on all fully-realized samples before it."""
    make_model = make_model or (lambda: GBTRanker())
    features = features or [c for c in FEATURES if c in ds.columns]
    models: dict[int, Ranker] = {}
    train = ds.dropna(subset=["fwd_ret", "y_rank"])
    for year in oos_years:
        sub = train[train["date"] <= train_cutoff(year)]
        models[year] = make_model().fit(sub[features].values, sub["y_rank"].values)
    return models


def predict_scores(ds: pd.DataFrame, models: dict[int, Ranker],
                   momentum_only: bool = False,
                   features: list[str] | None = None) -> pd.DataFrame:
    """Per (date, symbol) score for OOS rebalance dates."""
    features = features or [c for c in FEATURES if c in ds.columns]
    out = []
    for date, block in ds.groupby("date"):
        year = int(date.year)
        if year not in models:
            continue
        b = block.copy()
        b["score"] = (b["mom_12_1"] if momentum_only
                      else models[year].predict(b[features].values))
        out.append(b[["date", "symbol", "score", "fwd_ret"]])
    return pd.concat(out, ignore_index=True)


def information_coefficient(scores: pd.DataFrame) -> dict[str, float]:
    """Mean per-date Spearman rank IC of score vs realized forward return."""
    ics = []
    for _, block in scores.dropna(subset=["fwd_ret"]).groupby("date"):
        if len(block) < 20:
            continue
        ics.append(float(block["score"].rank().corr(block["fwd_ret"].rank())))
    arr = np.array(ics)
    return {"ic_mean": float(arr.mean()),
            "ic_tstat": float(arr.mean() / arr.std() * np.sqrt(len(arr))),
            "ic_hit": float((arr > 0).mean())}


def run_portfolio(scores: pd.DataFrame, close: pd.DataFrame,
                  top_n: int = TOP_N, cost_bps: float = COST_BPS) -> BacktestResult:
    ret = close.pct_change()
    dates = sorted(scores["date"].unique())
    all_days = close.index
    daily_net: list[pd.Series] = []
    daily_gross: list[pd.Series] = []
    holdings: dict[pd.Timestamp, list[str]] = {}
    turnovers: dict[pd.Timestamp, float] = {}
    prev_drifted = pd.Series(dtype=float)

    for i, d in enumerate(dates):
        picks = (scores[scores["date"] == d].nlargest(top_n, "score")["symbol"].tolist())
        holdings[d] = picks
        w = pd.Series(1.0 / len(picks), index=picks)
        union = w.index.union(prev_drifted.index)
        tw = float((w.reindex(union, fill_value=0.0)
                    - prev_drifted.reindex(union, fill_value=0.0)).abs().sum())
        turnovers[d] = tw

        end = dates[i + 1] if i + 1 < len(dates) else all_days[-1]
        days = all_days[(all_days > d) & (all_days <= end)]
        if len(days) == 0:
            continue
        r = ret.loc[days, picks].fillna(0.0)
        wealth = (1.0 + r).cumprod()
        port_wealth = wealth.mean(axis=1)  # EW buy-and-hold within period
        pr = port_wealth.pct_change()
        pr.iloc[0] = port_wealth.iloc[0] - 1.0
        gross = pr.copy()
        pr.iloc[0] -= tw * cost_bps / 1e4  # charge rebalance cost on day 1
        daily_net.append(pr)
        daily_gross.append(gross)

        drift = wealth.iloc[-1] * (1.0 / len(picks))
        prev_drifted = drift / drift.sum()

    net = pd.concat(daily_net)
    return BacktestResult(daily_returns=net, gross_returns=pd.concat(daily_gross),
                          holdings=holdings,
                          turnover=pd.Series(turnovers).sort_index())


def vol_managed(r: pd.Series, target_vol: float = 0.20, window: int = 63) -> pd.Series:
    """Scale exposure by target/trailing-realized vol, capped at 1 (rest in cash)."""
    realized = r.rolling(window).std() * np.sqrt(252)
    scale = (target_vol / realized).clip(upper=1.0).shift(1).fillna(1.0)
    return r * scale


def summarize(r: pd.Series, label: str, benchmark: pd.Series | None = None) -> dict[str, float]:
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {}
    total = float((1 + r).prod())
    years = n / 252.0
    cagr = total ** (1 / years) - 1
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    wealth = (1 + r).cumprod()
    dd = float((wealth / wealth.cummax() - 1).min())
    out = {"label": label, "cagr": cagr, "vol": vol, "sharpe": sharpe,
           "max_dd": dd, "years": years}
    if benchmark is not None:
        b = benchmark.reindex(r.index).fillna(0.0)
        active = r - b
        out["ann_active"] = float(active.mean() * 252)
        out["ir"] = float(active.mean() / active.std() * np.sqrt(252)) if active.std() > 0 else 0.0
    return out


def per_year_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, r in series.items():
        r = r.dropna()
        rows[name] = (1 + r).groupby(r.index.year).prod() - 1
    return pd.DataFrame(rows)
