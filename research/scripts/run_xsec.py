"""End-to-end cross-sectional walk-forward experiment.

Usage: python -m research.scripts.run_xsec  (from repo root, .env present)

Prints: strategy (ML ranker) vs mom-only baseline vs SPY vs EW-universe,
net of costs, OOS 2020+, with per-year returns. Writes picks and daily
returns to data/xsec_results/ for inspection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from research.xsec import backtest as bt          # noqa: E402
from research.xsec import features as ft          # noqa: E402
from research.xsec.data import fetch_daily, load_universe, to_panels  # noqa: E402

OOS_START_YEAR = 2020

# Liquidity-at-the-time eligibility screen. 300 ≈ top 60% of the ~500-name
# panel, matching the live book's liquidity_keep=0.60 (config/xsec.py).
# Without this screen the backtest inflates ~5%/yr from index-inclusion bias
# (measured 2026-07-09: unfiltered mom 33.2% CAGR vs 28.4% filtered) — the
# headline numbers are ALWAYS the filtered ones.
ELIGIBLE_TOP = 300


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    syms = load_universe(str(root / "data/sp500_constituents.csv")) + ["SPY"]
    raw = fetch_daily(syms, str(root / "data/xsec_daily.pkl"))
    panels = to_panels(raw)
    close, volume = panels["close"], panels["volume"]
    spy = close.pop("SPY")
    volume = volume.drop(columns=["SPY"], errors="ignore")
    print(f"panel: {close.shape[0]} days x {close.shape[1]} symbols, "
          f"{close.index[0].date()} .. {close.index[-1].date()}")

    dates = ft.month_end_dates(close.index)
    ds = ft.build_dataset(close, volume, dates, eligible_top=ELIGIBLE_TOP)
    print(f"dataset: {len(ds):,} samples over {ds['date'].nunique()} rebalance dates")

    oos_years = sorted({int(d.year) for d in dates if d.year >= OOS_START_YEAR})
    models = bt.yearly_models(ds, oos_years)

    ml_scores = bt.predict_scores(ds, models)
    mom_scores = bt.predict_scores(ds, models, momentum_only=True)
    ml = bt.run_portfolio(ml_scores, close)
    mom = bt.run_portfolio(mom_scores, close)

    idx = ml.daily_returns.index
    spy_ret = spy.pct_change().reindex(idx).fillna(0.0)
    ew_ret = close.pct_change().mean(axis=1).reindex(idx).fillna(0.0)

    series = {"ml_net": ml.daily_returns, "ml_gross": ml.gross_returns,
              "mom_net": mom.daily_returns, "spy": spy_ret, "ew_universe": ew_ret}
    print("\n=== summary (OOS, net of costs where noted) ===")
    for name, r in series.items():
        s = bt.summarize(r, name, benchmark=spy_ret if name != "spy" else None)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()})

    print("\n=== per-year returns ===")
    print(per := bt.per_year_table(series).round(4))
    print(f"\navg one-way turnover/rebalance: ml={ml.turnover.mean():.2f} "
          f"mom={mom.turnover.mean():.2f}")

    outdir = root / "data/xsec_results"
    outdir.mkdir(exist_ok=True)
    pd.DataFrame({k: v for k, v in series.items()}).to_csv(outdir / "daily_returns.csv")
    per.to_csv(outdir / "per_year.csv")
    pd.Series({str(k): ",".join(v) for k, v in ml.holdings.items()}).to_csv(
        outdir / "ml_holdings.csv")
    print(f"\nwrote {outdir}/")


if __name__ == "__main__":
    main()
