"""Screen the full Alpaca US-equity list and fetch 10y history for the top ~3000.

Screen order matters for bias: the LIQUIDITY screen uses recent data only to
bound the download (a stock must be listed today to be tradable tomorrow, so
"active today" is a deployment constraint, not lookahead) — but the
point-in-time eligibility used in the BACKTEST is recomputed per rebalance
from trailing dollar volume inside the fetched history. ETFs/funds are
excluded by issuer-name heuristics; a few stragglers are harmless.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402
from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.enums import AssetClass, AssetStatus  # noqa: E402
from alpaca.trading.requests import GetAssetsRequest  # noqa: E402

from research.xsec.data import fetch_daily  # noqa: E402

FUND_RE = re.compile(r"ETF|ETN|iShares|SPDR|ProShares|Direxion|VanEck|Invesco|"
                     r"WisdomTree|Xtrackers|Global X|Fund\b", re.IGNORECASE)
TOP_N = 3000
MIN_PRICE = 3.0


def candidate_symbols() -> list[str]:
    tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    assets = tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY,
                                                status=AssetStatus.ACTIVE))
    return sorted(a.symbol for a in assets
                  if a.tradable and a.symbol.isalpha() and len(a.symbol) <= 5
                  and not FUND_RE.search(a.name or ""))


def screen_by_liquidity(symbols: list[str]) -> list[str]:
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    start = datetime.now() - timedelta(days=90)
    stats = {}
    for i in range(0, len(symbols), 200):
        chunk = symbols[i: i + 200]
        try:
            df = dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                start=start, adjustment="all", feed="sip")).df
        except Exception as e:  # a bad symbol shouldn't kill the sweep
            print(f"chunk {i}: {e}", flush=True)
            continue
        if not len(df):
            continue
        g = df.groupby(level=0)
        med_dv = (df["close"] * df["volume"]).groupby(level=0).median()
        last_px = g["close"].last()
        n_days = g.size()
        for s in med_dv.index:
            if last_px[s] >= MIN_PRICE and n_days[s] >= 40:
                stats[s] = float(med_dv[s])
        print(f"screened {i + len(chunk)}/{len(symbols)}, kept {len(stats)}", flush=True)
        time.sleep(0.4)
    top = sorted(stats, key=stats.get, reverse=True)[:TOP_N]
    return sorted(top)


def main() -> None:
    uni_path = ROOT / "data/universe3000.csv"
    if uni_path.exists():
        syms = pd.read_csv(uni_path)["symbol"].tolist()
        print(f"universe cached: {len(syms)}")
    else:
        cands = candidate_symbols()
        print(f"candidates after name filter: {len(cands)}")
        syms = screen_by_liquidity(cands)
        pd.DataFrame({"symbol": syms}).to_csv(uni_path, index=False)
        print(f"screened universe: {len(syms)}")
    if "SPY" not in syms:
        syms = syms + ["SPY"]
    df = fetch_daily(syms, str(ROOT / "data/xsec_daily_3000.pkl"))
    print("history:", df.shape)


if __name__ == "__main__":
    main()
