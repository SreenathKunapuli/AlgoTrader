"""Cross-sectional features and forward-return labels on daily panels.

All features at rebalance date t use only data up to and including t; the
label is the NEXT 21 trading days' total return. Both are rank-transformed
to [0, 1] within each date's ELIGIBLE set — cross-sectional ML predicts
relative ordering, not levels, which sidesteps market-direction
non-stationarity (the killer of the intraday models).

The set is the documented cross-sectional canon, not invented factors:
momentum at three horizons (Jegadeesh-Titman), short-term reversal (Jegadeesh
1990), realized/idiosyncratic vol (Ang et al.), skewness and the MAX lottery
effect (Bali-Cakici-Whitelaw), Amihud illiquidity, 52-week high
(George-Hwang), trend consistency, size-via-dollar-volume, and market beta.
With ~10x the names of the S&P-only run, learned interactions between these
get their first real chance against the plain momentum baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 21  # forward trading days per holding period (~1 month)

# price/volume-only features (computable without a market series)
FEATURES = ["mom_12_1", "mom_6_1", "mom_3_1", "rev_1m", "rev_5d",
            "vol_3m", "vol_12m", "skew_3m", "max_ret_1m", "amihud_3m",
            "up_frac_6m", "dist_52w_high", "dollar_vol"]
# require a market (SPY) series
MARKET_FEATURES = ["beta_12m", "idio_vol_3m"]
ALL_FEATURES = FEATURES + MARKET_FEATURES


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month in the panel."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def compute_features(close: pd.DataFrame, volume: pd.DataFrame,
                     market: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    """Raw (un-ranked) feature panels [date x symbol], NaN where history short."""
    ret = close.pct_change()
    dollar = (close * volume).replace(0.0, np.nan)
    feats = {
        "mom_12_1": close.shift(21) / close.shift(252) - 1.0,
        "mom_6_1": close.shift(21) / close.shift(126) - 1.0,
        "mom_3_1": close.shift(21) / close.shift(63) - 1.0,
        "rev_1m": close / close.shift(21) - 1.0,
        "rev_5d": close / close.shift(5) - 1.0,
        "vol_3m": ret.rolling(63).std(),
        "vol_12m": ret.rolling(252).std(),
        "skew_3m": ret.rolling(63).skew(),
        "max_ret_1m": ret.rolling(21).max(),
        "amihud_3m": (ret.abs() / dollar).rolling(63, min_periods=40).mean(),
        "up_frac_6m": (ret > 0).rolling(126).mean(),
        "dist_52w_high": close / close.rolling(252).max() - 1.0,
        "dollar_vol": np.log(dollar.rolling(63).mean()),
    }
    if market is not None:
        mret = market.reindex(close.index).pct_change()
        cov = ret.rolling(252).cov(mret)
        beta = cov.div(mret.rolling(252).var(), axis=0)
        feats["beta_12m"] = beta
        resid = ret.sub(beta.mul(mret, axis=0))
        feats["idio_vol_3m"] = resid.rolling(63).std()
    return feats


def forward_return(close: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Total return over the NEXT `horizon` trading days (close is adjusted)."""
    return close.shift(-horizon) / close - 1.0


def cross_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank each row to (0, 1]; NaNs stay NaN. Robust to fat-tailed features."""
    return panel.rank(axis=1, pct=True)


def build_dataset(close: pd.DataFrame, volume: pd.DataFrame,
                  dates: pd.DatetimeIndex, market: pd.Series | None = None,
                  eligible_top: int | None = None) -> pd.DataFrame:
    """Stacked samples at rebalance dates: feature cols + fwd_ret + y_rank.

    A sample requires all features present (≥252d listed history). When
    `eligible_top` is set, each date keeps only the top-N names by trailing
    dollar volume AT THAT DATE (point-in-time liquidity screen) and ranks —
    features and label — within that eligible set. fwd_ret is NaN on
    trailing dates: usable for prediction, excluded from training.
    """
    feats = compute_features(close, volume, market)
    names = list(feats.keys())
    fwd = forward_return(close)
    rows = []
    for d in dates:
        if d not in close.index:
            continue
        block = pd.DataFrame({k: v.loc[d] for k, v in feats.items()}).dropna()
        if block.empty:
            continue
        if eligible_top is not None and len(block) > eligible_top:
            block = block.nlargest(eligible_top, "dollar_vol")
        block[names] = block[names].rank(pct=True)
        block["fwd_ret"] = fwd.loc[d].reindex(block.index)
        block["y_rank"] = block["fwd_ret"].rank(pct=True)
        block["date"] = d
        block["symbol"] = block.index
        rows.append(block)
    return pd.concat(rows, ignore_index=True)
