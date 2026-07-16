"""Live L1 feature construction — the §5.2.1 input spec (F=14).

Features per finalized 5-min bar, each z-scored over a rolling 60-bar
window (causal: window ends at the current bar). Feature list is frozen in
docs/ARCHITECTURE_DECISIONS.md. All computations at index i use bars[<=i] only.
"""

from __future__ import annotations

import math

import numpy as np

from ...data.bar_builder import Bar

FEATURES = [
    "quote_imb", "flow_imb", "rel_spread", "logret_1", "logret_3", "logret_6",
    "logret_12", "vwap_dev", "vol_z", "range_atr", "tradecount_z", "rv_pctile",
    "tod_sin", "tod_cos",
]
N_FEATURES = len(FEATURES)
ZSCORE_WINDOW = 60
ATR_PERIOD = 14


def _safe_log_ret(closes: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros(len(closes))
    if len(closes) > lag:
        out[lag:] = np.log(np.maximum(closes[lag:], 1e-9) / np.maximum(closes[:-lag], 1e-9))
    return out


def _rolling_z(x: np.ndarray, w: int = ZSCORE_WINDOW) -> np.ndarray:
    """Causal rolling z-score; warm-up uses expanding window."""
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        seg = x[lo : i + 1]
        sd = seg.std()
        out[i] = (x[i] - seg.mean()) / sd if sd > 1e-12 else 0.0
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = ATR_PERIOD) -> np.ndarray:
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        lo = max(0, i - period + 1)
        atr[i] = tr[lo : i + 1].mean()
    return np.asarray(np.maximum(atr, 1e-9))


def _rv_percentile(logret1: np.ndarray, rv_window: int = 20, hist: int = 390) -> np.ndarray:
    rv = np.zeros_like(logret1)
    for i in range(len(logret1)):
        lo = max(0, i - rv_window + 1)
        rv[i] = np.sqrt(np.mean(logret1[lo : i + 1] ** 2))
    out = np.zeros_like(rv)
    for i in range(len(rv)):
        lo = max(0, i - hist + 1)
        seg = rv[lo : i + 1]
        out[i] = (seg <= rv[i]).mean()
    return out


def build_features(bars: list[Bar]) -> np.ndarray:
    """[N, 14] feature matrix from finalized 5-min bars (causal throughout)."""
    n = len(bars)
    close = np.array([b.close for b in bars])
    high = np.array([b.high for b in bars])
    low = np.array([b.low for b in bars])
    vol = np.array([b.volume for b in bars], dtype=np.float64)
    vwap = np.array([b.vwap for b in bars])
    tcount = np.array([b.trade_count for b in bars], dtype=np.float64)
    spread = np.array([b.mean_spread for b in bars])
    qimb = np.array([b.mean_quote_imbalance for b in bars])
    fimb = np.array([b.flow_imbalance for b in bars])

    r1 = _safe_log_ret(close, 1)
    atr = _atr(high, low, close)
    tod = np.array([
        (b.ts.hour * 60 + b.ts.minute) / (24 * 60) * 2 * math.pi for b in bars
    ])

    cols = np.stack([
        qimb,
        fimb,
        spread / np.maximum(close, 1e-9),
        r1,
        _safe_log_ret(close, 3),
        _safe_log_ret(close, 6),
        _safe_log_ret(close, 12),
        (close - vwap) / np.maximum(close, 1e-9),
        _rolling_z(vol),
        (high - low) / atr,
        _rolling_z(tcount),
        _rv_percentile(r1),
        np.sin(tod),
        np.cos(tod),
    ], axis=1)

    # z-score the non-bounded columns over the rolling window
    for j in (2, 3, 4, 5, 6, 7, 9):
        cols[:, j] = _rolling_z(cols[:, j])
    assert cols.shape == (n, N_FEATURES)
    return np.asarray(np.nan_to_num(cols, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
