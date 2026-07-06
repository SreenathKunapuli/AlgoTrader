"""Feature engineering for LOB snapshots.

Two feature modes:

- ``raw40``: the classic DeepLOB input — 10 levels x (ask_p, ask_v, bid_p,
  bid_v), prices expressed relative to mid so the model learns book *shape*,
  not price level.
- ``extended``: raw40 plus engineered microstructure features (per-level
  imbalance, microprice deviation, spread, cumulative depth ratios, OFI,
  signed trade flow, rolling realized volatility). 62 features total.

Normalization is a separate, explicitly fitted step (`Normalizer`) so that
statistics come from the training slice only — fitting on the full series is
lookahead leakage and inflates test accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulator import SimResult

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Raw block helpers
# --------------------------------------------------------------------------- #
def split_book(snapshots: np.ndarray):
    """Return (ask_p, ask_v, bid_p, bid_v), each [N, levels]."""
    return (
        snapshots[:, 0::4],
        snapshots[:, 1::4],
        snapshots[:, 2::4],
        snapshots[:, 3::4],
    )


def mid_price(snapshots: np.ndarray) -> np.ndarray:
    return (snapshots[:, 0] + snapshots[:, 2]) / 2.0


# --------------------------------------------------------------------------- #
# Feature builders
# --------------------------------------------------------------------------- #
def build_raw40(snapshots: np.ndarray) -> np.ndarray:
    """Prices as offsets from mid (in ticks); volumes left raw.

    Keeping the DeepLOB [level x (p,v) pair] column layout intact matters:
    the model's first conv layer has stride 2 over feature columns and
    assumes price/volume adjacency.
    """
    ask_p, ask_v, bid_p, bid_v = split_book(snapshots)
    mid = mid_price(snapshots)[:, None]
    out = np.empty_like(snapshots)
    out[:, 0::4] = ask_p - mid
    out[:, 1::4] = ask_v
    out[:, 2::4] = bid_p - mid
    out[:, 3::4] = bid_v
    return out


def build_extended(
    snapshots: np.ndarray,
    buy_flow: np.ndarray,
    sell_flow: np.ndarray,
    vol_window: int = 50,
) -> tuple[np.ndarray, list[str]]:
    """raw40 + 22 engineered features -> [N, 62], plus feature names."""
    ask_p, ask_v, bid_p, bid_v = split_book(snapshots)
    n_levels = ask_p.shape[1]
    mid = mid_price(snapshots)

    feats: list[np.ndarray] = [build_raw40(snapshots)]
    names: list[str] = []
    for lvl in range(n_levels):
        names += [f"ask_p{lvl+1}", f"ask_v{lvl+1}", f"bid_p{lvl+1}", f"bid_v{lvl+1}"]

    # Per-level volume imbalance: (Vb - Va) / (Vb + Va)  -> 10 features
    imb = (bid_v - ask_v) / (bid_v + ask_v + EPS)
    feats.append(imb)
    names += [f"imb_l{l+1}" for l in range(n_levels)]

    # Spread (ticks) and microprice deviation from mid
    spread = (ask_p[:, 0] - bid_p[:, 0])[:, None]
    micro = (ask_p[:, 0] * bid_v[:, 0] + bid_p[:, 0] * ask_v[:, 0]) / (
        bid_v[:, 0] + ask_v[:, 0] + EPS
    )
    micro_dev = (micro - mid)[:, None]
    feats += [spread, micro_dev]
    names += ["spread", "microprice_dev"]

    # Cumulative depth imbalance at 1, 3, 5, 10 levels
    for k in (1, 3, 5, 10):
        vb = bid_v[:, :k].sum(axis=1)
        va = ask_v[:, :k].sum(axis=1)
        feats.append(((vb - va) / (vb + va + EPS))[:, None])
        names.append(f"cum_imb_{k}")

    # Mid-price return over 1 and 10 snapshots (ticks)
    for lag in (1, 10):
        r = np.zeros_like(mid)
        r[lag:] = mid[lag:] - mid[:-lag]
        feats.append(r[:, None])
        names.append(f"mid_ret_{lag}")

    # Order flow imbalance (Cont-Kukanov-Stoikov) at best level
    ofi = _order_flow_imbalance(bid_p[:, 0], bid_v[:, 0], ask_p[:, 0], ask_v[:, 0])
    feats.append(ofi[:, None])
    names.append("ofi")

    # Signed trade flow and its rolling sum (trade-flow persistence signal)
    signed = (buy_flow - sell_flow)[:, None]
    feats.append(signed)
    names.append("signed_flow")
    feats.append(_rolling_sum(signed[:, 0], 20)[:, None])
    names.append("signed_flow_20")

    # Rolling realized volatility of 1-step mid returns
    r1 = np.zeros_like(mid)
    r1[1:] = np.diff(mid)
    feats.append(_rolling_std(r1, vol_window)[:, None])
    names.append(f"rv_{vol_window}")

    out = np.concatenate(feats, axis=1)
    assert out.shape[1] == len(names), (out.shape, len(names))
    return out, names


def _order_flow_imbalance(bp, bv, ap, av) -> np.ndarray:
    """OFI_t per Cont et al. (2014): contribution of best-quote changes."""
    n = len(bp)
    ofi = np.zeros(n)
    dbp, dbv = np.diff(bp), np.diff(bv)
    dap, dav = np.diff(ap), np.diff(av)
    # bid side contribution
    e_bid = np.where(dbp > 0, bv[1:], np.where(dbp < 0, -bv[:-1], dbv))
    # ask side contribution (sign flipped)
    e_ask = np.where(dap < 0, av[1:], np.where(dap > 0, -av[:-1], dav))
    ofi[1:] = e_bid - e_ask
    return ofi


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([[0.0], x]))
    out = np.empty_like(x)
    out[:w] = c[1 : w + 1]
    out[w:] = c[w + 1 :] - c[1:-w]
    return out


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling std; warm-up region uses expanding window."""
    s1 = _rolling_sum(x, w)
    s2 = _rolling_sum(x * x, w)
    n = np.minimum(np.arange(1, len(x) + 1), w).astype(np.float64)
    var = s2 / n - (s1 / n) ** 2
    return np.sqrt(np.maximum(var, 0.0))


# --------------------------------------------------------------------------- #
# Normalization (fit on train only)
# --------------------------------------------------------------------------- #
@dataclass
class Normalizer:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Normalizer":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std = np.where(self.std < EPS, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Normalizer used before fit()")
        return ((x - self.mean) / self.std).astype(np.float32)


# --------------------------------------------------------------------------- #
# Top-level convenience
# --------------------------------------------------------------------------- #
def make_features(
    result: SimResult, mode: str = "extended"
) -> tuple[np.ndarray, list[str]]:
    if mode == "raw40":
        x = build_raw40(result.snapshots)
        names = []
        for lvl in range(x.shape[1] // 4):
            names += [
                f"ask_p{lvl+1}", f"ask_v{lvl+1}",
                f"bid_p{lvl+1}", f"bid_v{lvl+1}",
            ]
        return x, names
    if mode == "extended":
        return build_extended(result.snapshots, result.buy_flow, result.sell_flow)
    raise ValueError(f"unknown feature mode: {mode}")
