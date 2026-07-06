"""Data adapters: synthetic simulator, FI-2010, and LOBSTER.

All adapters return a `SimResult`-compatible object (snapshots in FI-2010
column order [ask_p1, ask_v1, bid_p1, bid_v1, ...], timestamps, and signed
trade flow), so the rest of the pipeline is source-agnostic.

Swap-in instructions:

FI-2010 (Ntakaris et al.):
    Download `Train_Dst_NoAuction_ZScore_CF_7.txt` etc. from
    https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
    and call `load_fi2010(path)`. Note the public files are already
    z-scored; pass `already_normalized=True` downstream and skip the
    Normalizer fit. FI-2010 has no trade-flow stream, so toxicity filtering
    is disabled (flows returned as zeros).

LOBSTER (https://lobsterdata.com — free samples available):
    Each instrument-day is a pair of CSVs:
      *_orderbook_10.csv : 4*levels columns [ask_p, ask_v, bid_p, bid_v]*L
      *_message_10.csv   : time, type, order_id, size, price, direction
    Call `load_lobster(orderbook_csv, message_csv)`. Prices are in
    dollars*10000 (i.e. hundredths of a cent); we keep them as-is and set
    tick_size accordingly.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .simulator import SimConfig, SimResult, simulate


# --------------------------------------------------------------------------- #
# Cleaning (real-data hygiene)
# --------------------------------------------------------------------------- #
def clean_snapshots(
    result: SimResult,
    max_spread_mult: float = 50.0,
    drop: bool = False,
) -> tuple[SimResult, dict]:
    """Repair anomalous LOB snapshots.

    Real exchange feeds (and, very rarely, the simulator) contain bad
    prints: crossed books (bid >= ask), zero/negative prices, and quotes
    with spreads orders of magnitude wider than normal (a momentary gap
    before liquidity refills). Training on these injects label noise, so we
    forward-fill them from the last good snapshot — the standard fix, since
    a stale-but-valid book is closer to truth than a crossed one.

    Returns the cleaned result and a report of what was touched. With
    `drop=True` the bad rows are removed instead of forward-filled (use only
    when timestamps need not stay contiguous).
    """
    s = result.snapshots
    ask1, bid1 = s[:, 0], s[:, 2]
    spread = ask1 - bid1

    med_spread = np.median(spread[spread > 0]) if np.any(spread > 0) else 1.0
    crossed = bid1 >= ask1
    nonpos = (ask1 <= 0) | (bid1 <= 0)
    wide = spread > max_spread_mult * med_spread
    bad = crossed | nonpos | wide

    report = {
        "n_total": int(len(s)),
        "n_crossed": int(crossed.sum()),
        "n_nonpositive": int(nonpos.sum()),
        "n_wide_spread": int(wide.sum()),
        "n_bad": int(bad.sum()),
        "median_spread": float(med_spread),
    }

    if not bad.any():
        return result, report

    if drop:
        keep = ~bad
        cleaned = dataclasses.replace(
            result,
            snapshots=s[keep],
            timestamps=result.timestamps[keep],
            buy_flow=result.buy_flow[keep],
            sell_flow=result.sell_flow[keep],
        )
        return cleaned, report

    # forward-fill bad rows from the last good snapshot
    new_s = s.copy()
    last_good = None
    for i in range(len(new_s)):
        if bad[i]:
            if last_good is not None:
                new_s[i] = new_s[last_good]
        else:
            last_good = i
    # if the first rows are bad, back-fill from the first good one
    if last_good is not None and bad[0]:
        first_good = int(np.argmax(~bad))
        new_s[: first_good] = new_s[first_good]

    cleaned = dataclasses.replace(result, snapshots=new_s)
    return cleaned, report


# --------------------------------------------------------------------------- #
# Synthetic
# --------------------------------------------------------------------------- #
def load_synthetic(
    n_events: int = 1_000_000,
    snapshot_every: int = 5,
    seed: int = 7,
) -> SimResult:
    return simulate(n_events, snapshot_every, SimConfig(seed=seed))


# --------------------------------------------------------------------------- #
# FI-2010
# --------------------------------------------------------------------------- #
def load_fi2010(path: str, max_cols: int | None = None) -> SimResult:
    """Parse an FI-2010 matrix file.

    File layout: rows are features, columns are time. Rows 0..39 are the
    LOB block in [ask_p, ask_v, bid_p, bid_v] x 10 order — identical to our
    convention. Rows 40..143 are pre-engineered features (ignored; we build
    our own). Rows 144..148 are labels for 5 horizons (ignored; we build
    our own from mid so horizon/alpha stay configurable).
    """
    raw = np.loadtxt(path)
    if max_cols:
        raw = raw[:, :max_cols]
    snapshots = raw[:40].T.astype(np.float64)          # [N, 40]
    n = len(snapshots)
    return SimResult(
        snapshots=snapshots,
        timestamps=np.arange(n, dtype=np.float64),     # event-indexed
        buy_flow=np.zeros(n),                          # FI-2010 has no trades
        sell_flow=np.zeros(n),
        tick_size=0.0001,
    )


# --------------------------------------------------------------------------- #
# LOBSTER
# --------------------------------------------------------------------------- #
def load_lobster(
    orderbook_csv: str,
    message_csv: str,
    levels: int = 10,
) -> SimResult:
    """Parse a LOBSTER orderbook/message file pair into our convention.

    Uses pandas for the read: np.loadtxt on a full-day 10-level orderbook
    (~90 MB, 400k rows) takes minutes; pandas does it in a couple of seconds.
    """
    import pandas as pd

    book = pd.read_csv(orderbook_csv, header=None).to_numpy(dtype=np.float64)
    msg = pd.read_csv(message_csv, header=None).to_numpy(dtype=np.float64)
    assert book.shape[1] >= 4 * levels, (
        f"need {levels} levels, file has {book.shape[1] // 4}"
    )
    assert len(book) == len(msg), "orderbook/message row mismatch"

    # LOBSTER orderbook column order is already [ask_p, ask_v, bid_p, bid_v]
    # per level — identical to our FI-2010 convention.
    snapshots = book[:, : 4 * levels].astype(np.float64)
    times = msg[:, 0]

    # Message type 4/5 = execution of a visible/hidden limit order. The
    # `direction` field is the side of the resting LIMIT order: -1 (a sell
    # limit) being executed means a buyer crossed -> buyer-initiated trade.
    mtype = msg[:, 1].astype(int)
    size = msg[:, 3]
    direction = msg[:, 5].astype(int)
    is_exec = (mtype == 4) | (mtype == 5)
    buy_flow = np.where(is_exec & (direction == -1), size, 0.0)
    sell_flow = np.where(is_exec & (direction == 1), size, 0.0)

    return SimResult(
        snapshots=snapshots,
        timestamps=times,
        buy_flow=buy_flow,
        sell_flow=sell_flow,
        # LOBSTER prices are in 1e-4 dollars (5859400 -> $585.94), so the
        # price->dollar multiplier used for PnL is 1e-4. (The actual trading
        # tick for AAPL is $0.01 = 100 LOBSTER units, but this field is the
        # dollar-conversion factor, not the min increment.)
        tick_size=1e-4,
    )
