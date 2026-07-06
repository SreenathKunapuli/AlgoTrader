"""Event-driven limit order book simulator.

Implements a Cont-Stoikov-Talreja style zero-intelligence book with two
realism upgrades that matter for downstream ML:

1. Market order arrivals follow a Hawkes process (self- and cross-excitation),
   producing the order-flow clustering real feeds exhibit. Without this,
   labels are too easy and models overfit to Poisson-flat flow.
2. Base intensities drift via a slow lognormal random walk, creating
   non-stationary regimes inside a single session.

The book is volume-aggregated per price level (no individual order IDs);
that is sufficient for snapshot-based prediction and orders of magnitude
faster than full price-time priority matching.

Output convention follows FI-2010 / DeepLOB: each snapshot row is
    [ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ask_v2, bid_p2, bid_v2, ...]
for `depth` levels, prices in ticks.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimConfig:
    tick_size: float = 0.01
    initial_mid_ticks: int = 10_000          # $100.00 at 1-cent ticks
    depth: int = 10                          # levels per side in snapshots
    book_depth: int = 30                     # levels we actively model

    # Limit order arrivals: rate at distance d (ticks) from opposite best
    # lambda_L(d) = limit_base / d**limit_decay, scaled by a spread-
    # responsive multiplier (queue-reactive flavour): when the spread is
    # wide, liquidity providers quote more aggressively. This is the
    # restoring force that keeps the book stable — without it the system
    # is bistable between an ever-thickening calm book and a hollowed-out
    # exploding one.
    limit_base: float = 1.2
    limit_decay: float = 0.6
    spread_response: float = 1.0      # multiplier slope per tick of spread
    spread_response_cap: float = 30.0 # cap on the multiplier

    # Cancellations: rate per unit volume at distance d. Calibrated so the
    # book volume is roughly stationary: too low and levels thicken until
    # the mid freezes, too high and the book hollows out and explodes.
    cancel_base: float = 0.03
    cancel_decay: float = 0.3

    # Market orders: Hawkes. mu = baseline, alpha = jump per event,
    # kappa = decay rate, cross = fraction of excitation sent to other side.
    # Keep alpha/kappa well below 1: near-critical branching produces
    # heavy-tailed bursts that march the book hundreds of ticks. exc_max
    # hard-caps the excitation state as a second line of defence.
    market_mu: float = 0.80
    hawkes_alpha: float = 0.60
    hawkes_kappa: float = 1.3
    hawkes_cross: float = 0.15
    hawkes_exc_max: float = 2.5

    # Fundamental anchor: a slow random walk representing value traders.
    # Market buy/sell baseline intensities tilt against displacement from
    # the anchor (exp(-beta * displacement)), which bounds the speed of
    # directional excursions. Without it, momentum bursts outrun the
    # book's ability to thicken behind the move, leaving hollow price
    # ranges that reversals teleport through (flash-crash artifacts).
    anchor_sigma: float = 0.02        # anchor RW step per event, ticks
    anchor_beta: float = 0.03         # reversion strength per tick displaced
    anchor_tilt_max: float = 3.0      # cap on the intensity multiplier

    # Order sizes ~ lognormal, rounded up to >= 1
    size_mean_log: float = 4.3               # median ~74 shares
    size_sigma_log: float = 0.8
    market_size_mult: float = 2.0            # market orders run larger
    # Price protection (LULD-style): a marketable order only executes
    # within this many ticks of a slow EMA of the mid; the rest is
    # cancelled. The reference MUST be a trailing average, not the
    # pre-trade best: per-order bands let consecutive orders leapfrog 8
    # ticks each and cascade through the whole book; a trailing-average
    # band caps the excursion speed of the price itself, which is exactly
    # what real limit-up/limit-down does.
    price_band_ticks: int = 25
    band_ema_alpha: float = 0.002     # EMA weight per event (~500-event window)

    # Regime drift: per-event lognormal random-walk step on base intensities
    regime_sigma: float = 0.0004
    regime_clip: tuple[float, float] = (0.4, 2.5)

    # Book seeding / replenishment
    seed_volume: float = 150.0
    replenish_min_levels: int = 15

    seed: int = 7


@dataclass
class SimResult:
    snapshots: np.ndarray      # [N, 4*depth] float64, FI-2010 column order
    timestamps: np.ndarray     # [N] event-time seconds
    buy_flow: np.ndarray       # [N] market-buy volume since previous snapshot
    sell_flow: np.ndarray      # [N] market-sell volume since previous snapshot
    tick_size: float = 0.01

    @property
    def mid(self) -> np.ndarray:
        return (self.snapshots[:, 0] + self.snapshots[:, 2]) / 2.0

    @property
    def spread(self) -> np.ndarray:
        return self.snapshots[:, 0] - self.snapshots[:, 2]


class _BookSide:
    """One side of the book: price level -> volume, with sorted price index."""

    __slots__ = ("is_bid", "vols", "prices")

    def __init__(self, is_bid: bool):
        self.is_bid = is_bid
        self.vols: dict[int, float] = {}
        self.prices: list[int] = []      # always sorted ascending

    def best(self) -> int:
        return self.prices[-1] if self.is_bid else self.prices[0]

    def add(self, price: int, vol: float) -> None:
        if price in self.vols:
            self.vols[price] += vol
        else:
            self.vols[price] = vol
            bisect.insort(self.prices, price)

    def remove(self, price: int, vol: float) -> float:
        """Remove up to `vol`; returns volume actually removed."""
        have = self.vols.get(price, 0.0)
        taken = min(have, vol)
        left = have - taken
        if left <= 1e-9:
            if price in self.vols:
                del self.vols[price]
                idx = bisect.bisect_left(self.prices, price)
                if idx < len(self.prices) and self.prices[idx] == price:
                    self.prices.pop(idx)
        else:
            self.vols[price] = left
        return taken

    def top_levels(self, n: int) -> list[tuple[int, float]]:
        if self.is_bid:
            sel = self.prices[-1 : -n - 1 : -1]
        else:
            sel = self.prices[:n]
        return [(p, self.vols[p]) for p in sel]

    def n_levels(self) -> int:
        return len(self.prices)


class LOBSimulator:
    def __init__(self, config: SimConfig | None = None):
        self.cfg = config or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------ #
    def run(self, n_events: int, snapshot_every: int = 5) -> SimResult:
        cfg = self.cfg
        rng = self.rng
        D = cfg.book_depth

        bids = _BookSide(is_bid=True)
        asks = _BookSide(is_bid=False)
        self._seed_book(bids, asks)

        # Precompute distance-dependent rates for d = 1..D
        dists = np.arange(1, D + 1, dtype=np.float64)
        limit_rates = cfg.limit_base / dists**cfg.limit_decay
        cancel_unit = cfg.cancel_base / dists**cfg.cancel_decay

        # Hawkes excitation state (excess intensity above baseline)
        exc_buy = 0.0
        exc_sell = 0.0

        anchor = float(cfg.initial_mid_ticks)
        ema_mid = float(cfg.initial_mid_ticks)
        regime = 1.0
        t = 0.0
        buy_since, sell_since = 0.0, 0.0

        n_snaps = n_events // snapshot_every
        snaps = np.empty((n_snaps, 4 * cfg.depth), dtype=np.float64)
        times = np.empty(n_snaps, dtype=np.float64)
        bflow = np.empty(n_snaps, dtype=np.float64)
        sflow = np.empty(n_snaps, dtype=np.float64)
        snap_i = 0

        # Event layout in the categorical draw:
        #   [0:D)        limit buy at distance d+1 from best ask
        #   [D:2D)       limit sell at distance d+1 from best bid
        #   [2D:3D)      cancel buy at level index d (from best bid)
        #   [3D:4D)      cancel sell at level index d (from best ask)
        #   4D           market buy
        #   4D+1         market sell
        rates = np.empty(4 * D + 2, dtype=np.float64)

        for ev in range(n_events):
            # --- regime drift ----------------------------------------- #
            regime *= math.exp(cfg.regime_sigma * rng.standard_normal())
            regime = min(max(regime, cfg.regime_clip[0]), cfg.regime_clip[1])

            # --- assemble current event rates -------------------------- #
            spread = asks.best() - bids.best()
            liq_mult = min(
                1.0 + cfg.spread_response * (spread - 1),
                cfg.spread_response_cap,
            )
            # Regime must scale limit, cancel AND market flows uniformly:
            # the book's equilibrium volume is inflow/cancel-rate, so
            # scaling arrivals without cancels thins or thickens the book
            # over slow regime excursions until it destabilizes. Uniform
            # scaling varies activity (wall-clock volatility clustering)
            # while leaving book-shape dynamics invariant.
            rates[0:D] = limit_rates * regime * liq_mult
            rates[D : 2 * D] = limit_rates * regime * liq_mult

            bid_top = bids.top_levels(D)
            ask_top = asks.top_levels(D)
            for i in range(D):
                rates[2 * D + i] = (
                    bid_top[i][1] * cancel_unit[i] * regime
                    if i < len(bid_top) else 0.0
                )
                rates[3 * D + i] = (
                    ask_top[i][1] * cancel_unit[i] * regime
                    if i < len(ask_top) else 0.0
                )

            anchor += cfg.anchor_sigma * rng.standard_normal()
            cur_mid = (bids.best() + asks.best()) / 2.0
            ema_mid += cfg.band_ema_alpha * (cur_mid - ema_mid)
            disp = cur_mid - anchor
            tilt = math.exp(-cfg.anchor_beta * disp)
            tilt = min(max(tilt, 1.0 / cfg.anchor_tilt_max), cfg.anchor_tilt_max)
            rates[4 * D] = cfg.market_mu * regime * tilt + exc_buy
            rates[4 * D + 1] = cfg.market_mu * regime / tilt + exc_sell

            total = rates.sum()
            dt = rng.exponential(1.0 / total)
            t += dt

            decay = math.exp(-cfg.hawkes_kappa * dt)
            exc_buy *= decay
            exc_sell *= decay

            # --- draw and apply event ---------------------------------- #
            k = rng.choice(4 * D + 2, p=rates / total)
            size = self._draw_size(rng)
            best_bid = bids.best()
            best_ask = asks.best()

            # Limit placement: a depth ladder anchored at the touch. The
            # ladder TOP for bids is best_ask - 1 (improvement allowed),
            # capped by the EMA band so a dislocated opposite quote cannot
            # drag placements far from fair value; depth k stacks downward
            # from there. This keeps the 30 ticks behind the touch dense
            # (every arrival feeds the ladder), which is what prevents
            # transient eat-throughs to the replenishment backstop.
            if k < D:                                   # limit buy
                top = min(best_ask - 1, int(ema_mid) + cfg.price_band_ticks)
                bids.add(top - k, size)
            elif k < 2 * D:                             # limit sell
                bottom = max(best_bid + 1, int(ema_mid) - cfg.price_band_ticks)
                asks.add(bottom + (k - D), size)
            elif k < 3 * D:                             # cancel buy
                i = k - 2 * D
                if i < len(bid_top):
                    bids.remove(bid_top[i][0], size)
            elif k < 4 * D:                             # cancel sell
                i = k - 3 * D
                if i < len(ask_top):
                    asks.remove(ask_top[i][0], size)
            elif k == 4 * D:                            # market buy
                msize = size * cfg.market_size_mult
                buy_since += self._execute_market(asks, msize, ema_mid)
                exc_buy = min(exc_buy + cfg.hawkes_alpha, cfg.hawkes_exc_max)
                exc_sell = min(
                    exc_sell + cfg.hawkes_alpha * cfg.hawkes_cross,
                    cfg.hawkes_exc_max,
                )
            else:                                       # market sell
                msize = size * cfg.market_size_mult
                sell_since += self._execute_market(bids, msize, ema_mid)
                exc_sell = min(exc_sell + cfg.hawkes_alpha, cfg.hawkes_exc_max)
                exc_buy = min(
                    exc_buy + cfg.hawkes_alpha * cfg.hawkes_cross,
                    cfg.hawkes_exc_max,
                )

            self._replenish(bids, asks)

            # --- snapshot ----------------------------------------------#
            if (ev + 1) % snapshot_every == 0 and snap_i < n_snaps:
                row = snaps[snap_i]
                a = asks.top_levels(cfg.depth)
                b = bids.top_levels(cfg.depth)
                for lvl in range(cfg.depth):
                    row[4 * lvl + 0] = a[lvl][0]
                    row[4 * lvl + 1] = a[lvl][1]
                    row[4 * lvl + 2] = b[lvl][0]
                    row[4 * lvl + 3] = b[lvl][1]
                times[snap_i] = t
                bflow[snap_i] = buy_since
                sflow[snap_i] = sell_since
                buy_since, sell_since = 0.0, 0.0
                snap_i += 1

        return SimResult(
            snapshots=snaps,
            timestamps=times,
            buy_flow=bflow,
            sell_flow=sflow,
            tick_size=cfg.tick_size,
        )

    # ------------------------------------------------------------------ #
    def _seed_book(self, bids: _BookSide, asks: _BookSide) -> None:
        cfg = self.cfg
        mid = cfg.initial_mid_ticks
        for d in range(1, cfg.book_depth + 1):
            vol = cfg.seed_volume * (1.0 + 0.3 * self.rng.standard_normal())
            bids.add(mid - d, max(vol, 50.0))
            vol = cfg.seed_volume * (1.0 + 0.3 * self.rng.standard_normal())
            asks.add(mid + d, max(vol, 50.0))

    def _draw_size(self, rng: np.random.Generator) -> float:
        return float(
            max(1.0, round(rng.lognormal(self.cfg.size_mean_log, self.cfg.size_sigma_log)))
        )

    def _execute_market(self, side: _BookSide, size: float, ref: float) -> float:
        """Fill within the LULD band around `ref`; cancel the rest.

        Returns the volume actually executed (only executed volume counts
        as trade flow — band-cancelled remainders never printed).
        """
        remaining = size
        band = self.cfg.price_band_ticks
        while remaining > 1e-9 and side.n_levels() > 1:
            best = side.best()
            if abs(best - ref) > band:
                break                        # price protection kicks in
            taken = side.remove(best, remaining)
            remaining -= taken
            if taken <= 1e-9:
                break
        return size - remaining

    def _replenish(self, bids: _BookSide, asks: _BookSide) -> None:
        """Keep a dense, contiguous ladder of `replenish_min_levels` right
        behind each best quote.

        The earlier version only extended the deep tail, so a burst that ate
        the near levels faster than limit flow refilled them left the best
        quote stranded on a far backstop level — a one-snapshot spread
        flicker. Filling contiguously from the touch outward models a market
        maker maintaining depth where trading actually happens and removes
        those gaps at the source.
        """
        cfg = self.cfg
        for side in (bids, asks):
            if side.n_levels() == 0:
                raise RuntimeError("book side emptied — increase seed volume")
            best = side.best()
            step = -1 if side.is_bid else 1
            for i in range(cfg.replenish_min_levels):
                price = best + step * i
                if price not in side.vols:
                    vol = cfg.seed_volume * (1.0 + 0.3 * self.rng.standard_normal())
                    side.add(price, max(vol, 50.0))


def simulate(n_events: int = 500_000, snapshot_every: int = 5,
             config: SimConfig | None = None) -> SimResult:
    return LOBSimulator(config).run(n_events, snapshot_every)
