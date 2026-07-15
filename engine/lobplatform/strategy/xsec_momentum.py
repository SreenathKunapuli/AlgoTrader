"""xsec_momentum — the validated cross-sectional momentum book, live.

What it does (mirrors research/xsec exactly): on the LAST trading day of
each month, ~30 minutes before close, fetch ~420 calendar days of daily
closes for the S&P universe, drop the bottom 40% by trailing dollar
volume, rank the rest by 12-1 momentum (close[-1-skip] / close[-lookback]
- 1), hold the top N equal-weight at alloc_pct of equity. Every order
passes RiskManager.approve with book="xsec".

Execution (redesigned 2026-07-10 — the original priced limit orders off
the 16-min-delayed SIP daily close, re-pegged once to the same stale
price, then gave up until next month; winners drifting into the close
systematically failed to fill):
  1. SIGNAL from delayed daily closes (fine for a monthly signal);
  2. ORDER PRICES from real-time IEX quotes (fetch_latest_quotes),
     falling back to the daily-close mark when a quote is missing;
  3. up to maker_rounds passive rounds (limit at touch, OrderManager
     re-pegs to mid then expires), reconciling between rounds;
  4. a final URGENT sweep that may cross the spread (the backtest assumes
     fills AT the close — paying ~2-5bps beats not filling);
  5. whatever still failed persists as `pending` targets, retried in the
     same close-window on following sessions;
  6. a month-end missed entirely (engine down) is caught up on the next
     session via calendar.last_completed_month_end.

Book separation invariants (enforced here + engine guards + risk checks):
- xsec positions carry Position.book == "xsec": the EOD flattener and the
  intraday signal-exit path skip them; ATR stops don't apply (monthly
  momentum lives or dies at rebalance; the account floor is the backstop).
- Symbols currently held by the intraday book are excluded from selection
  (next-ranked name takes the slot) — one position per symbol, one owner.
- Holdings persist to a JSON file from FILLS (broker truth), not intents;
  after any broker reconcile the retag loop restores book tags within one
  scheduler tick.

Circuit breaker: when the book's within-cycle drawdown vs cost basis
exceeds book_dd_halt_pct (set BEYOND the worst backtest month — see
config/xsec.py), state.xsec_buys_halted flips and RiskManager rejects new
xsec buys. Sells always pass: de-risk, never panic-liquidate.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import structlog

from ..config.settings import Settings
from ..config.xsec import XSEC, XsecConfig
from ..data import calendar
from ..data.history import fetch_daily_history, fetch_latest_closes, fetch_latest_quotes
from ..execution.order_manager import OrderManager
from ..persistence.repo import Repo
from ..pubsub import PubSub
from ..risk.risk_manager import OrderIntent, Rejection, RiskManager
from ..risk.state import PortfolioState

log = structlog.get_logger()

BOOK_FILE = "data/xsec_book.json"
MAKER_ROUND_WAIT_S = 75.0   # OrderManager: 30s re-peg + 30s expire + margin
SWEEP_WAIT_S = 45.0         # urgent: 30s then market-replace + margin


def momentum_ranks(history: dict[str, dict[str, Any]], cfg: XsecConfig = XSEC,
                   ) -> list[tuple[str, float]]:
    """[(symbol, mom_12_1)] sorted best-first, liquidity-filtered.

    Same math as research/xsec/features.py mom_12_1 at the panel's last row:
    close[-1-skip] / close[-lookback] - 1, requiring >= lookback closes.
    """
    eligible = {s: h for s, h in history.items()
                if len(h["closes"]) >= cfg.lookback_days and h["dollar_vol"] > 0}
    if not eligible:
        return []
    by_dv = sorted(eligible, key=lambda s: eligible[s]["dollar_vol"], reverse=True)
    keep = set(by_dv[: max(1, int(len(by_dv) * cfg.liquidity_keep))])
    scores = []
    for s in keep:
        c = eligible[s]["closes"]
        mom = c[-1 - cfg.skip_days] / c[-cfg.lookback_days] - 1.0
        scores.append((s, float(mom)))
    return sorted(scores, key=lambda x: x[1], reverse=True)


class XsecMomentumStrategy:
    def __init__(self, settings: Settings, state: PortfolioState,
                 risk: RiskManager, om: OrderManager, repo: Repo,
                 pubsub: PubSub, cfg: XsecConfig = XSEC,
                 book_file: str = BOOK_FILE) -> None:
        self.settings = settings
        self.state = state
        self.risk = risk
        self.om = om
        self.repo = repo
        self.pubsub = pubsub
        self.cfg = cfg
        self.book_file = Path(book_file)
        self.holdings: dict[str, int] = {}
        self.pending: dict[str, int] = {}   # unmet targets, retried next session
        self.last_rebalance_month: str = ""
        self._last_mark_refresh: datetime = datetime.min.replace(tzinfo=UTC)
        self._marks: dict[str, float] = {}  # daily-close fallback for quotes
        self._load_book()

    # ---------------- persistence ---------------- #
    def _load_book(self) -> None:
        if self.book_file.exists():
            data = json.loads(self.book_file.read_text())
            self.holdings = {k: int(v) for k, v in data.get("holdings", {}).items()}
            self.pending = {k: int(v) for k, v in data.get("pending", {}).items()}
            self.last_rebalance_month = data.get("last_rebalance_month", "")

    def _save_book(self) -> None:
        self.book_file.parent.mkdir(parents=True, exist_ok=True)
        self.book_file.write_text(json.dumps({
            "holdings": self.holdings,
            "pending": self.pending,
            "last_rebalance_month": self.last_rebalance_month}, indent=1))

    # ---------------- book tagging ---------------- #
    def retag(self) -> set[str]:
        """Restore book="xsec" on state positions after any reconcile; sync
        holdings quantities to broker truth. Returns the xsec symbol set
        (engine guards read it)."""
        for sym in list(self.holdings):
            pos = self.state.positions.get(sym)
            if pos is None:
                continue  # order may not have filled yet; keep intent
            pos.book = "xsec"
            pos.stop_price = None  # no ATR stops on this book
            # Initialise xsec_qty only when no intraday addon has been tracked yet.
            # Preserves the addon layer if retag runs mid-session during rebalance.
            if pos.xsec_qty == 0:
                pos.xsec_qty = pos.qty
            self.holdings[sym] = pos.xsec_qty   # holdings = xsec-owned shares only
        return set(self.holdings)

    # ---------------- universe ---------------- #
    def universe(self) -> list[str]:
        import csv
        path = Path(self.cfg.universe_csv)
        if not path.exists():
            log.error("xsec.universe_missing", path=str(path))
            return []
        with open(path) as f:
            return sorted({row["Symbol"].strip() for row in csv.DictReader(f)})

    # ---------------- targets ---------------- #
    def compute_targets(self, history: dict[str, dict[str, Any]],
                        equity: float) -> dict[str, int]:
        """Equal-weight top-N target quantities, excluding symbols the
        intraday book currently holds."""
        intraday_held = set(self.state.book_positions("intraday"))
        ranked = [(s, m) for s, m in momentum_ranks(history, self.cfg)
                  if s not in intraday_held]
        picks = ranked[: self.cfg.top_n]
        if not picks:
            return {}
        budget = self.cfg.alloc_pct * equity / len(picks)
        targets: dict[str, int] = {}
        for sym, _mom in picks:
            price = history[sym]["closes"][-1]
            qty = int(budget / price)
            if qty > 0:
                targets[sym] = qty
        return targets

    # ---------------- execution ---------------- #
    def _deltas(self, targets: dict[str, int]) -> dict[str, int]:
        """Signed share deltas needed to reach `targets` from broker truth.
        Symbols in targets only — a full rebalance passes explicit 0s for
        names to exit."""
        current = {s: p.qty for s, p in self.state.book_positions("xsec").items()}
        out: dict[str, int] = {}
        for sym, tgt in targets.items():
            delta = tgt - current.get(sym, 0)
            if delta != 0:
                out[sym] = delta
        return out

    async def _quotes_for(self, symbols: list[str]) -> dict[str, tuple[float, float]]:
        try:
            return await asyncio.to_thread(
                fetch_latest_quotes, self.settings.alpaca_api_key,
                self.settings.alpaca_secret_key, symbols)
        except Exception as exc:
            log.warning("xsec.quotes_failed", error=str(exc))
            return {}

    async def _submit_deltas(self, deltas: dict[str, int], now: datetime,
                             urgent: bool) -> int:
        """One submission pass over signed deltas: sells first (frees cash
        and book headroom), live-quote pricing with daily-close fallback."""
        quotes = await self._quotes_for(sorted(deltas))
        ordered = sorted(deltas.items(), key=lambda kv: kv[1])  # sells first
        submitted = 0
        for sym, delta in ordered:
            side: Literal["buy", "sell"] = "buy" if delta > 0 else "sell"
            bid_ask = quotes.get(sym)
            if bid_ask:
                mid = (bid_ask[0] + bid_ask[1]) / 2.0
                spread = bid_ask[1] - bid_ask[0]
            else:
                mid, spread = self._marks.get(sym, 0.0), 0.0
            if mid <= 0:
                log.warning("xsec.no_price", symbol=sym)
                continue
            intent = OrderIntent(symbol=sym, side=side, qty=abs(delta),
                                 price_hint=mid, reason="xsec_rebalance",
                                 book="xsec")
            approval = self.risk.approve(intent, now, urgent=urgent)
            if isinstance(approval, Rejection):
                self.repo.add_rejection(approval.reason, intent.as_dict())
                log.info("xsec.rejected", symbol=sym, reason=approval.reason)
                continue
            try:
                await self.om.submit(approval, now, mid=mid, spread=spread,
                                     strategy="xsec")
                submitted += 1
            except Exception as exc:
                log.error("xsec.submit_failed", symbol=sym, error=str(exc))
        return submitted

    async def _reconcile_and_retag(self) -> None:
        try:
            await self.om.reconcile_state()
        except Exception as exc:
            log.error("xsec.reconcile_failed", error=str(exc))
        self.retag()

    async def _execute_toward(self, targets: dict[str, int]) -> dict[str, int]:
        """Drive the book toward `targets`: passive rounds, then an urgent
        sweep. Returns the still-unmet portion of `targets`."""
        for rnd in range(self.cfg.maker_rounds):
            now = datetime.now(UTC)
            deltas = self._deltas(targets)
            if not deltas:
                return {}
            if not calendar.in_entry_window(now):
                break  # buys would be rejected; leave the rest to pending
            log.info("xsec.maker_round", round=rnd + 1, orders=len(deltas))
            await self._submit_deltas(deltas, now, urgent=False)
            await asyncio.sleep(MAKER_ROUND_WAIT_S)
            await self._reconcile_and_retag()

        deltas = self._deltas(targets)
        now = datetime.now(UTC)
        if deltas and self.cfg.market_fallback and calendar.in_entry_window(now):
            log.info("xsec.urgent_sweep", orders=len(deltas))
            await self._submit_deltas(deltas, now, urgent=True)
            await asyncio.sleep(SWEEP_WAIT_S)
            await self._reconcile_and_retag()
            deltas = self._deltas(targets)
        return {s: targets[s] for s in deltas}

    # ---------------- rebalance ---------------- #
    async def rebalance(self, now: datetime, target_month: str) -> None:
        syms = self.universe()
        if not syms:
            return
        log.info("xsec.rebalance_start", universe=len(syms), month=target_month)
        history = await asyncio.to_thread(
            fetch_daily_history, self.settings.alpaca_api_key,
            self.settings.alpaca_secret_key, syms, self.cfg.history_days)
        targets = self.compute_targets(history, self.state.equity)
        if not targets:
            log.warning("xsec.no_targets")
            return
        self._marks = {s: history[s]["closes"][-1] for s in history}

        # exits for names held but no longer selected are explicit 0-targets
        held = {s: p.qty for s, p in self.state.book_positions("xsec").items()}
        full_targets = {**{s: 0 for s in held if s not in targets}, **targets}

        # while executing, tag-ownership covers old holdings AND new intents
        self.holdings = {**held, **targets}

        unmet = await self._execute_toward(full_targets)

        # holdings from FILLS (broker truth after reconcile), never intents;
        # zombie sells (position remains, target 0) stay owned + pending
        actual = {s: p.qty for s, p in self.state.book_positions("xsec").items()
                  if p.qty != 0}
        self.holdings = actual
        self.pending = unmet
        self.last_rebalance_month = target_month
        self._save_book()
        log.info("xsec.rebalance_done", targets=len(targets),
                 filled=len([s for s in targets if s in actual]),
                 pending=len(unmet))
        await self.pubsub.publish("xsec", {
            "ts": now.isoformat(), "targets": targets,
            "filled": actual, "pending": unmet})

    async def retry_pending(self, now: datetime) -> None:
        """Re-attempt unmet targets from the last rebalance in the same
        close-window on a later session."""
        log.info("xsec.retry_pending", orders=len(self.pending))
        unmet = await self._execute_toward(dict(self.pending))
        actual = {s: p.qty for s, p in self.state.book_positions("xsec").items()
                  if p.qty != 0}
        self.holdings = actual
        self.pending = unmet
        self._save_book()

    # ---------------- marks + circuit breaker ---------------- #
    async def refresh_marks(self, now: datetime) -> None:
        held = list(self.state.book_positions("xsec"))
        if not held:
            return
        closes = await asyncio.to_thread(
            fetch_latest_closes, self.settings.alpaca_api_key,
            self.settings.alpaca_secret_key, held)
        for sym, px in closes.items():
            pos = self.state.positions.get(sym)
            if pos and pos.book == "xsec":
                pos.mark = px
        self._last_mark_refresh = now

    def update_breaker(self) -> None:
        """Halt NEW xsec buys when the within-cycle book drawdown (marks vs
        cost basis) breaches book_dd_halt_pct; auto-clears when it recovers.
        Sells are never blocked."""
        basis = self.state.book_cost_basis("xsec")
        upnl = self.state.book_unrealized("xsec")
        dd = (-upnl / basis) if basis > 0 and upnl < 0 else 0.0
        breached = dd >= self.cfg.book_dd_halt_pct
        if breached and not self.state.xsec_buys_halted:
            log.critical("xsec.breaker_tripped", book_dd=round(dd, 4),
                         limit=self.cfg.book_dd_halt_pct)
        elif not breached and self.state.xsec_buys_halted:
            log.info("xsec.breaker_cleared", book_dd=round(dd, 4))
        self.state.xsec_buys_halted = breached

    # ---------------- scheduler ---------------- #
    def _in_exec_window(self, now: datetime) -> bool:
        close = calendar.session_close(now)
        if close is None:
            return False
        start = close.timestamp() - self.cfg.rebalance_before_close_min * 60
        return now.timestamp() >= start and calendar.in_entry_window(now)

    def should_rebalance(self, now: datetime) -> bool:
        if self.state.halted:
            return False
        if now.strftime("%Y-%m") == self.last_rebalance_month:
            return False
        if not calendar.is_last_session_of_month(now):
            return False
        return self._in_exec_window(now)

    def missed_month(self, now: datetime) -> str | None:
        """The month of a month-end the engine slept through, if any."""
        if not self.cfg.catch_up_missed or self.state.halted:
            return None
        me = calendar.last_completed_month_end(now)
        if me is None:
            return None
        month = me.strftime("%Y-%m")
        return month if month > self.last_rebalance_month else None

    async def run(self) -> None:
        """Engine background task: tag books, refresh marks, run the
        breaker, fire monthly (or catch up / retry shortfalls)."""
        log.info("xsec.strategy_started", holdings=len(self.holdings),
                 pending=len(self.pending))
        while True:
            await asyncio.sleep(30)
            now = datetime.now(UTC)
            self.retag()
            self.update_breaker()
            if not calendar.is_session_open(now):
                continue
            if (now - self._last_mark_refresh).total_seconds() > 3600:
                try:
                    await self.refresh_marks(now)
                except Exception as exc:
                    log.error("xsec.mark_refresh_failed", error=str(exc))
            try:
                if self.should_rebalance(now):
                    await self.rebalance(now, now.strftime("%Y-%m"))
                elif (month := self.missed_month(now)) and self._in_exec_window(now):
                    await self.rebalance(now, month)
                elif self.pending and self._in_exec_window(now):
                    await self.retry_pending(now)
            except Exception as exc:
                log.error("xsec.rebalance_failed", error=str(exc))
