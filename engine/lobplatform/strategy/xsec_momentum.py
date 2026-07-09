"""xsec_momentum — the validated cross-sectional momentum book, live.

What it does (mirrors research/xsec exactly): on the LAST trading day of
each month, ~20 minutes before close, fetch ~420 calendar days of daily
closes for the S&P universe, drop the bottom 40% by trailing dollar
volume, rank the rest by 12-1 momentum (close[-1-skip] / close[-lookback]
- 1), hold the top N equal-weight at alloc_pct of equity. Maker-only
entries via the standard OrderManager path; every order passes
RiskManager.approve with book="xsec".

Book separation invariants (enforced here + engine guards + risk checks):
- xsec positions carry Position.book == "xsec": the EOD flattener and the
  intraday signal-exit path skip them; ATR stops don't apply (monthly
  momentum lives or dies at rebalance, the kill switch is the backstop).
- Symbols currently held by the intraday book are excluded from selection
  (next-ranked name takes the slot) — one position per symbol, one owner.
- Holdings persist to a JSON file; after any broker reconcile (startup,
  wake) the retag loop restores book tags within one scheduler tick.

Fills: there is no trade-update stream in this build — the position mirror
is maintained by broker reconciles. After submitting rebalance orders this
strategy schedules its own reconcile+retag pass so the book reflects fills
within ~3 minutes.
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
from ..data.history import fetch_daily_history, fetch_latest_closes
from ..execution.order_manager import OrderManager
from ..persistence.repo import Repo
from ..pubsub import PubSub
from ..risk.risk_manager import OrderIntent, Rejection, RiskManager
from ..risk.state import PortfolioState

log = structlog.get_logger()

BOOK_FILE = "data/xsec_book.json"
POST_REBALANCE_RECONCILE_S = 180.0


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
        self.last_rebalance_month: str = ""
        self._last_mark_refresh: datetime = datetime.min.replace(tzinfo=UTC)
        self._load_book()

    # ---------------- persistence ---------------- #
    def _load_book(self) -> None:
        if self.book_file.exists():
            data = json.loads(self.book_file.read_text())
            self.holdings = {k: int(v) for k, v in data.get("holdings", {}).items()}
            self.last_rebalance_month = data.get("last_rebalance_month", "")

    def _save_book(self) -> None:
        self.book_file.parent.mkdir(parents=True, exist_ok=True)
        self.book_file.write_text(json.dumps({
            "holdings": self.holdings,
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
            self.holdings[sym] = pos.qty
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

    # ---------------- rebalance ---------------- #
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

    async def rebalance(self, now: datetime) -> None:
        syms = self.universe()
        if not syms:
            return
        log.info("xsec.rebalance_start", universe=len(syms))
        history = await asyncio.to_thread(
            fetch_daily_history, self.settings.alpaca_api_key,
            self.settings.alpaca_secret_key, syms, self.cfg.history_days)
        targets = self.compute_targets(history, self.state.equity)
        if not targets:
            log.warning("xsec.no_targets")
            return
        current = {s: p.qty for s, p in self.state.book_positions("xsec").items()}
        marks = {s: history[s]["closes"][-1] for s in history}

        # sells first (frees cash and book headroom for the buys)
        orders: list[tuple[str, Literal["buy", "sell"], int, float]] = []
        for sym in current:
            tgt = targets.get(sym, 0)
            delta = tgt - current[sym]
            if delta < 0:
                orders.append((sym, "sell", -delta, marks.get(sym, 0.0)))
        for sym, tgt in targets.items():
            delta = tgt - current.get(sym, 0)
            if delta > 0:
                orders.append((sym, "buy", delta, marks[sym]))

        submitted = 0
        for sym, side, qty, price in orders:
            if price <= 0:
                continue
            intent = OrderIntent(symbol=sym, side=side, qty=qty, price_hint=price,
                                 reason="xsec_rebalance", book="xsec")
            approval = self.risk.approve(intent, now)
            if isinstance(approval, Rejection):
                self.repo.add_rejection(approval.reason, intent.as_dict())
                log.info("xsec.rejected", symbol=sym, reason=approval.reason)
                continue
            try:
                await self.om.submit(approval, now, mid=price, spread=0.0,
                                     strategy="xsec")
                submitted += 1
            except Exception as exc:
                log.error("xsec.submit_failed", symbol=sym, error=str(exc))

        self.holdings = dict(targets)
        self.last_rebalance_month = now.strftime("%Y-%m")
        self._save_book()
        log.info("xsec.rebalance_done", targets=len(targets), submitted=submitted)
        await self.pubsub.publish("xsec", {
            "ts": now.isoformat(), "targets": targets, "submitted": submitted})
        # fills land asynchronously; reconcile + retag brings the mirror true
        await asyncio.sleep(POST_REBALANCE_RECONCILE_S)
        try:
            await self.om.reconcile_state()
        except Exception as exc:
            log.error("xsec.post_reconcile_failed", error=str(exc))
        self.retag()
        self._save_book()

    # ---------------- marks ---------------- #
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

    # ---------------- scheduler ---------------- #
    def should_rebalance(self, now: datetime) -> bool:
        if self.state.halted:
            return False
        if now.strftime("%Y-%m") == self.last_rebalance_month:
            return False
        if not calendar.is_last_session_of_month(now):
            return False
        close = calendar.session_close(now)
        if close is None:
            return False
        start = close.timestamp() - self.cfg.rebalance_before_close_min * 60
        return now.timestamp() >= start and calendar.in_entry_window(now)

    async def run(self) -> None:
        """Engine background task: tag books, refresh marks, fire monthly."""
        log.info("xsec.strategy_started", holdings=len(self.holdings))
        while True:
            await asyncio.sleep(30)
            now = datetime.now(UTC)
            self.retag()
            if not calendar.is_session_open(now):
                continue
            if (now - self._last_mark_refresh).total_seconds() > 3600:
                try:
                    await self.refresh_marks(now)
                except Exception as exc:
                    log.error("xsec.mark_refresh_failed", error=str(exc))
            if self.should_rebalance(now):
                try:
                    await self.rebalance(now)
                except Exception as exc:
                    log.error("xsec.rebalance_failed", error=str(exc))
