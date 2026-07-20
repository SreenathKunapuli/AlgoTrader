"""Two-book separation tests: xsec momentum book vs intraday ensemble book.

The failure modes these guard against: the EOD flattener liquidating the
monthly book, the intraday path adjusting xsec positions (or vice versa),
one book consuming the other's risk budget, and rebalance math drifting
from the research implementation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from lobplatform.config.tiers import TIERS, Tier
from lobplatform.config.xsec import XsecConfig
from lobplatform.risk.risk_manager import OrderIntent, Rejection, RiskManager
from lobplatform.risk.state import PortfolioState, Position
from lobplatform.strategy.xsec_momentum import XsecMomentumStrategy, momentum_ranks

TIER = TIERS[Tier.MEDIUM]
ENTRY_TS = datetime(2026, 7, 8, 15, 0, tzinfo=UTC)  # Wed, mid-session


def make_state(equity: float = 100_000.0) -> PortfolioState:
    s = PortfolioState(equity=equity, cash=equity, day_start_equity=equity,
                       peak_equity=equity)
    return s


def make_strategy(state: PortfolioState, tmp_path, cfg: XsecConfig | None = None,
                  ) -> XsecMomentumStrategy:
    from unittest.mock import MagicMock

    return XsecMomentumStrategy(
        settings=MagicMock(), state=state, risk=RiskManager(TIER, state),
        om=MagicMock(), repo=MagicMock(), pubsub=MagicMock(),
        cfg=cfg or XsecConfig(), book_file=str(tmp_path / "book.json"))


# ---------------- momentum math ---------------- #
def test_momentum_ranks_match_research_formula():
    cfg = XsecConfig(lookback_days=252, skip_days=21, liquidity_keep=1.0)
    closes_a = [100.0] * 300
    closes_a[-252] = 100.0
    closes_a[-22] = 150.0  # mom = 150/100 - 1 = 0.5
    closes_b = [100.0] * 300
    closes_b[-252] = 100.0
    closes_b[-22] = 120.0  # mom = 0.2
    hist = {"A": {"closes": closes_a, "dollar_vol": 1e9},
            "B": {"closes": closes_b, "dollar_vol": 1e9},
            "SHORT_HIST": {"closes": [100.0] * 50, "dollar_vol": 1e9}}
    ranked = momentum_ranks(hist, cfg)
    assert [s for s, _ in ranked] == ["A", "B"]  # short history excluded
    assert ranked[0][1] == pytest.approx(0.5)
    assert ranked[1][1] == pytest.approx(0.2)


def test_liquidity_filter_drops_bottom():
    cfg = XsecConfig(lookback_days=10, skip_days=2, liquidity_keep=0.5)
    hist = {s: {"closes": [100.0] * 20, "dollar_vol": dv}
            for s, dv in [("BIG", 1e9), ("SMALL", 1e3)]}
    ranked = momentum_ranks(hist, cfg)
    assert [s for s, _ in ranked] == ["BIG"]


# ---------------- risk: book-aware approval ---------------- #
def test_xsec_intent_bypasses_tier_universe():
    state = make_state()
    risk = RiskManager(TIER, state)
    intent = OrderIntent(symbol="ZTS", side="buy", qty=10, price_hint=100.0,
                         reason="xsec_rebalance", book="xsec")
    approval = risk.approve(intent, ENTRY_TS)
    assert not isinstance(approval, Rejection)  # ZTS not in tier universe


def test_intraday_intent_still_universe_checked():
    state = make_state()
    risk = RiskManager(TIER, state)
    intent = OrderIntent(symbol="ZTS", side="buy", qty=10, price_hint=100.0)
    approval = risk.approve(intent, ENTRY_TS)
    assert isinstance(approval, Rejection)
    assert "not in tier universe" in approval.reason


def test_xsec_allocation_cap():
    state = make_state(100_000.0)
    risk = RiskManager(TIER, state)
    # alloc 50% = 50k; a 6k order fits per-position (6%) but book fills up
    for i in range(9):
        state.positions[f"S{i}"] = Position(symbol=f"S{i}", qty=55, entry_price=100.0,
                                            mark=100.0, book="xsec")
    # book gross = 9 * 5500 = 49500; adding 5500 breaches 50k
    intent = OrderIntent(symbol="NEW", side="buy", qty=55, price_hint=100.0, book="xsec")
    approval = risk.approve(intent, ENTRY_TS)
    assert isinstance(approval, Rejection)
    assert "allocation" in approval.reason


def test_xsec_book_does_not_consume_intraday_gross():
    state = make_state(100_000.0)
    risk = RiskManager(TIER, state)
    # 50k of xsec exposure; MEDIUM tier max gross is 80k
    for i in range(10):
        state.positions[f"X{i}"] = Position(symbol=f"X{i}", qty=50, entry_price=100.0,
                                            mark=100.0, book="xsec")
    intent = OrderIntent(symbol="AAPL", side="buy", qty=70, price_hint=100.0)
    approval = risk.approve(intent, ENTRY_TS)
    assert not isinstance(approval, Rejection)  # intraday budget untouched


def test_xsec_long_only_and_cross_book_lock():
    state = make_state()
    risk = RiskManager(TIER, state)
    # xsec book is long-only: sell intent must be rejected
    short = OrderIntent(symbol="ZTS", side="sell", qty=10, price_hint=100.0, book="xsec")
    r = risk.approve(short, ENTRY_TS)
    assert isinstance(r, Rejection) and "long-only" in r.reason

    state.positions["AAPL"] = Position(symbol="AAPL", qty=10, entry_price=100.0,
                                       mark=100.0, book="xsec")
    risk.add_to_universe(["AAPL"])

    # Intraday LONG add-on on an xsec long position is now allowed
    intraday_long = OrderIntent(symbol="AAPL", side="buy", qty=5, price_hint=100.0)
    r2 = risk.approve(intraday_long, ENTRY_TS)
    assert not isinstance(r2, Rejection), (
        f"expected approval, got: {r2.reason if isinstance(r2, Rejection) else ''}")

    # Intraday SHORT on an xsec position must still be rejected (cross-book conflict)
    intraday_short = OrderIntent(symbol="AAPL", side="sell", qty=5, price_hint=100.0)
    r3 = risk.approve(intraday_short, ENTRY_TS)
    assert isinstance(r3, Rejection) and "held by xsec" in r3.reason


# ---------------- strategy: targets, persistence, retag ---------------- #
def test_targets_equal_weight_and_exclude_intraday(tmp_path):
    state = make_state(100_000.0)
    state.positions["AAA"] = Position(symbol="AAA", qty=5, entry_price=50.0,
                                      mark=50.0, book="intraday")
    cfg = XsecConfig(top_n=2, alloc_pct=0.5, lookback_days=10, skip_days=2,
                     liquidity_keep=1.0)
    strat = make_strategy(state, tmp_path, cfg)
    hist = {}
    for sym, mom_px in [("AAA", 300.0), ("BBB", 200.0), ("CCC", 150.0), ("DDD", 120.0)]:
        closes = [100.0] * 20
        closes[-3] = mom_px  # position -1-skip
        closes[-1] = 100.0
        hist[sym] = {"closes": closes, "dollar_vol": 1e9}
    targets = strat.compute_targets(hist, equity=100_000.0)
    # AAA excluded (intraday-held); next best BBB, CCC; 25k each at px 100
    assert set(targets) == {"BBB", "CCC"}
    assert targets["BBB"] == 250


def test_book_file_roundtrip_and_retag(tmp_path):
    state = make_state()
    strat = make_strategy(state, tmp_path)
    strat.holdings = {"NVDA": 10, "LLY": 5}
    strat.last_rebalance_month = "2026-07"
    strat._save_book()

    # fresh instance loads the same book (restart survival)
    strat2 = make_strategy(state, tmp_path)
    assert strat2.holdings == {"NVDA": 10, "LLY": 5}
    assert strat2.last_rebalance_month == "2026-07"

    # reconcile adopted broker positions with default book -> retag fixes
    state.positions["NVDA"] = Position(symbol="NVDA", qty=10, entry_price=100.0,
                                       mark=100.0, stop_price=95.0)
    tagged = strat2.retag()
    assert state.positions["NVDA"].book == "xsec"
    assert state.positions["NVDA"].stop_price is None
    assert "NVDA" in tagged


def test_should_rebalance_gates(tmp_path):
    state = make_state()
    strat = make_strategy(state, tmp_path)
    # 2026-07-31 is a Friday and the last July session; 15:45 ET = 19:45 UTC
    month_end = datetime(2026, 7, 31, 19, 45, tzinfo=UTC)
    assert strat.should_rebalance(month_end)
    strat.last_rebalance_month = "2026-07"
    assert not strat.should_rebalance(month_end)  # once per month
    mid_month = datetime(2026, 7, 8, 19, 45, tzinfo=UTC)
    strat.last_rebalance_month = ""
    assert not strat.should_rebalance(mid_month)  # not last session
    too_early = datetime(2026, 7, 31, 14, 0, tzinfo=UTC)
    assert not strat.should_rebalance(too_early)  # before close-20min
    state.halted = True
    assert not strat.should_rebalance(month_end)


def test_book_file_is_valid_json_after_save(tmp_path):
    state = make_state()
    strat = make_strategy(state, tmp_path)
    strat.holdings = {"AAPL": 3}
    strat.pending = {"MSFT": 7}
    strat._save_book()
    data = json.loads(strat.book_file.read_text())
    assert data["holdings"] == {"AAPL": 3}
    assert data["pending"] == {"MSFT": 7}
    strat2 = make_strategy(state, tmp_path)
    assert strat2.pending == {"MSFT": 7}  # shortfalls survive restarts


# ---------------- catch-up, deltas, circuit breaker ---------------- #
def test_last_completed_month_end_calendar():
    from lobplatform.data import calendar as cal

    # Thursday 2026-07-02: June's final session was Tuesday 2026-06-30
    me = cal.last_completed_month_end(datetime(2026, 7, 2, 15, 0, tzinfo=UTC))
    assert me is not None and me.strftime("%Y-%m-%d") == "2026-06-30"
    # standing ON a month-end session, the last COMPLETED one is May's
    me2 = cal.last_completed_month_end(datetime(2026, 6, 30, 15, 0, tzinfo=UTC))
    assert me2 is not None and me2.strftime("%Y-%m") == "2026-05"


def test_missed_month_catch_up(tmp_path):
    state = make_state()
    strat = make_strategy(state, tmp_path)
    mid = datetime(2026, 7, 2, 19, 45, tzinfo=UTC)  # engine was down June 30
    assert strat.missed_month(mid) == "2026-06"
    strat.last_rebalance_month = "2026-06"          # already rebalanced: no-op
    assert strat.missed_month(mid) is None
    strat.last_rebalance_month = ""
    strat.cfg = XsecConfig(catch_up_missed=False)   # opt-out honored
    assert strat.missed_month(mid) is None
    strat.cfg = XsecConfig()
    state.halted = True                             # halted engine never trades
    assert strat.missed_month(mid) is None


def test_deltas_cover_buys_and_explicit_exits(tmp_path):
    state = make_state()
    strat = make_strategy(state, tmp_path)
    state.positions["OLD"] = Position(symbol="OLD", qty=10, entry_price=100.0,
                                      mark=100.0, book="xsec")
    state.positions["KEEP"] = Position(symbol="KEEP", qty=5, entry_price=100.0,
                                       mark=100.0, book="xsec")
    targets = {"NEW": 5, "KEEP": 5, "OLD": 0}
    assert strat._deltas(targets) == {"NEW": 5, "OLD": -10}


def test_breaker_trips_and_clears(tmp_path):
    state = make_state()
    strat = make_strategy(state, tmp_path)
    state.positions["AAA"] = Position(symbol="AAA", qty=100, entry_price=100.0,
                                      mark=55.0, book="xsec")  # -45% vs basis
    strat.update_breaker()
    assert state.xsec_buys_halted
    state.positions["AAA"].mark = 90.0                          # recovers
    strat.update_breaker()
    assert not state.xsec_buys_halted


def test_profiles_form_a_ladder():
    from lobplatform.config.tiers import TIERS as ALL_TIERS
    from lobplatform.config.xsec import XSEC_BY_TIER

    low, med, high = (XSEC_BY_TIER[t] for t in (Tier.LOW, Tier.MEDIUM, Tier.HIGH))
    assert low.top_n > med.top_n > high.top_n          # concentration ladder
    assert low.alloc_pct <= med.alloc_pct <= high.alloc_pct
    for tier, cfg in XSEC_BY_TIER.items():
        # equal weight must fit under the per-position cap (drift headroom)
        assert cfg.alloc_pct / cfg.top_n <= cfg.max_position_pct
        # account floor sits beyond the profile's backtest DD at allocation
        # (profiles.csv: top-50 -36.5%, top-20 -40.4%, top-10 -42.6%)
        backtest_dd = {50: 0.365, 20: 0.404, 10: 0.426}[cfg.top_n]
        assert ALL_TIERS[tier].max_drawdown_pct > backtest_dd * cfg.alloc_pct
