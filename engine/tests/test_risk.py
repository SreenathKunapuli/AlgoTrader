"""Sizing property tests + full RiskManager.approve() branch coverage."""

from __future__ import annotations

from datetime import UTC

from hypothesis import given
from hypothesis import strategies as st
from lobplatform.config.tiers import TIERS, Tier
from lobplatform.risk.risk_manager import Approval, OrderIntent, Rejection, RiskManager
from lobplatform.risk.sizing import size_position
from lobplatform.risk.state import Position

from .conftest import IN_SESSION

MED = TIERS[Tier.MEDIUM]
HIGH = TIERS[Tier.HIGH]


# ---------- sizing ---------- #
@given(equity=st.floats(1_000, 10_000_000), price=st.floats(1, 5_000),
       atr=st.floats(0.01, 500), vol=st.floats(0.3, 1.0))
def test_sizing_never_exceeds_caps(equity: float, price: float, atr: float, vol: float) -> None:
    for tier in TIERS.values():
        shares = size_position(tier, equity, price, atr, vol)
        assert shares >= 0
        assert shares * price <= tier.max_position_pct * equity + price  # floor slack
        risk = shares * tier.stop_atr_multiple * atr
        assert risk <= tier.risk_per_trade_pct * equity + tier.stop_atr_multiple * atr


def test_sizing_zero_on_degenerate() -> None:
    assert size_position(MED, 0, 100, 1) == 0
    assert size_position(MED, 100_000, 0, 1) == 0
    assert size_position(MED, 100_000, 100, 0) == 0
    assert size_position(MED, 100, 5000, 1) == 0  # can't afford one share


# ---------- approve() branches ---------- #
def _intent(symbol: str = "AAPL", side: str = "buy", qty: int = 10,
            price: float = 100.0, reason: str = "signal") -> OrderIntent:
    return OrderIntent(symbol=symbol, side=side, qty=qty, price_hint=price, reason=reason)  # type: ignore[arg-type]


def test_rejects_when_halted(state) -> None:  # type: ignore[no-untyped-def]
    state.halted = True
    r = RiskManager(MED, state).approve(_intent(), IN_SESSION)
    assert isinstance(r, Rejection) and "HALTED" in r.reason


def test_rejects_unknown_symbol(state) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(MED, state).approve(_intent(symbol="GME"), IN_SESSION)
    assert isinstance(r, Rejection) and "universe" in r.reason


def test_rejects_nonpositive_qty(state) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(MED, state).approve(_intent(qty=0), IN_SESSION)
    assert isinstance(r, Rejection)


def test_rejects_short_in_medium(state) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(MED, state).approve(_intent(side="sell"), IN_SESSION)
    assert isinstance(r, Rejection) and "short" in r.reason


def test_allows_short_in_high(state) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(HIGH, state).approve(_intent(side="sell"), IN_SESSION)
    assert isinstance(r, Approval)


def test_rejects_position_size(state) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(MED, state).approve(_intent(qty=200, price=100.0), IN_SESSION)
    # 200*100 = 20k > 10% of 100k
    assert isinstance(r, Rejection) and "position size" in r.reason


def test_rejects_gross_exposure(state) -> None:  # type: ignore[no-untyped-def]
    for _i, sym in enumerate(["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV"]):
        state.positions[sym] = Position(symbol=sym, qty=95, entry_price=100.0, mark=100.0)
    # gross = 8*9500 = 76k; +9.5k > 80% of 100k
    r = RiskManager(MED, state).approve(_intent(symbol="AAPL", qty=95), IN_SESSION)
    assert isinstance(r, Rejection) and "gross" in r.reason


def test_rejects_max_positions(state) -> None:  # type: ignore[no-untyped-def]
    for sym in MED.universe[:10]:
        state.positions[sym] = Position(symbol=sym, qty=1, entry_price=10.0, mark=10.0)
    r = RiskManager(MED, state).approve(_intent(symbol="UNH", qty=1, price=10.0), IN_SESSION)
    assert isinstance(r, Rejection) and "open positions" in r.reason


def test_rejects_daily_loss(state) -> None:  # type: ignore[no-untyped-def]
    state.intraday_realized_today = -2_500.0  # intraday book -2.5% on the day
    r = RiskManager(MED, state).approve(_intent(), IN_SESSION)
    assert isinstance(r, Rejection) and "daily loss" in r.reason


def test_daily_loss_does_not_block_xsec(state) -> None:  # type: ignore[no-untyped-def]
    state.intraday_realized_today = -2_500.0
    x = OrderIntent(symbol="ZTS", side="buy", qty=10, price_hint=100.0,
                    reason="xsec_rebalance", book="xsec")
    r = RiskManager(MED, state).approve(x, IN_SESSION)
    assert isinstance(r, Approval)  # intraday breach must not gate the monthly book


def test_rejects_drawdown(state) -> None:  # type: ignore[no-untyped-def]
    state.peak_equity = 160_000.0  # 100k now -> 37.5% DD > 35% MEDIUM floor
    state.day_start_equity = state.equity  # keep day pnl at 0
    r = RiskManager(MED, state).approve(_intent(), IN_SESSION)
    assert isinstance(r, Rejection) and "drawdown" in r.reason
    # the floor is account-wide: xsec entries are refused too
    x = OrderIntent(symbol="ZTS", side="buy", qty=10, price_hint=100.0, book="xsec")
    assert isinstance(RiskManager(MED, state).approve(x, IN_SESSION), Rejection)


def test_intraday_halt_scoped_to_intraday(state) -> None:  # type: ignore[no-untyped-def]
    state.intraday_halted = True
    rm = RiskManager(MED, state)
    r = rm.approve(_intent(), IN_SESSION)
    assert isinstance(r, Rejection) and "intraday" in r.reason
    x = OrderIntent(symbol="ZTS", side="buy", qty=10, price_hint=100.0, book="xsec")
    assert isinstance(rm.approve(x, IN_SESSION), Approval)


def test_xsec_breaker_blocks_buys_not_sells(state) -> None:  # type: ignore[no-untyped-def]
    state.xsec_buys_halted = True
    rm = RiskManager(MED, state)
    buy = OrderIntent(symbol="ZTS", side="buy", qty=10, price_hint=100.0, book="xsec")
    r = rm.approve(buy, IN_SESSION)
    assert isinstance(r, Rejection) and "breaker" in r.reason
    state.positions["ZTS"] = Position(symbol="ZTS", qty=10, entry_price=100.0,
                                      mark=100.0, book="xsec")
    sell = OrderIntent(symbol="ZTS", side="sell", qty=10, price_hint=100.0, book="xsec")
    assert isinstance(rm.approve(sell, IN_SESSION), Approval)  # de-risk always allowed


def test_rejects_outside_entry_window(state) -> None:  # type: ignore[no-untyped-def]
    from datetime import datetime

    weekend = datetime(2026, 6, 13, 15, 0, tzinfo=UTC)  # Saturday
    r = RiskManager(MED, state).approve(_intent(), weekend)
    assert isinstance(r, Rejection) and "window" in r.reason


def test_approves_valid_entry(state) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(MED, state).approve(_intent(), IN_SESSION)
    assert isinstance(r, Approval) and r.token


def test_exit_bypasses_windows_but_not_direction(state) -> None:  # type: ignore[no-untyped-def]
    state.positions["AAPL"] = Position(symbol="AAPL", qty=10, entry_price=100.0, mark=100.0)
    rm = RiskManager(MED, state)
    ok = rm.approve_exit(_intent(side="sell", qty=10, reason="stop"))
    assert isinstance(ok, Approval)
    bad = rm.approve_exit(_intent(side="buy", qty=10))  # would increase exposure
    assert isinstance(bad, Rejection)
