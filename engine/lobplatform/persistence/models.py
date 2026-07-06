"""SQLAlchemy 2.0 ORM models — the shared contract between engine and API.

Why: the API is a read/control plane over these tables; the engine is the
only writer (except the `commands` table, which the API writes and the
engine polls). All timestamps are UTC-naive-forbidden: timezone-aware UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    gross_exposure: Mapped[float] = mapped_column(Float)


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    broker_order_id: Mapped[str] = mapped_column(String(64), default="")
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(8))          # buy / sell
    qty: Mapped[int] = mapped_column(Integer)
    order_type: Mapped[str] = mapped_column(String(16))   # limit / market
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    filled_qty: Mapped[int] = mapped_column(Integer, default=0)
    fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str] = mapped_column(String(64), default="signal")  # signal/stop/eod/kill


class Trade(Base):
    """Closed round-trip."""

    __tablename__ = "trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(8))           # long / short
    qty: Mapped[int] = mapped_column(Integer)
    entry_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    exit_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float] = mapped_column(Float)
    signal_scores_json: Mapped[dict] = mapped_column(JSON, default=dict)  # type: ignore[type-arg]


class SignalRecord(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    scores_json: Mapped[dict] = mapped_column(JSON, default=dict)  # type: ignore[type-arg]
    ensemble: Mapped[float] = mapped_column(Float)


class EngineState(Base):
    """Singleton row (id=1). HALTED survives restarts by living here."""

    __tablename__ = "engine_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    status: Mapped[str] = mapped_column(String(16), default="STOPPED")
    tier: Mapped[str] = mapped_column(String(8), default="medium")
    halted_reason: Mapped[str] = mapped_column(Text, default="")
    day_start_equity: Mapped[float] = mapped_column(Float, default=0.0)
    peak_equity: Mapped[float] = mapped_column(Float, default=0.0)
    heartbeat_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_data_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RiskRejection(Base):
    __tablename__ = "risk_rejections"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    reason: Mapped[str] = mapped_column(String(128))
    intent_json: Mapped[dict] = mapped_column(JSON, default=dict)  # type: ignore[type-arg]


class Command(Base):
    """API -> engine control channel; engine polls every 2s and marks done."""

    __tablename__ = "commands"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    command: Mapped[str] = mapped_column(String(32))       # kill / reset / set_tier
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)  # type: ignore[type-arg]
    done: Mapped[bool] = mapped_column(Boolean, default=False)


class SignalHealth(Base):
    __tablename__ = "signal_health"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    signal: Mapped[str] = mapped_column(String(32), index=True)
    rolling_hit_rate: Mapped[float] = mapped_column(Float)
    attributed_pnl_20s: Mapped[float] = mapped_column(Float)
    weight_multiplier: Mapped[float] = mapped_column(Float, default=1.0)
    flagged_for_retrain: Mapped[bool] = mapped_column(Boolean, default=False)
