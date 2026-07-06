"""Repository: the engine's single DB access point.

Why: sync SQLAlchemy behind small, purpose-named methods keeps transactions
short and testable; the asyncio loop calls these via `asyncio.to_thread`
when on the hot path (they are all sub-millisecond on SQLite/Postgres).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    Base,
    Command,
    EngineState,
    EquitySnapshot,
    Order,
    RiskRejection,
    SignalHealth,
    SignalRecord,
    Trade,
    utcnow,
)


class Repo:
    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self._engine = create_engine(database_url, connect_args=connect_args, future=True)
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False)
        Base.metadata.create_all(self._engine)

    def session(self) -> Session:
        return self._session_factory()

    # --- engine state (singleton) ---
    def get_state(self) -> EngineState:
        with self.session() as s:
            state = s.get(EngineState, 1)
            if state is None:
                state = EngineState(id=1)
                s.add(state)
                s.commit()
            return state

    def update_state(self, **fields: Any) -> None:
        with self.session() as s:
            state = s.get(EngineState, 1) or EngineState(id=1)
            for k, v in fields.items():
                setattr(state, k, v)
            s.add(state)
            s.commit()

    # --- writes ---
    def add_equity_snapshot(self, ts: datetime, equity: float, cash: float, gross: float) -> None:
        with self.session() as s:
            s.add(EquitySnapshot(ts=ts, equity=equity, cash=cash, gross_exposure=gross))
            s.commit()

    def upsert_order(self, client_order_id: str, **fields: Any) -> None:
        with self.session() as s:
            row = s.scalar(select(Order).where(Order.client_order_id == client_order_id))
            if row is None:
                row = Order(client_order_id=client_order_id, **fields)
            else:
                for k, v in fields.items():
                    setattr(row, k, v)
            s.add(row)
            s.commit()

    def add_trade(self, **fields: Any) -> None:
        with self.session() as s:
            s.add(Trade(**fields))
            s.commit()

    def add_signal(self, ts: datetime, symbol: str, scores: dict[str, Any],
                   ensemble: float) -> None:
        with self.session() as s:
            s.add(SignalRecord(ts=ts, symbol=symbol, scores_json=scores, ensemble=ensemble))
            s.commit()

    def add_rejection(self, reason: str, intent: dict[str, Any]) -> None:
        with self.session() as s:
            s.add(RiskRejection(ts=utcnow(), reason=reason, intent_json=intent))
            s.commit()

    def add_signal_health(self, **fields: Any) -> None:
        with self.session() as s:
            s.add(SignalHealth(ts=utcnow(), **fields))
            s.commit()

    # --- commands channel ---
    def pending_commands(self) -> list[Command]:
        with self.session() as s:
            q = select(Command).where(Command.done.is_(False)).order_by(Command.id)
            return list(s.scalars(q))

    def mark_command_done(self, command_id: int) -> None:
        with self.session() as s:
            cmd = s.get(Command, command_id)
            if cmd:
                cmd.done = True
                s.commit()

    def enqueue_command(self, command: str, payload: dict[str, Any] | None = None) -> None:
        with self.session() as s:
            s.add(Command(command=command, payload_json=payload or {}))
            s.commit()
