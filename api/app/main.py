"""FastAPI app — read/control plane over the engine's DB (§6).

Never touches the broker; control commands go through the `commands`
table which the engine polls every 2s.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from lobplatform.config.settings import get_settings
from lobplatform.config.tiers import Tier
from lobplatform.persistence.models import (
    EquitySnapshot,
    Order,
    RiskRejection,
    SignalHealth,
    SignalRecord,
    Trade,
)
from lobplatform.persistence.repo import Repo
from pydantic import BaseModel
from sqlalchemy import select

from .auth import (
    check_login_rate,
    decode_token,
    issue_guest_token,
    issue_token,
    require_auth,
    require_owner,
)
from .metrics import compute_metrics, downsample

settings = get_settings()
repo = Repo(settings.resolved_database_url())
app = FastAPI(title="LOB Platform API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)

RANGES = {"1d": timedelta(days=1), "1w": timedelta(weeks=1),
          "1m": timedelta(days=30), "all": None}


def iso_utc(dt: datetime | None) -> str | None:
    """DB datetimes come back naive from SQLite; stamp them as the UTC they are."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()



class LoginBody(BaseModel):
    password: str


class ConfirmBody(BaseModel):
    confirm: bool = False


class TierBody(BaseModel):
    tier: str


@app.post("/auth/login")
def login(body: LoginBody, request: Request) -> dict[str, str]:
    check_login_rate(request)
    return {"token": issue_token(body.password)}


@app.post("/auth/guest")
def guest_login(request: Request) -> dict[str, str]:
    """No password required — returns a read-only token (8h expiry)."""
    check_login_rate(request)
    return {"token": issue_guest_token()}


@app.get("/auth/me")
def me(payload: dict = Depends(require_auth)) -> dict[str, str]:  # type: ignore[type-arg]
    return {"role": str(payload.get("role", "owner")), "sub": str(payload.get("sub", ""))}


@app.get("/account")
def account(_: dict = Depends(require_auth)) -> dict[str, Any]:  # type: ignore[type-arg]
    st = repo.get_state()
    with repo.session() as s:
        latest = s.scalars(select(EquitySnapshot)
                           .order_by(EquitySnapshot.id.desc()).limit(1)).first()
    equity = latest.equity if latest else 0.0
    cash = latest.cash if latest else 0.0
    gross = latest.gross_exposure if latest else 0.0
    day_pnl = equity - st.day_start_equity if st.day_start_equity else 0.0
    return {
        "equity": equity, "cash": cash, "gross_exposure": gross,
        "day_pnl": round(day_pnl, 2),
        "day_pnl_pct": round(day_pnl / st.day_start_equity * 100, 4)
        if st.day_start_equity else 0.0,
        "buying_power": cash,
    }


@app.get("/positions")
def positions(_: dict = Depends(require_auth)) -> list[dict[str, Any]]:  # type: ignore[type-arg]
    """Engine heartbeat persists its live position mirror every 10s."""
    return list(repo.get_state().positions_json or [])


@app.get("/orders")
def orders(status: str | None = None, limit: int = Query(100, le=1000),
           _: dict = Depends(require_auth)) -> list[dict[str, Any]]:  # type: ignore[type-arg]
    with repo.session() as s:
        q = select(Order).order_by(Order.id.desc()).limit(limit)
        if status:
            q = q.where(Order.status == status)
        return [{
            "client_order_id": o.client_order_id, "symbol": o.symbol, "side": o.side,
            "qty": o.qty, "type": o.order_type, "status": o.status,
            "ts": iso_utc(o.ts), "limit_price": o.limit_price,
            "filled_qty": o.filled_qty, "fill_price": o.fill_price, "reason": o.reason,
        } for o in s.scalars(q)]


@app.get("/trades")
def trades(since: str | None = None, limit: int = Query(100, le=1000),
           _: dict = Depends(require_auth)) -> list[dict[str, Any]]:  # type: ignore[type-arg]
    with repo.session() as s:
        q = select(Trade).order_by(Trade.exit_ts.desc()).limit(limit)
        if since:
            q = q.where(Trade.exit_ts >= datetime.fromisoformat(since))
        return [{
            "symbol": t.symbol, "side": t.side, "qty": t.qty,
            "entry_ts": iso_utc(t.entry_ts), "exit_ts": iso_utc(t.exit_ts),
            "entry_price": t.entry_price, "exit_price": t.exit_price,
            "pnl": round(t.pnl, 2), "signal_scores": t.signal_scores_json,
            "holding_seconds": (t.exit_ts - t.entry_ts).total_seconds(),
        } for t in s.scalars(q)]


@app.get("/equity-curve")
def equity_curve(range: str = "1d",
                 _: dict = Depends(require_auth)) -> list[dict[str, Any]]:  # type: ignore[type-arg]
    if range not in RANGES:
        raise HTTPException(422, f"range must be one of {list(RANGES)}")
    with repo.session() as s:
        q = select(EquitySnapshot).order_by(EquitySnapshot.id)
        span = RANGES[range]
        if span is not None:
            q = q.where(EquitySnapshot.ts >= datetime.now(UTC) - span)
        pts = [{"ts": iso_utc(r.ts), "equity": r.equity} for r in s.scalars(q)]
    return downsample(pts)


@app.get("/metrics")
def metrics(range: str = "all",
            _: dict = Depends(require_auth)) -> dict[str, Any]:  # type: ignore[type-arg]
    if range not in RANGES:
        raise HTTPException(422, f"range must be one of {list(RANGES)}")
    span = RANGES[range]
    cutoff = datetime.now(UTC) - span if span else None
    with repo.session() as s:
        eq_q = select(EquitySnapshot).order_by(EquitySnapshot.id)
        tr_q = select(Trade)
        if cutoff:
            eq_q = eq_q.where(EquitySnapshot.ts >= cutoff)
            tr_q = tr_q.where(Trade.exit_ts >= cutoff)
        equity = [(r.ts, r.equity) for r in s.scalars(eq_q)]
        pnls = [t.pnl for t in s.scalars(tr_q)]
    return compute_metrics(equity, pnls)


@app.get("/signals/latest")
def signals_latest(_: dict = Depends(require_auth)) -> list[dict[str, Any]]:  # type: ignore[type-arg]
    with repo.session() as s:
        rows = list(s.scalars(select(SignalRecord)
                              .order_by(SignalRecord.id.desc()).limit(200)))
    latest: dict[str, SignalRecord] = {}
    for r in rows:
        latest.setdefault(r.symbol, r)
    return [{"symbol": r.symbol, "ts": iso_utc(r.ts), "ensemble": r.ensemble,
             "per_signal": r.scores_json} for r in latest.values()]


@app.get("/signals/health")
def signals_health(_: dict = Depends(require_auth)) -> list[dict[str, Any]]:  # type: ignore[type-arg]
    with repo.session() as s:
        rows = list(s.scalars(select(SignalHealth)
                              .order_by(SignalHealth.id.desc()).limit(50)))
    return [{"signal": r.signal, "ts": iso_utc(r.ts),
             "rolling_hit_rate": r.rolling_hit_rate,
             "attributed_pnl_20s": r.attributed_pnl_20s,
             "weight_multiplier": r.weight_multiplier,
             "flagged_for_retrain": r.flagged_for_retrain} for r in rows]


@app.get("/engine/status")
def engine_status(_: dict = Depends(require_auth)) -> dict[str, Any]:  # type: ignore[type-arg]
    st = repo.get_state()
    hb_age = ((datetime.now(UTC) - st.heartbeat_ts.replace(tzinfo=UTC)).total_seconds()
              if st.heartbeat_ts else None)
    return {"status": st.status, "tier": st.tier, "halted_reason": st.halted_reason,
            "heartbeat_age_s": hb_age,
            "last_data_ts": iso_utc(st.last_data_ts),
            "trading_mode": settings.trading_mode}


@app.put("/config/tier")
def set_tier(body: TierBody, _: dict = Depends(require_owner)) -> dict[str, str]:  # type: ignore[type-arg]
    if body.tier not in [t.value for t in Tier]:
        raise HTTPException(422, "tier must be low|medium|high")
    if repo.get_state().status == "HALTED":
        raise HTTPException(409, "engine is HALTED; reset first")
    repo.enqueue_command("set_tier", {"tier": body.tier})
    return {"status": "queued", "tier": body.tier}


@app.post("/engine/kill")
def kill(body: ConfirmBody, _: dict = Depends(require_owner)) -> dict[str, str]:  # type: ignore[type-arg]
    if not body.confirm:
        raise HTTPException(422, 'requires {"confirm": true}')
    repo.enqueue_command("kill")
    return {"status": "kill queued"}


@app.post("/engine/reset")
def reset(body: ConfirmBody, _: dict = Depends(require_owner)) -> dict[str, str]:  # type: ignore[type-arg]
    if not body.confirm:
        raise HTTPException(422, 'requires {"confirm": true}')
    repo.enqueue_command("reset")
    return {"status": "reset queued"}


@app.get("/risk/rejections")
def rejections(limit: int = Query(50, le=500),
               _: dict = Depends(require_auth)) -> list[dict[str, Any]]:  # type: ignore[type-arg]
    with repo.session() as s:
        rows = list(s.scalars(select(RiskRejection)
                              .order_by(RiskRejection.id.desc()).limit(limit)))
    return [{"ts": iso_utc(r.ts), "reason": r.reason, "intent": r.intent_json}
            for r in rows]


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    try:
        st = repo.get_state()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(503, f"db unreachable: {exc}") from exc
    hb_ok = bool(st.heartbeat_ts and
                 (datetime.now(UTC) - st.heartbeat_ts.replace(tzinfo=UTC)).total_seconds() < 30)
    return {"db": "ok", "engine_heartbeat_fresh": hb_ok}


# ---------------- WebSocket ---------------- #
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket, token: str = Query("")) -> None:
    try:
        decode_token(token)
    except HTTPException:
        await ws.close(code=4401)
        return
    await ws.accept()
    try:
        sub_msg = json.loads(await ws.receive_text())
        channels: set[str] = set(sub_msg.get("subscribe", []))
    except Exception:
        await ws.close(code=4400)
        return

    async def db_poller() -> None:
        """Fallback realtime: poll the DB every 2s for fresh rows/state."""
        last_eq_id = 0
        last_status = ""
        while True:
            await asyncio.sleep(2)
            with repo.session() as s:
                if "equity" in channels:
                    for r in s.scalars(select(EquitySnapshot)
                                       .where(EquitySnapshot.id > last_eq_id)
                                       .order_by(EquitySnapshot.id)):
                        last_eq_id = max(last_eq_id, r.id)
                        await ws.send_json({"channel": "equity",
                                            "data": {"ts": iso_utc(r.ts),
                                                     "equity": r.equity,
                                                     "cash": r.cash,
                                                     "gross": r.gross_exposure},
                                            "ts": iso_utc(r.ts)})
            if "engine_status" in channels:
                st = repo.get_state()
                if st.status != last_status:
                    last_status = st.status
                    await ws.send_json({"channel": "engine_status",
                                        "data": {"status": st.status,
                                                 "reason": st.halted_reason},
                                        "ts": datetime.now(UTC).isoformat()})

    async def pinger() -> None:
        while True:
            await asyncio.sleep(15)
            await ws.send_json({"channel": "ping", "ts": datetime.now(UTC).isoformat()})

    tasks = [asyncio.create_task(db_poller()), asyncio.create_task(pinger())]
    try:
        while True:
            await ws.receive_text()  # keep connection open; ignore client chatter
    except Exception:
        pass
    finally:
        for t in tasks:
            t.cancel()
