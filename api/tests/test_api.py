"""API endpoint tests: auth gates, happy paths, invalid input, WS, metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from lobplatform.persistence.repo import Repo

import api.app.main as main_mod
from api.app.metrics import compute_metrics


@pytest.fixture()
def client_repo(monkeypatch):  # type: ignore[no-untyped-def]
    import api.app.auth as auth_mod

    repo = Repo("sqlite:///:memory:")
    monkeypatch.setattr(main_mod, "repo", repo)

    class _S:
        app_password = "pw"
        jwt_secret = "test-secret"
        jwt_expiry_hours = 24

    monkeypatch.setattr(auth_mod, "get_settings", lambda: _S())
    return repo


@pytest.fixture()
async def client(client_repo):  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=main_mod.app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


async def _token(c: AsyncClient) -> dict[str, str]:
    r = await c.post("/auth/login", json={"password": "pw"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}"}


async def test_login_wrong_password(client) -> None:  # type: ignore[no-untyped-def]
    r = await client.post("/auth/login", json={"password": "nope"})
    assert r.status_code == 401


async def test_endpoints_require_auth(client) -> None:  # type: ignore[no-untyped-def]
    for path in ["/account", "/positions", "/orders", "/trades", "/equity-curve",
                 "/metrics", "/signals/latest", "/signals/health", "/engine/status",
                 "/risk/rejections"]:
        r = await client.get(path)
        assert r.status_code == 401, path


async def test_full_surface_happy_path(client, client_repo) -> None:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    client_repo.update_state(status="RUNNING", tier="medium",
                             day_start_equity=100_000.0, heartbeat_ts=now,
                             positions_json=[{"symbol": "SPY", "qty": 5}])
    client_repo.add_equity_snapshot(now, 100_500.0, 60_000.0, 40_000.0)
    client_repo.add_trade(symbol="SPY", side="long", qty=5, entry_ts=now - timedelta(hours=2),
                          exit_ts=now, entry_price=100.0, exit_price=101.0, pnl=5.0,
                          signal_scores_json={"momentum": 0.4})
    client_repo.add_signal(now, "SPY", {"momentum": {"score": 0.5}}, 0.42)
    client_repo.add_rejection("exceeds max position size", {"symbol": "SPY"})
    client_repo.add_signal_health(signal="lob_flow", rolling_hit_rate=0.55,
                                  attributed_pnl_20s=12.0, weight_multiplier=1.0,
                                  flagged_for_retrain=False)
    h = await _token(client)

    acct = (await client.get("/account", headers=h)).json()
    assert acct["equity"] == 100_500.0 and acct["day_pnl"] == 500.0

    assert (await client.get("/positions", headers=h)).json()[0]["symbol"] == "SPY"
    trades = (await client.get("/trades", headers=h)).json()
    assert trades[0]["pnl"] == 5.0 and trades[0]["signal_scores"] == {"momentum": 0.4}
    curve = (await client.get("/equity-curve?range=all", headers=h)).json()
    assert len(curve) == 1
    sig = (await client.get("/signals/latest", headers=h)).json()
    assert sig[0]["symbol"] == "SPY" and sig[0]["ensemble"] == 0.42
    rej = (await client.get("/risk/rejections", headers=h)).json()
    assert "position size" in rej[0]["reason"]
    health = (await client.get("/signals/health", headers=h)).json()
    assert health[0]["signal"] == "lob_flow"
    status = (await client.get("/engine/status", headers=h)).json()
    assert status["status"] == "RUNNING" and status["trading_mode"] == "paper"
    hz = (await client.get("/healthz")).json()
    assert hz["db"] == "ok" and hz["engine_heartbeat_fresh"] is True


async def test_control_commands(client, client_repo) -> None:  # type: ignore[no-untyped-def]
    h = await _token(client)
    assert (await client.post("/engine/kill", json={}, headers=h)).status_code == 422
    r = await client.post("/engine/kill", json={"confirm": True}, headers=h)
    assert r.status_code == 200
    cmds = client_repo.pending_commands()
    assert cmds and cmds[0].command == "kill"

    assert (await client.put("/config/tier", json={"tier": "extreme"},
                             headers=h)).status_code == 422
    r = await client.put("/config/tier", json={"tier": "high"}, headers=h)
    assert r.status_code == 200
    assert client_repo.pending_commands()[-1].payload_json == {"tier": "high"}

    client_repo.update_state(status="HALTED")
    r = await client.put("/config/tier", json={"tier": "low"}, headers=h)
    assert r.status_code == 409
    r = await client.post("/engine/reset", json={"confirm": True}, headers=h)
    assert r.status_code == 200


async def test_invalid_range(client, client_repo) -> None:  # type: ignore[no-untyped-def]
    h = await _token(client)
    assert (await client.get("/equity-curve?range=5y", headers=h)).status_code == 422


def test_metrics_golden() -> None:
    """Fixture equity/trades -> known metric values."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    # equity: 100k -> 101k -> 100k -> 102k over 4 days
    eq = [(base + timedelta(days=i), v)
          for i, v in enumerate([100_000.0, 101_000.0, 100_000.0, 102_000.0])]
    pnls = [50.0, -25.0, 100.0, -25.0]
    m = compute_metrics(eq, pnls)
    assert m["total_return_pct"] == 2.0
    assert m["hit_rate"] == 0.5
    assert m["avg_win"] == 75.0 and m["avg_loss"] == -25.0
    # max DD: peak 101k -> trough 100k = 0.9901%
    assert abs(m["max_drawdown_pct"] - 0.9901) < 0.001
    assert m["n_trades"] == 4
    assert m["sharpe_daily_annualized"] != 0.0


async def test_ws_stream(client, client_repo) -> None:  # type: ignore[no-untyped-def]
    """WS: auth required; subscribed channel delivers a DB-polled event."""
    from starlette.testclient import TestClient

    now = datetime.now(UTC)
    with TestClient(main_mod.app) as tc:
        # bad token -> closed
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with tc.websocket_connect("/ws/stream?token=bad") as ws:
                ws.receive_json()
        # good token -> subscribe, then a fresh equity row arrives via poller
        token = tc.post("/auth/login", json={"password": "pw"}).json()["token"]
        with tc.websocket_connect(f"/ws/stream?token={token}") as ws:
            ws.send_json({"subscribe": ["equity", "engine_status"]})
            client_repo.add_equity_snapshot(now, 123_456.0, 1.0, 0.0)
            client_repo.update_state(status="RUNNING")
            msg = ws.receive_json()
            assert msg["channel"] in ("equity", "engine_status", "ping")
