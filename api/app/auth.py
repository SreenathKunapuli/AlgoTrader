"""Single-user JWT auth: APP_PASSWORD -> HS256 token, 24h expiry.

Login is rate-limited to 5 attempts/min per client IP (in-memory window —
single-process API, single user; a distributed limiter would be
overkill and add a Redis dependency to the auth path).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from lobplatform.config.settings import get_settings

_bearer = HTTPBearer(auto_error=False)
_attempts: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=5))


def check_login_rate(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _attempts[ip]
    if len(window) == 5 and now - window[0] < 60:
        raise HTTPException(429, "too many login attempts; wait a minute")
    window.append(now)


def issue_token(password: str) -> str:
    s = get_settings()
    if not password or password != s.app_password:
        raise HTTPException(401, "invalid password")
    payload = {"sub": "owner", "role": "owner",
               "exp": datetime.now(UTC) + timedelta(hours=s.jwt_expiry_hours)}
    return jwt.encode(payload, s.jwt_secret, algorithm="HS256")


def issue_guest_token() -> str:
    """Short-lived read-only token; no password required."""
    s = get_settings()
    payload = {"sub": "guest", "role": "guest",
               "exp": datetime.now(UTC) + timedelta(hours=8)}
    return jwt.encode(payload, s.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> dict[str, object]:
    try:
        return jwt.decode(token, get_settings().jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(401, f"invalid token: {exc}") from exc


def require_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, object]:
    if creds is None:
        raise HTTPException(401, "missing bearer token")
    return decode_token(creds.credentials)


def require_owner(
    payload: dict[str, object] = Depends(require_auth),
) -> dict[str, object]:
    """Rejects guest tokens — use on any write/control endpoint."""
    if payload.get("role") == "guest":
        raise HTTPException(403, "spectator access only — sign in as owner to make changes")
    return payload
