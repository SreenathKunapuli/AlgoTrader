"""Event bus: Redis pub-sub when REDIS_URL is set, else in-process asyncio.

The API's /ws/stream reads the same channels; in fallback mode the API
polls the DB instead (engine and API are separate processes).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

log = structlog.get_logger()

CHANNELS = ("equity", "positions", "orders", "signals", "engine_status")


class PubSub:
    def __init__(self, redis_url: str = "") -> None:
        self._redis: Any = None
        self._local: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {c: [] for c in CHANNELS}
        if redis_url:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(redis_url)

    async def publish(self, channel: str, data: dict[str, Any]) -> None:
        payload = {"channel": channel, "data": data}
        if self._redis is not None:
            try:
                await self._redis.publish(f"lob:{channel}", json.dumps(payload, default=str))
                return
            except Exception as exc:
                log.warning("pubsub.redis_publish_failed", error=str(exc))
        for q in self._local.get(channel, []):
            if not q.full():
                q.put_nowait(payload)

    def subscribe_local(self, channel: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self._local.setdefault(channel, []).append(q)
        return q
