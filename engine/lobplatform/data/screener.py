"""Daily screener: top gainers and most-active stocks via Alpaca REST.

Called once at session open + 60s to extend the intraday universe with names
that have real volatility/volume that day.  The fixed tier universe misses
single-stock catalysts (earnings beats, M&A, macro prints) that momentum and
mean-reversion can trade.

Alpaca v1beta1 endpoints used (Basic/free plan):
  GET /v1beta1/screener/stocks/movers        → gainers[] + losers[]
  GET /v1beta1/screener/stocks/most_actives  → most_actives[]
Each item contains at minimum: symbol, price, percent_change, volume.
"""

from __future__ import annotations

import re
from typing import Any

import requests
import structlog

log = structlog.get_logger()

_BASE = "https://data.alpaca.markets/v1beta1/screener/stocks"
_VALID_SYM = re.compile(r"^[A-Z]{1,5}$")   # plain US equity tickers only


def _get(path: str, api_key: str, secret_key: str,
         params: dict[str, Any] | None = None) -> dict[str, Any]:
    r = requests.get(
        f"{_BASE}/{path}",
        headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key},
        params=params,
        timeout=10,
    )
    r.raise_for_status()
    data: dict[str, Any] = r.json()
    return data


def scan_candidates(
    api_key: str,
    secret_key: str,
    exclude: set[str],
    max_n: int = 10,
    min_price: float = 8.0,
    min_dollar_volume: float = 5_000_000.0,
    min_pct_change: float = 3.0,
) -> list[str]:
    """Return up to max_n intraday candidates not already in `exclude`.

    Pulls top-25 gainers (% change) and top-25 most-active (share volume),
    scores each by sqrt(pct_change * dollar_volume) to rank names that combine
    real momentum with real liquidity, then filters and caps at max_n.

    Filters applied (in order):
      - valid US equity ticker regex (no dots, hyphens, warrants, preferred)
      - price >= min_price (default $8 — excludes micro-cap junk)
      - dollar_volume >= min_dollar_volume (default $5M — ensures tradeable float)
      - for gainers: pct_change >= min_pct_change (default +3% — real catalyst)
    Either endpoint failing is tolerated; the other still contributes.
    """
    gainers_raw: list[dict[str, Any]] = []
    actives_raw: list[dict[str, Any]] = []
    try:
        data = _get("movers", api_key, secret_key, params={"top": 25})
        gainers_raw = data.get("gainers", [])
    except Exception as exc:
        log.warning("screener.movers_failed", error=str(exc))
    try:
        data = _get("most_actives", api_key, secret_key,
                    params={"by": "volume", "top": 25})
        actives_raw = data.get("most_actives", [])
    except Exception as exc:
        log.warning("screener.actives_failed", error=str(exc))

    scored: list[tuple[float, dict[str, Any]]] = []
    seen_sym: set[str] = set()

    def _score(item: dict[str, Any], require_pct: bool) -> float | None:
        sym = str(item.get("symbol", ""))
        price = float(item.get("price", 0) or 0)
        volume = float(item.get("volume", 0) or 0)
        pct = abs(float(item.get("percent_change", 0) or 0))
        dv = price * volume
        if (sym in exclude or sym in seen_sym
                or not _VALID_SYM.match(sym)
                or price < min_price
                or dv < min_dollar_volume):
            return None
        if require_pct and pct < min_pct_change:
            return None
        return float((pct * dv) ** 0.5)   # geometric mean: momentum and liquidity

    for item in gainers_raw:
        s = _score(item, require_pct=True)
        if s is not None:
            seen_sym.add(item["symbol"])
            scored.append((s, item))

    for item in actives_raw:
        s = _score(item, require_pct=False)
        if s is not None:
            seen_sym.add(item["symbol"])
            scored.append((s, item))

    scored.sort(key=lambda t: t[0], reverse=True)
    out = [item["symbol"] for _, item in scored[:max_n]]

    log.info("screener.scan", raw=len(gainers_raw) + len(actives_raw),
             accepted=len(out), symbols=out[:5])
    return out
