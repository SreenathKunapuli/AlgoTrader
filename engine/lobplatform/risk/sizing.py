"""Position sizing — exact formula from §5.3.

shares = min(shares_by_risk, shares_by_cap), floored, whole shares only;
0 shares -> skip. The vol-regime multiplier scales the result.
"""

from __future__ import annotations

import math

from ..config.tiers import TierConfig


def size_position(
    tier: TierConfig, equity: float, price: float, atr: float, vol_mult: float = 1.0
) -> int:
    """Whole-share position size; 0 means skip the trade."""
    if price <= 0 or atr <= 0 or equity <= 0:
        return 0
    risk_dollars = tier.risk_per_trade_pct * equity
    stop_distance = tier.stop_atr_multiple * atr
    shares_by_risk = math.floor(risk_dollars / stop_distance)
    shares_by_cap = math.floor(tier.max_position_pct * equity / price)
    return max(0, math.floor(min(shares_by_risk, shares_by_cap) * vol_mult))
