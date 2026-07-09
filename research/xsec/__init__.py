"""Cross-sectional daily-horizon strategy research.

Why this exists: the cost_aware_experiment proved the intraday LOB signal is
real but only capturable passively, and the live 1-min ensemble competes in
the most efficient arena in markets with free delayed data. This package
tests the alternative with the strongest published evidence a retail-sized
account can actually harvest: cross-sectional equity factors (momentum,
reversal, low-vol) combined by a walk-forward ML ranker, monthly rebalance,
where costs are ~10bps not ~10 spreads.
"""
