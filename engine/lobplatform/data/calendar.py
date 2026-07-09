"""XNYS session logic via exchange_calendars.

Why: half-days and holidays break naive "9:30–16:00" assumptions; a real
calendar keeps the EOD flattener and no-trade windows correct year-round.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import exchange_calendars as xcals
import pandas as pd

_CAL = xcals.get_calendar("XNYS")


def is_session_open(ts: datetime) -> bool:
    """True when `ts` (UTC) is inside a regular XNYS session."""
    t = pd.Timestamp(ts)
    try:
        return bool(_CAL.is_open_on_minute(t))
    except Exception:
        return False


def session_close(ts: datetime) -> datetime | None:
    """Close time (UTC) of the session containing/next to `ts`, else None."""
    t = pd.Timestamp(ts)
    try:
        sess = _CAL.minute_to_session(t, direction="next")
        close: datetime = _CAL.session_close(sess).to_pydatetime().replace(tzinfo=UTC)
        return close
    except Exception:
        return None


def session_open(ts: datetime) -> datetime | None:
    t = pd.Timestamp(ts)
    try:
        sess = _CAL.minute_to_session(t, direction="next")
        op: datetime = _CAL.session_open(sess).to_pydatetime().replace(tzinfo=UTC)
        return op
    except Exception:
        return None


def in_entry_window(ts: datetime) -> bool:
    """No new entries first 5 min or last 10 min of the session."""
    if not is_session_open(ts):
        return False
    o, c = session_open(ts), session_close(ts)
    if o is None or c is None:
        return False
    return o + timedelta(minutes=5) <= ts <= c - timedelta(minutes=10)


def in_eod_flatten_window(ts: datetime) -> bool:
    """Last 5 minutes of the session (flattener fires at close-5min)."""
    if not is_session_open(ts):
        return False
    c = session_close(ts)
    return c is not None and ts >= c - timedelta(minutes=5)


def is_last_session_of_month(ts: datetime) -> bool:
    """True when `ts` falls in the month's final XNYS session (xsec rebalance day)."""
    t = pd.Timestamp(ts)
    try:
        sess = _CAL.minute_to_session(t, direction="none")
    except Exception:
        return False
    nxt = _CAL.next_session(sess)
    return bool(nxt.month != sess.month or nxt.year != sess.year)
