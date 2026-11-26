"""Time-domain guard for watermark cursors.

A connector's watermark must never be in the future. A future cursor silently
wedges the feed: the next run asks the source for "everything after <future
date>" and gets nothing — forever. (This is exactly how the Calendar feed died:
it saved the newest meeting's *start* time, which jumps ahead the moment a meeting
is scheduled for a future date.) `clamp_iso_to_now` caps any RFC3339 watermark at
the current instant so a future value can never be persisted — the one invariant
that kills this whole bug class.
"""
from __future__ import annotations

from datetime import datetime, timezone


def _parse_iso(ts: str) -> datetime | None:
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d


def clamp_iso_to_now(ts: str | None, now: datetime | None = None) -> str | None:
    """Return `ts` if it is at or before now; otherwise return now (RFC3339 'Z').

    Unparseable or empty inputs are returned unchanged — we never silently drop a
    cursor we don't understand. `now` is injectable for deterministic tests.
    """
    if not ts:
        return ts
    d = _parse_iso(ts)
    if d is None:
        return ts
    now = now or datetime.now(timezone.utc)
    return ts if d <= now else now.strftime("%Y-%m-%dT%H:%M:%SZ")
