"""The distill backfill valve: old git commits stay OUT of LLM distillation
(they remain in L1 for footprints/dig), fresh commits and every other source
pass through untouched."""
from remembrancerlib import Event

from distill.engine import _git_age_filter


def _ev(id, source, occurred_at):
    return Event(id=id, source=source, author="x", occurred_at=occurred_at,
                 payload={"subject": "x"})


def test_old_git_dropped_fresh_git_and_other_sources_kept():
    events = [
        _ev("git:old", "git", "2026-01-05T10:00:00-05:00"),
        _ev("git:new", "git", "2026-07-01T10:00:00-05:00"),
        _ev("gmail:old", "gmail", "2026-01-05T10:00:00Z"),
        _ev("slack:old", "slack", "2025-11-01T10:00:00Z"),
    ]
    kept, dropped = _git_age_filter(events, 14, now="2026-07-09")
    assert dropped == 1
    assert [e.id for e in kept] == ["git:new", "gmail:old", "slack:old"]


def test_valve_disabled_keeps_everything():
    events = [_ev("git:old", "git", "2020-01-01T00:00:00Z")]
    for off in (None, 0):
        kept, dropped = _git_age_filter(events, off, now="2026-07-09")
        assert dropped == 0 and kept == events


def test_git_event_missing_date_is_dropped_when_valve_on():
    # No occurred_at = can't prove freshness; the valve treats it as stale.
    kept, dropped = _git_age_filter([_ev("git:x", "git", None)], 14, now="2026-07-09")
    assert dropped == 1 and kept == []
