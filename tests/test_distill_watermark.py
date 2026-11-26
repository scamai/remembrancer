"""The distill watermark is a POSITION in a stream ordered by `occurred_at`.

That is only safe if the stream is append-ordered, and it is not: a recurring
calendar expansion files instances years ahead, so a live store can hold
thousands of future-dated day-files, decades ahead.

Taking the physically-last id would set the watermark to a 2040 event, after
which every event ingested afterwards is filed under a day <= today, sorts
BEFORE the watermark, and is never yielded again -- while the pass prints "no
distillable events past watermark; index regenerated." and exits 0.

Without this guard, the first pass that advances past a future-dated record
strands every later event.
"""
from datetime import datetime, timedelta, timezone

from distill.engine import _settled_watermark


class _Event:
    def __init__(self, event_id, occurred_at):
        self.id = event_id
        self.occurred_at = occurred_at


def _iso(when):
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


_NOW = datetime.now(timezone.utc)
_YESTERDAY = _iso(_NOW - timedelta(days=1))
_LAST_WEEK = _iso(_NOW - timedelta(days=7))
_FAR_FUTURE = "2040-07-02T10:30:00Z"


def test_the_watermark_never_lands_on_a_future_dated_record():
    stream = [
        _Event("slack:1", _LAST_WEEK),
        _Event("git:2", _YESTERDAY),
        _Event("gcal:recurring-2040", _FAR_FUTURE),
    ]
    assert _settled_watermark(stream, "prev") == "git:2", (
        "advancing to the 2040 instance strands every later real event"
    )


def test_many_trailing_future_records_are_all_skipped():
    stream = [_Event("slack:1", _LAST_WEEK)] + [
        _Event(f"gcal:{year}", f"{year}-03-01T09:00:00Z")
        for year in range(2027, 2041)
    ]
    assert _settled_watermark(stream, "prev") == "slack:1"


def test_a_settled_only_stream_advances_to_its_physical_end():
    stream = [_Event("a", _LAST_WEEK), _Event("b", _YESTERDAY)]
    assert _settled_watermark(stream, "prev") == "b"


def test_an_all_future_window_holds_the_watermark_where_it_was():
    """Skipping past records that have not happened yet is the bug; refusing to
    advance costs one re-scan."""
    stream = [_Event("gcal:x", _FAR_FUTURE), _Event("gcal:y", "2039-01-01T00:00:00Z")]
    assert _settled_watermark(stream, "prev") == "prev"


def test_an_empty_window_holds_the_watermark():
    assert _settled_watermark([], "prev") == "prev"
    assert _settled_watermark([], None) is None


def test_an_unparseable_date_is_treated_as_settled_not_as_the_future():
    """Refusing to advance on bad data would wedge the distiller permanently."""
    stream = [_Event("a", _LAST_WEEK), _Event("weird", "not-a-date")]
    assert _settled_watermark(stream, "prev") == "weird"


def test_a_naive_timestamp_is_read_as_utc_not_rejected():
    naive_past = (datetime.now(timezone.utc) - timedelta(days=2)).strftime(
        "%Y-%m-%dT%H:%M:%S")
    stream = [_Event("a", _LAST_WEEK), _Event("naive", naive_past)]
    assert _settled_watermark(stream, "prev") == "naive"
