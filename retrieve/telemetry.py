"""Per-call read-path telemetry, bound to one MCP read call through a contextvar.

``mcp/server.py`` opens a record around ``remembrancer_search``, ``remembrancer_get_note``
and ``remembrancer_timeline`` and writes it into that call's audit row;
the write and dig tools (``remembrancer_file_answer``, ``remembrancer_remember``,
``remembrancer_dig``, ``remembrancer_dig_live``, ``remembrancer_confirm_answer``) are recorded the
same way.
The retrieval core adds what only it can see:

- ``duration_ms``: ``search()``'s own measure, the same number it logs to
  ``state/queries.jsonl``, so the audit row and the query log agree;
- ``path``: which ranking path ``search()`` took, ``index`` or ``fallback``
  (the full-corpus load), which nothing measured before;
- ``audit_loads`` and ``audit_ms``: how many times the verification audit was
  loaded during the call, and the wall time spent loading it;
- ``contested_scan_ms``: wall time inside the contested-label index.

With no record bound (the CLI, benches, the nightly, the warm-up thread) every
hook is a no-op that costs one contextvar lookup. A thread started inside a
call does not inherit the record, so background work is never billed to it.
"""
from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

_CURRENT: "contextvars.ContextVar[Optional[dict[str, Any]]]" = contextvars.ContextVar(
    "remembrancer_read_telemetry", default=None)


def current() -> Optional[dict[str, Any]]:
    """The record bound to this call, or ``None`` outside a timed call."""
    return _CURRENT.get()


@contextmanager
def collect() -> Iterator[tuple[dict[str, Any], bool]]:
    """Bind a fresh record for one call and yield ``(record, owner)``.

    A nested ``collect()`` joins the record already bound and yields
    ``owner=False``, so a tool wrapper and the function it calls share one row.
    """
    existing = _CURRENT.get()
    if existing is not None:
        yield existing, False
        return
    record: dict[str, Any] = {}
    token = _CURRENT.set(record)
    try:
        yield record, True
    finally:
        _CURRENT.reset(token)


def query_log_allowed() -> bool:
    """False when the bound call asked to stay out of ``state/queries.jsonl``.

    The MCP layer sets ``query_log = False`` for callers that must never read as
    demand (a smoke-test probe); everything else logs exactly as before.
    """
    record = _CURRENT.get()
    return record is None or record.get("query_log") is not False


def set_value(key: str, value: Any) -> None:
    record = _CURRENT.get()
    if record is not None:
        record[key] = value


def add(key: str, amount: float = 1) -> None:
    record = _CURRENT.get()
    if record is not None:
        record[key] = record.get(key, 0) + amount


@contextmanager
def timed(key: str) -> Iterator[None]:
    """Add the wall time of the block, in milliseconds, to ``record[key]``."""
    record = _CURRENT.get()
    if record is None:
        yield
        return
    started = time.monotonic()
    try:
        yield
    finally:
        record[key] = record.get(key, 0.0) + (time.monotonic() - started) * 1000.0
