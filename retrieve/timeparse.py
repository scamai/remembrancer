#!/usr/bin/env python3
"""Deterministic time-range extraction for time-aware retrieval (no LLM).

The retrieval path is deliberately LLM-free, so the LongMemEval time-aware
recipe (index by date; extract the time range from the query; filter) gets a
deterministic parser instead of a model call. `parse_time_range` lifts a time
phrase out of a query — "last week", "in July 2026", "since 2026-08-01",
"between 2026-07-01 and 2026-08-01", "recently" — and returns the range plus
the remaining query text for BM25.

Fail-closed: unrecognized or unparseable phrasing returns None and the query
runs untime-filtered. Relative phrases resolve against an explicit `today`
(injected for tests).

Ranges are matched against a note's ``updated`` — the distiller's write date,
not the date of the fact — so a phrase is only given a narrow window when it
genuinely names a past day (``yesterday``, ``on 2026-08-03``). ``today`` is an
as-of qualifier and resolves to an upper bound; see the note on it below.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_ISO = r"\d{4}-\d{2}(?:-\d{2})?"
_MIN_DATE = date(2000, 1, 1)


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    end = (date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)) - timedelta(days=1)
    return start, end


def _iso_start(value: str) -> date | None:
    """'2026-08' -> first of month; '2026-08-15' -> that day."""
    try:
        parts = value.split("-")
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None


def _iso_end(value: str) -> date | None:
    """'2026-08' -> last of month; '2026-08-15' -> that day."""
    try:
        parts = value.split("-")
        if len(parts) == 2:
            return _month_bounds(int(parts[0]), int(parts[1]))[1]
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None


def parse_time_range(query: str, today: date | None = None
                     ) -> tuple[date, date, str] | None:
    """Extract at most one time range from a query.

    Returns (start, end, cleaned_query) — both dates inclusive — or None when
    the query carries no recognizable time phrase.
    """
    today = today or date.today()
    q = query or ""

    def _clean(span) -> str:
        return re.sub(r"\s+", " ", (q[: span[0]] + " " + q[span[1]:])).strip()

    # between A and B / A..B
    m = re.search(rf"\bbetween\s+({_ISO})\s+and\s+({_ISO})\b", q, re.I)
    if not m:
        m = re.search(rf"({_ISO})\s*\.\.\s*({_ISO})", q)
    if m:
        start, end = _iso_start(m.group(1)), _iso_end(m.group(2))
        if start and end and start <= end:
            return start, end, _clean(m.span())
        return None

    # since / after / before / until <date>
    m = re.search(rf"\b(?:since|after)\s+({_ISO})\b", q, re.I)
    if m:
        start = _iso_start(m.group(1))
        if start:
            return start, today, _clean(m.span())
        return None
    m = re.search(rf"\b(?:before|until)\s+({_ISO})\b", q, re.I)
    if m:
        end = _iso_end(m.group(1))
        if end:
            return _MIN_DATE, end, _clean(m.span())
        return None
    # on <date> — a single day
    m = re.search(rf"\bon\s+({_ISO})\b", q, re.I)
    if m:
        start, end = _iso_start(m.group(1)), _iso_end(m.group(1))
        if start and end:
            return start, end, _clean(m.span())
        return None

    # in <month> [of] <year> / <month>[,] <year>
    m = re.search(r"\b(?:in\s+)?(" + "|".join(_MONTHS) + r")(?:\s+(?:of\s+)?|,\s*)(\d{4})\b", q, re.I)
    if not m:
        m = re.search(r"\bin\s+(" + "|".join(_MONTHS) + r")\b(?!\s+\d{4})", q, re.I)
    if m:
        month = _MONTHS[m.group(1).lower()]
        if m.lastindex == 2 and m.group(2):
            start, end = _month_bounds(int(m.group(2)), month)
        else:
            # bare "in July": the most recent July not in the future
            year = today.year if month <= today.month else today.year - 1
            start, end = _month_bounds(year, month)
        return start, end, _clean(m.span())

    # last N days
    m = re.search(r"\b(?:last|past)\s+(\d+)\s+days?\b", q, re.I)
    if m:
        n = min(3650, int(m.group(1)))
        return today - timedelta(days=n), today, _clean(m.span())

    # fixed relative phrases (checked after the parameterized ones)
    phrases = (
        (r"\bday before yesterday\b", today - timedelta(days=2),
         today - timedelta(days=2)),
        (r"\byesterday\b", today - timedelta(days=1), today - timedelta(days=1)),
        # "today" is an AS-OF qualifier, not a one-day window. Two reasons, and
        # the second is the decisive one:
        #  (a) in a question it almost always means "as things stand" -- "what
        #      accuracy does the parser get today", "where does X stand today";
        #      the point reading ("what shipped today") is the rarer one and is
        #      better served by the event/timeline surfaces, which carry real
        #      event dates.
        #  (b) the field this range filters is a note's `updated` -- the
        #      DISTILLER's write date, not the date of the fact. A same-day
        #      window on it selects "notes last night's pass happened to
        #      rewrite", which is never what any reading of "today" means, so
        #      the filter silently drops the whole topic and hands the query to
        #      whatever the pipeline touched most recently. Measured on a
        #      benchmark question ("What accuracy does the invoice parser get
        #      today..."): the answer note ranked first in the whole corpus and
        #      was cut by this filter before ranking, delivering an unrelated
        #      top 5.
        # `yesterday` and an explicit `on <date>` keep their point reading:
        # naming a specific past day is their only reading.
        (r"\btoday\b", _MIN_DATE, today),
        (r"\bthis week\b", today - timedelta(days=today.weekday()), today),
        (r"\b(?:last|past) week\b", today - timedelta(days=7), today),
        (r"\bthis month\b", today.replace(day=1), today),
        (r"\b(?:last|past) month\b",
         *_month_bounds(today.year - (1 if today.month == 1 else 0),
                        12 if today.month == 1 else today.month - 1)),
        (r"\b(?:recently|recent)\b", today - timedelta(days=30), today),
        (r"\bthis year\b", date(today.year, 1, 1), today),
        (r"\blast year\b", date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)),
    )
    for pattern, start, end in phrases:
        m = re.search(pattern, q, re.I)
        if m:
            return start, end, _clean(m.span())
    return None
