"""Evidence-anchored freshness for retrieval rows.

A note is as fresh as the newest event it cites, not as fresh as the last tool
that rewrote its frontmatter.  An evidence-date builder derives that
date per note into ``state/evidence_dates.json`` (directly from the events a
note cites, or *derived* from the notes it supersedes, rolls up or links to
when it cites none); this module turns it into a tier against a per-type SLO
and falls back to ``updated`` — flagged as the weaker anchor — when a note has
no dateable evidence at all.  Anchor precedence: evidence > derived > updated.

Freshness and verification are orthogonal: a verified-but-stale note is both.
Nothing here ever raises on malformed dates; the worst case is ``unknown``.

**Settled notes.**  Most past-window notes are not out of date, they are
*finished*: a record of one meeting, one thread, one decision, with no newer
evidence anywhere in the store.  A refresh round attests those into
``state/settled_notes.json``; a row here adds ``settled``/``settled_reason``
(and ``successor`` when the topic continued in another note) to the freshness
dict so a reader can say "as of <date>, complete" instead of "stale".  The
attestation is **bound to the anchor it was made against**: it is honoured only
while the note's ``evidence_as_of`` still equals the attested date, so any newer
evidence silently voids it.  The tier vocabulary is deliberately unchanged —
settling relabels, it never hides a note from the stale counts.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os

DEFAULT_SLO_DAYS: dict[str, int] = {
    "state": 14,
    "project": 30, "artifact": 30, "entity": 30,
    "initiative": 45, "synthesis": 45, "page": 45, "source-summary": 45,
    "decision": 180, "reference": 180, "correction": 180,
}
FALLBACK_SLO_DAYS = 30
AGING_MULTIPLIER = 3
EVIDENCE_DATES_PATH = os.path.join("state", "evidence_dates.json")
SETTLED_NOTES_PATH = os.path.join("state", "settled_notes.json")
SETTLED_REASONS = ("no-newer-evidence", "superseded-in-practice",
                   "covered-elsewhere", "no-linkable-sources")


def freshness_cfg(cfg: dict) -> dict:
    """``retrieval.freshness`` with defaults: enabled, per-type SLO overrides."""
    block = ((cfg or {}).get("retrieval") or {}).get("freshness")
    if not isinstance(block, dict):
        block = {}
    slo = dict(DEFAULT_SLO_DAYS)
    overrides = block.get("slo_days")
    if isinstance(overrides, dict):
        for note_type, days in overrides.items():
            if isinstance(days, int) and not isinstance(days, bool) and days > 0:
                slo[str(note_type)] = days
    return {"enabled": block.get("enabled", True) is not False, "slo_days": slo}


def load_evidence_dates(root: str) -> dict:
    """``state/evidence_dates.json`` -> note name -> row; missing/malformed -> {}."""
    try:
        with open(os.path.join(root, EVIDENCE_DATES_PATH), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != 1:
        return {}
    notes = data.get("notes")
    return notes if isinstance(notes, dict) else {}


def load_settled(root: str) -> dict:
    """``state/settled_notes.json`` -> note name -> attestation; missing -> {}."""
    try:
        with open(os.path.join(root, SETTLED_NOTES_PATH), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != 1:
        return {}
    notes = data.get("notes")
    return notes if isinstance(notes, dict) else {}


def _parse_day(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def _today(now: datetime | date | None) -> date:
    if now is None:
        return datetime.now(timezone.utc).date()
    if isinstance(now, datetime):
        return now.date()
    return now


def _settled_fields(note, settled: dict | None, day: date) -> dict:
    """The ``settled`` block for *note*, or ``{}`` when no attestation binds it.

    An attestation is honoured only while the note's evidence anchor is still the
    one it was made against — newer evidence voids it without any bookkeeping.
    """
    row = (settled or {}).get(getattr(note, "name", ""))
    if not isinstance(row, dict):
        return {}
    if _parse_day(row.get("evidence_as_of")) != day:
        return {}
    reason = str(row.get("reason") or "")
    if reason not in SETTLED_REASONS:
        return {}
    out = {"settled": True, "settled_reason": reason}
    if row.get("settled_at"):
        out["settled_at"] = str(row["settled_at"])
    if row.get("successor"):
        out["successor"] = str(row["successor"])
    return out


def freshness_for(note, evidence_dates: dict, *, now: datetime | date | None = None,
                  cfg: dict | None = None, settled: dict | None = None) -> dict:
    """Return ``{tier, evidence_as_of, anchor, age_days, slo_days}`` for one note.

    Anchor precedence: the evidence row for ``note.name`` (``evidence`` when the
    note cites dateable events itself, ``derived`` when the builder anchored it
    on the notes it supersedes / rolls up / links to) → the note's own
    ``updated`` frontmatter → nothing (tier ``unknown``).  A derived row also
    passes ``rule`` and ``derived_from`` through so the caller can see the
    basis.  Tier: fresh while age ≤ SLO, aging while ≤ 3×SLO, stale beyond.
    """
    slo_days = freshness_cfg(cfg or {})["slo_days"]
    slo = slo_days.get(str(getattr(note, "type", "") or ""), FALLBACK_SLO_DAYS)

    anchor: str | None = None
    day: date | None = None
    basis: dict = {}
    row = (evidence_dates or {}).get(getattr(note, "name", ""))
    if isinstance(row, dict):
        day = _parse_day(row.get("evidence_as_of"))
        if day is not None:
            if row.get("anchor") == "derived":
                anchor = "derived"
                rule, sources = row.get("rule"), row.get("derived_from")
                if isinstance(rule, str) and rule:
                    basis["rule"] = rule
                if isinstance(sources, list) and sources:
                    basis["derived_from"] = [str(s) for s in sources]
            else:
                anchor = "evidence"
                # A future-only note is anchored on when Remembrancer learned of the
                # hold (the evidence-date builder's rule ``ingested``); pass
                # the basis through so "as of <date>" is not overstated.
                if row.get("rule") == "ingested":
                    basis["rule"] = "ingested"
                    if row.get("future_only") is True:
                        basis["future_only"] = True
    if day is None:
        day = _parse_day(getattr(note, "updated", ""))
        if day is not None:
            anchor = "updated"
    if day is None:
        return {"tier": "unknown", "evidence_as_of": None, "anchor": None,
                "age_days": None, "slo_days": slo}

    age = max(0, (_today(now) - day).days)
    if age <= slo:
        tier = "fresh"
    elif age <= AGING_MULTIPLIER * slo:
        tier = "aging"
    else:
        tier = "stale"
    return {"tier": tier, "evidence_as_of": day.isoformat(), "anchor": anchor,
            "age_days": age, "slo_days": slo, **basis,
            **_settled_fields(note, settled, day)}


class FreshnessContext:
    """Per-request bundle: evidence dates loaded once, applied to every row."""

    __slots__ = ("evidence_dates", "cfg", "now", "settled")

    def __init__(self, evidence_dates: dict, cfg: dict,
                 now: datetime | date | None = None, settled: dict | None = None):
        self.evidence_dates = evidence_dates
        self.cfg = cfg
        self.now = now
        self.settled = settled or {}

    @classmethod
    def load(cls, root: str, cfg: dict) -> "FreshnessContext | None":
        """``None`` when ``retrieval.freshness.enabled`` is false (callers skip)."""
        if not freshness_cfg(cfg)["enabled"]:
            return None
        return cls(load_evidence_dates(root), cfg, settled=load_settled(root))

    def for_note(self, note) -> dict:
        return freshness_for(note, self.evidence_dates, now=self.now, cfg=self.cfg,
                             settled=self.settled)
