"""The meeting-event contract — the ONE shared shape for a call in the pipeline.

A processed call (transcript → LLM brief) enters Remembrancer as a Layer-1 `meeting`
event, exactly like git/slack/gmail/gcal. This module is the single source of
truth for that event's shape so the two sides never drift:

  - a meeting assistant imports `build_meeting_event(...)` to EMIT the event
    (in-process, same as it imports remembrancerlib);
  - the distiller imports `is_meeting_event` / `parse_meeting_event` to CONSUME it.

The event carries the already-distilled brief (summary, action items, decisions)
plus links back to the raw transcript — NEVER the full transcript itself, so the
committed events/*.jsonl stays small.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remembrancerlib import Event, event_text, restrict_sensitive_acl  # noqa: E402

SOURCE = "meeting"
DOMAIN = "business"          # meetings live under context/business/
_EXCERPT_CHARS = 1024        # bounded transcript snippet for BM25 only
_UNKNOWN_OCCURRED_AT = "1970-01-01T00:00:00Z"


def render_text(title: str, summary: str,
                action_items: list[str] | None,
                decisions: list[str] | None) -> str:
    """Canonical free-text for BM25 + clustering. `_event_text` in the engine
    reads payload['text'], so we pack the whole distilled brief in here."""
    parts = [title or "", summary or ""]
    ai = [a for a in (action_items or []) if str(a).strip()]
    de = [d for d in (decisions or []) if str(d).strip()]
    if de:
        parts.append("Decisions:\n" + "\n".join(f"- {d}" for d in de))
    if ai:
        parts.append("Action items:\n" + "\n".join(f"- {a}" for a in ai))
    return "\n\n".join(p for p in parts if p).strip()


def build_meeting_event(
    *,
    source_id: str,
    title: str,
    summary: str,
    attendees: list[str],
    action_items: list[str] | None = None,
    decisions: list[str] | None = None,
    started_at: str | None = None,          # ISO8601; clamped to now if in the future
    author: str = "",
    icaluid: str | None = None,             # links to the gcal skeleton event
    source_url: str | None = None,          # original Bluedot/Granola URL (fallback link)
    transcript_ref: str | None = None,      # stable deep-link to the served transcript
    excerpt: str | None = None,             # raw transcript head; truncated here, never full
    acl: list[str] | None = None,
    updated_at: str | None = None,           # stable producer/source revision time
    source_revision: str | None = None,      # optional upstream revision/version id
) -> Event:
    """Construct the canonical immutable `meeting` Event.

    The compatibility prefix remains ``bluedot:<source_id>`` while a content
    revision suffix distinguishes a corrected transcript/brief.  Exact replays
    produce the same id and fingerprint; changed summary, decisions, transcript
    reference, ``updated_at``, or upstream ``source_revision`` produce a new L1
    correction event.

    The ACL defaults to ``["eng"]`` and is then narrowed by
    :func:`sensitive_acl`: sensitive brief or excerpt text is labelled
    ``secrets`` whatever ``acl`` was passed.
    """
    attendees = [a.strip() for a in (attendees or []) if a and a.strip()]
    payload: dict[str, Any] = {
        "kind": "meeting",
        "title": title or "",
        "summary": summary or "",
        "text": render_text(title, summary, action_items, decisions),
        "attendees": attendees,
        "action_items": [a for a in (action_items or []) if str(a).strip()],
        "decisions": [d for d in (decisions or []) if str(d).strip()],
    }
    if excerpt:
        payload["excerpt"] = excerpt[:_EXCERPT_CHARS]

    provenance: dict[str, Any] = {"bluedot": source_id, "domain": DOMAIN}
    if icaluid:
        provenance["icaluid"] = icaluid
    if source_url:
        provenance["source_url"] = source_url
    if transcript_ref:
        provenance["transcript_ref"] = transcript_ref

    effective_acl = sensitive_acl(
        payload, list(acl) if acl is not None else ["eng"],
    )
    occurred = _stable_occurred_at(started_at, updated_at)
    revision = _meeting_revision(
        source_id=source_id,
        payload=payload,
        provenance=provenance,
        author=author or (attendees[0] if attendees else ""),
        acl=effective_acl,
        started_at=started_at,
        updated_at=updated_at,
        source_revision=source_revision,
    )
    provenance["revision"] = revision
    if updated_at:
        provenance["updated_at"] = updated_at
    if source_revision:
        provenance["source_revision"] = source_revision

    kwargs: dict[str, Any] = dict(
        id=f"bluedot:{source_id}@{revision}",
        source=SOURCE,
        author=author or (attendees[0] if attendees else ""),
        payload=payload,
        occurred_at=occurred,
        acl=effective_acl,
        entities=[],            # resolved by the distiller, never guessed at emit time
        provenance=provenance,
    )
    return Event(**kwargs)


def sensitive_acl(payload: dict[str, Any], acl: list[str]) -> list[str]:
    """Fail closed for sensitive meeting text.

    The same check the gcal, gdrive, git, gmail, slack and upload connectors
    run.  The crm, okf and artifact connectors still skip it.  The scanned text is ``event_text`` of the payload (the view distill and
    retrieval read, as in the Slack and git connectors) plus the stored
    transcript excerpt, which ``event_text`` does not read but the committed
    event still carries.  A match (salary, visa, termination, ...) relabels
    the event ``secrets``; an explicit ``acl`` argument is narrowed the same
    way, never widened.

    The result is what ``_meeting_revision`` binds, as in the Slack connector:
    a non-sensitive meeting keeps its ACL and therefore its id, while a replay
    of an already-filed sensitive meeting mints one new ``secrets`` revision.
    Older revisions stay as they were (Layer 1 is immutable).
    """
    text = event_text({"payload": payload})
    excerpt = str(payload.get("excerpt") or "")
    return restrict_sensitive_acl(f"{text}\n{excerpt}" if excerpt else text, acl)


def _stable_occurred_at(started_at: str | None, updated_at: str | None) -> str:
    """Choose a replay-stable, non-future occurrence time.

    Completed meeting producers normally provide ``started_at``.  For malformed
    future/missing starts, a valid past ``updated_at`` is the best correction
    timestamp.  A fixed sentinel is safer than Event's wall-clock default, which
    would make an exact replay conflict with its original immutable record.
    """
    now = datetime.now(timezone.utc)
    for value in (started_at, updated_at):
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        if parsed <= now:
            return str(value)
    return _UNKNOWN_OCCURRED_AT


def _meeting_revision(**identity: Any) -> str:
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def is_meeting_event(ev: Event) -> bool:
    return getattr(ev, "source", "") == SOURCE or (ev.payload or {}).get("kind") == "meeting"


def parse_meeting_event(ev: Event) -> dict[str, Any]:
    """Normalized view of a meeting event for the distiller. Always returns the
    full key set (empty defaults) so callers never KeyError."""
    p = ev.payload or {}
    prov = ev.provenance or {}
    return {
        "source_id": prov.get("bluedot", ""),
        "title": p.get("title", ""),
        "summary": p.get("summary", ""),
        "attendees": list(p.get("attendees", []) or []),
        "action_items": list(p.get("action_items", []) or []),
        "decisions": list(p.get("decisions", []) or []),
        "excerpt": p.get("excerpt", ""),
        "icaluid": prov.get("icaluid", ""),
        "source_url": prov.get("source_url", ""),
        "transcript_ref": prov.get("transcript_ref", ""),
        "occurred_at": ev.occurred_at,
    }
