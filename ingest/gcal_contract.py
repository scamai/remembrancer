"""Closed, replayable contract for Google Calendar Layer-1 snapshots.

Calendar objects are mutable and the same logical occurrence can be observed
through several attendee calendars.  This contract separates the stable source
fact (UID, occurrence, source update, answer payload, and visibility) from the
calendar copy that happened to expose it.  The connector, distiller, and source
lifecycle verifier all use this module so a legacy or merely calendar-shaped
event cannot acquire unattended cancellation authority.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import re
from typing import Any

from ingest.source_contract import ContractError, event_valid, field as _field


SNAPSHOT_FORMAT = "event-v1"
SINGLE_OCCURRENCE = "single"
PAYLOAD_KEYS = frozenset({
    "title", "status", "attendees", "attachments", "hangoutLink",
})
PROVENANCE_KEYS = frozenset({
    "snapshot_format", "revision", "calId", "eventId", "iCalUID",
    "occurrence", "updated", "domain", "htmlLink",
})
STATUSES = frozenset({"confirmed", "tentative", "cancelled"})
DOMAINS = frozenset({"customers", "internal"})

_REVISION_RE = re.compile(r"[a-f0-9]{24}\Z")
_LINEAGE_TOKEN_RE = re.compile(r"[^\s,\[\]\r\n]+\Z")


class GcalContractError(ContractError):
    """A purported Calendar source snapshot violates the closed contract."""


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _valid_occurrence(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    if value == SINGLE_OCCURRENCE:
        return True
    if _parse_timestamp(value) is not None:
        return True
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _valid_event_time(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    if _parse_timestamp(value) is not None:
        return True
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _string_list(value: Any, *, canonical: bool = False) -> bool:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in value
    ):
        return False
    return not canonical or value == sorted(set(value))


def revision_material(
    *, author: str, occurred_at: str, payload: dict[str, Any], acl: list[str],
    entities: list[str], ical_uid: str, occurrence: str, updated: str,
    domain: str,
) -> dict[str, Any]:
    """Return exactly the source/security material bound by a revision.

    ``calId``, the per-attendee Google ``eventId``, and ``htmlLink`` are
    observation metadata.  They intentionally do not participate: otherwise
    two attendee copies of the same source fact would collide under different
    event ids.  The authenticated L1 row still preserves the first observation.
    """
    return {
        "snapshot_format": SNAPSHOT_FORMAT,
        "author": author,
        "occurred_at": occurred_at,
        "payload": payload,
        "acl": list(acl),
        "entities": list(entities),
        "iCalUID": ical_uid,
        "occurrence": occurrence,
        "updated": updated,
        "domain": domain,
    }


def snapshot_revision(
    *, author: str, occurred_at: str, payload: dict[str, Any], acl: list[str],
    entities: list[str], ical_uid: str, occurrence: str, updated: str,
    domain: str,
) -> str:
    """Return the immutable id revision for one logical source snapshot."""
    raw = json.dumps(
        revision_material(
            author=author, occurred_at=occurred_at, payload=payload, acl=acl,
            entities=entities, ical_uid=ical_uid, occurrence=occurrence,
            updated=updated, domain=domain,
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def validate_gcal_event(event: Any) -> None:
    """Validate exact source identity, revision, provenance, and payload."""
    if _field(event, "source") != "gcal":
        raise GcalContractError("source must be gcal")
    event_id = _field(event, "id")
    author = _field(event, "author")
    occurred_at = _field(event, "occurred_at")
    payload = _field(event, "payload")
    provenance = _field(event, "provenance")
    acl = _field(event, "acl")
    entities = _field(event, "entities")
    if not isinstance(event_id, str) or not event_id:
        raise GcalContractError("event id must be a non-empty string")
    if (
        not isinstance(author, str)
        or not author.strip()
        or author != author.strip()
    ):
        raise GcalContractError("author must be a non-empty string")
    if not _valid_event_time(occurred_at):
        raise GcalContractError("occurred_at must be a Calendar date or timestamp")
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        raise GcalContractError("Calendar payload has an invalid closed schema")
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
        raise GcalContractError("Calendar provenance has an invalid closed schema")
    if not _string_list(acl, canonical=True):
        raise GcalContractError("Calendar ACL must be a canonical string list")
    if not _string_list(entities, canonical=True):
        raise GcalContractError("Calendar entities must be a canonical string list")

    title = payload.get("title")
    status = payload.get("status")
    attendees = payload.get("attendees")
    attachments = payload.get("attachments")
    hangout = payload.get("hangoutLink")
    if not isinstance(title, str) or not title.strip():
        raise GcalContractError("Calendar title must be non-empty")
    if status not in STATUSES:
        raise GcalContractError("Calendar status is unsupported")
    if not _string_list(attendees, canonical=True):
        raise GcalContractError("Calendar attendees must be canonical")
    if not _string_list(attachments, canonical=True):
        raise GcalContractError("Calendar attachments must be canonical")
    if hangout is not None and (
        not isinstance(hangout, str) or not hangout.strip()
    ):
        raise GcalContractError("hangoutLink must be null or non-empty")

    if provenance.get("snapshot_format") != SNAPSHOT_FORMAT:
        raise GcalContractError("unsupported Calendar snapshot format")
    revision = provenance.get("revision")
    cal_id = provenance.get("calId")
    source_event_id = provenance.get("eventId")
    ical_uid = provenance.get("iCalUID")
    occurrence = provenance.get("occurrence")
    updated = provenance.get("updated")
    domain = provenance.get("domain")
    html_link = provenance.get("htmlLink")
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise GcalContractError("invalid Calendar snapshot revision")
    if not isinstance(cal_id, str) or not cal_id.strip():
        raise GcalContractError("calId must identify the observation calendar")
    if not isinstance(source_event_id, str) or not source_event_id.strip():
        raise GcalContractError("eventId must identify the observed Google copy")
    if (
        not isinstance(ical_uid, str)
        or _LINEAGE_TOKEN_RE.fullmatch(ical_uid) is None
    ):
        raise GcalContractError("iCalUID must be a stable non-empty identity")
    if not _valid_occurrence(occurrence):
        raise GcalContractError("occurrence must be a Calendar date or timestamp")
    if _parse_timestamp(updated) is None:
        raise GcalContractError("updated must be a timezone-aware source timestamp")
    if domain not in DOMAINS:
        raise GcalContractError("invalid Calendar routing domain")
    if html_link is not None and (
        not isinstance(html_link, str) or not html_link.strip()
    ):
        raise GcalContractError("htmlLink must be null or non-empty")

    expected = snapshot_revision(
        author=author,
        occurred_at=occurred_at,
        payload=payload,
        acl=acl,
        entities=entities,
        ical_uid=ical_uid,
        occurrence=occurrence,
        updated=updated,
        domain=domain,
    )
    if revision != expected or event_id != f"gcal:{expected}":
        raise GcalContractError(
            "Calendar event id/revision does not bind the source snapshot"
        )


def gcal_event_valid(event: Any) -> bool:
    return event_valid(validate_gcal_event, event)


def gcal_lineage(event: Any) -> str:
    """Return the stable occurrence lineage only for a valid contract event."""
    if not gcal_event_valid(event):
        return ""
    provenance = _field(event, "provenance")
    return f"gcal:{provenance['iCalUID']}:{provenance['occurrence']}"


def gcal_source_updated(event: Any) -> datetime | None:
    """Return source update time only when the entire contract is valid."""
    if not gcal_event_valid(event):
        return None
    return _parse_timestamp(_field(event, "provenance")["updated"])
