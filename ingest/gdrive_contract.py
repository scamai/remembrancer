"""Closed, replayable contract for Google Drive Layer-1 snapshots.

Drive files are mutable and an allowlisted file can disappear without leaving a
row in ``files.list``.  The connector therefore emits two kinds of immutable
snapshot: a live file revision and an inventory-backed tombstone.  Both bind the
stable file id, source time, answer-bearing payload, routing provenance, entity
projection, and final ACL into the event id.  A merely Drive-shaped legacy event
cannot acquire source-authoritative lifecycle powers through this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

from ingest.source_contract import ContractError, event_valid, field as _field


SNAPSHOT_FORMAT = "file-v1"
LIVE_STATUS = "active"
DELETED_STATUS = "unavailable"
DELETION_REASON = "absent-from-complete-allowlisted-inventory"
DOMAINS = frozenset({
    "customers", "prospect", "partners", "vendor", "investors",
    "advisors", "network", "recruiting", "research", "internal",
    "eng", "ml", "product", "business", "ops", "repos", "infra",
    "company", "people", "design", "synthesis",
})

LIVE_PAYLOAD_KEYS = frozenset({"title", "text", "status", "deleted"})
TOMBSTONE_PAYLOAD_KEYS = LIVE_PAYLOAD_KEYS | frozenset({"deletion_reason"})
PROVENANCE_KEYS = frozenset({
    "snapshot_format", "revision", "fileId", "source_time", "modifiedTime", "drive",
    "folder", "path", "mimeType", "webViewLink", "domain",
})

_REVISION_RE = re.compile(r"[a-f0-9]{24}\Z")
_FILE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,256}\Z")


class GdriveContractError(ContractError):
    """A purported Drive source snapshot violates the closed contract."""


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _canonical_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and item.strip() for item in value)
        and value == sorted(set(value))
    )


def revision_material(
    *, author: str, occurred_at: str, payload: dict[str, Any],
    provenance: dict[str, Any], acl: list[str], entities: list[str],
) -> dict[str, Any]:
    """Return the exact source/security projection covered by a revision."""
    stable_provenance = dict(provenance)
    stable_provenance.pop("revision", None)
    return {
        "snapshot_format": SNAPSHOT_FORMAT,
        "author": author,
        "occurred_at": occurred_at,
        "payload": payload,
        "provenance": stable_provenance,
        "acl": list(acl),
        "entities": list(entities),
    }


def snapshot_revision(
    *, author: str, occurred_at: str, payload: dict[str, Any],
    provenance: dict[str, Any], acl: list[str], entities: list[str],
) -> str:
    """Return the immutable revision for one exact Drive source snapshot."""
    raw = json.dumps(
        revision_material(
            author=author, occurred_at=occurred_at, payload=payload,
            provenance=provenance, acl=acl, entities=entities,
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def validate_gdrive_event(event: Any) -> None:
    """Validate exact source identity, revision, provenance, and payload."""
    if _field(event, "source") != "gdrive":
        raise GdriveContractError("source must be gdrive")
    event_id = _field(event, "id")
    author = _field(event, "author")
    occurred_at = _field(event, "occurred_at")
    payload = _field(event, "payload")
    provenance = _field(event, "provenance")
    acl = _field(event, "acl")
    entities = _field(event, "entities")

    if not isinstance(event_id, str) or not event_id:
        raise GdriveContractError("event id must be a non-empty string")
    if not isinstance(author, str) or not author.strip():
        raise GdriveContractError("author must be a non-empty string")
    if _timestamp(occurred_at) is None:
        raise GdriveContractError("occurred_at must be a timezone-aware timestamp")
    if not isinstance(payload, dict):
        raise GdriveContractError("Drive payload must be an object")
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
        raise GdriveContractError("Drive provenance has an invalid closed schema")
    if not _canonical_string_list(acl):
        raise GdriveContractError("Drive ACL must be a canonical string list")
    if not _canonical_string_list(entities):
        raise GdriveContractError("Drive entities must be a canonical string list")

    title = payload.get("title")
    text = payload.get("text")
    deleted = payload.get("deleted")
    if not isinstance(title, str) or not title.strip():
        raise GdriveContractError("Drive title must be non-empty")
    if not isinstance(text, str):
        raise GdriveContractError("Drive text must be a string")
    if not isinstance(deleted, bool):
        raise GdriveContractError("Drive deleted must be boolean")
    if deleted:
        if (
            set(payload) != TOMBSTONE_PAYLOAD_KEYS
            or payload.get("status") != DELETED_STATUS
            or payload.get("deletion_reason") != DELETION_REASON
            or text
        ):
            raise GdriveContractError("invalid Drive tombstone payload")
    elif (
        set(payload) != LIVE_PAYLOAD_KEYS
        or payload.get("status") != LIVE_STATUS
        or not text.strip()
    ):
        raise GdriveContractError("invalid live Drive payload")

    if provenance.get("snapshot_format") != SNAPSHOT_FORMAT:
        raise GdriveContractError("unsupported Drive snapshot format")
    revision = provenance.get("revision")
    file_id = provenance.get("fileId")
    source_time = provenance.get("source_time")
    modified_time = provenance.get("modifiedTime")
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise GdriveContractError("invalid Drive snapshot revision")
    if not isinstance(file_id, str) or _FILE_ID_RE.fullmatch(file_id) is None:
        raise GdriveContractError("fileId must be a canonical stable identity")
    if source_time != occurred_at or _timestamp(source_time) is None:
        raise GdriveContractError("source_time must equal the source occurrence time")
    if _timestamp(modified_time) is None:
        raise GdriveContractError("modifiedTime must be a source timestamp")
    if _timestamp(source_time) < _timestamp(modified_time):
        raise GdriveContractError("source_time cannot precede modifiedTime")
    for key in ("drive", "folder", "path", "mimeType", "domain"):
        value = provenance.get(key)
        if not isinstance(value, str):
            raise GdriveContractError(f"{key} must be a string")
    if (
        not provenance["drive"].strip()
        or not provenance["path"].strip()
        or not provenance["mimeType"].strip()
    ):
        raise GdriveContractError("drive, path, and mimeType must be non-empty")
    if provenance["domain"] not in DOMAINS:
        raise GdriveContractError("Drive domain is not a valid memory domain")
    link = provenance.get("webViewLink")
    if link is not None and (not isinstance(link, str) or not link.strip()):
        raise GdriveContractError("webViewLink must be null or non-empty")

    expected = snapshot_revision(
        author=author,
        occurred_at=occurred_at,
        payload=payload,
        provenance=provenance,
        acl=acl,
        entities=entities,
    )
    if revision != expected or event_id != f"gdrive:{expected}":
        raise GdriveContractError(
            "Drive event id/revision does not bind the source snapshot"
        )


def gdrive_event_valid(event: Any) -> bool:
    return event_valid(validate_gdrive_event, event)


def gdrive_lineage(event: Any) -> str:
    """Return a stable file lineage only for a valid contract event."""
    if not gdrive_event_valid(event):
        return ""
    return f"gdrive:{_field(event, 'provenance')['fileId']}"


def gdrive_source_time(event: Any) -> datetime | None:
    """Return source modification/deletion time for a valid contract event."""
    if not gdrive_event_valid(event):
        return None
    return _timestamp(_field(event, "provenance")["source_time"])
