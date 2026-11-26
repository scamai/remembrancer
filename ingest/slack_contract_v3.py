"""Closed, replayable contract for Slack ``thread-v3`` Layer-1 events.

``thread-v2`` is an immutable legacy contract.  Its revision deliberately stays
unchanged in :mod:`ingest.slack_contract`, so already-persisted events continue
to validate.  V3 is a separate authority boundary: its revision binds every
field used to identify, order, route, or authorize a mutable Slack snapshot.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

from ingest.slack_contract import SlackContractError, valid_thread_timestamp
from ingest.source_contract import event_valid, field as _field


SNAPSHOT_FORMAT = "thread-v3"
SNAPSHOT_PAYLOAD_KEYS = frozenset({
    "text", "reply_count", "message_count", "snapshot_complete",
    "snapshot_format", "resolution_reactions", "signal_eligible",
})
TOMBSTONE_PAYLOAD_KEYS = SNAPSHOT_PAYLOAD_KEYS | frozenset({
    "deleted", "subtype", "deletion_evidence",
})
PROVENANCE_KEYS = frozenset({
    "channel", "channel_id", "ts", "thread_ts", "revision", "permalink",
})
PROVENANCE_MATERIAL_KEYS = PROVENANCE_KEYS - {"revision"}
REACTION_KEYS = frozenset({"message_ts", "name", "count"})
DELETION_EVIDENCE_KEYS = frozenset({"kind", "code"})
DELETION_EVIDENCE = frozenset({
    ("api-error", "message_not_found"),
    ("source-subtype", "message_deleted"),
})

# Slack permits ``-`` and ``_`` anywhere in a public-channel name.  V3 binds
# the exact API name into the event id/revision, so do not reject a real source
# name merely because punctuation is its first character.  The frozen V2
# contract intentionally keeps its narrower legacy rule.
_CHANNEL_RE = re.compile(r"[a-z0-9_-]+\Z")
_CHANNEL_ID_RE = re.compile(r"[A-Z0-9]+\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{64}\Z")


def _plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def source_occurred_at(thread_ts: str) -> str:
    """Return the connector's canonical UTC event time for a Slack root ts."""
    if not valid_thread_timestamp(thread_ts):
        raise SlackContractError("invalid Slack thread timestamp")
    try:
        seconds = int(thread_ts.split(".", 1)[0])
        occurred = datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise SlackContractError("Slack thread timestamp is outside datetime range") from exc
    return occurred.strftime("%Y-%m-%dT%H:%M:%SZ")


def revision_material(
    *,
    author: str,
    occurred_at: str,
    payload: dict[str, Any],
    provenance: dict[str, Any],
    acl: list[str],
    entities: list[str],
) -> dict[str, Any]:
    """Return the exact v3 material hashed into a persisted revision.

    ``ingested_at`` is observation metadata and is intentionally excluded.  The
    immutable source identity, source time, content, routing/lineage provenance,
    ACL, and entity projection are all included.  A caller may pass provenance
    before or after adding ``revision``; the self-referential field is removed.
    """
    provenance_without_revision = dict(provenance)
    provenance_without_revision.pop("revision", None)
    return {
        "author": author,
        "occurred_at": occurred_at,
        "payload": payload,
        "provenance": provenance_without_revision,
        "acl": list(acl),
        "entities": list(entities),
    }


def snapshot_revision(
    *,
    author: str,
    occurred_at: str,
    payload: dict[str, Any],
    provenance: dict[str, Any],
    acl: list[str],
    entities: list[str],
) -> str:
    """Return the stable v3 revision for the complete persisted source fact."""
    raw = json.dumps(
        revision_material(
            author=author,
            occurred_at=occurred_at,
            payload=payload,
            provenance=provenance,
            acl=acl,
            entities=entities,
        ),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_string_list(value: Any, label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SlackContractError(f"Slack {label} must be a list of strings")


def _validate_reactions(value: Any) -> None:
    if not isinstance(value, list):
        raise SlackContractError("resolution_reactions must be a list")
    canonical: list[tuple[str, str, int]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != REACTION_KEYS:
            raise SlackContractError("resolution reaction has an invalid schema")
        message_ts = row.get("message_ts")
        name = row.get("name")
        count = row.get("count")
        if (
            not valid_thread_timestamp(message_ts)
            or not isinstance(name, str)
            or not name.strip()
            or not _plain_int(count)
            or count <= 0
        ):
            raise SlackContractError("resolution reaction has invalid values")
        canonical.append((message_ts, name, count))
    if canonical != sorted(canonical) or len(canonical) != len(set(canonical)):
        raise SlackContractError("resolution reactions must be unique and sorted")


def validate_thread_v3_event(event: Any) -> None:
    """Validate the exact v3 identity, source time, revision, and closed schema."""
    if _field(event, "source") != "slack":
        raise SlackContractError("source must be slack")
    event_id = _field(event, "id")
    author = _field(event, "author")
    occurred_at = _field(event, "occurred_at")
    provenance = _field(event, "provenance")
    payload = _field(event, "payload")
    event_acl = _field(event, "acl")
    event_entities = _field(event, "entities")
    if not isinstance(event_id, str) or not event_id:
        raise SlackContractError("event id must be a non-empty string")
    if not isinstance(author, str) or not author.strip():
        raise SlackContractError("author must be a non-empty string")
    if not isinstance(occurred_at, str) or not occurred_at:
        raise SlackContractError("occurred_at must be a non-empty string")
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
        raise SlackContractError("Slack provenance has an invalid schema")
    if not isinstance(payload, dict):
        raise SlackContractError("Slack payload must be an object")
    _validate_string_list(event_acl, "ACL")
    _validate_string_list(event_entities, "entities")

    channel = provenance.get("channel")
    channel_id = provenance.get("channel_id")
    ts = provenance.get("ts")
    thread_ts = provenance.get("thread_ts")
    revision = provenance.get("revision")
    permalink = provenance.get("permalink")
    if not isinstance(channel, str) or _CHANNEL_RE.fullmatch(channel) is None:
        raise SlackContractError("invalid Slack channel name")
    if not isinstance(channel_id, str) or _CHANNEL_ID_RE.fullmatch(channel_id) is None:
        raise SlackContractError("invalid Slack channel id")
    if not valid_thread_timestamp(ts) or thread_ts != ts:
        raise SlackContractError("thread_ts and ts must be the same Slack timestamp")
    if occurred_at != source_occurred_at(thread_ts):
        raise SlackContractError("occurred_at does not match the Slack thread timestamp")
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise SlackContractError("invalid Slack snapshot revision")
    if permalink is not None and (
        not isinstance(permalink, str) or not permalink.strip()
    ):
        raise SlackContractError("permalink must be null or a non-empty string")
    if event_id != f"slack:{channel}/{thread_ts}@{revision}":
        raise SlackContractError("event id does not bind channel, thread, and revision")

    deleted = payload.get("deleted") is True
    expected_keys = TOMBSTONE_PAYLOAD_KEYS if deleted else SNAPSHOT_PAYLOAD_KEYS
    if set(payload) != expected_keys:
        raise SlackContractError("Slack payload has an invalid closed schema")
    text = payload.get("text")
    reply_count = payload.get("reply_count")
    message_count = payload.get("message_count")
    if (
        not isinstance(text, str)
        or not text.strip()
        or not text.startswith(f"[#{channel}] ")
    ):
        raise SlackContractError("Slack text must be non-empty and channel-bound")
    if payload.get("snapshot_complete") is not True:
        raise SlackContractError("Slack snapshot is not complete")
    if payload.get("snapshot_format") != SNAPSHOT_FORMAT:
        raise SlackContractError("unsupported Slack snapshot format")
    if (
        not _plain_int(reply_count)
        or reply_count < 0
        or not _plain_int(message_count)
        or message_count < 0
        or not isinstance(payload.get("signal_eligible"), bool)
    ):
        raise SlackContractError("invalid Slack snapshot counts or signal state")
    _validate_reactions(payload.get("resolution_reactions"))

    if deleted:
        evidence = payload.get("deletion_evidence")
        if (
            payload.get("subtype") != "thread_deleted"
            or reply_count != 0
            or message_count != 0
            or payload.get("resolution_reactions") != []
            or payload.get("signal_eligible") is not False
            or not isinstance(evidence, dict)
            or set(evidence) != DELETION_EVIDENCE_KEYS
            or (evidence.get("kind"), evidence.get("code")) not in DELETION_EVIDENCE
            or text != f"[#{channel}] Slack thread {thread_ts} was deleted at the source."
        ):
            raise SlackContractError("invalid Slack deletion snapshot")
    elif message_count != reply_count + 1:
        raise SlackContractError("message_count must equal root plus replies")

    expected_revision = snapshot_revision(
        author=author,
        occurred_at=occurred_at,
        payload=payload,
        provenance=provenance,
        acl=event_acl,
        entities=event_entities,
    )
    if revision != expected_revision:
        raise SlackContractError(
            "Slack revision does not bind the complete thread-v3 source fact"
        )


def thread_v3_event_valid(event: Any) -> bool:
    return event_valid(validate_thread_v3_event, event)


def thread_v3_lineage(event: Any) -> str:
    """Return an authority-bearing lineage only for an exact valid v3 event."""
    if not thread_v3_event_valid(event):
        return ""
    provenance = _field(event, "provenance")
    return f"slack:{provenance['channel_id']}:{provenance['thread_ts']}"


def thread_v3_text_authority(event: Any) -> bool:
    """Whether a valid v3 snapshot may enter generic exact-text verification."""
    if not thread_v3_event_valid(event):
        return False
    payload = _field(event, "payload")
    return (
        payload.get("deleted") is not True
        and payload.get("resolution_reactions") == []
    )
