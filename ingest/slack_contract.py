"""Closed, replayable contract for Slack ``thread-v2`` Layer-1 events.

Both the connector and the verifier import this module.  Keeping revision,
identity, provenance, and payload validation behind one door prevents a forged
event that merely resembles a Slack payload from becoming source authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ingest.source_contract import ContractError, event_valid, field as _field


SNAPSHOT_FORMAT = "thread-v2"
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
REACTION_KEYS = frozenset({"message_ts", "name", "count"})
DELETION_EVIDENCE_KEYS = frozenset({"kind", "code"})
DELETION_EVIDENCE = frozenset({
    ("api-error", "message_not_found"),
    ("source-subtype", "message_deleted"),
})

_CHANNEL_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_CHANNEL_ID_RE = re.compile(r"[A-Z0-9]+\Z")
_THREAD_TS_RE = re.compile(r"[0-9]{1,20}\.[0-9]{1,20}\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{16}\Z")


class SlackContractError(ContractError):
    """A purported Slack source snapshot violates the closed contract."""


def snapshot_revision(
    author: str, payload: dict[str, Any], acl: list[str] | None = None,
) -> str:
    """Return the stable revision bound to the exact persisted source snapshot."""
    raw = json.dumps(
        {"author": author, "payload": payload, "acl": list(acl or [])},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def valid_thread_timestamp(value: Any) -> bool:
    """Whether ``value`` is a canonical Slack message/thread timestamp."""
    return isinstance(value, str) and _THREAD_TS_RE.fullmatch(value) is not None


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
        if (not valid_thread_timestamp(message_ts)
                or not isinstance(name, str) or not name.strip()
                or not _plain_int(count) or count <= 0):
            raise SlackContractError("resolution reaction has invalid values")
        canonical.append((message_ts, name, count))
    if canonical != sorted(canonical) or len(canonical) != len(set(canonical)):
        raise SlackContractError("resolution reactions must be unique and sorted")


def validate_thread_v2_event(event: Any) -> None:
    """Validate exact source identity, provenance, revision, and payload schema."""
    if _field(event, "source") != "slack":
        raise SlackContractError("source must be slack")
    event_id = _field(event, "id")
    author = _field(event, "author")
    provenance = _field(event, "provenance")
    payload = _field(event, "payload")
    if not isinstance(event_id, str) or not event_id:
        raise SlackContractError("event id must be a non-empty string")
    if not isinstance(author, str) or not author.strip():
        raise SlackContractError("author must be a non-empty string")
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
        raise SlackContractError("Slack provenance has an invalid schema")
    if not isinstance(payload, dict):
        raise SlackContractError("Slack payload must be an object")

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
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise SlackContractError("invalid Slack snapshot revision")
    if permalink is not None and (
            not isinstance(permalink, str) or not permalink.strip()):
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
    if (not isinstance(text, str) or not text.strip()
            or not text.startswith(f"[#{channel}] ")):
        raise SlackContractError("Slack text must be non-empty and channel-bound")
    if payload.get("snapshot_complete") is not True:
        raise SlackContractError("Slack snapshot is not complete")
    if payload.get("snapshot_format") != SNAPSHOT_FORMAT:
        raise SlackContractError("unsupported Slack snapshot format")
    if (not _plain_int(reply_count) or reply_count < 0
            or not _plain_int(message_count) or message_count < 0
            or not isinstance(payload.get("signal_eligible"), bool)):
        raise SlackContractError("invalid Slack snapshot counts or signal state")
    _validate_reactions(payload.get("resolution_reactions"))

    if deleted:
        evidence = payload.get("deletion_evidence")
        if (payload.get("subtype") != "thread_deleted"
                or reply_count != 0 or message_count != 0
                or payload.get("resolution_reactions") != []
                or payload.get("signal_eligible") is not False
                or not isinstance(evidence, dict)
                or set(evidence) != DELETION_EVIDENCE_KEYS
                or (evidence.get("kind"), evidence.get("code"))
                not in DELETION_EVIDENCE
                or text != f"[#{channel}] Slack thread {thread_ts} was deleted at the source."):
            raise SlackContractError("invalid Slack deletion snapshot")
    elif message_count != reply_count + 1:
        raise SlackContractError("message_count must equal root plus replies")

    event_acl = _field(event, "acl")
    if not isinstance(event_acl, list) or any(
        not isinstance(value, str) for value in event_acl
    ):
        raise SlackContractError("Slack ACL must be a list of strings")
    expected_revision = snapshot_revision(author, payload, event_acl)
    if revision != expected_revision:
        raise SlackContractError("Slack revision does not bind author and payload")


def thread_v2_event_valid(event: Any) -> bool:
    return event_valid(validate_thread_v2_event, event)


def thread_v2_text_authority(event: Any) -> bool:
    """Whether a valid snapshot may enter generic exact-text verification."""
    if not thread_v2_event_valid(event):
        return False
    payload = _field(event, "payload")
    return (
        payload.get("deleted") is not True
        and payload.get("resolution_reactions") == []
    )
