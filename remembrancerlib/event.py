"""Layer-1 events: immutable, append-only, the substrate of truth.

One JSON object per line in events/YYYY-MM-DD.jsonl (partitioned by occurred_at
date), never edited after write. Bi-temporal: occurred_at = when it happened,
ingested_at = when Remembrancer learned it (lets us answer "what did we know when" and
handle retroactive corrections). Dedup key is `id` (source:natural-id).
"""
from __future__ import annotations

import json
import os
import hashlib
import fcntl
import re
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

from .locks import PipelineBusyError, pipeline_mutation_lock
from .redact import redact_event, redact_payload


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Event:
    """A single Layer-1 record."""
    id: str                                  # stable dedup key, "source:natural-id"
    source: str                              # git | slack | gmail | gcal | gdrive | crm | ci | training | session
    author: str                              # who produced it
    payload: dict[str, Any]                  # source-specific body
    occurred_at: str = field(default_factory=_utcnow_iso)   # when it happened (ISO8601 UTC)
    ingested_at: str = field(default_factory=_utcnow_iso)   # when Remembrancer learned it
    acl: list[str] = field(default_factory=list)            # visibility labels; [] = public
    entities: list[str] = field(default_factory=list)       # e.g. ["customer:acme"]
    provenance: dict[str, Any] = field(default_factory=dict)  # path back to ground truth

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Event":
        return Event(
            id=d["id"], source=d["source"], author=d.get("author", ""),
            payload=d.get("payload", {}),
            occurred_at=d.get("occurred_at") or d.get("ts") or _utcnow_iso(),
            ingested_at=d.get("ingested_at") or _utcnow_iso(),
            acl=list(d.get("acl", [])), entities=list(d.get("entities", [])),
            provenance=d.get("provenance", {}),
        )


def event_text(event: Event | dict, max_chars: int | None = None) -> str:
    """Return one source-agnostic text view used by distill and retrieval.

    Connectors intentionally keep source-shaped payloads, so consumers must not
    assume that useful text always lives in ``payload.text``.  Preserve field
    order, de-duplicate repeated values, and optionally cap prompt/log size.
    """
    payload = ((event.get("payload") if isinstance(event, dict)
                else getattr(event, "payload", None)) or {})
    values: list[Any] = [
        payload.get("subject"), payload.get("title"), payload.get("text"),
        payload.get("snippet"), payload.get("summary"), payload.get("body"),
        payload.get("description"), payload.get("message"), payload.get("name"),
    ]
    files = payload.get("files")
    if isinstance(files, list):
        values.append(" ".join(str(item) for item in files))
    seen: set[str] = set()
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
    result = "\n".join(parts)
    if max_chars is None or len(result) <= max_chars:
        return result
    # A head-only clip systematically drops conclusions, replies, and action
    # items that appear late in long Slack/Gmail payloads.  Preserve both ends
    # and make the omission explicit; callers that need to conclude "absent"
    # must separately account for the incomplete middle.
    marker = "\n…[middle omitted]…\n"
    if max_chars <= len(marker) + 2:
        return result[:max_chars]
    available = max_chars - len(marker)
    head = (available + 1) // 2
    tail = available - head
    return result[:head] + marker + result[-tail:]


def existing_ids(path: str) -> set[str]:
    """Ids already present in a day-file (for idempotent re-ingest)."""
    ids: set[str] = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    return ids


class EventConflictError(RuntimeError):
    """A supposedly stable event id was reused for different immutable data."""


#: The refusal while an event-conflict repair is unfinished.
_REPAIR_PENDING = ("unfinished event-conflict repair journal blocks new appends "
                   "(event_conflict_repair_transaction.json in the state folder)")


class EventSchemaError(ValueError):
    """An appender attempted to write an event outside the L1 contract."""


class EventValidationError(RuntimeError):
    """The validated L1 prefix is absent, corrupt, or no longer immutable."""


EVENT_CHECKPOINT_VERSION = 1

_EVENT_REQUIRED_FIELDS = {
    "id", "source", "author", "occurred_at", "ingested_at", "payload",
    "acl", "entities", "provenance",
}
_EVENT_ALLOWED_SOURCES = {
    "git", "slack", "gmail", "gcal", "gdrive", "crm", "ci", "training",
    "session", "meeting", "artifact", "okf", "upload", "decision",
}
_EVENT_ID_PREFIX_ALIASES = {"meeting": {"meeting", "bluedot"}}
_DAY_PREFIX = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _parse_iso_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def _prepare_event_for_append(event: Event, position: int) -> tuple[Event, dict[str, Any], str, str]:
    """Validate and fully serialize one event before any event file is opened."""
    location = f"incoming event #{position}"
    if not isinstance(event, Event):
        raise EventSchemaError(f"{location}: expected Event, got {type(event).__name__}")
    # Credentials pasted into a source (a key shared in Slack, a token in a
    # doc) must never reach the immutable, git-committed L1 log.
    event, _redacted = redact_event(event)
    data = asdict(event)
    missing = sorted(_EVENT_REQUIRED_FIELDS - set(data))
    errors: list[str] = []
    if missing:
        errors.append(f"missing fields {','.join(missing)}")
    for field_name in ("id", "source", "author", "occurred_at", "ingested_at"):
        if not isinstance(data.get(field_name), str):
            errors.append(f"{field_name} must be a string")
    for field_name in ("payload", "provenance"):
        if not isinstance(data.get(field_name), dict):
            errors.append(f"{field_name} must be an object")
    for field_name in ("acl", "entities"):
        value = data.get(field_name)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value or []):
            errors.append(f"{field_name} must be a list of strings")

    source = data.get("source") if isinstance(data.get("source"), str) else ""
    event_id = data.get("id") if isinstance(data.get("id"), str) else ""
    if source not in _EVENT_ALLOWED_SOURCES:
        errors.append(f"unsupported source {source!r}")
    if not event_id or ":" not in event_id:
        errors.append("id must be a non-empty source:natural-id")
    else:
        prefix = event_id.split(":", 1)[0]
        if source and prefix not in _EVENT_ID_PREFIX_ALIASES.get(source, {source}):
            errors.append(f"id prefix {prefix!r} does not match source {source!r}")

    for field_name in ("occurred_at", "ingested_at"):
        if _parse_iso_timestamp(data.get(field_name)) is None:
            errors.append(f"{field_name} must be a parseable ISO-8601 timestamp")
    occurred_at = data.get("occurred_at") if isinstance(data.get("occurred_at"), str) else ""
    if not _DAY_PREFIX.match(occurred_at):
        errors.append("occurred_at must start with an ISO calendar date")

    if errors:
        raise EventSchemaError(f"{location}: " + "; ".join(errors))

    day = occurred_at[:10]
    # Parsing above rejects impossible dates; this explicit basename contract is
    # defense in depth against path traversal and platform-specific separators.
    filename = f"{day}.jsonl"
    if filename != os.path.basename(filename) or os.sep in day or (os.altsep and os.altsep in day):
        raise EventSchemaError(f"{location}: occurred_at cannot select an event path")
    try:
        encoded = event.to_json()
    except Exception as exc:
        raise EventSchemaError(f"{location}: event is not JSON serializable") from exc
    return event, data, filename, encoded


def event_validation_checkpoint_path(events_dir: str) -> str:
    """Return the canonical validation checkpoint beside ``events/``."""
    root = os.path.dirname(os.path.abspath(events_dir))
    return os.path.join(root, "state", "event_validation_checkpoint.json")


def event_store_lock_path(events_dir: str) -> str:
    """Return the single lock shared by appenders and the corpus guard."""
    root = os.path.dirname(os.path.abspath(events_dir))
    return os.path.join(root, "state", "event-store.lock")


@contextmanager
def event_store_lock(events_dir: str) -> Iterator[None]:
    """Serialize an event-store snapshot with all :func:`append_events` calls."""
    lock_path = event_store_lock_path(events_dir)
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _event_fingerprint_dict(data: dict[str, Any]) -> str:
    stable = dict(data)
    # Compare on the redacted payload: an event stored before ingest-time
    # redaction existed still holds the raw credential, and re-fetching the
    # same source object now yields the redacted form.  That is the same
    # immutable fact, not an id conflict.
    stable["payload"], _redacted = redact_payload(stable.get("payload"))
    # Re-fetching the same source object naturally gets a new ingestion time;
    # it is still the same immutable source event.
    stable.pop("ingested_at", None)
    # A Gmail message can be observed in more than one Workspace mailbox.  The
    # observation mailbox is useful provenance, but is not part of the immutable
    # source fact identified by its RFC Message-ID.  Treat it like ingestion time
    # so cross-mailbox fan-out remains idempotent while the first observation stays
    # traceable in the stored event.
    if stable.get("source") == "gmail" and isinstance(stable.get("provenance"), dict):
        provenance = dict(stable["provenance"])
        provenance.pop("mailbox", None)
        stable["provenance"] = provenance
    # A closed-contract Calendar occurrence can be observed through several
    # attendees' copies.  The revision binds stable UID/occurrence, source
    # update, answer payload, ACL, entities, and routing; these three fields are
    # observation-only metadata and must not turn the same immutable source
    # fact into an id conflict on a later calendar shard.
    if stable.get("source") == "gcal" and isinstance(stable.get("provenance"), dict):
        provenance = dict(stable["provenance"])
        if provenance.get("snapshot_format") == "event-v1":
            for key in ("calId", "eventId", "htmlLink"):
                provenance.pop(key, None)
            stable["provenance"] = provenance
    raw = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def append_events(events: Iterable[Event], out_dir: str) -> int:
    """Append immutable events, enforcing corpus-wide id uniqueness.

    A repeated id with identical source data is an idempotent no-op. Reusing an
    id for different data is a contract violation and fails loudly rather than
    making provenance resolve differently in timeline/search/graph consumers.
    """
    pending = list(events)
    if not pending:
        return 0
    # Validate the complete batch—including JSON serialization and output
    # filenames—before acquiring locks or creating/opening any event file.
    prepared = [
        _prepare_event_for_append(event, position)
        for position, event in enumerate(pending, 1)
    ]
    root = os.path.dirname(os.path.abspath(out_dir))
    with pipeline_mutation_lock(root, blocking=False):
        # We hold the pipeline lock: this is the moment anything that was
        # spooled while it was held can safely land. Best-effort — a drain
        # failure must never lose the caller's own write.
        _drain_spool_best_effort(root, out_dir)
        return _append_events_under_pipeline_lock(prepared, out_dir)


def drain_spool_holding_pipeline_lock(root: str, out_dir: str) -> int:
    """Append every event waiting in the spool; the caller ALREADY owns ``pipeline.lock``.

    For a caller that holds the lock through its own descriptor (the isolated
    runner's ``_PipelineLock``): ``append_events`` would open a second
    descriptor on the same file, and flock treats that as a competing holder,
    so it would report the pipeline busy against itself.

    One scan of the store covers every spool file. A file the store refuses
    goes to the spool's quarantine and the others still land (see
    ``spool.drain_spool``). Raises when the store itself cannot take appends;
    every spool file then stays on disk.
    """
    from .spool import drain_spool

    return drain_spool(
        root, lambda batches: _append_batches_under_pipeline_lock(batches, out_dir),
    )


def _drain_spool_best_effort(root: str, out_dir: str) -> int:
    """Drain inside ``append_events``: a failure never costs the caller its write.

    It is logged, not swallowed: a drain that fails on every call would
    otherwise look exactly like an empty spool.
    """
    try:
        return drain_spool_holding_pipeline_lock(root, out_dir)
    except Exception as exc:
        print(
            f"[remembrancerlib] remember spool drain failed, spool kept: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 0


#: Present while a run that pinned its publish waits for the next run to
#: recover it by merging exactly that publish. An event appended to a live
#: partition in between makes the recovering run fail once: its late-merge
#: reconcile carries the event across the fast-forward, then the rebuild sees
#: the live dirty-path set change. The next run publishes it. If the pinned
#: publish appended to that same partition, the two share no prefix, and every
#: recovery refuses until a person repairs the partition. Writes spool instead.
_PUBLISH_PRECOMMIT_MARKER = "state/nightly_publish_precommit.json"


def _publish_recovery_pending(root: str) -> bool:
    return os.path.lexists(os.path.join(root, *_PUBLISH_PRECOMMIT_MARKER.split("/")))


def append_events_or_spool(events, out_dir: str) -> dict[str, int]:
    """Append if the pipeline is free, otherwise spool durably.

    Agent writes (``remembrancer_remember``) are append-only corrections, so deferring
    one is safe; refusing one loses it. Callers get ``{"appended": n,
    "spooled": m}`` instead of a PipelineBusyError. The write also spools
    while a pinned publish awaits recovery, with the lock free.
    """
    pending = list(events)
    if not pending:
        return {"appended": 0, "spooled": 0}
    root = os.path.dirname(os.path.abspath(out_dir))
    from .spool import spool_events
    try:
        # The publisher writes its marker under this same lock. Keep the
        # marker check and append inside one mutation transaction.
        with pipeline_mutation_lock(root, blocking=False):
            if not _publish_recovery_pending(root):
                return {"appended": append_events(pending, out_dir), "spooled": 0}
    except PipelineBusyError:
        pass
    return {"appended": 0, "spooled": spool_events(root, pending)}


def _refuse_unfinished_repair(out_dir: str) -> None:
    pending_repair = os.path.join(
        os.path.dirname(os.path.abspath(out_dir)), "state",
        "event_conflict_repair_transaction.json",
    )
    if os.path.exists(pending_repair):
        raise EventConflictError(_REPAIR_PENDING)


def _known_fingerprints(
        out_dir: str, incoming_ids: set[str],
        conflicted: set[str] | None = None) -> dict[str, str]:
    """Read every partition once: the stored fingerprint of each incoming id.

    Damage to the store (a partial tail, invalid JSON) always raises. An
    incoming id the store already holds as two different records raises too,
    unless ``conflicted`` is given: the id is then added to it for the caller
    to refuse, and the scan goes on.
    """
    known: dict[str, str] = {}
    for fn in sorted(f for f in os.listdir(out_dir) if f.endswith(".jsonl")):
        existing_path = os.path.join(out_dir, fn)
        with open(existing_path, "rb") as raw_file:
            raw = raw_file.read()
        if raw and not raw.endswith(b"\n"):
            raise EventConflictError(f"event file has an unterminated partial tail: {existing_path}")
        with open(existing_path, encoding="utf-8") as existing_file:
            for line_number, line in enumerate(existing_file, 1):
                try:
                    data = json.loads(line)
                except Exception as exc:
                    raise EventConflictError(
                        f"event file contains invalid JSON at {existing_path}:{line_number}"
                    ) from exc
                event_id = data.get("id")
                if event_id in incoming_ids:
                    fingerprint = _event_fingerprint_dict(data)
                    previous = known.setdefault(event_id, fingerprint)
                    if previous != fingerprint:
                        if conflicted is None:
                            raise EventConflictError(
                                f"existing event id {event_id!r} already has conflicting records"
                            )
                        conflicted.add(event_id)
    return known


def _plan_append(
        prepared: list[tuple[Event, dict[str, Any], str, str]],
        fingerprints: dict[str, str]) -> dict[str, list[str]]:
    """Group a batch's new events by partition, skipping exact repeats.

    ``fingerprints`` gains every id the batch adds. Raises when an id is reused
    for different immutable data.
    """
    by_file: dict[str, list[str]] = {}
    for event, data, filename, encoded in prepared:
        fingerprint = _event_fingerprint_dict(data)
        if event.id in fingerprints:
            if fingerprints[event.id] != fingerprint:
                raise EventConflictError(
                    f"event id {event.id!r} was reused for different immutable data"
                )
            continue
        fingerprints[event.id] = fingerprint
        by_file.setdefault(filename, []).append(encoded)
    return by_file


def _write_planned(out_dir: str, by_file: dict[str, list[str]]) -> int:
    written = 0
    for filename, encoded_events in by_file.items():
        path = os.path.join(out_dir, filename)
        if os.path.commonpath((os.path.realpath(out_dir), os.path.realpath(path))) != os.path.realpath(out_dir):
            raise EventSchemaError(f"event output escapes events directory: {filename}")
        with open(path, "a", encoding="utf-8") as output:
            for encoded in encoded_events:
                output.write(encoded + "\n")
                written += 1
            output.flush()
            os.fsync(output.fileno())
    return written


def _append_events_under_pipeline_lock(
        prepared: list[tuple[Event, dict[str, Any], str, str]], out_dir: str) -> int:
    """Append a materialized batch while the caller owns ``pipeline.lock``."""
    os.makedirs(out_dir, exist_ok=True)
    with event_store_lock(out_dir):
        _refuse_unfinished_repair(out_dir)
        incoming_ids = {event.id for event, _data, _filename, _encoded in prepared}
        known = _known_fingerprints(out_dir, incoming_ids)
        by_file = _plan_append(prepared, dict(known))
        return _write_planned(out_dir, by_file)


def _append_batches_under_pipeline_lock(
        batches: list[list[Event]], out_dir: str) -> list[Exception | None]:
    """Append several batches after ONE scan of the store; the caller owns ``pipeline.lock``.

    Each batch lands whole or not at all, in order. A batch that breaks the L1
    contract by itself -- an event outside the schema, or an id reused for
    different data (against the store, an earlier batch, or itself) -- is
    refused: its slot in the returned list holds the error, and the other
    batches still land. ``None`` means the batch landed or was already there.
    Damage to the store itself (an unfinished repair, a partial tail, invalid
    JSON) and I/O errors raise, and then no batch is written.
    """
    refusals: list[Exception | None] = [None] * len(batches)
    prepared_batches: dict[int, list[tuple[Event, dict[str, Any], str, str]]] = {}
    for index, batch in enumerate(batches):
        try:
            prepared_batches[index] = [
                _prepare_event_for_append(event, position)
                for position, event in enumerate(batch, 1)
            ]
        except Exception as exc:
            refusals[index] = exc
    incoming_ids = {
        event.id
        for prepared in prepared_batches.values()
        for event, _data, _filename, _encoded in prepared
    }
    if not incoming_ids:
        return refusals
    os.makedirs(out_dir, exist_ok=True)
    with event_store_lock(out_dir):
        _refuse_unfinished_repair(out_dir)
        conflicted: set[str] = set()
        fingerprints = _known_fingerprints(out_dir, incoming_ids, conflicted)
        by_file: dict[str, list[str]] = {}
        for index, prepared in prepared_batches.items():
            stored_conflicts = sorted(
                {event.id for event, _data, _filename, _encoded in prepared} & conflicted
            )
            if stored_conflicts:
                refusals[index] = EventConflictError(
                    f"existing event id {stored_conflicts[0]!r} already has conflicting records"
                )
                continue
            trial = dict(fingerprints)
            try:
                planned = _plan_append(prepared, trial)
            except EventConflictError as exc:
                refusals[index] = exc
                continue
            fingerprints = trial
            for filename, encoded_events in planned.items():
                by_file.setdefault(filename, []).extend(encoded_events)
        _write_planned(out_dir, by_file)
    return refusals


def _iter_all_events(events_dir: str) -> Iterator[Event]:
    """Yield each logical event once in stable file/line order.

    ``id`` is the corpus-wide identity key.  Older connector versions wrote a
    small number of the same logical ids more than once (most notably Gmail
    thread ids).  L1 remains append-only, so readers resolve that historical
    overlap deterministically by keeping the first record.  New writes are
    protected by :func:`append_events`, which rejects conflicting reuse.
    """
    if not os.path.isdir(events_dir):
        return
    seen: set[str] = set()
    for fn in sorted(f for f in os.listdir(events_dir) if f.endswith(".jsonl")):
        with open(os.path.join(events_dir, fn)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = Event.from_dict(json.loads(line))
                except Exception:
                    continue
                if ev.id in seen:
                    continue
                seen.add(ev.id)
                yield ev


def load_event_validation_checkpoint(path: str) -> dict[str, Any]:
    """Load and strictly validate a corpus-guard checkpoint.

    The checkpoint is a capability boundary, not a best-effort cache.  A
    missing or malformed file therefore fails closed instead of falling back
    to the live (possibly unvalidated) JSONL tails.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            checkpoint = json.load(handle)
    except FileNotFoundError as exc:
        raise EventValidationError(
            f"event validation checkpoint is missing: {path}; "
            "run corpus_guard.py --bootstrap-checkpoint attended"
        ) from exc
    except Exception as exc:
        raise EventValidationError(
            f"event validation checkpoint is unreadable: {path}"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise EventValidationError("event validation checkpoint must be an object")
    if checkpoint.get("version") != EVENT_CHECKPOINT_VERSION:
        raise EventValidationError(
            f"unsupported event validation checkpoint version: "
            f"{checkpoint.get('version')!r}"
        )
    files = checkpoint.get("files")
    if not isinstance(files, dict):
        raise EventValidationError("event validation checkpoint 'files' must be an object")
    for filename, metadata in files.items():
        if (not isinstance(filename, str) or filename != os.path.basename(filename)
                or not filename.endswith(".jsonl")):
            raise EventValidationError(f"invalid checkpoint event filename: {filename!r}")
        if not isinstance(metadata, dict):
            raise EventValidationError(f"checkpoint metadata for {filename} must be an object")
        size = metadata.get("size")
        lines = metadata.get("lines")
        digest = metadata.get("sha256")
        if (not isinstance(size, int) or isinstance(size, bool) or size < 0
                or not isinstance(lines, int) or isinstance(lines, bool) or lines < 0
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise EventValidationError(f"invalid checkpoint metadata for {filename}")
    return checkpoint


def _validated_prefix_bytes(events_dir: str, checkpoint_path: str | None = None
                            ) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Read and authenticate every byte prefix named by the checkpoint."""
    checkpoint_path = checkpoint_path or event_validation_checkpoint_path(events_dir)
    checkpoint = load_event_validation_checkpoint(checkpoint_path)
    prefixes: dict[str, bytes] = {}
    for filename, metadata in sorted(checkpoint["files"].items()):
        path = os.path.join(events_dir, filename)
        try:
            with open(path, "rb") as handle:
                raw = handle.read(metadata["size"])
        except FileNotFoundError as exc:
            raise EventValidationError(
                f"validated event file was removed: {filename}"
            ) from exc
        if len(raw) != metadata["size"]:
            raise EventValidationError(
                f"validated event file was truncated: {filename}"
            )
        if raw and not raw.endswith(b"\n"):
            raise EventValidationError(
                f"validated prefix does not end at a record boundary: {filename}"
            )
        digest = hashlib.sha256(raw).hexdigest()
        if digest != metadata["sha256"]:
            raise EventValidationError(
                f"validated event prefix was modified: {filename}"
            )
        if raw.count(b"\n") != metadata["lines"]:
            raise EventValidationError(
                f"validated event line count changed: {filename}"
            )
        prefixes[filename] = raw
    return prefixes, checkpoint


def event_validation_status(events_dir: str, checkpoint_path: str | None = None
                            ) -> dict[str, Any]:
    """Report whether the live store is exactly at, ahead of, or invalid vs checkpoint."""
    checkpoint_path = checkpoint_path or event_validation_checkpoint_path(events_dir)
    if not os.path.isdir(events_dir):
        return {
            "ok": False, "current": False, "behind": False,
            "errors": [f"events directory is missing: {events_dir}"],
        }
    try:
        _prefixes, checkpoint = _validated_prefix_bytes(events_dir, checkpoint_path)
    except EventValidationError as exc:
        return {"ok": False, "current": False, "behind": False, "errors": [str(exc)]}
    checkpoint_files = checkpoint["files"]
    live_files = (sorted(name for name in os.listdir(events_dir) if name.endswith(".jsonl"))
                  if os.path.isdir(events_dir) else [])
    behind_files: list[str] = []
    errors: list[str] = []
    for filename in live_files:
        size = os.path.getsize(os.path.join(events_dir, filename))
        checked_size = int((checkpoint_files.get(filename) or {}).get("size", 0))
        if size < checked_size:
            errors.append(f"validated event file was truncated: {filename}")
        elif size > checked_size:
            behind_files.append(filename)
    for filename in checkpoint_files:
        if filename not in live_files:
            errors.append(f"validated event file was removed: {filename}")
    return {
        "ok": not errors,
        "current": not errors and not behind_files,
        "behind": not errors and bool(behind_files),
        "behind_files": behind_files,
        "errors": errors,
        "checked_at": checkpoint.get("checked_at"),
    }


def iter_validated_event_records(events_dir: str, checkpoint_path: str | None = None
                                 ) -> Iterator[Event]:
    """Yield every physical event record inside the authenticated prefixes.

    Unlike :func:`iter_validated_events`, this preserves historical duplicate
    ids so integrity-sensitive consumers (for example citation validation) can
    reject an ambiguous legacy id rather than silently accepting first-wins.
    """
    prefixes, _checkpoint = _validated_prefix_bytes(events_dir, checkpoint_path)
    for filename, raw in sorted(prefixes.items()):
        for line_number, line in enumerate(raw.splitlines(), 1):
            try:
                data = json.loads(line)
                event = Event.from_dict(data)
            except Exception as exc:
                raise EventValidationError(
                    f"validated event is invalid at {filename}:{line_number}"
                ) from exc
            yield event


def _iter_validated_all_events(events_dir: str, checkpoint_path: str | None = None
                               ) -> Iterator[Event]:
    seen: set[str] = set()
    for event in iter_validated_event_records(events_dir, checkpoint_path):
        if event.id in seen:
            continue
        seen.add(event.id)
        yield event


def iter_validated_events(events_dir: str, after_id: str | None = None,
                          checkpoint_path: str | None = None) -> Iterator[Event]:
    """Yield only the authenticated byte prefixes approved by corpus guard.

    Appenders may safely add bytes while this iterator runs: the checkpointed
    prefix is immutable, and any later tail is intentionally invisible until a
    subsequent successful guard pass advances the checkpoint.
    """
    if after_id is None:
        yield from _iter_validated_all_events(events_dir, checkpoint_path)
        return

    found = False
    for event in _iter_validated_all_events(events_dir, checkpoint_path):
        if found:
            yield event
        elif event.id == after_id:
            found = True
    if not found:
        warnings.warn(
            f"event watermark {after_id!r} was not found in validated prefix; "
            "replaying all validated events",
            RuntimeWarning,
            stacklevel=2,
        )
        yield from _iter_validated_all_events(events_dir, checkpoint_path)


def iter_events(events_dir: str, after_id: str | None = None) -> Iterator[Event]:
    """Yield events in stable file/line order, optionally after a watermark.

    A cursor can outlive its event after a restore, retention change, or manual
    repair.  Treating an unknown cursor as "already at EOF" silently wedges the
    distiller forever.  If ``after_id`` is absent, warn and replay the store;
    the successful pass will advance the caller's cursor back into this store.
    """
    if after_id is None:
        yield from _iter_all_events(events_dir)
        return

    found = False
    for ev in _iter_all_events(events_dir):
        if found:
            yield ev
        elif ev.id == after_id:
            found = True

    if not found:
        warnings.warn(
            f"event watermark {after_id!r} was not found; replaying all events",
            RuntimeWarning,
            stacklevel=2,
        )
        yield from _iter_all_events(events_dir)
