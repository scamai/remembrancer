#!/usr/bin/env python3
"""Authenticate and advance the append-only L1 event prefix.

Every successful pass records the byte size, SHA-256, and line count of each
``events/*.jsonl`` file.  The next pass proves that prefix is unchanged and
applies schema, conflict, volume, and future-horizon policy to the physical
tail, independent of mutable timestamps.  The checkpoint advances atomically
only after the complete store passes while holding the event-store writer lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from remembrancerlib.event import (  # noqa: E402
    EVENT_CHECKPOINT_VERSION,
    EventValidationError,
    _event_fingerprint_dict,
    event_store_lock,
    event_validation_checkpoint_path,
    load_event_validation_checkpoint,
)
from remembrancerlib.locks import serialized_pipeline_mutation  # noqa: E402


REQUIRED_FIELDS = {
    "id", "source", "author", "occurred_at", "ingested_at", "payload",
    "acl", "entities", "provenance",
}
ALLOWED_SOURCES = {
    "git", "slack", "gmail", "gcal", "gdrive", "crm", "ci", "training",
    "session", "meeting", "artifact", "okf", "upload", "decision",
}
ID_PREFIX_ALIASES = {"meeting": {"meeting", "bluedot"}}


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def _schema_errors(record: Any, location: str) -> list[str]:
    if not isinstance(record, dict):
        return [f"{location}: event record must be an object"]
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        errors.append(f"{location}: missing fields {','.join(missing)}")
    for field in ("id", "source", "author", "occurred_at", "ingested_at"):
        if field in record and not isinstance(record[field], str):
            errors.append(f"{location}: {field} must be a string")
    for field in ("payload", "provenance"):
        if field in record and not isinstance(record[field], dict):
            errors.append(f"{location}: {field} must be an object")
    for field in ("acl", "entities"):
        value = record.get(field)
        if field in record and (
            not isinstance(value, list)
            or any(not isinstance(item, str) for item in value)
        ):
            errors.append(f"{location}: {field} must be a list of strings")
    for field in ("occurred_at", "ingested_at"):
        if field in record and _parse_time(record[field]) is None:
            errors.append(f"{location}: {field} must be a parseable ISO-8601 timestamp")
    return errors


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".corpus-guard-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(directory, os.O_DIRECTORY)
        except (AttributeError, OSError):  # pragma: no cover - non-POSIX fallback
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _checkpoint_path(root: str, checkpoint_path: str | None) -> str:
    if not checkpoint_path:
        return event_validation_checkpoint_path(os.path.join(root, "events"))
    if os.path.isabs(checkpoint_path):
        return checkpoint_path
    return os.path.join(root, checkpoint_path)


@serialized_pipeline_mutation()
def inspect(root: str, *, since: str | None = None, max_new_events: int = 5000,
            max_new_gcal: int = 1000, future_days: int = 180,
            checkpoint_path: str | None = None,
            bootstrap_checkpoint: bool = False) -> dict[str, Any]:
    """Validate L1 and atomically advance its authenticated byte checkpoint.

    ``bootstrap_checkpoint`` is an explicit attended migration for a store that
    has no checkpoint.  It performs full strict validation, treats existing ID
    conflicts as historical warnings, and does not apply tail volume/future
    budgets.  It never overwrites an existing malformed or broken checkpoint.
    """
    root = os.path.abspath(root)
    events_dir = os.path.join(root, "events")
    checkpoint_path = _checkpoint_path(root, checkpoint_path)
    since_at = _parse_time(since) if since is not None else None
    if since is not None and since_at is None:
        raise ValueError(f"invalid --since timestamp: {since!r}")
    now = datetime.now(timezone.utc)
    future_cutoff = now + timedelta(days=future_days)
    errors: list[str] = []
    warnings: list[str] = []
    new_by_source: Counter[str] = Counter()
    seen: dict[str, tuple[str, bool]] = {}
    duplicate_ids: set[str] = set()
    conflicting_ids: set[str] = set()
    new_conflicts: set[str] = set()
    new_future_gcal: list[str] = []
    records = 0
    new_records = 0
    bootstrapped = False
    checkpoint_advanced = False

    with event_store_lock(events_dir):
        checkpoint: dict[str, Any] | None = None
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = load_event_validation_checkpoint(checkpoint_path)
            except EventValidationError as exc:
                errors.append(str(exc))
        elif bootstrap_checkpoint:
            checkpoint = {
                "version": EVENT_CHECKPOINT_VERSION,
                "checked_at": None,
                "files": {},
            }
            bootstrapped = True
        else:
            errors.append(
                f"event validation checkpoint is missing: {checkpoint_path}; "
                "run with --bootstrap-checkpoint attended"
            )

        old_files = checkpoint.get("files", {}) if checkpoint is not None else {}
        if not os.path.isdir(events_dir):
            if bootstrap_checkpoint:
                live_filenames: list[str] = []
            else:
                errors.append("events directory is missing")
                live_filenames: list[str] = []
        else:
            live_filenames = sorted(
                name for name in os.listdir(events_dir) if name.endswith(".jsonl")
            )
        for filename in sorted(set(old_files) - set(live_filenames)):
            errors.append(f"validated event file was removed: {filename}")

        next_files: dict[str, dict[str, Any]] = {}
        for filename in live_filenames:
            path = os.path.join(events_dir, filename)
            with open(path, "rb") as handle:
                raw = handle.read()
            old_metadata = old_files.get(filename) or {}
            old_size = int(old_metadata.get("size", 0))
            if old_size > len(raw):
                errors.append(f"validated event file was truncated: {filename}")
                old_size = len(raw)
            elif filename in old_files:
                prefix = raw[:old_size]
                if prefix and not prefix.endswith(b"\n"):
                    errors.append(
                        f"validated prefix does not end at a record boundary: {filename}"
                    )
                if hashlib.sha256(prefix).hexdigest() != old_metadata.get("sha256"):
                    errors.append(f"validated event prefix was modified: {filename}")
                if prefix.count(b"\n") != old_metadata.get("lines"):
                    errors.append(f"validated event line count changed: {filename}")
            if raw and not raw.endswith(b"\n"):
                errors.append(f"{filename}: unterminated partial tail")

            next_files[filename] = {
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "lines": raw.count(b"\n"),
            }
            byte_offset = 0
            for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
                byte_offset += len(line)
                physical_tail = byte_offset > old_size
                is_new = physical_tail and checkpoint is not None and not bootstrapped
                location = f"{filename}:{line_number}"
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    errors.append(f"{location}: invalid JSON")
                    continue
                records += 1
                errors.extend(_schema_errors(record, location))
                if not isinstance(record, dict):
                    continue
                event_id_value = record.get("id")
                source_value = record.get("source")
                event_id = event_id_value if isinstance(event_id_value, str) else ""
                source = source_value if isinstance(source_value, str) else ""
                if is_new:
                    new_records += 1
                    new_by_source[source] += 1
                if not event_id or ":" not in event_id:
                    errors.append(f"{location}: invalid event id")
                if source not in ALLOWED_SOURCES:
                    errors.append(f"{location}: unsupported source {source!r}")
                prefix = event_id.split(":", 1)[0] if ":" in event_id else ""
                allowed_prefixes = ID_PREFIX_ALIASES.get(source, {source})
                if source and prefix not in allowed_prefixes:
                    errors.append(
                        f"{location}: id prefix {prefix!r} does not match source {source!r}"
                    )
                fingerprint = _event_fingerprint_dict(record)
                if event_id in seen:
                    duplicate_ids.add(event_id)
                    prior_fingerprint, prior_new = seen[event_id]
                    if fingerprint != prior_fingerprint:
                        conflicting_ids.add(event_id)
                        if is_new or prior_new:
                            new_conflicts.add(event_id)
                else:
                    seen[event_id] = (fingerprint, is_new)
                if is_new and source == "gcal":
                    occurred = _parse_time(record.get("occurred_at"))
                    if occurred is not None and occurred > future_cutoff:
                        new_future_gcal.append(event_id)

        new_total = sum(new_by_source.values())
        if new_total > max_new_events:
            errors.append(f"new event budget exceeded: {new_total}>{max_new_events}")
        if new_by_source["gcal"] > max_new_gcal:
            errors.append(f"new gcal budget exceeded: {new_by_source['gcal']}>{max_new_gcal}")
        if new_future_gcal:
            errors.append(
                f"{len(new_future_gcal)} new gcal event(s) exceed {future_days}-day horizon"
            )
        if new_conflicts:
            errors.append(f"{len(new_conflicts)} newly appended id(s) conflict with L1 history")
        if conflicting_ids:
            warnings.append(
                f"historical append-only store contains {len(conflicting_ids)} "
                "conflicting id(s); readers deterministically keep the first"
            )

        if not errors and checkpoint is not None:
            next_checkpoint = {
                "version": EVENT_CHECKPOINT_VERSION,
                "checked_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "records": records,
                "files": next_files,
            }
            _atomic_write_json(checkpoint_path, next_checkpoint)
            checkpoint_advanced = True

    return {
        "ok": not errors,
        "checked_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "since": (since_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                  if since_at is not None else None),
        "checkpoint": checkpoint_path,
        "checkpoint_advanced": checkpoint_advanced,
        "bootstrapped": bootstrapped and checkpoint_advanced,
        "records": records,
        "logical_events": len(seen),
        "duplicate_ids": len(duplicate_ids),
        "conflicting_ids": len(conflicting_ids),
        "new_events": new_records,
        "new_by_source": dict(sorted(new_by_source.items())),
        "new_future_gcal": len(new_future_gcal),
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--since", help="audit metadata only; tail detection uses checkpoint bytes")
    parser.add_argument("--checkpoint", default="state/event_validation_checkpoint.json")
    parser.add_argument(
        "--bootstrap-checkpoint", action="store_true",
        help="attended first-run migration: validate all history and create checkpoint",
    )
    parser.add_argument("--max-new-events", type=int, default=5000)
    parser.add_argument("--max-new-gcal", type=int, default=1000)
    parser.add_argument("--future-days", type=int, default=180)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args(argv)
    result = inspect(
        os.path.abspath(args.root), since=args.since,
        max_new_events=args.max_new_events, max_new_gcal=args.max_new_gcal,
        future_days=args.future_days, checkpoint_path=args.checkpoint,
        bootstrap_checkpoint=args.bootstrap_checkpoint,
    )
    if args.json_path:
        _atomic_write_json(args.json_path, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
