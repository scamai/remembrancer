"""Content-bound deterministic retirement for authoritative source deletion.

Human approval and source lifecycle are intentionally separate authorities.  A
human may approve/reject any memory; an unattended producer may retire only a
machine-owned note after a closed source contract reports an explicit deletion.
Both durable copies (note frontmatter and an append-only journal) bind the exact
answer-relevant content and authenticated source event.  Raw ``state`` text is
never sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import re
import stat
from typing import Any


SOURCE_RETIREMENT_VERSION = 1
_PRODUCER_SOURCES = {
    "gcal-cancellation-v1": "gcal",
    "slack-thread-deletion-v1": "slack",
    "slack-thread-v3-deletion-v1": "slack",
    "gdrive-file-deletion-v1": "gdrive",
}
_ACTIVE_PRODUCERS = frozenset({
    "gcal-cancellation-v1",
    "slack-thread-v3-deletion-v1",
    "gdrive-file-deletion-v1",
})
_SHA256_RE = re.compile(r"[a-f0-9]{64}\Z")
_NOTE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
_MAX_AUDIT_BYTES = 32 * 1024 * 1024
_MAX_AUDIT_LINE_BYTES = 256 * 1024
_WRITE_SEAL = object()


@dataclass(frozen=True)
class SourceLifecycleAudit:
    latest: dict[str, dict]
    trustworthy: bool = True


@dataclass(frozen=True)
class _SourceLifecycleWriteCapability:
    note: str
    state: str
    transition_id: str
    content_fingerprint: str
    preimage_fingerprint: str
    _seal: object


def _canonical_hash(payload: dict) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _event_fingerprint(event: Any) -> str:
    serialized = event.to_json()
    if not isinstance(serialized, str) or not serialized:
        raise ValueError("source lifecycle event is not canonically serializable")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _source_preimage_fingerprint(note: Any) -> str:
    """Bind the exact persisted note before a retirement control write."""
    from .verification import verification_content_fingerprint

    proof = getattr(note, "source_retirement", None)
    if proof is not None and not isinstance(proof, dict):
        raise ValueError("source retirement preimage proof is not an object")
    return _canonical_hash({
        "content_fingerprint": verification_content_fingerprint(note),
        "state": getattr(note, "state", None),
        "superseded": getattr(note, "superseded", None),
        "source_retirement": proof,
    })


def _retirement_material(
    *, note: str, content_fingerprint: str, source: str, lineage: str,
    event_id: str, event_fingerprint: str, producer: str,
) -> dict:
    return {
        "version": SOURCE_RETIREMENT_VERSION,
        "note": note,
        "content_fingerprint": content_fingerprint,
        "source": source,
        "lineage": lineage,
        "event_id": event_id,
        "event_fingerprint": event_fingerprint,
        "producer": producer,
    }


def _valid_material(material: dict) -> bool:
    return bool(
        material.get("version") == SOURCE_RETIREMENT_VERSION
        and isinstance(material.get("note"), str)
        and _NOTE_NAME_RE.fullmatch(material["note"])
        and isinstance(material.get("content_fingerprint"), str)
        and _SHA256_RE.fullmatch(material["content_fingerprint"])
        and material.get("source") in set(_PRODUCER_SOURCES.values())
        and isinstance(material.get("lineage"), str)
        and material["lineage"].startswith(material["source"] + ":")
        and isinstance(material.get("event_id"), str)
        and material["event_id"].startswith(material["source"] + ":")
        and isinstance(material.get("event_fingerprint"), str)
        and _SHA256_RE.fullmatch(material["event_fingerprint"])
        and _PRODUCER_SOURCES.get(material.get("producer")) == material.get("source")
    )


def _embedded_retirement(note: Any) -> dict | None:
    proof = getattr(note, "source_retirement", None)
    expected = {
        "version", "retirement_id", "content_fingerprint", "source",
        "lineage", "event_id", "event_fingerprint", "producer",
    }
    if not isinstance(proof, dict) or set(proof) != expected:
        return None
    from .verification import verification_content_fingerprint

    material = _retirement_material(
        note=str(getattr(note, "name", "") or ""),
        content_fingerprint=str(proof.get("content_fingerprint") or ""),
        source=str(proof.get("source") or ""),
        lineage=str(proof.get("lineage") or ""),
        event_id=str(proof.get("event_id") or ""),
        event_fingerprint=str(proof.get("event_fingerprint") or ""),
        producer=str(proof.get("producer") or ""),
    )
    if (
        not _valid_material(material)
        or material["producer"] not in _ACTIVE_PRODUCERS
        or proof.get("version") != SOURCE_RETIREMENT_VERSION
        or proof.get("content_fingerprint") != verification_content_fingerprint(note)
        or proof.get("retirement_id") != _canonical_hash(material)
        or getattr(note, "state", None) != "superseded"
        or getattr(note, "superseded", None) is not True
    ):
        return None
    return proof


def _parse_audit(raw: bytes) -> SourceLifecycleAudit:
    if len(raw) > _MAX_AUDIT_BYTES or (raw and not raw.endswith(b"\n")):
        return SourceLifecycleAudit({}, trustworthy=False)
    latest: dict[str, dict] = {}
    for raw_line in raw.splitlines():
        if not raw_line or len(raw_line) > _MAX_AUDIT_LINE_BYTES:
            return SourceLifecycleAudit({}, trustworthy=False)
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return SourceLifecycleAudit({}, trustworthy=False)
        if not isinstance(row, dict) or row.get("action") not in {"retire", "restore"}:
            return SourceLifecycleAudit({}, trustworthy=False)
        material = {
            key: row.get(key)
            for key in (
                "version", "note", "content_fingerprint", "source", "lineage",
                "event_id", "event_fingerprint", "producer",
            )
        }
        if not _valid_material(material):
            return SourceLifecycleAudit({}, trustworthy=False)
        transition_id = row.get("transition_id")
        prior_retirement = row.get("retirement_id")
        expected = _canonical_hash({
            **material,
            "action": row["action"],
            "retirement_id": prior_retirement or "",
            "target_content_fingerprint": row.get("target_content_fingerprint"),
        })
        target_fingerprint = row.get("target_content_fingerprint")
        if (
            not isinstance(transition_id, str)
            or transition_id != expected
            or not isinstance(target_fingerprint, str)
            or _SHA256_RE.fullmatch(target_fingerprint) is None
            or (row["action"] == "retire"
                and target_fingerprint != material["content_fingerprint"])
            or (row["action"] == "retire" and prior_retirement != _canonical_hash(material))
            or (row["action"] == "restore"
                and (not isinstance(prior_retirement, str)
                     or _SHA256_RE.fullmatch(prior_retirement) is None))
        ):
            return SourceLifecycleAudit({}, trustworthy=False)
        latest[material["note"]] = row
    return SourceLifecycleAudit(latest)


def load_source_lifecycle_audit(root: str) -> SourceLifecycleAudit:
    from .locks import open_repository_state_directory

    try:
        state_descriptor, _created = open_repository_state_directory(
            root, create=False,
        )
    except OSError:
        return SourceLifecycleAudit({}, trustworthy=False)
    if state_descriptor < 0:
        return SourceLifecycleAudit({})
    try:
        descriptor = os.open(
            "source_lifecycle.jsonl",
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=state_descriptor,
        )
    except FileNotFoundError:
        os.close(state_descriptor)
        return SourceLifecycleAudit({})
    except OSError:
        os.close(state_descriptor)
        return SourceLifecycleAudit({}, trustworthy=False)
    try:
        with os.fdopen(descriptor, "rb") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                metadata = os.fstat(handle.fileno())
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size > _MAX_AUDIT_BYTES
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                ):
                    return SourceLifecycleAudit({}, trustworthy=False)
                raw = handle.read(_MAX_AUDIT_BYTES + 1)
            except OSError:
                return SourceLifecycleAudit({}, trustworthy=False)
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
    finally:
        os.close(state_descriptor)
    return _parse_audit(raw)


def source_retirement_valid(
    note: Any, audit: SourceLifecycleAudit | None = None,
) -> bool:
    proof = _embedded_retirement(note)
    if proof is None:
        return False
    if audit is None:
        return bool(getattr(note, "_source_retirement_audit_match", False))
    if not audit.trustworthy:
        return False
    row = audit.latest.get(str(getattr(note, "name", "") or ""))
    return bool(
        isinstance(row, dict)
        and row.get("action") == "retire"
        and row.get("retirement_id") == proof["retirement_id"]
        and row.get("content_fingerprint") == proof["content_fingerprint"]
        and row.get("event_id") == proof["event_id"]
        and row.get("event_fingerprint") == proof["event_fingerprint"]
        and row.get("producer") == proof["producer"]
    )


def bind_loaded_source_retirements(
    notes: list[Any], root: str,
) -> SourceLifecycleAudit:
    audit = load_source_lifecycle_audit(root)
    for note in notes:
        note._source_retirement_audit_match = source_retirement_valid(note, audit)
    return audit


def _append_row(root: str, row: dict) -> None:
    from .locks import open_repository_state_directory

    state_descriptor, _created = open_repository_state_directory(
        root, create=True,
    )
    encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            "source_lifecycle.jsonl", flags, 0o600,
            dir_fd=state_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("source lifecycle audit is not a regular file")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                from .verification_audit import read_descriptor_all
                existing = read_descriptor_all(descriptor, _MAX_AUDIT_BYTES)
                audit = _parse_audit(existing)
                if not audit.trustworthy:
                    raise RuntimeError(
                        "source lifecycle audit is malformed; refusing transition"
                    )
                if len(existing) + len(encoded) > _MAX_AUDIT_BYTES:
                    raise RuntimeError("source lifecycle audit exceeds safety bound")
                current = audit.latest.get(row["note"])
                if not (
                    isinstance(current, dict)
                    and current.get("transition_id") == row["transition_id"]
                ):
                    original_size = len(existing)
                    offset = 0
                    try:
                        while offset < len(encoded):
                            written = os.write(descriptor, encoded[offset:])
                            if written <= 0:
                                raise OSError(
                                    "source lifecycle audit append made no progress"
                                )
                            offset += written
                        os.fsync(descriptor)
                    except BaseException:
                        try:
                            os.ftruncate(descriptor, original_size)
                            os.fsync(descriptor)
                        except OSError:
                            pass
                        raise
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        os.fsync(state_descriptor)
    finally:
        os.close(state_descriptor)


def _event_lineage(event: Any, producer: str) -> str:
    if producer == "gcal-cancellation-v1":
        try:
            from ingest.gcal_contract import gcal_lineage
            return gcal_lineage(event)
        except (ImportError, AttributeError, TypeError, ValueError):
            return ""
    elif producer == "slack-thread-v3-deletion-v1":
        try:
            from ingest.slack_contract_v3 import thread_v3_lineage
            return thread_v3_lineage(event)
        except (ImportError, AttributeError, TypeError, ValueError):
            return ""
    elif producer == "gdrive-file-deletion-v1":
        try:
            from ingest.gdrive_contract import gdrive_lineage
            return gdrive_lineage(event)
        except (ImportError, AttributeError, TypeError, ValueError):
            return ""
    return ""


def _validated_transition_history(
    root: str, event: Any, lineage: str, producer: str, action: str,
) -> list[Any]:
    """Return authenticated lineage history only when ``event`` is latest."""
    from .event import _parse_iso_timestamp, iter_validated_events

    try:
        all_events = list(iter_validated_events(
            os.path.join(os.path.abspath(root), "events")
        ))
    except Exception as exc:
        raise PermissionError(
            "source lifecycle event is outside the authenticated L1 prefix"
        ) from exc
    stored = next(
        (candidate for candidate in all_events
         if candidate.id == getattr(event, "id", None)),
        None,
    )
    if stored is None or stored.to_json() != event.to_json():
        raise PermissionError(
            "source lifecycle event is outside the authenticated L1 prefix"
        )
    history = [
        candidate for candidate in all_events
        if _event_lineage(candidate, producer) == lineage
    ]
    if not history or _event_lineage(event, producer) != lineage:
        raise PermissionError("event does not match the owned source lineage")
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    def source_rank(item: tuple[int, Any]) -> tuple[int, datetime, datetime, int]:
        index, candidate = item
        observed = (
            _parse_iso_timestamp(getattr(candidate, "ingested_at", None)) or epoch
        )
        source_updated = epoch
        if getattr(candidate, "source", None) == "gcal":
            try:
                from ingest.gcal_contract import gcal_source_updated
                source_updated = gcal_source_updated(candidate) or epoch
            except (ImportError, TypeError, ValueError):
                source_updated = epoch
        elif getattr(candidate, "source", None) == "gdrive":
            try:
                from ingest.gdrive_contract import gdrive_source_time
                source_updated = gdrive_source_time(candidate) or epoch
            except (ImportError, TypeError, ValueError):
                source_updated = epoch
        terminal = int(
            getattr(candidate, "source", None) == "slack"
            and (getattr(candidate, "payload", None) or {}).get("deleted") is True
        )
        return terminal, source_updated, observed, index

    # Google mutations are ordered by the source's own update clock. Slack
    # deletion is terminal because Slack has no undelete operation, so a late
    # live replay can never outrank the authority-bearing tombstone. Ingestion
    # and physical order are only tie-breakers within the same source state.
    latest_index, latest = max(
        enumerate(history),
        key=source_rank,
    )
    del latest_index
    if getattr(event, "source", None) in {"gcal", "gdrive"}:
        if getattr(event, "source", None) == "gcal":
            from ingest.gcal_contract import gcal_source_updated
            source_time = gcal_source_updated
            source_label = "Calendar"
        else:
            from ingest.gdrive_contract import gdrive_source_time
            source_time = gdrive_source_time
            source_label = "Drive"
        newest_source_time = source_time(latest)
        tied = [
            candidate for candidate in history
            if source_time(candidate) == newest_source_time
        ]
        if len({candidate.id for candidate in tied}) > 1:
            raise PermissionError(
                f"source lifecycle has conflicting latest {source_label} snapshots"
            )
    if latest.to_json() != event.to_json():
        raise PermissionError("source lifecycle event is not the latest source revision")
    is_deleted = (
        (getattr(event, "payload", None) or {}).get("status") == "cancelled"
        if getattr(event, "source", None) == "gcal"
        else (getattr(event, "payload", None) or {}).get("deleted") is True
    )
    if (action == "retire") is not is_deleted:
        raise PermissionError("event does not authorize this source lifecycle transition")
    return history


def _validate_owned_note(
    note: Any, *, root: str, lineage: str, source: str,
    history: list[Any], transition_event: Any,
) -> None:
    """Independently prove one atomic proposed note owns this source object."""
    from .locks import pipeline_mutation_lock_held
    from .note import load_notes

    root = os.path.abspath(root)
    if not pipeline_mutation_lock_held(root):
        raise PermissionError("source lifecycle requires the pipeline mutation lock")
    if (
        getattr(note, "state", None) != "proposed"
        or getattr(note, "superseded", None) is not False
        or getattr(note, "approval", None) is not None
        or getattr(note, "source_retirement", None) is not None
        or getattr(note, "producer_lifecycle", None) is not None
        or getattr(note, "type", None) not in {
            "project", "decision", "state", "reference",
        }
        or list(getattr(note, "source_lineages", None) or []) != [lineage]
        or not lineage.startswith(source + ":")
        or any(
            str(item).startswith(("by:", "promote-by:", "reject-by:"))
            for item in (getattr(note, "provenance", None) or [])
        )
    ):
        raise PermissionError(
            "source retirement requires one exact machine-proposed lineage owner"
        )
    prior_ids = {
        candidate.id for candidate in history
        if candidate.id != getattr(transition_event, "id", None)
    }
    if not prior_ids.intersection(getattr(note, "provenance", None) or []):
        raise PermissionError(
            "source retirement owner does not cite an earlier lineage revision"
        )
    matches = [
        candidate for candidate in load_notes(os.path.join(root, "context"))
        if candidate.name == getattr(note, "name", None)
    ]
    if (
        len(matches) != 1
        or _source_preimage_fingerprint(matches[0])
        != _source_preimage_fingerprint(note)
    ):
        raise PermissionError(
            "source retirement target is not the exact persisted note preimage"
        )


def _source_material(
    note: Any, event: Any, lineage: str, producer: str, *, root: str, action: str,
) -> dict:
    from .verification import verification_content_fingerprint

    history = _validated_transition_history(
        root, event, lineage, producer, action,
    )
    source = str(getattr(event, "source", "") or "")
    if action == "retire":
        _validate_owned_note(
            note, root=root, lineage=lineage, source=source,
            history=history, transition_event=event,
        )
    event_id = str(getattr(event, "id", "") or "")
    material = _retirement_material(
        note=str(getattr(note, "name", "") or ""),
        content_fingerprint=verification_content_fingerprint(note),
        source=source,
        lineage=str(lineage or ""),
        event_id=event_id,
        event_fingerprint=_event_fingerprint(event),
        producer=str(producer or ""),
    )
    if not _valid_material(material):
        raise ValueError("source lifecycle transition violates the closed producer contract")
    return material


def _record_source_retirement(
    note: Any, *, root: str, event: Any, lineage: str, producer: str,
) -> _SourceLifecycleWriteCapability:
    """Journal an exact source retirement and return private write authority."""
    preimage_fingerprint = _source_preimage_fingerprint(note)
    material = _source_material(
        note, event, lineage, producer, root=root, action="retire",
    )
    retirement_id = _canonical_hash(material)
    transition_id = _canonical_hash({
        **material, "action": "retire", "retirement_id": retirement_id,
        "target_content_fingerprint": material["content_fingerprint"],
    })
    row = {
        **material,
        "action": "retire",
        "to": "superseded",
        "retirement_id": retirement_id,
        "target_content_fingerprint": material["content_fingerprint"],
        "transition_id": transition_id,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _append_row(root, row)
    note.source_retirement = {
        key: row[key]
        for key in (
            "version", "retirement_id", "content_fingerprint", "source",
            "lineage", "event_id", "event_fingerprint", "producer",
        )
    }
    note.state = "superseded"
    note.superseded = True
    note._source_retirement_audit_match = True
    return _SourceLifecycleWriteCapability(
        note=material["note"], state="superseded",
        transition_id=transition_id,
        content_fingerprint=material["content_fingerprint"],
        preimage_fingerprint=preimage_fingerprint,
        _seal=_WRITE_SEAL,
    )


def _record_source_restoration(
    note: Any, *, root: str, event: Any, lineage: str, producer: str,
    desired: Any | None = None,
) -> _SourceLifecycleWriteCapability:
    """Inline restoration is intentionally forbidden.

    A live revision is new factual content, not a lifecycle control.  Letting a
    deletion event mint authority for arbitrary ``desired`` bytes previously
    allowed body and ACL replacement.  The distiller now creates a fresh
    proposed owner for the live snapshot while retaining the retired note as
    immutable history.
    """
    del note, root, event, lineage, producer, desired
    raise PermissionError(
        "inline source restoration is disabled; create a new source-derived note"
    )


def source_lifecycle_write_authorized(
    note: Any, root: str, capability: object, *, existing_note: Any,
) -> bool:
    from .locks import pipeline_mutation_lock_held

    root = os.path.abspath(root)
    if (
        not isinstance(capability, _SourceLifecycleWriteCapability)
        or capability._seal is not _WRITE_SEAL
        or capability.note != getattr(note, "name", None)
        or capability.note != getattr(existing_note, "name", None)
        or capability.state != getattr(note, "state", None)
        or capability.state != "superseded"
        or not pipeline_mutation_lock_held(root)
    ):
        return False
    from .verification import verification_content_fingerprint

    if capability.content_fingerprint != verification_content_fingerprint(note):
        return False
    try:
        if capability.preimage_fingerprint != _source_preimage_fingerprint(existing_note):
            return False
    except (TypeError, ValueError):
        return False
    audit = load_source_lifecycle_audit(root)
    if not audit.trustworthy:
        return False
    row = audit.latest.get(capability.note)
    if not isinstance(row, dict) or row.get("transition_id") != capability.transition_id:
        return False
    return source_retirement_valid(note, audit)
