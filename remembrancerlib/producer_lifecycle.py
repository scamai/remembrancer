"""Content-bound lifecycle authority for deterministic producer notes.

This module deliberately does not teach the generic note writer about a new
authority.  It supplies the closed proof, append-only journal, and sealed write
capability that the write door can consume.  The first (and currently only)
contract is the contradiction linter:

* only an exact ``type: contradiction`` note with the deterministic structured
  pair and the existing ``generated-by:contradiction-lint`` ownership marker is
  eligible;
* only ``producer=contradiction-lint-v1`` may supersede or restore it;
* every transition is bound in both note frontmatter (``producer_lifecycle``)
  and ``state/producer_lifecycle.jsonl`` to the exact answer-content hash; and
* the audit row is durably appended before the in-memory note receives the
  proof.  A retry can finish an audit-first transition after a crash without
  appending a second row.

The embedded proof is a lifecycle control and must stay outside
``verification_content_fingerprint`` (as do human approval and source
retirement proofs); including it would make the content hash self-referential.
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


PRODUCER_LIFECYCLE_VERSION = 1
CONTRADICTION_PRODUCER = "contradiction-lint-v1"
CONTRADICTION_PROVENANCE_MARKER = "generated-by:contradiction-lint"
AUDIT_FILENAME = "producer_lifecycle.jsonl"

_ACTIONS = {
    "supersede": ("proposed", "superseded", True),
    "restore": ("superseded", "proposed", False),
}
_SHA256_RE = re.compile(r"[a-f0-9]{64}\Z")
_NOTE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
_UTC_SECOND_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_MAX_AUDIT_BYTES = 32 * 1024 * 1024
_MAX_AUDIT_LINE_BYTES = 256 * 1024
_WRITE_SEAL = object()
_EVIDENCE_SEAL = object()

_MATERIAL_KEYS = frozenset({
    "version", "note", "pair", "producer", "action", "from", "to",
    "content_fingerprint", "invariant_fingerprint", "preimage_fingerprint",
    "evidence_fingerprint", "prior_transition_id",
})
_PROOF_KEYS = _MATERIAL_KEYS | frozenset({"transition_id"})
_ROW_KEYS = _PROOF_KEYS | frozenset({"ts"})


@dataclass(frozen=True)
class ProducerLifecycleAudit:
    """Parsed journal state.

    ``latest`` is keyed by note name and ``transitions`` retains the complete
    validated chain for dual-binding checks and audit-first crash recovery.
    """

    latest: dict[str, dict]
    transitions: dict[str, dict]
    trustworthy: bool = True


@dataclass(frozen=True)
class _ProducerLifecycleWriteCapability:
    """Unforgeable-by-data authority for one exact, already-journaled write."""

    note: str
    action: str
    state: str
    transition_id: str
    content_fingerprint: str
    invariant_fingerprint: str
    preimage_fingerprint: str
    _seal: object


@dataclass(frozen=True)
class ContradictionLifecycleEvidence:
    """Sealed result of rerunning the deterministic detector on one snapshot."""

    note: str
    pair: tuple[str, str]
    active: bool
    detector: str
    snapshot_fingerprint: str
    _seal: object


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_fingerprint(note: Any) -> str:
    from .verification import verification_content_fingerprint

    value = verification_content_fingerprint(note)
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("note content fingerprint is not canonical")
    return value


_INVARIANT_FIELDS = (
    "name", "title", "description", "type", "domain", "project", "acl",
    "entities", "components", "provenance", "source_lineages", "updated",
    "body", "resource", "tags", "structured", "confidence", "approval",
    "source_retirement",
)


def _invariant_fingerprint(note: Any) -> str:
    """Bind every persisted field the lifecycle producer is forbidden to edit."""
    payload = {
        field: getattr(note, field, None)
        for field in _INVARIANT_FIELDS
    }
    # Match Note.to_markdown/from_frontmatter normalization so a capability
    # minted from the just-written producer object also binds the disk preimage.
    payload["title"] = str(payload.get("title") or payload.get("name") or "").strip()
    payload["body"] = str(payload.get("body") or "").rstrip()
    try:
        return _canonical_hash(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("note contains non-canonical lifecycle content") from exc


def _preimage_fingerprint(note: Any) -> str:
    """Bind the exact control state that must still exist at the write door."""
    proof = getattr(note, "producer_lifecycle", None)
    if proof is not None and not isinstance(proof, dict):
        raise ValueError("producer lifecycle preimage proof is not an object")
    return _canonical_hash({
        "invariant_fingerprint": _invariant_fingerprint(note),
        "state": getattr(note, "state", None),
        "superseded": getattr(note, "superseded", None),
        "producer_lifecycle": proof,
    })


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or "note"


def _expected_contradiction_name(pair: tuple[str, str]) -> str:
    """Mirror the linter's stable, collision-resistant name contract."""
    first, second = pair
    base = f"contradiction-{_slugify(first)}-vs-{_slugify(second)}"
    if len(base) <= 80:
        return base
    digest = hashlib.sha256(f"{first}\0{second}".encode("utf-8")).hexdigest()[:10]
    return f"{base[:69].rstrip('-')}-{digest}"


def _contradiction_identity(note: Any) -> tuple[str, tuple[str, str]]:
    """Return the closed note identity or raise before any journal mutation."""
    name = getattr(note, "name", None)
    note_type = getattr(note, "type", None)
    structured = getattr(note, "structured", None)
    provenance = getattr(note, "provenance", None)
    approval = getattr(note, "approval", None)
    if not isinstance(name, str) or _NOTE_NAME_RE.fullmatch(name) is None:
        raise PermissionError("producer lifecycle requires a canonical note name")
    if note_type != "contradiction":
        raise PermissionError("producer lifecycle is restricted to contradiction notes")
    if getattr(note, "domain", None) != "company":
        raise PermissionError("contradiction producer lifecycle requires domain=company")
    if getattr(note, "title", "") not in ("", name):
        raise PermissionError("contradiction title is not producer-canonical")
    for field in (
        "project", "resource", "tags", "components", "source_lineages",
    ):
        value = getattr(note, field, "" if field in {"project", "resource"} else [])
        if value not in ("", [], None):
            raise PermissionError(
                f"contradiction producer does not own non-empty {field}"
            )
    if not isinstance(structured, dict) or set(structured) != {"kind", "pair"}:
        raise PermissionError("contradiction structured metadata has an invalid schema")
    pair = structured.get("pair")
    if (
        structured.get("kind") != "contradiction"
        or not isinstance(pair, list)
        or len(pair) != 2
        or any(not isinstance(item, str) or _NOTE_NAME_RE.fullmatch(item) is None
               for item in pair)
        or pair[0] >= pair[1]
    ):
        raise PermissionError("contradiction pair is not canonical")
    normalized = (pair[0], pair[1])
    if name != _expected_contradiction_name(normalized):
        raise PermissionError("contradiction note name does not bind its structured pair")
    expected_description = (
        f"Auto-detected contradiction between {normalized[0]} and {normalized[1]}"
    )
    expected_body = (
        f"Auto-detected contradiction between [[{normalized[0]}]] and "
        f"[[{normalized[1]}]].\n\n"
        "Resolve by updating one of the source notes or reconciling the conflict."
    )
    if getattr(note, "description", None) != expected_description:
        raise PermissionError("contradiction description is not producer-canonical")
    if str(getattr(note, "body", "") or "").rstrip() != expected_body:
        raise PermissionError("contradiction body is not producer-canonical")
    if (
        not isinstance(provenance, list)
        or any(not isinstance(item, str) or not item for item in provenance)
        or len(provenance) != len(set(provenance))
    ):
        raise PermissionError("contradiction provenance is not a canonical list")
    generated = [
        item for item in provenance if item.startswith("generated-by:")
    ]
    if generated != [CONTRADICTION_PROVENANCE_MARKER]:
        raise PermissionError("contradiction provenance does not prove sole producer ownership")
    if (
        approval is not None
        or getattr(note, "source_retirement", None) is not None
        or any(
            item.startswith(("by:", "confirmed-by:", "machine-update:"))
            for item in provenance
        )
    ):
        raise PermissionError("producer lifecycle cannot control a human or proposal note")
    from .verification import producer_verification_valid

    if not producer_verification_valid(note):
        raise PermissionError(
            "contradiction producer verification is missing, stale, or malformed"
        )
    confidence = getattr(note, "confidence", None)
    verification = confidence.get("verification") if isinstance(confidence, dict) else None
    if (
        not isinstance(verification, dict)
        or verification.get("producer") != "contradiction-lint"
        or verification.get("method")
        != "producer-grounded-v1:contradiction-detector"
    ):
        raise PermissionError(
            "contradiction verification does not bind the deterministic detector"
        )
    return name, normalized


def _contradiction_lifecycle_evidence_from_notes(
    note: Any,
    *,
    notes: list[Any],
) -> ContradictionLifecycleEvidence:
    """Seal a detector result from an already-authoritative disk snapshot."""
    name, pair = _contradiction_identity(note)
    if not isinstance(notes, list):
        raise ValueError("contradiction lifecycle evidence requires a note list")
    groups: dict[str, list[Any]] = {}
    for candidate in notes:
        candidate_name = getattr(candidate, "name", None)
        if not isinstance(candidate_name, str) or _NOTE_NAME_RE.fullmatch(candidate_name) is None:
            raise ValueError("detector snapshot contains a non-canonical note name")
        groups.setdefault(candidate_name, []).append(candidate)
    if any(len(group) != 1 for group in groups.values()):
        raise ValueError("detector snapshot contains duplicate logical note names")
    physical = groups.get(name, [])
    if (
        len(physical) != 1
        or _preimage_fingerprint(physical[0]) != _preimage_fingerprint(note)
    ):
        raise PermissionError(
            "lifecycle note is not the exact note in the detector snapshot"
        )

    # Runtime imports avoid a module-import cycle when lint.engine consumes this
    # authority.  At call time its deterministic detector is fully initialized.
    from .note import current_note_names
    from lint.engine import find_contradictions

    current = current_note_names(notes)
    detector_notes = [candidate for candidate in notes if candidate.name in current]
    pairs = sorted(tuple(sorted(item)) for item in find_contradictions(detector_notes))
    if len(pairs) != len(set(pairs)):
        raise ValueError("detector emitted duplicate contradiction pairs")
    snapshot = {
        "detector": "lint.find-contradictions-v1",
        "pair": list(pair),
        "current_pairs": [list(item) for item in pairs],
        "inputs": [
            {
                "name": candidate.name,
                "content_fingerprint": _content_fingerprint(candidate),
                "state": getattr(candidate, "state", None),
                "superseded": getattr(candidate, "superseded", None),
                "current": candidate.name in current,
            }
            for candidate in sorted(notes, key=lambda item: item.name)
        ],
    }
    return ContradictionLifecycleEvidence(
        note=name,
        pair=pair,
        active=pair in set(pairs),
        detector="lint.find-contradictions-v1",
        snapshot_fingerprint=_canonical_hash(snapshot),
        _seal=_EVIDENCE_SEAL,
    )


def contradiction_lifecycle_evidence(
    note: Any,
    *,
    root: str,
) -> ContradictionLifecycleEvidence:
    """Rerun the detector over the exact on-disk corpus under its write lock.

    Accepting a caller-supplied note list here would turn a sealed object into
    self-attestation: omitting the two source notes could manufacture an
    ``absent`` verdict.  The lifecycle authority therefore loads L2 itself and
    also proves that the target object is the exact persisted preimage.
    """
    from .locks import pipeline_mutation_lock_held
    from .note import load_notes

    root = os.path.abspath(root)
    if not pipeline_mutation_lock_held(root):
        raise PermissionError(
            "contradiction lifecycle evidence requires the pipeline mutation lock"
        )
    return _contradiction_lifecycle_evidence_from_notes(
        note,
        notes=load_notes(os.path.join(root, "context")),
    )


def _validated_evidence(
    note: Any,
    *,
    action: str,
    evidence: object,
) -> ContradictionLifecycleEvidence:
    name, pair = _contradiction_identity(note)
    expected_active = action == "restore"
    if (
        not isinstance(evidence, ContradictionLifecycleEvidence)
        or evidence._seal is not _EVIDENCE_SEAL
        or evidence.note != name
        or evidence.pair != pair
        or evidence.active is not expected_active
        or evidence.detector != "lint.find-contradictions-v1"
        or not isinstance(evidence.snapshot_fingerprint, str)
        or _SHA256_RE.fullmatch(evidence.snapshot_fingerprint) is None
    ):
        predicate = "present" if expected_active else "absent"
        raise PermissionError(
            f"producer {action} requires sealed detector evidence that the pair is {predicate}"
        )
    return evidence


def _material(
    note: Any,
    *,
    action: str,
    producer: str,
    evidence: ContradictionLifecycleEvidence,
    prior_transition_id: str,
) -> dict[str, Any]:
    if producer != CONTRADICTION_PRODUCER:
        raise PermissionError(
            f"unsupported producer lifecycle authority: {producer!r}"
        )
    if action not in _ACTIONS:
        raise ValueError(f"unsupported producer lifecycle action: {action!r}")
    name, pair = _contradiction_identity(note)
    from_state, to_state, _superseded = _ACTIONS[action]
    return {
        "version": PRODUCER_LIFECYCLE_VERSION,
        "note": name,
        "pair": list(pair),
        "producer": producer,
        "action": action,
        "from": from_state,
        "to": to_state,
        "content_fingerprint": _content_fingerprint(note),
        "invariant_fingerprint": _invariant_fingerprint(note),
        "preimage_fingerprint": _preimage_fingerprint(note),
        "evidence_fingerprint": evidence.snapshot_fingerprint,
        "prior_transition_id": prior_transition_id,
    }


def _material_valid(material: dict[str, Any]) -> bool:
    if not isinstance(material, dict) or set(material) != _MATERIAL_KEYS:
        return False
    action = material.get("action")
    pair = material.get("pair")
    if action not in _ACTIONS:
        return False
    from_state, to_state, _superseded = _ACTIONS[action]
    if (
        material.get("version") != PRODUCER_LIFECYCLE_VERSION
        or material.get("producer") != CONTRADICTION_PRODUCER
        or material.get("from") != from_state
        or material.get("to") != to_state
        or not isinstance(material.get("note"), str)
        or _NOTE_NAME_RE.fullmatch(material["note"]) is None
        or not isinstance(pair, list)
        or len(pair) != 2
        or any(not isinstance(item, str) or _NOTE_NAME_RE.fullmatch(item) is None
               for item in pair)
        or pair[0] >= pair[1]
        or material["note"] != _expected_contradiction_name((pair[0], pair[1]))
        or not isinstance(material.get("content_fingerprint"), str)
        or _SHA256_RE.fullmatch(material["content_fingerprint"]) is None
        or not isinstance(material.get("invariant_fingerprint"), str)
        or _SHA256_RE.fullmatch(material["invariant_fingerprint"]) is None
        or not isinstance(material.get("preimage_fingerprint"), str)
        or _SHA256_RE.fullmatch(material["preimage_fingerprint"]) is None
        or not isinstance(material.get("evidence_fingerprint"), str)
        or _SHA256_RE.fullmatch(material["evidence_fingerprint"]) is None
        or not isinstance(material.get("prior_transition_id"), str)
        or (
            material["prior_transition_id"]
            and _SHA256_RE.fullmatch(material["prior_transition_id"]) is None
        )
    ):
        return False
    return True


def _row_material(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in _MATERIAL_KEYS}


def _row_valid(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict) or set(row) != _ROW_KEYS:
        return False
    material = _row_material(row)
    transition_id = row.get("transition_id")
    timestamp = row.get("ts")
    if not (
        _material_valid(material)
        and isinstance(transition_id, str)
        and transition_id == _canonical_hash(material)
        and isinstance(timestamp, str)
        and _UTC_SECOND_RE.fullmatch(timestamp)
    ):
        return False
    try:
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _untrusted_audit() -> ProducerLifecycleAudit:
    return ProducerLifecycleAudit({}, {}, trustworthy=False)


def _parse_audit(raw: bytes) -> ProducerLifecycleAudit:
    """Validate the complete per-note transition chain; malformed bytes close it."""
    if len(raw) > _MAX_AUDIT_BYTES or (raw and not raw.endswith(b"\n")):
        return _untrusted_audit()
    latest: dict[str, dict] = {}
    transitions: dict[str, dict] = {}
    for raw_line in raw.splitlines():
        if not raw_line or len(raw_line) > _MAX_AUDIT_LINE_BYTES:
            return _untrusted_audit()
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _untrusted_audit()
        if not _row_valid(row):
            return _untrusted_audit()
        transition_id = row["transition_id"]
        if transition_id in transitions:
            return _untrusted_audit()
        prior = latest.get(row["note"])
        expected_prior = prior["transition_id"] if prior else ""
        if row["prior_transition_id"] != expected_prior:
            return _untrusted_audit()
        if prior is None:
            if row["action"] != "supersede":
                return _untrusted_audit()
        else:
            expected_action = "restore" if prior["action"] == "supersede" else "supersede"
            if (
                row["action"] != expected_action
                or row["pair"] != prior["pair"]
                or row["producer"] != prior["producer"]
                or (row["action"] == "restore"
                    and row["content_fingerprint"] != prior["content_fingerprint"])
                or (row["action"] == "restore"
                    and row["invariant_fingerprint"] != prior["invariant_fingerprint"])
            ):
                return _untrusted_audit()
        transitions[transition_id] = row
        latest[row["note"]] = row
    return ProducerLifecycleAudit(latest, transitions)


def _audit_path(root: str) -> str:
    return os.path.join(os.path.abspath(root), "state", AUDIT_FILENAME)


def _open_state_directory(root: str, *, create: bool) -> tuple[int, bool]:
    """Open a real repository-local ``state`` directory without following it.

    Returning a directory descriptor lets every audit-file operation stay
    relative to the directory that was actually checked.  A repository whose
    ``state`` entry is a symlink is refused instead of writing lifecycle
    authority outside the repository.
    """
    from .locks import open_repository_state_directory

    return open_repository_state_directory(root, create=create)


def load_producer_lifecycle_audit(root: str) -> ProducerLifecycleAudit:
    """Read the private journal under a shared lock; any anomaly fails closed."""
    try:
        state_descriptor, _created = _open_state_directory(root, create=False)
    except OSError:
        return _untrusted_audit()
    if state_descriptor < 0:
        return ProducerLifecycleAudit({}, {})
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(AUDIT_FILENAME, flags, dir_fd=state_descriptor)
    except FileNotFoundError:
        os.close(state_descriptor)
        return ProducerLifecycleAudit({}, {})
    except OSError:
        os.close(state_descriptor)
        return _untrusted_audit()
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
                    return _untrusted_audit()
                raw = handle.read(_MAX_AUDIT_BYTES + 1)
            except OSError:
                return _untrusted_audit()
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
    except OSError:
        return _untrusted_audit()
    finally:
        os.close(state_descriptor)
    return _parse_audit(raw)


def _append_row(root: str, row: dict[str, Any]) -> dict[str, Any]:
    """Durably append one chain-valid row, returning the canonical stored row."""
    if not _row_valid(row):
        raise ValueError("producer lifecycle row violates the closed schema")
    state_descriptor, _created = _open_state_directory(root, create=True)
    try:
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            AUDIT_FILENAME, flags, 0o600, dir_fd=state_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("producer lifecycle audit is not a regular file")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                chunks: list[bytes] = []
                remaining = _MAX_AUDIT_BYTES + 1
                while remaining > 0:
                    chunk = os.read(descriptor, min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                existing = b"".join(chunks)
                audit = _parse_audit(existing)
                if not audit.trustworthy:
                    raise RuntimeError(
                        "producer lifecycle audit is malformed; refusing transition"
                    )
                current = audit.latest.get(row["note"])
                if (
                    isinstance(current, dict)
                    and current.get("transition_id") == row["transition_id"]
                ):
                    stored = current
                else:
                    expected_prior = current["transition_id"] if current else ""
                    if row["prior_transition_id"] != expected_prior:
                        raise RuntimeError(
                            "producer lifecycle changed concurrently; retry from current state"
                        )
                    encoded = (
                        json.dumps(
                            row, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False,
                        )
                        + "\n"
                    ).encode("utf-8")
                    if (
                        len(encoded) > _MAX_AUDIT_LINE_BYTES
                        or len(existing) + len(encoded) > _MAX_AUDIT_BYTES
                    ):
                        raise RuntimeError(
                            "producer lifecycle audit exceeds safety bound"
                        )
                    original_size = len(existing)
                    offset = 0
                    try:
                        while offset < len(encoded):
                            written = os.write(descriptor, encoded[offset:])
                            if written <= 0:
                                raise OSError(
                                    "producer lifecycle audit append made no progress"
                                )
                            offset += written
                        os.fsync(descriptor)
                    except BaseException:
                        # Never leave a torn JSONL tail that permanently poisons
                        # every later read. The exclusive lock makes rollback exact.
                        try:
                            os.ftruncate(descriptor, original_size)
                            os.fsync(descriptor)
                        except OSError:
                            pass
                        raise
                    stored = row
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        os.fsync(state_descriptor)
        return stored
    finally:
        os.close(state_descriptor)


def _embedded_transition(note: Any) -> dict[str, Any] | None:
    proof = getattr(note, "producer_lifecycle", None)
    if not isinstance(proof, dict) or set(proof) != _PROOF_KEYS:
        return None
    material = {key: proof.get(key) for key in _MATERIAL_KEYS}
    try:
        content_fingerprint = _content_fingerprint(note)
        invariant_fingerprint = _invariant_fingerprint(note)
    except (TypeError, ValueError):
        return None
    if (
        not _material_valid(material)
        or proof.get("transition_id") != _canonical_hash(material)
        or proof.get("note") != getattr(note, "name", None)
        or proof.get("content_fingerprint") != content_fingerprint
        or proof.get("invariant_fingerprint") != invariant_fingerprint
        or proof.get("to") != getattr(note, "state", None)
    ):
        return None
    expected_superseded = _ACTIONS[proof["action"]][2]
    if getattr(note, "superseded", None) is not expected_superseded:
        return None
    try:
        _name, pair = _contradiction_identity(note)
    except (PermissionError, ValueError):
        return None
    if proof.get("pair") != list(pair):
        return None
    return proof


def producer_lifecycle_valid(
    note: Any,
    audit: ProducerLifecycleAudit | None = None,
) -> bool:
    """Whether the latest frontmatter proof exactly matches the durable journal."""
    proof = _embedded_transition(note)
    if proof is None:
        return False
    if audit is None:
        return bool(getattr(note, "_producer_lifecycle_audit_match", False))
    if not audit.trustworthy:
        return False
    row = audit.latest.get(proof["note"])
    return bool(
        isinstance(row, dict)
        and row.get("transition_id") == proof["transition_id"]
        and all(row.get(key) == proof.get(key) for key in _PROOF_KEYS)
    )


def bind_loaded_producer_lifecycles(
    notes: list[Any], root: str,
) -> ProducerLifecycleAudit:
    """Cache only the result of a full dual-binding check on loaded notes."""
    audit = load_producer_lifecycle_audit(root)
    for note in notes:
        note._producer_lifecycle_audit_match = producer_lifecycle_valid(note, audit)
    return audit


def _bind_transition(note: Any, row: dict[str, Any]) -> None:
    """Apply the already-durable row to the in-memory note (the crash boundary)."""
    note.producer_lifecycle = {
        key: row[key] for key in _PROOF_KEYS
    }
    note.state = row["to"]
    note.superseded = _ACTIONS[row["action"]][2]
    note._producer_lifecycle_audit_match = True


def _proof_matches_row(proof: Any, row: dict[str, Any]) -> bool:
    """Match a prior embedded proof without requiring today's content hash."""
    return bool(
        isinstance(proof, dict)
        and set(proof) == _PROOF_KEYS
        and all(proof.get(key) == row.get(key) for key in _PROOF_KEYS)
    )


def _capability(row: dict[str, Any]) -> _ProducerLifecycleWriteCapability:
    return _ProducerLifecycleWriteCapability(
        note=row["note"],
        action=row["action"],
        state=row["to"],
        transition_id=row["transition_id"],
        content_fingerprint=row["content_fingerprint"],
        invariant_fingerprint=row["invariant_fingerprint"],
        preimage_fingerprint=row["preimage_fingerprint"],
        _seal=_WRITE_SEAL,
    )


def _pending_row(
    note: Any,
    *,
    audit: ProducerLifecycleAudit,
    action: str,
    producer: str,
) -> dict[str, Any] | None:
    """Find an audit-first row whose frontmatter application was interrupted."""
    latest = audit.latest.get(str(getattr(note, "name", "") or ""))
    if not isinstance(latest, dict):
        return None
    expected_from, _to, expected_superseded = _ACTIONS[action]
    if (
        latest.get("action") != action
        or latest.get("producer") != producer
        or latest.get("content_fingerprint") != _content_fingerprint(note)
        or latest.get("invariant_fingerprint") != _invariant_fingerprint(note)
        or latest.get("preimage_fingerprint") != _preimage_fingerprint(note)
        or getattr(note, "state", None) != expected_from
        or getattr(note, "superseded", None) is expected_superseded
    ):
        return None
    current_proof = getattr(note, "producer_lifecycle", None)
    prior_id = latest["prior_transition_id"]
    if prior_id:
        prior_row = audit.transitions.get(prior_id)
        if not isinstance(prior_row, dict) or not _proof_matches_row(
            current_proof, prior_row,
        ):
            return None
    elif current_proof is not None:
        return None
    return latest


def _record_transition(
    note: Any,
    *,
    root: str,
    action: str,
    producer: str,
) -> _ProducerLifecycleWriteCapability:
    from .locks import pipeline_mutation_lock_held

    root = os.path.abspath(root)
    if not pipeline_mutation_lock_held(root):
        raise PermissionError(
            "producer lifecycle transition requires the pipeline mutation lock"
        )
    if producer != CONTRADICTION_PRODUCER:
        raise PermissionError(
            f"unsupported producer lifecycle authority: {producer!r}"
        )
    if action not in _ACTIONS:
        raise ValueError(f"unsupported producer lifecycle action: {action!r}")
    _contradiction_identity(note)
    if not isinstance(getattr(note, "superseded", None), bool):
        raise PermissionError("producer lifecycle requires a boolean superseded flag")
    evidence = contradiction_lifecycle_evidence(note, root=root)
    evidence = _validated_evidence(note, action=action, evidence=evidence)
    audit = load_producer_lifecycle_audit(root)
    if not audit.trustworthy:
        raise RuntimeError("producer lifecycle audit is malformed; refusing transition")

    # Exact replay after the note write: return equivalent authority without a
    # duplicate journal row.
    if producer_lifecycle_valid(note, audit):
        current = audit.latest[getattr(note, "name")]
        if current["action"] == action and current["producer"] == producer:
            return _capability(current)

    # Audit-first crash recovery: the desired row is already current but the
    # frontmatter still represents its predecessor.
    pending = _pending_row(
        note, audit=audit, action=action, producer=producer,
    )
    if pending is not None:
        _bind_transition(note, pending)
        return _capability(pending)

    from_state, _to_state, expected_superseded = _ACTIONS[action]
    if (
        getattr(note, "state", None) != from_state
        or bool(getattr(note, "superseded", False)) == expected_superseded
    ):
        raise PermissionError(
            f"producer {action} requires state={from_state!r} and the matching flag"
        )

    latest = audit.latest.get(getattr(note, "name"))
    if action == "restore":
        # Hiding a note is sensitive: restoration is allowed only from the
        # exact supersession this producer previously bound in both copies.
        if not producer_lifecycle_valid(note, audit) or latest["action"] != "supersede":
            raise PermissionError(
                "producer restoration requires a current dual-bound supersession"
            )
    elif latest is not None:
        # A proposed flag may be deterministically refreshed after restoration.
        # Its old proof can have a stale content hash, but it must still name the
        # latest valid journal transition; ownership is independently rechecked
        # above against type/structured/provenance.
        proof = getattr(note, "producer_lifecycle", None)
        if latest.get("action") != "restore" or not _proof_matches_row(proof, latest):
            raise PermissionError(
                "producer supersession does not continue the current lifecycle chain"
            )
    elif getattr(note, "producer_lifecycle", None) is not None:
        raise PermissionError(
            "producer supersession cannot overwrite an unjournaled lifecycle proof"
        )

    prior_id = latest["transition_id"] if latest else ""
    material = _material(
        note, action=action, producer=producer,
        evidence=evidence,
        prior_transition_id=prior_id,
    )
    row = {
        **material,
        "transition_id": _canonical_hash(material),
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # Audit first.  No note control field changes before this durable append.
    stored = _append_row(root, row)
    _bind_transition(note, stored)
    return _capability(stored)


def record_contradiction_supersession(
    note: Any,
    *,
    root: str,
    producer: str = CONTRADICTION_PRODUCER,
) -> _ProducerLifecycleWriteCapability:
    """Journal and bind deterministic retirement of a resolved contradiction."""
    return _record_transition(
        note, root=root, action="supersede", producer=producer,
    )


def record_contradiction_restoration(
    note: Any,
    *,
    root: str,
    producer: str = CONTRADICTION_PRODUCER,
) -> _ProducerLifecycleWriteCapability:
    """Journal and bind restoration when the exact contradiction recurs."""
    return _record_transition(
        note, root=root, action="restore", producer=producer,
    )


def producer_lifecycle_write_authorized(
    note: Any,
    root: str,
    capability: object,
    *,
    existing_note: Any,
) -> bool:
    """Validate desired bytes and the persisted preimage against the journal."""
    from .locks import pipeline_mutation_lock_held

    root = os.path.abspath(root)
    try:
        content_fingerprint = _content_fingerprint(note)
        invariant_fingerprint = _invariant_fingerprint(note)
        preimage_fingerprint = _preimage_fingerprint(existing_note)
    except (TypeError, ValueError):
        return False
    if (
        not isinstance(capability, _ProducerLifecycleWriteCapability)
        or capability._seal is not _WRITE_SEAL
        or capability.note != getattr(note, "name", None)
        or capability.note != getattr(existing_note, "name", None)
        or capability.action not in _ACTIONS
        or capability.state != getattr(note, "state", None)
        or capability.content_fingerprint != content_fingerprint
        or capability.invariant_fingerprint != invariant_fingerprint
        or capability.preimage_fingerprint != preimage_fingerprint
        or not pipeline_mutation_lock_held(root)
    ):
        return False
    try:
        current_evidence = contradiction_lifecycle_evidence(
            existing_note, root=root,
        )
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError):
        return False
    if current_evidence.active is not (capability.action == "restore"):
        return False
    audit = load_producer_lifecycle_audit(root)
    if not audit.trustworthy or not producer_lifecycle_valid(note, audit):
        return False
    row = audit.latest.get(capability.note)
    return bool(
        isinstance(row, dict)
        and row.get("action") == capability.action
        and row.get("to") == capability.state
        and row.get("transition_id") == capability.transition_id
        and row.get("content_fingerprint") == capability.content_fingerprint
    )


def producer_lifecycle_capability_sealed(
    note: Any,
    capability: object,
) -> bool:
    """Whether a lifecycle seal may preserve an existing producer verdict.

    This is not write authorization. The note write door still performs the
    complete journal, preimage, detector, and lock checks below; this narrow
    predicate only prevents its generic machine-binding guard from destroying
    an unchanged producer proof before those checks can run.
    """
    try:
        content_fingerprint = _content_fingerprint(note)
    except (TypeError, ValueError):
        return False
    return bool(
        isinstance(capability, _ProducerLifecycleWriteCapability)
        and capability._seal is _WRITE_SEAL
        and capability.note == getattr(note, "name", None)
        and capability.action in _ACTIONS
        and capability.state == getattr(note, "state", None)
        and capability.content_fingerprint == content_fingerprint
    )
