"""Canonical binding between a verification verdict and the note bytes it judged.

Confidence and audit records are derived metadata.  They are deliberately
excluded from the fingerprint so writing a verdict does not change the content
being attested.  Human trust state and the legacy materialized ``superseded``
flag are lifecycle controls rather than factual content and are excluded too.
Every other persisted field that can affect an answer is bound.
"""

from __future__ import annotations

import copy
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
from typing import Any

from .acl import restricted_labels
from .event import _EVENT_ALLOWED_SOURCES, _EVENT_ID_PREFIX_ALIASES


_CONTENT_FIELDS = (
    "name",
    "title",
    "description",
    "type",
    "domain",
    "project",
    "acl",
    "entities",
    "components",
    "provenance",
    "source_lineages",
    "resource",
    "tags",
    "structured",
    "updated",
    "body",
)

_MACHINE_EVENT_PREFIXES = frozenset(
    set(_EVENT_ALLOWED_SOURCES).union(
        *(set(aliases) for aliases in _EVENT_ID_PREFIX_ALIASES.values())
    )
)
PRODUCER_GROUNDING_PREFIX = "producer-grounded-v1:"
PRODUCER_AUTHORITY_VERSION = 1
_SHA256_RE = re.compile(r"[a-f0-9]{64}\Z")
_PRODUCER_RE = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
_NOTE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
_GOVERNANCE_CLASS_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,127}\Z")
GOVERNANCE_HOLD_REASON_PREFIX = "governance-approval-required:"

HUMAN_APPROVAL_VERSION = 2
HUMAN_TRUST_STATES = frozenset({"verified", "superseded"})
_HUMAN_ACTION_STATE = {
    "promote": "verified",
    "demote": "proposed",
    "reject": "superseded",
}
AUTO_APPROVAL_VERSION = 1
AUTO_APPROVAL_METHOD = "auto-approve"
_MAX_TRUST_AUDIT_BYTES = 32 * 1024 * 1024
_MAX_TRUST_AUDIT_LINE_BYTES = 256 * 1024
_TRUST_WRITE_SEAL = object()
_PRODUCER_VERIFICATION_SEAL = object()
_GOVERNANCE_REWRITE_SEAL = object()
_AUTO_APPROVAL_SEAL = object()

# A machine-success grant is intentionally much narrower than the producer
# marker vocabulary.  New deterministic producers remain review-only until a
# validator that independently reloads their source authority is added here.
_CLOSED_PRODUCER_METHODS = frozenset(
    {
        ("contradiction-lint", "contradiction-detector"),
        ("artifact-publish", "artifact-event-projection"),
    }
)


@dataclass(frozen=True)
class HumanTrustAudit:
    """Latest human lifecycle transition per note plus log integrity state."""

    latest: dict[str, dict]
    trustworthy: bool = True


@dataclass(frozen=True)
class _HumanTrustWriteCapability:
    """One-shot-shaped authority for the exact transition already journaled."""

    note: str
    state: str
    approval_id: str
    content_fingerprint: str
    expected_existing_sha256: str
    _seal: object


@dataclass(frozen=True)
class _ProducerVerificationCapability:
    """Sealed authority for one exact producer result already in the audit.

    Caller-supplied ``sources`` and ``grounded=True`` are data, not authority.
    Only a closed validator in this module can create this object, and the
    write door accepts it only for the exact content/audit tuple it validated.
    """

    producer: str
    source_method: str
    source_fingerprint: str
    content_fingerprint: str
    audit_id: str
    _seal: object


@dataclass(frozen=True)
class _GovernanceRewriteCapability:
    """Authority for one exact rewrite of an already queued hard-held draft."""

    note: str
    review_class: str
    existing_sha256: str
    replacement_sha256: str
    _seal: object


@dataclass(frozen=True)
class _AutoApprovalCapability:
    """Sealed authority for one exact config-gated machine approval.

    Only :func:`authorize_auto_approval` can create this object, and the write
    door accepts it only for that note with ``state: verified`` at that exact
    content fingerprint and on-disk preimage.
    """

    note: str
    content_fingerprint: str
    audit_id: str
    expected_existing_sha256: str
    _seal: object


def _current_content_fingerprint(note: Any) -> str | None:
    """Return the canonical answer-content hash for a Note or index record."""
    if hasattr(note, "verification_content_fingerprint"):
        value = getattr(note, "verification_content_fingerprint", None)
        return value if isinstance(value, str) and _SHA256_RE.fullmatch(value) else None
    try:
        return verification_content_fingerprint(note)
    except Exception:
        return None


def _embedded_human_approval(note: Any) -> dict | None:
    """Validate the closed, content-bound approval block embedded in a note."""
    approval = getattr(note, "approval", None)
    if not isinstance(approval, dict) or set(approval) != {
        "version",
        "approval_id",
        "content_fingerprint",
        "identity",
        "state",
        "key_id",
        "signature",
    }:
        return None
    state = getattr(note, "state", None)
    current = _current_content_fingerprint(note)
    if (
        approval.get("version") != HUMAN_APPROVAL_VERSION
        or state not in HUMAN_TRUST_STATES
        or approval.get("state") != state
        or not isinstance(approval.get("approval_id"), str)
        or _SHA256_RE.fullmatch(approval["approval_id"]) is None
        or not isinstance(approval.get("content_fingerprint"), str)
        or _SHA256_RE.fullmatch(approval["content_fingerprint"]) is None
        or approval.get("content_fingerprint") != current
        or not isinstance(approval.get("identity"), str)
        or not approval["identity"].strip()
        or not isinstance(approval.get("key_id"), str)
        or not approval["key_id"].strip()
        or not isinstance(approval.get("signature"), str)
        or not approval["signature"].strip()
        or getattr(note, "superseded", None) is not (state == "superseded")
    ):
        return None
    return approval


_SIGNED_ROW_FIELDS = (
    "ts",
    "action",
    "note",
    "identity",
    "from",
    "to",
    "reason",
    "approval_version",
    "approval_id",
    "content_fingerprint",
    "expected_existing_sha256",
    "key_id",
)


def _decode_ed25519_key(value: str, *, expected_bytes: int) -> bytes:
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise PermissionError("human approval key is not canonical base64") from exc
    if len(raw) != expected_bytes:
        raise PermissionError("human approval key has an invalid length")
    return raw


def _approval_key_id(public_bytes: bytes) -> str:
    return hashlib.sha256(public_bytes).hexdigest()[:16]


def _approval_public_key(root: str):
    """Load the configured verifier key, deriving it from test/UI credentials.

    Production read paths should use ``trust.approval_public_key``.  Deriving
    from the private environment variable keeps the attended UI process and
    isolated tests self-contained without ever persisting private bytes in a
    note or journal.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    config_path = os.path.join(os.path.abspath(root), "config.json")
    try:
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        config = {}
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    trust = config.get("trust") or {} if isinstance(config, dict) else {}
    encoded_public = (
        trust.get("approval_public_key") if isinstance(trust, dict) else None
    )
    if isinstance(encoded_public, str) and encoded_public:
        try:
            public_bytes = _decode_ed25519_key(encoded_public, expected_bytes=32)
            public_key = Ed25519PublicKey.from_public_bytes(public_bytes)
        except (PermissionError, ValueError):
            return None
    else:
        # A signer is not a repository-wide trust root. Production readers
        # (MCP, nightly, CLI) do not hold the UI private credential, so allowing
        # the UI to derive a verifier key here would create approvals that only
        # that one process can recognize. Tests opt into derivation explicitly;
        # production must provision the matching public key in config.json.
        if os.environ.get("REMEMBRANCER_TEST_ALLOW_PRIVATE_APPROVAL_DERIVATION", "") != "1":
            return None
        encoded_private = os.environ.get("REMEMBRANCER_APPROVAL_PRIVATE_KEY", "")
        if not encoded_private:
            return None
        try:
            private_bytes = _decode_ed25519_key(encoded_private, expected_bytes=32)
            private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
            public_key = private_key.public_key()
            public_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        except (PermissionError, ValueError):
            return None
    key_id = _approval_key_id(public_bytes)
    configured_id = trust.get("approval_key_id") if isinstance(trust, dict) else None
    if configured_id not in (None, "", key_id):
        return None
    return public_key, key_id


def _approval_private_key(root: str):
    """Load UI-only signing authority; never fall back to repository data."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    encoded = os.environ.get("REMEMBRANCER_APPROVAL_PRIVATE_KEY", "")
    if not encoded:
        raise PermissionError(
            "human trust transitions require the attended UI signing credential"
        )
    private_bytes = _decode_ed25519_key(encoded, expected_bytes=32)
    private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public = _approval_public_key(root)
    if public is None or public[1] != _approval_key_id(public_bytes):
        raise PermissionError(
            "human approval signing key does not match configured verifier"
        )
    return private_key, public[1]


def _signed_row_bytes(row: dict) -> bytes:
    material = {field: row.get(field) for field in _SIGNED_ROW_FIELDS}
    return json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _parse_human_trust_audit(raw: bytes, verifier=None) -> HumanTrustAudit:
    """Parse both legacy rows and new bound rows; malformed bytes fail closed."""
    if len(raw) > _MAX_TRUST_AUDIT_BYTES or (raw and not raw.endswith(b"\n")):
        return HumanTrustAudit({}, trustworthy=False)
    latest: dict[str, dict] = {}
    seen_approval_ids: dict[str, str] = {}
    for raw_line in raw.splitlines():
        if not raw_line or len(raw_line) > _MAX_TRUST_AUDIT_LINE_BYTES:
            return HumanTrustAudit({}, trustworthy=False)
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return HumanTrustAudit({}, trustworthy=False)
        if not isinstance(row, dict):
            return HumanTrustAudit({}, trustworthy=False)
        name = row.get("note")
        action = row.get("action")
        target = row.get("to")
        identity = row.get("identity")
        if (
            not isinstance(name, str)
            or _NOTE_NAME_RE.fullmatch(name) is None
            or action not in _HUMAN_ACTION_STATE
            or target != _HUMAN_ACTION_STATE[action]
            or not isinstance(identity, str)
        ):
            return HumanTrustAudit({}, trustworthy=False)

        legacy_bound_fields = {
            "approval_id",
            "content_fingerprint",
            "approval_version",
        }
        bound_fields = {
            "approval_id",
            "content_fingerprint",
            "approval_version",
            "expected_existing_sha256",
            "key_id",
            "signature",
        }
        present = bound_fields.intersection(row)
        if present == legacy_bound_fields:
            approval_id = row.get("approval_id")
            fingerprint = row.get("content_fingerprint")
            if (
                row.get("approval_version") != 1
                or not isinstance(approval_id, str)
                or _SHA256_RE.fullmatch(approval_id) is None
                or not isinstance(fingerprint, str)
                or _SHA256_RE.fullmatch(fingerprint) is None
            ):
                return HumanTrustAudit({}, trustworthy=False)
            prior_name = seen_approval_ids.setdefault(approval_id, name)
            if prior_name != name:
                return HumanTrustAudit({}, trustworthy=False)
        elif present and present != bound_fields:
            return HumanTrustAudit({}, trustworthy=False)
        elif present:
            approval_id = row.get("approval_id")
            fingerprint = row.get("content_fingerprint")
            if (
                row.get("approval_version") != HUMAN_APPROVAL_VERSION
                or not isinstance(approval_id, str)
                or _SHA256_RE.fullmatch(approval_id) is None
                or not isinstance(fingerprint, str)
                or _SHA256_RE.fullmatch(fingerprint) is None
                or not isinstance(row.get("expected_existing_sha256"), str)
                or _SHA256_RE.fullmatch(row["expected_existing_sha256"]) is None
                or not isinstance(row.get("key_id"), str)
                or not row["key_id"]
                or not isinstance(row.get("signature"), str)
                or set(row) != set(_SIGNED_ROW_FIELDS) | {"signature"}
                or verifier is None
                or row["key_id"] != verifier[1]
            ):
                return HumanTrustAudit({}, trustworthy=False)
            try:
                signature = base64.b64decode(row["signature"], validate=True)
                verifier[0].verify(signature, _signed_row_bytes(row))
            except Exception:
                return HumanTrustAudit({}, trustworthy=False)
            prior_name = seen_approval_ids.setdefault(approval_id, name)
            if prior_name != name:
                return HumanTrustAudit({}, trustworthy=False)
        # A well-formed legacy row remains useful activity history, but it has
        # no content binding and therefore can never prove current trust.
        latest[name] = row
    return HumanTrustAudit(latest)


def load_human_trust_audit(root: str) -> HumanTrustAudit:
    """Read the append-only human trust journal under a shared lock."""
    from .locks import open_repository_state_directory

    try:
        state_descriptor, _created = open_repository_state_directory(
            root,
            create=False,
        )
    except OSError:
        return HumanTrustAudit({}, trustworthy=False)
    if state_descriptor < 0:
        return HumanTrustAudit({})
    try:
        descriptor = os.open(
            "trust_audit.jsonl",
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=state_descriptor,
        )
    except FileNotFoundError:
        os.close(state_descriptor)
        return HumanTrustAudit({})
    except OSError:
        os.close(state_descriptor)
        return HumanTrustAudit({}, trustworthy=False)
    try:
        with os.fdopen(descriptor, "rb") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                metadata = os.fstat(handle.fileno())
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size > _MAX_TRUST_AUDIT_BYTES
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                ):
                    return HumanTrustAudit({}, trustworthy=False)
                raw = handle.read(_MAX_TRUST_AUDIT_BYTES + 1)
            except OSError:
                return HumanTrustAudit({}, trustworthy=False)
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
    finally:
        os.close(state_descriptor)
    return _parse_human_trust_audit(raw, _approval_public_key(root))


def human_trust_valid(
    note: Any,
    audit: HumanTrustAudit | None = None,
) -> bool:
    """Whether a human lifecycle state is bound in both durable copies."""
    approval = _embedded_human_approval(note)
    if approval is None:
        return False
    if audit is None:
        # `load_notes` sets this private read-time result after checking the
        # durable journal. Recompute the content binding above on every call so
        # an in-memory body mutation immediately invalidates the approval.
        return bool(getattr(note, "_human_trust_audit_match", False))
    if not audit.trustworthy:
        return False
    row = audit.latest.get(str(getattr(note, "name", "") or ""))
    return bool(
        isinstance(row, dict)
        and row.get("approval_version") == HUMAN_APPROVAL_VERSION
        and row.get("approval_id") == approval["approval_id"]
        and row.get("content_fingerprint") == approval["content_fingerprint"]
        and row.get("identity") == approval["identity"]
        and row.get("to") == approval["state"]
        and row.get("key_id") == approval["key_id"]
        and row.get("signature") == approval["signature"]
    )


def bind_loaded_human_trust(notes: list[Any], root: str) -> HumanTrustAudit:
    """Attach the journal-validation result used by lifecycle helpers."""
    audit = load_human_trust_audit(root)
    for note in notes:
        note._human_trust_audit_match = human_trust_valid(note, audit)
    return audit


def _append_human_trust_row(root: str, row: dict) -> None:
    """Durably append a transition before the matching note bytes are written."""
    from .locks import open_repository_state_directory

    encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    verifier = _approval_public_key(root)
    if (
        len(encoded) > _MAX_TRUST_AUDIT_LINE_BYTES
        or not _parse_human_trust_audit(encoded, verifier).trustworthy
    ):
        raise ValueError("human trust row violates the signed journal schema")
    state_descriptor, _created = open_repository_state_directory(
        root,
        create=True,
    )
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            "trust_audit.jsonl",
            flags,
            0o600,
            dir_fd=state_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("human trust audit is not a regular file")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                from .verification_audit import read_descriptor_all

                existing = read_descriptor_all(
                    descriptor,
                    _MAX_TRUST_AUDIT_BYTES,
                )
                if not _parse_human_trust_audit(existing, verifier).trustworthy:
                    raise RuntimeError(
                        "human trust audit is malformed; refusing transition"
                    )
                if len(existing) + len(encoded) > _MAX_TRUST_AUDIT_BYTES:
                    raise RuntimeError("human trust audit exceeds safety bound")
                original_size = len(existing)
                offset = 0
                try:
                    while offset < len(encoded):
                        written = os.write(descriptor, encoded[offset:])
                        if written <= 0:
                            raise OSError("human trust audit append made no progress")
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


def _record_human_trust_transition(
    note: Any,
    *,
    root: str,
    action: str,
    identity: str,
    previous_state: str,
    target_state: str,
    expected_existing_sha256: str,
    reason: str = "",
) -> _HumanTrustWriteCapability:
    """Journal and bind one authorized transition, returning exact write authority."""
    from .locks import pipeline_mutation_lock_held

    root = os.path.abspath(root)
    if not pipeline_mutation_lock_held(root):
        raise PermissionError(
            "human trust transition requires the pipeline mutation lock"
        )
    name = str(getattr(note, "name", "") or "")
    config_path = os.path.join(root, "config.json")
    try:
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        config = {}
    except (OSError, json.JSONDecodeError) as exc:
        raise PermissionError(
            "cannot authorize human trust against malformed config"
        ) from exc
    if not isinstance(config, dict):
        raise PermissionError("cannot authorize human trust against malformed config")
    trust = config.get("trust") or {}
    identities = config.get("identities") or {}
    if not isinstance(trust, dict) or not isinstance(identities, dict):
        raise PermissionError("cannot authorize human trust against malformed config")
    approvers = trust.get("approver_identities", [])
    grants = identities.get(identity, []) if isinstance(identity, str) else []
    note_acl = list(getattr(note, "acl", None) or [])
    if (
        not isinstance(approvers, list)
        or identity not in approvers
        or identity not in identities
        or not isinstance(grants, list)
        or (note_acl and not set(note_acl).intersection(str(value) for value in grants))
    ):
        raise PermissionError("identity is not authorized for this trust transition")
    if (
        _NOTE_NAME_RE.fullmatch(name) is None
        or action not in _HUMAN_ACTION_STATE
        or _HUMAN_ACTION_STATE[action] != target_state
        or not isinstance(identity, str)
        or not identity.strip()
        or not isinstance(expected_existing_sha256, str)
        or _SHA256_RE.fullmatch(expected_existing_sha256) is None
        or len(str(reason or "").encode("utf-8")) > 8192
    ):
        raise ValueError("invalid human trust transition")
    current = verification_content_fingerprint(note)
    audit = load_human_trust_audit(root)
    if not audit.trustworthy:
        raise RuntimeError("human trust audit is malformed; refusing transition")
    latest = audit.latest.get(name)
    if (
        isinstance(latest, dict)
        and latest.get("approval_version") == HUMAN_APPROVAL_VERSION
        and latest.get("action") == action
        and latest.get("identity") == identity
        and latest.get("from") == previous_state
        and latest.get("to") == target_state
        and latest.get("reason") == str(reason or "")
        and latest.get("content_fingerprint") == current
        and latest.get("expected_existing_sha256") == expected_existing_sha256
    ):
        approval = None
        if target_state in HUMAN_TRUST_STATES:
            approval = {
                "version": HUMAN_APPROVAL_VERSION,
                "approval_id": latest["approval_id"],
                "content_fingerprint": current,
                "identity": identity,
                "state": target_state,
                "key_id": latest["key_id"],
                "signature": latest["signature"],
            }
        note.approval = approval
        note._human_trust_audit_match = target_state in HUMAN_TRUST_STATES
        return _HumanTrustWriteCapability(
            note=name,
            state=target_state,
            approval_id=latest["approval_id"],
            content_fingerprint=current,
            expected_existing_sha256=expected_existing_sha256,
            _seal=_TRUST_WRITE_SEAL,
        )

    private_key, key_id = _approval_private_key(root)
    approval_id = secrets.token_hex(32)
    row = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "action": action,
        "note": name,
        "identity": identity,
        "from": previous_state,
        "to": target_state,
        "reason": str(reason or ""),
        "approval_version": HUMAN_APPROVAL_VERSION,
        "approval_id": approval_id,
        "content_fingerprint": current,
        "expected_existing_sha256": expected_existing_sha256,
        "key_id": key_id,
    }
    row["signature"] = base64.b64encode(
        private_key.sign(_signed_row_bytes(row))
    ).decode("ascii")
    approval = None
    if target_state in HUMAN_TRUST_STATES:
        approval = {
            "version": HUMAN_APPROVAL_VERSION,
            "approval_id": approval_id,
            "content_fingerprint": current,
            "identity": identity,
            "state": target_state,
            "key_id": key_id,
            "signature": row["signature"],
        }
    _append_human_trust_row(root, row)
    note.approval = approval
    note._human_trust_audit_match = target_state in HUMAN_TRUST_STATES
    return _HumanTrustWriteCapability(
        note=name,
        state=target_state,
        approval_id=approval_id,
        content_fingerprint=current,
        expected_existing_sha256=expected_existing_sha256,
        _seal=_TRUST_WRITE_SEAL,
    )


def human_trust_write_authorized(
    note: Any,
    root: str,
    capability: object,
    *,
    existing_text: str | None,
) -> bool:
    """Validate that the exact requested write follows the latest durable row."""
    from .locks import pipeline_mutation_lock_held

    if (
        not isinstance(capability, _HumanTrustWriteCapability)
        or capability._seal is not _TRUST_WRITE_SEAL
        or capability.note != getattr(note, "name", None)
        or capability.state != getattr(note, "state", None)
        or capability.content_fingerprint != verification_content_fingerprint(note)
        or not isinstance(existing_text, str)
        or capability.expected_existing_sha256
        != hashlib.sha256(existing_text.encode("utf-8")).hexdigest()
        or not pipeline_mutation_lock_held(root)
    ):
        return False
    audit = load_human_trust_audit(root)
    if not audit.trustworthy:
        return False
    row = audit.latest.get(capability.note)
    if not isinstance(row, dict) or (
        row.get("approval_id") != capability.approval_id
        or row.get("content_fingerprint") != capability.content_fingerprint
        or row.get("to") != capability.state
    ):
        return False
    if capability.state in HUMAN_TRUST_STATES:
        return human_trust_valid(note, audit)
    return capability.state == "proposed" and getattr(note, "approval", None) is None


def verification_content_fingerprint(note: Any) -> str:
    """Return the stable SHA-256 for all answer-relevant note content."""
    payload = {field: getattr(note, field, None) for field in _CONTENT_FIELDS}
    # Match Note.to_markdown/from_frontmatter normalization so a verdict made
    # immediately before the atomic write still binds the reloaded note.
    for field in (
        "name",
        "title",
        "description",
        "type",
        "domain",
        "project",
        "resource",
        "updated",
    ):
        payload[field] = str(payload.get(field) or "").strip()
    payload["title"] = payload.get("title") or payload.get("name") or ""
    for field in (
        "acl",
        "entities",
        "components",
        "provenance",
        "source_lineages",
        "tags",
    ):
        payload[field] = [
            str(value).strip().strip("'\"")
            for value in (payload.get(field) or [])
            if str(value).strip().strip("'\"")
        ]
    # Text-mode reads normalize CRLF/bare-CR source material to LF. Bind the
    # pre-write draft to that same durable representation so a source-native
    # Windows document cannot make the governance queue mistake its own first
    # persistence for an unauthorized content rewrite.
    payload["body"] = (
        str(payload.get("body") or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .rstrip()
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def invalidate_verification(
    note: Any,
    *,
    reason: str = "content-changed",
) -> dict:
    """Replace any machine verdict with an explicit fail-closed current binding.

    This helper is used immediately before an in-place content/trust mutation is
    written.  Keeping a concrete ``skipped`` verdict prevents an older audit row
    from making the changed draft answer-eligible while it awaits review.
    """
    prior = getattr(note, "confidence", None)
    confidence = copy.deepcopy(prior) if isinstance(prior, dict) else {}
    try:
        prior_score = float(confidence.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        prior_score = 0.0
    confidence["score"] = min(max(prior_score, 0.0), 0.2)
    confidence["level"] = "low"
    confidence["human"] = "none"
    confidence["verification"] = {
        "status": "skipped",
        "checked": 0,
        "supported": 0,
        "unsupported": 0,
        "reason": str(reason or "content-changed"),
        "content_fingerprint": verification_content_fingerprint(note),
    }
    note.confidence = confidence
    return confidence


def governance_hold_class(note: Any) -> str | None:
    """Return the persisted mandatory-review class, failing closed on drift.

    The reason and the structured marker are redundant on purpose.  Either one
    is enough to retain the hold; disagreement is treated as an invalid hold,
    never as permission to publish.
    """
    confidence = getattr(note, "confidence", None)
    if not isinstance(confidence, dict):
        return None
    verification = confidence.get("verification")
    reason = verification.get("reason") if isinstance(verification, dict) else None
    reason_class: str | None = None
    if isinstance(reason, str) and reason.startswith(GOVERNANCE_HOLD_REASON_PREFIX):
        candidate = reason[len(GOVERNANCE_HOLD_REASON_PREFIX) :]
        reason_class = (
            candidate if _GOVERNANCE_CLASS_RE.fullmatch(candidate) else "invalid-hold"
        )
    marker = confidence.get("governance_hold")
    marker_class: str | None = None
    if isinstance(marker, dict) and marker.get("required") is True:
        candidate = marker.get("class")
        marker_class = (
            candidate
            if (
                isinstance(candidate, str) and _GOVERNANCE_CLASS_RE.fullmatch(candidate)
            )
            else "invalid-hold"
        )
    if reason_class is None:
        return marker_class
    if marker_class is None or marker_class == reason_class:
        return reason_class
    return "invalid-hold"


def apply_governance_hold(note: Any, cls: str) -> bool:
    """Install/rebind one mandatory-review hold on the current note content."""
    review_class = str(cls or "").strip()
    if _GOVERNANCE_CLASS_RE.fullmatch(review_class) is None:
        raise ValueError("governance hold class is invalid")
    reason = GOVERNANCE_HOLD_REASON_PREFIX + review_class
    confidence = getattr(note, "confidence", None)
    verification = (
        confidence.get("verification") if isinstance(confidence, dict) else None
    )
    if (
        isinstance(verification, dict)
        and verification.get("status") == "skipped"
        and verification.get("reason") == reason
        and verification.get("content_fingerprint")
        == verification_content_fingerprint(note)
        and governance_hold_class(note) == review_class
    ):
        return False
    prior = dict(verification) if isinstance(verification, dict) else None
    invalidate_verification(note, reason=reason)
    if prior is not None and not str(prior.get("reason") or "").startswith(
        GOVERNANCE_HOLD_REASON_PREFIX
    ):
        note.confidence["governance_prior_verification"] = prior
    note.confidence["governance_hold"] = {
        "required": True,
        "class": review_class,
        "reason": reason,
    }
    return True


def _issue_governance_rewrite_capability(
    note: Any,
    *,
    existing_text: str,
    review_class: str,
) -> object:
    """Seal one exact held-draft preimage/replacement pair for governance.

    The caller must independently prove that ``existing_text`` is the queued
    on-disk preimage.  This private constructor merely makes the shared write
    door enforce that the capability cannot be reused after either side of the
    rewrite changes.
    """
    if not isinstance(existing_text, str):
        raise TypeError("governance rewrite preimage must be text")
    review_class = str(review_class or "").strip()
    if _GOVERNANCE_CLASS_RE.fullmatch(review_class) is None:
        raise ValueError("governance hold class is invalid")
    apply_governance_hold(note, review_class)
    replacement = note.to_markdown()
    return _GovernanceRewriteCapability(
        note=str(getattr(note, "name", "") or ""),
        review_class=review_class,
        existing_sha256=hashlib.sha256(existing_text.encode("utf-8")).hexdigest(),
        replacement_sha256=hashlib.sha256(replacement.encode("utf-8")).hexdigest(),
        _seal=_GOVERNANCE_REWRITE_SEAL,
    )


def governance_rewrite_authorized(
    note: Any,
    *,
    existing_text: str | None,
    replacement_text: str,
    review_class: str,
    capability: object,
) -> bool:
    """Validate an exact, non-replayable rewrite of a hard-held proposal."""
    return bool(
        isinstance(capability, _GovernanceRewriteCapability)
        and capability._seal is _GOVERNANCE_REWRITE_SEAL
        and capability.note == getattr(note, "name", None)
        and capability.review_class == review_class
        and isinstance(existing_text, str)
        and capability.existing_sha256
        == hashlib.sha256(existing_text.encode("utf-8")).hexdigest()
        and capability.replacement_sha256
        == hashlib.sha256(replacement_text.encode("utf-8")).hexdigest()
    )


def machine_authored(note: Any) -> bool:
    """Whether durable source/producer metadata identifies a machine draft.

    A copied ``by:`` citation does not turn generated output into human
    authorship. Mutable source ownership, event ids, and explicit deterministic
    producer markers remain authoritative even when such a citation is also
    present.
    """
    provenance = [str(value) for value in (getattr(note, "provenance", None) or [])]
    return bool(
        getattr(note, "source_lineages", None)
        or getattr(note, "producer_lifecycle", None)
        or any(
            value.startswith(("generated-by:", "machine-update:"))
            or value.partition(":")[0] in _MACHINE_EVENT_PREFIXES
            for value in provenance
        )
    )


def ensure_machine_verification_binding(
    note: Any,
    *,
    allow_persisted_producer: bool = False,
) -> bool:
    """Fail-close a proposed machine note lacking a current content binding.

    Returns ``True`` when the confidence block was replaced. Direct human
    drafts (``by:`` provenance) and legacy/manual notes without recognizable
    machine provenance retain their historical representation.
    """
    if getattr(note, "state", None) != "proposed":
        return False
    provenance = [str(value) for value in (getattr(note, "provenance", None) or [])]
    confidence = getattr(note, "confidence", None)
    verification = (
        confidence.get("verification") if isinstance(confidence, dict) else None
    )
    current = verification_content_fingerprint(note)
    status = verification.get("status") if isinstance(verification, dict) else None
    bound_fingerprint = (
        verification.get("content_fingerprint")
        if isinstance(verification, dict)
        else None
    )
    # Authorship is not approval. Generic CoVe can audit claims but cannot
    # self-attest the persisted note's slug/type/domain/ACL classification.
    # Consequently only a validated deterministic producer proof may carry a
    # machine ``verified`` status through the write door. Human approval is the
    # separate ``state: verified`` lifecycle path handled above.
    producer_success = bool(
        producer_verification_write_authorized(note)
        or (allow_persisted_producer and producer_verification_valid(note))
    )
    if status == "verified" and bound_fingerprint is not None:
        if not producer_success:
            invalidate_verification(
                note, reason="generic-cove-cannot-self-attest-classification"
            )
            return True

    if any(value.startswith("by:") for value in provenance):
        return False
    machine_owned = machine_authored(note)
    if not machine_owned:
        return False
    safe_failure = status in {"flagged", "skipped"} and bound_fingerprint == current
    valid_success = (
        status == "verified" and bound_fingerprint == current and producer_success
    )
    if safe_failure or valid_success:
        return False
    invalidate_verification(note, reason="machine-content-awaits-verification")
    return True


def _producer_audit_id(
    producer: str,
    method: str,
    source_fingerprint: str,
    content_fingerprint: str,
    status: str,
) -> str:
    payload = json.dumps(
        {
            "producer": producer,
            "method": method,
            "source_fingerprint": source_fingerprint,
            "content_fingerprint": content_fingerprint,
            "status": status,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def producer_source_fingerprint(
    producer: str,
    source_method: str,
    sources: object,
) -> str:
    """Hash one canonical producer/source envelope.

    This function deliberately supplies identity, not authority.  A hash of
    caller-controlled JSON cannot make a source trustworthy; it is useful only
    after a closed validator independently selected ``sources``.
    """
    payload = {
        "producer": producer,
        "source_method": source_method,
        "sources": sources,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _producer_audit_matches(note: Any, audit: object, values: dict) -> bool:
    """Require the producer success in both frontmatter and the shared log."""
    if audit is None or not bool(getattr(audit, "trustworthy", False)):
        return False
    name = str(getattr(note, "name", "") or "")
    statuses = getattr(audit, "statuses", {})
    fingerprints = getattr(audit, "content_fingerprints", {})
    audit_ids = getattr(audit, "audit_ids", {})
    return bool(
        statuses.get(name) == "verified"
        and fingerprints.get(name) == values.get("content_fingerprint")
        and audit_ids.get(name) == values.get("audit_id")
    )


def _closed_producer_note_shape_valid(
    note: Any,
    producer: str,
    source_method: str,
) -> bool:
    """Revalidate persisted fields owned by an allowlisted producer."""
    if (producer, source_method) == ("artifact-publish", "artifact-event-projection"):
        from distill.artifact_verification import note_shape_valid

        return note_shape_valid(note)
    if (producer, source_method) != (
        "contradiction-lint",
        "contradiction-detector",
    ):
        return False
    # Index records intentionally omit bodies/structured metadata. They are
    # only a discovery optimization and retrieval rechecks the loaded Note
    # before returning it. Do not reject the candidate before that full check.
    if hasattr(note, "verification"):
        return True
    structured = getattr(note, "structured", None)
    pair = structured.get("pair") if isinstance(structured, dict) else None
    if (
        not isinstance(pair, list)
        or len(pair) != 2
        or any(
            not isinstance(item, str) or _NOTE_NAME_RE.fullmatch(item) is None
            for item in pair
        )
        or pair != sorted(pair)
        or pair[0] == pair[1]
        or structured != {"kind": "contradiction", "pair": pair}
    ):
        return False

    def slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80] or "note"

    base = f"contradiction-{slug(pair[0])}-vs-{slug(pair[1])}"
    if len(base) <= 80:
        expected_name = base
    else:
        digest = hashlib.sha256(f"{pair[0]}\0{pair[1]}".encode("utf-8")).hexdigest()[
            :10
        ]
        expected_name = f"{base[:69].rstrip('-')}-{digest}"
    name = getattr(note, "name", None)
    expected_body = (
        f"Auto-detected contradiction between [[{pair[0]}]] and [[{pair[1]}]].\n\n"
        "Resolve by updating one of the source notes or reconciling the conflict."
    )
    provenance = list(getattr(note, "provenance", None) or [])
    generated = [
        item
        for item in provenance
        if isinstance(item, str) and item.startswith("generated-by:")
    ]
    return bool(
        name == expected_name
        and getattr(note, "description", None)
        == f"Auto-detected contradiction between {pair[0]} and {pair[1]}"
        and getattr(note, "type", None) == "contradiction"
        and getattr(note, "domain", None) == "company"
        and getattr(note, "title", "") in ("", name)
        and getattr(note, "project", "") in ("", None)
        and list(getattr(note, "components", None) or []) == []
        and list(getattr(note, "source_lineages", None) or []) == []
        and getattr(note, "resource", "") in ("", None)
        and list(getattr(note, "tags", None) or []) == []
        and str(getattr(note, "body", "") or "").rstrip() == expected_body
        and getattr(note, "state", None) in {"proposed", "superseded"}
        and generated == ["generated-by:contradiction-lint"]
        and not any(
            isinstance(item, str) and item.startswith(("by:", "machine-update:"))
            for item in provenance
        )
        and getattr(note, "approval", None) is None
        and getattr(note, "source_retirement", None) is None
    )


def producer_verification_write_authorized(note: Any) -> bool:
    """Validate the in-memory sealed capability for a fresh producer write."""
    capability = getattr(note, "_producer_verification_capability", None)
    if (
        not isinstance(capability, _ProducerVerificationCapability)
        or capability._seal is not _PRODUCER_VERIFICATION_SEAL
        or capability.content_fingerprint != verification_content_fingerprint(note)
    ):
        return False
    confidence = getattr(note, "confidence", None)
    verification = (
        confidence.get("verification") if isinstance(confidence, dict) else None
    )
    return bool(
        isinstance(verification, dict)
        and (capability.producer, capability.source_method) in _CLOSED_PRODUCER_METHODS
        and _closed_producer_note_shape_valid(
            note,
            capability.producer,
            capability.source_method,
        )
        and getattr(note, "producer_lifecycle", None) is None
        and verification.get("status") == "verified"
        and verification.get("producer") == capability.producer
        and verification.get("method")
        == PRODUCER_GROUNDING_PREFIX + capability.source_method
        and verification.get("source_fingerprint") == capability.source_fingerprint
        and verification.get("content_fingerprint") == capability.content_fingerprint
        and verification.get("audit_id") == capability.audit_id
        and verification.get("grounded") is True
    )


def producer_grounding_capability_valid(
    note: Any,
    capability: object,
    *,
    producer: str,
    source_method: str,
    source_fingerprint: str,
    audit_id: str,
) -> bool:
    """Check a closed validator's seal before a confidence block is built."""
    return bool(
        isinstance(capability, _ProducerVerificationCapability)
        and capability._seal is _PRODUCER_VERIFICATION_SEAL
        and capability.producer == producer
        and capability.source_method == source_method
        and capability.source_fingerprint == source_fingerprint
        and capability.content_fingerprint == verification_content_fingerprint(note)
        and capability.audit_id == audit_id
        and (producer, source_method) in _CLOSED_PRODUCER_METHODS
        and _closed_producer_note_shape_valid(note, producer, source_method)
        and getattr(note, "producer_lifecycle", None) is None
    )


def _append_producer_verification_audit(root: str, row: dict) -> None:
    """Append through the shared rollback-evident audit/head transaction."""
    from .verification_audit import append_verification_audit_record
    from retrieve.verification import load_verification_audit

    path = os.path.join(os.path.abspath(root), "state", "verification.jsonl")
    if not append_verification_audit_record(row, path):
        raise OSError(
            "producer verification audit/head append failed; refusing authority"
        )
    committed = load_verification_audit(root)
    if not (
        committed.trustworthy
        and committed.statuses.get(row.get("note")) == row.get("status")
        and committed.content_fingerprints.get(row.get("note"))
        == row.get("content_fingerprint")
        and committed.audit_ids.get(row.get("note")) == row.get("audit_id")
    ):
        raise RuntimeError("producer verification audit/head commit did not revalidate")


def authorize_contradiction_producer_verification(
    note: Any,
    *,
    root: str,
) -> tuple[_ProducerVerificationCapability, object, str]:
    """Independently replay and seal the one closed producer contract.

    The caller cannot supply source notes.  The validator reloads the current
    corpus under the pipeline lock, reruns the contradiction detector, derives
    the exact output projection, and requires both inputs to carry current
    strong trust before it appends a success row.
    """
    from .locks import pipeline_mutation_lock_held
    from .acl import combine_acl
    from .note import current_note_names, load_notes
    from lint.engine import find_contradictions, _contradiction_note_name
    from tools.source_eligibility import sources_strongly_verified

    root = os.path.abspath(root)
    if not pipeline_mutation_lock_held(root):
        raise PermissionError(
            "closed producer verification requires the pipeline mutation lock"
        )
    context_dir = os.path.join(root, "context")
    physical = load_notes(context_dir)
    groups: dict[str, list[Any]] = {}
    for candidate in physical:
        groups.setdefault(candidate.name, []).append(candidate)
    structured = getattr(note, "structured", None)
    pair = structured.get("pair") if isinstance(structured, dict) else None
    if (
        not isinstance(pair, list)
        or len(pair) != 2
        or any(
            not isinstance(item, str) or _NOTE_NAME_RE.fullmatch(item) is None
            for item in pair
        )
        or pair != sorted(pair)
        or pair[0] == pair[1]
    ):
        raise PermissionError("contradiction validator requires a canonical pair")
    normalized = (pair[0], pair[1])
    current = current_note_names(physical)
    sources = []
    for name in normalized:
        matches = groups.get(name, [])
        if len(matches) != 1 or name not in current:
            raise PermissionError(
                "contradiction validator requires unique current source notes"
            )
        sources.append(matches[0])
    detector_notes = [candidate for candidate in physical if candidate.name in current]
    detected = {tuple(sorted(item)) for item in find_contradictions(detector_notes)}
    if normalized not in detected:
        raise PermissionError(
            "contradiction pair is not reproducible from current source bytes"
        )
    if not sources_strongly_verified(root, sources):
        raise PermissionError(
            "contradiction sources lack current authenticated verification"
        )

    expected_provenance = sorted(
        {
            item
            for source in sources
            for item in (source.provenance or [])
            if not str(item).startswith(("by:", "generated-by:", "machine-update:"))
        }
    ) + ["generated-by:contradiction-lint"]
    expected_entities = sorted(
        {entity for source in sources for entity in (source.entities or [])}
    )
    expected_body = (
        f"Auto-detected contradiction between [[{normalized[0]}]] and "
        f"[[{normalized[1]}]].\n\n"
        "Resolve by updating one of the source notes or reconciling the conflict."
    )
    expected_acl = combine_acl(*(list(source.acl or []) for source in sources))
    exact = bool(
        getattr(note, "name", None) == _contradiction_note_name(*normalized)
        and getattr(note, "description", None)
        == f"Auto-detected contradiction between {normalized[0]} and {normalized[1]}"
        and getattr(note, "type", None) == "contradiction"
        and getattr(note, "domain", None) == "company"
        and getattr(note, "title", "") in ("", getattr(note, "name", None))
        and getattr(note, "project", "") in ("", None)
        and list(getattr(note, "acl", None) or []) == expected_acl
        and list(getattr(note, "entities", None) or []) == expected_entities
        and list(getattr(note, "components", None) or []) == []
        and list(getattr(note, "provenance", None) or []) == expected_provenance
        and list(getattr(note, "source_lineages", None) or []) == []
        and getattr(note, "resource", "") in ("", None)
        and list(getattr(note, "tags", None) or []) == []
        and structured == {"kind": "contradiction", "pair": list(normalized)}
        and str(getattr(note, "body", "") or "").rstrip() == expected_body
        and getattr(note, "state", None) == "proposed"
        and getattr(note, "superseded", False) is False
        and getattr(note, "approval", None) is None
        and getattr(note, "source_retirement", None) is None
        and getattr(note, "producer_lifecycle", None) is None
    )
    if not exact:
        raise PermissionError(
            "contradiction output is not the exact deterministic projection"
        )

    source_material = {
        "detector": "lint.find-contradictions-v1",
        "pair": list(normalized),
        "sources": [
            {
                "name": source.name,
                "content_fingerprint": verification_content_fingerprint(source),
                "state": getattr(source, "state", None),
                "approval_id": (
                    (getattr(source, "approval", None) or {}).get("approval_id")
                    if isinstance(getattr(source, "approval", None), dict)
                    else None
                ),
                "verification_audit_id": (
                    (
                        (getattr(source, "confidence", None) or {}).get("verification")
                        or {}
                    ).get("audit_id")
                    if isinstance(getattr(source, "confidence", None), dict)
                    else None
                ),
            }
            for source in sources
        ],
    }
    producer = "contradiction-lint"
    source_method = "contradiction-detector"
    source_fingerprint = producer_source_fingerprint(
        producer,
        source_method,
        source_material,
    )
    content_fingerprint = verification_content_fingerprint(note)
    method = PRODUCER_GROUNDING_PREFIX + source_method
    audit_id = _producer_audit_id(
        producer,
        method,
        source_fingerprint,
        content_fingerprint,
        "verified",
    )
    from retrieve.verification import load_verification_audit

    audit = load_verification_audit(root)
    if not audit.trustworthy:
        raise RuntimeError(
            "verification audit is malformed; refusing producer authority"
        )
    if not (
        audit.statuses.get(note.name) == "verified"
        and audit.content_fingerprints.get(note.name) == content_fingerprint
        and audit.audit_ids.get(note.name) == audit_id
    ):
        _append_producer_verification_audit(
            root,
            {
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "authority_version": PRODUCER_AUTHORITY_VERSION,
                "audit_id": audit_id,
                "note": note.name,
                "producer": producer,
                "method": method,
                "source_fingerprint": source_fingerprint,
                "content_fingerprint": content_fingerprint,
                "checked": 1,
                "supported": 1,
                "unsupported": 0,
                "status": "verified",
                "grounded": True,
                "reason": (
                    "closed replay of deterministic contradiction detector over "
                    "authenticated current source notes"
                ),
            },
        )
    capability = _ProducerVerificationCapability(
        producer=producer,
        source_method=source_method,
        source_fingerprint=source_fingerprint,
        content_fingerprint=content_fingerprint,
        audit_id=audit_id,
        _seal=_PRODUCER_VERIFICATION_SEAL,
    )
    return (
        capability,
        source_material,
        (
            "closed replay of deterministic contradiction detector over "
            "authenticated current source notes"
        ),
    )


def authorize_artifact_producer_verification(
    note: Any,
    *,
    root: str,
) -> tuple[_ProducerVerificationCapability, dict]:
    """Seal only an exact canonical artifact replay from authenticated L1.

    No source events or caller-provided success bit cross this interface. The
    dedicated validator reloads source authority and refuses human notes,
    replacement proposals, edited projections and ambiguous source histories.
    """
    from distill.artifact_verification import (
        MARKER,
        METHOD,
        PRODUCER,
        ArtifactProjectionRefused,
        replay_source_material,
    )
    from remembrancerlib.event import EventValidationError

    try:
        sources = replay_source_material(note, root=root)
    except (OSError, ValueError, EventValidationError) as exc:
        raise ArtifactProjectionRefused(str(exc)) from exc
    original_provenance = list(note.provenance)
    if MARKER not in note.provenance:
        note.provenance = [*note.provenance, MARKER]
    source_fingerprint = producer_source_fingerprint(PRODUCER, METHOD, sources)
    content_fingerprint = verification_content_fingerprint(note)
    method = PRODUCER_GROUNDING_PREFIX + METHOD
    audit_id = _producer_audit_id(
        PRODUCER,
        method,
        source_fingerprint,
        content_fingerprint,
        "verified",
    )
    verification = {
        "authority_version": PRODUCER_AUTHORITY_VERSION,
        "producer": PRODUCER,
        "method": method,
        "source_fingerprint": source_fingerprint,
        "content_fingerprint": content_fingerprint,
        "audit_id": audit_id,
        "checked": 1,
        "supported": 1,
        "unsupported": 0,
        "status": "verified",
        "grounded": True,
        "reason": "exact canonical artifact projection from authenticated latest L1 publish",
    }
    try:
        from retrieve.verification import load_verification_audit

        audit = load_verification_audit(root)
        if not audit.trustworthy:
            raise RuntimeError(
                "verification audit is malformed; refusing producer authority"
            )
        if not (
            audit.statuses.get(note.name) == "verified"
            and audit.content_fingerprints.get(note.name) == content_fingerprint
            and audit.audit_ids.get(note.name) == audit_id
        ):
            _append_producer_verification_audit(
                root,
                {
                    "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "note": note.name,
                    **verification,
                },
            )
    except Exception:
        note.provenance = original_provenance
        raise
    return _ProducerVerificationCapability(
        producer=PRODUCER,
        source_method=METHOD,
        source_fingerprint=source_fingerprint,
        content_fingerprint=content_fingerprint,
        audit_id=audit_id,
        _seal=_PRODUCER_VERIFICATION_SEAL,
    ), verification


def producer_verification_valid(note: Any, audit: object | None = None) -> bool:
    """Validate a producer-grounded result, including its derived audit id.

    Both full ``Note`` objects and v2 index records expose the required fields;
    this keeps index-first and authoritative full-scan retrieval identical.
    """
    if hasattr(note, "verification"):
        values = {
            "status": getattr(note, "verification", None),
            "producer": getattr(note, "verification_producer", None),
            "method": getattr(note, "verification_method", None),
            "source_fingerprint": getattr(
                note, "verification_source_fingerprint", None
            ),
            "content_fingerprint": getattr(
                note, "embedded_verification_content_fingerprint", None
            ),
            "audit_id": getattr(note, "verification_audit_id", None),
            "grounded": getattr(note, "verification_grounded", None),
            "reason": getattr(note, "verification_reason", None),
            "checked": getattr(note, "verification_checked", None),
            "supported": getattr(note, "verification_supported", None),
            "unsupported": getattr(note, "verification_unsupported", None),
        }
        current = getattr(note, "verification_content_fingerprint", None)
    else:
        confidence = getattr(note, "confidence", None)
        if not isinstance(confidence, dict):
            return False
        verification = confidence.get("verification")
        if not isinstance(verification, dict):
            return False
        values = verification
        current = verification_content_fingerprint(note)

    producer = values.get("producer")
    method = values.get("method")
    status = values.get("status")
    source_fingerprint = values.get("source_fingerprint")
    content_fingerprint = values.get("content_fingerprint")
    audit_id = values.get("audit_id")
    grounded = values.get("grounded")
    reason = values.get("reason")
    try:
        checked = int(values.get("checked"))
        supported = int(values.get("supported"))
        unsupported = int(values.get("unsupported"))
    except (TypeError, ValueError):
        return False

    provenance = [str(item) for item in (getattr(note, "provenance", None) or [])]
    markers = {
        value[len("generated-by:") :]
        for value in provenance
        if value.startswith("generated-by:")
    }
    if (
        not isinstance(producer, str)
        or _PRODUCER_RE.fullmatch(producer) is None
        or markers != {producer}
        or not isinstance(method, str)
        or not method.startswith(PRODUCER_GROUNDING_PREFIX)
        or _PRODUCER_RE.fullmatch(method[len(PRODUCER_GROUNDING_PREFIX) :]) is None
        or status not in {"verified", "skipped"}
        or not isinstance(grounded, bool)
        or grounded != (status == "verified")
        or not isinstance(reason, str)
        or not reason.strip()
        or not isinstance(source_fingerprint, str)
        or _SHA256_RE.fullmatch(source_fingerprint) is None
        or not isinstance(content_fingerprint, str)
        or _SHA256_RE.fullmatch(content_fingerprint) is None
        or content_fingerprint != current
        or not isinstance(audit_id, str)
        or _SHA256_RE.fullmatch(audit_id) is None
        or min(checked, supported, unsupported) < 0
        or supported + unsupported > checked
        or (grounded and (checked < 1 or supported < 1 or unsupported != 0))
        or (not grounded and supported != 0)
    ):
        return False
    if audit_id != _producer_audit_id(
        producer,
        method,
        source_fingerprint,
        content_fingerprint,
        status,
    ):
        return False
    if status != "verified":
        # A structurally bound refusal is useful terminal/review metadata but
        # is never an answer grant, so it needs no authority journal.
        return True
    source_method = method[len(PRODUCER_GROUNDING_PREFIX) :]
    if (producer, source_method) not in _CLOSED_PRODUCER_METHODS:
        return False
    if not _closed_producer_note_shape_valid(note, producer, source_method):
        return False
    if producer_verification_write_authorized(note):
        return True
    if audit is None:
        # Full Note objects loaded from disk carry an absolute/relative path;
        # internal lifecycle checks can recover the same audit used by
        # retrieval. Index records have no repository root and must receive the
        # audit explicitly from the retrieval caller.
        path = str(getattr(note, "path", "") or "")
        marker = f"{os.sep}context{os.sep}"
        absolute = os.path.abspath(path) if path else ""
        if marker in absolute:
            root = absolute.split(marker, 1)[0]
            try:
                from retrieve.verification import load_verification_audit

                audit = load_verification_audit(root)
            except Exception:
                audit = None
    return _producer_audit_matches(note, audit, values)


def auto_approval_enabled(cfg: Any) -> bool:
    """Whether config enables machine auto-approval (only literal ``true``)."""
    trust = cfg.get("trust") if isinstance(cfg, dict) else None
    if not isinstance(trust, dict):
        return False
    return trust.get("auto_approve") is True


def _auto_approval_acl_denied(note: Any, labels: frozenset[str] | None) -> bool:
    """Whether a restricted label rules machine approval out; unknown labels rule it out."""
    if labels is None:
        return True
    acl = getattr(note, "acl", None) or []
    if isinstance(acl, str):
        acl = [acl]
    return bool(set(str(value) for value in acl) & labels)


def _note_repository_root(note: Any) -> str | None:
    """Recover the repository root from a loaded note's context path."""
    path = str(getattr(note, "path", "") or "")
    marker = f"{os.sep}context{os.sep}"
    absolute = os.path.abspath(path) if path else ""
    if marker in absolute:
        return absolute.split(marker, 1)[0]
    return None


def _auto_approval_audit_id(name: str, content_fingerprint: str) -> str:
    """Derive the deterministic audit id for one exact machine approval."""
    payload = json.dumps(
        {
            "method": AUTO_APPROVAL_METHOD,
            "note": name,
            "content_fingerprint": content_fingerprint,
            "status": "verified",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def authorize_auto_approval(
    note: Any,
    *,
    cfg: dict,
    audit_reason: str,
) -> _AutoApprovalCapability:
    """Seal one exact machine approval for a persisted, held proposal.

    Policy lives here and nowhere else: the reviewed config gate must be
    literal ``true`` and the note's ACL must carry no restricted label
    (``remembrancerlib.acl.restricted_labels``).  The write door and the read gate revalidate
    both predicates, so a stale capability cannot outlive the gate.  The
    journaled row binds the note's current content fingerprint; any later
    content edit invalidates the approval exactly like the signed human path.
    """
    from .locks import pipeline_mutation_lock_held
    from .note import Note, parse_frontmatter

    if not auto_approval_enabled(cfg):
        raise PermissionError("machine auto-approval is not enabled in config")
    name = str(getattr(note, "name", "") or "")
    if _NOTE_NAME_RE.fullmatch(name) is None:
        raise ValueError("invalid auto-approval note name")
    if _auto_approval_acl_denied(note, restricted_labels(cfg)):
        raise PermissionError("notes with a restricted label are never auto-approved")
    reason = str(audit_reason or "")
    if not reason.strip() or len(reason.encode("utf-8")) > 8192:
        raise ValueError("invalid auto-approval reason")
    root = _note_repository_root(note)
    if root is None:
        raise PermissionError("auto-approval requires a repository-bound note")
    if not pipeline_mutation_lock_held(root):
        raise PermissionError(
            "machine auto-approval requires the pipeline mutation lock"
        )
    path = str(getattr(note, "path", "") or "")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PermissionError("auto-approval requires the persisted held note") from exc
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    preimage = b"".join(chunks)
    try:
        existing_fm, existing_body = parse_frontmatter(preimage.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise PermissionError("auto-approval target is not UTF-8") from exc
    persisted = Note.from_frontmatter(existing_fm, existing_body, path)
    current = verification_content_fingerprint(note)
    if (
        getattr(persisted, "state", None) != "proposed"
        or verification_content_fingerprint(persisted) != current
    ):
        raise PermissionError(
            "auto-approval candidate does not match the persisted proposal"
        )
    audit_id = _auto_approval_audit_id(name, current)
    from retrieve.verification import load_verification_audit

    audit = load_verification_audit(root)
    if not audit.trustworthy:
        raise RuntimeError("verification audit is malformed; refusing auto-approval")
    if not (
        audit.statuses.get(name) == "verified"
        and audit.content_fingerprints.get(name) == current
        and audit.audit_ids.get(name) == audit_id
    ):
        _append_producer_verification_audit(
            root,
            {
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "authority_version": AUTO_APPROVAL_VERSION,
                "audit_id": audit_id,
                "note": name,
                "method": AUTO_APPROVAL_METHOD,
                "content_fingerprint": current,
                "checked": 1,
                "supported": 1,
                "unsupported": 0,
                "status": "verified",
                "reason": reason,
            },
        )
    # The embedded marker is the note-side copy of the two-phase binding.  It
    # deliberately preserves a durable governance hold: the approval releases
    # the hold at read time without erasing its history.
    prior = getattr(note, "confidence", None)
    confidence = copy.deepcopy(prior) if isinstance(prior, dict) else {}
    confidence["verification"] = {
        "status": "verified",
        "method": AUTO_APPROVAL_METHOD,
        "content_fingerprint": current,
        "audit_id": audit_id,
        "checked": 1,
        "supported": 1,
        "unsupported": 0,
        "reason": reason,
    }
    note.confidence = confidence
    note._auto_approval_match = True
    return _AutoApprovalCapability(
        note=name,
        content_fingerprint=current,
        audit_id=audit_id,
        expected_existing_sha256=hashlib.sha256(preimage).hexdigest(),
        _seal=_AUTO_APPROVAL_SEAL,
    )


def auto_approval_write_authorized(
    note: Any,
    root: str,
    capability: object,
    *,
    existing_text: str | None,
) -> bool:
    """Validate that the exact requested write follows the durable machine row."""
    from retrieve.maintenance import read_config

    from .locks import pipeline_mutation_lock_held

    if (
        not isinstance(capability, _AutoApprovalCapability)
        or capability._seal is not _AUTO_APPROVAL_SEAL
        or capability.note != getattr(note, "name", None)
        or getattr(note, "state", None) != "verified"
        or capability.content_fingerprint != verification_content_fingerprint(note)
        or not isinstance(existing_text, str)
        or capability.expected_existing_sha256
        != hashlib.sha256(existing_text.encode("utf-8")).hexdigest()
        or _auto_approval_acl_denied(note, restricted_labels(read_config(root)))
        or not pipeline_mutation_lock_held(root)
    ):
        return False
    from retrieve.verification import load_verification_audit

    audit = load_verification_audit(root)
    if not audit.trustworthy or audit.auto_approve is not True:
        return False
    if not (
        audit.statuses.get(capability.note) == "verified"
        and audit.content_fingerprints.get(capability.note)
        == capability.content_fingerprint
        and audit.audit_ids.get(capability.note) == capability.audit_id
    ):
        return False
    return auto_approval_valid(note, audit)


def auto_approval_valid(note: Any, audit: object | None = None) -> bool:
    """Validate a config-gated machine approval bound in both durable copies.

    Both full ``Note`` objects and v2 index records expose the required
    fields; this keeps index-first and authoritative full-scan retrieval
    identical.  The gate travels with the audit's load-time config verdict, so
    disabling ``trust.auto_approve`` immediately revokes read-time honoring.
    """
    # Without an audit, the bound read-time result below carries the config's
    # labels; with one, the labels it loaded from config.
    labels = restricted_labels(None) if audit is None else getattr(audit, "restricted_labels", None)
    if getattr(note, "state", None) != "verified" or _auto_approval_acl_denied(note, labels):
        return False
    if hasattr(note, "verification"):
        values = {
            "status": getattr(note, "verification", None),
            "method": getattr(note, "verification_method", None),
            "content_fingerprint": getattr(
                note, "embedded_verification_content_fingerprint", None
            ),
            "audit_id": getattr(note, "verification_audit_id", None),
        }
    else:
        confidence = getattr(note, "confidence", None)
        if not isinstance(confidence, dict):
            return False
        verification = confidence.get("verification")
        if not isinstance(verification, dict):
            return False
        values = verification
    name = str(getattr(note, "name", "") or "")
    fingerprint = values.get("content_fingerprint")
    audit_id = values.get("audit_id")
    if (
        _NOTE_NAME_RE.fullmatch(name) is None
        or values.get("status") != "verified"
        or values.get("method") != AUTO_APPROVAL_METHOD
        or not isinstance(fingerprint, str)
        or _SHA256_RE.fullmatch(fingerprint) is None
        or fingerprint != _current_content_fingerprint(note)
        or not isinstance(audit_id, str)
        or audit_id != _auto_approval_audit_id(name, fingerprint)
    ):
        return False
    if audit is None:
        # `load_notes` sets this private read-time result after checking the
        # durable journal and the config gate.  Recompute the content binding
        # above on every call so an in-memory body mutation immediately
        # invalidates the approval.
        return bool(getattr(note, "_auto_approval_match", False))
    if (
        not bool(getattr(audit, "trustworthy", False))
        or getattr(audit, "auto_approve", False) is not True
    ):
        return False
    return bool(
        getattr(audit, "statuses", {}).get(name) == "verified"
        and getattr(audit, "content_fingerprints", {}).get(name) == fingerprint
        and getattr(audit, "audit_ids", {}).get(name) == audit_id
    )


def bind_loaded_auto_approvals(
    notes: list[Any],
    root: str,
    verification_audit: Any = None,
) -> None:
    """Attach the journal/gate validation result used by lifecycle helpers.

    A caller that already holds the audit for ``root`` passes it, so one read
    is not paid twice; omitting it loads the audit exactly as before.
    """
    from retrieve.verification import load_verification_audit

    audit = (
        verification_audit
        if verification_audit is not None
        else load_verification_audit(root)
    )
    for note in notes:
        note._auto_approval_match = auto_approval_valid(note, audit)
