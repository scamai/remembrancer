"""Fail-closed source selection shared by generated memory projections."""
from __future__ import annotations

import json
import os
from typing import Iterable, Any

from retrieve.verification import (
    load_verification_audit,
    verification_status,
    verification_allows_answer,
    verification_gate_enabled,
)
from remembrancerlib.verification import (
    human_trust_valid,
    producer_verification_valid,
    verification_content_fingerprint,
)


def is_generated_projection(note: Any) -> bool:
    """Whether a note is a machine projection rather than source authority."""
    return any(
        isinstance(value, str) and value.startswith("generated-by:")
        for value in (getattr(note, "provenance", None) or [])
    )


def eligible_source_notes(root: str, notes: Iterable[Any]) -> list[Any]:
    """Return current notes safe to quote into another generated projection.

    This is intentionally stricter than merely checking an embedded status:
    it uses the same content-bound audit resolution as answer retrieval and
    excludes generated pages so one machine claim cannot recursively become
    authority for the next.  A malformed existing config/audit aborts the
    producer rather than causing it to publish an empty or laundered view.
    """
    config_path = os.path.join(os.path.abspath(root), "config.json")
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError:
        config = {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("source eligibility cannot trust config.json") from exc
    if not isinstance(config, dict):
        raise ValueError("source eligibility requires an object config.json")

    audit = load_verification_audit(root)
    if not audit.trustworthy:
        raise ValueError("source eligibility cannot trust verification audit")
    enabled = verification_gate_enabled(config)
    return [
        note for note in notes
        if (not is_generated_projection(note)
            and verification_allows_answer(note, audit, enabled=enabled))
    ]


def _source_strongly_verified(note: Any, audit: Any) -> bool:
    """Check one note against an already-loaded trustworthy audit."""
    if (getattr(note, "state", None) == "verified"
            and human_trust_valid(note)):
        return True
    if producer_verification_valid(note, audit):
        verification = (getattr(note, "confidence", None) or {}).get("verification", {})
        if verification.get("status") == "verified":
            return True
    if verification_status(note, audit) != "verified":
        return False
    confidence = getattr(note, "confidence", None)
    verification = confidence.get("verification") if isinstance(confidence, dict) else None
    current = verification_content_fingerprint(note)
    audit_id = audit.audit_ids.get(getattr(note, "name", ""))
    if (not isinstance(verification, dict)
            or verification.get("content_fingerprint") != current
            or not verification.get("audit_id")
            or verification.get("audit_id") != audit_id
            or audit.content_fingerprints.get(getattr(note, "name", "")) != current):
        return False
    return True


def strongly_verified_source_notes(root: str, notes: Iterable[Any]) -> list[Any]:
    """Return only sources with current human/CoVe/producer verification."""
    audit = load_verification_audit(root)
    if not audit.trustworthy:
        return []
    return [note for note in notes if _source_strongly_verified(note, audit)]


def sources_strongly_verified(root: str, notes: Iterable[Any]) -> bool:
    """Whether every factual source carries current, content-bound trust.

    Legacy/pending notes may remain discoverable for compatibility, but a
    projection must not convert that absence of a failure into a new producer
    ``verified`` verdict. Empty source sets are not factual authority.
    """
    rows = list(notes)
    if not rows:
        return False
    return len(strongly_verified_source_notes(root, rows)) == len(rows)
