"""Shared write gate for deterministic/generated Layer-2 projections.

``generated-by:`` is an ownership capability, never a truth signal.  Every
generated write therefore receives a content-bound verification result at the
same write door. Public :class:`GeneratedGrounding` values describe a claim or
refusal but cannot grant trust. Only an allowlisted validator that independently
reloads source authority can attach the sealed capability required for
``verified``; every other write is explicitly ``skipped`` and routed to review.
No generated UPDATE may inherit a verdict for older bytes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace

from remembrancerlib import (
    Note, combine_acl, current_note_names, effective_state, index_notes_by_name,
    load_notes, producer_verification_valid, verification_content_fingerprint,
    write_note,
)
from remembrancerlib.verification import (
    authorize_contradiction_producer_verification,
    producer_grounding_capability_valid,
    producer_source_fingerprint,
)


_SOURCE_METHOD_RE = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
PRODUCER_GROUNDING_PREFIX = "producer-grounded-v1:"


def producer_marker(producer: str) -> str:
    producer = str(producer or "").strip()
    if _SOURCE_METHOD_RE.fullmatch(producer) is None:
        raise ValueError("producer must match [a-z0-9][a-z0-9-]*")
    return f"generated-by:{producer}"


def producer_owned(note: Note, producer: str) -> bool:
    """Only an unapproved note with the exact marker is mutable by a producer."""
    marker = producer_marker(producer)
    producer_markers = {
        str(item) for item in (note.provenance or [])
        if str(item).startswith("generated-by:")
    }
    return (
        note.state == "proposed"
        and producer_markers == {marker}
        and not any(str(item).startswith("by:") for item in (note.provenance or []))
    )


@dataclass(frozen=True)
class GeneratedWrite:
    action: str
    note_name: str
    target_name: str = ""


@dataclass(frozen=True)
class GeneratedGrounding:
    """Attestation input for one complete generated note.

    ``sources`` must be canonical-JSON data copied from the authoritative input
    records used by the renderer.  It is hashed, never stored in frontmatter.
    ``grounded`` is only the caller's requested outcome. It becomes a machine
    success only when ``capability`` is the sealed result of the matching
    closed source replayer. A hash of caller-supplied JSON is identity, never
    authority.
    """

    source_method: str
    sources: object
    grounded: bool
    reason: str
    # Only a closed source replayer can supply a capability bearing the
    # in-process seal.  This field is intentionally opaque and never persisted.
    capability: object | None = None


def generated_grounding(
    source_method: str,
    sources: object,
    *,
    grounded: bool,
    reason: str,
) -> GeneratedGrounding:
    """Construct a review claim/refusal; this API never seals a success."""
    method = str(source_method or "").strip()
    if _SOURCE_METHOD_RE.fullmatch(method) is None:
        raise ValueError(
            "generated grounding source_method must match "
            "[a-z0-9][a-z0-9-]*"
        )
    if not isinstance(grounded, bool):
        raise ValueError("generated grounding grounded must be a bool")
    explanation = " ".join(str(reason or "").split())
    if not explanation:
        raise ValueError("generated grounding requires a reason")
    return GeneratedGrounding(method, sources, grounded, explanation)


def contradiction_producer_grounding(
    note: Note,
    *,
    context_dir: str,
) -> GeneratedGrounding:
    """Return a success only after independently replaying the closed detector.

    Source notes are deliberately not accepted as arguments.  The authority
    reloads them from ``context_dir`` under the pipeline mutation lock and
    appends the matching audit row before returning its sealed capability.
    """
    root = os.path.dirname(os.path.abspath(context_dir))
    try:
        capability, sources, reason = authorize_contradiction_producer_verification(
            note, root=root,
        )
    except PermissionError as exc:
        return generated_grounding(
            "contradiction-detector",
            {"validator": "closed-replay-refused"},
            grounded=False,
            reason=str(exc),
        )
    return GeneratedGrounding(
        source_method="contradiction-detector",
        sources=sources,
        grounded=True,
        reason=reason,
        capability=capability,
    )


def _source_fingerprint(producer: str, grounding: GeneratedGrounding) -> tuple[str, bool]:
    """Hash canonical source records; invalid material can never become verified."""
    try:
        return producer_source_fingerprint(
            producer, grounding.source_method, grounding.sources,
        ), True
    except (TypeError, ValueError):
        fallback = {
            "producer": producer,
            "source_method": grounding.source_method,
            "invalid_sources": True,
        }
        encoded = json.dumps(fallback, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest(), False


def _grounding_audit_id(
    *,
    producer: str,
    method: str,
    source_fingerprint: str,
    content_fingerprint: str,
    status: str,
) -> str:
    """Stable producer audit id shared with the closed authority journal."""
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


def _producer_confidence(
    note: Note,
    *,
    producer: str,
    grounding: GeneratedGrounding | None,
) -> dict:
    """Build the complete content-bound confidence block for ``note``."""
    if grounding is None:
        grounding = generated_grounding(
            "unattested",
            {"producer": producer},
            grounded=False,
            reason="producer supplied no source-authority grounding",
        )
    source_fingerprint, canonical_sources = _source_fingerprint(producer, grounding)
    requested_grounded = grounding.grounded and canonical_sources
    reason = grounding.reason
    if grounding.grounded and not canonical_sources:
        reason = "grounding source material is not canonical JSON"
    content_fingerprint = verification_content_fingerprint(note)
    method = PRODUCER_GROUNDING_PREFIX + grounding.source_method
    requested_status = "verified" if requested_grounded else "skipped"
    audit_id = _grounding_audit_id(
        producer=producer,
        method=method,
        source_fingerprint=source_fingerprint,
        content_fingerprint=content_fingerprint,
        status=requested_status,
    )
    grounded = bool(
        requested_grounded
        and producer_grounding_capability_valid(
            note,
            grounding.capability,
            producer=producer,
            source_method=grounding.source_method,
            source_fingerprint=source_fingerprint,
            audit_id=audit_id,
        )
    )
    status = "verified" if grounded else "skipped"
    if requested_grounded and not grounded:
        reason = (
            "caller-supplied grounding has no closed producer authority; "
            "attended review required"
        )
        audit_id = _grounding_audit_id(
            producer=producer,
            method=method,
            source_fingerprint=source_fingerprint,
            content_fingerprint=content_fingerprint,
            status=status,
        )

    # Reuse the established confidence envelope and add producer-specific proof
    # fields.  ``skipped`` is explicitly low/review-required even if legacy
    # pseudo-provenance would otherwise inflate the corroboration score.
    from distill.confidence import compute_confidence

    vres = SimpleNamespace(
        status=status,
        checked=1 if grounded else 0,
        supported=1 if grounded else 0,
        unsupported=0,
        content_fingerprint=content_fingerprint,
        audit_id=audit_id,
    )
    confidence = compute_confidence(note, vres=vres)
    verification = confidence["verification"]
    verification.update({
        "producer": producer,
        "method": method,
        "source_fingerprint": source_fingerprint,
        "content_fingerprint": content_fingerprint,
        "audit_id": audit_id,
        "grounded": grounded,
        "reason": reason,
    })
    if not grounded:
        verification["review_required"] = True
        confidence["score"] = min(float(confidence.get("score", 0.0) or 0.0), 0.2)
        confidence["level"] = "low"
    return confidence


def _valid_matching_proof(note: Note, confidence: object, *, producer: str) -> bool:
    """Whether the central read gate accepts this exact producer proof."""
    if not isinstance(confidence, dict):
        return False
    verification = confidence.get("verification")
    if not isinstance(verification, dict):
        return False
    return (
        confidence == note.confidence
        and verification.get("producer") == producer
        and producer_verification_valid(note)
    )


def producer_grounding_terminal(note: Note) -> str | None:
    """Return ``verified``/``review`` for a valid producer result, else ``None``.

    The backfill queue uses this to avoid sending structurally eventless
    projections through generic CoVe three times.  Invalid or stale blocks are
    not terminal and remain fail-closed in retrieval.
    """
    producer = generated_producer(note)
    if producer is None:
        return None
    if not _valid_matching_proof(note, note.confidence, producer=producer):
        return None
    verification = note.confidence["verification"]
    return "verified" if verification["status"] == "verified" else "review"


def generated_producer(note: Note) -> str | None:
    """Return the one explicit generated owner, never inferring from shape/name."""
    markers = [
        str(value)[len("generated-by:"):]
        for value in (note.provenance or [])
        if str(value).startswith("generated-by:")
    ]
    if len(markers) != 1 or not markers[0]:
        return None
    return markers[0]


def bind_generated_verification(
    note: Note,
    *,
    producer: str,
    grounding: GeneratedGrounding | None,
    existing: Note | None = None,
) -> dict:
    """Mutate ``note`` with a fresh proof/refusal for its exact current content.

    Used by the attended/backfill path for legacy generated pages.  Ownership
    must already be explicit and exact; this function never mints a producer
    marker or infers one from a filename.
    """
    if generated_producer(note) != producer:
        raise ValueError("generated verification owner does not match exact producer marker")
    _apply_grounding(
        note,
        producer=producer,
        grounding=grounding,
        existing=existing,
    )
    return note.confidence


def _apply_grounding(
    note: Note,
    *,
    producer: str,
    grounding: GeneratedGrounding | None,
    existing: Note | None = None,
) -> None:
    candidate = _producer_confidence(note, producer=producer, grounding=grounding)
    # Preserve the exact prior envelope (especially computed_at) on an
    # idempotent replay, but only after independently proving it binds these
    # bytes and equals the newly derived stable audit id.
    if (
        existing is not None
        and _valid_matching_proof(existing, existing.confidence, producer=producer)
        and existing.confidence["verification"].get("audit_id")
        == candidate["verification"].get("audit_id")
    ):
        note.confidence = copy.deepcopy(existing.confidence)
    else:
        note.confidence = candidate
    verification = note.confidence.get("verification", {})
    if (
        isinstance(grounding, GeneratedGrounding)
        and verification.get("status") == "verified"
        and producer_grounding_capability_valid(
            note,
            grounding.capability,
            producer=producer,
            source_method=grounding.source_method,
            source_fingerprint=verification.get("source_fingerprint", ""),
            audit_id=verification.get("audit_id", ""),
        )
    ):
        note._producer_verification_capability = grounding.capability


def resolve_generated_target(
    name: str, *, producer: str, notes_by_name: dict[str, Note],
) -> Note | None:
    """Resolve a fixed logical name to its unique current producer-owned head."""
    existing = notes_by_name.get(name)
    if existing is None:
        return None
    current = current_note_names(notes_by_name.values())
    if existing.name in current:
        return existing
    marker = producer_marker(producer)

    def descends_from(note: Note, target: str, seen: set[str]) -> bool:
        if note.name in seen:
            return False
        seen.add(note.name)
        for relation, parent_name in note.typed_links:
            if relation != "supersedes":
                continue
            if parent_name == target:
                return True
            parent = notes_by_name.get(parent_name)
            if parent is not None and descends_from(parent, target, seen):
                return True
        return False

    successors = [
        note for note in notes_by_name.values()
        if note.name in current
        and marker in (note.provenance or [])
        and descends_from(note, existing.name, set())
    ]
    # A newly proposed revision does not activate its supersedes edge. During
    # review both the approved generated head and its pending machine child are
    # therefore in ``current_note_names``. Resolve the stable logical target to
    # the one approved head; ``_propose_replacement`` will then find and refresh
    # that pending child in place. Without this preference the third generator
    # pass sees two successors and can never converge.
    approved = [note for note in successors if effective_state(note) == "verified"]
    if len(approved) == 1:
        return approved[0]
    if len(approved) > 1 or len(successors) != 1:
        raise ValueError(
            f"{producer} refuses historical fixed-name target {existing.name!r}; "
            f"owned current successors={len(successors)}, approved={len(approved)}"
        )
    return successors[0]


def persist_generated_note(
    desired: Note,
    *,
    producer: str,
    context_dir: str,
    notes_by_name: dict[str, Note] | None = None,
    detail: str = "generated projection refresh requires review",
    grounding: GeneratedGrounding | None = None,
) -> GeneratedWrite:
    """Create/refresh only producer-owned drafts; propose every collision.

    Fixed-name generators must never infer ownership from a name, type, folder,
    or prose shape.  The explicit ``generated-by:`` marker is the capability.
    Approved, human-authored, legacy-unmarked, and other-producer notes remain
    byte-for-byte untouched and receive a replacement proposal instead.
    """
    if notes_by_name is None:
        notes_by_name = index_notes_by_name(
            load_notes(context_dir), operation=f"{producer} generated write",
        )
    desired = copy.deepcopy(desired)
    desired.path = ""
    desired.state = "proposed"
    desired.superseded = False
    # Confidence is finalized only after target/proposal naming and link
    # materialization.  Until then retain no caller-supplied verdict: accepting
    # one would let a producer reuse proof for different content.
    desired.confidence = None
    marker = producer_marker(producer)
    desired.provenance = list(dict.fromkeys([
        *(
            item for item in (desired.provenance or [])
            if not str(item).startswith("by:")
            and not str(item).startswith("generated-by:")
        ),
        marker,
    ]))

    requested_name = desired.name
    existing = resolve_generated_target(
        requested_name, producer=producer, notes_by_name=notes_by_name,
    )
    if existing is None:
        _apply_grounding(desired, producer=producer, grounding=grounding)
        import govern
        govern.persist_machine_note(
            desired,
            context_dir,
            requester=f"generated:{producer}",
            reason=f"generated projection: {producer}",
        )
        notes_by_name[desired.name] = desired
        return GeneratedWrite("CREATE", desired.name)

    if existing.name != requested_name:
        desired.name = existing.name

    if producer_owned(existing, producer):
        if (desired.domain, desired.project) != (existing.domain, existing.project):
            raise ValueError(
                f"{producer} refuses implicit generated-note move for {existing.name!r}: "
                f"{existing.domain}/{existing.project} -> {desired.domain}/{desired.project}"
            )
        desired.name = existing.name
        desired.title = desired.title or existing.title or existing.name
        _apply_grounding(
            desired, producer=producer, grounding=grounding, existing=existing,
        )
        if (desired.to_markdown() == existing.to_markdown()
                or (
                    desired.description == existing.description
                    and desired.type == existing.type
                    and desired.domain == existing.domain
                    and desired.project == existing.project
                    and desired.acl == existing.acl
                    and desired.entities == existing.entities
                    and desired.components == existing.components
                    and desired.provenance == existing.provenance
                    and desired.resource == existing.resource
                    and desired.tags == existing.tags
                    and desired.structured == existing.structured
                    and desired.confidence == existing.confidence
                    and desired.body.strip() == existing.body.strip()
                )):
            return GeneratedWrite("IGNORE", existing.name)
        import govern
        govern.persist_machine_note(
            desired,
            context_dir,
            requester=f"generated:{producer}",
            reason=f"generated projection refresh: {producer}",
        )
        notes_by_name[desired.name] = desired
        return GeneratedWrite("UPDATE", desired.name)

    from distill.engine import _propose_replacement

    # Even a supersedes edge reveals the target's name/existence.  Narrow the
    # proposal to an audience safe for both artifacts; incompatible OR scopes
    # cannot be represented and combine_acl fails closed.
    desired.acl = combine_acl(list(existing.acl or []), list(desired.acl or []))
    decision = _propose_replacement(
        existing, desired, context_dir, notes_by_name,
        f"generated:{producer}:{desired.name}", detail=detail,
        finalize_note=lambda proposal, prior=None: _apply_grounding(
            proposal,
            producer=producer,
            grounding=grounding,
            existing=prior,
        ),
    )
    return GeneratedWrite(decision.action, decision.note_name, existing.name)
