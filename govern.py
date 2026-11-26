"""Governance — the human approval inbox for high-stakes agent writes.

Caller-side governance (NOT in the frozen remembrancerlib core). After a write lands, `route()` classifies
it; high-stakes writes (restricted labels, externally-authored, decisions, customer deal-status) are
enqueued to an append-only ledger `state/approval_queue.jsonl` for a human to approve/reject. The
inbox actions call promote/reject. Routine writes auto-publish (+ a random audit sample).

Event-sourced: the queue is a FOLD over the append-only log (mirrors remembrancer's paradigm). Pending =
items whose latest event is `enqueued`/`escalated`.
"""
from __future__ import annotations

import copy
import hashlib
import fcntl
import json
import logging
import os
import random
import re
import stat
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

_log = logging.getLogger("remembrancer.govern")
#: What the reconcile log tells the operator to do about a decided note that is still proposed.
_REAPPLY_HINT = "re-apply the decision or edit the note"

# decision per class
APPROVAL, AUTO, SAMPLE = "approval-required", "auto", "auto+sample"
# The org's own mail domains.  The authority is ``config.json
# governance.internal_domains``.  A root that sets none has none: every email
# author is outside, so every email-sourced write is held for review.
DEFAULT_INTERNAL_DOMAINS: tuple[str, ...] = ()
_SAMPLE_RATE = 0.05
_SLA_HOURS = 48
_QUEUE_LOCK = threading.RLock()
_MAX_QUEUE_BYTES = 64 * 1024 * 1024
_MAX_QUEUE_LINE_BYTES = 1024 * 1024
_QUEUE_EVENTS = frozenset({
    "enqueued", "escalated", "approved", "rejected", "edited", "withdrawn",
    # Distinguishable machine resolution recorded by the config-gated
    # auto-approval drainer (remembrancerlib.verification.authorize_auto_approval).
    "approved-auto",
})
#: Latest events under which an item is still OPEN: it asks a human to decide
#: THESE bytes.  The one definition of "pending" -- ``pending()`` (the inbox and
#: auto-approval), ``reconcile_completed()``, ``enqueue_for_review()``'s reopen
#: check, the bound-pending guard (``bound_pending_names`` /
#: ``note_bound_pending``, through ``_open_binding_items``),
#: a decision-review queue and an approvals UI (its review page and decide
#: POST) all read it, so no reader can drift.
#: Adding an event here OFFERS its items for a decision again.  So an event for
#: a decision that is made but not yet applied (say, a deferred click-time
#: signed approval) does not belong here: it would put an already-decided item
#: back in the inbox, in auto-approval and on the decide POST.  Such an item
#: still binds its bytes until it is applied, so it belongs in a guard-only
#: superset read by ``_open_binding_items`` alone (for example
#: ``_BINDING_OPEN | _DECIDED_UNAPPLIED``), never in this set.
_BINDING_OPEN = frozenset({"enqueued", "escalated"})


class ApprovalQueueError(RuntimeError):
    """The approval ledger is unsafe, corrupt, or internally inconsistent."""


class ApprovalItemResolvedError(RuntimeError):
    """The exact bytes being enqueued already have a resolved approval item.

    Queue ids are deterministic in the note's content fingerprint and on-disk
    preimage, so a note whose current bytes were already approved / rejected /
    edited can never receive a second item.  A proposed note in that position
    (the decision was recorded on the queue but the note never transitioned)
    is a human follow-up, not a crash the reconciler can repair -- it must be
    reported once, not raised every cycle.
    """


@dataclass
class RouteMeta:
    requester: str = ""            # provenance by:<identity> (who wrote it)
    source: str | None = None      # caller-declared origin, e.g. email:x@outside.example / slack:@jdoe
    confidence: float | None = None  # note.confidence, else None
    supersedes: str | None = None    # for edit-in-place / correction writes


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _queue_path(root: str) -> str:
    return os.path.join(root, "state", "approval_queue.jsonl")


def _persisted_approval_binding(current) -> dict[str, str]:
    """Bind one independently loaded proposed note to its exact file bytes."""
    from remembrancerlib.verification import verification_content_fingerprint

    if getattr(current, "state", None) != "proposed":
        raise RuntimeError("only an exact proposed note may enter approval")
    expected_fingerprint = verification_content_fingerprint(current)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(current.path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("approval target is not a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return {
        "content_fingerprint": expected_fingerprint,
        "expected_existing_sha256": digest.hexdigest(),
    }


def _approval_binding(note, root: str, *, corpus=None) -> tuple[object, dict[str, str]]:
    """Bind a queue decision to the exact, independently loaded disk preimage.

    ``corpus`` exists only for READ-ONLY listings, which need the same check
    over many items at once. Loading the whole corpus per item is correct for a
    single decision and quadratic for a list: rendering a large approval inbox
    called this once per item, pegged the UI's CPU for minutes, and blocked
    every other request behind it.

    Every MUTATION path -- approve, reject, the governance rewrite capability --
    deliberately passes nothing, so each still re-reads the note from disk at the
    moment it acts. That independence is the whole point of this function and a
    caller that "optimises" it away is removing the check, not speeding it up.
    """
    from remembrancerlib import load_notes
    from remembrancerlib.verification import verification_content_fingerprint

    root = os.path.abspath(root)
    candidates = (load_notes(os.path.join(root, "context"))
                  if corpus is None else corpus)
    matches = [
        candidate for candidate in candidates
        if candidate.name == getattr(note, "name", None)
    ]
    if len(matches) != 1:
        raise RuntimeError("approval target is missing, suppressed, or ambiguous")
    current = matches[0]
    binding = _persisted_approval_binding(current)
    if verification_content_fingerprint(note) != binding["content_fingerprint"]:
        raise RuntimeError("approval caller does not match the persisted note")
    return current, binding


def approval_item_matches(item: dict, note, root: str,
                         *, corpus=None) -> tuple[bool, dict[str, str]]:
    """Whether a pending item still names the exact bytes it originally queued.

    ``corpus`` is a read-only optimisation for listings; see _approval_binding.
    """
    if not isinstance(item, dict):
        return False, {}
    try:
        _current, binding = _approval_binding(note, root, corpus=corpus)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False, {}
    return _binding_matches(item, binding), binding


def _binding_matches(item: dict, binding: dict[str, str]) -> bool:
    """The exact-preimage test: every binding field is on the item, as a string.

    Shared by the trust path above and the enrichment guard
    (:func:`bound_pending_names`), so the two cannot disagree about which
    bytes an item binds.  An empty binding (the note could not be bound) never
    matches.
    """
    return bool(binding) and all(
        isinstance(item.get(key), str) and item.get(key) == value
        for key, value in binding.items()
    )


def _item_id(
    name: str,
    binding: dict[str, str],
    *,
    diff: dict | None = None,
) -> str:
    material = {
        "name": name,
        "content_fingerprint": binding["content_fingerprint"],
        "expected_existing_sha256": binding["expected_existing_sha256"],
    }
    # The historical single-target queue id is intentionally unchanged.  A
    # multi-target aggregate did not exist under that contract, so its new id
    # additionally binds the exact lifecycle diff and cannot be replayed with
    # a different target set.
    if isinstance((diff or {}).get("supersedes"), list):
        material["diff"] = diff
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return "aq_" + hashlib.sha256(encoded).hexdigest()[:32]


# ─── Outside-authored sources ───

#: Event sources whose ``author`` is the message's From header.  Only these can
#: make a distilled note ``external``: a git author's email or a calendar
#: organizer is not content an outside party sent us.
EMAIL_EVENT_SOURCES = frozenset({"gmail"})
# One address-shaped token: local part and domain, split at header separators.
# The second alternative consumes a whole local-part run that is not followed
# by an address in one step.  Without it the scan restarted inside that run at
# every offset, quadratic on a long separator-free header (n=32,000: 3.8 s);
# with it the matches are exactly the first alternative's, in linear time.  Its
# matches carry an empty domain and are dropped by :func:`_author_addresses`.
_ADDRESS_RE = re.compile(
    r'([^\s<>()\[\],;:"@]*)@([^\s<>(),;:"@]+)|[^\s<>()\[\],;:"@]+')


def domain_list(raw) -> tuple[str, ...]:
    """A configured mail-domain list: a string or a list of strings, each
    lower-cased and stripped of a leading ``@`` or ``.``, duplicates dropped.
    Any other value names no domain."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return ()
    domains = (d.strip().lower().lstrip("@.") for d in raw if isinstance(d, str))
    return tuple(dict.fromkeys(d for d in domains if d))


def internal_domains(cfg) -> tuple[str, ...]:
    """The org domains from ``config.json governance.internal_domains``.

    Absent: :data:`DEFAULT_INTERNAL_DOMAINS`, which is empty.  Present but not
    a string or a list: ``()``.  Both fail closed (every email author is
    outside, so every email-sourced write is held for review rather than
    published).
    """
    governance = cfg.get("governance") if isinstance(cfg, dict) else None
    if not isinstance(governance, dict) or "internal_domains" not in governance:
        return DEFAULT_INTERNAL_DOMAINS
    return domain_list(governance["internal_domains"])


def _author_addresses(author) -> list[str]:
    """Every address-shaped token in a From header, lower-cased, in order.

    Deliberately not an RFC 5322 parse of the one real mailbox: a display name
    that carries an address counts too, and so does every token of a header a
    parser would reject.  The caller treats ANY outside token as outside, so
    this can over-hold a note for review but never hide an outside author
    behind an internal-looking display name.
    """
    if not isinstance(author, str) or "@" not in author:
        return []
    addresses = []
    for local, domain in _ADDRESS_RE.findall(author):
        domain = domain.lower().rstrip(".")
        if domain:
            addresses.append(f"{local.lower()}@{domain}")
    return addresses


def external_email_source(authors, cfg) -> str | None:
    """``email:<address>`` for the first outside author in ``authors``, else None.

    Outside means :func:`_external`: an address whose domain is neither an org
    domain (:func:`internal_domains`) nor a subdomain of one -- the same test
    :func:`classify` applies to the returned source.
    """
    internal = internal_domains(cfg)
    for author in authors or ():
        for address in _author_addresses(author):
            source = f"email:{address}"
            if _external(source, internal):
                return source
    return None


def external_event_source(events, root: str) -> str | None:
    """The outside author a note distilled from ``events`` is attributed to.

    Only :data:`EMAIL_EVENT_SOURCES` events count; config is read only when one
    is present.  Distill passes the result as ``source=`` to
    :func:`persist_machine_note`, which holds the note as ``external``.
    """
    authors = [getattr(ev, "author", None) for ev in events or ()
               if getattr(ev, "source", None) in EMAIL_EVENT_SOURCES]
    if not authors:
        return None
    return external_email_source(authors, _config(root))


def _external(source: str | None, internal) -> bool:
    """True if the source is an external email/slack author (domain ∉ internal). ASI06 surface."""
    if not source:
        return False
    s = source.lower()
    if s.startswith("email:") and "@" in s:
        dom = s.split("@")[-1].strip()
        return not any(dom == d or dom.endswith("." + d) for d in internal)
    # a slack: source is internal by default (our workspace); other prefixes: unknown → not external
    return False


def classify(note, meta: RouteMeta, cfg: dict) -> tuple[str, str]:
    """First-match, top-down (highest stakes wins). Returns (class, decision)."""
    from remembrancerlib.acl import restricted_labels
    from retrieve.maintenance import automatic
    if automatic(cfg):
        return "automatic", AUTO
    g = (cfg or {}).get("governance", {}) or {}
    internal = internal_domains(cfg)
    acl = set(note.acl if not isinstance(note.acl, str) else [note.acl])
    if acl & restricted_labels(cfg):
        # The class keeps the name our approval queue already stores.
        return "secrets-founders", APPROVAL
    if g.get("external_is_high_stakes", True) and _external(meta.source, internal):
        return "external", APPROVAL
    if note.type == "decision":
        return "decision", APPROVAL
    kind = (note.structured or {}).get("kind") if isinstance(note.structured, dict) else None
    if note.domain in ("customers", "customer") and (note.type == "state" or kind in ("deal", "deal_status", "status")):
        return "customer-deal", APPROVAL
    if acl:                                    # other ACL-restricted → auto but always sampled
        return "restricted", SAMPLE
    return "routine", (SAMPLE if random.random() < _SAMPLE_RATE else AUTO)


def _config(root: str) -> dict:
    try:
        from retrieve.query import _load_config
        cfg = _load_config(os.path.abspath(root))
    except Exception:
        cfg = {}
    return cfg if isinstance(cfg, dict) else {}


def _hold_note(note, cls: str) -> bool:
    """Install an unconditional answer hold while preserving verifier detail."""
    from remembrancerlib.verification import apply_governance_hold

    return apply_governance_hold(note, cls)


def prepare(note, meta: RouteMeta, root: str) -> dict:
    """Apply a fail-closed hold before the caller's first note write.

    Callers that construct a note should invoke this before persistence. Route
    repeats the check against independently loaded bytes as defense in depth.
    """
    cfg = _config(root)
    cls, decision = classify(note, meta, cfg)
    if decision == APPROVAL:
        _hold_note(note, cls)
    return {"class": cls, "decision": decision}


def prepare_machine(note, root: str, *, requester: str = "machine",
                    source: str | None = None) -> dict:
    """Install a pre-write hold for an approval-classified machine draft."""
    from remembrancerlib.verification import governance_hold_class

    if getattr(note, "state", None) != "proposed":
        return {"class": "not-proposed", "decision": AUTO}
    durable_class = governance_hold_class(note)
    if durable_class is not None:
        _hold_note(note, durable_class)
        return {"class": durable_class, "decision": APPROVAL}
    return prepare(
        note,
        RouteMeta(requester=requester, source=source),
        root,
    )


def persist_machine_note(
    note,
    context_dir: str,
    *,
    requester: str = "machine",
    source: str | None = None,
    reason: str = "machine-derived memory",
    **write_kwargs,
) -> str:
    """Persist a producer note and close every required approval transaction.

    High-stakes output is held before its first byte, then exact-enqueued. A
    later producer refresh may rewrite a held proposal only from its still
    pending exact preimage; the replacement is enqueued before the stale item
    is retired. Queue failure therefore leaves hidden, retryable bytes rather
    than readable or falsely approved memory.
    """
    from remembrancerlib import load_notes
    from remembrancerlib.note import write_note as core_write_note
    from remembrancerlib.verification import (
        apply_governance_hold,
        governance_hold_class,
    )

    root = os.path.dirname(os.path.abspath(context_dir))
    if getattr(note, "state", None) is None:
        note.state = "proposed"
    matches = [
        candidate for candidate in load_notes(context_dir)
        if candidate.name == getattr(note, "name", None)
    ]
    current = matches[0] if len(matches) == 1 else None
    current_class = governance_hold_class(current) if current is not None else None
    if current_class is not None and getattr(note, "state", None) == "proposed":
        apply_governance_hold(note, current_class)
        routing = {"class": current_class, "decision": APPROVAL}
    else:
        routing = prepare_machine(
            note, root, requester=requester, source=source,
        )

    review_capability = write_kwargs.pop("_review_capability", None)
    if (
        routing["decision"] == APPROVAL
        and current_class is not None
        and getattr(note, "state", None) == "proposed"
    ):
        review_capability = authorize_review_rewrite(
            note, root, routing["class"],
        )
    path = core_write_note(
        note,
        context_dir,
        _review_capability=review_capability,
        **write_kwargs,
    )
    if routing["decision"] == APPROVAL:
        targets = sorted({
            target for relation, target in getattr(note, "typed_links", [])
            if relation == "supersedes" and target
        })
        enqueue_for_review(
            note,
            requester=requester,
            reason=reason,
            root=root,
            cls=routing["class"],
            supersedes=targets or None,
            source=source,
        )
    return path


def prepare_for_review(note, cls: str) -> dict:
    """Install a mandatory review hold before the note's first persistence.

    Some producer workflows require attended review regardless of the ordinary
    governance classifier (feedback drafts and edits to verified pages are the
    two current callers).  They must apply the same hold that
    :func:`enqueue_for_review` will independently reassert after loading the
    persisted bytes, otherwise a newly written proposal has a brief readable
    window before its queue record exists.
    """
    review_class = str(cls or "manual-review").strip() or "manual-review"
    _hold_note(note, review_class)
    return {"class": review_class, "decision": APPROVAL}


def authorize_review_rewrite(note, root: str, cls: str) -> object | None:
    """Authorize one exact rewrite of an already queued hard-held draft.

    Ordinary producers must byte-preserve held proposals.  An attended editor
    may change one only when the current bytes still have an exact pending
    queue item; the returned opaque capability binds both those bytes and the
    fully rendered replacement.  The caller must then enqueue the replacement,
    which resolves the stale item after the new binding is durable.
    """
    from remembrancerlib import load_notes
    from remembrancerlib.verification import (
        _issue_governance_rewrite_capability,
        apply_governance_hold,
        governance_hold_class,
    )

    root = os.path.abspath(root)
    review_class = str(cls or "").strip()
    matches = [
        candidate for candidate in load_notes(os.path.join(root, "context"))
        if candidate.name == getattr(note, "name", None)
    ]
    if len(matches) != 1 or matches[0].state != "proposed":
        raise RuntimeError("review rewrite target is missing, ambiguous, or not proposed")
    current = matches[0]
    if governance_hold_class(current) != review_class:
        raise RuntimeError("review rewrite class does not match the durable hold")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(current.path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("review rewrite target is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        existing_text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("review rewrite target is not UTF-8") from exc
    apply_governance_hold(note, review_class)
    if note.to_markdown() == existing_text:
        # Crash recovery: the replacement bytes may already be durable while
        # their queue append failed. No rewrite authority is needed to replay
        # identical bytes; enqueue_for_review will finish and retire the stale
        # binding below.
        return None
    exact = [
        item for item in pending(root)
        if item.get("note") == current.name
        and item.get("class") == review_class
        and approval_item_matches(item, current, root)[0]
    ]
    if len(exact) != 1:
        raise RuntimeError("review rewrite requires one exact pending approval item")
    return _issue_governance_rewrite_capability(
        note,
        existing_text=existing_text,
        review_class=review_class,
    )


def _persist_required_hold(note, cls: str, root: str):
    """Return the exact persisted proposed note after installing its hold."""
    from remembrancerlib import load_notes, write_note
    from remembrancerlib.verification import governance_hold_class

    root = os.path.abspath(root)
    matches = [
        candidate for candidate in load_notes(os.path.join(root, "context"))
        if candidate.name == getattr(note, "name", None)
    ]
    if len(matches) != 1 or matches[0].state != "proposed":
        raise RuntimeError("approval target is missing, ambiguous, or not proposed")
    current = matches[0]
    current_class = governance_hold_class(current)
    if current_class is not None and current_class != cls:
        raise RuntimeError("approval class conflicts with the durable review hold")
    if _hold_note(current, cls):
        write_note(current, os.path.join(root, "context"))
    return current


def _carried_external_source(root: str, note_name: str) -> str | None:
    """The outside author the newest earlier queue item for a note recorded.

    A held note keeps its class through later rewrites (the hold is monotonic),
    but those rewrites carry no source: distill UPDATEs pass none by design,
    and neither does the crash reconciler.  Reading the author back
    from the queue keeps the approvals inbox naming it for as long as the hold
    lasts.  The scan runs newest first over the note's items, pending,
    replaced (``edited``) or ``withdrawn``, and stops at an approved or
    rejected one: that decision ended the proposal it tracked, and a later
    note reusing the name must not inherit its author.  Only an item flagged
    ``external`` whose source is an ``email:`` address is carried: an internal
    author or a producer's reason string never is.  It labels an item; it
    never changes a class or a hold.
    """
    for item in reversed(list(_fold(root).values())):
        if item.get("note") != note_name:
            continue
        if item.get("latest") in _PROPOSAL_DECIDED:
            return None
        source = item.get("source")
        if (
            item.get("external") is True
            and isinstance(source, str)
            and source.startswith("email:")
        ):
            return source
    return None


def _resolve_replaced_pending(root: str, note_name: str, replacement_id: str) -> None:
    """Close stale queue bindings only after their replacement is durable."""
    for item in pending(root):
        item_id = item.get("id")
        if item.get("note") != note_name or item_id == replacement_id:
            continue
        record(
            root,
            item_id,
            "edited",
            "governance-requeue",
            replacement_id=replacement_id,
        )


def route(note, meta: RouteMeta, root: str, _rng=None) -> dict:
    """Classify a just-written note and durably enqueue governed writes.

    Queue failure is deliberately fail-closed: callers must not report that a
    high-stakes write is governed when no durable approval record exists.
    Returns ``{class, decision, id?}``.
    """
    cfg = _config(root)
    cls, decision = classify(note, meta, cfg)
    if decision == AUTO:
        return {"class": cls, "decision": decision}
    if decision == APPROVAL:
        note = _persist_required_hold(note, cls, root)
    note, binding = _approval_binding(note, root)
    item_id = _item_id(note.name, binding)
    internal = internal_domains(cfg)
    rec = {
        "id": item_id, "event": "enqueued", "ts": _now(), "note": note.name,
        "class": cls, "decision": decision, "requester": meta.requester or "",
        "source": meta.source, "external": _external(meta.source, internal),
        "confidence": meta.confidence,
        **binding,
        "acl": list(note.acl) if not isinstance(note.acl, str) else [note.acl],
        "diff": {"kind": "supersede" if meta.supersedes else "create",
                 "supersedes": meta.supersedes, "new": (note.body or "")[:2000]},
        "sla_hours": _SLA_HOURS,
        "due": (datetime.now(timezone.utc) + timedelta(hours=_SLA_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _append(root, rec)
    return {"class": cls, "decision": decision, "id": item_id}


def enqueue_for_review(note, requester: str, reason: str, root: str,
                       cls: str = "feedback-draft",
                       supersedes: str | list[str] | None = None,
                       source: str | None = None) -> str:
    """Force a just-written note into the approval queue regardless of `classify` (routine writes
    normally auto-publish). Used by feedback and aggregate synthesis producers:
    every such draft must be reviewed by a human before it becomes canonical.
    Single-target diffs retain their historical string representation; aggregate
    diffs carry a sorted list. Returns the queue item id (also useful to link back).

    ``source`` is the caller-declared origin (e.g. ``email:x@outside.example``
    from distill); when given, the item records it instead of
    ``reason``, so the approvals inbox names the outside author.  A re-enqueue
    without one (a distill UPDATE of a held note, or the crash
    reconciler) carries the outside author an earlier item for the note
    recorded (:func:`_carried_external_source`).  An ``external`` item is
    always flagged ``external``, since that class is reached only from an
    outside author.  The source is not part of the item id."""
    linked_targets = sorted({
        target for relation, target in getattr(note, "typed_links", [])
        if relation == "supersedes" and target
    })
    if supersedes is None:
        supplied_targets = linked_targets
    elif isinstance(supersedes, str):
        supplied_targets = [supersedes] if supersedes else []
    elif isinstance(supersedes, list):
        if any(not isinstance(target, str) or not target for target in supersedes):
            raise ValueError("approval supersedes metadata contains an invalid target")
        supplied_targets = sorted(set(supersedes))
    else:
        raise ValueError("approval supersedes metadata must be a string or list")
    if linked_targets != supplied_targets:
        raise ValueError("approval supersedes metadata must match the proposal edge")

    supersedes_diff: str | list[str] | None
    if len(linked_targets) == 1:
        # Preserve the existing queue schema for ordinary one-note proposals.
        supersedes_diff = linked_targets[0]
    elif linked_targets:
        supersedes_diff = linked_targets
    else:
        supersedes_diff = None

    note = _persist_required_hold(note, cls, root)
    note, binding = _approval_binding(note, root)
    diff = {
        "kind": "supersede" if supersedes_diff else "create",
        "supersedes": supersedes_diff,
        "new": (note.body or "")[:2000],
    }
    item_id = _item_id(note.name, binding, diff=diff)
    internal = internal_domains(_config(root))
    if source is None:
        source = _carried_external_source(root, note.name)
    rec = {
        "id": item_id, "event": "enqueued", "ts": _now(), "note": note.name,
        "class": cls, "decision": APPROVAL, "requester": requester or "",
        "source": source or reason,
        "external": cls == "external"
        or (bool(source) and _external(source, internal)),
        "confidence": None,
        **binding,
        "acl": list(note.acl) if not isinstance(note.acl, str) else [note.acl],
        "diff": diff,
        "sla_hours": _SLA_HOURS,
        "due": (datetime.now(timezone.utc) + timedelta(hours=_SLA_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    existing = get_item(root, item_id)
    if existing is not None:
        same = (existing.get("note") == rec["note"]
                and existing.get("class") == rec["class"]
                and existing.get("acl") == rec["acl"]
                and existing.get("diff") == rec["diff"])
        if not same:
            raise RuntimeError("approval id collision has different content")
        if existing.get("latest") in _BINDING_OPEN:
            _resolve_replaced_pending(root, note.name, item_id)
            return item_id
        if existing.get("latest") != "withdrawn":
            raise ApprovalItemResolvedError(
                "refusing to reopen a resolved approval item"
            )
    _append(root, rec)
    _resolve_replaced_pending(root, note.name, item_id)
    return item_id


def _append(root: str, rec: dict) -> None:
    """Append one complete, durable queue event or raise without a partial row."""
    payload = (json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(payload) > _MAX_QUEUE_LINE_BYTES:
        raise ValueError("approval queue event exceeds safety bound")
    from remembrancerlib.locks import open_repository_state_directory

    with _QUEUE_LOCK:
        state_fd, _created = open_repository_state_directory(root, create=True)
        try:
            fd = os.open(
                "approval_queue.jsonl",
                os.O_RDWR | os.O_CREAT | os.O_APPEND
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=state_fd,
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                metadata = os.fstat(fd)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ApprovalQueueError("approval queue is not a regular file")
                os.fchmod(fd, 0o600)
                original_size = metadata.st_size
                if original_size + len(payload) > _MAX_QUEUE_BYTES:
                    raise ApprovalQueueError("approval queue exceeds safety bound")
                progress = [0]
                try:
                    while progress[0] < len(payload):
                        written = os.write(fd, payload[progress[0]:])
                        if written <= 0:
                            raise OSError("approval queue append made no progress")
                        progress[0] += written
                    os.fsync(fd)
                except BaseException:
                    # Never truncate a non-cooperating writer's tail. Roll back
                    # only the exact bytes proven to be this append.
                    if os.fstat(fd).st_size == original_size + progress[0]:
                        os.lseek(fd, original_size, os.SEEK_SET)
                        chunks: list[bytes] = []
                        remaining = progress[0]
                        while remaining:
                            chunk = os.read(fd, remaining)
                            if not chunk:
                                break
                            chunks.append(chunk)
                            remaining -= len(chunk)
                        if b"".join(chunks) == payload[:progress[0]]:
                            os.ftruncate(fd, original_size)
                            os.fsync(fd)
                    raise
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
            os.fsync(state_fd)
        finally:
            os.close(state_fd)


def record(root: str, item_id: str, event: str, actor: str, **extra) -> None:
    """Append a resolution event (approved/approved-auto/rejected/edited/escalated). Never rewrites."""
    if (
        not isinstance(item_id, str)
        or not item_id
        or len(item_id) > 128
        or event not in (_QUEUE_EVENTS - {"enqueued"})
        or not isinstance(actor, str)
        or not actor.strip()
    ):
        raise ValueError("invalid approval queue resolution event")
    _append(root, {"id": item_id, "event": event, "ts": _now(), "actor": actor, **extra})


_RESOLVED = {"approved", "rejected", "edited"}
# A human or the auto-approval drainer decided the proposal an item tracked.
_PROPOSAL_DECIDED = frozenset({"approved", "approved-auto", "rejected"})


def _fold(root: str) -> dict[str, dict]:
    """Fold the ledger by id: keep the enqueue record + its latest event."""
    from remembrancerlib.locks import open_repository_state_directory
    from remembrancerlib.verification_audit import read_descriptor_all

    items: dict[str, dict] = {}
    try:
        state_fd, _created = open_repository_state_directory(root, create=False)
    except OSError as exc:
        raise ApprovalQueueError("approval queue state directory is unsafe") from exc
    if state_fd < 0:
        return {}
    try:
        try:
            queue_fd = os.open(
                "approval_queue.jsonl",
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=state_fd,
            )
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise ApprovalQueueError("approval queue is unsafe") from exc
        try:
            with _QUEUE_LOCK:
                fcntl.flock(queue_fd, fcntl.LOCK_SH)
                metadata = os.fstat(queue_fd)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                ):
                    raise ApprovalQueueError(
                        "approval queue must be a private regular file"
                    )
                raw = read_descriptor_all(queue_fd, _MAX_QUEUE_BYTES)
                if raw and not raw.endswith(b"\n"):
                    raise ApprovalQueueError("approval queue has a partial tail")
        finally:
            try:
                fcntl.flock(queue_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(queue_fd)
    finally:
        os.close(state_fd)
    for encoded in raw.splitlines():
        if not encoded or len(encoded) > _MAX_QUEUE_LINE_BYTES:
            raise ApprovalQueueError("approval queue contains an invalid line")
        try:
            r = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApprovalQueueError("approval queue contains malformed JSON") from exc
        if not isinstance(r, dict):
            raise ApprovalQueueError("approval queue row is not an object")
        i = r.get("id")
        event = r.get("event")
        if (
            not isinstance(i, str)
            or not i
            or len(i) > 128
            or not isinstance(event, str)
            or event not in _QUEUE_EVENTS
        ):
            raise ApprovalQueueError("approval queue row has invalid identity/event")
        if event == "enqueued":
            if not isinstance(r.get("note"), str) or not r["note"]:
                raise ApprovalQueueError("approval enqueue row has no note")
            binding_values = (
                r.get("content_fingerprint"), r.get("expected_existing_sha256"),
            )
            if any(value is not None for value in binding_values) and not all(
                isinstance(value, str)
                and len(value) == 64
                and all(char in "0123456789abcdef" for char in value)
                for value in binding_values
            ):
                raise ApprovalQueueError("approval enqueue binding is malformed")
            items.setdefault(i, {}).update({k: v for k, v in r.items() if k != "event"})
            items[i]["latest"] = "enqueued"
        else:
            if not isinstance(r.get("actor"), str) or not r["actor"]:
                raise ApprovalQueueError("approval resolution row has no actor")
            items.setdefault(i, {})["latest"] = r.get("event", "")
            items[i]["last_actor"] = r.get("actor", "")
    return items


def _completed_resolution(root: str, item: dict) -> dict | None:
    """Prove an audit-first trust transition already completed this item."""
    expected_sha = item.get("expected_existing_sha256")
    note_name = item.get("note")
    if not (
        isinstance(expected_sha, str)
        and len(expected_sha) == 64
        and isinstance(note_name, str)
        and note_name
    ):
        return None
    from remembrancerlib import (
        human_trust_valid,
        load_human_trust_audit,
        load_notes,
    )

    audit = load_human_trust_audit(root)
    if not audit.trustworthy:
        return None
    row = audit.latest.get(note_name)
    if not isinstance(row, dict):
        return None
    mapping = {
        ("promote", "verified"): "approved",
        ("reject", "superseded"): "rejected",
    }
    event = mapping.get((row.get("action"), row.get("to")))
    actor = row.get("identity")
    if (
        event is None
        or row.get("from") != "proposed"
        or row.get("expected_existing_sha256") != expected_sha
        or not isinstance(actor, str)
        or not actor
    ):
        return None
    matches = [
        note for note in load_notes(os.path.join(os.path.abspath(root), "context"))
        if note.name == note_name
    ]
    if len(matches) != 1 or not human_trust_valid(matches[0], audit):
        return None
    if getattr(matches[0], "state", None) != row.get("to"):
        return None
    return {"id": item["id"], "note": note_name, "event": event, "actor": actor}


def reconcile_completed(root: str) -> list[dict]:
    """Finish queue copy two after an audit-first note transition crash.

    Promote/reject commits the signed human-trust row and note before the queue
    resolution.  If the final append fails, the next inbox read can prove that
    exact preimage transition from the trust journal and idempotently close the
    secondary queue without asking the reviewer to approve new bytes.
    """
    recovered: list[dict] = []
    for item in _fold(root).values():
        if item.get("latest") not in _BINDING_OPEN:
            continue
        resolution = _completed_resolution(root, item)
        if resolution is None:
            continue
        record(
            root, resolution["id"], resolution["event"], resolution["actor"],
            recovered=True,
        )
        recovered.append(resolution)
    return recovered


def pending(
    root: str,
    visible_names: set[str] | None = None,
    *,
    reconcile: bool = True,
) -> list[dict]:
    """Items awaiting a decision (latest event in :data:`_BINDING_OPEN`), ACL-gated by
    visible_names, newest first."""
    if reconcile:
        reconcile_completed(root)
    out = []
    for it in _fold(root).values():
        if it.get("latest") not in _BINDING_OPEN:
            continue
        if visible_names is not None and it.get("note") and it["note"] not in visible_names:
            continue
        out.append(it)
    out.sort(key=lambda i: i.get("ts", ""), reverse=True)
    return out


def get_item(root: str, item_id: str) -> dict | None:
    return _fold(root).get(item_id)


# ─── Bound-pending guard (enrichment must not stale a human's binding) ───

def _open_binding_items(root: str) -> list[dict]:
    """Items whose latest event is in :data:`_BINDING_OPEN`; read-only.

    No reconcile and no append (``pending()`` does both).  ``_fold`` raises a
    bare ``OSError`` for a ledger it cannot read (over the size bound, a short
    read, a failed lock); that is as untrustworthy as a malformed ledger, so it
    is re-raised as :class:`ApprovalQueueError` and callers keep ONE refusal
    path instead of crashing on the other.
    """
    try:
        items = _fold(root)
    except OSError as exc:
        raise ApprovalQueueError("approval queue is unreadable") from exc
    return [item for item in items.values() if item.get("latest") in _BINDING_OPEN]


def _load_persisted_note(path: str):
    """Parse one note file the way :func:`remembrancerlib.load_notes` parses each file.

    Same text-mode read and frontmatter reader, so the content fingerprint is
    the one a full load would compute.  No suppression or trust binding: the
    binding test needs only content fields, state and the file's bytes.
    """
    from remembrancerlib import Note, parse_frontmatter

    with open(path) as handle:
        frontmatter, body = parse_frontmatter(handle.read())
    return Note.from_frontmatter(frontmatter, body, path)


def bound_pending_names(root: str, corpus) -> frozenset[str]:
    """Names whose exact current bytes an open approval item binds.

    ``corpus`` is the caller's own independently loaded notes (one
    ``load_notes`` per run).  The test per item is the one
    :func:`approval_item_matches` applies: exactly one note of that name,
    still ``proposed``, whose content fingerprint and on-disk sha both equal
    the item's.  Read-only: no reconcile, no writes, one ledger fold plus one
    file read per note an open item names, so O(notes + queue rows) where
    ``approval_item_matches`` per item is O(items x notes).  Raises
    :class:`ApprovalQueueError` when the ledger is unsafe or unreadable;
    callers fail closed.

    Pipeline enrichment (verification and confidence rewrites) must skip these
    notes: rewriting them changes ``expected_existing_sha256`` and silently
    invalidates the decision a human is about to make.
    """
    by_name: dict[str, list] = {}
    for note in corpus:
        by_name.setdefault(getattr(note, "name", None), []).append(note)
    bound: set[str] = set()
    bindings: dict[str, dict[str, str]] = {}
    for item in _open_binding_items(root):
        candidates = by_name.get(item.get("note")) or []
        if len(candidates) != 1:
            # Missing, suppressed or ambiguous: approval_item_matches refuses
            # the item as well, so no human can act on these bytes through it.
            continue
        note = candidates[0]
        if note.name not in bindings:
            try:
                bindings[note.name] = _persisted_approval_binding(note)
            except (OSError, RuntimeError, TypeError, ValueError):
                bindings[note.name] = {}
        if _binding_matches(item, bindings[note.name]):
            bound.add(note.name)
    return frozenset(bound)


#: Whether :func:`note_bound_pending` has already said, in this process, that
#: it is failing closed.  Without the warning every distill verification commit
#: turns ledger-only and nothing says why -- silence lets an audit outage go
#: unnoticed -- and one line per commit would flood a long run, so it is said
#: once.
_bound_guard_failed_closed_reported = False


def note_bound_pending(root: str, note) -> bool:
    """Write-time single-note form of :func:`bound_pending_names`.

    Re-reads the ledger and the note's file now, so it also sees an item the
    caller's own run enqueued a moment ago.  The binding is computed from the
    file as it is on disk, never from the caller's in-memory copy: an item
    binds persisted bytes, and :func:`_persisted_approval_binding` is only
    sound for an independently loaded note.  A copy that drifted from its
    file would otherwise fingerprint differently, read as "not bound", and let
    the write through.  Fails CLOSED: an unsafe or unreadable ledger answers
    True (do not rewrite), with one warning per process.
    """
    global _bound_guard_failed_closed_reported
    name = getattr(note, "name", None)
    path = getattr(note, "path", None)
    try:
        items = [item for item in _open_binding_items(root) if item.get("note") == name]
    except ApprovalQueueError as exc:
        if not _bound_guard_failed_closed_reported:
            _bound_guard_failed_closed_reported = True
            _log.warning(
                "approval queue under %s is untrustworthy (%s); verification "
                "commits stay ledger-only and rewrite no note until it is "
                "repaired (approval-queue-untrustworthy)",
                root, exc,
            )
        return True
    if not items or not path:
        return False
    try:
        binding = _persisted_approval_binding(_load_persisted_note(path))
    except (OSError, RuntimeError, TypeError, ValueError):
        # No exact proposed preimage at that path, so no item can bind it.
        return False
    return any(_binding_matches(item, binding) for item in items)


def reconcile_required_reviews(root: str, *, apply: bool = False) -> dict:
    """Plan or apply hard holds + exact queue items for current high-risk notes.

    This is the crash reconciler and legacy migration for the producer-side
    governance transaction. It deliberately includes recognizable machine and
    legacy/manual proposed memory: classification determines stakes, while the
    report keeps authorship visible for attended triage.
    """
    from remembrancerlib import (
        current_note_names,
        effective_state,
        load_notes,
        machine_authored,
        pipeline_mutation_lock,
    )
    from remembrancerlib.note import write_note as core_write_note
    from remembrancerlib.verification import governance_hold_class

    root = os.path.abspath(root)
    context_dir = os.path.join(root, "context")
    result = {
        "schema_version": 1,
        "mode": "apply" if apply else "dry-run",
        "required": 0,
        "machine": 0,
        "legacy_or_manual": 0,
        "already_held": 0,
        "exact_pending": 0,
        "held": 0,
        "enqueued": 0,
        # Notes whose exact current bytes already carry a resolved queue item
        # (approved / rejected / edited) while the note itself is still
        # ``proposed``.  Nothing can be enqueued for them again; they are a
        # human follow-up and are reported, never retried as an error.
        "resolved": 0,
        "resolved_notes": [],
        "errors": [],
        "classes": {},
    }

    def run() -> None:
        from retrieve.maintenance import automatic
        if automatic(_config(root)):
            # The derived maintenance projection owns exception triage. Keep
            # existing holds/queue bytes for audit and immediate policy rollback.
            return
        notes = load_notes(context_dir)
        current = current_note_names(notes)
        cfg = _config(root)
        pending_items = pending(root)
        pending_by_note_class: dict[tuple[str, str], list[dict]] = {}
        for item in pending_items:
            key = (item.get("note"), item.get("class"))
            pending_by_note_class.setdefault(key, []).append(item)
        for original in sorted(notes, key=lambda item: item.name):
            if original.name not in current or effective_state(original) == "verified":
                continue
            if original.state in {"verified", "superseded"}:
                continue
            held_class = governance_hold_class(original)
            cls, decision = classify(
                original,
                RouteMeta(requester="governance-reconcile"),
                cfg,
            )
            if held_class is not None:
                cls, decision = held_class, APPROVAL
            if decision != APPROVAL:
                continue
            result["required"] += 1
            result["classes"][cls] = result["classes"].get(cls, 0) + 1
            if machine_authored(original):
                result["machine"] += 1
            else:
                result["legacy_or_manual"] += 1
            if held_class is not None:
                result["already_held"] += 1
            candidates = pending_by_note_class.get((original.name, cls), [])
            try:
                binding = (
                    _persisted_approval_binding(original) if candidates else {}
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                binding = {}
            exact = [
                item for item in candidates
                if binding and all(
                    isinstance(item.get(key), str)
                    and item.get(key) == value
                    for key, value in binding.items()
                )
            ]
            if len(exact) == 1:
                result["exact_pending"] += 1
                continue
            if not apply:
                continue
            try:
                if held_class is None:
                    note = copy.deepcopy(original)
                    note.state = "proposed"
                    note.superseded = False
                    prepare_for_review(note, cls)
                    core_write_note(note, context_dir)
                    result["held"] += 1
                else:
                    # Crash recovery starts from bytes that already carry the
                    # monotonic hard hold. Rewriting those identical bytes is
                    # both unnecessary and forbidden without an exact review
                    # capability; only the missing queue copy needs repair.
                    note = original
                targets = sorted({
                    target for relation, target in note.typed_links
                    if relation == "supersedes" and target
                })
                enqueue_for_review(
                    note,
                    requester="governance-reconcile",
                    reason="high-stakes memory migration/reconciliation",
                    root=root,
                    cls=cls,
                    supersedes=targets or None,
                )
                result["enqueued"] += 1
            except ApprovalItemResolvedError:
                result["resolved"] += 1
                result["resolved_notes"].append(original.name)
                _log.info(
                    "governance reconcile: %r already has a resolved approval "
                    "item for its current bytes; note is still proposed -- %s",
                    original.name, _REAPPLY_HINT,
                )
            except Exception as exc:
                result["errors"].append({
                    "note": original.name,
                    "error": str(exc),
                })

    if apply:
        with pipeline_mutation_lock(root, blocking=False):
            run()
    else:
        run()
    result["complete"] = not result["errors"] and (
        not apply
        or result["required"]
        == result["exact_pending"] + result["enqueued"] + result["resolved"]
    )
    return result
