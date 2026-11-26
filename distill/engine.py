#!/usr/bin/env python3
"""The distillation engine — Layer-1 events -> Layer-2 durable notes (PARADIGM §4).

The heart of Remembrancer: lossy compression of an event flood toward what stays true.
Reads events past the `distill` watermark, CLUSTERS related ones, RESOLVES
customer entities against the registry, and per cluster picks exactly one of:

    CREATE  — a durable fact/decision/state not yet captured -> new atomic Note.
    UPDATE  — cluster matches an existing note -> append provenance, bump updated.
    LINK    — cluster relates two existing notes -> add [[links]].
    IGNORE  — pure chatter (the majority) -> no note; the event still lives in L1.

Distillation is a WRITE-time operation. With no API key it runs on a fully
deterministic heuristic. With `--llm` and an LLM configured (distill/llm_env.py)
it MAY call an LLM to sharpen a note's description/body — but only as an
enhancement layer over the same heuristic decision; the engine never calls an
LLM at read time.

CLI:  python3 distill/engine.py [--root .] [--llm] [--rebuild]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

# Import the frozen shared core. Support running both as a module
# (`python3 -m distill.engine`) and as a script (`python3 distill/engine.py`).
try:
    from remembrancerlib import (
        Event,
        iter_validated_events as iter_events,
        Note,
        write_note,
        regen_index,
        load_notes,
        index_notes_by_name,
        slugify,
        ACLConflict,
        Cursors,
        EventValidationError,
        acl_can_reveal,
        combine_acl,
        current_note_names,
        effective_state,
        event_text,
        concise_description,
        durable_replace_json,
        source_retirement_valid,
    )
    from distill.entities import (
        EntityResolver,
        registry_types,
        Canonicalizer,
        emit_proposals,
    )
    from distill.llm import LLMClient, llm_requirement
    from distill.consolidate import consolidate, grounded_evidence_quotes
    from remembrancerlib.source_lifecycle import (
        _record_source_retirement,
    )
    from remembrancerlib.locks import serialized_pipeline_mutation
except ImportError:  # pragma: no cover - script-relative fallback
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from remembrancerlib import (
        Event,
        iter_validated_events as iter_events,
        Note,
        write_note,
        regen_index,
        load_notes,
        index_notes_by_name,
        slugify,
        ACLConflict,
        Cursors,
        EventValidationError,
        acl_can_reveal,
        combine_acl,
        current_note_names,
        effective_state,
        event_text,
        concise_description,
        durable_replace_json,
        source_retirement_valid,
    )
    from distill.entities import (
        EntityResolver,
        registry_types,
        Canonicalizer,
        emit_proposals,
    )
    from distill.llm import LLMClient, llm_requirement
    from distill.consolidate import consolidate, grounded_evidence_quotes
    from remembrancerlib.source_lifecycle import (
        _record_source_retirement,
    )
    from remembrancerlib.locks import serialized_pipeline_mutation


def write_note(note, context_dir: str, **kwargs) -> str:
    """Distillation's single governed persistence boundary."""
    import govern

    return govern.persist_machine_note(
        note,
        context_dir,
        requester="distill",
        reason="distillation output",
        **kwargs,
    )


def _cluster_govern_source(cl: "Cluster", context_dir: str) -> str | None:
    """The outside author a NEW note distilled from ``cl`` is attributed to.

    ``email:<address>`` of the cluster's first outside-authored
    email event (``govern.external_event_source``), else None.  Every path that
    creates a note from a cluster passes it as ``source=`` to the write door, so
    ``govern.classify`` holds the note as ``external`` (never auto-granted).
    UPDATEs of existing notes do not pass it: holding a note that was visible
    before an outside email touched it is the owner's call.
    """
    import govern

    return govern.external_event_source(
        cl.events, os.path.dirname(os.path.abspath(context_dir))
    )


# Optional BM25 retrieval — degrade gracefully if module missing.
try:
    from retrieve.bm25 import rank as _bm25_rank
except ImportError:  # pragma: no cover

    def _bm25_rank(query, docs):  # type: ignore[misc]
        return list(enumerate([0.0] * len(docs)))


# Meeting-event contract (shared with a meeting assistant). Guarded so a
# distill still runs if the ingest package is momentarily unavailable — meetings
# then simply flow the generic path (which routes them to `business` by
# provenance domain) instead of the dedicated enrichment handler.
try:
    from ingest.meeting_events import is_meeting_event, parse_meeting_event
except Exception:  # pragma: no cover - ingest package absent/broken
    is_meeting_event = None  # type: ignore[assignment]
    parse_meeting_event = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Heuristic vocabulary
# ---------------------------------------------------------------------------

# An event is "durable-ish" (worth a note) if it carries a decision or a
# concrete unit of work. git commits and decision/CI/training/session events
# qualify; pure slack/gmail chatter does not, UNLESS its text signals a
# decision/ship/outcome. Kept deterministic and auditable (see rules.md).
#
# `upload` is DURABLE by source: a human uploading a document (like an OKF
# bundle) is durable *intent* — the person deliberately curated that content
# into memory. Uploads are rare and intentional, so every uploaded doc should
# distill into a proposed note (routed by domain) for the human to keep/drop,
# even when its text carries no decision keyword. Without this, a plain (non-
# decision) uploaded doc has no durable event → IGNORE → 0 proposed notes.
DURABLE_SOURCES = {
    "git",
    "ci",
    "training",
    "decision",
    "session",
    "gcal",
    "gdrive",
    "crm",
    "upload",
    "meeting",
    "artifact",
}

# `decid`, `migrat`, `adopt` and `deprecat` are stems, so they take `\w*`: with
# the group's closing `\b` a bare stem matched only the stem itself, which made
# "decided", "migrated", "deprecated" and "adoption" non-durable while the
# future-tense "will ship" was durable. The leading `\b` stays, so
# "undecided" and "indecision" still do not match.
DECISION_RE = re.compile(
    r"\b(decid\w*|decision|chose|chosen|agreed|ship(?:ped|s|ping)?|launch(?:ed)?|"
    r"rolled out|released?|approved?|migrat\w*|adopt\w*|deprecat\w*|fix(?:ed|es)?|"
    r"resolv(?:ed|es)?|root[- ]?cause|postmortem|outcome|eval(?:uation)?|"
    r"baseline|threshold|policy|standard)\b",
    re.IGNORECASE,
)

# Pure-chatter markers that, absent any durable signal, push toward IGNORE.
CHATTER_RE = re.compile(
    r"^\s*(thanks|thank you|lol|nice|ok(ay)?|got it|\+1|congrats|gm|good morning|"
    r"ping|bump|standup|fyi)\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "with",
    "by",
    "this",
    "that",
    "it",
    "its",
    "we",
    "i",
    "you",
    "they",
    "add",
    "added",
    "update",
    "updated",
    "fix",
    "fixed",
    "new",
    "use",
    "using",
    "from",
    "into",
    "via",
    "customer",
}


def _event_text(ev: Event) -> str:
    """Best free-text representation of an event payload across sources."""
    return event_text(ev)


def _is_durable(ev: Event) -> bool:
    if ev.source in DURABLE_SOURCES:
        return True
    return bool(DECISION_RE.search(_event_text(ev)))


def _is_pure_chatter(ev: Event) -> bool:
    if _is_durable(ev):
        return False
    return bool(CHATTER_RE.search(_event_text(ev))) or len(_event_text(ev)) < 12


# --- DISTILL SKIP HOOK: exclude machine-sent email from becoming memory -------
# OPT-IN. Memory should distill HUMAN interaction, not machine noise (newsletters,
# notifications, receipts). This gate runs a message-level human-vs-machine
# classifier, when one is installed, over gmail events and drops machine-sent
# ones BEFORE clustering. Enabled by config distill.skip_machine_email == True (or
# run(..., skip_machine=True)). Non-gmail events are always kept. Degrades to a
# no-op if the classifier import fails.
def _machine_email_filter(events: list[Event], min_confidence: float = 0.8):
    """Partition events into (kept, dropped). dropped = gmail events the classifier
    labels machine-sent at >= min_confidence. Returns (kept, dropped_records)."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tools.human_vs_machine import is_machine_sent  # noqa: E402
    except Exception:  # pragma: no cover - classifier absent -> keep everything
        return events, []
    kept, dropped = [], []
    for ev in events:
        if ev.source != "gmail":
            kept.append(ev)
            continue
        v = is_machine_sent(
            ev.to_dict() if hasattr(ev, "to_dict") else _event_as_dict(ev)
        )
        if v.is_machine and v.confidence >= min_confidence:
            dropped.append(
                {
                    "id": ev.id,
                    "author": ev.author,
                    "subject": (ev.payload or {}).get("subject", ""),
                    "confidence": round(v.confidence, 2),
                    "reason": v.reason,
                }
            )
        else:
            kept.append(ev)
    return kept, dropped


def _event_as_dict(ev: Event) -> dict:
    """Minimal dict view of an Event for the classifier (which reads author/payload/
    provenance)."""
    return {
        "author": ev.author,
        "payload": ev.payload or {},
        "provenance": ev.provenance or {},
        "entities": ev.entities or [],
        "source": ev.source,
    }


# --- DISTILL BACKFILL VALVE: don't spend LLM calls on a git-history backfill --
# An org-wide git backfill floods one run with tens of thousands of OLD commits.
# Footprints/timeline/dig want the full history (they read raw L1 events), but
# distilling months of stale commits into notes is LLM cost without memory value.
# Enabled by config distill.git_max_age_days (absent/0 = keep everything). Only
# git events are ever dropped; the distill cursor still advances past them, so
# they are skipped exactly once, and fresh commits keep distilling nightly.
def _git_age_filter(events: list[Event], max_age_days, now: str | None = None):
    """Partition (kept, dropped_count): drop source=git events whose occurred_at
    date is more than max_age_days before `now` (ISO date, default today UTC)."""
    if not max_age_days:
        return events, 0
    import datetime as _dt

    today = (
        _dt.datetime.fromisoformat(now).date()
        if now
        else _dt.datetime.now(_dt.timezone.utc).date()
    )
    cutoff = (today - _dt.timedelta(days=int(max_age_days))).isoformat()
    kept = [
        ev
        for ev in events
        if ev.source != "git" or (ev.occurred_at or "")[:10] >= cutoff
    ]
    return kept, len(events) - len(kept)


def _gcal_future_filter(
    events: list[Event], max_future_days=180, now: str | None = None
):
    """Partition calendar events that fall outside the useful memory horizon.

    Google Calendar's ``singleEvents`` API expands recurring meetings.  A
    missing API bound once admitted years of future occurrences into L1.  Raw
    events stay immutable, but future instances beyond this horizon must never
    consume clustering/model work or become claims about current plans.

    ``max_future_days`` of ``0``/``None`` disables the gate.  Invalid timestamps
    are retained so a separate schema validator can report them rather than
    silently discarding evidence.
    """
    if not max_future_days:
        return events, 0
    import datetime as _dt

    if now:
        anchor = _dt.datetime.fromisoformat(str(now).replace("Z", "+00:00"))
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=_dt.timezone.utc)
    else:
        # Deterministic anchor: use the latest ingested_at among the
        # events being filtered, falling back to wall-clock only when
        # no events have a usable timestamp. This prevents the test from
        # becoming date-dependent: as long as the ingested_at is stable
        # (which it should be in tests and in real data), the cutoff
        # window is stable too.
        candidates: list = []
        for event in events:
            ts = getattr(event, "ingested_at", None)
            if not ts:
                continue
            try:
                d = _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if d.tzinfo is None:
                    d = d.replace(tzinfo=_dt.timezone.utc)
                candidates.append(d)
            except Exception:
                pass
        if candidates:
            anchor = max(candidates)
        else:
            anchor = _dt.datetime.now(_dt.timezone.utc)
    cutoff = anchor + _dt.timedelta(days=int(max_future_days))
    kept: list[Event] = []
    dropped = 0
    for event in events:
        if event.source != "gcal":
            kept.append(event)
            continue
        try:
            occurred = _dt.datetime.fromisoformat(
                str(event.occurred_at).replace("Z", "+00:00")
            )
            if occurred.tzinfo is None:
                occurred = occurred.replace(tzinfo=_dt.timezone.utc)
        except (TypeError, ValueError):
            kept.append(event)
            continue
        if occurred > cutoff:
            dropped += 1
        else:
            kept.append(event)
    return kept, dropped


def _gcal_personal_block_filter(events: list[Event], patterns):
    """Partition (kept, dropped): drop source=gcal events whose title matches a
    personal time-block pattern (e.g. ``^focus:\\s`` → "focus: deep work").

    These are one person's calendar holds, not company memory.  The attendee
    gates in ``ingest.gcal`` miss them because such blocks frequently carry an
    internal co-attendee (``jdoe`` + ``asmith``), defeating the min-attendee and
    empty-attendee heuristics.  The distinguishing signal is the title.  Raw
    events stay immutable in L1 (footprints/dig still see them); they are only
    kept out of clustering, notes, and model work.

    Empty/``None`` ``patterns`` (or all-invalid patterns) disable the gate.
    """
    return _payload_pattern_filter(events, "gcal", "title", patterns)


def _git_subject_filter(events: list[Event], patterns):
    """Partition (kept, dropped): drop source=git events whose commit subject
    matches a pattern (e.g. ``^\\[nightly\\] memory update\\b``).

    A pipeline that commits its own output (notes, events, state) back to a
    repository it also ingests would otherwise distill those commits like any
    other work: their changed paths name the notes they rewrote, so they take
    those notes' entities, and every run adds another note about the memory
    itself.  Raw events stay immutable in L1 (recent git activity, footprints
    and dig still see them); they are only kept out of clustering, notes, and
    model work.

    Empty/``None`` ``patterns`` (or all-invalid patterns) disable the gate.
    """
    return _payload_pattern_filter(events, "git", "subject", patterns)


def _payload_pattern_filter(events: list[Event], source: str, field: str, patterns):
    """Partition (kept, dropped): drop ``source`` events whose ``payload[field]``
    matches any of ``patterns`` (case-insensitive regexes, searched).  Events of
    other sources, and events with no such text, are kept.  Empty/``None``
    ``patterns`` (or all-invalid patterns) disable the gate."""
    if not patterns:
        return events, 0
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            continue
    if not compiled:
        return events, 0
    kept: list[Event] = []
    dropped = 0
    for event in events:
        text = ""
        payload = getattr(event, "payload", None)
        if event.source == source and isinstance(payload, dict):
            text = str(payload.get(field) or "").strip()
        if text and any(rx.search(text) for rx in compiled):
            dropped += 1
        else:
            kept.append(event)
    return kept, dropped


def _slack_low_signal_filter(events: list[Event]):
    """Partition (kept, dropped): drop source=slack events persisted with
    ``payload.signal_eligible == false``.

    With ``slack.persist_low_signal`` on, standalone human messages (no
    replies, no resolution emoji) reach L1 so the channel record is COMPLETE
    for transcripts and event search — but they are exactly the noise the
    thread signal gate exists to keep out of clustering/notes, so the gate
    re-applies here at distill intake.  A missing flag means an eligible or
    pre-flag event and is kept.  Tombstones (``payload.deleted``) always
    flow: a deletion must keep superseding whatever notes cite the thread.
    """
    kept: list[Event] = []
    dropped = 0
    for event in events:
        payload = getattr(event, "payload", None)
        if (
            event.source == "slack"
            and isinstance(payload, dict)
            and payload.get("signal_eligible") is False
            and not payload.get("deleted")
        ):
            dropped += 1
        else:
            kept.append(event)
    return kept, dropped


# --- DISTILL INTAKE: the one policy door for both distillation entry points ---
def distill_intake(
    events: list[Event],
    cfg: dict[str, Any] | None,
    *,
    skip_machine: bool | None = None,
) -> tuple[list[Event], dict[str, Any]]:
    """Apply distill's intake policy to an L1 event stream.

    Returns ``(kept, report)``.  This is the ONLY intake: ``run()`` (the
    incremental / rebuild engine) and the resumable production digest both
    call it, so a valve added here reaches
    both, so a valve added here must stay shared: if either copy drifts and
    drops the calendar personal-block valve or the Slack low-signal valve,
    ``focus:`` blocks and ``signal_eligible=false`` messages reach production
    clusters through the drifted path while the other keeps them out.

    Filters, in this order (``run()``'s order, kept on purpose):

    1. machine-sent gmail, when ``distill.skip_machine_email`` is true or the
       caller passes ``skip_machine=True`` (an explicit argument wins);
    2. the latest complete snapshot per mutable source object
       (``_latest_mutable_snapshots``);
    3. old git commits (``distill.git_max_age_days``);
    4. far-future calendar instances (``distill.gcal_max_future_days``, else
       ``gcal.lookahead_days``, else 180);
    5. calendar personal blocks (``distill.gcal_drop_title_patterns``);
    6. Slack messages persisted with ``signal_eligible=false``; tombstones
       (``payload.deleted``) always flow;
    7. git commits whose subject matches ``distill.git_drop_subject_patterns``,
       such as the commits a pipeline makes to publish its own output.

    Machine mail is filtered BEFORE snapshot collapse.  That was ``run()``'s
    order; the digest used to collapse first.  With today's sources the two
    orders yield identical events: the machine classifier judges only gmail,
    and gmail messages are individual (cumulative) events that
    ``_latest_mutable_snapshots`` never collapses.  The order is pinned so the
    two doors cannot drift apart again.  If a future source is BOTH
    snapshot-collapsed and machine-judged, revisit it: dropping a machine-sent
    newest revision first would let an older human revision of the same object
    stand in as current, the hazard ``_latest_mutable_snapshots``'s caller
    comment describes for value filters.  Two intake-parity tests fail the
    day that happens: the order canary (a human message followed by newer machine mail in one
    thread) and the check that no ``_SNAPSHOT_LINEAGE_SOURCES`` source is
    judged by ``_machine_email_filter``.

    The report always carries the would-be machine-mail drop count, even when
    the filter is off, so a dry run shows its impact.  ``policy_excluded``
    lists the events filters 5 to 7 removed: they are well-formed, current
    events that policy keeps out of clustering, so the digest can find ledger
    rows it had already started on them before the policy reached it.
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    distill_cfg = cfg.get("distill") or {}
    if skip_machine is None:
        skip_machine = bool(distill_cfg.get("skip_machine_email"))
    events = list(events)
    kept, machine_dropped = _machine_email_filter(events)
    if skip_machine:
        events = kept
    # Select the newest complete revision *before* the value-based filters
    # below.  Otherwise a current calendar revision moved beyond the horizon is
    # removed first and an older in-horizon revision is resurrected as truth.
    before_snapshot_collapse = len(events)
    events = _latest_mutable_snapshots(events)
    mutable_revisions_superseded = before_snapshot_collapse - len(events)
    # Backfill valve: keep old git commits OUT of LLM distillation (they stay in
    # the L1 store for footprints/dig).
    events, git_age_dropped = _git_age_filter(
        events, distill_cfg.get("git_max_age_days")
    )
    gcal_horizon = distill_cfg.get(
        "gcal_max_future_days",
        (cfg.get("gcal") or {}).get("lookahead_days", 180),
    )
    events, gcal_future_dropped = _gcal_future_filter(events, gcal_horizon)
    # Personal-block valve: one person's calendar holds ("focus: deep work") are not
    # company memory and the ingest attendee gates miss them (internal
    # co-attendees).  They stay in L1.
    before_policy = events
    events, gcal_block_dropped = _gcal_personal_block_filter(
        events, distill_cfg.get("gcal_drop_title_patterns") or []
    )
    # Low-signal Slack valve: persisted standalone messages stay in L1
    # (complete transcripts / event search) but never reach clustering/notes.
    events, slack_low_signal_dropped = _slack_low_signal_filter(events)
    # Self-ingestion valve: commits that publish the pipeline's own output stay
    # in L1 (recent git activity still lists them) but never become notes.
    events, git_subject_dropped = _git_subject_filter(
        events, distill_cfg.get("git_drop_subject_patterns") or []
    )
    kept_ids = {id(event) for event in events}
    policy_excluded = [event for event in before_policy if id(event) not in kept_ids]
    report = {
        "machine_skip": {
            "enabled": bool(skip_machine),
            "dropped_events": len(machine_dropped),
            "examples": machine_dropped[:10],
        },
        "mutable_revisions_superseded": mutable_revisions_superseded,
        "git_age_dropped": git_age_dropped,
        "gcal_horizon": gcal_horizon,
        "gcal_future_dropped": gcal_future_dropped,
        "gcal_block_dropped": gcal_block_dropped,
        "slack_low_signal_dropped": slack_low_signal_dropped,
        "git_subject_dropped": git_subject_dropped,
        "policy_excluded": policy_excluded,
    }
    return events, report


def _keywords(text: str, k: int = 8) -> list[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", (text or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for t in toks:
        if t in STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@dataclass
class Cluster:
    key: str
    events: list[Event] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    # Eng-component / initiative ids routed OUT of `entities` — see
    # `_canonicalize_cluster_entities`. Disjoint from `entities` by construction.
    components: list[str] = field(default_factory=list)
    # All immutable event ids in the same mutable source object, including
    # revisions/messages that predate this incremental pass.  A later Slack
    # thread revision or Gmail reply must update the note already bound to that
    # source lineage even when its corrected text no longer resembles the old
    # description.  Populated by ``run`` only for closed lineage contracts.
    lineage_provenance_ids: list[str] = field(default_factory=list)
    # Stable mutable-object identities represented by this cluster.  New notes
    # persist these separately from provenance citations so future corrections
    # have explicit ownership rather than guessing from a shared citation.
    source_lineages: list[str] = field(default_factory=list)
    # Registry-authoritative note domain for the deterministic path, set by
    # _decide_and_apply (None = no external registry entity -> use cl.domain).
    _reg_domain: str | None = None

    @property
    def acl(self) -> list[str]:
        return _safe_acl(*(ev.acl for ev in self.events))

    @property
    def domain(self) -> str:
        # Most common domain hint across events; prefer an explicit provenance
        # domain, then the source's natural folder.
        counts: dict[str, int] = defaultdict(int)
        for ev in self.events:
            d = (ev.provenance or {}).get("domain")
            if not d:
                d = _source_domain(ev)
            counts[d] += 1
        return max(counts, key=lambda d: (counts[d], d)) if counts else "company"

    @property
    def text(self) -> str:
        return " ".join(_event_text(ev) for ev in self.events).strip()

    @property
    def provenance_ids(self) -> list[str]:
        return [ev.id for ev in self.events]


class MutableLineageConflict(RuntimeError):
    """One mutable source object is already claimed by several current notes."""

    def __init__(self, cluster_key: str, note_names: list[str]):
        self.cluster_key = cluster_key
        self.note_names = sorted(note_names)
        super().__init__(
            f"mutable source lineage {cluster_key!r} is bound to multiple current notes: "
            + ", ".join(self.note_names)
        )


class MutableLineageOwnershipRequired(RuntimeError):
    """A legacy citation exists, but no note has explicit source ownership."""

    def __init__(self, cluster_key: str, note_names: list[str]):
        self.cluster_key = cluster_key
        self.note_names = sorted(note_names)
        super().__init__(
            f"mutable source lineage {cluster_key!r} needs ownership review: "
            + ", ".join(self.note_names)
        )


def _safe_acl(*acls: list[str]) -> list[str]:
    """ACL safe for every input under Remembrancer's OR-grant semantics.

    A union of ``customers`` and ``eng`` grants both groups access to a note
    containing each other's input. Restrict to their common grant instead;
    incompatible scopes cannot be represented by Remembrancer's OR labels and raise
    ``ACLConflict``. Public inputs do not widen a restricted input.
    """
    return combine_acl(*acls)


def _acl_compatible(*acls: list[str]) -> bool:
    try:
        combine_acl(*acls)
        return True
    except ACLConflict:
        return False


# Repository -> context domain, from the git repositories the config declares
# (``repo_domains``).  run() binds it from its root's config.json.  Callers
# that skip run() (a digest script, maintenance tools) read the code folder's
# config.json on first use, as they read the code folder's registry.
_REPO_DOMAIN: dict[str, str] | None = None
_SOURCE_DOMAIN = {
    "git": "eng",
    "ci": "eng",
    "session": "eng",
    "slack": "eng",
    "training": "ml",
    "gmail": "network",
    "crm": "customers",
    "gcal": "company",
    "gdrive": "company",
    "meeting": "business",
    "artifact": "internal",
}
# NOTE: gmail DEFAULTS to `network` (neutral), NOT customers — the customers/prospect/
# investor/... bucket must be EARNED via the entity registry (_registry_domain). Defaulting
# unknown gmail senders to customers is what mislabeled a batch of one-time/tooling notes as
# "customers" (since fixed).


def repo_domains(cfg: Any) -> dict[str, str]:
    """Repository name -> context domain, from the config's git repositories.

    Reads the entries the git connector ingests, with its precedence
    (``ingest/git_events.py``, ``GitConnector``): ``repos`` when the key is
    present, else ``monorepos``.  An entry is named as the connector names its
    events' ``provenance.repo``: ``name`` (``repo`` in ``monorepos``), else the
    last part of ``path``.  An entry with neither, or without a string
    ``domain``, maps nothing; its events take their source's domain.
    """
    if not isinstance(cfg, dict):
        return {}
    entries = cfg.get("repos")
    if entries is None:
        monorepos = cfg.get("monorepos")
        entries = [
            {"name": m.get("repo") or m.get("path"), "path": m.get("path"),
             "domain": m.get("domain")}
            for m in (monorepos if isinstance(monorepos, list) else [])
            if isinstance(m, dict)
        ]
    out: dict[str, str] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        name, path, domain = entry.get("name"), entry.get("path"), entry.get("domain")
        if not name and isinstance(path, str) and path.strip("/"):
            name = os.path.basename(os.path.abspath(path.rstrip("/")))
        if isinstance(name, str) and name and isinstance(domain, str) and domain:
            out[name] = domain
    return out


def _bind_repo_domains(cfg: Any) -> None:
    global _REPO_DOMAIN
    _REPO_DOMAIN = repo_domains(cfg)


# The code folder, whose config.json callers that skip run() read.
_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _repo_domain_map() -> dict[str, str]:
    if _REPO_DOMAIN is None:
        _bind_repo_domains(_load_config(_CODE_ROOT))
    return _REPO_DOMAIN


def _source_domain(ev: Event) -> str:
    repo = (ev.provenance or {}).get("repo")
    repo_domain = _repo_domain_map()
    if repo and repo in repo_domain:
        return repo_domain[repo]
    return _SOURCE_DOMAIN.get(ev.source, "company")


# ---------------------------------------------------------------------------
# Registry-type -> note domain routing (makes the re-route STICK)
# ---------------------------------------------------------------------------
#
# The registry's relationship type for a cluster's PRIMARY external entity is the
# AUTHORITY for which domain folder its note belongs in. When a cluster has an
# external registry entity, this map overrides the LLM's domain guess (and the coarse
# sender-based default) so the next distill never undoes the re-route. Internal-
# subject clusters (no external registry entity) keep their LLM/subject domain.
TYPE_DOMAIN = {
    "customer": "customers",
    "prospect": "prospect",
    "vendor": "vendor",
    "investor": "investors",
    "partner": "partners",
    "advisor": "advisors",
    "friend": "network",
    "champion": "network",
    "academic": "research",
    "recruiting": "recruiting",
}
# Types that are NOT external parties: a cluster routed solely by one of these
# keeps the LLM/subject domain rather than being forced to a relationship folder.
_INTERNAL_TYPES = {"self", "person"}


def _registry_domain(cl: "Cluster") -> str | None:
    """Authoritative note domain for a cluster, derived from its PRIMARY external
    registry entity's type. Returns:

      * a relationship domain (customers/vendor/investors/...) to OVERRIDE the LLM when
        the cluster's first external registry entity is a known external party,
      * the sentinel "__skip__" when the cluster's ONLY external registry entity is
        `noise` (the note should not be created at all),
      * None when the cluster has no external registry entity (internal-subject) —
        the caller keeps the LLM/subject domain.
    """
    rt = registry_types(_REGISTRY_TYPES_PATH)
    canon = _canonicalizer()
    saw_noise = False
    for ent in cl.entities:
        # Write-time canonicalization (below) turns cluster entities into
        # `<type>:<slug>` ids ("investor:initech-ventures"). Those CANNOT round-
        # trip through type_for: its prefix-strip leaves the bare slug, which no
        # longer matches multi-word registry names ("initech ventures"). The
        # canonicalizer minted the id, so when it recognises the entity, ask IT
        # for the type; everything else (raw strings, pre-seeded ids from the
        # separate customers.yml authority) keeps the type_for path.
        cid = canon.canonical_id(ent)
        t = canon.type_of(cid) if cid else rt.type_for(ent)
        if t is None or t in _INTERNAL_TYPES:
            continue
        if t == "noise":
            saw_noise = True
            continue
        dom = TYPE_DOMAIN.get(t)
        if dom:
            return dom  # first external party wins (entities are ordered)
    if saw_noise:
        return "__skip__"  # only-noise external cluster -> drop
    return None  # internal-subject -> keep LLM/subject domain


# run() points this at <root>/registry/entities.yml.  Callers that use the
# engine without run() (a digest script, maintenance tools) read the code folder's own
# registry.
_REGISTRY_TYPES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "registry",
    "entities.yml",
)

# ---------------------------------------------------------------------------
# Write-time entity canonicalization (cluster-union region below)
# ---------------------------------------------------------------------------
#
# Note frontmatter today holds raw LLM/email surface strings ("Jane Doe", "jane
# doe", "123456789012"). This maps every entity a cluster acquires through
# `Canonicalizer` (registry/entities.yml) BEFORE it lands on cl.entities, so
# notes carry one stable id per real-world entity instead of a bag of aliases.

# Resolved at run() time to <root>/state/entity_unresolved.jsonl; module-relative
# default mirrors _REGISTRY_TYPES_PATH so ad-hoc/script callers still work.
_ENTITY_UNRESOLVED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "state",
    "entity_unresolved.jsonl",
)

# Every registry relationship type the engine knows how to route (TYPE_DOMAIN)
# plus the two internal types (self/person) — the full set of `<type>:<slug>`
# prefixes a canonical id can carry. Used to recognise an id that is ALREADY
# canonical-shaped even when it comes from a DIFFERENT authority than
# Canonicalizer's own registry/entities.yml — e.g. `customer:acme`, pre-seeded
# by EntityResolver against registry/customers.yml, a separate file. Such an id
# must pass through unchanged rather than being treated as unresolved noise
# just because Canonicalizer doesn't recognise it as one of ITS entries.
# `system` (internal infrastructure hosts) is registry-only: canonicalizable,
# never domain-routing, hence absent from TYPE_DOMAIN and added by hand here.
_KNOWN_ENTITY_PREFIXES = set(TYPE_DOMAIN) | _INTERNAL_TYPES | {"system"}

# Cached Canonicalizer keyed by _REGISTRY_TYPES_PATH (mirrors the
# registry_types() singleton pattern): rebuilt whenever run()/tests point the
# path at a different registry, shared by cluster_events and _registry_domain.
_CANONICALIZER: "Canonicalizer | None" = None


def _canonicalizer() -> Canonicalizer:
    global _CANONICALIZER
    if _CANONICALIZER is None or _CANONICALIZER.registry_path != _REGISTRY_TYPES_PATH:
        _CANONICALIZER = Canonicalizer(_REGISTRY_TYPES_PATH)
    return _CANONICALIZER


# ---------------------------------------------------------------------------
# ComponentResolver — eng-component / initiative routing
# ---------------------------------------------------------------------------
#
# A raw cluster-entity string the Canonicalizer can't resolve to a real-world
# person/org/system may still name an engineering component
# (registry/eng_components.yml, keyed by `repo` + its `entities:` aliases) or a
# cross-repo initiative (registry/initiatives.yml, keyed by id/title/aliases).
# Built ONCE per `cluster_events` pass (mirrors EntityResolver/Canonicalizer),
# word-boundary matching in EntityResolver's alias-regex discipline — but with
# hyphen continuation blocked on both sides (see _component_boundary_pattern).
# Both registries are OPTIONAL: a repo with neither file resolves nothing, so
# behaviour is byte-identical to before this task when they're absent (e.g.
# every tmp-dir test root that only seeds registry/entities.yml).
# Stdlib-only: the components side reuses ingest.repo_profiles.load_components
# (which has its own PyYAML-optional fallback parser); the initiatives side
# tries PyYAML via _safe_load_yaml and falls back to a stdlib line parser —
# this module never hard-requires PyYAML.

# run() points these at <root>/registry/{eng_components,initiatives}.yml;
# callers that skip run() read the code folder's own files, as for
# _REGISTRY_TYPES_PATH.
_ENG_COMPONENTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "registry",
    "eng_components.yml",
)
_INITIATIVES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "registry",
    "initiatives.yml",
)

# The two kind prefixes ComponentResolver mints — disjoint from
# `_KNOWN_ENTITY_PREFIXES` (Canonicalizer's vocabulary), so a raw string
# already shaped like one of these (a re-processed cluster, or a connector
# that pre-seeded a component id directly) routes straight to `components`
# without a registry lookup.
_KIND_COMPONENT_PREFIXES = frozenset({"component", "initiative"})


def _component_boundary_pattern(words: list[str]) -> "re.Pattern[str] | None":
    """Case-insensitive word-boundary alternation over `words`, in
    `distill.entities.EntityResolver`'s alias-regex discipline — EXCEPT that
    hyphen continuation is blocked on both sides. '-' is not \\w, so the
    EntityResolver-style ``(?<![\\w@])...(?!\\w)`` boundary would let
    "parser-v8" match INSIDE "parser-v8-experimental"; component
    identifiers routinely carry -v2/-experimental/-staging suffixes, and a
    suffixed variant is a DIFFERENT identifier. This tightening is
    ComponentResolver-local: EntityResolver's own matcher (person/org names —
    a different input domain) is deliberately untouched."""
    uniq = sorted({w for w in words if w}, key=len, reverse=True)
    if not uniq:
        return None
    alt = "|".join(re.escape(w) for w in uniq)
    return re.compile(r"(?<![\w@-])(?:" + alt + r")(?![\w-])", re.IGNORECASE)


def _safe_load_yaml(path: str):
    """Best-effort PyYAML load: None when PyYAML is absent, the file is
    missing, or parsing fails — callers fall back to a stdlib parser. The
    import is inside the function so this module never hard-requires PyYAML
    (same discipline as ingest.repo_profiles.load_components)."""
    if not os.path.exists(path):
        return None
    try:
        import yaml  # type: ignore

        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _parse_initiatives_fallback(path: str) -> list[dict]:
    """Stdlib line parser for registry/initiatives.yml's documented block
    shape (`initiatives:` header, `- id: ...` records, `key: value` scalars,
    inline `[a, b]` lists) — mirrors distill.entities._parse_registry_fallback's
    discipline. Only used when _safe_load_yaml yields nothing usable."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return []
    entries: list[dict] = []
    cur: dict | None = None
    in_initiatives = False
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped == "initiatives:":
            in_initiatives = True
            continue
        if not in_initiatives:
            continue
        if stripped.startswith("- "):
            cur = {}
            entries.append(cur)
            stripped = stripped[2:].strip()
            if not stripped:
                continue
        if cur is None or ":" not in stripped:
            continue
        k, v = stripped.split(":", 1)
        k, v = k.strip(), v.strip()
        if v.startswith("[") and v.endswith("]"):
            cur[k] = [_strip_quotes(x) for x in v[1:-1].split(",") if x.strip()]
        else:
            cur[k] = _strip_quotes(v)
    return entries


def _strip_quotes(s: str) -> str:
    return s.strip().strip("\"'")


class ComponentResolver:
    """Raw identifier -> `component:<repo>` / `initiative:<id>`, or None.

    Built once from registry/eng_components.yml (each entry's `repo` plus its
    `entities:` aliases, e.g. "parser-v8" / "Parser" both resolve to
    `component:parser-v8`) and registry/initiatives.yml (each entry's `id`,
    `title`, and `aliases`, e.g. "billing revamp" -> `initiative:
    billing-revamp`). Missing/unreadable registries degrade to zero
    matches — callers see the same unresolved-queue behaviour as before this
    resolver existed. PyYAML is optional on both sides (see the loaders).
    """

    def __init__(
        self,
        components_path: str = _ENG_COMPONENTS_PATH,
        initiatives_path: str = _INITIATIVES_PATH,
    ):
        self.components_path = components_path
        self.initiatives_path = initiatives_path
        self._matchers: list[tuple[str, "re.Pattern[str]"]] = []
        for repo, words in self._load_components(components_path):
            rx = _component_boundary_pattern(words)
            if rx:
                self._matchers.append((f"component:{repo}", rx))
        for iid, words in self._load_initiatives(initiatives_path):
            rx = _component_boundary_pattern(words)
            if rx:
                self._matchers.append((f"initiative:{iid}", rx))

    @staticmethod
    def _load_components(path: str) -> list[tuple[str, list[str]]]:
        """registry/eng_components.yml -> [(repo, [repo, *entities]), ...].

        Delegates the file parsing to `ingest.repo_profiles.load_components`
        — the established loader for this registry, PyYAML-optional (it has
        its own stdlib flow-mapping fallback) and [] on a missing file. The
        import is lazy and guarded so a broken/absent ingest package can
        never take the distiller down (this facet just resolves nothing)."""
        try:
            from ingest.repo_profiles import load_components as _load_eng_components
        except Exception:  # pragma: no cover - ingest package absent/broken
            return []
        try:
            entries = _load_eng_components(path)
        except Exception:  # pragma: no cover - defensive: loader itself is best-effort
            return []
        out: list[tuple[str, list[str]]] = []
        for ent in entries:
            if not isinstance(ent, dict):
                continue
            repo = str(ent.get("repo") or "").strip()
            if not repo:
                continue
            words = [repo] + [
                str(e).strip() for e in (ent.get("entities") or []) if str(e).strip()
            ]
            out.append((repo, words))
        return out

    @staticmethod
    def _load_initiatives(path: str) -> list[tuple[str, list[str]]]:
        """registry/initiatives.yml -> [(id, [id, title, *aliases]), ...].

        PyYAML when available (via _safe_load_yaml); otherwise the stdlib
        block parser (_parse_initiatives_fallback). [] on a missing file."""
        if not os.path.exists(path):
            return []
        data = _safe_load_yaml(path)
        if isinstance(data, dict):
            entries = data.get("initiatives") or []
        else:
            entries = _parse_initiatives_fallback(path)
        out: list[tuple[str, list[str]]] = []
        for ent in entries:
            if not isinstance(ent, dict):
                continue
            iid = str(ent.get("id") or "").strip()
            if not iid:
                continue
            words = [iid, str(ent.get("title") or "").strip()]
            words += [
                str(a).strip() for a in (ent.get("aliases") or []) if str(a).strip()
            ]
            out.append((iid, [w for w in words if w]))
        return out

    def resolve_id(self, s: str) -> str | None:
        """First matcher (registry load order: components, then initiatives)
        whose pattern matches `s`, or None."""
        if not s:
            return None
        for cid, rx in self._matchers:
            if rx.search(s):
                return cid
        return None


# Cached ComponentResolver keyed by (_ENG_COMPONENTS_PATH, _INITIATIVES_PATH)
# (mirrors the `_canonicalizer()` singleton pattern): rebuilt whenever
# run()/tests point either path at different registries, shared by
# `cluster_events`.
_COMPONENT_RESOLVER: "ComponentResolver | None" = None


def _component_resolver() -> ComponentResolver:
    global _COMPONENT_RESOLVER
    if (
        _COMPONENT_RESOLVER is None
        or _COMPONENT_RESOLVER.components_path != _ENG_COMPONENTS_PATH
        or _COMPONENT_RESOLVER.initiatives_path != _INITIATIVES_PATH
    ):
        _COMPONENT_RESOLVER = ComponentResolver(_ENG_COMPONENTS_PATH, _INITIATIVES_PATH)
    return _COMPONENT_RESOLVER


def _looks_already_canonical(s: str) -> bool:
    """True if `s` already has the shape `<known-type>:<slug>`."""
    if ":" not in s:
        return False
    head, _, tail = s.partition(":")
    return head.strip().lower() in _KNOWN_ENTITY_PREFIXES and bool(tail.strip())


def _plausible_unresolved(s: str) -> bool:
    """True for a raw identifier worth queuing for human triage: has at least
    one letter (bare numerics like a phone number are machine junk, not an
    entity) and isn't registry `noise` (free-mail/infra/edu domains, or a
    human-triaged noise verdict).

    Drop strings that are too short to be a real
    entity name (< 3 chars), bare email-handle prefixes (the full email still
    goes through the canonicalizer correctly upstream), or pure ALL-CAPS
    acronyms like "FYI" / "OK". These were leaking onto `entities:` frontmatter
    and bloating retrieval hits with non-entity tokens.
    """
    if not re.search(r"[a-zA-Z]", s or ""):
        return False
    if len(s.strip()) < 3:
        return False
    if s.startswith("@"):
        return False
    if s.isupper():
        return False
    rt = registry_types(_REGISTRY_TYPES_PATH)
    return rt.type_for(s) != "noise"


def _append_unresolved(path: str, entries: list[dict[str, Any]]) -> None:
    """Append newly-seen unresolved identifiers, deduped against every
    identifier already queued (in this file OR earlier in this same call) so
    replaying the same events never produces duplicate queue lines."""
    if not entries:
        return
    existing: set[str] = set()
    if os.path.exists(path):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing.add(json.loads(line).get("identifier"))
                    except Exception:
                        continue
        except Exception:  # pragma: no cover - unreadable queue file -> best-effort
            pass
    fresh = [e for e in entries if e["identifier"] not in existing]
    if not fresh:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for e in fresh:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _has_external_party(raw: list[str], canonicalizer: Canonicalizer) -> bool:
    """True if any raw entity resolves (or is already canonical-shaped) to an
    externally-registered party — a type in TYPE_DOMAIN, not self/person.

    Mirrors `_registry_domain`'s external-party check at canonicalization time:
    a cluster carrying such a party will later be re-routed OUT of "company"
    (to customers/vendor/partner/...), so a self id kept on the company
    exception would leak into that relationship note's frontmatter — and, if
    first-seen, into its slug via `_name_for_cluster`.
    """
    for s in raw:
        if not s:
            continue
        cid = canonicalizer.canonical_id(s)
        if cid is None and _looks_already_canonical(s):
            # Pre-seeded id from another authority (e.g. customer:widgetco via
            # customers.yml) that THIS registry doesn't know. Intentional
            # asymmetry with _registry_domain: the GATE trusts the prefix and
            # counts it as external (so a self id never rides along with it),
            # but the ROUTER still returns None for it (type_for can't classify
            # it) and the note keeps its LLM/subject domain.
            cid = s
        if not cid:
            continue
        t = cid.split(":", 1)[0].strip().lower()
        if t in TYPE_DOMAIN:  # TYPE_DOMAIN keys and _INTERNAL_TYPES are disjoint
            return True
    return False


def _canonicalize_cluster_entities(
    raw: list[str],
    domain: str,
    canonicalizer: Canonicalizer,
    unresolved_path: str,
    queued_this_run: set[str],
    first_day: dict[str, str],
    counts: dict[str, int],
    component_resolver: "ComponentResolver | None" = None,
    emit_unresolved: bool = True,
) -> tuple[list[str], list[str]]:
    """Map a cluster's first-seen-ordered raw entity strings to canonical
    entity ids AND eng-component/initiative ids. Returns `(entities,
    components)` — the two facets are disjoint by construction (a
    `component:`/`initiative:` id NEVER lands in `entities`).

    Per raw string, in order:
      1. Canonicalizer-resolvable & not self  -> entities (canonical id, order
         preserved, deduped)
      2. Canonicalizer-resolvable & self      -> DROPPED, except exactly one
         self id survives when `domain` is "company" AND no externally-
         registered party coexists in the facet (an external party means the
         note will be re-routed by _registry_domain — self must not ride
         along into it)
      3. already `component:`/`initiative:`-shaped, OR ComponentResolver-
         resolvable (registry/eng_components.yml repos+aliases,
         registry/initiatives.yml ids/titles/aliases) -> components
      4. already-canonical-shaped but unknown to THIS registry (e.g. a
         pre-seeded `customer:acme` from the separate customers.yml
         authority) -> entities, unchanged
      5. machine junk (no letter, or registry noise) -> dropped silently
      6. unresolvable but plausible  -> kept RAW in entities + queued to
         entity_unresolved.jsonl

    Order is preserved (first-seen) WITHIN each returned list because
    `_cluster_key` and domain routing both depend on `entities[0]`.
    """
    if component_resolver is None:
        component_resolver = _component_resolver()
    keep_self_ok = domain == "company" and not _has_external_party(raw, canonicalizer)
    out: list[str] = []
    comps: list[str] = []
    kept_self = False
    to_queue: list[dict[str, Any]] = []
    for s in raw:
        if not s:
            continue
        cid = canonicalizer.canonical_id(s)
        if cid:
            if canonicalizer.is_self(s):
                if keep_self_ok and not kept_self:
                    kept_self = True
                    if cid not in out:
                        out.append(cid)
                continue  # self dropped everywhere else
            if cid not in out:
                out.append(cid)
            continue
        # Component/initiative routing: already kind-prefixed (a re-processed
        # cluster, or a connector pre-seeding a component id directly) OR
        # resolvable via the eng-component/initiative registries.
        head = s.split(":", 1)[0].strip().lower() if ":" in s else ""
        if head in _KIND_COMPONENT_PREFIXES:
            if s not in comps:
                comps.append(s)
            continue
        comp_id = component_resolver.resolve_id(s)
        if comp_id:
            if comp_id not in comps:
                comps.append(comp_id)
            continue
        if _looks_already_canonical(s):
            if s not in out:
                out.append(s)
            continue
        if _plausible_unresolved(s):
            if s not in out:
                out.append(s)
            if s not in queued_this_run:
                queued_this_run.add(s)
                to_queue.append(
                    {
                        "identifier": s,
                        "first_seen": first_day.get(s, ""),
                        "count": counts.get(s, 1),
                    }
                )
        # else: machine junk (bare numeric, registry noise) -> drop silently
    if to_queue and emit_unresolved:
        _append_unresolved(unresolved_path, to_queue)
    return out, comps


def _is_bulk_noise_cluster(cl: "Cluster") -> bool:
    """True if the cluster is bulk/unsubscribe/newsletter noise with no durable
    human content — so it should be IGNORED, never routed to customers/prospect.

    The registry `noise` bucket only catches known ESP/free-mail domains; a bulk
    email also carries our own domain's address, so `_registry_domain` returns
    None (keeps "customers") for an unregistered mailing-list domain. This gate
    is the missing fix: it runs a note-level noise detector, when one is
    installed, over a synthetic note built from the cluster's text + its events. Degrades to
    a no-op (returns False) if the detector import fails, so behaviour is never
    worse than today.
    """
    if not cl.events:
        return False  # no events -> nothing to classify (e.g. synthetic test clusters)
    try:
        from distill.noise import is_bulk_noise  # noqa: E402
    except Exception:  # pragma: no cover - detector absent -> keep prior behaviour
        return False
    synthetic = {
        "name": _name_for_cluster(cl),
        "description": _description(cl),
        "body": cl.text,
        "entities": cl.entities,
    }
    events = [_event_as_dict(ev) for ev in cl.events]
    verdict = is_bulk_noise(synthetic, events)
    return bool(verdict.get("noise")) and verdict.get("confidence", 0.0) >= 0.85


def _cluster_key(ev: Event, entities: list[str]) -> str:
    """Deterministic clustering key for one event.

    Group related events by, in priority order:
      (a) provenance THREAD / repo / file (one PR thread, one Slack thread),
      (b) a shared resolved ENTITY (all Acme signal on a day coalesces),
      (c) same DAY + same SOURCE (a coarse catch-all so fragments still merge).
    """
    prov = ev.provenance or {}
    day = (ev.occurred_at or "")[:10]

    # (a) explicit thread / conversation anchors
    thread = (
        prov.get("thread_ts")
        or prov.get("thread")
        or prov.get("threadId")
        or prov.get("thread_id")
        or prov.get("ts")
    )
    if ev.source == "slack" and (prov.get("channel_id") or prov.get("channel")):
        chan = prov.get("channel_id") or prov.get("channel")
        return f"slack:{chan}:{thread or day}"
    if ev.source == "gmail" and thread:
        return f"gmail:{thread}"
    if ev.source == "crm" and (prov.get("objectId") or prov.get("recordId")):
        object_id = prov.get("objectId") or prov.get("recordId")
        return f"crm:{prov.get('object') or 'record'}:{object_id}"
    if ev.source == "upload":
        upload_id = prov.get("upload_id") or prov.get("uploadId") or ev.id
        return f"upload:{upload_id}"
    if prov.get("repo"):
        # cluster a repo's commits for the day together (one piece of work)
        return f"repo:{prov['repo']}:{day}"
    if prov.get("fileId") or prov.get("file_id"):
        return f"drive:{prov.get('fileId') or prov.get('file_id')}"
    # Calendar objects are mutable snapshots.  Their stable source identity must
    # outrank an attendee/entity grouping: combining several same-party meetings
    # into one cluster makes a correction lineage bind to multiple existing notes
    # and either mutates the wrong one or fails the whole production digest.
    if ev.source == "gcal" and (
        prov.get("iCalUID") or prov.get("eventId") or prov.get("event_id")
    ):
        event_id = prov.get("iCalUID") or prov.get("eventId") or prov.get("event_id")
        occurrence = prov.get("occurrence") or ""
        return f"gcal:{event_id}:{occurrence}"

    # (b) shared entity
    if entities:
        return f"entity:{entities[0]}:{day}"

    # (c) same day + same source
    return f"{ev.source}:{day}"


def _mutable_source_lineage(ev: Event) -> str:
    """Return a stable mutable-object key, or ``""`` for immutable/open sources."""
    provenance = ev.provenance or {}
    if ev.source == "slack":
        channel = provenance.get("channel_id") or provenance.get("channel")
        thread = provenance.get("thread_ts") or provenance.get("ts")
        if channel and thread:
            return f"slack:{channel}:{thread}"
    if ev.source == "gmail":
        thread = provenance.get("threadId") or provenance.get("thread_id")
        if thread:
            return f"gmail:{thread}"
    if ev.source == "gdrive":
        file_id = provenance.get("fileId") or provenance.get("file_id")
        if file_id:
            return f"gdrive:{file_id}"
    if ev.source == "crm":
        object_id = provenance.get("objectId") or provenance.get("recordId")
        if object_id:
            return f"crm:{provenance.get('object') or 'record'}:{object_id}"
    if ev.source == "gcal":
        event_id = (
            provenance.get("iCalUID")
            or provenance.get("eventId")
            or provenance.get("event_id")
        )
        occurrence = provenance.get("occurrence") or ""
        if event_id:
            return f"gcal:{event_id}:{occurrence}"
    return ""


# CRM emits activity deltas rather than a closed object snapshot; collapsing
# those rows would erase earlier stage/summary changes.  Only connectors with a
# complete replaceable snapshot contract belong here.
_SNAPSHOT_LINEAGE_SOURCES = frozenset({"slack", "gdrive", "gcal"})


def _latest_mutable_snapshots(events: list[Event]) -> list[Event]:
    """Keep only the latest complete snapshot per mutable source object.

    Gmail thread events are individual messages and therefore cumulative; they
    deliberately stay outside this filter.  Slack/GDrive/GCal revisions are
    complete current-object snapshots. Feeding an old and a corrected snapshot
    into one rebuild cluster can resurrect the old description or make the
    verifier see mutually exclusive sources. Calendar is special: a stale
    attendee copy may be ingested after a newer cancellation, so a valid closed
    contract ranks by source ``updated`` time before observation time. Drive's
    closed inventory contract similarly ranks modification/deletion/restoration
    source time, so a replayed old listing cannot revive a removed file. Slack
    v3 is similarly the sole authority-bearing snapshot contract: legacy rows
    remain readable as a compatibility fallback but cannot outrank a valid v3
    revision.
    """
    from remembrancerlib.event import _parse_iso_timestamp

    selected: dict[str, tuple[tuple[int, datetime, datetime, int], int, Event]] = {}
    passthrough: list[tuple[int, Event]] = []
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    for index, event in enumerate(events):
        lineage = _mutable_source_lineage(event)
        if not lineage or event.source not in _SNAPSHOT_LINEAGE_SOURCES:
            passthrough.append((index, event))
            continue
        observed = _parse_iso_timestamp(getattr(event, "ingested_at", None)) or epoch
        if event.source in {"gcal", "gdrive"}:
            try:
                if event.source == "gcal":
                    from ingest.gcal_contract import gcal_source_updated

                    source_updated = gcal_source_updated(event)
                else:
                    from ingest.gdrive_contract import gdrive_source_time

                    source_updated = gdrive_source_time(event)
            except (ImportError, TypeError, ValueError):
                source_updated = None
            rank = (
                2 if source_updated is not None else 1,
                source_updated or epoch,
                observed,
                index,
            )
        else:
            try:
                from ingest.slack_contract_v3 import thread_v3_event_valid
                from ingest.slack_contract import thread_v2_event_valid

                deleted = (event.payload or {}).get("deleted") is True
                if thread_v3_event_valid(event):
                    # Slack has no undelete operation.  Once the closed v3
                    # contract observes deletion, a late/stale live replay must
                    # never revive that source object.
                    contract_rank = 3 if deleted else 2
                elif thread_v2_event_valid(event):
                    # V2 remains a readable fallback, but its deletion bit has
                    # no authority and cannot suppress a legacy live snapshot.
                    contract_rank = -1 if deleted else 1
                else:
                    contract_rank = -2 if deleted else 0
            except (ImportError, AttributeError, TypeError, ValueError):
                contract_rank = -2
            rank = (contract_rank, epoch, observed, index)
        prior = selected.get(lineage)
        if prior is None or rank > prior[0]:
            selected[lineage] = (rank, index, event)
    kept = passthrough + [(index, event) for _rank, index, event in selected.values()]
    return [event for _index, event in sorted(kept, key=lambda pair: pair[0])]


def cluster_mutable_events(
    events: list[Event],
    resolver: EntityResolver,
    *,
    lineage_events: "Iterable[Event] | None" = None,
    canonicalizer: "Canonicalizer | None" = None,
    unresolved_path: str | None = None,
    component_resolver: "ComponentResolver | None" = None,
    emit_unresolved: bool = True,
) -> tuple[list[Cluster], int]:
    """Cluster current snapshots and bind them to their complete L1 lineage.

    This is the production clustering door for both the incremental engine and
    the resumable digest.  Mutable connectors append corrected snapshots under
    new event IDs, so semantic similarity alone is not a safe idempotency key:
    every resulting cluster carries *all* event IDs for the same source object.
    The write path can then force a correction onto the note that already cites
    an older revision, even when the corrected wording is completely different.

    ``lineage_events`` should be the authenticated full L1 prefix.  It defaults
    to the supplied event batch for focused callers.  The returned integer is
    the number of older complete snapshots removed before clustering.
    """
    current_events = _latest_mutable_snapshots(list(events))
    history = current_events if lineage_events is None else list(lineage_events)
    terminal_slack_lineages = {
        _mutable_source_lineage(event)
        for event in history
        if _is_slack_tombstone(event)
    }
    if terminal_slack_lineages:
        current_events = [
            event
            for event in current_events
            if not (
                event.source == "slack"
                and not _is_slack_tombstone(event)
                and _mutable_source_lineage(event) in terminal_slack_lineages
            )
        ]
    clusters = cluster_events(
        current_events,
        resolver,
        canonicalizer=canonicalizer,
        unresolved_path=unresolved_path,
        component_resolver=component_resolver,
        emit_unresolved=emit_unresolved,
    )
    requested_lineages = {
        lineage
        for event in current_events
        if (lineage := _mutable_source_lineage(event))
    }
    if requested_lineages:
        lineage_ids: dict[str, list[str]] = {
            lineage: [] for lineage in requested_lineages
        }
        for event in history:
            lineage = _mutable_source_lineage(event)
            if lineage in lineage_ids and event.id not in lineage_ids[lineage]:
                lineage_ids[lineage].append(event.id)
        for cluster in clusters:
            lineages = {
                lineage
                for event in cluster.events
                if (lineage := _mutable_source_lineage(event))
            }
            cluster.source_lineages = sorted(lineages)
            cluster.lineage_provenance_ids = [
                event_id
                for lineage in sorted(lineages)
                for event_id in lineage_ids.get(lineage, [])
            ]
    return clusters, len(events) - len(current_events)


def _is_slack_tombstone(event: Event) -> bool:
    """Whether an authority-bearing v3 snapshot records source disappearance."""
    try:
        from ingest.slack_contract_v3 import thread_v3_event_valid
    except (ImportError, AttributeError):
        return False
    return thread_v3_event_valid(event) and (event.payload or {}).get("deleted") is True


def _is_slack_deletion_claim(event: Event) -> bool:
    """Whether a Slack row claims deletion, regardless of its authority version."""
    return (
        event.source == "slack"
        and isinstance(event.payload, dict)
        and event.payload.get("deleted") is True
    )


def _is_gdrive_tombstone(event: Event) -> bool:
    """Whether a complete Drive inventory proves source disappearance."""
    try:
        from ingest.gdrive_contract import gdrive_event_valid
    except (ImportError, AttributeError):
        return False
    return gdrive_event_valid(event) and (event.payload or {}).get("deleted") is True


def _filed_current_note(
    cluster: Cluster,
    notes: list[Note],
    *,
    current_names: set[str] | None = None,
) -> Note | None:
    """Return a current note already covering this exact snapshot, replay-only.

    Coverage proves that the current event was filed; it does not confer
    mutable-source ownership.  This keeps old aggregate citations from
    becoming write authority when a digest ledger is lost.
    """
    provenance = set(cluster.provenance_ids)
    if not provenance:
        return None
    current = (
        current_names if current_names is not None else _current_distill_names(notes)
    )
    holders = sorted(
        (
            note
            for note in notes
            if note.name in current and provenance.issubset(set(note.provenance or []))
        ),
        key=lambda note: note.name,
    )
    return holders[0] if holders else None


def _lineage_bound_notes(
    cluster: Cluster,
    notes: list[Note],
    *,
    current_names: set[str] | None = None,
) -> list[Note]:
    """Return only explicit mutable-source owners; legacy citations need review."""
    lineage_ids = set(cluster.lineage_provenance_ids)
    source_lineages = set(cluster.source_lineages)
    if not lineage_ids and not source_lineages:
        return []
    if (
        _filed_current_note(
            cluster,
            notes,
            current_names=current_names,
        )
        is not None
    ):
        return []
    current = (
        current_names if current_names is not None else _current_distill_names(notes)
    )
    eligible = [
        note for note in notes if note.name in current and _is_distill_target(note)
    ]
    explicit = [
        note
        for note in eligible
        if source_lineages.intersection(getattr(note, "source_lineages", None) or [])
    ]
    if len(explicit) > 1:
        raise MutableLineageConflict(cluster.key, [note.name for note in explicit])
    if explicit:
        bound = explicit
    else:
        legacy = [
            note
            for note in eligible
            if note.name in current
            and not (getattr(note, "source_lineages", None) or [])
            and lineage_ids.intersection(note.provenance or [])
        ]
        # Ambiguity is what needs a human, and ambiguity means MORE THAN ONE
        # current citer.  Refusing a *sole* citer was unsatisfiable rather than
        # strict: a note acquires `source_lineages` only inside `_apply_update`,
        # which runs after this binding succeeds, so no production code path
        # could ever clear the requirement and the object's later revisions were
        # dropped for good.  With one current distill target citing the lineage and no
        # explicit owner anywhere, there is no other note the revision could
        # belong to; binding it here makes the ownership explicit on the next
        # write, after which the ordinary `explicit` branch takes over.
        if len(legacy) > 1:
            raise MutableLineageOwnershipRequired(
                cluster.key,
                [note.name for note in legacy],
            )
        bound = legacy
    if bound and not _acl_compatible(bound[0].acl, cluster.acl):
        raise ACLConflict(
            f"mutable source lineage {cluster.key!r} changed to an incompatible ACL"
        )
    return bound


def _lineage_confined(note: Note, lineage_event_ids: set) -> bool:
    """Whether every event this note cites belongs to one mutable source object.

    Both source-deletion handlers promise to retire "one atomic note whose sole
    mutable lineage is this file".  The frozen core enforces half of that
    (`_validate_owned_note` requires `source_lineages == [lineage]`); the other
    half — that the note is not *also* carrying unrelated material — used to
    hold only because explicit ownership was minted exclusively by CREATE of a
    fresh atomic note.  Now that a sole legacy citer can be granted ownership,
    state the invariant instead of inheriting it, or one deleted thread could
    retire a page that had absorbed a dozen other sources.

    Every explicit owner today is confined, so
    this refuses nothing that is retired today.
    """
    return set(str(item) for item in (note.provenance or [])).issubset(
        lineage_event_ids
    )


def _partition_incompatible_acl_cluster(cluster: Cluster) -> list[Cluster]:
    """Split a subject cluster when no OR-grant ACL can contain all events.

    This is intentionally rare, but
    it is a security boundary: relabeling an ``eng`` + ``customers`` thread as
    ``secrets`` invents a reader group rather than enforcing both grants.
    Public events may safely join a restricted partition and are assigned once.
    """
    groups: list[list[Event]] = []
    for event in cluster.events:
        for group in groups:
            try:
                combine_acl(*(item.acl for item in group), event.acl)
            except ACLConflict:
                continue
            group.append(event)
            break
        else:
            groups.append([event])
    if len(groups) <= 1:
        return [cluster]
    out: list[Cluster] = []
    for group in groups:
        material = "\0".join(sorted(event.id for event in group))
        suffix = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
        out.append(Cluster(key=f"{cluster.key}:acl-{suffix}", events=group))
    return out


def cluster_events(
    events: list[Event],
    resolver: EntityResolver,
    canonicalizer: "Canonicalizer | None" = None,
    unresolved_path: str | None = None,
    component_resolver: "ComponentResolver | None" = None,
    emit_unresolved: bool = True,
) -> list[Cluster]:
    """Group events into clusters and union resolved entities onto each.

    Two-pass so that entity sharing can merge clusters that the first-pass key
    split: after the primary keying, any clusters sharing a customer entity on
    the same day are merged into the entity cluster.

    `canonicalizer`/`unresolved_path`/`component_resolver` default to the
    module globals (`_REGISTRY_TYPES_PATH` / `_ENTITY_UNRESOLVED_PATH` /
    `_ENG_COMPONENTS_PATH`+`_INITIATIVES_PATH`), read at CALL time so `run()`
    and tests can point them at a per-root registry/state file.
    """
    if canonicalizer is None:
        canonicalizer = _canonicalizer()
    if unresolved_path is None:
        unresolved_path = _ENTITY_UNRESOLVED_PATH
    if component_resolver is None:
        component_resolver = _component_resolver()

    # Pass 1: primary key.
    by_key: dict[str, Cluster] = {}
    ev_entities: dict[str, list[str]] = {}
    for ev in events:
        ents = resolver.resolve(
            text=_event_text(ev), author=ev.author, provenance=ev.provenance
        )
        # honor any entities a connector already pre-seeded
        for e in ev.entities:
            if e not in ents:
                ents.append(e)
        ev_entities[ev.id] = ents
        key = _cluster_key(ev, ents)
        cl = by_key.setdefault(key, Cluster(key=key))
        cl.events.append(ev)

    clusters = [
        partition
        for cluster in by_key.values()
        for partition in _partition_incompatible_acl_cluster(cluster)
    ]

    # Union RAW entities onto each cluster (first-seen order), then canonicalize
    # the whole facet in one pass so ordering, the self/company exception, and
    # unresolved-queue dedup all see the cluster as a unit.
    queued_this_run: set[str] = set()
    for cl in clusters:
        raw: list[str] = []
        first_day: dict[str, str] = {}
        counts: dict[str, int] = defaultdict(int)
        for ev in cl.events:
            day = (ev.occurred_at or "")[:10]
            for e in ev_entities.get(ev.id, []):
                raw.append(e)
                first_day.setdefault(e, day)
                counts[e] += 1
        cl.entities, cl.components = _canonicalize_cluster_entities(
            raw,
            cl.domain,
            canonicalizer,
            unresolved_path,
            queued_this_run,
            first_day,
            counts,
            component_resolver,
            emit_unresolved,
        )

    return sorted(clusters, key=lambda c: c.key)


# ---------------------------------------------------------------------------
# Note matching (UPDATE/LINK detection)
# ---------------------------------------------------------------------------


def _name_for_cluster(cl: Cluster) -> str:
    kws = _keywords(cl.text, k=4)
    if cl.entities:
        ent = cl.entities[0].split(":", 1)[-1]
        base = ent + "-" + "-".join(kws[:3]) if kws else ent + "-activity"
    else:
        base = "-".join(kws[:4]) if kws else (cl.domain + "-activity")
    return slugify(base)


def _similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Deterministic projections are rebuilt by their named producer. They are
# BM25-bait for the source facts they summarize, so the generic distiller must
# never pick one as UPDATE/LINK/SUPERSEDE target: the next producer refresh
# would discard the generic event evidence after its digest row is complete.
# Legacy dossier/initiative pages are also excluded by type until ownership
# migration can prove and stamp their producer. CREATE is unaffected: a durable
# cluster still gets its own atomic contributing note.
_NON_DISTILL_TARGET_TYPES = ("entity", "initiative", "correction")


def _is_generated_projection(note: Note) -> bool:
    return any(
        str(item).startswith("generated-by:") for item in (note.provenance or [])
    )


def _is_distill_target(note: Note) -> bool:
    return note.type not in _NON_DISTILL_TARGET_TYPES and not _is_generated_projection(
        note
    )


def _current_distill_names(notes) -> set[str]:
    """Names eligible for mutation/linking in the global write-time view."""
    return current_note_names(list(notes))


def _match_existing(cl: Cluster, notes: list[Note]) -> tuple[Note | None, list[Note]]:
    """Return (best UPDATE candidate or None, list of LINK-related notes).

    A note is an UPDATE match if it shares any entity with the cluster AND its
    description/name keywords overlap strongly (Jaccard >= 0.5) OR it shares a
    provenance id with the cluster (same source thread already distilled).
    Notes that overlap weakly (>= 0.2, or >= 0.1 when they share an entity)
    become LINK candidates.

    A third tier — same primary entity at the same
    primary position AND same domain AND a shared project namespace AND content
    overlap >= 0.15 — also counts as an UPDATE candidate. This folds
    same-project, same-owner events (e.g. billing-revamp shards about different
    sub-decisions) into one note instead of proliferating per-decision shards.

    Dossier/initiative pages are excluded from consideration entirely (see
    `_NON_DISTILL_TARGET_TYPES`) — neither an UPDATE/SUPERSEDE target nor a LINK
    partner may ever be a `type: entity` / `type: initiative` page.
    """
    bound = _lineage_bound_notes(cl, notes)
    if bound:
        return bound[0], []
    current = _current_distill_names(notes)
    notes = [
        n
        for n in notes
        if n.name in current
        and _is_distill_target(n)
        and _acl_compatible(n.acl, cl.acl)
    ]
    cl_kws = set(_keywords(cl.text, k=10))
    cl_prov = set(cl.provenance_ids)
    cl_ents = set(cl.entities)
    cl_primary = cl.entities[:1]
    cl_domain = getattr(cl, "domain", None)
    # Cluster's predicted name (used to extract project namespace for the third
    # tier). LLM-path notes have semantic names like `ml-billing-revamp-...`;
    # heuristic-path names follow `entity-keywords[:3]` like `jane-doe-billing-
    # revamp-regression`. Both feed the same project-namespace heuristic.
    cl_pred_name = _name_for_cluster(cl)
    cl_project_ns = _project_namespace(cl_pred_name)

    best: Note | None = None
    best_score = 0.0
    related: list[Note] = []
    for n in notes:
        n_kws = set(_keywords(n.name + " " + n.description, k=12))
        sim = _similarity(cl_kws, n_kws)
        shares_prov = bool(cl_prov & set(n.provenance))
        shares_ent = bool(cl_ents & set(n.entities))
        same_owner_primary = bool(cl_primary) and n.entities[:1] == cl_primary
        same_domain = cl_domain is not None and n.domain == cl_domain
        # Project-namespace match: the cluster's predicted project tokens must
        # appear in the candidate note's name tokens. This is the semantic
        # "same project" signal — replaces the looser "any content overlap"
        # which could match two unrelated ml projects for the same owner.
        n_project_ns = _project_namespace(n.name)
        shares_project = bool(cl_project_ns & n_project_ns) if cl_project_ns else False

        if shares_prov or (shares_ent and sim >= 0.5):
            if sim + (0.5 if shares_prov else 0.0) > best_score:
                best_score = sim + (0.5 if shares_prov else 0.0)
                best = n
        elif same_owner_primary and same_domain and shares_project and sim >= 0.15:
            # Same primary owner + same domain + same project namespace + any
            # content overlap: update the existing note rather than CREATE a
            # new shard. Project-namespace match is the semantic gate that
            # prevents unrelated projects (e.g. billing-revamp matching a
            # face-detection note owned by the same person) from colliding.
            score = sim + 0.3
            if score > best_score:
                best_score = score
                best = n
        elif sim >= 0.2 or (shares_ent and sim >= 0.1):
            related.append(n)
    return best, related


def _project_namespace(name: str) -> set[str]:
    """Extract the project namespace tokens from a note slug.

    The first 3 hyphen-separated segments of a slug encode the project (e.g.
    `ml-billing-revamp-v4-dataset-quality-issues` -> `billing`, `revamp`).
    Domain slugs (`ml`, `eng`) are < 3 chars and dropped — they're a routing
    prefix, not a content signal; the existing _keywords() regex
    `[a-zA-Z][a-zA-Z0-9_-]{2,}` already filters them out of content matching,
    so we align here for consistency. Heuristic-path slugs follow
    `entity-keywords[:3]` (e.g. `jane-doe-billing-revamp-...`) which still
    includes `billing`/`revamp` in positions 2-3. Empty/short names return
    empty set (no project signal -> tier-3 cannot fire).
    """
    if not name:
        return set()
    parts = name.split("-")[:3]
    return {p for p in parts if len(p) >= 3 and not p.isdigit()}


# ---------------------------------------------------------------------------
# Layer-3 project inference: the LLM proposes a new project namespace from cluster
# content. Opt-in via config distill.auto_create_project (default OFF) because
# LLM calls are slow and quota-costly; gated so the distiller doesn't create
# folders at LLM call rate.
# ---------------------------------------------------------------------------
# Examples in the tool description steer the slugs the model proposes.
_PROJECT_SLUG_EXAMPLES = "'billing-revamp', 'search-revamp'"
PROJECT_PROPOSAL_TOOL = {
    "name": "propose_projects",
    "description": "Propose 1-3 slug-safe project namespace candidates for this cluster.",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"1-3 slug-safe project namespace candidates (e.g., {_PROJECT_SLUG_EXAMPLES}). Each ≤3 hyphen-separated lowercase-alphanumeric segments, slugified.",
            },
            "reasoning": {
                "type": "string",
                "description": "Why these candidates fit the cluster content. Must be grounded in the cited events.",
            },
        },
        "required": ["candidates", "reasoning"],
    },
}


def _propose_project_via_llm(
    cl: Cluster, dom: str, existing_names: set[str], client
) -> str:
    """Layer 3: ask the LLM to propose a new project namespace for this cluster.

    Returns the first valid candidate that doesn't collide with existing
    project names in this domain. Returns "" when the LLM declines, no candidate
    is valid, or the LLM is unavailable. Safe-by-default: every candidate
    passes through `_normalize_project_slug` and `_looks_like_project`
    before being returned, so the LLM can't fabricate a non-slug-safe name.
    """
    if client is None or not getattr(client, "available", lambda: False)():
        return ""
    event_lines = []
    for ev in (cl.events or [])[:6]:
        text = _event_text(ev)[:200]
        event_lines.append(f"- {ev.id} ({ev.source}): {text}")
    prompt = (
        "Propose 1-3 slug-safe project namespace candidates for the cluster "
        "below. Each candidate must be 1-3 hyphen-separated lowercase-"
        f"alphanumeric segments (e.g., {_PROJECT_SLUG_EXAMPLES}, "
        "'kernel-security'). Ground your reasoning in the cited events. "
        "Do NOT invent projects unrelated to the events. Avoid duplicating "
        "existing project names in this domain.\n\n"
        f"DOMAIN: {dom}\n"
        f"EXISTING PROJECTS IN THIS DOMAIN (do not duplicate): "
        f"{sorted(existing_names) or '(none)'}\n\n"
        "EVENTS:\n" + "\n".join(event_lines)
    )
    try:
        out = client.decide(prompt, PROJECT_PROPOSAL_TOOL, max_tokens=400)
    except Exception:
        return ""
    cands = out.get("candidates") or []
    for c in cands:
        ns = _normalize_project_slug(c)
        if ns and ns not in existing_names and _looks_like_project(ns):
            return ns
    return ""


def _normalize_project_slug(s: str) -> str:
    """Coerce an LLM-proposed name into a valid project namespace.

    Rules: 1-3 hyphen-separated lowercase-alphanumeric segments, each 2-30
    chars. Whitespace and special characters collapse to hyphens. Returns ""
    when input is empty, has no usable segments, or is too long.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s).strip("-")
    if not s:
        return ""
    parts = [p for p in s.split("-") if p]
    if not parts or len(parts) > 3:
        return ""
    if not all(2 <= len(p) <= 30 for p in parts):
        return ""
    return "-".join(parts)


def _looks_like_project(s: str) -> bool:
    """Sanity check on a proposed namespace.

    A single common word like 'model' or 'data' is too generic to be a
    project; require either 2+ segments (multi-word) or one segment of
    length ≥ 5 (specific enough).
    """
    parts = s.split("-")
    if len(parts) >= 2:
        return True
    return len(parts[0]) >= 5


# ---------------------------------------------------------------------------
# Note construction
# ---------------------------------------------------------------------------


def _note_type(cl: Cluster) -> str:
    txt = cl.text.lower()
    if re.search(
        r"\b(decid|decision|chose|agreed|adopt|deprecat|policy|standard)\b", txt
    ):
        return "decision"
    if re.search(r"\b(state|status|current|now|baseline|version)\b", txt):
        return "state"
    if re.search(r"\b(doc|reference|guide|how[- ]?to|spec)\b", txt):
        return "reference"
    return "project"


def _description(cl: Cluster) -> str:
    """Return one bounded retrieval line, never an embedded document dump."""
    durable = [ev for ev in cl.events if _is_durable(ev)]
    src = durable[0] if durable else cl.events[0]
    payload = src.payload if isinstance(src.payload, dict) else {}
    line = ""
    for field_name in ("subject", "title", "name", "summary"):
        candidate = payload.get(field_name)
        if candidate is not None and str(candidate).strip():
            heading = concise_description(candidate)
            substantive = re.sub(
                r"^(?:(?:re|fw|fwd)\s*:\s*)+",
                "",
                heading,
                flags=re.IGNORECASE,
            )
            if sum(character.isalnum() for character in substantive) >= 8:
                line = heading
                break
    line = concise_description(
        line or _event_text(src), fallback=f"{cl.domain} activity"
    )
    if cl.entities:
        names = ", ".join(e.split(":", 1)[-1] for e in cl.entities)
        if names.lower() not in line.lower():
            line = f"{line} ({names})"
    return concise_description(line, fallback=f"{cl.domain} activity")


def _body(
    cl: Cluster, description: str, links: list[str], valid_names: set[str] | None = None
) -> str:
    lines = [description, ""]
    lines.append("Distilled from:")
    for ev in cl.events:
        t = _event_text(ev)
        t = (t[:100] + "…") if len(t) > 100 else t
        lines.append(f"- `{ev.id}` ({ev.source}) {t}".rstrip())
    if links:
        if valid_names is not None:
            links = [link for link in links if link in valid_names]
        if links:
            lines.append("")
            lines.append("Related: " + " ".join(f"[[{link}]]" for link in links))
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Per-cluster action
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    action: str  # CREATE | UPDATE | LINK | IGNORE
    cluster_key: str
    note_name: str = ""
    detail: str = ""


# ---------------------------------------------------------------------------
# Meeting enrichment (meeting → company-context)
# ---------------------------------------------------------------------------
#
# A `meeting` event carries an already-distilled brief (summary, action_items,
# decisions) + attendee emails + a transcript deep-link. It does NOT flow the
# generic clustering path — it gets a dedicated, deterministic handler so:
#   * the note name is STABLE (`meeting-<date>-<slug>`) → replay UPDATEs, never
#     duplicates (idempotent on the `bluedot:<source_id>` event id);
#   * the SIGNAL GATE (≥1 non-internal attendee resolves to a registry entity)
#     decides plain-note vs. enriched-note, so an internal all-hands never spams
#     dossiers;
#   * decisions/action_items land as a first-class `structured` block WITHOUT any
#     new LLM call (they ride in the event payload already).

_MEETING_NOTE_TYPE = "source-summary"


# Leading `word:` segments that are task keywords, not owners.
_ACTION_ITEM_NONOWNERS = {
    "todo",
    "action",
    "action item",
    "action items",
    "next",
    "next step",
    "next steps",
    "note",
    "notes",
    "fyi",
    "decision",
    "decisions",
    "reminder",
    "deadline",
    "due",
    "task",
    "tasks",
    "goal",
    "goals",
    "update",
    "follow up",
    "followup",
    "follow-up",
    "ask",
    "asks",
    "blocker",
    "blockers",
    "risk",
    "risks",
}


def _parse_action_item(text: str) -> dict[str, Any]:
    """One action-item string → `{text, owner, due, open}`. Owner/due are parsed
    ONLY when unambiguously present ("Jane: ..." → owner "Jane"; an ISO date →
    due); otherwise "" (never guessed). `open` is always True (a fresh todo)."""
    text = str(text).strip()
    owner = ""
    m = re.match(r"^([A-Za-z][\w.\-]*(?:\s+[A-Za-z][\w.\-]*){0,2})\s*:\s+\S", text)
    if m:
        cand = m.group(1).strip()
        # Drop task-keyword prefixes ("TODO:", "Follow up:", "Next step:") that a
        # bare `word:` regex would misread as an owner — never guess a non-name.
        if cand.lower() not in _ACTION_ITEM_NONOWNERS:
            owner = cand
    due = ""
    dm = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if dm:
        due = dm.group(1)
    return {"text": text, "owner": owner, "due": due, "open": True}


def _meeting_note_name(parsed: dict[str, Any], ev: Event) -> str:
    """Stable, deterministic note name `meeting-<date>-<slug>` (design)."""
    date = str(parsed.get("occurred_at") or ev.occurred_at or "")[:10]
    label = parsed.get("title") or parsed.get("source_id") or ev.id
    parts = ["meeting"]
    if date:
        parts.append(date)
    tail = slugify(str(label))
    if tail:
        parts.append(tail)
    return slugify("-".join(parts))


def _meeting_body(parsed: dict[str, Any], wikilinks: list[str]) -> str:
    lines: list[str] = []
    summary = (parsed.get("summary") or "").strip()
    if summary:
        lines += [summary, ""]
    decisions = [str(d).strip() for d in parsed.get("decisions", []) if str(d).strip()]
    if decisions:
        lines.append("## Decisions")
        lines += [f"- {d}" for d in decisions]
        lines.append("")
    action_items = [
        str(a).strip() for a in parsed.get("action_items", []) if str(a).strip()
    ]
    if action_items:
        lines.append("## Action items")
        lines += [f"- {a}" for a in action_items]
        lines.append("")
    if wikilinks:
        lines.append("Related: " + " ".join(f"[[{link}]]" for link in wikilinks))
    return "\n".join(lines).rstrip() + "\n"


def _meeting_wikilinks(
    entities: list[str], notes_by_name: dict[str, Note], self_name: str
) -> list[str]:
    """Existing deal/customer notes that already carry a resolved entity — the
    counterparty's note to [[link]] from this meeting. Dossier/initiative pages
    are excluded (they are regenerated, not distill targets). Deterministic
    (sorted), capped so a note body never explodes."""
    ents = set(entities)
    current = _current_distill_names(notes_by_name.values())
    out = []
    for name in sorted(notes_by_name):
        if name == self_name or name not in current:
            continue
        n = notes_by_name[name]
        if n.type in _NON_DISTILL_TARGET_TYPES:
            continue
        if ents & set(n.entities or []):
            out.append(name)
    return out[:5]


def _meeting_note_acl(existing_acl: list[str], event_acl: list[str]) -> list[str]:
    """ACL of a meeting note after one more revision of its meeting.

    ``_safe_acl`` refuses scopes no OR-grant label can hold, and in general
    that is right: relabelling ``eng`` + ``customers`` as ``secrets`` would
    invent a reader group.  One pair is expected here and has a fail-closed
    answer.  A meeting whose brief or excerpt is sensitive is labelled
    ``secrets`` at ingest (``ingest.meeting_events.sensitive_acl``), while
    other revisions of the same meeting are not: one emitted ``eng`` before
    that check existed, or a corrected brief without the sensitive words.
    The note then stays, or becomes, ``secrets``: sticky, as the Slack
    connector keeps a whole thread ``secrets`` once any revision is,
    but here only for the note; a corrected brief's L1 event is still
    emitted ``eng``.  Meetings are re-scanned every pass, so both orders
    occur.  Refusing would leave the ``eng`` note serving the sensitive brief:
    the nightly digest parks the event as a handler error for
    review, and ``run`` aborts the whole pass.  Every other incompatible pair
    still raises ``ACLConflict``.

    Safe only because every identity holding ``secrets`` holds every label
    (a property of ``config.json``'s grants).  Narrowing works only
    for a note with no wiki-links either way: ``write_note`` refuses a
    ``secrets`` note that links to an ``eng`` note, or that an ``eng`` note
    links to, with ``ValueError``.  That write fails closed and the old note
    keeps its bytes; a linked note needs a governed correction.
    """
    try:
        return _safe_acl(existing_acl, event_acl)
    except ACLConflict:
        if "secrets" in (existing_acl or []) or "secrets" in (event_acl or []):
            return ["secrets"]
        raise


def _distill_meeting(
    ev: Event,
    resolver: EntityResolver,
    notes_by_name: dict[str, Note],
    context_dir: str,
    indexes: "Any | None" = None,
) -> Decision:
    """Dedicated handler for one meeting event (SIGNAL GATE → plain | enrich).

    Idempotent: keyed on the stable note name, so a replay (or --rebuild)
    UPDATEs the same note rather than creating a duplicate, and a human-edited
    body is preserved (only machine-owned metadata + provenance are refreshed).
    """
    parsed = parse_meeting_event(ev)
    base_name = _meeting_note_name(parsed, ev)
    current = _current_distill_names(notes_by_name.values())
    # Physical-event provenance is the replay boundary.  Once an immutable
    # event has been filed, a later sweep must not mutate a historical note or
    # mint a duplicate merely because a verified correction now supersedes it.
    filed = [
        note for note in notes_by_name.values() if ev.id in (note.provenance or [])
    ]
    if filed:
        holder = sorted(filed, key=lambda note: note.name)[0]
        if holder.name not in current:
            return Decision(
                "IGNORE",
                f"meeting:{parsed.get('source_id') or ev.id}",
                note_name=holder.name,
                detail="replay: event already filed in history",
            )
        name = holder.name
    elif base_name in current or base_name not in notes_by_name:
        name = base_name
    else:
        # The stable base is historical.  Continue the source-owned stream in
        # a separate current revision; never rewrite the replaced bytes.
        revisions = sorted(
            note.name
            for note in notes_by_name.values()
            if note.name in current
            and note.type == _MEETING_NOTE_TYPE
            and note.name.startswith(base_name + "-revision")
        )
        if revisions:
            name = revisions[-1]
        else:
            name = base_name + "-revision"
            i = 2
            while name in notes_by_name:
                name = f"{base_name}-revision-{i}"
                i += 1
    resource = parsed.get("transcript_ref") or parsed.get("source_url") or ""
    acl = list(ev.acl) or ["eng"]
    description = (parsed.get("title") or parsed.get("summary") or "Meeting").strip()[
        :160
    ]

    # SIGNAL GATE: reconcile non-internal attendees → registry entities via the
    # DEEP resolver: traverse email→person→works_at org→type,
    # and queue ambiguous attendees (link-email / map-domain / merge-org …) for
    # human review. Attendees are bare emails here (name unknown → "").
    pairs = [("", e) for e in parsed.get("attendees", [])]
    # JOIN (meeting-chain reconciler): the CALENDAR is the
    # authoritative attendee source — Bluedot frequently captures only the host, so
    # external invitees never reach resolution. Union the matched gcal event's full
    # invitee list (exact iCalUID when the emit stamped one, else title+time) before
    # resolving. Best-effort: a miss (or reconcile absent) leaves the Bluedot list
    # untouched and never breaks distillation. Registry auto-create for newly-seen
    # domains is handled by the idempotent backfill tool, NOT the nightly hot path.
    if indexes is not None:
        try:
            from distill import reconcile as _rec

            rec, _via, _score = _rec.match_calendar(
                parsed.get("title", ""),
                str(getattr(ev, "occurred_at", "") or ""),
                indexes.gcal,
                parsed.get("icaluid") or None,
                internal_domains=getattr(resolver, "_internal_domains", set()),
            )
            if rec is not None:
                have = {e for _, e in pairs}
                for e in rec.attendees:
                    if e and e not in have:
                        pairs.append(("", e))
                        have.add(e)
        except Exception:
            pass
    result = resolver.resolve_attendees_deep(pairs)
    entities = result["entities"]
    # Ambiguous attendees enrich FUTURE runs only; a queue-write failure is
    # swallowed inside emit_proposals and never breaks distillation.
    emit_proposals(
        result.get("proposals") or [],
        _ENTITY_UNRESOLVED_PATH,
        source_event=ev.id,
        today=_today(),
    )
    if entities:
        structured: dict[str, Any] | None = {
            "kind": "meeting",
            "decisions": [
                str(d).strip() for d in parsed.get("decisions", []) if str(d).strip()
            ],
            "action_items": [
                _parse_action_item(a)
                for a in parsed.get("action_items", [])
                if str(a).strip()
            ],
        }
        wikilinks = [
            link
            for link in _meeting_wikilinks(entities, notes_by_name, name)
            if link in notes_by_name and acl_can_reveal(acl, notes_by_name[link].acl)
        ]
    else:
        # No external registry entity: plain note only (no dossier spam).
        structured = None
        wikilinks = []

    existing = notes_by_name.get(name)
    if existing is not None:
        # Preserve any human-edited body while refreshing machine-owned
        # metadata.  Approved/human-authored bytes are immutable: their
        # refresh is a complete proposed replacement that needs review.
        desired = copy.deepcopy(existing)
        new_acl = _meeting_note_acl(existing.acl, acl)
        before = (
            list(existing.entities),
            list(existing.provenance),
            list(existing.acl),
            existing.resource,
            existing.structured,
            existing.domain,
            existing.type,
            existing.title,
            existing.body,
        )
        for e in entities:
            if e not in desired.entities:
                desired.entities.append(e)
        if ev.id not in desired.provenance:
            desired.provenance.append(ev.id)
        desired.acl = new_acl
        if resource:
            desired.resource = resource
        # Non-destructive: only refresh the structured block when this pass
        # produced one. A later resolve regression (attendee dropped from the
        # registry → structured None) must not wipe a prior enrichment.
        if structured is not None:
            desired.structured = structured
        desired.domain = "business"
        desired.type = _MEETING_NOTE_TYPE
        if parsed.get("title"):
            desired.title = parsed["title"]
        # The links were filtered against the EVENT's ACL; a sticky `secrets`
        # note must re-check them against its own.
        new_links = [
            link for link in wikilinks
            if f"[[{link}]]" not in desired.body
            and acl_can_reveal(new_acl, notes_by_name[link].acl)
        ]
        if new_links:
            desired.body = (
                desired.body.rstrip()
                + "\nRelated: "
                + " ".join(f"[[{link}]]" for link in new_links)
                + "\n"
            )
        after = (
            list(desired.entities),
            list(desired.provenance),
            list(desired.acl),
            desired.resource,
            desired.structured,
            desired.domain,
            desired.type,
            desired.title,
            desired.body,
        )
        if after == before:
            return Decision(
                "IGNORE",
                f"meeting:{parsed.get('source_id') or ev.id}",
                note_name=name,
                detail="replay: no material change",
            )
        desired.updated = _today()
        cluster_key = f"meeting:{parsed.get('source_id') or ev.id}"
        if _requires_replacement_proposal(existing):
            return _propose_replacement(
                existing,
                desired,
                context_dir,
                notes_by_name,
                cluster_key,
                detail="meeting refresh requires review",
            )
        desired.state = "proposed"
        write_note(desired, context_dir)
        notes_by_name[name] = desired
        return Decision(
            "UPDATE",
            f"meeting:{parsed.get('source_id') or ev.id}",
            note_name=name,
            detail=f"entities={entities}",
        )

    note = Note(
        name=name,
        title=parsed.get("title") or "",
        description=description or "Meeting",
        type=_MEETING_NOTE_TYPE,
        domain="business",
        acl=acl,
        entities=list(entities),
        provenance=[ev.id],
        body=_meeting_body(parsed, wikilinks),
        resource=resource,
        structured=structured,
        state="proposed",
    )
    write_note(note, context_dir)
    notes_by_name[name] = note
    return Decision(
        "CREATE",
        f"meeting:{parsed.get('source_id') or ev.id}",
        note_name=name,
        detail=("enriched " if entities else "plain ") + f"entities={entities}",
    )


# Shared with distill.verify, which needs it to recognise these notes as
# deterministic projections. A second copy of the literal is how the Overview
# A second copy of the literal drifts from the module that defines the charts; one
# definition, imported.
GCAL_CANCELLATION_PREFIX = "calendar-cancelled-"
#: These notes are written with ``type="correction"``, which agent
#: ``remembrancer_remember`` writes also use — so the type alone never identifies one.
GCAL_CANCELLATION_NOTE_TYPE = "correction"


def _is_gcal_cancellation(ev: Event) -> bool:
    return ev.source == "gcal" and (ev.payload or {}).get("status") == "cancelled"


def _authoritative_gcal_snapshot(ev: Event) -> bool:
    """Whether ``ev`` satisfies the connector's complete source contract."""
    try:
        from ingest.gcal_contract import gcal_event_valid

        return gcal_event_valid(ev)
    except (ImportError, TypeError, ValueError):
        return False


def _source_owned_gcal_note(note: Note, prior_ids: set[str]) -> bool:
    """Whether an authoritative cancellation may explicitly retire ``note``.

    Provenance alone is not ownership: a human synthesis can cite the same
    event.  Generic distill-owned calendar notes carry the exact ``(gcal)``
    evidence trail emitted by :func:`_body`; human-authored and aggregate note
    types remain immutable and receive the separate correction note instead.
    """
    if note.type not in {"project", "decision", "state", "reference"}:
        return False
    if any(str(item).startswith("by:") for item in (note.provenance or [])):
        return False
    body = note.body or ""
    return any(
        event_id in (note.provenance or []) and f"`{event_id}` (gcal)" in body
        for event_id in prior_ids
    )


def _gcal_cancellation_name(stable: str) -> str:
    """Build a readable, path-safe name that retains full source identity.

    ``slugify`` truncates filenames at 80 characters. Calendar instance ids can
    share a long recurring-series prefix and differ only in the occurrence
    suffix, so prefixing an already-long id used to collapse distinct weeks onto
    one file. Keep room for the historical ``-revision-N`` suffix and bind the
    complete untruncated source id with a deterministic digest.
    """
    prefix = GCAL_CANCELLATION_PREFIX
    readable = slugify(stable)
    candidate = prefix + readable
    max_base_length = 64
    if len(candidate) <= max_base_length:
        return candidate
    digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]
    readable_budget = max_base_length - len(prefix) - len(digest) - 1
    head = readable[:readable_budget].rstrip("-") or "event"
    return f"{prefix}{head}-{digest}"


def _distill_gcal_cancellation(
    ev: Event, notes_by_name: dict[str, Note], all_events: list[Event], context_dir: str
) -> Decision:
    """Create an auditable correction and retire a traceable active meeting note.

    This is an intentional exception to graph-projected supersession: a
    Calendar cancellation is an authoritative source-state transition, so the
    affected meeting is explicitly retired (state plus compatibility flag)
    without waiting for a correction-head review.
    """
    provenance = ev.provenance or {}
    authoritative = _authoritative_gcal_snapshot(ev)
    event_id = provenance.get("eventId")
    uid = provenance.get("iCalUID")
    occurrence = provenance.get("occurrence")
    prior_ids: set[str] = set()
    for candidate in all_events:
        if candidate.source != "gcal" or _is_gcal_cancellation(candidate):
            continue
        cp = candidate.provenance or {}
        same_event = bool(event_id and cp.get("eventId") == event_id)
        same_occurrence = bool(
            uid
            and cp.get("iCalUID") == uid
            and occurrence
            and cp.get("occurrence") == occurrence
        )
        # New non-recurring snapshots use the stable ``single`` sentinel.
        # Migration-era rows used the mutable scheduled start instead.  UID is
        # still unambiguous for that singleton, so retain those citations for a
        # review link without granting them retirement authority.
        same_legacy_singleton = bool(
            authoritative
            and occurrence == "single"
            and uid
            and cp.get("iCalUID") == uid
        )
        if same_event or same_occurrence or same_legacy_singleton:
            prior_ids.add(candidate.id)

    current = _current_distill_names(notes_by_name.values())
    retired: list[str] = []
    review_targets: list[str] = []
    lineage = _mutable_source_lineage(ev)
    current_explicit = [
        note
        for note in notes_by_name.values()
        if note.name in current
        and lineage in (note.source_lineages or [])
        and _is_distill_target(note)
    ]
    if len(current_explicit) > 1:
        raise MutableLineageConflict(
            f"gcal-cancel:{event_id or uid or ev.id}",
            [note.name for note in current_explicit],
        )
    retired_replay = [
        note
        for note in notes_by_name.values()
        if authoritative
        and source_retirement_valid(note)
        and (note.source_retirement or {}).get("lineage") == lineage
        and (note.source_retirement or {}).get("event_id") == ev.id
    ]
    explicit = current_explicit or retired_replay[:1]
    if explicit:
        prior = explicit[0]
        # A deterministic producer cannot change human/legacy sensitive truth.
        # The correction below remains reviewable, while the approved bytes stay
        # untouched until a human chooses to supersede them.
        already_retired = authoritative and source_retirement_valid(prior)
        if already_retired:
            retired.append(prior.name)
        elif authoritative and prior.state not in {"verified", "superseded"}:
            root = os.path.dirname(os.path.abspath(context_dir))
            try:
                capability = _record_source_retirement(
                    prior,
                    root=root,
                    event=ev,
                    lineage=lineage,
                    producer="gcal-cancellation-v1",
                )
            except (PermissionError, ValueError):
                # A migrated owner may cite a legacy source row.  The new
                # cancellation remains review evidence, but without an earlier
                # closed-contract snapshot it cannot retire that memory.
                review_targets.append(prior.name)
            else:
                write_note(
                    prior,
                    context_dir,
                    _retirement_capability=capability,
                )
                retired.append(prior.name)
        else:
            review_targets.append(prior.name)

    # A pre-contract distiller-owned note can be identified well enough to put
    # it in the attended correction proposal, but never well enough to mutate
    # it.  This also bridges the historical start-time lineage to the new stable
    # singleton lineage during migration.
    review_targets.extend(
        note.name
        for note in notes_by_name.values()
        if note.name in current
        and note.name not in review_targets
        and note.name not in retired
        and _source_owned_gcal_note(note, prior_ids)
    )

    # The authenticated cancellation event plus the dual-bound retired note is
    # already a complete audit trail.  A second current "deleted" note would
    # become stale if the source later reappears, so create a review correction
    # only when automatic retirement was not authorized.
    if retired and not review_targets:
        return Decision(
            "SUPERSEDE",
            f"gcal-cancel:{event_id or uid or ev.id}",
            note_name=retired[0],
            detail=f"source cancellation retired={','.join(sorted(retired))}",
        )

    filed_before = [
        note
        for note in notes_by_name.values()
        if note.type == "correction" and ev.id in (note.provenance or [])
    ]
    if filed_before:
        holder = sorted(filed_before, key=lambda note: note.name)[0]
        return Decision(
            "IGNORE",
            f"gcal-cancel:{event_id or uid or ev.id}",
            note_name=holder.name,
            detail="replay: cancellation event already filed",
        )

    title = str((ev.payload or {}).get("title") or "Calendar event")
    stable = str(event_id or uid or ev.id)
    base_name = _gcal_cancellation_name(stable)
    # Immutable provenance is the replay boundary.  A historical correction is
    # never reopened; a genuinely new source revision continues under a fresh
    # current revision name.
    if (
        base_name in _current_distill_names(notes_by_name.values())
        or base_name not in notes_by_name
    ):
        name = base_name
    else:
        revisions = sorted(
            note.name
            for note in notes_by_name.values()
            if note.name in _current_distill_names(notes_by_name.values())
            and note.type == "correction"
            and note.name.startswith(base_name + "-revision")
        )
        if revisions:
            name = revisions[-1]
        else:
            name = base_name + "-revision"
            suffix = 2
            while name in notes_by_name:
                name = f"{base_name}-revision-{suffix}"
                suffix += 1
    links = " ".join(f"[[supersedes:{target}]]" for target in sorted(review_targets))
    body = f"Calendar marked this event as cancelled: {title}."
    if links:
        body += "\n\nSupersedes: " + links
    body += f"\n\nDistilled from: `{ev.id}`\n"
    existing = notes_by_name.get(name)
    if existing is None:
        superseded_acl = list(ev.acl)
        if review_targets:
            superseded_acl = _safe_acl(
                list(ev.acl),
                *(
                    notes_by_name[target_name].acl
                    for target_name in review_targets
                    if target_name in notes_by_name
                ),
            )
        note = Note(
            name=name,
            description=f"Calendar cancellation: {title}"[:160],
            type="correction",
            domain=provenance.get("domain") or "company",
            acl=superseded_acl,
            entities=list(ev.entities),
            provenance=[ev.id],
            state="proposed",
            body=body,
        )
        write_note(note, context_dir)
        notes_by_name[name] = note
        action = "CREATE"
    else:
        desired = copy.deepcopy(existing)
        desired.acl = _safe_acl(existing.acl, ev.acl)
        if ev.id not in desired.provenance:
            desired.provenance.append(ev.id)
        desired.body = body
        desired.description = f"Calendar cancellation: {title}"[:160]
        desired.type = "correction"
        desired.domain = provenance.get("domain") or existing.domain or "company"
        before = (
            list(existing.acl),
            list(existing.provenance),
            existing.body,
            existing.description,
            existing.type,
            existing.domain,
        )
        after = (
            list(desired.acl),
            list(desired.provenance),
            desired.body,
            desired.description,
            desired.type,
            desired.domain,
        )
        if after == before:
            return Decision(
                "IGNORE",
                f"gcal-cancel:{stable}",
                note_name=existing.name,
                detail=f"replay: no material change; retired={','.join(sorted(retired)) or '-'}",
            )
        if _requires_replacement_proposal(existing):
            return _propose_replacement(
                existing,
                desired,
                context_dir,
                notes_by_name,
                f"gcal-cancel:{stable}",
                detail="calendar correction refresh requires review",
            )
        desired.state = "proposed"
        write_note(desired, context_dir)
        notes_by_name[name] = desired
        note = desired
        action = "UPDATE"
    return Decision(
        action,
        f"gcal-cancel:{stable}",
        note_name=note.name,
        detail=f"retired={','.join(sorted(retired)) or '-'}",
    )


def _distill_slack_tombstone(
    ev: Event,
    notes_by_name: dict[str, Note],
    all_events: list[Event],
    context_dir: str,
) -> Decision:
    """File one closed-contract Slack deletion without mutating cited notes.

    Only an explicitly owned, machine-proposed atomic note may be retired
    automatically.  Legacy citations and multiple owners stop in review; an
    approved source note remains byte-for-byte intact and is referenced by a
    proposed correction that a human may choose to approve.
    """
    if not _is_slack_tombstone(ev):
        raise ValueError("Slack tombstone violates the closed source contract")
    lineage = _mutable_source_lineage(ev)
    history_ids = {
        candidate.id
        for candidate in all_events
        if _mutable_source_lineage(candidate) == lineage and candidate.id != ev.id
    }
    current = _current_distill_names(notes_by_name.values())
    current_explicit = [
        note
        for note in notes_by_name.values()
        if note.name in current
        and lineage in (note.source_lineages or [])
        and _is_distill_target(note)
    ]
    if len(current_explicit) > 1:
        raise MutableLineageConflict(
            f"slack-delete:{lineage}",
            [note.name for note in current_explicit],
        )
    retired_replay = [
        note
        for note in notes_by_name.values()
        if source_retirement_valid(note)
        and (note.source_retirement or {}).get("lineage") == lineage
        and (note.source_retirement or {}).get("event_id") == ev.id
    ]
    explicit = current_explicit or retired_replay[:1]
    legacy = [
        note
        for note in notes_by_name.values()
        if note.name in current
        and not (note.source_lineages or [])
        and history_ids.intersection(note.provenance or [])
        and _is_distill_target(note)
    ]
    if not explicit and legacy:
        raise MutableLineageOwnershipRequired(
            f"slack-delete:{lineage}",
            [note.name for note in legacy],
        )

    review_targets: list[str] = []
    retired: list[str] = []
    if explicit:
        prior = explicit[0]
        proof = prior.source_retirement or {}
        already_retired = (
            source_retirement_valid(prior)
            and proof.get("lineage") == lineage
            and proof.get("event_id") == ev.id
        )
        if not already_retired and not _lineage_confined(prior, history_ids | {ev.id}):
            # Owns this thread, but has absorbed other sources too: a deletion
            # of one thread may not retire a page that is also the memory of
            # others.  Preserve bytes and route the transition to review.
            review_targets.append(prior.name)
        elif not already_retired and prior.state not in {"verified", "superseded"}:
            root = os.path.dirname(os.path.abspath(context_dir))
            try:
                capability = _record_source_retirement(
                    prior,
                    root=root,
                    event=ev,
                    lineage=lineage,
                    producer="slack-thread-v3-deletion-v1",
                )
            except (PermissionError, ValueError):
                # A v2/pre-contract owner has no authority-bearing earlier v3
                # revision. Preserve it byte-for-byte and route the transition
                # to attended review instead of blocking every later nightly.
                review_targets.append(prior.name)
            else:
                write_note(
                    prior,
                    context_dir,
                    _retirement_capability=capability,
                )
                retired.append(prior.name)
        elif already_retired:
            retired.append(prior.name)
        else:
            review_targets.append(prior.name)

    if retired and not review_targets:
        return Decision(
            "SUPERSEDE",
            f"slack-delete:{lineage}",
            note_name=retired[0],
            detail=f"source deletion retired={','.join(sorted(retired))}",
        )

    provenance = ev.provenance or {}
    channel = str(provenance.get("channel") or provenance.get("channel_id") or "slack")
    thread_ts = str(provenance.get("thread_ts") or provenance.get("ts") or "thread")
    stable = slugify(f"{channel}-{thread_ts}")
    name = f"slack-thread-deleted-{stable}"
    links = " ".join(f"[[supersedes:{target}]]" for target in sorted(review_targets))
    body = f"Slack reports that thread {thread_ts} in #{channel} was deleted at the source."
    if links:
        body += "\n\nReview source transition: " + links
    body += f"\n\nDistilled from: `{ev.id}`\n"
    description = concise_description(
        f"Slack source deletion: #{channel} thread {thread_ts}",
    )
    existing = notes_by_name.get(name)
    if existing is not None and ev.id in (existing.provenance or []):
        return Decision(
            "IGNORE",
            f"slack-delete:{lineage}",
            note_name=name,
            detail=f"replay: deletion already filed; retired={','.join(retired) or '-'}",
        )
    if existing is None:
        correction = Note(
            name=name,
            description=description,
            type="correction",
            domain="internal",
            acl=list(ev.acl),
            entities=list(ev.entities),
            provenance=[ev.id],
            state="proposed",
            body=body,
        )
        write_note(correction, context_dir)
        notes_by_name[name] = correction
        action = "CREATE"
    else:
        desired = copy.deepcopy(existing)
        desired.acl = _safe_acl(existing.acl, ev.acl)
        desired.provenance = list(dict.fromkeys([*existing.provenance, ev.id]))
        desired.description = description
        desired.body = body
        desired.type = "correction"
        desired.domain = "internal"
        desired.state = "proposed"
        if _requires_replacement_proposal(existing):
            return _propose_replacement(
                existing,
                desired,
                context_dir,
                notes_by_name,
                f"slack-delete:{lineage}",
                detail="Slack deletion correction refresh requires review",
            )
        write_note(desired, context_dir)
        notes_by_name[name] = desired
        correction = desired
        action = "UPDATE"
    return Decision(
        action,
        f"slack-delete:{lineage}",
        note_name=correction.name,
        detail=f"retired={','.join(retired) or '-'}",
    )


def _distill_gdrive_tombstone(
    ev: Event,
    notes_by_name: dict[str, Note],
    all_events: list[Event],
    context_dir: str,
) -> Decision:
    """Apply one inventory-proven Drive disappearance under exact ownership.

    The closed event may retire only one atomic, machine-proposed note whose
    sole mutable lineage is this file. Human-approved, legacy, aggregate, and
    otherwise ambiguous pages remain immutable and are routed to review.
    """
    if not _is_gdrive_tombstone(ev):
        raise ValueError("Drive tombstone violates the closed source contract")
    lineage = _mutable_source_lineage(ev)
    history_ids = {
        candidate.id
        for candidate in all_events
        if _mutable_source_lineage(candidate) == lineage and candidate.id != ev.id
    }
    current = _current_distill_names(notes_by_name.values())
    current_explicit = [
        note
        for note in notes_by_name.values()
        if note.name in current
        and lineage in (note.source_lineages or [])
        and _is_distill_target(note)
    ]
    if len(current_explicit) > 1:
        raise MutableLineageConflict(
            f"gdrive-delete:{lineage}",
            [note.name for note in current_explicit],
        )
    retired_replay = [
        note
        for note in notes_by_name.values()
        if source_retirement_valid(note)
        and (note.source_retirement or {}).get("lineage") == lineage
        and (note.source_retirement or {}).get("event_id") == ev.id
    ]
    explicit = current_explicit or retired_replay[:1]
    legacy = [
        note
        for note in notes_by_name.values()
        if note.name in current
        and not (note.source_lineages or [])
        and history_ids.intersection(note.provenance or [])
        and _is_distill_target(note)
    ]
    if not explicit and legacy:
        raise MutableLineageOwnershipRequired(
            f"gdrive-delete:{lineage}",
            [note.name for note in legacy],
        )

    review_targets: list[str] = []
    retired: list[str] = []
    if explicit:
        prior = explicit[0]
        proof = prior.source_retirement or {}
        already_retired = (
            source_retirement_valid(prior)
            and proof.get("lineage") == lineage
            and proof.get("event_id") == ev.id
        )
        if already_retired:
            retired.append(prior.name)
        elif not _lineage_confined(prior, history_ids | {ev.id}):
            # Owns this file, but has absorbed other sources too: a deletion of
            # one file may not retire a page that is also the memory of others.
            review_targets.append(prior.name)
        elif prior.state not in {"verified", "superseded"}:
            root = os.path.dirname(os.path.abspath(context_dir))
            try:
                capability = _record_source_retirement(
                    prior,
                    root=root,
                    event=ev,
                    lineage=lineage,
                    producer="gdrive-file-deletion-v1",
                )
            except (PermissionError, ValueError):
                # An explicit marker is insufficient when its cited source row
                # predates the closed Drive contract or its ownership is not
                # exactly atomic. Preserve bytes and create attended evidence.
                review_targets.append(prior.name)
            else:
                write_note(
                    prior,
                    context_dir,
                    _retirement_capability=capability,
                )
                retired.append(prior.name)
        else:
            review_targets.append(prior.name)

    if retired and not review_targets:
        return Decision(
            "SUPERSEDE",
            f"gdrive-delete:{lineage}",
            note_name=retired[0],
            detail=f"source deletion retired={','.join(sorted(retired))}",
        )

    provenance = ev.provenance or {}
    file_id = str(provenance.get("fileId") or "file")
    title = str((ev.payload or {}).get("title") or file_id)
    name = f"gdrive-file-unavailable-{slugify(file_id)}"
    links = " ".join(f"[[supersedes:{target}]]" for target in sorted(review_targets))
    body = f"Google Drive no longer exposes the allowlisted source file: {title}."
    if links:
        body += "\n\nReview source transition: " + links
    body += f"\n\nDistilled from: `{ev.id}`\n"
    description = concise_description(f"Drive source unavailable: {title}")
    cluster_key = f"gdrive-delete:{lineage}"
    existing = notes_by_name.get(name)
    if existing is not None and ev.id in (existing.provenance or []):
        return Decision(
            "IGNORE",
            cluster_key,
            note_name=name,
            detail=f"replay: deletion already filed; retired={','.join(retired) or '-'}",
        )
    domain = str(provenance.get("domain") or "company")
    if existing is None:
        correction = Note(
            name=name,
            description=description,
            type="correction",
            domain=domain,
            acl=list(ev.acl),
            entities=list(ev.entities),
            provenance=[ev.id],
            state="proposed",
            body=body,
        )
        write_note(correction, context_dir)
        notes_by_name[name] = correction
        action = "CREATE"
    else:
        desired = copy.deepcopy(existing)
        desired.acl = _safe_acl(existing.acl, ev.acl)
        desired.provenance = list(dict.fromkeys([*existing.provenance, ev.id]))
        desired.description = description
        desired.body = body
        desired.type = "correction"
        desired.domain = domain
        desired.state = "proposed"
        if _requires_replacement_proposal(existing):
            return _propose_replacement(
                existing,
                desired,
                context_dir,
                notes_by_name,
                cluster_key,
                detail="Drive deletion correction refresh requires review",
            )
        write_note(desired, context_dir)
        notes_by_name[name] = desired
        correction = desired
        action = "UPDATE"
    return Decision(
        action,
        cluster_key,
        note_name=correction.name,
        detail=f"retired={','.join(retired) or '-'}",
    )


# ---------------------------------------------------------------------------
# Artifact enrichment (published pages → notes)
# ---------------------------------------------------------------------------
#
# An `artifact` event records one publish of a static internal page (title,
# description, url, slug, version, text_excerpt, source host/dir/repo). Like
# meetings, it does NOT flow the generic clustering path — a dedicated,
# deterministic handler keeps the note name STABLE (`artifact-<slug>`) so a
# replay or re-publish (v2, v3, …) UPDATEs the same note rather than minting
# duplicates (idempotent on the `artifact:<slug>:v<n>` event id).
#
# Version order. Which publish a note projects is decided by
# the artifact's own version, never by processing order. The hub manifest's
# integer `version` is the order; the connector mints the event id
# `artifact:<slug>:v<version>` from it, and a note's provenance retains only
# ids, so the id is read on both sides of the comparison. An event older than
# the newest version its note lineage already cites adds its citation and
# labels but cannot replace title, description, body, tags, resource, domain
# or `updated`; the newest version itself may re-project (the repair replay).
# Without this guard, whichever version ran last won: a fraction of multi-version
# notes served an older version.
# Manifest version and published_at agree across multi-version slugs;
# that matters because `distill/verify.py` picks the latest
# source by event time, so a disagreeing manifest would leave the note
# unverified (no model call) rather than serving it wrong.

_ARTIFACT_NOTE_TYPE = "artifact"
_ARTIFACT_EVENT_ID_RE = re.compile(r"artifact:(?P<slug>.+):v(?P<version>\d+)")


def _artifact_content_section(p: dict) -> list[str]:
    """`## Content` lines: the raw excerpt (capped), verbatim.

    Deliberately model-free: the
    artifact note is a source-native projection of its publish event, and the
    deterministic-projection verification clause in `distill/verify.py` proves
    exactly that by re-derivation. An LLM-paraphrased body can never satisfy a
    re-derivation proof, and exact-quote CoVe structurally flags paraphrase —
    so key-fact sharpening moved out of the factual body for good."""
    excerpt = (p.get("text_excerpt") or "").strip()
    if not excerpt:
        return []
    return ["## Content", excerpt[:2000], ""]


def _artifact_body(p: dict, author: str = "") -> str:
    lines: list[str] = []
    desc = (p.get("description") or "").strip()
    if desc:
        lines += [desc, ""]
    lines += _artifact_content_section(p)
    if author:
        lines += [f"Published by: {author}", ""]
    # Only an edit by another account adds a line, so every event published
    # before versioning (no `published_by`) still projects byte-identically.
    published_by = p.get("published_by")
    if published_by and published_by != author:
        who = p.get("editor") or published_by
        versions = f"v{p.get('version', '')}"
        if p.get("edit_of"):
            versions += f", replacing v{p['edit_of']}"
        edited = f"Edited by: {who} ({versions})"
        if p.get("note"):
            edited += f" — {p['note']}"
        lines += [edited, ""]
    src = p.get("source") or {}
    built = " · ".join(
        str(x) for x in [src.get("host"), src.get("dir"), src.get("repo")] if x
    )
    if built:
        lines += [f"Built from: {built}", ""]
    lines.append(f"Live page: {p.get('url', '')}")
    return "\n".join(lines).rstrip() + "\n"


def _artifact_id_version(event_id: object) -> tuple[str, int] | None:
    """``(slug, version)`` of an ``artifact:<slug>:v<N>`` event id, else None."""
    match = _ARTIFACT_EVENT_ID_RE.fullmatch(str(event_id or ""))
    if match is None:
        return None
    return match.group("slug"), int(match.group("version"))


def _artifact_newest_cited(slug: str, notes: Iterable[Note]) -> int | None:
    """The highest version of ``slug`` that any of ``notes`` cites, if any."""
    versions = [
        parsed[1]
        for note in notes
        for pid in (note.provenance or [])
        if (parsed := _artifact_id_version(pid)) is not None and parsed[0] == slug
    ]
    return max(versions, default=None)


def _artifact_lineage(
    existing: Note, notes_by_name: dict[str, Note], current: set[str]
) -> list[Note]:
    """``existing`` plus the pending machine proposal that would receive its
    refresh: a reviewed note's newest version can live only in that proposal."""
    marker = _machine_update_marker(existing.name)
    return [
        existing,
        *(
            note
            for note in notes_by_name.values()
            if note.name != existing.name
            and note.name in current
            and marker in (note.provenance or [])
        ),
    ]


def _artifact_project(note: Note, ev: Event) -> Note:
    """A copy of ``note`` carrying ``ev``'s projection and proposal lineage.

    The one projection rule for CREATE, refresh and the dry-run repair tool.
    Source content is replaced; an owned pending replacement edge is retained.
    Provenance, entities and ACL are left alone: they accumulate separately.
    """
    p = ev.payload or {}
    projected = copy.deepcopy(note)
    projected.type = _ARTIFACT_NOTE_TYPE
    projected.domain = p.get("domain") or note.domain or "internal"
    projected.body = _artifact_body(p, str(ev.author or ""))
    # The payload is replaceable; a pending machine proposal's lineage is not.
    # Replaying its own event selects the proposal as the holder, bypassing
    # _propose_replacement, so keep its existing, ownership-matched edge.
    if note.state == "proposed":
        for relation, target in note.typed_links:
            edge = f"[[supersedes:{target}]]"
            if (
                relation == "supersedes"
                and _machine_update_marker(target) in (note.provenance or [])
                and edge not in projected.body
            ):
                projected.body = projected.body.rstrip() + f"\n\nLinks: {edge}\n"
    projected.description = (p.get("title") or note.name).strip()[:160]
    projected.tags = [str(t) for t in (p.get("tags") or []) if t]
    if p.get("title"):
        projected.title = p["title"]
    if p.get("url"):
        projected.resource = p["url"]
    projected.updated = str(ev.occurred_at or "")[:10] or _today()
    return projected


def _distill_artifact(
    ev: Event, notes_by_name: dict[str, Note], context_dir: str
) -> Decision:
    """Dedicated handler for one artifact-publish event.

    Idempotent: keyed on the stable note name `artifact-<slug>`, so a replay
    (or --rebuild, or a v2 re-publish) UPDATEs the same note rather than
    creating a duplicate. Unlike meeting notes, the BODY is machine-owned —
    a re-publish means the page's content changed, so the body is refreshed
    from the newest version's payload; provenance/entities/acl accumulate.
    An older version than the lineage already cites (see "Version order"
    above) only adds its citation and labels, and against a reviewed note it
    is ignored outright rather than opening a provenance-only review proposal.
    """
    p = ev.payload or {}
    base_name = slugify(f"artifact-{p.get('slug') or ev.id}")
    current = _current_distill_names(notes_by_name.values())
    filed = [
        note for note in notes_by_name.values() if ev.id in (note.provenance or [])
    ]
    if filed:
        holder = sorted(filed, key=lambda note: note.name)[0]
        if holder.name not in current:
            return Decision(
                "IGNORE",
                ev.id,
                note_name=holder.name,
                detail="replay: event already filed in history",
            )
        name = holder.name
    elif base_name in current or base_name not in notes_by_name:
        name = base_name
    else:
        revisions = sorted(
            note.name
            for note in notes_by_name.values()
            if note.name in current
            and note.type == _ARTIFACT_NOTE_TYPE
            and note.name.startswith(base_name + "-revision")
        )
        if revisions:
            name = revisions[-1]
        else:
            name = base_name + "-revision"
            i = 2
            while name in notes_by_name:
                name = f"{base_name}-revision-{i}"
                i += 1
    url = p.get("url", "")
    detail = f"artifact v{p.get('version')}"

    existing = notes_by_name.get(name)
    if existing is not None:
        # Refresh the machine-owned body/metadata and accumulate provenance.
        # Once a human approved or authored the note, publish changes become a
        # reviewable replacement rather than an in-place rewrite.
        own = _artifact_id_version(ev.id)
        newest = (
            _artifact_newest_cited(
                own[0], _artifact_lineage(existing, notes_by_name, current)
            )
            if own is not None
            else None
        )
        older = own is not None and newest is not None and own[1] < newest
        if older:
            if _requires_replacement_proposal(existing):
                return Decision(
                    "IGNORE",
                    ev.id,
                    note_name=name,
                    detail=f"older artifact v{own[1]} < v{newest}: kept newest",
                )
            detail = f"artifact v{own[1]} < v{newest}: provenance only"
            desired = copy.deepcopy(existing)
        else:
            desired = _artifact_project(existing, ev)
        new_acl = _safe_acl(existing.acl, list(ev.acl))
        for pid in (ev.id, url):
            if pid and pid not in desired.provenance:
                desired.provenance.append(pid)
        for e in ev.entities:
            if e not in desired.entities:
                desired.entities.append(e)
        desired.acl = new_acl
        before = (
            existing.type,
            existing.domain,
            existing.body,
            existing.description,
            existing.title,
            existing.resource,
            list(existing.provenance),
            list(existing.entities),
            list(existing.acl),
            existing.updated,
            list(existing.tags),
        )
        after = (
            desired.type,
            desired.domain,
            desired.body,
            desired.description,
            desired.title,
            desired.resource,
            list(desired.provenance),
            list(desired.entities),
            list(desired.acl),
            desired.updated,
            list(desired.tags),
        )
        if after == before:
            return Decision(
                "IGNORE", ev.id, note_name=name, detail="replay: no material change"
            )
        if _requires_replacement_proposal(existing):
            return _propose_replacement(
                existing,
                desired,
                context_dir,
                notes_by_name,
                ev.id,
                detail="artifact refresh requires review",
            )
        desired.state = "proposed"
        write_note(desired, context_dir)
        notes_by_name[name] = desired
        return Decision("UPDATE", ev.id, note_name=name, detail=detail)

    note = _artifact_project(
        Note(
            name=name,
            description="",
            domain="",
            acl=list(ev.acl),
            entities=list(ev.entities),
            provenance=[x for x in (ev.id, url) if x],
            state="proposed",
        ),
        ev,
    )
    write_note(note, context_dir)
    notes_by_name[name] = note
    return Decision("CREATE", ev.id, note_name=name, detail=detail)


def _decide_and_apply(
    cl: Cluster,
    notes_by_name: dict[str, Note],
    context_dir: str,
    client=None,
    auto_create_project: bool = False,
) -> Decision:
    """Choose ONE action for the cluster and apply it to disk.

    Deterministic heuristic (runs with NO API key):
      * IGNORE  if the cluster has no durable event (pure chatter).
      * UPDATE  if it matches an existing note (shared provenance or
                strong entity+keyword overlap).
      * LINK    if it doesn't itself warrant a new note but relates two
                existing notes (no durable signal but >=2 related notes).
      * CREATE  otherwise — a durable cluster not yet represented.
    """
    notes = list(notes_by_name.values())
    filed = _filed_current_note(cl, notes)
    if filed is not None:
        return Decision(
            "IGNORE",
            cl.key,
            note_name=filed.name,
            detail="replay: cluster provenance already filed",
        )
    durable_events = [ev for ev in cl.events if _is_durable(ev)]
    # Human uploads are durable intent: never dropped by a noise gate. Curation of an
    # uploaded doc happens at the human review step (keep/drop), not in the distiller.
    upload = _is_upload_cluster(cl)

    # Registry routing (same authority as the LLM path): drop a noise-only cluster,
    # else carry the relationship domain into _apply_create via cl._reg_domain.
    reg_dom = _registry_domain(cl)
    if reg_dom == "__skip__":
        if upload:
            reg_dom = None  # upload → keep, route by subject
        else:
            return Decision("IGNORE", cl.key, detail="registry:noise-only")
    # Bulk/unsubscribe gate: a cluster that is mailing-list noise (even when it
    # carries our own domain's address, which defeats the registry-only test)
    # never becomes a customer/prospect note. Uploads bypass this — a human chose them.
    if not upload and _is_bulk_noise_cluster(cl):
        return Decision("IGNORE", cl.key, detail="noise:bulk/unsubscribe")
    cl._reg_domain = reg_dom  # type: ignore[attr-defined]

    # Mutable snapshots may update only their explicit owner.  With no owner,
    # a new durable object gets a fresh atomic note; semantic similarity alone
    # cannot authorize mutation of a legacy or unrelated note.
    if cl.source_lineages:
        bound = _lineage_bound_notes(cl, notes)
        if bound:
            return _apply_update(cl, bound[0], context_dir, notes_by_name)
        if not durable_events:
            return Decision("IGNORE", cl.key, detail="pure chatter")
        return _apply_create(
            cl,
            context_dir,
            notes_by_name,
            [],
            client=client,
            auto_create_project=auto_create_project,
        )

    match, related = _match_existing(cl, notes)
    related_names = [n.name for n in related][:6]

    # No durable signal -> not a new note.
    if not durable_events:
        if match is not None:
            return _apply_update(cl, match, context_dir, notes_by_name)
        if len(related_names) >= 2:
            return _apply_link(related, context_dir, notes_by_name)
        return Decision("IGNORE", cl.key, detail="pure chatter")

    # Durable signal but already represented -> UPDATE.
    if match is not None:
        return _apply_update(cl, match, context_dir, notes_by_name, related_names)

    # New durable fact -> CREATE. Three-layer project inference is
    # threaded through here so the deterministic CREATE path can propose new
    # namespaces when auto_create_project is enabled.
    return _apply_create(
        cl,
        context_dir,
        notes_by_name,
        related_names,
        client=client,
        auto_create_project=auto_create_project,
    )


def _apply_create(
    cl: Cluster,
    context_dir: str,
    notes_by_name: dict[str, Note],
    links: list[str],
    client=None,
    auto_create_project: bool = False,
) -> Decision:
    name = _name_for_cluster(cl)
    # avoid name collisions within this pass
    base = name
    i = 2
    while name in notes_by_name:
        name = f"{base}-{i}"
        i += 1

    desc = _description(cl)
    current = _current_distill_names(notes_by_name.values())
    links = [
        name
        for name in links
        if name in current and acl_can_reveal(cl.acl, notes_by_name[name].acl)
    ]
    body = _body(cl, desc, links, current)

    # Registry type is authoritative for the folder when set; else the cluster's
    # source/subject domain.
    dom = getattr(cl, "_reg_domain", None) or cl.domain
    # Three-layer project inference: peer -> namespace -> LLM-proposed.
    # Layer 3 only fires when auto_create_project is True (config-gated to
    # avoid runaway folder creation at LLM call rate).
    project = _infer_project(
        cl, dom, notes_by_name, client=client, auto_create=auto_create_project
    )
    note = Note(
        name=name,
        description=desc,
        type=_note_type(cl),
        domain=dom,
        project=project,
        acl=cl.acl,
        entities=cl.entities,
        components=cl.components,
        provenance=cl.provenance_ids,
        source_lineages=list(cl.source_lineages),
        body=body,
        state="proposed",
    )
    write_note(note, context_dir, source=_cluster_govern_source(cl, context_dir))
    notes_by_name[name] = note
    return Decision(
        "CREATE", cl.key, note_name=name, detail=f"domain={dom} entities={cl.entities}"
    )


def _infer_project(
    cl: Cluster,
    dom: str,
    notes_by_name: dict[str, Note],
    client=None,
    auto_create: bool = False,
) -> str:
    """Return the project namespace to file a new note under, or "" for flat.

    Three-layer inference:

      Layer 1 — same-owner-same-domain peer with project set.
        Bootstrap-friendly: an owner's event inherits the project from
        that owner's other billing-revamp notes (same owner + same domain + project set).

      Layer 2 — project-namespace match across ALL same-domain peers.
        Multi-contributor support: a teammate's billing-revamp event inherits the project
        when any peer's `_project_namespace` tokens overlap with the cluster's
        predicted namespace. Doesn't require ownership match.

      Layer 3 — the LLM proposes a new namespace from cluster content.
        Opt-in via config `distill.auto_create_project` (default OFF).
        Returns the first valid candidate that doesn't collide with existing
        project names in this domain. Gated so the distiller doesn't create
        folders at LLM call rate (slow + quota cost).
    """
    cl_primary = cl.entities[:1]
    current = _current_distill_names(notes_by_name.values())
    current_notes = [note for note in notes_by_name.values() if note.name in current]

    # Layer 1: same-owner-same-domain peer with project set.
    if cl_primary:
        primary = cl_primary[0]
        for n in current_notes:
            if n.domain != dom:
                continue
            if n.entities[:1] != [primary]:
                continue
            if n.project:
                return n.project

    # Layer 2: project-namespace match across all same-domain peers with
    # `project` set. Captures multi-contributor projects: anyone whose
    # predicted slug shares a namespace token with an existing project note
    # in the same domain inherits that project.
    cl_pred_name = _name_for_cluster(cl)
    cl_ns = _project_namespace(cl_pred_name)
    if cl_ns:
        for n in current_notes:
            if n.domain != dom or not n.project:
                continue
            n_ns = _project_namespace(n.name)
            if cl_ns & n_ns:
                return n.project

    # Layer 3: the LLM proposes a new namespace (opt-in via auto_create).
    if auto_create and client is not None:
        existing = {n.project for n in current_notes if n.domain == dom and n.project}
        proposed = _propose_project_via_llm(cl, dom, existing, client)
        if proposed:
            return proposed

    return ""


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _machine_update_marker(target_name: str) -> str:
    return f"machine-update:{target_name}"


def _requires_replacement_proposal(note: Note) -> bool:
    """Machine writes never mutate approved or directly human-authored bytes."""
    machine_owned = any(
        str(item).startswith("machine-update:") for item in (note.provenance or [])
    )
    return note.state == "verified" or (
        not machine_owned
        and any(str(item).startswith("by:") for item in (note.provenance or []))
    )


def _propose_replacement(
    existing: Note,
    desired: Note,
    context_dir: str,
    notes_by_name: dict[str, Note],
    cluster_key: str,
    detail: str = "machine update requires review",
    finalize_note=None,
) -> Decision:
    """Write/update one machine-owned proposed head, preserving *existing*.

    ``desired`` is the complete replacement snapshot.  A durable marker lets a
    later machine pass update only its own pending proposal; human/import heads
    are never silently co-opted.  Promotion remains the sole operation that
    makes the edge current.
    """
    current = _current_distill_names(notes_by_name.values())
    if existing.name not in current:
        raise ValueError(f"refusing replacement of historical note {existing.name!r}")
    marker = _machine_update_marker(existing.name)
    heads = [
        note
        for note in notes_by_name.values()
        if effective_state(note) != "superseded"
        and ("supersedes", existing.name) in note.typed_links
    ]
    owned = [note for note in heads if marker in (note.provenance or [])]
    if len(heads) != len(owned) or len(owned) > 1:
        raise ValueError(
            f"unretired non-machine or parallel head already targets {existing.name!r}"
        )

    desired = copy.deepcopy(desired)
    desired.path = ""
    desired.title = desired.title or existing.title or existing.name
    desired.state = "proposed"
    desired.superseded = False
    desired.confidence = None
    desired.body = desired.body.rstrip()
    if f"[[supersedes:{existing.name}]]" not in desired.body:
        desired.body += f"\n\nLinks: [[supersedes:{existing.name}]]\n"
    # ``by:`` identifies the author of the *old* note, not this machine
    # proposal.  The supersedes edge carries lineage; copying authorship would
    # falsely make confidence/UI treat generated prose as human-written.
    desired.provenance = list(
        dict.fromkeys(
            [
                *(
                    item
                    for item in (desired.provenance or [])
                    if not str(item).startswith("by:")
                ),
                marker,
            ]
        )
    )

    if owned:
        head = owned[0]
        if head.name not in current:
            raise ValueError(f"machine proposal {head.name!r} is no longer current")
        desired.name = head.name
        desired.provenance = list(
            dict.fromkeys(
                [
                    *(
                        item
                        for item in (head.provenance or [])
                        if not str(item).startswith("by:")
                    ),
                    *(desired.provenance or []),
                ]
            )
        )
        if finalize_note is not None:
            finalize_note(desired, head)
        # ``write_note`` fail-closes every proposed machine artifact by adding
        # a content-bound skipped verdict when no producer proof was supplied.
        # Apply that normalization before the idempotency comparison; otherwise
        # an unchanged pending proposal compares ``None`` to the persisted
        # skipped block, is reported as UPDATE on every pass, and never
        # converges even though the bytes written are identical.
        from remembrancerlib import ensure_machine_verification_binding

        ensure_machine_verification_binding(desired)
        comparable_fields = (
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
            "updated",
            "resource",
            "tags",
            "superseded",
            "structured",
            "state",
            "confidence",
        )
        same_metadata = all(
            getattr(desired, field_name) == getattr(head, field_name)
            for field_name in comparable_fields
        )
        if same_metadata and desired.body.strip() == head.body.strip():
            return Decision(
                "IGNORE",
                cluster_key,
                note_name=head.name,
                detail=f"pending replacement already current; target={existing.name}",
            )
        write_note(desired, context_dir)
        notes_by_name[desired.name] = desired
        return Decision(
            "UPDATE",
            cluster_key,
            note_name=desired.name,
            detail=f"pending replacement refreshed; target={existing.name}",
        )

    base = slugify(f"{existing.name}-update")
    name = base
    suffix = 2
    while name in notes_by_name:
        name = f"{base}-{suffix}"
        suffix += 1
    desired.name = name
    if finalize_note is not None:
        finalize_note(desired, None)
    from remembrancerlib import ensure_machine_verification_binding

    ensure_machine_verification_binding(desired)
    write_note(desired, context_dir)
    notes_by_name[name] = desired
    return Decision(
        "SUPERSEDE",
        cluster_key,
        note_name=name,
        detail=f"{detail}; target={existing.name}",
    )


def _apply_update(
    cl: Cluster,
    note: Note,
    context_dir: str,
    notes_by_name: dict[str, Note],
    links: list[str] | None = None,
) -> Decision:
    if note.name not in _current_distill_names(notes_by_name.values()):
        raise ValueError(f"refusing UPDATE of historical note {note.name!r}")
    if not _is_distill_target(note):
        raise ValueError(
            f"refusing generic UPDATE of producer-owned projection {note.name!r}"
        )
    if _requires_replacement_proposal(note):
        desired = copy.deepcopy(note)
        desired.description = _description(cl)
        desired.acl = _safe_acl(note.acl, cl.acl)
        desired.entities = list(dict.fromkeys([*note.entities, *cl.entities]))
        desired.components = list(dict.fromkeys([*note.components, *cl.components]))
        desired.source_lineages = list(
            dict.fromkeys(
                [
                    *note.source_lineages,
                    *cl.source_lineages,
                ]
            )
        )
        added = [pid for pid in cl.provenance_ids if pid not in note.provenance]
        desired.provenance = list(note.provenance) + added
        if added:
            trail = "\n".join(f"- `{pid}`" for pid in added)
            desired.body = desired.body.rstrip() + f"\n\nUpdate {_today()}:\n{trail}\n"
        if links:
            current = _current_distill_names(notes_by_name.values())
            new_links = [
                name
                for name in links
                if name != note.name
                and name in current
                and acl_can_reveal(desired.acl, notes_by_name[name].acl)
                and f"[[{name}]]" not in desired.body
            ]
            if new_links:
                desired.body = (
                    desired.body.rstrip()
                    + "\nRelated: "
                    + " ".join(f"[[{name}]]" for name in new_links)
                    + "\n"
                )
        before = (
            note.description,
            list(note.acl),
            list(note.entities),
            list(note.components),
            list(note.provenance),
            list(note.source_lineages),
            note.body,
        )
        after = (
            desired.description,
            list(desired.acl),
            list(desired.entities),
            list(desired.components),
            list(desired.provenance),
            list(desired.source_lineages),
            desired.body,
        )
        if after == before:
            return Decision(
                "IGNORE", cl.key, note_name=note.name, detail="replay: no new evidence"
            )
        desired.updated = _today()
        return _propose_replacement(
            note,
            desired,
            context_dir,
            notes_by_name,
            cl.key,
        )
    # Append new provenance, union entities/acl, bump updated. Note reflects
    # current truth; history lives in the events.
    new_acl = _safe_acl(note.acl, cl.acl)
    before = (
        list(note.provenance),
        list(note.source_lineages),
        list(note.entities),
        list(note.components),
        list(note.acl),
        note.description,
        note.body,
    )
    added = [p for p in cl.provenance_ids if p not in note.provenance]
    note.provenance = note.provenance + added
    note.source_lineages = list(
        dict.fromkeys(
            [
                *note.source_lineages,
                *cl.source_lineages,
            ]
        )
    )
    for e in cl.entities:
        if e not in note.entities:
            note.entities.append(e)
    for c in cl.components:
        if c not in note.components:
            note.components.append(c)
    note.acl = new_acl
    # Retrieval intentionally ranks name + description, not the append-only
    # evidence trail.  Keep the newest distilled truth in description whenever
    # this cluster contributes new immutable evidence.  A human-authored memory
    # (``by:`` provenance) remains authoritative and is never silently rewritten.
    human_authored = any(str(item).startswith("by:") for item in note.provenance)
    if added and not human_authored:
        note.description = _description(cl)
    # extend body with a dated provenance trail for the new events
    if added:
        trail = "\n".join(f"- `{p}`" for p in added)
        note.body = note.body.rstrip() + f"\n\nUpdate {_today()}:\n{trail}\n"
    if links:
        current = _current_distill_names(notes_by_name.values())
        new_links = [
            link
            for link in links
            if link != note.name
            and link in current
            and acl_can_reveal(note.acl, notes_by_name[link].acl)
            and f"[[{link}]]" not in note.body
        ]
        if new_links:
            note.body = (
                note.body.rstrip()
                + "\nRelated: "
                + " ".join(f"[[{link}]]" for link in new_links)
                + "\n"
            )
    after = (
        list(note.provenance),
        list(note.source_lineages),
        list(note.entities),
        list(note.components),
        list(note.acl),
        note.description,
        note.body,
    )
    if after == before:
        return Decision(
            "IGNORE", cl.key, note_name=note.name, detail="replay: no new evidence"
        )
    note.updated = _today()
    note.state = "proposed"
    try:
        write_note(note, context_dir)
    except ValueError as exc:
        # The shared write door refuses a note whose body links a target its
        # readers cannot retrieve. That guard is correct and must stay — but a
        # refusal is about ONE note, and letting it propagate once killed a
        # whole shard: one bad link failed the distill stage and refused
        # the publish for two consecutive runs.
        #
        # The links added just above are already ACL-filtered by
        # `acl_can_reveal`, so what trips the door here is a PRE-EXISTING link
        # — typically on a regenerated dossier page — surfacing when an
        # unrelated update touches the note. Nothing this pass can do makes
        # that note writable, so the cluster is left un-applied and loud;
        # bounded attempts route it to attended review, which is the venue
        # that can actually fix the link or the ACL.
        #
        # One blocked note must not discard the run.
        print(f"[distill] UPDATE refused for {note.name!r}: {exc}", file=sys.stderr)
        return Decision(
            "IGNORE", cl.key, note_name=note.name, detail=f"write-refused: {exc}"
        )
    notes_by_name[note.name] = note
    return Decision(
        "UPDATE", cl.key, note_name=note.name, detail=f"+{len(added)} provenance"
    )


def _apply_link(
    related: list[Note], context_dir: str, notes_by_name: dict[str, Note]
) -> Decision:
    # Add reciprocal [[links]] among related notes that don't yet link.
    current = _current_distill_names(notes_by_name.values())
    related = [
        note for note in related if note.name in current and _is_distill_target(note)
    ]
    changed = []
    for n in related:
        others = [
            m.name
            for m in related
            if m.name != n.name
            and acl_can_reveal(n.acl, m.acl)
            and acl_can_reveal(m.acl, n.acl)
            and f"[[{m.name}]]" not in n.body
        ]
        if others:
            desired = copy.deepcopy(n)
            desired.body = (
                desired.body.rstrip()
                + "\nRelated: "
                + " ".join(f"[[{m}]]" for m in others)
                + "\n"
            )
            desired.updated = _today()
            if _requires_replacement_proposal(n):
                proposed = _propose_replacement(
                    n,
                    desired,
                    context_dir,
                    notes_by_name,
                    "link",
                    detail="structural link update requires review",
                )
                changed.append(f"{n.name}->{proposed.note_name}")
            else:
                desired.state = "proposed"
                try:
                    write_note(desired, context_dir)
                except ValueError as exc:
                    # Same hazard as `_apply_update`: the new links here are
                    # ACL-filtered both ways, but a PRE-EXISTING link on this
                    # note can still trip the write door, and one refusal must
                    # not abort the shard. Skip this note, keep linking the
                    # rest, and say so.
                    print(
                        f"[distill] LINK refused for {n.name!r}: {exc}", file=sys.stderr
                    )
                    continue
                notes_by_name[n.name] = desired
                changed.append(n.name)
    if not changed:
        return Decision("IGNORE", "link-noop", detail="already linked")
    return Decision("LINK", "link", detail=f"linked {', '.join(changed)}")


# ---------------------------------------------------------------------------
# LLM consolidation path
# ---------------------------------------------------------------------------


def _candidates(cluster: Cluster, notes: list[Note], k: int) -> list[Note]:
    """Top-K BM25-similar existing notes for the cluster's text.

    Deterministic projections and legacy dossier/initiative pages are excluded
    from the candidate pool (see `_is_distill_target` / `_match_existing`) and
    are never offered as UPDATE/LINK/SUPERSEDE targets.
    """
    bound = _lineage_bound_notes(cluster, notes)
    if cluster.source_lineages and not bound:
        return []
    current = _current_distill_names(notes)
    notes = [
        n
        for n in notes
        if n.name in current
        and _is_distill_target(n)
        and _acl_compatible(n.acl, cluster.acl)
    ]
    if not notes:
        return []
    query = (
        " ".join(
            (ev.payload.get("subject") or ev.payload.get("text") or "")
            for ev in cluster.events
        )
        + " "
        + " ".join(cluster.entities)
    )
    docs = [f"{n.description} {n.body}" for n in notes]
    ranked = _bm25_rank(query, docs)
    ordered = list(bound)
    seen = {note.name for note in ordered}
    ordered.extend(
        notes[index] for index, _score in ranked if notes[index].name not in seen
    )
    # A lineage-bound target is always offered even if the configured semantic
    # pool is tiny.  There is at most one; ambiguity fails closed above.
    return ordered[: max(k, len(bound))]


def _apply_supersede(
    decision,
    cl: Cluster,
    notes_by_name: dict[str, Note],
    context_dir: str,
    client=None,
    auto_create_project: bool = False,
) -> "Decision":
    """Write a proposed replacement and its durable supersession edge.

    The target remains current until an approver promotes the replacement to
    ``verified``. Retrieval then projects the edge dynamically for each caller;
    no edge-derived ``superseded: true`` flag is written to the target.
    """
    target_name = str(decision.superseded_note or "").strip()
    current = _current_distill_names(notes_by_name.values())
    if target_name not in current:
        raise ValueError(f"refusing SUPERSEDE of non-current note {target_name!r}")
    target = notes_by_name[target_name]
    if not _is_distill_target(target):
        raise ValueError(
            f"refusing generic SUPERSEDE of producer-owned projection {target_name!r}"
        )
    if not acl_can_reveal(cl.acl, target.acl):
        raise ACLConflict(
            f"replacement ACL {cl.acl!r} could reveal target {target_name!r} "
            f"outside {target.acl!r}"
        )
    if any(
        relation == "supersedes" and edge_target == target_name
        for note in notes_by_name.values()
        if note.name != (decision.name or "")
        for relation, edge_target in note.typed_links
    ):
        raise ValueError(f"parallel supersession head already targets {target_name!r}")

    name = decision.name or _name_for_cluster(cl)
    base, i = name, 2
    while name in notes_by_name:
        name = f"{base}-{i}"
        i += 1
    body = decision.body or _body(cl, decision.description or "", [], current)
    # Append a durable typed edge so the supersession survives --rebuild and lint.
    body = body.rstrip() + f"\n\nLinks: [[supersedes:{target_name}]]\n"
    # Inherit project from same-entity-same-domain
    # peers (same heuristic as _apply_create's CREATE path). The supersede
    # target's project is the natural anchor; fall back to peer-inference if
    # the target isn't tracked in `notes_by_name` for any reason.
    sup_target_name = target_name
    project = ""
    if sup_target_name and sup_target_name in notes_by_name:
        project = notes_by_name[sup_target_name].project
    if not project:
        project = _infer_project(
            cl,
            decision.domain or cl.domain,
            notes_by_name,
            client=client,
            auto_create=auto_create_project,
        )
    note = Note(
        name=name,
        description=decision.description or _description(cl),
        type=_note_type(cl),
        domain=decision.domain or cl.domain,  # routed by what it is about
        project=project,
        acl=cl.acl,
        # Entity identity is deterministic/canonical; model output never
        # introduces principals into frontmatter.
        entities=cl.entities,
        components=cl.components,  # the LLM's ConsolidateDecision doesn't carry components this task
        provenance=cl.provenance_ids,
        source_lineages=list(cl.source_lineages),
        body=body,
        state="proposed",
    )
    write_note(note, context_dir, source=_cluster_govern_source(cl, context_dir))
    notes_by_name[name] = note
    return Decision(
        "SUPERSEDE", cl.key, note_name=name, detail=f"superseded={target_name}"
    )


def _render_links(links: list, valid_names: set[str] | None = None) -> str:
    """Render link items into typed-link markers: [[relation:name]].

    The LLM returns `links` inconsistently — sometimes a list of dicts
    ({"relation","name"}), sometimes a list of bare name strings. Accept both so a
    well-formed decision is never dropped by a crash (and ignore other shapes).

    When `valid_names` is provided, drop links whose
    target name isn't in the set — guards against the LLM hallucinating markers like
    `[[relates:name]]` that point to non-existent notes. The body must NOT
    contain unresolvable `[[foo]]` text (it clutters read-time path links and
    confuses lint).
    """
    parts = []
    for lnk in links or []:
        if isinstance(lnk, str):
            rel, nm = "relates", lnk.strip()
        elif isinstance(lnk, dict):
            rel = lnk.get("relation") or "relates"
            nm = (lnk.get("name") or "").strip()
        else:
            continue
        if not nm:
            continue
        # Historical LLM prompts used ``related-to``.  Keep that harmless
        # semantic as a canonical ``relates`` edge while still rejecting any
        # state-changing relation supplied through LINK.
        if rel == "related-to":
            rel = "relates"
        if rel not in {"relates", "extends"}:
            continue  # LINK must never smuggle supersession/contradiction state.
        if valid_names is not None and nm not in valid_names:
            continue
        parts.append(f"[[{rel}:{nm}]]")
    return " ".join(parts)


def _apply_decision(
    decision,
    cl: Cluster,
    notes_by_name: dict[str, Note],
    context_dir: str,
    client=None,
    auto_create_project: bool = False,
    allowed_targets: "set[str] | None" = None,
) -> "Decision | None":
    """Map a ConsolidationDecision onto disk. Returns a Decision or None (=> fallback)."""
    a = decision.action
    filed = _filed_current_note(cl, list(notes_by_name.values()))
    if filed is not None:
        return Decision(
            "IGNORE",
            cl.key,
            note_name=filed.name,
            detail="replay: cluster provenance already filed",
        )
    lineage_bound = _lineage_bound_notes(cl, list(notes_by_name.values()))
    if lineage_bound and (
        a != "UPDATE" or decision.target_note != lineage_bound[0].name
    ):
        # Mutable-source corrections are not optional semantic suggestions.
        # Fall through to the deterministic matcher, which updates the one
        # source-bound note (or proposes a replacement for approved bytes).
        return None
    if cl.source_lineages and not lineage_bound and a != "CREATE":
        return None
    # Human uploads are durable intent: never dropped (noise gate or an LLM IGNORE). The
    # human keeps/drops an uploaded doc at the review step, so distill must always
    # yield a note for it — an LLM IGNORE falls back to the deterministic CREATE.
    upload = _is_upload_cluster(cl)
    # Registry routing is the AUTHORITY: a cluster whose only external entity is
    # `noise` never becomes a note; a cluster with an external registry party is
    # forced into that party's relationship domain, overriding the LLM's guess.
    reg_dom = _registry_domain(cl)
    if reg_dom == "__skip__":
        if upload:
            reg_dom = None  # upload → keep, route by subject
        else:
            return Decision("IGNORE", cl.key, detail="registry:noise-only")
    # Bulk/unsubscribe gate (same authority as the deterministic path): drop a
    # mailing-list-noise cluster before it can be written as a customer note.
    if a in ("CREATE", "SUPERSEDE") and not upload and _is_bulk_noise_cluster(cl):
        return Decision("IGNORE", cl.key, detail="noise:bulk/unsubscribe")
    if reg_dom:
        decision.domain = reg_dom
    if a == "IGNORE":
        if upload:
            return None  # upload: fall back → deterministic CREATE
        return Decision("IGNORE", cl.key, detail="llm:ignore")
    # The ledger speeds up a replay but is not the idempotency boundary.  If it
    # is restored or lost, a note already covering every source id proves this
    # cluster's one factual action was filed; never mint a ``-2`` duplicate.
    current_names = _current_distill_names(notes_by_name.values())
    cluster_provenance = set(cl.provenance_ids)
    covered = next(
        (
            note
            for note in notes_by_name.values()
            if note.name in current_names
            if cluster_provenance and cluster_provenance.issubset(set(note.provenance))
        ),
        None,
    )
    if covered is not None and a in ("CREATE", "SUPERSEDE"):
        return Decision(
            "IGNORE",
            cl.key,
            note_name=covered.name,
            detail="replay: cluster provenance already filed",
        )
    # Weak-model blast-radius gate: factual actions require grounded evidence.
    cluster_ids = {ev.id for ev in cl.events}
    source_text = "\n".join(_event_text(ev) for ev in cl.events)
    evidence_ids = set(getattr(decision, "evidence_event_ids", None) or [])
    evidence_quotes = grounded_evidence_quotes(
        getattr(decision, "evidence_quotes", None) or [], source_text
    )
    if (
        not evidence_ids
        or not evidence_quotes
        or not evidence_ids.issubset(cluster_ids)
    ):
        return None
    if a in ("CREATE", "SUPERSEDE") and not _is_durable_cluster(cl):
        return None
    # CONTENT guard: a CREATE/SUPERSEDE with no description or body has nothing
    # to write; fall back to the deterministic path.  (The provenance_ids check
    # would always pass since clusters always have >=1 event — this is the real
    # gate.)
    if a in ("CREATE", "SUPERSEDE") and not (decision.description or decision.body):
        return None  # no content to write -> deterministic fallback
    if a == "CREATE":
        name = decision.name or _name_for_cluster(cl)
        base, i = name, 2
        while name in notes_by_name:
            name = f"{base}-{i}"
            i += 1
        body = decision.body or _body(cl, decision.description or "", [], current_names)
        if decision.links:
            safe_names = {
                note.name
                for note in notes_by_name.values()
                if note.name in current_names and acl_can_reveal(cl.acl, note.acl)
            }
            link_str = _render_links(decision.links, safe_names)
            if link_str:
                body = body.rstrip() + "\nLinks: " + link_str + "\n"
        # An LLM CREATE inherits project from
        # same-entity-same-domain peers (same heuristic as the deterministic
        # _apply_create path). The LLM itself does not pick a project — that's a
        # deterministic decision based on existing structure.
        project = _infer_project(
            cl,
            decision.domain or cl.domain,
            notes_by_name,
            client=client,
            auto_create=auto_create_project,
        )
        note = Note(
            name=name,
            description=decision.description or _description(cl),
            type=_note_type(cl),
            domain=decision.domain or cl.domain,  # routed by what it is about
            project=project,
            acl=cl.acl,
            # Entity resolution is deterministic/canonical; never let the model
            # inject arbitrary principals or external parties into frontmatter.
            entities=cl.entities,
            components=cl.components,  # the LLM's ConsolidationDecision doesn't carry components this task
            provenance=cl.provenance_ids,
            source_lineages=list(cl.source_lineages),
            body=body,
            state="proposed",
        )
        write_note(note, context_dir, source=_cluster_govern_source(cl, context_dir))
        notes_by_name[name] = note
        return Decision("CREATE", cl.key, note_name=name, detail="llm")
    # Belt-and-suspenders: even though `_candidates` never offers a dossier/
    # initiative page as a candidate, guard the target lookup itself so a
    # hallucinated/guessed target_note can never land an UPDATE/LINK on one —
    # falls through to `return None` -> deterministic fallback (CREATE/IGNORE).
    if (
        a in ("UPDATE", "LINK")
        and decision.target_note in current_names
        and (allowed_targets is not None and decision.target_note in allowed_targets)
        and notes_by_name[decision.target_note].type not in _NON_DISTILL_TARGET_TYPES
    ):
        tgt = notes_by_name[decision.target_note]
        if a == "UPDATE":
            # The deterministic matcher is the authority for factual mutation.
            # Apply only the deterministic update, while accepting the model's
            # correctly grounded routing decision so the quality ledger can
            # converge instead of retrying the same UPDATE forever.
            best, related = _match_existing(cl, list(notes_by_name.values()))
            if best is not tgt:
                return None
            return _apply_update(
                cl,
                tgt,
                context_dir,
                notes_by_name,
                [note.name for note in related][:6],
            )
        # LINK is structural only: it may add validated edges, never facts,
        # entities, provenance, or a replacement description.
        before = tgt.body
        desired = copy.deepcopy(tgt)
        # Union new components from the CLUSTER (decision has no components
        # facet this task — the LLM path doesn't route eng-components).
        if decision.links:
            safe_names = {
                note.name
                for note in notes_by_name.values()
                if note.name in current_names and acl_can_reveal(tgt.acl, note.acl)
            }
            link_str = _render_links(decision.links, safe_names)
            if link_str and link_str not in desired.body:
                desired.body = desired.body.rstrip() + "\nLinks: " + link_str + "\n"
        if desired.body == before:
            return Decision(
                "IGNORE",
                cl.key,
                note_name=tgt.name,
                detail="llm: link replay already filed",
            )
        desired.updated = _today()
        if _requires_replacement_proposal(tgt):
            proposed = _propose_replacement(
                tgt,
                desired,
                context_dir,
                notes_by_name,
                cl.key,
                detail="LLM structural link update requires review",
            )
            return Decision(
                a,
                cl.key,
                note_name=proposed.note_name,
                detail=f"llm: proposed link for {tgt.name}",
            )
        desired.state = "proposed"
        write_note(desired, context_dir)
        notes_by_name[desired.name] = desired
        return Decision(a, cl.key, note_name=desired.name, detail="llm")
    if a == "SUPERSEDE":
        # Unattended supersession is too destructive for a small model. A future
        # review queue may retain this proposal, but nightly keeps old truth live.
        # Translate the grounded classification through the deterministic policy
        # once, then mark the cluster quality-complete; retrying an intentionally
        # disallowed destructive action can never produce convergence.
        fallback = _decide_and_apply(
            cl,
            notes_by_name,
            context_dir,
            client=client,
            auto_create_project=auto_create_project,
        )
        fallback.detail = f"llm:supersede-policy-fallback; {fallback.detail}"
        return fallback
    return None  # unknown / target-missing => fallback


def _is_durable_cluster(cl: Cluster) -> bool:
    """True if the cluster contains at least one durable event."""
    return any(_is_durable(ev) for ev in cl.events)


def _is_upload_cluster(cl: Cluster) -> bool:
    """True if any event in the cluster came from a human upload (source='upload').

    A human upload is durable *intent* — the person deliberately curated this content
    into memory — so an upload cluster must ALWAYS yield a proposed note: it is never
    dropped by a noise gate or an LLM IGNORE. The human keeps or drops it at the review
    step (that is where curation happens for uploads, not in the distiller)."""
    return any(ev.source == "upload" for ev in cl.events)


def _settled_watermark(stream_events, fallback):
    """The newest event that is not dated in the future.

    A positional watermark is only safe over an append-ordered stream. This one
    is ordered by `occurred_at` date, which calendar expansion can put years
    ahead, so the last PHYSICAL record is not the last SETTLED one.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    for event in reversed(stream_events):
        raw = str(getattr(event, "occurred_at", "") or "")
        try:
            when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            # Unparseable is not evidence of the future; treat it as settled
            # rather than refusing to advance at all.
            return event.id
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when <= now:
            return event.id
    # Every event in the window is future-dated: hold the watermark where it is
    # rather than skipping past records that have not happened yet.
    return fallback


def _load_config(root: str) -> dict[str, Any]:
    """Load config.json from root; return empty dict on any error."""
    path = os.path.join(root, "config.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PreparedVerification:
    """A side-effect-free CoVe result bound to one exact note snapshot."""

    note_name: str
    content_fingerprint: str
    result: Any
    #: Whether the model was actually reached. A CoVe that never got an answer
    #: out of the LLM is not evidence about the note, so the caller must be able to
    #: tell it apart from "the model answered and the note could not be
    #: checked" -- charging a retry attempt for the former would turn a
    #: provider outage into stuck clusters.
    model_reached: bool = True


def _verification_source_events(
    note: Note,
    cl,
    event_pool: "list[Event] | dict[str, Event] | None" = None,
) -> list[Event]:
    """Resolve the complete source set without touching audit or note state."""
    if isinstance(event_pool, dict):
        by_id = event_pool
    else:
        pool = (
            list(event_pool)
            if event_pool is not None
            else list(getattr(cl, "events", []) or [])
        )
        by_id = {event.id: event for event in pool}
    return [by_id[event_id] for event_id in note.provenance if event_id in by_id]


def _prepare_verification(
    note_name: str,
    cl,
    notes_by_name: dict,
    client,
    event_pool: "list[Event] | dict[str, Event] | None" = None,
    *,
    parallel_chunks: bool = True,
) -> _PreparedVerification:
    """Run model-only CoVe against an immutable note snapshot.

    No audit, confidence, note, queue, or ledger writes happen here.  The
    production digest can therefore fan this phase out across different notes,
    then commit every result through the serial governed write door.
    """
    from distill.verify import verify_note
    from remembrancerlib import verification_content_fingerprint

    note = notes_by_name.get(note_name)
    if note is None:
        return _PreparedVerification(note_name, "", None)
    snapshot = copy.deepcopy(note)
    source_events = _verification_source_events(snapshot, cl, event_pool)
    outcome: dict = {}
    vres = verify_note(
        snapshot,
        source_events,
        client=client,
        audit=False,
        parallel_chunks=parallel_chunks,
        outcome=outcome,
    )
    return _PreparedVerification(
        note_name,
        verification_content_fingerprint(snapshot),
        vres,
        bool(outcome.get("model_reached", True)),
    )


def _commit_prepared_verification(
    prepared: _PreparedVerification,
    notes_by_name: dict,
    context_dir: str,
    stale_days: int,
) -> str:
    """Durably commit one prepared result, or refuse a stale snapshot.

    The audit append precedes the confidence write exactly as required by the
    verification contract.  Callers serialize this function in source order.
    """
    import govern
    from distill.verify import VerificationResult, _audit_result
    from distill.confidence import compute_confidence
    from lint.engine import contradictions_by_note
    from remembrancerlib import verification_content_fingerprint

    note = notes_by_name.get(prepared.note_name)
    if note is None:
        return "skipped"
    current_fingerprint = verification_content_fingerprint(note)
    if prepared.content_fingerprint != current_fingerprint:
        return "stale"
    vres = prepared.result
    corpus_root = os.path.dirname(os.path.abspath(context_dir))
    verdict_unusable = (
        vres is None
        or getattr(vres, "content_fingerprint", None) != current_fingerprint
        or not getattr(vres, "audit_id", None)
    )
    audit_written = False
    if not verdict_unusable:
        audit_written = _audit_result(
            note,
            vres,
            path=os.path.join(corpus_root, "state", "verification.jsonl"),
        )
        if not audit_written:
            # An audit append that fails is INFRASTRUCTURE, not a verdict, and
            # folding it into "skipped" is how an audit outage can stay
            # invisible: every note read "skipped", which sounds
            # like the verifier declining rather than the ledger being broken.
            # Ledger rotation should keep segment headroom, so
            # reaching here means something a human has to look at.
            print(
                f"[distill] VERIFICATION AUDIT UNWRITABLE for {note.name!r}: the "
                "verdict was computed but could not be recorded, so it is being "
                "persisted as 'skipped'. This is a ledger failure, not a "
                "verification result.",
                file=sys.stderr,
            )
    if govern.note_bound_pending(corpus_root, note):
        # Ledger-only, like the re-score held lane: an open approval item binds
        # these exact bytes, and a confidence-only rewrite would stale the
        # human's binding -- or, when the bytes later oscillate back to an
        # earlier ``edited`` item's, orphan it for good, because
        # enqueue_for_review never reopens a resolved id. The audit row above
        # is the durable verdict: auto_approve reads the ledger's latest row,
        # and decision_review prefers that row over a held note's embedded
        # ``governance_prior_verification`` whenever their audit ids differ
        # (the embedded block no longer refreshes here). A content change
        # still requeues through write_note.
        return vres.status if not verdict_unusable and audit_written else "skipped"
    if verdict_unusable or not audit_written:
        # Persist the quality failure on the note itself. A missing key, missing
        # evidence, malformed model envelope, or failed audit append must never
        # leave a freshly written note looking merely "unverified".
        vres = VerificationResult(
            checked=0,
            supported=0,
            unsupported=0,
            status="skipped",
            content_fingerprint=current_fingerprint,
        )
    contradicted = bool(
        contradictions_by_note(list(notes_by_name.values())).get(note.name)
    )
    note.confidence = compute_confidence(
        note, vres=vres, contradicted=contradicted, stale_days=stale_days
    )
    write_note(note, context_dir)
    return vres.status


def _verify_and_score(
    note_name: str,
    cl,
    notes_by_name: dict,
    context_dir: str,
    client,
    stale_days: int,
    event_pool: "list[Event] | dict[str, Event] | None" = None,
) -> str:
    """Post-write verification hook with model computation separated from durable commit."""
    prepared = _prepare_verification(
        note_name,
        cl,
        notes_by_name,
        client,
        event_pool,
    )
    return _commit_prepared_verification(
        prepared,
        notes_by_name,
        context_dir,
        stale_days,
    )


def _validated_target_events(events_dir: str, requested: "list[Event]") -> list[Event]:
    """Resolve explicit upload requests through the authenticated L1 prefix."""
    validated = {event.id: event for event in iter_events(events_dir)}
    resolved: list[Event] = []
    for event in requested:
        stored = validated.get(event.id)
        if stored is None:
            raise EventValidationError(
                f"targeted event {event.id!r} is outside the validated L1 prefix"
            )
        if stored.to_json() != event.to_json():
            raise EventValidationError(
                f"targeted event {event.id!r} differs from its validated L1 record"
            )
        resolved.append(stored)
    return resolved


@serialized_pipeline_mutation(root_argument="root")
def run(
    root: str = ".",
    use_llm: bool | None = None,
    rebuild: bool = False,
    skip_machine: bool | None = None,
    only_events: "list[Event] | None" = None,
) -> dict[str, Any]:
    """Run one distillation pass. Returns a summary dict.

    Reads events past the `distill` watermark (or ALL events when rebuild=True),
    clusters, resolves entities, applies one action per cluster, regenerates
    INDEX.md, and advances the watermark to the last processed event id.

    Args:
        root:    Remembrancer repo root directory.
        use_llm: If None, auto-detect via LLMClient().available(). Pass True/False
                 to override.
        rebuild: When True, ignore the stored `distill` watermark and reprocess all
                 events from scratch; the watermark is still advanced at the end.
        only_events: When given, distill EXACTLY these events after proving they
                 are present in the validated L1 prefix, and DO NOT advance the
                 global `distill` watermark. This is how uploads are
                 distilled: the watermark can point at a FUTURE-dated calendar event
                 (events are filed by meeting date), which would otherwise make a
                 freshly-uploaded event — filed under today — sort *before* the
                 watermark and be skipped. Targeted distill sidesteps that entirely.
    """
    root = os.path.abspath(root)
    events_dir = os.path.join(root, "events")
    context_dir = os.path.join(root, "context")
    registry_path = os.path.join(root, "registry", "customers.yml")
    # Every registry this run reads is THIS root's: the registry-type authority
    # (entities.yml and its split files) and the eng-component/initiative
    # registries.  A root without one has none; the loaders read a missing file
    # as empty.  Falling back to the code folder's own registry/ made a run on
    # any other root read that folder's registry instead.
    global \
        _REGISTRY_TYPES_PATH, \
        _ENTITY_UNRESOLVED_PATH, \
        _ENG_COMPONENTS_PATH, \
        _INITIATIVES_PATH
    _REGISTRY_TYPES_PATH = os.path.join(root, "registry", "entities.yml")
    _ENG_COMPONENTS_PATH = os.path.join(root, "registry", "eng_components.yml")
    _INITIATIVES_PATH = os.path.join(root, "registry", "initiatives.yml")
    # The unresolved-entity queue is an OUTPUT under THIS root's state/ dir, so
    # canonicalization never writes cross-root.
    _ENTITY_UNRESOLVED_PATH = os.path.join(root, "state", "entity_unresolved.jsonl")
    cursors = Cursors(os.path.join(root, "state", "cursors.json"))

    # Resolve use_llm from LLMClient.available() when not explicitly set.
    client = LLMClient()
    if use_llm is None:
        use_llm = client.available()

    # Rebuild ignores the stored watermark.
    # only_events (uploads) bypasses the watermark entirely — see the docstring.
    targeted = only_events is not None
    after_id = None if rebuild else cursors.get("distill")
    stream_events = (
        _validated_target_events(events_dir, list(only_events))
        if targeted
        else list(iter_events(events_dir, after_id=after_id))
    )
    # Keep the physical stream head before any policy filter.  A pass containing
    # only machine mail or stale git still consumed those records and must
    # advance past them exactly once.
    #
    # But NOT past a far-future record. The stream is ordered by day-file name,
    # and day-files are named from `occurred_at` -- so a recurring calendar
    # expansion files instances years ahead. The live store can hold thousands
    # of such files, ending far in the future. Taking the physical last id would
    # set the watermark to a 2040 event, after which EVERY event ingested
    # afterwards is filed under a day <= today, sorts BEFORE the watermark, and
    # is never yielded again -- while the pass reports "no distillable events
    # past watermark; index regenerated." and exits 0. All Slack, git, Gmail,
    # Drive and upload memory would stop, silently and permanently. A backfill
    # after an outage has the same shape: it files events under their real
    # source dates, which now sort before the watermark.
    #
    # Clamping to the newest settled record keeps the watermark a position in
    # the part of the stream that IS append-ordered. Future-dated instances are
    # re-yielded each pass and dropped again by the same `gcal_max_future_days`
    # filter that already drops them, so nothing is double-distilled.
    stream_last_id = _settled_watermark(stream_events, after_id)
    new_events = list(stream_events)

    # Read candidate_k from config.json distill.candidate_k (default 40).
    cfg = _load_config(root)
    # Notes route by THIS root's repositories (``repo_domains``).
    _bind_repo_domains(cfg)
    verify_on: bool = bool((cfg.get("distill") or {}).get("verify"))
    if verify_on and (not use_llm or not client.available()):
        raise RuntimeError(
            "distill.verify is enabled but the LLM is unavailable (it needs "
            f"{llm_requirement()}); refusing an unverified write pass"
        )

    # One intake for both distillation doors (see distill_intake): machine mail
    # (opt-in via distill.skip_machine_email or run(skip_machine=), impact
    # ALWAYS reported), snapshot collapse, then the git-age, calendar-horizon,
    # calendar personal-block and Slack low-signal valves.
    new_events, intake = distill_intake(new_events, cfg, skip_machine=skip_machine)
    machine_skip = intake["machine_skip"]
    mutable_revisions_superseded = intake["mutable_revisions_superseded"]
    git_age_dropped = intake["git_age_dropped"]
    gcal_future_dropped = intake["gcal_future_dropped"]
    if git_age_dropped:
        print(
            f"[distill] git backfill valve: skipped {git_age_dropped} "
            f"git events older than distill.git_max_age_days",
            file=sys.stderr,
        )
    if gcal_future_dropped:
        print(
            f"[distill] calendar horizon: skipped {gcal_future_dropped} gcal events "
            f"more than {intake['gcal_horizon']} days in the future",
            file=sys.stderr,
        )
    if intake["gcal_block_dropped"]:
        print(
            f"[distill] calendar personal-block valve: skipped "
            f"{intake['gcal_block_dropped']} gcal events matching "
            f"distill.gcal_drop_title_patterns",
            file=sys.stderr,
        )
    if intake["slack_low_signal_dropped"]:
        print(
            f"[distill] slack low-signal valve: skipped "
            f"{intake['slack_low_signal_dropped']} slack events with "
            f"signal_eligible=false",
            file=sys.stderr,
        )
    if intake["git_subject_dropped"]:
        print(
            f"[distill] self-ingestion valve: skipped "
            f"{intake['git_subject_dropped']} git events matching "
            f"distill.git_drop_subject_patterns",
            file=sys.stderr,
        )
    candidate_k: int = (cfg.get("distill") or {}).get("candidate_k", 40)
    # Max-quality mode: when distill.thorough is true, the LLM judges EVERY cluster
    # (the regex durability gate is a cost filter, not a quality one — the LLM is a better
    # noise judge than DECISION_RE). Cluster reaches the LLM = thorough OR durable.
    thorough: bool = bool((cfg.get("distill") or {}).get("thorough"))
    # Layer-3 project inference: opt-in via config. Default OFF.
    # When True, the distiller calls the LLM to propose new project namespaces when
    # no peer-based or namespace match is found. Slow + quota-costly.
    auto_create_project: bool = bool(
        (cfg.get("distill") or {}).get("auto_create_project")
    )
    # Credibility: opt-in post-write verification + confidence. Default OFF — 2 extra LLM
    # calls per WRITTEN note; flip on only after an offline backfill
    # validates quality. Confidence is only computed here in the verify path; the backfill tool
    # scores the whole corpus deterministically without a key.
    stale_days: int = int((cfg.get("distill") or {}).get("stale_days", 180))
    verification_events = (
        {event.id: event for event in iter_events(events_dir)} if verify_on else None
    )

    summary: dict[str, Any] = {
        "events_read": len(stream_events),
        "processed": len(stream_events),  # includes policy-filtered stream records
        "clusters": 0,
        "actions": {"CREATE": 0, "UPDATE": 0, "LINK": 0, "IGNORE": 0, "SUPERSEDE": 0},
        "decisions": [],
        "last_event_id": after_id,
        "use_llm": bool(use_llm),
        "rebuild": rebuild,
        "machine_skip": machine_skip,
        "git_age_dropped": git_age_dropped,
        "gcal_future_dropped": gcal_future_dropped,
        "gcal_block_dropped": intake["gcal_block_dropped"],
        "slack_low_signal_dropped": intake["slack_low_signal_dropped"],
        "git_subject_dropped": intake["git_subject_dropped"],
        "mutable_revisions_superseded": mutable_revisions_superseded,
        "verification": {"verified": 0, "flagged": 0, "skipped": 0},
    }

    def verify_decision(decision: Decision, cluster=None) -> None:
        """Verify every factual write, including a replay after a crash.

        Dedicated meeting/artifact/cancellation handlers do not flow through
        the generic cluster loop.  An idempotent replay can also return IGNORE
        after the first note write but before its audit/frontmatter commit.
        Both paths must complete CoVe before the watermark advances.
        """
        if not (use_llm and verify_on and decision.note_name):
            return
        note = notes_by_name.get(decision.note_name)
        if note is None or note.state == "verified" or source_retirement_valid(note):
            return
        verification = (
            note.confidence.get("verification")
            if isinstance(note.confidence, dict)
            else None
        )
        prior_status = (
            verification.get("status") if isinstance(verification, dict) else None
        )
        factual_write = decision.action in ("CREATE", "UPDATE", "SUPERSEDE")
        replay_incomplete = decision.action == "IGNORE" and prior_status not in {
            "verified",
            "flagged",
        }
        if not (factual_write or replay_incomplete):
            return
        try:
            status = _verify_and_score(
                decision.note_name,
                cluster,
                notes_by_name,
                context_dir,
                client,
                stale_days,
                verification_events,
            )
        except Exception as exc:
            status = "skipped"
            print(
                f"[distill] verification skipped for {decision.note_name}: {exc}",
                file=sys.stderr,
            )
        if status not in summary["verification"]:
            status = "skipped"
        if status in {"flagged", "skipped"}:
            print(
                f"[distill] verification {status} for {decision.note_name}",
                file=sys.stderr,
            )
        summary["verification"][status] += 1

    # Meeting events get a dedicated, deterministic enrichment handler rather
    # than the generic clustering path: stable note name,
    # attendee signal-gate, first-class decisions/action-items. Everything else
    # flows the normal pipeline unchanged.
    if is_meeting_event is not None:
        pipeline_events = [ev for ev in new_events if not is_meeting_event(ev)]
        if targeted or rebuild:
            meeting_events = [ev for ev in new_events if is_meeting_event(ev)]
        else:
            # WATERMARK-IMMUNE: unlike the high-volume sources, meetings are few and
            # _distill_meeting is idempotent (stable note name + provenance dedup), so
            # we scan ALL meeting events every pass. A recap that arrives days late (or
            # a manual import of an old call) files into a past day-file that sorts
            # BEFORE the watermark; relying on the cursor would silently drop it.
            meeting_events = [
                ev for ev in iter_events(events_dir) if is_meeting_event(ev)
            ]
    else:  # pragma: no cover - ingest package absent
        meeting_events, pipeline_events = [], new_events

    # Cancellations are corrections, never normal durable meeting facts.
    cancelled_gcal_events = [ev for ev in pipeline_events if _is_gcal_cancellation(ev)]
    pipeline_events = [ev for ev in pipeline_events if not _is_gcal_cancellation(ev)]

    slack_tombstone_events = [ev for ev in pipeline_events if _is_slack_tombstone(ev)]
    non_authoritative_slack_deletions = [
        ev
        for ev in pipeline_events
        if _is_slack_deletion_claim(ev) and not _is_slack_tombstone(ev)
    ]
    # A v2/forged deletion remains immutable L1 history, but it is neither a
    # lifecycle transition nor ordinary prose to distill.  Consuming it without
    # a write prevents the generic cluster path from turning a legacy tombstone
    # into an effective deletion by another name.
    pipeline_events = [ev for ev in pipeline_events if not _is_slack_deletion_claim(ev)]
    summary["slack_non_authoritative_deletions_dropped"] = len(
        non_authoritative_slack_deletions
    )

    gdrive_tombstone_events = [ev for ev in pipeline_events if _is_gdrive_tombstone(ev)]
    pipeline_events = [ev for ev in pipeline_events if not _is_gdrive_tombstone(ev)]

    # Artifact events get the same dedicated-handler treatment: stable note
    # name `artifact-<slug>`, replay-safe UPDATE,
    # deterministic body — never the generic clustering path.
    artifact_events = [ev for ev in pipeline_events if ev.source == "artifact"]
    pipeline_events = [ev for ev in pipeline_events if ev.source != "artifact"]

    if not new_events and not meeting_events:
        regen_index(context_dir)
        # `stream_last_id` is None when every record in the window is dated in
        # the future and there is no prior watermark to hold: there is nothing
        # settled to point at yet, so leave the cursor unset.
        if not targeted and stream_events and isinstance(stream_last_id, str):
            cursors.set("distill", stream_last_id)
        summary["last_event_id"] = stream_last_id
        print("[distill] no distillable events past watermark; index regenerated.")
        return summary

    resolver = EntityResolver(registry_path)

    # Calendar (gcal) attendee attribution.
    # gcal events arrive tagged with RAW email-domain strings (ingest _external_orgs),
    # which no note/query resolves to a canonical org. Route them through the SAME
    # deep attendee resolver as the Bluedot meeting path, in-memory BEFORE clustering
    # (events are immutable on disk; canonicalization already mutates ev.entities in
    # memory before clustering, so this rides the established pattern). Deterministic,
    # NO LLM. When the resolver yields canonical ids, they replace the dead raw-domain
    # tags; when it yields nothing, the raw domain is LEFT intact (graceful degrade).
    for ev in pipeline_events:
        if getattr(ev, "source", "") != "gcal":
            continue
        attendees = (ev.payload or {}).get("attendees") or []
        if not attendees:
            continue
        gres = resolver.resolve_attendees_deep([("", e) for e in attendees])
        if gres["entities"]:
            ev.entities = list(gres["entities"])
        emit_proposals(
            gres.get("proposals") or [],
            _ENTITY_UNRESOLVED_PATH,
            source_event=ev.id,
            today=_today(),
        )

    lineage_events = list(
        verification_events.values()
        if verification_events is not None
        else iter_events(events_dir)
    )
    clusters, _pipeline_revisions_superseded = cluster_mutable_events(
        pipeline_events,
        resolver,
        lineage_events=lineage_events,
    )
    summary["clusters"] = len(clusters)

    # Load existing notes once; keep an in-pass index by name so CREATE/UPDATE
    # within the same run see each other.
    notes_by_name = index_notes_by_name(
        load_notes(context_dir),
        operation="distillation",
    )

    lineage_conflicts: list[dict[str, Any]] = []
    for cluster in clusters:
        if not cluster.lineage_provenance_ids:
            continue
        try:
            _lineage_bound_notes(cluster, list(notes_by_name.values()))
        except (MutableLineageConflict, MutableLineageOwnershipRequired) as exc:
            lineage_conflicts.append(
                {
                    "cluster": cluster.key,
                    "lineage_provenance_ids": cluster.lineage_provenance_ids,
                    "notes": exc.note_names,
                    "reason": (
                        "mutable-lineage-ambiguous"
                        if isinstance(exc, MutableLineageConflict)
                        else "mutable-lineage-ownership-required"
                    ),
                }
            )
    for event in [*slack_tombstone_events, *gdrive_tombstone_events]:
        lineage = _mutable_source_lineage(event)
        source_prefix = "slack" if event.source == "slack" else "gdrive"
        review_cluster = Cluster(
            key=f"{source_prefix}-delete:{lineage}",
            events=[event],
            source_lineages=[lineage],
            lineage_provenance_ids=[
                candidate.id
                for candidate in lineage_events
                if _mutable_source_lineage(candidate) == lineage
            ],
        )
        try:
            _lineage_bound_notes(review_cluster, list(notes_by_name.values()))
        except (MutableLineageConflict, MutableLineageOwnershipRequired) as exc:
            lineage_conflicts.append(
                {
                    "cluster": review_cluster.key,
                    "lineage_provenance_ids": review_cluster.lineage_provenance_ids,
                    "notes": exc.note_names,
                    "reason": (
                        "mutable-lineage-ambiguous"
                        if isinstance(exc, MutableLineageConflict)
                        else "mutable-lineage-ownership-required"
                    ),
                }
            )
    review_path = os.path.join(root, "state", "mutable_lineage_review.json")
    if lineage_conflicts:
        durable_replace_json(
            review_path,
            {"version": 1, "review_required": lineage_conflicts},
            mode=0o600,
            indent=2,
            sort_keys=True,
        )
        first = lineage_conflicts[0]
        error_type = (
            MutableLineageConflict
            if first["reason"] == "mutable-lineage-ambiguous"
            else MutableLineageOwnershipRequired
        )
        raise error_type(first["cluster"], first["notes"])
    if os.path.exists(review_path):
        durable_replace_json(
            review_path,
            {"version": 1, "review_required": []},
            mode=0o600,
            indent=2,
            sort_keys=True,
        )

    # A cancellation may reference an earlier calendar revision that predates
    # the incremental distill watermark, so only then scan the L1 calendar log.
    all_calendar_events = list(iter_events(events_dir)) if cancelled_gcal_events else []
    for ev in cancelled_gcal_events:
        dec = _distill_gcal_cancellation(
            ev, notes_by_name, all_calendar_events, context_dir
        )
        verify_decision(dec)
        summary["actions"][dec.action] = summary["actions"].get(dec.action, 0) + 1
        summary["decisions"].append(
            {
                "action": dec.action,
                "cluster": dec.cluster_key,
                "note": dec.note_name,
                "detail": dec.detail,
            }
        )
    for ev in slack_tombstone_events:
        dec = _distill_slack_tombstone(ev, notes_by_name, lineage_events, context_dir)
        verify_decision(dec)
        summary["actions"][dec.action] = summary["actions"].get(dec.action, 0) + 1
        summary["decisions"].append(
            {
                "action": dec.action,
                "cluster": dec.cluster_key,
                "note": dec.note_name,
                "detail": dec.detail,
            }
        )
    for ev in gdrive_tombstone_events:
        dec = _distill_gdrive_tombstone(
            ev,
            notes_by_name,
            lineage_events,
            context_dir,
        )
        verify_decision(dec)
        summary["actions"][dec.action] = summary["actions"].get(dec.action, 0) + 1
        summary["decisions"].append(
            {
                "action": dec.action,
                "cluster": dec.cluster_key,
                "note": dec.note_name,
                "detail": dec.detail,
            }
        )

    # Build the calendar/gmail indexes ONCE for the meeting→calendar join, but only
    # when there are meetings to reconcile (scanning gmail is not free) and the
    # reconciler is importable. Best-effort — a failure degrades to Bluedot-only.
    mtg_indexes = None
    if meeting_events:
        try:
            from distill import reconcile as _rec

            # gcal_only: the forward hook only uses indexes.gcal for the join, so
            # skip the gmail corroboration scan it never reads.
            mtg_indexes = _rec.build_indexes(
                events_dir, gcal_only=True, events=list(iter_events(events_dir))
            )
        except Exception:
            mtg_indexes = None

    # Meetings first (deal/customer notes they [[link]] to already exist on disk).
    for ev in meeting_events:
        dec = _distill_meeting(
            ev, resolver, notes_by_name, context_dir, indexes=mtg_indexes
        )
        verify_decision(dec)
        summary["actions"][dec.action] = summary["actions"].get(dec.action, 0) + 1
        summary["decisions"].append(
            {
                "action": dec.action,
                "cluster": dec.cluster_key,
                "note": dec.note_name,
                "detail": dec.detail,
            }
        )

    # Artifacts: dedicated deterministic handler (model-free projection).
    for ev in artifact_events:
        dec = _distill_artifact(ev, notes_by_name, context_dir)
        verify_decision(dec)
        summary["actions"][dec.action] = summary["actions"].get(dec.action, 0) + 1
        summary["decisions"].append(
            {
                "action": dec.action,
                "cluster": dec.cluster_key,
                "note": dec.note_name,
                "detail": dec.detail,
            }
        )

    for cl in clusters:
        # LLM consolidation path for durable clusters when use_llm is True.
        dec: Decision | None = None
        if use_llm and (thorough or _is_durable_cluster(cl)):
            cands = _candidates(cl, list(notes_by_name.values()), candidate_k)
            decision = consolidate(cl, cands, client)
            dec = (
                _apply_decision(
                    decision,
                    cl,
                    notes_by_name,
                    context_dir,
                    client=client,
                    auto_create_project=auto_create_project,
                    allowed_targets={n.name for n in cands},
                )
                if decision
                else None
            )
        if dec is None:
            # Deterministic fallback (also used when use_llm is False).
            dec = _decide_and_apply(
                cl,
                notes_by_name,
                context_dir,
                client=client,
                auto_create_project=auto_create_project,
            )

        # Credibility hook (opt-in): verify a freshly written note against ITS source events,
        # fold the verdict + contradiction + corroboration into a confidence block, re-write it.
        # Post-write, minimal blast radius: only CREATE/UPDATE/SUPERSEDE produce a factual note;
        # IGNORE/LINK need no fact check. A skipped verification is quality-incomplete and keeps
        # the watermark stationary so the authenticated source cluster can be retried.
        verify_decision(dec, cl)

        summary["actions"][dec.action] = summary["actions"].get(dec.action, 0) + 1
        summary["decisions"].append(
            {
                "action": dec.action,
                "cluster": cl.key,
                "note": dec.note_name,
                "detail": dec.detail,
            }
        )

    regen_index(context_dir)

    if summary["verification"]["skipped"]:
        raise RuntimeError(
            f"verification skipped for {summary['verification']['skipped']} written note(s); "
            "watermark not advanced"
        )

    last_id = stream_last_id
    # Targeted (upload) distill must NOT move the global watermark — it processed an
    # out-of-band subset, not the head of the stream. Nor may a pass that only
    # re-swept backdated meetings (no new head events) move it.
    if not targeted and stream_events and isinstance(last_id, str):
        cursors.set("distill", last_id)
    summary["last_event_id"] = last_id

    a = summary["actions"]
    rebuild_tag = " [REBUILD]" if rebuild else (" [TARGETED]" if targeted else "")
    wm = "watermark unchanged (targeted)" if targeted else f"watermark -> {last_id}"
    print(
        f"[distill]{rebuild_tag} {len(new_events)} events -> {len(clusters)} clusters | "
        f"CREATE={a['CREATE']} UPDATE={a['UPDATE']} LINK={a['LINK']} "
        f"IGNORE={a['IGNORE']} | {wm}"
    )
    for d in summary["decisions"]:
        if d["action"] != "IGNORE":
            print(f"    {d['action']:7} {d['note'] or d['cluster']}  {d['detail']}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Remembrancer distillation engine (L1 events -> L2 notes)."
    )
    ap.add_argument("--root", default=".", help="Remembrancer repo root (default: .)")
    ap.add_argument(
        "--llm",
        action="store_true",
        help="Allow an LLM to sharpen descriptions/bodies at WRITE time "
        f"(requires {llm_requirement()}; heuristic fallback otherwise).",
    )
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore stored distill watermark and reprocess ALL events from scratch.",
    )
    args = ap.parse_args(argv)
    run(root=args.root, use_llm=args.llm if args.llm else None, rebuild=args.rebuild)
    return 0


# ---------------------------------------------------------------------------
# AGENTS.md spec loader
# ---------------------------------------------------------------------------

_SPEC_DEFAULTS: dict[str, Any] = {
    "contradiction_policy": "flag",
    "file_back_enabled": True,
}

_BOOL_TRUE = {"true", "yes", "1", "on"}
_BOOL_FALSE = {"false", "no", "0", "off"}


def load_spec(root: str = ".") -> dict[str, Any]:
    """Parse the ``conventions`` fenced block from AGENTS.md and return a dict.

    Reads ``<root>/AGENTS.md``, locates a fenced block delimited by
    ````conventions`` and `````, and parses ``key: value`` lines within it.
    Boolean values (``true``/``false``) are coerced to Python booleans.
    Any key absent from the block, or when AGENTS.md does not exist or has no
    such block, falls back to the safe defaults:

        {"contradiction_policy": "flag", "file_back_enabled": True}

    Stdlib-only; no third-party dependencies.
    """
    result: dict[str, Any] = dict(_SPEC_DEFAULTS)
    agents_path = os.path.join(root, "AGENTS.md")
    if not os.path.isfile(agents_path):
        return result

    try:
        with open(agents_path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return result

    # Locate the ```conventions ... ``` fenced block.
    in_block = False
    for line in lines:
        stripped = line.strip()
        if not in_block:
            if stripped == "```conventions":
                in_block = True
            continue
        # End of fenced block.
        if stripped == "```":
            break
        # Parse key: value
        m = re.match(r"^([A-Za-z0-9_]+)\s*:\s*(.+)$", stripped)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        lower = raw.lower()
        if lower in _BOOL_TRUE:
            result[key] = True
        elif lower in _BOOL_FALSE:
            result[key] = False
        else:
            result[key] = raw

    return result


if __name__ == "__main__":
    raise SystemExit(main())
