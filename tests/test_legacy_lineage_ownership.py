"""A sole legacy citer of a mutable source object is its unambiguous owner.

Before this, `_lineage_bound_notes` raised `MutableLineageOwnershipRequired` for
*any* legacy citation, and no production code path could ever satisfy the
requirement: a note gains `source_lineages` only inside `_apply_update`, which
runs after binding succeeds.  Every such cluster therefore dead-lettered into
the digest ledger's attended-review queue permanently.

Ambiguity is the thing worth stopping on, and ambiguity means *more than one*
current citer.  Exactly one current distill target citing the lineage, with no
explicit owner anywhere, leaves no alternative target — refusing it only drops
the source object's later revisions on the floor.

Granting ownership to a note that also cites other material would, on its own,
widen what a source deletion may retire, so the retirement path here asserts the
atomicity its own docstring already claims ("one atomic note whose sole mutable
lineage is this file"): an owner whose provenance escapes the lineage is routed
to attended review instead of being retired.  Every explicit owner was
confined when this was added, so the guard is insurance for the ownership
grants above.
"""
from __future__ import annotations

import pytest

from distill import engine
from ingest.slack_contract_v3 import snapshot_revision, source_occurred_at
from remembrancerlib import Event, Note, append_events, load_notes
from tests.event_validation_helpers import guarded_run


LEGACY_LINEAGE = "slack:C1:30.000"


def _legacy_slack_event(event_id: str, text: str) -> Event:
    return Event(
        id=event_id,
        source="slack",
        author="alice",
        payload={"text": text},
        occurred_at="2026-07-15T12:00:00Z",
        acl=["eng"],
        provenance={"channel": "engineering", "channel_id": "C1",
                    "thread_ts": "30.000", "ts": "30.000"},
    )


def _legacy_cluster() -> tuple[engine.Cluster, str]:
    old_id = "slack:engineering/30.000@old"
    new = _legacy_slack_event(
        "slack:engineering/30.000@new", "Decision: production is disabled.",
    )
    cluster = engine.Cluster(
        key=LEGACY_LINEAGE,
        events=[new],
        lineage_provenance_ids=[old_id, new.id],
        source_lineages=[LEGACY_LINEAGE],
    )
    return cluster, old_id


def test_sole_legacy_citer_is_bound_as_the_lineage_owner():
    cluster, old_id = _legacy_cluster()
    legacy = Note(
        name="legacy", description="Legacy citation", provenance=[old_id],
        state="proposed",
    )

    assert engine._lineage_bound_notes(cluster, [legacy]) == [legacy]


def test_two_legacy_citers_still_require_attended_review():
    cluster, old_id = _legacy_cluster()
    first = Note(
        name="first", description="Legacy citation", provenance=[old_id],
        state="proposed",
    )
    second = Note(
        name="second", description="Another legacy citation",
        provenance=[old_id], state="proposed",
    )

    with pytest.raises(engine.MutableLineageOwnershipRequired):
        engine._lineage_bound_notes(cluster, [first, second])


def test_explicit_owner_still_beats_a_sole_legacy_citer():
    cluster, old_id = _legacy_cluster()
    owner = Note(
        name="owner", description="Explicit owner", provenance=[old_id],
        source_lineages=[LEGACY_LINEAGE], state="proposed",
    )
    legacy = Note(
        name="legacy", description="Legacy citation", provenance=[old_id],
        state="proposed",
    )

    assert engine._lineage_bound_notes(cluster, [owner, legacy]) == [owner]


# ── Slack v3 snapshots for the end-to-end paths ──────────────────────────────

def _v3_snapshot(text: str, *, deleted: bool = False, observed: str) -> Event:
    channel, thread_ts = "engineering", "30.000"
    if deleted:
        payload = {
            "text": f"[#{channel}] Slack thread {thread_ts} was deleted at the source.",
            "reply_count": 0,
            "message_count": 0,
            "snapshot_complete": True,
            "snapshot_format": "thread-v3",
            "resolution_reactions": [],
            "signal_eligible": False,
            "deleted": True,
            "subtype": "thread_deleted",
            "deletion_evidence": {"kind": "api-error", "code": "message_not_found"},
        }
    else:
        payload = {
            "text": f"[#{channel}] {text}",
            "reply_count": 0,
            "message_count": 1,
            "snapshot_complete": True,
            "snapshot_format": "thread-v3",
            "resolution_reactions": [],
            "signal_eligible": True,
        }
    occurred_at = source_occurred_at(thread_ts)
    provenance = {
        "channel": channel, "channel_id": "C1", "ts": thread_ts,
        "thread_ts": thread_ts, "permalink": None,
    }
    revision = snapshot_revision(
        author="alice", occurred_at=occurred_at, payload=payload,
        provenance=provenance, acl=["eng"], entities=[],
    )
    provenance["revision"] = revision
    return Event(
        id=f"slack:{channel}/{thread_ts}@{revision}",
        source="slack", author="alice", occurred_at=occurred_at,
        ingested_at=observed, payload=payload, acl=["eng"],
        provenance=provenance,
    )


def _rewrite_frontmatter(root, name: str, line_prefix: str, replacement: str | None):
    """Edit one note file on disk the way a pre-contract corpus already reads.

    These fixtures have to produce states the current governed writer refuses to
    mint (a legacy citer with no ownership; an owner that outgrew its lineage),
    so they edit the stored markdown directly rather than through `write_note`.
    """
    context = str(root / "context")
    path = next(item.path for item in load_notes(context) if item.name == name)
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    out = []
    for line in lines:
        if line.startswith(line_prefix):
            if replacement is not None:
                out.append(replacement)
            continue
        out.append(line)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out) + "\n")


def test_legacy_citer_regains_ownership_and_absorbs_the_next_revision(tmp_path):
    """The whole loop: a legacy note is updated and becomes an explicit owner."""
    first = _v3_snapshot(
        "Decision: production launch is enabled.", observed="2026-07-15T12:00:00Z",
    )
    assert append_events([first], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)
    owner = next(
        note for note in load_notes(str(tmp_path / "context"))
        if note.type != "correction"
    )
    assert owner.source_lineages == [LEGACY_LINEAGE]
    # A pre-contract citer carries neither ownership nor this pass's scoring;
    # dropping both is what makes the stored bytes read as legacy corpus.
    _rewrite_frontmatter(tmp_path, owner.name, "source_lineages:", None)
    _rewrite_frontmatter(tmp_path, owner.name, "confidence:", None)
    reloaded = next(
        note for note in load_notes(str(tmp_path / "context"))
        if note.name == owner.name
    )
    assert reloaded.source_lineages == []

    second = _v3_snapshot(
        "Correction: production launch is disabled.",
        observed="2026-07-15T13:00:00Z",
    )
    assert append_events([second], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)

    current = {note.name: note for note in load_notes(str(tmp_path / "context"))}
    updated = current[owner.name]
    assert updated.source_lineages == [LEGACY_LINEAGE]
    assert second.id in updated.provenance
    assert "Correction: production launch is disabled" in updated.description


def test_source_deletion_never_retires_an_owner_that_outgrew_its_lineage(tmp_path):
    """The retirement path enforces the atomicity its docstring claims."""
    first = _v3_snapshot(
        "Decision: production launch is enabled.", observed="2026-07-15T12:00:00Z",
    )
    assert append_events([first], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)
    context = str(tmp_path / "context")
    owner = next(
        note for note in load_notes(context) if note.type != "correction"
    )
    _rewrite_frontmatter(
        tmp_path, owner.name, "provenance:",
        f"provenance: [{first.id}, gmail:unrelated-thread@rev]",
    )
    reloaded = next(
        note for note in load_notes(context) if note.name == owner.name
    )
    assert reloaded.source_lineages == [LEGACY_LINEAGE]
    assert "gmail:unrelated-thread@rev" in reloaded.provenance

    deleted = _v3_snapshot("", deleted=True, observed="2026-07-15T13:00:00Z")
    assert append_events([deleted], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)

    after = {note.name: note for note in load_notes(context)}
    kept = after[owner.name]
    assert kept.effective_state != "superseded"
    assert kept.source_retirement is None
    correction = next(
        note for note in after.values()
        if note.name.startswith("slack-thread-deleted-")
    )
    assert f"[[supersedes:{owner.name}]]" in correction.body


def test_source_deletion_still_retires_a_confined_owner(tmp_path):
    """The guard above must not disturb the atomic retirement it protects."""
    first = _v3_snapshot(
        "Decision: production launch is enabled.", observed="2026-07-15T12:00:00Z",
    )
    assert append_events([first], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)
    owner = next(
        note for note in load_notes(str(tmp_path / "context"))
        if note.type != "correction"
    )

    deleted = _v3_snapshot("", deleted=True, observed="2026-07-15T13:00:00Z")
    assert append_events([deleted], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)

    after = {note.name: note for note in load_notes(str(tmp_path / "context"))}
    assert after[owner.name].effective_state == "superseded"
    assert after[owner.name].source_retirement["event_id"] == deleted.id
