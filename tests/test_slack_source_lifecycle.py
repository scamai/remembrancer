"""Closed Slack deletion/restoration flows through governed source lifecycle."""
from __future__ import annotations

from distill import engine
from ingest.slack_contract import snapshot_revision as snapshot_revision_v2
from ingest.slack_contract_v3 import (
    snapshot_revision,
    source_occurred_at,
)
from remembrancerlib import Event, Note, append_events, load_notes
from remembrancerlib import source_lifecycle as lifecycle
from remembrancerlib.verification import verification_content_fingerprint
from tests.event_validation_helpers import guarded_run


def _snapshot(
    text: str,
    *,
    deleted: bool = False,
    observed: str,
    channel: str = "engineering",
    channel_id: str = "C1",
) -> Event:
    thread_ts = "30.000"
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
            "deletion_evidence": {
                "kind": "api-error", "code": "message_not_found",
            },
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
        "channel": channel,
        "channel_id": channel_id,
        "ts": thread_ts,
        "thread_ts": thread_ts,
        "permalink": None,
    }
    revision = snapshot_revision(
        author="alice",
        occurred_at=occurred_at,
        payload=payload,
        provenance=provenance,
        acl=["eng"],
        entities=[],
    )
    provenance["revision"] = revision
    return Event(
        id=f"slack:{channel}/{thread_ts}@{revision}",
        source="slack",
        author="alice",
        occurred_at=occurred_at,
        ingested_at=observed,
        payload=payload,
        acl=["eng"],
        provenance=provenance,
    )


def _legacy_v2_deletion(*, observed: str) -> Event:
    channel = "engineering"
    thread_ts = "30.000"
    payload = {
        "text": f"[#{channel}] Slack thread {thread_ts} was deleted at the source.",
        "reply_count": 0,
        "message_count": 0,
        "snapshot_complete": True,
        "snapshot_format": "thread-v2",
        "resolution_reactions": [],
        "signal_eligible": False,
        "deleted": True,
        "subtype": "thread_deleted",
        "deletion_evidence": {
            "kind": "api-error", "code": "message_not_found",
        },
    }
    revision = snapshot_revision_v2("slack", payload, ["eng"])
    return Event(
        id=f"slack:{channel}/{thread_ts}@{revision}",
        source="slack",
        author="slack",
        occurred_at=source_occurred_at(thread_ts),
        ingested_at=observed,
        payload=payload,
        acl=["eng"],
        provenance={
            "channel": channel,
            "channel_id": "C1",
            "ts": thread_ts,
            "thread_ts": thread_ts,
            "revision": revision,
            "permalink": None,
        },
    )


def _legacy_v2_snapshot(text: str, *, observed: str) -> Event:
    channel = "engineering"
    thread_ts = "30.000"
    payload = {
        "text": f"[#{channel}] {text}",
        "reply_count": 0,
        "message_count": 1,
        "snapshot_complete": True,
        "snapshot_format": "thread-v2",
        "resolution_reactions": [],
        "signal_eligible": True,
    }
    revision = snapshot_revision_v2("alice", payload, ["eng"])
    return Event(
        id=f"slack:{channel}/{thread_ts}@{revision}",
        source="slack",
        author="alice",
        occurred_at=source_occurred_at(thread_ts),
        ingested_at=observed,
        payload=payload,
        acl=["eng"],
        provenance={
            "channel": channel,
            "channel_id": "C1",
            "ts": thread_ts,
            "thread_ts": thread_ts,
            "revision": revision,
            "permalink": None,
        },
    )


def test_slack_delete_retires_owner_and_late_live_snapshot_cannot_restore(tmp_path):
    initial = _snapshot(
        "Decision: production launch is enabled.",
        observed="2026-07-15T12:00:00Z",
    )
    assert append_events([initial], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)
    first = load_notes(str(tmp_path / "context"))
    source_note = next(note for note in first if note.type != "correction")
    assert source_note.source_lineages == ["slack:C1:30.000"]

    deleted = _snapshot("", deleted=True, observed="2026-07-15T13:00:00Z")
    assert append_events([deleted], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)
    after_delete = {note.name: note for note in load_notes(str(tmp_path / "context"))}
    retired = after_delete[source_note.name]
    assert retired.effective_state == "superseded"
    assert retired.source_retirement["event_id"] == deleted.id
    assert retired.source_retirement["producer"] == "slack-thread-v3-deletion-v1"
    assert not any(note.type == "correction" for note in after_delete.values())

    restored = _snapshot(
        "Correction: production launch is disabled.",
        observed="2026-07-15T14:00:00Z",
    )
    assert append_events([restored], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)
    after_restore = {note.name: note for note in load_notes(str(tmp_path / "context"))}
    assert after_restore[source_note.name].effective_state == "superseded"
    live_owners = [
        note for note in after_restore.values()
        if note.effective_state != "superseded"
        and note.source_lineages == ["slack:C1:30.000"]
    ]
    assert live_owners == []
    assert not any(restored.id in note.provenance for note in after_restore.values())
    history = lifecycle._validated_transition_history(
        str(tmp_path),
        deleted,
        "slack:C1:30.000",
        "slack-thread-v3-deletion-v1",
        "retire",
    )
    assert {event.id for event in history} == {initial.id, deleted.id, restored.id}


def test_malformed_slack_deletion_never_enters_source_lifecycle(tmp_path):
    event = Event(
        id="slack:engineering/30.000@forged",
        source="slack",
        author="alice",
        payload={"deleted": True, "text": "forged deletion"},
        provenance={
            "channel": "engineering", "channel_id": "C1",
            "thread_ts": "30.000", "ts": "30.000",
        },
    )
    assert engine._is_slack_tombstone(event) is False


def test_legacy_v2_deletion_is_consumed_without_retiring_memory(tmp_path):
    initial = _snapshot(
        "Decision: production launch is enabled.",
        observed="2026-07-15T12:00:00Z",
    )
    assert append_events([initial], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)
    source_note = next(
        note for note in load_notes(str(tmp_path / "context"))
        if note.type != "correction"
    )

    legacy_deletion = _legacy_v2_deletion(observed="2026-07-15T13:00:00Z")
    assert append_events([legacy_deletion], str(tmp_path / "events")) == 1
    summary = guarded_run(engine, root=str(tmp_path), use_llm=False)

    after = {note.name: note for note in load_notes(str(tmp_path / "context"))}
    assert engine._is_slack_tombstone(legacy_deletion) is False
    assert summary["slack_non_authoritative_deletions_dropped"] == 1
    assert after[source_note.name].effective_state != "superseded"
    assert after[source_note.name].source_retirement is None
    assert not any(note.type == "correction" for note in after.values())


def test_v3_deletion_cannot_retire_owner_from_another_lineage(tmp_path):
    initial = _snapshot(
        "Decision: production launch is enabled.",
        observed="2026-07-15T12:00:00Z",
    )
    assert append_events([initial], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)
    source_note = next(
        note for note in load_notes(str(tmp_path / "context"))
        if note.type != "correction"
    )

    other_lineage_deletion = _snapshot(
        "",
        deleted=True,
        observed="2026-07-15T13:00:00Z",
        channel="security",
        channel_id="C2",
    )
    assert append_events([other_lineage_deletion], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)

    after = {note.name: note for note in load_notes(str(tmp_path / "context"))}
    assert after[source_note.name].effective_state != "superseded"
    assert after[source_note.name].source_retirement is None
    corrections = [note for note in after.values() if note.type == "correction"]
    assert len(corrections) == 1
    assert corrections[0].source_lineages == []


def test_first_v3_deletion_preserves_v2_owner_and_routes_review(tmp_path):
    legacy = _legacy_v2_snapshot(
        "Decision: production launch is enabled.",
        observed="2026-07-15T12:00:00Z",
    )
    assert append_events([legacy], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)
    source_note = next(
        note for note in load_notes(str(tmp_path / "context"))
        if note.type != "correction"
    )

    deletion = _snapshot("", deleted=True, observed="2026-07-15T13:00:00Z")
    assert append_events([deletion], str(tmp_path / "events")) == 1
    guarded_run(engine, root=str(tmp_path), use_llm=False)

    after = {note.name: note for note in load_notes(str(tmp_path / "context"))}
    assert after[source_note.name].effective_state != "superseded"
    assert after[source_note.name].source_retirement is None
    corrections = [note for note in after.values() if note.type == "correction"]
    assert len(corrections) == 1
    assert f"[[supersedes:{source_note.name}]]" in corrections[0].body


def test_legacy_v2_lifecycle_proof_is_parseable_but_no_longer_active():
    note = Note(
        name="legacy-slack-owner",
        description="Legacy Slack memory",
        type="state",
        domain="eng",
        state="superseded",
        superseded=True,
    )
    content_fingerprint = verification_content_fingerprint(note)
    material = lifecycle._retirement_material(
        note=note.name,
        content_fingerprint=content_fingerprint,
        source="slack",
        lineage="slack:C1:30.000",
        event_id="slack:engineering/30.000@legacy",
        event_fingerprint="a" * 64,
        producer="slack-thread-deletion-v1",
    )
    note.source_retirement = {
        "version": lifecycle.SOURCE_RETIREMENT_VERSION,
        "retirement_id": lifecycle._canonical_hash(material),
        "content_fingerprint": content_fingerprint,
        "source": "slack",
        "lineage": "slack:C1:30.000",
        "event_id": "slack:engineering/30.000@legacy",
        "event_fingerprint": "a" * 64,
        "producer": "slack-thread-deletion-v1",
    }
    note._source_retirement_audit_match = True

    assert lifecycle._valid_material(material) is True
    assert lifecycle.source_retirement_valid(note) is False
    assert note.effective_state == "proposed"
