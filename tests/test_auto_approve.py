"""Config-gated machine auto-approval is durable, content-bound, and revocable."""
from __future__ import annotations

import json

import pytest

import govern
from remembrancerlib import (
    Note,
    effective_state,
    load_notes,
    regen_index,
    verification_content_fingerprint,
    write_note,
)
from remembrancerlib.locks import pipeline_mutation_lock
from remembrancerlib.verification import (
    _auto_approval_audit_id,
    authorize_auto_approval,
)
from retrieve import query as Q
from retrieve.verification import load_verification_audit
from tests.trust_helpers import write_raw_legacy_note


def _config(root, *, gate: bool | None = None, restricted: list[str] | None = None) -> None:
    trust = {
        "approver_identities": ["reviewer"],
        "hide_verification_failures": True,
        "proposed_rank_penalty": 0.5,
    }
    if gate is not None:
        trust["auto_approve"] = gate
    config = {
        "identities": {"reviewer": ["eng", "secrets", "founders"]},
        "trust": trust,
        "retrieval": {"query_log": False},
    }
    if restricted is not None:
        config["governance"] = {"restricted_labels": restricted}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _held_note(root, name="held-note", *, acl=("eng",), body="Held fact") -> str:
    """Persist one proposed note under a governance hold and queue it."""
    note = Note(
        name=name,
        description=f"autoapprove {name}",
        type="state",
        domain="eng",
        acl=list(acl),
        state="proposed",
        provenance=["by:reviewer"],
        body=body,
        confidence={"verification": {"status": "flagged"}},
    )
    write_note(note, str(root / "context"))
    return govern.enqueue_for_review(
        note, requester="test", reason="needs review",
        root=str(root), cls="manual-review",
    )


def _approve(root, name="held-note", reason="checks-passed") -> None:
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    with pipeline_mutation_lock(str(root)):
        note = next(
            candidate
            for candidate in load_notes(str(root / "context"))
            if candidate.name == name
        )
        capability = authorize_auto_approval(
            note, cfg=cfg, audit_reason=reason,
        )
        note.state = "verified"
        write_note(
            note, str(root / "context"), _auto_capability=capability,
        )


def _stored(root, name="held-note") -> Note:
    return next(
        candidate
        for candidate in load_notes(str(root / "context"))
        if candidate.name == name
    )


def test_gate_off_refuses_authority_and_hold_stays_denied(tmp_path):
    _config(tmp_path)  # no auto_approve key at all: the default is off
    item_id = _held_note(tmp_path)

    with pytest.raises(PermissionError, match="not enabled in config"):
        _approve(tmp_path)
    assert load_verification_audit(str(tmp_path)).statuses.get("held-note") is None
    assert effective_state(_stored(tmp_path)) == "proposed"
    assert Q.get_note("held-note", identity="reviewer", root=str(tmp_path)) is None
    assert [item["id"] for item in govern.pending(str(tmp_path))] == [item_id]


def test_gate_on_approves_held_note_and_closes_queue(tmp_path):
    _config(tmp_path, gate=True)
    item_id = _held_note(tmp_path)
    # The hold denies reads until a governed approval exists.
    assert Q.get_note("held-note", identity="reviewer", root=str(tmp_path)) is None

    _approve(tmp_path)

    stored = _stored(tmp_path)
    assert effective_state(stored) == "verified"
    audit = load_verification_audit(str(tmp_path))
    assert audit.trustworthy and audit.auto_approve is True
    assert audit.statuses.get("held-note") == "verified"
    # The machine verdict is distinguishable from a human approval in the
    # audit trail and binds the current content fingerprint.
    rows = [
        json.loads(line)
        for line in (tmp_path / "state" / "verification.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    row = rows[-1]
    assert row["note"] == "held-note"
    assert row["method"] == "auto-approve"
    assert row["reason"] == "checks-passed"
    assert row["content_fingerprint"] == \
        verification_content_fingerprint(stored)
    assert row["audit_id"] == audit.audit_ids["held-note"]
    assert stored.approval is None

    assert Q.get_note("held-note", identity="reviewer", root=str(tmp_path)) is not None
    regen_index(str(tmp_path / "context"))
    assert {row["name"] for row in Q.search(
        "autoapprove", identity="reviewer", root=str(tmp_path),
    )} == {"held-note"}
    assert {row["name"] for row in Q.search_full_scan(
        "autoapprove", identity="reviewer", root=str(tmp_path),
    )} == {"held-note"}

    govern.record(str(tmp_path), item_id, "approved-auto", "auto-approve")
    assert govern.pending(str(tmp_path)) == []
    assert govern.get_item(str(tmp_path), item_id)["latest"] == "approved-auto"


def test_gate_flip_off_revokes_read_honoring(tmp_path):
    _config(tmp_path, gate=True)
    _held_note(tmp_path)
    _approve(tmp_path)
    assert Q.get_note("held-note", identity="reviewer", root=str(tmp_path)) is not None

    _config(tmp_path, gate=False)
    assert load_verification_audit(str(tmp_path)).auto_approve is False
    assert effective_state(_stored(tmp_path)) == "proposed"
    assert Q.get_note("held-note", identity="reviewer", root=str(tmp_path)) is None


@pytest.mark.parametrize("acl", [("secrets",), ("founders",), ("eng", "secrets")])
def test_restricted_labels_refused_even_with_gate_on(tmp_path, acl):
    # secrets always; founders because this config lists it.
    _config(tmp_path, gate=True, restricted=["founders"])
    _held_note(tmp_path, name="sensitive-note", acl=acl)

    with pytest.raises(PermissionError, match="never auto-approved"):
        _approve(tmp_path, name="sensitive-note")
    assert load_verification_audit(str(tmp_path)).statuses.get("sensitive-note") is None
    assert effective_state(_stored(tmp_path, name="sensitive-note")) == "proposed"
    assert Q.get_note(
        "sensitive-note", identity="reviewer", root=str(tmp_path),
    ) is None


def test_edit_after_approval_invalidates_it(tmp_path):
    _config(tmp_path, gate=True)
    _held_note(tmp_path)
    _approve(tmp_path)
    assert Q.get_note("held-note", identity="reviewer", root=str(tmp_path)) is not None

    # Any post-approval content change breaks the fingerprint binding, exactly
    # like the signed human path.
    path = tmp_path / "context" / "eng" / "held-note.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("Held fact", "Tampered fact"),
        encoding="utf-8",
    )
    regen_index(str(tmp_path / "context"))
    assert effective_state(_stored(tmp_path)) == "proposed"
    assert Q.get_note("held-note", identity="reviewer", root=str(tmp_path)) is None
    assert Q.search("autoapprove", identity="reviewer", root=str(tmp_path)) == []

    # A write through the door carrying edited content is refused even when it
    # replays the stale sealed capability.
    edited = Note(
        name="held-note",
        description="autoapprove held-note",
        type="state",
        domain="eng",
        acl=["eng"],
        state="verified",
        provenance=["by:reviewer"],
        body="Edited through the door",
    )
    with pytest.raises(PermissionError, match="content-bound trust transition"):
        write_note(edited, str(tmp_path / "context"))


def test_forged_auto_approval_without_journal_is_not_honored(tmp_path):
    _config(tmp_path, gate=True)
    forged = Note(
        name="forged-auto",
        description="autoapprove forged",
        type="state",
        domain="eng",
        acl=["eng"],
        state="verified",
        provenance=["by:reviewer"],
        body="Forged machine approval",
    )
    fingerprint = verification_content_fingerprint(forged)
    # Even a structurally perfect embedded marker (correct derived audit id
    # and fingerprint) is not authority without the journaled row.
    forged.confidence = {"verification": {
        "status": "verified",
        "method": "auto-approve",
        "content_fingerprint": fingerprint,
        "audit_id": _auto_approval_audit_id(forged.name, fingerprint),
        "checked": 1,
        "supported": 1,
        "unsupported": 0,
        "reason": "forged",
    }}
    write_raw_legacy_note(tmp_path, forged)

    assert effective_state(_stored(tmp_path, name="forged-auto")) == "proposed"
    assert Q.get_note(
        "forged-auto", identity="reviewer", root=str(tmp_path),
    ) is None

    # A hand-appended journal row without the chained rollback anchor breaks
    # the head binding: the audit fails closed and nothing is honored.
    _held_note(tmp_path)
    _approve(tmp_path)
    audit_path = tmp_path / "state" / "verification.jsonl"
    with open(audit_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "ts": "2026-01-01T00:00:00Z",
            "authority_version": 1,
            "audit_id": _auto_approval_audit_id(forged.name, fingerprint),
            "note": forged.name,
            "method": "auto-approve",
            "content_fingerprint": fingerprint,
            "checked": 1,
            "supported": 1,
            "unsupported": 0,
            "status": "verified",
            "reason": "forged",
        }) + "\n")
    audit = load_verification_audit(str(tmp_path))
    assert audit.trustworthy is False
    assert Q.get_note(
        "forged-auto", identity="reviewer", root=str(tmp_path),
    ) is None
    assert Q.get_note("held-note", identity="reviewer", root=str(tmp_path)) is None


def test_demote_covers_auto_approved_notes(tmp_path):
    _config(tmp_path, gate=True)
    _held_note(tmp_path)
    _approve(tmp_path)
    assert effective_state(_stored(tmp_path)) == "verified"

    result = Q.demote(
        "held-note", identity="reviewer", reason="back to review",
        root=str(tmp_path),
    )
    assert result["ok"] and result["state"] == "proposed"

    stored = _stored(tmp_path)
    assert stored.state == "proposed"
    assert stored.approval is None
    verification = (stored.confidence or {}).get("verification") or {}
    assert verification.get("status") == "skipped"
    assert verification.get("method") is None
    assert effective_state(stored) == "proposed"
    # Demotion invalidates the machine verdict: the stale audit row can no
    # longer make the changed draft answer-eligible, and the durable
    # governance hold applies again.
    assert Q.get_note("held-note", identity="reviewer", root=str(tmp_path)) is None
