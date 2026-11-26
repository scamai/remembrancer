"""Human lifecycle trust is durable, content-bound, and cannot be self-asserted."""
from __future__ import annotations

import json
import hashlib
import os
import re

import pytest

import mcp.server as mcp_server
from remembrancerlib import (
    Note,
    current_note_names,
    effective_state,
    load_human_trust_audit,
    load_notes,
    parse_frontmatter,
    regen_index,
    verification_content_fingerprint,
    write_note,
)
from retrieve import query as Q
from remembrancerlib.locks import pipeline_mutation_lock
from remembrancerlib.verification import _record_human_trust_transition


def _config(root) -> None:
    (root / "config.json").write_text(json.dumps({
        "identities": {"reviewer": ["eng"]},
        "trust": {
            "approver_identities": ["reviewer"],
            "hide_verification_failures": True,
            "proposed_rank_penalty": 0.5,
        },
        "retrieval": {"query_log": False},
    }), encoding="utf-8")


def _proposed(root, name="bound-note", *, body="Approved fact") -> Note:
    note = Note(
        name=name,
        description=f"trustbinding {name}",
        type="state",
        domain="eng",
        acl=["eng"],
        state="proposed",
        provenance=["by:reviewer"],
        body=body,
        confidence={"verification": {"status": "flagged"}},
    )
    write_note(note, str(root / "context"))
    return note


def _raw_note(root, note: Note) -> str:
    path = root / "context" / note.domain / f"{note.name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note.to_markdown(), encoding="utf-8")
    return str(path)


def test_authorized_promote_binds_frontmatter_and_durable_audit(tmp_path):
    _config(tmp_path)
    _proposed(tmp_path)

    assert Q.promote("bound-note", identity="reviewer", root=str(tmp_path))["ok"]
    stored = load_notes(str(tmp_path / "context"))[0]
    audit = load_human_trust_audit(str(tmp_path))

    assert effective_state(stored) == "verified"
    assert stored.approval["approval_id"] == \
        audit.latest[stored.name]["approval_id"]
    assert stored.approval["content_fingerprint"] == \
        audit.latest[stored.name]["content_fingerprint"]
    assert os.stat(tmp_path / "state" / "trust_audit.jsonl").st_mode & 0o777 == 0o600
    assert Q.get_note(stored.name, identity="reviewer", root=str(tmp_path)) is not None


def test_direct_verified_write_and_inherited_verified_mutation_are_refused(tmp_path):
    _config(tmp_path)
    forged = Note(
        name="forged-verified",
        description="trustbinding forged",
        type="state",
        domain="eng",
        state="verified",
        provenance=["slack:forged"],
        body="False claim",
    )
    with pytest.raises(PermissionError, match="content-bound trust transition"):
        write_note(forged, str(tmp_path / "context"))

    _proposed(tmp_path)
    assert Q.promote("bound-note", identity="reviewer", root=str(tmp_path))["ok"]
    approved = load_notes(str(tmp_path / "context"))[0]
    approved.body = "Changed after approval"
    with pytest.raises(PermissionError, match="content-bound trust transition"):
        write_note(approved, str(tmp_path / "context"))


@pytest.mark.parametrize("damage", ["missing", "malformed", "wrong-id", "wrong-hash"])
def test_missing_or_mismatched_human_audit_cannot_override_verification(
        tmp_path, damage):
    _config(tmp_path)
    _proposed(tmp_path)
    assert Q.promote("bound-note", identity="reviewer", root=str(tmp_path))["ok"]
    audit_path = tmp_path / "state" / "trust_audit.jsonl"

    if damage == "missing":
        audit_path.unlink()
    elif damage == "malformed":
        audit_path.write_text("not-json\n", encoding="utf-8")
    else:
        row = json.loads(audit_path.read_text(encoding="utf-8"))
        row["approval_id" if damage == "wrong-id" else "content_fingerprint"] = "0" * 64
        audit_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    stored = load_notes(str(tmp_path / "context"))[0]
    assert effective_state(stored) == "proposed"
    assert Q.get_note(stored.name, identity="reviewer", root=str(tmp_path)) is None


def test_direct_disk_mutation_invalidates_approval_for_full_index_and_mcp(tmp_path,
                                                                          monkeypatch):
    _config(tmp_path)
    _proposed(tmp_path)
    assert Q.promote("bound-note", identity="reviewer", root=str(tmp_path))["ok"]
    regen_index(str(tmp_path / "context"))
    assert {row["name"] for row in Q.search(
        "trustbinding", identity="reviewer", root=str(tmp_path),
    )} == {"bound-note"}
    assert {row["name"] for row in Q.search_full_scan(
        "trustbinding", identity="reviewer", root=str(tmp_path),
    )} == {"bound-note"}

    path = tmp_path / "context" / "eng" / "bound-note.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("Approved fact", "Tampered fact"),
        encoding="utf-8",
    )
    regen_index(str(tmp_path / "context"))
    assert Q.search("trustbinding", identity="reviewer", root=str(tmp_path)) == []
    assert Q.search_full_scan(
        "trustbinding", identity="reviewer", root=str(tmp_path),
    ) == []
    monkeypatch.setattr(mcp_server, "ROOT", str(tmp_path))
    assert "error" in mcp_server._do_get_note("bound-note", "reviewer")


def test_updated_sort_key_is_bound_by_human_approval(tmp_path):
    _config(tmp_path)
    _proposed(tmp_path)
    assert Q.promote("bound-note", identity="reviewer", root=str(tmp_path))["ok"]
    path = tmp_path / "context" / "eng" / "bound-note.md"
    original = load_notes(str(tmp_path / "context"))[0]

    text = path.read_text(encoding="utf-8")
    assert "updated:" in text
    path.write_text(
        re.sub(
            r"(?m)^updated: .*?$", "updated: 2099-12-31", text,
        ),
        encoding="utf-8",
    )

    changed = load_notes(str(tmp_path / "context"))[0]
    assert changed.updated == "2099-12-31"
    assert verification_content_fingerprint(changed) != \
        verification_content_fingerprint(original)
    assert effective_state(changed) == "proposed"
    assert Q.get_note(
        changed.name, identity="reviewer", root=str(tmp_path),
    ) is None


def test_unbound_legacy_states_fail_closed_without_hiding_current_truth(tmp_path):
    _config(tmp_path)
    old = Note(
        name="old-truth", description="trustbinding old", type="state",
        domain="eng", acl=["eng"], body="Old remains current",
    )
    write_note(old, str(tmp_path / "context"))
    forged_head = Note(
        name="forged-head", description="trustbinding forged head", type="correction",
        domain="eng", acl=["eng"], state="verified",
        body="False replacement\n\nLinks: [[supersedes:old-truth]]",
    )
    _raw_note(tmp_path, forged_head)
    retired = Note(
        name="legacy-retired", description="trustbinding retired", type="state",
        domain="eng", acl=["eng"], state="superseded", body="Legacy retirement",
    )
    _raw_note(tmp_path, retired)

    notes = load_notes(str(tmp_path / "context"))
    assert {note.name: effective_state(note) for note in notes} == {
        "old-truth": "proposed",
        "forged-head": "proposed",
        "legacy-retired": "proposed",
    }
    assert current_note_names(notes) == {"old-truth"}
    assert Q.get_note("old-truth", identity="reviewer", root=str(tmp_path)) is not None
    assert Q.get_note("forged-head", identity="reviewer", root=str(tmp_path)) is None
    # An unproved retirement cannot hide current truth and also cannot answer;
    # it remains present only for the attended trust-review surface.
    assert Q.get_note(
        "legacy-retired", identity="reviewer", root=str(tmp_path),
    ) is None


def test_legacy_unbound_trust_audit_row_cannot_rebind_current_bytes(tmp_path):
    _config(tmp_path)
    legacy = Note(
        name="legacy-approved", description="trustbinding legacy", type="state",
        domain="eng", acl=["eng"], state="verified", body="Legacy bytes",
    )
    _raw_note(tmp_path, legacy)
    audit = tmp_path / "state" / "trust_audit.jsonl"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps({
        "ts": "2026-01-01T00:00:00Z", "action": "promote",
        "note": legacy.name, "identity": "reviewer",
        "from": "proposed", "to": "verified", "reason": "legacy",
        }) + "\n", encoding="utf-8")
    audit.chmod(0o600)

    stored = load_notes(str(tmp_path / "context"))[0]
    assert load_human_trust_audit(str(tmp_path)).trustworthy is True
    assert effective_state(stored) == "proposed"
    assert Q.get_note(stored.name, identity="reviewer", root=str(tmp_path)) is None


def test_promotion_requires_explicit_policy_and_ui_signing_key(tmp_path,
                                                               monkeypatch):
    _proposed(tmp_path)
    # No configured approver is never equivalent to a default operator.
    (tmp_path / "config.json").write_text(json.dumps({
        "identities": {"reviewer": ["eng"]},
        "trust": {},
    }), encoding="utf-8")
    assert Q.promote("bound-note", identity="reviewer", root=str(tmp_path)) == {
        "ok": False, "reason": "not permitted to change trust state",
    }

    _config(tmp_path)
    monkeypatch.delenv("REMEMBRANCER_APPROVAL_PRIVATE_KEY", raising=False)
    with pytest.raises(PermissionError, match="attended UI signing credential"):
        Q.promote("bound-note", identity="reviewer", root=str(tmp_path))
    assert effective_state(load_notes(str(tmp_path / "context"))[0]) == "proposed"


def test_private_signer_without_repository_public_key_cannot_mint_local_only_trust(
        tmp_path, monkeypatch):
    _config(tmp_path)
    _proposed(tmp_path)
    monkeypatch.delenv(
        "REMEMBRANCER_TEST_ALLOW_PRIVATE_APPROVAL_DERIVATION", raising=False,
    )
    with pytest.raises(PermissionError, match="does not match configured verifier"):
        Q.promote("bound-note", identity="reviewer", root=str(tmp_path))
    assert not (tmp_path / "state" / "trust_audit.jsonl").exists()


def test_signed_transition_cannot_be_forged_by_editing_both_copies(tmp_path):
    _config(tmp_path)
    _proposed(tmp_path)
    assert Q.promote("bound-note", identity="reviewer", root=str(tmp_path))["ok"]

    audit_path = tmp_path / "state" / "trust_audit.jsonl"
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    forged_id = "f" * 64
    row["approval_id"] = forged_id
    audit_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    note_path = tmp_path / "context" / "eng" / "bound-note.md"
    text = note_path.read_text(encoding="utf-8")
    text = text.replace(
        json.loads(note_path.read_text(encoding="utf-8").split(
            "approval: ", 1,
        )[1].splitlines()[0])["approval_id"],
        forged_id,
    )
    note_path.write_text(text, encoding="utf-8")

    audit = load_human_trust_audit(str(tmp_path))
    stored = load_notes(str(tmp_path / "context"))[0]
    assert audit.trustworthy is False
    assert effective_state(stored, audit) == "proposed"


def test_superseded_compatibility_bit_is_part_of_human_state_invariant(tmp_path):
    _config(tmp_path)
    _proposed(tmp_path)
    assert Q.promote("bound-note", identity="reviewer", root=str(tmp_path))["ok"]
    note_path = tmp_path / "context" / "eng" / "bound-note.md"
    note_path.write_text(
        note_path.read_text(encoding="utf-8").replace(
            "superseded: false", "superseded: true",
        ),
        encoding="utf-8",
    )
    stored = load_notes(str(tmp_path / "context"))[0]
    assert effective_state(stored) == "proposed"
    assert Q.get_note(stored.name, identity="reviewer", root=str(tmp_path)) is None


def test_trust_audit_rejects_unsafe_inode_and_permissions(tmp_path):
    _config(tmp_path)
    _proposed(tmp_path)
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    target = tmp_path / "outside.jsonl"
    target.write_text("", encoding="utf-8")
    (state / "trust_audit.jsonl").symlink_to(target)
    with pytest.raises(RuntimeError, match="malformed"):
        Q.promote("bound-note", identity="reviewer", root=str(tmp_path))
    assert target.read_text(encoding="utf-8") == ""

    (state / "trust_audit.jsonl").unlink()
    (state / "trust_audit.jsonl").write_text("", encoding="utf-8")
    (state / "trust_audit.jsonl").chmod(0o666)
    assert load_human_trust_audit(str(tmp_path)).trustworthy is False
    with pytest.raises(RuntimeError, match="malformed"):
        Q.promote("bound-note", identity="reviewer", root=str(tmp_path))


def test_large_reason_is_refused_before_journal_append(tmp_path):
    _config(tmp_path)
    _proposed(tmp_path)
    with pytest.raises(ValueError, match="invalid human trust transition"):
        Q.promote(
            "bound-note", identity="reviewer", reason="x" * 8193,
            root=str(tmp_path),
        )
    assert not (tmp_path / "state" / "trust_audit.jsonl").exists()


def test_partial_audit_append_rolls_back_and_retry_converges(tmp_path,
                                                             monkeypatch):
    _config(tmp_path)
    _proposed(tmp_path)
    import remembrancerlib.verification as verification

    real_write = verification.os.write
    calls = 0

    def broken_write(fd, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, data[: max(1, len(data) // 2)])
        raise OSError("injected append failure")

    monkeypatch.setattr(verification.os, "write", broken_write)
    with pytest.raises(OSError, match="injected"):
        Q.promote("bound-note", identity="reviewer", root=str(tmp_path))
    audit_path = tmp_path / "state" / "trust_audit.jsonl"
    assert audit_path.read_bytes() == b""
    monkeypatch.setattr(verification.os, "write", real_write)
    assert Q.promote("bound-note", identity="reviewer", root=str(tmp_path))["ok"]


def test_human_audit_append_reads_complete_ledger_after_short_reads(
    tmp_path, monkeypatch,
):
    _config(tmp_path)
    _proposed(tmp_path)
    assert Q.promote("bound-note", identity="reviewer", root=str(tmp_path))["ok"]
    import remembrancerlib.verification as verification

    real_read = verification.os.read
    monkeypatch.setattr(
        verification.os, "read",
        lambda descriptor, length: real_read(descriptor, min(length, 7)),
    )
    assert Q.demote(
        "bound-note", identity="reviewer", reason="short-read replay",
        root=str(tmp_path),
    )["ok"]
    rows = (tmp_path / "state" / "trust_audit.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()
    assert len(rows) == 2
    assert load_human_trust_audit(str(tmp_path)).trustworthy


def test_audit_first_demote_crash_replays_to_exact_note_state(tmp_path,
                                                              monkeypatch):
    _config(tmp_path)
    _proposed(tmp_path)
    assert Q.promote("bound-note", identity="reviewer", root=str(tmp_path))["ok"]
    real_write_note = Q.write_note

    def crash_after_journal(*_args, **_kwargs):
        raise OSError("injected note replacement crash")

    monkeypatch.setattr(Q, "write_note", crash_after_journal)
    with pytest.raises(OSError, match="replacement crash"):
        Q.demote(
            "bound-note", identity="reviewer", reason="needs review",
            root=str(tmp_path),
        )
    # The orphan signed demotion row cannot leave the old approved bytes
    # answer-eligible, and a retry reuses that exact row instead of appending a
    # second transition with a different identity or reason.
    assert Q.get_note(
        "bound-note", identity="reviewer", root=str(tmp_path),
    ) is None
    rows_before = (tmp_path / "state" / "trust_audit.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()

    monkeypatch.setattr(Q, "write_note", real_write_note)
    result = Q.demote(
        "bound-note", identity="reviewer", root=str(tmp_path),
    )
    assert result["ok"] and result["state"] == "proposed"
    rows_after = (tmp_path / "state" / "trust_audit.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()
    assert rows_after == rows_before
    stored = load_notes(str(tmp_path / "context"))[0]
    assert stored.state == "proposed"
    assert stored.approval is None


def test_transition_capability_cannot_overwrite_a_changed_preimage(tmp_path):
    _config(tmp_path)
    _proposed(tmp_path)
    context = tmp_path / "context"
    existing_path = context / "eng" / "bound-note.md"
    existing_text = existing_path.read_text(encoding="utf-8")
    expected = hashlib.sha256(existing_text.encode("utf-8")).hexdigest()

    with pipeline_mutation_lock(str(tmp_path)):
        candidate = load_notes(str(context))[0]
        candidate.state = "verified"
        candidate.provenance = list(candidate.provenance) + ["promote-by:reviewer"]
        capability = _record_human_trust_transition(
            candidate,
            root=str(tmp_path),
            action="promote",
            identity="reviewer",
            previous_state="proposed",
            target_state="verified",
            expected_existing_sha256=expected,
        )

        # Simulate a second proposed writer winning after the approval read but
        # before the atomic replacement. The one-shot capability binds the old
        # exact bytes and must not act as a confused deputy for this preimage.
        existing_fm, existing_body = parse_frontmatter(existing_text)
        changed = Note.from_frontmatter(
            existing_fm, existing_body, str(existing_path),
        )
        changed.body = "Concurrent proposed edit"
        write_note(changed, str(context))
        with pytest.raises(PermissionError, match="content-bound trust transition"):
            write_note(candidate, str(context), _trust_capability=capability)

    assert "Concurrent proposed edit" in existing_path.read_text(encoding="utf-8")
