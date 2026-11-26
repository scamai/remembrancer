import json

from remembrancerlib import (
    Note, load_notes, verification_content_fingerprint, write_note,
)
from retrieve import query as Q
from retrieve.verification import VerificationAudit, verification_status
from tools.generated_notes import (
    GeneratedGrounding, generated_grounding, persist_generated_note,
    producer_grounding_terminal, producer_owned,
)


def _generated(body="machine v1"):
    return Note(name="fixed-page", description="generated", domain="eng", body=body)


def _write_verified(tmp_path, note, identity="alice"):
    grants = list(note.acl or [])
    (tmp_path / "config.json").write_text(json.dumps({
        "identities": {identity: grants},
        "trust": {"approver_identities": [identity]},
    }))
    note.state = "proposed"
    path = write_note(note, str(tmp_path / "context"))
    result = Q.promote(note.name, identity=identity, root=str(tmp_path))
    assert result["ok"], result
    return path


def test_producer_token_matches_the_central_verifier_contract():
    import pytest
    from tools.generated_notes import producer_marker

    assert producer_marker("stable-producer-2") == "generated-by:stable-producer-2"
    for invalid in ("", "Uppercase", "has space", "has_underscore", "has:colon"):
        with pytest.raises(ValueError, match="must match"):
            producer_marker(invalid)


def test_generator_refreshes_only_its_explicit_pending_note(tmp_path):
    context = str(tmp_path / "context")
    first = persist_generated_note(
        _generated(), producer="test-producer", context_dir=context,
    )
    notes = {note.name: note for note in load_notes(context)}
    second = persist_generated_note(
        _generated("machine v2"), producer="test-producer", context_dir=context,
        notes_by_name=notes,
    )

    assert first.action == "CREATE"
    assert second.action == "UPDATE"
    stored = {note.name: note for note in load_notes(context)}["fixed-page"]
    assert producer_owned(stored, "test-producer")
    assert stored.body == "machine v2\n"


def test_generator_collision_with_verified_human_becomes_proposal(tmp_path):
    context = str(tmp_path / "context")
    human = Note(
        name="fixed-page", description="human", domain="eng", state="verified",
        provenance=["by:alice"], body="HUMAN BYTES",
    )
    path = _write_verified(tmp_path, human)
    before = open(path, "rb").read()

    result = persist_generated_note(
        _generated("machine replacement"), producer="test-producer",
        context_dir=context,
    )

    assert open(path, "rb").read() == before
    assert result.target_name == "fixed-page"
    proposal = {note.name: note for note in load_notes(context)}[result.note_name]
    assert proposal.state == "proposed"
    assert ("supersedes", "fixed-page") in proposal.typed_links
    assert "generated-by:test-producer" in proposal.provenance
    assert not any(item.startswith("by:") for item in proposal.provenance)


def test_other_producer_marker_does_not_grant_overwrite(tmp_path):
    context = str(tmp_path / "context")
    existing = _generated("other producer")
    existing.state = "proposed"
    existing.provenance = ["generated-by:other"]
    path = write_note(existing, context)
    before = open(path, "rb").read()

    result = persist_generated_note(
        _generated("new producer"), producer="test-producer", context_dir=context,
    )

    assert open(path, "rb").read() == before
    assert result.target_name == "fixed-page"


def test_mixed_producer_markers_are_not_owned(tmp_path):
    note = _generated()
    note.state = "proposed"
    note.provenance = [
        "generated-by:test-producer", "generated-by:other",
    ]
    assert producer_owned(note, "test-producer") is False


def test_same_producer_without_explicit_proposed_state_is_not_mutable(tmp_path):
    context = str(tmp_path / "context")
    legacy = _generated("legacy")
    legacy.state = None
    legacy.provenance = ["generated-by:test-producer"]
    path = write_note(legacy, context)
    before = open(path, "rb").read()

    result = persist_generated_note(
        _generated("replacement"), producer="test-producer", context_dir=context,
    )

    assert open(path, "rb").read() == before
    assert result.target_name == legacy.name


def test_generated_collision_combines_acl_or_fails_closed(tmp_path):
    context = str(tmp_path / "context")
    restricted = Note(
        name="fixed-page", description="secret", domain="eng",
        acl=["secrets"], state="verified", body="secret",
    )
    _write_verified(tmp_path, restricted)
    desired = _generated("eng aggregate")
    desired.acl = ["eng"]

    import pytest
    from remembrancerlib import ACLConflict
    with pytest.raises(ACLConflict):
        persist_generated_note(
            desired, producer="test-producer", context_dir=context,
        )


def test_approved_generated_head_receives_next_reviewable_revision(tmp_path):
    context = str(tmp_path / "context")
    human = Note(
        name="fixed-page", description="human", domain="eng", state="verified",
        provenance=["by:alice"], body="human",
    )
    _write_verified(tmp_path, human)
    first = persist_generated_note(
        _generated("machine v1"), producer="test-producer", context_dir=context,
    )
    notes = {note.name: note for note in load_notes(context)}
    approved = notes[first.note_name]
    result = Q.promote(approved.name, identity="alice", root=str(tmp_path))
    assert result["ok"], result

    second = persist_generated_note(
        _generated("machine v2"), producer="test-producer", context_dir=context,
    )
    third = persist_generated_note(
        _generated("machine v3"), producer="test-producer", context_dir=context,
    )

    stored = {note.name: note for note in load_notes(context)}
    assert second.target_name == approved.name
    assert ("supersedes", approved.name) in stored[second.note_name].typed_links
    assert stored[second.note_name].state == "proposed"
    assert third.action == "UPDATE"
    assert third.note_name == second.note_name
    assert stored[third.note_name].body.startswith("machine v3")
    assert len([
        note for note in stored.values()
        if ("supersedes", approved.name) in note.typed_links
    ]) == 1


def test_unattested_generated_write_is_explicit_content_bound_review(tmp_path):
    context = str(tmp_path / "context")

    persist_generated_note(
        _generated(), producer="test-producer", context_dir=context,
    )

    stored = load_notes(context)[0]
    verification = stored.confidence["verification"]
    assert verification["status"] == "skipped"
    assert verification["grounded"] is False
    assert verification["producer"] == "test-producer"
    assert verification["method"] == "producer-grounded-v1:unattested"
    assert verification["content_fingerprint"] == verification_content_fingerprint(stored)
    assert verification["review_required"] is True
    assert producer_grounding_terminal(stored) == "review"


def test_caller_claimed_deterministic_grounding_cannot_mint_verified(tmp_path):
    context = str(tmp_path / "context")
    proof = generated_grounding(
        "fixture-record",
        {"record": {"id": 7, "value": "machine v1"}},
        grounded=True,
        reason="exact deterministic fixture rendering",
    )
    first = persist_generated_note(
        _generated(), producer="test-producer", context_dir=context,
        grounding=proof,
    )
    initial = load_notes(context)[0]
    before = initial.confidence
    second = persist_generated_note(
        _generated(), producer="test-producer", context_dir=context,
        grounding=proof,
    )
    stored = load_notes(context)[0]

    assert first.action == "CREATE"
    assert second.action == "IGNORE"
    assert stored.confidence == before
    # ``sources`` is caller-controlled data, not source authority.  A generic
    # producer must not be able to self-attest simply by choosing grounded=True.
    assert stored.confidence["verification"]["status"] == "skipped"
    assert stored.confidence["verification"]["grounded"] is False
    assert producer_grounding_terminal(stored) == "review"


def test_terminal_check_cannot_outvote_the_central_producer_validator(tmp_path):
    context = str(tmp_path / "context")
    proof = generated_grounding(
        "fixture-record", {"record": "trusted"}, grounded=True,
        reason="exact deterministic fixture rendering",
    )
    persist_generated_note(
        _generated(), producer="test-producer", context_dir=context,
        grounding=proof,
    )
    stored = load_notes(context)[0]
    # The derived audit id and content hash still match, but this claim-count
    # envelope is impossible even for a review-only result. Retrieval rejects it, so
    # backlog classification must reject it too rather than stranding the note.
    stored.confidence["verification"]["supported"] = 1

    assert producer_grounding_terminal(stored) is None


def test_content_change_rebinds_proof_and_default_refresh_cannot_reuse_verified(tmp_path):
    context = str(tmp_path / "context")
    proof = generated_grounding(
        "fixture-record", {"record": "v1"}, grounded=True,
        reason="exact deterministic fixture rendering",
    )
    persist_generated_note(
        _generated("machine v1"), producer="test-producer", context_dir=context,
        grounding=proof,
    )
    old = load_notes(context)[0]
    old_verification = old.confidence["verification"]

    persist_generated_note(
        _generated("machine v2"), producer="test-producer", context_dir=context,
    )
    current = load_notes(context)[0]
    current_verification = current.confidence["verification"]

    assert current_verification["status"] == "skipped"
    assert current_verification["content_fingerprint"] != old_verification["content_fingerprint"]
    assert current_verification["audit_id"] != old_verification["audit_id"]
    # A stale successful audit for the prior bytes cannot resurrect this
    # explicit current-content failure.
    stale_audit = VerificationAudit(
        {current.name: "verified"},
        {current.name: old_verification["content_fingerprint"]},
        {current.name: old_verification["audit_id"]},
    )
    assert verification_status(current, stale_audit) == "skipped"


def test_noncanonical_grounding_sources_fail_closed(tmp_path):
    context = str(tmp_path / "context")
    proof = generated_grounding(
        "fixture-record", {"not-json": object()}, grounded=True,
        reason="caller claimed deterministic rendering",
    )

    persist_generated_note(
        _generated(), producer="test-producer", context_dir=context,
        grounding=proof,
    )

    verification = load_notes(context)[0].confidence["verification"]
    assert verification["status"] == "skipped"
    assert verification["grounded"] is False
    assert verification["reason"] == "grounding source material is not canonical JSON"


def test_forged_grounding_capability_cannot_mint_success(tmp_path):
    context = str(tmp_path / "context")
    forged = GeneratedGrounding(
        source_method="contradiction-detector",
        sources={"detector": "fabricated", "pair": ["alpha", "beta"]},
        grounded=True,
        reason="caller forged every public field",
        capability={"_seal": object()},
    )

    persist_generated_note(
        _generated(), producer="contradiction-lint", context_dir=context,
        grounding=forged,
    )

    verification = load_notes(context)[0].confidence["verification"]
    assert verification["status"] == "skipped"
    assert verification["grounded"] is False


def test_collision_proposal_proof_binds_final_name_and_supersedes_link(tmp_path):
    context = str(tmp_path / "context")
    _write_verified(tmp_path, Note(
        name="fixed-page", description="human", domain="eng", state="verified",
        provenance=["by:alice"], body="human truth",
    ))
    proof = generated_grounding(
        "fixture-record", {"record": "replacement"}, grounded=True,
        reason="exact deterministic fixture rendering",
    )

    result = persist_generated_note(
        _generated("machine replacement"), producer="test-producer",
        context_dir=context, grounding=proof,
    )

    proposal = {note.name: note for note in load_notes(context)}[result.note_name]
    assert result.note_name != "fixed-page"
    assert ("supersedes", "fixed-page") in proposal.typed_links
    assert proposal.confidence is not None
    assert (proposal.confidence["verification"]["content_fingerprint"]
            == verification_content_fingerprint(proposal))
