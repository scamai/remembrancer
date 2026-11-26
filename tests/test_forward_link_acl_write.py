"""Wiki-link names obey the same ACL boundary as their target notes."""
from __future__ import annotations

import json

import pytest

from remembrancerlib import Note, load_notes, write_note
from retrieve import query as Q
from tests.trust_helpers import configure_approver, write_raw_legacy_note


def _note(name: str, *, acl=None, body: str = "body") -> Note:
    return Note(
        name=name,
        description=f"description for {name}",
        type="state",
        domain="eng",
        acl=list(acl or []),
        state="proposed",
        body=body,
    )


def test_write_door_rejects_outgoing_link_that_reveals_restricted_target(tmp_path):
    context = str(tmp_path / "context")
    write_note(_note("secret-roadmap", acl=["secrets"]), context)

    with pytest.raises(ValueError, match="would reveal restricted target"):
        write_note(
            _note("engineering-page", acl=["eng"], body="See [[secret-roadmap]]."),
            context,
        )

    assert {note.name for note in load_notes(context)} == {"secret-roadmap"}


def test_write_door_allows_acl_safe_and_truly_dangling_links(tmp_path):
    context = str(tmp_path / "context")
    write_note(_note("secret-roadmap", acl=["secrets"]), context)
    write_note(
        _note("secret-index", acl=["secrets"], body="See [[secret-roadmap]]."),
        context,
    )
    write_note(
        _note("future-index", acl=["eng"], body="Reserved [[not-created-yet]]."),
        context,
    )

    assert {note.name for note in load_notes(context)} == {
        "future-index", "secret-index", "secret-roadmap",
    }


def test_target_creation_cannot_turn_existing_orphan_into_acl_disclosure(tmp_path):
    context = str(tmp_path / "context")
    write_note(
        _note("public-plan", acl=["eng"], body="Future [[secret-roadmap]]."),
        context,
    )

    with pytest.raises(ValueError, match="would reveal restricted target"):
        write_note(_note("secret-roadmap", acl=["secrets"]), context)

    assert {note.name for note in load_notes(context)} == {"public-plan"}


def test_target_acl_cannot_be_narrowed_past_existing_inbound_link(tmp_path):
    context = str(tmp_path / "context")
    write_note(_note("roadmap"), context)
    write_note(
        _note("engineering-page", acl=["eng"], body="See [[roadmap]]."),
        context,
    )
    roadmap = next(note for note in load_notes(context) if note.name == "roadmap")
    roadmap.acl = ["secrets"]

    with pytest.raises(ValueError, match="would reveal restricted target"):
        write_note(roadmap, context)

    stored = next(note for note in load_notes(context) if note.name == "roadmap")
    assert stored.acl == []


def test_create_page_inherits_shared_forward_link_acl_guard(tmp_path):
    context = str(tmp_path / "context")
    (tmp_path / "config.json").write_text(json.dumps({
        "identities": {"alice": ["eng"]},
        "retrieval": {"file_back": {"enabled": True, "min_chars": 10}},
    }), encoding="utf-8")
    write_note(_note("secret-roadmap", acl=["secrets"]), context)

    with pytest.raises(ValueError, match="would reveal restricted target"):
        Q.create_page(
            "Engineering plan",
            "This engineering page is long enough and links to [[secret-roadmap]].",
            domain="eng",
            identity="alice",
            root=str(tmp_path),
        )


def test_promotion_rejects_legacy_unsafe_link_before_signed_audit(tmp_path):
    context = str(tmp_path / "context")
    write_note(_note("secret-roadmap", acl=["secrets"]), context)
    write_raw_legacy_note(
        tmp_path,
        _note("legacy-page", acl=["eng"], body="See [[secret-roadmap]]."),
    )
    configure_approver(tmp_path, "reviewer")
    audit_path = tmp_path / "state" / "trust_audit.jsonl"
    before = audit_path.read_bytes() if audit_path.exists() else b""

    result = Q.promote("legacy-page", identity="reviewer", root=str(tmp_path))

    assert result["ok"] is False
    assert "would reveal restricted target" in result["reason"]
    assert (audit_path.read_bytes() if audit_path.exists() else b"") == before

