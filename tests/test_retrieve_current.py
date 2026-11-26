"""Tests for edge-aware retrieval: current_only hides superseded notes.

Note B has a [[supersedes:A]] typed-link in its body.
search(current_only=True) must exclude A (the superseded target) but include B.
"""
import os
import tempfile
import json

import pytest

from remembrancerlib import Note, write_note
from retrieve.query import get_note, promote, search


@pytest.fixture
def supersede_remembrancer():
    """Temp Remembrancer layout with two notes where B supersedes A.

    - note-a: the older note (will be superseded)
    - note-b: has [[supersedes:note-a]] in body (the current version)
    """
    with tempfile.TemporaryDirectory() as tmp:
        context_dir = os.path.join(tmp, "context")
        os.makedirs(context_dir, exist_ok=True)
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "identities": {"reviewer": []},
                "trust": {"approver_identities": ["reviewer"]},
            }, f)

        note_a = Note(
            name="note-a",
            description="Old onboarding process documentation",
            domain="company",
            acl=[],
            body="This is the old onboarding process.",
        )
        write_note(note_a, context_dir)

        note_b = Note(
            name="note-b",
            description="Updated onboarding process documentation",
            domain="company",
            acl=[],
            body="This is the new onboarding process.\n\n[[supersedes:note-a]]",
            state="proposed",
        )
        write_note(note_b, context_dir)
        result = promote("note-b", identity="reviewer", root=tmp)
        assert result["ok"], result

        yield tmp


def test_current_only_excludes_superseded(supersede_remembrancer):
    """search(current_only=True) must exclude the superseded target (note-a)."""
    hits = search("onboarding", current_only=True, root=supersede_remembrancer)
    names = {h["name"] for h in hits}
    assert "note-a" not in names, "note-a is superseded and must be hidden with current_only=True"
    assert "note-b" in names, "note-b is the superseding note and must appear"


def test_hide_superseded_default_hides_edge_target(supersede_remembrancer):
    """Retrieval contract: even with current_only=False, the DEFAULT (hide_superseded=True)
    drops a supersedes-edge target (note-a) so folded shards stop diluting the slice; the
    superseding note (note-b) stays."""
    hits = search("onboarding", current_only=False, root=supersede_remembrancer)
    names = {h["name"] for h in hits}
    assert "note-a" not in names, "superseded edge-target must be hidden by default"
    assert "note-b" in names


def test_hide_superseded_false_restores_legacy(supersede_remembrancer):
    """Opt out via retrieval.hide_superseded:false → exactly the earlier behavior (both surface)."""
    import json
    with open(os.path.join(supersede_remembrancer, "config.json"), "w") as f:
        json.dump({"identities": {}, "retrieval": {"hide_superseded": False}}, f)
    hits = search("onboarding", current_only=False, root=supersede_remembrancer)
    names = {h["name"] for h in hits}
    assert "note-a" in names and "note-b" in names


def test_no_supersedes_links_unaffected(tmp_remembrancer):
    """current_only=True on a graph with no supersedes edges returns all matching notes."""
    # tmp_remembrancer has public-note and ml-only-note, neither supersedes the other
    hits = search("detector", current_only=True, root=tmp_remembrancer)
    names = {h["name"] for h in hits}
    assert "public-note" in names
    assert "ml-only-note" in names


def test_current_only_superseder_not_excluded(supersede_remembrancer):
    """The note doing the superseding (note-b) must NOT be excluded."""
    hits = search("onboarding", current_only=True, root=supersede_remembrancer)
    names = {h["name"] for h in hits}
    assert "note-b" in names


def test_get_note_uses_current_projection_unless_history_requested(supersede_remembrancer):
    assert get_note("note-a", root=supersede_remembrancer) is None
    historical = get_note("note-a", root=supersede_remembrancer, include_history=True)
    assert historical and historical["name"] == "note-a"
    assert get_note("note-b", root=supersede_remembrancer)["name"] == "note-b"
