"""Supersession lint reports graph state without materializing target flags."""
from remembrancerlib import Note, load_notes, write_note
from lint.engine import find_supersession_issues, mark_superseded
from retrieve.query import demote
from tests.trust_helpers import write_raw_legacy_note, write_reviewed_note


def make_note(name, body="", domain="eng", superseded=False, state=None, acl=None):
    return Note(name=name, description="test note", type="concept", domain=domain,
                body=body, superseded=superseded, state=state, acl=acl or [])


class TestSupersessionProjection:
    def test_verified_edge_is_reported_without_rewriting_target(self, tmp_path):
        ctx = str(tmp_path / "context")
        write_reviewed_note(tmp_path, make_note("a", state="verified"))
        write_reviewed_note(tmp_path, make_note(
            "b", body="[[supersedes:a]]", state="verified"
        ))

        assert mark_superseded(load_notes(ctx), ctx) == ["a"]
        fresh = {note.name: note for note in load_notes(ctx)}
        assert fresh["a"].superseded is False
        assert fresh["a"].state == "verified"

    def test_proposed_and_legacy_heads_are_inactive(self, tmp_path):
        ctx = str(tmp_path / "context")
        write_reviewed_note(tmp_path, make_note("old", state="verified"))
        write_note(make_note(
            "pending", body="[[supersedes:old]]", state="proposed"
        ), ctx)
        write_note(make_note("legacy", body="[[supersedes:old]]"), ctx)

        assert mark_superseded(load_notes(ctx), ctx) == []

    def test_demote_immediately_removes_active_edge(self, tmp_path):
        ctx = str(tmp_path / "context")
        old = make_note("old", state="verified")
        head = make_note("head", body="[[supersedes:old]]", state="verified")
        write_reviewed_note(tmp_path, old)
        write_reviewed_note(tmp_path, head)
        assert mark_superseded(load_notes(ctx), ctx) == ["old"]

        assert demote("head", identity="reviewer", root=str(tmp_path))["ok"]
        assert mark_superseded(load_notes(ctx), ctx) == []
        assert {note.name: note for note in load_notes(ctx)}["old"].superseded is False

    def test_explicit_retirement_is_not_created_or_cleared(self, tmp_path):
        ctx = str(tmp_path / "context")
        old = make_note("old", superseded=True, state="superseded")
        head = make_note("head", body="[[supersedes:old]]", state="proposed")
        write_raw_legacy_note(tmp_path, old)
        write_note(head, ctx)

        assert mark_superseded(load_notes(ctx), ctx) == []
        fresh = {note.name: note for note in load_notes(ctx)}
        assert fresh["old"].state == "superseded"
        assert fresh["old"].superseded is True

    def test_dangling_edge_is_ignored(self, tmp_path):
        ctx = str(tmp_path / "context")
        write_raw_legacy_note(tmp_path, make_note(
            "head", body="[[supersedes:ghost]]", state="verified"
        ))
        assert mark_superseded(load_notes(ctx), ctx) == []

    def test_returns_sorted_targets(self, tmp_path):
        ctx = str(tmp_path / "context")
        for name in ("a", "b"):
            write_reviewed_note(tmp_path, make_note(name, state="verified"))
        write_reviewed_note(tmp_path, make_note(
            "head", body="[[supersedes:b]] [[supersedes:a]]", state="verified"
        ))
        assert mark_superseded(load_notes(ctx), ctx) == ["a", "b"]

    def test_acl_incompatible_edge_is_not_reported(self, tmp_path):
        ctx = str(tmp_path / "context")
        write_reviewed_note(
            tmp_path, make_note("target", state="verified", acl=["eng"]),
        )
        write_raw_legacy_note(tmp_path, make_note(
            "head", body="[[supersedes:target]]", state="verified", acl=["secrets"]
        ))
        assert mark_superseded(load_notes(ctx), ctx) == []

    def test_legacy_parallel_heads_and_cycles_are_reported(self):
        notes = [
            make_note("target", state="verified"),
            make_note("head-a", body="[[supersedes:target]]", state="verified"),
            make_note("head-b", body="[[supersedes:target]]", state="verified"),
            make_note("cycle-a", body="[[supersedes:cycle-b]]", state="verified"),
            make_note("cycle-b", body="[[supersedes:cycle-a]]", state="verified"),
        ]

        issues = find_supersession_issues(notes)

        assert issues == []
