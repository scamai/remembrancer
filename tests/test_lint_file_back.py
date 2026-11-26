import os

from lint.engine import file_back_links, run
from remembrancerlib import Note, load_notes
from tests.trust_helpers import write_reviewed_note
from tests.trust_helpers import write_raw_legacy_note


def _persist(context: str, note: Note) -> str:
    # File-back tests intentionally build legacy/unsafe link states as fixtures.
    return write_raw_legacy_note(os.path.dirname(context), note)


def _note(name, *, body="", acl=None):
    return Note(
        name=name, description=f"note {name}", type="project", domain="eng",
        body=body, acl=acl or [],
    )


def test_file_back_adds_untyped_reciprocal_link(tmp_path):
    context = str(tmp_path / "context")
    source = _note("source", body="[[supports:target]]")
    target = _note("target")
    _persist(context, source)
    _persist(context, target)

    result = file_back_links([source, target], context)

    fresh = {note.name: note for note in load_notes(context)}
    assert result["written"] == ["target"]
    assert ("relates", "source") in fresh["target"].typed_links
    assert "[[supports:source]]" not in fresh["target"].body


def test_file_back_is_idempotent(tmp_path):
    context = str(tmp_path / "context")
    source = _note("source", body="[[target]]")
    target = _note("target")
    _persist(context, source)
    _persist(context, target)

    first = file_back_links([source, target], context)
    second = file_back_links(load_notes(context), context)

    assert first["written"] == ["target"]
    assert second["written"] == []
    fresh = {note.name: note for note in load_notes(context)}
    assert fresh["target"].body.count("[[source]]") == 1


def test_file_back_disabled_by_agents_convention(tmp_path):
    context = str(tmp_path / "context")
    _persist(context, _note("source", body="[[target]]"))
    _persist(context, _note("target"))
    (tmp_path / "AGENTS.md").write_text(
        "```conventions\nfile_back_enabled: false\n```\n", encoding="utf-8"
    )

    summary = run(root=str(tmp_path))

    fresh = {note.name: note for note in load_notes(context)}
    assert summary["file_back"]["enabled"] is False
    assert "[[source]]" not in fresh["target"].body


def test_file_back_never_leaks_restricted_source_name_to_public_target(tmp_path):
    context = str(tmp_path / "context")
    source = _note("secret-source", body="[[public-target]]", acl=["secrets"])
    target = _note("public-target")
    _persist(context, source)
    _persist(context, target)

    result = file_back_links([source, target], context)

    fresh = {note.name: note for note in load_notes(context)}
    assert result["written"] == []
    assert result["skipped_acl"] == [("secret-source", "public-target")]
    assert "secret-source" not in fresh["public-target"].body


def test_file_back_allows_public_source_name_in_restricted_target(tmp_path):
    context = str(tmp_path / "context")
    source = _note("public-source", body="[[secret-target]]")
    target = _note("secret-target", acl=["secrets"])
    _persist(context, source)
    _persist(context, target)

    result = file_back_links([source, target], context)

    fresh = {note.name: note for note in load_notes(context)}
    assert result["written"] == ["secret-target"]
    assert "[[public-source]]" in fresh["secret-target"].body


def test_file_back_acl_subset_rule_for_or_grants(tmp_path):
    context = str(tmp_path / "context")
    safe_source = _note("safe-source", body="[[eng-target]]", acl=["eng", "secrets"])
    unsafe_source = _note("unsafe-source", body="[[mixed-target]]", acl=["eng"])
    eng_target = _note("eng-target", acl=["eng"])
    mixed_target = _note("mixed-target", acl=["eng", "customers"])
    notes = [safe_source, unsafe_source, eng_target, mixed_target]
    for note in notes:
        _persist(context, note)

    result = file_back_links(notes, context)

    fresh = {note.name: note for note in load_notes(context)}
    assert "[[safe-source]]" in fresh["eng-target"].body
    assert "unsafe-source" not in fresh["mixed-target"].body
    assert ("unsafe-source", "mixed-target") in result["skipped_acl"]


def test_file_back_write_cap_converges_across_passes(tmp_path):
    context = str(tmp_path / "context")
    notes = []
    for index in range(3):
        source = _note(f"source-{index}", body=f"[[target-{index}]]")
        target = _note(f"target-{index}")
        notes.extend((source, target))
        _persist(context, source)
        _persist(context, target)

    first = file_back_links(notes, context, max_writes=2)
    second = file_back_links(load_notes(context), context, max_writes=2)

    assert len(first["written"]) == 2
    assert first["deferred"] == 1
    assert len(second["written"]) == 1
    assert second["deferred"] == 0


def test_file_back_to_verified_note_is_a_reviewable_replacement(tmp_path):
    context = str(tmp_path / "context")
    source = _note("source", body="[[target]]")
    target = _note("target")
    source.state = "verified"
    target.state = "verified"
    write_reviewed_note(tmp_path, source)
    target_path, _reviewed = write_reviewed_note(tmp_path, target)
    current = {note.name: note for note in load_notes(context)}
    source, target = current[source.name], current[target.name]
    before = open(target_path, "rb").read()

    first = file_back_links([source, target], context)
    second = file_back_links(load_notes(context), context)

    assert open(target_path, "rb").read() == before
    assert first["review_required_links"] == 1
    assert len(first["review_proposals"]) == 1
    proposal_name = first["review_proposals"][0]["proposal"]
    proposal = {note.name: note for note in load_notes(context)}[proposal_name]
    assert proposal.state == "proposed"
    assert ("supersedes", "target") in proposal.typed_links
    assert ("relates", "source") in proposal.typed_links
    assert second["written"] == []
    assert second["review_required_links"] == 1


def test_review_held_target_does_not_crash_file_back(tmp_path, monkeypatch):
    """Regression: a governance-held target's PermissionError from
    write_note must route to the review queue, not kill the lint stage."""
    import lint.engine as engine
    context = str(tmp_path / "context")
    source = _note("source", body="[[supports:target]]")
    target = _note("target")
    _persist(context, source)
    _persist(context, target)

    real_write = engine.write_note
    def held_write(note, ctx, **kw):
        if note.name == "target":
            raise PermissionError("review-held proposal rewrite requires an "
                                  "exact governance capability and requeue")
        return real_write(note, ctx, **kw)
    monkeypatch.setattr(engine, "write_note", held_write)

    result = engine.file_back_links([source, target], context)
    assert "target" not in result["written"]
    assert result["review_required_links"] >= 1


# --- one blocked target must not abort the corpus pass ------------------------
#
# `_propose_replacement` refuses when the target already has an unretired head
# that is not a machine update — a second competing replacement would be wrong,
# so the refusal is correct. Letting it propagate is not: when a note such as
# `weekly-sync` is targeted by a CALENDAR note whose provenance is a calendar
# event rather than a machine-update marker, the ValueError escapes lint and
# `exit 1` discards the whole pipeline run, including a digest that has already
# merged its shard ledgers. A corpus can hold several notes with more than one
# unretired head, so it recurs.

def test_a_refused_replacement_defers_that_target_and_the_pass_continues(
        tmp_path, monkeypatch):
    import lint.engine as le
    context = str(tmp_path / "context")
    source = _note("fb-source", body="[[supports:fb-blocked]] [[supports:fb-ok]]")
    blocked = _note("fb-blocked")
    ok = _note("fb-ok")
    for note in (source, blocked, ok):
        _persist(context, note)

    # every target needs the proposal path; the blocked one refuses
    monkeypatch.setattr(le, "_requires_replacement_proposal", lambda target: True)

    def _propose(target, desired, ctx, by_name, kind, detail=""):
        if target.name == "fb-blocked":
            raise ValueError(
                "unretired non-machine or parallel head already targets "
                f"{target.name!r}")
        return le.ReplacementDecision(action="CREATE", note_name=target.name + "-v2") \
            if hasattr(le, "ReplacementDecision") else _Decision(target.name + "-v2")

    class _Decision:
        def __init__(self, note_name):
            self.action, self.note_name = "CREATE", note_name

    monkeypatch.setattr(le, "_propose_replacement", _propose)

    result = le.file_back_links(load_notes(context), context)

    # the refusal did not raise, and the OTHER target was still processed
    proposals = {p["target"]: p["proposal"] for p in result["review_proposals"]}
    assert "fb-blocked" in proposals, "the blocked target must be reported"
    assert proposals["fb-blocked"] == "", (
        "a refused target carries no proposal name — it goes to attended review")
    assert result["review_required_links"] >= 1


def test_a_refused_replacement_is_reported_not_silently_dropped(tmp_path, monkeypatch):
    """Deferring must be visible; silently skipping would hide corpus damage."""
    import lint.engine as le
    context = str(tmp_path / "context")
    source = _note("fb2-source", body="[[supports:fb2-blocked]]")
    blocked = _note("fb2-blocked")
    for note in (source, blocked):
        _persist(context, note)
    monkeypatch.setattr(le, "_requires_replacement_proposal", lambda target: True)
    monkeypatch.setattr(le, "_propose_replacement", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("unretired non-machine or parallel head already targets 'x'")))

    result = le.file_back_links(load_notes(context), context)

    assert result["review_required_links"] >= 1
    assert any(p["target"] == "fb2-blocked" for p in result["review_proposals"])
