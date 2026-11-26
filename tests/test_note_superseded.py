"""Tests for the Note.superseded field (the field only, no lint logic)."""
import os
from remembrancerlib import Note, write_note
from remembrancerlib.note import parse_frontmatter
from tests.trust_helpers import write_raw_legacy_note


def test_superseded_defaults_false():
    n = Note(name="x", description="d", type="concept", domain="eng")
    assert n.superseded is False


def test_superseded_emitted_in_markdown():
    n = Note(name="x", description="d", type="concept", domain="eng")
    n.superseded = True
    md = n.to_markdown()
    assert "superseded: true" in md


def test_superseded_false_emitted_in_markdown():
    n = Note(name="x", description="d", type="concept", domain="eng")
    md = n.to_markdown()
    assert "superseded: false" in md


def test_superseded_parsed_from_string_true(tmp_path):
    """A note written with superseded: true must be parsed back correctly."""
    n = Note(name="alpha", description="d", type="concept", domain="eng")
    n.superseded = True
    n.state = "superseded"
    path = write_raw_legacy_note(tmp_path, n)
    with open(path) as f:
        fm, body = parse_frontmatter(f.read())
    assert fm.get("superseded") in ("true", True)


def test_superseded_roundtrip(tmp_path):
    """Write a superseded note and load it back; superseded must be True."""
    from remembrancerlib import load_notes
    n = Note(name="beta", description="d", type="concept", domain="eng")
    n.superseded = True
    n.state = "superseded"
    write_raw_legacy_note(tmp_path, n)
    notes = load_notes(str(tmp_path / "context"))
    assert len(notes) == 1
    assert notes[0].superseded is True


def test_superseded_roundtrip_false(tmp_path):
    """Write a normal note and load it back; superseded must be False."""
    from remembrancerlib import load_notes
    n = Note(name="gamma", description="d", type="concept", domain="eng")
    ctx = str(tmp_path / "context")
    write_note(n, ctx)
    notes = load_notes(ctx)
    assert notes[0].superseded is False


def test_old_note_without_superseded_parses(tmp_path):
    """Additive: a note file without the superseded field must still load with superseded=False."""
    from remembrancerlib import load_notes
    # Write a note manually without the superseded field
    domain_dir = str(tmp_path / "context" / "eng")
    os.makedirs(domain_dir, exist_ok=True)
    path = os.path.join(domain_dir, "old-note.md")
    with open(path, "w") as f:
        f.write("---\nname: old-note\ndescription: old\ntype: concept\ndomain: eng\nupdated: 2024-01-01\n---\n\nBody here.\n")
    notes = load_notes(str(tmp_path / "context"))
    assert len(notes) == 1
    assert notes[0].superseded is False
