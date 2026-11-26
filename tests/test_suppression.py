"""Tests for the suppression (tombstone) hooks in remembrancerlib.note.

Two choke points must honor ``state/suppressed.json`` (path derived from the
parent of context_dir):

  * ``load_notes`` — a tombstoned note is invisible on read.
  * ``write_note`` — a tombstoned note is a NO-OP (returns existing path), so a
    distiller pass cannot resurrect a deleted memory.

A missing file leaves both behaviours unchanged. A present but malformed file
must fail closed so one truncated JSON write cannot resurrect every deletion.
"""
import json
import os

import pytest

from remembrancerlib import Note, load_notes, write_note, load_suppressed


def _seed(tmp_path):
    """Create a minimal repo: <root>/context/ with two notes."""
    context_dir = os.path.join(str(tmp_path), "context")
    os.makedirs(context_dir, exist_ok=True)
    keep = Note(name="keep-me", description="kept", domain="company", acl=[],
                body="I stay.")
    # A schema-valid name whose filename slug is different lets the tests cover
    # tombstoning by exact note id and by canonical file slug independently.
    drop = Note(name="delete--me", description="doomed", domain="company", acl=[],
                body="I go.")
    write_note(keep, context_dir)
    write_note(drop, context_dir)
    return context_dir


def _write_suppressed(tmp_path, names):
    state_dir = os.path.join(str(tmp_path), "state")
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "suppressed.json"), "w") as f:
        json.dump(names, f)


# --------------------------------------------------------------------------- #
# load_suppressed
# --------------------------------------------------------------------------- #
def test_load_suppressed_absent_is_empty(tmp_path):
    assert load_suppressed(str(tmp_path)) == set()


def test_load_suppressed_list(tmp_path):
    _write_suppressed(tmp_path, ["a", "b"])
    assert load_suppressed(str(tmp_path)) == {"a", "b"}


def test_load_suppressed_object_form(tmp_path):
    state_dir = os.path.join(str(tmp_path), "state")
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "suppressed.json"), "w") as f:
        json.dump({"suppressed": ["x"], "meta": "ignored"}, f)
    assert load_suppressed(str(tmp_path)) == {"x"}


def test_load_suppressed_invalid_fails_closed(tmp_path):
    state_dir = os.path.join(str(tmp_path), "state")
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "suppressed.json"), "w") as f:
        f.write("{ not valid json")
    with pytest.raises(RuntimeError, match="suppression state is unreadable"):
        load_suppressed(str(tmp_path))


def test_load_suppressed_invalid_shape_fails_closed(tmp_path):
    state_dir = os.path.join(str(tmp_path), "state")
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "suppressed.json"), "w") as f:
        json.dump({"wrong": []}, f)
    with pytest.raises(RuntimeError, match="no 'suppressed' list"):
        load_suppressed(str(tmp_path))


# --------------------------------------------------------------------------- #
# load_notes — read choke point
# --------------------------------------------------------------------------- #
def test_load_notes_no_suppression_sees_all(tmp_path):
    context_dir = _seed(tmp_path)
    names = {n.name for n in load_notes(context_dir)}
    assert names == {"keep-me", "delete--me"}


def test_load_notes_skips_suppressed_by_name(tmp_path):
    context_dir = _seed(tmp_path)
    _write_suppressed(tmp_path, ["delete--me"])
    names = {n.name for n in load_notes(context_dir)}
    assert names == {"keep-me"}


def test_load_notes_skips_suppressed_by_slug(tmp_path):
    """Tombstoning by slug (delete-me) must also hide the note."""
    context_dir = _seed(tmp_path)
    _write_suppressed(tmp_path, ["delete-me"])
    names = {n.name for n in load_notes(context_dir)}
    assert names == {"keep-me"}


def test_load_notes_normalizes_historical_display_name_tombstone(tmp_path):
    context_dir = _seed(tmp_path)
    _write_suppressed(tmp_path, ["Delete Me"])
    names = {n.name for n in load_notes(context_dir)}
    assert names == {"keep-me"}


def test_load_notes_absent_file_unchanged(tmp_path):
    context_dir = _seed(tmp_path)
    # No suppressed.json at all.
    assert len(load_notes(context_dir)) == 2


# --------------------------------------------------------------------------- #
# write_note — write choke point (distiller resurrection guard)
# --------------------------------------------------------------------------- #
def test_write_note_noops_when_suppressed(tmp_path):
    context_dir = _seed(tmp_path)
    # Remove the file on disk to prove write does NOT recreate it.
    doomed_path = os.path.join(context_dir, "company", "delete-me.md")
    _write_suppressed(tmp_path, ["delete--me"])
    os.remove(doomed_path)
    assert not os.path.exists(doomed_path)

    resurrect = Note(name="delete--me", description="doomed", domain="company",
                     acl=[], body="distiller tries again")
    returned = write_note(resurrect, context_dir)

    assert returned == doomed_path          # returns the path it WOULD have used
    assert not os.path.exists(doomed_path)  # but never actually wrote it


def test_write_note_noops_when_suppressed_by_slug(tmp_path):
    context_dir = _seed(tmp_path)
    doomed_path = os.path.join(context_dir, "company", "delete-me.md")
    _write_suppressed(tmp_path, ["delete-me"])
    os.remove(doomed_path)

    write_note(Note(name="delete--me", description="x", domain="company", body="y"),
               context_dir)
    assert not os.path.exists(doomed_path)


def test_write_note_writes_when_not_suppressed(tmp_path):
    context_dir = _seed(tmp_path)
    _write_suppressed(tmp_path, ["something-else"])
    n = Note(name="fresh", description="new", domain="company", body="hi")
    path = write_note(n, context_dir)
    assert os.path.exists(path)
    assert n.name in {x.name for x in load_notes(context_dir)}


def test_write_note_absent_file_unchanged(tmp_path):
    context_dir = _seed(tmp_path)
    n = Note(name="fresh", description="new", domain="company", body="hi")
    path = write_note(n, context_dir)
    assert os.path.exists(path)
