import pytest

from remembrancerlib import Note, index_notes_by_name, load_notes, write_note


def test_note_name_index_fails_closed_on_two_physical_notes():
    first = Note(name="same-memory", description="first", path="context/eng/a.md")
    second = Note(name="same-memory", description="second", path="context/ml/b.md")

    with pytest.raises(ValueError, match="ambiguous note names") as exc:
        index_notes_by_name([first, second], operation="distillation")

    assert "context/eng/a.md" in str(exc.value)
    assert "context/ml/b.md" in str(exc.value)


def test_note_name_index_preserves_unique_notes():
    notes = [Note(name="a", description="one"), Note(name="b", description="two")]
    assert index_notes_by_name(notes) == {"a": notes[0], "b": notes[1]}


def test_write_door_rejects_same_logical_name_in_another_domain(tmp_path):
    context = str(tmp_path / "context")
    write_note(Note(name="same-memory", description="first", domain="company"), context)

    with pytest.raises(FileExistsError, match="duplicate logical note name"):
        write_note(Note(name="same-memory", description="second", domain="eng"), context)

    assert len(load_notes(context)) == 1
