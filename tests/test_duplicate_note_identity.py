"""Duplicate note names fail closed until a governed corpus migration resolves them."""
from remembrancerlib import Note, write_note
from retrieve.query import get_note, search


def _root(tmp_path):
    (tmp_path / "context").mkdir()
    write_note(Note(name="same-id", description="alpha unique", domain="eng", body="alpha"),
               str(tmp_path / "context"))
    # Duplicate logical names are legacy/corrupt input.  The write choke point
    # correctly refuses to create one, so materialize the invalid historical
    # file directly to exercise the readers' fail-closed behavior.
    duplicate = Note(
        name="same-id", description="beta unique", domain="ml", body="beta",
    )
    duplicate_path = tmp_path / "context" / "ml" / "same-id.md"
    duplicate_path.parent.mkdir()
    duplicate_path.write_text(duplicate.to_markdown())
    return str(tmp_path)


def test_get_note_refuses_ambiguous_global_name(tmp_path):
    assert get_note("same-id", root=_root(tmp_path)) is None


def test_search_does_not_return_unopenable_ambiguous_result(tmp_path):
    assert search("alpha unique", root=_root(tmp_path)) == []
