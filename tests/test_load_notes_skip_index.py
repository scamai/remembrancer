"""Test that load_notes skips both INDEX.md and index.md files."""
import os
import tempfile

from remembrancerlib import Note, load_notes, write_note


def test_load_notes_skips_index_md_with_frontmatter():
    """Test that index.md files with frontmatter are NOT loaded as notes.

    This is a regression test:
    load_notes() was only skipping INDEX.md (uppercase) but not index.md (lowercase).
    If index.md had frontmatter, it would be wrongly ingested as a note.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # `write_note` derives repo_root as the PARENT of context_dir and takes
        # its suppression lock under `<repo_root>/state/`. Passing the temp dir
        # itself made repo_root `/tmp`, so this test tried to create and write
        # `/tmp/state/suppressed.lock`, a path every user of a shared machine
        # can reach and another user may already own (PermissionError). Nest the
        # context dir the way the repo does, so state stays inside the fixture.
        context_dir = os.path.join(tmp, "context")
        os.makedirs(context_dir, exist_ok=True)

        # Write a real note that should be loaded
        real_note = Note(
            name="real-note",
            description="A real note that should be loaded",
            domain="company",
            body="This is a real note.",
        )
        write_note(real_note, context_dir)

        # Write an index.md file WITH frontmatter (the risky case)
        index_path = os.path.join(context_dir, "index.md")
        with open(index_path, "w") as f:
            f.write("---\n")
            f.write("type: index\n")
            f.write("name: index\n")
            f.write("---\n")
            f.write("This is the index file content.\n")

        # Load notes and verify
        loaded = load_notes(context_dir)

        # Should have exactly 1 note (the real one)
        assert len(loaded) == 1
        assert loaded[0].name == "real-note"
        assert loaded[0].description == "A real note that should be loaded"

        # Verify no loaded note has path pointing to index.md
        for note in loaded:
            assert "index.md" not in note.path
            assert note.path.endswith("real-note.md")
