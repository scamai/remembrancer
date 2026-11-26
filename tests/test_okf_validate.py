"""TDD tests for tools.okf_validate — OKF bundle validator.

validate_bundle(context_dir) -> list[str]
  - Flags every .md note (except index.md/INDEX.md) that has no non-empty 'type'
    frontmatter field with "<path>: missing required 'type'".
  - Returns an empty list when all notes are valid.
"""
import os


from remembrancerlib import Note, write_note
from tools.okf_validate import validate_bundle


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_raw(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_valid_note_not_flagged(tmp_path):
    """A note that carries a non-empty 'type' must NOT appear in violations."""
    context_dir = str(tmp_path / "context")
    note = Note(
        name="okf-valid",
        description="A note with a proper type field",
        type="project",
        domain="company",
        body="Some content.",
    )
    write_note(note, context_dir)

    violations = validate_bundle(context_dir)

    assert violations == [], f"Expected no violations, got: {violations}"


def test_note_missing_type_flagged(tmp_path):
    """A raw .md file with no 'type' frontmatter must be flagged."""
    context_dir = str(tmp_path / "context")
    bad_path = os.path.join(context_dir, "eng", "no-type.md")
    _write_raw(bad_path, "---\nname: no-type\ndescription: broken\n---\n\nBody text.\n")

    violations = validate_bundle(context_dir)

    assert len(violations) == 1
    assert "missing required 'type'" in violations[0]
    assert "no-type.md" in violations[0]


def test_mixed_bundle_flags_only_bad(tmp_path):
    """When the context dir has one valid and one invalid note, only the
    invalid note is flagged."""
    context_dir = str(tmp_path / "context")

    # Valid note via remembrancerlib (has type="reference")
    good = Note(
        name="good-note",
        description="has type",
        type="reference",
        domain="company",
        body="All good.",
    )
    write_note(good, context_dir)

    # Invalid note: hand-written, no type field
    bad_path = os.path.join(context_dir, "company", "bad-note.md")
    _write_raw(bad_path, "---\nname: bad-note\ndescription: missing type\n---\n\nOops.\n")

    violations = validate_bundle(context_dir)

    assert len(violations) == 1
    assert "missing required 'type'" in violations[0]
    assert "bad-note.md" in violations[0]


def test_index_files_are_skipped(tmp_path):
    """index.md and INDEX.md must be ignored even when they lack a 'type'."""
    context_dir = str(tmp_path / "context")
    os.makedirs(context_dir, exist_ok=True)

    for fname in ("index.md", "INDEX.md"):
        _write_raw(
            os.path.join(context_dir, fname),
            "---\nname: index\n---\n\n# Index\n",
        )

    violations = validate_bundle(context_dir)
    assert violations == [], f"Index files should be skipped, got: {violations}"


def test_empty_type_is_flagged(tmp_path):
    """A note with type: '' (empty string) must also be flagged."""
    context_dir = str(tmp_path / "context")
    bad_path = os.path.join(context_dir, "eng", "empty-type.md")
    _write_raw(bad_path, "---\nname: empty-type\ntype: \ndescription: empty\n---\n\nBody.\n")

    violations = validate_bundle(context_dir)

    assert len(violations) == 1
    assert "missing required 'type'" in violations[0]


def test_empty_context_dir_returns_no_violations(tmp_path):
    """An empty context directory is a trivially valid bundle."""
    context_dir = str(tmp_path / "context")
    os.makedirs(context_dir, exist_ok=True)

    violations = validate_bundle(context_dir)
    assert violations == []
