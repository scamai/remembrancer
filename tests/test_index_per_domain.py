"""Tests for per-domain index.md + root index.md."""
import os
from remembrancerlib import Note, write_note, regen_index


def test_per_domain_index(tmp_path):
    ctx = str(tmp_path / "context")
    write_note(Note(name="a", description="aa", type="concept", domain="eng"), ctx)
    write_note(Note(name="b", description="bb", type="concept", domain="ml"), ctx)
    regen_index(ctx)
    assert os.path.exists(os.path.join(ctx, "eng", "index.md"))
    assert os.path.exists(os.path.join(ctx, "ml", "index.md"))
    assert os.path.exists(os.path.join(ctx, "index.md"))            # root
    root = open(os.path.join(ctx, "index.md")).read()
    assert "eng" in root and "ml" in root


def test_root_index_contains_counts(tmp_path):
    ctx = str(tmp_path / "context")
    write_note(Note(name="note1", description="d1", type="concept", domain="eng"), ctx)
    write_note(Note(name="note2", description="d2", type="concept", domain="eng"), ctx)
    write_note(Note(name="note3", description="d3", type="concept", domain="ml"), ctx)
    regen_index(ctx)
    root = open(os.path.join(ctx, "index.md")).read()
    # Root should mention counts
    assert "2" in root   # eng has 2 notes
    assert "1" in root   # ml has 1 note


def test_domain_index_contains_entries(tmp_path):
    ctx = str(tmp_path / "context")
    write_note(Note(name="alpha", description="first note", type="concept", domain="eng"), ctx)
    regen_index(ctx)
    domain_idx = open(os.path.join(ctx, "eng", "index.md")).read()
    # OKF path link: [title](/domain/slug.md) — description · type
    assert "/eng/alpha.md" in domain_idx
    assert "first note" in domain_idx
    assert "concept" in domain_idx


def test_legacy_INDEX_still_written(tmp_path):
    """Back-compat: flat INDEX.md must still exist after regen_index."""
    ctx = str(tmp_path / "context")
    write_note(Note(name="x", description="dx", type="concept", domain="eng"), ctx)
    regen_index(ctx)
    assert os.path.exists(os.path.join(ctx, "INDEX.md"))


def test_project_subdir_domain_index_links(tmp_path):
    """A note with project="search-ranking" lives at
    context/ml/search-ranking/<slug>.md, and the per-domain index.md link must
    include the project segment."""
    ctx = str(tmp_path / "context")
    write_note(Note(name="v3-progress", description="v3 progress",
                    type="project", domain="ml", project="search-ranking"), ctx)
    regen_index(ctx)
    # File written to nested path.
    assert os.path.exists(os.path.join(ctx, "ml", "search-ranking", "v3-progress.md"))
    # Per-domain index link includes project segment.
    domain_idx = open(os.path.join(ctx, "ml", "index.md")).read()
    assert "/ml/search-ranking/v3-progress.md" in domain_idx
    assert "v3 progress" in domain_idx


def test_project_subdir_does_not_break_legacy_flat_notes(tmp_path):
    """When project="" (default), write_note uses
    the existing 2-segment join. Existing flat notes keep their paths."""
    ctx = str(tmp_path / "context")
    write_note(Note(name="flat", description="flat note",
                    type="project", domain="ml"), ctx)  # no project
    assert os.path.exists(os.path.join(ctx, "ml", "flat.md"))
    regen_index(ctx)
    domain_idx = open(os.path.join(ctx, "ml", "index.md")).read()
    # Flat note uses /ml/flat.md (no project segment).
    assert "/ml/flat.md" in domain_idx


def test_project_subdir_index_groups_by_project(tmp_path):
    """Notes within the same project subfolder
    appear together (sorted by project name then slug) in the per-domain
    index, even when flat notes from the same domain also exist."""
    ctx = str(tmp_path / "context")
    write_note(Note(name="flat-other", description="flat other",
                    type="project", domain="ml"), ctx)
    write_note(Note(name="v3", description="v3 progress",
                    type="project", domain="ml", project="search-ranking"), ctx)
    write_note(Note(name="v4", description="v4 progress",
                    type="project", domain="ml", project="search-ranking"), ctx)
    regen_index(ctx)
    domain_idx = open(os.path.join(ctx, "ml", "index.md")).read()
    # Both project notes have the project-segment link.
    assert "/ml/search-ranking/v3.md" in domain_idx
    assert "/ml/search-ranking/v4.md" in domain_idx
    # Flat note keeps its old-style link.
    assert "/ml/flat-other.md" in domain_idx
