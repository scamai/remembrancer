"""Tests for AGENTS.md spec loader (distill.engine.load_spec).

TDD: these tests are written FIRST and must fail before implementation,
then pass after load_spec is added to distill/engine.py.
"""
import textwrap


from distill.engine import load_spec


# ---------------------------------------------------------------------------
# Defaults — no AGENTS.md present
# ---------------------------------------------------------------------------

def test_load_spec_defaults_when_absent(tmp_path):
    """load_spec returns safe defaults when AGENTS.md does not exist."""
    result = load_spec(root=str(tmp_path))
    assert result == {
        "contradiction_policy": "flag",
        "file_back_enabled": True,
    }


# ---------------------------------------------------------------------------
# Override — AGENTS.md with a conventions block
# ---------------------------------------------------------------------------

def test_load_spec_reads_conventions_block(tmp_path):
    """load_spec parses key: value lines inside the ```conventions block."""
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text(
        textwrap.dedent("""\
        # Remembrancer AGENTS.md

        Some prose here.

        ```conventions
        contradiction_policy: reconcile
        file_back_enabled: true
        ```

        More prose.
        """)
    )
    result = load_spec(root=str(tmp_path))
    assert result["contradiction_policy"] == "reconcile"


def test_load_spec_boolean_coercion(tmp_path):
    """file_back_enabled: false is coerced to Python False."""
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text(
        textwrap.dedent("""\
        ```conventions
        contradiction_policy: flag
        file_back_enabled: false
        ```
        """)
    )
    result = load_spec(root=str(tmp_path))
    assert result["file_back_enabled"] is False


def test_load_spec_missing_key_falls_back_to_default(tmp_path):
    """Keys absent from the conventions block fall back to their safe defaults."""
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text(
        textwrap.dedent("""\
        ```conventions
        contradiction_policy: reconcile
        ```
        """)
    )
    result = load_spec(root=str(tmp_path))
    # file_back_enabled not in block -> default True
    assert result["file_back_enabled"] is True
    assert result["contradiction_policy"] == "reconcile"


def test_load_spec_empty_conventions_block(tmp_path):
    """An empty conventions block returns all defaults."""
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text("```conventions\n```\n")
    result = load_spec(root=str(tmp_path))
    assert result == {
        "contradiction_policy": "flag",
        "file_back_enabled": True,
    }


def test_load_spec_no_conventions_block_in_file(tmp_path):
    """AGENTS.md exists but has no conventions block -> safe defaults."""
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text("# Remembrancer AGENTS.md\n\nJust prose, no fenced block.\n")
    result = load_spec(root=str(tmp_path))
    assert result == {
        "contradiction_policy": "flag",
        "file_back_enabled": True,
    }
