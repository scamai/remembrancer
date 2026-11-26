"""Tests for query->file-back synthesis note writing.

file_answer(question, answer, source_names, root) writes a type:synthesis note
and returns its path. Gated by config retrieval.file_back.{enabled, min_chars}.
"""
import json
import os
import tempfile

import pytest

from remembrancerlib import Note, load_notes, write_note
import govern
from retrieve import query as Q
from retrieve.query import _synthesis_name, file_answer
from tests.trust_helpers import configure_approver


@pytest.fixture
def tmp_remembrancer_with_config():
    """Temp Remembrancer layout with file_back enabled, min_chars=50."""
    with tempfile.TemporaryDirectory() as tmp:
        context_dir = os.path.join(tmp, "context")
        os.makedirs(context_dir, exist_ok=True)

        # Write a source note
        source = Note(
            name="acme-profile",
            description="Acme customer profile",
            domain="company",
            acl=[],
            body="Acme onboards via the X flow and uses the Y system.",
        )
        write_note(source, context_dir)

        config = {
            "identities": {"writer": ["eng", "founders"]},
            "retrieval": {
                "candidate_pool": 25,
                "deliver_max_notes": 5,
                "max_slice_tokens": 4000,
                "file_back": {"enabled": True, "min_chars": 50},
            }
        }
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(config, f)

        yield tmp


@pytest.fixture
def tmp_remembrancer_file_back_disabled():
    """Temp Remembrancer layout with file_back disabled."""
    with tempfile.TemporaryDirectory() as tmp:
        context_dir = os.path.join(tmp, "context")
        os.makedirs(context_dir, exist_ok=True)

        config = {
            "retrieval": {
                "file_back": {"enabled": False, "min_chars": 50},
            }
        }
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(config, f)

        yield tmp


@pytest.fixture
def tmp_remembrancer_high_min_chars():
    """Temp Remembrancer layout with file_back enabled but high min_chars threshold."""
    with tempfile.TemporaryDirectory() as tmp:
        context_dir = os.path.join(tmp, "context")
        os.makedirs(context_dir, exist_ok=True)

        config = {
            "retrieval": {
                "file_back": {"enabled": True, "min_chars": 1000},
            }
        }
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(config, f)

        yield tmp


def test_file_back_creates_synthesis_note(tmp_remembrancer_with_config):
    """file_answer with a long-enough answer creates a type:synthesis note file."""
    long_answer = "Acme onboards via the X flow and uses the Y system for all customer data. " * 3
    p = file_answer(
        "how does acme onboarding work?",
        long_answer,
        ["acme-profile"],
        identity="writer",
        root=tmp_remembrancer_with_config,
    )
    assert p, "file_answer must return a non-empty path"
    assert os.path.isfile(p), f"file does not exist at {p}"
    content = open(p).read()
    assert "synthesis" in content, "written note must contain 'synthesis' (type field)"
    assert long_answer.strip()[:50] in content or "synthesis" in content


def test_file_back_note_has_correct_type(tmp_remembrancer_with_config):
    """The written note must have type: synthesis in its frontmatter."""
    long_answer = "Acme onboards via the X flow and uses the Y system for all customer data. " * 3
    p = file_answer(
        "how does acme onboarding work?",
        long_answer,
        ["acme-profile"],
        identity="writer",
        root=tmp_remembrancer_with_config,
    )
    content = open(p).read()
    assert "type: synthesis" in content


def test_file_back_note_has_provenance(tmp_remembrancer_with_config):
    """The written note must include provenance with the source names."""
    long_answer = "Acme onboards via the X flow and uses the Y system for all customer data. " * 3
    p = file_answer(
        "how does acme onboarding work?",
        long_answer,
        ["acme-profile"],
        identity="writer",
        root=tmp_remembrancer_with_config,
    )
    content = open(p).read()
    assert "acme-profile" in content, "provenance must reference source note names"


def test_file_back_disabled_returns_empty(tmp_remembrancer_file_back_disabled):
    """file_answer returns '' when file_back.enabled is False."""
    long_answer = "Acme onboards via the X flow and uses the Y system for all customer data. " * 3
    result = file_answer(
        "how does acme onboarding work?",
        long_answer,
        ["acme-profile"],
        root=tmp_remembrancer_file_back_disabled,
    )
    assert result == "", "file_answer must return '' when file_back is disabled"


def test_file_back_short_answer_returns_empty(tmp_remembrancer_with_config):
    """file_answer returns '' when answer is shorter than min_chars."""
    short_answer = "Short."
    result = file_answer(
        "how does acme onboarding work?",
        short_answer,
        ["acme-profile"],
        root=tmp_remembrancer_with_config,
    )
    assert result == "", "file_answer must return '' when answer is shorter than min_chars"


def test_file_back_answer_at_min_chars(tmp_remembrancer_with_config):
    """file_answer writes note when answer length is exactly min_chars (50)."""
    # min_chars=50, so exactly 50 chars should pass
    exact_answer = "A" * 50
    p = file_answer(
        "how does acme onboarding work?",
        exact_answer,
        ["acme-profile"],
        identity="writer",
        root=tmp_remembrancer_with_config,
    )
    assert p != "", "answer at exactly min_chars must create a note"
    assert os.path.isfile(p)


def test_file_back_answer_just_below_min_chars(tmp_remembrancer_with_config):
    """file_answer returns '' when answer is one char below min_chars (49 chars)."""
    short_answer = "A" * 49
    result = file_answer(
        "how does acme onboarding work?",
        short_answer,
        ["acme-profile"],
        root=tmp_remembrancer_with_config,
    )
    assert result == "", "answer below min_chars must not create a note"


def test_file_back_note_in_synthesis_dir(tmp_remembrancer_with_config):
    """The written synthesis note must be inside context/company/ directory."""
    long_answer = "Acme onboards via the X flow and uses the Y system for all customer data. " * 3
    p = file_answer(
        "how does acme onboarding work?",
        long_answer,
        ["acme-profile"],
        identity="writer",
        root=tmp_remembrancer_with_config,
    )
    # The note should be under context/<domain>/<slug>.md
    assert "context" in p, "file must be inside the context directory"


def test_file_back_returns_path_string(tmp_remembrancer_with_config):
    """file_answer must return a str path (not None or other type)."""
    long_answer = "Acme onboards via the X flow and uses the Y system for all customer data. " * 3
    p = file_answer(
        "how does acme onboarding work?",
        long_answer,
        ["acme-profile"],
        identity="writer",
        root=tmp_remembrancer_with_config,
    )
    assert isinstance(p, str)


def test_file_back_high_min_chars_skips(tmp_remembrancer_high_min_chars):
    """file_answer returns '' when answer is shorter than a high min_chars threshold."""
    medium_answer = "Acme onboards via the X flow. " * 5  # ~150 chars, below 1000
    result = file_answer(
        "how does acme onboarding work?",
        medium_answer,
        ["acme-profile"],
        root=tmp_remembrancer_high_min_chars,
    )
    assert result == ""


def test_mcp_file_answer_tool_wraps_file_answer(tmp_remembrancer_with_config, monkeypatch):
    """_do_file_answer in mcp/server.py calls file_answer and returns the path."""
    import sys
    # Ensure worktree root is on path for imports
    wt_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if wt_root not in sys.path:
        sys.path.insert(0, wt_root)

    # Patch ROOT in server to point to our tmp dir
    import mcp.server as srv
    monkeypatch.setattr(srv, "ROOT", tmp_remembrancer_with_config)

    long_answer = "Acme onboards via the X flow and uses the Y system for all customer data. " * 3
    result = srv._do_file_answer(
        question="how does acme onboarding work?",
        answer=long_answer,
        sources=["acme-profile"],
        as_identity="writer",
        root=tmp_remembrancer_with_config,
    )
    assert result.get("path"), "MCP _do_file_answer must return a non-empty path"
    assert os.path.isfile(result["path"])


def test_file_back_cannot_publish_restricted_source(tmp_path):
    """A caller must not turn a secrets source into public verified memory."""
    context = tmp_path / "context"
    context.mkdir()
    write_note(Note(name="secret-source", description="secret", domain="company",
                    acl=["secrets"], body="confidential detail"), str(context))
    (tmp_path / "config.json").write_text(json.dumps({
        "identities": {"eng-user": ["eng"], "secret-user": ["secrets"]},
        "retrieval": {"file_back": {"enabled": True, "min_chars": 1}},
    }))
    with pytest.raises(PermissionError, match="not permitted"):
        file_answer("what is secret?", "confidential synthesized answer",
                    ["secret-source"], identity="eng-user", root=str(tmp_path))

    path = file_answer("what is secret?", "confidential synthesized answer",
                       ["secret-source"], identity="secret-user", root=str(tmp_path))
    fm = open(path, encoding="utf-8").read()
    assert "acl: [secrets]" in fm
    assert "state: proposed" in fm


def test_file_back_requires_real_citation(tmp_remembrancer_with_config):
    with pytest.raises(ValueError, match="requires at least one source"):
        file_answer("uncited", "A sufficiently long answer that must not be filed.", [],
                    identity="writer", root=tmp_remembrancer_with_config)


def test_file_back_never_overwrites_existing_note(tmp_path):
    """A public source must not choose a question that replaces secret memory."""
    context = tmp_path / "context"
    context.mkdir()
    source = Note(name="public-source", description="public", domain="eng", acl=[],
                  body="safe source")
    protected = Note(name="initiative-remembrancer", description="private", domain="ml",
                     acl=["secrets"], body="do not replace")
    write_note(source, str(context))
    protected_path = write_note(protected, str(context))
    (tmp_path / "config.json").write_text(json.dumps({
        "identities": {"writer": ["eng"]},
        "retrieval": {"file_back": {"enabled": True, "min_chars": 1}},
    }))

    path = file_answer("initiative remembrancer", "safe synthesized answer", ["public-source"],
                       identity="writer", root=str(tmp_path))
    assert path and path != protected_path
    assert "do not replace" in open(protected_path, encoding="utf-8").read()


def test_file_back_refuses_reserved_name_collision(tmp_path):
    context = tmp_path / "context"
    context.mkdir()
    source = Note(name="public-source", description="public", domain="eng", acl=[],
                  body="safe source")
    write_note(source, str(context))
    answer = "safe synthesized answer"
    name = _synthesis_name("question", answer, [source])
    write_note(Note(name=name, description="human collision", domain="eng", acl=[],
                    body="human content"), str(context))
    (tmp_path / "config.json").write_text(json.dumps({
        "identities": {"writer": ["eng"]},
        "retrieval": {"file_back": {"enabled": True, "min_chars": 1}},
    }))

    with pytest.raises(ValueError, match="already exists"):
        file_answer("question", answer, ["public-source"], identity="writer",
                    root=str(tmp_path))


def test_file_back_synthesis_is_hidden_until_signed_approval(
    tmp_remembrancer_with_config,
):
    answer = "fileanswerholdword grounded synthesis awaiting exact human review " * 3
    path = file_answer(
        "How should this synthesis be governed?", answer, ["acme-profile"],
        identity="writer", root=tmp_remembrancer_with_config,
    )
    note = next(
        row for row in load_notes(os.path.join(tmp_remembrancer_with_config, "context"))
        if row.path == path
    )
    verification = note.confidence["verification"]
    assert verification["reason"] == (
        "governance-approval-required:file-answer-synthesis"
    )
    assert Q.get_note(note.name, identity="writer", root=tmp_remembrancer_with_config) is None
    assert Q.search(
        "fileanswerholdword", identity="writer", root=tmp_remembrancer_with_config,
    ) == []
    pending = govern.pending(tmp_remembrancer_with_config)
    assert len(pending) == 1 and pending[0]["note"] == note.name

    configure_approver(tmp_remembrancer_with_config, "reviewer")
    item = govern.get_item(tmp_remembrancer_with_config, pending[0]["id"])
    result = Q.promote(
        note.name, identity="reviewer", root=tmp_remembrancer_with_config,
        expected_content_fingerprint=item["content_fingerprint"],
        expected_existing_sha256=item["expected_existing_sha256"],
    )
    assert result["ok"], result
    govern.record(tmp_remembrancer_with_config, item["id"], "approved", "reviewer")
    assert Q.get_note(
        note.name, identity="reviewer", root=tmp_remembrancer_with_config,
    ) is not None


def test_file_back_queue_failure_leaves_hold_and_exact_retry_enqueues(
    tmp_remembrancer_with_config, monkeypatch,
):
    question = "Can a failed queue publish this synthesis?"
    answer = "fileanswerqueueword held answer with enough grounded content " * 3
    real_append = govern._append
    monkeypatch.setattr(
        govern, "_append",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("queue offline")),
    )
    with pytest.raises(OSError, match="queue offline"):
        file_answer(
            question, answer, ["acme-profile"], identity="writer",
            root=tmp_remembrancer_with_config,
        )
    held = next(
        note for note in load_notes(os.path.join(tmp_remembrancer_with_config, "context"))
        if note.name.startswith("synthesis-")
    )
    assert Q.get_note(
        held.name, identity="writer", root=tmp_remembrancer_with_config,
    ) is None

    monkeypatch.setattr(govern, "_append", real_append)
    replay = file_answer(
        question, answer, ["acme-profile"], identity="writer",
        root=tmp_remembrancer_with_config,
    )
    assert replay == held.path
    pending = govern.pending(tmp_remembrancer_with_config)
    assert len(pending) == 1 and pending[0]["note"] == held.name
