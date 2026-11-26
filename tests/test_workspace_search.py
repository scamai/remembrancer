"""Git-aware, bounded local-workspace evidence retrieval."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )
    return result.stdout.strip()


@pytest.fixture
def workspace_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "internal"
    repo.mkdir()
    _git(repo, "init")
    (repo / ".gitignore").write_text("ignored/\n*.ignored\n")
    (repo / "remembrancer").mkdir()
    (repo / "remembrancer" / "README.md").write_text("remembrancer root\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "design.md").write_text("# Design\ncommitted-orchid baseline\n")
    (repo / "service.py").write_text("STATE = 'clean-saffron'\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    return repo, repo / "remembrancer"


def _settings(**overrides):
    base = {
        "enabled": True,
        "max_results": 10,
        "snippet_chars": 400,
        "max_files": 100,
        "max_file_bytes": 4096,
        "max_index_bytes": 65536,
        "workspaces": [
            {
                "id": "internal-test",
                "root": "..",
                "include": ["**/*.md", "**/*.py", ".gitignore"],
                "exclude": ["remembrancer/context/**", "remembrancer/events/**", "remembrancer/state/**"],
            }
        ],
    }
    base.update(overrides)
    return base


def _search(query: str, remembrancer_root: Path, **settings):
    from retrieve.workspace import clear_workspace_cache, search_workspaces

    if settings.pop("clear", True):
        clear_workspace_cache()
    return search_workspaces(
        query,
        remembrancer_root=str(remembrancer_root),
        settings=_settings(**settings),
    )


def test_indexes_committed_staged_modified_and_untracked(workspace_repo):
    repo, remembrancer_root = workspace_repo

    (repo / "docs" / "design.md").write_text("# Design\nmodified-cobalt is current\n")
    (repo / "service.py").write_text("STATE = 'staged-topaz'\n")
    _git(repo, "add", "service.py")
    (repo / "proposal.md").write_text("untracked-vermilion proposal\n")
    (repo / "ignored").mkdir()
    (repo / "ignored" / "hidden.md").write_text("ignored-vermilion\n")

    committed = _search("remembrancer root", remembrancer_root)
    modified = _search("modified-cobalt", remembrancer_root)
    staged = _search("staged-topaz", remembrancer_root)
    untracked = _search("untracked-vermilion", remembrancer_root)
    ignored = _search("ignored-vermilion", remembrancer_root)

    assert committed["results"][0]["worktree_state"] == ["committed"]
    assert modified["results"][0]["worktree_state"] == ["modified"]
    assert staged["results"][0]["worktree_state"] == ["staged"]
    assert untracked["results"][0]["worktree_state"] == ["untracked"]
    assert "ignored/hidden.md" not in {row["path"] for row in ignored["results"]}
    assert modified["workspaces"][0]["dirty"] is True
    assert len(modified["workspaces"][0]["head"]) == 40
    assert modified["workspaces"][0]["branch"]
    assert len(modified["workspaces"][0]["manifest_sha256"]) == 64


def test_staged_and_modified_states_can_coexist(workspace_repo):
    repo, remembrancer_root = workspace_repo
    path = repo / "service.py"
    path.write_text("STATE = 'staged-indigo'\n")
    _git(repo, "add", "service.py")
    path.write_text("STATE = 'staged-indigo plus modified-cyan'\n")

    out = _search("modified-cyan", remembrancer_root)

    assert out["results"][0]["worktree_state"] == ["staged", "modified"]


def test_subdirectory_workspace_paths_are_relative_to_configured_root(workspace_repo):
    repo, remembrancer_root = workspace_repo
    from retrieve.workspace import clear_workspace_cache, search_workspaces

    clear_workspace_cache()
    settings = _settings(
        workspaces=[
            {
                "id": "remembrancer-only",
                "root": ".",
                "include": ["**/*.md", "README.md"],
                "exclude": ["context/**", "events/**", "state/**"],
            }
        ]
    )
    out = search_workspaces(
        "remembrancer root",
        remembrancer_root=str(remembrancer_root),
        settings=settings,
    )

    assert out["results"][0]["path"] == "README.md"
    assert "docs/design.md" not in {row["path"] for row in out["results"]}


def test_nul_delimited_paths_preserve_newlines_and_tabs(workspace_repo):
    repo, remembrancer_root = workspace_repo
    newline_path = repo / "docs" / "line\nbreak.md"
    tab_path = repo / "docs" / "tab\tfile.md"
    newline_path.write_text("newline-path-silver\n")
    tab_path.write_text("tab-path-gold\n")

    newline = _search("newline-path-silver", remembrancer_root)
    tab = _search("tab-path-gold", remembrancer_root)

    assert newline["results"][0]["path"] == "docs/line\nbreak.md"
    assert tab["results"][0]["path"] == "docs/tab\tfile.md"


def test_detached_head_is_reported(workspace_repo):
    repo, remembrancer_root = workspace_repo
    _git(repo, "checkout", "--detach")

    out = _search("committed-orchid", remembrancer_root)
    metadata = out["workspaces"][0]

    assert metadata["detached"] is True
    assert metadata["branch"] is None


def test_content_and_manifest_hash_change_without_head_change(workspace_repo):
    repo, remembrancer_root = workspace_repo
    path = repo / "docs" / "design.md"
    before = _search("committed-orchid", remembrancer_root)
    head = before["workspaces"][0]["head"]
    before_manifest = before["workspaces"][0]["manifest_sha256"]
    before_hash = before["results"][0]["content_sha256"]

    path.write_text("# Design\nchanged-orchid baseline\n")
    after = _search("changed-orchid", remembrancer_root, clear=False)

    assert after["workspaces"][0]["head"] == head
    assert after["workspaces"][0]["manifest_sha256"] != before_manifest
    assert after["results"][0]["content_sha256"] != before_hash


def test_same_size_mtime_restored_mutation_is_detected(workspace_repo):
    repo, remembrancer_root = workspace_repo
    path = repo / "docs" / "design.md"
    before = _search("committed-orchid", remembrancer_root)
    old_hash = before["results"][0]["content_sha256"]
    old_stat = path.stat()
    replacement = "# Design\nmutatedxx-orchid baseline\n"
    original = path.read_text()
    assert len(replacement) == len(original)
    path.write_text(replacement)
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))

    after = _search("mutatedxx-orchid", remembrancer_root, clear=False)

    assert after["results"]
    assert after["results"][0]["content_sha256"] != old_hash


def test_removed_cached_file_disappears(workspace_repo):
    repo, remembrancer_root = workspace_repo
    path = repo / "proposal.md"
    path.write_text("temporary-cerulean\n")
    assert _search("temporary-cerulean", remembrancer_root)["results"]

    path.unlink()
    after = _search("temporary-cerulean", remembrancer_root, clear=False)

    assert after["results"] == []


def test_excludes_secrets_raw_memory_binary_invalid_utf8_and_symlinks(workspace_repo):
    repo, remembrancer_root = workspace_repo
    forbidden = "forbidden-ruby"
    (repo / ".env").write_text(f"API_KEY={forbidden}\n")
    (repo / "private.pem").write_text(f"{forbidden}\n")
    for dirname in ("context", "events", "state"):
        target = repo / "remembrancer" / dirname
        target.mkdir()
        (target / "hidden.md").write_text(f"{forbidden}\n")
    (repo / "binary.py").write_bytes(b"\x00" + forbidden.encode())
    (repo / "invalid.py").write_bytes(b"\xff\xfe" + forbidden.encode())
    outside = repo.parent / "outside.md"
    outside.write_text(f"{forbidden}\n")
    (repo / "linked.md").symlink_to(outside)
    os.mkfifo(repo / "named-pipe.md")

    out = _search(forbidden, remembrancer_root)

    assert out["results"] == []


def test_rejects_root_escape_and_absolute_root(workspace_repo):
    _repo, remembrancer_root = workspace_repo
    from retrieve.workspace import clear_workspace_cache, search_workspaces

    clear_workspace_cache()
    escaped = _settings(
        workspaces=[
            {
                "id": "escape",
                "root": "../../..",
                "include": ["**/*.md"],
            }
        ]
    )
    absolute = _settings(
        workspaces=[
            {
                "id": "absolute",
                "root": "/tmp",
                "include": ["**/*.md"],
            }
        ]
    )

    for settings in (escaped, absolute):
        out = search_workspaces(
            "anything",
            remembrancer_root=str(remembrancer_root),
            settings=settings,
        )
        assert out["searched"] is False
        assert out["results"] == []
        assert out["reason"] == "unsafe_root"


def test_oversized_file_is_skipped_and_budget_is_explicit(workspace_repo):
    repo, remembrancer_root = workspace_repo
    (repo / "huge.md").write_text("oversized-amber " * 200)
    skipped = _search(
        "oversized-amber",
        remembrancer_root,
        max_file_bytes=64,
    )
    assert skipped["results"] == []

    (repo / "a.md").write_text("budget-lilac\n")
    (repo / "b.md").write_text("budget-lilac\n")
    partial = _search("budget-lilac", remembrancer_root, max_files=1)
    assert partial["partial"] is True
    assert partial["reason"] == "file_budget"


def test_snippet_has_lines_is_bounded_and_redacts_secret_assignment(workspace_repo):
    repo, remembrancer_root = workspace_repo
    (repo / "secrets_example.py").write_text(
        'before = 1\nAPI_KEY = "sample_value_that_must_not_escape"\nafter = 2\n'
    )

    out = _search("API_KEY", remembrancer_root, snippet_chars=90)
    row = out["results"][0]

    assert row["line_start"] == 1
    assert row["line_end"] >= 2
    assert len(row["snippet"]) <= 90
    assert "sample_value_that_must_not_escape" not in row["snippet"]
    assert "<redacted>" in row["snippet"]
    assert row["trust"] == "unverified-local-worktree"


def test_snippet_redacts_authorization_uri_tokens_and_private_keys(workspace_repo):
    repo, remembrancer_root = workspace_repo
    (repo / "docs" / "credential-examples.md").write_text(
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345\n"
        "postgres://reader:correct-horse-battery@db.local/app\n"  # trufflehog:ignore: a placeholder this test redacts
        "token ghp_abcdefghijklmnopqrstuvwxyz0123456789\n"
    )
    out = _search("Authorization", remembrancer_root)
    snippet = out["results"][0]["snippet"]

    assert "abcdefghijklmnopqrstuvwxyz012345" not in snippet
    assert "correct-horse-battery" not in snippet
    assert snippet.count("<redacted>") >= 2
    token = _search("token", remembrancer_root)
    token_snippet = token["results"][0]["snippet"]
    assert "ghp_" not in token_snippet
    assert "<redacted>" in token_snippet

    (repo / "docs" / "key-example.md").write_text(
        "private sample\n-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n"
    )
    private = _search("private sample", remembrancer_root)
    assert private["results"][0]["snippet"] == "[redacted secret-like content]"


def test_git_commands_share_the_total_time_budget(workspace_repo, monkeypatch):
    _repo, remembrancer_root = workspace_repo
    import retrieve.workspace as workspace

    observed: list[float] = []

    def timeout(*_args, **kwargs):
        observed.append(float(kwargs["timeout"]))
        raise subprocess.TimeoutExpired(cmd="git", timeout=kwargs["timeout"])

    workspace.clear_workspace_cache()
    monkeypatch.setattr(workspace.subprocess, "run", timeout)
    started = workspace.time.monotonic()
    out = workspace.search_workspaces(
        "anything",
        remembrancer_root=str(remembrancer_root),
        settings=_settings(max_seconds=0.1),
    )
    elapsed = workspace.time.monotonic() - started

    assert out["searched"] is False
    assert out["reason"] == "time_budget"
    assert out["partial"] is False
    assert observed and max(observed) <= 0.11
    assert elapsed < 0.5


def test_disabled_settings_do_not_touch_workspace(workspace_repo):
    _repo, remembrancer_root = workspace_repo
    from retrieve.workspace import search_workspaces

    out = search_workspaces(
        "committed-orchid",
        remembrancer_root=str(remembrancer_root),
        settings={"enabled": False},
    )

    assert out == {
        "searched": False,
        "results": [],
        "total_count": 0,
        "returned": 0,
        "partial": False,
        "workspaces": [],
        "reason": "disabled",
    }
