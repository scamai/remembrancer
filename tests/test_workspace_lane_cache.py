"""The workspace-evidence lane's process cache.

Five properties, each against a real Git fixture:

* **Equivalence.** The cached lane returns exactly the payload of the pre-cache
  algorithm, reproduced here as ``_reference_search``: every Git command, a
  fresh read of every selected file and a fresh ``retrieve.bm25.rank`` on every
  call, snippets for every hit, then sort and page. Checked across clean,
  staged, unstaged, untracked, deleted, renamed, committed, HEAD-only,
  ``reset --soft``, detached and exclude-file states, on both the first call
  after a change and the warm call after it.
* **Freshness.** The cache key is the index file's content plus HEAD; the
  worktree-dependent sets are recomputed on every call.
* **Deadline.** Indexing, ranking and the page's snippets stop at the call's
  deadline and return ``partial`` results instead of running past it.
* **Lock.** No Git command, file read, indexing, ranking or snippet runs while
  ``_CACHE_LOCK`` is held.
* **Secret paths.** A secret file the include list matches stays out of every
  payload on both selection paths: tracked (memoized per index state) and
  untracked (recomputed each call).

Kept mutations (``test_kept_mutation_*``) load ``retrieve/workspace.py`` with
one guard removed and assert the defect comes back, so each property test is
shown to tell the fixed code from the broken one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import pytest

import retrieve.workspace as ws
from retrieve.bm25 import rank as reference_rank

WORKSPACE_SOURCE = Path(ws.__file__)


# ─── Fixture repository ──────────────────────────────────────────────────────

SECRET_TOKEN = "tracked-secret-garnet"
TRACKED_SECRET_PATHS = (
    "credentials.json",
    "keys.pem",
    "deploy/service-account.json",
    "ops/.ssh/notes.md",
)


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
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "internal"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / ".gitignore").write_text("ignored/\n*.ignored\n")
    (repo / "remembrancer").mkdir()
    (repo / "remembrancer" / "README.md").write_text("remembrancer root shared readme\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "design.md").write_text(
        "# Design\nclean-orchid baseline shared shared\nranking token notes\n"
    )
    (repo / "docs" / "other.md").write_text("other-umber shared\nsecond line\n")
    (repo / "service.py").write_text(
        "STATE = 'clean-saffron'\nAPI_KEY = \"sample_value_that_must_not_escape\"\n"
    )
    (repo / "notes.txt").write_text("plain notes about ranking and design\n")
    for index in range(6):
        (repo / "docs" / f"bulk-{index}.md").write_text(
            f"bulk document {index} shared filler\n" + "padding line\n" * index
        )
    (repo / ".env").write_text("API_KEY=forbidden-ruby\n")
    # Committed secret files that the include list DOES match (``**/*.json``,
    # ``**/*.pem``, ``**/*.md``): only the secret-path filter keeps them out,
    # on the tracked selection path that the cache memoizes.
    for path in TRACKED_SECRET_PATHS:
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(f'{{"note": "{SECRET_TOKEN} shared"}}\n')
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "fixture")
    return repo


@pytest.fixture(autouse=True)
def _clear_cache():
    ws.clear_workspace_cache()
    yield
    ws.clear_workspace_cache()


def _settings(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "enabled": True,
        "max_results": 20,
        "snippet_chars": 120,
        "max_files": 100,
        "max_file_bytes": 4096,
        "max_index_bytes": 65536,
        "max_seconds": 30,
        "workspaces": [
            {
                "id": "internal-test",
                "root": "..",
                "include": [
                    "**/*.md",
                    "**/*.py",
                    "**/*.txt",
                    "**/*.json",
                    "**/*.pem",
                    ".gitignore",
                ],
                "exclude": ["remembrancer/context/**", "remembrancer/events/**"],
            }
        ],
    }
    settings.update(overrides)
    return settings


def _search(module, repo: Path, query: str, **overrides: Any) -> dict[str, Any]:
    return module.search_workspaces(
        query,
        remembrancer_root=str(repo / "remembrancer"),
        settings=_settings(**overrides),
    )


# ─── Reference: the pre-cache algorithm, uncached ───────────────────────────────


def _reference_state(root: Path, deadline: float) -> dict[str, Any]:
    """The pre-cache ``_workspace_state``: all eight Git commands, in order."""
    repo_top = Path(
        ws._git_text(root, "rev-parse", "--show-toplevel", deadline=deadline)
    ).resolve(strict=True)
    prefix_path = root.relative_to(repo_top)
    prefix = "" if str(prefix_path) == "." else prefix_path.as_posix()

    def paths(*args: str) -> set[str]:
        return ws._local_paths(
            ws._nul_paths(ws._run_git(repo_top, *args, deadline=deadline)), prefix
        )

    tracked = paths("ls-files", "-z")
    untracked = paths("ls-files", "--others", "--exclude-standard", "-z")
    staged = paths("diff", "--cached", "--name-only", "-z", "HEAD")
    modified = paths("diff", "--name-only", "-z")
    oids = ws._stage_oids(repo_top, prefix, deadline=deadline)
    head = ws._git_text(repo_top, "rev-parse", "HEAD", deadline=deadline)
    if not re.fullmatch(r"[0-9a-f]{40,64}", head):
        raise ws.GitWorkspaceError("git_head")
    branch = ws._git_optional_text(
        repo_top, "symbolic-ref", "--short", "-q", "HEAD", deadline=deadline
    )
    return {
        "tracked": tracked,
        "untracked": untracked,
        "staged": staged,
        "modified": modified,
        "oids": oids,
        "head": head,
        "branch": branch,
        "detached": not bool(branch),
    }


def _reference_refresh(root, workspace, limits, deadline):
    """The pre-cache ``_refresh_workspace`` with no document cache at all."""
    import hashlib

    state = _reference_state(root, deadline)
    include = tuple(i for i in workspace.get("include", []) if isinstance(i, str))
    exclude = tuple(i for i in workspace.get("exclude", []) if isinstance(i, str))
    all_paths = sorted(state["tracked"] | state["untracked"])
    selected = [p for p in all_paths if ws._selected(p, include, exclude)]
    partial = False
    reason = None
    if len(selected) > limits["max_files"]:
        selected = selected[: limits["max_files"]]
        partial = True
        reason = "file_budget"
    documents: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    indexed = 0
    for path in selected:
        worktree_state = ws._states(path, state)
        read = ws._safe_read(root, path, limits["max_file_bytes"])
        if read is None:
            continue
        text, content_hash = read
        size = len(text.encode("utf-8"))
        if indexed + size > limits["max_index_bytes"]:
            partial = True
            reason = reason or "index_budget"
            break
        indexed += size
        documents[path] = (text, content_hash, worktree_state)
    ordered = [(path, *documents[path]) for path in sorted(documents)]
    manifest = "\n".join(
        f"{path}\0{','.join(states)}\0{sha}" for path, _text, sha, states in ordered
    ).encode()
    metadata = {
        "id": str(workspace["id"]),
        "head": state["head"],
        "branch": state["branch"] or None,
        "detached": state["detached"],
        "dirty": bool(state["untracked"] or state["staged"] or state["modified"]),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    return ordered, metadata, partial, reason


def _reference_search(
    query: str,
    *,
    remembrancer_root: str,
    settings: Mapping[str, Any],
    refreshed: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """``refreshed`` memoizes each workspace's uncached refresh across queries
    run against one unchanged state (a test-time saving only)."""
    if not isinstance(settings, Mapping) or settings.get("enabled") is not True:
        return ws._empty("disabled")
    workspaces = settings.get("workspaces")
    if not isinstance(workspaces, list) or not workspaces:
        return ws._empty("disabled")
    limits = ws._limits(settings)
    deadline = time.monotonic() + 600
    rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    partial = False
    reason = None
    valid = 0
    memo = {} if refreshed is None else refreshed
    for position, workspace in enumerate(workspaces):
        if not isinstance(workspace, Mapping):
            continue
        workspace_id = workspace.get("id")
        if not isinstance(workspace_id, str) or not ws._WORKSPACE_ID.fullmatch(
            workspace_id
        ):
            continue
        if position not in memo:
            try:
                root = ws._same_repository_root(
                    Path(remembrancer_root), workspace.get("root"), deadline=deadline
                )
                memo[position] = _reference_refresh(root, workspace, limits, deadline)
            except ws.UnsafeRootError:
                memo[position] = "unsafe_root"
            except ws.GitWorkspaceError as exc:
                memo[position] = str(exc)
        outcome = memo[position]
        if isinstance(outcome, str):
            reason = reason or outcome
            partial = partial or outcome == "time_budget"
            continue
        documents, metadata, was_partial, partial_reason = outcome
        valid += 1
        meta.append(metadata)
        partial = partial or was_partial
        if partial_reason:
            reason = reason or partial_reason
        ranking_docs = [f"{path} {path}\n{text}" for path, text, _s, _w in documents]
        for index, score in reference_rank(query, ranking_docs):
            if score <= 0:
                continue
            path, text, sha, states = documents[index]
            snippet, line_start, line_end = ws._snippet(
                text, query, limits["snippet_chars"]
            )
            rows.append(
                {
                    "workspace": workspace_id,
                    "path": path,
                    "line_start": line_start,
                    "line_end": line_end,
                    "snippet": snippet,
                    "score": round(float(score), 6),
                    "content_sha256": sha,
                    "worktree_state": list(states),
                    "trust": "unverified-local-worktree",
                }
            )
    if valid == 0:
        return ws._empty(reason or "disabled")
    rows.sort(key=lambda row: (-float(row["score"]), row["workspace"], row["path"]))
    page = rows[: limits["max_results"]]
    result = {
        "searched": True,
        "results": page,
        "total_count": len(rows),
        "returned": len(page),
        "partial": partial,
        "workspaces": meta,
    }
    if reason:
        result["reason"] = reason
    return result


QUERIES = (
    "shared",
    "clean-orchid",
    "ranking token",
    "staged-topaz",
    "unstaged-cobalt",
    "untracked-vermilion",
    "API_KEY",
    "design",
    "",
    "bulk document",
    SECRET_TOKEN,
)


def _assert_equivalent(
    module, repo: Path, label: str = "", *, queries=QUERIES, **overrides
) -> None:
    """First call after a change, then the warm call, both equal the reference.

    The failure message starts with ``label|`` so a kept mutation can pin the
    state at which its defect shows.
    """
    refreshed: dict[int, Any] = {}
    expected = {
        query: _reference_search(
            query,
            remembrancer_root=str(repo / "remembrancer"),
            settings=_settings(**overrides),
            refreshed=refreshed,
        )
        for query in queries
    }
    for attempt in ("first", "warm"):
        for query in queries:
            actual = _search(module, repo, query, **overrides)
            assert actual == expected[query], f"{label}|{attempt}|{query!r}"


# ─── Kept-mutation loader ────────────────────────────────────────────────────


def _mutant(original: str, replacement: str):
    """``retrieve/workspace.py`` with one guard removed, as a fresh module.

    The anchor must occur exactly once, so a refactor that moves the guard
    fails the test instead of silently mutating nothing.
    """
    source = WORKSPACE_SOURCE.read_text(encoding="utf-8")
    assert (
        source.count(original) == 1
    ), f"mutation anchor {original!r} must occur exactly once"
    name = f"workspace_mutant_{abs(hash(original))}"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(WORKSPACE_SOURCE)
    sys.modules[name] = module  # dataclasses resolve annotations through here
    try:
        exec(
            compile(
                source.replace(original, replacement), str(WORKSPACE_SOURCE), "exec"
            ),
            module.__dict__,
        )
    finally:
        sys.modules.pop(name, None)
    return module


# ─── Equivalence and freshness across workspace states ───────────────────────


def _walk_states(module, repo: Path) -> None:
    """Drive the fixture through every state change; check each one."""
    design = repo / "docs" / "design.md"
    service = repo / "service.py"
    proposal = repo / "proposal.md"
    other = repo / "docs" / "other.md"

    _assert_equivalent(module, repo, "clean")  # clean

    design.write_text("# Design\nstaged-topaz shared shared\nranking token notes\n")
    _git(repo, "add", "docs/design.md")
    _assert_equivalent(module, repo, "staged change")  # staged change

    service.write_text("STATE = 'unstaged-cobalt'\nshared = 1\n")
    _assert_equivalent(module, repo, "unstaged change")  # unstaged change

    design.write_text("# Design\nstaged-topaz then unstaged-cobalt shared\n")
    _assert_equivalent(
        module, repo, "staged and modified"
    )  # staged and modified on one file

    proposal.write_text("untracked-vermilion shared proposal\n")
    _assert_equivalent(module, repo, "untracked file")  # untracked file

    proposal.write_text("untracked-vermilion edited, now longer shared shared\n")
    _assert_equivalent(module, repo, "untracked edited")  # untracked file edited

    other.unlink()
    _assert_equivalent(module, repo, "unstaged deletion")  # unstaged deletion

    _git(repo, "rm", "-q", "--cached", "docs/bulk-0.md")
    _assert_equivalent(
        module, repo, "staged deletion"
    )  # staged deletion (file becomes untracked)

    _git(repo, "mv", "docs/bulk-1.md", "docs/renamed-bulk.md")
    _assert_equivalent(module, repo, "staged rename")  # staged rename

    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "exclude").write_text("proposal.md\n")
    _assert_equivalent(
        module, repo, "exclude file"
    )  # untracked set changes, index does not

    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "second")
    _assert_equivalent(module, repo, "new commit")  # new commit, clean again

    _git(repo, "commit", "-q", "--allow-empty", "-m", "head only")
    _assert_equivalent(module, repo, "empty commit")  # HEAD moves, index unchanged

    _git(repo, "reset", "-q", "--soft", "HEAD~2")
    _assert_equivalent(
        module, repo, "reset --soft"
    )  # HEAD back, index unchanged: all staged

    _git(repo, "checkout", "-q", "--detach")
    _assert_equivalent(module, repo, "detached")  # detached HEAD


def test_cached_lane_matches_reference_in_every_workspace_state(repo):
    _walk_states(ws, repo)


def test_cached_lane_matches_reference_under_budgets_and_subdirectory_roots(repo):
    (repo / "proposal.md").write_text("untracked-vermilion shared\n")
    (repo / "docs" / "design.md").write_text("# Design\nunstaged-cobalt shared\n")
    _assert_equivalent(ws, repo, max_files=3)  # file_budget
    _assert_equivalent(ws, repo, max_index_bytes=1024, max_file_bytes=200)
    _assert_equivalent(ws, repo, max_results=2, snippet_chars=80)
    two = [
        {
            "id": "whole-repo",
            "root": "..",
            "include": ["**/*.md", "**/*.py"],
            "exclude": [],
        },
        {"id": "remembrancer-only", "root": ".", "include": ["**/*.md"], "exclude": []},
        {"id": "escape", "root": "../../..", "include": ["**/*.md"]},
    ]
    _assert_equivalent(ws, repo, workspaces=two)
    # A different policy for the same root is a different cache entry.
    narrow = [{"id": "narrow", "root": "..", "include": ["docs/*.md"]}]
    _assert_equivalent(ws, repo, workspaces=narrow)
    _assert_equivalent(ws, repo)


def test_same_size_same_mtime_edit_of_a_committed_file_is_seen(repo):
    path = repo / "docs" / "design.md"
    _assert_equivalent(ws, repo)
    old_stat = path.stat()
    original = path.read_text()
    replacement = original.replace("clean-orchid", "dirty-orchid")
    assert len(replacement) == len(original)
    path.write_text(replacement)
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    _assert_equivalent(ws, repo, queries=("dirty-orchid", "clean-orchid", "shared"))


def test_concurrent_calls_return_the_reference_payload(repo):
    (repo / "proposal.md").write_text("untracked-vermilion shared\n")
    expected = {
        query: _reference_search(
            query, remembrancer_root=str(repo / "remembrancer"), settings=_settings()
        )
        for query in QUERIES
    }
    errors: list[BaseException] = []
    mismatches: list[str] = []

    def worker(offset: int) -> None:
        try:
            for step in range(3 * len(QUERIES)):
                query = QUERIES[(offset + step) % len(QUERIES)]
                if _search(ws, repo, query) != expected[query]:
                    mismatches.append(query)
                if step % 7 == 0:
                    ws.clear_workspace_cache()
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    # A worker still running past the join checked nothing yet.
    assert not any(thread.is_alive() for thread in threads)
    assert not errors
    assert not mismatches


# ─── Secret-path exclusion on both selection paths ───────────────────────────

UNTRACKED_SECRET_PATHS = ("staging/credentials.json", "staging/keys.pem")


def _secret_leaks(module, repo: Path) -> list[str]:
    """Every way a secret file shows up in a payload, ``attempt|query|what``.

    The fixture's committed secrets exercise the tracked selection (computed
    on the first call, served from the cache on the warm one); the files
    written here exercise the untracked selection, recomputed on every call.
    Each holds ``SECRET_TOKEN`` in text no snippet redaction would hide.
    """
    for path in UNTRACKED_SECRET_PATHS:
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(f'{{"note": "{SECRET_TOKEN} shared"}}\n')
    secrets = set(TRACKED_SECRET_PATHS) | set(UNTRACKED_SECRET_PATHS)
    module.clear_workspace_cache()
    leaks: list[str] = []
    for attempt in ("first", "warm"):
        for query in (SECRET_TOKEN, "note", "shared"):
            out = _search(module, repo, query)
            assert out["searched"] is True, out
            leaks.extend(
                f"{attempt}|{query}|{row['path']}"
                for row in out["results"]
                if row["path"] in secrets
            )
            if SECRET_TOKEN in json.dumps(out):
                leaks.append(f"{attempt}|{query}|token")
    return leaks


def test_secret_files_matched_by_include_never_reach_results(repo):
    assert _secret_leaks(ws, repo) == []


# ─── What a warm call does and does not re-run ───────────────────────────────


class _Counts:
    def __init__(self, module, monkeypatch):
        self.git: list[tuple[str, ...]] = []
        self.calls = {"_safe_read": 0, "_term_stats": 0, "_build_corpus": 0}
        real_git = module._run_git

        def git(root, *args, **kwargs):
            self.git.append(args)
            return real_git(root, *args, **kwargs)

        monkeypatch.setattr(module, "_run_git", git)
        for name in self.calls:
            real = getattr(module, name)

            def counted(*args, _real=real, _name=name, **kwargs):
                self.calls[_name] += 1
                return _real(*args, **kwargs)

            monkeypatch.setattr(module, name, counted)

    def reset(self) -> None:
        self.git.clear()
        for name in self.calls:
            self.calls[name] = 0

    def ran(self, *args: str) -> bool:
        return tuple(args) in self.git


def test_warm_call_reuses_index_state_documents_and_corpus(repo, monkeypatch):
    first = _search(ws, repo, "shared")
    counts = _Counts(ws, monkeypatch)
    second = _search(ws, repo, "shared")

    assert second == first
    assert counts.calls == {"_safe_read": 0, "_term_stats": 0, "_build_corpus": 0}
    assert not counts.ran("ls-files", "-z")
    assert not counts.ran("ls-files", "--stage", "-z")
    assert not counts.ran("diff", "--cached", "--name-only", "-z", "HEAD")
    # The worktree-dependent sets are never served from the cache.
    assert counts.ran("ls-files", "--others", "--exclude-standard", "-z")
    assert counts.ran("diff", "--name-only", "-z")


def test_one_changed_file_reindexes_only_that_document(repo, monkeypatch):
    _search(ws, repo, "shared")
    counts = _Counts(ws, monkeypatch)
    (repo / "docs" / "design.md").write_text("# Design\nchanged-orchid shared\n")

    out = _search(ws, repo, "changed-orchid")

    assert out["results"][0]["worktree_state"] == ["modified"]
    assert counts.calls == {"_safe_read": 1, "_term_stats": 1, "_build_corpus": 1}


def test_index_touch_without_content_change_keeps_the_cache(repo, monkeypatch):
    _search(ws, repo, "shared")
    index = Path(_git(repo, "rev-parse", "--absolute-git-dir")) / "index"
    os.utime(index, None)
    counts = _Counts(ws, monkeypatch)
    _search(ws, repo, "shared")
    assert not counts.ran("ls-files", "-z")


def _index_key_sees_same_stat_content_change(module, tmp_path: Path) -> bool:
    """Rewrite an index-sized file in place, same size, mtime restored."""
    index = tmp_path / "index"
    index.write_bytes(b"A" * 4096)
    before = module._index_key(tmp_path, "", "0" * 40, index)
    old = index.stat()
    with open(index, "r+b") as handle:  # same inode, same size
        handle.write(b"B")
    os.utime(index, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert index.stat().st_ino == old.st_ino and index.stat().st_size == old.st_size
    after = module._index_key(tmp_path, "", "0" * 40, index)
    return before is not None and after is not None and before != after


def test_index_key_is_the_index_content_not_its_stat(tmp_path):
    assert _index_key_sees_same_stat_content_change(ws, tmp_path)


def _unreadable_index_is_never_reused(module, repo: Path, monkeypatch) -> bool:
    monkeypatch.setattr(module, "_index_file_sha256", lambda _path: None)
    module.clear_workspace_cache()
    _search(module, repo, "clean-orchid")
    (repo / "docs" / "design.md").write_text("# Design\nstaged-topaz shared\n")
    _git(repo, "add", "docs/design.md")
    out = _search(module, repo, "staged-topaz")
    return bool(out["results"]) and out["results"][0]["worktree_state"] == ["staged"]


def test_an_index_that_cannot_be_hashed_reruns_git_every_call(repo, monkeypatch):
    assert _unreadable_index_is_never_reused(ws, repo, monkeypatch)


def _head_moved_during_git_is_not_cached(module, repo: Path, monkeypatch) -> bool:
    """HEAD moves away and back while the index-derived commands run.

    The first call computes the staged set against the moved HEAD; it must not
    be cached under the original HEAD, or the next call (HEAD and index exactly
    as before) is served a staged set that is no longer true.
    """
    (repo / "docs" / "design.md").write_text("# Design\nsecond-commit shared\n")
    _git(repo, "commit", "-q", "-am", "second")
    second = _git(repo, "rev-parse", "HEAD")
    first = _git(repo, "rev-parse", "HEAD~1")
    real_git = module._run_git
    moved = {"done": False}

    def git(root, *args, **kwargs):
        if args[:2] == ("diff", "--cached") and not moved["done"]:
            moved["done"] = True
            _git(repo, "reset", "-q", "--soft", first)
        return real_git(root, *args, **kwargs)

    monkeypatch.setattr(module, "_run_git", git)
    module.clear_workspace_cache()
    _search(module, repo, "second-commit")
    assert moved["done"]
    _git(repo, "reset", "-q", "--soft", second)  # index untouched throughout
    out = _search(module, repo, "second-commit")
    return out["results"][0]["worktree_state"] == ["committed"]


def test_head_moving_during_git_commands_is_not_cached(repo, monkeypatch):
    assert _head_moved_during_git_is_not_cached(ws, repo, monkeypatch)


# ─── Deadline ────────────────────────────────────────────────────────────────

BUDGET = 1.0
SLACK = 0.45  # git fixture calls plus scheduler noise; well under one sleep run
SLEEP = 0.05


@pytest.fixture
def wide_repo(repo: Path) -> Path:
    for index in range(40):
        (repo / "docs" / f"wide-{index:02d}.md").write_text(
            f"widecorpus term {index}\n"
        )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "wide")
    return repo


def _slow_rank_call(module, repo: Path, monkeypatch) -> tuple[dict, float]:
    _search(module, repo, "widecorpus")  # warm: corpus built, ranking left
    real = module._score_document

    def slow(*args, **kwargs):
        time.sleep(SLEEP)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "_score_document", slow)
    started = time.monotonic()
    out = _search(module, repo, "widecorpus", max_seconds=BUDGET)
    return out, time.monotonic() - started


def test_ranking_past_the_deadline_returns_partial_results_in_budget(
    wide_repo, monkeypatch
):
    out, elapsed = _slow_rank_call(ws, wide_repo, monkeypatch)

    assert elapsed < BUDGET + SLACK
    assert out["searched"] is True
    assert out["partial"] is True
    assert out["reason"] == "time_budget"
    assert out["results"], "the documents scored before the deadline are returned"
    assert out["total_count"] < 40


def _slow_index_call(module, repo: Path, monkeypatch) -> tuple[dict, float]:
    real = module._term_stats

    def slow(*args, **kwargs):
        time.sleep(SLEEP)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "_term_stats", slow)
    started = time.monotonic()
    out = _search(module, repo, "widecorpus", max_seconds=BUDGET)
    return out, time.monotonic() - started


def test_indexing_past_the_deadline_is_partial_and_the_next_call_resumes(
    wide_repo, monkeypatch
):
    out, elapsed = _slow_index_call(ws, wide_repo, monkeypatch)

    assert elapsed < BUDGET + SLACK
    assert out["searched"] is True
    assert out["partial"] is True
    assert out["reason"] == "time_budget"
    assert out["results"] == []

    monkeypatch.undo()
    counts = _Counts(ws, monkeypatch)
    resumed = _search(ws, wide_repo, "widecorpus")
    (entry,) = ws._CACHE.values()
    # Statistics computed before the deadline were kept: only the rest ran.
    assert 0 < counts.calls["_term_stats"] < len(entry.documents)
    assert resumed == _reference_search(
        "widecorpus", remembrancer_root=str(wide_repo / "remembrancer"), settings=_settings()
    )


def test_corpus_statistics_stop_at_an_expired_deadline():
    stats = [ws._TermStats(freqs={"a": 1}, length=1)] * 3
    assert ws._build_corpus("fp", stats, time.monotonic() - 1) is None
    built = ws._build_corpus("fp", stats, time.monotonic() + 30)
    assert built is not None and built.count == 3


class _SlowKeys(dict):
    def keys(self):
        time.sleep(0.005)
        return super().keys()


def test_document_frequency_pass_stops_at_the_deadline():
    stats = [ws._TermStats(freqs=_SlowKeys(a=1), length=1) for _ in range(200)]
    started = time.monotonic()
    assert ws._build_corpus("fp", stats, started + 0.1) is None
    # 200 slow documents take 1 s; the check every 64 documents stops at ~0.32 s.
    assert time.monotonic() - started < 0.7


def _slow_idf_build(module, monkeypatch):
    import math
    from types import SimpleNamespace

    calls = {"n": 0}

    def slow_log(value):
        calls["n"] += 1
        if calls["n"] % 64 == 0:
            time.sleep(0.005)
        return math.log(value)

    monkeypatch.setattr(module, "math", SimpleNamespace(log=slow_log))
    stats = [
        module._TermStats(freqs={f"t{i}": 1 for i in range(10_000)}, length=10_000)
    ]
    # Unchecked, the 10,000 terms take ~0.8 s and return a corpus; the check
    # every 4,096 terms stops at ~0.32 s.
    built = module._build_corpus("fp", stats, time.monotonic() + 0.1)
    return built, calls["n"]


def test_idf_pass_stops_at_the_deadline(monkeypatch):
    built, logs = _slow_idf_build(ws, monkeypatch)
    assert built is None
    assert logs < 10_000


def test_snippets_past_the_grace_drop_rows_and_flag_partial(wide_repo, monkeypatch):
    _search(ws, wide_repo, "widecorpus")
    real = ws._snippet

    def slow(*args, **kwargs):
        time.sleep(0.3)
        return real(*args, **kwargs)

    monkeypatch.setattr(ws, "_snippet", slow)
    started = time.monotonic()
    out = _search(ws, wide_repo, "widecorpus", max_seconds=0.5, max_results=20)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5 + ws._SNIPPET_GRACE_SECONDS + 0.3 + SLACK
    assert out["partial"] is True and out["reason"] == "time_budget"
    assert 0 < out["returned"] < 20
    assert out["total_count"] == 40


def _tie_order(module, repo: Path, monkeypatch) -> list[str]:
    """Two rows whose scores differ only past the 6th decimal.

    The page is sorted on the ROUNDED score, then workspace, then path, as the
    pre-cache code sorted its finished rows: the later path, though it ranked
    higher by raw score, must not come first.
    """
    first = _search(module, repo, "shared")
    paths = sorted(row["path"] for row in first["results"])
    module.clear_workspace_cache()
    real = module._refresh_workspace

    def refresh(**kwargs):
        ordered, *rest, ranking = real(**kwargs)
        return [ordered[0], ordered[1]], *rest, ranking

    monkeypatch.setattr(module, "_refresh_workspace", refresh)
    monkeypatch.setattr(
        module,
        "_rank",
        lambda *_a, **_k: ([(1, 1.0000002), (0, 1.0000001)], True),
    )
    out = _search(module, repo, "shared")
    assert {row["score"] for row in out["results"]} == {1.0}
    assert paths  # the fixture has hits at all
    return [row["path"] for row in out["results"]]


def test_rows_tied_after_rounding_are_ordered_by_path(repo, monkeypatch):
    order = _tie_order(ws, repo, monkeypatch)
    assert order == sorted(order)


def test_cached_corpus_scores_equal_reference_bm25_bit_for_bit():
    paths = ["a.md", "b.md", "c.py", "empty.md", "z/deep path.txt"]
    texts = [
        "shared shared ranking token",
        "shared ranking\nprocesses processing processed",
        "TOKEN tokenization and shared",
        "",
        "ranking ranking ranking unrelated words here",
    ]
    documents = [
        ws._Document(
            path=path,
            text=text,
            content_sha256=str(index),
            worktree_state=("committed",),
            source_key="",
            size=len(text),
        )
        for index, (path, text) in enumerate(zip(paths, texts))
    ]
    stats = [ws._term_stats(document) for document in documents]
    corpus = ws._build_corpus("fp", stats, time.monotonic() + 30)
    ranking_docs = [ws._ranking_text(document) for document in documents]
    assert ranking_docs[0] == "a.md a.md\nshared shared ranking token"
    for query in (
        "shared ranking",
        "missing-term",
        "TOKEN token",
        "shared shared",
        "processes",
        "md",
        "deep",
    ):
        ranked, complete = ws._rank(query, corpus, stats, time.monotonic() + 30)
        assert complete
        assert ranked == reference_rank(query, ranking_docs), query
    assert ws._rank("", corpus, stats, time.monotonic() + 30) == ([], True)
    assert all(score == 0 for _i, score in reference_rank("", ranking_docs))


# ─── Lock ────────────────────────────────────────────────────────────────────


def _lock_is_free(module) -> bool:
    """Whether another thread could take ``_CACHE_LOCK`` right now."""
    result: list[bool] = []

    def probe() -> None:
        got = module._CACHE_LOCK.acquire(blocking=False)
        if got:
            module._CACHE_LOCK.release()
        result.append(got)

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join(5)
    return bool(result and result[0])


def _held_during_heavy_steps(module, repo: Path, monkeypatch) -> set[str]:
    """Names of the heavy steps that ran while the lock was held."""
    held: set[str] = set()
    for name in (
        "_run_git",
        "_safe_read",
        "_term_stats",
        "_score_document",
        "_snippet",
    ):
        real = getattr(module, name)

        def watched(*args, _real=real, _name=name, **kwargs):
            if not _lock_is_free(module):
                held.add(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, name, watched)
    module.clear_workspace_cache()
    _search(module, repo, "shared")  # cold: git, reads, indexing, ranking
    _search(module, repo, "shared")  # warm: ranking and snippets
    return held


def test_no_heavy_step_runs_under_the_cache_lock(repo, monkeypatch):
    assert _held_during_heavy_steps(ws, repo, monkeypatch) == set()


def _concurrent_call_finishes_during_slow_rank(module, repo: Path, monkeypatch) -> bool:
    _search(module, repo, "shared")
    real = module._score_document
    blocked = threading.Event()
    release = threading.Event()

    def blocking(corpus, stats, q_terms):
        if q_terms == ["alpha"] and not blocked.is_set():
            blocked.set()
            release.wait(10)
        return real(corpus, stats, q_terms)

    monkeypatch.setattr(module, "_score_document", blocking)
    slow = threading.Thread(target=_search, args=(module, repo, "alpha"))
    fast = None
    slow.start()
    try:
        assert blocked.wait(5)
        done = threading.Event()
        fast = threading.Thread(
            target=lambda: (_search(module, repo, "shared"), done.set())
        )
        fast.start()
        finished = done.wait(3)
    finally:
        release.set()
        slow.join(10)
        if fast is not None:
            fast.join(10)
    return finished


def test_a_slow_rank_does_not_block_a_concurrent_call(repo, monkeypatch):
    assert _concurrent_call_finishes_during_slow_rank(ws, repo, monkeypatch)


# ─── Kept mutations: each removes one guard and the defect returns ───────────

RANK_CHECK = (
    "    for position, document_stats in enumerate(stats):\n"
    "        if time.monotonic() > deadline:\n"
    "            complete = False\n"
    "            break\n"
)
STATS_CHECK = "if document_stats is None and time.monotonic() <= deadline:"
RANK_CALL = "        scored, complete = _rank(query, corpus, stats, deadline)\n"


def test_kept_mutation_ranking_without_the_deadline_overruns(wide_repo, monkeypatch):
    mutant = _mutant(
        RANK_CHECK,
        "    for position, document_stats in enumerate(stats):\n"
        "        complete = True\n",
    )
    out, elapsed = _slow_rank_call(mutant, wide_repo, monkeypatch)
    assert elapsed >= 40 * SLEEP
    assert out["total_count"] == 40  # every document was scored, budget or not


def test_kept_mutation_indexing_without_the_deadline_overruns(wide_repo, monkeypatch):
    mutant = _mutant(STATS_CHECK, "if document_stats is None:")
    out, elapsed = _slow_index_call(mutant, wide_repo, monkeypatch)
    assert elapsed >= 40 * SLEEP


def test_kept_mutation_document_frequency_without_the_deadline_overruns():
    mutant = _mutant(
        "        if position % 64 == 0 and time.monotonic() > deadline:\n"
        "            return None\n",
        "",
    )
    stats = [mutant._TermStats(freqs=_SlowKeys(a=1), length=1) for _ in range(200)]
    started = time.monotonic()
    mutant._build_corpus("fp", stats, started + 0.1)
    assert time.monotonic() - started >= 200 * 0.005


def test_kept_mutation_idf_without_the_deadline_returns_a_late_corpus(monkeypatch):
    mutant = _mutant(
        "        if position % 4096 == 0 and time.monotonic() > deadline:\n"
        "            return None\n",
        "",
    )
    built, logs = _slow_idf_build(mutant, monkeypatch)
    assert built is not None and logs == 10_000


def test_kept_mutation_snippets_without_the_grace_check_overrun(wide_repo, monkeypatch):
    mutant = _mutant(
        "        if time.monotonic() > snippet_deadline:\n"
        "            partial = True\n"
        '            reason = reason or "time_budget"\n'
        "            break\n",
        "",
    )
    _search(mutant, wide_repo, "widecorpus")
    real = mutant._snippet
    monkeypatch.setattr(
        mutant, "_snippet", lambda *a, **k: (time.sleep(0.1), real(*a, **k))[1]
    )
    started = time.monotonic()
    out = _search(mutant, wide_repo, "widecorpus", max_seconds=0.5, max_results=20)
    assert time.monotonic() - started >= 20 * 0.1
    assert out["returned"] == 20


def test_kept_mutation_page_sort_without_the_path_tie_break(repo, monkeypatch):
    mutant = _mutant(
        "    candidates.sort(key=lambda row: (-row[0], row[1], row[2].path))\n",
        "    candidates.sort(key=lambda row: (-row[0], row[1]))\n",
    )
    order = _tie_order(mutant, repo, monkeypatch)
    assert order != sorted(order)


def test_kept_mutation_ranking_under_the_lock_blocks_others(repo, monkeypatch):
    mutant = _mutant(
        RANK_CALL,
        "        with _CACHE_LOCK:\n"
        "            scored, complete = _rank(query, corpus, stats, deadline)\n",
    )
    assert "_score_document" in _held_during_heavy_steps(mutant, repo, monkeypatch)
    monkeypatch.undo()
    assert not _concurrent_call_finishes_during_slow_rank(mutant, repo, monkeypatch)


def test_kept_mutation_the_old_whole_call_lock_is_caught(repo, monkeypatch):
    """The pre-cache shape: one lock around every workspace's refresh and rank."""
    mutant = _mutant(
        "    for workspace in workspaces:\n        if not isinstance(workspace, Mapping):",
        "    _CACHE_LOCK.acquire()\n"
        "    for workspace in workspaces:\n        if not isinstance(workspace, Mapping):",
    )
    try:
        held = _held_during_heavy_steps(mutant, repo, monkeypatch)
    finally:
        while True:  # the mutant never releases what it took
            try:
                mutant._CACHE_LOCK.release()
            except RuntimeError:
                break
    assert {"_run_git", "_safe_read", "_term_stats", "_score_document"} <= held


def test_kept_mutation_index_key_without_head_serves_a_stale_staged_set(repo):
    mutant = _mutant(
        "    return (str(repo_top), prefix, str(index_path), head, digest)\n",
        "    return (str(repo_top), prefix, str(index_path), digest)\n",
    )
    with pytest.raises(AssertionError, match=r"^reset --soft\|"):
        _walk_states(mutant, repo)


def test_kept_mutation_stat_keyed_index_misses_a_content_change(tmp_path):
    mutant = _mutant(
        "    digest = _index_file_sha256(index_path)\n",
        "    info = index_path.stat()\n"
        "    digest = f'{info.st_ino}:{info.st_size}:{info.st_mtime_ns}'\n",
    )
    assert not _index_key_sees_same_stat_content_change(mutant, tmp_path)


def test_kept_mutation_reusing_an_unprovable_key_serves_stale_state(repo, monkeypatch):
    mutant = _mutant(
        "    if key is not None and previous is not None and previous.key == key:\n",
        "    if previous is not None and previous.key == key:\n",
    )
    assert not _unreadable_index_is_never_reused(mutant, repo, monkeypatch)


def test_kept_mutation_without_the_after_check_caches_a_mixed_state(repo, monkeypatch):
    mutant = _mutant(
        "        if _index_key(repo_top, prefix, head_after, index_path) != key:\n",
        "        if False:\n",
    )
    assert not _head_moved_during_git_is_not_cached(mutant, repo, monkeypatch)


def test_kept_mutation_fingerprint_without_content_reuses_a_stale_corpus(repo):
    mutant = _mutant(
        '        digest.update(document.content_sha256.encode("ascii"))\n', ""
    )
    with pytest.raises(AssertionError, match=r"^staged change\|"):
        _walk_states(mutant, repo)


def test_kept_mutation_term_stats_keyed_by_path_alone_go_stale(repo):
    mutant = _mutant(
        "        stats_key = (document.path, document.content_sha256)\n",
        "        stats_key = (document.path, '')\n",
    )
    with pytest.raises(AssertionError, match=r"^staged change\|"):
        _walk_states(mutant, repo)


def test_kept_mutation_reusing_selection_across_index_states_drops_files(repo):
    mutant = _mutant(
        "    if previous.index_state is not None and index_state is previous.index_state:\n",
        "    if previous.index_state is not None:\n",
    )
    with pytest.raises(AssertionError, match=r"^staged rename\|"):
        _walk_states(mutant, repo)


def _leaked_paths(leaks: list[str]) -> set[str]:
    return {leak.split("|")[2] for leak in leaks} - {"token"}


def test_kept_mutation_tracked_selection_without_the_secret_filter(repo):
    mutant = _mutant(
        "p for p in index_state.tracked if _selected(p, include, exclude)",
        "p for p in index_state.tracked"
        " if _matches(p, include) and not _matches(p, exclude)",
    )
    leaks = _secret_leaks(mutant, repo)
    # Every committed secret leaks, on the cold call and the cached warm call.
    assert _leaked_paths(leaks) == set(TRACKED_SECRET_PATHS)
    assert f"first|{SECRET_TOKEN}|credentials.json" in leaks
    assert f"warm|{SECRET_TOKEN}|credentials.json" in leaks


def test_kept_mutation_untracked_selection_without_the_secret_filter(repo):
    mutant = _mutant(
        'p for p in state["untracked"] if _selected(p, include, exclude)',
        'p for p in state["untracked"]'
        " if _matches(p, include) and not _matches(p, exclude)",
    )
    assert _leaked_paths(_secret_leaks(mutant, repo)) == set(UNTRACKED_SECRET_PATHS)
