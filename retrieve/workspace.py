"""Bounded, Git-aware evidence retrieval from configured local workspaces.

This is a read-only query-time overlay. It never writes Remembrancer memory, never
executes repository content, and never follows links outside the configured Git
top-level. ACL selection happens at the MCP boundary before this module is
called.

Process cache. One entry per (root, policy) holds:

* the Git state that is a pure function of the index file and HEAD -- tracked
  paths, stage-0 OIDs, the staged set and the policy-selected tracked paths --
  keyed on the SHA-256 of the index file's bytes plus the HEAD OID. The
  untracked and unstaged-modified sets depend on the worktree, which no index
  key can vouch for, so ``ls-files --others`` and ``diff`` run on every call;
* documents by blob OID (clean files) as before, and per-document BM25 term
  statistics keyed on (path, content SHA-256);
* the corpus-level BM25 statistics (IDF, average length) keyed on the ordered
  (path, content SHA-256) list, so an unchanged document set is never re-indexed.

``_CACHE_LOCK`` guards only reading and replacing an entry. Git, file reads,
indexing, ranking and snippets run outside it, and every one of those steps
checks the call's deadline, so a slow step returns ``partial`` results instead
of running past ``max_seconds``. Scores, order and snippets are identical to an
uncached ``retrieve.bm25.rank`` over the same documents.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from retrieve.bm25 import B, K1, tokenize

_HARD_MAX_RESULTS = 20
_HARD_MAX_SNIPPET_CHARS = 2_000
_HARD_MAX_FILES = 10_000
_HARD_MAX_FILE_BYTES = 1_048_576
_HARD_MAX_INDEX_BYTES = 268_435_456
_HARD_MAX_SECONDS = 30.0
_GIT_TIMEOUT_SECONDS = 10
_WORKSPACE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")

_SECRET_BASENAMES = frozenset(
    {
        "credentials.json",
        "service-account.json",
        "service_account.json",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }
)
_SECRET_SUFFIXES = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".kdbx",
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"password|passwd)\b\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_AUTHORIZATION_VALUE = re.compile(
    r"(?i)(\bauthorization\b\s*:\s*(?:bearer|basic)\s+)" r"[A-Za-z0-9._~+/=-]+"
)
_URI_CREDENTIAL = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^/\s:@]+:)[^@\s/]+(@)")
_KNOWN_TOKEN = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|"
    r"sk-[A-Za-z0-9_-]{20,}|"
    r"AKIA[0-9A-Z]{16}|"
    r"xox[baprs]-[A-Za-z0-9-]{20,})\b"
)
_PRIVATE_KEY_MARKER = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")


class WorkspaceError(RuntimeError):
    """Base class for stable local-provider refusal."""


class UnsafeRootError(WorkspaceError):
    """Configured root is not a relative path inside Remembrancer's Git repository."""


class GitWorkspaceError(WorkspaceError):
    """Git could not provide an authoritative workspace view."""


@dataclass(frozen=True)
class _Document:
    path: str
    text: str
    content_sha256: str
    worktree_state: tuple[str, ...]
    source_key: str
    size: int  # UTF-8 bytes of ``text``, counted once at read time


@dataclass(frozen=True)
class _IndexState:
    """Git state derived from the index file and HEAD alone.

    ``key`` is None when freshness could not be proven (the index file was
    unreadable, oversized, or changed while Git ran); such a state is used for
    its own call and never reused.
    """

    key: tuple[str, ...] | None
    tracked: frozenset[str]
    staged: frozenset[str]
    oids: Mapping[str, str]


@dataclass(frozen=True)
class _TermStats:
    """BM25 per-document statistics for one ranking document."""

    freqs: Mapping[str, int]
    length: int


@dataclass(frozen=True)
class _Corpus:
    """BM25 corpus statistics for one ordered document set."""

    fingerprint: str
    count: int
    avgdl: float
    idf: Mapping[str, float]


@dataclass(frozen=True)
class _Cache:
    """One immutable snapshot; a call replaces it, never mutates it."""

    documents: Mapping[str, _Document] = field(default_factory=dict)
    stats: Mapping[tuple[str, str], _TermStats] = field(default_factory=dict)
    index_state: _IndexState | None = None
    selected_tracked: tuple[str, ...] = ()
    corpus: _Corpus | None = None


_CACHE_LOCK = threading.RLock()
_CACHE: dict[tuple[str, str], _Cache] = {}
_EMPTY_CACHE = _Cache()
# The index file is hashed whole to key the cache; past this size the key is
# not computed and every call re-runs the index-derived Git commands.
_MAX_GIT_INDEX_FILE_BYTES = 268_435_456
_SNIPPET_GRACE_SECONDS = 0.25


def clear_workspace_cache() -> None:
    """Clear the process-local content cache (tests and explicit maintenance)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _empty(reason: str) -> dict[str, Any]:
    return {
        "searched": False,
        "results": [],
        "total_count": 0,
        "returned": 0,
        "partial": False,
        "workspaces": [],
        "reason": reason,
    }


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _git_timeout(deadline: float | None) -> float:
    if deadline is None:
        return float(_GIT_TIMEOUT_SECONDS)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise GitWorkspaceError("time_budget")
    return max(0.01, min(float(_GIT_TIMEOUT_SECONDS), remaining))


def _run_git(root: Path, *args: str, deadline: float | None = None) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            timeout=_git_timeout(deadline),
        )
    except subprocess.TimeoutExpired as exc:
        raise GitWorkspaceError("time_budget") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitWorkspaceError("git_error") from exc
    return result.stdout


def _git_text(root: Path, *args: str, deadline: float | None = None) -> str:
    try:
        return (
            _run_git(
                root,
                *args,
                deadline=deadline,
            )
            .decode("utf-8", "strict")
            .strip()
        )
    except UnicodeDecodeError as exc:
        raise GitWorkspaceError("git_encoding") from exc


def _git_optional_text(root: Path, *args: str, deadline: float | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            timeout=_git_timeout(deadline),
        )
    except subprocess.TimeoutExpired as exc:
        raise GitWorkspaceError("time_budget") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitWorkspaceError("git_error") from exc
    if result.returncode == 1:
        return ""
    if result.returncode != 0:
        raise GitWorkspaceError("git_error")
    try:
        return result.stdout.decode("utf-8", "strict").strip()
    except UnicodeDecodeError as exc:
        raise GitWorkspaceError("git_encoding") from exc


def _nul_paths(raw: bytes) -> set[str]:
    paths: set[str] = set()
    for item in raw.split(b"\0"):
        if not item:
            continue
        path = os.fsdecode(item)
        try:
            path.encode("utf-8", "strict")
        except UnicodeEncodeError:
            continue
        if _safe_relative_parts(path) is not None:
            paths.add(path)
    return paths


def _local_path(path: str, prefix: str) -> str | None:
    if not prefix:
        return path
    marker = prefix + "/"
    if not path.startswith(marker):
        return None
    local = path[len(marker) :]
    return local if _safe_relative_parts(local) is not None else None


def _local_paths(paths: set[str], prefix: str) -> set[str]:
    localized: set[str] = set()
    for path in paths:
        local = _local_path(path, prefix)
        if local is not None:
            localized.add(local)
    return localized


def _stage_oids(
    root: Path, prefix: str, deadline: float | None = None
) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in _run_git(root, "ls-files", "--stage", "-z", deadline=deadline).split(
        b"\0"
    ):
        if not item or b"\t" not in item:
            continue
        meta, raw_path = item.split(b"\t", 1)
        fields = meta.split()
        if len(fields) != 3 or fields[2] != b"0":
            continue
        path = os.fsdecode(raw_path)
        try:
            path.encode("utf-8", "strict")
        except UnicodeEncodeError:
            continue
        local = _local_path(path, prefix)
        if local is not None:
            result[local] = fields[1].decode("ascii", "strict")
    return result


def _safe_relative_parts(path: str) -> tuple[str, ...] | None:
    if not isinstance(path, str) or not path or "\x00" in path:
        return None
    pure = PurePosixPath(path)
    if pure.is_absolute():
        return None
    parts = pure.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    return tuple(parts)


def _matches(path: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            continue
        if fnmatch.fnmatchcase(path, pattern):
            return True
        if pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:]):
            return True
    return False


def _secret_path(path: str) -> bool:
    parts = [part.lower() for part in PurePosixPath(path).parts]
    if not parts:
        return True
    base = parts[-1]
    if base == ".env" or base.startswith(".env."):
        return True
    if base in _SECRET_BASENAMES:
        return True
    if any(base.endswith(suffix) for suffix in _SECRET_SUFFIXES):
        return True
    return any(part in {".git", ".ssh", ".gnupg"} for part in parts)


def _selected(path: str, include: Sequence[str], exclude: Sequence[str]) -> bool:
    if _secret_path(path):
        return False
    if not include or not _matches(path, include):
        return False
    return not _matches(path, exclude)


def _same_repository_root(
    remembrancer_root: Path, raw_root: Any, deadline: float | None = None
) -> Path:
    if not isinstance(raw_root, str) or not raw_root or "\x00" in raw_root:
        raise UnsafeRootError("unsafe_root")
    relative = Path(raw_root)
    if relative.is_absolute() or any(part == ".." for part in relative.parts[1:]):
        # A single leading ".." is supported for the monorepo parent. The
        # same-repository check below remains the authority.
        if raw_root != "..":
            raise UnsafeRootError("unsafe_root")
    try:
        remembrancer = remembrancer_root.resolve(strict=True)
        repo_top = Path(
            _git_text(
                remembrancer,
                "rev-parse",
                "--show-toplevel",
                deadline=deadline,
            )
        ).resolve(strict=True)
        candidate = (remembrancer / relative).resolve(strict=True)
    except GitWorkspaceError:
        raise
    except OSError as exc:
        raise UnsafeRootError("unsafe_root") from exc
    try:
        candidate.relative_to(repo_top)
    except ValueError as exc:
        raise UnsafeRootError("unsafe_root") from exc
    candidate_top = Path(
        _git_text(
            candidate,
            "rev-parse",
            "--show-toplevel",
            deadline=deadline,
        )
    ).resolve(strict=True)
    if candidate_top != repo_top:
        raise UnsafeRootError("unsafe_root")
    return candidate


def _safe_read(root: Path, path: str, max_bytes: int) -> tuple[str, str] | None:
    parts = _safe_relative_parts(path)
    if parts is None:
        return None
    directory_fds: list[int] = []
    file_fd: int | None = None
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    try:
        current_fd = os.open(str(root), directory_flags)
        directory_fds.append(current_fd)
        for component in parts[:-1]:
            current_fd = os.open(
                component,
                directory_flags,
                dir_fd=current_fd,
            )
            directory_fds.append(current_fd)
        file_fd = os.open(
            parts[-1],
            os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0),
            dir_fd=current_fd,
        )
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            return None
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(file_fd, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(file_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            getattr(before, "st_mtime_ns", int(before.st_mtime * 1e9)),
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            getattr(after, "st_mtime_ns", int(after.st_mtime * 1e9)),
        )
        if identity_before != identity_after or len(data) > max_bytes:
            return None
        if b"\x00" in data:
            return None
        try:
            text = data.decode("utf-8", "strict")
        except UnicodeDecodeError:
            return None
        return text, hashlib.sha256(data).hexdigest()
    except OSError:
        return None
    finally:
        if file_fd is not None:
            try:
                os.close(file_fd)
            except OSError:
                pass
        for descriptor in reversed(directory_fds):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _policy_fingerprint(workspace: Mapping[str, Any], limits: Mapping[str, Any]) -> str:
    material = {
        "version": 1,
        "id": workspace.get("id"),
        "root": workspace.get("root"),
        "include": workspace.get("include"),
        "exclude": workspace.get("exclude"),
        "max_file_bytes": limits["max_file_bytes"],
        "max_files": limits["max_files"],
        "max_index_bytes": limits["max_index_bytes"],
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _index_file_sha256(path: Path) -> str | None:
    """SHA-256 of the whole Git index file, or None when it cannot be read.

    Git replaces the index by renaming a finished lock file over it, so an open
    descriptor always sees one complete version.
    """
    try:
        with open(path, "rb") as handle:
            info = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_size > _MAX_GIT_INDEX_FILE_BYTES
            ):
                return None
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = handle.read(1_048_576)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_GIT_INDEX_FILE_BYTES:
                    return None
                digest.update(chunk)
            return digest.hexdigest()
    except OSError:
        return None


def _git_index_path(repo_top: Path, deadline: float | None) -> Path | None:
    """Where this checkout's index lives (linked worktrees and GIT_INDEX_FILE)."""
    try:
        raw = _git_text(repo_top, "rev-parse", "--git-path", "index", deadline=deadline)
    except GitWorkspaceError as exc:
        if str(exc) == "time_budget":
            raise
        return None
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else repo_top / path


def _index_key(
    repo_top: Path, prefix: str, head: str, index_path: Path | None
) -> tuple[str, ...] | None:
    if index_path is None:
        return None
    digest = _index_file_sha256(index_path)
    if digest is None:
        return None
    return (str(repo_top), prefix, str(index_path), head, digest)


def _index_state(
    repo_top: Path,
    prefix: str,
    head: str,
    deadline: float | None,
    previous: _IndexState | None,
) -> _IndexState:
    """Tracked paths, stage-0 OIDs and the staged set, reused while provably fresh.

    All three are functions of the index file's bytes and of HEAD's tree, so a
    state cached under (index SHA-256, HEAD OID) is exact. A state is cached
    only if the key read after the Git commands equals the key read before
    them; otherwise it serves this call and is recomputed on the next.
    """
    index_path = _git_index_path(repo_top, deadline)
    key = _index_key(repo_top, prefix, head, index_path)
    if key is not None and previous is not None and previous.key == key:
        return previous
    tracked = _local_paths(
        _nul_paths(_run_git(repo_top, "ls-files", "-z", deadline=deadline)),
        prefix,
    )
    staged = _local_paths(
        _nul_paths(
            _run_git(
                repo_top,
                "diff",
                "--cached",
                "--name-only",
                "-z",
                "HEAD",
                deadline=deadline,
            )
        ),
        prefix,
    )
    oids = _stage_oids(repo_top, prefix, deadline=deadline)
    if key is not None:
        head_after = _git_text(repo_top, "rev-parse", "HEAD", deadline=deadline)
        if _index_key(repo_top, prefix, head_after, index_path) != key:
            key = None
    return _IndexState(
        key=key,
        tracked=frozenset(tracked),
        staged=frozenset(staged),
        oids=oids,
    )


def _workspace_state(
    root: Path,
    deadline: float | None = None,
    previous: _IndexState | None = None,
) -> tuple[dict[str, Any], _IndexState]:
    repo_top = Path(
        _git_text(
            root,
            "rev-parse",
            "--show-toplevel",
            deadline=deadline,
        )
    ).resolve(strict=True)
    prefix_path = root.relative_to(repo_top)
    prefix = "" if str(prefix_path) == "." else prefix_path.as_posix()
    # The untracked and unstaged sets depend on the worktree, which no index
    # key can vouch for: both commands run on every call.
    untracked = _local_paths(
        _nul_paths(
            _run_git(
                repo_top,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                deadline=deadline,
            )
        ),
        prefix,
    )
    # `git diff` may rewrite a stat-dirty index (even with GIT_OPTIONAL_LOCKS=0),
    # so it runs before the index file is hashed for the key.
    modified = _local_paths(
        _nul_paths(_run_git(repo_top, "diff", "--name-only", "-z", deadline=deadline)),
        prefix,
    )
    head = _git_text(repo_top, "rev-parse", "HEAD", deadline=deadline)
    if not re.fullmatch(r"[0-9a-f]{40,64}", head):
        raise GitWorkspaceError("git_head")
    branch = _git_optional_text(
        repo_top,
        "symbolic-ref",
        "--short",
        "-q",
        "HEAD",
        deadline=deadline,
    )
    index_state = _index_state(repo_top, prefix, head, deadline, previous)
    state = {
        "tracked": index_state.tracked,
        "untracked": untracked,
        "staged": index_state.staged,
        "modified": modified,
        "oids": index_state.oids,
        "head": head,
        "branch": branch,
        "detached": not bool(branch),
    }
    return state, index_state


def _states(path: str, state: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    if path in state["untracked"]:
        values.append("untracked")
        return tuple(values)
    if path in state["staged"]:
        values.append("staged")
    if path in state["modified"]:
        values.append("modified")
    if not values:
        values.append("committed")
    return tuple(values)


def _ranking_text(document: _Document) -> str:
    """The string BM25 indexes for a document (the path counts twice)."""
    return f"{document.path} {document.path}\n{document.text}"


def _term_stats(document: _Document) -> _TermStats:
    tokens = tokenize(_ranking_text(document))
    return _TermStats(freqs=Counter(tokens), length=len(tokens))


def _corpus_fingerprint(ordered: Sequence[_Document]) -> str:
    """Identity of the ranked corpus: the ordered (path, content hash) list.

    The content hash fixes a document's text, and the path is part of its
    ranking text, so equal fingerprints index equal strings in equal order.
    """
    digest = hashlib.sha256(b"workspace-bm25-v1\0")
    for document in ordered:
        digest.update(document.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.content_sha256.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _build_corpus(
    fingerprint: str, stats: Sequence[_TermStats], deadline: float
) -> _Corpus | None:
    """Corpus statistics exactly as ``retrieve.bm25.BM25`` computes them.

    Returns None if the deadline passes first.
    """
    document_frequency: Counter[str] = Counter()
    for position, document_stats in enumerate(stats):
        if position % 64 == 0 and time.monotonic() > deadline:
            return None
        document_frequency.update(document_stats.freqs.keys())
    count = len(stats)
    idf: dict[str, float] = {}
    for position, (term, n_q) in enumerate(document_frequency.items()):
        if position % 4096 == 0 and time.monotonic() > deadline:
            return None
        idf[term] = math.log(1.0 + (count - n_q + 0.5) / (n_q + 0.5))
    avgdl = (sum(item.length for item in stats) / count) if count else 0.0
    return _Corpus(fingerprint=fingerprint, count=count, avgdl=avgdl, idf=idf)


def _refresh_workspace(
    *,
    root: Path,
    workspace: Mapping[str, Any],
    limits: Mapping[str, Any],
    deadline: float,
) -> tuple[
    list[_Document],
    dict[str, Any],
    bool,
    str | None,
    tuple[_Corpus, list[_TermStats]] | None,
]:
    """Refresh one workspace's documents and BM25 inputs.

    Holds ``_CACHE_LOCK`` only to read the previous snapshot and to publish the
    new one. The last element is None when the corpus could not be indexed
    inside the deadline (the call is then partial); term statistics computed
    before the deadline are still published, so the next call resumes there.
    """
    cache_key = (str(root), _policy_fingerprint(workspace, limits))
    with _CACHE_LOCK:
        previous = _CACHE.get(cache_key, _EMPTY_CACHE)
    state, index_state = _workspace_state(
        root, deadline=deadline, previous=previous.index_state
    )
    include = tuple(
        item for item in workspace.get("include", []) if isinstance(item, str)
    )
    exclude = tuple(
        item for item in workspace.get("exclude", []) if isinstance(item, str)
    )
    # Selection is per path, so filtering the tracked and untracked sets apart
    # and merging them equals filtering their sorted union.
    if previous.index_state is not None and index_state is previous.index_state:
        selected_tracked = previous.selected_tracked
    else:
        selected_tracked = tuple(
            sorted(p for p in index_state.tracked if _selected(p, include, exclude))
        )
    selected_untracked = {
        p for p in state["untracked"] if _selected(p, include, exclude)
    }
    if selected_untracked:
        selected = sorted(selected_untracked.union(selected_tracked))
    else:
        selected = list(selected_tracked)
    partial = False
    reason: str | None = None
    if len(selected) > limits["max_files"]:
        selected = selected[: limits["max_files"]]
        partial = True
        reason = "file_budget"

    documents: dict[str, _Document] = {}
    indexed_bytes = 0

    for path in selected:
        if time.monotonic() > deadline:
            partial = True
            reason = reason or "time_budget"
            break
        worktree_state = _states(path, state)
        clean = worktree_state == ("committed",)
        oid = state["oids"].get(path) if clean else None
        source_key = f"blob:{oid}" if oid else ""
        cached = previous.documents.get(path)
        if cached is not None and source_key and cached.source_key == source_key:
            document = cached
        else:
            read = _safe_read(root, path, limits["max_file_bytes"])
            if read is None:
                continue
            text, content_hash = read
            source_key = source_key or (
                "worktree:" + content_hash + ":" + ",".join(worktree_state)
            )
            document = _Document(
                path=path,
                text=text,
                content_sha256=content_hash,
                worktree_state=worktree_state,
                source_key=source_key,
                size=len(text.encode("utf-8")),
            )
        size = document.size
        if indexed_bytes + size > limits["max_index_bytes"]:
            partial = True
            reason = reason or "index_budget"
            break
        indexed_bytes += size
        documents[path] = document

    ordered = [documents[path] for path in sorted(documents)]
    fingerprint = _corpus_fingerprint(ordered)
    stats_by_key: dict[tuple[str, str], _TermStats] = {}
    stats: list[_TermStats] = []
    for document in ordered:
        stats_key = (document.path, document.content_sha256)
        document_stats = previous.stats.get(stats_key)
        if document_stats is None and time.monotonic() <= deadline:
            document_stats = _term_stats(document)
        if document_stats is not None:
            stats_by_key[stats_key] = document_stats
            stats.append(document_stats)
    corpus: _Corpus | None = None
    if len(stats) == len(ordered):
        if previous.corpus is not None and previous.corpus.fingerprint == fingerprint:
            corpus = previous.corpus
        else:
            corpus = _build_corpus(fingerprint, stats, deadline)
    if corpus is None:
        partial = True
        reason = reason or "time_budget"

    with _CACHE_LOCK:
        _CACHE[cache_key] = _Cache(
            documents=documents,
            stats=stats_by_key,
            index_state=index_state,
            selected_tracked=selected_tracked,
            corpus=corpus,
        )
    manifest_material = "\n".join(
        f"{doc.path}\0{','.join(doc.worktree_state)}\0{doc.content_sha256}"
        for doc in ordered
    ).encode()
    metadata = {
        "id": str(workspace["id"]),
        "head": state["head"],
        "branch": state["branch"] or None,
        "detached": state["detached"],
        "dirty": bool(state["untracked"] or state["staged"] or state["modified"]),
        "manifest_sha256": hashlib.sha256(manifest_material).hexdigest(),
    }
    ranking = (corpus, stats) if corpus is not None else None
    return ordered, metadata, partial, reason, ranking


def _score_document(
    corpus: _Corpus, stats: _TermStats, q_terms: Sequence[str]
) -> float:
    """``retrieve.bm25.BM25.score`` for one document, operation for operation."""
    freqs = stats.freqs
    dl = stats.length
    denom_norm = K1 * (1.0 - B + B * (dl / corpus.avgdl)) if corpus.avgdl else K1
    score = 0.0
    for term in q_terms:
        f = freqs.get(term, 0)
        if f == 0:
            continue
        idf = corpus.idf.get(term, 0.0)
        score += idf * (f * (K1 + 1.0)) / (f + denom_norm)
    return score


def _rank(
    query: str,
    corpus: _Corpus,
    stats: Sequence[_TermStats],
    deadline: float,
) -> tuple[list[tuple[int, float]], bool]:
    """``retrieve.bm25.rank`` over a cached corpus, stopping at the deadline.

    Returns (index, score) pairs in rank order and whether every document was
    scored; on a deadline, the documents scored so far are ranked.
    """
    q_terms = tokenize(query)
    if not q_terms or not stats:
        return [], True
    scored: list[tuple[int, float]] = []
    complete = True
    for position, document_stats in enumerate(stats):
        if time.monotonic() > deadline:
            complete = False
            break
        scored.append((position, _score_document(corpus, document_stats, q_terms)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored, complete


def _redact(text: str) -> str:
    if _PRIVATE_KEY_MARKER.search(text):
        return "[redacted secret-like content]"
    text = _SECRET_ASSIGNMENT.sub(r'\1"<redacted>"', text)
    text = _AUTHORIZATION_VALUE.sub(r"\1<redacted>", text)
    text = _URI_CREDENTIAL.sub(r"\1<redacted>\2", text)
    return _KNOWN_TOKEN.sub("<redacted>", text)


def _snippet(text: str, query: str, cap: int) -> tuple[str, int, int]:
    lines = text.splitlines() or [""]
    query_lower = query.lower().strip()
    best_index = 0
    if query_lower:
        exact = next(
            (i for i, line in enumerate(lines) if query_lower in line.lower()),
            None,
        )
        if exact is not None:
            best_index = exact
        else:
            q_terms = set(tokenize(query))
            best_index = max(
                range(len(lines)),
                key=lambda i: (
                    len(q_terms.intersection(tokenize(lines[i]))),
                    -i,
                ),
            )
    start = max(0, best_index - 1)
    end = min(len(lines), best_index + 2)
    rendered = _redact("\n".join(lines[start:end]))
    if len(rendered) > cap:
        rendered = rendered[:cap]
    return rendered, start + 1, end


def _limits(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "max_results": _bounded_int(
            settings.get("max_results"), 5, 1, _HARD_MAX_RESULTS
        ),
        "snippet_chars": _bounded_int(
            settings.get("snippet_chars"), 700, 80, _HARD_MAX_SNIPPET_CHARS
        ),
        "max_files": _bounded_int(settings.get("max_files"), 6_000, 1, _HARD_MAX_FILES),
        "max_file_bytes": _bounded_int(
            settings.get("max_file_bytes"), 262_144, 64, _HARD_MAX_FILE_BYTES
        ),
        "max_index_bytes": _bounded_int(
            settings.get("max_index_bytes"), 134_217_728, 1_024, _HARD_MAX_INDEX_BYTES
        ),
        "max_seconds": _bounded_float(
            settings.get("max_seconds"), 10.0, 0.1, _HARD_MAX_SECONDS
        ),
    }


def search_workspaces(
    query: str,
    *,
    remembrancer_root: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Search explicitly configured, already ACL-filtered workspaces.

    Returns only bounded unverified local evidence. Absolute paths and raw
    exception text never appear in the response.
    """
    if not isinstance(settings, Mapping) or settings.get("enabled") is not True:
        return _empty("disabled")
    workspaces = settings.get("workspaces")
    if not isinstance(workspaces, list) or not workspaces:
        return _empty("disabled")
    limits = _limits(settings)
    deadline = time.monotonic() + limits["max_seconds"]
    candidates: list[tuple[float, str, _Document]] = []
    workspace_meta: list[dict[str, Any]] = []
    partial = False
    reason: str | None = None
    valid = 0

    for workspace in workspaces:
        if not isinstance(workspace, Mapping):
            continue
        workspace_id = workspace.get("id")
        if (
            not isinstance(workspace_id, str)
            or _WORKSPACE_ID.fullmatch(workspace_id) is None
        ):
            continue
        try:
            root = _same_repository_root(
                Path(remembrancer_root),
                workspace.get("root"),
                deadline=deadline,
            )
            documents, metadata, was_partial, partial_reason, ranking = (
                _refresh_workspace(
                    root=root,
                    workspace=workspace,
                    limits=limits,
                    deadline=deadline,
                )
            )
        except UnsafeRootError:
            reason = reason or "unsafe_root"
            continue
        except GitWorkspaceError as exc:
            code = (
                str(exc)
                if str(exc)
                in {
                    "git_error",
                    "git_encoding",
                    "git_head",
                    "time_budget",
                }
                else "git_error"
            )
            reason = reason or code
            partial = partial or code == "time_budget"
            continue
        valid += 1
        workspace_meta.append(metadata)
        partial = partial or was_partial
        if partial_reason:
            reason = reason or partial_reason
        if ranking is None:
            continue
        corpus, stats = ranking
        scored, complete = _rank(query, corpus, stats, deadline)
        if not complete:
            partial = True
            reason = reason or "time_budget"
        for index, score in scored:
            if score <= 0:
                continue
            candidates.append((round(float(score), 6), workspace_id, documents[index]))

    if valid == 0:
        return _empty(reason or "disabled")
    # Same key and the same stable pre-order as sorting finished rows, so the
    # page is identical; snippets are then cut for the page only. A call whose
    # ranking stopped at the deadline still gets snippets for the rows it
    # ranked: the page (<= _HARD_MAX_RESULTS rows) may run _SNIPPET_GRACE_SECONDS
    # past the deadline, and rows past that are dropped and flagged partial.
    candidates.sort(key=lambda row: (-row[0], row[1], row[2].path))
    total = len(candidates)
    page: list[dict[str, Any]] = []
    snippet_deadline = deadline + _SNIPPET_GRACE_SECONDS
    for score, workspace_id, document in candidates[: limits["max_results"]]:
        if time.monotonic() > snippet_deadline:
            partial = True
            reason = reason or "time_budget"
            break
        snippet, line_start, line_end = _snippet(
            document.text,
            query,
            limits["snippet_chars"],
        )
        page.append(
            {
                "workspace": workspace_id,
                "path": document.path,
                "line_start": line_start,
                "line_end": line_end,
                "snippet": snippet,
                "score": score,
                "content_sha256": document.content_sha256,
                "worktree_state": list(document.worktree_state),
                "trust": "unverified-local-worktree",
            }
        )
    result: dict[str, Any] = {
        "searched": True,
        "results": page,
        "total_count": total,
        "returned": len(page),
        "partial": partial,
        "workspaces": workspace_meta,
    }
    if reason:
        result["reason"] = reason
    return result
