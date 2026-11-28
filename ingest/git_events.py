#!/usr/bin/env python3
"""Remembrancer connector: git history -> Layer-1 events (stdlib only, no external deps).

Refactored onto the shared `Connector` 4-stage contract (fetch/filter/normalize +
watermark cursor) while keeping the original push-time CLI working:

    # push-time Action / local one-shot (one or more repos, optional --since):
    python3 ingest/git_events.py --repo /path/to/repo [--repo ...] --since 2026-06-01

    # config-driven .run() path (reads config.json `monorepos`):
    python3 ingest/git_events.py --config config.json

Config `monorepos` entries need NO local `path`: when an entry only names a GitHub
repo ("acme/product"), the connector maintains a bare mirror clone under
`state/git_mirrors/` (created with `git clone --mirror`, refreshed with
`git remote update --prune` on every run). Auth is the `gh` credential helper,
passed explicitly per invocation, so a host whose ambient git config holds no
github.com credential still mirrors private repos -- without it every private
mirror fails with "could not read Username" while `gh api orgs/<org>/repos` works.
An explicit `path` (local worktree) still wins; `url` overrides the
GitHub-derived clone URL; `since` may be set per-entry or globally.

Mirror refreshes cross the network, and the network fails transiently: in the
three days to 2026-11-26 seven DIFFERENT repositories each timed out once at
120 s (intermittent TLS stalls to github.com from the pipeline host), and each single
timeout made the whole git source report an error.  So a transient failure
(timeout, connection reset, TLS or 5xx stderr) is retried with backoff, at most
`MIRROR_ATTEMPTS` times per repository and `MIRROR_RETRY_BUDGET` retries per run
(a GitHub outage must not cost every repository three timeouts), and when the
first `MIRROR_CIRCUIT_BREAKER_FAILURES` refreshes all miss with none succeeding
the rest of the run skips mirror refreshes and reports a source error (a
GitHub outage must not cost hundreds of repositories one timeout each either).
Rate limiting (429, GitHub's rate-limit messages) counts as transient; a
rejected certificate or missing credentials never does.  A repository
that is still unavailable after its retries is recorded in
`<mirrors_dir>/.refresh-failures.json` and tolerated as a per-repository miss;
it becomes a SOURCE error only when it is persistent (the same repository fails
`PERSISTENT_FAILURE_RUNS` runs in a row), systemic (at least half of this run's
mirror refreshes failed), or not transient at all (authentication, a missing
repository), or when the failure record cannot be written.  Entries for
repositories no longer configured are pruned from the record, but only after a
run that listed and tried every configured repository.

git is the *quiet* source: one non-merge commit = one event. The filter() valve
drops merge commits (already excluded at fetch) and anything missing a sha. The
event id is the FULL 40-char sha (`git:<sha>`) — short prefixes collide at org
scale — so re-runs are idempotent (`append_events` dedups on `id`) and the same
commit reached via two repos (git-subtree monorepo imports preserve shas) is
stored exactly once. Author is captured as "Name <email>" so downstream identity
matching (people_registry.py) can attribute by email, not display name.

Sensitive-content ACL (`remembrancerlib.restrict_sensitive_acl`, which relabels salary /
visa / termination ... text as `secrets` in the Gmail, Calendar, Drive, upload and
Slack connectors) is OFF for git by default.  Turn it on with
`{"git": {"sensitive_acl": true}}` in config.json once a reviewer has labelled the
existing regex hits and at least 20% are true positives: "termination" or "visa" in
a commit is usually a process or a file path, not HR content.  When on,
it checks the same text distill reads (`event_text`: subject + changed paths).  The
event id is the commit sha, so turning it on affects commits ingested afterwards
only; commits already in Layer 1 keep their label.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

# Make `remembrancerlib` / `ingest` importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remembrancerlib import Event, event_text, iter_events, restrict_sensitive_acl  # noqa: E402
from ingest.base import Connector  # noqa: E402

# Unit separator (0x1f) between fields, record separator (0x1e) between commits.
FMT = "%H%x1f%aI%x1f%an%x1f%ae%x1f%s"
SEP = "\x1f"
REC = "\x1e"
DEFAULT_MIRRORS_DIR = os.path.join("state", "git_mirrors")
DEFAULT_COMMAND_TIMEOUT_SECONDS = 120
# Transient mirror failures: tries per repository, backoff before retry n
# (BACKOFF * 2**(n-1) seconds), retries per connector run, and how many
# consecutive runs one repository may miss before it is a source error.
MIRROR_ATTEMPTS = 3
MIRROR_RETRY_BACKOFF_SECONDS = 5.0
MIRROR_RETRY_BUDGET = 4  # worst case: 4 extra timeouts (~9 min at 120 s) per run
PERSISTENT_FAILURE_RUNS = 3
REFRESH_FAILURES_FILE = ".refresh-failures.json"
# After this many transient mirror misses with not one refresh succeeding,
# GitHub or the network is down: stop refreshing for the rest of the run
# instead of spending a full timeout on each of hundreds of repositories.
MIRROR_CIRCUIT_BREAKER_FAILURES = 5
# stderr that names a network condition rather than a property of the
# repository (auth, missing repo, corrupt mirror), so a retry can succeed.
# TLS words count only beside a transport failure: a bare "ssl" would also
# match "SSL certificate problem", which no retry fixes.  Rate limiting
# (429, GitHub's primary and secondary limits) passes with time.
_TRANSIENT_STDERR = re.compile(
    r"could not resolve host|connection (timed out|reset|refused)|"
    r"failed to connect|operation timed out|timed out|early eof|rpc failed|"
    r"remote end hung up|unexpected disconnect|empty reply|"
    r"gnutls recv error|gnutls_handshake|ssl_read|ssl_write|ssl_connect|"
    r"ssl_error_syscall|\b(ssl|tls)\b[^\n]*\b(connection|handshake)\b|"
    r"returned error: (5\d\d|429)|http (5\d\d|429)|internal server error|"
    r"bad gateway|service unavailable|temporarily unavailable|"
    r"too many requests|secondary rate limit|api rate limit exceeded",
    re.IGNORECASE,
)
# stderr that is permanent whatever else it says: a certificate the client
# rejects, or credentials it lacks.  Checked first, so a line that also says
# "connection" is not retried.
_PERMANENT_STDERR = re.compile(
    r"certificate problem|certificate verify failed|"
    r"certificate verification failed|could not read username|"
    r"authentication failed|\bnot found\b",
    re.IGNORECASE,
)


class TransientGitError(RuntimeError):
    """A network-bound git or gh failure (mirror refresh or clone, org
    listing) that a later attempt may not repeat."""


class RetryBudget:
    """Retries left for this run, shared by every repository's mirror."""

    def __init__(self, remaining=MIRROR_RETRY_BUDGET):
        self.remaining = max(0, int(remaining))

    def take(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


def _command_timeout(value) -> float:
    """Return a bounded per-repository git timeout from untrusted config."""
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        timeout = DEFAULT_COMMAND_TIMEOUT_SECONDS
    return min(600.0, max(10.0, timeout))


class GitConnector(Connector):
    """Pull commit history from one or more repos into L1 events.

    Config shape (any combination is accepted):
      - {"repos": [{"path": "...", "name": "acme/product", "domain": "product"}]}
      - {"monorepos": [{"repo": "acme/product", "domain": "product",
                        "path": "...", "url": "...", "since": "2026-01-01"}]}
      - {"git_org": {"org": "acme", "since": null, "exclude": ["big-noise-repo"],
                     "include_archived": true}}
    `git_org` enumerates EVERY repo in the org (via `gh api`, forks always skipped;
    archived repos included by default — departed contributors' history lives
    there) and ingests each one that isn't already covered by an explicit entry.
    Explicit monorepos entries win, so their `domain` hints are preserved.
    `path` / `url` / per-entry `since` are all optional for monorepos entries —
    a bare mirror of github.com/<repo> is maintained automatically when absent.
    `since` / `acl` / `mirrors_dir` may be set on the config too.
    """

    name = "git"
    cursor_key = "git"

    def __init__(self, config, **kw):
        super().__init__(config, **kw)
        # accept either "repos" or config.json's "monorepos"
        repos = config.get("repos")
        if repos is None:
            repos = []
            for m in config.get("monorepos", []):
                repos.append({
                    "path": m.get("path"),
                    "name": m.get("repo") or m.get("path"),
                    "url": m.get("url"),
                    "domain": m.get("domain"),
                    "since": m.get("since"),
                })
        self.repos = repos
        self.since = config.get("since")
        self.acl_override = config.get("acl")  # optional explicit acl list
        # Off until a reviewer has checked the existing hits (module docstring).
        git_cfg = config.get("git")
        self.sensitive_acl = (
            isinstance(git_cfg, dict) and git_cfg.get("sensitive_acl") is True
        )
        # Isolated nightly checkouts are disposable, while the organization's
        # bare mirrors are a large derived cache.  The outer runner may point
        # this at a persistent, lock-protected cache without making the cache
        # part of durable Memory state.
        self.mirrors_dir = (
            os.environ.get("REMEMBRANCER_GIT_MIRRORS_DIR")
            or config.get("mirrors_dir")
            or DEFAULT_MIRRORS_DIR
        )
        self.had_errors = False
        # Mirror outcomes of this run: names refreshed, and transient misses
        # (name -> message) that are judged together once fetch() has finished.
        self.retry_budget = RetryBudget()
        self._sleep = time.sleep  # tests replace it to skip the backoff wait
        self._mirror_ok: list[str] = []
        self._mirror_transient: dict[str, str] = {}
        # Every mirror name this run tried (any outcome), whether the entry
        # list was complete (the org listing succeeded), and whether the
        # circuit breaker skipped the rest: the record is pruned to the names
        # tried only when the run saw every configured repository.
        self._mirror_tried: set[str] = set()
        self._entries_complete = True
        self._breaker_open = False
        org_cfg = config.get("git_org") or {}
        self.command_timeout_seconds = _command_timeout(
            os.environ.get("REMEMBRANCER_GIT_COMMAND_TIMEOUT_SECONDS")
            or org_cfg.get("command_timeout_seconds")
            or config.get("command_timeout_seconds")
        )
        self.org = org_cfg.get("org")
        self.org_exclude = set(org_cfg.get("exclude") or [])
        self.org_since = org_cfg.get("since")
        self.org_include_archived = org_cfg.get("include_archived", True)
        # url_base override lets tests (and GH Enterprise) point at another root.
        self.org_url_base = org_cfg.get("url_base") or "https://github.com"

    def _org_entries(self):
        """Repo entries for every org repo not already explicitly configured.
        Forks are always skipped (their history isn't the org's work)."""
        explicit = {r.get("name") for r in self.repos}
        entries = []
        try:
            org_repos = _with_retries(
                lambda: _org_repos(self.org, timeout=self.command_timeout_seconds),
                budget=self.retry_budget,
                sleep=self._sleep,
            )
        except RuntimeError as exc:
            self.had_errors = True
            self._entries_complete = False
            print(f"[git] {exc}", file=sys.stderr)
            return entries
        for r in org_repos:
            full = f"{self.org}/{r.get('name')}"
            if r.get("fork") or full in explicit or r.get("name") in self.org_exclude:
                continue
            if r.get("archived") and not self.org_include_archived:
                continue
            entries.append({
                "name": full,
                "url": f"{self.org_url_base}/{full}.git",
                "domain": None,
                "since": self.org_since,
            })
        return entries

    # --- stage 1: fetch ----------------------------------------------------
    def fetch(self, cursor):
        """Yield raw commit tuples from each configured repo. `cursor` is unused
        for the windowing (we rely on --since + id-dedup), but kept for contract.

        Existing L1 ids are the durable authority for commits observed by an
        earlier run.  Repository ordering can change as explicit entries are
        added or the organization listing evolves; re-selecting provenance for
        an existing ``git:<sha>`` would otherwise conflict with that immutable
        first observation.  New shas are still de-duplicated across this run,
        with explicit repositories winning as before.

        Explicit repositories are traversed first and therefore own provenance
        when the same immutable commit is reachable from multiple repositories
        (for example after a subtree import).  De-duplicating here is required:
        append_events correctly rejects one git:<sha> id with two different
        provenance.repo fingerprints.
        """
        entries = list(self.repos)
        if self.org:
            entries.extend(self._org_entries())
        seen_shas: set[str] = {
            event.id.split(":", 1)[1]
            for event in iter_events(self.events_dir)
            if event.source == "git" and event.id.startswith("git:")
        }
        for r in entries:
            path = r.get("path")
            if not path or not _is_git_repo(path):
                # No usable local worktree: fall back to a maintained bare mirror
                # when the entry names a remote ("org/repo" or explicit url).
                name = r.get("name") or ""
                url = r.get("url") or (f"https://github.com/{name}.git" if "/" in name else None)
                if url:
                    if self._breaker_open:
                        continue  # counted once, in _trip_breaker
                    self._mirror_tried.add(name or url)
                    try:
                        path = _with_retries(
                            lambda: _ensure_mirror(
                                name,
                                url,
                                self.mirrors_dir,
                                timeout=self.command_timeout_seconds,
                            ),
                            budget=self.retry_budget,
                            sleep=self._sleep,
                        )
                        self._mirror_ok.append(name or url)
                    except TransientGitError as exc:
                        # Judged after the loop: one miss is tolerated, a
                        # persistent or systemic one is a source error.
                        self._mirror_transient[name or url] = str(exc)
                        if (not self._mirror_ok and len(self._mirror_transient)
                                >= MIRROR_CIRCUIT_BREAKER_FAILURES):
                            self._trip_breaker()
                        continue
                    except (OSError, RuntimeError) as exc:
                        self.had_errors = True
                        print(f"[git] mirror unavailable for {name or url}: {exc}",
                              file=sys.stderr)
                        continue
                else:
                    path = None
            if not path or not _is_git_repo(path):
                self.had_errors = True
                print(f"[git] skip non-repo path: {r.get('path') or r.get('name')}",
                      file=sys.stderr)
                continue
            repo_name = r.get("name") or os.path.basename(os.path.abspath(path.rstrip("/")))
            domain = r.get("domain")
            since = r.get("since") or self.since
            try:
                for h, ts, author, subject, files in _commits(
                    path, since, timeout=self.command_timeout_seconds,
                ):
                    if h in seen_shas:
                        continue
                    seen_shas.add(h)
                    yield {
                        "sha": h, "ts": ts, "author": author, "subject": subject,
                        "files": files, "repo": repo_name, "domain": domain,
                    }
            except RuntimeError as exc:
                self.had_errors = True
                print(f"[git] {repo_name}: {exc}", file=sys.stderr)
        self._judge_mirror_failures()

    def _trip_breaker(self):
        """Stop refreshing mirrors for the rest of the run: the first
        MIRROR_CIRCUIT_BREAKER_FAILURES refreshes all missed and none worked,
        so the rest would each cost a full timeout for the same answer."""
        self._breaker_open = True
        self.had_errors = True
        print(f"[git] {len(self._mirror_transient)} mirror refreshes failed "
              "and none succeeded; skipping the remaining mirror refreshes "
              "this run [source error: GitHub or the network is down]",
              file=sys.stderr)

    def _judge_mirror_failures(self):
        """Decide whether this run's transient mirror misses are a source error.

        One repository missing one run is coverage that the next run restores
        (ids are full shas and `--since` re-reads the window), so it is logged,
        not raised.  It is a source error when the same repository has now
        missed PERSISTENT_FAILURE_RUNS runs in a row, when at least half of the
        run's mirror refreshes missed (GitHub or the network is down, not one
        repository), or when the streak record cannot be kept, because then
        persistence cannot be proven either way.
        """
        failed = self._mirror_transient
        if not failed and not self._mirror_ok:
            return  # no mirror-backed repository this run: nothing to record
        path = os.path.join(self.mirrors_dir, REFRESH_FAILURES_FILE)
        record = _load_failure_record(path)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for name in self._mirror_ok:
            record.pop(name, None)
        if self._entries_complete and not self._breaker_open:
            # A repository removed from the org or excluded since its last
            # miss would otherwise keep its streak in the file forever.  Not
            # when the run saw only part of the list: a failed org listing or
            # a tripped breaker would erase streaks it never re-tested.
            for name in [k for k in record if k not in self._mirror_tried]:
                del record[name]
        for name, message in failed.items():
            prior = record.get(name) or {}
            try:
                runs = int(prior.get("consecutive_runs", 0)) + 1
            except (TypeError, ValueError):
                runs = 1
            record[name] = {
                "consecutive_runs": runs,
                "first_failed_at": prior.get("first_failed_at") or now,
                "last_failed_at": now,
                "last_error": message[:500],
            }
        try:
            _write_failure_record(path, record)
        except OSError as exc:
            if failed:
                self.had_errors = True
                print(f"[git] cannot record mirror failures in {path}: {exc}",
                      file=sys.stderr)
        attempted = len(failed) + len(self._mirror_ok)
        # With one mirror-backed repository, one miss is 1 of 1 and therefore
        # systemic: a single-repository config has no healthy peers to show
        # the network is up, so it keeps the old behaviour (a source error).
        systemic = bool(failed) and 2 * len(failed) >= attempted
        for name in sorted(failed):
            runs = record[name]["consecutive_runs"]
            persistent = runs >= PERSISTENT_FAILURE_RUNS
            if persistent or systemic:
                self.had_errors = True
                why = (f"{runs} consecutive runs" if persistent else
                       f"{len(failed)} of {attempted} mirrors failed this run")
                print(f"[git] mirror unavailable for {name}: {failed[name]} "
                      f"[source error: {why}]", file=sys.stderr)
            else:
                print(f"[git] mirror unavailable for {name}: {failed[name]} "
                      f"[tolerated: run {runs} of {PERSISTENT_FAILURE_RUNS} "
                      f"before it is a source error]", file=sys.stderr)

    # --- stage 2: filter (anti-blowup valve) -------------------------------
    def filter(self, raw):
        # git is quiet; merges already excluded at fetch. Keep any real commit.
        return bool(raw.get("sha"))

    # --- stage 3: normalize ------------------------------------------------
    def normalize(self, raw):
        repo = raw["repo"]
        sha = raw["sha"]
        # governance key: repo (or repo/top-folder) drives acl labeling.
        domain, acl = self.labeler.label(self.name, key=repo, domain_hint=raw.get("domain"))
        if self.acl_override:
            acl = list(self.acl_override)
        payload = {"subject": raw.get("subject", ""), "files": raw.get("files", [])[:50]}
        if self.sensitive_acl:
            acl = restrict_sensitive_acl(event_text({"payload": payload}), acl)
        return Event(
            # FULL sha: 7-char prefixes collide at org scale, and the full sha is
            # what makes subtree-duplicated commits (same sha, two repos) dedup.
            id=f"git:{sha}",
            source="git",
            author=raw.get("author", ""),
            occurred_at=raw.get("ts"),
            acl=acl,
            entities=[],  # distiller resolves entities
            provenance={"repo": repo, "ref": sha, "url": None},
            payload=payload,
        )


def _load_failure_record(path):
    """The per-repository miss streaks; an absent or unreadable file is empty
    (a lost record only delays escalation by a run or two, it never hides a
    systemic outage, which is judged within the run)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _write_failure_record(path, record):
    """Atomic write next to the mirrors (the cache the runner lock protects)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    # A crash between open and replace leaves `<file>.tmp.<pid>`; sweep those
    # (the runner lock means no other writer is mid-write).
    prefix = os.path.basename(path) + ".tmp."
    for leftover in os.listdir(directory):
        if leftover.startswith(prefix):
            try:
                os.unlink(os.path.join(directory, leftover))
            except OSError:
                pass
    try:
        with open(tmp, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except OSError:
        if os.path.lexists(tmp):
            os.unlink(tmp)
        raise


def _org_repos(org, *, timeout=DEFAULT_COMMAND_TIMEOUT_SECONDS):
    """All repos in a GitHub org via `gh api` (name/fork/archived per line).
    A failed listing is source-health failure, not an empty organization.  The
    caller still processes explicit repositories but records a non-zero CLI
    result so nightly cannot mistake missing coverage for source silence."""
    timeout = _command_timeout(timeout)
    args = [
        "gh", "api", f"orgs/{org}/repos?per_page=100&type=all", "--paginate",
        "--jq", ".[] | {name, fork, archived}",
    ]
    try:
        res = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TransientGitError(
            f"org listing timed out after {timeout:g}s for {org}"
        ) from exc
    if res.returncode != 0:
        raise _mirror_failure(
            f"org listing failed for {org}: {res.stderr.strip()[:200]}",
            res.stderr,
        )
    out = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"org listing returned invalid JSON for {org}"
                ) from exc
    return out


def _is_git_repo(path):
    """True for a worktree or bare repo AT `path` -- never one above it.

    `git rev-parse` searches UPWARD by design, so any directory that merely sits
    inside a checkout answers "true" for its parent's repository. That is not
    hypothetical: a bulk copy that leaves mirror directories without `refs/`
    makes every one of them resolve to an enclosing checkout's `.git`. The
    connector then reads that checkout's history and files it under the other
    repository's name, with provenance that was never true.

    Comparing the resolved git dir against `path` is what makes the answer mean
    "a repository is here" rather than "a repository is somewhere above here".
    """
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "--absolute-git-dir"],
            capture_output=True, text=True,
        )
    except Exception:
        return False
    if out.returncode != 0:
        return False
    resolved = os.path.realpath(out.stdout.strip())
    here = os.path.realpath(path)
    # Bare mirror: the git dir IS the path. Worktree: it is <path>/.git.
    return resolved in (here, os.path.join(here, ".git"))


# A private repo's mirror needs credentials. The connector
# already shells out to `gh` for the org listing, so the helper adds no new
# dependency and no new secret, and it does not depend on whatever the host's
# ambient git config happens to hold.
_GIT_CREDENTIAL_ARGS = ("-c", "credential.helper=!gh auth git-credential")


def _mirror_failure(message, stderr=""):
    """The error for a failed mirror command: transient when stderr names a
    network condition, so the caller may retry it."""
    stderr = stderr or ""
    transient = (_TRANSIENT_STDERR.search(stderr)
                 and not _PERMANENT_STDERR.search(stderr))
    return (TransientGitError if transient else RuntimeError)(message)


def _with_retries(call, *, attempts=MIRROR_ATTEMPTS, budget=None, sleep=None):
    """Run `call()`, retrying a TransientGitError with doubling backoff.

    At most `attempts` tries in all, and only while `budget` (a RetryBudget
    shared across the run; None means unlimited) has retries left.  The last
    error is re-raised saying which limit stopped it: the attempt cap or the
    run's retry budget.  Any other
    exception propagates at once: retrying a bad credential or a missing
    repository only spends another timeout.
    """
    sleep = time.sleep if sleep is None else sleep
    attempts = max(1, int(attempts))
    attempt = 1
    while True:
        try:
            return call()
        except TransientGitError as exc:
            if attempt >= attempts:
                raise TransientGitError(
                    f"{exc} (attempt {attempt} of {attempts})"
                ) from exc
            if budget is not None and not budget.take():
                raise TransientGitError(
                    f"{exc} (attempt {attempt}; run retry budget exhausted)"
                ) from exc
            delay = MIRROR_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"[git] {exc}; retrying in {delay:g}s "
                  f"(attempt {attempt + 1} of {attempts})", file=sys.stderr)
            sleep(delay)
            attempt += 1


def _ensure_mirror(
    name, url, mirrors_dir, *, timeout=DEFAULT_COMMAND_TIMEOUT_SECONDS,
):
    """Clone-or-refresh a bare mirror, raising when freshness is unprovable.

    One try.  A network-bound failure raises TransientGitError, which the
    connector retries through `_with_retries` under its run budget."""
    timeout = _command_timeout(timeout)
    os.makedirs(mirrors_dir, exist_ok=True)
    base = (name or url).replace("/", "__")
    if base.endswith(".git"):
        base = base[:-4]
    dest = os.path.join(mirrors_dir, base + ".git")
    if os.path.isdir(dest) and not _is_git_repo(dest):
        # A directory that is not a repository must be replaced, not refreshed.
        # Refreshing it "succeeds" against whatever repository encloses it.
        if os.path.islink(dest):
            os.unlink(dest)
        else:
            shutil.rmtree(dest)
    if os.path.isdir(dest):
        try:
            res = subprocess.run(
                ["git", *_GIT_CREDENTIAL_ARGS, "-C", dest,
                 "remote", "update", "--prune"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TransientGitError(
                f"mirror refresh timed out after {timeout:g}s for {name or url}"
            ) from exc
        if res.returncode != 0:
            raise _mirror_failure(
                f"mirror refresh failed for {name}: {res.stderr.strip()}",
                res.stderr,
            )
        return dest
    # Prefer a blobless mirror (log/--name-only never need blobs); some
    # transports/old gits reject --filter, so fall back to a plain mirror.
    for args in (["git", *_GIT_CREDENTIAL_ARGS, "clone", "--mirror",
                  "--filter=blob:none", url, dest],
                 ["git", *_GIT_CREDENTIAL_ARGS, "clone", "--mirror", url, dest]):
        try:
            res = subprocess.run(
                args, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            if os.path.lexists(dest):
                if os.path.isdir(dest) and not os.path.islink(dest):
                    shutil.rmtree(dest)
                else:
                    os.unlink(dest)
            raise TransientGitError(
                f"mirror clone timed out after {timeout:g}s for {name or url}"
            ) from exc
        if res.returncode == 0:
            return dest
        # A transport may reject the filtered clone after creating a partial
        # destination.  The unfiltered fallback must start clean.
        if os.path.lexists(dest):
            if os.path.isdir(dest) and not os.path.islink(dest):
                shutil.rmtree(dest)
            else:
                os.unlink(dest)
    raise _mirror_failure(
        f"mirror clone failed for {name or url}: {res.stderr.strip()}",
        res.stderr,
    )


def _commits(repo, since, *, timeout=DEFAULT_COMMAND_TIMEOUT_SECONDS):
    """Yield branch-reachable non-merge commits with author and touched files.

    Single `git log --name-only` pass — one subprocess per repo, not per commit
    (a monorepo backfill is tens of thousands of commits). `author` is rendered
    as "Name <email>" so identity matching can key on the email. Both local
    branch refs (bare mirrors store GitHub branches there) and remote-tracking
    branch refs (ordinary worktrees) are traversed. Tags, stash, notes, and
    provider-specific pull-request refs are deliberately excluded.
    """
    # A freshly-created bare mirror is a valid git repository but may not have
    # a HEAD yet.  That is an empty source, not an operational error; avoid the
    # noisy "ambiguous argument HEAD" warning and simply yield nothing.
    timeout = _command_timeout(timeout)
    try:
        head = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--verify", "HEAD"],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git HEAD check timed out after {timeout:g}s for {repo}"
        ) from exc
    if head.returncode != 0:
        return

    args = ["git", "-C", repo, "log", "--branches", "--remotes",
            "--no-merges", "--name-only",
            f"--pretty=format:{REC}{FMT}"]
    if since:
        args.append(f"--since={since}")
    # `git log --since` is a traversal CUTOFF, not a filter: it always emits the
    # tip commit even when the tip is older than the bound. Re-check ISO-dated
    # bounds in Python so a quiet repo doesn't leak stale tip commits.
    since_day = since[:10] if since and re.match(r"^\d{4}-\d{2}-\d{2}", since) else None
    try:
        res = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git log timed out after {timeout:g}s for {repo}"
        ) from exc
    if res.returncode != 0:
        raise RuntimeError(f"git log failed: {res.stderr.strip()}")
    for record in res.stdout.split(REC):
        if not record.strip():
            continue
        head, _, tail = record.partition("\n")
        parts = head.split(SEP)
        if len(parts) != 5:
            continue
        h, ts, an, ae, subject = parts
        if since_day and ts and ts[:10] < since_day:
            continue
        author = f"{an} <{ae}>" if ae else an
        files = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        yield h, ts, author, subject, files


# ---------------------------------------------------------------------------
# CLI: keep the original `--repo ... --since ...` push-time path, plus `.run()`.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="git -> Remembrancer L1 events")
    ap.add_argument("--repo", action="append", help="repo path (repeatable)")
    ap.add_argument("--since", default=None)
    ap.add_argument("--acl", default=None, help="comma-separated acl labels (override governance)")
    ap.add_argument("--out-dir", default="events", help="event store dir (default: events)")
    ap.add_argument("--config", default="config.json", help="config.json for the .run() path")
    args = ap.parse_args()

    if args.repo:
        # Explicit-repo path (push-time Action / local one-shot).
        config = {
            "repos": [{"path": p} for p in args.repo],
            "since": args.since,
        }
        if args.acl:
            config["acl"] = [a.strip() for a in args.acl.split(",") if a.strip()]
    else:
        # Config-driven path: read monorepos from config.json.
        config = {}
        if os.path.exists(args.config):
            import json
            with open(args.config) as f:
                config = json.load(f)
        if args.since:
            config["since"] = args.since
        if args.acl:
            config["acl"] = [a.strip() for a in args.acl.split(",") if a.strip()]
        if (not config.get("repos") and not config.get("monorepos")
                and not (config.get("git_org") or {}).get("org")):
            print("[git] no --repo, monorepos, or git_org in config; nothing to do.",
                  file=sys.stderr)
            return 0

    conn = GitConnector(config, events_dir=args.out_dir)
    conn.run()
    return 1 if conn.had_errors else 0


if __name__ == "__main__":
    sys.exit(main())
