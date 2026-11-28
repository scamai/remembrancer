"""Regression coverage for Git mirror retries and source health."""
import json
import os
import subprocess
import sys

import pytest

import ingest.git_events as ge
from ingest.git_events import (
    MIRROR_RETRY_BACKOFF_SECONDS,
    PERSISTENT_FAILURE_RUNS,
    REFRESH_FAILURES_FILE,
    GitConnector,
    RetryBudget,
    TransientGitError,
)
from remembrancerlib import iter_events

_REAL_RUN = subprocess.run


def _git(cwd, *args):
    _REAL_RUN(["git", "-C", cwd, *args], check=True, capture_output=True)


def _make_repo(root, n_commits=2):
    os.makedirs(root, exist_ok=True)
    _REAL_RUN(["git", "init", "-q", root], check=True, capture_output=True)
    _git(root, "config", "user.name", "Test Author")
    _git(root, "config", "user.email", "author@example.invalid")
    for i in range(n_commits):
        with open(os.path.join(root, f"f{i}.txt"), "w") as f:
            f.write(f"{os.path.basename(root)} v{i}\n")
        _git(root, "add", ".")
        _git(root, "commit", "-q", "-m", f"feat: {os.path.basename(root)} {i}")
    return root


@pytest.fixture(scope="module")
def repos(tmp_path_factory):
    """Four real repositories, two commits each, keyed by "acme/<name>"."""
    root = tmp_path_factory.mktemp("origins")
    return {
        f"acme/r{i}": _make_repo(str(root / f"r{i}"))
        for i in range(4)
    }


def _connector(tmp_path, names):
    conn = GitConnector(
        {
            "monorepos": [
                {"repo": n, "url": f"https://example.invalid/{n}.git"}
                for n in names
            ],
            "mirrors_dir": str(tmp_path / "mirrors"),
        },
        events_dir=str(tmp_path / "events"),
    )
    conn.sleeps = []
    conn._sleep = conn.sleeps.append
    return conn


def _fake_mirrors(monkeypatch, repos, failing, calls=None, error=TransientGitError):
    """`_ensure_mirror` that serves `repos` and raises for names in `failing`."""
    def ensure(name, url, mirrors_dir, *, timeout):
        if calls is not None:
            calls.append(name)
        if name in failing:
            raise error(f"mirror refresh timed out after {timeout:g}s for {name}")
        return repos[name]
    monkeypatch.setattr(ge, "_ensure_mirror", ensure)


def _record(tmp_path):
    path = tmp_path / "mirrors" / REFRESH_FAILURES_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def _event_repos(tmp_path):
    return sorted({
        e.provenance["repo"] for e in iter_events(str(tmp_path / "events"))
    })


# ─── retry and backoff ──────────────────────────────────────────────────────

def test_a_refresh_that_times_out_once_is_retried_and_ingests(tmp_path, monkeypatch):
    """Real `_ensure_mirror` against a real mirror: the first `remote update`
    stalls, the retry succeeds, and the run is clean."""
    origin = _make_repo(str(tmp_path / "origin"))
    mirrors = str(tmp_path / "mirrors")
    ge._ensure_mirror("acme/widget", origin, mirrors)  # existing mirror
    stalls = []

    def run(args, **kwargs):
        if "remote" in args and "update" in args and not stalls:
            stalls.append(args)
            raise subprocess.TimeoutExpired(args, kwargs.get("timeout"))
        return _REAL_RUN(args, **kwargs)

    monkeypatch.setattr(ge.subprocess, "run", run)
    conn = GitConnector(
        {"monorepos": [{"repo": "acme/widget", "url": origin}],
         "mirrors_dir": mirrors},
        events_dir=str(tmp_path / "events"),
    )
    sleeps = []
    conn._sleep = sleeps.append

    assert conn.run() == 2
    assert conn.had_errors is False
    assert len(stalls) == 1
    assert sleeps == [MIRROR_RETRY_BACKOFF_SECONDS]
    assert _record(tmp_path) == {}


def test_backoff_doubles_and_gives_up_after_the_attempt_limit(tmp_path, monkeypatch, repos):
    calls = []
    _fake_mirrors(monkeypatch, repos, failing={"acme/r0"}, calls=calls)
    conn = _connector(tmp_path, ["acme/r0"])
    conn.run()
    assert calls == ["acme/r0"] * ge.MIRROR_ATTEMPTS
    assert conn.sleeps == [
        MIRROR_RETRY_BACKOFF_SECONDS * 2 ** i for i in range(ge.MIRROR_ATTEMPTS - 1)
    ]


def test_retry_budget_bounds_the_whole_run_during_an_outage(tmp_path, monkeypatch, repos):
    """GitHub down: every repository fails, but only `budget` retries are spent
    across the run, not MIRROR_ATTEMPTS per repository."""
    calls = []
    _fake_mirrors(monkeypatch, repos, failing=set(repos), calls=calls)
    conn = _connector(tmp_path, sorted(repos))
    conn.retry_budget = RetryBudget(2)
    conn.run()
    assert len(calls) == len(repos) + 2
    assert conn.had_errors is True  # 4 of 4 failed: systemic


def test_a_non_transient_failure_is_not_retried_and_is_a_source_error(
        tmp_path, monkeypatch, repos):
    calls = []
    _fake_mirrors(monkeypatch, repos, failing={"acme/r0"}, calls=calls,
                  error=RuntimeError)
    conn = _connector(tmp_path, sorted(repos))
    conn.run()
    assert calls.count("acme/r0") == 1
    assert conn.sleeps == []
    assert conn.had_errors is True
    assert _record(tmp_path) == {}


# ─── per-repository miss vs source error ────────────────────────────────────

def test_one_transient_miss_among_healthy_repos_is_tolerated(tmp_path, monkeypatch, repos):
    _fake_mirrors(monkeypatch, repos, failing={"acme/r0"})
    conn = _connector(tmp_path, sorted(repos))
    assert conn.run() == 6  # the three healthy repositories, two commits each
    assert conn.had_errors is False
    assert _event_repos(tmp_path) == ["acme/r1", "acme/r2", "acme/r3"]
    entry = _record(tmp_path)["acme/r0"]
    assert entry["consecutive_runs"] == 1
    assert "timed out" in entry["last_error"]


def test_cli_exits_zero_for_a_tolerated_miss(tmp_path, monkeypatch, repos):
    """The CLI reports a source error through its exit status."""
    _fake_mirrors(monkeypatch, repos, failing={"acme/r0"})
    monkeypatch.setattr(ge.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ge, "_org_repos", lambda _org, **_kw: [
        {"name": name.split("/", 1)[1]} for name in sorted(repos)
    ])
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "git_org": {"org": "acme"},
        "mirrors_dir": str(tmp_path / "mirrors"),
    }))
    monkeypatch.setattr(sys, "argv", [
        "git_events.py", "--config", str(config),
        "--out-dir", str(tmp_path / "events"),
    ])
    assert ge.main() == 0
    assert _event_repos(tmp_path) == ["acme/r1", "acme/r2", "acme/r3"]


def test_the_same_repository_missing_run_after_run_becomes_a_source_error(
        tmp_path, monkeypatch, repos):
    _fake_mirrors(monkeypatch, repos, failing={"acme/r0"})
    outcomes = []
    for _ in range(PERSISTENT_FAILURE_RUNS):
        conn = _connector(tmp_path, sorted(repos))
        conn.run()
        outcomes.append(conn.had_errors)
    assert outcomes == [False] * (PERSISTENT_FAILURE_RUNS - 1) + [True]
    assert _record(tmp_path)["acme/r0"]["consecutive_runs"] == PERSISTENT_FAILURE_RUNS


def test_a_successful_refresh_resets_the_streak(tmp_path, monkeypatch, repos):
    names = sorted(repos)
    for failing in ({"acme/r0"}, {"acme/r0"}, set(), {"acme/r0"}):
        _fake_mirrors(monkeypatch, repos, failing=failing)
        conn = _connector(tmp_path, names)
        conn.run()
    assert conn.had_errors is False
    assert _record(tmp_path)["acme/r0"]["consecutive_runs"] == 1


def test_half_of_the_mirrors_failing_is_a_source_error_at_once(tmp_path, monkeypatch, repos):
    _fake_mirrors(monkeypatch, repos, failing={"acme/r0", "acme/r1"})
    conn = _connector(tmp_path, sorted(repos))
    conn.run()
    assert conn.had_errors is True


def test_an_unwritable_failure_record_is_a_source_error(tmp_path, monkeypatch, repos):
    """Without the record a persistent miss could never escalate."""
    (tmp_path / "mirrors" / REFRESH_FAILURES_FILE).mkdir(parents=True)
    _fake_mirrors(monkeypatch, repos, failing={"acme/r0"})
    conn = _connector(tmp_path, sorted(repos))
    conn.run()
    assert conn.had_errors is True


def test_a_corrupt_failure_record_reads_as_empty(tmp_path, monkeypatch, repos):
    path = tmp_path / "mirrors" / REFRESH_FAILURES_FILE
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    _fake_mirrors(monkeypatch, repos, failing={"acme/r0"})
    conn = _connector(tmp_path, sorted(repos))
    conn.run()
    assert conn.had_errors is False
    assert _record(tmp_path)["acme/r0"]["consecutive_runs"] == 1


# ─── classification of real command failures ───────────────────────────────

def _existing_mirror(tmp_path, monkeypatch):
    (tmp_path / "mirrors" / "acme__widget.git").mkdir(parents=True)
    monkeypatch.setattr(ge, "_is_git_repo", lambda _path: True)


@pytest.mark.parametrize("stderr,transient", [
    ("fatal: unable to access 'https://github.com/acme/widget/': "
     "GnuTLS recv error (-110): The TLS connection was non-properly terminated.", True),
    ("fatal: unable to access 'https://github.com/acme/widget/': "
     "Could not resolve host: github.com", True),
    ("error: RPC failed; HTTP 502 curl 22 The requested URL returned error: 502", True),
    ("fatal: could not read Username for 'https://github.com': "
     "terminal prompts disabled", False),
    ("remote: Repository not found.\nfatal: repository "
     "'https://github.com/acme/widget/' not found", False),
    # TLS words are transient only beside a transport failure.
    ("fatal: unable to access 'https://github.com/acme/widget/': "
     "OpenSSL SSL_read: Connection reset by peer, errno 104", True),
    ("fatal: unable to access 'https://github.com/acme/widget/': "
     "SSL certificate problem: unable to get local issuer certificate", False),
    ("fatal: unable to access 'https://github.com/acme/widget/': "
     "server certificate verification failed. CAfile: none CRLfile: none", False),
    ("fatal: unable to access 'https://github.com/acme/widget/': "
     "SSL connection error: certificate verify failed", False),
    ("fatal: unable to access 'https://github.com/acme/widget/': "
     "SSL: no alternative certificate subject name matches target host name", False),
    # Rate limiting passes with time; a bare 403 is an access problem.
    ("fatal: unable to access 'https://github.com/acme/widget/': "
     "The requested URL returned error: 429", True),
    ("remote: You have exceeded a secondary rate limit.\n"
     "fatal: unable to access 'https://github.com/acme/widget/': "
     "The requested URL returned error: 403", True),
    ("fatal: unable to access 'https://github.com/acme/widget/': "
     "The requested URL returned error: 403", False),
])
def test_refresh_stderr_decides_whether_a_retry_can_help(
        tmp_path, monkeypatch, stderr, transient):
    _existing_mirror(tmp_path, monkeypatch)
    monkeypatch.setattr(
        ge.subprocess, "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 1, "", stderr),
    )
    with pytest.raises(RuntimeError) as info:
        ge._ensure_mirror("acme/widget", "https://example.invalid/w.git",
                          str(tmp_path / "mirrors"))
    assert isinstance(info.value, TransientGitError) is transient


def test_a_refresh_timeout_is_transient(tmp_path, monkeypatch):
    _existing_mirror(tmp_path, monkeypatch)

    def stall(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(ge.subprocess, "run", stall)
    with pytest.raises(TransientGitError, match="timed out after 120s"):
        ge._ensure_mirror("acme/widget", "https://example.invalid/w.git",
                          str(tmp_path / "mirrors"))


def test_a_clone_timeout_is_transient_and_leaves_no_partial_mirror(tmp_path, monkeypatch):
    mirrors = tmp_path / "mirrors"

    def stall(args, **kwargs):
        os.makedirs(args[-1], exist_ok=True)  # a partial clone destination
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(ge.subprocess, "run", stall)
    with pytest.raises(TransientGitError, match="mirror clone timed out"):
        ge._ensure_mirror("acme/widget", "https://example.invalid/w.git", str(mirrors))
    assert not (mirrors / "acme__widget.git").exists()


def test_an_org_listing_that_stalls_once_is_retried(tmp_path, monkeypatch):
    stalls = []

    def run(args, **kwargs):
        if not stalls:
            stalls.append(args)
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        return subprocess.CompletedProcess(
            args, 0, '{"name": "widget", "fork": false, "archived": false}\n', "")

    monkeypatch.setattr(ge.subprocess, "run", run)
    conn = GitConnector({"git_org": {"org": "acme"}},
                        events_dir=str(tmp_path / "events"))
    sleeps = []
    conn._sleep = sleeps.append
    assert [e["name"] for e in conn._org_entries()] == ["acme/widget"]
    assert conn.had_errors is False
    assert sleeps == [MIRROR_RETRY_BACKOFF_SECONDS]


def test_org_listing_rate_limit_is_transient(monkeypatch):
    monkeypatch.setattr(
        ge.subprocess, "run",
        lambda args, **kw: subprocess.CompletedProcess(
            args, 1, "", "gh: API rate limit exceeded for user ID 1. (HTTP 403)"),
    )
    with pytest.raises(TransientGitError):
        ge._org_repos("acme")


# ─── retry messages ─────────────────────────────────────────────────────────

def _always_transient():
    raise TransientGitError("mirror refresh timed out after 120s for acme/r0")


def test_the_final_message_names_the_attempt_cap():
    with pytest.raises(TransientGitError, match=r"\(attempt 3 of 3\)$"):
        ge._with_retries(_always_transient, attempts=3, sleep=lambda _s: None)


def test_the_final_message_names_an_exhausted_budget():
    with pytest.raises(TransientGitError,
                       match=r"\(attempt 1; run retry budget exhausted\)$"):
        ge._with_retries(_always_transient, attempts=3, budget=RetryBudget(0),
                         sleep=lambda _s: None)


# ─── circuit breaker ───────────────────────────────────────────────────────

def _down_names(n):
    return [f"acme/down{i}" for i in range(n)]


def test_an_outage_stops_refreshing_after_the_breaker_threshold(
        tmp_path, monkeypatch, repos):
    """Every repository times out and none succeeds: after the breaker's
    threshold the rest are skipped, not each given its own timeout."""
    names = _down_names(ge.MIRROR_CIRCUIT_BREAKER_FAILURES + 3)
    calls = []
    _fake_mirrors(monkeypatch, repos, failing=set(names), calls=calls)
    conn = _connector(tmp_path, names)
    conn.retry_budget = RetryBudget(0)
    conn.run()
    assert calls == names[:ge.MIRROR_CIRCUIT_BREAKER_FAILURES]
    assert conn.had_errors is True
    assert sorted(_record(tmp_path)) == sorted(calls)


def test_the_breaker_stays_closed_once_a_refresh_has_succeeded(
        tmp_path, monkeypatch, repos):
    names = ["acme/r0"] + _down_names(ge.MIRROR_CIRCUIT_BREAKER_FAILURES + 1)
    calls = []
    _fake_mirrors(monkeypatch, repos, failing=set(names[1:]), calls=calls)
    conn = _connector(tmp_path, names)
    conn.retry_budget = RetryBudget(0)
    conn.run()
    assert calls == names


# ─── failure record housekeeping ───────────────────────────────────────────

def _seed_record(tmp_path, names):
    path = tmp_path / "mirrors" / REFRESH_FAILURES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        n: {"consecutive_runs": 2, "first_failed_at": "2026-11-24T00:00:00+00:00",
            "last_failed_at": "2026-11-25T00:00:00+00:00", "last_error": "x"}
        for n in names
    }))


def test_a_repository_no_longer_configured_is_pruned_from_the_record(
        tmp_path, monkeypatch, repos):
    _seed_record(tmp_path, ["acme/gone"])
    _fake_mirrors(monkeypatch, repos, failing={"acme/r0"})
    conn = _connector(tmp_path, sorted(repos))
    conn.run()
    assert sorted(_record(tmp_path)) == ["acme/r0"]


def test_streaks_survive_a_run_whose_org_listing_failed(tmp_path, monkeypatch, repos):
    """The listing failed, so the org repositories were never re-tested:
    pruning them would silently reset their streaks."""
    _seed_record(tmp_path, ["acme/org-only"])
    _fake_mirrors(monkeypatch, repos, failing={"acme/r0"})

    def no_listing(_org, *, timeout):
        raise RuntimeError("org listing failed for acme: HTTP 401")

    monkeypatch.setattr(ge, "_org_repos", no_listing)
    conn = _connector(tmp_path, sorted(repos))
    conn.org = "acme"
    conn.run()
    assert sorted(_record(tmp_path)) == ["acme/org-only", "acme/r0"]


def test_streaks_survive_a_run_the_breaker_cut_short(tmp_path, monkeypatch, repos):
    names = _down_names(ge.MIRROR_CIRCUIT_BREAKER_FAILURES + 1)
    _seed_record(tmp_path, [names[-1]])
    _fake_mirrors(monkeypatch, repos, failing=set(names))
    conn = _connector(tmp_path, names)
    conn.retry_budget = RetryBudget(0)
    conn.run()
    assert _record(tmp_path)[names[-1]]["consecutive_runs"] == 2  # untouched


def test_a_leftover_tmp_file_from_a_crashed_write_is_swept(tmp_path, monkeypatch, repos):
    leftover = tmp_path / "mirrors" / f"{REFRESH_FAILURES_FILE}.tmp.99999"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("{")
    _fake_mirrors(monkeypatch, repos, failing={"acme/r0"})
    conn = _connector(tmp_path, sorted(repos))
    conn.run()
    assert not leftover.exists()
    assert "acme/r0" in _record(tmp_path)
