import json
import re
import subprocess
import sys
import os

import pytest

from remembrancerlib import Event

# Absolute path to the repo root so scripts/run.py resolves regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_run_pipeline_end_to_end(tmp_path, tmp_git_repo):
    r = subprocess.run(
        [sys.executable, os.path.join(_REPO_ROOT, "scripts", "run.py"),
         "--ingest-git", tmp_git_repo],
        capture_output=True, text=True,
        cwd=str(tmp_path),
    )
    assert r.returncode == 0, r.stderr
    assert os.path.exists(os.path.join(str(tmp_path), "context", "INDEX.md"))


def test_repeated_ingest_git_flags_keep_every_repo(tmp_path, monkeypatch):
    from scripts import run

    seen = []

    class FakeConnector:
        def __init__(self, config, events_dir):
            seen.extend(repo["path"] for repo in config["repos"])

        def run(self):
            return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "GitConnector", FakeConnector)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--ingest-git", "repo-a", "--ingest-git", "repo-b", "--no-distill"],
    )

    run.main()

    assert seen == ["repo-a", "repo-b"]


def test_run_refuses_to_bootstrap_preexisting_unvalidated_history(tmp_path, monkeypatch):
    from dataclasses import asdict
    from scripts import run

    events = tmp_path / "events"
    events.mkdir()
    (events / "2026-07-15.jsonl").write_text(
        json.dumps(asdict(Event(
            id="slack:history", source="slack", author="person",
            payload={"text": "preexisting"},
            occurred_at="2026-07-15T12:00:00Z",
            ingested_at="2026-07-15T12:01:00Z",
        ))) + "\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run.py", "--no-distill"])

    with pytest.raises(RuntimeError, match="attended bootstrap required"):
        run.main()
    assert not (tmp_path / "state" / "event_validation_checkpoint.json").exists()


def _dated_repo(path, author, email, commits):
    """A git repo whose commits carry the given dates and subjects."""
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    for date, subject in commits:
        env = {**os.environ, "GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": email,
               "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": email,
               "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
        subprocess.run(["git", "-C", str(path), "-c", "commit.gpgsign=false", "commit", "-q",
                        "--allow-empty", "-m", subject], check=True, env=env)
    out = subprocess.run(["git", "-C", str(path), "log", "--format=%H"], check=True,
                         capture_output=True, text=True)
    return set(out.stdout.split())


def _run_ingest(work, repo):
    r = subprocess.run([sys.executable, os.path.join(_REPO_ROOT, "scripts", "run.py"),
                        "--ingest-git", str(repo)], capture_output=True, text=True, cwd=str(work))
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stderr


def _cited_commits(work):
    """Every git sha a note's provenance names."""
    cited = set()
    for d, _, names in os.walk(work / "context"):
        for n in names:
            if n.endswith(".md"):
                with open(os.path.join(d, n), encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("provenance:"):
                            cited.update(re.findall(r"git:([0-9a-f]{40})", line))
    return cited


def test_a_repository_ingested_later_is_distilled_behind_the_watermark(tmp_path):
    """Events are filed by their own dates and the distill watermark is a
    position in that order, so a second repository's older commits land behind
    it. The run distills them in a targeted pass instead of skipping them."""
    first = _dated_repo(tmp_path / "a", "Dana Okafor", "dana@acme.example", [
        ("2026-11-05T10:00:00Z", "feat(ledger-service): posting API with double-entry checks"),
        ("2026-11-07T10:00:00Z", "docs: Sam Rivera becomes on-call owner for ledger-service"),
    ])
    second = _dated_repo(tmp_path / "b", "Avery Chen", "avery@acme.example", [
        ("2026-11-01T10:00:00Z", "feat(billing-portal): invoice PDF export"),
        ("2026-11-03T10:00:00Z", "docs: Avery Chen owns billing-portal and its on-call rota"),
        ("2026-11-09T10:00:00Z", "fix(billing-portal): currency rounding in invoice totals"),
    ])
    work = tmp_path / "work"
    work.mkdir()

    log = _run_ingest(work, tmp_path / "a")
    assert "behind the distill watermark" not in log  # a first run has no watermark
    assert first <= _cited_commits(work)

    log = _run_ingest(work, tmp_path / "b")
    # The two commits older than the first repository's newest; the third is
    # past the watermark, where the normal pass reads it.
    assert "[run] distilling 2 ingested events behind the distill watermark" in log
    assert first | second <= _cited_commits(work)

    notes_before = sorted(p.name for p in work.joinpath("context").rglob("*.md"))
    log = _run_ingest(work, tmp_path / "b")  # nothing new: nothing to catch up
    assert "behind the distill watermark" not in log
    assert sorted(p.name for p in work.joinpath("context").rglob("*.md")) == notes_before


def test_a_rebuild_needs_no_catch_up_pass(tmp_path):
    """--rebuild already reprocesses every event; a second, targeted pass would
    distill the late repository's commits twice."""
    _dated_repo(tmp_path / "a", "Dana Okafor", "dana@acme.example", [
        ("2026-11-07T10:00:00Z", "docs: Sam Rivera becomes on-call owner for ledger-service")])
    late = _dated_repo(tmp_path / "b", "Avery Chen", "avery@acme.example", [
        ("2026-11-01T10:00:00Z", "feat(billing-portal): invoice PDF export")])
    work = tmp_path / "work"
    work.mkdir()
    _run_ingest(work, tmp_path / "a")
    r = subprocess.run([sys.executable, os.path.join(_REPO_ROOT, "scripts", "run.py"),
                        "--ingest-git", str(tmp_path / "b"), "--rebuild"],
                       capture_output=True, text=True, cwd=str(work))
    assert r.returncode == 0, r.stderr[-3000:]
    assert "behind the distill watermark" not in r.stderr
    assert late <= _cited_commits(work)


# ─── The verification audit is sealed before a run appends ───────────────────
#
# A verdict append never rotates the audit (it refuses once the file is full),
# so the pipeline run seals a nearly full audit first, under its lock.

def _audit_with_one_verdict(root):
    from remembrancerlib.verification_audit import append_verification_audit_record

    (root / "state").mkdir(parents=True, exist_ok=True)
    path = str(root / "state" / "verification.jsonl")
    assert append_verification_audit_record({
        "note": "n", "status": "verified", "checked": 1, "supported": 1,
        "contradicted": 0, "audit_id": "n-0", "content_fingerprint": "a" * 64,
    }, path)
    return path


def _staged_run(tmp_path, monkeypatch, rotate):
    """Run main() on an ingest of one repository, recording each step in order."""
    from scripts import run

    steps = []

    class FakeConnector:
        def __init__(self, config, events_dir):
            pass

        def run(self):
            steps.append("ingest")
            return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REMEMBRANCER_PIPELINE_LOCK_HELD", raising=False)
    monkeypatch.setattr(run, "GitConnector", FakeConnector)
    monkeypatch.setattr(run.distiller, "run", lambda **kw: steps.append("distill"))
    monkeypatch.setattr(run, "rotate_verification_audit_if_full",
                        lambda path: rotate(path, steps))
    monkeypatch.setattr(sys, "argv", ["run.py", "--ingest-git", "repo-a"])
    run.main()
    return steps


def test_the_run_seals_a_full_audit_under_its_lock_before_it_ingests(tmp_path, monkeypatch,
                                                                    capsys):
    from remembrancerlib.locks import pipeline_mutation_lock_held
    from remembrancerlib.verification_audit import rotate_verification_audit_if_full

    path = _audit_with_one_verdict(tmp_path)

    def rotate(audit, steps):
        steps.append(("rotate", os.path.abspath(audit), pipeline_mutation_lock_held(".")))
        return rotate_verification_audit_if_full(audit, high_water=1)  # "full" here

    steps = _staged_run(tmp_path, monkeypatch, rotate)

    assert steps == [("rotate", path, True), "ingest", "distill"]
    assert (tmp_path / "state" / "verification.1.jsonl").exists(), "segment 1 is sealed"
    assert os.path.getsize(path) == 0, "a valid empty audit is left for this run's verdicts"
    assert "[run] sealed verification audit segment 1" in capsys.readouterr().err


def test_an_audit_below_the_mark_is_left_as_it_is(tmp_path, monkeypatch):
    from remembrancerlib.verification_audit import rotate_verification_audit_if_full

    path = _audit_with_one_verdict(tmp_path)
    before = open(path, "rb").read()

    steps = _staged_run(tmp_path, monkeypatch,
                        lambda audit, steps: rotate_verification_audit_if_full(audit))

    assert steps == ["ingest", "distill"]
    assert open(path, "rb").read() == before
    assert not (tmp_path / "state" / "verification.1.jsonl").exists()


def test_a_rotation_that_fails_warns_and_the_run_goes_on(tmp_path, monkeypatch, capsys):
    def rotate(audit, steps):
        raise OSError("refusing to rotate a verification audit that does not match its head")

    steps = _staged_run(tmp_path, monkeypatch, rotate)

    assert steps == ["ingest", "distill"]
    err = capsys.readouterr().err
    assert "[run] verification audit not rotated: refusing to rotate" in err
