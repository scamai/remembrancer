"""Git sensitive-content ACL: off by default, on with {"git": {"sensitive_acl": true}}.

It stays off by default until an operator has reviewed the existing regex hits;
these tests pin that the flag is the only switch and that it reaches `run()`.
"""
import subprocess

import pytest

from ingest.git_events import GitConnector
from remembrancerlib import Cursors, iter_events


def _commit_raw(subject, files=("src/app.py",)):
    return {"repo": "acme/widget", "sha": "a" * 40, "author": "A <a@example.com>",
            "ts": "2026-10-16T00:00:00Z", "subject": subject, "files": list(files)}


@pytest.mark.parametrize("config", [
    {},
    {"git": {}},
    {"git": {"sensitive_acl": False}},
    {"git": {"sensitive_acl": "true"}},  # only the JSON boolean turns it on
    {"git": ["sensitive_acl"]},
])
def test_git_sensitive_acl_is_off_unless_enabled(config):
    connector = GitConnector({"repos": [], **config})
    assert connector.sensitive_acl is False
    assert connector.normalize(_commit_raw("fix: salary band table")).acl == ["eng"]


def test_git_sensitive_acl_on_relabels_only_sensitive_commits():
    connector = GitConnector({"repos": [], "git": {"sensitive_acl": True}})
    assert connector.normalize(_commit_raw("fix: salary band table")).acl == ["secrets"]
    # changed paths are part of the text distill reads, so they count too
    assert connector.normalize(
        _commit_raw("docs: refresh", files=["docs/visa-process.md"])).acl == ["secrets"]
    assert connector.normalize(_commit_raw("feat: faster detector")).acl == ["eng"]


def test_git_sensitive_acl_also_narrows_an_explicit_acl_override():
    connector = GitConnector({"repos": [], "acl": ["ml"], "git": {"sensitive_acl": True}})
    assert connector.normalize(_commit_raw("chore: payroll export")).acl == ["secrets"]
    assert connector.normalize(_commit_raw("chore: lint")).acl == ["ml"]


def _git(cwd, *args):
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True)


@pytest.mark.parametrize("enabled, expected", [
    (False, {"fix: salary band table": ["eng"], "feat: faster detector": ["eng"]}),
    (True, {"fix: salary band table": ["secrets"], "feat: faster detector": ["eng"]}),
])
def test_git_run_reads_the_flag_from_config(tmp_path, enabled, expected):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    _git(str(repo), "config", "user.name", "Test Author")
    _git(str(repo), "config", "user.email", "author@example.com")
    for index, subject in enumerate(expected):
        (repo / f"f{index}.txt").write_text(f"{subject}\n")
        _git(str(repo), "add", ".")
        _git(str(repo), "commit", "-q", "-m", subject)

    connector = GitConnector(
        {"repos": [{"path": str(repo), "name": "acme/widget"}],
         "git": {"sensitive_acl": enabled}},
        events_dir=str(tmp_path / "events"),
        cursors=Cursors(str(tmp_path / "state" / "cursors.json")),
    )
    assert connector.run() == 2
    labels = {event.payload["subject"]: event.acl
              for event in iter_events(str(tmp_path / "events"))}
    assert labels == expected
