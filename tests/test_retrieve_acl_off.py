# tests/test_retrieve_acl_off.py
# ACL fail-open fix: a missing/empty identity DENIES by default (public-only);
# full access is reachable only via the explicit REMEMBRANCER_ALLOW_FULL_ACCESS flag.
from retrieve.query import search  # fixture writes 1 public + 1 acl:[ml] note


def test_anon_denies_by_default(tmp_remembrancer, monkeypatch):
    monkeypatch.delenv("REMEMBRANCER_ALLOW_FULL_ACCESS", raising=False)
    names = {h["name"] for h in search("detector", identity=None, root=tmp_remembrancer)}
    assert "public-note" in names
    assert "ml-only-note" not in names          # FAIL-CLOSED: no anon access to acl-labeled notes


def test_full_access_only_with_explicit_flag(tmp_remembrancer, monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_ALLOW_FULL_ACCESS", "1")
    names = {h["name"] for h in search("detector", identity=None, root=tmp_remembrancer)}
    assert "ml-only-note" in names and "public-note" in names   # opt-in full access restored
