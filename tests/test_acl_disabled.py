"""A legacy ACL-disable environment variable must never bypass access control."""
from retrieve import query as Q


def test_legacy_acl_disabled_variable_is_ignored(monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_ACL_DISABLED", "1")
    monkeypatch.delenv("REMEMBRANCER_ALLOW_FULL_ACCESS", raising=False)
    assert Q.grants_for("some-random-person") == set()
    assert Q.grants_for(None) is Q.DENY


def test_normal_acl_scope_is_unchanged(monkeypatch):
    monkeypatch.delenv("REMEMBRANCER_ACL_DISABLED", raising=False)
    monkeypatch.delenv("REMEMBRANCER_ALLOW_FULL_ACCESS", raising=False)
    # unknown identity → public-only (∅), not full access
    g = Q.grants_for("unknown-x")
    assert g is not Q.FULL_ACCESS and g == set()
