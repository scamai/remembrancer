"""Query logging (Karpathy's log.md): every search records the query + result quality
to state/queries.jsonl, so the usage-feedback loop can find memory gaps.
"""
import json
import os
import pytest
from remembrancerlib import Note, write_note
from retrieve import query as Q


@pytest.fixture(autouse=True)
def _real_query_log(monkeypatch):
    """The suite-wide conftest disables query-log persistence (REMEMBRANCER_QUERY_LOG=off)
    so tests never pollute the checkout's real state/. These tests exercise the
    writer itself, so re-enable it (writes land under tmp_path roots)."""
    monkeypatch.delenv("REMEMBRANCER_QUERY_LOG", raising=False)


def _seed(root):
    ctx = os.path.join(root, "context")
    os.makedirs(ctx, exist_ok=True)
    write_note(Note(name="customer-acme-deal", description="Acme Corp signed pilot deal",
                    type="decision", domain="customers", acl=["customers"],
                    entities=["acme.com"], provenance=["gmail:x"],
                    body="Acme Corp signed a pilot deal with us."), ctx)
    # Logging is sensitive and therefore requires an explicit literal true.
    json.dump({"retrieval": {"query_log": True}},
              open(os.path.join(root, "config.json"), "w"))


def _log_rows(root):
    p = os.path.join(root, "state", "queries.jsonl")
    return [json.loads(line) for line in open(p)] if os.path.exists(p) else []


def test_search_logs_query_with_result_quality(tmp_path):
    root = str(tmp_path)
    _seed(root)
    Q.search("Acme pilot deal", identity=None, root=root)
    rows = _log_rows(root)
    assert len(rows) == 1
    r = rows[0]
    assert r["query"] == "Acme pilot deal"
    assert r["n_results"] >= 1 and r["top_score"] > 0
    assert "customer-acme-deal" in r["delivered"]
    assert isinstance(r["duration_ms"], int) and r["duration_ms"] >= 0


def test_zero_result_query_is_logged_as_missing_signal(tmp_path):
    root = str(tmp_path)
    _seed(root)
    Q.search("zzzqqq nonexistent unrelated terms", identity=None, root=root)
    r = _log_rows(root)[-1]
    assert r["n_results"] == 0 and r["top_score"] == 0.0   # the MISSING signal


def test_query_log_can_be_disabled(tmp_path):
    root = str(tmp_path)
    _seed(root)
    json.dump({"retrieval": {"query_log": False}},
              open(os.path.join(root, "config.json"), "w"))
    Q.search("Acme", identity=None, root=root)
    assert _log_rows(root) == []


def test_query_log_defaults_off_when_setting_is_absent(tmp_path):
    root = str(tmp_path)
    _seed(root)
    json.dump({"retrieval": {}}, open(os.path.join(root, "config.json"), "w"))
    Q.search("confidential question", identity=None, root=root)
    assert _log_rows(root) == []


def test_query_log_rejects_truthy_non_boolean_setting(tmp_path):
    root = str(tmp_path)
    _seed(root)
    json.dump({"retrieval": {"query_log": "true"}},
              open(os.path.join(root, "config.json"), "w"))
    Q.search("confidential question", identity=None, root=root)
    assert _log_rows(root) == []
