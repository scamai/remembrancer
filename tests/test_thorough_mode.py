"""Max-quality mode: distill.thorough=true makes the LLM judge EVERY cluster, not just
the regex-durable ones. A non-durable email cluster should reach the LLM only when thorough.
"""
import json
import os
import types
from distill import engine
from distill.consolidate import ConsolidationDecision
from remembrancerlib import Event, append_events
from tests.event_validation_helpers import guarded_run


def _seed_nondurable(root):
    # gmail is not a durable source, and this text has no DECISION_RE keyword -> the
    # default durability gate would skip it (no LLM call).
    append_events([Event(id="gmail:t1", source="gmail", author="x@acme.io",
                   payload={"subject": "quick hello", "snippet": "just saying hi"},
                   acl=["customers"], entities=["acme.io"],
                   provenance={"threadId": "t1", "domain": "customers"})],
                  os.path.join(root, "events"))


def _write_config(root, thorough):
    json.dump({"distill": {"thorough": thorough}},
              open(os.path.join(root, "config.json"), "w"))


def _count_consolidate_calls(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(engine, "LLMClient",
                        lambda: types.SimpleNamespace(available=lambda: True))

    def fake(cl, cand, client):
        calls["n"] += 1
        return ConsolidationDecision(action="IGNORE")
    monkeypatch.setattr(engine, "consolidate", fake)
    return calls


def test_nondurable_cluster_skips_the_llm_by_default(tmp_path, monkeypatch):
    root = str(tmp_path); _seed_nondurable(root); _write_config(root, thorough=False)
    calls = _count_consolidate_calls(monkeypatch)
    guarded_run(engine, root=root, use_llm=True)
    assert calls["n"] == 0          # durability gate skips it — no LLM call


def test_thorough_sends_nondurable_cluster_to_the_llm(tmp_path, monkeypatch):
    root = str(tmp_path); _seed_nondurable(root); _write_config(root, thorough=True)
    calls = _count_consolidate_calls(monkeypatch)
    guarded_run(engine, root=root, use_llm=True)
    assert calls["n"] == 1          # thorough -> the LLM judges every cluster
