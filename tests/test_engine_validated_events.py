import os
import json

import pytest

from distill import engine
from remembrancerlib import Event, EventValidationError, append_events
from tools.corpus_guard import inspect


def _event(event_id):
    return Event(
        id=event_id,
        source="slack",
        author="person",
        payload={"text": "status update"},
        occurred_at="2026-07-15T12:00:00Z",
        ingested_at="2026-07-15T12:01:00Z",
    )


def test_engine_rebuild_consumes_only_validated_prefix(tmp_path):
    root = str(tmp_path)
    events_dir = os.path.join(root, "events")
    append_events([_event("slack:validated")], events_dir)
    assert inspect(root, bootstrap_checkpoint=True)["ok"]
    append_events([_event("slack:async-tail")], events_dir)

    first = engine.run(root=root, use_llm=False, rebuild=True)
    assert first["events_read"] == 1
    assert first["last_event_id"] == "slack:validated"

    assert inspect(root)["ok"]
    second = engine.run(root=root, use_llm=False, rebuild=True)
    assert second["events_read"] == 2
    assert second["last_event_id"] == "slack:async-tail"


def test_targeted_distill_rejects_unvalidated_or_mismatched_event(tmp_path):
    root = str(tmp_path)
    events_dir = os.path.join(root, "events")
    validated = _event("slack:validated")
    append_events([validated], events_dir)
    assert inspect(root, bootstrap_checkpoint=True)["ok"]

    tail = _event("slack:tail")
    append_events([tail], events_dir)
    with pytest.raises(EventValidationError, match="outside the validated"):
        engine.run(root=root, use_llm=False, only_events=[tail])

    forged = _event("slack:validated")
    forged.payload = {"text": "forged caller copy"}
    with pytest.raises(EventValidationError, match="differs from"):
        engine.run(root=root, use_llm=False, only_events=[forged])


def test_verification_enabled_refuses_keyless_write_pass(tmp_path, monkeypatch):
    root = str(tmp_path)
    events_dir = os.path.join(root, "events")
    append_events([_event("slack:needs-verification")], events_dir)
    assert inspect(root, bootstrap_checkpoint=True)["ok"]
    (tmp_path / "config.json").write_text(
        json.dumps({"distill": {"verify": True}}), encoding="utf-8"
    )

    class Client:
        def available(self):
            return False

    monkeypatch.setattr(engine, "LLMClient", Client)
    with pytest.raises(RuntimeError, match=r"the LLM is unavailable \(it needs "
                                           r"REMEMBRANCER_LLM_API_KEY and REMEMBRANCER_LLM_MODEL"):
        engine.run(root=root, use_llm=False)
    assert not (tmp_path / "context").exists()
    assert not (tmp_path / "state" / "cursors.json").exists()
