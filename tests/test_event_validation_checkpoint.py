import json
import threading
from dataclasses import asdict

import pytest

from remembrancerlib import (
    Event,
    EventValidationError,
    PipelineBusyError,
    append_events,
    event_validation_status,
    iter_validated_events,
)
from tools.corpus_guard import inspect
import tools.corpus_guard as corpus_guard


def _append(root, event):
    events = root / "events"
    events.mkdir(exist_ok=True)
    with open(events / "2026-07-15.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")


def _event(event_id):
    return Event(
        id=event_id,
        source="slack",
        author="person",
        payload={"text": event_id},
        occurred_at="2026-07-15T12:00:00Z",
        ingested_at="2026-07-15T12:01:00Z",
    )


def test_validated_reader_hides_async_tail_until_next_guard(tmp_path):
    _append(tmp_path, _event("slack:validated"))
    assert inspect(str(tmp_path), bootstrap_checkpoint=True)["ok"]
    assert [event.id for event in iter_validated_events(str(tmp_path / "events"))] == [
        "slack:validated"
    ]

    _append(tmp_path, _event("slack:async-tail"))
    status = event_validation_status(str(tmp_path / "events"))
    assert status["ok"] is True
    assert status["behind"] is True
    assert status["current"] is False
    assert [event.id for event in iter_validated_events(str(tmp_path / "events"))] == [
        "slack:validated"
    ]

    assert inspect(str(tmp_path))["ok"]
    assert event_validation_status(str(tmp_path / "events"))["current"] is True
    assert [event.id for event in iter_validated_events(str(tmp_path / "events"))] == [
        "slack:validated", "slack:async-tail"
    ]


def test_validated_reader_fails_closed_without_checkpoint_or_after_mutation(tmp_path):
    _append(tmp_path, _event("slack:one"))
    with pytest.raises(EventValidationError, match="checkpoint is missing"):
        list(iter_validated_events(str(tmp_path / "events")))

    assert inspect(str(tmp_path), bootstrap_checkpoint=True)["ok"]
    path = tmp_path / "events" / "2026-07-15.jsonl"
    path.write_bytes(path.read_bytes().replace(b"slack:one", b"slack:won", 1))
    with pytest.raises(EventValidationError, match="prefix was modified"):
        list(iter_validated_events(str(tmp_path / "events")))


def test_malformed_checkpoint_never_falls_back_or_gets_overwritten(tmp_path):
    _append(tmp_path, _event("slack:one"))
    state = tmp_path / "state"
    state.mkdir()
    checkpoint = state / "event_validation_checkpoint.json"
    checkpoint.write_text(json.dumps({
        "version": 1,
        "files": {"../outside.jsonl": {"size": 0, "lines": 0, "sha256": "0" * 64}},
    }))
    before = checkpoint.read_bytes()

    with pytest.raises(EventValidationError, match="filename"):
        list(iter_validated_events(str(tmp_path / "events")))
    result = inspect(str(tmp_path), bootstrap_checkpoint=True)
    assert result["ok"] is False
    assert result["checkpoint_advanced"] is False
    assert checkpoint.read_bytes() == before


def test_validated_reader_watermark_recovery_is_prefix_scoped(tmp_path):
    _append(tmp_path, _event("slack:one"))
    _append(tmp_path, _event("slack:two"))
    assert inspect(str(tmp_path), bootstrap_checkpoint=True)["ok"]
    _append(tmp_path, _event("slack:unvalidated"))

    assert [event.id for event in iter_validated_events(
        str(tmp_path / "events"), after_id="slack:one"
    )] == ["slack:two"]
    with pytest.warns(RuntimeWarning, match="validated prefix"):
        replayed = list(iter_validated_events(
            str(tmp_path / "events"), after_id="slack:unvalidated"
        ))
    assert [event.id for event in replayed] == ["slack:one", "slack:two"]


def test_guard_checkpoint_and_appender_share_one_snapshot_lock(tmp_path, monkeypatch):
    append_events([_event("slack:one")], str(tmp_path / "events"))
    assert inspect(str(tmp_path), bootstrap_checkpoint=True)["ok"]

    at_checkpoint_write = threading.Event()
    release_guard = threading.Event()
    append_finished = threading.Event()
    append_errors = []
    original_write = corpus_guard._atomic_write_json

    def paused_write(path, payload):
        at_checkpoint_write.set()
        assert release_guard.wait(2)
        original_write(path, payload)

    monkeypatch.setattr(corpus_guard, "_atomic_write_json", paused_write)
    guard_result = {}

    def run_guard():
        guard_result.update(inspect(str(tmp_path)))

    guard_thread = threading.Thread(target=run_guard)
    guard_thread.start()
    assert at_checkpoint_write.wait(2)

    def run_append():
        try:
            append_events([_event("slack:after-snapshot")], str(tmp_path / "events"))
        except Exception as exc:  # captured for an assertion in the main thread
            append_errors.append(exc)
        finally:
            append_finished.set()

    append_thread = threading.Thread(target=run_append)
    append_thread.start()
    # Interactive appenders fail fast while corpus_guard owns the outer pipeline
    # transaction.  They must not block and later leak into its authenticated
    # snapshot.
    assert append_finished.wait(1)
    assert len(append_errors) == 1
    assert isinstance(append_errors[0], PipelineBusyError)
    release_guard.set()
    guard_thread.join(2)
    append_thread.join(2)
    assert guard_result["ok"] is True
    assert append_finished.is_set()

    # The rejected append did not enter the checkpoint.  A caller retry after
    # the guarded transaction succeeds as a new, unvalidated tail.
    assert [event.id for event in iter_validated_events(str(tmp_path / "events"))] == [
        "slack:one"
    ]
    assert event_validation_status(str(tmp_path / "events"))["current"] is True
    assert append_events(
        [_event("slack:after-snapshot")], str(tmp_path / "events")
    ) == 1
    assert event_validation_status(str(tmp_path / "events"))["behind"] is True
