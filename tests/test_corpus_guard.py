import json
from dataclasses import asdict

from remembrancerlib import Event
from tools.corpus_guard import inspect


def _write(root, records, mode="w"):
    events = root / "events"
    events.mkdir(exist_ok=True)
    with open(events / "2026-07-15.jsonl", mode) as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _event(event_id, source="slack", occurred="2026-07-15T12:00:00Z",
           ingested="2026-07-15T12:01:00Z"):
    return asdict(Event(id=event_id, source=source, author="person", payload={"text": "x"},
                        occurred_at=occurred, ingested_at=ingested))


def _bootstrap(root):
    result = inspect(str(root), bootstrap_checkpoint=True)
    assert result["ok"] is True, result
    assert result["bootstrapped"] is True
    return result


def test_guard_requires_explicit_attended_bootstrap(tmp_path):
    _write(tmp_path, [_event("slack:old")])
    result = inspect(str(tmp_path), since="2026-07-15T12:00:00Z")
    assert result["ok"] is False
    assert result["checkpoint_advanced"] is False
    assert any("--bootstrap-checkpoint" in error for error in result["errors"])

    bootstrapped = _bootstrap(tmp_path)
    assert bootstrapped["new_events"] == 0
    checkpoint = json.loads(
        (tmp_path / "state" / "event_validation_checkpoint.json").read_text()
    )
    metadata = checkpoint["files"]["2026-07-15.jsonl"]
    assert metadata["size"] > 0
    assert metadata["lines"] == 1
    assert len(metadata["sha256"]) == 64


def test_guard_counts_physical_tail_not_ingested_timestamp(tmp_path):
    _write(tmp_path, [_event("slack:old")])
    _bootstrap(tmp_path)
    # This record deliberately claims an old ingestion time.  It is still new
    # because its bytes are beyond the authenticated prefix.
    _write(tmp_path, [_event("slack:late", ingested="2020-01-01T00:00:00Z"),
                      _event("gcal:near", "gcal", "2026-08-01T00:00:00Z")], mode="a")
    result = inspect(str(tmp_path), since="2040-01-01T00:00:00Z")
    assert result["ok"] is True
    assert result["new_events"] == 2
    assert result["new_by_source"] == {"gcal": 1, "slack": 1}


def test_guard_rejects_tail_far_future_calendar_and_volume_without_advancing(tmp_path):
    _write(tmp_path, [_event("slack:old")])
    _bootstrap(tmp_path)
    checkpoint_path = tmp_path / "state" / "event_validation_checkpoint.json"
    before = checkpoint_path.read_bytes()
    records = [_event(f"gcal:{index}", "gcal", "2040-01-01T00:00:00Z")
               for index in range(3)]
    _write(tmp_path, records, mode="a")
    result = inspect(str(tmp_path), max_new_events=2, max_new_gcal=2, future_days=180)
    assert result["ok"] is False
    assert result["checkpoint_advanced"] is False
    assert checkpoint_path.read_bytes() == before
    assert result["new_future_gcal"] == 3
    assert any("event budget" in error for error in result["errors"])
    assert any("gcal budget" in error for error in result["errors"])


def test_guard_warns_bootstrap_conflict_but_rejects_tail_conflict(tmp_path):
    old_first = _event("gmail:legacy", "gmail")
    old_second = _event("gmail:legacy", "gmail")
    old_second["payload"] = {"text": "historical conflict"}
    _write(tmp_path, [old_first, old_second])
    legacy = _bootstrap(tmp_path)
    assert legacy["conflicting_ids"] == 1
    assert legacy["warnings"]

    current = _event("gmail:legacy", "gmail", ingested="2020-01-01T00:00:00Z")
    current["payload"] = {"text": "new conflict"}
    _write(tmp_path, [current], mode="a")
    result = inspect(str(tmp_path))
    assert result["ok"] is False
    assert any("newly appended" in error for error in result["errors"])


def test_guard_rejects_prefix_mutation_and_truncation(tmp_path):
    _write(tmp_path, [_event("slack:one"), _event("slack:two")])
    _bootstrap(tmp_path)
    path = tmp_path / "events" / "2026-07-15.jsonl"
    original = path.read_bytes()
    path.write_bytes(original.replace(b'"slack:one"', b'"slack:won"', 1))
    mutated = inspect(str(tmp_path))
    assert mutated["ok"] is False
    assert any("prefix was modified" in error for error in mutated["errors"])

    path.write_bytes(original[: len(original) // 2])
    truncated = inspect(str(tmp_path))
    assert truncated["ok"] is False
    assert any("truncated" in error for error in truncated["errors"])


def test_guard_strictly_validates_types_and_timestamps(tmp_path):
    bad = _event("slack:bad")
    bad["author"] = 42
    bad["occurred_at"] = "not-a-time"
    bad["payload"] = "not-an-object"
    bad["acl"] = ["eng", 1]
    _write(tmp_path, [bad])
    result = inspect(str(tmp_path), bootstrap_checkpoint=True)
    assert result["ok"] is False
    joined = "\n".join(result["errors"])
    assert "author must be a string" in joined
    assert "occurred_at must be a parseable" in joined
    assert "payload must be an object" in joined
    assert "acl must be a list of strings" in joined
    assert not (tmp_path / "state" / "event_validation_checkpoint.json").exists()


def test_guard_allows_meeting_legacy_prefix_alias(tmp_path):
    _write(tmp_path, [_event("bluedot:call@revision", "meeting")])
    assert inspect(str(tmp_path), bootstrap_checkpoint=True)["ok"] is True
