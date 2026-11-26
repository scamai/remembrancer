import pytest

from remembrancerlib import (
    Event, EventConflictError, EventSchemaError, append_events, iter_events,
)


def test_iter_events_unknown_watermark_warns_and_replays_all(tmp_path):
    events_dir = str(tmp_path / "events")
    append_events(
        [
            Event(id="git:one", source="git", author="a", payload={}),
            Event(id="git:two", source="git", author="b", payload={}),
        ],
        events_dir,
    )

    with pytest.warns(RuntimeWarning, match="watermark.*not found"):
        ids = [event.id for event in iter_events(events_dir, after_id="git:missing")]

    assert ids == ["git:one", "git:two"]


def test_iter_events_known_watermark_yields_only_following_events(tmp_path):
    events_dir = str(tmp_path / "events")
    append_events(
        [
            Event(id="git:one", source="git", author="a", payload={}),
            Event(id="git:two", source="git", author="b", payload={}),
        ],
        events_dir,
    )

    assert [event.id for event in iter_events(events_dir, after_id="git:one")] == ["git:two"]


def test_iter_events_drops_historical_duplicate_ids_in_stable_order(tmp_path):
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    (events_dir / "2026-01-01.jsonl").write_text(
        Event(id="gmail:legacy-thread", source="gmail", author="first",
              payload={"text": "first observation"},
              occurred_at="2026-01-01T00:00:00Z").to_json() + "\n"
    )
    (events_dir / "2026-01-02.jsonl").write_text(
        Event(id="gmail:legacy-thread", source="gmail", author="second",
              payload={"text": "conflicting legacy observation"},
              occurred_at="2026-01-02T00:00:00Z").to_json() + "\n" +
        Event(id="gmail:new", source="gmail", author="third", payload={},
              occurred_at="2026-01-02T01:00:00Z").to_json() + "\n"
    )

    events = list(iter_events(str(events_dir)))
    assert [(event.id, event.author) for event in events] == [
        ("gmail:legacy-thread", "first"),
        ("gmail:new", "third"),
    ]
    assert [event.id for event in iter_events(
        str(events_dir), after_id="gmail:legacy-thread"
    )] == ["gmail:new"]


def test_event_ids_are_unique_across_day_files(tmp_path):
    events_dir = str(tmp_path / "events")
    first = Event(id="gmail:stable", source="gmail", author="a", payload={"text": "first"},
                  occurred_at="2026-01-01T00:00:00Z")
    append_events([first], events_dir)

    replay = Event(id="gmail:stable", source="gmail", author="a", payload={"text": "first"},
                   occurred_at="2026-01-01T00:00:00Z")
    assert append_events([replay], events_dir) == 0

    conflict = Event(id="gmail:stable", source="gmail", author="a", payload={"text": "changed"},
                     occurred_at="2026-01-02T00:00:00Z")
    with pytest.raises(EventConflictError, match="reused"):
        append_events([conflict], events_dir)


def test_append_refuses_unterminated_partial_tail(tmp_path):
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    (events_dir / "2026-01-01.jsonl").write_text('{"id":"git:partial"')
    with pytest.raises(EventConflictError, match="partial tail"):
        append_events([Event(id="git:new", source="git", author="a", payload={})],
                      str(events_dir))
    assert list(iter_events(str(events_dir))) == []


@pytest.mark.parametrize("occurred_at", [
    "../state/x", "2026-99-99T00:00:00Z", "not-a-time", "",
])
def test_append_rejects_invalid_or_traversing_event_date(tmp_path, occurred_at):
    events_dir = tmp_path / "events"
    event = Event(
        id="session:bad-date", source="session", author="attacker", payload={},
        occurred_at=occurred_at,
    )
    with pytest.raises(EventSchemaError, match="occurred_at"):
        append_events([event], str(events_dir))
    assert not events_dir.exists()
    assert not (tmp_path / "state" / "x.jsonl").exists()


def test_append_validates_and_serializes_whole_batch_before_first_write(tmp_path):
    events_dir = tmp_path / "events"
    valid = Event(
        id="session:first", source="session", author="tester", payload={},
        occurred_at="2026-07-15T00:00:00Z",
    )
    invalid_later = Event(
        id="session:second", source="session", author="tester",
        payload={"bad": object()}, occurred_at="2026-07-16T00:00:00Z",
    )
    with pytest.raises(EventSchemaError, match="not JSON serializable"):
        append_events([valid, invalid_later], str(events_dir))
    assert not events_dir.exists()


@pytest.mark.parametrize("event", [
    Event(id="git:wrong", source="gmail", author="a", payload={}),
    Event(id="no-prefix", source="session", author="a", payload={}),
    Event(id="mystery:one", source="mystery", author="a", payload={}),
])
def test_append_rejects_source_and_id_contract_violations(tmp_path, event):
    with pytest.raises(EventSchemaError):
        append_events([event], str(tmp_path / "events"))
    assert not (tmp_path / "events").exists()
