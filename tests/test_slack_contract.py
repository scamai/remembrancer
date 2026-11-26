"""Closed-contract tests shared by Slack ingest and source verification."""
from copy import deepcopy

import pytest

from distill import verify as V
from ingest.slack_contract import (
    SlackContractError,
    snapshot_revision,
    thread_v2_event_valid,
    thread_v2_text_authority,
    validate_thread_v2_event,
)
from remembrancerlib import Event


def _event(*, payload=None, author="U1"):
    payload = payload or {
        "text": "[#engineering] U1: Model status passed",
        "reply_count": 0,
        "message_count": 1,
        "snapshot_complete": True,
        "snapshot_format": "thread-v2",
        "resolution_reactions": [],
        "signal_eligible": True,
    }
    revision = snapshot_revision(author, payload, [])
    return Event(
        id=f"slack:engineering/30.000@{revision}",
        source="slack",
        author=author,
        occurred_at="1970-01-01T00:00:30Z",
        payload=payload,
        provenance={
            "channel": "engineering",
            "channel_id": "C1",
            "ts": "30.000",
            "thread_ts": "30.000",
            "revision": revision,
            "permalink": None,
        },
    )


def _replace_revision(event):
    revision = snapshot_revision(event.author, event.payload, event.acl)
    event.provenance["revision"] = revision
    event.id = f"slack:engineering/30.000@{revision}"


def test_connector_contract_is_the_verifier_allowlist():
    event = _event()

    assert validate_thread_v2_event(event) is None
    assert thread_v2_event_valid(event) is True
    assert thread_v2_text_authority(event) is True
    assert V._snapshot_source_eligible(event) is True


def test_snapshot_revision_binds_acl_policy():
    event = _event()
    public_revision = snapshot_revision(event.author, event.payload, [])
    restricted_revision = snapshot_revision(event.author, event.payload, ["eng"])

    assert public_revision != restricted_revision


def test_thread_v2_revision_algorithm_is_frozen_for_persisted_events():
    event = _event()

    assert event.provenance["revision"] == "c3be41a142fb4bed"


@pytest.mark.parametrize("forgery", [
    "payload_only",
    "author_after_revision",
    "payload_after_revision",
    "extra_payload_field",
    "id_channel",
    "provenance_channel",
    "provenance_thread",
    "extra_provenance_field",
    "wrong_source",
])
def test_forged_slack_like_events_fail_closed_in_contract_and_verifier(forgery):
    event = deepcopy(_event())
    if forgery == "payload_only":
        event = {"id": "slack:1", "source": "slack", "payload": event.payload}
    elif forgery == "author_after_revision":
        event.author = "Mallory"
    elif forgery == "payload_after_revision":
        event.payload["text"] += " altered"
    elif forgery == "extra_payload_field":
        event.payload["hidden_state"] = "failed"
        _replace_revision(event)
    elif forgery == "id_channel":
        event.id = event.id.replace("engineering", "random")
    elif forgery == "provenance_channel":
        event.provenance["channel"] = "random"
    elif forgery == "provenance_thread":
        event.provenance["thread_ts"] = "31.000"
    elif forgery == "extra_provenance_field":
        event.provenance["trusted"] = True
    elif forgery == "wrong_source":
        event.source = "session"

    with pytest.raises(SlackContractError):
        validate_thread_v2_event(event)
    assert thread_v2_event_valid(event) is False
    assert thread_v2_text_authority(event) is False
    assert V._snapshot_source_eligible(event) is False


def test_reaction_snapshot_is_valid_but_not_generic_text_authority():
    event = _event()
    event.payload["resolution_reactions"] = [{
        "message_ts": "30.000", "name": "white_check_mark", "count": 1,
    }]
    _replace_revision(event)

    validate_thread_v2_event(event)
    assert thread_v2_text_authority(event) is False
    assert V._snapshot_source_eligible(event) is False


@pytest.mark.parametrize("evidence", [
    {"kind": "api-error", "code": "message_not_found"},
    {"kind": "source-subtype", "code": "message_deleted"},
])
def test_explicit_deletion_evidence_is_structurally_valid_but_not_text_authority(evidence):
    payload = {
        "text": "[#engineering] Slack thread 30.000 was deleted at the source.",
        "reply_count": 0,
        "message_count": 0,
        "snapshot_complete": True,
        "snapshot_format": "thread-v2",
        "resolution_reactions": [],
        "signal_eligible": False,
        "deleted": True,
        "subtype": "thread_deleted",
        "deletion_evidence": evidence,
    }
    event = _event(payload=payload, author="slack")

    validate_thread_v2_event(event)
    assert thread_v2_text_authority(event) is False
    assert V._snapshot_source_eligible(event) is False


@pytest.mark.parametrize("evidence", [
    None,
    {},
    {"kind": "api-error", "code": "unknown"},
    {"kind": "inference", "code": "empty_messages"},
])
def test_inferred_or_missing_deletion_evidence_is_rejected(evidence):
    event = _event(payload={
        "text": "[#engineering] Slack thread 30.000 was deleted at the source.",
        "reply_count": 0,
        "message_count": 0,
        "snapshot_complete": True,
        "snapshot_format": "thread-v2",
        "resolution_reactions": [],
        "signal_eligible": False,
        "deleted": True,
        "subtype": "thread_deleted",
        "deletion_evidence": evidence,
    }, author="slack")

    with pytest.raises(SlackContractError):
        validate_thread_v2_event(event)
