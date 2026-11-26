"""Which labels hold a note for a person, and which labels a person's note gets, come from config.

The engine knows one sensitive label, ``secrets``. A deployment adds its own
restricted labels in ``governance.restricted_labels``: governance holds a note
that carries one for a person, no machine approves it, and no visibility valve
shows it while it is held. ``authored_acl`` maps the domain of a note a person
writes through remember or create_page to its labels; a domain with no entry
gets ``secrets``.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

import govern
from remembrancerlib import Note, effective_state, load_notes, verification_content_fingerprint, write_note
from remembrancerlib.acl import restricted_labels
from remembrancerlib.locks import pipeline_mutation_lock
from remembrancerlib.verification import authorize_auto_approval, auto_approval_valid, governance_hold_class
from remembrancerlib.verification_audit import append_verification_audit_record
from retrieve import query as Q
from retrieve.verification import (labeled_customer_deal_visible, labeled_decision_visible,
                                   load_verification_audit)
from tests.test_auto_approve import _approve, _config, _held_note, _stored


def test_secrets_is_always_restricted_and_config_only_adds_labels():
    assert restricted_labels(None) == {"secrets"}
    assert restricted_labels({}) == {"secrets"}
    assert restricted_labels({"governance": {"restricted_labels": ["board", " legal "]}}) == {
        "secrets", "board", "legal"}
    assert restricted_labels({"governance": {"restricted_labels": []}}) == {"secrets"}
    # Anything but names in a list adds nothing.
    assert restricted_labels({"governance": {"restricted_labels": "board"}}) == {"secrets"}
    assert restricted_labels({"governance": {"restricted_labels": ["", 7, None, "board"]}}) == {
        "secrets", "board"}
    assert restricted_labels({"governance": "board"}) == {"secrets"}
    assert restricted_labels(["governance"]) == {"secrets"}


def test_governance_holds_the_labels_config_names_and_no_other():
    meta = govern.RouteMeta()
    founders = Note(name="a", description="d", domain="eng", type="state", acl=["founders"], body="b")
    board = Note(name="b", description="d", domain="eng", type="state", acl=["eng", "board"], body="b")
    secret = Note(name="c", description="d", domain="eng", type="state", acl=["secrets"], body="b")
    assert govern.classify(founders, meta, {}) == ("restricted", govern.SAMPLE)
    assert govern.classify(board, meta, {"governance": {"restricted_labels": ["board"]}}) == (
        "secrets-founders", govern.APPROVAL)
    assert govern.classify(secret, meta, {}) == ("secrets-founders", govern.APPROVAL)


def test_a_persons_note_gets_its_domains_labels_from_config():
    assert Q._authored_acl("eng") == ["eng"]
    assert Q._authored_acl("company") == ["secrets"]
    assert Q._authored_acl("company", "", {"authored_acl": {"company": ["staff"]}}) == ["staff"]
    assert Q._authored_acl("eng", "", {"authored_acl": {"eng": ["platform"]}}) == ["platform"]
    assert Q._authored_acl("eng", "", {"authored_acl": {"eng": []}}) == []
    # An entry that is not a list of names is ignored.
    assert Q._authored_acl("company", "", {"authored_acl": {"company": "staff"}}) == ["secrets"]
    assert Q._authored_acl("eng", "", {"authored_acl": {"eng": [1]}}) == ["eng"]
    assert Q._authored_acl("eng", "", {"authored_acl": ["eng"]}) == ["eng"]
    # Sensitive text still lands in secrets, whatever the domain maps to.
    assert Q._authored_acl("company", "the offer letter for Sam", {"authored_acl": {"company": ["staff"]}}) == [
        "secrets"]


def test_the_audit_loads_the_labels_from_config(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"governance": {"restricted_labels": ["board"]}}))
    assert load_verification_audit(str(tmp_path)).restricted_labels == {"secrets", "board"}
    assert append_verification_audit_record({
        "ts": "2026-11-11T10:00:00Z", "audit_id": "audit-one", "note": "one",
        "checked": 1, "supported": 1, "unsupported": 0, "status": "verified",
        "content_fingerprint": "0" * 64, "unsupported_claims": [], "verdicts": [],
    }, str(tmp_path / "state" / "verification.jsonl"))
    loaded = load_verification_audit(str(tmp_path))
    assert loaded.trustworthy and loaded.statuses == {"one": "verified"}
    assert loaded.restricted_labels == {"secrets", "board"}
    (tmp_path / "config.json").write_text("{not json")
    assert load_verification_audit(str(tmp_path)).restricted_labels == {"secrets"}


def _reader_config(root, **governance):
    """A reader who holds ``eng``, under the governance settings given. Unapproved notes answer, as
    in attended mode, so only the governance hold and its valves decide what the reader gets."""
    (root / "config.json").write_text(json.dumps({"identities": {"reader": ["eng"]},
                                                  "trust": {"hide_verification_failures": False},
                                                  "governance": governance}))


def _held(root, note, hold_class):
    """Write ``note`` under a config that restricts none of its labels, so the write door holds it as
    ``hold_class``; the tests restrict a label afterwards."""
    _reader_config(root)
    (root / "state").mkdir(exist_ok=True)
    write_note(note, str(root / "context"))
    [stored] = [n for n in load_notes(str(root / "context")) if n.name == note.name]
    assert governance_hold_class(stored) == hold_class
    return stored


def test_the_customer_deal_valve_refuses_a_configured_label_and_an_audit_without_labels(tmp_path):
    note = _held(tmp_path, Note(name="deal", description="d", domain="customers", type="state", state="proposed",
                                acl=["eng", "board"], provenance=["gmail:m1"], body="Beta access follow-up."),
                 "customer-deal")
    valve = {"customer_deal_visibility": "labeled", "customer_deal_label_verification": "not-disputed"}
    _reader_config(tmp_path, **valve)
    audit = load_verification_audit(str(tmp_path))
    assert labeled_customer_deal_visible(note, audit) is True
    assert labeled_customer_deal_visible(note, dataclasses.replace(audit, restricted_labels=None)) is False
    assert Q.get_note("deal", identity="reader", root=str(tmp_path)) is not None
    _reader_config(tmp_path, restricted_labels=["board"], **valve)
    assert labeled_customer_deal_visible(note, load_verification_audit(str(tmp_path))) is False
    assert Q.get_note("deal", identity="reader", root=str(tmp_path)) is None


def test_the_decision_valve_refuses_a_configured_label_and_an_audit_without_labels(tmp_path):
    note = _held(tmp_path, Note(name="dec", description="d", domain="company", type="decision", state="proposed",
                                acl=["eng", "board"], provenance=["slack:c/1@a"],
                                body="We decided to ship the pilot."),
                 "decision")
    # The decision label index, as a builder writes it for a decision it labels.
    (tmp_path / "state" / "decision_visibility.json").write_text(json.dumps({"version": 1, "notes": {
        "dec": {"class": "decision", "content_fingerprint": verification_content_fingerprint(note)}}}))
    audit = load_verification_audit(str(tmp_path))
    assert labeled_decision_visible(note, audit) is True
    assert labeled_decision_visible(note, dataclasses.replace(audit, restricted_labels=None)) is False
    assert Q.get_note("dec", identity="reader", root=str(tmp_path)) is not None
    _reader_config(tmp_path, restricted_labels=["board"])
    assert labeled_decision_visible(note, load_verification_audit(str(tmp_path))) is False
    assert Q.get_note("dec", identity="reader", root=str(tmp_path)) is None


def test_remember_and_a_new_page_take_their_labels_from_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "identities": {"writer": ["staff"]},
        "authored_acl": {"company": ["staff"]},
        "retrieval": {"file_back": {"min_chars": 20}},
    }))
    assert Q.remember("The on-call pager moves to the platform team on Monday.", identity="writer",
                      root=str(tmp_path))
    assert Q.create_page("On-call rotation", "How the on-call pager rotates, step by step, for everyone.",
                         identity="writer", root=str(tmp_path))
    assert {n.type: n.acl for n in load_notes(str(tmp_path / "context"))} == {
        "synthesis": ["staff"], "page": ["staff"]}
    # A domain that neither config nor the defaults name gets secrets, which this writer lacks.
    assert Q.remember("The board meets on the first Monday of each month.", domain="investors",
                      identity="writer", root=str(tmp_path)) == ""


def test_a_machine_approval_honours_only_an_audit_that_carries_labels(tmp_path):
    _config(tmp_path, gate=True)
    _held_note(tmp_path, name="board-note", acl=("eng", "board"))
    _approve(tmp_path, name="board-note")
    note = _stored(tmp_path, name="board-note")
    audit = load_verification_audit(str(tmp_path))
    assert auto_approval_valid(note, audit) is True
    assert auto_approval_valid(note, dataclasses.replace(audit, restricted_labels=None)) is False
    board = restricted_labels({"governance": {"restricted_labels": ["board"]}})
    assert auto_approval_valid(note, dataclasses.replace(audit, restricted_labels=board)) is False


def test_the_write_door_rechecks_the_labels_config_restricts_when_it_writes(tmp_path):
    _config(tmp_path, gate=True)
    _held_note(tmp_path, name="board-note", acl=("eng", "board"))
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    with pipeline_mutation_lock(str(tmp_path)):
        note = _stored(tmp_path, name="board-note")
        capability = authorize_auto_approval(note, cfg=cfg, audit_reason="checks-passed")
        # The config restricts the label after the grant and before the write.
        _config(tmp_path, gate=True, restricted=["board"])
        note.state = "verified"
        with pytest.raises(PermissionError):
            write_note(note, str(tmp_path / "context"), _auto_capability=capability)
    assert effective_state(_stored(tmp_path, name="board-note")) == "proposed"
    assert Q.get_note("board-note", identity="reviewer", root=str(tmp_path)) is None


def test_founders_is_an_ordinary_label_where_config_does_not_restrict_it(tmp_path):
    _config(tmp_path, gate=True)
    _held_note(tmp_path, name="founders-note", acl=("founders",))
    _approve(tmp_path, name="founders-note")
    assert load_verification_audit(str(tmp_path)).statuses.get("founders-note") == "verified"
    assert Q.get_note("founders-note", identity="reviewer", root=str(tmp_path)) is not None
