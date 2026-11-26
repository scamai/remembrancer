from remembrancerlib import Note, active_supersession_targets, current_note_names
from remembrancerlib.verification import verification_content_fingerprint


def _note(name, *, body="", state="verified", acl=None):
    note = Note(name=name, description=name, domain="eng", body=body,
                state=state, acl=acl or [])
    if state == "verified":
        # Projection behavior is the unit under test.  Model the result of
        # bind_loaded_human_trust explicitly so a raw, unaudited state string
        # is not mistaken for governed verification.
        note.approval = {
            "version": 2,
            "approval_id": "a" * 64,
            "content_fingerprint": verification_content_fingerprint(note),
            "identity": "projection-fixture",
            "state": "verified",
            "key_id": "projection-fixture",
            "signature": "projection-fixture",
        }
        note._human_trust_audit_match = True
    return note


def test_duplicate_logical_names_are_excluded_from_current_projection():
    notes = [_note("duplicate"), _note("duplicate"), _note("safe")]
    assert current_note_names(notes) == {"safe"}


def test_parallel_verified_heads_do_not_hide_shared_target():
    notes = [
        _note("target"),
        _note("head-a", body="[[supersedes:target]]"),
        _note("head-b", body="[[supersedes:target]]"),
    ]
    assert "target" not in active_supersession_targets(notes)
    assert current_note_names(notes) == {"target", "head-a", "head-b"}


def test_legacy_supersession_cycle_activates_no_edges():
    notes = [
        _note("a", body="[[supersedes:b]]"),
        _note("b", body="[[supersedes:a]]"),
    ]
    assert active_supersession_targets(notes) == set()
    assert current_note_names(notes) == {"a", "b"}


def test_acl_incompatible_edge_does_not_hide_target():
    notes = [
        _note("target", acl=["eng"]),
        _note("head", body="[[supersedes:target]]", acl=["secrets"]),
    ]
    assert active_supersession_targets(notes) == set()
    assert current_note_names(notes) == {"target", "head"}
