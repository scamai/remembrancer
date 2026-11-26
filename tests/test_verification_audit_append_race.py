"""An append that waited on a rotation's lock must not write into the sealed pair.

`_append_once` opens the current audit and head by name and then blocks on
LOCK_EX.  If `rotate_verification_audit` holds those locks meanwhile, it renames
the pair into sealed segment N and leaves a fresh empty current pair.  The
appender's descriptors then point at the SEALED files; appending there moves the
sealed head past the manifest's final_head_id, and every reader fails closed.
These tests run the rotation inside that window, deterministically, by hooking
the appender's first LOCK_EX.
"""
import fcntl
import os


import remembrancerlib.verification_audit as VA
from remembrancerlib.verification_audit import (
    append_verification_audit_record,
    rotate_verification_audit,
    segment_audit_name,
)


def _record(note, status="verified", index=0):
    return {"note": note, "status": status, "checked": 1, "supported": 1,
            "contradicted": 0, "audit_id": f"{note}-{index}",
            "content_fingerprint": f"{index:064x}"}


def _rotate_inside_the_first_lock_wait(monkeypatch, path):
    """Run one rotation the first time the appender takes LOCK_EX on its audit."""
    real_flock = fcntl.flock
    state = {"rotated": False}

    def flock(descriptor, operation):
        if (operation == fcntl.LOCK_EX and not state["rotated"]
                and os.fstat(descriptor).st_ino == os.stat(path).st_ino):
            state["rotated"] = True
            assert rotate_verification_audit(path) == 1
        return real_flock(descriptor, operation)

    monkeypatch.setattr(VA.fcntl, "flock", flock)
    return state


def test_an_append_racing_a_rotation_lands_in_the_new_current_segment(
        tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    path = str(state_dir / "verification.jsonl")
    assert append_verification_audit_record(_record("sealed", index=1), path)
    raced = _rotate_inside_the_first_lock_wait(monkeypatch, path)

    assert append_verification_audit_record(_record("raced", index=2), path)
    assert raced["rotated"], "the hook must actually have rotated mid-append"
    monkeypatch.undo()

    sealed = (state_dir / segment_audit_name(1)).read_bytes()
    assert b'"raced"' not in sealed, "the record was written into the sealed segment"
    assert b'"raced"' in (state_dir / "verification.jsonl").read_bytes()

    from retrieve.verification import load_verification_audit, reset_audit_cache
    reset_audit_cache()
    audit = load_verification_audit(str(tmp_path))
    assert audit.trustworthy
    assert audit.statuses == {"sealed": "verified", "raced": "verified"}


def test_without_the_recheck_the_race_breaks_every_read(tmp_path, monkeypatch):
    """Kept mutation: with the name check disabled, the same race writes into
    the sealed pair and the reader fails closed.  If this ever passes as
    trustworthy, the test above no longer proves the recheck matters."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    path = str(state_dir / "verification.jsonl")
    assert append_verification_audit_record(_record("sealed", index=1), path)
    monkeypatch.setattr(VA, "_names_still_point_at", lambda *_a: True)
    _rotate_inside_the_first_lock_wait(monkeypatch, path)
    append_verification_audit_record(_record("raced", index=2), path)
    monkeypatch.undo()

    from retrieve.verification import load_verification_audit, reset_audit_cache
    reset_audit_cache()
    assert not load_verification_audit(str(tmp_path)).trustworthy


def test_a_ledger_that_never_settles_refuses_instead_of_writing(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    path = str(state_dir / "verification.jsonl")
    assert append_verification_audit_record(_record("first", index=1), path)
    monkeypatch.setattr(VA, "_names_still_point_at", lambda *_a: False)
    before = (state_dir / "verification.jsonl").read_bytes()
    assert append_verification_audit_record(_record("never", index=2), path) is False
    assert (state_dir / "verification.jsonl").read_bytes() == before
