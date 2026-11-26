"""The 128 MiB cap was a wall; it is now a valve.

An append that would cross MAX_AUDIT_BYTES used to raise, and
`distill/engine.py` caught that and substituted
`VerificationResult(0, 0, 0, "skipped")`. So a full ledger silently turned every
verdict into "skipped": no note could reach `verified`, auto-approval
(`trust.auto_approve`, which requires exactly that) approved nothing, and the
approval queue grew without bound. Once the free space fell below one typical
record, every append failed.

An audit ledger is evidence, so rotation SEALS history rather than compacting
it, and the reader folds the segments back. These tests exist mostly to prove
that a segment cannot be deleted, reordered or edited without the read failing
closed.

Rotation is the PIPELINE PRE-FLIGHT's job, never the appender's: rotation
shrinks the current segment, and an appender may be running inside a snapshot
transaction whose publisher rejects an append-only log that lost bytes
(append-time rotation inside that stage would make every publish refuse).  The
appender refuses when full; the
pre-flight (`rotate_verification_audit_if_full`) seals at a high-water mark
with 16 MiB of headroom so the wall is never reached inside a run.
"""
import json
import os

import pytest

from remembrancerlib.verification_audit import (
    MAX_AUDIT_BYTES,
    ROTATION_HIGH_WATER_BYTES,
    append_verification_audit_record,
    rotate_verification_audit_if_full,
    encode_segments,
    head_path_for_audit,
    parse_segments,
    rotate_verification_audit,
    segment_audit_name,
    segments_path_for_audit,
)


def _record(note, status="verified", index=0, pad=0):
    row = {
        "note": note, "status": status, "checked": 3, "supported": 3,
        "contradicted": 0, "audit_id": f"{note}-{index}",
        "content_fingerprint": "a" * 64,
    }
    if pad:
        # Real rows carry the per-claim trail and run ~1,360 bytes.
        row["claims"] = ["c" * 40] * pad
    return row


def _encoded_size(rec):
    return len(
        (json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))


def _audit(tmp_path):
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    return str(state / "verification.jsonl")


def _fill_to_cap(path, keep_free):
    """Grow the audit to `keep_free` bytes of headroom, head kept consistent."""
    from remembrancerlib.verification_audit import encode_head, make_head, parse_heads

    filler = {"note": "filler", "status": "flagged", "checked": 1,
              "supported": 0, "contradicted": 1, "audit_id": "filler",
              "content_fingerprint": "b" * 64}
    line = (json.dumps(filler, ensure_ascii=False, sort_keys=True) + "\n").encode()
    target = MAX_AUDIT_BYTES - keep_free
    body = line * (target // len(line))
    with open(path, "wb") as handle:
        handle.write(body)
    os.chmod(path, 0o600)
    head_file = head_path_for_audit(path)
    with open(head_file, "wb") as handle:
        handle.write(encode_head(make_head(body, None)))
    os.chmod(head_file, 0o600)
    heads, ok = parse_heads(open(head_file, "rb").read())
    assert ok and heads
    return MAX_AUDIT_BYTES - len(body)          # the ACTUAL headroom left


def test_a_full_segment_refuses_the_record_rotation_is_the_preflights_job(tmp_path):
    """Append must NEVER rotate: it may be running inside a snapshot
    transaction, and shrinking an append-only log there makes the publisher
    refuse the whole run.  Full segment -> refuse; the pre-flight rotates."""
    path = _audit(tmp_path)
    headroom = _fill_to_cap(path, keep_free=200)
    record = _record("after-the-wall", pad=30)
    assert _encoded_size(record) > headroom, (
        f"the fixture must not fit: {_encoded_size(record)}B into {headroom}B"
    )

    assert append_verification_audit_record(record, path) is False
    assert os.path.getsize(path) >= MAX_AUDIT_BYTES - 300, (
        "a refused append must not shrink or rotate the segment"
    )
    assert not os.path.exists(str(tmp_path / "state" / segment_audit_name(1)))

    # The pre-flight seals the segment; service resumes with history intact.
    assert rotate_verification_audit_if_full(path) == 1
    assert append_verification_audit_record(record, path) is True

    from retrieve.verification import load_verification_audit
    audit = load_verification_audit(str(tmp_path))
    assert audit.trustworthy
    assert audit.statuses.get("after-the-wall") == "verified"
    assert audit.statuses.get("filler") == "flagged", "sealed history must survive"


def test_the_preflight_rotates_only_at_the_high_water_mark(tmp_path):
    path = _audit(tmp_path)
    assert rotate_verification_audit_if_full(path) == 0, "missing file is a no-op"

    assert append_verification_audit_record(_record("small"), path)
    assert rotate_verification_audit_if_full(path) == 0, "small file is a no-op"
    assert not os.path.exists(str(tmp_path / "state" / segment_audit_name(1)))

    # At the mark it seals.  (Tiny explicit mark; the default is cap-16MiB.)
    assert rotate_verification_audit_if_full(path, high_water=1) == 1
    assert (tmp_path / "state" / segment_audit_name(1)).exists()
    assert ROTATION_HIGH_WATER_BYTES < MAX_AUDIT_BYTES


def test_rotation_seals_the_old_segment_and_records_it(tmp_path):
    path = _audit(tmp_path)
    headroom = _fill_to_cap(path, keep_free=200)
    record = _record("x", pad=30)
    assert _encoded_size(record) > headroom
    assert rotate_verification_audit_if_full(path) == 1
    append_verification_audit_record(record, path)

    sealed = tmp_path / "state" / segment_audit_name(1)
    assert sealed.exists() and sealed.stat().st_size > 0
    assert (tmp_path / "state" / "verification.1.head.jsonl").exists()
    assert os.stat(sealed).st_mode & 0o777 == 0o600

    rows, ok = parse_segments(open(segments_path_for_audit(path), "rb").read())
    assert ok and len(rows) == 1
    assert rows[0]["index"] == 1 and rows[0]["name"] == segment_audit_name(1)

    # The new current segment exists and is small.
    assert os.path.exists(path)
    assert os.path.getsize(path) < 10_000


def test_the_reader_folds_segments_newest_wins(tmp_path):
    path = _audit(tmp_path)
    assert append_verification_audit_record(_record("n", "flagged", 1), path)
    rotate_verification_audit(path)
    assert append_verification_audit_record(_record("n", "verified", 2), path)

    from retrieve.verification import load_verification_audit
    audit = load_verification_audit(str(tmp_path))
    assert audit.trustworthy
    assert audit.statuses["n"] == "verified", "the current segment must win"


def test_reading_straight_after_a_rotation_is_valid_not_closed(tmp_path):
    """Between the rename and the next append there must still be a valid,
    empty current segment; an absent one denies every retrieval."""
    path = _audit(tmp_path)
    append_verification_audit_record(_record("kept"), path)
    rotate_verification_audit(path)

    assert os.path.exists(path) and os.path.getsize(path) == 0
    from retrieve.verification import load_verification_audit
    audit = load_verification_audit(str(tmp_path))
    assert audit.trustworthy
    assert audit.statuses["kept"] == "verified"


# --------------------------------------------------------------------------- #
# tamper cases: the point of sealing is that it cannot be quietly undone
# --------------------------------------------------------------------------- #
def _rotated(tmp_path):
    path = _audit(tmp_path)
    append_verification_audit_record(_record("sealed-note", "verified"), path)
    rotate_verification_audit(path)
    append_verification_audit_record(_record("current-note", "verified"), path)
    return path


def _load(tmp_path):
    from retrieve.verification import load_verification_audit
    return load_verification_audit(str(tmp_path))


def test_a_deleted_sealed_segment_fails_closed(tmp_path):
    _rotated(tmp_path)
    assert _load(tmp_path).trustworthy
    os.remove(tmp_path / "state" / segment_audit_name(1))
    assert not _load(tmp_path).trustworthy


def test_an_edited_sealed_segment_fails_closed(tmp_path):
    path = _rotated(tmp_path)
    sealed = tmp_path / "state" / segment_audit_name(1)
    body = sealed.read_bytes().replace(b'"verified"', b'"flagged"')
    sealed.write_bytes(body)
    assert not _load(tmp_path).trustworthy, (
        "a rewritten verdict in sealed history must deny the whole read"
    )


def test_a_truncated_sealed_segment_fails_closed(tmp_path):
    _rotated(tmp_path)
    sealed = tmp_path / "state" / segment_audit_name(1)
    sealed.write_bytes(b"")
    assert not _load(tmp_path).trustworthy


def test_a_manifest_that_drops_a_segment_fails_closed(tmp_path):
    path = _rotated(tmp_path)
    manifest = segments_path_for_audit(path)
    with open(manifest, "wb") as handle:
        handle.write(encode_segments([]))
    # An emptied manifest orphans a segment that still exists on disk.
    assert (tmp_path / "state" / segment_audit_name(1)).exists()
    audit = _load(tmp_path)
    assert audit.statuses.get("sealed-note") is None, (
        "history removed from the manifest must not silently vanish from the read"
    )


def test_a_manifest_with_a_wrong_head_id_fails_closed(tmp_path):
    path = _rotated(tmp_path)
    manifest = segments_path_for_audit(path)
    rows, ok = parse_segments(open(manifest, "rb").read())
    assert ok
    rows[0]["final_head_id"] = "c" * 64
    with open(manifest, "wb") as handle:
        handle.write(encode_segments(rows))
    assert not _load(tmp_path).trustworthy


def test_a_non_contiguous_manifest_is_refused(tmp_path):
    rows = [{"index": 2, "name": segment_audit_name(2), "final_head_id": "d" * 64}]
    parsed, ok = parse_segments(encode_segments(rows))
    assert not ok, "segment indexes must be 1..N in order"


def test_a_corrupt_manifest_is_refused(tmp_path):
    assert parse_segments(b"{not json")[1] is False
    assert parse_segments(b'{"version":99,"segments":[]}')[1] is False
    assert parse_segments(b"")[1] is True, "never rotated is an honest empty state"


# --------------------------------------------------------------------------- #
# rotation refuses rather than destroying
# --------------------------------------------------------------------------- #
def test_rotation_refuses_an_audit_that_does_not_match_its_head(tmp_path):
    path = _audit(tmp_path)
    append_verification_audit_record(_record("a"), path)
    with open(path, "ab") as handle:
        handle.write(b'{"note":"smuggled","status":"verified"}\n')
    with pytest.raises(OSError, match="does not match"):
        rotate_verification_audit(path)
    assert os.path.exists(path), "the unmatched audit must be left in place"


def test_rotation_refuses_to_overwrite_an_existing_segment(tmp_path):
    path = _audit(tmp_path)
    append_verification_audit_record(_record("a"), path)
    # An orphan from a crash between the renames and the manifest write.
    (tmp_path / "state" / segment_audit_name(1)).write_bytes(b"orphan\n")
    with pytest.raises(OSError, match="refusing to overwrite"):
        rotate_verification_audit(path)


def test_rotating_an_empty_audit_is_a_no_op(tmp_path):
    path = _audit(tmp_path)
    open(path, "w").close()
    os.chmod(path, 0o600)
    assert rotate_verification_audit(path) == 0
    assert not (tmp_path / "state" / segment_audit_name(1)).exists()


def test_two_rotations_chain(tmp_path):
    path = _audit(tmp_path)
    append_verification_audit_record(_record("one", "flagged"), path)
    rotate_verification_audit(path)
    append_verification_audit_record(_record("two", "flagged"), path)
    rotate_verification_audit(path)
    append_verification_audit_record(_record("three", "verified"), path)

    rows, ok = parse_segments(open(segments_path_for_audit(path), "rb").read())
    assert ok and [row["index"] for row in rows] == [1, 2]
    audit = _load(tmp_path)
    assert audit.trustworthy
    assert audit.statuses == {"one": "flagged", "two": "flagged",
                              "three": "verified"}
