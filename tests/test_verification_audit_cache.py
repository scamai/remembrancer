"""The sealed-segment fold cache must be invisible to every verdict.

`load_verification_audit` used to re-read, re-hash and re-parse every sealed
segment on every call, so search latency grew with the ledger.  The cache reuses a
verified fold only while every segment and head file is byte-for-byte the file
that was verified, and the danger it introduces is a tamper made AFTER the cache
is warm.  So every tamper test here warms the cache first, then tampers, then
reads.  A test that tampered before the first read would pass with no cache at
all and would prove nothing about this one.

`test_dropping_ctime_from_the_key_hides_a_same_size_edit` is the kept mutation:
it shows the same-size-edit test fails when the key loses ctime, so that test
is checking the key rather than passing by construction.
"""
import json
import os
import shutil
import time

import pytest

from remembrancerlib.verification_audit import (
    append_verification_audit_record,
    encode_segments,
    parse_segments,
    rotate_verification_audit,
    segment_audit_name,
)
import retrieve.verification as V
import retrieve.contested as C


def _record(note, status="verified", index=0, legacy=False):
    row = {"note": note, "status": status, "checked": 1, "supported": 1,
           "contradicted": 0}
    if not legacy:
        row["audit_id"] = f"{note}-{index}"
        row["content_fingerprint"] = f"{index:064x}"
    return row


def _audit(tmp_path):
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    return str(state / "verification.jsonl")


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    V.reset_audit_cache()
    C.reset_cache()
    yield
    V.reset_audit_cache()
    C.reset_cache()


@pytest.fixture
def settled(monkeypatch):
    """Treat every file as settled so fresh fixtures are cacheable."""
    monkeypatch.setattr(V, "_SEALED_CACHE_MIN_AGE_NS", -(10 ** 18))
    monkeypatch.setattr(C, "_SEGMENT_MIN_AGE_NS", -(10 ** 18))


def _three_segments(tmp_path):
    """Two sealed segments plus a current one, with later rows overriding."""
    path = _audit(tmp_path)
    append_verification_audit_record(_record("a", "flagged", 1), path)
    append_verification_audit_record(_record("b", "verified", 1), path)
    append_verification_audit_record(_record("c", "verified", 1), path)
    rotate_verification_audit(path)
    append_verification_audit_record(_record("a", "verified", 2), path)
    # A legacy row in a LATER segment does not clear the binding from an earlier
    # one (pre-existing cross-segment update semantics, preserved here) ...
    append_verification_audit_record(_record("b", "flagged", legacy=True), path)
    # ... while inside one segment the latest row's missing binding wins.
    append_verification_audit_record(_record("e", "verified", 2), path)
    append_verification_audit_record(_record("e", "verified", legacy=True), path)
    rotate_verification_audit(path)
    append_verification_audit_record(_record("c", "skipped", 3), path)
    append_verification_audit_record(_record("d", "verified", 3), path)
    return path


def _reference(tmp_path):
    """The pre-cache algorithm: parse every file in manifest order, fold."""
    state = tmp_path / "state"
    rows, ok = parse_segments((state / "verification.segments.json").read_bytes())
    assert ok
    statuses, fingerprints, ids = {}, {}, {}
    for raw in [(state / r["name"]).read_bytes() for r in rows] + [
            (state / "verification.jsonl").read_bytes()]:
        piece = V._parse_verification_audit_bytes(raw)
        assert piece.trustworthy
        statuses.update(piece.statuses)
        fingerprints.update(piece.content_fingerprints)
        ids.update(piece.audit_ids)
    return statuses, fingerprints, ids


def _load(tmp_path):
    return V.load_verification_audit(str(tmp_path))


def _same(audit, reference):
    statuses, fingerprints, ids = reference
    assert dict(audit.statuses) == statuses
    assert dict(audit.content_fingerprints) == fingerprints
    assert dict(audit.audit_ids) == ids


def _count_parses(monkeypatch):
    calls = []
    original = V._parse_verification_audit_bytes

    def counting(raw):
        calls.append(len(raw))
        return original(raw)

    monkeypatch.setattr(V, "_parse_verification_audit_bytes", counting)
    return calls


# ─── equivalence ─────────────────────────────────────────────────────────────
def test_cold_and_warm_reads_equal_the_pre_cache_fold(tmp_path, settled):
    _three_segments(tmp_path)
    reference = _reference(tmp_path)
    assert reference[0] == {"a": "verified", "b": "flagged",
                            "c": "skipped", "d": "verified", "e": "verified"}
    assert "e" not in reference[1], "in-file legacy row must drop the binding"
    assert reference[1]["b"] == f"{1:064x}", "cross-segment binding survives"
    cold = _load(tmp_path)
    warm = _load(tmp_path)
    assert cold.trustworthy and warm.trustworthy
    _same(cold, reference)
    _same(warm, reference)


def test_a_warm_read_parses_no_sealed_segment(tmp_path, settled, monkeypatch):
    _three_segments(tmp_path)
    _load(tmp_path)                                   # warm
    calls = _count_parses(monkeypatch)
    V._CURRENT_PARSE_CACHE.clear()                    # isolate the sealed half
    assert _load(tmp_path).trustworthy
    assert len(calls) == 1, f"only the current ledger may be parsed: {calls}"


def test_an_unchanged_current_ledger_is_not_reparsed(tmp_path, settled, monkeypatch):
    _three_segments(tmp_path)
    _load(tmp_path)
    calls = _count_parses(monkeypatch)
    assert _load(tmp_path).trustworthy
    assert calls == [], "identical current bytes must reuse their parse"


def test_young_files_are_never_cached(tmp_path, monkeypatch):
    """Default min age: a fold of just-written files is not stored, because a
    write within the same coarse timestamp tick could leave ctime unchanged."""
    _three_segments(tmp_path)
    _load(tmp_path)
    assert V._SEALED_CACHE == {}
    calls = _count_parses(monkeypatch)
    V._CURRENT_PARSE_CACHE.clear()
    _load(tmp_path)
    assert len(calls) == 3, "both sealed segments and the current one re-parse"


def test_an_append_to_the_current_ledger_is_visible(tmp_path, settled):
    path = _three_segments(tmp_path)
    assert _load(tmp_path).statuses.get("f") is None
    append_verification_audit_record(_record("f", "flagged", 4), path)
    audit = _load(tmp_path)
    assert audit.trustworthy and audit.statuses["f"] == "flagged"
    _same(audit, _reference(tmp_path))


def test_a_new_rotation_is_folded_in(tmp_path, settled):
    path = _three_segments(tmp_path)
    _load(tmp_path)
    rotate_verification_audit(path)
    append_verification_audit_record(_record("a", "flagged", 9), path)
    audit = _load(tmp_path)
    assert audit.trustworthy and audit.statuses["a"] == "flagged"
    _same(audit, _reference(tmp_path))


def test_callers_cannot_mutate_the_cached_fold(tmp_path, settled):
    """Neither a caller nor the reader's own merge of the current ledger may
    write into the cached fold: a later read would serve the change."""
    _three_segments(tmp_path)
    _load(tmp_path)                                   # cold: fills the cache
    hit = _load(tmp_path)                             # warm: served from it
    assert V._SEALED_CACHE
    hit.statuses["a"] = "skipped"                     # a misbehaving caller
    hit.statuses["zz-injected"] = "verified"
    again = _load(tmp_path)
    assert again.statuses["a"] == "verified"
    assert "zz-injected" not in again.statuses
    cached_statuses = next(iter(V._SEALED_CACHE.values()))[1][0]
    assert "d" not in cached_statuses, (
        "the current ledger's rows must not be merged into the cached fold")


# ─── tampering after the cache is warm ───────────────────────────────────────
def _warm(tmp_path):
    _three_segments(tmp_path)
    assert _load(tmp_path).trustworthy
    assert V._SEALED_CACHE, "fixture must actually warm the cache"
    # Coarse kernel timestamps tick every 1-10 ms; step past one so a tamper
    # below cannot land in the same tick as the fixture's last write.
    time.sleep(0.05)
    return tmp_path / "state"


def _same_size_edit(sealed):
    """Flip one verdict without changing size, then put mtime back."""
    before = os.stat(sealed)
    body = sealed.read_bytes()
    assert b'"flagged"' in body
    edited = body.replace(b'"flagged"', b'"skipped"', 1)
    assert len(edited) == len(body)
    sealed.write_bytes(edited)
    os.utime(sealed, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_a_same_size_edit_with_mtime_restored_fails_closed(tmp_path, settled):
    state = _warm(tmp_path)
    _same_size_edit(state / segment_audit_name(1))
    assert not _load(tmp_path).trustworthy


def test_dropping_ctime_from_the_key_hides_a_same_size_edit(
        tmp_path, settled, monkeypatch):
    """Kept mutation for the test above: without ctime the cache serves the
    old fold and the edit goes unseen.  If this ever starts failing, the
    same-size-edit test no longer proves the key needs ctime."""
    def without_ctime(m):
        return (m.st_dev, m.st_ino, m.st_size, m.st_mtime_ns, m.st_mode)

    monkeypatch.setattr(V, "_stat_key", without_ctime)
    V.reset_audit_cache()
    state = _warm(tmp_path)
    _same_size_edit(state / segment_audit_name(1))
    assert _load(tmp_path).trustworthy, "mutation must hide the tamper"


@pytest.mark.parametrize("name", [segment_audit_name(1), "verification.1.head.jsonl"])
def test_a_chmod_fails_closed(tmp_path, settled, name):
    state = _warm(tmp_path)
    os.chmod(state / name, 0o644)
    assert not _load(tmp_path).trustworthy


def test_a_deleted_segment_fails_closed(tmp_path, settled):
    state = _warm(tmp_path)
    os.remove(state / segment_audit_name(2))
    assert not _load(tmp_path).trustworthy


def test_a_segment_replaced_by_a_copy_of_itself_is_rechecked(tmp_path, settled,
                                                              monkeypatch):
    """A new inode is a miss even with identical bytes; the full check runs and
    passes, so the verdicts are unchanged."""
    state = _warm(tmp_path)
    sealed = state / segment_audit_name(1)
    copy = state / "copy.tmp"
    shutil.copy2(sealed, copy)
    os.chmod(copy, 0o600)
    os.replace(copy, sealed)
    reference = _reference(tmp_path)
    calls = _count_parses(monkeypatch)
    V._CURRENT_PARSE_CACHE.clear()
    audit = _load(tmp_path)
    assert audit.trustworthy
    assert len(calls) == 3, "a replaced segment must be re-read and re-verified"
    _same(audit, reference)


def test_swapped_segments_fail_closed(tmp_path, settled):
    state = _warm(tmp_path)
    one, two = state / segment_audit_name(1), state / segment_audit_name(2)
    tmp = state / "swap.tmp"
    os.replace(one, tmp)
    os.replace(two, one)
    os.replace(tmp, two)
    assert not _load(tmp_path).trustworthy


def test_a_symlinked_segment_fails_closed(tmp_path, settled):
    state = _warm(tmp_path)
    sealed = state / segment_audit_name(1)
    target = tmp_path / "elsewhere.jsonl"
    shutil.copy2(sealed, target)
    os.remove(sealed)
    os.symlink(target, sealed)
    assert not _load(tmp_path).trustworthy


def test_an_edited_head_file_fails_closed(tmp_path, settled):
    state = _warm(tmp_path)
    head = state / "verification.1.head.jsonl"
    head.write_bytes(head.read_bytes() + b"\n")
    os.chmod(head, 0o600)
    assert not _load(tmp_path).trustworthy


def test_a_manifest_with_a_wrong_head_id_fails_closed(tmp_path, settled):
    state = _warm(tmp_path)
    manifest = state / "verification.segments.json"
    rows, ok = parse_segments(manifest.read_bytes())
    assert ok
    rows[0]["final_head_id"] = "c" * 64
    manifest.write_bytes(encode_segments(rows))
    assert not _load(tmp_path).trustworthy


def test_a_manifest_that_drops_a_segment_hides_its_history(tmp_path, settled):
    state = _warm(tmp_path)
    manifest = state / "verification.segments.json"
    rows, ok = parse_segments(manifest.read_bytes())
    manifest.write_bytes(encode_segments(rows[1:]))
    audit = _load(tmp_path)
    # Dropping row 1 breaks the 1..N contiguity rule, so the read fails closed.
    assert not audit.trustworthy


def test_an_in_place_edit_of_the_current_ledger_fails_closed(tmp_path, settled):
    """The current ledger's parse is cached by content, but its head check runs
    on every load."""
    path = _three_segments(tmp_path)
    assert _load(tmp_path).trustworthy
    body = open(path, "rb").read().replace(b'"skipped"', b'"flagged"', 1)
    with open(path, "wb") as handle:
        handle.write(body)
    os.chmod(path, 0o600)
    assert not _load(tmp_path).trustworthy


# ─── contested index ─────────────────────────────────────────────────────────
def _contested_row(note, flagged=True, index=0):
    row = _record(note, "flagged" if flagged else "verified", index)
    row["verdicts"] = ([{"verdict": "contradicted", "claim": f"claim {index}",
                         "evidence_quote": "q", "event_ids": ["e1"]}]
                       if flagged else [])
    return row


def test_contested_rescans_only_the_current_ledger_after_an_append(
        tmp_path, settled, monkeypatch):
    path = _audit(tmp_path)
    append_verification_audit_record(_contested_row("x", True, 1), path)
    append_verification_audit_record(_contested_row("y", True, 1), path)
    rotate_verification_audit(path)
    append_verification_audit_record(_contested_row("z", True, 2), path)
    root = str(tmp_path)
    assert set(C.contested_index(root)) == {"x", "y", "z"}

    scanned = []
    original = C._scan_one
    monkeypatch.setattr(C, "_scan_one",
                        lambda p: (scanned.append(os.path.basename(p)), original(p))[1])
    # A clean row for x in the current ledger erases the sealed finding.
    append_verification_audit_record(_contested_row("x", False, 3), path)
    index = C.contested_index(root)
    assert set(index) == {"y", "z"}
    assert scanned == ["verification.jsonl"], scanned
    assert index == C._scan(C.ledger_paths(root)), "cached fold == full scan"


# ─── one audit load per read call ────────────────────────────────────────────
def _corpus(tmp_path):
    from remembrancerlib import Note, regen_index, write_note
    from remembrancerlib.verification_audit import encode_head, make_head

    (tmp_path / "config.json").write_text(json.dumps({
        "identities": {"reviewer": ["eng"]},
        "retrieval": {"query_log": False},
        "trust": {"proposed_rank_penalty": 0.5},
    }), encoding="utf-8")
    for index in range(4):
        write_note(Note(name=f"note-{index}", description="loadcount gateword",
                        domain="eng", type="state", state="proposed",
                        body="answer body"), str(tmp_path / "context"))
    path = _audit(tmp_path)
    raw = "".join(json.dumps({"note": f"note-{i}", "status": "verified"}) + "\n"
                  for i in range(4)).encode()
    open(path, "wb").write(raw)
    os.chmod(path, 0o600)
    head = tmp_path / "state" / "verification.head.jsonl"
    head.write_bytes(encode_head(make_head(raw, None)))
    os.chmod(head, 0o600)
    regen_index(str(tmp_path / "context"))
    return str(tmp_path)


def _count_loads(monkeypatch):
    calls = []
    original = V._load_sealed_fold

    def counting(descriptor):
        calls.append(descriptor)
        return original(descriptor)

    monkeypatch.setattr(V, "_load_sealed_fold", counting)
    return calls


@pytest.mark.parametrize("call", ["search", "search_full_scan", "get_note"])
def test_each_read_call_loads_the_audit_once(tmp_path, monkeypatch, call):
    from retrieve import query as Q

    root = _corpus(tmp_path)
    assert Q.search("gateword", root=root), "fixture must be searchable"
    calls = _count_loads(monkeypatch)
    if call == "get_note":
        assert Q.get_note("note-1", root=root) is not None
    else:
        assert getattr(Q, call)("gateword", root=root)
    assert len(calls) == 1, f"{call} loaded the audit {len(calls)} times"


# ─── single-flight, prefix fold, failed reads, warm-up ──────────────────────
def test_a_rotation_folds_only_the_new_segment(tmp_path, settled, monkeypatch):
    path = _three_segments(tmp_path)
    _load(tmp_path)                                   # cache holds 2 sealed
    rotate_verification_audit(path)                   # seals segment 3
    append_verification_audit_record(_record("a", "flagged", 9), path)
    reference = _reference(tmp_path)
    calls = _count_parses(monkeypatch)
    V._CURRENT_PARSE_CACHE.clear()
    audit = _load(tmp_path)
    assert audit.trustworthy
    assert len(calls) == 2, f"only segment 3 and the current ledger: {calls}"
    _same(audit, reference)


def test_concurrent_misses_read_the_segments_once(tmp_path, settled, monkeypatch):
    import threading
    _three_segments(tmp_path)
    seen = []
    original = V._verify_segment
    gate = threading.Event()

    def slow(descriptor, row):
        seen.append(row["name"])
        gate.wait(1)
        return original(descriptor, row)

    monkeypatch.setattr(V, "_verify_segment", slow)
    results = []
    threads = [threading.Thread(target=lambda: results.append(_load(tmp_path)))
               for _ in range(4)]
    for thread in threads:
        thread.start()
    time.sleep(0.1)
    gate.set()
    for thread in threads:
        thread.join()
    assert all(r.trustworthy for r in results) and len(results) == 4
    assert len(seen) == 2, f"each sealed segment must be read once: {seen}"


def test_a_load_holds_at_most_one_segment_pair_open(tmp_path, settled, monkeypatch):
    _three_segments(tmp_path)
    open_now, peak = [0], [0]
    real_open, real_close = V._open_pair, V._close_pair

    def counting_open(descriptor, row):
        got = real_open(descriptor, row)
        if got is not None:
            open_now[0] += 1
            peak[0] = max(peak[0], open_now[0])
        return got

    def counting_close(descriptors):
        open_now[0] -= 1
        return real_close(descriptors)

    monkeypatch.setattr(V, "_open_pair", counting_open)
    monkeypatch.setattr(V, "_close_pair", counting_close)
    assert _load(tmp_path).trustworthy               # cold: pass 1 + pass 2
    assert _load(tmp_path).trustworthy               # warm: pass 1
    assert peak[0] == 1 and open_now[0] == 0


def test_contested_does_not_cache_a_failed_segment_read(tmp_path, settled, monkeypatch):
    path = _audit(tmp_path)
    append_verification_audit_record(_contested_row("x", True, 1), path)
    rotate_verification_audit(path)
    append_verification_audit_record(_contested_row("y", True, 2), path)
    root = str(tmp_path)
    original = C._scan_one
    failed = []

    def flaky(p):
        if os.path.basename(p) == segment_audit_name(1) and not failed:
            failed.append(p)
            return None
        return original(p)

    monkeypatch.setattr(C, "_scan_one", flaky)
    assert set(C.contested_index(root)) == {"y"}, "x's segment failed once"
    # No ledger change: an incomplete index must not have been cached.
    assert set(C.contested_index(root)) == {"x", "y"}
    append_verification_audit_record(_contested_row("z", True, 3), path)
    assert set(C.contested_index(root)) == {"x", "y", "z"}, "the retry must recover x"


def test_warm_audit_caches_fills_the_cache(tmp_path, settled, monkeypatch):
    _three_segments(tmp_path)
    monkeypatch.setenv("REMEMBRANCER_WARM_AUDIT", "1")
    thread = V.warm_audit_caches(str(tmp_path))
    assert thread is not None
    thread.join(10)
    assert V._SEALED_CACHE and C._CACHE
    monkeypatch.setenv("REMEMBRANCER_WARM_AUDIT", "0")
    assert V.warm_audit_caches(str(tmp_path)) is None
