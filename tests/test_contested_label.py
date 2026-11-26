"""The read-time ``contested`` label.

A note whose latest verification row CONTRADICTS one of its claims is the one
case where the reader genuinely needs to know something: not "this is hidden",
not "a person owes a decision", but "the source says the opposite, here is the
quote".  So the dispute is served WITH its evidence, sourced from the
verification ledger, and the note file is never rewritten to carry it.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieve import contested as contested_mod  # noqa: E402


def _row(note, status, verdicts, ts="2026-09-04T00:00:00Z"):
    return json.dumps({
        "ts": ts, "note": note, "status": status, "checked": len(verdicts),
        "supported": 0, "unsupported": len(verdicts), "model": "test-model",
        "verdicts": verdicts,
    }, sort_keys=True)


def _contradicted(claim="the deal closed in June", quote="we signed in August",
                  event_ids=("e2",), **extra):
    return {"claim": claim, "verdict": "contradicted", "evidence_quote": quote,
            "event_ids": list(event_ids), "severity": "high", **extra}


def _corpus(tmp_path, rows, *, segments=None):
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    (state / "verification.jsonl").write_text("\n".join(rows) + "\n",
                                              encoding="utf-8")
    if segments:
        for name, seg_rows in segments.items():
            (state / name).write_text("\n".join(seg_rows) + "\n", encoding="utf-8")
        (state / "verification.segments.json").write_text(json.dumps({
            "version": 1,
            "segments": [{"index": i, "name": name}
                         for i, name in enumerate(segments, start=1)],
        }), encoding="utf-8")
    return str(tmp_path)


@pytest.fixture(autouse=True)
def _clean_cache():
    contested_mod.reset_cache()
    yield
    contested_mod.reset_cache()


def test_flagged_row_with_a_contradiction_yields_claim_quote_and_events(tmp_path):
    root = _corpus(tmp_path, [_row("n", "flagged", [_contradicted()])])
    assert contested_mod.contested_for(root, "n") == [{
        "claim": "the deal closed in June",
        "evidence_quote": "we signed in August",
        "event_ids": ["e2"],
    }]
    assert contested_mod.is_contested(root, "n")


def test_unsupported_is_not_contested(tmp_path):
    """The common case: the verifier found no support, which is not a dispute."""
    root = _corpus(tmp_path, [_row("n", "flagged", [
        {"claim": "c", "verdict": "unsupported", "evidence_quote": "q",
         "event_ids": ["e1"]}])])
    assert contested_mod.contested_for(root, "n") == []


def test_a_validation_error_is_a_parser_failure_not_a_dispute(tmp_path):
    root = _corpus(tmp_path, [_row("n", "flagged", [
        _contradicted(validation_error="quote not found in source")])])
    assert contested_mod.contested_for(root, "n") == []


def test_only_the_latest_row_counts(tmp_path):
    """Re-verified clean this morning means not contested, whatever last month said."""
    root = _corpus(tmp_path, [
        _row("n", "flagged", [_contradicted()], ts="2026-08-01T00:00:00Z"),
        _row("n", "verified", [], ts="2026-09-01T00:00:00Z"),
    ])
    assert contested_mod.contested_for(root, "n") == []


def test_a_later_segment_row_supersedes_a_sealed_one(tmp_path):
    root = _corpus(
        tmp_path,
        [_row("n", "flagged", [_contradicted(claim="new claim")])],
        segments={"verification.1.jsonl": [
            _row("n", "flagged", [_contradicted(claim="old claim")])]},
    )
    assert [c["claim"] for c in contested_mod.contested_for(root, "n")] == ["new claim"]


def test_a_sealed_segment_alone_still_produces_a_label(tmp_path):
    root = _corpus(
        tmp_path,
        [_row("other", "verified", [])],
        segments={"verification.1.jsonl": [
            _row("n", "flagged", [_contradicted()])]},
    )
    assert contested_mod.is_contested(root, "n")


def test_at_most_three_claims_are_carried(tmp_path):
    root = _corpus(tmp_path, [_row("n", "flagged", [
        _contradicted(claim=f"claim {i}") for i in range(9)])])
    claims = contested_mod.contested_for(root, "n")
    assert len(claims) == contested_mod.MAX_CONTESTED == 3
    assert [c["claim"] for c in claims] == ["claim 0", "claim 1", "claim 2"]


def test_missing_or_damaged_ledger_yields_no_label(tmp_path):
    assert contested_mod.contested_for(str(tmp_path), "n") == []
    state = tmp_path / "state"
    state.mkdir()
    (state / "verification.jsonl").write_text("{not json\n", encoding="utf-8")
    contested_mod.reset_cache()
    assert contested_mod.contested_for(str(tmp_path), "n") == []


def test_index_is_memoised_until_the_ledger_changes(tmp_path):
    root = _corpus(tmp_path, [_row("n", "flagged", [_contradicted()])])
    calls = []
    original = contested_mod._scan_one

    def counting(path):
        calls.append(path)
        return original(path)

    # One root, one ledger file: each count is one file scan.
    contested_mod._scan_one = counting
    try:
        contested_mod.contested_index(root)
        contested_mod.contested_index(root)
        assert len(calls) == 1
        # An append invalidates it: the signature carries size and mtime.
        with open(os.path.join(root, "state", "verification.jsonl"), "a") as fh:
            fh.write(_row("m", "flagged", [_contradicted()]) + "\n")
        os.utime(os.path.join(root, "state", "verification.jsonl"), (0, 0))
        assert contested_mod.is_contested(root, "m")
        assert len(calls) == 2
    finally:
        contested_mod._scan_one = original


# ─── the bounded tail read the pipeline uses ─────────────────────────────────

def test_tail_read_finds_the_row_the_verifier_just_appended(tmp_path):
    root = _corpus(tmp_path, [
        _row("n", "verified", []),
        _row("n", "flagged", [_contradicted()]),
    ])
    assert contested_mod.tail_contested(root, "n")


def test_tail_read_returns_none_when_the_tail_holds_no_row_for_the_note(tmp_path):
    root = _corpus(tmp_path, [_row("other", "flagged", [_contradicted()])])
    assert contested_mod.tail_contested(root, "n") is None


def test_tail_read_returns_empty_for_a_non_flagged_latest_row(tmp_path):
    root = _corpus(tmp_path, [
        _row("n", "flagged", [_contradicted()]),
        _row("n", "verified", []),
    ])
    assert contested_mod.tail_contested(root, "n") == []


def test_tail_read_drops_the_partial_first_line(tmp_path):
    root = _corpus(tmp_path, [
        _row("n", "flagged", [_contradicted()]),
        _row("other", "verified", []),
    ])
    # A window small enough to slice the first record in half must not resurrect it.
    assert contested_mod.tail_contested(root, "n", tail_bytes=120) is None


# ─── the config switch ───────────────────────────────────────────────────────

def test_policy_switch_reads_the_digest_block():
    assert contested_mod.review_policy_settles({"digest": {"review_policy": "settle"}})
    for cfg in ({}, None, {"digest": {}}, {"digest": {"review_policy": "hold"}},
                {"digest": []}, "nope"):
        assert not contested_mod.review_policy_settles(cfg)
