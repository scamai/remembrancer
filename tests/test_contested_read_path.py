"""The contested label reaches the two retrieval row builders.

The digest ledger's ``contested`` state and this label are the same fact seen
from two ends, so they share one switch (``digest.review_policy: "settle"``).
With the key absent neither retrieval path reads the verification ledger at all
and every row is byte-identical to the pre-policy contract — that is the
rollback, and it is asserted here rather than assumed.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from remembrancerlib import Note, write_note
from retrieve import contested as contested_mod
from retrieve.query import search, search_full_scan

import mcp.server as srv


def _row(note, status, verdicts):
    return json.dumps({
        "ts": "2026-09-04T00:00:00Z", "note": note, "status": status,
        "checked": len(verdicts), "supported": 0, "unsupported": len(verdicts),
        "model": "test-model", "verdicts": verdicts,
    }, sort_keys=True)


_CONTRADICTED = {
    "claim": "the pilot starts in June",
    "verdict": "contradicted",
    "evidence_quote": "the pilot was pushed to September",
    "event_ids": ["gmail:abc"],
}
_UNSUPPORTED = {
    "claim": "the pilot has a champion",
    "verdict": "unsupported",
    "evidence_quote": "",
    "event_ids": [],
}


@pytest.fixture
def corpus(request):
    """Two notes — one contradicted, one merely unsupported — and a config."""
    policy = getattr(request, "param", "settle")
    with tempfile.TemporaryDirectory() as tmp:
        context_dir = os.path.join(tmp, "context")
        os.makedirs(context_dir)
        os.makedirs(os.path.join(tmp, "state"))
        # Setting under test: the verification gate
        # hides nothing, so a flagged note answers and the label is what tells
        # the reader about the dispute.
        config = {"identities": {"reviewer": []},
                  "trust": {"hide_verification_failures": False}}
        if policy is not None:
            config["digest"] = {"review_policy": policy}
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(config, fh)
        for name, description in (
            ("disputed-pilot", "Acme pilot onboarding schedule"),
            ("quiet-pilot", "Acme pilot onboarding staffing"),
        ):
            write_note(Note(name=name, description=description, domain="customers",
                            acl=[], body="Body."), context_dir)
        with open(os.path.join(tmp, "state", "verification.jsonl"), "w",
                  encoding="utf-8") as fh:
            fh.write(_row("disputed-pilot", "flagged", [_CONTRADICTED]) + "\n")
            fh.write(_row("quiet-pilot", "flagged", [_UNSUPPORTED]) + "\n")
        os.chmod(os.path.join(tmp, "state", "verification.jsonl"), 0o600)
        contested_mod.reset_cache()
        yield tmp
        contested_mod.reset_cache()


def _by_name(rows):
    return {row["name"]: row for row in rows}


# ─── retrieve/query.py row assembly ──────────────────────────────────────────

def test_search_rows_carry_the_contradicted_claim_and_its_evidence(corpus):
    rows = _by_name(search("onboarding", root=corpus))
    assert rows["disputed-pilot"]["contested"] == [{
        "claim": "the pilot starts in June",
        "evidence_quote": "the pilot was pushed to September",
        "event_ids": ["gmail:abc"],
    }]
    # Flagged-but-not-contradicted is the common case and gets no label.
    assert "contested" not in rows["quiet-pilot"]


def test_full_scan_recovery_path_labels_identically(corpus):
    rows = _by_name(search_full_scan("onboarding", root=corpus))
    assert rows["disputed-pilot"]["contested"][0]["claim"] == "the pilot starts in June"
    assert "contested" not in rows["quiet-pilot"]


def test_the_note_file_is_never_rewritten_for_the_label(corpus):
    path = os.path.join(corpus, "context", "customers", "disputed-pilot.md")
    before = open(path, "rb").read()
    search("onboarding", root=corpus)
    assert open(path, "rb").read() == before


@pytest.mark.parametrize("corpus", [None, "hold"], indirect=True)
def test_absent_policy_produces_no_label_and_reads_no_ledger(corpus, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the verification ledger must not be scanned")

    monkeypatch.setattr(contested_mod, "_scan", explode)
    rows = _by_name(search("onboarding", root=corpus))
    assert rows and all("contested" not in row for row in rows.values())


# ─── mcp/server.py _note_to_summary ──────────────────────────────────────────

def test_note_to_summary_adds_the_label_when_the_policy_is_on(corpus):
    root = srv._contested_root(corpus)
    assert root == os.path.abspath(corpus)
    row = srv._note_to_summary({"name": "disputed-pilot"}, root)
    assert row["contested"][0]["evidence_quote"] == "the pilot was pushed to September"
    assert "contested" not in srv._note_to_summary({"name": "quiet-pilot"}, root)


@pytest.mark.parametrize("corpus", [None, "hold"], indirect=True)
def test_note_to_summary_is_unchanged_without_the_policy(corpus):
    assert srv._contested_root(corpus) is None
    assert "contested" not in srv._note_to_summary({"name": "disputed-pilot"}, None)
    assert "contested" not in srv._note_to_summary({"name": "disputed-pilot"})


def test_note_to_summary_keeps_a_label_the_retrieval_row_already_computed():
    """A retrieve/query row arrives with the identical key; don't recompute it."""
    row = srv._note_to_summary(
        {"name": "n", "contested": [{"claim": "c", "evidence_quote": "q",
                                     "event_ids": []}]})
    assert row["contested"] == [{"claim": "c", "evidence_quote": "q",
                                 "event_ids": []}]
