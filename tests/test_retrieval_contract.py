"""Operational retrieval contract: index-first, bounded text, ACL, and ordering."""
from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path

import pytest

from remembrancerlib import Note, regen_index, write_note
from retrieve import query as Q
from retrieve.index import (_decode_index_metadata, encode_index_metadata,
                            load_index_snapshot)


def _write(root: Path, **overrides) -> Path:
    values = {
        "name": "note-a",
        "description": "shared retrieval phrase",
        "domain": "eng",
        "body": "body",
    }
    values.update(overrides)
    note = Note(**values)
    return Path(write_note(note, str(root / "context")))


def _config(root: Path, retrieval=None, identities=None) -> None:
    (root / "config.json").write_text(json.dumps({
        "identities": identities or {"reviewer": []},
        "retrieval": retrieval or {},
        "trust": {"approver_identities": ["reviewer"]},
    }))


def _review(root: Path, name: str, state: str) -> None:
    transition = Q.promote if state == "verified" else Q.reject
    result = transition(name, identity="reviewer", root=str(root))
    assert result["ok"], result


def _rewrite_index_metadata(index_path: Path, mutate) -> None:
    """Rewrite one row and its unkeyed generation digest for tamper tests."""
    lines = index_path.read_text().splitlines()
    for index, line in enumerate(lines):
        if "<!-- remembrancer-index-v2:" not in line:
            continue
        metadata = _decode_index_metadata(line)
        assert metadata is not None
        mutate(metadata)
        prefix = line[:line.index("<!-- remembrancer-index-v2:")]
        lines[index] = prefix + encode_index_metadata(metadata)
        break
    tokens = [
        line.split("<!-- remembrancer-index-v2:", 1)[1].split(" -->", 1)[0]
        for line in lines if "<!-- remembrancer-index-v2:" in line
    ]
    digest = hashlib.sha256("\n".join(tokens).encode("ascii")).hexdigest()
    for index, line in enumerate(lines):
        if line.startswith("<!-- remembrancer-index-v2-digest:"):
            lines[index] = f"<!-- remembrancer-index-v2-digest:{digest} -->"
            break
    index_path.write_text("\n".join(lines) + "\n")


def test_fresh_domain_indexes_discover_candidates_without_full_parse(tmp_path, monkeypatch):
    match_path = _write(tmp_path, name="matching-note", description="quasar launch decision")
    miss_path = _write(tmp_path, name="unrelated-note", description="ordinary status update")
    regen_index(str(tmp_path / "context"))

    opened_notes: list[str] = []
    real_open = builtins.open

    def tracking_open(file, *args, **kwargs):
        path = os.fspath(file)
        if path.endswith(".md") and os.path.basename(path) not in {"index.md", "INDEX.md"}:
            opened_notes.append(os.path.abspath(path))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(Q, "load_notes", lambda *_: (_ for _ in ()).throw(
        AssertionError("fresh v2 indexes must not invoke a full corpus scan")))
    monkeypatch.setattr(builtins, "open", tracking_open)

    hits = Q.search("quasar", root=str(tmp_path))
    assert [hit["name"] for hit in hits] == ["matching-note"]
    assert os.path.abspath(match_path) in opened_notes
    # Snapshot validation hashes every note without parsing it. Only matching
    # candidates are loaded as Notes; load_notes remains forbidden above.
    assert os.path.abspath(miss_path) in opened_notes


def test_committed_index_is_portable_across_checkout_timestamp_changes(tmp_path):
    note_path = _write(tmp_path, name="portable-note", description="portable catalog")
    regen_index(str(tmp_path / "context"))
    index_path = tmp_path / "context" / "eng" / "index.md"
    rendered = index_path.read_bytes()

    # Simulate Git materializing the exact bytes in a different checkout.
    os.utime(note_path, ns=(1_700_000_000_000_000_000,) * 2)
    snapshot = load_index_snapshot(str(tmp_path / "context"))
    assert snapshot is not None
    assert [record.name for record in snapshot.records] == ["portable-note"]

    regen_index(str(tmp_path / "context"))
    assert index_path.read_bytes() == rendered
    metadata = next(
        _decode_index_metadata(line)
        for line in index_path.read_text().splitlines()
        if "<!-- remembrancer-index-v2:" in line
    )
    assert metadata is not None
    assert "mtime_ns" not in metadata
    assert "ctime_ns" not in metadata


def test_same_size_note_rewrite_invalidates_index_even_with_restored_mtime(tmp_path):
    note_path = _write(tmp_path, name="digest-note", description="alpha token")
    regen_index(str(tmp_path / "context"))
    original_stat = note_path.stat()
    original = note_path.read_bytes()
    changed = original.replace(b"alpha token", b"omega token")
    assert len(changed) == len(original)
    note_path.write_bytes(changed)
    os.utime(note_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert load_index_snapshot(str(tmp_path / "context")) is None


def test_same_size_acl_rewrite_falls_back_without_disclosure(tmp_path):
    _config(tmp_path, identities={"eng-user": ["eng"]})
    note_path = _write(
        tmp_path, name="acl-digest-note", description="aclhashword", acl=["eng"]
    )
    regen_index(str(tmp_path / "context"))
    original_stat = note_path.stat()
    original = note_path.read_bytes()
    changed = original.replace(b"acl: [eng]", b"acl: [ops]")
    assert len(changed) == len(original)
    note_path.write_bytes(changed)
    os.utime(note_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert Q.search("aclhashword", identity="eng-user", root=str(tmp_path)) == []


def test_self_consistent_acl_index_tamper_is_blocked_by_authoritative_reload(
        tmp_path, monkeypatch):
    _config(tmp_path, identities={"eng-user": ["eng"]})
    _write(
        tmp_path, name="private-note", description="aclforgeryword", acl=["secrets"]
    )
    regen_index(str(tmp_path / "context"))
    index_path = tmp_path / "context" / "eng" / "index.md"
    _rewrite_index_metadata(index_path, lambda metadata: metadata.update(acl=[]))

    calls = 0
    real_load_notes = Q.load_notes

    def counted_load_notes(path, **kwargs):
        nonlocal calls
        calls += 1
        return real_load_notes(path, **kwargs)

    monkeypatch.setattr(Q, "load_notes", counted_load_notes)
    assert Q.search("aclforgeryword", identity="eng-user", root=str(tmp_path)) == []
    assert calls == 1


def test_snapshot_fails_closed_when_suppression_state_is_malformed(tmp_path):
    _write(tmp_path, name="suppressed-risk", description="must not resurrect")
    regen_index(str(tmp_path / "context"))
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "suppressed.json").write_text("{not-json")

    with pytest.raises(RuntimeError, match="suppression state is unreadable"):
        load_index_snapshot(str(tmp_path / "context"))


def test_stale_index_falls_back_and_recovers_new_description(tmp_path, monkeypatch):
    note_path = _write(tmp_path, name="mutable-note", description="old wording")
    regen_index(str(tmp_path / "context"))
    index_path = tmp_path / "context" / "eng" / "index.md"

    _write(tmp_path, name="mutable-note", description="freshly coined zephyrium")
    # Make the staleness relation deterministic even on coarse/virtual filesystems.
    index_mtime = index_path.stat().st_mtime_ns
    os.utime(note_path, ns=(index_mtime + 1_000_000, index_mtime + 1_000_000))

    calls = 0
    real_load_notes = Q.load_notes

    def counted_load_notes(path, **kwargs):
        nonlocal calls
        calls += 1
        return real_load_notes(path, **kwargs)

    monkeypatch.setattr(Q, "load_notes", counted_load_notes)
    assert [hit["name"] for hit in Q.search("zephyrium", root=str(tmp_path))] == [
        "mutable-note"
    ]
    assert calls == 1


def test_tampered_index_metadata_falls_back_without_false_negative(tmp_path, monkeypatch):
    _write(tmp_path, name="authoritative-note", description="real zephyrium wording")
    regen_index(str(tmp_path / "context"))
    index_path = tmp_path / "context" / "eng" / "index.md"
    lines = index_path.read_text().splitlines()
    for index, line in enumerate(lines):
        if "<!-- remembrancer-index-v2:" not in line:
            continue
        metadata = _decode_index_metadata(line)
        assert metadata is not None
        metadata["description"] = "tampered decoy wording"
        prefix = line[:line.index("<!-- remembrancer-index-v2:")]
        lines[index] = prefix + encode_index_metadata(metadata)
        break
    index_path.write_text("\n".join(lines) + "\n")

    calls = 0
    real_load_notes = Q.load_notes

    def counted_load_notes(path, **kwargs):
        nonlocal calls
        calls += 1
        return real_load_notes(path, **kwargs)

    monkeypatch.setattr(Q, "load_notes", counted_load_notes)
    assert [hit["name"] for hit in Q.search("zephyrium", root=str(tmp_path))] == [
        "authoritative-note"
    ]
    assert Q.search("decoy", root=str(tmp_path)) == []
    assert calls == 2


def test_regen_index_excludes_tombstoned_files_from_complete_snapshot(tmp_path):
    _write(tmp_path, name="deleted-memory", description="must remain deleted")
    regen_index(str(tmp_path / "context"))
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "suppressed.json").write_text(json.dumps([
        "deleted-memory"
    ]))

    regen_index(str(tmp_path / "context"))
    snapshot = load_index_snapshot(str(tmp_path / "context"))
    assert snapshot is not None
    assert snapshot.records == ()
    assert "deleted-memory" not in (tmp_path / "context" / "eng" / "index.md").read_text()


def test_regen_clears_generated_domain_index_after_last_note_is_removed(tmp_path):
    note_path = _write(tmp_path, name="last-note", description="last live note")
    regen_index(str(tmp_path / "context"))
    note_path.unlink()

    regen_index(str(tmp_path / "context"))
    domain_index = (tmp_path / "context" / "eng" / "index.md").read_text()
    assert "last-note" not in domain_index
    assert "remembrancer-index-v2-digest" in domain_index
    snapshot = load_index_snapshot(str(tmp_path / "context"))
    assert snapshot is not None and snapshot.records == ()


def test_index_round_trips_legacy_invalid_and_verified_states(tmp_path):
    _config(tmp_path)
    _write(tmp_path, name="legacy-note", description="legacy state", state=None)
    _write(tmp_path, name="verified-note", description="verified state", state="proposed")
    _review(tmp_path, "verified-note", "verified")
    invalid_path = _write(
        tmp_path, name="invalid-note", description="invalid state", state="proposed"
    )
    invalid_path.write_text(
        invalid_path.read_text().replace("state: proposed", "state: invalid-value")
    )

    regen_index(str(tmp_path / "context"))
    snapshot = load_index_snapshot(str(tmp_path / "context"))
    assert snapshot is not None
    states = {record.name: record.state for record in snapshot.records}
    assert states == {
        "invalid-note": "invalid-value",
        "legacy-note": None,
        "verified-note": "verified",
    }


def test_bm25_uses_name_description_and_curated_keys(tmp_path):
    """Ranking contract: tags and curated index keys are ranked too.

    Stage-1 ranks name + description + tags + TF-IDF-selected distinctive body
    terms carried in the index row. Titles remain display-only; RAW body text
    is still never indexed wholesale — only distinctive terms survive the
    TF-IDF cut, so "orion"-style generic body prose cannot flood results while
    a truly distinctive body term can find its note.
    """
    _write(
        tmp_path,
        name="stable-memory-name",
        title="Human Nebulaword Title",
        description="searchable synopsis",
        body="bodyonlysingularity appears nowhere else",
    )
    regen_index(str(tmp_path / "context"))

    assert Q.search("stable memory", root=str(tmp_path))[0]["name"] == "stable-memory-name"
    assert Q.search("synopsis", root=str(tmp_path))[0]["name"] == "stable-memory-name"
    assert Q.search("Nebulaword", root=str(tmp_path)) == []            # title: display-only
    # a distinctive body term finds the note via its index keys
    assert Q.search("bodyonlysingularity", root=str(tmp_path))[0]["name"] == "stable-memory-name"


def test_equal_bm25_scores_prefer_more_recent_update(tmp_path):
    _write(tmp_path, name="older-entry", description="identical rankword",
           updated="2026-06-01")
    _write(tmp_path, name="newer-entry", description="identical rankword",
           updated="2026-07-14")
    regen_index(str(tmp_path / "context"))

    assert [hit["name"] for hit in Q.search("rankword", root=str(tmp_path))] == [
        "newer-entry", "older-entry"
    ]


def test_opted_in_superseded_history_is_rank_demoted(tmp_path):
    _config(tmp_path, retrieval={"hide_superseded": False})
    _write(tmp_path, name="current-entry", description="same demotionword")
    _write(tmp_path, name="historic-entry", description="same demotionword",
           state="proposed")
    _review(tmp_path, "historic-entry", "superseded")
    regen_index(str(tmp_path / "context"))

    hits = Q.search("demotionword", root=str(tmp_path))
    assert [hit["name"] for hit in hits] == ["current-entry", "historic-entry"]
    assert hits[0]["score"] > hits[1]["score"]


def test_per_request_history_overrides_default_and_current_filter(tmp_path):
    _config(tmp_path, retrieval={"hide_superseded": True})
    _write(tmp_path, name="current-entry", description="requesthistory word",
           state="proposed")
    _review(tmp_path, "current-entry", "verified")
    _write(tmp_path, name="historic-entry", description="requesthistory word",
           state="proposed")
    _review(tmp_path, "historic-entry", "superseded")
    regen_index(str(tmp_path / "context"))

    assert [hit["name"] for hit in Q.search(
        "requesthistory", root=str(tmp_path), current_only=True
    )] == ["current-entry"]
    hits = Q.search(
        "requesthistory", root=str(tmp_path), current_only=True,
        include_history=True,
    )
    assert [hit["name"] for hit in hits] == ["current-entry", "historic-entry"]
    assert hits[0]["score"] > hits[1]["score"]

    full = Q.search_full_scan(
        "requesthistory", root=str(tmp_path), current_only=True,
        include_history=True,
    )
    assert [hit["name"] for hit in full] == ["current-entry", "historic-entry"]


def test_fresh_index_still_enforces_authoritative_note_acl(tmp_path, monkeypatch):
    _config(tmp_path, identities={"eng-user": ["eng"]})
    _write(tmp_path, name="public-memory", description="aclword", acl=[])
    _write(tmp_path, name="secret-memory", description="aclword", acl=["secrets"])
    regen_index(str(tmp_path / "context"))
    monkeypatch.setattr(Q, "load_notes", lambda *_: (_ for _ in ()).throw(
        AssertionError("ACL gating should not require a full fallback")))

    names = {hit["name"] for hit in Q.search(
        "aclword", identity="eng-user", root=str(tmp_path)
    )}
    assert names == {"public-memory"}


def test_search_only_writes_optional_state_audit(tmp_path, monkeypatch):
    # The suite-wide conftest sets REMEMBRANCER_QUERY_LOG=off so tests can't pollute a
    # real checkout; this test exercises the writer itself, so re-enable it.
    monkeypatch.delenv("REMEMBRANCER_QUERY_LOG", raising=False)
    _config(tmp_path, retrieval={"query_log": True})
    _write(tmp_path, name="readonly-note", description="readonlyword")
    regen_index(str(tmp_path / "context"))
    (tmp_path / "events").mkdir()
    (tmp_path / "events" / "2026-07-15.jsonl").write_text('{"id":"session:x"}\n')
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry" / "entities.yml").write_text("people: []\n")

    protected_roots = [tmp_path / "context", tmp_path / "events", tmp_path / "registry"]
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for root in protected_roots for path in root.rglob("*") if path.is_file()
    }

    assert Q.search("readonlyword", root=str(tmp_path))

    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for root in protected_roots for path in root.rglob("*") if path.is_file()
    }
    assert after == before
    rows = (tmp_path / "state" / "queries.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["query"] == "readonlyword"
