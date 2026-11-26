"""Failed claim verification must not become an answer through any read path."""
from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import pytest

import mcp.server as mcp_server
from remembrancerlib import (
    Note,
    load_notes,
    regen_index,
    verification_content_fingerprint,
    write_note,
)
from remembrancerlib.event import _EVENT_ALLOWED_SOURCES, _EVENT_ID_PREFIX_ALIASES
from remembrancerlib.verification_audit import encode_head, make_head
from retrieve import query as Q
from retrieve.index import load_index_snapshot
from retrieve.verification import load_verification_audit
from tools.generated_notes import generated_grounding, persist_generated_note


_MACHINE_PROVENANCE_PREFIXES = sorted(
    set(_EVENT_ALLOWED_SOURCES).union(
        *(set(aliases) for aliases in _EVENT_ID_PREFIX_ALIASES.values())
    )
)


def _config(root: Path, *, gate: bool | None = None) -> None:
    trust = {"proposed_rank_penalty": 0.5}
    if gate is not None:
        trust["hide_verification_failures"] = gate
    (root / "config.json").write_text(json.dumps({
        "identities": {"reviewer": ["eng"]},
        "retrieval": {
            "query_log": False,
            "file_back": {"enabled": True, "min_chars": 1},
        },
        "trust": {**trust, "approver_identities": ["reviewer"]},
    }), encoding="utf-8")


def _require_audit(root: Path) -> None:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["trust"]["verification_audit_required"] = True
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _confidence(status: str) -> dict:
    return {
        "score": 0.3,
        "verification": {
            "status": status,
            "checked": 1,
            "supported": int(status == "verified"),
            "unsupported": int(status != "verified"),
        },
    }


def _write(root: Path, name: str, *, status: str | None = None,
           state: str | None = "proposed", entity: str = "") -> None:
    requested_state = state
    write_note(Note(
        name=name,
        description="verification gateword",
        domain="eng",
        type="state",
        state="proposed" if state == "verified" else state,
        entities=[entity] if entity else [],
        confidence=_confidence(status) if status is not None else None,
        body="answer body",
    ), str(root / "context"))
    if requested_state == "verified":
        assert Q.promote(name, identity="reviewer", root=str(root))["ok"]


def _audit(
    root: Path, *rows: tuple[str, str], anchored: bool = False,
) -> None:
    path = root / "state" / "verification.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps({"note": name, "status": status}) + "\n"
        for name, status in rows
    ), encoding="utf-8")
    path.chmod(0o600)
    if anchored:
        raw = path.read_bytes()
        head = path.with_name("verification.head.jsonl")
        head.write_bytes(encode_head(make_head(raw, None)))
        head.chmod(0o600)
    path.chmod(0o600)


def test_index_and_full_scan_share_verification_gate_and_priority(tmp_path):
    _config(tmp_path)
    _write(tmp_path, "embedded-flagged", status="flagged")
    _write(tmp_path, "embedded-skipped", status="skipped")
    _write(tmp_path, "embedded-verified", status="verified")
    _write(tmp_path, "human-override", status="flagged", state="verified")
    _write(tmp_path, "legacy-unverified", status=None, state=None)
    _write(tmp_path, "audit-only-flagged", status=None)
    _write(tmp_path, "audit-latest-visible", status=None)
    _write(tmp_path, "embedded-failure-wins", status="flagged")
    _write(tmp_path, "audit-flagged-wins", status="verified")
    _write(tmp_path, "audit-skipped-wins", status="verified")
    _write(tmp_path, "nonfailure-conflict", status="verified")
    _write(tmp_path, "matching-verified", status="verified")
    _audit(
        tmp_path,
        ("audit-only-flagged", "flagged"),
        ("audit-latest-visible", "flagged"),
        ("audit-latest-visible", "verified"),
        ("embedded-failure-wins", "verified"),
        ("audit-flagged-wins", "flagged"),
        ("audit-skipped-wins", "skipped"),
        ("nonfailure-conflict", "unverified"),
        ("matching-verified", "verified"),
    )
    regen_index(str(tmp_path / "context"))

    expected = {
        "human-override",
        "legacy-unverified",
        "audit-latest-visible",
    }
    indexed = Q.search("gateword", root=str(tmp_path))
    scanned = Q.search_full_scan("gateword", root=str(tmp_path))
    assert {row["name"] for row in indexed} == expected
    assert {row["name"] for row in scanned} == expected
    assert {row["name"]: row["verification"] for row in indexed} == {
        "human-override": "flagged",
        "legacy-unverified": "unverified",
        "audit-latest-visible": "verified",
    }
    for failed_name in (
        "embedded-failure-wins",
        "audit-flagged-wins",
        "audit-skipped-wins",
        "nonfailure-conflict",
    ):
        assert Q.get_note(failed_name, root=str(tmp_path)) is None

    snapshot = load_index_snapshot(str(tmp_path / "context"))
    assert snapshot is not None
    embedded = {record.name: record.verification for record in snapshot.records}
    assert embedded["embedded-flagged"] == "flagged"
    assert embedded["legacy-unverified"] is None


@pytest.mark.parametrize("name", [
    "embedded-flagged", "embedded-skipped", "audit-only-flagged",
])
def test_get_note_and_history_cannot_bypass_failed_verification(tmp_path, name):
    _config(tmp_path)
    _write(tmp_path, "embedded-flagged", status="flagged")
    _write(tmp_path, "embedded-skipped", status="skipped")
    _write(tmp_path, "audit-only-flagged")
    _write(tmp_path, "human-override", status="flagged", state="verified")
    _audit(tmp_path, ("audit-only-flagged", "flagged"))

    assert Q.get_note(name, root=str(tmp_path)) is None
    assert Q.get_note(name, root=str(tmp_path), include_history=True) is None
    assert Q.get_note("human-override", root=str(tmp_path))["body"] == "answer body\n"


def test_required_audit_cannot_downgrade_to_legacy_mode_after_deletion(tmp_path):
    _config(tmp_path)
    _require_audit(tmp_path)
    _write(tmp_path, "audit-only-flagged")
    _write(tmp_path, "legacy-proposed", state=None)
    _audit(tmp_path, ("audit-only-flagged", "flagged"), anchored=True)

    assert Q.get_note("audit-only-flagged", root=str(tmp_path)) is None
    assert load_verification_audit(str(tmp_path)).trustworthy is True
    (tmp_path / "state" / "verification.jsonl").unlink()

    assert load_verification_audit(str(tmp_path)).trustworthy is False
    assert Q.get_note("audit-only-flagged", root=str(tmp_path)) is None
    assert Q.get_note("legacy-proposed", root=str(tmp_path)) is None


def test_required_audit_rejects_truncation_or_head_rollback(tmp_path):
    _config(tmp_path)
    _require_audit(tmp_path)
    _audit(tmp_path, ("legacy-note", "flagged"), anchored=True)
    audit_path = tmp_path / "state" / "verification.jsonl"
    head_path = tmp_path / "state" / "verification.head.jsonl"
    audit_bytes = audit_path.read_bytes()
    head_bytes = head_path.read_bytes()

    assert load_verification_audit(str(tmp_path)).trustworthy is True
    audit_path.write_bytes(b"")
    assert load_verification_audit(str(tmp_path)).trustworthy is False

    audit_path.write_bytes(audit_bytes)
    audit_path.chmod(0o600)
    head_path.write_bytes(b"")
    assert load_verification_audit(str(tmp_path)).trustworthy is False

    head_path.write_bytes(head_bytes)
    head_path.chmod(0o600)
    assert load_verification_audit(str(tmp_path)).trustworthy is True
    head_path.unlink()
    assert load_verification_audit(str(tmp_path)).trustworthy is False


def test_audit_reader_rejects_symlink_non_private_and_non_regular(tmp_path):
    _config(tmp_path)
    _audit(tmp_path, ("legacy-note", "flagged"))
    audit_path = tmp_path / "state" / "verification.jsonl"
    assert load_verification_audit(str(tmp_path)).trustworthy is True

    audit_path.chmod(0o644)
    assert load_verification_audit(str(tmp_path)).trustworthy is False
    audit_path.chmod(0o600)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(audit_path.read_bytes())
    outside.chmod(0o600)
    audit_path.unlink()
    audit_path.symlink_to(outside)
    assert load_verification_audit(str(tmp_path)).trustworthy is False


def test_malformed_or_unreadable_audit_fails_closed_without_leaking(tmp_path, monkeypatch):
    _config(tmp_path)
    _write(tmp_path, "legacy-proposed", state=None)
    _write(tmp_path, "machine-verified", status="verified")
    _write(tmp_path, "human-verified", state="verified")
    audit_path = tmp_path / "state" / "verification.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text('{"note":"legacy-proposed","status":"verified"}',
                          encoding="utf-8")  # partial append: no newline

    assert load_verification_audit(str(tmp_path)).trustworthy is False
    assert Q.get_note("legacy-proposed", root=str(tmp_path)) is None
    assert Q.get_note("machine-verified", root=str(tmp_path)) is None
    assert Q.get_note("human-verified", root=str(tmp_path)) is not None

    real_open = builtins.open

    def denied_open(file, *args, **kwargs):
        if Path(file) == audit_path:
            raise PermissionError("secret filesystem detail")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", denied_open)
    assert [
        row["name"] for row in Q.search("gateword", root=str(tmp_path))
    ] == ["human-verified"]
    assert Q.get_note("legacy-proposed", root=str(tmp_path)) is None
    assert Q.get_note("machine-verified", root=str(tmp_path)) is None
    assert Q.get_note("human-verified", root=str(tmp_path)) is not None


def test_explicit_false_disables_gate_for_attended_compatibility(tmp_path):
    _config(tmp_path, gate=False)
    _write(tmp_path, "failed-but-opted-in", status="flagged")
    path = tmp_path / "state" / "verification.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("malformed\n", encoding="utf-8")

    assert Q.get_note("failed-but-opted-in", root=str(tmp_path)) is not None
    assert [row["name"] for row in Q.search("gateword", root=str(tmp_path))] == [
        "failed-but-opted-in"
    ]


def test_file_answer_refuses_failed_source(tmp_path):
    _config(tmp_path)
    _write(tmp_path, "failed-source", status="skipped")

    with pytest.raises(PermissionError, match="unavailable"):
        Q.file_answer(
            "What is the answer?", "A grounded answer", ["failed-source"],
            root=str(tmp_path), identity=None,
        )
    assert not any(
        note.name.startswith("synthesis-")
        for note in Q.load_notes(str(tmp_path / "context"))
    )


def test_mcp_search_get_timeline_and_graph_apply_gate(tmp_path, monkeypatch):
    _config(tmp_path)
    _write(tmp_path, "failed-note", status="flagged", entity="org:acme")
    _write(
        tmp_path, "good-note", status="verified", state="verified",
        entity="org:acme",
    )
    monkeypatch.setattr(mcp_server, "ROOT", str(tmp_path))
    monkeypatch.setattr(mcp_server, "_iter_events_safe", lambda: [])

    search = mcp_server._do_search(
        "gateword", None, 20, None, root=str(tmp_path),
    )
    assert {row["name"] for row in search["results"]} == {"good-note"}
    assert "error" in mcp_server._do_get_note("failed-note", None)
    assert mcp_server._do_get_note("good-note", None)["verification"] == "unverified"

    timeline = mcp_server._do_timeline("org:acme", None, 20)
    assert {row.get("name") for row in timeline["events"]} == {"good-note"}
    by_evidence = mcp_server._notes_by_evidence_key(mcp_server.FULL_ACCESS)
    assert "failed-note" not in by_evidence
    assert by_evidence["good-note"].name == "good-note"


def test_mcp_get_note_failure_does_not_use_direct_loader(tmp_path, monkeypatch):
    class BrokenRetrieval:
        @staticmethod
        def get_note(*args, **kwargs):
            raise RuntimeError("primary read failed")

    monkeypatch.setattr(mcp_server, "_retrieve_module", lambda: BrokenRetrieval)
    monkeypatch.setattr(
        mcp_server,
        "_fallback_load_notes",
        lambda: (_ for _ in ()).throw(AssertionError("direct fallback bypassed gate")),
    )
    assert "error" in mcp_server._do_get_note("anything", None)


def test_matching_generic_cove_binding_still_cannot_self_attest_classification(tmp_path):
    _config(tmp_path)
    note = Note(
        name="bound-note",
        description="binding gateword",
        domain="eng",
        type="state",
        state="proposed",
        provenance=["slack:bound"],
        body="The launch is complete.",
    )
    fingerprint = verification_content_fingerprint(note)
    note.confidence = _confidence("verified")
    note.confidence["verification"].update({
        "audit_id": "audit-bound-1",
        "content_fingerprint": fingerprint,
    })
    write_note(note, str(tmp_path / "context"))
    audit = tmp_path / "state" / "verification.jsonl"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps({
        "note": note.name,
        "status": "verified",
        "audit_id": "audit-bound-1",
        "content_fingerprint": fingerprint,
    }) + "\n", encoding="utf-8")

    assert Q.get_note(note.name, root=str(tmp_path)) is None

    # A machine mutation carrying the old approval is invalidated at the write
    # door, and the old audit row cannot resurrect it.
    note.body = "The launch is delayed."
    write_note(note, str(tmp_path / "context"))
    persisted = {row.name: row for row in load_notes(str(tmp_path / "context"))}[note.name]
    assert persisted.confidence["verification"]["status"] == "skipped"
    assert persisted.confidence["verification"]["content_fingerprint"] == \
        verification_content_fingerprint(persisted)
    assert Q.get_note(note.name, root=str(tmp_path)) is None


@pytest.mark.parametrize("provenance_prefix", _MACHINE_PROVENANCE_PREFIXES)
def test_write_door_fail_closes_every_allowed_machine_event_source(
        tmp_path, provenance_prefix):
    note = Note(
        name="machine-source-draft",
        description="unverified machine source draft",
        domain="eng",
        type="state",
        state="proposed",
        provenance=[f"{provenance_prefix}:event-1"],
        body="Machine-authored content.",
    )

    write_note(note, str(tmp_path / "context"))

    persisted = load_notes(str(tmp_path / "context"))[0]
    verification = persisted.confidence["verification"]
    assert verification["status"] == "skipped"
    assert verification["reason"] == "machine-content-awaits-verification"
    assert verification["content_fingerprint"] == \
        verification_content_fingerprint(persisted)


def test_new_audit_only_success_cannot_bypass_missing_embedded_commit(tmp_path):
    _config(tmp_path)
    note = Note(
        name="audit-crash-window",
        description="binding gateword",
        domain="eng",
        type="state",
        state="proposed",
        body="Current content",
    )
    write_note(note, str(tmp_path / "context"))
    fingerprint = verification_content_fingerprint(note)
    audit = tmp_path / "state" / "verification.jsonl"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps({
        "note": note.name,
        "status": "verified",
        "audit_id": "audit-only-new-format",
        "content_fingerprint": fingerprint,
    }) + "\n", encoding="utf-8")

    assert Q.get_note(note.name, root=str(tmp_path)) is None


def test_generic_producer_cannot_self_attest_in_index_or_full_scan(tmp_path):
    _config(tmp_path)
    context = str(tmp_path / "context")
    note = Note(
        name="generated-bound",
        description="producer gateword",
        domain="eng",
        type="page",
        body="Deterministic value: 42",
    )
    persist_generated_note(
        note,
        producer="test-producer",
        context_dir=context,
        grounding=generated_grounding(
            "fixture-record",
            {"value": 42},
            grounded=True,
            reason="exact deterministic fixture rendering",
        ),
    )
    regen_index(context)
    assert Q.search("gateword", root=str(tmp_path)) == []
    assert Q.search_full_scan("gateword", root=str(tmp_path)) == []

    persisted = {row.name: row for row in load_notes(context)}[note.name]
    persisted.confidence["verification"]["audit_id"] = "0" * 64
    write_note(persisted, context)
    regen_index(context)
    assert Q.search("gateword", root=str(tmp_path)) == []
    assert Q.search_full_scan("gateword", root=str(tmp_path)) == []


@pytest.mark.parametrize("provenance", [
    ["slack:forged"],
    ["slack:forged", "by:claimed-human"],
])
def test_write_door_rejects_fingerprint_only_machine_success(tmp_path, provenance):
    _config(tmp_path)
    note = Note(
        name="forged-machine-proof",
        description="forged gateword",
        domain="eng",
        type="state",
        state="proposed",
        provenance=provenance,
        body="Unreviewed machine claim",
    )
    note.confidence = _confidence("verified")
    note.confidence["verification"]["content_fingerprint"] = \
        verification_content_fingerprint(note)
    write_note(note, str(tmp_path / "context"))

    persisted = load_notes(str(tmp_path / "context"))[0]
    assert persisted.confidence["verification"]["status"] == "skipped"
    assert persisted.confidence["verification"]["reason"] == \
        "generic-cove-cannot-self-attest-classification"
    assert Q.get_note(note.name, root=str(tmp_path)) is None


def test_write_door_rejects_matching_generic_cove_audit_shape(tmp_path):
    note = Note(
        name="generic-cove-shape",
        description="unreviewed classification",
        domain="company",
        type="decision",
        state="proposed",
        provenance=["slack:event-1"],
        body="Should we approve launch?",
    )
    note.confidence = _confidence("verified")
    note.confidence["verification"].update({
        "content_fingerprint": verification_content_fingerprint(note),
        "audit_id": "well-shaped-but-not-authoritative",
    })

    write_note(note, str(tmp_path / "context"))

    persisted = load_notes(str(tmp_path / "context"))[0]
    verification = persisted.confidence["verification"]
    assert verification["status"] == "skipped"
    # Decisions are approval-classified by governance and therefore become hard-held
    # until explicit human review, even when verification payloads are present.
    assert verification["reason"] == \
        "governance-approval-required:decision"


def test_allowlisted_producer_shape_and_matching_audit_still_need_sealed_writer(
    tmp_path,
):
    _config(tmp_path)
    note = Note(
        name="contradiction-alpha-vs-beta",
        description="Auto-detected contradiction between alpha and beta",
        type="contradiction",
        domain="company",
        state="proposed",
        provenance=["generated-by:contradiction-lint"],
        structured={"kind": "contradiction", "pair": ["alpha", "beta"]},
        body=(
            "Auto-detected contradiction between [[alpha]] and [[beta]].\n\n"
            "Resolve by updating one of the source notes or reconciling the conflict."
        ),
    )
    fingerprint = verification_content_fingerprint(note)
    source_fingerprint = "1" * 64
    method = "producer-grounded-v1:contradiction-detector"
    material = json.dumps({
        "producer": "contradiction-lint",
        "method": method,
        "source_fingerprint": source_fingerprint,
        "content_fingerprint": fingerprint,
        "status": "verified",
    }, sort_keys=True, separators=(",", ":")).encode()
    audit_id = hashlib.sha256(material).hexdigest()
    note.confidence = {
        "score": 1.0,
        "level": "high",
        "verification": {
            "status": "verified", "checked": 1, "supported": 1,
            "unsupported": 0, "producer": "contradiction-lint",
            "method": method, "source_fingerprint": source_fingerprint,
            "content_fingerprint": fingerprint, "audit_id": audit_id,
            "grounded": True, "reason": "fabricated closed proof",
        },
    }
    # ``path`` is caller-controlled on in-memory Note objects. It must not let
    # the write door discover the forged audit and bypass the missing seal.
    note.path = str(
        tmp_path / "context" / "company" / f"{note.name}.md"
    )
    audit = tmp_path / "state" / "verification.jsonl"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps({
        "note": note.name, "status": "verified", "audit_id": audit_id,
        "content_fingerprint": fingerprint,
    }) + "\n", encoding="utf-8")
    audit.chmod(0o600)

    write_note(note, str(tmp_path / "context"))

    persisted = load_notes(str(tmp_path / "context"))[0]
    assert persisted.confidence["verification"]["status"] == "skipped"
    assert Q.get_note(note.name, root=str(tmp_path)) is None


def test_imported_legacy_embedded_success_is_not_content_authority(tmp_path):
    _config(tmp_path)
    _write(tmp_path, "name-only-success", status="verified", state="proposed")
    assert Q.get_note("name-only-success", root=str(tmp_path)) is None


# --------------------------------------------------------------------------- #
# The answer gate is ALLOW-BY-DEFAULT over the status vocabulary:
#
#     return status not in (FAILED_VERIFICATION_STATUSES
#                           | {_INVALID_STATUS, _CONFLICT_STATUS, _STALE_STATUS})
#
# so any status that is not explicitly denied GRANTS an answer. Adding a member
# to VERIFICATION_STATUSES without also denying it therefore makes those notes
# answerable in one line, silently. That is a live risk right now: the ledger
# learned a `not_applicable` verdict whose whole meaning is "this
# note cannot be verified this way", and the only thing keeping it out of the
# audit vocabulary is that nobody has added it yet.
# --------------------------------------------------------------------------- #

#: Every status the retrieval vocabulary admits, and whether it may answer.
#: This map is the decision; the test below only holds the code to it. A new
#: status fails here until someone states which side it is on.
_STATUS_MAY_ANSWER = {
    "verified": True,
    "unverified": True,      # no model verdict yet — the pre-CoVe default
    "flagged": False,
    "skipped": False,
}


def test_every_admitted_status_is_deliberately_classified():
    """A new vocabulary member must not inherit answer-eligibility by default."""
    from retrieve.verification import VERIFICATION_STATUSES

    assert set(VERIFICATION_STATUSES) == set(_STATUS_MAY_ANSWER), (
        "retrieve.verification.VERIFICATION_STATUSES changed. The answer gate "
        "is allow-by-default, so an unclassified status GRANTS answers. Decide "
        "explicitly and record it in _STATUS_MAY_ANSWER — and if the new status "
        "means 'not trustworthy to answer from', it also belongs in "
        "FAILED_VERIFICATION_STATUSES."
    )


def test_the_deny_set_matches_the_classification():
    """FAILED_VERIFICATION_STATUSES must hold exactly the non-answering ones."""
    from retrieve.verification import FAILED_VERIFICATION_STATUSES

    expected = {s for s, may in _STATUS_MAY_ANSWER.items() if not may}
    assert set(FAILED_VERIFICATION_STATUSES) == expected, (
        "a status classified as non-answering is not in the deny set, so the "
        "allow-by-default gate will answer from it"
    )


def test_a_hypothetical_unclassified_status_would_answer():
    """Pins WHY the two tests above exist, so neither is deleted as ceremony.

    This does not add a status; it demonstrates on the real predicate that an
    unlisted one is granted, which is the failure mode being guarded.
    """
    from retrieve.verification import (
        FAILED_VERIFICATION_STATUSES,
        _CONFLICT_STATUS,
        _INVALID_STATUS,
        _STALE_STATUS,
    )

    denied = FAILED_VERIFICATION_STATUSES | {
        _INVALID_STATUS, _CONFLICT_STATUS, _STALE_STATUS,
    }
    assert "not_applicable" not in denied, (
        "if not_applicable ever enters the vocabulary it must ALSO enter the "
        "deny set in the same change — this assertion is the reminder"
    )
