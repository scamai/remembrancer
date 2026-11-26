"""Mutable-source ownership is explicit, canonical, and trust-bound."""
from __future__ import annotations

import pytest

from lint.engine import find_schema_issues
from remembrancerlib import (
    Note,
    human_trust_valid,
    load_notes,
    regen_index,
    verification_content_fingerprint,
    write_note,
)
from retrieve.index import load_index_snapshot
from tests.trust_helpers import write_reviewed_note


def test_source_lineages_round_trip_and_legacy_absence(tmp_path):
    context = tmp_path / "context"
    note = Note(
        name="owned", description="Owned snapshot",
        source_lineages=["slack:C1:30.000", "gcal:weekly:"],
    )
    write_note(note, str(context))
    loaded = load_notes(str(context))[0]
    assert loaded.source_lineages == note.source_lineages

    legacy = Note(name="legacy", description="Legacy citation")
    raw = legacy.to_markdown()
    assert "source_lineages:" not in raw
    path = context / "company" / "legacy.md"
    path.write_text(raw, encoding="utf-8")
    by_name = {item.name: item for item in load_notes(str(context))}
    assert by_name["legacy"].source_lineages == []


@pytest.mark.parametrize(
    "raw_value",
    [
        "slack:C1:30.000",
        "[] # forged-comment",
        "[slack:C1:30.000, slack:C1:30.000]",
        "[unknown:key]",
        "[slack:C1:30.000, bad key]",
    ],
)
def test_malformed_source_lineages_never_become_authority(tmp_path, raw_value):
    context = tmp_path / "context"
    path = context / "company" / "forged.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\n"
        "name: forged\n"
        "description: Forged ownership\n"
        "type: project\n"
        "domain: company\n"
        "acl: []\nentities: []\ncomponents: []\nprovenance: [slack:old]\n"
        f"source_lineages: {raw_value}\n"
        "updated: 2026-07-15\nsuperseded: false\n---\n\nBody\n",
        encoding="utf-8",
    )
    loaded = load_notes(str(context))[0]
    assert loaded.source_lineages == []
    assert any(
        issue["code"] == "invalid_source_lineages"
        for issue in find_schema_issues(str(context))
    )
    with pytest.raises(ValueError, match="source_lineages"):
        write_note(loaded, str(context))


def test_source_ownership_mutation_invalidates_human_content_binding(tmp_path):
    note = Note(name="approved", description="Approved fact", state="proposed")
    _path, approved = write_reviewed_note(tmp_path, note, state="verified")
    before = verification_content_fingerprint(approved)
    approved.source_lineages = ["gdrive:file-1"]
    assert verification_content_fingerprint(approved) != before
    assert human_trust_valid(approved) is False


def test_index_metadata_preserves_source_and_lifecycle_fields(tmp_path):
    context = tmp_path / "context"
    note = Note(
        name="indexed-owner", description="Indexed owner",
        source_lineages=["crm:deal:deal-1"],
    )
    write_note(note, str(context))
    regen_index(str(context))
    snapshot = load_index_snapshot(str(context))
    assert snapshot is not None
    record = snapshot.records[0]
    assert record.source_lineages == ("crm:deal:deal-1",)
    assert record.source_retirement is None
    assert record.producer_lifecycle is None
