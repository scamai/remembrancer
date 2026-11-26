import json
from pathlib import Path

import pytest

from lint.engine import (
    QuarantineLimitExceeded,
    VALID_DOMAINS,
    find_schema_issues,
    quarantine_invalid_notes,
    run,
)
from remembrancerlib import MAX_DESCRIPTION_CHARS, Note, concise_description, write_note
from tests.trust_helpers import write_reviewed_note


def _write_raw_note(path: Path, frontmatter: str, body: str = "body") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")


def test_schema_accepts_current_producer_types_and_domains(tmp_path):
    context = tmp_path / "context"
    for index, (note_type, domain) in enumerate((
        ("artifact", "company"),
        ("initiative", "people"),
        ("correction", "design"),
        ("synthesis", "synthesis"),
    )):
        write_note(Note(
            name=f"producer-{index}", description="valid", type=note_type,
            domain=domain,
        ), str(context))

    assert find_schema_issues(str(context)) == []
    assert {"people", "design", "company", "synthesis"} <= set(VALID_DOMAINS)


def test_schema_reports_required_slug_type_and_domain_issues(tmp_path):
    context = tmp_path / "context"
    _write_raw_note(
        context / "unknown" / "bad.md",
        "name: Bad_Name\ndescription:\ntype: mystery\ndomain: unknown",
    )

    issues = find_schema_issues(str(context))
    codes = {str(issue["code"]) for issue in issues}
    assert codes == {"missing_description", "invalid_slug", "invalid_type", "invalid_domain"}
    required = next(issue for issue in issues if issue["code"] == "missing_description")
    assert required["quarantinable"] is True
    assert all(
        issue["quarantinable"] is False
        for issue in issues if issue["code"] != "missing_description"
    )


def test_run_is_audit_only_by_default(tmp_path):
    context = tmp_path / "context"
    bad = context / "eng" / "bad.md"
    _write_raw_note(bad, "name: bad\ndescription:\ntype: project\ndomain: eng")

    summary = run(root=str(tmp_path))

    assert bad.is_file()
    assert summary["quarantined"] == []
    assert any(issue["code"] == "missing_description" for issue in summary["schema_issues"])


def test_quarantine_moves_required_failure_and_writes_manifest(tmp_path):
    context = tmp_path / "context"
    bad = context / "eng" / "bad.md"
    original = "---\nname: bad\ndescription:\ntype: project\ndomain: eng\n---\n\nbody\n"
    bad.parent.mkdir(parents=True)
    bad.write_text(original, encoding="utf-8")

    summary = run(root=str(tmp_path), quarantine=True)

    assert not bad.exists()
    assert len(summary["quarantined"]) == 1
    record = summary["quarantined"][0]
    quarantined = tmp_path / str(record["quarantine_path"])
    assert quarantined.read_text(encoding="utf-8") == original
    manifest = tmp_path / "state" / "quarantine" / "manifest.jsonl"
    manifest_record = json.loads(manifest.read_text(encoding="utf-8").strip())
    assert manifest_record["original_path"] == "context/eng/bad.md"
    assert manifest_record["issues"] == ["missing_description"]
    assert len(manifest_record["sha256"]) == 64
    assert summary["remaining_schema_issues"] == []


def test_quarantine_cap_fails_before_any_move(tmp_path):
    context = tmp_path / "context"
    for name in ("one", "two"):
        _write_raw_note(
            context / "eng" / f"{name}.md",
            f"name: {name}\ndescription:\ntype: project\ndomain: eng",
        )
    issues = find_schema_issues(str(context))

    with pytest.raises(QuarantineLimitExceeded, match="safety cap"):
        quarantine_invalid_notes(issues, str(tmp_path), max_notes=1)

    assert (context / "eng" / "one.md").is_file()
    assert (context / "eng" / "two.md").is_file()
    assert not (tmp_path / "state" / "quarantine" / "manifest.jsonl").exists()


def test_quarantine_never_overwrites_a_prior_quarantined_copy(tmp_path):
    context = tmp_path / "context"
    bad = context / "eng" / "bad.md"
    frontmatter = "name: bad\ndescription:\ntype: project\ndomain: eng"
    _write_raw_note(bad, frontmatter, body="first")
    first = quarantine_invalid_notes(find_schema_issues(str(context)), str(tmp_path))

    _write_raw_note(bad, frontmatter, body="second")
    second = quarantine_invalid_notes(find_schema_issues(str(context)), str(tmp_path))

    first_path = tmp_path / str(first[0]["quarantine_path"])
    second_path = tmp_path / str(second[0]["quarantine_path"])
    assert first_path != second_path
    assert "first" in first_path.read_text(encoding="utf-8")
    assert "second" in second_path.read_text(encoding="utf-8")
    manifest = tmp_path / "state" / "quarantine" / "manifest.jsonl"
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 2


def test_invalid_slug_is_report_only_even_when_quarantine_enabled(tmp_path):
    context = tmp_path / "context"
    note = context / "eng" / "legacy-name.md"
    _write_raw_note(
        note,
        "name: Legacy_Name\ndescription: legacy\ntype: project\ndomain: eng",
    )
    original = note.read_bytes()

    summary = run(root=str(tmp_path), quarantine=True)

    assert note.is_file()
    assert note.read_bytes() == original
    assert summary["quarantined"] == []
    assert any(issue["code"] == "invalid_slug" for issue in summary["schema_issues"])


def test_superseded_boolean_requires_explicit_historical_state(tmp_path):
    context = tmp_path / "context"
    _write_raw_note(
        context / "eng" / "legacy.md",
        "name: legacy\ndescription: legacy\ntype: state\ndomain: eng\n"
        "superseded: true",
    )
    assert [issue["code"] for issue in find_schema_issues(str(context))] == [
        "unscoped_superseded_flag"
    ]

    _write_raw_note(
        context / "eng" / "explicit.md",
        "name: explicit\ndescription: explicit\ntype: state\ndomain: eng\n"
        "state: superseded\nsuperseded: true",
    )
    issues = find_schema_issues(str(context))
    assert [issue["path"] for issue in issues] == ["eng/legacy.md"]


def test_write_gate_rejects_edge_materialized_boolean(tmp_path):
    with pytest.raises(ValueError, match="state: superseded"):
        write_note(Note(
            name="bad", description="bad", type="state", domain="eng",
            superseded=True,
        ), str(tmp_path / "context"))

    with pytest.raises(PermissionError, match="content-bound trust transition"):
        write_note(Note(
            name="historic", description="historic", type="state", domain="eng",
            superseded=True, state="superseded",
        ), str(tmp_path / "context"))
    path, _reviewed = write_reviewed_note(tmp_path, Note(
        name="historic", description="historic", type="state", domain="eng",
        state="superseded",
    ))
    assert Path(path).is_file()


def test_schema_reports_every_path_for_duplicate_logical_name(tmp_path):
    context = tmp_path / "context"
    frontmatter = "name: duplicate\ndescription: valid\ntype: project"
    _write_raw_note(context / "eng" / "duplicate.md", frontmatter + "\ndomain: eng")
    _write_raw_note(
        context / "company" / "duplicate.md",
        frontmatter + "\ndomain: company",
    )

    issues = [issue for issue in find_schema_issues(str(context))
              if issue["code"] == "duplicate_name"]

    assert [issue["path"] for issue in issues] == [
        "company/duplicate.md", "eng/duplicate.md",
    ]
    assert all(issue["quarantinable"] is False for issue in issues)


def test_description_contract_bounds_writes_and_reports_legacy_dump(tmp_path):
    context = tmp_path / "context"
    normalized = concise_description("  title\n" + "word " * 100)
    assert "\n" not in normalized
    assert len(normalized) <= MAX_DESCRIPTION_CHARS

    with pytest.raises(ValueError, match="description exceeds"):
        write_note(Note(
            name="too-long", description="x" * (MAX_DESCRIPTION_CHARS + 1),
            type="reference", domain="eng",
        ), str(context))

    legacy = context / "eng" / "legacy.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        "---\nname: legacy\ndescription: "
        + "x" * (MAX_DESCRIPTION_CHARS + 1)
        + "\ntype: reference\ndomain: eng\n---\n\nfull source stays here\n",
        encoding="utf-8",
    )
    issues = find_schema_issues(str(context))
    assert [issue["code"] for issue in issues] == ["description_too_long"]
