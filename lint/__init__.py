"""lint — static analysis for the Remembrancer context store.

Checks: schema/quarantine, orphans, dangling links, ACL-safe file-back,
superseded marking, contradiction detection, gap reporting, and an
orchestrator that ties them all together.

Public API:
    find_orphans(notes) -> list[str]
    find_dangling(notes) -> list[tuple[str,str]]
    mark_superseded(notes, context_dir) -> list[str]
    find_contradictions(notes) -> list[tuple[str,str]]
    write_contradiction_notes(pairs, context_dir) -> list[str]
    gap_report(notes) -> list[str]
    run(root=".") -> dict
"""
from .engine import (
    VALID_DOMAINS,
    QuarantineLimitExceeded,
    find_schema_issues,
    quarantine_invalid_notes,
    find_orphans,
    find_dangling,
    find_unsafe_forward_links,
    analyze_dossier_acl,
    file_back_links,
    mark_superseded,
    find_contradictions,
    reconcile_contradictions,
    write_contradiction_notes,
    gap_report,
    run,
)

__all__ = [
    "VALID_DOMAINS",
    "QuarantineLimitExceeded",
    "find_schema_issues",
    "quarantine_invalid_notes",
    "find_orphans",
    "find_dangling",
    "find_unsafe_forward_links",
    "analyze_dossier_acl",
    "file_back_links",
    "mark_superseded",
    "find_contradictions",
    "reconcile_contradictions",
    "write_contradiction_notes",
    "gap_report",
    "run",
]
