"""lint.engine — Remembrancer static-analysis checks.

All functions are stdlib-only, deterministic, and idempotent.

Public API:
    find_orphans(notes) -> list[str]
    find_dangling(notes) -> list[tuple[str,str]]
    mark_superseded(notes, context_dir) -> list[str]
    find_contradictions(notes) -> list[tuple[str,str]]
    write_contradiction_notes(pairs, context_dir) -> list[str]
    gap_report(notes) -> list[str]
    find_entity_issues(notes, registry_path=...) -> dict[str, list]
    find_dossier_acl_drift(notes, registry_path=...) -> list[tuple[str, list, list]]
    write_entity_lint_report(issues, dossier_drift, context_dir) -> str
    run(root=".") -> dict
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from remembrancerlib import (
    ACLConflict,
    MAX_DESCRIPTION_CHARS,
    Note,
    acl_can_reveal,
    combine_acl,
    current_note_names,
    effective_state,
    index_notes_by_name,
    load_notes,
    parse_frontmatter,
    regen_index,
    slugify,
    supersession_head_active,
    write_note,
)
from remembrancerlib.note import (
    SOURCE_LINEAGE_RE,
    VALID_DOMAINS,
    VALID_NAME_RE,
    VALID_STATES,
    VALID_TYPES,
)
from remembrancerlib.dossier import (
    DOSSIER_METADATA_MARKER,
    DOSSIER_PROFILE_PLACEHOLDER,
    DOSSIER_PROFILE_VERSION,
    DOSSIER_SELECTOR_VERSION,
    dossier_base_name,
    dossier_profile_claims_fingerprint,
    dossier_profile_description,
    dossier_profile_prompt_material,
    dossier_slug,
    dossier_source_fingerprint,
    extract_dossier_profile,
    normalize_dossier_profile_claims,
    render_dossier_profile,
)
from remembrancerlib.registry_trust import (
    audit_registry_records,
    registry_record_quarantine_reasons,
)
from remembrancerlib.producer_lifecycle import (
    _contradiction_identity,
    contradiction_lifecycle_evidence,
    producer_lifecycle_valid,
    record_contradiction_restoration,
    record_contradiction_supersession,
)
from remembrancerlib.locks import pipeline_mutation_lock, serialized_pipeline_mutation
from distill.engine import (
    ComponentResolver,
    _parse_initiatives_fallback,
    _propose_replacement,
    _requires_replacement_proposal,
    _safe_load_yaml,
    load_spec,
)
from distill.entities import canonicalize, Canonicalizer, _load_registry_records
from tools.generated_notes import (
    bind_generated_verification,
    contradiction_producer_grounding,
)
from tools.source_eligibility import eligible_source_notes


_INDEX_FILENAMES = {"INDEX.md", "index.md", "_lint_report.md"}
_REQUIRED_ISSUE_CODES = {
    "missing_name", "missing_description", "missing_type",
}


class QuarantineLimitExceeded(RuntimeError):
    """Raised before mutation when a quarantine pass exceeds its safety cap."""


def _schema_issue(
    rel_path: str,
    code: str,
    message: str,
    *,
    field: str = "",
    quarantinable: bool = False,
) -> dict[str, object]:
    return {
        "path": rel_path,
        "code": code,
        "field": field,
        "message": message,
        "quarantinable": quarantinable,
    }


def find_schema_issues(context_dir: str) -> list[dict[str, object]]:
    """Validate note frontmatter without mutating the corpus.

    Required fields are read from the raw frontmatter rather than from
    :class:`Note`, whose backwards-compatible defaults intentionally hide a
    missing ``name``/``type``.  Results are stable, JSON-serialisable records.

    Only missing required fields are automatically quarantinable.  Invalid
    historical slugs/domains/types are reported for an explicit migration;
    this prevents a first lint rollout from silently relocating a large
    existing corpus.
    """
    issues: list[dict[str, object]] = []
    name_paths: dict[str, list[str]] = {}
    context_abs = os.path.abspath(context_dir)
    if not os.path.isdir(context_abs):
        return issues

    for dirpath, dirnames, filenames in os.walk(context_abs):
        dirnames.sort()
        for filename in sorted(filenames):
            if not filename.endswith(".md") or filename in _INDEX_FILENAMES:
                continue
            path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(path, context_abs).replace(os.sep, "/")
            try:
                with open(path, encoding="utf-8") as fh:
                    fm, _body = parse_frontmatter(fh.read())
            except (OSError, UnicodeError) as exc:
                issues.append(_schema_issue(
                    rel_path, "unreadable", f"note cannot be read: {exc}"
                ))
                continue

            if not fm:
                issues.append(_schema_issue(
                    rel_path,
                    "missing_frontmatter",
                    "note has no parseable frontmatter",
                ))
                continue

            for field in ("name", "description", "type"):
                value = str(fm.get(field, "") or "").strip()
                if not value:
                    issues.append(_schema_issue(
                        rel_path,
                        f"missing_{field}",
                        f"required field {field!r} is empty or absent",
                        field=field,
                        quarantinable=True,
                    ))

            description = str(fm.get("description", "") or "").strip()
            if len(description) > MAX_DESCRIPTION_CHARS:
                issues.append(_schema_issue(
                    rel_path,
                    "description_too_long",
                    f"description has {len(description)} characters; maximum is "
                    f"{MAX_DESCRIPTION_CHARS}",
                    field="description",
                ))

            name = str(fm.get("name", "") or "").strip()
            if name:
                name_paths.setdefault(name, []).append(rel_path)
            if name and VALID_NAME_RE.fullmatch(name) is None:
                issues.append(_schema_issue(
                    rel_path,
                    "invalid_slug",
                    f"name {name!r} must match [a-z0-9][a-z0-9-]*",
                    field="name",
                ))

            note_type = str(fm.get("type", "") or "").strip()
            if note_type and note_type not in VALID_TYPES:
                issues.append(_schema_issue(
                    rel_path,
                    "invalid_type",
                    f"type {note_type!r} is not a supported producer type",
                    field="type",
                ))

            if "state" in fm and fm.get("state") not in VALID_STATES:
                issues.append(_schema_issue(
                    rel_path,
                    "invalid_state",
                    f"state {fm.get('state')!r} is not a supported trust state",
                    field="state",
                ))

            if "source_lineages" in fm:
                raw_lineages = fm.get("source_lineages")
                invalid_lineages = (
                    not isinstance(raw_lineages, list)
                    or any(
                        not isinstance(value, str)
                        or SOURCE_LINEAGE_RE.fullmatch(value) is None
                        for value in (raw_lineages if isinstance(raw_lineages, list) else [])
                    )
                    or (
                        isinstance(raw_lineages, list)
                        and len(set(raw_lineages)) != len(raw_lineages)
                    )
                )
                if invalid_lineages:
                    issues.append(_schema_issue(
                        rel_path,
                        "invalid_source_lineages",
                        "source_lineages must be a unique list of canonical mutable-source keys",
                        field="source_lineages",
                    ))

            if "producer_lifecycle" in fm:
                parsed_note = Note.from_frontmatter(fm, _body, path)
                if getattr(
                    parsed_note, "_producer_lifecycle_schema_valid", True,
                ) is not True:
                    issues.append(_schema_issue(
                        rel_path,
                        "invalid_producer_lifecycle",
                        "producer_lifecycle must be one compact JSON object",
                        field="producer_lifecycle",
                    ))

            raw_superseded = fm.get("superseded", False)
            has_superseded_flag = (
                raw_superseded is True
                or str(raw_superseded).strip().lower() in {"true", "yes", "1"}
            )
            if has_superseded_flag and fm.get("state") != "superseded":
                issues.append(_schema_issue(
                    rel_path,
                    "unscoped_superseded_flag",
                    "superseded: true requires explicit state: superseded; "
                    "edge-derived target flags must be migrated",
                    field="superseded",
                ))

            domain = str(fm.get("domain", "") or "").strip()
            if not domain:
                issues.append(_schema_issue(
                    rel_path,
                    "missing_domain",
                    "domain is empty or absent",
                    field="domain",
                ))
            elif domain not in VALID_DOMAINS:
                issues.append(_schema_issue(
                    rel_path,
                    "invalid_domain",
                    f"domain {domain!r} is not supported",
                    field="domain",
                ))

    for name, paths in sorted(name_paths.items()):
        if len(paths) < 2:
            continue
        locations = ", ".join(sorted(paths))
        for rel_path in sorted(paths):
            issues.append(_schema_issue(
                rel_path,
                "duplicate_name",
                f"logical note name {name!r} is duplicated at: {locations}",
                field="name",
            ))

    return sorted(issues, key=lambda issue: (str(issue["path"]), str(issue["code"])))


def quarantine_invalid_notes(
    issues: list[dict[str, object]],
    root: str,
    *,
    max_notes: int = 10,
) -> list[dict[str, object]]:
    """Move notes missing required fields into an auditable quarantine.

    Files move from ``context/...`` to ``state/quarantine/context/...`` and an
    append-only JSONL manifest records the original path, destination, issue
    codes and SHA-256.  The safety cap is checked for the whole batch before
    the first move, so a bad validator/config cannot cause a mass migration.

    Slug/domain/type violations remain report-only: they often require a
    coordinated link/path migration and are not safe to repair by moving a
    file in isolation.
    """
    if max_notes < 0:
        raise ValueError("max_notes must be non-negative")

    root_abs = os.path.abspath(root)
    context_abs = os.path.join(root_abs, "context")
    by_path: dict[str, set[str]] = {}
    for issue in issues:
        code = str(issue.get("code", ""))
        if code not in _REQUIRED_ISSUE_CODES or not issue.get("quarantinable"):
            continue
        rel_path = str(issue.get("path", ""))
        if rel_path:
            by_path.setdefault(rel_path, set()).add(code)

    if len(by_path) > max_notes:
        raise QuarantineLimitExceeded(
            f"refusing to quarantine {len(by_path)} notes; safety cap is {max_notes}"
        )

    quarantine_root = os.path.join(root_abs, "state", "quarantine")
    manifest_path = os.path.join(quarantine_root, "manifest.jsonl")
    records: list[dict[str, object]] = []

    for rel_path, codes in sorted(by_path.items()):
        source = os.path.abspath(os.path.join(context_abs, rel_path.replace("/", os.sep)))
        if os.path.commonpath((context_abs, source)) != context_abs:
            raise ValueError(f"quarantine source escapes context: {rel_path!r}")
        if os.path.islink(source):
            raise ValueError(f"refusing to quarantine symlink: {rel_path!r}")
        context_real = os.path.realpath(context_abs)
        source_real = os.path.realpath(source)
        if os.path.commonpath((context_real, source_real)) != context_real:
            raise ValueError(f"quarantine source resolves outside context: {rel_path!r}")
        if not os.path.isfile(source):
            continue

        with open(source, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()

        destination = os.path.join(quarantine_root, "context", rel_path.replace("/", os.sep))
        if os.path.exists(destination):
            stem, suffix = os.path.splitext(destination)
            candidate = f"{stem}-{digest[:12]}{suffix}"
            counter = 2
            while os.path.exists(candidate):
                candidate = f"{stem}-{digest[:12]}-{counter}{suffix}"
                counter += 1
            destination = candidate
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        os.makedirs(quarantine_root, exist_ok=True)

        record: dict[str, object] = {
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
            "original_path": os.path.relpath(source, root_abs).replace(os.sep, "/"),
            "quarantine_path": os.path.relpath(destination, root_abs).replace(os.sep, "/"),
            "issues": sorted(codes),
            "sha256": digest,
        }

        os.replace(source, destination)
        try:
            line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            fd = os.open(manifest_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                view = memoryview(line)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short write while appending quarantine manifest")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            # Preserve the stronger invariant: never move a note without an
            # audit record.  Best-effort rollback, then surface the failure.
            if os.path.exists(destination) and not os.path.exists(source):
                os.makedirs(os.path.dirname(source), exist_ok=True)
                os.replace(destination, source)
            raise
        records.append(record)

    return records


def write_schema_lint_report(
    issues: list[dict[str, object]],
    quarantined: list[dict[str, object]],
    context_dir: str,
) -> str:
    """Append a compact schema/quarantine section to the lint report."""
    os.makedirs(context_dir, exist_ok=True)
    report_path = os.path.join(context_dir, "_lint_report.md")
    lines = ["", "## Note Schema", ""]
    if not issues:
        lines.append("_(no note schema issues detected)_")
    else:
        for issue in issues[:100]:
            lines.append(
                f"- `{issue['path']}` — **{issue['code']}**: {issue['message']}"
            )
        if len(issues) > 100:
            lines.append(f"- _(+{len(issues) - 100} more)_")
    lines += ["", "### Quarantine", ""]
    if quarantined:
        for record in quarantined:
            lines.append(
                f"- `{record['original_path']}` → `{record['quarantine_path']}` "
                f"(sha256 `{str(record['sha256'])[:12]}…`)"
            )
    else:
        lines.append("_(no notes quarantined; schema audit is non-destructive by default)_")
    lines.append("")
    with open(report_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return report_path


def _load_schema_valid_notes(
    context_dir: str,
    issues: list[dict[str, object]],
) -> list[Note]:
    """Load notes while withholding schema offenders from auto-mutations."""
    context_abs = os.path.abspath(context_dir)
    excluded = {
        os.path.abspath(os.path.join(context_abs, str(issue["path"]).replace("/", os.sep)))
        for issue in issues
        if issue.get("path")
    }
    return [note for note in load_notes(context_dir) if os.path.abspath(note.path) not in excluded]


# ---------------------------------------------------------------------------
# Orphan + dangling-link detection
# ---------------------------------------------------------------------------

def find_orphans(notes: list[Note]) -> list[str]:
    """Return names of notes that have no inbound link from any other note.

    A note is an orphan if no other note's ``typed_links`` targets it.
    Notes with ``type == "contradiction"`` are excluded — they are lint-generated
    and would otherwise pollute the orphan signal on every re-run.
    """
    all_names: set[str] = {n.name for n in notes if n.type != "contradiction"}
    # Build the set of names that appear as link targets
    referenced: set[str] = set()
    for note in notes:
        for _rel, target in note.typed_links:
            referenced.add(target)
    return sorted(name for name in all_names if name not in referenced)


def find_dangling(notes: list[Note]) -> list[tuple[str, str]]:
    """Return (source_name, target_name) pairs where the target note does not exist.

    Only link targets that match no known note name are considered dangling.
    Pairs are deduped and returned in sorted order (mirrors find_contradictions).
    """
    all_names: set[str] = {n.name for n in notes}
    seen: set[tuple[str, str]] = set()
    dangling: list[tuple[str, str]] = []
    for note in notes:
        for _rel, target in note.typed_links:
            if target not in all_names:
                pair = (note.name, target)
                if pair not in seen:
                    seen.add(pair)
                    dangling.append(pair)
    return sorted(dangling)


def _backlink_acl_safe(source: Note, target: Note) -> bool:
    """Whether exposing ``source.name`` inside ``target`` is ACL-safe.

    ACL labels are OR grants.  A public source can be named anywhere.  A
    restricted source may only be named in a restricted target whose every
    grant label is also a source grant; otherwise a target-only reader could
    learn the existence/name of memory they cannot retrieve.
    """
    return acl_can_reveal(target.acl, source.acl)


def find_unsafe_forward_links(notes: list[Note]) -> list[tuple[str, str, str]]:
    """Return links whose readable container reveals a less-visible target."""
    groups: dict[str, list[Note]] = {}
    for note in notes:
        groups.setdefault(note.name, []).append(note)
    by_name = {name: group[0] for name, group in groups.items() if len(group) == 1}
    unsafe: set[tuple[str, str, str]] = set()
    for source in notes:
        for relation, target_name in source.typed_links:
            target = by_name.get(target_name)
            if target is None:
                continue
            if not acl_can_reveal(source.acl, target.acl):
                unsafe.add((source.name, relation, target_name))
    return sorted(unsafe)


def file_back_links(
    notes: list[Note],
    context_dir: str,
    *,
    enabled: bool = True,
    max_writes: int = 50,
    max_links_per_note: int = 50,
) -> dict[str, object]:
    """Add missing reciprocal wiki-links with ACL-safe bounded writes.

    Backlinks are deliberately untyped ``[[source]]`` links.  Mirroring a
    directional relation such as ``supersedes`` would invert its meaning.
    Writes are capped per pass so enabling the feature on a legacy corpus
    cannot rewrite thousands of files at once; subsequent passes converge.
    """
    result: dict[str, object] = {
        "enabled": bool(enabled),
        "written": [],
        "skipped_acl": [],
        "skipped_acl_total": 0,
        "ambiguous_names": [],
        "deferred": 0,
        "deferred_links": 0,
        "review_required_links": 0,
        "review_proposals": [],
    }
    if not enabled:
        return result
    if max_writes < 0:
        raise ValueError("max_writes must be non-negative")
    if max_links_per_note < 1:
        raise ValueError("max_links_per_note must be positive")

    name_groups: dict[str, list[Note]] = {}
    for note in notes:
        name_groups.setdefault(note.name, []).append(note)
    ambiguous_names = sorted(name for name, group in name_groups.items() if len(group) != 1)
    result["ambiguous_names"] = ambiguous_names
    ambiguous_set = set(ambiguous_names)
    by_name = {name: group[0] for name, group in name_groups.items() if len(group) == 1}
    pending: dict[str, set[str]] = {}
    skipped_acl: set[tuple[str, str]] = set()
    machine_proposals = {
        note.name for note in notes
        if any(str(item).startswith("machine-update:")
               for item in (note.provenance or []))
    }
    for source in sorted(notes, key=lambda note: note.name):
        # Pending replacement proposals are review artifacts, not established
        # graph nodes. Filing their control/source links back would recursively
        # create proposals (and can even manufacture self-links).
        if source.name in ambiguous_set or source.name in machine_proposals:
            continue
        for _relation, target_name in source.typed_links:
            if not target_name or target_name == source.name:
                continue
            target = by_name.get(target_name)
            if target is None or target.name in machine_proposals:
                continue
            target_links = {name for _rel, name in target.typed_links}
            if source.name in target_links:
                continue
            if not _backlink_acl_safe(source, target):
                skipped_acl.add((source.name, target.name))
                continue
            pending.setdefault(target.name, set()).add(source.name)

    target_names = sorted(pending)
    to_write = target_names[:max_writes]
    result["deferred"] = max(0, len(target_names) - len(to_write))
    result["skipped_acl_total"] = len(skipped_acl)
    result["skipped_acl"] = sorted(skipped_acl)[:100]
    written: list[str] = []
    review_proposals: list[dict[str, str]] = []
    review_required_links = 0
    deferred_links = 0
    for target_name in to_write:
        target = by_name[target_name]
        all_sources = sorted(pending[target_name])
        sources = all_sources[:max_links_per_note]
        deferred_links += len(all_sources) - len(sources)
        desired = copy.deepcopy(target)
        prefix = desired.body.rstrip()
        separator = "\n\n" if prefix else ""
        desired.body = prefix + separator + "Backlinks: " + " ".join(
            f"[[{source_name}]]" for source_name in sources
        ) + "\n"
        if _requires_replacement_proposal(target):
            try:
                decision = _propose_replacement(
                    target, desired, context_dir, by_name, "file-back",
                    detail="backlink update requires review",
                )
            except ValueError:
                # `_propose_replacement` refuses when the target already has an
                # unretired head that is not a machine update -- a second,
                # competing replacement would be genuinely wrong, so the refusal
                # is correct. What is not correct is letting ONE such target
                # abort the whole corpus pass: this is the same failure the
                # PermissionError branch below already handles ("one held target
                # must not crash the whole lint stage").
                #
                # A note can be targeted by a CALENDAR note whose provenance is
                # a gcal event rather than a machine-update marker; the guard
                # then fires and the ValueError propagates out of lint ->
                # `stage=lint` -> `exit 1` -> the entire scheduled run is
                # discarded, including digest work already merged. Several notes
                # can carry more than one unretired head, so this recurs.
                #
                # Route it to the attended queue, exactly as the sibling branch.
                review_required_links += len(sources)
                review_proposals.append({
                    "target": target_name,
                    "proposal": "",
                })
                continue
            review_required_links += len(sources)
            review_proposals.append({
                "target": target_name,
                "proposal": decision.note_name,
            })
            if decision.action != "IGNORE":
                written.append(decision.note_name)
        else:
            desired.state = "proposed"
            try:
                write_note(desired, context_dir)
            except PermissionError:
                # A review-held proposal cannot be rewritten without its exact
                # governance capability. One held target must not crash the
                # whole lint stage —
                # route it into the attended file-back review queue instead.
                review_required_links += len(sources)
                review_proposals.append({
                    "target": target_name,
                    "proposal": "",
                })
                continue
            by_name[target_name] = desired
            written.append(target_name)
    result["written"] = written
    result["review_required_links"] = review_required_links
    result["review_proposals"] = review_proposals
    # Targets deferred wholesale also defer all their missing links.
    deferred_links += sum(len(pending[name]) for name in target_names[max_writes:])
    result["deferred_links"] = deferred_links
    return result


# ---------------------------------------------------------------------------
# Superseded marking
# ---------------------------------------------------------------------------

def mark_superseded(notes: list[Note], context_dir: str) -> list[str]:
    """Report targets of active supersession edges without rewriting them.

    The historical function name is retained for API compatibility.  Current
    truth is now projected dynamically at retrieval time from caller-visible,
    explicitly verified heads.  Lint must never materialize an edge by setting
    ``superseded: true`` on its target: doing so would survive a later demotion
    or tombstone and incorrectly keep the old note hidden.

    ``context_dir`` is intentionally unused.  Explicit retirement producers
    (for example an authoritative Calendar cancellation) may still write both
    ``state: superseded`` and the compatibility boolean on the note itself.
    """
    del context_dir
    name_map: dict[str, Note] = {n.name: n for n in notes}
    active_targets: set[str] = set()
    for note in notes:
        for rel, target in note.typed_links:
            if rel != "supersedes" or target not in name_map:
                continue
            if (supersession_head_active(note)
                    and acl_can_reveal(note.acl, name_map[target].acl)):
                active_targets.add(target)
    return sorted(active_targets)


def find_supersession_issues(notes: list[Note]) -> list[dict[str, str]]:
    """Audit legacy verified edges that current projection must ignore."""
    groups: dict[str, list[Note]] = {}
    for note in notes:
        groups.setdefault(note.name, []).append(note)
    by_name = {name: group[0] for name, group in groups.items() if len(group) == 1}
    issues: set[tuple[str, str, str]] = set()
    valid_edges: list[tuple[str, str]] = []
    heads_by_target: dict[str, set[str]] = {}
    for head in by_name.values():
        if not supersession_head_active(head):
            continue
        for relation, target_name in head.typed_links:
            if relation != "supersedes":
                continue
            target = by_name.get(target_name)
            if target_name == head.name:
                issues.add(("self_cycle", head.name, target_name))
            elif target is None:
                issues.add(("dangling_target", head.name, target_name))
            elif not acl_can_reveal(head.acl, target.acl):
                issues.add(("acl_incompatible", head.name, target_name))
            else:
                valid_edges.append((head.name, target_name))
                heads_by_target.setdefault(target_name, set()).add(head.name)
    for target, heads in heads_by_target.items():
        if len(heads) > 1:
            for head in heads:
                issues.add(("parallel_heads", head, target))

    graph: dict[str, set[str]] = {}
    for head, target in valid_edges:
        graph.setdefault(head, set()).add(target)
    visiting: list[str] = []
    visited: set[str] = set()

    def walk(node: str) -> None:
        if node in visiting:
            cycle = visiting[visiting.index(node):] + [node]
            for left, right in zip(cycle, cycle[1:]):
                issues.add(("cycle", left, right))
            return
        if node in visited:
            return
        visiting.append(node)
        for target in graph.get(node, set()):
            walk(target)
        visiting.pop()
        visited.add(node)

    for node in sorted(graph):
        walk(node)
    return [
        {"code": code, "head": head, "target": target}
        for code, head, target in sorted(issues)
    ]


# ---------------------------------------------------------------------------
# Contradiction detection + contradiction notes
# ---------------------------------------------------------------------------

_ANTONYM_PAIRS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"enabled"}), frozenset({"disabled"})),
    (frozenset({"active"}), frozenset({"inactive"})),
    (frozenset({"available"}), frozenset({"unavailable"})),
    (frozenset({"allowed"}), frozenset({"blocked", "forbidden", "denied"})),
    (frozenset({"approved"}), frozenset({"rejected"})),
    (frozenset({"supported"}), frozenset({"unsupported"})),
    (frozenset({"required"}), frozenset({"optional"})),
    (frozenset({"open"}), frozenset({"closed"})),
    (frozenset({"online"}), frozenset({"offline"})),
    (frozenset({"public"}), frozenset({"private"})),
    (frozenset({"present"}), frozenset({"absent"})),
    (frozenset({"complete", "completed"}), frozenset({"incomplete"})),
    (frozenset({"resolved"}), frozenset({"unresolved"})),
    (frozenset({"success", "successful"}), frozenset({"failure", "failed"})),
    (frozenset({"true"}), frozenset({"false"})),
    (frozenset({"valid"}), frozenset({"invalid"})),
)
_POLARITY_WORDS = frozenset(
    word for positive, negative in _ANTONYM_PAIRS for word in positive | negative
)
_CONTRADICTION_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "have", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "this", "to", "was", "were", "will", "with",
})


def _description_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _polarity_context(
    words: list[str],
    polarity: frozenset[str],
    ignored: frozenset[str] | set[str],
    *,
    radius: int = 4,
) -> set[str]:
    """Meaningful words close to a polarity claim, not anywhere in the note."""
    context: set[str] = set()
    for index, word in enumerate(words):
        if word not in polarity:
            continue
        start = max(0, index - radius)
        end = min(len(words), index + radius + 1)
        context.update(words[start:end])
    return context - set(ignored)


def _same_entity_antonym(a: Note, b: Note) -> bool:
    """Conservative same-entity + antonym heuristic.

    Besides a shared canonical-looking entity token, the descriptions must
    carry opposite sides of one explicit antonym pair.  A shared subject token
    is also required unless both descriptions reduce to the entity itself and
    the polarity word.  This avoids flagging unrelated properties of a large
    dossier merely because one is "enabled" and another "disabled".
    """
    if a.type == "contradiction" or b.type == "contradiction":
        return False
    if effective_state(a) == "superseded" or effective_state(b) == "superseded":
        return False
    entities_a = {str(entity).strip().lower() for entity in a.entities if str(entity).strip()}
    entities_b = {str(entity).strip().lower() for entity in b.entities if str(entity).strip()}
    shared_entities = entities_a & entities_b
    if not shared_entities:
        return False

    words_a = re.findall(r"[a-z0-9]+", (a.description or "").lower())
    words_b = re.findall(r"[a-z0-9]+", (b.description or "").lower())
    tokens_a = set(words_a)
    tokens_b = set(words_b)
    entity_words: set[str] = set()
    for entity in shared_entities:
        entity_words.update(_description_tokens(entity))
    ignored = _POLARITY_WORDS | _CONTRADICTION_STOPWORDS | entity_words

    for positive, negative in _ANTONYM_PAIRS:
        a_positive = bool(tokens_a & positive)
        a_negative = bool(tokens_a & negative)
        b_positive = bool(tokens_b & positive)
        b_negative = bool(tokens_b & negative)
        # Descriptions containing both polarities describe a transition or
        # comparison; treating those as one side creates noisy false flags.
        if (a_positive and not a_negative and b_negative and not b_positive) or (
            a_negative and not a_positive and b_positive and not b_negative
        ):
            a_side = positive if a_positive else negative
            b_side = negative if b_negative else positive
            topic_a = _polarity_context(words_a, a_side, ignored)
            topic_b = _polarity_context(words_b, b_side, ignored)
            if (topic_a & topic_b) or (not topic_a and not topic_b):
                return True
    return False

def find_contradictions(notes: list[Note]) -> list[tuple[str, str]]:
    """Return explicit and heuristic contradiction pairs.

    Each pair is normalised so the lexicographically smaller name comes first,
    and duplicates are eliminated.  In addition to a typed ``contradicts``
    edge, two live notes are candidates when they share an entity and their
    descriptions contain opposite, subject-aligned antonyms.
    """
    current = current_note_names(notes)
    notes = [note for note in notes if note.name in current]
    seen: set[tuple[str, str]] = set()
    for note in notes:
        for rel, target in note.typed_links:
            if rel == "contradicts" and target in current and target != note.name:
                seen.add(tuple(sorted((note.name, target))))

    # Generate heuristic candidates through the entity index rather than an
    # O(N^2) corpus scan.  A pair sharing multiple entities is checked once.
    by_entity: dict[str, list[Note]] = {}
    for note in notes:
        if note.type == "contradiction":
            continue
        for entity in {str(item).strip().lower() for item in note.entities if str(item).strip()}:
            by_entity.setdefault(entity, []).append(note)
    heuristic_candidates: set[tuple[str, str]] = set()
    note_by_name = {note.name: note for note in notes}
    for entity_notes in by_entity.values():
        ordered = sorted(entity_notes, key=lambda note: note.name)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                if left.name != right.name:
                    heuristic_candidates.add((left.name, right.name))
    for left_name, right_name in sorted(heuristic_candidates):
        if _same_entity_antonym(note_by_name[left_name], note_by_name[right_name]):
            seen.add((left_name, right_name))
    return sorted(seen)


def contradictions_by_note(notes: list[Note]) -> dict[str, list[str]]:
    """Per-note view of `find_contradictions`: name -> sorted [names it contradicts].

    Symmetric — both endpoints of every `contradicts` pair get the edge, so a note that is
    merely the *target* of a contradiction still surfaces it. Pure/additive: consumed by the
    confidence score (which it caps) and inline signal badges. Built from the same typed-link
    logic as `find_contradictions`, so the two never disagree.
    """
    out: dict[str, set[str]] = {}
    for a, b in find_contradictions(notes):
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return {name: sorted(targets) for name, targets in out.items()}


def _load_contradiction_policy(context_dir: str) -> str:
    """Read lint.contradiction_policy from config.json; default 'flag'.

    Checks context_dir/config.json first, then walks up parent directories.
    """
    # First: check directly inside context_dir
    direct_cfg = os.path.join(os.path.abspath(context_dir), "config.json")
    if os.path.isfile(direct_cfg):
        try:
            with open(direct_cfg) as f:
                cfg = json.load(f)
            return cfg.get("lint", {}).get("contradiction_policy", "flag")
        except (json.JSONDecodeError, OSError):
            return "flag"

    # Fallback: walk up from context_dir's parent
    candidate = os.path.dirname(os.path.abspath(context_dir))
    for _ in range(5):
        cfg_path = os.path.join(candidate, "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                return cfg.get("lint", {}).get("contradiction_policy", "flag")
            except (json.JSONDecodeError, OSError):
                return "flag"
        candidate = os.path.dirname(candidate)
    return "flag"


def reconcile_contradictions(
    pairs: list[tuple[str, str]],
    notes: list[Note],
    context_dir: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Resolve only contradictions with an unambiguous supersession signal.

    A date alone is not sufficient evidence that the newer prose is true.  We
    therefore auto-reconcile only when exactly one endpoint explicitly
    ``supersedes`` the other through an active verified head. The target is
    hidden dynamically for callers who can see that head; reconciliation does
    not materialize a boolean on the target. Every
    uncertain/ambiguous pair is returned for the policy's required ``flag``
    fallback.
    """
    by_name = {note.name: note for note in notes}
    resolved: list[tuple[str, str]] = []
    unresolved: list[tuple[str, str]] = []
    for pair in pairs:
        a_name, b_name = pair
        a = by_name.get(a_name)
        b = by_name.get(b_name)
        if a is None or b is None:
            unresolved.append(pair)
            continue
        a_supersedes_b = ("supersedes", b_name) in a.typed_links
        b_supersedes_a = ("supersedes", a_name) in b.typed_links
        if a_supersedes_b == b_supersedes_a:
            unresolved.append(pair)
            continue
        head = a if a_supersedes_b else b
        target = b if a_supersedes_b else a
        if (not supersession_head_active(head)
                or not acl_can_reveal(head.acl, target.acl)):
            unresolved.append(pair)
            continue
        resolved.append(pair)
    return sorted(resolved), sorted(unresolved)


def _safe_combined_acl(notes: list[Note]) -> list[str]:
    """Return a conservative ACL for a note that reveals all input notes."""
    return combine_acl(*(list(note.acl or []) for note in notes))


def _contradiction_note_name(a: str, b: str) -> str:
    """Stable collision-resistant name that remains within slugify's 80 chars."""
    base = f"contradiction-{slugify(a)}-vs-{slugify(b)}"
    if len(base) <= 80:
        return base
    digest = hashlib.sha256(f"{a}\0{b}".encode("utf-8")).hexdigest()[:10]
    return f"{base[:69].rstrip('-')}-{digest}"


_CONTRADICTION_PRODUCER_ID = "contradiction-lint"
_CONTRADICTION_PRODUCER = f"generated-by:{_CONTRADICTION_PRODUCER_ID}"


def _contradiction_machine_owned(note: Note) -> bool:
    """True only for the exact closed contradiction producer contract."""
    try:
        _contradiction_identity(note)
    except (PermissionError, ValueError):
        return False
    return True


def _sweep_contradiction_lifecycle(
    pairs: list[tuple[str, str]], notes: list[Note], context_dir: str,
) -> list[str]:
    """Resolve stale generated flags and reopen one only if its pair returns."""
    current = {tuple(sorted(pair)) for pair in pairs}
    changed: list[str] = []
    notes_by_name = index_notes_by_name(notes, operation="contradiction lifecycle")
    root = os.path.dirname(os.path.abspath(context_dir))
    for note in list(notes):
        structured = note.structured or {}
        pair = structured.get("pair")
        if (note.type != "contradiction"
                or structured.get("kind") != "contradiction"
                or not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(value, str) for value in pair)):
            continue
        normalized = tuple(sorted(pair))
        machine_owned = _contradiction_machine_owned(note)
        evidence = (
            contradiction_lifecycle_evidence(note, root=root)
            if machine_owned else None
        )
        is_active = evidence.active if evidence is not None else normalized in current
        if is_active and note.state == "superseded":
            if not producer_lifecycle_valid(note):
                # Raw/human historical state is outside deterministic authority.
                continue
            capability = record_contradiction_restoration(
                note, root=root,
            )
            write_note(note, context_dir, _producer_capability=capability)
            notes_by_name[note.name] = note
            changed.append(note.name)
            continue
        elif not is_active and note.state != "superseded":
            desired = copy.deepcopy(note)
            desired.state = "superseded"
            desired.superseded = True
        else:
            continue
        if _requires_replacement_proposal(note):
            # The proposal remains non-destructive; approval is the only event
            # that can hide the human-resolved flag.
            desired.superseded = False
            desired.state = "proposed"
            desired.body = desired.body.rstrip() + (
                f"\n\nLifecycle review: contradiction {'recurred' if is_active else 'is no longer detected'}.\n"
            )
            decision = _propose_replacement(
                note, desired, context_dir, notes_by_name, "contradiction-lifecycle",
                detail="contradiction lifecycle change requires review",
            )
            changed.append(decision.note_name)
        elif machine_owned:
            capability = record_contradiction_supersession(
                note, root=root,
            )
            write_note(note, context_dir, _producer_capability=capability)
            notes_by_name[note.name] = note
            changed.append(note.name)
        # A legacy/user-authored proposed collision without an ownership marker
        # is report-only; lint never claims it by rewriting control fields.
    return changed


def write_contradiction_notes(
    pairs: list[tuple[str, str]],
    context_dir: str,
    policy: str | None = None,
    notes: list[Note] | None = None,
) -> list[str]:
    """Serialize contradiction flag creation and lifecycle transitions."""
    root = os.path.dirname(os.path.abspath(context_dir))
    with pipeline_mutation_lock(root):
        return _write_contradiction_notes_locked(
            pairs, context_dir, policy=policy, notes=notes,
        )


def _write_contradiction_notes_locked(
    pairs: list[tuple[str, str]],
    context_dir: str,
    policy: str | None = None,
    notes: list[Note] | None = None,
) -> list[str]:
    """For each contradiction pair, create a ``type=contradiction`` note.

    ``reconcile`` first applies only evidence-backed supersession and then
    falls back to writing flags for every uncertain pair.  ``suppress`` is the
    sole policy that deliberately emits nothing.  Returns note names written.

    Idempotent: writing the same pair twice overwrites cleanly (same slug).

    Args:
        pairs: Sorted (a, b) contradiction pairs from find_contradictions.
        context_dir: Path to the context directory where notes are stored.
        policy: When provided, overrides the internal config.json lookup.
                Pass ``"flag"`` to force note creation, ``"reconcile"`` to
                attempt safe resolution, or ``"suppress"`` to silence it.
                When ``None`` (default),
                the policy is read from config.json as before.
        notes: Source notes, used for safe reconciliation and ACL derivation.
    """
    if policy is None:
        policy = _load_contradiction_policy(context_dir)
    if policy == "suppress":
        return []
    if policy == "reconcile" and notes is not None:
        _resolved, pairs = reconcile_contradictions(pairs, notes, context_dir)
    # Missing source notes make reconciliation uncertain, so fail closed to a
    # visible flag.  Unknown policies also use the safe/default flag behavior.

    all_notes = load_notes(context_dir)
    disk_names = {note.name for note in all_notes}
    for source in notes or []:
        if source.name not in disk_names:
            all_notes.append(source)
            disk_names.add(source.name)
    written: list[str] = _sweep_contradiction_lifecycle(pairs, all_notes, context_dir)
    by_name: dict[str, Note] = {}
    duplicate_names: set[str] = set()
    for source in all_notes:
        if source.name in by_name:
            duplicate_names.add(source.name)
        else:
            by_name[source.name] = source
    for a, b in pairs:
        note_name = _contradiction_note_name(a, b)
        if a in duplicate_names or b in duplicate_names:
            raise ACLConflict("ambiguous contradiction source cannot be ACL-scoped")
        source_notes = [by_name[name] for name in (a, b) if name in by_name]
        if len(source_notes) != 2:
            raise ACLConflict("contradiction source metadata is required for safe ACL derivation")
        existing = [note for note in all_notes if note.name == note_name]
        expected_pair = sorted([a, b])
        if existing and (len(existing) != 1
                         or existing[0].type != "contradiction"
                         or (existing[0].structured or {}).get("pair") != expected_pair):
            raise FileExistsError(
                f"refusing to overwrite non-owned contradiction name {note_name!r}"
            )
        note = Note(
            name=note_name,
            description=f"Auto-detected contradiction between {a} and {b}",
            type="contradiction",
            domain="company",
            acl=_safe_combined_acl(source_notes),
            entities=sorted({entity for source in source_notes for entity in source.entities}),
            provenance=(
                sorted({
                    item
                    for source in source_notes
                    for item in source.provenance
                    if not str(item).startswith((
                        "by:", "generated-by:", "machine-update:",
                    ))
                })
                + [_CONTRADICTION_PRODUCER]
            ),
            state="proposed",
            structured={"kind": "contradiction", "pair": expected_pair},
            body=(
                f"Auto-detected contradiction between [[{a}]] and [[{b}]].\n\n"
                f"Resolve by updating one of the source notes or reconciling the conflict."
            ),
        )
        if existing:
            prior = existing[0]
            if _requires_replacement_proposal(prior):
                notes_by_name = index_notes_by_name(
                    all_notes, operation="contradiction flagging",
                )
                decision = _propose_replacement(
                    prior, note, context_dir, notes_by_name, "contradiction-flag",
                    detail="contradiction flag refresh requires review",
                    finalize_note=lambda proposal, old=None: bind_generated_verification(
                        proposal,
                        producer=_CONTRADICTION_PRODUCER_ID,
                        grounding=contradiction_producer_grounding(
                            proposal, context_dir=context_dir,
                        ),
                        existing=old,
                    ),
                )
                if decision.note_name not in written:
                    written.append(decision.note_name)
                continue
            if not _contradiction_machine_owned(prior):
                raise FileExistsError(
                    f"refusing to overwrite non-owned contradiction name {note_name!r}"
                )
            note.path = prior.path
            bind_generated_verification(
                note,
                producer=_CONTRADICTION_PRODUCER_ID,
                grounding=contradiction_producer_grounding(
                    note, context_dir=context_dir,
                ),
                existing=prior,
            )
            if (note.to_markdown() == prior.to_markdown()
                    or (note.description == prior.description
                        and note.acl == prior.acl
                        and note.entities == prior.entities
                        and note.provenance == prior.provenance
                        and note.state == prior.state
                        and note.structured == prior.structured
                        and note.confidence == prior.confidence
                        and note.body.strip() == prior.body.strip())):
                continue
        else:
            bind_generated_verification(
                note,
                producer=_CONTRADICTION_PRODUCER_ID,
                grounding=contradiction_producer_grounding(
                    note, context_dir=context_dir,
                ),
            )
        write_note(note, context_dir)
        if note_name not in written:
            written.append(note_name)
    return written


# ---------------------------------------------------------------------------
# Gap report (missing entity pages / open questions)
# ---------------------------------------------------------------------------

def gap_report(notes: list[Note]) -> list[str]:
    """Identify entities referenced in notes that lack a ``type=entity`` page.

    Returns a list of human-readable strings:
        "missing entity page: <entity>"

    This function only computes the gap list; it does not write any file.
    Call ``write_gap_report(gaps, context_dir)`` to persist the report to
    ``context/_lint_report.md``.
    """
    # Collect all entity pages
    entity_pages: set[str] = {n.name for n in notes if n.type == "entity"}

    # Collect all referenced entities from note.entities lists
    referenced_entities: set[str] = set()
    for note in notes:
        for entity in note.entities:
            entity = entity.strip()
            if entity:
                referenced_entities.add(entity)

    missing = sorted(referenced_entities - entity_pages)
    gaps: list[str] = [f"missing entity page: {e}" for e in missing]
    return gaps


def write_gap_report(gaps: list[str], context_dir: str) -> str:
    """Write the gap report to context/_lint_report.md; return the path."""
    os.makedirs(context_dir, exist_ok=True)
    report_path = os.path.join(context_dir, "_lint_report.md")
    lines = [
        "# Remembrancer Lint Report",
        "",
        "## Missing Entity Pages",
        "",
    ]
    if gaps:
        for g in gaps:
            lines.append(f"- {g}")
            # Suggest a question
            entity = g.replace("missing entity page: ", "")
            lines.append(f"  - Suggested: _What is `{entity}`? Create an entity note._")
    else:
        lines.append("_(no missing entity pages detected)_")
    lines.append("")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


# ---------------------------------------------------------------------------
# Entity canonicalization guardrail
# ---------------------------------------------------------------------------

_ENTITY_LINT_TOP_N = 10

_RAW_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_BARE_NUMERIC_RE = re.compile(r"^\d+$")

# Legacy relationship-prefixed sections (distill.entities._TYPE_SECTIONS minus
# "people" and "system", which keep their own prefix unchanged post-split).
# Post-split, an org's id becomes `org:<slug>` with `relationship: [...]`; a
# note that still stores one of these old section-prefixed ids
# (`customer:acme-corp`, `self:initech`, ...) needs rewriting to the new id.
_LEGACY_ORG_PREFIXES = (
    "self", "vendor", "investor", "partner", "advisor",
    "champion", "customer", "friend", "academic", "recruiting", "prospect",
)


def _split_registry_active(registry_path: str) -> bool:
    """True once `registry/orgs.yml` exists next to `registry_path` — the
    signal `distill.entities` uses to detect the split registry
    is live. Legacy-prefixed ids must never be flagged before this is true."""
    d = os.path.dirname(registry_path) or "."
    return os.path.isfile(os.path.join(d, "orgs.yml"))


def _registered_components_and_initiatives(registry_path: str) -> tuple[set[str], set[str]]:
    """(registered component repos, registered initiative ids), lowercased —
    from registry/eng_components.yml + registry/initiatives.yml next to
    `registry_path`. Missing files -> empty sets (ComponentResolver's own
    loaders already degrade gracefully to `[]`)."""
    d = os.path.dirname(registry_path) or "."
    components = {
        repo.strip().lower()
        for repo, _ in ComponentResolver._load_components(os.path.join(d, "eng_components.yml"))
    }
    initiatives = {
        iid.strip().lower()
        for iid, _ in ComponentResolver._load_initiatives(os.path.join(d, "initiatives.yml"))
    }
    return components, initiatives


def find_entity_issues(
    notes: list[Note], registry_path: str = "registry/entities.yml"
) -> dict[str, list[tuple[str, ...]]]:
    """Flag notes whose ``entities:``/``components:`` values are raw/unresolved/
    misrouted instead of the canonical ids `distill.entities.Canonicalizer` /
    `distill.engine.ComponentResolver` would produce.

    Offender classes (entity-graph taxonomy):
      raw_email             — a bare email address (e.g. "jane@example.com").
      bare_numeric          — a bare numeric id (e.g. a Slack/Drive numeric id
                              like "123456789012").
      unresolved_alias      — a known alias/name that ``canonicalize`` resolves to
                              a *different* string — it should have been stored as
                              that canonical id. Entries are
                              ``(note_name, entity, canonical_id)``.
      self_outside_company  — an entity that resolves to a self entity
                              (``Canonicalizer.is_self``: a ``self:`` id in
                              legacy mode, or an ``org:`` id whose
                              ``relationship`` includes ``self`` post-split) on a
                              note whose ``domain`` is not ``"company"`` (self
                              entities are dropped from the facet outside
                              ``domain: company``). Entries are
                              ``(note_name, entity, canonical_id)``.
      wrong_facet           — an ``entities:`` value that is (or resolves via
                              ``Canonicalizer.kind_of`` to) a ``component:``/
                              ``initiative:`` id — that belongs in the
                              ``components:`` facet instead.
                              Entries are ``(note_name, entity, kind)``.
      legacy_prefixed       — an ``entities:`` value still using an OLD
                              relationship-section prefix (``customer:``,
                              ``vendor:``, ..., ``self:``) that resolves to a
                              NEW ``org:<slug>`` id — only flagged once the split
                              registry (``registry/orgs.yml``) is live, so this
                              never false-positives pre-migration. Entries are
                              ``(note_name, entity, canonical_id)``.
      unregistered_component — a ``components:`` value shaped ``component:<repo>``
                              / ``initiative:<id>`` whose ``<repo>``/``<id>``
                              isn't in ``registry/eng_components.yml`` /
                              ``registry/initiatives.yml``. Entries are
                              ``(note_name, component)``.

    A string ``canonicalize`` cannot resolve (returns ``None``) is left unflagged —
    it may be a legitimate not-yet-registered mention (the gap report's concern,
    not this guardrail's). Each class list is deduped and sorted.
    """
    raw_email: list[tuple[str, str]] = []
    bare_numeric: list[tuple[str, str]] = []
    unresolved_alias: list[tuple[str, str, str]] = []
    self_outside_company: list[tuple[str, str, str]] = []
    wrong_facet: list[tuple[str, str, str]] = []
    legacy_prefixed: list[tuple[str, str, str]] = []
    unregistered_component: list[tuple[str, str]] = []

    canon = Canonicalizer(registry_path)
    split_active = _split_registry_active(registry_path)
    registered_components, registered_initiatives = _registered_components_and_initiatives(
        registry_path
    )

    for note in notes:
        for raw_entity in note.entities:
            entity = raw_entity.strip()
            if not entity:
                continue
            if _RAW_EMAIL_RE.match(entity):
                raw_email.append((note.name, entity))
                continue
            if _BARE_NUMERIC_RE.match(entity):
                bare_numeric.append((note.name, entity))
                continue

            kind = canon.kind_of(entity)
            if kind in ("component", "initiative"):
                wrong_facet.append((note.name, entity, kind))
                continue

            if split_active and ":" in entity:
                prefix = entity.split(":", 1)[0].strip().lower()
                if prefix in _LEGACY_ORG_PREFIXES:
                    legacy_cid = canon.canonical_id(entity)
                    if legacy_cid and legacy_cid != entity:
                        legacy_prefixed.append((note.name, entity, legacy_cid))
                        continue

            canonical_id = canon.canonical_id(entity)
            if not canonical_id:
                continue
            if canon.is_self(canonical_id) and note.domain != "company":
                self_outside_company.append((note.name, entity, canonical_id))
                continue
            if canonical_id.lower() != entity.lower():
                unresolved_alias.append((note.name, entity, canonical_id))

        for raw_component in note.components:
            component = raw_component.strip()
            if not component or ":" not in component:
                continue
            prefix, _, tail = component.partition(":")
            prefix = prefix.strip().lower()
            tail = tail.strip().lower()
            if prefix == "component" and tail not in registered_components:
                unregistered_component.append((note.name, component))
            elif prefix == "initiative" and tail not in registered_initiatives:
                unregistered_component.append((note.name, component))

    return {
        "raw_email": sorted(set(raw_email)),
        "bare_numeric": sorted(set(bare_numeric)),
        "unresolved_alias": sorted(set(unresolved_alias)),
        "self_outside_company": sorted(set(self_outside_company)),
        "wrong_facet": sorted(set(wrong_facet)),
        "legacy_prefixed": sorted(set(legacy_prefixed)),
        "unregistered_component": sorted(set(unregistered_component)),
    }


def analyze_registry_trust(registry_path: str) -> dict[str, object]:
    """Audit split-registry identity shape and field-level authority."""
    registry_dir = os.path.dirname(registry_path) or "."
    if not os.path.exists(os.path.join(registry_dir, "orgs.yml")):
        return {"quarantined_ids": [], "blocking": [], "review": []}
    split = {}
    for filename, key in (("people.yml", "people"), ("orgs.yml", "orgs"),
                          ("systems.yml", "systems")):
        document = _safe_load_yaml(os.path.join(registry_dir, filename))
        rows = document.get(key) if isinstance(document, dict) else []
        split[key] = [row for row in (rows or []) if isinstance(row, dict)]
    return audit_registry_records(split)


def write_registry_review_queue(
    root: str,
    registry_audit: dict[str, object],
    entity_issues: dict[str, list[tuple[str, ...]]],
    gaps: list[str],
    *,
    max_items: int = 200,
) -> str:
    """Write a bounded deterministic attended-review queue under ``state/``."""
    items: list[dict[str, object]] = []
    for severity in ("blocking", "review"):
        for row in registry_audit.get(severity) or []:
            if isinstance(row, dict):
                items.append({"kind": "registry", "severity": severity, **row})
    high_impact = {
        "raw_email", "self_outside_company", "wrong_facet",
        "legacy_prefixed", "unregistered_component",
    }
    for issue_class, rows in sorted(entity_issues.items()):
        for row in rows:
            items.append({
                "kind": "entity-facet",
                "severity": "isolated" if issue_class in high_impact else "review",
                "issue_class": issue_class,
                "values": list(row),
            })
    for gap in gaps:
        items.append({"kind": "missing-entity-page", "severity": "review", "detail": gap})
    priority = {"blocking": 0, "isolated": 1, "review": 2}
    items.sort(key=lambda item: (
        priority.get(str(item.get("severity") or "review"), 3),
        json.dumps(item, ensure_ascii=False, sort_keys=True),
    ))
    total = len(items)
    payload = {
        "schema_version": 1,
        "total": total,
        "returned": min(total, max_items),
        "truncated": total > max_items,
        "quarantined_ids": list(registry_audit.get("quarantined_ids") or []),
        "items": items[:max_items],
    }
    state_dir = os.path.join(os.path.abspath(root), "state")
    os.makedirs(state_dir, exist_ok=True)
    path = os.path.join(state_dir, "registry_review_queue.json")
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _normalize_entity_for_matching(entity: str, registry_path: str) -> str:
    """Best-effort canonicalization for contributing-note matching.

    Falls back to the lowercased raw string when ``canonicalize`` can't resolve
    it (no registry entry, or the registry file itself is absent) — so tests and
    real notes both match by whatever identity they actually share, and this
    never crashes on an unregistered entity.
    """
    entity = entity.strip()
    return canonicalize(entity, registry_path=registry_path) or entity.lower()


# Dossiers AND initiative pages are the same split-page
# family (one page generator writes both from the same AUTO/NOTES
# contract) — an initiative page's acl can drift from its contributing notes
# exactly like an entity dossier's, so both types are checked here.
_DOSSIER_LIKE_TYPES = ("entity", "initiative")
_DOSSIER_AUDIENCE_MARKER = "audience_partitioned"
_DOSSIER_PRODUCER_MARKER = "generated-by:entity-pages"


def _initiative_specs(registry_path: str) -> dict[str, dict[str, object]]:
    """Load the initiative source selectors next to the entity registry.

    This is intentionally independent from the page generator.  Lint
    must be able to reconstruct an initiative's sources even if that producer
    is broken or its page metadata has been tampered with.
    """
    path = os.path.join(os.path.dirname(registry_path) or ".", "initiatives.yml")
    doc = _safe_load_yaml(path)
    entries = doc.get("initiatives") if isinstance(doc, dict) else None
    if not isinstance(entries, list):
        entries = _parse_initiatives_fallback(path)
    specs: dict[str, dict[str, object]] = {}
    duplicate_ids: set[str] = set()
    for raw in entries or []:
        if not isinstance(raw, dict):
            continue
        iid = str(raw.get("id") or "").strip()
        if not iid or iid in duplicate_ids:
            continue
        if iid in specs:
            # Duplicate ids are not a trustworthy selector. Leaving the id out
            # makes generated metadata fail closed below instead of choosing a
            # registry row by load order.
            specs.pop(iid, None)
            duplicate_ids.add(iid)
            continue
        specs[iid] = {
            "id": iid,
            "components": [
                str(value).strip() for value in (raw.get("components") or [])
                if str(value).strip()
            ],
            "aliases": [
                str(value).strip() for value in (raw.get("aliases") or [])
                if str(value).strip()
            ],
            "acl": [
                str(value).strip() for value in (raw.get("acl") or [])
                if str(value).strip()
            ],
        }
    return specs


def _initiative_alias_pattern(words: list[str]) -> "re.Pattern[str] | None":
    """Mirror the producer's documented metadata-only alias matcher."""
    words = sorted({word for word in words if word}, key=len, reverse=True)
    if not words:
        return None
    alt = "|".join(re.escape(word) for word in words)
    return re.compile(r"(?<![\w@])(?:" + alt + r")(?![\w])", re.IGNORECASE)


def _initiative_contributes(note: Note, spec: dict[str, object]) -> bool:
    """Independently apply the initiative registry's stable source selector."""
    iid = str(spec.get("id") or "").strip()
    raw_components = {
        str(value).strip().lower() for value in (spec.get("components") or [])
        if str(value).strip()
    }
    component_ids = {f"component:{value}" for value in raw_components}
    facets = {
        str(value).strip().lower() for value in (note.components or [])
        if str(value).strip()
    }
    entities = {
        str(value).strip().lower() for value in (note.entities or [])
        if str(value).strip()
    }
    structured_match = bool(
        component_ids & facets
        or (f"initiative:{iid}" in facets if iid else False)
        or raw_components & entities
    )
    rx = _initiative_alias_pattern(list(spec.get("aliases") or []))
    metadata = " ".join((note.name or "", note.title or "", note.description or ""))
    return structured_match or bool(rx is not None and rx.search(metadata))


def _registry_entity_acls(
    registry_path: str, canon: Canonicalizer,
) -> dict[str, list[str]]:
    """Index registry subject ACLs once, independent of generated metadata."""
    registry_dir = os.path.dirname(registry_path) or "."
    sources: list[tuple[str, str]]
    split_mode = os.path.exists(os.path.join(registry_dir, "orgs.yml"))
    if split_mode:
        sources = [("people.yml", "people"), ("orgs.yml", "orgs")]
    else:
        sources = [
            (os.path.basename(registry_path), section)
            for section in (
                "people", "vendor", "investor", "partner", "advisor",
                "champion", "customer", "friend", "academic", "recruiting",
                "prospect",
            )
        ]
    indexed: dict[str, list[str]] = {}
    for filename, section in sources:
        path = filename if os.path.isabs(filename) else os.path.join(registry_dir, filename)
        for row in _load_registry_records(path, section):
            if split_mode:
                kind = "person" if section == "people" else "org"
                if registry_record_quarantine_reasons(row, kind):
                    continue
            raw_id = str(row.get("id") or row.get("name") or "").strip()
            if not raw_id:
                continue
            resolved = canon.canonical_id(raw_id) or raw_id
            raw_acl = row.get("acl") or []
            acl = sorted({str(value).strip() for value in raw_acl if str(value).strip()})
            indexed.setdefault(resolved, acl)
    return indexed


def _generated_dossier_metadata(
    note: Note,
    registry_path: str,
    initiative_specs: dict[str, dict[str, object]],
    entity_acls: dict[str, list[str]],
) -> tuple[dict[str, object] | None, str | None]:
    """Validate producer metadata and its binding to the physical page name."""
    if _DOSSIER_PRODUCER_MARKER not in (note.provenance or []):
        return None, None
    structured = note.structured if isinstance(note.structured, dict) else {}
    has_versioned_metadata = (
        DOSSIER_METADATA_MARKER in (note.provenance or [])
        or any(key in structured for key in (
            "kind", "subject_id", "selector_version", "source_fingerprint",
            "profile_version", "profile_claims", "profile_fingerprint",
            "profile_sources", "profile_prompt_incomplete",
            _DOSSIER_AUDIENCE_MARKER, "audience_acl",
        ))
    )
    if not has_versioned_metadata:
        # Compatibility for producer-owned pages written before dossier schema
        # v1. Their ACL is still independently recomputed below; the next
        # generator pass upgrades them in place. Once either the durable schema
        # marker or any v1 field exists, partial/removal tampering fails closed.
        return None, None
    expected_kind = f"{note.type}_dossier"
    if structured.get("kind") != expected_kind:
        return None, f"invalid generated dossier kind; expected {expected_kind!r}"
    subject_id = str(structured.get("subject_id") or "").strip()
    if not subject_id:
        return None, "generated dossier metadata is missing subject_id"
    if structured.get("selector_version") != DOSSIER_SELECTOR_VERSION:
        return None, "generated dossier selector_version is missing or unsupported"
    fingerprint = structured.get("source_fingerprint")
    if (not isinstance(fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None):
        return None, "generated dossier source_fingerprint is missing or malformed"
    if structured.get("profile_version") != DOSSIER_PROFILE_VERSION:
        return None, "generated dossier profile_version is missing or unsupported"
    profile_claims = structured.get("profile_claims")
    if not isinstance(profile_claims, dict):
        return None, "generated dossier profile_claims is missing or malformed"
    profile_fingerprint = structured.get("profile_fingerprint")
    if (not isinstance(profile_fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", profile_fingerprint) is None):
        return None, "generated dossier profile_fingerprint is missing or malformed"
    profile_sources = structured.get("profile_sources")
    if (not isinstance(profile_sources, list)
            or any(not isinstance(value, str) or not value for value in profile_sources)
            or profile_sources != sorted(set(profile_sources))):
        return None, "generated dossier profile_sources is malformed"
    if not isinstance(structured.get("profile_prompt_incomplete"), bool):
        return None, "generated dossier profile_prompt_incomplete is malformed"

    try:
        base_name = dossier_base_name(note.type, subject_id)
    except ValueError as exc:
        return None, str(exc)
    partitioned = structured.get(_DOSSIER_AUDIENCE_MARKER) is True
    declared_audience: list[str] = []
    stable_names = {base_name}
    if partitioned:
        raw_acl = structured.get("audience_acl")
        if not isinstance(raw_acl, list) or any(
                not isinstance(label, str) or not label.strip() for label in raw_acl):
            return None, "generated dossier audience_acl is malformed"
        declared_audience = sorted(set(label.strip() for label in raw_acl))
        if len(declared_audience) == 1:
            stable_names.add(
                f"{base_name}-audience-{dossier_slug(declared_audience[0])}"
            )
    if note.name not in stable_names:
        # Approved/human-owned fixed-name pages are never overwritten. The
        # generator files a proposed ``<target>-update[-N]`` successor instead;
        # bind that alternate physical name to its durable supersedes edge and
        # machine-update capability rather than rejecting a valid review head.
        proposal_targets = [
            target for target in stable_names
            if f"machine-update:{target}" in (note.provenance or [])
            and ("supersedes", target) in note.typed_links
            and (
                note.name == f"{target}-update"
                or note.name.startswith(f"{target}-update-")
            )
        ]
        if len(proposal_targets) != 1:
            qualifier = "subject/audience" if partitioned else "subject"
            return None, f"generated dossier {qualifier} metadata does not match page name"

    if note.type == "entity":
        dossier_ids = {
            _normalize_entity_for_matching(value, registry_path)
            for value in (note.entities or []) if str(value).strip()
        }
        normalized_subject = _normalize_entity_for_matching(subject_id, registry_path)
        if normalized_subject not in dossier_ids:
            return None, "generated entity subject_id does not match its entities facet"
        if subject_id not in entity_acls:
            return None, "generated entity subject_id is absent from the registry"
    elif note.type == "initiative":
        spec = initiative_specs.get(subject_id)
        if spec is None:
            return None, "generated initiative subject_id is absent or ambiguous in registry"
        expected_components = sorted(
            f"component:{str(value).strip()}"
            for value in (spec.get("components") or []) if str(value).strip()
        )
        if sorted(set(note.components or [])) != expected_components:
            return None, "generated initiative components do not match its registry selector"

    return {
        "schema_marked": DOSSIER_METADATA_MARKER in (note.provenance or []),
        "subject_id": subject_id,
        "partitioned": partitioned,
        "audience_acl": declared_audience,
        "source_fingerprint": fingerprint,
        "profile_claims": profile_claims,
        "profile_fingerprint": profile_fingerprint,
        "profile_sources": profile_sources,
        "profile_prompt_incomplete": structured["profile_prompt_incomplete"],
    }, None


def _safe_dossier_acl(acls: list[list[str]]) -> list[str]:
    """Return the ACL that may expose every non-secret contributor.

    Remembrancer labels are OR grants. A literal union would let a reader of one
    scope see facts sourced from another, so only common restricted labels are
    safe; an unrepresentable intersection aborts dossier generation for review.
    """
    return combine_acl(*acls)


# A dossier conflict is either an ACL-safety failure this transaction can still
# cause (blocking), or a state only an attended repair can clear (deferred). The
# distinction is the record's, not the caller's: publish gating, the lint report
# and the attended queue all read the same field.
DOSSIER_CONFLICT_BLOCKING = "blocking"
DOSSIER_CONFLICT_DEFERRED_STALE = "deferred-stale-fingerprint"
DOSSIER_CONFLICT_DEFERRED_LEGACY_SPLIT = "deferred-legacy-split"
_DEFERRED_DOSSIER_DISPOSITIONS = frozenset({
    DOSSIER_CONFLICT_DEFERRED_STALE,
    DOSSIER_CONFLICT_DEFERRED_LEGACY_SPLIT,
})


def _dossier_conflict_record(
    dossier: Note,
    contributors: list[tuple[str, list[str]]],
    reason: str,
    disposition: str = DOSSIER_CONFLICT_BLOCKING,
) -> dict[str, object]:
    return {
        "name": dossier.name,
        "current_acl": sorted(set(dossier.acl or [])),
        "entities": sorted(set(dossier.entities or [])),
        "contributors": [
            {"name": name, "acl": sorted(set(acl))}
            for name, acl in sorted(contributors)
        ],
        "reason": reason,
        "disposition": disposition,
    }


def _infer_initiative_spec(
    note: Note, specs: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    """Resolve a legacy/unmarked initiative page by its fixed logical name."""
    matches = []
    for iid, spec in specs.items():
        base_name = dossier_base_name("initiative", iid)
        if note.name == base_name or note.name.startswith(f"{base_name}-audience-"):
            matches.append(spec)
    return matches[0] if len(matches) == 1 else None


def find_dossier_acl_drift(
    notes: list[Note], registry_path: str = "registry/entities.yml"
) -> list[tuple[str, list[str], list[str]]]:
    """Flag ``type: entity`` / ``type: initiative`` dossier-like notes whose
    ``acl`` has drifted from the safe common scope of their contributing
    notes' ``acl`` labels.

    A dossier's contributing notes are every *other*, non-dossier-like note
    that shares at least one entity id with it (matched after best-effort
    canonicalization, so raw and already-canonical entity strings both match).
    Notes labeled ``secrets`` are excluded from the rollup entirely — never
    widened into a broader page.

    Returns ``(dossier_name, current_acl, expected_acl)`` for each drifted
    dossier, sorted by name. Returns ``[]`` when there are no dossier/
    initiative notes yet — the common case until a page generator
    starts producing them; this check must never require dossiers to exist.

    N.B.: this lint runs BEFORE the nightly page regen, so a freshly-drifted
    acl may lag one nightly cycle before the page generator picks it up —
    see the report caption in `write_entity_lint_report`.
    """
    return analyze_dossier_acl(notes, registry_path=registry_path)["drift"]


def analyze_dossier_acl(
    notes: list[Note], registry_path: str = "registry/entities.yml"
) -> dict[str, list]:
    """Return dossier ACL drift and unrepresentable contributor conflicts.

    Remembrancer ACL labels are OR grants.  When contributors have no common label,
    there is no safe ACL value for a page containing all of their content.
    Lint must report that state as data instead of raising ``ACLConflict`` and
    aborting before its JSON summary can be written.

    Producer-marked pages bind a stable registry target and complete source
    fingerprint. Lint independently reconstructs entity contributors from
    canonical facets and initiative contributors from the initiative registry's
    component/alias selector. Audience facets are checked against only the
    public/source rows their declared label may reveal. Unmarked pages retain
    the stricter all-contributor audit, so a hand-authored structured field
    cannot bypass it. The legacy :func:`find_dossier_acl_drift` API remains a
    list-only view of the ``drift`` bucket. ``conflicts`` contains
    JSON-serialisable records for attended split/quarantine repair.
    """
    dossiers = [n for n in notes if n.type in _DOSSIER_LIKE_TYPES]
    if not dossiers:
        return {"drift": [], "conflicts": []}

    canon = Canonicalizer(registry_path)
    initiative_specs = _initiative_specs(registry_path)
    entity_acls = _registry_entity_acls(registry_path, canon)
    raw_source_pool = [
        note for note in notes
        if note.type not in _DOSSIER_LIKE_TYPES
        and "secrets" not in (note.acl or [])
    ]
    registry_dir = os.path.dirname(os.path.abspath(registry_path))
    source_root = (os.path.dirname(registry_dir)
                   if os.path.basename(registry_dir) == "registry"
                   else registry_dir)
    source_pool = eligible_source_notes(source_root, raw_source_pool)
    # Entity normalization is registry-backed and comparatively expensive.
    # Build the inverse facet index once instead of repeating it for every
    # dossier page (a corpus can hold many).
    entity_source_index: dict[str, list[Note]] = {}
    for note in source_pool:
        normalized_ids = {
            _normalize_entity_for_matching(value, registry_path)
            for value in (note.entities or []) if str(value).strip()
        }
        for normalized_id in normalized_ids:
            entity_source_index.setdefault(normalized_id, []).append(note)
    drift: list[tuple[str, list[str], list[str]]] = []
    conflicts: list[dict[str, object]] = []
    for dossier in dossiers:
        metadata, metadata_issue = _generated_dossier_metadata(
            dossier, registry_path, initiative_specs, entity_acls,
        )
        if metadata_issue:
            conflicts.append(_dossier_conflict_record(dossier, [], metadata_issue))
            continue
        partitioned = bool(metadata and metadata["partitioned"])
        declared_audience = list(metadata["audience_acl"]) if metadata else []

        subject_acl: list[str] | None = None
        initiative_spec: dict[str, object] | None = None
        dossier_ids: set[str] = set()
        if dossier.type == "initiative":
            if metadata:
                initiative_spec = initiative_specs.get(str(metadata["subject_id"]))
            else:
                initiative_spec = _infer_initiative_spec(dossier, initiative_specs)
            if initiative_spec is not None:
                subject_acl = list(initiative_spec.get("acl") or [])
            else:
                # Compatibility path for historical hand-authored initiative
                # pages that used an entity selector before components existed.
                dossier_ids = {
                    _normalize_entity_for_matching(e, registry_path)
                    for e in dossier.entities if e.strip()
                }
        else:
            if metadata:
                dossier_ids = {
                    _normalize_entity_for_matching(
                        str(metadata["subject_id"]), registry_path,
                    )
                }
                # Metadata validation already proved this target exists.
                subject_acl = entity_acls.get(str(metadata["subject_id"]))
            else:
                dossier_ids = {
                    _normalize_entity_for_matching(e, registry_path)
                    for e in dossier.entities if e.strip()
                }
                if len(dossier_ids) == 1:
                    subject_id = next(iter(dossier_ids))
                    if subject_id in entity_acls:
                        subject_acl = entity_acls[subject_id]

        if initiative_spec is not None:
            candidates = [
                note for note in source_pool
                if _initiative_contributes(note, initiative_spec)
            ]
        elif dossier_ids:
            by_identity: dict[int, Note] = {}
            for dossier_id in dossier_ids:
                for note in entity_source_index.get(dossier_id, []):
                    by_identity[id(note)] = note
            candidates = list(by_identity.values())
        else:
            candidates = []
        source_notes = [
            note for note in candidates
            if not partitioned or acl_can_reveal(
                declared_audience, list(note.acl or []),
            )
        ]

        contributors = [
            (note.name, list(note.acl or [])) for note in source_notes
        ]
        if metadata:
            actual_fingerprint = dossier_source_fingerprint(source_notes)
            if actual_fingerprint != metadata["source_fingerprint"]:
                conflicts.append(_dossier_conflict_record(
                    dossier,
                    contributors,
                    "generated dossier source_fingerprint does not match current registry sources",
                    DOSSIER_CONFLICT_DEFERRED_STALE,
                ))
                continue
            raw_claims = metadata["profile_claims"]
            # Re-verification must reproduce the material the claim was grounded
            # against. The cross-note budget evicts a different subset as the
            # entity gains notes, so pull the CITED sources in first — otherwise
            # a valid claim decays to "ungrounded" purely because the window
            # moved, blocking publish.
            cited_sources = sorted({
                claim["source_note"]
                for value in (raw_claims.values() if isinstance(raw_claims, dict) else [])
                for claim in (value if isinstance(value, list) else [value])
                if isinstance(claim, dict) and isinstance(claim.get("source_note"), str)
            })
            _prompt_text, profile_excerpts, prompt_incomplete = (
                dossier_profile_prompt_material(source_notes, require_names=cited_sources)
            )
            subject_id = str(metadata.get("subject_id") or "")
            subject_labels = [subject_id, dossier.title or dossier.name]
            if dossier.type == "entity" and subject_id:
                subject_labels.append(canon.display_name(subject_id))
            grounded_claims = normalize_dossier_profile_claims(
                raw_claims, profile_excerpts, subject_labels,
            )
            expected_sources = sorted({
                claim["source_note"]
                for value in grounded_claims.values()
                for claim in (value if isinstance(value, list) else [value])
                if isinstance(claim, dict)
            })
            profile_issue = ""
            if grounded_claims != raw_claims:
                profile_issue = (
                    "generated dossier profile contains an ungrounded or "
                    "semantically unsupported field/source/quote"
                )
            elif metadata["profile_fingerprint"] != dossier_profile_claims_fingerprint(
                    grounded_claims):
                profile_issue = "generated dossier profile_fingerprint does not match claims"
            elif metadata["profile_sources"] != expected_sources:
                profile_issue = "generated dossier profile_sources do not match claims"
            elif metadata["profile_prompt_incomplete"] != prompt_incomplete:
                profile_issue = "generated dossier prompt completeness metadata is stale"
            elif metadata["schema_marked"]:
                rendered_profile = render_dossier_profile(grounded_claims)
                expected_body_profile = rendered_profile or DOSSIER_PROFILE_PLACEHOLDER
                if extract_dossier_profile(dossier.body) != expected_body_profile:
                    profile_issue = "generated dossier Profile body does not match grounded claims"
                else:
                    last_activity = max(
                        (str(note.updated or "") for note in source_notes), default="",
                    )
                    expected_description = dossier_profile_description(
                        rendered_profile,
                        dossier.title or dossier.name,
                        len(source_notes),
                        last_activity,
                    )
                    if dossier.description != expected_description:
                        profile_issue = (
                            "generated dossier description does not match grounded claims"
                        )
            if profile_issue:
                conflicts.append(_dossier_conflict_record(
                    dossier, contributors, profile_issue,
                ))
                continue

        contributor_acls = [acl for _name, acl in contributors]
        # No contributors means there is no evidence for widening a generated
        # page to public. It may be a registry-only profile or an aggregate whose
        # sources were quarantined. Preserve its current fail-closed ACL; the
        # ordinary orphan report covers disconnected pages separately.
        if not contributor_acls and metadata is None:
            continue
        try:
            acl_inputs: list[list[str]] = []
            if partitioned:
                acl_inputs.append(declared_audience)
            if subject_acl is not None:
                acl_inputs.append(subject_acl)
            acl_inputs.extend(contributor_acls)
            expected = _safe_dossier_acl(acl_inputs)
        except ACLConflict as exc:
            # A producer-marked page was written by the generator in this very
            # transaction, so an unrepresentable ACL there is a live bug and must
            # keep blocking publish. An unmarked pre-partition page is different:
            # nothing in the pipeline rewrites it, its fail-closed ACL is left
            # exactly as it is, and only the attended audience-split workflow can
            # resolve it. Blocking on that froze EVERY publish behind one legacy
            # page, and many such pages can sit latent. Over-exposure that
            # already exists in a page body is a separate, still
            # fatal check: find_unsafe_forward_links rejects any link a page's ACL
            # may not reveal, so deferring here narrows the queue, not the gate.
            conflicts.append(_dossier_conflict_record(
                dossier, contributors, str(exc),
                DOSSIER_CONFLICT_BLOCKING if metadata
                else DOSSIER_CONFLICT_DEFERRED_LEGACY_SPLIT,
            ))
            continue
        current = sorted(set(dossier.acl or []))
        if current != expected:
            drift.append((dossier.name, current, expected))

    return {
        "drift": sorted(drift),
        "conflicts": sorted(conflicts, key=lambda item: str(item["name"])),
    }


def write_entity_lint_report(
    issues: dict[str, list[tuple[str, ...]]],
    dossier_drift: list[tuple[str, list[str], list[str]]],
    context_dir: str,
    top_n: int = _ENTITY_LINT_TOP_N,
    dossier_conflicts: list[dict[str, object]] | None = None,
    registry_audit: dict[str, object] | None = None,
) -> str:
    """Append the entity-canonicalization section to context/_lint_report.md.

    Appends (rather than overwriting) so it composes with `write_gap_report`'s
    section in `run()`; renders cleanly with zero offenders and zero dossiers
    or when called standalone against a fresh context dir.
    """
    os.makedirs(context_dir, exist_ok=True)
    report_path = os.path.join(context_dir, "_lint_report.md")

    class_labels = [
        ("raw_email", "Raw email addresses"),
        ("bare_numeric", "Bare numeric ids"),
        ("unresolved_alias", "Aliases that should be canonical ids"),
        ("self_outside_company", "Self entities outside domain: company"),
        ("wrong_facet", "entities: should be components: (component/initiative id)"),
        ("legacy_prefixed", "Legacy relationship-prefixed ids (post split-registry)"),
        ("unregistered_component", "components: id not in the registry"),
    ]

    lines = ["", "## Entity Canonicalization", ""]
    total = sum(len(issues.get(key, [])) for key, _ in class_labels)
    if total == 0:
        lines.append("_(no entity canonicalization issues detected)_")
    else:
        for key, label in class_labels:
            offenders = issues.get(key, [])
            lines.append(f"### {label} ({len(offenders)})")
            if not offenders:
                lines.append("_(none)_")
            else:
                for entry in offenders[:top_n]:
                    if len(entry) == 3:
                        note_name, entity, canonical_id = entry
                        lines.append(f"- `{note_name}`: `{entity}` → should be `{canonical_id}`")
                    else:
                        note_name, entity = entry
                        lines.append(f"- `{note_name}`: `{entity}`")
                remaining = len(offenders) - top_n
                if remaining > 0:
                    lines.append(f"- _(+{remaining} more)_")
            lines.append("")

    registry_audit = registry_audit or {
        "quarantined_ids": [], "blocking": [], "review": [],
    }
    lines.append("### Registry trust quarantine / review")
    lines.append("")
    quarantined_ids = list(registry_audit.get("quarantined_ids") or [])
    lines.append(
        f"- Semantically quarantined identities: {len(quarantined_ids)}"
    )
    for identifier in quarantined_ids[:top_n]:
        lines.append(f"  - `{identifier}`")
    review_rows = list(registry_audit.get("review") or [])
    lines.append(f"- Unverified high-impact registry fields: {len(review_rows)}")
    for row in review_rows[:top_n]:
        if isinstance(row, dict):
            lines.append(
                f"  - `{row.get('entity_id', '')}` `{row.get('field', '')}`: "
                f"{row.get('reason', '')}"
            )
    lines.append("- Full bounded queue: `state/registry_review_queue.json`")
    lines.append("")
    lines.append("### Dossier ACL drift")
    lines.append("")
    lines.append("_(nightly runs dossier generation before this lint; findings describe "
                 "the current post-generation corpus and must be reviewed now)_")
    lines.append("")
    if not dossier_drift:
        lines.append("_(no dossier ACL drift detected)_")
    else:
        for name, current, expected in dossier_drift[:top_n]:
            lines.append(f"- `{name}`: acl={current} but contributing notes imply {expected}")
        remaining = len(dossier_drift) - top_n
        if remaining > 0:
            lines.append(f"- _(+{remaining} more)_")
    lines.append("")

    dossier_conflicts = dossier_conflicts or []
    lines.append("### Dossier ACL conflicts")
    lines.append("")
    if not dossier_conflicts:
        lines.append("_(no unrepresentable dossier ACL conflicts detected)_")
    else:
        for conflict in dossier_conflicts[:top_n]:
            labels = sorted({
                tuple(item.get("acl") or [])
                for item in (conflict.get("contributors") or [])
                if isinstance(item, dict)
            })
            gate = ("deferred to attended repair"
                    if _dossier_conflict_is_deferred(conflict)
                    else "blocking publication")
            lines.append(
                f"- `{conflict.get('name', '')}`: no OR-grant ACL can contain "
                f"contributor scopes {labels}; split or quarantine the aggregate "
                f"({gate})"
            )
        remaining = len(dossier_conflicts) - top_n
        if remaining > 0:
            lines.append(f"- _(+{remaining} more)_")
    lines.append("")

    with open(report_path, "a") as f:
        # A generated report may be committed by a scheduled job. Keep one
        # terminal newline, not a blank terminal line that fails diff-check.
        f.write("\n".join(lines).rstrip() + "\n")
    return report_path


# ---------------------------------------------------------------------------
# Run orchestrator
# ---------------------------------------------------------------------------

@serialized_pipeline_mutation(root_argument="root")
def run(
    root: str = ".",
    *,
    quarantine: bool = False,
    max_quarantine: int = 10,
    max_file_back_writes: int = 50,
) -> dict:
    """Run all lint checks and apply safe auto-fixes.

    Schema validation is always audit-only unless ``quarantine=True``.  Even
    then, only notes missing required fields are moved and the batch cap is
    checked before mutation.  Cross-reference file-back is bounded per pass
    and controlled by AGENTS.md's ``file_back_enabled`` convention.

    Returns a summary dict with keys:
        orphans: list[str]
        dangling: list[tuple[str,str]]
        superseded: list[str]
        contradictions: list[tuple[str,str]]
        gaps: list[str]
    """
    context_dir = os.path.join(root, "context")
    schema_issues = find_schema_issues(context_dir)
    quarantined: list[dict[str, object]] = []
    quarantine_blocked = ""
    if quarantine:
        try:
            quarantined = quarantine_invalid_notes(
                schema_issues, root, max_notes=max_quarantine
            )
        except QuarantineLimitExceeded as exc:
            quarantine_blocked = str(exc)
    remaining_schema_issues = find_schema_issues(context_dir) if quarantined else schema_issues

    # A report-only schema offender must not be incidentally rewritten by a
    # supersede/backlink auto-fix (which could normalise or lose malformed raw
    # frontmatter before a human reviews it).
    notes = _load_schema_valid_notes(context_dir, remaining_schema_issues)

    orphans = find_orphans(notes)
    dangling = find_dangling(notes)
    unsafe_forward_links = find_unsafe_forward_links(notes)

    # Supersession is report-only here; retrieval projects these active edges
    # per caller and lint never rewrites their targets.
    superseded = mark_superseded(notes, context_dir)
    supersession_issues = find_supersession_issues(notes)

    # Generated contradiction flags describe current truth only. Historical
    # supersession targets and retired notes remain available for audit, but
    # must not become source authority for a new verified projection.
    current = current_note_names(notes)
    current_notes = [note for note in notes if note.name in current]
    contradictions = find_contradictions(current_notes)
    # Resolve behavioral policies once: AGENTS.md is authoritative.
    spec = load_spec(root)
    policy = spec.get("contradiction_policy", "flag")
    reconciled: list[tuple[str, str]] = []
    contradiction_flags: list[str]
    if policy == "reconcile":
        reconciled, unresolved = reconcile_contradictions(
            contradictions, current_notes, context_dir
        )
        contradiction_flags = write_contradiction_notes(
            unresolved, context_dir, policy="flag", notes=current_notes
        )
    else:
        contradiction_flags = write_contradiction_notes(
            contradictions, context_dir, policy=str(policy), notes=current_notes
        )

    gaps = gap_report(current_notes)
    write_gap_report(gaps, context_dir)
    write_schema_lint_report(schema_issues, quarantined, context_dir)

    registry_path = os.path.join(root, "registry", "entities.yml")
    entity_issues = find_entity_issues(notes, registry_path=registry_path)
    registry_audit = analyze_registry_trust(registry_path)
    registry_review_queue = write_registry_review_queue(
        root, registry_audit, entity_issues, gaps,
    )
    # Dossiers are generated from current supersession heads only; auditing
    # historical targets as live contributors would manufacture fingerprint
    # drift that the producer can never reproduce.
    dossier_acl = analyze_dossier_acl(current_notes, registry_path=registry_path)
    dossier_acl_drift = dossier_acl["drift"]
    dossier_acl_conflicts = dossier_acl["conflicts"]
    write_entity_lint_report(
        entity_issues, dossier_acl_drift, context_dir,
        dossier_conflicts=dossier_acl_conflicts,
        registry_audit=registry_audit,
    )

    # Include contradiction notes created above.  The pre-sweep orphan/dangling
    # summary intentionally describes what lint observed at pass start; the
    # next pass observes the newly filed backlinks.
    file_back_notes = _load_schema_valid_notes(context_dir, remaining_schema_issues)
    file_back_current = current_note_names(file_back_notes)
    file_back = file_back_links(
        [note for note in file_back_notes if note.name in file_back_current],
        context_dir,
        enabled=bool(spec.get("file_back_enabled", True)),
        max_writes=max_file_back_writes,
    )

    regen_index(context_dir)

    return {
        "orphans": orphans,
        "dangling": dangling,
        "unsafe_forward_links": unsafe_forward_links,
        "superseded": superseded,
        "supersession_issues": supersession_issues,
        "contradictions": contradictions,
        "reconciled": reconciled,
        "contradiction_flags": contradiction_flags,
        "gaps": gaps,
        "schema_issues": schema_issues,
        "remaining_schema_issues": remaining_schema_issues,
        "quarantined": quarantined,
        "quarantine_blocked": quarantine_blocked,
        "file_back": file_back,
        "entity_issues": entity_issues,
        "registry_audit": registry_audit,
        "registry_review_queue": registry_review_queue,
        "dossier_acl_drift": dossier_acl_drift,
        "dossier_acl_conflicts": dossier_acl_conflicts,
        "dossier_acl_blocking": _fatal_dossier_conflicts(dossier_acl_conflicts),
        "dossier_acl_deferred": _deferred_dossier_conflicts(dossier_acl_conflicts),
    }


def _dossier_conflict_is_deferred(conflict) -> bool:
    """Whether a conflict is an attended queue item rather than a publish gate.

    Two states cannot be resolved by the run that observes them, so blocking on
    either stops every unrelated memory update indefinitely:

    * ``source_fingerprint`` staleness — the generated page lags the registry,
      and registry state also moves inside a transaction, so the attended
      domain-migration workflow regenerates its own targets.
    * an unrepresentable ACL on an *unmarked* pre-partition page — nothing in the
      pipeline rewrites it, so its fail-closed ACL simply stays; only the
      attended audience-split workflow can resolve it.

    Everything else still blocks, including an unrepresentable ACL on a
    producer-marked page: the generator wrote that page in this transaction.
    Content that is already over-exposed is gated separately and remains fatal
    (:func:`find_unsafe_forward_links`), so deferring here shortens the gate's
    reach over pages it cannot fix, not over unsafe content.

    Records serialised before dispositions existed are classified by reason.
    """
    record = conflict or {}
    disposition = str(record.get("disposition") or "")
    if disposition:
        return disposition in _DEFERRED_DOSSIER_DISPOSITIONS
    return "source_fingerprint" in str(record.get("reason", ""))


def _fatal_dossier_conflicts(conflicts) -> list:
    """Return only the conflicts that must refuse publication."""
    return [
        conflict for conflict in (conflicts or [])
        if not _dossier_conflict_is_deferred(conflict)
    ]


def _deferred_dossier_conflicts(conflicts) -> list:
    """Return the conflicts routed to attended repair instead of the gate."""
    return [
        conflict for conflict in (conflicts or [])
        if _dossier_conflict_is_deferred(conflict)
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python3 -m lint.engine",
        description="Run Remembrancer lint checks on a context directory.",
    )
    parser.add_argument("--root", default=".", help="Project root (default: .)")
    parser.add_argument(
        "--quarantine",
        action="store_true",
        help="Move notes missing required fields to state/quarantine (default: audit only).",
    )
    parser.add_argument(
        "--max-quarantine",
        type=int,
        default=10,
        help="Fail closed before moving more than this many notes (default: 10).",
    )
    parser.add_argument(
        "--max-file-back-writes",
        type=int,
        default=50,
        help="Maximum backlink note rewrites in one pass (default: 50).",
    )
    parser.add_argument(
        "--fail-on-schema",
        action="store_true",
        help="Exit non-zero when schema issues remain after optional quarantine.",
    )
    parser.add_argument(
        "--fail-on-unsafe-links",
        action="store_true",
        help=("Exit non-zero while a forward wiki-link leaks a less-visible target "
              "or a dossier combines unrepresentable ACL scopes."),
    )
    parser.add_argument(
        "--json-summary",
        help="Atomically write the complete machine-readable summary to this path.",
    )
    args = parser.parse_args()

    summary = run(
        root=args.root,
        quarantine=args.quarantine,
        max_quarantine=args.max_quarantine,
        max_file_back_writes=args.max_file_back_writes,
    )
    print("=== Remembrancer Lint Summary ===")
    print(f"Orphans ({len(summary['orphans'])}): {summary['orphans']}")
    print(f"Dangling links ({len(summary['dangling'])}): {summary['dangling']}")
    print(f"Unsafe forward links ({len(summary['unsafe_forward_links'])}): "
          f"{summary['unsafe_forward_links'][:100]}")
    print(f"Superseded ({len(summary['superseded'])}): {summary['superseded']}")
    print(f"Contradictions ({len(summary['contradictions'])}): {summary['contradictions']}")
    print(f"Reconciled ({len(summary['reconciled'])}): {summary['reconciled']}")
    print(f"Gaps ({len(summary['gaps'])}): {summary['gaps']}")
    print(f"Schema issues ({len(summary['schema_issues'])}): {summary['schema_issues']}")
    print(f"Quarantined ({len(summary['quarantined'])}): {summary['quarantined']}")
    if summary["quarantine_blocked"]:
        print(f"Quarantine blocked: {summary['quarantine_blocked']}")
    print(f"File-back: {summary['file_back']}")
    entity_issue_count = sum(len(v) for v in summary["entity_issues"].values())
    print(f"Entity issues ({entity_issue_count}): {summary['entity_issues']}")
    registry_audit = summary["registry_audit"]
    print(f"Registry quarantined ({len(registry_audit['quarantined_ids'])}): "
          f"{registry_audit['quarantined_ids'][:100]}")
    print(f"Registry review fields ({len(registry_audit['review'])}); "
          f"queue={summary['registry_review_queue']}")
    print(f"Dossier ACL drift ({len(summary['dossier_acl_drift'])}): {summary['dossier_acl_drift']}")
    print(f"Dossier ACL conflicts ({len(summary['dossier_acl_conflicts'])}): "
          f"{summary['dossier_acl_conflicts'][:100]}")
    print(f"Dossier ACL deferred to attended repair "
          f"({len(summary['dossier_acl_deferred'])}): "
          f"{[c.get('name') for c in summary['dossier_acl_deferred'][:100]]}")
    if args.json_summary:
        output = os.path.abspath(args.json_summary)
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        temporary = f"{output}.tmp.{os.getpid()}"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if summary["quarantine_blocked"]:
        sys.exit(2)
    if args.fail_on_schema and summary["remaining_schema_issues"]:
        sys.exit(1)
    if args.fail_on_unsafe_links and (
            summary["unsafe_forward_links"]
            or summary["dossier_acl_blocking"]
            or summary["supersession_issues"]):
        sys.exit(1)
    sys.exit(0)
