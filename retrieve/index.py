"""Read and validate Remembrancer's per-domain retrieval indexes.

The Markdown indexes are both human-readable navigation and a cheap candidate
catalog.  ``regen_index`` appends an opaque, versioned metadata comment to each
row.  A portable content digest (rather than checkout-local inode timestamps)
lets retrieval validate that catalog without parsing every note.  Legacy,
missing, malformed, or stale indexes deliberately return ``None`` so callers
can fall back to a full corpus scan rather than lose recall.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
import re

from remembrancerlib import load_suppressed, slugify


INDEX_VERSION_MARKER = "<!-- remembrancer-index-v2 -->"
INDEX_META_PREFIX = "<!-- remembrancer-index-v2:"
_INDEX_META_RE = re.compile(r"<!-- remembrancer-index-v2:([A-Za-z0-9_=-]+) -->\s*$")
_INDEX_DIGEST_RE = re.compile(r"^<!-- remembrancer-index-v2-digest:([a-f0-9]{64}) -->$")
_SKIP_MARKDOWN = {"INDEX.md", "index.md", "_lint_report.md"}


@dataclass(frozen=True)
class IndexRecord:
    """One note row from a validated per-domain index."""

    path: str
    name: str
    title: str
    description: str
    updated: str
    acl: tuple[str, ...]
    entities: tuple[str, ...]
    provenance: tuple[str, ...]
    source_lineages: tuple[str, ...]
    state: str | None
    approval: dict | None
    source_retirement: dict | None
    producer_lifecycle: dict | None
    verification: str | None
    verification_content_fingerprint: str
    embedded_verification_content_fingerprint: str | None
    verification_audit_id: str | None
    verification_method: str | None
    verification_producer: str | None
    verification_source_fingerprint: str | None
    verification_grounded: bool
    verification_reason: str | None
    verification_checked: int
    verification_supported: int
    verification_unsupported: int
    superseded: bool
    supersedes: tuple[str, ...]
    content_sha256: str
    size: int
    # Optional (absent in pre-2026-08 index rows ⇒ empty): curated write-time
    # tags, ranked by retrieve.query._ranking_text.
    tags: tuple[str, ...] = ()
    # Optional (absent in older rows ⇒ empty): TF-IDF-selected distinctive body
    # terms, so stage-1 ranking sees body-resident facts without storing bodies
    # in the index. Raw words; the query-time stemmer handles the rest.
    index_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class IndexSnapshot:
    """A complete, fresh view of every indexed note path."""

    records: tuple[IndexRecord, ...]


def encode_index_metadata(metadata: dict) -> str:
    """Return a comment-safe, deterministic representation for an index row."""
    raw = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    token = base64.urlsafe_b64encode(raw).decode("ascii")
    return f"{INDEX_META_PREFIX}{token} -->"


def _decode_index_metadata(line: str) -> dict | None:
    match = _INDEX_META_RE.search(line)
    if match is None:
        return None
    try:
        raw = base64.b64decode(match.group(1), altchars=b"-_", validate=True)
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _safe_relative_path(value: object) -> str | None:
    """Normalize a metadata path and reject absolute/traversing paths."""
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    normalized = os.path.normpath(value).replace(os.sep, "/")
    if (normalized.startswith("/") or normalized in (".", "..")
            or normalized.startswith("../")):
        return None
    return normalized


def _string_tuple(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return tuple(value)


def _record(metadata: dict) -> IndexRecord | None:
    path = _safe_relative_path(metadata.get("path"))
    acl = _string_tuple(metadata.get("acl"))
    entities = _string_tuple(metadata.get("entities"))
    provenance = _string_tuple(metadata.get("provenance"))
    source_lineages = _string_tuple(metadata.get("source_lineages"))
    supersedes = _string_tuple(metadata.get("supersedes"))
    # Absent in pre-2026-08 rows: treated as empty, never a validation failure.
    tags = _string_tuple(metadata.get("tags", []))
    if tags is None:
        return None
    index_keys = _string_tuple(metadata.get("index_keys", []))
    if index_keys is None:
        return None
    approval = metadata.get("approval", {})
    source_retirement = metadata.get("source_retirement", {})
    producer_lifecycle = metadata.get("producer_lifecycle", {})
    scalar_fields = ("name", "title", "description", "updated", "state",
                     "verification", "verification_content_fingerprint",
                     "embedded_verification_content_fingerprint",
                     "verification_audit_id", "verification_method",
                     "verification_producer", "verification_source_fingerprint",
                     "verification_reason", "content_sha256")
    integer_fields = (
        "size", "verification_checked", "verification_supported",
        "verification_unsupported",
    )
    if (path is None or acl is None or entities is None or provenance is None
            or source_lineages is None
            or supersedes is None
            or not isinstance(approval, dict)
            or not isinstance(source_retirement, dict)
            or not isinstance(producer_lifecycle, dict)
            or any(not isinstance(metadata.get(field), str) for field in scalar_fields)
            or any(not isinstance(metadata.get(field), int) or metadata.get(field) < 0
                   for field in integer_fields)
            or re.fullmatch(r"[a-f0-9]{64}", metadata.get("content_sha256", "")) is None
            or not isinstance(metadata.get("superseded"), bool)
            or not isinstance(metadata.get("verification_grounded"), bool)):
        return None
    return IndexRecord(
        path=path,
        name=metadata["name"],
        title=metadata["title"],
        description=metadata["description"],
        updated=metadata["updated"],
        acl=acl,
        entities=entities,
        provenance=provenance,
        source_lineages=source_lineages,
        state=metadata["state"] or None,
        approval=dict(approval) or None,
        source_retirement=dict(source_retirement) or None,
        producer_lifecycle=dict(producer_lifecycle) or None,
        verification=metadata["verification"] or None,
        verification_content_fingerprint=metadata["verification_content_fingerprint"],
        embedded_verification_content_fingerprint=(
            metadata["embedded_verification_content_fingerprint"] or None
        ),
        verification_audit_id=metadata["verification_audit_id"] or None,
        verification_method=metadata["verification_method"] or None,
        verification_producer=metadata["verification_producer"] or None,
        verification_source_fingerprint=(
            metadata["verification_source_fingerprint"] or None
        ),
        verification_grounded=metadata["verification_grounded"],
        verification_reason=metadata["verification_reason"] or None,
        verification_checked=metadata["verification_checked"],
        verification_supported=metadata["verification_supported"],
        verification_unsupported=metadata["verification_unsupported"],
        superseded=metadata["superseded"],
        supersedes=supersedes,
        content_sha256=metadata["content_sha256"],
        size=metadata["size"],
        tags=tags,
        index_keys=index_keys,
    )


def _actual_markdown_paths(context_dir: str) -> dict[str, tuple[int, str]] | None:
    """Return live note paths and portable byte identity.

    Filesystem timestamps cannot be committed as index freshness tokens: Git
    assigns new ctime/mtime values in every checkout.  Hashing the raw bytes is
    portable and still materially cheaper than parsing the complete corpus.
    The before/after stat comparison rejects a file rewritten while it is read.
    """
    actual: dict[str, tuple[int, str]] = {}
    repo_root = os.path.dirname(os.path.abspath(context_dir))
    # Use the canonical fail-closed tombstone loader. Treating a corrupt file
    # as an empty set would resurrect suppressed notes in the index snapshot.
    suppressed = {slugify(value) for value in load_suppressed(repo_root)}
    try:
        for dirpath, _, files in os.walk(context_dir):
            for filename in files:
                if not filename.endswith(".md") or filename in _SKIP_MARKDOWN:
                    continue
                if os.path.splitext(filename)[0] in suppressed:
                    continue
                path = os.path.join(dirpath, filename)
                rel = os.path.relpath(path, context_dir).replace(os.sep, "/")
                with open(path, "rb") as handle:
                    before = os.fstat(handle.fileno())
                    digest = hashlib.sha256()
                    size = 0
                    while chunk := handle.read(1024 * 1024):
                        size += len(chunk)
                        digest.update(chunk)
                    after = os.fstat(handle.fileno())
                before_identity = (
                    before.st_dev, before.st_ino, before.st_size,
                    before.st_mtime_ns, before.st_ctime_ns,
                )
                after_identity = (
                    after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns,
                )
                if before_identity != after_identity or size != after.st_size:
                    return None
                actual[rel] = (size, digest.hexdigest())
    except OSError:
        return None
    return actual


def load_index_snapshot(context_dir: str) -> IndexSnapshot | None:
    """Load a complete fresh v2 snapshot, or ``None`` to request full fallback.

    Freshness is conservative: the indexed path set and exact byte identity
    must match the live note set, and every domain row set must have a valid
    generation digest.  Adding, deleting, moving, or changing a note/index
    therefore requests an authoritative full scan; merely touching a file does
    not, so a clean checkout remains valid.
    """
    context_dir = os.path.abspath(context_dir)
    actual = _actual_markdown_paths(context_dir)
    if actual is None:
        return None
    if not actual:
        return IndexSnapshot(())

    records: list[IndexRecord] = []
    try:
        entries = list(os.scandir(context_dir))
    except OSError:
        return None
    for entry in sorted(entries, key=lambda item: item.name):
        if not entry.is_dir(follow_symlinks=False):
            continue
        index_path = os.path.join(entry.path, "index.md")
        if not os.path.isfile(index_path):
            continue
        try:
            with open(index_path, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            if INDEX_VERSION_MARKER not in lines:
                return None
        except OSError:
            return None
        digest_values = [match.group(1) for line in lines
                         if (match := _INDEX_DIGEST_RE.fullmatch(line))]
        if len(digest_values) != 1:
            return None
        row_tokens: list[str] = []
        for line in lines:
            if INDEX_META_PREFIX not in line:
                continue
            metadata = _decode_index_metadata(line)
            record = _record(metadata) if metadata is not None else None
            if record is None or record.path.split("/", 1)[0] != entry.name:
                return None
            match = _INDEX_META_RE.search(line)
            if match is None:
                return None
            row_tokens.append(match.group(1))
            records.append(record)
        calculated = hashlib.sha256("\n".join(row_tokens).encode("ascii")).hexdigest()
        if calculated != digest_values[0]:
            return None

    indexed = {record.path for record in records}
    if len(indexed) != len(records) or indexed != set(actual):
        return None
    by_path = {record.path: record for record in records}
    for path, identity in actual.items():
        record = by_path[path]
        if identity != (record.size, record.content_sha256):
            return None
    # Detect a concurrent add/delete/rewrite while the indexes were read.
    if _actual_markdown_paths(context_dir) != actual:
        return None
    return IndexSnapshot(tuple(records))
