"""Closed replay of canonical artifact notes from authenticated L1 records.

This validator accepts no caller-supplied events. A process cache reuses only an
authenticated event snapshot whose checkpoint and every input inode/size/mtime/
ctime still match. Proposals and human-owned notes deliberately stay outside
this contract: a faithful page excerpt is never authority to supersede a note.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import threading

from remembrancerlib.event import (
    iter_validated_event_records,
    load_event_validation_checkpoint,
)
from remembrancerlib.locks import pipeline_mutation_lock_held

PRODUCER = "artifact-publish"
METHOD = "artifact-event-projection"
MARKER = "generated-by:" + PRODUCER
_CACHE: dict[str, tuple[tuple, tuple]] = {}
_CACHE_LOCK = threading.Lock()


class ArtifactProjectionRefused(PermissionError):
    """Source proof was unavailable; this is not an audit storage failure."""


def note_shape_valid(note) -> bool:
    from remembrancerlib.verification import governance_hold_class

    if hasattr(note, "verification"):
        # Retrieval rechecks the full note and its exact audit binding.
        return getattr(note, "type", None) == "artifact"
    provenance = list(getattr(note, "provenance", None) or [])
    return bool(
        getattr(note, "type", None) == "artifact"
        and getattr(note, "state", None) == "proposed"
        and not getattr(note, "superseded", False)
        and getattr(note, "approval", None) is None
        and getattr(note, "source_retirement", None) is None
        and getattr(note, "producer_lifecycle", None) is None
        and not getattr(note, "project", None)
        and not getattr(note, "components", None)
        and not getattr(note, "structured", None)
        and not getattr(note, "source_lineages", None)
        and not getattr(note, "typed_links", [])
        and governance_hold_class(note) is None
        and all(isinstance(p, str) for p in provenance)
        and not any(p.startswith(("by:", "machine-update:")) for p in provenance)
        and {p for p in provenance if p.startswith("generated-by:")} <= {MARKER}
    )


def _snapshot_key(root: str) -> tuple:
    checkpoint = Path(root, "state", "event_validation_checkpoint.json")
    metadata = load_event_validation_checkpoint(str(checkpoint))
    paths = [
        checkpoint,
        *(Path(root, "events", name) for name in sorted(metadata["files"])),
    ]
    signatures = []
    for path in paths:
        stat = path.stat()
        signatures.append(
            (
                str(path),
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        )
    return tuple(signatures)


def _artifact_records(root: str) -> tuple:
    with _CACHE_LOCK:
        before = _snapshot_key(root)
        cached = _CACHE.get(root)
        if cached is not None and cached[0] == before:
            return cached[1]
        records = tuple(
            event
            for event in iter_validated_event_records(os.path.join(root, "events"))
            if event.source == "artifact"
        )
        if _snapshot_key(root) != before:
            raise PermissionError("artifact source snapshot changed during validation")
        # Keep one corpus per worker; never retain production and scratch stores.
        _CACHE.clear()
        _CACHE[root] = (before, records)
        return records


def replay_source_material(note, *, root: str) -> dict:
    """Prove every persisted projection field before a success can be sealed."""
    from distill.engine import _artifact_body, _artifact_id_version
    from distill.verify import _latest_projection_events
    from remembrancerlib import combine_acl, slugify

    root = os.path.abspath(root)
    if not pipeline_mutation_lock_held(root):
        raise PermissionError(
            "artifact verification requires the pipeline mutation lock"
        )
    if not note_shape_valid(note):
        raise PermissionError("artifact is not a canonical unreviewed projection")
    ids = [p for p in note.provenance if p.startswith("artifact:")]
    parsed = [_artifact_id_version(p) for p in ids]
    if not ids or any(p is None for p in parsed) or len({p[0] for p in parsed}) != 1:
        raise PermissionError("artifact citations must name one versioned lineage")
    slug = parsed[0][0]
    if note.name != slugify("artifact-" + slug):
        raise PermissionError("artifact note name does not match its source lineage")
    records = []
    seen = set()
    for event in _artifact_records(root):
        version = _artifact_id_version(event.id)
        if version is None or version[0] != slug:
            continue
        if event.id in seen:
            raise PermissionError("ambiguous duplicate artifact source id")
        seen.add(event.id)
        payload = event.payload
        if (
            payload.get("slug") != slug
            or type(payload.get("version")) is not int
            or payload["version"] != version[1]
            or version[1] < 1
        ):
            raise PermissionError("artifact source version disagrees with its identity")
        for key in ("title", "description", "url", "text_excerpt", "domain"):
            if key in payload and not isinstance(payload[key], str):
                raise PermissionError("artifact source text field has an invalid shape")
        if (
            not isinstance(payload.get("source", {}), dict)
            or not isinstance(payload.get("tags", []), list)
            or any(not isinstance(tag, str) for tag in payload.get("tags", []))
        ):
            raise PermissionError("artifact source metadata has an invalid shape")
        records.append(event)
    if not records or not set(ids) <= seen:
        raise PermissionError(
            "artifact citation is outside the authenticated source prefix"
        )
    newest = max(records, key=lambda event: _artifact_id_version(event.id)[1])
    dated = _latest_projection_events(records, lambda event: True)
    if newest.id not in ids or len(dated) != 1 or dated[0].id != newest.id:
        raise PermissionError(
            "artifact version and source time do not establish the same newest event"
        )
    cited = sorted(
        (event for event in records if event.id in ids),
        key=lambda event: _artifact_id_version(event.id)[1],
    )
    payload = newest.payload
    domain = "internal"
    for event in cited:
        domain = event.payload.get("domain") or domain
    allowed_provenance = (
        set(ids)
        | {MARKER}
        | {event.payload["url"] for event in cited if event.payload.get("url")}
    )
    exact = (
        set(note.provenance) <= allowed_provenance
        and note.body == _artifact_body(payload, newest.author)
        and note.description == (payload.get("title") or note.name).strip()[:160]
        and note.title == (payload.get("title") or note.name)
        and note.resource == payload.get("url", "")
        and note.domain == domain
        and note.updated == newest.occurred_at[:10]
        and note.tags == [tag for tag in payload.get("tags", []) if tag]
        and sorted(note.acl) == sorted(combine_acl(*(event.acl for event in cited)))
        and sorted(note.entities)
        == sorted({entity for event in cited for entity in event.entities})
    )
    if not exact:
        raise PermissionError(
            "artifact fields are not the exact authenticated projection"
        )
    return {
        "validator": METHOD,
        "newest_event": newest.id,
        "sources": [
            {
                "id": event.id,
                "sha256": hashlib.sha256(event.to_json().encode()).hexdigest(),
            }
            for event in cited
        ],
        "checkpoint_sha256": hashlib.sha256(
            Path(root, "state", "event_validation_checkpoint.json").read_bytes()
        ).hexdigest(),
    }
