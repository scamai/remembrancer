"""Durable review queue for grounded fallback answers.

Only an answer explicitly confirmed by a configured identity can become an L2
note. Queue mutation and note persistence form one locked transaction: notes are
durable before rows are marked promoted, and any ordinary failure restores both
the queue and every note byte-for-byte. A crash after the note write but before
the queue commit is repaired idempotently by the next pass.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from remembrancerlib import (  # noqa: E402
    Note, acl_can_reveal, current_note_names, human_trust_valid,
    index_notes_by_name, load_notes, pipeline_mutation_lock, slugify, write_note,
)
import govern  # noqa: E402

DEFAULT = os.path.join(ROOT, "state", "fallback_candidates.jsonl")


def _key(question: str, answer: str) -> str:
    return hashlib.sha256(f"{question}\x00{answer}".encode("utf-8")).hexdigest()


def _root_for(path: str, root: str | None = None) -> str:
    if root:
        return os.path.abspath(root)
    parent = os.path.dirname(os.path.abspath(path))
    return os.path.dirname(parent) if os.path.basename(parent) == "state" else parent


@contextmanager
def _queue_lock(path: str, *, exclusive: bool):
    """Lock a stable sidecar inode so atomic queue replacement stays serialized."""
    lock_path = os.path.abspath(path) + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_rows_unlocked(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed fallback queue line {lineno}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"fallback queue line {lineno} is not an object")
            rows.append(row)
    return rows


def list_candidates(path: str = DEFAULT) -> list[dict]:
    with _queue_lock(path, exclusive=False):
        return _read_rows_unlocked(path)


def log_candidate(question, answer, citations, tier, path=DEFAULT,
                  identity=None, acl=None, root: str | None = None):
    if not citations:
        return  # grounding rule: never queue an uncited answer
    if not isinstance(citations, list) or any(not isinstance(c, str) for c in citations):
        raise ValueError("fallback citations must be a list of event ids")
    if acl is not None and (
        not isinstance(acl, list) or any(not isinstance(label, str) for label in acl)
    ):
        raise ValueError("fallback ACL must be a list")
    effective_root = _root_for(path, root)
    with pipeline_mutation_lock(effective_root, blocking=False):
        with _queue_lock(path, exclusive=True):
            rows = _read_rows_unlocked(path)
            candidate_key = _key(str(question), str(answer))
            if any(_key(str(row.get("question", "")), str(row.get("answer", "")))
                   == candidate_key for row in rows):
                return
            row = {
                "question": str(question),
                "answer": str(answer),
                "citations": list(dict.fromkeys(citations)),
                "tier": str(tier),
                "identity": str(identity or "").strip(),
                "promoted": False,
            }
            # [] is a real public ACL, while an omitted ACL is unknown. Keep
            # that distinction so a legacy/direct caller can never silently
            # turn missing policy into a public synthesis during promotion.
            if acl is not None:
                row["acl"] = list(acl)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def _load_identities(root: str) -> dict[str, list[str]]:
    path = os.path.join(root, "config.json")
    try:
        with open(path, encoding="utf-8") as handle:
            cfg = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PermissionError("cannot authenticate fallback confirmation identity") from exc
    identities = cfg.get("identities")
    if not isinstance(identities, dict):
        raise PermissionError("fallback promotion requires configured identities")
    return identities


def _validated_note(row: dict, identities: dict[str, list[str]]) -> tuple[Note, str]:
    if row.get("tier") != "user-confirmed":
        raise PermissionError("only user-confirmed fallback rows may auto-promote")
    identity = str(row.get("identity") or "").strip()
    if not identity or identity not in identities:
        raise PermissionError("confirmation identity is missing or unconfigured")
    grants = identities[identity]
    if not isinstance(grants, list) or any(not isinstance(label, str) for label in grants):
        raise PermissionError("confirmation identity has malformed grants")
    if "acl" not in row or not isinstance(row["acl"], list) or any(
        not isinstance(label, str) for label in row["acl"]
    ):
        raise ValueError("confirmed fallback row has no trustworthy ACL")
    acl = list(dict.fromkeys(label.strip() for label in row["acl"] if label.strip()))
    if acl and not set(acl).intersection(grants):
        raise PermissionError("confirmation identity is not authorized for the fallback ACL")
    citations = row.get("citations")
    if not isinstance(citations, list) or not citations or any(
        not isinstance(citation, str) or not citation.strip() for citation in citations
    ):
        raise ValueError("confirmed fallback row requires citations")
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not question or not answer:
        raise ValueError("confirmed fallback row requires question and answer")
    fallback_key = _key(question, answer)
    name = f"fallback-{slugify(question)[:36]}-{fallback_key[:16]}"
    description = " ".join(answer.split())[:200]
    note = Note(
        name=name,
        description=description,
        type="synthesis",
        domain="synthesis",
        acl=acl,
        body=answer,
        provenance=[
            *list(dict.fromkeys(citations)),
            f"fallback-key:{fallback_key}",
            f"confirmed-by:{identity}",
        ],
        state="proposed",
    )
    # A configured identity confirms that this candidate should enter review;
    # it is not the signed trust authority that may publish synthesized facts.
    govern.prepare_for_review(note, "fallback-synthesis")
    return note, fallback_key


def _fsync_dir(path: str) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace(path: str, payload: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".fallback-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _queue_payload(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def _commit_queue(path: str, rows: list[dict]) -> None:
    """Commit a complete JSONL image; split out for failure-injection tests."""
    _atomic_replace(path, _queue_payload(rows))


def _restore_path(path: str, before: bytes | None) -> None:
    if before is None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            return
        _fsync_dir(os.path.dirname(path))
        return
    _atomic_replace(path, before)


def _ensure_durable_note(
    note: Note, fallback_key: str, context_dir: str,
) -> tuple[str, bytes | None, bool, Note]:
    notes = index_notes_by_name(
        load_notes(context_dir), operation="fallback confirmed promotion",
    )
    current = current_note_names(notes.values())
    for relation, target_name in note.typed_links:
        if relation == "supersedes":
            raise ValueError("fallback answers cannot create lifecycle edges")
        target = notes.get(target_name)
        if (target is None or target.name not in current
                or not acl_can_reveal(note.acl, target.acl)):
            raise PermissionError(
                f"fallback answer contains hidden or non-current link {target_name!r}"
            )
    existing = notes.get(note.name)
    if existing is not None:
        marker = f"fallback-key:{fallback_key}"
        expected_provenance = set(note.provenance or [])
        if (
            existing.name not in current
            or marker not in (existing.provenance or [])
            or not expected_provenance.issubset(set(existing.provenance or []))
            or existing.type != note.type
            or existing.domain != note.domain
            or list(existing.acl or []) != list(note.acl or [])
            or existing.description != note.description
            or existing.body.strip() != note.body.strip()
        ):
            raise FileExistsError(f"fallback note identity collision: {note.name}")
        # Crash recovery / retry: the exact durable note already exists. Its
        # trust state may have been promoted by a human, so never demote it;
        # raw/legacy authority claims are not accepted as completion.
        if existing.state == "verified" and human_trust_valid(existing):
            return existing.path, None, False, existing
        if existing.state != "proposed":
            raise PermissionError(
                "fallback synthesis has no valid reviewable trust state"
            )
        return existing.path, None, False, existing

    path = os.path.join(context_dir, note.domain, f"{slugify(note.name)}.md")
    before = open(path, "rb").read() if os.path.exists(path) else None
    try:
        written = write_note(note, context_dir)
        if not os.path.isfile(written):
            raise RuntimeError("fallback note was suppressed or not durably written")
        stored = index_notes_by_name(
            load_notes(context_dir), operation="fallback post-write verification",
        ).get(note.name)
        if stored is None or f"fallback-key:{fallback_key}" not in (stored.provenance or []):
            raise RuntimeError("fallback note durability verification failed")
    except Exception:
        _restore_path(path, before)
        raise
    return written, before, True, stored


def _promote_rows_locked(rows: list[dict], indices: list[int], path: str, root: str) -> list[str]:
    identities = _load_identities(root)
    prepared: list[tuple[int, Note, str]] = []
    for index in indices:
        if index < 0 or index >= len(rows):
            raise IndexError(f"no candidate #{index}")
        if rows[index].get("promoted"):
            continue
        note, fallback_key = _validated_note(rows[index], identities)
        prepared.append((index, note, fallback_key))

    context_dir = os.path.join(root, "context")
    queue_before = open(path, "rb").read() if os.path.exists(path) else None
    snapshots: list[tuple[str, bytes | None, str | None]] = []
    newly_enqueued: set[str] = set()
    queue_before_ids = {
        str(item.get("id")) for item in govern.pending(root)
        if isinstance(item.get("id"), str)
    }
    promoted_names: list[str] = []
    updated = copy.deepcopy(rows)
    try:
        for index, note, fallback_key in prepared:
            note_path, before, wrote, stored = _ensure_durable_note(
                note, fallback_key, context_dir,
            )
            approval_id: str | None = None
            if not (stored.state == "verified" and human_trust_valid(stored)):
                try:
                    approval_id = govern.enqueue_for_review(
                        stored,
                        requester=str(rows[index].get("identity") or "fallback"),
                        reason=f"fallback:{fallback_key}", root=root,
                        cls="fallback-synthesis",
                    )
                except Exception:
                    if wrote:
                        _restore_path(note_path, before)
                    raise
                if approval_id not in queue_before_ids:
                    newly_enqueued.add(approval_id)
            if wrote:
                snapshots.append((note_path, before, approval_id))
            updated[index]["promoted"] = True
            updated[index]["promoted_note"] = note.name
            if approval_id is not None:
                updated[index]["approval_id"] = approval_id
            promoted_names.append(note.name)
        _commit_queue(path, updated)
    except Exception as exc:
        withdrawal_failures: set[str] = set()
        for approval_id in sorted(newly_enqueued):
            try:
                govern.record(
                    root, approval_id, "withdrawn", "fallback-queue",
                    reason="fallback transaction rolled back",
                )
            except Exception:
                withdrawal_failures.add(approval_id)
        for note_path, before, approval_id in reversed(snapshots):
            # If the pending approval could not be withdrawn, retain its exact
            # hard-held target for safe retry instead of creating a dangling
            # actionable queue item.
            if approval_id is not None and approval_id not in newly_enqueued:
                continue
            if approval_id not in withdrawal_failures:
                _restore_path(note_path, before)
        if queue_before is None:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        else:
            _atomic_replace(path, queue_before)
        if withdrawal_failures:
            raise RuntimeError(
                "fallback rollback retained held notes because approval "
                "withdrawal failed"
            ) from exc
        raise
    return promoted_names


def promote(index, apply, path=DEFAULT, root: str | None = None):
    if not apply:
        rows = list_candidates(path)
        if index < 0 or index >= len(rows):
            return f"no candidate #{index}"
        return f"DRY-RUN would promote: {str(rows[index].get('question', ''))[:60]}"

    effective_root = _root_for(path, root)
    with pipeline_mutation_lock(effective_root, blocking=False):
        with _queue_lock(path, exclusive=True):
            rows = _read_rows_unlocked(path)
            if index < 0 or index >= len(rows):
                return f"no candidate #{index}"
            if rows[index].get("promoted"):
                return f"already promoted #{index}"
            names = _promote_rows_locked(rows, [index], path, effective_root)
    return f"promoted #{index} -> context/synthesis/{names[0]}.md"


def promote_confirmed(apply, path=DEFAULT, tiers=("user-confirmed",),
                      root: str | None = None):
    """Batch-promote confirmed rows as one serialized, rollback-safe commit."""
    if not apply:
        rows = list_candidates(path)
        pending = [row for row in rows if row.get("tier") in tiers and not row.get("promoted")]
        return {"promoted": 0, "skipped": len(rows) - len(pending), "would_promote": len(pending)}

    effective_root = _root_for(path, root)
    with pipeline_mutation_lock(effective_root, blocking=False):
        with _queue_lock(path, exclusive=True):
            rows = _read_rows_unlocked(path)
            indices = [
                index for index, row in enumerate(rows)
                if row.get("tier") in tiers and not row.get("promoted")
            ]
            names = _promote_rows_locked(rows, indices, path, effective_root)
    return {"promoted": len(names), "skipped": len(rows) - len(indices)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fallback candidate review queue.")
    parser.add_argument("--list", action="store_true", help="print candidates")
    parser.add_argument("--promote-confirmed", action="store_true",
                        help="promote user-confirmed candidates (dry-run without --apply)")
    parser.add_argument("--apply", action="store_true", help="actually write notes")
    parser.add_argument("--path", default=DEFAULT, help="candidates jsonl path")
    parser.add_argument("--root", default=ROOT, help="Remembrancer repository root")
    arguments = parser.parse_args()

    if arguments.list:
        for i, row in enumerate(list_candidates(arguments.path)):
            flag = "PROMOTED" if row.get("promoted") else "pending "
            print(f"#{i} [{flag}] ({row.get('tier')}) {str(row.get('question', ''))[:70]}")
    elif arguments.promote_confirmed:
        result = promote_confirmed(
            apply=arguments.apply, path=arguments.path, root=arguments.root,
        )
        mode = "APPLY" if arguments.apply else "DRY-RUN"
        suffix = f" would_promote={result.get('would_promote', 0)}" if not arguments.apply else ""
        print(
            f"[{mode}] promote-confirmed: promoted={result['promoted']} "
            f"skipped={result['skipped']}{suffix}"
        )
    else:
        parser.print_help()
