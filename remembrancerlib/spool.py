"""Durable spool for agent writes that arrive while the pipeline holds its lock.

``append_events`` takes ``pipeline.lock`` with ``blocking=False``, so any write
during a nightly/catch-up transaction failed outright. That was tolerable when
the pipeline ran once a night; it became near-total once a catch-up run could
last as long as its timer interval, holding the lock almost continuously.
Agents then lost every ``remembrancer_remember`` payload with "memory pipeline is
busy".

Design mirrors ``artifact_publish_lock``: a SEPARATE inode
(``state/remember_spool.lock``) that the pipeline never acquires, so spooling
can never report busy. Spooled events are drained on the next successful
pipeline-lock acquisition: by the pipeline run itself, which drains the live
spool right after it takes the lock (``event.drain_spool_holding_pipeline_lock``),
and by any later ``append_events`` that wins the lock. Draining is replay-safe
because the event id is the dedup key — the append skips an incoming event
whose id already exists with an identical fingerprint, and refuses only when
the same id carries different immutable data.

A drain never wedges and never deletes a fact. A spool file the store refuses
moves to ``quarantine/`` with its reason beside it, and the other files still
drain; lines that do not parse go to ``quarantine/`` as well. Nothing drains
from quarantine on its own: it waits for a person.

The spool is on disk, not in memory: a crash between accepting a write and
draining it must not lose the fact.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from typing import Callable, Iterator, Optional

from .event import Event
from .locks import remember_spool_lock
from .redact import redact_event

SPOOL_DIRNAME = "remember_spool"
QUARANTINE_DIRNAME = "quarantine"
#: Beside each quarantined file: why the drain set it aside.
REASON_SUFFIX = ".reason"


def spool_dir(root: str) -> str:
    return os.path.join(os.path.abspath(root), "state", SPOOL_DIRNAME)


def quarantine_dir(root: str) -> str:
    return os.path.join(spool_dir(root), QUARANTINE_DIRNAME)


def _spool_files(root: str) -> list[str]:
    directory = spool_dir(root)
    try:
        names = sorted(f for f in os.listdir(directory) if f.endswith(".jsonl"))
    except FileNotFoundError:
        return []
    return [os.path.join(directory, name) for name in names]


def spool_depth(root: str) -> int:
    """Number of events currently waiting to be drained."""
    total = 0
    for path in _spool_files(root):
        try:
            with open(path, encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
        except OSError:
            continue
    return total


def spool_events(root: str, events) -> int:
    """Persist events for later draining. Never blocks on the pipeline."""
    # Redact before the spool file exists: it sits on disk until drained.
    pending = [redact_event(e)[0] for e in events]
    if not pending:
        return 0
    directory = spool_dir(root)
    # Timestamp-ordered name keeps drain order stable; uuid avoids collisions
    # between concurrent writers that both hold the spool lock in turn.
    name = f"{int(time.time() * 1000):015d}-{uuid.uuid4().hex}.jsonl"
    target = os.path.join(directory, name)
    with remember_spool_lock(root):
        os.makedirs(directory, exist_ok=True)
        # The lock creates and syncs state when missing. Persist the spool
        # directory entry there before acknowledging any of its files.
        _fsync_directory(os.path.dirname(directory))
        # tmp + os.replace: a crash mid-write must never leave a partial line
        # that would fail to parse and wedge every future drain.
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".spool-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for event in pending:
                    handle.write(event.to_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
            _fsync_directory(directory)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
    return len(pending)


def iter_spooled(root: str) -> Iterator[Event]:
    """Yield every spooled event, oldest file first."""
    for path in _spool_files(root):
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield Event.from_dict(json.loads(line))
                    except Exception:
                        # A single unparseable record must not block the rest.
                        continue
        except OSError:
            continue


def quarantined_files(root: str) -> list[str]:
    """Spool files and unparseable lines the drain set aside, oldest name first."""
    directory = quarantine_dir(root)
    try:
        names = sorted(f for f in os.listdir(directory) if f.endswith(".jsonl"))
    except FileNotFoundError:
        return []
    return [os.path.join(directory, name) for name in names]


def _report(message: str) -> None:
    print(f"[remembrancerlib] remember spool: {message}", file=sys.stderr, flush=True)


def _fsync_directory(directory: str) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_spool_file(path: str) -> tuple[list[Event], list[bytes]]:
    """One spool file's events, and its raw lines that do not parse as one."""
    with open(path, "rb") as handle:
        data = handle.read()
    events: list[Event] = []
    unparseable: list[bytes] = []
    for raw in data.split(b"\n"):
        if not raw.strip():
            continue
        try:
            events.append(Event.from_dict(json.loads(raw.decode("utf-8"))))
        except Exception:
            unparseable.append(raw)
    return events, unparseable


def _quarantine(root: str, name: str, payload: Optional[bytes], source: Optional[str],
                reason: str) -> None:
    """Durably place ``name`` (and its reason) in quarantine.

    Either ``payload`` is written as the file, or ``source`` is moved there.
    The reason goes first, so a crash leaves at worst a reason with no file;
    the next drain then quarantines the same file again.
    """
    from .atomic import durable_replace_bytes

    directory = quarantine_dir(root)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    _fsync_directory(spool_dir(root))
    target = os.path.join(directory, name)
    durable_replace_bytes(target + REASON_SUFFIX, (reason + "\n").encode("utf-8"), mode=0o600)
    if payload is not None:
        durable_replace_bytes(target, payload, mode=0o600)
    else:
        os.replace(source, target)
        _fsync_directory(directory)
    _report(f"quarantined {name}: {reason}")


DrainOutcome = Optional[BaseException]


def drain_spool(root: str,
                append_batches: Callable[[list[list[Event]]], list[DrainOutcome]]) -> int:
    """Append every spooled event, then clear the spool. Returns the events drained.

    ``append_batches`` is called by the caller that ALREADY owns
    ``pipeline.lock``; this function never acquires it. It gets one batch per
    spool file, in spool order, checks them all against the store in one scan,
    and returns one entry per batch: ``None`` when the batch landed (or was
    already there), else the error that refused it. It raises when the store
    itself cannot take appends; the drain then stops and every file stays.

    A refused file moves to quarantine with its reason, and the files after it
    still drain. Lines that do not parse are written to quarantine before their
    file is removed. Files are removed only after their events are durably
    appended, so a crash mid-drain replays them (idempotently) rather than
    dropping them.
    """
    if not _spool_files(root):
        return 0
    drained = 0
    with remember_spool_lock(root):
        loaded: list[tuple[str, list[Event], list[bytes]]] = []
        for path in _spool_files(root):
            try:
                events, unparseable = _read_spool_file(path)
            except OSError as exc:
                _report(f"left {os.path.basename(path)} in the spool, unreadable: {exc}")
                continue
            loaded.append((path, events, unparseable))
        if not loaded:
            return 0
        outcomes = list(append_batches([events for _path, events, _bad in loaded]))
        if len(outcomes) != len(loaded):
            raise RuntimeError(
                f"spool drain got {len(outcomes)} outcomes for {len(loaded)} files"
            )
        for (path, events, unparseable), refusal in zip(loaded, outcomes):
            name = os.path.basename(path)
            if refusal is not None:
                _quarantine(root, name, None, path,
                            f"refused by the event store: {type(refusal).__name__}: {refusal}")
                continue
            if unparseable:
                stem = name[:-len(".jsonl")]
                _quarantine(
                    root, f"{stem}.unparseable.jsonl",
                    b"".join(raw + b"\n" for raw in unparseable), None,
                    f"{len(unparseable)} line(s) of {name} did not parse as an event; "
                    f"its other {len(events)} event(s) drained",
                )
            try:
                os.unlink(path)
            except OSError:
                pass
            drained += len(events)
    return drained
