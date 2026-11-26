"""Contradicted-verdict lookup over the append-only verification ledger.

``retrieve/verification.py`` answers *whether* a note's latest audit row failed.
This module answers *what* the verifier contradicted — the claim, the evidence
quote and the event ids behind it — so a note with a real contradiction can be
**served with a label** instead of parked in a review queue nobody consumes.

Why it is a separate reader rather than more fields on ``VerificationAudit``:

* the gate loader is a security surface that fails closed on any anomaly, and
  it deliberately keeps only the small, fixed set of fields the gate needs;
* this is a *presentation* lookup — it never widens or narrows what a reader is
  allowed to see, and a damaged ledger simply yields no label;
* note files are **never rewritten** to carry it.  The verification ledger is
  the single source of truth for the label, exactly as it is for the digest
  ledger's ``contested`` terminal state, so the read path and the pipeline can
  never disagree about which claim is in dispute.

Cost.  The ledger can grow large, so the scan is byte-
oriented: a regex pulls the note name out of each physical line and only lines
that also mention a ``contradicted`` verdict are JSON-parsed.  The result is
memoised per corpus root against the (size, mtime, ctime, inode) signature of
every ledger file.  The fold of the sealed segments is also memoised, so an
append to the current ledger re-scans only that file, not every sealed segment,
and a rotation folds only the new segment
onto the cached prefix.  A sealed fold is kept only when every file in it was
read successfully and had been still for ``_SEGMENT_MIN_AGE_NS`` (the racy-git
rule); a failed read is never cached, so the next call retries it.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

from retrieve import telemetry

#: How many contradicted claims a row carries.  A label is a pointer to the
#: dispute, not a transcript of it.
MAX_CONTESTED = 3

#: Emitted by ``distill.verify``; the only verdict token that means "the source
#: says the opposite", as opposed to "the source does not say this".
CONTRADICTED = "contradicted"

_NOTE_NAME_RE = re.compile(rb'"note"\s*:\s*"([a-z0-9][a-z0-9-]*)"')
_CONTRADICTED_RE = re.compile(rb'"verdict"\s*:\s*"contradicted"')
_STATUS_FLAGGED_RE = re.compile(rb'"status"\s*:\s*"flagged"')

_SEGMENT_MANIFEST = "verification.segments.json"
_CURRENT_LEDGER = "verification.jsonl"

_CACHE: dict[str, tuple[tuple, dict[str, list[dict]]]] = {}
_CACHE_LOCK = threading.Lock()
# root -> (signature entries of the sealed segments, their folded index).
_SEALED_FOLDS: dict[str, tuple[tuple, dict[str, list[dict]]]] = {}
_SEGMENT_MIN_AGE_NS = 2_000_000_000


def ledger_paths(root: str) -> list[str]:
    """Sealed segments in manifest order, then the current ledger.

    Same ordering contract the gate loader uses: a later row for a note
    supersedes an earlier one, so the current file must be scanned last.
    """
    state = os.path.join(os.path.abspath(root), "state")
    paths: list[str] = []
    manifest = os.path.join(state, _SEGMENT_MANIFEST)
    try:
        with open(manifest, encoding="utf-8") as handle:
            rows = (json.load(handle) or {}).get("segments") or []
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        rows = []
    if isinstance(rows, list):
        for row in rows:
            name = row.get("name") if isinstance(row, dict) else None
            if not isinstance(name, str) or "/" in name or name.startswith("."):
                continue
            candidate = os.path.join(state, name)
            if os.path.exists(candidate):
                paths.append(candidate)
    current = os.path.join(state, _CURRENT_LEDGER)
    if os.path.exists(current):
        paths.append(current)
    return paths


def _signature(paths: list[str]) -> tuple:
    signature = []
    for path in paths:
        try:
            info = os.stat(path)
        except OSError:
            return ()
        signature.append((path, info.st_size, info.st_mtime_ns, info.st_ino,
                          info.st_ctime_ns))
    return tuple(signature)


def _contested_verdicts(row: dict) -> list[dict]:
    """The contradicted verdicts of one audit row, trimmed to what a label needs."""
    out: list[dict] = []
    for verdict in (row.get("verdicts") or []):
        if not isinstance(verdict, dict):
            continue
        if verdict.get("verdict") != CONTRADICTED:
            continue
        # A validation error produces ``skipped``, never a real contradiction;
        # labeling one would advertise a parser failure as a dispute.
        if verdict.get("validation_error"):
            continue
        out.append({
            "claim": str(verdict.get("claim") or ""),
            "evidence_quote": str(verdict.get("evidence_quote") or ""),
            "event_ids": [str(item) for item in (verdict.get("event_ids") or [])],
        })
        if len(out) >= MAX_CONTESTED:
            break
    return out


def _scan_one(path: str) -> dict[str, list[dict] | None] | None:
    """Each note's LATEST row in one file: its contested verdicts, or ``None``.

    ``None`` records that the note's last row here is clean, so a later file's
    clean row erases an earlier file's finding when the maps are folded.  The
    whole result is ``None`` when the file could not be opened.
    """
    last: dict[str, list[dict] | None] = {}
    try:
        handle = open(path, "rb")
    except OSError:
        return None
    with handle:
        for line in handle:
            match = _NOTE_NAME_RE.search(line)
            if match is None:
                continue
            name = match.group(1).decode("ascii")
            if not (_CONTRADICTED_RE.search(line)
                    and _STATUS_FLAGGED_RE.search(line)):
                last[name] = None
                continue
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                last[name] = None
                continue
            if not isinstance(row, dict) or row.get("note") != name:
                last[name] = None
                continue
            if row.get("status") != "flagged":
                last[name] = None
                continue
            last[name] = _contested_verdicts(row) or None
    return last


def _apply(latest: dict[str, list[dict]],
           last: dict[str, list[dict] | None]) -> None:
    for name, verdicts in last.items():
        if verdicts:
            latest[name] = verdicts
        else:
            latest.pop(name, None)


def _scan(paths: list[str]) -> dict[str, list[dict]]:
    """Latest-row contested verdicts per note, oldest segment first.

    Only the LATEST row counts: a note that was contradicted last month and
    re-verified clean this morning is not contested, so a later row without a
    contradiction must erase the earlier finding rather than accumulate with it.
    An unreadable file contributes nothing, as before.
    """
    latest: dict[str, list[dict]] = {}
    for path in paths:
        _apply(latest, _scan_one(path) or {})
    return latest


def _scan_cached(root: str, paths: list[str],
                 signature: tuple) -> tuple[dict[str, list[dict]], bool]:
    """``_scan`` reusing the fold of the sealed prefix.

    Returns the index and whether every file was read; an incomplete index is
    served but never cached.
    """
    now_ns = time.time_ns()
    sealed = [(path, entry) for path, entry in zip(paths, signature)
              if os.path.basename(path) != _CURRENT_LEDGER]
    current = [(path, entry) for path, entry in zip(paths, signature)
               if os.path.basename(path) == _CURRENT_LEDGER]
    sealed_signature = tuple(entry for _path, entry in sealed)
    with _CACHE_LOCK:
        cached = _SEALED_FOLDS.get(root)
    if cached is not None and cached[0] == sealed_signature:
        prefix, base = cached[0], dict(cached[1])
    elif (cached is not None and len(cached[0]) < len(sealed_signature)
            and sealed_signature[:len(cached[0])] == cached[0]):
        prefix, base = cached[0], dict(cached[1])
    else:
        prefix, base = (), {}
    complete = True
    for path, _entry in sealed[len(prefix):]:
        last = _scan_one(path)
        if last is None:
            complete = False
            continue
        _apply(base, last)
    settled = all(max(entry[2], entry[4]) <= now_ns - _SEGMENT_MIN_AGE_NS
                  for entry in sealed_signature[len(prefix):])
    if complete and settled and sealed_signature != prefix:
        with _CACHE_LOCK:
            _SEALED_FOLDS[root] = (sealed_signature, dict(base))
    latest = base
    for path, _entry in current:
        last = _scan_one(path)
        if last is None:
            complete = False
            continue
        _apply(latest, last)
    return latest, complete


def contested_index(root: str) -> dict[str, list[dict]]:
    """note name -> contradicted claims of its latest audit row (never empty).

    Best-effort by construction: an unreadable or absent ledger yields ``{}``,
    which means "no note carries a contested label" — the pre-policy behaviour.
    Its wall time is added to the bound read-call record as
    ``contested_scan_ms`` (``retrieve.telemetry``).
    """
    with telemetry.timed("contested_scan_ms"):
        return _contested_index(root)


def _contested_index(root: str) -> dict[str, list[dict]]:
    root = os.path.abspath(root)
    paths = ledger_paths(root)
    signature = _signature(paths)
    with _CACHE_LOCK:
        cached = _CACHE.get(root)
        if cached is not None and cached[0] == signature and signature:
            return cached[1]
    complete = True
    try:
        if signature:
            index, complete = _scan_cached(root, paths, signature)
        else:
            index = _scan(paths)
    except Exception:
        index, complete = {}, False
    if complete:
        with _CACHE_LOCK:
            _CACHE[root] = (signature, index)
    return index


def tail_contested(root: str, name: str, *,
                   tail_bytes: int = 1 << 20) -> list[dict] | None:
    """Contested claims from the note's newest row inside the ledger tail.

    The cheap counterpart to ``contested_index`` for the pipeline: the verifier
    has *just* appended this note's row, so it sits at the very end of the
    current ledger and a bounded tail read is exact.  Returns ``None`` when the
    tail carries no row for the note — the caller then has no evidence either
    way and must not invent a contradiction.
    """
    path = os.path.join(os.path.abspath(root), "state", _CURRENT_LEDGER)
    start = 0
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = max(0, size - max(0, int(tail_bytes)))
            handle.seek(start)
            raw = handle.read()
    except OSError:
        return None
    lines = raw.split(b"\n")
    if start > 0 and lines:
        lines.pop(0)                      # first line is a partial record
    target = name.encode("ascii", "ignore")
    for line in reversed(lines):
        if not line or b'"note"' not in line:
            continue
        match = _NOTE_NAME_RE.search(line)
        if match is None or match.group(1) != target:
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(row, dict) or row.get("note") != name:
            return None
        if row.get("status") != "flagged":
            return []
        return _contested_verdicts(row)
    return None


def contested_for(root: str, name: str | None) -> list[dict]:
    """The contested label for one note, or ``[]`` when it is not contested."""
    if not name:
        return []
    return list(contested_index(root).get(str(name), []))


def is_contested(root: str, name: str | None) -> bool:
    return bool(contested_for(root, name))


def review_policy_settles(config) -> bool:
    """Whether ``digest.review_policy`` is ``settle`` in this config.

    The read-time label and the pipeline's ``contested`` terminal state are the
    same decision seen from two ends, so they share one switch: with the key
    absent every row is byte-identical to the pre-policy contract and no ledger
    is read at all.
    """
    block = (config or {}).get("digest") if isinstance(config, dict) else None
    if not isinstance(block, dict):
        return False
    return block.get("review_policy") == "settle"


def reset_cache() -> None:
    """Drop the memoised index (tests, and callers that just wrote the ledger)."""
    with _CACHE_LOCK:
        _CACHE.clear()
        _SEALED_FOLDS.clear()
