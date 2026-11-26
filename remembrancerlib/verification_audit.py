"""Rollback-evident head records for the verification audit ledger.

``verification.jsonl`` is append-only, but a missing or truncated file used to
look exactly like a repository that had never enabled verification.  The
tracked ``verification_audit_required`` cutover bit prevents deletion only when
the companion head is also required.  Each head row binds the complete audit
prefix and chains to the previous head, so an accidental delete/truncate of
either file fails closed instead of reviving a prior verdict.

This is an integrity anchor, not an authenticity boundary against an attacker
who can rewrite both runtime ledgers and execute code as the Remembrancer service
user.  That stronger boundary belongs in off-host/WORM backup policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
from typing import Any


HEAD_VERSION = 1
HEAD_RELATIVE_PATH = "state/verification.head.jsonl"
_HEAD_KEYS = frozenset({
    "version", "audit_size", "audit_sha256", "previous_head", "head_id",
})
_SHA256_RE = re.compile(r"[a-f0-9]{64}\Z")
MAX_HEAD_BYTES = 128 * 1024 * 1024
MAX_HEAD_LINE_BYTES = 4 * 1024
MAX_AUDIT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class VerificationAuditHead:
    audit_size: int
    audit_sha256: str
    previous_head: str
    head_id: str


def head_path_for_audit(audit_path: str) -> str:
    """Return the sibling rollback-anchor path for an audit filename."""
    absolute = os.path.abspath(audit_path)
    if absolute.endswith(".jsonl"):
        return absolute[:-6] + ".head.jsonl"
    return absolute + ".head.jsonl"


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _head_material(
    *, audit_size: int, audit_sha256: str, previous_head: str,
) -> dict[str, Any]:
    return {
        "version": HEAD_VERSION,
        "audit_size": audit_size,
        "audit_sha256": audit_sha256,
        "previous_head": previous_head,
    }


def make_head(audit: bytes, previous: VerificationAuditHead | None) -> dict[str, Any]:
    """Return the next exact prefix head for ``audit``."""
    previous_id = previous.head_id if previous is not None else ""
    material = _head_material(
        audit_size=len(audit),
        audit_sha256=hashlib.sha256(audit).hexdigest(),
        previous_head=previous_id,
    )
    return {**material, "head_id": _canonical_hash(material)}


def encode_head(row: dict[str, Any]) -> bytes:
    return (
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def parse_heads(raw: bytes) -> tuple[list[VerificationAuditHead], bool]:
    """Strictly parse a complete chained head ledger."""
    if (
        len(raw) > MAX_HEAD_BYTES
        or not raw
        or not raw.endswith(b"\n")
    ):
        return [], False
    heads: list[VerificationAuditHead] = []
    previous: VerificationAuditHead | None = None
    for encoded in raw.splitlines():
        if not encoded or len(encoded) > MAX_HEAD_LINE_BYTES:
            return [], False
        try:
            row = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return [], False
        if not isinstance(row, dict) or set(row) != _HEAD_KEYS:
            return [], False
        size = row.get("audit_size")
        digest = row.get("audit_sha256")
        previous_id = row.get("previous_head")
        head_id = row.get("head_id")
        if (
            row.get("version") != HEAD_VERSION
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or not isinstance(previous_id, str)
            or (previous_id and _SHA256_RE.fullmatch(previous_id) is None)
            or not isinstance(head_id, str)
            or _SHA256_RE.fullmatch(head_id) is None
        ):
            return [], False
        if previous is None:
            if previous_id:
                return [], False
        elif (
            previous_id != previous.head_id
            or size <= previous.audit_size
        ):
            return [], False
        material = _head_material(
            audit_size=size,
            audit_sha256=digest,
            previous_head=previous_id,
        )
        if head_id != _canonical_hash(material):
            return [], False
        current = VerificationAuditHead(
            audit_size=size,
            audit_sha256=digest,
            previous_head=previous_id,
            head_id=head_id,
        )
        heads.append(current)
        previous = current
    return heads, True


def head_matches_audit(head: VerificationAuditHead, audit: bytes) -> bool:
    return bool(
        head.audit_size == len(audit)
        and head.audit_sha256 == hashlib.sha256(audit).hexdigest()
    )


def head_matches_prefix(head: VerificationAuditHead, audit: bytes) -> bool:
    return bool(
        head.audit_size <= len(audit)
        and head.audit_sha256
        == hashlib.sha256(audit[:head.audit_size]).hexdigest()
    )


def read_descriptor_all(descriptor: int, maximum: int) -> bytes:
    """Read a locked regular-file descriptor fully despite legal short reads."""
    metadata = os.fstat(descriptor)
    if metadata.st_size > maximum:
        raise OSError("ledger exceeds safety bound")
    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = metadata.st_size
    chunks: list[bytes] = []
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise OSError("short ledger read")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.fstat(descriptor).st_size != metadata.st_size:
        raise OSError("ledger size changed while locked")
    return b"".join(chunks)


def _append_all(
    descriptor: int,
    payload: bytes,
    progress: list[int] | None = None,
) -> int:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError(f"short ledger append: {offset}/{len(payload)}")
        offset += written
        if progress is not None:
            progress[0] = offset
    return offset


def _read_from_current(descriptor: int, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _rollback_exact_append(
    descriptor: int, original_size: int, payload: bytes, written: int,
) -> None:
    """Rollback only bytes proven to be this writer's exact partial suffix."""
    if written <= 0 or os.fstat(descriptor).st_size != original_size + written:
        return
    os.lseek(descriptor, original_size, os.SEEK_SET)
    if _read_from_current(descriptor, written) != payload[:written]:
        return
    os.ftruncate(descriptor, original_size)
    os.fsync(descriptor)


# --------------------------------------------------------------------------- #
# segment rotation — the 128 MiB cap was a wall, not a valve
# --------------------------------------------------------------------------- #
# Before this, an append that would cross MAX_AUDIT_BYTES raised, and
# distill/engine.py caught that and substituted VerificationResult(0,0,0,
# "skipped"). So the ledger filling up silently turned every verdict into
# "skipped" -- no note could ever reach `verified` again, `auto_approve` (which
# requires exactly that) drained nothing, and the approval queue grew without
# bound. Once the ledger sat closer to the cap than one typical record, every
# append failed and verification stopped, silently.
#
# Rotation seals the full segment and starts an empty one. Nothing is dropped:
# an audit ledger is evidence, so history is retained rather than compacted.
#
# The cross-segment link cannot live in the new segment's genesis head --
# `parse_heads` requires the first head of a file to carry `previous_head: ""`,
# and relaxing that would let a truncated ledger pass as a complete one. It
# lives in a small manifest instead, written atomically at rotation, so
# `parse_heads` is untouched: each segment still self-authenticates against its
# own head file, and the manifest records their order and final head ids so a
# deleted or reordered segment is detectable.
SEGMENTS_RELATIVE_PATH = "state/verification.segments.json"
SEGMENT_VERSION = 1
MAX_SEGMENTS_BYTES = 4 * 1024 * 1024
_SEGMENT_NAME_RE = re.compile(r"\Averification\.([1-9][0-9]{0,5})\.jsonl\Z")


# The pre-flight seals the segment BEFORE a transaction snapshots it, so a
# stage never needs to rotate mid-run.  The headroom below the cap must
# comfortably exceed one run's append volume (a heavy run writes a few
# MiB).  16 MiB is >10x that.
ROTATION_HEADROOM_BYTES = 16 * 1024 * 1024
ROTATION_HIGH_WATER_BYTES = MAX_AUDIT_BYTES - ROTATION_HEADROOM_BYTES


def rotate_verification_audit_if_full(
    audit_path: str,
    high_water: int = ROTATION_HIGH_WATER_BYTES,
) -> int:
    """Seal the segment when it is at/above the high-water mark.

    The ONLY sanctioned caller is the pipeline pre-flight: live tree, under
    the pipeline lock, before the initial append-only snapshot.  Rotating
    anywhere else can race an in-flight isolated transaction's copyback.
    Returns the sealed segment index, or 0 when below the mark or missing.
    """
    try:
        size = os.path.getsize(audit_path)
    except FileNotFoundError:
        return 0
    if size < high_water:
        return 0
    return rotate_verification_audit(audit_path)


def segments_path_for_audit(audit_path: str) -> str:
    """The manifest that records the sealed segments beside an audit file."""
    absolute = os.path.abspath(audit_path)
    directory = os.path.dirname(absolute)
    base = os.path.basename(absolute)
    stem = base[:-6] if base.endswith(".jsonl") else base
    return os.path.join(directory, stem + ".segments.json")


def segment_audit_name(index: int) -> str:
    return f"verification.{index}.jsonl"


def parse_segments(raw: bytes) -> tuple[list[dict[str, Any]], bool]:
    """Strictly parse the sealed-segment manifest.

    Fails closed on anything unexpected: a manifest that cannot be trusted must
    deny the history it claims to order, not silently shorten it.
    """
    if not raw:
        return [], True                     # never rotated: an honest empty state
    if len(raw) > MAX_SEGMENTS_BYTES:
        return [], False
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [], False
    if not isinstance(document, dict) or document.get("version") != SEGMENT_VERSION:
        return [], False
    rows = document.get("segments")
    if not isinstance(rows, list):
        return [], False
    parsed: list[dict[str, Any]] = []
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or set(row) != {"index", "name", "final_head_id"}:
            return [], False
        index, name, final = row["index"], row["name"], row["final_head_id"]
        if (
            not isinstance(index, int) or isinstance(index, bool)
            or index != position                      # contiguous, in order
            or not isinstance(name, str)
            or name != segment_audit_name(index)
            or not isinstance(final, str)
            or _SHA256_RE.fullmatch(final) is None
        ):
            return [], False
        parsed.append({"index": index, "name": name, "final_head_id": final})
    return parsed, True


def encode_segments(rows: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        {"version": SEGMENT_VERSION, "segments": rows},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _write_segments_manifest(directory_fd: int, name: str,
                             rows: list[dict[str, Any]]) -> None:
    """Replace the manifest atomically within the state directory."""
    payload = encode_segments(rows)
    temporary = name + ".tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
        0o600, dir_fd=directory_fd,
    )
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.rename(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    os.fsync(directory_fd)


def _create_empty_segment(directory_fd: int, audit_name: str,
                          head_name: str) -> None:
    """An empty audit plus a genesis head over zero bytes."""
    audit_descriptor = os.open(
        audit_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600, dir_fd=directory_fd,
    )
    os.fsync(audit_descriptor)
    os.close(audit_descriptor)
    head_descriptor = os.open(
        head_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600, dir_fd=directory_fd,
    )
    try:
        payload = encode_head(make_head(b"", None))
        written = 0
        while written < len(payload):
            written += os.write(head_descriptor, payload[written:])
        os.fsync(head_descriptor)
    finally:
        os.close(head_descriptor)
    os.fsync(directory_fd)


def rotate_verification_audit(audit_path: str) -> int:
    """Seal the current audit segment and start an empty one. Returns the index.

    Callers hold no locks: this takes its own, audit-first then head, the same
    order as the appender, so the two cannot deadlock.
    """
    audit_path = os.path.abspath(audit_path)
    directory = os.path.dirname(audit_path)
    audit_name = os.path.basename(audit_path)
    head_name = os.path.basename(head_path_for_audit(audit_path))
    segments_name = os.path.basename(segments_path_for_audit(audit_path))

    directory_fd = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    audit_descriptor = head_descriptor = None
    try:
        flags = (os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_CLOEXEC", 0))
        audit_descriptor = os.open(audit_name, flags, 0o600, dir_fd=directory_fd)
        head_descriptor = os.open(head_name, flags, 0o600, dir_fd=directory_fd)
        fcntl.flock(audit_descriptor, fcntl.LOCK_EX)
        fcntl.flock(head_descriptor, fcntl.LOCK_EX)

        existing = read_descriptor_all(audit_descriptor, MAX_AUDIT_BYTES)
        head_raw = read_descriptor_all(head_descriptor, MAX_HEAD_BYTES)
        heads, trustworthy = parse_heads(head_raw)
        if not existing:
            return 0                       # nothing to seal
        if not trustworthy or not heads or not head_matches_audit(heads[-1], existing):
            raise OSError(
                "refusing to rotate a verification audit that does not match "
                "its head; attended repair required"
            )

        try:
            manifest_descriptor = os.open(
                segments_name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            rows: list[dict[str, Any]] = []
        else:
            try:
                rows, ok = parse_segments(
                    read_descriptor_all(manifest_descriptor, MAX_SEGMENTS_BYTES))
            finally:
                os.close(manifest_descriptor)
            if not ok:
                raise OSError(
                    "verification segment manifest is unreadable; refusing to "
                    "rotate over history that cannot be ordered"
                )

        index = len(rows) + 1
        sealed_audit = segment_audit_name(index)
        sealed_head = os.path.basename(head_path_for_audit(
            os.path.join(directory, sealed_audit)))
        for candidate in (sealed_audit, sealed_head):
            try:
                os.stat(candidate, dir_fd=directory_fd)
            except FileNotFoundError:
                continue
            raise OSError(f"refusing to overwrite existing segment {candidate!r}")

        # Seal, then publish. If the process dies between the renames and the
        # manifest write, the next call re-derives the same index and the
        # orphaned pair is caught by the overwrite guard above -- loudly, rather
        # than by silently clobbering evidence.
        os.rename(audit_name, sealed_audit,
                  src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.rename(head_name, sealed_head,
                  src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
        _write_segments_manifest(
            directory_fd, segments_name,
            rows + [{"index": index, "name": sealed_audit,
                     "final_head_id": heads[-1].head_id}],
        )
        # Leave a VALID empty segment behind, not an absent one. Between the
        # rename and the next append the reader would otherwise find no audit
        # file at all and fail closed -- denying every retrieval for as long as
        # the gap lasts. An empty audit plus its genesis head reads as "nothing
        # verified in this segment", which is true and safe.
        _create_empty_segment(directory_fd, audit_name, head_name)
        return index
    finally:
        for descriptor in (head_descriptor, audit_descriptor):
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(descriptor)
        os.close(directory_fd)


class _AuditSegmentFull(Exception):
    """This segment cannot take the record; rotate and retry."""


def append_verification_audit_record(rec: dict[str, Any], audit_path: str) -> bool:
    """Append one verdict; a full segment refuses -- rotation is not ours to do.

    Rotation SHRINKS the current segment, and this appender may be running
    inside the isolated nightly transaction, whose publish gate rightly
    refuses an authority log that lost bytes ("deleted its prefix").  Append-time rotation inside the stage once
    made every publish refuse.  The one safe
    rotation point is the pipeline pre-flight -- live tree, under the
    pipeline lock, before the initial append-only snapshot
    (``rotate_verification_audit_if_full``).  With the pre-flight's 16 MiB
    high-water headroom, a refusal here means one run appended more than the
    whole headroom: loud, rare, and self-healing at the next pre-flight.
    """
    try:
        return _append_once(rec, audit_path)
    except _AuditSegmentFull:
        print(
            "[verification-audit] segment is full and append-time rotation is "
            "forbidden inside transactions; the record was NOT written. The "
            "next pipeline pre-flight rotation restores headroom.",
            file=sys.stderr,
        )
        return False


_APPEND_REOPEN_ATTEMPTS = 5


def _names_still_point_at(directory_fd: int, name: str, descriptor: int) -> bool:
    """True when ``name`` in the directory is the very file ``descriptor`` holds."""
    try:
        by_name = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    held = os.fstat(descriptor)
    return (by_name.st_dev, by_name.st_ino) == (held.st_dev, held.st_ino)


def _append_once(rec: dict[str, Any], audit_path: str) -> bool:
    """Append one verdict and its rollback head as a fail-closed pair.

    Locks are always acquired audit-first, then head.  A reader may observe a
    crash boundary only as a mismatch, which denies retrieval.  Ordinary write
    errors roll both exact suffixes back to the prior matching pair.
    """
    directory_fd: int | None = None
    audit_descriptor: int | None = None
    head_descriptor: int | None = None
    try:
        audit_path = os.path.abspath(audit_path)
        directory = os.path.dirname(audit_path)
        os.makedirs(directory, exist_ok=True)
        directory_fd = os.open(
            directory,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        flags = (
            os.O_APPEND
            | os.O_CREAT
            | os.O_RDWR
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        head_path = head_path_for_audit(audit_path)
        if os.path.dirname(head_path) != directory:
            raise OSError("verification head must be beside its audit")
        audit_name = os.path.basename(audit_path)
        head_name = os.path.basename(head_path)
        # Open by name, then lock -- and only then confirm the names still
        # point at the locked files.  A rotation that held the locks while we
        # waited has renamed this pair into a SEALED segment; appending to it
        # would break the manifest's final_head_id and fail every reader
        # closed.  So on a mismatch, drop the stale pair and reopen.
        for _attempt in range(_APPEND_REOPEN_ATTEMPTS):
            audit_descriptor = os.open(audit_name, flags, 0o600, dir_fd=directory_fd)
            head_descriptor = os.open(head_name, flags, 0o600, dir_fd=directory_fd)
            fcntl.flock(audit_descriptor, fcntl.LOCK_EX)
            fcntl.flock(head_descriptor, fcntl.LOCK_EX)
            if (_names_still_point_at(directory_fd, audit_name, audit_descriptor)
                    and _names_still_point_at(directory_fd, head_name, head_descriptor)):
                break
            for descriptor in (head_descriptor, audit_descriptor):
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(descriptor)
            audit_descriptor = head_descriptor = None
        else:
            raise OSError("verification ledger kept moving while appending")
        audit_metadata = os.fstat(audit_descriptor)
        head_metadata = os.fstat(head_descriptor)
        if (
            not stat.S_ISREG(audit_metadata.st_mode)
            or not stat.S_ISREG(head_metadata.st_mode)
        ):
            raise OSError("verification ledger is not a regular file")
        os.fchmod(audit_descriptor, 0o600)
        os.fchmod(head_descriptor, 0o600)
        existing = read_descriptor_all(audit_descriptor, MAX_AUDIT_BYTES)
        head_raw = read_descriptor_all(head_descriptor, MAX_HEAD_BYTES)
        heads: list[VerificationAuditHead] = []
        if head_raw:
            heads, trustworthy = parse_heads(head_raw)
            if (
                not trustworthy
                or not heads
                or not head_matches_audit(heads[-1], existing)
            ):
                raise OSError(
                    "verification audit/head mismatch; attended repair required"
                )
        elif existing:
            raise OSError("existing verification audit lacks a rollback anchor")

        if not heads:
            genesis_payload = encode_head(make_head(existing, None))
            genesis_size = os.fstat(head_descriptor).st_size
            genesis_written = 0
            genesis_progress = [0]
            try:
                genesis_written = _append_all(
                    head_descriptor, genesis_payload, genesis_progress,
                )
                os.fsync(head_descriptor)
            except BaseException:
                genesis_written = genesis_progress[0]
                _rollback_exact_append(
                    head_descriptor, genesis_size, genesis_payload,
                    genesis_written,
                )
                raise
            heads, trustworthy = parse_heads(genesis_payload)
            if not trustworthy or not heads:
                raise OSError("failed to construct verification audit genesis")

        payload = (
            json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        if len(existing) + len(payload) > MAX_AUDIT_BYTES:
            raise _AuditSegmentFull(
                "verification audit segment is full; rotation required")
        audit_original_size = len(existing)
        head_original_size = os.fstat(head_descriptor).st_size
        audit_written = 0
        head_payload = b""
        head_written = 0
        audit_progress = [0]
        head_progress = [0]
        try:
            audit_written = _append_all(
                audit_descriptor, payload, audit_progress,
            )
            os.fsync(audit_descriptor)
            head_payload = encode_head(make_head(existing + payload, heads[-1]))
            if head_original_size + len(head_payload) > MAX_HEAD_BYTES:
                raise OSError("verification head exceeds safety bound")
            head_written = _append_all(
                head_descriptor, head_payload, head_progress,
            )
            os.fsync(head_descriptor)
        except BaseException:
            audit_written = audit_progress[0]
            head_written = head_progress[0]
            _rollback_exact_append(
                head_descriptor, head_original_size, head_payload, head_written,
            )
            _rollback_exact_append(
                audit_descriptor, audit_original_size, payload, audit_written,
            )
            raise
        os.fsync(directory_fd)
        return True
    except _AuditSegmentFull:
        raise                          # the caller rotates; this is not a failure
    except Exception:
        return False
    finally:
        if head_descriptor is not None:
            try:
                fcntl.flock(head_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(head_descriptor)
        if audit_descriptor is not None:
            try:
                fcntl.flock(audit_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(audit_descriptor)
        if directory_fd is not None:
            os.close(directory_fd)
