"""Read-time verification quality gate.

Verification is written in two places during the gradual rollout:

* new notes carry ``confidence.verification.status`` in frontmatter;
* older notes may only have an append-only row in
  ``state/verification.jsonl``.

Frontmatter and the latest audit row are combined conservatively.  A failed
status on either side wins; other disagreements fail closed so a crash between
the two writes cannot resurrect stale machine approval.  A damaged or unreadable
audit log likewise cannot prove that an unreviewed note was not flagged, so
callers fail closed for non-human-approved notes.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
import re
import stat
import threading
import time
from typing import Mapping

from retrieve import telemetry
from remembrancerlib.acl import restricted_labels
from remembrancerlib.verification import (
    HumanTrustAudit,
    auto_approval_valid,
    governance_hold_class,
    human_trust_valid,
    producer_verification_valid,
    verification_content_fingerprint,
)
from remembrancerlib.verification_audit import (
    SEGMENTS_RELATIVE_PATH,
    HEAD_RELATIVE_PATH,
    MAX_HEAD_BYTES,
    head_matches_audit,
    parse_heads,
    read_descriptor_all,
)


VERIFICATION_STATUSES = frozenset({
    "verified", "flagged", "skipped", "unverified",
})
FAILED_VERIFICATION_STATUSES = frozenset({"flagged", "skipped"})
# The labeled-decision index a decision-review tool writes.
DECISION_VISIBILITY_RELATIVE_PATH = os.path.join("state", "decision_visibility.json")
DECISION_VISIBILITY_POLICIES = frozenset({"hold", "labeled"})
DEFAULT_DECISION_VISIBILITY = "labeled"
_DECISION_HOLD_REASON = "governance-approval-required:decision"
# The same relief valve for the ``customer-deal`` hold class, which
# had none.  It is OFF by default -- absent config is today's behavior, and
# turning it on is a one-line owner decision, not an engineering project.
CUSTOMER_DEAL_VISIBILITY_POLICIES = frozenset({"hold", "labeled"})
DEFAULT_CUSTOMER_DEAL_VISIBILITY = "hold"
CUSTOMER_DEAL_LABEL_VERIFICATION_BARS = frozenset({"clean", "not-disputed"})
DEFAULT_CUSTOMER_DEAL_LABEL_VERIFICATION = "clean"
_CUSTOMER_DEAL_CLASS = "customer-deal"
_CUSTOMER_DEAL_HOLD_REASON = "governance-approval-required:customer-deal"
_MAX_DECISION_INDEX_BYTES = 16 * 1024 * 1024
_INVALID_STATUS = "__invalid__"
_CONFLICT_STATUS = "__conflict__"
_STALE_STATUS = "__stale__"
_NOTE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
_MAX_AUDIT_BYTES = 128 * 1024 * 1024
_MAX_AUDIT_LINE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class VerificationAudit:
    """Latest valid audit status by note, plus the integrity verdict.

    ``auto_approve`` carries the load-time config gate verdict for machine
    auto-approval so every read surface honors or revokes it uniformly.
    """

    statuses: Mapping[str, str]
    content_fingerprints: Mapping[str, str] | None = None
    audit_ids: Mapping[str, str] | None = None
    trustworthy: bool = True
    auto_approve: bool = False
    automatic_admission: bool = False
    # Note name -> content fingerprint of every held decision the
    # ``governance.decision_visibility: labeled`` policy lets answer (still
    # ``proposed``).  ``None`` means the policy is ``hold`` (or the index could
    # not be read), which is byte-identical to the pre-policy gate.
    labeled_decisions: Mapping[str, str] | None = None
    # The ``customer-deal`` valve.  Unlike decisions this carries the
    # policy itself rather than a prebuilt index -- every condition it checks is
    # readable from the note (or index row) about to be served, so both read
    # paths decide identically without a builder.  The defaults are the
    # pre-policy gate: an audit built by any other caller (or by ``_bad_audit``)
    # withholds exactly as it does today.
    customer_deal_visibility: str = DEFAULT_CUSTOMER_DEAL_VISIBILITY
    customer_deal_label_verification: str = DEFAULT_CUSTOMER_DEAL_LABEL_VERIFICATION
    # The labels whose held notes no valve shows (``remembrancerlib.acl.restricted_labels``),
    # read from config with the policies above.  An audit built any other way
    # carries None, and then no valve shows any held note.
    restricted_labels: frozenset[str] | None = None

    def __post_init__(self):
        # Preserve the small constructor used by older call sites/tests while
        # exposing binding metadata for new-format rows.
        object.__setattr__(self, "content_fingerprints", self.content_fingerprints or {})
        object.__setattr__(self, "audit_ids", self.audit_ids or {})


def _bad_audit() -> VerificationAudit:
    # Never retain or surface the parser/permission error: retrieval denial is
    # intentionally indistinguishable from an absent note.
    return VerificationAudit({}, trustworthy=False)


def verification_gate_enabled(config: dict) -> bool:
    """Return the fail-closed config switch (only literal ``false`` disables)."""
    trust = config.get("trust", {}) if isinstance(config, dict) else {}
    if not isinstance(trust, dict):
        trust = {}
    return trust.get("hide_verification_failures") is not False


def _verification_audit_required(root: str) -> bool:
    """Return the tracked, monotonic cutover marker for the audit ledger.

    A pre-verification fixture/repository may legitimately have no ledger.
    Once production sets this reviewed config bit, deleting the ignored runtime
    log can never downgrade the corpus back to that legacy mode.
    """
    path = os.path.join(os.path.abspath(root), "config.json")
    try:
        with open(path, encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        return False
    except (OSError, json.JSONDecodeError, TypeError):
        return True
    if not isinstance(config, dict):
        return True
    trust = config.get("trust")
    if trust is None:
        return False
    if not isinstance(trust, dict):
        return True
    return trust.get("verification_audit_required") is True


def _auto_approve_enabled(root: str) -> bool:
    """Return the reviewed machine auto-approval gate (literal ``true`` only).

    Unlike the audit-required cutover bit, a missing or malformed config fails
    closed to disabled: the gate grants authority, so ambiguity must deny it.
    """
    path = os.path.join(os.path.abspath(root), "config.json")
    try:
        with open(path, encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if not isinstance(config, dict):
        return False
    trust = config.get("trust")
    if not isinstance(trust, dict):
        return False
    return trust.get("auto_approve") is True


def decision_visibility_policy(config: dict) -> str:
    """Return ``governance.decision_visibility``: ``labeled`` unless config says ``hold``.

    Any value outside the vocabulary fails closed to ``hold``.
    """
    governance = config.get("governance") if isinstance(config, dict) else None
    value = governance.get("decision_visibility") if isinstance(governance, dict) else None
    if value is None:
        return DEFAULT_DECISION_VISIBILITY
    return value if value in DECISION_VISIBILITY_POLICIES else "hold"


def customer_deal_visibility_policy(config: dict) -> str:
    """Return ``governance.customer_deal_visibility``: ``hold`` unless ``labeled``.

    The mirror image of ``decision_visibility`` with the opposite
    default: absent config withholds, and any value outside the vocabulary
    fails closed to ``hold``.
    """
    governance = config.get("governance") if isinstance(config, dict) else None
    value = governance.get("customer_deal_visibility") \
        if isinstance(governance, dict) else None
    if value is None:
        return DEFAULT_CUSTOMER_DEAL_VISIBILITY
    return value if value in CUSTOMER_DEAL_VISIBILITY_POLICIES else "hold"


def customer_deal_label_verification(config: dict) -> str:
    """Return ``governance.customer_deal_label_verification``; unknown → ``clean``.

    ``clean`` requires a real verifier verdict bound to the bytes being served;
    ``not-disputed`` also admits notes the verifier has never judged, which is
    what attended-compatibility mode already grants every other note type.
    """
    governance = config.get("governance") if isinstance(config, dict) else None
    value = governance.get("customer_deal_label_verification") \
        if isinstance(governance, dict) else None
    if value is None:
        return DEFAULT_CUSTOMER_DEAL_LABEL_VERIFICATION
    return value if value in CUSTOMER_DEAL_LABEL_VERIFICATION_BARS \
        else DEFAULT_CUSTOMER_DEAL_LABEL_VERIFICATION


def _load_customer_deal_policy(root: str) -> tuple[str, str]:
    """``(policy, verification bar)`` from config; unreadable config withholds."""
    try:
        with open(os.path.join(os.path.abspath(root), "config.json"),
                  encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        config = {}
    except (OSError, json.JSONDecodeError, TypeError):
        return DEFAULT_CUSTOMER_DEAL_VISIBILITY, DEFAULT_CUSTOMER_DEAL_LABEL_VERIFICATION
    if not isinstance(config, dict):
        return DEFAULT_CUSTOMER_DEAL_VISIBILITY, DEFAULT_CUSTOMER_DEAL_LABEL_VERIFICATION
    return customer_deal_visibility_policy(config), customer_deal_label_verification(config)


def labeled_customer_deal_visible(note, audit: VerificationAudit) -> bool:
    """Whether a governance-held ``customer-deal`` note may pass with its label.

    Least privilege and symmetric with ``labeled_decision_visible``:
    the ``labeled`` policy, a ``customer-deal`` hold (never another class), a
    ``proposed`` note with no restricted label, and a verification bar
    evaluated against the bytes about to be served.

    Every condition is read from the note itself, so the index-first and
    full-scan read paths reach the same verdict.  The decision valve's extra condition --
    at least one human-authored source among the cited events -- is *not*
    enforced here: it needs the event corpus, so it belongs to an index builder
    the gate does not own.  That is one reason this valve defaults to ``hold``.
    """
    if getattr(audit, "customer_deal_visibility", DEFAULT_CUSTOMER_DEAL_VISIBILITY) \
            != "labeled":
        return False
    if getattr(note, "state", None) != "proposed":
        return False
    acl = getattr(note, "acl", None) or []
    if isinstance(acl, str):
        acl = [acl]
    labels = getattr(audit, "restricted_labels", None)
    if labels is None or set(str(value) for value in acl) & labels:
        return False
    hold_class = governance_hold_class(note)
    if hold_class is None:
        # An index row carries the reason only; a full note is checked through
        # ``governance_hold_class`` above, which fails closed on hold drift.
        if getattr(note, "verification_reason", None) != _CUSTOMER_DEAL_HOLD_REASON:
            return False
    elif hold_class != _CUSTOMER_DEAL_CLASS:
        return False
    return _customer_deal_verification_bar_met(note, audit)


def _customer_deal_verification_bar_met(note, audit: VerificationAudit) -> bool:
    """Apply the configured bar to the ledger verdict bound to the current bytes.

    A verdict recorded for other bytes is inapplicable, never a veto and never a
    grant.  ``flagged`` on the current bytes is a dispute and withholds under
    both bars.
    """
    current = _current_content_fingerprint(note)
    if current is None:
        return False
    name = str(getattr(note, "name", "") or "")
    bound = audit.content_fingerprints.get(name) == current
    status = audit.statuses.get(name) if bound else None
    audit_id = audit.audit_ids.get(name) if bound else None
    if status == "flagged":
        return False
    bar = getattr(audit, "customer_deal_label_verification",
                  DEFAULT_CUSTOMER_DEAL_LABEL_VERIFICATION)
    if bar == "not-disputed":
        return True
    return bool(audit_id) and status in ("verified", "skipped")


def _load_labeled_decisions(root: str) -> Mapping[str, str] | None:
    """The labeled-decision index, or ``None`` when the policy is ``hold``.

    A missing, oversized or malformed index under the ``labeled`` policy yields
    an empty mapping: no decision is labeled until the next rebuild, never the
    reverse.  Only rows with a well-formed name, a sha256 fingerprint and the
    ``decision`` class are honored.
    """
    root = os.path.abspath(root)
    try:
        with open(os.path.join(root, "config.json"), encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        config = {}
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if decision_visibility_policy(config) != "labeled":
        return None
    path = os.path.join(root, DECISION_VISIBILITY_RELATIVE_PATH)
    try:
        if os.path.getsize(path) > _MAX_DECISION_INDEX_BYTES:
            return {}
        with open(path, encoding="utf-8") as index_file:
            payload = json.load(index_file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    notes = payload.get("notes") if isinstance(payload, dict) else None
    if not isinstance(notes, dict):
        return {}
    labeled: dict[str, str] = {}
    for name, row in notes.items():
        if (not isinstance(name, str) or _NOTE_NAME_RE.fullmatch(name) is None
                or not isinstance(row, dict) or row.get("class") != "decision"):
            continue
        fingerprint = row.get("content_fingerprint")
        if isinstance(fingerprint, str) and re.fullmatch(r"[a-f0-9]{64}", fingerprint):
            labeled[name] = fingerprint
    return labeled


def labeled_decision_visible(note, audit: VerificationAudit) -> bool:
    """Whether a governance-held decision may pass the hold with its label.

    Requires the ``labeled`` policy, a ``decision`` hold (never another
    class), a ``proposed`` note with no restricted label, and an index entry
    bound to the exact bytes about to be served.
    """
    labeled = getattr(audit, "labeled_decisions", None)
    if not labeled:
        return False
    name = str(getattr(note, "name", "") or "")
    if getattr(note, "state", None) != "proposed":
        return False
    acl = getattr(note, "acl", None) or []
    if isinstance(acl, str):
        acl = [acl]
    labels = getattr(audit, "restricted_labels", None)
    if labels is None or set(str(value) for value in acl) & labels:
        return False
    hold_class = governance_hold_class(note)
    if hold_class is None:
        # Index rows carry the reason only; the full note is rechecked below
        # through ``governance_hold_class`` before retrieval returns it.
        if getattr(note, "verification_reason", None) != _DECISION_HOLD_REASON:
            return False
    elif hold_class != "decision":
        return False
    current = _current_content_fingerprint(note)
    return current is not None and labeled.get(name) == current


# ─── Sealed-segment and current-ledger caches ───
#
# Every read surface calls ``load_verification_audit``, and every call used to
# re-read, re-hash and re-parse every sealed segment, so read latency grew with
# the ledger's history.  Sealed segments never change after rotation, so their
# verified fold is cached here.
#
# The invariant the cache rests on is the manifest: it is re-read and re-parsed
# on every load, and each row's ``final_head_id`` (a sha256 chain over the
# segment's size and sha256) is part of the cache key, so a hit can only ever
# return the fold of exactly the bytes the manifest commits to.  On top of that,
# every load still opens each segment and head without following symlinks,
# share-locks it and fstat-checks it for type and mode 0600 (one pair at a time),
# and reuses the fold only while every file's (dev, ino, size, mtime_ns,
# ctime_ns) is unchanged; any difference is a miss that re-runs the full
# read -> sha256 -> head check -> parse for the changed suffix.  ctime cannot be
# set by an unprivileged writer and moves on every write, chmod and rename.  A
# fold is cached only once all of its files had been still for
# ``_SEALED_CACHE_MIN_AGE_NS`` before they were read (the racy-git rule): a write
# inside the same coarse timestamp tick as the file's last change could
# otherwise leave ctime unchanged.  Misses are single-flight per state directory,
# so concurrent readers share one fold instead of each folding alone, and a
# rotation (the manifest only ever grows) folds just the new segment onto the
# cached prefix.  The current ledger is re-read and re-hashed on every load; only
# its JSON parse is reused, keyed by the sha256 of the exact bytes, which cannot
# change a result.
_SEALED_CACHE_MIN_AGE_NS = 2_000_000_000
_CACHE_LOCK = threading.Lock()
_SEALED_CACHE: dict[tuple[int, int], tuple[tuple, tuple[dict, dict, dict]]] = {}
_MISS_LOCKS: dict[tuple[int, int], threading.Lock] = {}
_CURRENT_PARSE_CACHE: dict[str, "VerificationAudit"] = {}


def reset_audit_cache() -> None:
    """Drop every cache (tests, and anything that rewrites a ledger in place)."""
    with _CACHE_LOCK:
        _SEALED_CACHE.clear()
        _MISS_LOCKS.clear()
        _CURRENT_PARSE_CACHE.clear()


def _stat_key(metadata: os.stat_result) -> tuple:
    # No st_mode: the type/0600 check runs on every load, hit or miss, and a
    # chmod moves ctime anyway.
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def _key_settled(segment_key: tuple, now_ns: int) -> bool:
    """Both files of one segment key were last changed >= the min age ago."""
    _name, _head_id, audit_key, head_key = segment_key
    limit = now_ns - _SEALED_CACHE_MIN_AGE_NS
    return max(audit_key[3], audit_key[4]) <= limit \
        and max(head_key[3], head_key[4]) <= limit


def _close_pair(descriptors: list[int]) -> None:
    for descriptor in reversed(descriptors):
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def _open_pair(state_descriptor: int, row: dict) -> tuple[list[int], tuple] | None:
    """Open, share-lock and type/mode-check one segment and its head.

    Returns the descriptors (caller closes them with ``_close_pair``) and the
    segment's cache key, or ``None`` on any failure, with nothing left open.
    """
    from remembrancerlib.verification_audit import head_path_for_audit

    audit_name = row["name"]
    head_name = os.path.basename(head_path_for_audit("/x/" + audit_name))
    descriptors: list[int] = []
    try:
        for name in (audit_name, head_name):
            descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                                 dir_fd=state_descriptor)
            descriptors.append(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_SH)
        metadata = [os.fstat(descriptor) for descriptor in descriptors]
    except OSError:
        _close_pair(descriptors)
        return None
    for item in metadata:
        if not stat.S_ISREG(item.st_mode) or stat.S_IMODE(item.st_mode) != 0o600:
            _close_pair(descriptors)
            return None
    key = (audit_name, row["final_head_id"],
           _stat_key(metadata[0]), _stat_key(metadata[1]))
    return descriptors, key


def _verify_segment(state_descriptor: int, row: dict):
    """Read, head-check and parse one sealed segment: ``(key, piece)`` or None."""
    opened = _open_pair(state_descriptor, row)
    if opened is None:
        return None
    descriptors, key = opened
    try:
        raw = read_descriptor_all(descriptors[0], _MAX_AUDIT_BYTES)
        heads, trustworthy = parse_heads(
            read_descriptor_all(descriptors[1], MAX_HEAD_BYTES))
    except OSError:
        return None
    finally:
        _close_pair(descriptors)
    if (not trustworthy or not heads
            or not head_matches_audit(heads[-1], raw)
            or heads[-1].head_id != row["final_head_id"]):
        return None
    piece = _parse_verification_audit_bytes(raw)
    if not piece.trustworthy:
        return None
    return key, piece


def _miss_lock(state_key: tuple[int, int]) -> threading.Lock:
    with _CACHE_LOCK:
        return _MISS_LOCKS.setdefault(state_key, threading.Lock())


def _cached_copy(state_key: tuple[int, int], keys: tuple):
    with _CACHE_LOCK:
        cached = _SEALED_CACHE.get(state_key)
    if cached is not None and cached[0] == keys:
        statuses, fingerprints, audit_ids = cached[1]
        return dict(statuses), dict(fingerprints), dict(audit_ids)
    return None


def _load_sealed_fold(
    state_descriptor: int,
) -> tuple[tuple[dict, dict, dict] | None, bool]:
    """The fold of every sealed audit segment, oldest first, each authenticated.

    Rotation seals a full segment and starts an empty one, so the history a
    verdict rests on is spread over several files. Each segment is verified
    against its own head file exactly as the current one is, and the manifest
    pins their order and final head ids -- a deleted, reordered or swapped
    segment fails closed rather than silently shortening the record.

    Returns ``((statuses, fingerprints, audit_ids), True)``, folded with the same
    later-row-wins ``update`` the reader applies, or ``(None, False)`` on any
    failure.  The returned maps are private copies.
    """
    from remembrancerlib.verification_audit import MAX_SEGMENTS_BYTES, parse_segments

    try:
        manifest_descriptor = os.open(
            os.path.basename(SEGMENTS_RELATIVE_PATH),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=state_descriptor,
        )
    except FileNotFoundError:
        return ({}, {}, {}), True             # never rotated
    except OSError:
        return None, False
    try:
        rows, ok = parse_segments(
            read_descriptor_all(manifest_descriptor, MAX_SEGMENTS_BYTES))
    except OSError:
        return None, False
    finally:
        os.close(manifest_descriptor)
    if not ok:
        return None, False
    if not rows:
        return ({}, {}, {}), True

    # Pass 1, every load: the current key of every segment, one pair at a time.
    keys: list[tuple] = []
    for row in rows:
        opened = _open_pair(state_descriptor, row)
        if opened is None:
            return None, False
        descriptors, key = opened
        _close_pair(descriptors)
        keys.append(key)
    try:
        state_metadata = os.fstat(state_descriptor)
    except OSError:
        return None, False
    state_key = (state_metadata.st_dev, state_metadata.st_ino)
    hit = _cached_copy(state_key, tuple(keys))
    if hit is not None:
        return hit, True

    # Pass 2, on a miss: single-flight, and only the suffix the cache lacks.
    with _miss_lock(state_key):
        hit = _cached_copy(state_key, tuple(keys))
        if hit is not None:
            return hit, True
        with _CACHE_LOCK:
            cached = _SEALED_CACHE.get(state_key)
        prefix: tuple = ()
        statuses: dict[str, str] = {}
        fingerprints: dict[str, str] = {}
        audit_ids: dict[str, str] = {}
        if (cached is not None and len(cached[0]) < len(keys)
                and tuple(keys[:len(cached[0])]) == cached[0]):
            prefix = cached[0]
            statuses, fingerprints, audit_ids = (
                dict(cached[1][0]), dict(cached[1][1]), dict(cached[1][2]))
        now_ns = time.time_ns()               # before the reads it vouches for
        verified_keys = list(prefix)
        for row in rows[len(prefix):]:
            got = _verify_segment(state_descriptor, row)
            if got is None:
                return None, False
            key, piece = got
            statuses.update(piece.statuses)
            fingerprints.update(piece.content_fingerprints or {})
            audit_ids.update(piece.audit_ids or {})
            verified_keys.append(key)
        if all(_key_settled(key, now_ns) for key in verified_keys[len(prefix):]):
            with _CACHE_LOCK:
                _SEALED_CACHE[state_key] = (
                    tuple(verified_keys),
                    (dict(statuses), dict(fingerprints), dict(audit_ids)))
        return (statuses, fingerprints, audit_ids), True


def warm_audit_caches(root: str) -> threading.Thread | None:
    """Fill the audit and contested caches off the request path.

    A long-lived server calls this once at start, so its first request does not
    pay the cold fold, which grows with every sealed segment; a request that
    arrives meanwhile waits on the single-flight lock instead of starting its own
    read.  Best effort: any failure is swallowed and the first real request does
    the work.  ``REMEMBRANCER_WARM_AUDIT=0`` disables it (the test suites set that).
    """
    if os.environ.get("REMEMBRANCER_WARM_AUDIT", "1").strip() == "0":
        return None

    def _warm() -> None:
        try:
            load_verification_audit(root)
            from retrieve.contested import contested_index
            contested_index(root)
        except Exception:
            pass

    thread = threading.Thread(target=_warm, name="remembrancer-audit-warm", daemon=True)
    thread.start()
    return thread


def _parse_current_cached(raw: bytes) -> "VerificationAudit":
    """``_parse_verification_audit_bytes`` memoised on the sha256 of ``raw``.

    The bytes were already read under the shared lock and checked against the
    head; identical bytes parse identically, so this changes no verdict.
    """
    digest = hashlib.sha256(raw).hexdigest()
    with _CACHE_LOCK:
        cached = _CURRENT_PARSE_CACHE.get(digest)
    if cached is not None:
        return cached
    piece = _parse_verification_audit_bytes(raw)
    if piece.trustworthy:
        with _CACHE_LOCK:
            _CURRENT_PARSE_CACHE.clear()
            _CURRENT_PARSE_CACHE[digest] = piece
    return piece


def load_verification_audit(root: str) -> VerificationAudit:
    """Load the latest audit row per note, failing closed on any corruption.

    Counted and timed into the bound read-call record (``retrieve.telemetry``:
    ``audit_loads``, ``audit_ms``), so an MCP audit row shows how many loads one
    call paid and what they cost. No record bound, no effect.
    """
    telemetry.add("audit_loads")
    with telemetry.timed("audit_ms"):
        return _load_verification_audit(root)


def _load_verification_audit(root: str) -> VerificationAudit:
    """Load the latest audit row per note, failing closed on any corruption.

    A missing file is normal only before the tracked verification-audit cutover.
    Once ``trust.verification_audit_required`` is enabled, absence is corruption
    rather than a downgrade to legacy mode.  Every physical line
    must be a complete, bounded JSON object with a valid note name and status.
    A shared lock prevents observing a partial append from the compliant writer.
    """
    from remembrancerlib.locks import open_repository_state_directory

    try:
        state_descriptor, _created = open_repository_state_directory(
            root, create=False,
        )
    except OSError:
        return _bad_audit()
    if state_descriptor < 0:
        from retrieve.maintenance import automatic, read_config
        return _bad_audit() if _verification_audit_required(root) else VerificationAudit(
            {}, automatic_admission=automatic(read_config(root)))
    required = _verification_audit_required(root)
    audit_descriptor: int | None = None
    head_descriptor: int | None = None
    try:
        audit_descriptor = os.open(
            "verification.jsonl",
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=state_descriptor,
        )
    except FileNotFoundError:
        audit_descriptor = None
    except OSError:
        os.close(state_descriptor)
        return _bad_audit()
    try:
        head_descriptor = os.open(
            os.path.basename(HEAD_RELATIVE_PATH),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=state_descriptor,
        )
    except FileNotFoundError:
        head_descriptor = None
    except OSError:
        if audit_descriptor is not None:
            os.close(audit_descriptor)
        os.close(state_descriptor)
        return _bad_audit()

    if audit_descriptor is None:
        if head_descriptor is not None:
            os.close(head_descriptor)
        os.close(state_descriptor)
        if required or head_descriptor is not None:
            return _bad_audit()
        customer_deal_policy, customer_deal_bar = _load_customer_deal_policy(root)
        from retrieve.maintenance import automatic, read_config
        return VerificationAudit(
            {}, automatic_admission=automatic(read_config(root)),
            labeled_decisions=_load_labeled_decisions(root),
            customer_deal_visibility=customer_deal_policy,
            customer_deal_label_verification=customer_deal_bar,
            restricted_labels=restricted_labels(read_config(root)))
    if required and head_descriptor is None:
        os.close(audit_descriptor)
        os.close(state_descriptor)
        return _bad_audit()

    try:
        try:
            fcntl.flock(audit_descriptor, fcntl.LOCK_SH)
            if head_descriptor is not None:
                fcntl.flock(head_descriptor, fcntl.LOCK_SH)
            audit_metadata = os.fstat(audit_descriptor)
            if (
                not stat.S_ISREG(audit_metadata.st_mode)
                or stat.S_IMODE(audit_metadata.st_mode) != 0o600
            ):
                return _bad_audit()
            raw = read_descriptor_all(audit_descriptor, _MAX_AUDIT_BYTES)
            if head_descriptor is not None:
                head_metadata = os.fstat(head_descriptor)
                if (
                    not stat.S_ISREG(head_metadata.st_mode)
                    or stat.S_IMODE(head_metadata.st_mode) != 0o600
                ):
                    return _bad_audit()
                head_raw = read_descriptor_all(head_descriptor, MAX_HEAD_BYTES)
                heads, trustworthy = parse_heads(head_raw)
                if (
                    not trustworthy
                    or not heads
                    or not head_matches_audit(heads[-1], raw)
                ):
                    return _bad_audit()
        except OSError:
            return _bad_audit()
        finally:
            if head_descriptor is not None:
                try:
                    fcntl.flock(head_descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            try:
                fcntl.flock(audit_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        sealed_fold, segments_ok = _load_sealed_fold(state_descriptor)
    finally:
        if head_descriptor is not None:
            os.close(head_descriptor)
        os.close(audit_descriptor)
        os.close(state_descriptor)

    if not segments_ok or sealed_fold is None:
        return _bad_audit()
    # Oldest segment first, current last: a later row for a note supersedes an
    # earlier one, exactly as it does within a single file. Merging the parsed
    # maps rather than concatenating the bytes keeps the cost proportional to
    # the note count, not to the length of history.
    statuses, fingerprints, audit_ids = sealed_fold
    piece = _parse_current_cached(raw)
    if not piece.trustworthy:
        return piece
    statuses.update(piece.statuses)
    fingerprints.update(piece.content_fingerprints or {})
    audit_ids.update(piece.audit_ids or {})
    parsed = VerificationAudit(statuses, fingerprints, audit_ids)
    if not parsed.trustworthy:
        return parsed
    customer_deal_policy, customer_deal_bar = _load_customer_deal_policy(root)
    from retrieve.maintenance import automatic, read_config
    return VerificationAudit(
        parsed.statuses,
        parsed.content_fingerprints,
        parsed.audit_ids,
        auto_approve=_auto_approve_enabled(root),
        automatic_admission=automatic(read_config(root)),
        labeled_decisions=_load_labeled_decisions(root),
        customer_deal_visibility=customer_deal_policy,
        customer_deal_label_verification=customer_deal_bar,
        restricted_labels=restricted_labels(read_config(root)),
    )


def _parse_verification_audit_bytes(raw: bytes) -> VerificationAudit:
    """Parse one complete audit prefix without consulting filesystem state."""
    if len(raw) > _MAX_AUDIT_BYTES or (raw and not raw.endswith(b"\n")):
        return _bad_audit()
    statuses: dict[str, str] = {}
    content_fingerprints: dict[str, str] = {}
    audit_ids: dict[str, str] = {}
    for raw_line in raw.splitlines():
        if not raw_line or len(raw_line) > _MAX_AUDIT_LINE_BYTES:
            return _bad_audit()
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _bad_audit()
        if not isinstance(row, dict):
            return _bad_audit()
        name = row.get("note")
        status = row.get("status")
        if (not isinstance(name, str) or _NOTE_NAME_RE.fullmatch(name) is None
                or not isinstance(status, str)
                or status not in VERIFICATION_STATUSES):
            return _bad_audit()
        fingerprint = row.get("content_fingerprint")
        audit_id = row.get("audit_id")
        # Legacy rows have neither binding field.  New rows must have both;
        # accepting a half-written/hand-edited binding would defeat the
        # audit/frontmatter two-phase contract.
        if (fingerprint is None) != (audit_id is None):
            return _bad_audit()
        if fingerprint is not None:
            if (not isinstance(fingerprint, str)
                    or re.fullmatch(r"[a-f0-9]{64}", fingerprint) is None
                    or not isinstance(audit_id, str)
                    or not audit_id or len(audit_id) > 128):
                return _bad_audit()
            content_fingerprints[name] = fingerprint
            audit_ids[name] = audit_id
        else:
            # The latest row is authoritative.  Do not retain binding metadata
            # from an older row when a legacy row follows it.
            content_fingerprints.pop(name, None)
            audit_ids.pop(name, None)
        statuses[name] = status
    return VerificationAudit(statuses, content_fingerprints, audit_ids)


def _current_content_fingerprint(note) -> str | None:
    """Read the canonical current-content hash from a note or validated index."""
    if hasattr(note, "verification_content_fingerprint"):
        value = getattr(note, "verification_content_fingerprint", None)
        return value if isinstance(value, str) and value else None
    try:
        return verification_content_fingerprint(note)
    except Exception:
        return None


def _embedded_binding(note) -> tuple[str | None, str | None, str | None] | None:
    """Return ``(content fingerprint, audit id, method)`` or invalid ``None``."""
    if hasattr(note, "verification"):
        fingerprint = getattr(
            note, "embedded_verification_content_fingerprint", None
        )
        audit_id = getattr(note, "verification_audit_id", None)
        method = getattr(note, "verification_method", None)
    else:
        confidence = getattr(note, "confidence", None)
        if not isinstance(confidence, dict):
            return (None, None, None) if confidence is None else None
        verification = confidence.get("verification")
        if not isinstance(verification, dict):
            return (None, None, None) if verification is None else None
        fingerprint = verification.get("content_fingerprint")
        audit_id = verification.get("audit_id")
        method = verification.get("method")
    if fingerprint in (None, "") and audit_id in (None, "") and method in (None, ""):
        return None, None, None
    if (not isinstance(fingerprint, str)
            or re.fullmatch(r"[a-f0-9]{64}", fingerprint) is None
            or (audit_id is not None and (not isinstance(audit_id, str)
                                          or not audit_id or len(audit_id) > 128))
            or (method is not None and not isinstance(method, str))):
        return None
    return fingerprint, audit_id, method


def embedded_verification_status(note) -> str | None:
    """Return a note/index row's embedded status, or ``None`` when absent.

    ``IndexRecord`` exposes a normalized ``verification`` scalar.  A real
    ``Note`` instead carries the nested confidence object.  Structurally
    present but invalid data returns an internal invalid marker so it cannot
    bypass the gate by masquerading as legacy absence.
    """
    if hasattr(note, "verification"):
        value = getattr(note, "verification", None)
        if value in (None, ""):
            return None
        return value if value in VERIFICATION_STATUSES else _INVALID_STATUS

    confidence = getattr(note, "confidence", None)
    if confidence is None:
        return None
    if not isinstance(confidence, dict):
        return _INVALID_STATUS
    if "verification" not in confidence:
        return None
    verification = confidence.get("verification")
    if not isinstance(verification, dict):
        return _INVALID_STATUS
    if "status" not in verification:
        return _INVALID_STATUS
    status = verification.get("status")
    return status if status in VERIFICATION_STATUSES else _INVALID_STATUS


def verification_status(note, audit: VerificationAudit) -> str | None:
    """Resolve embedded and latest-audit status conservatively.

    The verifier appends its audit row before replacing note frontmatter.  A
    crash between those writes can therefore leave a stale embedded ``verified``
    next to a newer ``flagged``/``skipped`` audit row.  No machine verdict may
    upgrade another machine verdict: a failure on either side wins, and any
    other disagreement is an internal conflict that fails the answer gate.
    """
    embedded = embedded_verification_status(note)
    if embedded == _INVALID_STATUS:
        return embedded
    name = str(getattr(note, "name", "") or "")
    audited = audit.statuses.get(name)
    current_fingerprint = _current_content_fingerprint(note)
    binding = _embedded_binding(note)
    if binding is None:
        return _INVALID_STATUS
    embedded_fingerprint, embedded_audit_id, method = binding
    producer_grounded = bool(
        isinstance(method, str)
        and method.startswith("producer-grounded-v1:")
        and producer_verification_valid(note, audit)
    )
    if embedded == "verified" and not producer_grounded:
        # Name-only legacy frontmatter is not proof of the current bytes. Keep
        # audit-only legacy compatibility, but never let an imported/hand-edited
        # proposed page self-assert machine approval.
        if embedded_fingerprint is None or embedded_audit_id is None:
            return _INVALID_STATUS
    if embedded_fingerprint is not None:
        if current_fingerprint is None or embedded_fingerprint != current_fingerprint:
            return _STALE_STATUS

    audited_fingerprint = audit.content_fingerprints.get(name)
    audited_id = audit.audit_ids.get(name)
    if audited_fingerprint is not None:
        if current_fingerprint is None or audited_fingerprint != current_fingerprint:
            if producer_grounded and embedded_fingerprint == current_fingerprint:
                # A deterministic producer has replaced the old content and
                # supplied a complete current proof. Bound audit verdicts for
                # prior bytes are inapplicable, not a veto on the new bytes.
                audited = None
                audited_fingerprint = None
                audited_id = None
            else:
                return _STALE_STATUS
    if embedded_audit_id is not None and not producer_grounded:
        # CoVe approval is valid only when both durable copies identify the
        # same verdict. A missing/deleted/newer row is not equivalent proof.
        if audited_id != embedded_audit_id or audited_fingerprint != embedded_fingerprint:
            if audited in FAILED_VERIFICATION_STATUSES:
                return audited
            if embedded in FAILED_VERIFICATION_STATUSES:
                return embedded
            return _CONFLICT_STATUS
    elif audited_id is not None and embedded_audit_id is None:
        # Audit append succeeded but the note write did not. Fail closed for
        # the new two-phase format; legacy audit-only rows remain supported.
        if audited in FAILED_VERIFICATION_STATUSES:
            return audited
        return _CONFLICT_STATUS
    if embedded is None:
        return audited
    if audited is None or embedded == audited:
        return embedded
    if audited in FAILED_VERIFICATION_STATUSES:
        return audited
    if embedded in FAILED_VERIFICATION_STATUSES:
        return embedded
    return _CONFLICT_STATUS


def public_verification_status(note, audit: VerificationAudit) -> str:
    """Return a stable, non-internal status suitable for result metadata."""
    status = verification_status(note, audit)
    return status if status in VERIFICATION_STATUSES else "unverified"


def verification_allows_answer(
    note,
    audit: VerificationAudit,
    *,
    enabled: bool = True,
    human_audit: HumanTrustAudit | None = None,
) -> bool:
    """Whether a note may participate in a default answer/read projection.

    Explicit human approval (``state: verified``) overrides model verification.
    Authorship markers do not: a human-authored draft remains proposed until an
    approver changes its trust state.  When the gate is disabled explicitly,
    this helper preserves the historical read behavior.
    """
    if human_trust_valid(note, human_audit):
        # Only a content-bound ``verified`` transition overrides a model
        # failure. A valid ``superseded`` transition is lifecycle history and
        # is filtered separately; it never upgrades answer eligibility.
        if getattr(note, "state", None) == "verified":
            return True
    elif auto_approval_valid(note, audit):
        # A config-gated machine approval is verified-equivalent while the
        # gate is on and both durable copies bind the current bytes.  Like
        # the signed transition above, it yields the governance hold below;
        # with the gate off it is never valid and behavior is unchanged.
        return True
    elif getattr(note, "state", None) in {"verified", "superseded"}:
        # A raw/legacy/forged state is not ordinary unverified compatibility.
        # It explicitly claims authority without proof and therefore fails
        # closed until an approver rebinds the current bytes.
        return False
    # Automatic acquisition is separate from verification. Legacy holds stay
    # auditable/rollbackable; this neither grants trust nor changes ACLs or
    # signed lifecycle decisions. Source content remains labeled as proposed.
    if audit.automatic_admission and audit.trustworthy:
        return True
    # Governance approval is a hard publication hold, not an ordinary model
    # failure toggle.  It remains enforced even when an attended compatibility
    # deployment disables generic verification filtering; only a content-bound
    # human approval above may override it.
    #
    # Exception: under ``governance.decision_visibility: labeled`` a
    # ``decision``-class hold whose current bytes are in the labeled index
    # (human-authored source and nothing disputed by the verifier, or -- with
    # ``governance.decision_min_machine_sources`` -- enough real
    # provenance event ids and no contradiction bound to those bytes) is
    # NOT dropped here; it continues into the
    # ordinary gate below as the ``proposed``/``skipped`` note it is, so it
    # answers only where any other unapproved note would (attended
    # compatibility mode), rank-demoted by the proposed penalty.
    #
    # The same exception applies to a ``customer-deal`` hold under
    # ``governance.customer_deal_visibility: labeled`` (default ``hold`` --
    # absent config is byte-identical to before the valve existed).
    governance_reason = getattr(note, "verification_reason", None)
    if (
        isinstance(governance_reason, str)
        and governance_reason.startswith("governance-approval-required:")
    ) or governance_hold_class(note) is not None:
        if not (labeled_decision_visible(note, audit)
                or labeled_customer_deal_visible(note, audit)):
            return False
    if not enabled:
        return True
    if not audit.trustworthy:
        return False
    # A matching generic CoVe audit/frontmatter pair proves only that a model
    # judged prose claims. It cannot attest the note's semantic/security
    # projections (slug, type, domain, ACL), so it is never an answer grant.
    # Deterministic producer proofs are structurally revalidated here; human
    # approval already returned above through ``state: verified``.
    if (embedded_verification_status(note) == "verified"
            and not producer_verification_valid(note, audit)):
        return False
    status = verification_status(note, audit)
    return status not in (
        FAILED_VERIFICATION_STATUSES | {_INVALID_STATUS, _CONFLICT_STATUS, _STALE_STATUS}
    )
