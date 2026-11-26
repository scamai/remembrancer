#!/usr/bin/env python3
"""Remembrancer retrieval — index-first, ACL-gated, BM25-ranked (stdlib only, no embeddings).

The permission gate AND the recall->precision dial:

  1. Consult fresh per-domain ``index.md`` files for candidates; use a full
     corpus scan only when those indexes are absent, legacy, malformed, or stale.
  2. Filter to what the acting identity is allowed to read (config.json ACL grants).
  3. Optionally scope to one entity (--entity customer:acme).
  4. BM25-rank on ``name + description + tags + index_keys``, never on the title.
     A second stage re-ranks the top ``retrieval.body_rerank_pool`` of those on
     note *bodies* and blends the two (``body_rerank_weight``); bodies are read
     for that pool only and are still never returned by this call.  With
     ``retrieval.ranking.query_stopwords`` the QUERY is pruned of English
     function words first (documents are untouched).
  5. Return a RECALL-WIDE candidate pool (retrieval.candidate_pool, default 25),
     but PRINT index-first: name · acl · entities · provenance · one-line description
     — never full bodies. Mark the top `deliver_max_notes` as the precision set to open.

The agent (or human) is the reranker: it reads the index lines, then calls
get_note(name) only on the few it actually needs. No LLM call on this path.

Usage:
    python3 retrieve/query.py --as jdoe --q "release checklist" [--k N] [--entity customer:acme]

Importable surface (reused by the MCP module):
    search(query, identity, k=None, entity=None, root=".", current_only=False) -> list[dict]
    search_full_scan(...) -> list[dict]  # authoritative recovery path, no index
    get_note(name, identity, root=".") -> dict | None
    file_answer(question, answer, source_names, root=".") -> str
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextvars import ContextVar
from datetime import date, datetime, timezone
from functools import lru_cache, wraps
import hashlib
import json
import os
import re
import sys
import time

# Make the repo root importable regardless of how this file was launched.
# `python3 retrieve/query.py` (its own documented usage, and how a coverage
# harness invokes it as a subprocess) puts
# `retrieve/` on sys.path[0], not the repo root, so `from remembrancerlib import
# ...` below only resolves when the CALLER's environment happens to carry
# `PYTHONPATH=.` -- pytest gets that for free from pyproject.toml, a bare
# shell or systemd unit does not. Without it every subprocess call fails with
# empty stdout, a harness that reads that as "no answer" counts each failure
# as a genuine miss, and a coverage score of zero reads as real.
# Same guarded, append-not-insert pattern as `mcp/server.py`.
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remembrancerlib import (
    Note,
    combine_acl,
    load_notes,
    load_suppressed,
    parse_frontmatter,
    slugify,
    write_note,
    effective_state,
    active_supersession_targets,
    current_note_names,
    acl_can_reveal,
    FULL_ACCESS,
    DENY,
    HumanTrustAudit,
    load_human_trust_audit,
    bind_loaded_source_retirements,
    bind_loaded_producer_lifecycles,
    bind_loaded_auto_approvals,
)
from remembrancerlib.locks import serialized_pipeline_mutation
from remembrancerlib.verification import _record_human_trust_transition

from integrations.artifact_graph import load_graph as _load_artifact_graph
from integrations.artifact_graph import node as _artifact_node
from retrieve.bm25 import BM25, strip_stopwords
from retrieve.event_search import search_events
from retrieve.freshness import FreshnessContext, load_evidence_dates
from retrieve.index import IndexRecord, IndexSnapshot, load_index_snapshot
from retrieve import telemetry
from retrieve.timeparse import parse_time_range
from retrieve.verification import (
    VerificationAudit,
    embedded_verification_status,
    load_verification_audit,
    public_verification_status,
    verification_allows_answer,
    verification_gate_enabled,
)

# Repo root: two levels up from this file (retrieve/query.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

# Defaults if config is missing the retrieval block.
_DEFAULT_CANDIDATE_POOL = 25
_DEFAULT_DELIVER_MAX = 5
_DEFAULT_MAX_SLICE_TOKENS = 4000
_SUPERSEDED_RANK_PENALTY = 0.25

# Optional ranking terms (gated, default OFF). Both knobs are byte-neutral when
# disabled.
_GRAPH_LATEST_BOOST = 1.15  # cap: a latest-in-topic page never gains more than this
_RECENCY_TYPES = frozenset({"state", "project"})  # the volatile note types only
_RECENCY_HALF_LIFE_SLO_MULTIPLE = 3  # half-life = 3 x the type's freshness SLO
_ARTIFACT_NOTE_PREFIX = "artifact-"
_DEFAULT_RANKING = {
    "graph_aware": False,
    "query_stopwords": False,
    "recency": {"enabled": False, "weight": 0.15, "floor": 0.3, "tie_window": 0.15},
}


def _load_config(root: str) -> dict:
    path = os.path.join(root, "config.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _retrieval_cfg(cfg: dict) -> dict:
    r = cfg.get("retrieval", {}) or {}
    return {
        "candidate_pool": int(r.get("candidate_pool", _DEFAULT_CANDIDATE_POOL)),
        "deliver_max_notes": int(r.get("deliver_max_notes", _DEFAULT_DELIVER_MAX)),
        "max_slice_tokens": int(r.get("max_slice_tokens", _DEFAULT_MAX_SLICE_TOKENS)),
        # Hide superseded notes (folded shards + explicitly-superseded) from current retrieval.
        # Default True — a note marked superseded should not surface as current truth. Set
        # retrieval.hide_superseded:false to restore the earlier behavior exactly.
        "hide_superseded": bool(r.get("hide_superseded", True)),
        # Two-stage body re-rank: after the frontmatter BM25 picks top
        # candidates, re-rank the top body_rerank_pool of them using a second
        # BM25 pass over the note's body text. body_rerank_weight blends
        # the two scores. Set body_rerank_pool=0 to disable.
        "body_rerank_pool": int(r.get("body_rerank_pool", 50)),
        "body_rerank_weight": float(r.get("body_rerank_weight", 0.5)),
        # Graph-aware artifact ranking + narrow evidence-recency term (both default OFF).
        "ranking": _ranking_cfg(r.get("ranking")),
    }


def _unit(value, default: float) -> float:
    """A float clamped to [0, 1]; malformed -> default."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:  # NaN
        return default
    return min(1.0, max(0.0, out))


def _ranking_cfg(block) -> dict:
    """``retrieval.ranking`` with defaults: ``{graph_aware, query_stopwords,
    recency:{enabled, weight, floor, tie_window}}``.  Anything malformed falls
    back to OFF / the default value."""
    block = block if isinstance(block, dict) else {}
    rec_in = block.get("recency")
    rec_in = rec_in if isinstance(rec_in, dict) else {}
    defaults = _DEFAULT_RANKING["recency"]
    return {
        "graph_aware": block.get("graph_aware") is True,
        "query_stopwords": block.get("query_stopwords") is True,
        "recency": {
            "enabled": rec_in.get("enabled") is True,
            "weight": _unit(
                rec_in.get("weight", defaults["weight"]), defaults["weight"]
            ),
            "floor": _unit(rec_in.get("floor", defaults["floor"]), defaults["floor"]),
            "tie_window": _unit(
                rec_in.get("tie_window", defaults["tie_window"]), defaults["tie_window"]
            ),
        },
    }


def _rank_query(query: str, rcfg: dict) -> str:
    """The text BM25 actually scores, after the gated query-stopword pruning.

    Query-side only, and applied identically in both ranking paths and in both
    BM25 stages (frontmatter + body re-rank), so the two stages never disagree
    about what the question was.  Flag off == the query byte-for-byte.
    """
    if not (rcfg.get("ranking") or {}).get("query_stopwords"):
        return query
    return strip_stopwords(query)


class RankingContext:
    """Per-request bundle for the gated, optional ranking terms.

    ``None`` (from :meth:`load`) means "nothing enabled" and every ranking path
    then behaves byte-for-byte as before.  ``graph`` is the artifact lifecycle
    graph (only loaded when ``graph_aware`` is on); ``freshness`` is an
    evidence-dates context (only when ``recency.enabled`` is on — independent of
    ``retrieval.freshness.enabled``, which governs the row *label*).
    """

    __slots__ = ("cfg", "graph", "freshness")

    def __init__(
        self, cfg: dict, graph: dict | None, freshness: FreshnessContext | None
    ):
        self.cfg = cfg
        self.graph = graph
        self.freshness = freshness

    @classmethod
    def load(
        cls, root: str, cfg: dict, *, include_history: bool = False
    ) -> "RankingContext | None":
        rk = _ranking_cfg((cfg.get("retrieval") or {}).get("ranking"))
        if include_history:
            return None  # history views must show the old pages un-demoted
        graph_aware = rk["graph_aware"]
        recency = rk["recency"]["enabled"]
        if not graph_aware and not recency:
            return None
        graph = _load_artifact_graph(root) if graph_aware else None
        fresh = FreshnessContext(load_evidence_dates(root), cfg) if recency else None
        return cls(rk, graph, fresh)


def _graph_factor(note, ranking: "RankingContext | None") -> float:
    """Graph-aware term for ``artifact-<slug>`` notes: a page with an accepted
    successor takes the superseded penalty; the latest page of a multi-page topic
    gets a small boost (capped at ``_GRAPH_LATEST_BOOST``).  Non-artifact notes
    and unknown pages are untouched."""
    if ranking is None or ranking.graph is None:
        return 1.0
    name = str(getattr(note, "name", "") or "")
    if not name.startswith(_ARTIFACT_NOTE_PREFIX):
        return 1.0
    page = _artifact_node(ranking.graph, name[len(_ARTIFACT_NOTE_PREFIX) :])
    if not page:
        return 1.0
    if page.get("superseded_by"):
        return _SUPERSEDED_RANK_PENALTY
    if page.get("latest_in_topic"):
        topic = (ranking.graph.get("topics") or {}).get(page.get("topic") or "") or {}
        if len(topic.get("pages") or []) > 1:
            return _GRAPH_LATEST_BOOST
    return 1.0


def _apply_recency(
    ranked: list[tuple[object, float]], ranking: "RankingContext | None"
) -> list[tuple[object, float]]:
    """Narrow evidence-recency term (optional): only ``state``/``project`` notes, only
    among candidates within ``tie_window`` of the top score, multiplied by
    ``(1-w) + w * max(0.5 ** (age / half_life), floor)`` where ``age`` is days
    since the note's *evidence* date (``updated``-anchored or undatable notes
    keep 1.0).  Half-life is 3 x the type's freshness SLO.  Order-preserving
    on the list; the caller sorts afterwards."""
    if ranking is None or ranking.freshness is None or not ranked:
        return ranked
    rc = ranking.cfg["recency"]
    top = max(score for _, score in ranked)
    if top <= 0:
        return ranked
    cut = top * (1.0 - rc["tie_window"])
    weight, floor = rc["weight"], rc["floor"]
    out: list[tuple[object, float]] = []
    for note, score in ranked:
        if score >= cut and str(getattr(note, "type", "") or "") in _RECENCY_TYPES:
            fresh = ranking.freshness.for_note(note)
            age = fresh.get("age_days")
            if fresh.get("anchor") in ("evidence", "derived") and age is not None:
                half_life = _RECENCY_HALF_LIFE_SLO_MULTIPLE * max(
                    1, int(fresh["slo_days"])
                )
                decay = max(0.5 ** (age / half_life), floor)
                score = score * ((1.0 - weight) + weight * decay)
        out.append((note, score))
    return out


def _trust_cfg(cfg: dict) -> dict:
    """Trust-tier knobs.

    Fresh agent writes and legacy notes without an explicit approval both fail
    closed to ``proposed``.  The rank penalty keeps them discoverable while
    preferring evidence-reviewed memory.
    """
    t = cfg.get("trust", {}) or {}
    if not isinstance(t, dict):
        t = {}
    try:
        penalty = float(t.get("proposed_rank_penalty", 0.5))
    except (TypeError, ValueError):
        penalty = 0.5
    return {
        "proposed_rank_penalty": min(1.0, max(0.0, penalty)),
        # Only the literal JSON false disables the quality gate. Missing,
        # malformed, or truthy-string values retain the fail-closed default.
        "hide_verification_failures": verification_gate_enabled(cfg),
    }


# ACL principals. Two DISTINCT sentinels so "see everything" can never be reached by
# omission (the old fail-open bug: None meant both "no identity" AND "full access").
_TRUTHY = {"1", "true", "yes", "on"}


def full_access_enabled() -> bool:
    """True only when REMEMBRANCER_ALLOW_FULL_ACCESS is explicitly truthy. NEVER set in production —
    the threat model is the network edge (nginx→identity), not a local shell operator."""
    return os.environ.get("REMEMBRANCER_ALLOW_FULL_ACCESS", "").strip().lower() in _TRUTHY


def grants_for(identity: str | None, root: str = "."):
    """ACL labels this identity may read, from config.json identities.

    Default-DENY: a missing/empty identity (anonymous / unmapped edge caller) sees only public
    notes — identical to how an unknown non-empty identity already behaves. Full access is
    reachable via REMEMBRANCER_ALLOW_FULL_ACCESS for an anonymous local CLI caller only.
    There is deliberately no process-wide ACL-disable mode: a service misconfiguration must
    never turn every authenticated identity into full access.
    """
    ident = (identity or "").strip()
    if not ident:  # None or empty/whitespace
        return FULL_ACCESS if full_access_enabled() else DENY
    cfg = _load_config(os.path.abspath(root))
    return set(cfg.get("identities", {}).get(ident, []))  # unknown → ∅ → public-only


def visible(note, grants) -> bool:
    """A note is visible if grants is FULL_ACCESS, its acl is [] (public), or acl ∩ grants."""
    if grants is FULL_ACCESS:  # explicit full-access only
        return True
    acl = note.acl
    if isinstance(acl, str):
        acl = [acl]
    if not acl:  # [] == public
        return True
    return bool(set(acl) & grants)


# --- canonical entity matching -------------------------------------------------
# Mirrors mcp/server.py's `_entity_in`: note frontmatter `entities:` historically
# holds raw surface strings ("Jane Doe", "jane doe", "jane@example.com"); this is the
# PREFERRED search path (mcp/server.py calls into `search()` here rather than
# reimplementing it), so once notes store canonical ids, exact-string
# matching alone breaks entity-scoped search entirely. Guarded import so this
# module keeps importing cleanly even if distill/ isn't on the path yet.
try:
    from distill.entities import Canonicalizer
except Exception:  # pragma: no cover - keeps retrieve/ self-contained
    Canonicalizer = None


_REGISTRY_READS: ContextVar[dict | None] = ContextVar("remembrancer_registry_reads", default=None)


def _registry_read_scope(function):
    """Fingerprint each registry file once per retrieval, including nested reads."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _REGISTRY_READS.get() is not None:
            return function(*args, **kwargs)
        token = _REGISTRY_READS.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _REGISTRY_READS.reset(token)
    return wrapped


def _registry_file_revision(path: str) -> tuple | None:
    try:
        with open(path, "rb") as handle:
            stat = os.fstat(handle.fileno())
            return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
                    stat.st_ctime_ns, hashlib.sha256(handle.read()).digest())
    except OSError:
        return None


def _registry_revision(paths: tuple[str, ...]) -> tuple:
    """Notice content changes even when a filesystem reuses all stat fields.

    Rapid equal-size writes can have identical mtime AND ctime on ext4. Content
    fingerprints prevent stale aliases in that case. The per-retrieval scope
    amortizes file reads across every candidate; parsed YAML stays cached.
    """
    reads = _REGISTRY_READS.get()
    revision = []
    for path in paths:
        path = os.path.abspath(path)
        if reads is not None and path in reads:
            value = reads[path]
        else:
            value = _registry_file_revision(path)
            if reads is not None:
                reads[path] = value
        revision.append(value)
    return tuple(revision)


def _entity_registry_revision(registry_path: str) -> tuple:
    directory = os.path.dirname(registry_path)
    # Canonicalizer switches to split mode when orgs.yml appears. Bind both
    # layouts so transitions and updates to any split input invalidate it.
    return _registry_revision(
        (registry_path,)
        + tuple(
            os.path.join(directory, filename)
            for filename in ("orgs.yml", "people.yml", "systems.yml", "noise.yml")
        )
    )


@lru_cache(maxsize=8)
def _canonicalizer(registry_path: str, revision: tuple):
    return Canonicalizer(registry_path) if Canonicalizer is not None else None


def canonicalize(
    identifier: str, registry_path: str = "registry/entities.yml"
) -> str | None:
    resolver = _canonicalizer(registry_path, _entity_registry_revision(registry_path))
    return resolver.canonical_id(identifier) if resolver is not None else None


# --- an account's Slack room is evidence the note is about that account --------
# `entities:` is empty on a large share of Slack-derived notes, because the
# resolver could only see the orgs in registry/customers.yml when those notes
# were distilled.  The room the
# conversation happened in is right there in the provenance
# (`slack:ext-acme/...`), and the registry now records which rooms
# belong to which account.  So an entity-scoped search falls back to it: asking
# about Acme returns what was said in Acme's room even when the note names
# no entity.  This only ever ADDS results — it runs after both the exact and the
# canonical comparison have already missed.
_CHANNEL_MAP_CACHE: dict[str, tuple[tuple, dict[str, set[str]]]] = {}
_PROV_CHANNEL_RE = re.compile(r"slack:([^/\s,\]]+)/")


def _norm_channel(name: str) -> str:
    return str(name or "").strip().lower().lstrip("#").replace("_", "-")


@_registry_read_scope
def _entity_channels(root: str) -> dict[str, set[str]]:
    """canonical entity id -> the Slack rooms registered to it."""
    path = os.path.join(root, "registry", "customers.yml")
    reg_path = os.path.join(root, "registry", "entities.yml")
    signature = _registry_revision((path,)) + _entity_registry_revision(reg_path)
    cached = _CHANNEL_MAP_CACHE.get(root)
    if cached is not None and cached[0] == signature:
        return cached[1]
    mapping: dict[str, set[str]] = {}
    try:
        import yaml

        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except Exception:  # noqa: BLE001
        _CHANNEL_MAP_CACHE[root] = (signature, mapping)
        return mapping
    entries = raw.get("customers") if isinstance(raw, dict) else raw
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        rooms = {
            _norm_channel(c)
            for c in (entry.get("slack_channels") or [])
            if str(c).strip()
        }
        if not rooms:
            continue
        ident = str(entry.get("id") or "")
        for key in {ident, canonicalize(ident, reg_path) or ident}:
            if key:
                mapping.setdefault(key, set()).update(rooms)
    _CHANNEL_MAP_CACHE[root] = (signature, mapping)
    return mapping


def _note_channels(note) -> set[str]:
    prov = getattr(note, "provenance", None) or []
    if isinstance(prov, str):
        prov = [prov]
    out: set[str] = set()
    for item in prov:
        for hit in _PROV_CHANNEL_RE.findall(str(item)):
            out.add(_norm_channel(hit))
    return out


@_registry_read_scope
def _matches_entity(note, entity: str | None, root: str = ROOT) -> bool:
    """`entity in note.entities`, canonical-id aware (same discipline as
    mcp/server.py's `_entity_in`).

    Exact string match wins first (unchanged behaviour). If that misses, both
    the query and each stored value are resolved through `canonicalize` and
    compared by canonical id — so "Jane Doe", "jane doe" and "jane@example.com" all
    match a query of "Janet Doe"/"person:jane-doe". Falls back to no-match
    whenever either side is unresolvable, so unmigrated/unknown strings never
    spuriously collide just because they both fail to canonicalize.
    """
    if not entity:
        return True
    ents = note.entities
    if isinstance(ents, str):
        ents = [ents]
    ents = ents or []
    if entity in ents:
        return True
    reg_path = os.path.join(root, "registry", "entities.yml")
    q_cid = canonicalize(entity, reg_path)
    if q_cid is not None and any(canonicalize(v, reg_path) == q_cid for v in ents):
        return True
    rooms = _entity_channels(root).get(q_cid or entity) or _entity_channels(root).get(
        entity
    )
    return bool(rooms and (rooms & _note_channels(note)))


def _build_superseded_set(
    notes: list,
    human_audit: HumanTrustAudit | None = None,
) -> set[str]:
    """Return targets of active supersession edges from caller-visible heads.

    Callers must pass only heads visible to the acting identity.  The shared
    lifecycle predicate additionally requires explicit ``state: verified``;
    proposed, invalid, rejected, tombstoned, and legacy state-less heads do
    not change current truth.
    """
    return active_supersession_targets(notes, human_audit)


def _contested_root(root: str, cfg: dict) -> str | None:
    """The root to read contested labels from, or ``None`` to read none.

    One switch, shared with the pipeline: ``digest.review_policy: "settle"``
    means a contradicted claim is served with the evidence against it instead
    of parked in a queue.  Absent, no verification ledger is read on this path
    and every row is byte-identical to the pre-policy contract.
    """
    from retrieve.contested import review_policy_settles

    return root if review_policy_settles(cfg) else None


def _note_to_entry(
    note,
    score: float,
    root: str,
    verification_audit: VerificationAudit,
    human_audit: HumanTrustAudit | None = None,
    freshness: FreshnessContext | None = None,
    contested_root: str | None = None,
    memory_view=None,
) -> dict:
    """Index-first entry: metadata + score + path, NO body.

    ``freshness`` is loaded once per request (``FreshnessContext.load``); it is
    ``None`` when ``retrieval.freshness.enabled`` is false, and then the key is
    omitted so the row is byte-identical to the pre-freshness contract.

    ``contested_root`` is the corpus root when ``digest.review_policy`` is
    ``settle``, and ``None`` otherwise (then no ledger is read and the row is
    byte-identical to the pre-policy contract).  When set, a note whose latest
    verification audit row is ``flagged`` with at least one ``contradicted``
    verdict carries a ``contested`` list of ``{claim, evidence_quote,
    event_ids}`` — the dispute is SERVED, with the evidence against it, instead
    of being hidden behind a queue.  The note file is never rewritten for this.
    """
    try:
        rel_path = os.path.relpath(note.path, root) if note.path else ""
    except ValueError:
        rel_path = note.path or ""
    entry = {
        "name": note.name,
        "title": getattr(note, "title", "") or note.name,
        "description": note.description,
        "type": note.type,
        "domain": note.domain,
        "acl": list(note.acl) if not isinstance(note.acl, str) else [note.acl],
        "entities": list(note.entities)
        if not isinstance(note.entities, str)
        else [note.entities],
        "provenance": list(note.provenance)
        if not isinstance(note.provenance, str)
        else [note.provenance],
        "updated": getattr(note, "updated", ""),
        "resource": getattr(note, "resource", ""),
        "tags": (
            list(note.tags)
            if not isinstance(getattr(note, "tags", []), str)
            else [note.tags]
        ),
        "state": effective_state(note, human_audit),
        "verification": public_verification_status(note, verification_audit),
        "score": round(float(score), 4),
        "path": rel_path,
    }
    from retrieve.maintenance import MemoryView, read_config, unresolved_claims
    memory = (memory_view or MemoryView(root, read_config(root))).for_note(note)
    if memory:
        entry["memory"] = memory
    if freshness is not None:
        entry["freshness"] = freshness.for_note(note)
    if contested_root is not None:
        # Imported here, not at module scope: with the policy absent this
        # module never touches the verification ledger at all.
        from retrieve.contested import contested_for

        claims = unresolved_claims(contested_for(contested_root, note.name), memory)
        if claims:
            entry["contested"] = claims
    return entry


def _ranking_text(note) -> str:
    """Return the complete BM25 document allowed by the retrieval contract.

    Titles, bodies, entities, and provenance are intentionally excluded:
    write-time distillation must put the current searchable truth in the stable
    name and description. Tags ARE included: they are
    write-time curated metadata of the same kind — the artifact pipeline stamps
    them from the publish manifest — and excluding them made curated facets
    unsearchable. Index keys are TF-IDF-selected
    distinctive body terms carried in the index row: they let stage-1 ranking
    see body-resident facts without storing bodies in the index. Only
    tag-carrying / key-carrying notes are affected.
    """
    name = str(getattr(note, "name", "") or "")
    description = str(getattr(note, "description") or "")
    raw_tags = getattr(note, "tags", None) or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    tags = " ".join(str(t) for t in raw_tags if t)
    keys = " ".join(str(t) for t in (getattr(note, "index_keys", None) or []) if t)
    return f"{name} {description} {tags} {keys}".strip()


def _ranking_text_body(note) -> str:
    """Body-only ranking text. Used in the second-stage body re-rank that
    refines BM25 ranking of the top-K candidates.

    The body is loaded from disk only for the top body_rerank_pool
    candidates (default 50), so we cap the body size in this string to
    avoid the BM25 tokenize pass dominating a single very long note.
    A 4000-char cap captures the bulk of note bodies and is well under
    the distiller's typical body length.
    """
    body = str(getattr(note, "body", "") or "")
    return body[:4000]


def _updated_in_range(updated: str, start, end) -> bool:
    """True when the note's updated day (YYYY-MM-DD prefix) is in [start, end].
    Notes without a parseable updated date cannot satisfy a time range."""
    try:
        day = date.fromisoformat(str(updated or "")[:10])
    except ValueError:
        return False
    return start <= day <= end


def _apply_time_filter(items: list, time_range) -> list:
    """Keep only items updated inside the parsed time range.

    Fallback: if the range would empty the candidate set, return the
    unfiltered list — a time phrase must never zero out an otherwise-good
    query (deterministic parsing has false positives; a recalled set is
    always safer than an empty one).
    """
    if time_range is None:
        return items
    start, end, _cleaned = time_range
    filtered = [
        item
        for item in items
        if _updated_in_range(str(getattr(item, "updated", "") or ""), start, end)
    ]
    return filtered or items


def _updated_timestamp(note) -> float:
    """Best-effort sortable timestamp; malformed/missing dates sort oldest."""
    value = str(getattr(note, "updated", "") or "").strip()
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (ValueError, OverflowError):
        return 0.0


def _is_superseded(
    note,
    edge_targets: set[str],
    human_audit: HumanTrustAudit | None = None,
) -> bool:
    """Whether a note is historical for this caller.

    ``superseded: true`` is retained as compatibility/audit metadata for an
    explicitly retired note, but is not an independent truth source.  This is
    what makes demoting or tombstoning the verified head restore its target
    immediately without a lint rewrite.
    """
    return (
        effective_state(note, human_audit) == "superseded" or note.name in edge_targets
    )


def _eligible(
    note,
    *,
    ambiguous_names: set[str],
    grants,
    entity: str | None,
    root: str,
    current_only: bool,
    hide_superseded: bool,
    verified_only: bool,
    edge_targets: set[str],
    verification_audit: VerificationAudit,
    human_audit: HumanTrustAudit,
    hide_verification_failures: bool,
) -> bool:
    """Apply the same security and lifecycle gates to index rows and notes."""
    stale = _is_superseded(note, edge_targets, human_audit)
    return (
        note.name not in ambiguous_names
        and visible(note, grants)
        and _matches_entity(note, entity, root=root)
        and (not current_only or not stale)
        and (not hide_superseded or not stale)
        and verification_allows_answer(
            note,
            verification_audit,
            enabled=hide_verification_failures,
            human_audit=human_audit,
        )
        and (
            not verified_only
            or effective_state(note, human_audit, verification_audit) == "verified"
        )
    )


def _adjust_score(
    note,
    raw_score: float,
    edge_targets: set[str],
    proposed_penalty: float,
    human_audit: HumanTrustAudit,
    ranking: "RankingContext | None" = None,
) -> float:
    """Soft-rank non-current notes when the caller opts into seeing history.

    Graph-signal boost: a note that SUPERSEDES others is a consolidated
    knowledge hub (the distiller collapses related threads into it). A small
    multiplicative boost rewards that consolidation; the BM25 frontmatter
    score still dominates, so this only re-ranks near-ties.

    ``ranking`` (optional, gated) adds the artifact-graph term: superseded pages take
    the same penalty as superseded notes, the latest page of a topic a small
    boost.  ``None`` leaves the score byte-identical to the score without it.
    """
    score = raw_score
    if effective_state(note, human_audit) == "proposed":
        score *= proposed_penalty
    if _is_superseded(note, edge_targets, human_audit):
        score *= _SUPERSEDED_RANK_PENALTY
    # Hub boost: count this note's outgoing ``supersedes`` typed links.
    # Capped at 1.5x so a single mega-note can't dominate the ranking.
    hub_edges = sum(
        1 for relation, _ in (note.typed_links or []) if relation == "supersedes"
    )
    if hub_edges > 0:
        score *= 1.0 + min(0.5, 0.1 * hub_edges)
    if ranking is not None:
        score *= _graph_factor(note, ranking)
    return score


def _sort_ranked(ranked: list[tuple[object, float]]) -> list[tuple[object, float]]:
    """Relevance first; ties prefer newer truth, then stable name."""
    return sorted(
        ranked,
        key=lambda pair: (-pair[1], -_updated_timestamp(pair[0]), pair[0].name),
    )


def _record_is_suppressed(record: IndexRecord, suppressed: set[str]) -> bool:
    return record.name in suppressed or slugify(record.name) in suppressed


def _load_index_records(
    context_dir: str, records: list[IndexRecord],
    verification_audit: VerificationAudit | None = None,
) -> dict[str, Note] | None:
    """Load and validate selected note files from a fresh index snapshot.

    Only candidates and supersession heads are opened.  Any mismatch requests a
    full fallback; retrieval never silently trusts an inconsistent index row.
    """
    loaded: dict[str, Note] = {}
    context_abs = os.path.abspath(context_dir)
    for record in records:
        if record.path in loaded:
            continue
        path = os.path.abspath(os.path.join(context_abs, *record.path.split("/")))
        try:
            if os.path.commonpath((context_abs, path)) != context_abs:
                return None
            with open(path, "rb") as handle:
                raw = handle.read()
                stat = os.fstat(handle.fileno())
            if (
                stat.st_size != record.size
                or hashlib.sha256(raw).hexdigest() != record.content_sha256
            ):
                return None
            fm, body = parse_frontmatter(raw.decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return None
        if not fm:
            return None
        note = Note.from_frontmatter(fm, body, path)
        if (
            note.name != record.name
            or (note.title or note.name) != record.title
            or note.description != record.description
            or (note.updated or "") != record.updated
            or tuple(note.acl or []) != record.acl
            or tuple(note.entities or []) != record.entities
            or tuple(note.provenance or []) != record.provenance
            or tuple(note.source_lineages or []) != record.source_lineages
            or note.state != record.state
            or note.approval != record.approval
            or note.source_retirement != record.source_retirement
            or note.producer_lifecycle != record.producer_lifecycle
            or embedded_verification_status(note) != record.verification
            or bool(note.superseded) != record.superseded
            or tuple(
                target
                for relation, target in note.typed_links
                if relation == "supersedes"
            )
            != record.supersedes
        ):
            return None
        loaded[record.path] = note
    bind_loaded_source_retirements(
        list(loaded.values()),
        os.path.dirname(context_abs),
    )
    bind_loaded_producer_lifecycles(
        list(loaded.values()),
        os.path.dirname(context_abs),
    )
    bind_loaded_auto_approvals(
        list(loaded.values()),
        os.path.dirname(context_abs),
        verification_audit,
    )
    return loaded


def _rank_full_corpus(
    query: str,
    all_notes: list[Note],
    *,
    grants,
    entity: str | None,
    root: str,
    current_only: bool,
    hide_superseded: bool,
    verified_only: bool,
    proposed_penalty: float,
    verification_audit: VerificationAudit,
    human_audit: HumanTrustAudit,
    hide_verification_failures: bool,
    time_range=None,
    ranking: "RankingContext | None" = None,
) -> list[tuple[Note, float]]:
    """Authoritative fallback for missing/legacy/stale indexes."""
    name_counts = Counter(note.name for note in all_notes)
    ambiguous_names = {name for name, count in name_counts.items() if count > 1}
    visible_heads = [
        note
        for note in all_notes
        if note.name not in ambiguous_names and visible(note, grants)
    ]
    edge_targets = _build_superseded_set(visible_heads, human_audit)
    eligible = [
        note
        for note in all_notes
        if _eligible(
            note,
            ambiguous_names=ambiguous_names,
            grants=grants,
            entity=entity,
            root=root,
            current_only=current_only,
            hide_superseded=hide_superseded,
            verified_only=verified_only,
            edge_targets=edge_targets,
            verification_audit=verification_audit,
            human_audit=human_audit,
            hide_verification_failures=hide_verification_failures,
        )
    ]
    # Parity with the index path: notes lack the index row's index_keys, so
    # compute the same TF-IDF body terms on the fly and attach them. The df
    # pass must run over the FULL corpus, exactly like regen — computing it
    # over the eligible subset would make the document (and therefore the
    # ranking) depend on the caller's ACL view, breaking parity with the
    # index-first path this backs up.
    from remembrancerlib import compute_index_keys

    keys_by_path = compute_index_keys(all_notes)
    for note in eligible:
        note.index_keys = keys_by_path.get(note.path, [])
    eligible = _apply_time_filter(eligible, time_range)
    # A query that was ONLY a time phrase has no BM25 terms; every eligible
    # note scores uniformly and _sort_ranked's recency tie-break orders them.
    raw_scores = (
        BM25([_ranking_text(note) for note in eligible]).score(query)
        if query.strip()
        else [1.0] * len(eligible)
    )
    ranked = [
        (
            note,
            _adjust_score(
                note,
                score,
                edge_targets,
                proposed_penalty,
                human_audit,
                ranking,
            ),
        )
        for note, score in zip(eligible, raw_scores)
        if score > 0
    ]
    return _sort_ranked(_apply_recency(ranked, ranking))


def _rank_index_snapshot(
    query: str,
    snapshot: IndexSnapshot,
    *,
    context_dir: str,
    grants,
    entity: str | None,
    root: str,
    current_only: bool,
    hide_superseded: bool,
    verified_only: bool,
    proposed_penalty: float,
    verification_audit: VerificationAudit,
    human_audit: HumanTrustAudit,
    hide_verification_failures: bool,
    rcfg: dict | None = None,
    time_range=None,
    ranking: "RankingContext | None" = None,
) -> list[tuple[Note, float]] | None:
    """Rank through a complete fresh index, loading only rows that can surface.

    The index contains no note bodies.  Supersession-head files are opened to
    verify their ACL and typed links before they may hide another note.
    ``None`` means an inconsistency was found and the caller must full-scan.
    """
    repo_root = os.path.dirname(os.path.abspath(context_dir))
    suppressed = load_suppressed(repo_root)
    records = [
        record
        for record in snapshot.records
        if not _record_is_suppressed(record, suppressed)
    ]
    name_counts = Counter(record.name for record in records)
    ambiguous_names = {name for name, count in name_counts.items() if count > 1}

    head_records = [record for record in records if record.supersedes]
    # Source/producer lifecycle validity is dual-bound to append-only journals
    # and deliberately is not self-attested by index metadata.  Load every row
    # carrying such a control before the *first* eligibility pass; otherwise a
    # restored producer note looks superseded in the index path and is dropped
    # before it can ever be authoritatively rebound.
    control_records = [
        record
        for record in records
        if record.source_retirement or record.producer_lifecycle
    ]
    preload_by_path = {
        record.path: record for record in (*head_records, *control_records)
    }
    loaded = _load_index_records(
        context_dir, list(preload_by_path.values()), verification_audit)
    if loaded is None:
        return None
    visible_heads = [
        loaded[record.path]
        for record in head_records
        if record.name not in ambiguous_names and visible(loaded[record.path], grants)
    ]
    edge_targets = _build_superseded_set(visible_heads, human_audit)

    eligible_records = [
        record
        for record in records
        if _eligible(
            loaded.get(record.path, record),
            ambiguous_names=ambiguous_names,
            grants=grants,
            entity=entity,
            root=root,
            current_only=current_only,
            hide_superseded=hide_superseded,
            verified_only=verified_only,
            edge_targets=edge_targets,
            verification_audit=verification_audit,
            human_audit=human_audit,
            hide_verification_failures=hide_verification_failures,
        )
    ]
    eligible_records = _apply_time_filter(eligible_records, time_range)
    # A query that was ONLY a time phrase has no BM25 terms; every eligible
    # record scores uniformly and the recency tie-break orders them.
    raw_scores = (
        BM25([_ranking_text(record) for record in eligible_records]).score(query)
        if query.strip()
        else [1.0] * len(eligible_records)
    )
    scored_records = [
        (record, score)
        for record, score in zip(eligible_records, raw_scores)
        if score > 0
    ]
    # Rank-order the candidates BEFORE anything slices them: this list is built in
    # index order, and the body re-rank pool below takes the first ``body_pool``
    # entries. Without this sort the second stage re-ranked whichever notes the
    # index happened to store first rather than the best-scoring ones (the
    # measured mean overlap with the true top-50 was 41.7%).
    scored_records.sort(
        key=lambda pair: pair[1], reverse=True
    )  # stable: ties keep index order
    candidates = [record for record, _ in scored_records]
    candidate_notes = _load_index_records(context_dir, candidates, verification_audit)
    if candidate_notes is None:
        return None
    loaded.update(candidate_notes)

    # Two-stage retrieval: re-rank the top ``body_rerank_pool`` candidates
    # using a BM25 over the note's full body text. Body is not in the
    # snapshot (would bloat the tracked index file), so this stage loads it
    # on-demand from the loaded Note objects — only the top few hundred
    # candidates, not the full corpus. Default pool = 50; configurable.
    if rcfg is None:
        body_pool, body_weight = 0, 0.0
    else:
        body_pool = int(rcfg.get("body_rerank_pool", 50))
        body_weight = float(rcfg.get("body_rerank_weight", 0.5))
    if body_pool > 0 and body_weight > 0 and len(scored_records) > 1:
        top_for_body = scored_records[:body_pool]
        body_docs = []
        body_notes: list[Note] = []
        for record, _ in top_for_body:
            note = loaded[record.path]
            if note is None:
                body_docs.append("")
                body_notes.append(None)
                continue
            body_docs.append(_ranking_text_body(note))
            body_notes.append(note)
        body_scores = BM25(body_docs).score(query)
        # Blend: new_score = frontmatter_score + body_weight * body_score
        for i, (record, fs) in enumerate(top_for_body):
            bs = body_scores[i] if i < len(body_scores) else 0.0
            scored_records[i] = (record, fs + body_weight * bs)

    ranked: list[tuple[Note, float]] = []
    for record, raw_score in scored_records:
        note = loaded[record.path]
        # Reapply every gate to authoritative frontmatter before returning a
        # row.  Index ACL metadata is only a discovery optimization.
        if not _eligible(
            note,
            ambiguous_names=ambiguous_names,
            grants=grants,
            entity=entity,
            root=root,
            current_only=current_only,
            hide_superseded=hide_superseded,
            verified_only=verified_only,
            edge_targets=edge_targets,
            verification_audit=verification_audit,
            human_audit=human_audit,
            hide_verification_failures=hide_verification_failures,
        ):
            continue
        ranked.append(
            (
                note,
                _adjust_score(
                    note,
                    raw_score,
                    edge_targets,
                    proposed_penalty,
                    human_audit,
                    ranking,
                ),
            )
        )
    return _sort_ranked(_apply_recency(ranked, ranking))


@_registry_read_scope
def search(
    query: str,
    identity: str | None = None,
    k: int | None = None,
    entity: str | None = None,
    root: str = ".",
    current_only: bool = False,
    verified_only: bool = False,
    include_history: bool = False,
) -> list[dict]:
    """ACL-gated, BM25-ranked recall-wide candidate pool of index entries (no bodies).

    Returns up to `k` entries (default = config retrieval.candidate_pool) ranked by
    BM25 over ``name + description + tags``,
    restricted to notes the identity may read and optionally scoped to one entity.
    A validated per-domain index supplies candidates; stale/legacy/missing indexes
    fall back to the full corpus. Entries are index-first dicts:
    {name, description, type, domain, acl, entities, provenance, score, path}.

    A missing identity sees public notes only unless the local-only
    ``REMEMBRANCER_ALLOW_FULL_ACCESS`` flag is explicitly enabled.

    When current_only=True, any note that is the TARGET of a ``supersedes``
    typed-link from any other note is excluded from results (edge-aware filter).
    The note that *does* the superseding is kept; only the superseded target is
    hidden.

    Note: ``current_only`` is placed after ``root`` so existing positional callers
    of the form ``search(q, identity, k, entity, root)`` are not broken.
    """
    root = os.path.abspath(root)
    _t0 = time.monotonic()
    cfg = _load_config(root)
    rcfg = _retrieval_cfg(cfg)
    pool = max(0, int(k if k is not None else rcfg["candidate_pool"]))

    grants = grants_for(identity, root)
    context_dir = os.path.join(root, "context")
    # API compatibility: default retrieval hides superseded history; callers
    # can set hide_superseded:false to include it, in which case it receives a
    # soft rank penalty rather than competing as current truth.
    # A per-request history opt-in overrides the default/current filters
    # without weakening the repository-wide default for other callers.
    hide = False if include_history else bool(rcfg.get("hide_superseded", True))
    current = False if include_history else current_only
    tcfg = _trust_cfg(cfg)
    proposed_penalty = tcfg["proposed_rank_penalty"]
    hide_verification_failures = tcfg["hide_verification_failures"]
    verification_audit = load_verification_audit(root)
    human_audit = load_human_trust_audit(root)

    snapshot = load_index_snapshot(context_dir)
    # Time-aware retrieval: lift an optional time range out of the query and
    # filter candidates by their updated day; the cleaned remainder goes to
    # BM25. Deterministic (no LLM on the read path); a range that empties the
    # candidate set falls back to unfiltered inside the rank functions.
    time_range = parse_time_range(query)
    rank_query = _rank_query(time_range[2] if time_range else query, rcfg)
    # Optional ranking terms (default OFF; None keeps the base order byte-for-byte).
    ranking = RankingContext.load(root, cfg, include_history=include_history)
    ranked = None
    rank_path = "fallback"
    if snapshot is not None:
        ranked = _rank_index_snapshot(
            rank_query,
            snapshot,
            context_dir=context_dir,
            grants=grants,
            entity=entity,
            root=root,
            current_only=current,
            hide_superseded=hide,
            verified_only=verified_only,
            proposed_penalty=proposed_penalty,
            verification_audit=verification_audit,
            human_audit=human_audit,
            hide_verification_failures=hide_verification_failures,
            rcfg=rcfg,
            time_range=time_range,
            ranking=ranking,
        )
        if ranked is not None:
            rank_path = "index"
    if ranked is None:
        ranked = _rank_full_corpus(
            rank_query,
            load_notes(context_dir, verification_audit=verification_audit),
            grants=grants,
            entity=entity,
            root=root,
            current_only=current,
            hide_superseded=hide,
            verified_only=verified_only,
            proposed_penalty=proposed_penalty,
            verification_audit=verification_audit,
            human_audit=human_audit,
            hide_verification_failures=hide_verification_failures,
            time_range=time_range,
            ranking=ranking,
        )

    from retrieve.maintenance import MemoryView
    memory_view = MemoryView(root, cfg)
    ranked = memory_view.rank(ranked, include_history=include_history)
    freshness = FreshnessContext.load(root, cfg)
    contested_root = _contested_root(root, cfg)
    entries = [
        _note_to_entry(
            note,
            score,
            root,
            verification_audit,
            human_audit,
            freshness=freshness,
            contested_root=contested_root,
            memory_view=memory_view,
        )
        for note, score in ranked[:pool]
    ]
    # Search itself never mutates context/events/registry.  This optional audit
    # signal is deliberately isolated under state/ and is best-effort.  The same
    # duration goes to the bound MCP read-call record (retrieve.telemetry), so the
    # audit row and the query log carry one number, not two measurements of it.
    duration_ms = int((time.monotonic() - _t0) * 1000)
    telemetry.set_value("duration_ms", duration_ms)
    telemetry.set_value("path", rank_path)
    _log_query(
        root,
        query,
        identity,
        entries,
        cfg,
        duration_ms=duration_ms,
    )
    return entries


@_registry_read_scope
def search_full_scan(
    query: str,
    identity: str | None = None,
    k: int | None = None,
    entity: str | None = None,
    root: str = ".",
    current_only: bool = False,
    verified_only: bool = False,
    include_history: bool = False,
) -> list[dict]:
    """Run the authoritative retrieval contract directly over the full corpus.

    This is the recovery surface for callers whose index-first search raised an
    operational exception.  It intentionally reuses the same ACL, suppression,
    ambiguity, trust, lifecycle, supersession and BM25 helpers as ``search``;
    the only difference is that it does not consult per-domain indexes.
    """
    root = os.path.abspath(root)
    _t0 = time.monotonic()
    cfg = _load_config(root)
    rcfg = _retrieval_cfg(cfg)
    tcfg = _trust_cfg(cfg)
    pool = max(0, int(k if k is not None else rcfg["candidate_pool"]))
    verification_audit = load_verification_audit(root)
    human_audit = load_human_trust_audit(root)
    time_range = parse_time_range(query)
    ranked = _rank_full_corpus(
        _rank_query(time_range[2] if time_range else query, rcfg),
        load_notes(os.path.join(root, "context"),
                   verification_audit=verification_audit),
        grants=grants_for(identity, root),
        entity=entity,
        root=root,
        current_only=False if include_history else current_only,
        hide_superseded=(
            False if include_history else bool(rcfg.get("hide_superseded", True))
        ),
        verified_only=verified_only,
        proposed_penalty=tcfg["proposed_rank_penalty"],
        verification_audit=verification_audit,
        human_audit=human_audit,
        hide_verification_failures=tcfg["hide_verification_failures"],
        time_range=time_range,
        ranking=RankingContext.load(root, cfg, include_history=include_history),
    )
    from retrieve.maintenance import MemoryView
    memory_view = MemoryView(root, cfg)
    ranked = memory_view.rank(ranked, include_history=include_history)
    freshness = FreshnessContext.load(root, cfg)
    contested_root = _contested_root(root, cfg)
    entries = [
        _note_to_entry(
            note,
            score,
            root,
            verification_audit,
            human_audit,
            freshness=freshness,
            contested_root=contested_root,
            memory_view=memory_view,
        )
        for note, score in ranked[:pool]
    ]
    _log_query(
        root,
        query,
        identity,
        entries,
        cfg,
        duration_ms=int((time.monotonic() - _t0) * 1000),
    )
    return entries


def dig_events(query: str, grants, cfg: dict, events_dir: str = "events") -> list[dict]:
    """L1 raw-event dig — explicit, user-insist-triggered escalation (NOT automatic).

    The distilled wiki (L2) stays the default source of truth; this only runs when the
    user insists the info exists. Returns grounded raw-event candidates to be shown to the
    user with the question 'does this answer your question?' — never asserted as fact
    until the user confirms (then it's written back to the wiki). BM25 score is NOT a
    confidence signal here (answerable & unanswerable queries overlap on score), so there
    is no score gate: we surface candidates and let human confirmation decide."""
    r = cfg.get("retrieval", {})
    rows = search_events(query, grants, k=r.get("l1_k", 8), events_dir=events_dir)
    for row in rows:
        row["tier"] = "L1-raw-event"
    return rows


def _log_query(
    root: str,
    query: str,
    identity: str | None,
    entries: list[dict],
    cfg: dict,
    duration_ms: int | None = None,
) -> None:
    """Append the query + its result quality to ``state/queries.jsonl`` — the activity
    log (Karpathy's ``log.md``) that the usage-feedback loop reads to find memory gaps.
    A query that returns nothing / scores low is the strongest signal of a missing or
    unfindable note. Best-effort: never breaks retrieval.
    Logging is fail-closed: it runs only when ``retrieval.query_log`` is the
    literal JSON boolean ``true``.  Missing, malformed, or truthy string values
    do not persist potentially sensitive queries."""
    try:
        if (cfg.get("retrieval") or {}).get("query_log") is not True:
            return
        override = os.environ.get("REMEMBRANCER_QUERY_LOG")
        if override == "off":
            return  # tests/benches disable persistence so local runs never
            # pollute a checkout's (or the server's) query stream
        if not telemetry.query_log_allowed():
            return  # the same switch, per MCP call (e.g. a monitoring probe)
        rec = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "query": query,
            "identity": identity,
            "n_results": len(entries),
            "top_score": round(entries[0]["score"], 3) if entries else 0.0,
            "delivered": [e["name"] for e in entries[:5]],
        }
        if duration_ms is not None:
            rec["duration_ms"] = int(duration_ms)
        path = override or os.path.join(root, "state", "queries.jsonl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def get_note(
    name: str,
    identity: str | None = None,
    root: str = ".",
    full: bool = False,
    include_history: bool = False,
) -> dict | None:
    """Open one note by name, ACL-checked.

    Returns the full entry including `body`, or None if the note doesn't exist or
    the identity isn't allowed to read it (the same gate `search` enforces).

    By default the body is truncated to ~max_slice_tokens*4 chars — right for an agent
    retrieval slice (bounded context). Pass full=True for the HUMAN wiki, which must
    render the whole note (a long reference page would otherwise cut off mid-section).

    When identity is None (default), only public notes are visible, unless
    REMEMBRANCER_ALLOW_FULL_ACCESS grants an anonymous local caller full access (see
    ``grants_for``).
    """
    root = os.path.abspath(root)
    cfg = _load_config(root)
    rcfg = _retrieval_cfg(cfg)
    tcfg = _trust_cfg(cfg)
    verification_audit = load_verification_audit(root)
    human_audit = load_human_trust_audit(root)
    grants = grants_for(identity, root)

    context_dir = os.path.join(root, "context")
    all_notes = load_notes(context_dir, verification_audit=verification_audit)
    matches = [n for n in all_notes if n.name == name]
    if len(matches) != 1:
        return None  # duplicate stable ids are ambiguous; never pick first/last.
    n = matches[0]
    if not visible(n, grants):
        return None  # ACL denial is indistinguishable from absence by design.
    if not verification_allows_answer(
        n,
        verification_audit,
        enabled=tcfg["hide_verification_failures"],
        human_audit=human_audit,
    ):
        return None
    if not include_history:
        visible_notes = [note for note in all_notes if visible(note, grants)]
        if _is_superseded(
            n,
            active_supersession_targets(visible_notes, human_audit),
            human_audit,
        ):
            return None
    entry = _note_to_entry(
        n,
        0.0,
        root,
        verification_audit,
        human_audit,
        freshness=FreshnessContext.load(root, cfg),
    )
    body = n.body or ""
    if not full:
        char_cap = rcfg["max_slice_tokens"] * 4  # ~4 chars/token heuristic.
        if len(body) > char_cap:
            body = body[:char_cap].rstrip() + "\n\n…[truncated]"
    entry["body"] = body
    entry["structured"] = getattr(n, "structured", None)
    entry.pop("score", None)
    from retrieve.maintenance import record_open
    record_open(root, name, cfg)
    return entry


def _file_back_cfg(cfg: dict) -> dict:
    """Return the file_back sub-config with safe defaults."""
    fb = (cfg.get("retrieval") or {}).get("file_back") or {}
    return {
        "enabled": bool(fb.get("enabled", True)),
        "min_chars": int(fb.get("min_chars", 200)),
    }


def _synthesis_acl(sources: list[Note]) -> list[str]:
    """Return the narrowest ACL that can safely contain every source.

    Remembrancer ACL labels are OR grants, so copying/unioning labels would make a
    synthesis visible to people who could not see every source.  A common
    grant is safe; incompatible labels have no safe OR representation and the
    synthesis is refused.
    """
    return combine_acl(*(list(note.acl or []) for note in sources))


def _file_answer_sources(
    source_names: list[str], identity: str | None, root: str
) -> list[Note]:
    """Resolve named source notes once, reject ambiguity and enforce ACLs."""
    if not source_names:
        raise ValueError("file_answer requires at least one source note")
    context_dir = os.path.join(root, "context")
    by_name: dict[str, list[Note]] = {}
    for note in load_notes(context_dir):
        by_name.setdefault(note.name, []).append(note)
    grants = grants_for(identity, root)
    cfg = _load_config(root)
    tcfg = _trust_cfg(cfg)
    verification_audit = load_verification_audit(root)
    human_audit = load_human_trust_audit(root)
    out: list[Note] = []
    seen: set[str] = set()
    for raw_name in source_names:
        name = str(raw_name or "").strip()
        if not name:
            raise ValueError("file_answer source name is empty")
        if name in seen:
            continue
        seen.add(name)
        candidates = by_name.get(name, [])
        if len(candidates) != 1:
            raise ValueError(f"file_answer source {name!r} is missing or ambiguous")
        source = candidates[0]
        if not visible(source, grants):
            raise PermissionError(f"not permitted to cite source {name!r}")
        if not verification_allows_answer(
            source,
            verification_audit,
            enabled=tcfg["hide_verification_failures"],
            human_audit=human_audit,
        ):
            raise PermissionError(f"source {name!r} is unavailable")
        out.append(source)
    if not out:
        raise ValueError("file_answer requires at least one source note")
    return out


def _synthesis_name(question: str, answer: str, sources: list[Note]) -> str:
    """Return a content-addressed, reserved name for a file-back synthesis.

    The question slug alone is an untrusted namespace: it can equal any existing
    human or restricted note name.  The reserved prefix and answer/source digest
    make a new synthesis additive, while a later exact replay resolves to the
    same note without writing over unrelated memory.
    """
    slug = slugify(question)[:44] or "answer"
    material = "\0".join([question, answer, *[source.name for source in sources]])
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"synthesis-{slug}-{digest}"


@serialized_pipeline_mutation()
def file_answer(
    question: str,
    answer: str,
    source_names: list[str],
    root: str = ".",
    identity: str | None = None,
) -> str:
    """Write a synthesised answer back as a ``type: synthesis`` note.

    The note is stored under its source domain (or ``company`` for mixed
    sources) with a reserved, content-addressed ``synthesis-`` name.  It never
    overwrites a pre-existing note.

    Returns the absolute path of the written note on success, or ``""`` when:
    - ``retrieval.file_back.enabled`` is ``false`` in ``config.json``, or
    - ``len(answer) < retrieval.file_back.min_chars``.

    Args:
        question:     The original question whose answer is being filed.
        answer:       The synthesised answer body to persist.
        source_names: Names of the source notes the answer was derived from.
                      Every source must exist exactly once and be visible to
                      ``identity``; they are written into provenance.
        identity: Acting identity used to enforce source ACLs. Omitted means
                  public-source-only, never implicit full access.
        root:         Repository root; ``config.json`` and ``context/`` are
                      resolved relative to it.
    """
    root = os.path.abspath(root)
    cfg = _load_config(root)
    fb = _file_back_cfg(cfg)

    if not fb["enabled"]:
        return ""
    if len(answer) < fb["min_chars"]:
        return ""

    sources = _file_answer_sources(source_names, identity, root)
    source_domains = {note.domain for note in sources}
    # Preserve a single source domain when it is unambiguous; a mixed-source
    # synthesis belongs in company but stays protected by the derived ACL.
    domain = next(iter(source_domains)) if len(source_domains) == 1 else "company"

    name = _synthesis_name(question, answer, sources)
    derived_acl = _synthesis_acl(sources)
    if not _writer_authorized(identity, derived_acl, root):
        raise PermissionError(
            "file_answer requires an authorized writer for the derived ACL"
        )
    context_dir = os.path.join(root, "context")
    existing = [
        candidate for candidate in load_notes(context_dir) if candidate.name == name
    ]
    if existing:
        # Exact replays are idempotent.  A collision with any altered/human
        # note must fail closed instead of letting a generated answer replace it.
        prior = existing[0] if len(existing) == 1 else None
        if (
            prior is not None
            and prior.type == "synthesis"
            and prior.body.rstrip() == answer.rstrip()
            and list(prior.provenance or []) == [source.name for source in sources]
            and list(prior.acl or []) == derived_acl
        ):
            from remembrancerlib import human_trust_valid

            if prior.state == "verified" and human_trust_valid(prior):
                return prior.path
            if prior.state != "proposed":
                raise ValueError(
                    f"file_answer synthesis {name!r} has no valid reviewable state"
                )
            import govern

            govern.enqueue_for_review(
                prior,
                requester=str(identity or "file-answer"),
                reason="file-answer synthesis retry",
                root=root,
                cls="file-answer-synthesis",
            )
            return prior.path
        raise ValueError(f"file_answer synthesis name {name!r} already exists")

    note = Note(
        name=name,
        description=question,
        type="synthesis",
        domain=domain,
        acl=derived_acl,
        provenance=[note.name for note in sources],
        # A generated answer is not independently verified merely because its
        # citations exist. A human may promote it after review.
        state="proposed",
        body=answer,
    )
    import govern

    govern.prepare_for_review(note, "file-answer-synthesis")
    path = write_note(note, context_dir)
    govern.enqueue_for_review(
        note,
        requester=str(identity or "file-answer"),
        reason="file-answer synthesis",
        root=root,
        cls="file-answer-synthesis",
    )
    return path


def _remember_domain(supersedes: str | None, context_dir: str) -> str:
    """Resolve the domain for a superseding note: the superseded note's domain."""
    matches = [note for note in load_notes(context_dir) if note.name == supersedes]
    if len(matches) > 1:
        raise ValueError(f"ambiguous note name {supersedes!r}")
    if matches:
        return matches[0].domain
    return "company"


# The labels a note gets when a person writes it through remember or
# create_page, by its domain.  ``authored_acl`` in config.json adds domains or
# overrides these; a domain neither names gets ``secrets``, so a write nobody
# mapped stays closed.
_DOMAIN_WRITE_ACL = {
    "eng": ["eng"],
    "product": ["eng"],
    "repos": ["eng"],
    "design": ["eng"],
    "ml": ["ml"],
    "research": ["ml"],
    "customers": ["customers"],
    "prospect": ["customers"],
    "partners": ["customers"],
    "vendor": ["customers"],
    "network": ["customers"],
    "infra": ["infra"],
    "ops": ["infra"],
    "synthesis": ["eng"],
}


def _domain_write_acl(cfg: dict | None) -> dict[str, list[str]]:
    """The domain-to-labels map for authored notes: the defaults, then config's ``authored_acl``.

    An entry that is not a list of strings is ignored, so its domain keeps the
    default, or ``secrets`` when it has none.
    """
    mapping = {domain: list(acl) for domain, acl in _DOMAIN_WRITE_ACL.items()}
    configured = cfg.get("authored_acl") if isinstance(cfg, dict) else None
    if isinstance(configured, dict):
        for domain, acl in configured.items():
            if isinstance(acl, list) and all(isinstance(label, str) for label in acl):
                mapping[str(domain)] = list(acl)
    return mapping


def _authored_acl(domain: str, text: str = "", cfg: dict | None = None) -> list[str]:
    """Fail-closed ACL for direct human/file-back writes."""
    acl = list(_domain_write_acl(cfg).get(domain, ["secrets"]))
    try:
        from remembrancerlib.acl import restrict_sensitive_acl

        return restrict_sensitive_acl(text, acl)
    except Exception:
        return acl


def _writer_authorized(identity: str | None, note_acl: list[str], root: str) -> bool:
    """Require a server-mapped writer and an applicable destination grant."""
    actor_grants = grants_for(identity, root)
    if actor_grants is FULL_ACCESS:
        return True
    if not actor_grants:
        return False
    return not note_acl or bool(set(note_acl) & set(actor_grants))


def _append_audit(root: str, record: dict) -> None:
    """Append one JSON line to state/agent_edits_audit.jsonl. Best-effort:
    an audit failure must never prevent the note write from succeeding."""
    try:
        path = os.path.join(root, "state", "agent_edits_audit.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def can_promote(identity, note, cfg) -> bool:
    """Only explicitly configured human approvers may change trust state."""
    allowed = (cfg.get("trust") or {}).get("approver_identities", [])
    return (
        bool(identity)
        and isinstance(allowed, list)
        and all(isinstance(value, str) for value in allowed)
        and identity in set(allowed)
    )


def _supersession_validation_error(note: Note, notes: list[Note], grants) -> str:
    """Return why activating ``note``'s supersession edges is unsafe.

    Promotion is the lifecycle commit point.  Validate the complete graph while
    holding the shared pipeline lock so a verified head can never hide a
    missing, historical, invisible, ACL-incompatible, or cyclic target.
    """
    edges = list(
        dict.fromkeys(
            target for relation, target in note.typed_links if relation == "supersedes"
        )
    )
    if not edges:
        return ""

    by_name: dict[str, list[Note]] = {}
    for candidate in notes:
        by_name.setdefault(candidate.name, []).append(candidate)
    current = current_note_names(notes)
    if note.name not in current:
        return "supersession head is not current"

    for target_name in edges:
        if not target_name or target_name == note.name:
            return "supersession target must be a different note"
        targets = by_name.get(target_name, [])
        if len(targets) != 1:
            return f"supersession target {target_name!r} is missing or ambiguous"
        target = targets[0]
        if target_name not in current:
            return f"supersession target {target_name!r} is not current"
        if not visible(target, grants):
            return f"not permitted to supersede target {target_name!r}"
        if not acl_can_reveal(list(note.acl or []), list(target.acl or [])):
            return f"replacement ACL could reveal target {target_name!r}"

    # Active verified edges plus the candidate edge(s).  A path back to the
    # candidate would make current truth depend on traversal order.
    graph: dict[str, set[str]] = {}
    for candidate in notes:
        if candidate.name != note.name and effective_state(candidate) != "verified":
            continue
        if effective_state(candidate) == "superseded":
            continue
        graph.setdefault(candidate.name, set()).update(
            target
            for relation, target in candidate.typed_links
            if relation == "supersedes" and target in by_name
        )
    graph[note.name] = set(edges)
    frontier = list(edges)
    seen: set[str] = set()
    while frontier:
        cursor = frontier.pop()
        if cursor == note.name:
            return "supersession would create a cycle"
        if cursor in seen:
            continue
        seen.add(cursor)
        frontier.extend(graph.get(cursor, ()))
    return ""


@serialized_pipeline_mutation()
def _transition(
    name: str,
    target_state: str,
    action: str,
    identity: str | None,
    reason: str,
    root: str,
    *,
    expected_content_fingerprint: str | None = None,
    expected_existing_sha256: str | None = None,
) -> dict:
    """Shared promote/demote/reject: find → ACL-check → policy-check → set_state → write → audit."""
    from remembrancerlib import effective_state as _es
    from remembrancerlib.note import set_state as _set_state

    root = os.path.abspath(root)
    cfg = _load_config(root)
    context_dir = os.path.join(root, "context")
    notes = load_notes(context_dir)
    matches = [candidate for candidate in notes if candidate.name == name]
    if not matches:
        return {"ok": False, "reason": f"note {name!r} not found or suppressed"}
    if len(matches) != 1:
        return {"ok": False, "reason": f"note name {name!r} is ambiguous"}
    note = matches[0]
    try:
        with open(note.path, "rb") as current_file:
            existing_bytes = current_file.read()
    except OSError:
        return {"ok": False, "reason": "cannot read the current note preimage"}
    import hashlib as _hashlib

    current_existing_sha256 = _hashlib.sha256(existing_bytes).hexdigest()
    from remembrancerlib.verification import verification_content_fingerprint

    current_content_fingerprint = verification_content_fingerprint(note)
    supplied = (
        expected_content_fingerprint is not None or expected_existing_sha256 is not None
    )
    if supplied:
        # Attended approval is consent to the bytes rendered in the inbox, not
        # to whichever same-name note happens to exist at submit time.
        supplied_preimage = expected_existing_sha256
        if (
            not isinstance(expected_content_fingerprint, str)
            or not isinstance(supplied_preimage, str)
            or expected_content_fingerprint != current_content_fingerprint
            or supplied_preimage != current_existing_sha256
        ):
            return {
                "ok": False,
                "reason": "approval target changed after review; reload and review again",
                "stale": True,
            }
    if not visible(note, grants_for(identity, root)):
        return {"ok": False, "reason": "not permitted to read this note"}
    if not can_promote(identity, note, cfg):
        return {"ok": False, "reason": "not permitted to change trust state"}
    from remembrancerlib import forward_link_validation_error

    link_error = forward_link_validation_error(note, notes)
    if link_error:
        # Refuse before appending the signed lifecycle row.  The write door
        # repeats this check as defense in depth for every non-human producer.
        return {"ok": False, "reason": link_error}
    prev = _es(note)
    raw_prev = (
        note.state
        if note.state in {"proposed", "verified", "superseded"}
        else "proposed"
    )
    from remembrancerlib import load_human_trust_audit

    trust_audit = load_human_trust_audit(root)
    pending = trust_audit.latest.get(note.name) if trust_audit.trustworthy else None
    pending = (
        pending
        if (
            isinstance(pending, dict)
            and pending.get("approval_version") == 2
            and pending.get("action") == action
            and pending.get("identity") == identity
            and pending.get("from") == raw_prev
            and pending.get("to") == target_state
            and pending.get("expected_existing_sha256") == current_existing_sha256
            and (not reason or pending.get("reason") == reason)
        )
        else None
    )
    allowed_transitions = {
        ("proposed", "verified"),
        ("verified", "proposed"),
        ("proposed", "superseded"),
    }
    if pending is None and (prev, target_state) not in allowed_transitions:
        return {
            "ok": False,
            "reason": f"invalid trust transition {prev}->{target_state}",
        }
    if target_state == "verified":
        error = _supersession_validation_error(note, notes, grants_for(identity, root))
        if error:
            return {"ok": False, "reason": error}
    _set_state(note, target_state)
    note.superseded = target_state == "superseded"
    stamp = f"{action}-by:{identity}"
    if stamp not in note.provenance:
        note.provenance = list(note.provenance) + [stamp]
    if action == "demote":
        from remembrancerlib import invalidate_verification

        invalidate_verification(note, reason="human-demoted-for-reverification")
    # The append-only transition record is fsync'd first. A crash before the
    # atomic note write leaves only an orphan row, which cannot grant trust;
    # the inverse ordering could expose unjournaled verified bytes.
    capability = _record_human_trust_transition(
        note,
        root=root,
        action=action,
        identity=identity or "",
        previous_state=raw_prev if pending is not None else prev,
        target_state=target_state,
        expected_existing_sha256=current_existing_sha256,
        reason=(pending.get("reason", "") if pending is not None else reason),
    )
    write_note(note, context_dir, _trust_capability=capability)
    return {"ok": True, "note": name, "from": prev, "state": target_state}


def promote(
    name: str,
    identity: str | None = None,
    reason: str = "",
    root: str = ".",
    *,
    expected_content_fingerprint: str | None = None,
    expected_existing_sha256: str | None = None,
) -> dict:
    """Promote a note to `verified` (canonical). ACL-checked, policy-gated, audited."""
    return _transition(
        name,
        "verified",
        "promote",
        identity,
        reason,
        root,
        expected_content_fingerprint=expected_content_fingerprint,
        expected_existing_sha256=expected_existing_sha256,
    )


def demote(
    name: str, identity: str | None = None, reason: str = "", root: str = "."
) -> dict:
    """Demote a note back to `proposed` (needs re-verification)."""
    return _transition(name, "proposed", "demote", identity, reason, root)


def reject(
    name: str,
    identity: str | None = None,
    reason: str = "",
    root: str = ".",
    *,
    expected_content_fingerprint: str | None = None,
    expected_existing_sha256: str | None = None,
) -> dict:
    """Reject a proposed note → `superseded` (retrieval-demoted; kept for history)."""
    return _transition(
        name,
        "superseded",
        "reject",
        identity,
        reason,
        root,
        expected_content_fingerprint=expected_content_fingerprint,
        expected_existing_sha256=expected_existing_sha256,
    )


@serialized_pipeline_mutation()
def remember(
    fact: str,
    supersedes: str | None = None,
    entities: list[str] | None = None,
    domain: str | None = None,
    identity: str | None = None,
    source: str | None = None,
    root: str = ".",
) -> str:
    """Persist a human-asserted memory as a local note (no git).

    Two modes:
      * ``supersedes=None`` — add a net-new ``type: synthesis`` note.
      * ``supersedes=<name>`` — propose a replacement: write a proposed
        ``type: correction`` note carrying ``[[supersedes:<name>]]``. It hides
        the old note only after an authorized promotion to ``verified``;
        history remains available through ``include_history=True``.

    The superseding note is given a DISTINCT slug (``<base>-<8hexhash>``) so
    ``write_note`` never overwrites the note it supersedes. Returns the absolute
    path of the written note, or ``""`` when gated:
      * ``retrieval.file_back.enabled`` is ``false``, or
      * ``len(fact) < retrieval.file_back.min_chars``.

    Every successful write appends one line to
    ``state/agent_edits_audit.jsonl`` (best-effort; never fatal).

    Args:
        fact:       The memory / correction body to persist.
        supersedes: Name of the note this replaces; ``None`` for net-new.
        entities:   Optional entity tags, e.g. ``["contact:jane-doe"]``.
        domain:     Domain override; defaults to the superseded note's domain
                    (or ``"company"`` for net-new).
        identity:   The server-VERIFIED caller (from the authenticated edge
                    login or a configured pin); stamped as ``by:<identity>`` in provenance.
        source:     Optional caller-DECLARED origin of the fact, e.g.
                    ``"slack:@jdoe"``, ``"email:x@y.com"``, ``"call:2026-01-15"``.
                    Unlike ``identity`` this is self-reported by the calling
                    agent (not server-verified); stamped as ``source:<source>``.
        root:       Repository root; ``config.json`` and ``context/`` resolve
                    relative to it.

    Provenance shape: ``["by:<identity>", "source:<source>"?, "at:<date>"]`` —
    ``source`` omitted when not given.
    """
    import hashlib

    root = os.path.abspath(root)
    cfg = _load_config(root)
    fb = _file_back_cfg(cfg)

    if not fb["enabled"]:
        return ""
    if len(fact) < fb["min_chars"]:
        return ""

    context_dir = os.path.join(root, "context")
    import govern

    route_meta = govern.RouteMeta(
        requester=identity or "",
        source=source,
        supersedes=supersedes,
    )
    actor_grants = grants_for(identity, root)
    if actor_grants is not FULL_ACCESS and not actor_grants:
        return ""

    all_notes = load_notes(context_dir)
    target = None
    if supersedes:
        matches = [note for note in all_notes if note.name == supersedes]
        if len(matches) != 1:
            return ""
        target = matches[0]
        if not visible(target, grants_for(identity, root)):
            return ""
        if target.name not in current_note_names(all_notes):
            return ""

    # Domain: explicit > superseded note's domain > "company".
    if domain:
        note_domain = domain
    elif target is not None:
        note_domain = target.domain
    else:
        note_domain = "company"

    note_type = "correction" if supersedes else "synthesis"

    try:
        from remembrancerlib.note import _today as _note_today

        today = _note_today()
    except Exception:
        import datetime

        today = datetime.date.today().isoformat()
    # Labeled provenance: server-verified caller (by:), optional caller-declared
    # origin (source:), and the write date (at:).
    provenance = [f"by:{identity or 'unknown'}"]
    if source:
        provenance.append(f"source:{source}")
    provenance.append(f"at:{today}")

    body = fact
    if supersedes:
        body = body.rstrip() + f"\n\nLinks: [[supersedes:{supersedes}]]\n"

    # Distinct slug: never collide with the superseded note's file (else
    # write_note would OVERWRITE the original).
    h = hashlib.sha1(fact.encode()).hexdigest()[:8]
    base = slugify(supersedes) if supersedes else slugify(fact)[:48]
    name = f"{base}-{h}"
    if supersedes:
        assert slugify(name) != slugify(
            supersedes
        ), "superseding note slug must differ from the superseded note's slug"

    if target is not None:
        note_acl = list(target.acl)
        note_entities = (
            list(entities) if entities is not None else list(target.entities)
        )
    else:
        note_acl = _authored_acl(note_domain, fact, cfg)
        note_entities = list(entities or [])
    if not _writer_authorized(identity, note_acl, root):
        return ""

    if target is not None:
        # Do not create competing correction heads.  An exact proposed replay
        # is idempotent; every materially different correction must chain from
        # the existing head after it is selected/approved.
        for candidate in all_notes:
            if effective_state(candidate) == "superseded":
                continue
            if ("supersedes", target.name) not in candidate.typed_links:
                continue
            if (
                candidate.state == "proposed"
                and candidate.body == body
                and list(candidate.acl or []) == note_acl
                and f"by:{identity or 'unknown'}" in (candidate.provenance or [])
            ):
                govern.route(candidate, route_meta, root)
                return candidate.path
            return ""

    same_name = [candidate for candidate in all_notes if candidate.name == name]
    if same_name:
        prior = same_name[0] if len(same_name) == 1 else None
        if (
            prior is not None
            and prior.type == note_type
            and prior.body == body
            and list(prior.acl or []) == note_acl
            and list(prior.entities or []) == note_entities
            and f"by:{identity or 'unknown'}" in (prior.provenance or [])
        ):
            govern.route(prior, route_meta, root)
            return prior.path
        base_name = name
        suffix = 2
        while any(candidate.name == name for candidate in all_notes):
            name = f"{base_name}-{suffix}"
            suffix += 1

    note = Note(
        name=name,
        description=fact[:120],
        type=note_type,
        domain=note_domain,
        acl=note_acl,
        provenance=provenance,
        entities=note_entities,
        body=body,
        state="proposed",
    )
    govern.prepare(note, route_meta, root)
    path = write_note(note, context_dir)

    from datetime import datetime, timezone

    _append_audit(
        root,
        {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "identity": identity,
            "source": source,
            "supersedes": supersedes,
            "fact": fact,
            "path": path,
        },
    )
    # Queue publication is part of the write contract.  If it fails, propagate
    # the error; ``prepare`` already made approval-required bytes unanswerable,
    # so retry can safely finish the durable queue without a visibility gap.
    govern.route(note, route_meta, root)
    return path


@serialized_pipeline_mutation()
def create_page(
    title: str,
    body: str,
    domain: str | None = None,
    identity: str | None = None,
    source: str | None = None,
    entities: list[str] | None = None,
    root: str = ".",
) -> str:
    """Author a first-class wiki PAGE (type: page) — a human-written document (runbook,
    onboarding, architecture note), NOT an event-derived note. Distinct from remember(): the
    caller supplies a real human TITLE (so it never renders as a slug), the slug is derived from
    that title (hash-suffixed only on collision, never overwriting an existing note), and the type
    is `page`. Gated by retrieval.file_back.enabled and a small body minimum. Provenance is stamped
    [by:, source:?, at:]; the write is audited and routed through governance like any other.
    Returns the written path, or "" when gated (disabled / no title / body too short)."""
    root = os.path.abspath(root)
    cfg = _load_config(root)
    fb = _file_back_cfg(cfg)
    if not fb["enabled"]:
        return ""
    title = (title or "").strip()
    body = body or ""
    if not title or len(body.strip()) < 40:  # a page needs a title and a real body
        return ""

    context_dir = os.path.join(root, "context")
    import govern

    route_meta = govern.RouteMeta(requester=identity or "", source=source)
    note_domain = domain or "company"
    from remembrancerlib.note import VALID_DOMAINS

    if note_domain not in VALID_DOMAINS:
        return ""
    note_acl = _authored_acl(note_domain, body, cfg)
    if not _writer_authorized(identity, note_acl, root):
        return ""

    try:
        from remembrancerlib.note import _today as _note_today

        today = _note_today()
    except Exception:
        import datetime

        today = datetime.date.today().isoformat()
    provenance = [f"by:{identity or 'unknown'}"]
    if source:
        provenance.append(f"source:{source}")
    provenance.append(f"at:{today}")

    # Stable, human-derived slug; exact retries are idempotent and a changed
    # page gets a hash suffix. Names are corpus-global ids, not merely paths.
    import hashlib

    base = slugify(title)[:64] or "page"
    name = base
    existing = [note for note in load_notes(context_dir) if note.name == name]
    if len(existing) == 1:
        prior = existing[0]
        if (
            prior.type == "page"
            and prior.title == title
            and prior.domain == note_domain
            and list(prior.acl or []) == note_acl
            and prior.body.rstrip() == body.rstrip()
        ):
            govern.route(prior, route_meta, root)
            return prior.path
    if existing:
        name = f"{base}-{hashlib.sha1((title + body).encode()).hexdigest()[:6]}"
        hashed = [note for note in load_notes(context_dir) if note.name == name]
        if len(hashed) == 1:
            prior = hashed[0]
            if (
                prior.type == "page"
                and prior.title == title
                and prior.domain == note_domain
                and list(prior.acl or []) == note_acl
                and prior.body.rstrip() == body.rstrip()
            ):
                govern.route(prior, route_meta, root)
                return prior.path
        if hashed:
            raise ValueError(f"page name {name!r} already exists")

    note = Note(
        name=name,
        title=title,  # a REAL human title (display_title uses it)
        description=(body.strip().splitlines()[0] if body.strip() else title)[:160],
        type="page",
        domain=note_domain,
        acl=note_acl,
        provenance=provenance,
        entities=list(entities or []),
        body=body,
        state="proposed",
    )
    govern.prepare(note, route_meta, root)
    path = write_note(note, context_dir)

    from datetime import datetime, timezone

    _append_audit(
        root,
        {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "identity": identity,
            "source": source,
            "supersedes": None,
            "fact": f"[page] {title}",
            "path": path,
        },
    )
    govern.route(note, route_meta, root)
    return path


def main() -> None:
    # The CLI is a LOCAL, shell-privileged operator tool (SSH into the server) — the threat model is
    # the network edge, not the local shell — so it opts into full-access ergonomics. Serving
    # paths (MCP/UI) never set this, so they default-deny.
    os.environ.setdefault("REMEMBRANCER_ALLOW_FULL_ACCESS", "1")
    ap = argparse.ArgumentParser(
        description="Remembrancer retrieval (index-first, ACL-gated, BM25)."
    )
    ap.add_argument(
        "--as",
        dest="identity",
        default=None,
        help="acting identity (config.json); omit for full-access (local CLI only)",
    )
    ap.add_argument("--q", required=True, help="query string")
    ap.add_argument(
        "--k", type=int, default=None, help="candidate pool size (default: config)"
    )
    ap.add_argument(
        "--entity", default=None, help="scope to one entity, e.g. customer:acme"
    )
    ap.add_argument(
        "--current",
        action="store_true",
        default=False,
        help="exclude notes that are the target of a 'supersedes' edge (show only current notes)",
    )
    ap.add_argument(
        "--history",
        action="store_true",
        default=False,
        help="include superseded history for this request (rank-demoted)",
    )
    ap.add_argument(
        "--insist",
        action="store_true",
        default=False,
        help="user-insist escalation: dig the raw events (L1) instead of the wiki, "
        "returning grounded candidates to confirm with the user",
    )
    args = ap.parse_args()

    cfg = _load_config(ROOT)
    rcfg = _retrieval_cfg(cfg)
    deliver_max = rcfg["deliver_max_notes"]
    grants = grants_for(args.identity, ROOT)

    entries = search(
        args.q,
        args.identity,
        k=args.k,
        entity=args.entity,
        current_only=args.current,
        include_history=args.history,
        root=ROOT,
    )

    scope = f" entity={args.entity}" if args.entity else ""
    current_tag = (
        " [--history]" if args.history else " [--current]" if args.current else ""
    )
    grants_display = (
        "ALL (full access)"
        if grants is FULL_ACCESS
        else "DENY (public-only)"
        if not grants
        else str(sorted(grants))
    )
    print(
        f"identity={args.identity} grants={grants_display}{scope}{current_tag} · "
        f"q={args.q!r} · {len(entries)} candidates (pool, recall-wide)"
    )

    # --insist: explicit user-driven escalation to L1 raw events. Wiki-first by default;
    # this only runs when the caller asserts the wiki answer is missing/insufficient.
    if getattr(args, "insist", False):
        dig = dig_events(args.q, grants, cfg, events_dir=os.path.join(ROOT, "events"))
        print(
            "\n⤷ INSIST — L1 raw-event dig (grounded candidates, UNVERIFIED — confirm "
            "with the user before trusting):"
        )
        if not dig:
            print("   (nothing in raw events either — honest miss)")
        for h in dig:
            print(f"   • {h['id']}  [{h['date']}]  (bm25={h['score']})  {h['snippet']}")
        print(
            '\n   → Ask the user: "does this answer your question?" — on yes, write it '
            "back to the wiki (tools/fallback_queue.log_candidate)."
        )
        return

    if not entries:
        print("(no matches)")
        return

    for i, e in enumerate(entries):
        marker = "★ DELIVER" if i < deliver_max else "  pool   "
        ents = f" · entities:{e['entities']}" if e["entities"] else ""
        fresh = e.get("freshness")
        fresh_tag = ""
        if fresh:
            fresh_tag = f" · {fresh['tier']}"
            if fresh["evidence_as_of"]:
                fresh_tag += f" (as of {fresh['evidence_as_of']}, {fresh['anchor']})"
            # A settled note is past its window because it is FINISHED, not
            # because it is out of date (retrieve/freshness.py).
            if fresh.get("settled"):
                fresh_tag += " · settled"
                if fresh.get("successor"):
                    fresh_tag += f" → {fresh['successor']}"
        print(f"\n{marker}  [{i + 1:>2}] {e['name']}  (bm25={e['score']}){fresh_tag}")
        print(f"            acl:{e['acl']}{ents}")
        print(f"            provenance:{e['provenance']}")
        print(f"            {e['description']}")
        print(f"            [{e['path']}]")

    n_deliver = min(deliver_max, len(entries))
    print(
        f"\n→ precision set: open the top {n_deliver} (★) via get_note(); "
        f"the rest stay index-only (progressive disclosure)."
    )


if __name__ == "__main__":
    main()
