"""Fail-closed ACL proof helpers for the materialized entity graph.

The graph is a derived cache.  Consumers must not trust an edge's declared
``acl`` merely because it came from ``state/entity_graph.json``: every edge is
re-derived from its endpoint ACLs and, for evidence-backed claims, the current
authenticated evidence corpus.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .acl import ACLConflict, combine_acl
from .dossier import substantive_exact_quote
from .registry_trust import registry_edge_proofs_valid


GRAPH_SCHEMA_VERSION = 3
PROOF_EVIDENCE = "evidence"
PROOF_REGISTRY = "registry"

_RELATION_CUES = {
    "works_at": (
        "works at", "work at", "worked at", "works for", "work for",
        "worked for", "joined", "employed by", "serves as", "is ceo",
        "is cto", "is cfo", "is founder", "is head", "任职", "就职", "加入", "工作于",
    ),
    "introduced": (
        "introduced", "introduction", "referred", "connected", "intro by",
        "介绍", "引荐", "牵线",
    ),
    "champion_of": (
        "champion", "advocate", "sponsor", "champions", "支持者", "倡导", "推动",
    ),
    "founder_of": (
        "founded", "founder", "co founded", "co founder", "started",
        "创立", "创始", "联合创始",
    ),
    "invested_in": (
        "invested in", "investor in", "investment in", "backed", "funded",
        "投资", "出资", "领投",
    ),
    "partner_of": (
        "partners with", "partner of", "partnership with", "合作伙伴", "合作",
    ),
}


def _semantic_text(value: Any) -> str:
    return re.sub(r"[^\w]+", " ", str(value or "").casefold()).strip()


def relation_quote_supports(
    edge_type: str,
    quote: str,
    src_labels: Iterable[str],
    dst_labels: Iterable[str],
) -> bool:
    """Conservatively require endpoints + a type cue in one quoted clause.

    Exact substring validation proves provenance but not meaning: an exact
    quote such as "Alice met Acme" must not support ``works_at``.  This small
    deterministic semantic gate intentionally prefers false negatives over a
    relationship that the cited sentence does not state.
    """
    cues = _RELATION_CUES.get(edge_type)
    if not cues or substantive_exact_quote(quote, quote) is None:
        return False
    src = {_semantic_text(value) for value in src_labels if _semantic_text(value)}
    dst = {_semantic_text(value) for value in dst_labels if _semantic_text(value)}
    if not src or not dst:
        return False
    clauses = re.split(r"[.!?;\n。！？；]+", quote)
    normalized_cues = tuple(_semantic_text(cue) for cue in cues)
    for raw_clause in clauses:
        clause = _semantic_text(raw_clause)
        padded = f" {clause} "
        if (clause and any(f" {label} " in padded for label in src)
                and any(f" {label} " in padded for label in dst)
                and any(f" {cue} " in padded for cue in normalized_cues)):
            return True
    return False


def edge_quotes_grounded_against_notes(
    edge: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    notes_by_name: Mapping[str, Any],
) -> bool:
    """Re-prove an extracted edge against current non-generated source notes."""
    if edge.get("source") != "extracted":
        return True
    proofs = normalize_edge_evidence_quotes(edge)
    if proofs is None:
        return False
    src_node = nodes_by_id.get(edge.get("src")) or {}
    dst_node = nodes_by_id.get(edge.get("dst")) or {}
    src_labels = [edge.get("src"), src_node.get("label"), *(src_node.get("aliases") or [])]
    dst_labels = [edge.get("dst"), dst_node.get("label"), *(dst_node.get("aliases") or [])]
    for proof in proofs:
        note = notes_by_name.get(proof["source_note"])
        if (note is None or getattr(note, "name", None) != proof["source_note"]
                or any(str(value).startswith("generated-by:")
                       for value in (getattr(note, "provenance", None) or []))):
            return False
        ids = [f"note:{note.name}"]
        ids.extend(
            str(value).strip()
            for value in (getattr(note, "provenance", None) or [])
            if str(value).strip()
            and not str(value).startswith("by:")
            and not str(value).startswith("at:")
        )
        excerpt = (
            f"{getattr(note, 'description', '') or ''}\n"
            f"{str(getattr(note, 'body', '') or '')[:1200]}"
        ).strip()
        if (proof["evidence_id"] not in ids
                or substantive_exact_quote(proof["quote"], excerpt) != proof["quote"]
                or not relation_quote_supports(
                    str(edge.get("type") or ""), proof["quote"], src_labels, dst_labels,
                )):
            return False
    return True


def normalize_edge_evidence_quotes(edge: dict[str, Any]) -> list[dict[str, str]] | None:
    """Validate the durable quote-proof envelope on an extracted edge.

    Exact membership in the prompt excerpt is deliberately checked by the
    producer and again by the query consumer, both of which have the source
    notes.  This shared structural check keeps malformed or incomplete proof
    out of the graph before either side consults content.
    """
    if not isinstance(edge, dict):
        return None
    evidence = edge.get("evidence")
    quotes = edge.get("evidence_quotes")
    if not isinstance(evidence, list) or not evidence or not isinstance(quotes, list):
        return None
    evidence_ids: list[str] = []
    for raw in evidence:
        if not isinstance(raw, str) or not raw.strip():
            return None
        evidence_ids.append(raw.strip())
    if len(evidence_ids) != len(set(evidence_ids)):
        return None
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in quotes:
        if not isinstance(raw, dict):
            return None
        evidence_id = raw.get("evidence_id")
        source_note = raw.get("source_note")
        quote = raw.get("quote")
        if (not isinstance(evidence_id, str) or evidence_id not in evidence_ids
                or evidence_id in seen or not isinstance(source_note, str)
                or not source_note.strip() or not isinstance(quote, str)
                or substantive_exact_quote(quote, quote) is None):
            return None
        seen.add(evidence_id)
        normalized.append({
            "evidence_id": evidence_id,
            "source_note": source_note.strip(),
            "quote": quote.strip(),
        })
    if seen != set(evidence_ids):
        return None
    return sorted(normalized, key=lambda item: evidence_ids.index(item["evidence_id"]))


def normalize_acl(value: Any) -> list[str] | None:
    """Return a canonical explicit ACL, or ``None`` for malformed metadata."""
    if not isinstance(value, list):
        return None
    labels: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw.strip():
            return None
        labels.append(raw.strip())
    return sorted(set(labels))


def graph_evidence_catalog(notes: Iterable[Any], events: Iterable[Any]) -> dict[str, list[str]]:
    """Build an unambiguous evidence-id -> ACL map.

    Callers are responsible for supplying only unsuppressed/current notes and
    corpus-guard-authenticated events.  Duplicate ids/names are removed from
    the result so they cannot be selected nondeterministically.
    """
    catalog: dict[str, list[str]] = {}
    ambiguous: set[str] = set()

    def add(key: str, acl: Any) -> None:
        key = str(key or "").strip()
        normalized = normalize_acl(list(acl or []))
        if not key or normalized is None or key in ambiguous:
            return
        if key in catalog:
            catalog.pop(key, None)
            ambiguous.add(key)
            return
        catalog[key] = normalized

    for note in notes:
        add(f"note:{getattr(note, 'name', '')}", getattr(note, "acl", []))
    for event in events:
        if isinstance(event, dict):
            add(event.get("id", ""), event.get("acl", []))
        else:
            add(getattr(event, "id", ""), getattr(event, "acl", []))
    return catalog


def derive_edge_acl(
    edge: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    evidence_catalog: dict[str, list[str]],
) -> tuple[list[str], list[list[str]]] | None:
    """Verify an edge's structural proof and derive its only safe ACL.

    ``registry`` proof is allowed only for curated/deterministic edges with no
    evidence; endpoint ACLs are its authority.  ``evidence`` proof requires a
    non-empty, unique list whose every id resolves in the supplied current
    corpus.  Incompatible OR-grant scopes are unrepresentable and fail closed.
    """
    if not isinstance(edge, dict):
        return None
    src = edge.get("src")
    dst = edge.get("dst")
    if not isinstance(src, str) or not src or not isinstance(dst, str) or not dst:
        return None
    src_node = nodes_by_id.get(src)
    dst_node = nodes_by_id.get(dst)
    if not isinstance(src_node, dict) or not isinstance(dst_node, dict):
        return None
    src_acl = normalize_acl(src_node.get("acl"))
    dst_acl = normalize_acl(dst_node.get("acl"))
    if src_acl is None or dst_acl is None:
        return None

    evidence = edge.get("evidence")
    if not isinstance(evidence, list):
        return None
    evidence_ids: list[str] = []
    for raw in evidence:
        if not isinstance(raw, str) or not raw.strip():
            return None
        evidence_ids.append(raw.strip())
    if len(evidence_ids) != len(set(evidence_ids)):
        return None

    proof = edge.get("proof")
    source = edge.get("source")
    evidence_acls: list[list[str]] = []
    if proof == PROOF_REGISTRY:
        if (evidence_ids or source not in {"curated", "derived"}
                or not registry_edge_proofs_valid(edge)):
            return None
    elif proof == PROOF_EVIDENCE:
        if not evidence_ids or source not in {"extracted", "derived"}:
            return None
        if source == "extracted" and normalize_edge_evidence_quotes(edge) is None:
            return None
        for evidence_id in evidence_ids:
            acl = evidence_catalog.get(evidence_id)
            if acl is None:
                return None
            evidence_acls.append(acl)
    else:
        return None

    try:
        derived = combine_acl(src_acl, dst_acl, *evidence_acls)
    except ACLConflict:
        return None
    return normalize_acl(derived) or [], evidence_acls


def edge_acl_is_proven(
    edge: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    evidence_catalog: dict[str, list[str]],
) -> tuple[list[str], list[list[str]]] | None:
    """Return proof details only when the declared edge ACL is exact."""
    declared = normalize_acl(edge.get("acl"))
    if declared is None:
        return None
    derived = derive_edge_acl(edge, nodes_by_id, evidence_catalog)
    if derived is None or derived[0] != declared:
        return None
    return derived
