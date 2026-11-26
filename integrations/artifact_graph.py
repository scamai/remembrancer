"""Read-side of the artifact lifecycle graph (``state/artifact_graph.json``).

The graph is built outside the engine; this module only
reads the JSON. Consumers: ``remembrancer_graph("artifact:<slug>" | "topic:<id>")``
(via the entity-graph merge), ``remembrancer_search_artifacts`` rows
(``topic`` / ``superseded_by`` / ``latest_in_topic``) and a topics listing page.

A missing or malformed graph is never an error — readers get the empty
structure and every lookup answers "nothing known".
"""
from __future__ import annotations

import json
import os
from typing import Any

# One authority for the edge vocabulary: the ledger module defines it (it is
# what a relation decision may name), and readers import it from here so they
# never keep a second copy that drifts.
from remembrancerlib.artifact_relations import RELATION_TYPES  # noqa: F401
from remembrancerlib.artifact_relations import fold as fold_relations
from remembrancerlib.artifact_relations import load_decisions as load_relation_decisions

GRAPH_VERSION = 1
GRAPH_REL_PATH = os.path.join("state", "artifact_graph.json")
ARTIFACT_PREFIX = "artifact:"
# The rule name the builder stamps on a succession the PUBLISHER declared
# (`supersedes=` at publish time). It is the one relation the ledger cannot
# undo, so the read-side fold shields it exactly as the builder does.
DECLARED_RULE = "manifest_declared"
STATUS_ACCEPTED = "accepted"


def _empty() -> dict[str, Any]:
    return {"version": GRAPH_VERSION, "computed_at": None, "nodes": {},
            "edges": [], "topics": {}, "proposals": []}


def _fold_ledger(root: str, graph: dict[str, Any]) -> dict[str, Any]:
    """Apply the relation ledger to a freshly-loaded graph, in place.

    The builder folds the ledger when it runs; folding again here means a
    retraction lands on the NEXT read rather than at the next build —
    the property that makes "negate this decision" feel immediate. Node
    ``superseded_by`` is recomputed from the folded edges by the same rule the
    builder uses, so a ledger assertion flips it exactly like a structural one.

    Best-effort by the same contract as the load itself: any failure leaves the
    graph as it was on disk rather than raising.
    """
    try:
        decisions = load_relation_decisions(root)
        if not decisions:
            return graph
        protected = frozenset(
            (e.get("type"), e.get("from"), e.get("to")) for e in graph["edges"]
            if e.get("rule") == DECLARED_RULE)
        edges = fold_relations(graph["edges"], decisions, protected=protected)
    except Exception:
        return graph
    graph["edges"] = edges
    accepted: dict[str, set[str]] = {}
    for edge in edges:
        if edge.get("type") != "supersedes" or edge.get("status") != STATUS_ACCEPTED:
            continue
        src = str(edge.get("from") or "")
        if src.startswith(ARTIFACT_PREFIX):
            accepted.setdefault(str(edge.get("to")), set()).add(
                src[len(ARTIFACT_PREFIX):])
    for node_id, node in graph["nodes"].items():
        node["superseded_by"] = sorted(accepted.get(node_id, ()))
    return graph


def load_graph(root: str, *, fold_ledger: bool = True) -> dict[str, Any]:
    """Load the graph under ``root``; missing/malformed -> empty structure.

    ``fold_ledger`` (default on) applies ``state/artifact_relations.jsonl`` over
    the stored edges at query time — pass ``False`` to see the file exactly as
    the last build wrote it.
    """
    path = os.path.join(root, GRAPH_REL_PATH)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return _empty()
    if (not isinstance(data, dict) or not isinstance(data.get("nodes"), dict)
            or not isinstance(data.get("edges"), list)):
        return _empty()
    graph = {
        "version": data.get("version", GRAPH_VERSION),
        "computed_at": data.get("computed_at"),
        "nodes": {k: v for k, v in data["nodes"].items() if isinstance(v, dict)},
        "edges": [e for e in data["edges"]
                  if isinstance(e, dict) and e.get("from") and e.get("to")],
        "topics": {k: v for k, v in (data.get("topics") or {}).items()
                   if isinstance(v, dict)} if isinstance(data.get("topics"), dict) else {},
        "proposals": [p for p in (data.get("proposals") or []) if isinstance(p, dict)]
        if isinstance(data.get("proposals"), list) else [],
    }
    # An empty graph answers "nothing known" — folding a ledger over it would
    # invent edges between pages the graph has never heard of.
    if fold_ledger and graph["nodes"]:
        graph = _fold_ledger(root, graph)
    return graph


def _artifact_id(slug_or_id: str) -> str:
    text = str(slug_or_id or "").strip()
    return text if text.startswith(ARTIFACT_PREFIX) else ARTIFACT_PREFIX + text


def node(graph: dict[str, Any], slug: str) -> dict[str, Any] | None:
    """The ``artifact:<slug>`` node (``slug`` may carry the prefix), or None."""
    return (graph.get("nodes") or {}).get(_artifact_id(slug))


def topic_of(graph: dict[str, Any], slug: str) -> str | None:
    n = node(graph, slug)
    return n.get("topic") if n else None


def superseded_by(graph: dict[str, Any], slug: str) -> list[str]:
    """Slugs of accepted successors of ``slug`` (empty when it is current)."""
    n = node(graph, slug)
    return list(n.get("superseded_by") or []) if n else []


def _stub(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    """A node dict for any id: artifact nodes as stored, topics from the
    topics table, foreign ids (person/org/system/component) as stubs."""
    found = (graph.get("nodes") or {}).get(node_id)
    if found is not None:
        return found
    topic = (graph.get("topics") or {}).get(node_id)
    if topic is not None:
        return {"id": node_id, "kind": "topic",
                "label": topic.get("label") or node_id[len("topic:"):],
                "pages": list(topic.get("pages") or []), "latest": topic.get("latest"),
                "tags": list(topic.get("tags") or [])}
    kind = node_id.split(":", 1)[0] if ":" in node_id else "unknown"
    return {"id": node_id, "kind": kind, "label": node_id.split(":", 1)[-1]}


def neighbors(graph: dict[str, Any], node_id: str, depth: int = 1,
              types=None) -> dict[str, Any]:
    """BFS from ``node_id`` (``artifact:<slug>``, bare slug, or ``topic:<id>``).

    Same shape as ``remembrancer_graph`` returns for entity nodes: ``{entity,
    resolved, nodes, edges, depth}`` (+ ``note`` when unresolved). ``types``
    restricts traversal to those edge types. Foreign endpoints (people, orgs,
    systems, components) appear as stub nodes — they live in the entity graph.
    """
    depth = max(1, int(depth))
    nodes = graph.get("nodes") or {}
    topics = graph.get("topics") or {}
    raw = str(node_id or "").strip()
    resolved = None
    for candidate in (raw, _artifact_id(raw)):
        if candidate in nodes or candidate in topics:
            resolved = candidate
            break
    if resolved is None:
        return {"entity": node_id, "resolved": None, "nodes": [], "edges": [],
                "depth": depth, "note": "node not found in artifact graph"}
    wanted = set(types) if types else None
    edges = [e for e in graph.get("edges") or []
             if wanted is None or e.get("type") in wanted]
    visited = {resolved}
    frontier = {resolved}
    out_edges: list[dict] = []
    seen: set[tuple] = set()
    for _hop in range(depth):
        if not frontier:
            break
        nxt: set[str] = set()
        for e in edges:
            src, dst = e["from"], e["to"]
            if src not in frontier and dst not in frontier:
                continue
            key = (e.get("type"), src, dst, e.get("rule"), e.get("at"))
            if key in seen:
                continue
            seen.add(key)
            out_edges.append(e)
            for end in (src, dst):
                if end not in visited:
                    nxt.add(end)
        visited |= nxt
        frontier = nxt
    touched = {resolved} | {e["from"] for e in out_edges} | {e["to"] for e in out_edges}
    return {"entity": node_id, "resolved": resolved,
            "nodes": [_stub(graph, i) for i in sorted(touched)],
            "edges": out_edges, "depth": depth}


def topic_timeline(graph: dict[str, Any], topic_id: str) -> dict[str, Any]:
    """Pages of a topic in publish order with version counts and the latest.

    ``topic_id`` may be ``topic:<key>`` or an artifact slug (its topic is
    used). Unknown -> ``{"topic": None, "pages": [], "latest": None}``.
    """
    topics = graph.get("topics") or {}
    key = str(topic_id or "").strip()
    if key not in topics:
        key = topic_of(graph, key) or ""
    topic = topics.get(key)
    if topic is None:
        return {"topic": None, "pages": [], "latest": None, "tags": []}
    pages = []
    for slug in topic.get("pages") or []:
        n = node(graph, slug) or {}
        pages.append({
            "slug": slug,
            "title": n.get("title") or slug,
            "url": n.get("url") or "",
            "published_at": n.get("published_at") or "",
            "version": n.get("version"),
            "versions": len(n.get("versions") or []),
            "superseded_by": list(n.get("superseded_by") or []),
            "retired": bool(n.get("retired")),
            "latest": bool(n.get("latest_in_topic")),
        })
    return {"topic": key, "label": topic.get("label"), "pages": pages,
            "latest": topic.get("latest"), "tags": list(topic.get("tags") or [])}
