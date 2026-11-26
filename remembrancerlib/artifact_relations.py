"""The artifact relation ledger — append-only decisions folded over the graph.

The graph builder derives edges structurally; this module records
DECISIONS about those edges (assert / retract) and folds them over the derived
list as the last build step. Append-only: a decision is never rewritten, so
every past state of the graph is reconstructible and every edge answers "who
decided this, and why".

Precedence, low to high: a structural rule < an LLM's adjudication (decider
``LLM_DECIDER``) < a human or agent assertion via MCP. A publisher's own
``supersedes=`` declaration outranks all three and is passed to :func:`fold`
as ``protected``.
"""
from __future__ import annotations

import fcntl
import json
import os
import uuid
from datetime import datetime, timezone

RELATION_TYPES: tuple[str, ...] = (
    "supersedes", "extends", "version_of", "same_topic", "links_to",
    "mentions", "authored_by", "handed_off", "handed_off_to", "about_component",
)
LEDGER_REL_PATH = os.path.join("state", "artifact_relations.jsonl")
ARTIFACT_PREFIX = "artifact:"
OPS = ("assert", "retract")

TIER_RULE, TIER_LLM, TIER_ACTOR = 0, 1, 2
#: The decider an LLM's adjudication is recorded under.
LLM_DECIDER = "llm"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _qualify(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text if ":" in text else ARTIFACT_PREFIX + text


def decision_id(now: datetime | None = None, nonce: str | None = None) -> str:
    stamp = (now or _utcnow()).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"rel-{stamp}-{nonce or uuid.uuid4().hex[:6]}"


def tier(by: str) -> int:
    """Precedence tier of a decider. ``rule:*`` < ``LLM_DECIDER`` < any remembrancer identity."""
    who = str(by or "").strip()
    if who.startswith("rule:"):
        return TIER_RULE
    if who == LLM_DECIDER:
        return TIER_LLM
    return TIER_ACTOR


def append_decision(root: str, *, op: str, type: str, from_slug: str,
                    to_slug: str, by: str, reason: str | None = None,
                    evidence: list[str] | None = None,
                    replaces_decision: str | None = None,
                    now: datetime | None = None) -> dict:
    """Append one decision. Raises ValueError on an invalid op/type/endpoint."""
    if op not in OPS:
        raise ValueError(f"op must be one of {OPS}, got {op!r}")
    if type not in RELATION_TYPES:
        raise ValueError(f"type must be one of {RELATION_TYPES}, got {type!r}")
    src, dst = _qualify(from_slug), _qualify(to_slug)
    if not src or not dst:
        raise ValueError("from_slug and to_slug are required")
    if src == dst:
        raise ValueError("a relation needs two different endpoints")
    if not str(by or "").strip():
        raise ValueError("by is required — every decision names its decider")
    moment = now or _utcnow()
    record = {
        "id": decision_id(moment),
        "op": op,
        "type": type,
        "from": src,
        "to": dst,
        "by": str(by).strip(),
        "at": _iso(moment),
        "reason": (reason or "").strip() or None,
        "evidence": list(dict.fromkeys(evidence or [])),
        "replaces_decision": replaces_decision or None,
    }
    path = os.path.join(root, LEDGER_REL_PATH)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    return record


def load_decisions(root: str) -> list[dict]:
    """Every well-formed decision, in file order. Missing file -> []."""
    path = os.path.join(root, LEDGER_REL_PATH)
    out: list[dict] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (isinstance(row, dict) and row.get("op") in OPS
                        and row.get("type") in RELATION_TYPES
                        and row.get("from") and row.get("to") and row.get("by")):
                    out.append(row)
    except OSError:
        return []
    return out


def _key(row: dict) -> tuple[str, str, str]:
    return (row["type"], row["from"], row["to"])


def current_relations(decisions: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Winning decision per ``(type, from, to)``: highest tier, then latest, then
    latest in file order. Ties inside a tier are broken by ``at`` then position."""
    best: dict[tuple[str, str, str], tuple[tuple[int, str, int], dict]] = {}
    for position, row in enumerate(decisions):
        rank = (tier(row["by"]), str(row.get("at") or ""), position)
        key = _key(row)
        if key not in best or rank > best[key][0]:
            best[key] = (rank, row)
    return {key: row for key, (_, row) in best.items()}


def fold(edges: list[dict], decisions: list[dict], *,
         protected: frozenset[tuple[str, str, str]] = frozenset()) -> list[dict]:
    """Apply the ledger to structurally-derived ``edges``; returns a new list.

    A winning ``retract`` drops the edge (unless its triple is ``protected`` — a
    publisher declaration, which outranks the ledger). A winning ``assert`` marks
    an existing edge accepted and stamps ``decision_id``/``decided_by``, or adds
    the edge when no structural rule produced it.
    """
    winners = current_relations(decisions)
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in edges:
        key = (edge.get("type"), edge.get("from"), edge.get("to"))
        decision = winners.get(key)
        if decision is None or key in protected:
            out.append(edge)
            continue
        seen.add(key)
        if decision["op"] == "retract":
            continue
        merged = dict(edge)
        merged.update({"status": "accepted", "confidence": 1.0,
                       "rule": f"ledger:{decision['by']}",
                       "decision_id": decision["id"],
                       "decided_by": decision["by"],
                       "at": decision.get("at")})
        if decision.get("reason"):
            merged["reason"] = decision["reason"]
        out.append(merged)
    for key, decision in winners.items():
        if key in seen or decision["op"] != "assert" or key in protected:
            continue
        out.append({"from": decision["from"], "to": decision["to"],
                    "type": decision["type"], "confidence": 1.0,
                    "status": "accepted", "rule": f"ledger:{decision['by']}",
                    "evidence": list(decision.get("evidence") or []),
                    "at": decision.get("at"), "decision_id": decision["id"],
                    "decided_by": decision["by"],
                    **({"reason": decision["reason"]} if decision.get("reason") else {})})
    return out


def decisions_for(decisions: list[dict], slug: str) -> list[dict]:
    """Every decision touching ``slug`` (either endpoint), oldest first."""
    node = _qualify(slug)
    return [row for row in decisions if row["from"] == node or row["to"] == node]
