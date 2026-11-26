"""Automatic admission and a rebuildable, evidence-bound memory organization.

Admission is not verification. This projection never signs approvals, rewrites
source notes, or deletes history. Later evidence can explain a state transition;
an inference is labeled and invalidated when any contributing note changes.
Demand chooses work priority, never truth or access permissions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from remembrancerlib import (
    Note,
    acl_can_reveal,
    effective_state,
    event_text,
    load_suppressed,
    parse_frontmatter,
)
from remembrancerlib.note import slugify
from remembrancerlib.verification import verification_content_fingerprint

STATE_PATH = "state/memory_maintenance.json"
ACCESS_PATH = "state/memory_access.jsonl"
LOG_LIMIT = 8 * 1024 * 1024
SNAPSHOT_LIMIT = 32 * 1024 * 1024
WINDOW_DAYS = 30
_CACHE: dict[str, tuple[tuple, dict]] = {}
_STATES = {
    "enabled": ("enabled", 1),
    "disabled": ("enabled", 0),
    "active": ("active", 1),
    "inactive": ("active", 0),
    "open": ("open", 1),
    "closed": ("open", 0),
    "online": ("online", 1),
    "offline": ("online", 0),
    "available": ("available", 1),
    "unavailable": ("available", 0),
    "resolved": ("resolved", 1),
    "unresolved": ("resolved", 0),
    "completed": ("complete", 1),
    "complete": ("complete", 1),
    "incomplete": ("complete", 0),
}
_UNCERTAIN = {
    "not",
    "never",
    "may",
    "might",
    "could",
    "should",
    "will",
    "if",
    "planned",
    "proposed",
    "was",
    "previously",
}


def automatic(cfg: dict) -> bool:
    block = cfg.get("governance") if isinstance(cfg, dict) else None
    return isinstance(block, dict) and block.get("admission_mode") == "automatic"


def read_config(root: str) -> dict:
    try:
        data = json.loads((Path(root) / "config.json").read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def corpus_stamp(root: str):
    """Bind indexes, validated event prefix, suppression and trust generations."""
    result = {}
    for relative in (
        "context/INDEX.md",
        "state/event_validation_checkpoint.json",
        "state/suppressed.json",
        "state/trust_audit.jsonl",
        "state/source_lifecycle.jsonl",
        "state/producer_lifecycle.jsonl",
        "state/verification.head.jsonl",
    ):
        try:
            s = (Path(root) / relative).stat()
            result[relative] = [s.st_ino, s.st_mtime_ns, s.st_ctime_ns, s.st_size]
        except OSError:
            result[relative] = None
    return result


def _fingerprint(note) -> str:
    # The verification hash intentionally excludes lifecycle controls. A
    # relationship must also expire when its source is retired or demoted.
    material = [
        verification_content_fingerprint(note),
        note.state,
        getattr(note, "approval", None),
        getattr(note, "source_retirement", None),
        getattr(note, "producer_lifecycle", None),
    ]
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def _time(value) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (
            dt.replace(tzinfo=timezone.utc)
            if dt.tzinfo is None
            else dt.astimezone(timezone.utc)
        )
    except (ValueError, TypeError):
        return None


def _tokens(value: str) -> list[str]:
    return re.findall(r"[\w]+", value.casefold())


def _log_rows(path: Path) -> tuple[list[dict], str]:
    """Bounded, tolerant operational input; a missing log is unknown demand."""
    try:
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            if size > LOG_LIMIT:
                handle.seek(size - LOG_LIMIT)
                handle.readline()
            raw = handle.read(LOG_LIMIT)
    except OSError:
        return [], "unavailable"
    rows = []
    malformed = False
    for line in raw.splitlines():
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
            else:
                malformed = True
        except (ValueError, UnicodeDecodeError):
            malformed = True
    return rows, "partial" if size > LOG_LIMIT or malformed else "observed"


def record_open(root: str, name: str, cfg: dict) -> None:
    """Called only AFTER authorized successful get_note; no query or identity."""
    from retrieve import telemetry

    if (
        not automatic(cfg)
        or (cfg.get("retrieval") or {}).get("query_log") is not True
        or os.environ.get("REMEMBRANCER_QUERY_LOG") == "off"
        or not telemetry.query_log_allowed()
    ):
        return
    try:
        path = Path(root) / ACCESS_PATH
        path.parent.mkdir(exist_ok=True)
        fd = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600
        )
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
                return
            data = (
                json.dumps(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "kind": "open",
                        "name": name,
                    }
                ).encode()
                + b"\n"
            )
            os.write(fd, data)
        finally:
            os.close(fd)
    except OSError:
        pass  # A failed usage log must not fail an otherwise valid read.


def demand_index(root: str, notes: list, now: datetime) -> tuple[dict, dict]:
    counts = {
        n.name: {"opens": 0, "searches": 0, "exposures": 0, "last_used": None}
        for n in notes
    }
    start = now - timedelta(days=WINDOW_DAYS)
    access, access_coverage = _log_rows(Path(root) / ACCESS_PATH)
    queries, query_coverage = _log_rows(Path(root) / "state/queries.jsonl")
    audit, audit_coverage = _log_rows(Path(root) / "state/mcp_audit.jsonl")
    # Match dual-writer records by note and second, not a global cutover time:
    # rotation, partial history and an old/future row cannot erase real usage.
    observed_opens = {
        (r.get("name"), t.replace(microsecond=0))
        for r in access
        if r.get("kind") == "open"
        and isinstance(r.get("name"), str)
        and (t := _time(r.get("ts")))
        and start <= t <= now
    }

    def bump(name, field, when):
        if isinstance(name, str) and name in counts:
            counts[name][field] += 1
            counts[name]["last_used"] = max(
                counts[name]["last_used"] or "", when.isoformat()
            )

    for row in access:
        when = _time(row.get("ts"))
        if when and start <= when <= now and row.get("kind") == "open":
            bump(row.get("name"), "opens", when)
    # Historical MCP opens fill the period before the dedicated access log.
    # Never double-count a request recorded by both writers after deployment.
    entities = defaultdict(set)
    for note in notes:
        for entity in note.entities:
            entities[entity.casefold()].add(note.name)
    for row in audit:
        when = _time(row.get("ts"))
        if not isinstance(row.get("query"), str) or not isinstance(
            row.get("n_results"), int
        ):
            continue
        if (
            when
            and start <= when <= now
            and row.get("tool") == "remembrancer_get_note"
            and (row.get("query"), when.replace(microsecond=0)) not in observed_opens
            and row.get("n_results", 0) == 1
            and row.get("caller") != "mcp_smoke"
        ):
            bump(row.get("query"), "opens", when)
        if (
            when
            and start <= when <= now
            and row.get("tool") == "remembrancer_timeline"
            and row.get("n_results", 0) > 0
            and row.get("caller") != "mcp_smoke"
        ):
            for name in entities.get(str(row.get("query") or "").casefold(), ()):
                bump(name, "searches", when)
    term_notes: dict[str, set[str]] = defaultdict(set)
    for note in notes:
        for term in set(_tokens(note.name + " " + note.description)):
            if len(term) >= 4:
                term_notes[term].add(note.name)
    for row in queries:
        when = _time(row.get("ts"))
        if not when or not start <= when <= now or row.get("caller") == "mcp_smoke":
            continue
        delivered = (
            row.get("delivered") if isinstance(row.get("delivered"), list) else []
        )
        for name in set(v for v in delivered if isinstance(v, str)):
            bump(name, "exposures", when)
        matched: Counter = Counter()
        for term in set(_tokens(str(row.get("query") or ""))):
            # Ambient words cannot make every page look requested.
            if len(term_notes.get(term, ())) <= max(10, len(notes) // 20):
                matched.update(term_notes.get(term, ()))
        for name, n in matched.items():
            if n >= 2:
                bump(name, "searches", when)
    coverage = {
        "window_days": WINDOW_DAYS,
        "opens": access_coverage,
        "queries": query_coverage,
        "historical_opens": audit_coverage,
    }
    return counts, coverage


def _property(note, text: str | None = None) -> tuple | None:
    if note.type not in {"state", "decision", "project", "correction"}:
        return None
    words = _tokens(note.description if text is None else text)
    if set(words) & _UNCERTAIN:
        return None
    values = [(i, _STATES[w]) for i, w in enumerate(words) if w in _STATES]
    if len(values) != 1 or not note.entities:
        return None
    i, (field, value) = values[0]
    subject = tuple(
        w
        for j, w in enumerate(words)
        if j != i and w not in {"is", "are", "now", "currently", "the"}
    )
    if len(subject) < 2:
        return None
    return (note.domain, tuple(sorted(note.entities)), subject, field), value


def _literal_support(claim: str, text: str) -> bool:
    """A complete source sentence/line, never a substring inside a denial."""
    wanted = _tokens(claim)
    return bool(wanted) and any(
        _tokens(part) == wanted for part in re.split(r"[\n.!?。！？]+", text)
    )


def _dated_assertion(ev, statement: str) -> datetime | None:
    # A Slack event dates the thread root, while its flattened text can include
    # months of replies. That is not a timestamp for each quoted assertion.
    if ev.source == "slack":
        payload = ev.payload if isinstance(ev.payload, dict) else {}
        if payload.get("message_count") != 1 or payload.get("reply_count") != 0:
            return None
    return _time(ev.occurred_at) if _literal_support(statement, event_text(ev)) else None


def _dispute_key(verdict) -> tuple | None:
    if not isinstance(verdict, dict):
        return None
    claim, quote, ids = verdict.get("claim"), verdict.get("evidence_quote"), verdict.get("event_ids")
    if (not isinstance(claim, str) or not isinstance(quote, str)
            or not isinstance(ids, list) or not ids or not all(isinstance(v, str) for v in ids)):
        return None
    return claim, quote, tuple(sorted(ids))


def unresolved_claims(claims: list, memory: dict | None) -> list:
    """Apply only the exact explanations from an already bound MemoryView."""
    rows = memory.get("dispute_explanations") if isinstance(memory, dict) else None
    if not isinstance(rows, list):
        return list(claims)
    explained = {key for v in rows if (key := _dispute_key(v)) is not None}
    return [v for v in claims if _dispute_key(v) not in explained]


def _record_context(note, verdicts, evidence) -> list[dict]:
    """Reclassify demonstrated category/extraction errors, never certify facts."""
    from distill import verify as V
    from retrieve.contested import MAX_CONTESTED

    # The presentation reader caps its list. At the cap, unseen claims may
    # still conflict: never close the case from a possibly truncated prefix.
    if not verdicts or len(verdicts) >= MAX_CONTESTED:
        return []
    alternatives = (note.structured or {}).get("alternatives", []) if isinstance(note.structured, dict) else []
    alternatives = alternatives if isinstance(alternatives, list) else []
    explanations = []
    for verdict in verdicts:
        claim, quote = verdict.get("claim", ""), verdict.get("evidence_quote", "")
        ids = verdict.get("event_ids")
        if not isinstance(claim, str) or not isinstance(quote, str) or not quote.strip() or not isinstance(ids, list) or not ids:
            continue
        sources = {}
        for eid in ids:
            ev = evidence.get(eid)
            if ev is None or eid not in note.provenance or not acl_can_reveal(note.acl, ev.acl):
                break
            text = event_text(ev)
            if " ".join(quote.split()) not in " ".join(text.split()):
                break
            sources[eid] = text
        if len(sources) != len(set(ids)):
            continue
        _admissible, kind = V.claim_admissible(claim, name=note.name, description=note.description)
        reason = None
        if kind in {"meta:title_restatement", "meta:structural"}:
            reason = "record-description-or-viewpoint"
        elif any(isinstance(v, str) and _tokens(claim) == _tokens("alternatives: " + v) for v in alternatives):
            reason = "recorded-alternative"
        else:
            claim_tokens, quote_tokens = _tokens(claim), _tokens(quote)
            restates = len(quote_tokens) >= 3 and any(
                claim_tokens[i:i + len(quote_tokens)] == quote_tokens
                for i in range(len(claim_tokens) - len(quote_tokens) + 1)
            )
            windows = V._enclosing_sentences(quote, ids, sources)
            # A fragment can match while its surrounding sentence assigns it
            # to someone else or a different date. Require that whole local
            # assertion to be a restatement too, not just non-negated words.
            def contained(text):
                tokens = _tokens(text)
                return bool(tokens) and any(
                    claim_tokens[i:i + len(tokens)] == tokens
                    for i in range(len(claim_tokens) - len(tokens) + 1)
                )
            if restates and len(windows) == len(sources) and all(
                contained(text) and not V._quote_disagrees(claim, text) for text in windows
            ):
                reason = "quoted-fragment-restates-claim"
            symbols = re.findall(r"`([A-Za-z_]\w*)`", claim)
            definition = re.search(r"\bdef\s+(_[A-Za-z_]\w*)\s*\(", quote)
            if len(symbols) == 1 and definition and definition[1] == "_" + symbols[0]:
                if f"`{definition[1]}`" in note.body and f"`{symbols[0]}`" not in note.body:
                    reason = "extracted-identifier-differs-from-record"
        if reason:
            explanations.append({"claim": claim, "evidence_quote": quote,
                                 "event_ids": list(ids), "reason": reason})
    return explanations


def _viewpoint(note, row: dict) -> dict | None:
    """Only a sourced, standalone attributed view; mixed factual notes stay open."""
    if not row["source_ids"]:
        return None
    body = re.sub(r"\[\[[^\]]+\]\]", "", note.body)
    if _tokens(body) != _tokens(note.description):
        return None
    match = re.match(
        r"^([\w][\w .'-]{0,79}?)\s+(?:prefers?|recommends?|suggests?|proposes?|believes?|thinks?)\b",
        note.description,
        re.I,
    ) or re.match(r"^([\w·]{1,40}?)(?:认为|觉得|倾向于|建议|主张)", note.description)
    if not match:
        return None
    return {
        "name": note.name,
        "speaker": match.group(1).strip(),
        "statement": note.description,
        "source_ids": row["source_ids"],
        "from": row["grounded_from"],
        "through": row["grounded_to"],
    }


def _ordered_source_observations(note, row, verdicts, evidence, now):
    """Explain a disputed state with dated source observations, not a true/false vote.

    Every disputed claim must be the note's complete, source-supported state
    assertion. Every opposing quote must be a complete line about that SAME
    property in an authorized cited event. Same-time opposites remain disputed.
    The result explains chronology; it does not infer an unstated cause.
    """
    from retrieve.contested import MAX_CONTESTED

    prop = _property(note)
    if prop is None or not row["source_ids"] or not verdicts or len(verdicts) >= MAX_CONTESTED:
        return []
    observations = {}

    def add(eid, statement, value):
        ev = evidence.get(eid)
        when = _dated_assertion(ev, statement) if ev is not None else None
        if (
            ev is None or eid not in note.provenance
            or not acl_can_reveal(note.acl, ev.acl)
            or when is None or when > now
            or not _literal_support(statement, event_text(ev))
        ):
            return False
        observations[(eid, value)] = {
            "claim": statement, "source_ids": [eid],
            "from": when.isoformat(), "through": when.isoformat(),
        }
        return True

    if not all(add(eid, note.description, prop[1]) for eid in row["source_ids"]):
        return []
    for verdict in verdicts:
        quote = verdict.get("evidence_quote", "")
        other = _property(note, quote)
        ids = verdict.get("event_ids")
        if (
            _tokens(verdict.get("claim", "")) != _tokens(note.description)
            or other is None or other[0] != prop[0] or other[1] == prop[1]
            or not isinstance(ids, list) or not ids
            or not all(add(eid, quote, other[1]) for eid in ids)
        ):
            return []
    by_time = defaultdict(set)
    for (_, value), observation in observations.items():
        by_time[observation["from"]].add(value)
    if any(len(values) > 1 for values in by_time.values()) or len(by_time) < 2:
        return []
    return sorted(observations.values(), key=lambda r: (r["from"], r["source_ids"]))


def _rename(note) -> tuple[str, str, str] | None:
    # A closed statement shape, mapped to two existing entity ids. Do not
    # globally rewrite aliases: namespaces and old product lines may survive.
    match = re.match(
        r"^([\w.-]+) is now ([\w.-]+)(?=[:.,\s]|$)", note.description, re.I
    )
    if not match:
        return None
    entities = {
        re.sub(r"\W", "", e.split(":", 1)[-1]).casefold(): e for e in note.entities
    }
    old, new = (entities.get(re.sub(r"\W", "", v).casefold()) for v in match.groups())
    return (old, new, match.group()) if old and new and old != new else None


def build_snapshot(
    root: str,
    notes: list,
    events,
    *,
    now: datetime | None = None,
    contested: dict | None = None,
) -> dict:
    """Pure organization over supplied validated events; no model or source writes.

    State transitions require same property + non-overlapping event times +
    literal source support. A later file mtime, model opinion, or popularity
    never establishes a transition. An unresolved pair has one shared work item.
    """
    now = now or datetime.now(timezone.utc)
    generation = corpus_stamp(root)
    duplicates = Counter(n.name for n in notes)
    by_name = {n.name: n for n in notes if duplicates[n.name] == 1}
    demand, coverage = demand_index(root, list(by_name.values()), now)
    wanted = {eid for n in by_name.values() for eid in n.provenance}
    evidence = {}
    ambiguous = set()
    for ev in events:
        if ev.id not in wanted:
            continue
        if ev.id in evidence:
            ambiguous.add(ev.id)
        evidence[ev.id] = ev
    rows = {}
    groups: dict[tuple, list] = defaultdict(list)
    properties = {}
    for name, note in by_name.items():
        src = [
            evidence[eid]
            for eid in note.provenance
            if eid in evidence
            and eid not in ambiguous
            and acl_can_reveal(note.acl, evidence[eid].acl)
        ]
        times = [
            t for ev in src if (t := _time(ev.occurred_at)) is not None and t <= now
        ]
        grounded = [
            ev
            for ev in src
            if _literal_support(note.description, event_text(ev))
            and (t := _time(ev.occurred_at)) is not None
            and t <= now
            and acl_can_reveal(note.acl, ev.acl)
        ]
        grounded_times = [when for ev in grounded
                          if (when := _dated_assertion(ev, note.description)) is not None]
        fingerprint = _fingerprint(note)
        row = {
            "fingerprint": fingerprint,
            "path": os.path.relpath(note.path, root) if note.path else "",
            "acl": list(note.acl),
            "domain": note.domain,
            "entities": list(note.entities),
            "status": "recorded",
            "attention": "none",
            "demand": demand[name],
            "first_event_at": min(times).isoformat() if times else None,
            "last_event_at": max(times).isoformat() if times else None,
            "grounded_from": min(grounded_times).isoformat()
            if grounded_times
            else None,
            "grounded_to": max(grounded_times).isoformat() if grounded_times else None,
            "source_ids": sorted(ev.id for ev in grounded),
            "timeline": [],
            "relationships": [],
            "dependencies": {},
        }
        rows[name] = row
        prop = _property(note)
        if prop:
            key, value = prop
            # Audience partitions prevent a private event changing public truth.
            key = (*key, tuple(sorted(note.acl)))
            groups[key].append(name)
            properties[name] = value
    # The verifier can compare a dated old state to a newer source without
    # knowing that both were correct observations at their respective times.
    # Explain only closed, source-grounded cases before escalating a conflict.
    contested = dict(contested or {})
    safe_evidence = {eid: ev for eid, ev in evidence.items() if eid not in ambiguous}
    for name, verdicts in list(contested.items()):
        if name not in rows:
            continue
        history = _ordered_source_observations(
            by_name[name], rows[name], verdicts, safe_evidence, now,
        )
        if history:
            rows[name]["source_history"] = history
            rows[name]["reason"] = "ordered-source-observations"
            rows[name]["dispute_explanations"] = [
                {**v, "reason": "ordered-source-observations"} for v in verdicts
            ]
            del contested[name]
        else:
            explanations = _record_context(by_name[name], verdicts, safe_evidence)
            if explanations:
                rows[name]["dispute_explanations"] = explanations
                remaining = unresolved_claims(verdicts, rows[name])
                if remaining:
                    contested[name] = remaining
                else:
                    rows[name]["reason"] = "record-context-explains-label"
                    del contested[name]
    viewpoints = {
        name: view for name, note in by_name.items()
        if (view := _viewpoint(note, rows[name])) is not None
    }
    cases = []
    # Explicit producer supersedes links are a causal proposal. If both ends
    # cite literal support and the new evidence follows the old evidence, the
    # proposal can organize history without pretending to be a human approval.
    successors = defaultdict(set)
    successor_edges = {}
    for name, note in by_name.items():
        for relation, target in note.typed_links:
            if relation == "supersedes":
                successors[target].add(name)
    for name, note in by_name.items():
        for relation, target in note.typed_links:
            if relation != "supersedes" or target not in rows or target == name:
                continue
            other = by_name[target]
            if not (
                set(note.entities) & set(other.entities)
                and acl_can_reveal(note.acl, other.acl)
                and acl_can_reveal(other.acl, note.acl)
            ):
                continue
            if len(successors[target]) != 1:
                continue
            a, b = rows[target], rows[name]
            if (
                name not in (contested or {})
                and target not in (contested or {})
                and a["grounded_to"]
                and b["grounded_from"]
                and a["grounded_to"] < b["grounded_from"]
            ):
                successor_edges[target] = name
    for start in sorted(set(successor_edges) - set(successor_edges.values())):
        chain = [start]
        while chain[-1] in successor_edges and len(chain) <= 32:
            chain.append(successor_edges[chain[-1]])
        if len(chain) > 32 or len(set(chain)) != len(chain):
            continue
        # Branching/merging proposals are ambiguous even when their dates differ.
        if any(list(successor_edges.values()).count(n) > 1 for n in chain):
            continue
        timeline = [
            {
                "name": n,
                "from": rows[n]["grounded_from"],
                "through": rows[n]["grounded_to"],
            }
            for n in chain
        ]
        for n in chain:
            rows[n].update(
                status="current-inferred" if n == chain[-1] else "historical-inferred",
                reason="explicit-successor-later-source",
                timeline=timeline,
                dependencies={
                    v: {"fingerprint": rows[v]["fingerprint"], "path": rows[v]["path"]}
                    for v in chain
                    if v != n
                },
            )
    topics = []
    for key, names in sorted(groups.items(), key=lambda pair: str(pair[0])):
        topic_id = hashlib.sha256(repr(key).encode()).hexdigest()[:20]
        ordered = sorted(
            names, key=lambda name: (rows[name]["grounded_to"] or "", name)
        )
        topics.append(
            {
                "id": topic_id,
                "domain": key[0],
                "entities": list(key[1]),
                "subject": " ".join(key[2]),
                "field": key[3],
                "notes": ordered,
            }
        )
        if len(names) < 2 or len({properties[n] for n in names}) < 2:
            continue
        # Limit dependencies/response size. A broad group is an unresolved case,
        # never permission to choose a winner from a truncated evidence window.
        resolvable = (
            len(names) <= 32
            and all(rows[n]["grounded_from"] for n in names)
            and not any(n in (contested or {}) for n in names)
        )
        if resolvable:
            resolvable = all(
                rows[a]["grounded_to"] < rows[b]["grounded_from"]
                for a, b in zip(ordered, ordered[1:])
            )
        timeline = (
            [
                {
                    "name": n,
                    "from": rows[n]["grounded_from"],
                    "through": rows[n]["grounded_to"],
                }
                for n in ordered
            ]
            if len(names) <= 32
            else []
        )
        for name in names:
            row = rows[name]
            row["topic"] = topic_id
            row["timeline"] = timeline
            row["dependencies"] = (
                {
                    n: {"fingerprint": rows[n]["fingerprint"], "path": rows[n]["path"]}
                    for n in names
                    if n != name
                }
                if len(names) <= 32
                else {}
            )
            row["status"] = (
                ("current-inferred" if name == ordered[-1] else "historical-inferred")
                if resolvable
                else "unresolved"
            )
            row["reason"] = (
                "later-source-same-property"
                if resolvable
                else "evidence-does-not-order-conflict"
            )
        if not resolvable:
            cases.append((topic_id, names, "conflicting-state"))
    # Preserve explicit contradictions, including those outside the conservative
    # state grammar. They can be deferred; they are never silently called true.
    seen = {frozenset(names) for _, names, _ in cases}
    for name, note in by_name.items():
        for relation, target in note.typed_links:
            if relation != "contradicts" or target not in rows or target == name:
                continue
            other = by_name[target]
            if not (
                acl_can_reveal(note.acl, other.acl)
                and acl_can_reveal(other.acl, note.acl)
            ):
                continue
            pair = frozenset((name, target))
            if all(n in viewpoints and n not in contested for n in pair):
                # Preserve who held which view, with equal standing. A factual
                # source disagreement (including a false attribution) still
                # reaches the independent contested lane below.
                perspectives = [viewpoints[n] for n in sorted(pair)]
                for n in pair:
                    rows[n]["reason"] = "attributed-viewpoints"
                    existing = {v["name"]: v for v in rows[n].get("perspectives", [])}
                    existing.update({v["name"]: v for v in perspectives})
                    rows[n]["perspectives"] = [existing[k] for k in sorted(existing)]
                    rows[n]["dependencies"].update({
                        v: {"fingerprint": rows[v]["fingerprint"], "path": rows[v]["path"]}
                        for v in pair if v != n
                    })
                continue
            explained = all(
                rows[n]["status"] in {"historical-inferred", "current-inferred"}
                and pair <= {entry["name"] for entry in rows[n]["timeline"]}
                for n in pair
            )
            if pair not in seen and not explained:
                seen.add(pair)
                cases.append(
                    (
                        hashlib.sha256("|".join(sorted(pair)).encode()).hexdigest()[
                            :20
                        ],
                        sorted(pair),
                        "explicit-conflict",
                    )
                )
    for name in contested or {}:
        if name in rows:
            cases.append(("claim-" + name, [name], "source-disagrees"))
    exceptions = []
    for case_id, names, reason in cases:
        requested = any(demand[n]["opens"] or demand[n]["searches"] for n in names)
        important = any(
            by_name[n].type == "decision"
            or by_name[n].domain in {"customers", "security"}
            for n in names
        )
        attention = "needs-human" if requested or important else "deferred"
        for name in names:
            rows[name]["status"] = "unresolved"
            if rows[name]["attention"] != "needs-human":
                rows[name]["attention"] = attention
        exceptions.append(
            {
                "id": case_id,
                "notes": names,
                "reason": reason,
                "attention": attention,
                "next_action": "investigate-sources" if attention == "needs-human" else "wait-for-evidence-or-demand",
                "attempted": ["source-grounding", "event-order", "same-property"],
                "priority": 10 * int(important)
                + sum(5 * demand[n]["opens"] + demand[n]["searches"] for n in names),
            }
        )
    exceptions.sort(key=lambda r: (-r["priority"], r["id"]))
    # Explicit, grounded renames are useful relations even where no state
    # grammar applies. A signed existing correction can supply the assertion;
    # its reported date stays separate from an event's occurrence date.
    relationships = []
    entity_notes = defaultdict(list)
    for note in by_name.values():
        for entity in note.entities:
            entity_notes[entity].append(note)
    for name, note in by_name.items():
        rename = _rename(note)
        if not rename or name in (contested or {}):
            continue
        old, new, assertion = rename
        if effective_state(note) != "verified" and not rows[name]["grounded_from"]:
            continue
        reported = next(
            (
                _time(p[3:])
                for p in note.provenance
                if p.startswith("at:") and _time(p[3:])
            ),
            None,
        )
        relation = {
            "kind": "renamed-to",
            "from_entity": old,
            "to_entity": new,
            "source_note": name,
            "assertion": assertion,
            "status": "source-asserted",
            "occurred_at": rows[name]["grounded_from"],
            "reported_at": reported.isoformat()
            if reported and reported <= now
            else None,
        }
        relationships.append(relation)
        for target in entity_notes[old]:
            if acl_can_reveal(target.acl, note.acl):
                row = rows[target.name]
                row["relationships"].append(relation)
                if target.name != name:
                    row["dependencies"][name] = {
                        "fingerprint": rows[name]["fingerprint"],
                        "path": rows[name]["path"],
                    }
    framework = {}
    for note in by_name.values():
        space = framework.setdefault(note.domain, {})
        for entity in note.entities or ["unassigned"]:
            space.setdefault(entity, []).append(note.name)
    for space in framework.values():
        for names in space.values():
            names.sort()
    return {
        "version": 1,
        "computed_at": now.isoformat(),
        "policy": "automatic",
        "corpus_stamp": generation,
        "demand_coverage": coverage,
        "notes": rows,
        "topics": topics,
        "exceptions": exceptions,
        "framework": framework,
        "relationships": relationships,
        "summary": {
            "recorded": len(rows),
            "topics": len(topics),
            "historical_inferred": sum(
                r["status"] == "historical-inferred" for r in rows.values()
            ),
            "relationships": len(relationships),
            "source_history_explained": sum(bool(r.get("source_history")) for r in rows.values()),
            "viewpoints_preserved": sum(bool(r.get("perspectives")) for r in rows.values()),
            "record_context_explained": sum(r.get("reason") == "record-context-explains-label" for r in rows.values()),
            "needs_human": sum(r["attention"] == "needs-human" for r in exceptions),
            "deferred": sum(r["attention"] == "deferred" for r in exceptions),
        },
    }


class MemoryView:
    """Read-only, bounded projection; stale relationships become ordinary notes."""

    def __init__(self, root: str, cfg: dict, *, snapshot: dict | None = None):
        self.root = os.path.abspath(root)
        self.enabled = automatic(cfg)
        self.snapshot = (
            snapshot if snapshot is not None else self._load() if self.enabled else {}
        )
        if self.snapshot.get("corpus_stamp") != corpus_stamp(root):
            self.snapshot = {}
        self.checked: dict[str, tuple[tuple, tuple]] = {}

    def _load(self) -> dict:
        path = Path(self.root) / STATE_PATH
        try:
            info = path.stat()
            key = (info.st_ino, info.st_mtime_ns, info.st_size)
            if info.st_size > SNAPSHOT_LIMIT:
                return {}
            cached = _CACHE.get(str(path))
            if cached and cached[0] == key:
                return cached[1]
            data = json.loads(path.read_text())
            if (
                not isinstance(data, dict)
                or data.get("version") != 1
                or not isinstance(data.get("notes"), dict)
                or not isinstance(data.get("exceptions"), list)
                or any(
                    not isinstance(v, dict)
                    or not isinstance(v.get("notes"), list)
                    or any(not isinstance(n, str) for n in v["notes"])
                    for v in data["exceptions"]
                )
            ):
                return {}
            _CACHE[str(path)] = key, data
            if len(_CACHE) > 64:
                _CACHE.pop(next(iter(_CACHE)))
            return data
        except (OSError, ValueError):
            return {}

    def _dependency(self, name, binding):
        try:
            if not isinstance(binding, dict):
                return None, None
            path = Path(self.root) / binding["path"]
            if not path.resolve().is_relative_to(
                (Path(self.root) / "context").resolve()
            ):
                return None, None
            info = path.stat()
            signature = (info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_size)
            cached = self.checked.get(name)
            if cached and cached[0] == signature:
                return cached[1]
            fm, body = parse_frontmatter(path.read_text())
            note = Note.from_frontmatter(fm, body, str(path))
            result = (note, _fingerprint(note)) if note.name == name else (None, None)
        except (OSError, ValueError, KeyError, TypeError):
            return None, None
        self.checked[name] = signature, result
        return result

    def for_note(self, note) -> dict:
        if not self.enabled:
            return {}
        default = {"admission": "automatic", "status": "recorded", "attention": "none"}
        row = self.snapshot.get("notes", {}).get(note.name)
        if not isinstance(row, dict) or row.get("fingerprint") != _fingerprint(note):
            return default
        dependencies = row.get("dependencies", {})
        if not isinstance(dependencies, dict) or len(dependencies) > 64:
            return default
        if dependencies:
            try:
                suppressed = {slugify(n) for n in load_suppressed(self.root)}
            except RuntimeError:
                return default
        for name, binding in dependencies.items():
            if slugify(name) in suppressed:
                return default
            other, fingerprint = self._dependency(name, binding)
            if (
                other is None
                or fingerprint != binding.get("fingerprint")
                or not acl_can_reveal(note.acl, other.acl)
            ):
                return default
        keys = (
            "status",
            "attention",
            "demand",
            "first_event_at",
            "last_event_at",
            "timeline",
            "topic",
            "reason",
            "relationships",
            "perspectives",
            "source_history",
            "dispute_explanations",
        )
        return {
            **default,
            **{key: row[key] for key in keys if key in row},
            "framework": {
                "domain": note.domain,
                "type": note.type,
                "entities": list(note.entities),
            },
            "assessed_at": self.snapshot.get("computed_at"),
        }

    def rank(self, ranked, *, include_history=False):
        if not self.enabled or include_history:
            return ranked
        adjusted = [
            (
                note,
                score
                * (
                    0.2
                    if self.for_note(note).get("status") == "historical-inferred"
                    else 1.0
                ),
            )
            for note, score in ranked
        ]
        return sorted(adjusted, key=lambda pair: -pair[1])
