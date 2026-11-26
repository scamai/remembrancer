"""Layer-2 notes: distilled, labeled, atomic markdown — the derived cache.

One file per durable fact at context/<domain>/<slug>.md, frontmatter + body +
[[links]]. INDEX.md lists one line per note (the cheap thing retrieval loads
first). L2 is a pure function of L1: it can be deleted and rebuilt from events.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .acl import acl_can_reveal
from .verification import (
    apply_governance_hold,
    auto_approval_valid,
    auto_approval_write_authorized,
    bind_loaded_auto_approvals,
    bind_loaded_human_trust,
    ensure_machine_verification_binding,
    governance_hold_class,
    governance_rewrite_authorized,
    human_trust_valid,
    human_trust_write_authorized,
    machine_authored,
    verification_content_fingerprint,
)
from .source_lifecycle import (
    bind_loaded_source_retirements,
    source_lifecycle_write_authorized,
    source_retirement_valid,
)
from .producer_lifecycle import (
    bind_loaded_producer_lifecycles,
    producer_lifecycle_capability_sealed,
    producer_lifecycle_valid,
    producer_lifecycle_write_authorized,
)

VALID_TYPES = (
    "project", "reference", "decision", "state", "entity", "concept",
    "source-summary", "synthesis", "contradiction", "page",
    # Current deterministic/repo-owned producers.
    "artifact", "initiative", "correction",
)
VALID_DOMAINS = (
    "customers", "prospect", "partners", "vendor", "investors",
    "advisors", "network", "recruiting", "research", "internal",
    "eng", "ml", "product", "business", "ops", "repos", "infra",
    "company",
    # Deterministic producer-owned presentation domains.
    "people", "design", "synthesis",
)
VALID_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
SOURCE_LINEAGE_RE = re.compile(
    r"(?:slack|gmail|gdrive|crm|gcal):[^\s,\[\]\r\n]+\Z"
)
MAX_DESCRIPTION_CHARS = 280


def concise_description(text: Any, *, fallback: str = "") -> str:
    """Normalize retrieval metadata to one bounded, non-empty line.

    Full source material belongs in the append-only body/L1 event.  Allowing a
    document or CSV to occupy ``description`` poisons BM25, every index, and
    later model candidate prompts. Producers may use this helper deliberately;
    the write door still rejects overlong values rather than silently hiding a
    producer bug.
    """
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        normalized = re.sub(r"\s+", " ", str(fallback or "")).strip()
    if len(normalized) <= MAX_DESCRIPTION_CHARS:
        return normalized
    prefix = normalized[: MAX_DESCRIPTION_CHARS - 1].rstrip()
    # Prefer a word boundary when it does not throw away most of the useful
    # prefix; scripts without spaces safely use the exact character boundary.
    boundary = prefix.rfind(" ")
    if boundary >= int(MAX_DESCRIPTION_CHARS * 0.7):
        prefix = prefix[:boundary].rstrip()
    return prefix + "…"

# Trust lifecycle.  Missing or invalid state is unreviewed, never an
# implicit human approval.  Legacy notes remain searchable but carry the
# proposed rank penalty until an approver explicitly promotes them.
VALID_STATES = ("proposed", "verified", "superseded")
DEFAULT_EFFECTIVE_STATE = "proposed"


def effective_state(note, human_audit=None, verification_audit=None) -> str:
    """Return only a lifecycle state backed by a governed transition proof."""
    s = getattr(note, "state", None)
    if s == "proposed":
        producer_control = getattr(note, "producer_lifecycle", None)
        producer_schema_valid = getattr(
            note, "_producer_lifecycle_schema_valid", True,
        )
        # A producer restoration makes ``proposed`` current only while both
        # durable copies remain valid.  If its journal/proof is removed,
        # malformed, or stale, hide the flag instead of silently accepting raw
        # state text as an equivalent restore authority.
        if (
            not producer_schema_valid
            or (producer_control is not None and not producer_lifecycle_valid(note))
        ):
            return "superseded"
        return s
    if s in ("verified", "superseded") and human_trust_valid(note, human_audit):
        return s
    # A config-gated machine approval is verified-equivalent only while the
    # gate remains on and both durable copies bind the current bytes.  It is
    # deliberately narrower than human trust: it can never retire a note.
    if s == "verified" and auto_approval_valid(note, verification_audit):
        return s
    # A closed deterministic source producer may retire only its own factual
    # snapshot.  This is deliberately narrower than human trust: it can prove
    # ``superseded`` but can never mint a verified fact.
    if s == "superseded" and source_retirement_valid(note):
        return s
    if s == "superseded" and producer_lifecycle_valid(note):
        return s
    return DEFAULT_EFFECTIVE_STATE


def supersession_head_active(note, human_audit=None) -> bool:
    """Whether this note may make a ``supersedes`` edge current.

    Supersession is a trust-gated graph projection, not a materialized flag on
    the target.  Only an explicitly verified head may change current truth.
    Missing/invalid legacy state fails closed to ``proposed`` and therefore
    cannot activate an edge until an approver promotes the head.
    """
    return (
        effective_state(note, human_audit) == "verified"
        and not bool(getattr(note, "superseded", False))
    )


def active_supersession_targets(notes, human_audit=None) -> set[str]:
    """Return valid targets hidden by active heads in this visible note set.

    Callers that enforce an identity ACL must pass only caller-visible notes;
    write-time/global consumers pass the whole loaded corpus. Self-links and
    dangling edges never change lifecycle state. Promotion prevents new cycles,
    while this projection remains deterministic for legacy data.
    """
    materialized = list(notes)
    groups: dict[str, list[Any]] = {}
    for note in materialized:
        groups.setdefault(str(getattr(note, "name", "") or ""), []).append(note)
    unique = {name: group[0] for name, group in groups.items()
              if name and len(group) == 1}
    edges: list[tuple[str, str]] = []
    for note in unique.values():
        if not supersession_head_active(note, human_audit):
            continue
        head_name = note.name
        for relation, target in getattr(note, "typed_links", []) or []:
            target_note = unique.get(target)
            if (relation == "supersedes" and target_note is not None
                    and target != head_name
                    and acl_can_reveal(note.acl, target_note.acl)):
                edges.append((head_name, target))

    # Two verified heads for one target are an unresolved branch, not current
    # truth.  Ignore every ambiguous edge until an approver retires one head.
    heads_by_target: dict[str, set[str]] = {}
    for head, target in edges:
        heads_by_target.setdefault(target, set()).add(head)
    edges = [edge for edge in edges if len(heads_by_target[edge[1]]) == 1]

    # Legacy files may predate promotion's cycle guard.  A cycle has no causal
    # head, so none of its edges may hide memory.  Detect all DFS stack members
    # and discard every edge touching them.
    graph: dict[str, set[str]] = {}
    for head, target in edges:
        graph.setdefault(head, set()).add(target)
    visiting: list[str] = []
    visited: set[str] = set()
    cycle_nodes: set[str] = set()

    def walk(node: str) -> None:
        if node in visiting:
            cycle_nodes.update(visiting[visiting.index(node):])
            return
        if node in visited:
            return
        visiting.append(node)
        for target in graph.get(node, set()):
            walk(target)
        visiting.pop()
        visited.add(node)

    for node in sorted(graph):
        walk(node)
    return {
        target for head, target in edges
        if head not in cycle_nodes and target not in cycle_nodes
    }


def current_note_names(notes, human_audit=None) -> set[str]:
    """Project one visible note set to names eligible as current truth."""
    materialized = list(notes)
    groups: dict[str, int] = {}
    for note in materialized:
        groups[note.name] = groups.get(note.name, 0) + 1
    unique_names = {name for name, count in groups.items() if count == 1}
    targets = active_supersession_targets(materialized, human_audit)
    return {
        note.name for note in materialized
        if note.name in unique_names
        # A raw sensitive lifecycle claim that cannot reproduce its effective
        # state is neither current truth nor a write-time mutation target.  In
        # particular, do not turn an unbound legacy ``superseded`` row into a
        # current ``proposed`` row merely because effective_state fails closed.
        # Keeping this comparison against effective_state also lets another
        # governed proof authority opt in by teaching effective_state how to
        # validate that exact transition.
        and (
            getattr(note, "state", None) not in ("verified", "superseded")
            or effective_state(note, human_audit) == note.state
        )
        and effective_state(note, human_audit) != "superseded" and note.name not in targets
    }


def set_state(note, state: str | None):
    """Validated in-place state setter. Raises ValueError on an invalid value — this is a
    control operation, so fail loud rather than silently no-op."""
    if state is not None and state not in VALID_STATES:
        raise ValueError(f"invalid state {state!r}; must be one of {VALID_STATES} or None")
    note.state = state
    return note
LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
# Typed link regex: optional generic relation prefix, e.g.
# [[supersedes:name]], [[supports:name]], or [[name]].  Relations are open
# vocabulary; individual consumers decide which semantics they understand.
TYPED_LINK_RE = re.compile(r"\[\[(?:([^:\]\s]+):)?([^\]]+)\]\]")


def resolve_links(body: str, name_to_path: dict) -> str:
    """Rewrite wiki-style links to OKF path links.

    - ``[[name]]``     → ``[name](/domain/slug.md)`` when name is in name_to_path
    - ``[[rel:name]]`` → ``[name](/domain/slug.md)`` stripping the rel: prefix
    - Unknown names are left as-is (``[[name]]`` or ``[[rel:name]]``)
    """
    def _replace(m: re.Match) -> str:
        inner = m.group(1)
        # Check for rel: prefix
        colon = inner.find(":")
        if colon != -1:
            name = inner[colon + 1:]
        else:
            name = inner
        name = name.strip()
        path = name_to_path.get(name)
        if path is None:
            return m.group(0)  # leave unchanged
        return f"[{name}]({path})"

    return LINK_RE.sub(_replace, body)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:80] or "note"


def _compact_json_line(obj) -> str:
    """Serialize a dict as ONE compact frontmatter line that round-trips through the scalar
    parser. json.dumps escapes \\n\\r\\t but leaves U+2028/U+2029/U+0085 raw, and the
    frontmatter parser splits on str.splitlines() which breaks on those — escape them so the
    value stays a single physical line and round-trips losslessly. Shared by `structured` and
    `confidence` (both additive optional-JSON fields)."""
    js = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    return (js.replace("\u2028", "\\u2028")
              .replace("\u2029", "\\u2029")
              .replace("\u0085", "\\u0085"))


def _as_json_obj(v):
    """Parse a frontmatter value that should be a JSON object: accept a dict as-is, or a
    `{`-prefixed JSON string. Malformed ⇒ None (the note still loads). Mirrors the additive
    `structured`/`confidence` contract."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip().startswith("{"):
        try:
            d = json.loads(v)
            return d if isinstance(d, dict) else None
        except Exception:
            return None
    return None


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


#: One entry of a block list: `- item` (or a bare `-`), at any indentation.
_BLOCK_ITEM_RE = re.compile(r"\s*-(?:\s+(.*))?")


def _list_item(raw: str) -> str:
    # Strip optional YAML quoting per item: a Python-repr-style list like
    # ['eng'] must parse to the same labels as [eng] — unstripped quotes
    # made ACL matching fail silently (notes invisible to every identity).
    return raw.strip().strip("'\"")


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse `---`-delimited YAML-ish frontmatter. Stdlib-only: handles scalars,
    [a, b] inline lists and block lists (a top-level `key:` with no value,
    followed by `- item` lines), which is sufficient for the note contract."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block, body = text[3:end].strip(), text[end + 4:]
    fm: dict[str, Any] = {}
    list_key = None     # the key whose block list the next `- item` lines extend
    for line in block.splitlines():
        if list_key is not None:
            # YAML reads `acl:` followed by `  - team-x` as ['team-x']. These
            # lines used to be skipped, which left acl empty, and an empty acl
            # is public: a hand-written restricted note was readable by anyone.
            item = _BLOCK_ITEM_RE.fullmatch(line)
            if item:
                if not isinstance(fm[list_key], list):
                    fm[list_key] = []
                value = _list_item(item.group(1) or "")
                if value:
                    fm[list_key].append(value)
                continue
            if not line.strip() or line.lstrip().startswith("#"):
                continue    # blank and comment lines do not end a YAML list
            list_key = None
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            v = [item for item in map(_list_item, v[1:-1].split(",")) if item]
        elif not v and not line[:1].isspace():
            list_key = k.strip()
        fm[k.strip()] = v
    return fm, body.lstrip("\n")


@dataclass
class Note:
    name: str
    description: str
    type: str = "project"
    domain: str = "company"                       # folder: eng|ml|customers|infra|company|product|research|business
    # Project namespace — when set, write_note writes to
    # context/<domain>/<project>/<slug>.md instead of context/<domain>/<slug>.md.
    # The distiller infers this from existing
    # same-entity-same-domain peers; absent (None/empty) ⇒ flat layout.
    # Absent from frontmatter on legacy notes → "" (back-compat).
    project: str = ""
    acl: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    # Eng-component / initiative facet: `component:<repo>`
    # / `initiative:<id>` ids the distiller routed OUT of `entities` (see
    # distill.engine.ComponentResolver). Disjoint from `entities` by
    # construction. Absent in old notes -> [] (back-compat on read).
    components: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)   # event ids
    # Explicit ownership of mutable source objects.  Provenance is citation;
    # only this field authorizes a later source revision to update the note.
    # Absent on legacy notes so old citations never become authority by accident.
    source_lineages: list[str] = field(default_factory=list)
    updated: str = field(default_factory=_today)
    body: str = ""
    path: str = ""                                 # filled on load
    # OKF optional fields (timestamp alias = updated)
    title: str = ""
    resource: str = ""
    tags: list[str] = field(default_factory=list)
    # Compatibility/audit marker for an explicitly retired note. Current truth
    # is governed by state + caller-visible verified graph edges; this boolean
    # is never materialized from an edge and is not an independent read gate.
    superseded: bool = False
    # Optional machine-structured payload the UI renders as a visual (capacity bars,
    # tables, status panels, …). Serialized as ONE compact JSON line in frontmatter so
    # it round-trips through the scalar parser without any nested-YAML risk. Always a
    # JSON object ({...}, never [...]) so the parser never mistakes it for a list.
    # Absent (None) ⇒ note renders exactly as before.
    structured: dict | None = None
    # Trust lifecycle: proposed → verified → superseded.  Absent
    # (None) fails closed to PROPOSED while preserving legacy files byte-for-byte.
    # Agents write `proposed`; humans promote to `verified`. Retrieval
    # prefers verified. Serialized only when non-None.
    state: str | None = None
    # Credibility: deterministic confidence signal, one compact JSON line only when present
    # (same additive precedent as `structured`; absent ⇒ note byte-identical). Frozen shape:
    # {score: 0..1, level: high|medium|low, sources: int, verification: {...}, human: str,
    #  contradicted: bool, computed_at: YYYY-MM-DD}. Consumed by signal badges. Orthogonal to
    # `state`: `state` is the human/trust decision, `confidence` is the derived evidence strength.
    confidence: dict | None = None
    # Content-bound human lifecycle proof. It is valid only when the latest
    # durable state/trust_audit.jsonl row carries the same id and content hash.
    approval: dict | None = None
    # Content- and journal-bound proof that a closed source contract reported
    # deletion/cancellation.  It is a lifecycle control, not answer content.
    source_retirement: dict | None = None
    # Deterministic producer-owned lifecycle proof (currently contradiction
    # flags only).  Like approval/source_retirement it is journal-bound control.
    producer_lifecycle: dict | None = None

    @property
    def effective_state(self) -> str:
        return effective_state(self)

    @property
    def links(self) -> list[str]:
        return LINK_RE.findall(self.body)

    @property
    def typed_links(self) -> list[tuple[str, str]]:
        """Return (relation, name) for every [[rel:name]] or [[name]] in body.

        Plain ``[[name]]`` defaults to relation ``"relates"``.
        Relation names are open vocabulary.  Plain links default to
        ``"relates"``; semantic consumers only act on relations they know.
        """
        return [
            ((rel.strip().lower() if rel else "relates"), name.strip())
            for rel, name in TYPED_LINK_RE.findall(self.body)
        ]

    def to_markdown(self) -> str:
        def fmt_list(xs): return "[" + ", ".join(xs) + "]"
        fm = [
            "---",
            f"name: {self.name}",
            f"title: {self.title or self.name}",
            f"description: {self.description}",
            f"resource: {self.resource}".rstrip(),
            f"tags: {fmt_list(self.tags)}",
            f"type: {self.type}",
            f"domain: {self.domain}",
            f"acl: {fmt_list(self.acl)}",
            f"entities: {fmt_list(self.entities)}",
            f"components: {fmt_list(self.components)}",
            f"provenance: {fmt_list(self.provenance)}",
            f"updated: {self.updated}",
            f"superseded: {'true' if self.superseded else 'false'}",
        ]
        # Project namespace: only when non-empty (additive precedent: keeps
        # legacy notes byte-identical when no project subfolder).
        if self.project:
            fm.append(f"project: {self.project}")
        if self.source_lineages:
            fm.append(f"source_lineages: {fmt_list(self.source_lineages)}")
        # structured: one compact JSON line, only when present (keeps existing notes byte-identical)
        if self.structured:
            fm.append("structured: " + _compact_json_line(self.structured))
        # Trust state is additive.  An absent line stays byte-compatible but is
        # interpreted as proposed, not silently approved.
        if self.state in VALID_STATES:
            fm.append(f"state: {self.state}")
        if self.approval:
            fm.append("approval: " + _compact_json_line(self.approval))
        if self.source_retirement:
            fm.append("source_retirement: " + _compact_json_line(self.source_retirement))
        if self.producer_lifecycle:
            fm.append("producer_lifecycle: " + _compact_json_line(self.producer_lifecycle))
        # confidence: one compact JSON line, only when present (same additive precedent)
        if self.confidence:
            fm.append("confidence: " + _compact_json_line(self.confidence))
        fm += [
            "---",
            "",
            self.body.rstrip() + "\n",
        ]
        return "\n".join(fm)

    @staticmethod
    def from_frontmatter(fm: dict[str, Any], body: str, path: str = "") -> "Note":
        def as_list(v): return v if isinstance(v, list) else ([v] if v else [])
        def as_bool(v): return v is True or str(v).strip().lower() == "true"

        name = fm.get("name", os.path.splitext(os.path.basename(path))[0])
        raw_lineages = fm.get("source_lineages", [])
        raw_producer_lifecycle = fm.get("producer_lifecycle")
        parsed_producer_lifecycle = _as_json_obj(raw_producer_lifecycle)
        producer_lifecycle_schema_valid = (
            "producer_lifecycle" not in fm
            or parsed_producer_lifecycle is not None
        )
        lineage_schema_valid = (
            isinstance(raw_lineages, list)
            and all(
                isinstance(value, str)
                and SOURCE_LINEAGE_RE.fullmatch(value) is not None
                for value in raw_lineages
            )
            and len(set(raw_lineages)) == len(raw_lineages)
        )
        note = Note(
            name=name,
            description=fm.get("description", ""),
            type=fm.get("type", "project"),
            domain=fm.get("domain", os.path.basename(os.path.dirname(path)) or "company"),
            project=str(fm.get("project", "") or ""),
            acl=as_list(fm.get("acl", [])),
            entities=as_list(fm.get("entities", [])),
            components=as_list(fm.get("components", [])),
            provenance=as_list(fm.get("provenance", [])),
            # A forged scalar must never be normalized into ownership authority.
            # Lint reports it, while every runtime consumer sees no owner.
            source_lineages=list(raw_lineages) if lineage_schema_valid else [],
            updated=fm.get("updated", _today()),
            body=body, path=path,
            title=fm.get("title", name),
            resource=fm.get("resource", ""),
            tags=as_list(fm.get("tags", [])),
            superseded=as_bool(fm.get("superseded", False)),
            structured=_as_json_obj(fm.get("structured")),
            # Preserve an explicit invalid value so trust-sensitive consumers
            # can distinguish it from a truly absent legacy field.  It still
            # fails closed through ``effective_state`` and the write door.
            state=(fm.get("state") if "state" in fm else None),
            confidence=_as_json_obj(fm.get("confidence")),
            approval=_as_json_obj(fm.get("approval")),
            source_retirement=_as_json_obj(fm.get("source_retirement")),
            producer_lifecycle=parsed_producer_lifecycle,
        )
        note._source_lineages_schema_valid = lineage_schema_valid
        note._producer_lifecycle_schema_valid = producer_lifecycle_schema_valid
        return note


def _suppressed_path(repo_root: str) -> str:
    """Path to the tombstone file: ``<repo_root>/state/suppressed.json``."""
    return os.path.join(repo_root, "state", "suppressed.json")


def load_suppressed(repo_root: str) -> set[str]:
    """Load the set of suppressed (tombstoned) note names/slugs.

    A tool that deletes a memory writes ``state/suppressed.json``; the engine
    only reads it.
    Suppressed notes are honored at BOTH choke points: ``load_notes`` skips them
    on read, and ``write_note`` no-ops on them so the distiller cannot resurrect
    a deleted memory. Format is a JSON list of names (slug/name), or an object
    with a top-level ``"suppressed"`` list.  A missing file yields an empty set
    for repositories that never delete anything.  An unreadable or malformed
    file fails closed: silently treating corruption as an empty tombstone set
    would resurrect every deleted memory at once.
    """
    path = _suppressed_path(repo_root)
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return set()
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"suppression state is unreadable: {path}") from exc
    if isinstance(data, dict):
        if "suppressed" not in data:
            raise RuntimeError(f"suppression state has no 'suppressed' list: {path}")
        data = data["suppressed"]
    if not isinstance(data, list):
        raise RuntimeError(f"suppression state is not a list: {path}")
    return {str(x) for x in data}


def _suppressed_slugs(suppressed: set[str]) -> set[str]:
    """Slug-normalize a tombstone set ONCE, for callers that test many names.

    Hoisted out of :func:`_is_suppressed` because the per-name form re-slugified
    every tombstone for every candidate, so with a large corpus and many
    tombstones `load_notes` spent most of a retrieval's wall clock on regex
    substitutions. Precomputing turns that O(notes x tombstones) into O(notes + tombstones).
    """
    return {slugify(item) for item in suppressed}


def _is_suppressed(note_name: str, suppressed: set[str],
                   suppressed_slugs: set[str] | None = None) -> bool:
    """True if either the note's name or its slug is tombstoned.

    `suppressed_slugs` is an optional precomputed :func:`_suppressed_slugs` result;
    pass it when testing many names against the same tombstone set. Omitting it is
    semantically identical, just slower.
    """
    if not suppressed:
        return False
    if note_name in suppressed:
        return True
    # Normalize BOTH sides.  Historical/UI tombstones can contain a display
    # name while deterministic producers use its slug; comparing a normalized
    # candidate to raw entries lets the same note resurrect under that slug.
    if suppressed_slugs is None:
        suppressed_slugs = _suppressed_slugs(suppressed)
    return slugify(note_name) in suppressed_slugs


def is_suppressed(note_name: str, repo_root: str) -> bool:
    """Return whether *note_name* is durably tombstoned in *repo_root*.

    This is the public producer guard for tools that manage more than a single
    note file (for example, a note plus custom indexes) and therefore cannot
    rely only on :func:`write_note`'s no-op behavior.
    """
    return _is_suppressed(note_name, load_suppressed(repo_root))


@contextmanager
def suppression_lock(repo_root: str):
    """Serialize tombstone mutations with every governed note producer.

    The lock file itself is disposable runtime state.  Holding this lock across
    the tombstone check and note write closes the delete-vs-regenerate race in
    which both operations could otherwise succeed from the same stale read.
    """
    state_dir = os.path.join(os.path.abspath(repo_root), "state")
    os.makedirs(state_dir, exist_ok=True)
    lock_path = os.path.join(state_dir, "suppressed.lock")
    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_notes(context_dir: str, *, verification_audit=None) -> list[Note]:
    # Suppression is honored at this choke point: any note whose name or slug is
    # tombstoned in <repo_root>/state/suppressed.json is skipped on read. The
    # repo root is the parent of context_dir. Missing file → empty set (no-op).
    repo_root = os.path.dirname(os.path.abspath(context_dir))
    suppressed = load_suppressed(repo_root)
    # Normalize the tombstone set ONCE — see _suppressed_slugs.
    suppressed_slugs = _suppressed_slugs(suppressed)
    notes: list[Note] = []
    for dirpath, _, files in os.walk(context_dir):
        for fn in files:
            if not fn.endswith(".md") or fn in ("INDEX.md", "index.md"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path) as f:
                fm, body = parse_frontmatter(f.read())
            if not fm:
                continue
            note = Note.from_frontmatter(fm, body, path)
            if _is_suppressed(note.name, suppressed, suppressed_slugs):
                continue  # tombstoned — invisible on read
            notes.append(note)
    bind_loaded_human_trust(notes, repo_root)
    bind_loaded_source_retirements(notes, repo_root)
    bind_loaded_producer_lifecycles(notes, repo_root)
    bind_loaded_auto_approvals(notes, repo_root, verification_audit)
    return notes


def index_notes_by_name(notes, *, operation: str = "note mutation") -> dict[str, Note]:
    """Build a name index only when every logical note name is unambiguous.

    A dict comprehension silently chooses one of two same-named files and can
    make a writer update the wrong domain/path.  Mutating workflows use this
    helper as a fail-closed boundary; lint can still inspect the raw list to
    report or quarantine the duplicates.
    """
    indexed: dict[str, Note] = {}
    collisions: dict[str, list[str]] = {}
    for note in notes:
        previous = indexed.get(note.name)
        if previous is None:
            indexed[note.name] = note
            continue
        collisions.setdefault(note.name, [previous.path or "<memory>"]).append(
            note.path or "<memory>"
        )
    if collisions:
        detail = "; ".join(
            f"{name}: {', '.join(paths)}"
            for name, paths in sorted(collisions.items())
        )
        raise ValueError(f"{operation} refused ambiguous note names: {detail}")
    return indexed


def forward_link_validation_error(note: Note, notes) -> str:
    """Return why writing *note* would violate wiki-link confidentiality.

    A link exposes its target's stable name even when retrieval would deny the
    target itself.  Validate the candidate as a replacement for any existing
    same-named note and cover both directions affected by the write:

    * every target named by the candidate must be unambiguous and safe for all
      readers of the candidate; and
    * an existing dangling link must not become a disclosure when a newly
      created (or newly restricted) target makes that name resolvable.

    Truly dangling links remain legal so the linter can report them without
    destroying author intent.  A self-link reveals no additional identity.
    """
    materialized = list(notes)
    same_name = [candidate for candidate in materialized
                 if candidate.name == note.name]
    if len(same_name) > 1:
        return f"note name {note.name!r} is ambiguous"

    # Only enforce inbound-link safety for writes that introduce a new note
    # or materially change the target's visibility. Existing notes with
    # unchanged ACL can keep their existing backlinks, even if an inbound
    # edge is already unsafe, because this rewrite cannot create new
    # disclosure by itself.
    existing_target = same_name[0] if same_name else None
    acl_changed = False
    if existing_target is not None and existing_target.acl != note.acl:
        acl_changed = True

    staged = [candidate for candidate in materialized
              if candidate.name != note.name]
    staged.append(note)
    groups: dict[str, list[Note]] = {}
    for candidate in staged:
        groups.setdefault(candidate.name, []).append(candidate)

    def edge_error(source: Note, target_name: str) -> str:
        if not target_name or target_name == source.name:
            return ""
        targets = groups.get(target_name, [])
        if len(targets) > 1:
            return (
                f"wiki-link target {target_name!r} from {source.name!r} "
                "is ambiguous"
            )
        if len(targets) == 1 and not acl_can_reveal(
            list(source.acl or []), list(targets[0].acl or []),
        ):
            return (
                f"wiki-link from {source.name!r} would reveal "
                f"restricted target {target_name!r}"
            )
        return ""

    # The candidate's outgoing links may have changed.
    for _relation, target_name in note.typed_links:
        error = edge_error(note, target_name)
        if error:
            return error

    # The candidate may also resolve an old orphan or narrow a target ACL.
    for source in staged:
        if source is note:
            continue
        for _relation, target_name in source.typed_links:
            if target_name != note.name:
                continue
            if existing_target is not None and not acl_changed:
                continue
            error = edge_error(source, target_name)
            if error:
                return error
    return ""


def write_note(
    note: Note, context_dir: str, *, _trust_capability=None,
    _retirement_capability=None, _producer_capability=None,
    _review_capability=None, _auto_capability=None,
) -> str:
    """Write a note to context/<domain>/<slug>.md (or context/<domain>/<project>/<slug>.md
    if `note.project` is set) and return its path.

    Honors suppression at the WRITE choke point: if the note's name or slug is
    tombstoned in ``state/suppressed.json`` this is a NO-OP — it returns the
    existing path (or the path it *would* have) without writing, so a distiller
    pass cannot resurrect a memory someone deleted.
    """
    repo_root = os.path.dirname(os.path.abspath(context_dir))
    if (
        note.state == "proposed"
        and _trust_capability is None
        and machine_authored(note)
    ):
        # Classification is caller-side policy, but the shared byte door is
        # the final place that can guarantee a high-stakes machine draft is
        # hidden *before* persistence. Central producers additionally exact-
        # enqueue through govern.persist_machine_note; the nightly reconciler
        # closes any crash between these two durable copies.
        import govern
        govern.prepare_machine(note, repo_root, requester="write-door")

    # Every proposed machine write reaches this choke point. If its current
    # content is not already bound to an explicit verdict, persist a skipped
    # binding before any bytes hit disk; a crash can never expose an unmarked
    # draft or resurrect a verdict for previous content.
    ensure_machine_verification_binding(
        note,
        allow_persisted_producer=producer_lifecycle_capability_sealed(
            note, _producer_capability,
        ),
    )

    with suppression_lock(repo_root):
        return _write_note_under_suppression_lock(
            note, context_dir, _trust_capability=_trust_capability,
            _retirement_capability=_retirement_capability,
            _producer_capability=_producer_capability,
            _review_capability=_review_capability,
            _auto_capability=_auto_capability,
        )


def _write_note_under_suppression_lock(
    note: Note, context_dir: str, *, _trust_capability=None,
    _retirement_capability=None, _producer_capability=None,
    _review_capability=None, _auto_capability=None,
) -> str:
    """Implementation for :func:`write_note`; caller holds suppression_lock."""
    def _safe_path_segment(value: str, field_name: str) -> str:
        value = str(value or "")
        if (not value or value in (".", "..") or os.path.isabs(value)
                or "/" in value or "\\" in value):
            raise ValueError(f"unsafe {field_name} path segment: {value!r}")
        return value

    domain = _safe_path_segment(note.domain, "domain")
    project = _safe_path_segment(note.project, "project") if note.project else ""
    context_abs = os.path.abspath(context_dir)
    domain_dir = os.path.join(context_dir, domain, project) if project \
                 else os.path.join(context_dir, domain)
    if os.path.commonpath((context_abs, os.path.abspath(domain_dir))) != context_abs:
        raise ValueError("note path escapes context directory")
    path = os.path.join(domain_dir, slugify(note.name) + ".md")

    repo_root = os.path.dirname(os.path.abspath(context_dir))
    if _is_suppressed(note.name, load_suppressed(repo_root)):
        note.path = path
        return path  # tombstoned — refuse to (re)write

    # Enforce the same schema at the write choke point that lint audits on
    # externally supplied/legacy Markdown.  This prevents a producer from
    # creating a new violation that a later pass would merely quarantine.
    name = str(note.name or "").strip()
    description = str(note.description or "").strip()
    note_type = str(note.type or "").strip()
    if not name or VALID_NAME_RE.fullmatch(name) is None:
        raise ValueError("note name must match [a-z0-9][a-z0-9-]*")
    if not description:
        raise ValueError("note description must be non-empty")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ValueError(
            f"note description exceeds {MAX_DESCRIPTION_CHARS} characters"
        )
    if note_type not in VALID_TYPES:
        raise ValueError(f"unsupported note type: {note_type!r}")
    if domain not in VALID_DOMAINS:
        raise ValueError(f"unsupported note domain: {domain!r}")
    if note.state is not None and note.state not in VALID_STATES:
        raise ValueError(f"unsupported note state: {note.state!r}")
    if note.superseded and note.state != "superseded":
        raise ValueError(
            "superseded: true requires explicit state: superseded; "
            "graph edges must not materialize target flags"
        )
    if getattr(note, "_source_lineages_schema_valid", True) is not True:
        raise ValueError("source_lineages frontmatter is malformed")
    if getattr(note, "_producer_lifecycle_schema_valid", True) is not True:
        raise ValueError("producer_lifecycle frontmatter is malformed")

    def _safe_line(value: Any, field_name: str) -> None:
        text = str(value or "")
        if any(ch in text for ch in ("\r", "\n", "\u0085", "\u2028", "\u2029")):
            raise ValueError(f"unsafe multiline frontmatter value for {field_name}")
        if any(ord(ch) < 32 and ch != "\t" for ch in text):
            raise ValueError(f"unsafe control character in frontmatter {field_name}")

    for field_name in ("name", "title", "description", "resource", "type", "domain",
                       "project", "updated", "state"):
        _safe_line(getattr(note, field_name, ""), field_name)
    for field_name in (
        "tags", "acl", "entities", "components", "provenance", "source_lineages",
    ):
        values = getattr(note, field_name, [])
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise ValueError(f"frontmatter list {field_name} must contain only strings")
        for item in values:
            _safe_line(item, field_name)
            if "," in str(item):
                raise ValueError(f"comma is not representable in frontmatter list {field_name}")
    if len(set(note.source_lineages)) != len(note.source_lineages):
        raise ValueError("source_lineages must not contain duplicates")
    if any(SOURCE_LINEAGE_RE.fullmatch(value) is None for value in note.source_lineages):
        raise ValueError("source_lineages contains a non-canonical mutable source key")

    # Wiki-link text is itself metadata: a readable source note must never
    # reveal the stable name of a target its readers cannot retrieve.  Keep
    # this at the shared write door so authored pages, distillation, compact,
    # file-back, and future producers all inherit the same invariant.
    link_error = forward_link_validation_error(note, load_notes(context_dir))
    if link_error:
        raise ValueError(link_error)

    # Logical names are corpus-global identities.  The destination-path check
    # below catches only same-folder slug collisions; without this scan the
    # same name can exist in two domains and every name-indexed writer/read
    # becomes ambiguous.  Valid names map to one deterministic basename, so we
    # only open matching candidates rather than reading the whole corpus.
    target_abs = os.path.abspath(path)
    candidate_filename = os.path.basename(path)
    if os.path.isdir(context_dir):
        for candidate_dir, _dirs, filenames in os.walk(context_dir):
            if candidate_filename not in filenames:
                continue
            candidate = os.path.abspath(os.path.join(candidate_dir, candidate_filename))
            if candidate == target_abs:
                continue
            try:
                with open(candidate, encoding="utf-8") as candidate_file:
                    candidate_fm, _ = parse_frontmatter(candidate_file.read())
            except (OSError, UnicodeError):
                continue
            if str(candidate_fm.get("name", "") or "").strip() == name:
                raise FileExistsError(
                    f"refusing duplicate logical note name {name!r}: "
                    f"already exists at {candidate}"
                )

    existing_text = None
    existing_note = None
    if os.path.exists(path):
        # Governance/trust capabilities bind the exact on-disk preimage bytes.
        # Disable universal-newline translation here so a CRLF source body is
        # checked against the same text that the capability sealed.
        with open(path, encoding="utf-8", newline="") as existing_file:
            existing_text = existing_file.read()
            existing_fm, existing_body = parse_frontmatter(existing_text)
        existing_name = existing_fm.get("name")
        if not existing_name:
            raise FileExistsError(f"refusing to overwrite unparseable existing note at {path}")
        if existing_name and existing_name != note.name:
            raise FileExistsError(
                f"refusing slug collision: {note.name!r} would overwrite {existing_name!r} at {path}"
            )
        existing_note = Note.from_frontmatter(existing_fm, existing_body, path)

    human_transition_write = human_trust_write_authorized(
        note, repo_root, _trust_capability, existing_text=existing_text,
    )
    auto_transition_write = auto_approval_write_authorized(
        note, repo_root, _auto_capability, existing_text=existing_text,
    )
    # A mandatory review hold is monotonic across every ordinary proposed
    # rewrite.  Confidence metadata is not part of the factual fingerprint, so
    # without this write-door rule a caller could clear the block while keeping
    # the queued content identity stale.  Only the sealed signed-human
    # lifecycle transition above may remove it; content edits rebind the hold
    # to their new fingerprint and must be requeued by their workflow.  A
    # sealed machine auto-approval preserves the durable hold rather than
    # rebinding it, and releases it only at the read gate while the config
    # gate remains on.
    held_class = None
    if (
        existing_note is not None
        and existing_note.state == "proposed"
        and not human_transition_write
        and not auto_transition_write
    ):
        held_class = governance_hold_class(existing_note)
        if held_class is not None:
            apply_governance_hold(note, held_class)

    markdown = note.to_markdown()
    if (
        held_class is not None
        and note.state == "proposed"
        and existing_text != markdown
        and not governance_rewrite_authorized(
            note,
            existing_text=existing_text,
            replacement_text=markdown,
            review_class=held_class,
            capability=_review_capability,
        )
    ):
        raise PermissionError(
            "review-held proposal rewrite requires an exact governance "
            "capability and requeue"
        )
    parsed, _body = parse_frontmatter(markdown)
    round_trip = Note.from_frontmatter(parsed, _body, path)
    for field_name in (
        "name", "domain", "acl", "provenance", "source_lineages", "state",
        "approval", "source_retirement",
        "producer_lifecycle",
    ):
        if getattr(round_trip, field_name) != getattr(note, field_name):
            raise ValueError(f"frontmatter round-trip mismatch for {field_name}")

    sensitive_states = {"verified", "superseded"}
    transition_write = human_transition_write or auto_transition_write
    transition_write = transition_write or source_lifecycle_write_authorized(
        note, repo_root, _retirement_capability, existing_note=existing_note,
    )
    transition_write = transition_write or producer_lifecycle_write_authorized(
        note, repo_root, _producer_capability, existing_note=existing_note,
    )
    if not transition_write:
        # Ordinary writers may replay identical historical bytes, but cannot
        # mint a trust state, mutate trusted/historical content in place, or
        # inherit approval after changing any answer-relevant field.
        if existing_text is not None and existing_text == markdown:
            note.path = path
            return path
        if note.state in sensitive_states or (
            existing_note is not None and existing_note.state in sensitive_states
        ):
            raise PermissionError(
                "verified/superseded note writes require an authorized, "
                "content-bound trust transition"
            )

    os.makedirs(domain_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".remembrancer-note-", suffix=".tmp", dir=domain_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(markdown)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(
            domain_dir,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    note.path = path
    return path


MAX_INDEX_KEYS = 16

_INDEX_KEY_WORD = re.compile(r"[a-z][a-z\-']{2,}")
_INDEX_KEY_HEX = re.compile(r"^[0-9a-f]{6,}$")
_INDEX_KEY_STOP = frozenset(
    "the a an and or of to in for on with at by from is are was were be been "
    "this that these those it its as if then than so we our you your they their "
    "he she his her i me my not no new update updated note notes http https com "
    "www org net io ai".split()
)


def _index_key_terms(text: str) -> list[str]:
    """Raw-word vocabulary for index keys (never pre-stemmed).

    Query-time BM25 applies its own stemmer, so keys must be raw words —
    pre-stemmed tokens would match nothing. Filters out ids, SHAs, and
    digit-carrying tokens a human would never type as a query term.
    """
    out = []
    for t in _INDEX_KEY_WORD.findall((text or "").lower()):
        if (t in _INDEX_KEY_STOP or _INDEX_KEY_HEX.match(t)
                or any(c.isdigit() for c in t)):
            continue
        out.append(t)
    return out


def compute_index_keys(notes, max_keys: int = MAX_INDEX_KEYS) -> dict[str, list[str]]:
    """Pick each note's most distinctive body terms for index-first retrieval.

    Body text is deliberately absent from the per-domain index rows (it would
    bloat the tracked files), so stage-1 ranking could only see name,
    description, and tags — a fact stated only in a note's body was unfindable
    (measured body-fact R@5 15.1% vs 98.5% for description terms on a
    production corpus). Carrying a few TF-IDF-selected body terms in the index row closes
    most of that gap at ~150 bytes per note. This is the LongMemEval
    index-side key-expansion result (+9.4% recall) adapted to BM25: keys
    AUGMENT the existing document rather than replacing it (keyphrase-only
    indexing is measured to be much worse).

    Terms already present in the indexed name/description/tags are excluded so
    keys carry only additive signal. Returns {note.path: [term, ...]}.
    """
    import math
    from collections import Counter

    notes = [n for n in notes if getattr(n, "path", None)]
    df: Counter[str] = Counter()
    per_note_terms: dict[str, list[str]] = {}
    for n in notes:
        terms = _index_key_terms((getattr(n, "body", "") or "")[:4000])
        per_note_terms[n.path] = terms
        df.update(set(terms))
    total = max(1, len(notes))
    out: dict[str, list[str]] = {}
    for n in notes:
        already = set(_index_key_terms(
            f"{n.name} {getattr(n, 'description', '') or ''} "
            f"{' '.join(str(t) for t in (getattr(n, 'tags', None) or []))}"))
        tf = Counter(t for t in per_note_terms[n.path] if t not in already)
        scored = sorted(tf, key=lambda t: -(tf[t] * math.log(1 + total / (1 + df[t]))))
        out[n.path] = scored[:max_keys]
    return out


def render_index_artifacts(context_dir: str) -> dict[str, str]:
    """Render every derived index without changing the filesystem.

    Keeping rendering separate from persistence gives producers a genuine
    read-only drift check and ensures no one reimplements the v2 integrity
    metadata with a weaker bespoke index format.
    """
    notes = sorted(load_notes(context_dir), key=lambda n: (n.domain, n.name))
    keys_by_path = compute_index_keys(notes)
    artifacts: dict[str, str] = {}

    # --- Legacy flat INDEX.md (back-compat) ---
    flat_lines = [
        "# Remembrancer context index",
        "",
        "One line per note — the cheap thing retrieval loads first to decide what to pull.",
        "Format: `- [[name]] — description · type · acl:[...] · entities:[...]`",
        "",
    ]
    cur_domain = None
    for n in notes:
        if n.domain != cur_domain:
            cur_domain = n.domain
            flat_lines.append(f"\n## {cur_domain}\n")
        ents = (" · entities:" + str(n.entities)) if n.entities else ""
        flat_lines.append(f"- [[{n.name}]] — {n.description} · {n.type} · acl:{n.acl}{ents}")
    if not notes:
        flat_lines.append("_(empty — populated by the first distillation pass)_")
    flat_path = os.path.join(context_dir, "INDEX.md")
    artifacts[flat_path] = "\n".join(flat_lines) + "\n"

    # --- Group notes by domain ---
    from collections import defaultdict
    by_domain: dict[str, list[Note]] = defaultdict(list)
    for n in notes:
        by_domain[n.domain].append(n)

    def index_metadata_comment(note: Note) -> tuple[str, str]:
        """Opaque v2 row metadata consumed by the read-only retrieval path.

        Base64 keeps arbitrary Unicode/descriptions from accidentally closing
        the HTML comment.  The visible Markdown row remains unchanged.
        """
        rel_path = os.path.relpath(note.path, context_dir).replace(os.sep, "/")
        with open(note.path, "rb") as source:
            raw_note = source.read()
        verification = ""
        verification_content = ""
        verification_audit_id = ""
        verification_method = ""
        verification_producer = ""
        verification_source_fingerprint = ""
        verification_grounded = False
        verification_reason = ""
        verification_checked = 0
        verification_supported = 0
        verification_unsupported = 0
        if isinstance(note.confidence, dict):
            verification_block = note.confidence.get("verification")
            if isinstance(verification_block, dict):
                status = verification_block.get("status")
                if isinstance(status, str):
                    verification = status
                fingerprint = verification_block.get("content_fingerprint")
                if isinstance(fingerprint, str):
                    verification_content = fingerprint
                audit_id = verification_block.get("audit_id")
                if isinstance(audit_id, str):
                    verification_audit_id = audit_id
                method = verification_block.get("method")
                if isinstance(method, str):
                    verification_method = method
                producer = verification_block.get("producer")
                if isinstance(producer, str):
                    verification_producer = producer
                source_fingerprint = verification_block.get("source_fingerprint")
                if isinstance(source_fingerprint, str):
                    verification_source_fingerprint = source_fingerprint
                if isinstance(verification_block.get("grounded"), bool):
                    verification_grounded = verification_block["grounded"]
                reason = verification_block.get("reason")
                if isinstance(reason, str):
                    verification_reason = reason
                for key in ("checked", "supported", "unsupported"):
                    value = verification_block.get(key)
                    if isinstance(value, int) and value >= 0:
                        if key == "checked":
                            verification_checked = value
                        elif key == "supported":
                            verification_supported = value
                        else:
                            verification_unsupported = value
        metadata = {
            "acl": list(note.acl or []),
            "description": note.description,
            "entities": list(note.entities or []),
            "provenance": list(note.provenance or []),
            "source_lineages": list(note.source_lineages or []),
            "tags": list(note.tags or []),
            "index_keys": list(keys_by_path.get(note.path, [])),
            "name": note.name,
            "path": rel_path,
            # Preserve whether trust was explicit. Retrieval treats both an
            # absent value and explicit ``proposed`` as proposed for ranking.
            # Empty string preserves a truly absent legacy state without
            # emitting JSON null (which cannot satisfy the scalar schema).
            # Explicit invalid strings remain observable and fail closed.
            "state": note.state or "",
            "approval": dict(note.approval or {}),
            "source_retirement": dict(note.source_retirement or {}),
            "producer_lifecycle": dict(note.producer_lifecycle or {}),
            # Embedded quality status lets index-first filtering use the same
            # gate as the authoritative full scan without opening every body.
            # The append-only audit remains a read-time fallback for old notes.
            "verification": verification,
            "verification_content_fingerprint": verification_content_fingerprint(note),
            "embedded_verification_content_fingerprint": verification_content,
            "verification_audit_id": verification_audit_id,
            "verification_method": verification_method,
            "verification_producer": verification_producer,
            "verification_source_fingerprint": verification_source_fingerprint,
            "verification_grounded": verification_grounded,
            "verification_reason": verification_reason,
            "verification_checked": verification_checked,
            "verification_supported": verification_supported,
            "verification_unsupported": verification_unsupported,
            "superseded": bool(note.superseded),
            "supersedes": [target for relation, target in note.typed_links
                           if relation == "supersedes"],
            "title": note.title or note.name,
            "updated": note.updated or "",
            "content_sha256": hashlib.sha256(raw_note).hexdigest(),
            "size": len(raw_note),
        }
        raw = json.dumps(metadata, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
        token = base64.urlsafe_b64encode(raw).decode("ascii")
        return f"<!-- remembrancer-index-v2:{token} -->", token

    # --- Per-domain index.md ---
    # Re-render generated indexes for domains that have become empty after a
    # deletion/tombstone, so stale rows do not disable index-first forever.
    generated_domains: set[str] = set()
    try:
        for entry in os.scandir(context_dir):
            if not entry.is_dir(follow_symlinks=False):
                continue
            old_index = os.path.join(entry.path, "index.md")
            try:
                with open(old_index, encoding="utf-8") as old:
                    if "<!-- remembrancer-index-v2 -->" in old.read():
                        generated_domains.add(entry.name)
            except OSError:
                continue
    except OSError:
        pass

    for domain in sorted(set(by_domain) | generated_domains):
        domain_notes = by_domain.get(domain, [])
        domain_dir = os.path.join(context_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        d_lines = [
            f"# {domain} index",
            "",
            f"Notes in the `{domain}` domain.",
            "",
            "<!-- remembrancer-index-v2 -->",
            "",
        ]
        row_tokens: list[str] = []
        for n in sorted(domain_notes, key=lambda x: (x.project or "", x.name)):
            slug = slugify(n.name)
            title = n.title or n.name
            # When the note lives in a project
            # subfolder, the link must include the project segment.
            link_path = f"/{domain}/{n.project}/{slug}.md" if n.project else f"/{domain}/{slug}.md"
            # Group notes by project subfolder in
            # the index when any project-subfolder notes exist in this domain.
            metadata_comment, token = index_metadata_comment(n)
            row_tokens.append(token)
            d_lines.append(
                f"- [{title}]({link_path}) — {n.description} · {n.type} "
                f"{metadata_comment}"
            )
        rows_digest = hashlib.sha256("\n".join(row_tokens).encode("ascii")).hexdigest()
        d_lines.append(f"<!-- remembrancer-index-v2-digest:{rows_digest} -->")
        domain_idx_path = os.path.join(domain_dir, "index.md")
        artifacts[domain_idx_path] = "\n".join(d_lines) + "\n"

    # --- Root index.md listing domains with counts ---
    root_lines = [
        "# Remembrancer context root index",
        "",
        "Domains and note counts.",
        "",
    ]
    for domain, domain_notes in sorted(by_domain.items()):
        count = len(domain_notes)
        root_lines.append(f"- [{domain}](/{domain}/index.md) — {count} note{'s' if count != 1 else ''}")
    if not by_domain:
        root_lines.append("_(empty — populated by the first distillation pass)_")
    root_idx_path = os.path.join(context_dir, "index.md")
    artifacts[root_idx_path] = "\n".join(root_lines) + "\n"

    return artifacts


def _atomic_write_text(path: str, content: str) -> None:
    """Durably replace one derived text artifact without exposing truncation."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=".remembrancer-index-", suffix=".tmp", dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        try:
            directory_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def regen_index(context_dir: str) -> str:
    """Atomically regenerate all corpus indexes from note frontmatter.

    The files are derived projections. Each replacement is atomic; if a
    process dies between files, the next idempotent pass repairs the remaining
    drift without any reader observing a truncated index.
    """
    artifacts = render_index_artifacts(context_dir)
    for path, content in artifacts.items():
        try:
            with open(path, encoding="utf-8") as existing:
                if existing.read() == content:
                    continue
        except FileNotFoundError:
            pass
        _atomic_write_text(path, content)

    return os.path.join(context_dir, "INDEX.md")
