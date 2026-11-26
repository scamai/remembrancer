"""Entity resolution — the customer-360 axis (a note's `entities`).

The customer registry (registry/customers.yml) is the ENTITY authority, exactly
as policy/governance.yml is the ACL authority. We resolve mentions / email
addresses / Drive folders / Slack channels in an event's text+provenance to a
stable `customer:<slug>` id — deterministically, never by guess — so "Acme",
"Acme Corp" and @acme.com all collapse to one entity.

Stdlib-only. PyYAML is used if available; otherwise a tiny line parser handles
the known registry shape (copying the fallback pattern from remembrancerlib/acl.py).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import govern
from remembrancerlib.registry_trust import (
    registry_field_verified,
    registry_record_quarantine_reasons,
)

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


def _strip(s: str) -> str:
    return s.strip().strip('"\'')


def _norm_name(s: str) -> str:
    """Normalize a display name for matching: lowercase alnum tokens, single-spaced.
    'Jane Doe (MIT)' → 'jane doe mit'."""
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))


# Generic name tokens that must NOT drive a merge-org proposal on their own — they
# collide across unrelated orgs (e.g. many orgs contain "data"/"tech"/"labs").
_GENERIC_ORG_TOKENS = {
    "data", "tech", "labs", "mail", "group", "global", "cloud", "systems", "digital",
    "media", "capital", "ventures", "partners", "solutions", "software", "network",
    "security", "health", "financial", "services", "consulting", "technologies",
    "international", "holdings", "studio", "studios", "world", "online", "company",
}


def _registrable_root(domain: str) -> str:
    """Coarse registrable root for merge/look-alike detection: the label just left
    of the public suffix. 'mail.example.com'→'example', 'example.co.kr'→'example',
    'example-mail.com'→'example-mail'. Deterministic and good enough to *propose*
    (never to auto-merge)."""
    parts = [p for p in (domain or "").lower().split(".") if p]
    if len(parts) < 2:
        return domain or ""
    # common two-label public suffixes (co.kr, com.cn, co.uk, …)
    _MULTI = {"co", "com", "org", "net", "gov", "edu", "ac"}
    if len(parts) >= 3 and parts[-2] in _MULTI and len(parts[-1]) == 2:
        return parts[-3]
    return parts[-2]


def emit_proposals(proposals: "list[dict[str, Any]]", path: str,
                   source_event: str | None = None, today: str | None = None) -> int:
    """Append attribution proposals to the entity_unresolved queue, de-duped by
    (kind, identifier). Backward-compatible superset of the existing
    {identifier, first_seen, count} record — existing consumers keep reading
    `identifier`; reviewers get kind/evidence/candidate. Never raises."""
    if not proposals:
        return 0
    try:
        existing: dict[tuple, dict] = {}
        order: list[tuple] = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    key = (r.get("kind", ""), r.get("identifier", ""))
                    if key not in existing:
                        order.append(key)
                    existing[key] = r
        wrote = 0
        changed = False
        for p in proposals:
            key = (p.get("kind", ""), p.get("identifier", ""))
            if key in existing:
                rec = existing[key]
                # Full meeting/calendar histories are deliberately replayed so
                # attribution can improve when the registry changes.  A replay
                # of the *same* event is not new evidence and must not inflate
                # the review queue or rewrite it every digest run.  Preserve
                # the old singular field for compatibility while recording all
                # distinct observations needed for exact idempotence.
                known_sources = {
                    str(value) for value in (rec.get("source_events") or []) if value
                }
                if rec.get("source_event"):
                    known_sources.add(str(rec["source_event"]))
                if source_event and source_event not in known_sources:
                    rec["source_events"] = sorted(known_sources | {source_event})
                    rec["count"] = int(rec.get("count", 1)) + 1
                    changed = True
                elif source_event is None:
                    # Callers without a stable source id retain the historical
                    # occurrence-count behavior; they cannot promise replay
                    # idempotence.
                    rec["count"] = int(rec.get("count", 1)) + 1
                    changed = True
            else:
                rec = {"identifier": p.get("identifier", ""), "kind": p.get("kind", ""),
                       "evidence": p.get("evidence", {}), "candidate": p.get("candidate"),
                       "first_seen": today or "", "count": 1}
                if p.get("ambiguous"):
                    rec["ambiguous"] = True
                if source_event:
                    rec["source_event"] = source_event
                existing[key] = rec
                order.append(key)
                wrote += 1
                changed = True
        if not changed:
            return wrote
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Unique temp name per process: parallel worker processes write this
        # queue concurrently, and a shared fixed
        # ``path + ".tmp"`` made one child's os.replace() steal the other's
        # temp file (FileNotFoundError spam). A concurrently lost update is
        # acceptable here — the queue is advisory and deliberately replayed —
        # but a crashed write is not.
        import tempfile
        fd, tmp = tempfile.mkstemp(
            prefix=".entity-unresolved-", suffix=".tmp",
            dir=os.path.dirname(path) or ".",
        )
        try:
            with os.fdopen(fd, "w") as f:
                for key in order:
                    f.write(json.dumps(existing[key], ensure_ascii=False) + "\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return wrote
    except Exception as ex:  # never break attribution on a queue-write failure
        print(f"[entities] emit_proposals failed: {ex}", file=__import__("sys").stderr)
        return 0


def _parse_registry_fallback(path: str, top_key: str = "customers") -> list[dict[str, Any]]:
    """Minimal parser for a registry split file without PyYAML.

    Understands the documented shape (default `top_key="customers"`, but the
    same block grammar covers people.yml / orgs.yml — pass their top-level key):

        customers:
          - id: customer:acme
            name: Acme Corp
            aliases: ["Acme", "Acme Corporation"]
            email_domains: ["acme.com"]
            drive_folders: ["Customers/Acme"]
            slack_channels: ["acme-deal"]
            acl: ["customers"]

    Inline `[a, b]` lists and bare scalars are both handled; multi-line `- item`
    list blocks are handled too. Anything exotic should use PyYAML (which is the
    happy path); this only exists so the engine never hard-depends on a dep.
    """
    customers: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    pending_list_key: str | None = None
    in_customers = False

    def parse_inline_list(v: str) -> list[str]:
        return [_strip(x) for x in v[1:-1].split(",") if _strip(x)]

    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())

            if stripped.rstrip().endswith(":") and stripped[:-1] == top_key:
                in_customers = True
                continue
            if not in_customers:
                continue

            # New customer record begins with `- id: ...` (or just `-`).
            if stripped.startswith("- "):
                # If we were collecting a multi-line list, that list ends here
                # only if this dash is at the *list* indent. Heuristic: a record
                # dash carries a `key:` after it.
                rest = stripped[2:]
                if pending_list_key and cur is not None and ":" not in rest:
                    cur.setdefault(pending_list_key, []).append(_strip(rest))
                    continue
                # start a new customer
                cur = {}
                customers.append(cur)
                pending_list_key = None
                if ":" in rest:
                    k, v = rest.split(":", 1)
                    k, v = k.strip(), v.strip()
                    if v.startswith("[") and v.endswith("]"):
                        cur[k] = parse_inline_list(v)
                    elif v:
                        cur[k] = _strip(v)
                    else:
                        pending_list_key = k
                continue

            if cur is None:
                continue

            if ":" in stripped:
                k, v = stripped.split(":", 1)
                k, v = k.strip(), v.strip()
                if v.startswith("[") and v.endswith("]"):
                    cur[k] = parse_inline_list(v)
                    pending_list_key = None
                elif v:
                    cur[k] = _strip(v)
                    pending_list_key = None
                else:
                    pending_list_key = k
            elif stripped.startswith("- ") and pending_list_key:
                cur.setdefault(pending_list_key, []).append(_strip(stripped[2:]))
    return customers


def _load_registry_records(path: str, top_key: str) -> list[dict[str, Any]]:
    """Load a `<top_key>:` list from a registry split file (PyYAML when
    available, stdlib fallback otherwise). [] on a missing/malformed file."""
    if not os.path.exists(path):
        return []
    if _HAVE_YAML:
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return []
        if isinstance(data, dict):
            return [r for r in (data.get(top_key, []) or []) if isinstance(r, dict)]
        return []
    return _parse_registry_fallback(path, top_key)


def _load_customers(path: str) -> list[dict[str, Any]]:
    return _load_registry_records(path, "customers")


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


# ---------------------------------------------------------------------------
# Split-registry loader: `RegistryTypes` and
# `Canonicalizer` both read registry/{people,orgs,systems,noise}.yml when
# registry/orgs.yml exists (produced by splitting entities.yml), falling
# back to the monolithic registry/entities.yml otherwise (so nothing
# breaks mid-migration). Both classes call this once in `__init__`; it is
# never re-read per lookup.
# ---------------------------------------------------------------------------

def _split_dir_for(registry_path: str) -> str:
    d = os.path.dirname(registry_path)
    return d if d else "."


def _read_split_doc(registry_dir: str, filename: str, key: str) -> list[Any]:
    path = os.path.join(registry_dir, filename)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
    except Exception:
        return []
    if isinstance(doc, dict):
        return list(doc.get(key) or [])
    return []


def _load_split_files(registry_dir: str) -> dict[str, Any] | None:
    """None -> no split files (caller should fall back to entities.yml).

    The split is considered "present" once registry/orgs.yml exists —
    people/systems/noise are read best-effort even if
    momentarily missing (each just contributes nothing).
    """
    orgs_path = os.path.join(registry_dir, "orgs.yml")
    if not _HAVE_YAML or not os.path.exists(orgs_path):
        return None
    noise_doc: dict[str, Any] = {}
    noise_path = os.path.join(registry_dir, "noise.yml")
    if os.path.exists(noise_path):
        try:
            with open(noise_path) as f:
                noise_doc = yaml.safe_load(f) or {}
        except Exception:
            noise_doc = {}
    if not isinstance(noise_doc, dict):
        noise_doc = {}
    return {
        "people": _read_split_doc(registry_dir, "people.yml", "people"),
        "orgs": _read_split_doc(registry_dir, "orgs.yml", "orgs"),
        "systems": _read_split_doc(registry_dir, "systems.yml", "systems"),
        "noise": noise_doc,
    }


# Relationship-type vocabulary (old section names == new orgs' relationship[0])
# mapped to the NEW kind vocabulary for `kind_of()`. Identical in split mode
# and fallback mode, since `type_of()` returns this same vocabulary either way.
_KIND_PREFIXES = frozenset({"person", "org", "system", "component", "initiative"})

_RELATIONSHIP_TYPE_TO_KIND = {
    "person": "person", "system": "system", "self": "org",
    "vendor": "org", "investor": "org", "partner": "org", "advisor": "org",
    "champion": "org", "customer": "org", "friend": "org", "academic": "org",
    "recruiting": "org", "prospect": "org", "org": "org",
}


def configured_internal_domains(cfg: Any) -> list[str]:
    """The org's own email domains for attendee resolution, from a loaded
    ``config.json``: ``entities.internal_domains``, else
    ``governance.internal_domains``, else none.

    The ``entities`` key exists for a deployment whose resolver must keep a
    different list from governance's. With no domain configured, every
    attendee is resolved as an outsider.
    """
    if not isinstance(cfg, dict):
        return []
    for section in ("entities", "governance"):
        block = cfg.get(section)
        if isinstance(block, dict) and "internal_domains" in block:
            return list(govern.domain_list(block["internal_domains"]))
    return []


def internal_domains_for_registry(registry_path: str) -> list[str]:
    """:func:`configured_internal_domains` for the memory root that holds
    ``registry_path``. A registry file lives at ``<root>/registry/<file>``, and
    that root's ``config.json`` names the org's domains. No readable config
    names none."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(registry_path)))
    try:
        with open(os.path.join(root, "config.json"), encoding="utf-8") as f:
            return configured_internal_domains(json.load(f))
    except (OSError, ValueError):
        return []


class EntityResolver:
    """Deterministic resolver: text + provenance -> [customer:<slug>, ...].

    Matching is case-insensitive and word-boundary-aware for aliases (so "Acme"
    does not match "acmeson"), substring for email domains and folder/channel
    paths. Built once per distillation pass and reused across all events.

    ``internal_domains`` defaults to the ones configured for the registry's
    own memory root (:func:`internal_domains_for_registry`).
    """

    def __init__(self, registry_path: str = "registry/customers.yml",
                 internal_domains: list[str] | None = None):
        self.registry_path = registry_path
        self.customers = _load_customers(registry_path)
        # Internal (self) email domain(s) — attendees on these are never external
        # signal (an internal all-hands must stay a plain note, never enrich).
        if internal_domains is None:
            internal_domains = internal_domains_for_registry(registry_path)
        self._internal_domains = {
            d.strip().lower() for d in internal_domains if d and d.strip()
        }
        # Pre-compile per-customer matchers.
        self._alias_res: list[tuple[str, re.Pattern[str]]] = []
        self._domains: list[tuple[str, list[str]]] = []
        self._folders: list[tuple[str, list[str]]] = []
        self._channels: list[tuple[str, list[str]]] = []
        self._acl: dict[str, list[str]] = {}
        # Attendee-resolution lookups: a full
        # email → customer, then a bare domain → customer.
        self._customer_email: dict[str, str] = {}
        self._customer_domain: dict[str, str] = {}
        for c in self.customers:
            cid = str(c.get("id", "")).strip()
            if not cid:
                continue
            self._acl[cid] = _as_list(c.get("acl"))
            names = _as_list(c.get("name")) + _as_list(c.get("aliases"))
            words = sorted({n for n in names if n}, key=len, reverse=True)
            if words:
                alt = "|".join(re.escape(w) for w in words)
                self._alias_res.append(
                    (cid, re.compile(r"(?<![\w@])(?:" + alt + r")(?![\w])", re.IGNORECASE))
                )
            doms = [d.lower() for d in _as_list(c.get("email_domains"))]
            self._domains.append((cid, doms))
            self._folders.append((cid, [d.lower() for d in _as_list(c.get("drive_folders"))]))
            self._channels.append((cid, [d.lower() for d in _as_list(c.get("slack_channels"))]))
            for d in doms:
                if d:
                    self._customer_domain.setdefault(d, cid)
            # Champion/contact addresses often live on a free-mail domain that
            # domain-matching would miss.
            for contact in (c.get("contacts") or []):
                if isinstance(contact, dict):
                    em = str(contact.get("email", "")).strip().lower()
                    if "@" in em:
                        self._customer_email.setdefault(em, cid)

        # --- people.yml / orgs.yml (attendee reconciliation) -----------------
        reg_dir = os.path.dirname(registry_path) or "."
        self._person_email: dict[str, str] = {}   # full email -> person:<slug>
        self._ambiguous_person_emails: set[str] = set()
        self._person_internal: set[str] = set()    # ids flagged internal:true
        self._person_org: dict[str, str] = {}      # person:<slug> -> works_at org:<slug>
        self._person_has_email: set[str] = set()   # persons with >=1 email on file
        self._name_persons: dict[str, list[str]] = {}  # normalized name -> [person ids]
        for p in _load_registry_records(os.path.join(reg_dir, "people.yml"), "people"):
            if registry_record_quarantine_reasons(p, "person"):
                continue
            pid = str(p.get("id", "")).strip()
            if not pid:
                continue
            # Positive internal/self/ACL declarations are safety boundaries:
            # preserve their restrictive effect even before relationship fact
            # review. Treating an unreviewed employee as external is unsafe.
            if bool(p.get("internal")):
                self._person_internal.add(pid)
            wa = str(p.get("works_at", "")).strip()
            if wa.startswith("org:"):
                self._person_org[pid] = wa
            emails = [e.strip().lower() for e in _as_list(p.get("emails")) if "@" in e]
            if emails:
                self._person_has_email.add(pid)
            for e in emails:
                _register_unambiguous(
                    self._person_email, self._ambiguous_person_emails, e, pid,
                )
            names = [p.get("name", ""), *_as_list(p.get("aliases"))]
            for nm in names:
                key = _norm_name(nm)
                if key:
                    self._name_persons.setdefault(key, [])
                    if pid not in self._name_persons[key]:
                        self._name_persons[key].append(pid)
        self._org_domain: dict[str, str] = {}      # domain -> org:<slug>
        self._ambiguous_org_domains: set[str] = set()
        self._org_self: set[str] = set()           # ids whose relationship is self
        self._org_type: dict[str, str] = {}        # org:<slug> -> relationship type
        self._root_orgs: dict[str, list[str]] = {} # registrable-root -> [org ids]  (merge detection)
        self._nametoken_orgs: dict[str, list[str]] = {}  # org name/domain token -> [org ids]
        for o in _load_registry_records(os.path.join(reg_dir, "orgs.yml"), "orgs"):
            if registry_record_quarantine_reasons(o, "org"):
                continue
            oid = str(o.get("id", "")).strip()
            if not oid:
                continue
            raw_rel = [r.lower() for r in _as_list(o.get("relationship"))]
            rel = raw_rel if registry_field_verified(o, "relationship") else []
            if "self" in raw_rel:
                self._org_self.add(oid)
            typ = next((r for r in rel if r != "self"), None)
            if typ:
                self._org_type[oid] = typ
            for tok in _norm_name(str(o.get("name", ""))).split():
                if len(tok) >= 4:              # skip short/ambiguous tokens
                    self._nametoken_orgs.setdefault(tok, [])
                    if oid not in self._nametoken_orgs[tok]:
                        self._nametoken_orgs[tok].append(oid)
            for d in _as_list(o.get("email_domains")):
                d = d.strip().lower()
                if d:
                    _register_unambiguous(
                        self._org_domain, self._ambiguous_org_domains, d, oid,
                    )
                    root = _registrable_root(d)
                    self._root_orgs.setdefault(root, [])
                    if oid not in self._root_orgs[root]:
                        self._root_orgs[root].append(oid)

        # free-mail / infra domains → never proposed as an org (registry noise authority)
        self._noise_domains: set[str] = set()
        try:
            import yaml as _yaml  # type: ignore
            nz = _yaml.safe_load(open(os.path.join(reg_dir, "noise.yml"))) or {}
            for k in ("free_email", "infra"):
                for d in (nz.get(k) or []):
                    self._noise_domains.add(str(d).strip().lower())
        except Exception:
            self._noise_domains = {"gmail.com", "googlemail.com", "outlook.com",
                                   "hotmail.com", "yahoo.com", "icloud.com", "qq.com"}

    def acl_for(self, entity_id: str) -> list[str]:
        return list(self._acl.get(entity_id, []))

    def resolve_attendees(self, emails: list[str]) -> list[str]:
        """Backward-compatible flat-list wrapper over `resolve_attendees_deep`.
        Accepts bare emails (names default to "") and returns just the tagged
        canonical entity ids — the original flat-list contract, unchanged for
        existing callers."""
        pairs = [("", e) for e in (emails or [])]
        return self.resolve_attendees_deep(pairs)["entities"]

    def resolve_attendees_deep(self, attendees: "list[tuple[str, str]]") -> "dict[str, Any]":
        """Full attendee attribution: traverse email→person→works_at org→type
        DETERMINISTICALLY, and detect (never auto-apply) ambiguous cases as
        proposals for the review queue.

        `attendees` = list of (name, email) pairs (name may be "").

        Returns:
          { "entities":    [canonical ids to tag]  — person + org, EXACT hits only,
            "primary_org": org id that drives the note's domain/dossier | None,
            "primary_type":its relationship (customer|investor|partner|…) | None,
            "proposals":   [ {kind, identifier, evidence, candidate} ]  — ambiguous,
            "trace":       [ per-attendee resolution record ]  — for the accuracy audit }

        NO LLM: this is the attribution hot path and stays deterministic.
        """
        entities: list[str] = []
        seen: set[str] = set()
        proposals: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        primary_org: str | None = None
        primary_type: str | None = None

        def _tag(cid: str):
            if cid and cid not in seen:
                seen.add(cid)
                entities.append(cid)

        for name, raw in attendees or []:
            e = (raw or "").strip().lower()
            rec: dict[str, Any] = {"email": e, "name": (name or "").strip(),
                                   "person": None, "org": None, "type": None,
                                   "via": None, "resolved": False}
            if not e or "@" not in e:
                trace.append(rec); continue
            domain = e.rsplit("@", 1)[-1]
            if domain in self._internal_domains:
                rec["via"] = "internal-domain"; trace.append(rec); continue

            # 1. exact email → person  (2. + works_at org  3. + org type)
            pid = self._person_email.get(e)
            if pid is not None:
                if pid in self._person_internal:
                    rec["via"] = "internal-person"; trace.append(rec); continue
                rec.update(person=pid, via="email→person", resolved=True)
                _tag(pid)
                oid = self._person_org.get(pid)
                if oid and oid not in self._org_self:
                    _tag(oid); rec["org"] = oid
                    rec["type"] = self._org_type.get(oid)
                    if primary_org is None:
                        primary_org, primary_type = oid, rec["type"]
                trace.append(rec); continue

            # 4. exact email/domain → customer or org (no person link on file)
            cid = (self._customer_email.get(e) or self._customer_domain.get(domain)
                   or self._org_domain.get(domain))
            if cid is not None and cid not in self._org_self:
                rec.update(org=cid, via="domain→org", resolved=True,
                           type=self._org_type.get(cid))
                _tag(cid)
                if primary_org is None:
                    primary_org, primary_type = cid, rec["type"]
                # The org resolved, but this attendee's EMAIL is not yet linked to a
                # person. Still propose the person-linkage (link-email / new-person at
                # `cid`) — these are people at known orgs with empty `emails:`. Domain proposals are NOT emitted
                # here (the domain already resolved).
                pprop = self._propose_person(name, e, works_at=cid)
                if pprop is not None:
                    rec["via"] = "domain→org+proposed:" + pprop["kind"]
                    proposals.append(pprop)
                trace.append(rec); continue

            # --- unresolved → propose (never tag): person-linkage first, then domain
            prop = self._propose_person(name, e, works_at=None) \
                or self._propose_domain(name, e, domain)
            if prop is not None:
                rec["via"] = "proposed:" + prop["kind"]
                proposals.append(prop)
            trace.append(rec)

        return {"entities": entities, "primary_org": primary_org,
                "primary_type": primary_type, "proposals": proposals, "trace": trace}

    def _propose_person(self, name: str, email: str,
                        works_at: "str | None") -> "dict[str, Any] | None":
        """Person-linkage proposal for an attendee whose EMAIL is not on any person
        record. Fires whether or not the domain resolved to an org (the people
        it is for are at KNOWN orgs with empty `emails:`):
          - exact name → one known person lacking this email → `link-email`
          - exact name → ≥2 people                          → `link-email` (ambiguous)
          - a name but no person match, and we know an org   → `new-person` at `works_at`
        Returns None when there's no name to anchor on."""
        nkey = _norm_name(name)
        if not nkey:
            return None
        matches = self._name_persons.get(nkey, [])
        if len(matches) == 1:
            pid = matches[0]
            if pid in self._person_internal:
                return None
            # Defensive only: both call sites reach here only after the exact
            # email→person lookup already missed, so `email` is never in the map.
            if email in self._person_email:        # unreachable in practice; kept safe
                return None
            return {"kind": "link-email", "identifier": email,
                    "evidence": {"name": name, "matches_person": pid,
                                 "person_has_email": pid in self._person_has_email,
                                 "works_at": works_at},
                    "candidate": pid}
        if len(matches) > 1:
            return {"kind": "link-email", "identifier": email,
                    "evidence": {"name": name, "ambiguous_matches": matches},
                    "candidate": None, "ambiguous": True}
        # no person by that name, but we know the org → propose creating the person
        if works_at is not None:
            return {"kind": "new-person", "identifier": email,
                    "evidence": {"name": name, "works_at": works_at}, "candidate": None}
        return None

    def _propose_domain(self, name: str, email: str,
                        domain: str) -> "dict[str, Any] | None":
        """Domain-mapping proposal — ONLY when the domain resolved to no org/customer.
        Emits `merge-org` when a look-alike org exists (fragmentation), else
        `map-domain`. Never fires for free-mail / infra / internal domains."""
        if domain in self._noise_domains or domain in self._internal_domains:
            return None
        if self._org_domain.get(domain) is not None or self._customer_domain.get(domain) is not None:
            return None
        root = _registrable_root(domain)
        # look-alike orgs: same registrable root, OR an org whose NAME contains the
        # root token (catches example.com ↔ org:example-mail-com fragmentation).
        # The name-token match is skipped for GENERIC roots ("data", "tech", "mail",
        # "group", "labs", …) that would collide with unrelated orgs → bogus merge
        # proposals. Same-registrable-root always counts (a real domain signal).
        lookalikes = list(self._root_orgs.get(root, []))
        if root not in _GENERIC_ORG_TOKENS:
            for oid in self._nametoken_orgs.get(root, []):
                if oid not in lookalikes:
                    lookalikes.append(oid)
        if lookalikes:
            return {"kind": "merge-org", "identifier": domain,
                    "evidence": {"registrable_root": root, "existing_orgs": lookalikes,
                                 "name": name},
                    "candidate": lookalikes[0]}
        return {"kind": "map-domain", "identifier": domain,
                "evidence": {"registrable_root": root, "name": name}, "candidate": None}

    def resolve(self, text: str = "", author: str | None = None,
                provenance: dict[str, Any] | None = None) -> list[str]:
        """Return matched `customer:<slug>` ids (deduped, registry order).

        Inputs scanned:
          - `text`        : free text (event payload subject/text/body).
          - `author`      : matched against email_domains (an @acme.com sender).
          - `provenance`  : dict; its string values are scanned for email
                            domains, drive_folders and slack_channels.
        """
        text = text or ""
        prov = provenance or {}
        prov_blob = " ".join(str(v) for v in prov.values() if v is not None)
        author = author or ""
        lower_blob = (text + " " + prov_blob + " " + author).lower()

        hits: list[str] = []
        seen: set[str] = set()

        def add(cid: str) -> None:
            if cid and cid not in seen:
                seen.add(cid)
                hits.append(cid)

        # 1) aliases / names — word-boundary match on the free text only
        #    (avoid matching ids embedded in urls/paths spuriously).
        for cid, rx in self._alias_res:
            if rx.search(text):
                add(cid)

        # 2) email domains — author + anything in text/provenance
        for cid, domains in self._domains:
            for d in domains:
                if d and ("@" + d in lower_blob or d in lower_blob):
                    add(cid)
                    break

        # 3) drive folders (provenance path or text)
        for cid, folders in self._folders:
            for fdr in folders:
                if fdr and fdr in lower_blob:
                    add(cid)
                    break

        # 4) slack channels (provenance.channel or text)
        for cid, channels in self._channels:
            for ch in channels:
                if ch and ch in lower_blob:
                    add(cid)
                    break

        # Preserve registry order for stability.
        order = {str(c.get("id", "")): i for i, c in enumerate(self.customers)}
        hits.sort(key=lambda cid: order.get(cid, 1 << 30))
        return hits


# ---------------------------------------------------------------------------
# Registry relationship-type lookup (the AUTHORITY for an entity's type)
# ---------------------------------------------------------------------------
#
# registry/entities.yml groups entities under top-level RELATIONSHIP sections
# (vendor, investor, partner, customer, friend, champion, academic, recruiting,
# prospect, advisor, self, people) plus a `noise` block (free_email / infra /
# edu_suffix). The SECTION is the authority for that entity's type. This lookup
# maps any identifier an event carries — a name, an alias, an email domain, or a
# bare email address — to its registry type, so the distiller can route a note by
# its primary entity's type (engine.TYPE_DOMAIN) instead of dumping every external
# party into `customers`. Deterministic.

# Sections whose entries name a relationship type. `people` -> "person",
# everything else keeps the section name as the type. `system` holds internal
# infrastructure/hosts (db-1, gpu-2, ...) so their mentions canonicalize like any
# other entity instead of fragmenting into case variants; TYPE_DOMAIN deliberately
# has no entry for it (a system mention never re-routes a note's domain).
_TYPE_SECTIONS = (
    "self", "people", "vendor", "investor", "partner", "advisor",
    "champion", "customer", "friend", "academic", "recruiting", "prospect",
    "system",
)


class RegistryTypes:
    """Identifier -> registry relationship type (vendor/investor/customer/...).

    Built once from registry/entities.yml. `type_for("acme.com")`,
    `type_for("Acme Corp")` and `type_for("sales@acme.com")` all resolve via the
    same authority. Returns None for unrecognised identifiers (so the caller keeps
    the LLM/subject domain). 'noise' identifiers resolve to "noise" (skip).
    """

    def __init__(self, registry_path: str = "registry/entities.yml"):
        self.registry_path = registry_path
        self._dom: dict[str, str] = {}   # email_domain -> type
        self._tok: dict[str, str] = {}   # name/alias (lower) -> type
        self._noise_free: set[str] = set()
        self._noise_infra: set[str] = set()
        self._noise_edu: list[str] = []
        self._noise_review: set[str] = set()  # human-triaged noise (domains/emails/names)
        self._load()

    def _load(self) -> None:
        split = _load_split_files(_split_dir_for(self.registry_path))
        if split is not None:
            self._load_from_split(split)
            return
        self._load_from_legacy()

    def _load_from_legacy(self) -> None:
        if not os.path.exists(self.registry_path):
            return
        if _HAVE_YAML:
            try:
                with open(self.registry_path) as f:
                    reg = yaml.safe_load(f) or {}
            except Exception:
                return
        else:  # pragma: no cover - happy path uses PyYAML
            return
        if not isinstance(reg, dict):
            return
        for section in _TYPE_SECTIONS:
            t = "person" if section == "people" else section
            for ent in (reg.get(section) or []):
                if not isinstance(ent, dict):
                    continue
                for d in _as_list(ent.get("email_domains")):
                    if d:
                        self._dom.setdefault(d.lower(), t)
                names = _as_list(ent.get("name")) + _as_list(ent.get("aliases"))
                for a in names:
                    if a:
                        self._tok.setdefault(a.lower(), t)
        noise = reg.get("noise") or {}
        if isinstance(noise, dict):
            self._load_noise_block(noise)

    def _load_from_split(self, split: dict[str, Any]) -> None:
        token_owner: dict[str, str] = {}
        domain_owner: dict[str, str] = {}
        ambiguous_tokens: set[str] = set()
        ambiguous_domains: set[str] = set()

        def register(mapping: dict[str, str], owners: dict[str, str],
                     ambiguous: set[str], key: str, value: str, owner: str) -> None:
            if not key or key in ambiguous:
                return
            prior = owners.get(key)
            if prior is None or prior == owner:
                owners[key] = owner
                mapping[key] = value
            else:
                owners.pop(key, None)
                mapping.pop(key, None)
                ambiguous.add(key)

        for ent in split["people"]:
            if (not isinstance(ent, dict)
                    or registry_record_quarantine_reasons(ent, "person")):
                continue
            names = _as_list(ent.get("name")) + _as_list(ent.get("aliases"))
            owner = str(ent.get("id") or "").strip().lower()
            for a in names:
                if a:
                    register(self._tok, token_owner, ambiguous_tokens,
                             a.lower(), "person", owner)
            # Full emails live in `emails:` now, not `email_domains:`; they
            # never fed `self._dom` usefully before either — internal-domain routing already comes from the
            # `self` org's email_domains below.

        for ent in split["orgs"]:
            if (not isinstance(ent, dict)
                    or registry_record_quarantine_reasons(ent, "org")):
                continue
            rel = (_as_list(ent.get("relationship"))
                   if registry_field_verified(ent, "relationship") else [])
            # `org_type` (entity type: investor/academic/recruiting/brand)
            # wins over `relationship` (engagement: customer/prospect/...) when
            # both are present. The entity type is what a graph view colours
            # by; engagement status stays on the record.
            otype = _as_list(ent.get("org_type"))
            t = otype[0] if otype else (rel[0] if rel else "org")
            owner = str(ent.get("id") or "").strip().lower()
            for d in _as_list(ent.get("email_domains")):
                if d:
                    register(self._dom, domain_owner, ambiguous_domains,
                             d.lower(), t, owner)
            names = _as_list(ent.get("name")) + _as_list(ent.get("aliases"))
            for a in names:
                if a:
                    register(self._tok, token_owner, ambiguous_tokens,
                             a.lower(), t, owner)

        for ent in split["systems"]:
            if (not isinstance(ent, dict)
                    or registry_record_quarantine_reasons(ent, "system")):
                continue
            names = _as_list(ent.get("name")) + _as_list(ent.get("aliases"))
            owner = str(ent.get("id") or "").strip().lower()
            for a in names:
                if a:
                    register(self._tok, token_owner, ambiguous_tokens,
                             a.lower(), "system", owner)

        noise = split["noise"] or {}
        if isinstance(noise, dict):
            self._load_noise_block(noise)

    def _load_noise_block(self, noise: dict[str, Any]) -> None:
        self._noise_free = {x.lower() for x in _as_list(noise.get("free_email"))}
        self._noise_infra = {x.lower() for x in _as_list(noise.get("infra"))}
        self._noise_edu = [s.lower() for s in _as_list(noise.get("edu_suffix"))]
        # `noise.review` is where human "this is noise" verdicts land (a
        # correction tool appends them). It was previously loaded
        # by NOTHING, so every curated noise call was silently ignored and the
        # sender kept minting notes. Real registry entities still win (checked
        # before this set), so an accidental overlap can't demote a real party.
        #
        # Post-split, noise.yml's `review:` list mixes two shapes: the old
        # flat free-text bucket (plain domain/email/name strings) AND the
        # entity-shaped triage records the registry split couldn't place
        # (dicts with `entity`, `name` and other fields) — pull the matchable
        # string out of either shape.
        review_raw = noise.get("review") or []
        review_keys: set[str] = set()
        for item in review_raw:
            if isinstance(item, dict):
                for key in ("entity", "name"):
                    v = item.get(key)
                    if v:
                        review_keys.add(str(v).strip().lower())
            elif item:
                review_keys.add(str(item).strip().lower())
        self._noise_review = review_keys

    def type_for(self, identifier: str) -> str | None:
        """Map one identifier to its registry type, or None if unrecognised.

        Accepts an entity id (`customer:acme` -> strips the prefix), a domain
        (`acme.com`), a name/alias (`Acme Corp`), or a bare email
        (`x@acme.com`). Returns a type string (vendor/investor/customer/partner/
        advisor/champion/friend/academic/recruiting/prospect/self/person/noise)
        or None.
        """
        s = (identifier or "").strip().lower()
        if not s:
            return None
        # An engine entity id may be prefixed (`customer:acme`, `vendor:initech`).
        # The prefix is the resolver's bucket, NOT authoritative here — strip it
        # and classify the bare identifier against the registry.
        if ":" in s and " " not in s:
            head, tail = s.split(":", 1)
            if tail:
                s = tail
        # domain-like (a dotted token with no spaces)
        if "." in s and " " not in s and "@" not in s:
            if s in self._noise_free or s in self._noise_infra:
                return "noise"
            if any(s.endswith(suf) for suf in self._noise_edu):
                return "noise"
            for d, t in self._dom.items():
                if s == d or s.endswith("." + d):
                    return t
            if s in self._noise_review:   # human-triaged noise domain (real entities won above)
                return "noise"
            return None
        # bare email -> classify its domain
        if "@" in s:
            if s in self._noise_review:
                return "noise"
            return self.type_for(s.split("@")[-1])
        # name / alias
        t = self._tok.get(s)
        if t:
            return t
        if s in self._noise_review:       # human-triaged noise name
            return "noise"
        return None


# ---------------------------------------------------------------------------
# Canonicalizer — the single authority mapping ANY identifier to a stable
# `<type>:<slug>` id
# ---------------------------------------------------------------------------
#
# Note frontmatter `entities:` today holds raw surface strings ("Jane Doe",
# "jane doe", "jane@example.com"). Everything downstream that needs to know "is
# this the same entity as that" (MCP timeline, the UI lens, the distill
# engine, the backfill tool) needs one stable id per entity, not a bag of
# aliases. Canonicalizer is that authority: build the lookup once from
# registry/entities.yml, then resolve any surface form deterministically.
#
# Known types are the same sections RegistryTypes reads (`people` -> "person",
# everything else keeps its section name), so a canonical id's prefix always
# matches `type_for()`'s vocabulary.
_KNOWN_TYPES = frozenset(
    "person" if section == "people" else section for section in _TYPE_SECTIONS
)


def _slugify(name: str) -> str:
    """Lowercase, non-alphanumeric runs collapse to one '-', trimmed."""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def _normalize_token(s: str) -> str:
    """Casefold and drop every non-alphanumeric character.

    This is deliberately more aggressive than whitespace-collapse: it is what
    makes "Jane Doe", "jane doe" and "janedoe" collapse to the same lookup key
    (registry alias lists mix spaced and unspaced forms).
    """
    return re.sub(r"[^a-z0-9]", "", s.casefold())


def _register_unambiguous(
    mapping: dict[str, str], ambiguous: set[str], key: str, value: str,
) -> None:
    """Register one exact identity token, dropping cross-entity collisions."""
    if not key or key in ambiguous:
        return
    prior = mapping.get(key)
    if prior is None or prior == value:
        mapping[key] = value
    else:
        mapping.pop(key, None)
        ambiguous.add(key)


class Canonicalizer:
    """Identifier (name / alias / email / domain / existing canonical id) ->
    stable `<type>:<slug>` id, backed by registry/entities.yml.

    All lookups are dict-based and built ONCE in `__init__` — no per-call
    registry scans. Resolution never guesses by substring: "Jane Smith" and
    "Jane Park" must not collapse onto "Jane Doe" just because they share a
    first name, and "Initech Security" must not collapse onto the self entity
    "Initech Inc." just because both contain the word "Initech". An
    unrecognised identifier is None, never a best-effort partner id.

    Follows this file's PyYAML-with-fallback pattern: with no PyYAML, the
    lookup is simply empty and `canonical_id` returns None for everything
    rather than crashing (mirrors `RegistryTypes._load`).
    """

    def __init__(self, registry_path: str = "registry/entities.yml"):
        self.registry_path = registry_path
        self._by_canonical: dict[str, dict[str, Any]] = {}
        self._email_exact: dict[str, str] = {}   # full email (lower) -> id
        self._domain_exact: dict[str, str] = {}  # plain domain (lower) -> id
        self._token_exact: dict[str, str] = {}   # normalized name/alias -> id
        self._ambiguous_emails: set[str] = set()
        self._ambiguous_domains: set[str] = set()
        self._ambiguous_tokens: set[str] = set()
        # A legacy relationship-prefixed id (`customer:globex`, or a
        # pre-split compound's full slug like `partner:initech-jane-doe`)
        # -> the new kind-based id. Empty in fallback mode (never consulted).
        self._legacy_alias: dict[str, str] = {}
        self._ambiguous_legacy_aliases: set[str] = set()
        self._load()

    def _load(self) -> None:
        split = _load_split_files(_split_dir_for(self.registry_path))
        if split is not None:
            self._load_from_split(split)
            return
        self._load_from_legacy()

    def _load_from_legacy(self) -> None:
        if not os.path.exists(self.registry_path) or not _HAVE_YAML:
            return
        try:
            with open(self.registry_path) as f:
                reg = yaml.safe_load(f) or {}
        except Exception:
            return
        if not isinstance(reg, dict):
            return
        for section in _TYPE_SECTIONS:
            t = "person" if section == "people" else section
            for ent in (reg.get(section) or []):
                if not isinstance(ent, dict):
                    continue
                name = _strip(str(ent.get("name") or ""))
                slug = _slugify(name)
                if not slug:
                    continue
                cid = f"{t}:{slug}"
                self._by_canonical.setdefault(cid, {"type": t, "name": name})
                for tok in [name] + _as_list(ent.get("aliases")):
                    key = _normalize_token(tok)
                    if key:
                        self._token_exact.setdefault(key, cid)
                # Quirk: `people` entries store FULL email addresses inside
                # `email_domains` alongside (or instead of) bare domains — a
                # value containing '@' is an exact address, not a domain.
                for d in _as_list(ent.get("email_domains")):
                    d = d.strip().lower()
                    if not d:
                        continue
                    if "@" in d:
                        self._email_exact.setdefault(d, cid)
                    else:
                        self._domain_exact.setdefault(d, cid)

    def _load_from_split(self, split: dict[str, Any]) -> None:
        primary_owners: dict[str, set[str]] = {}
        for bucket, kind in (("people", "person"), ("orgs", "org"), ("systems", "system")):
            for row in split[bucket]:
                if (not isinstance(row, dict)
                        or registry_record_quarantine_reasons(row, kind)):
                    continue
                owner = str(row.get("id") or "").strip().lower()
                key = _normalize_token(str(row.get("name") or ""))
                if owner and key:
                    primary_owners.setdefault(key, set()).add(owner)

        def register_token(token: str, owner: str, *, primary: bool) -> None:
            key = _normalize_token(token)
            if not key:
                return
            primaries = primary_owners.get(key, set())
            if primary:
                if len(primaries) == 1:
                    self._ambiguous_tokens.discard(key)
                    self._token_exact[key] = owner
                else:
                    self._token_exact.pop(key, None)
                    self._ambiguous_tokens.add(key)
                return
            if primaries:
                return
            _register_unambiguous(
                self._token_exact, self._ambiguous_tokens, key, owner,
            )

        for ent in split["people"]:
            if (not isinstance(ent, dict)
                    or registry_record_quarantine_reasons(ent, "person")):
                continue
            pid = str(ent.get("id") or "").strip().lower()
            if not pid:
                continue
            name = _strip(str(ent.get("name") or ""))
            self._by_canonical.setdefault(pid, {"type": "person", "name": name})
            aliases = _as_list(ent.get("aliases"))
            for index, tok in enumerate([name] + aliases):
                register_token(tok, pid, primary=index == 0)
                slug_alias = _slugify(tok)
                if slug_alias:
                    _register_unambiguous(
                        self._legacy_alias, self._ambiguous_legacy_aliases,
                        f"person:{slug_alias}", pid,
                    )
            emails = _as_list(ent.get("emails"))
            for e in emails:
                e = e.strip().lower()
                if "@" in e:
                    _register_unambiguous(
                        self._email_exact, self._ambiguous_emails, e, pid,
                    )

        for ent in split["orgs"]:
            if (not isinstance(ent, dict)
                    or registry_record_quarantine_reasons(ent, "org")):
                continue
            oid = str(ent.get("id") or "").strip().lower()
            if not oid:
                continue
            name = _strip(str(ent.get("name") or ""))
            declared_rel = _as_list(ent.get("relationship"))
            rel = declared_rel if registry_field_verified(ent, "relationship") else []
            # `org_type` (entity type: investor/academic/recruiting/brand)
            # wins over `relationship` (engagement: customer/prospect/...)
            # when both are present. The user's intent for entity type is
            # the canvas's color signal; engagement status stays on the
            # record as `relationship`.
            otype = _as_list(ent.get("org_type"))
            t = otype[0] if otype else (rel[0] if rel else "org")
            self._by_canonical.setdefault(
                oid, {"type": t, "name": name, "relationship": rel,
                      "org_type": otype,
                      "declared_relationship": declared_rel}
            )
            aliases = _as_list(ent.get("aliases"))
            for index, tok in enumerate([name] + aliases):
                register_token(tok, oid, primary=index == 0)
            for d in _as_list(ent.get("email_domains")):
                d = d.strip().lower()
                if not d:
                    continue
                if "@" in d:
                    _register_unambiguous(
                        self._email_exact, self._ambiguous_emails, d, oid,
                    )
                else:
                    _register_unambiguous(
                        self._domain_exact, self._ambiguous_domains, d, oid,
                    )
            # Every historical relationship prefix (customer:/partner:/
            # advisor:/champion:/friend:/... incl. self:) + the SAME slug the
            # pre-split entity used resolves to the new org id. "Same slug"
            # is read expansively here: the org's own name AND every alias
            # (which always includes the full pre-split compound string,
            # e.g. "Initech (Jane Doe)") are each slugified and aliased
            # under every relationship prefix — this is what makes a
            # compound entry's OLD canonical id (`partner:initech-jane-
            # doe`, slugified from the whole compound name) resolve, not
            # just the new org's own (shorter) slug.
            for tok in ([name] + aliases if declared_rel else []):
                slug_alias = _slugify(tok)
                if not slug_alias:
                    continue
                for legacy_type in declared_rel:
                    _register_unambiguous(
                        self._legacy_alias, self._ambiguous_legacy_aliases,
                        f"{legacy_type}:{slug_alias}", oid,
                    )

        for ent in split["systems"]:
            if (not isinstance(ent, dict)
                    or registry_record_quarantine_reasons(ent, "system")):
                continue
            sid = str(ent.get("id") or "").strip().lower()
            if not sid:
                continue
            name = _strip(str(ent.get("name") or ""))
            self._by_canonical.setdefault(sid, {"type": "system", "name": name})
            aliases = _as_list(ent.get("aliases"))
            for index, tok in enumerate([name] + aliases):
                register_token(tok, sid, primary=index == 0)
                slug_alias = _slugify(tok)
                if slug_alias:
                    _register_unambiguous(
                        self._legacy_alias, self._ambiguous_legacy_aliases,
                        f"system:{slug_alias}", sid,
                    )

    def _domain_lookup(self, domain: str) -> str | None:
        """Exact match, then walk up to the parent domain (`dev.example.com` ->
        `example.com`), stopping at the registrable two-label domain. Domains that
        are free-email/infra/edu/review noise, or simply unregistered, fall
        through to None here — there is nothing to invent."""
        domain = domain.strip(". ")
        if not domain:
            return None
        if domain in self._domain_exact:
            return self._domain_exact[domain]
        parts = domain.split(".")
        for i in range(1, len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in self._domain_exact:
                return self._domain_exact[parent]
        return None

    def canonical_id(self, identifier: str) -> str | None:
        """Resolve any surface identifier to its `<type>:<slug>` id, or None.

        Deterministic pipeline:
          1. Already `<known-type>:<slug>` and it resolves -> return it
             normalized (lowercased).
          2. Contains '@': the full address may be registered verbatim
             (people entries store full emails in `email_domains`); else
             resolve its domain part.
          3. Domain-shaped (dotted, no spaces): exact match, then parent-
             domain walk.
          4. Otherwise: casefold + strip separators, match against every
             entry's name + aliases across all `_TYPE_SECTIONS`.
          5/6. Anything else (bare numerics, noise, unrecognised) -> None.
        """
        if not identifier:
            return None
        s = _strip(str(identifier))
        if not s:
            return None

        # 1. Already a canonical id? (new kind-based id, e.g. `org:initech`,
        #    `person:jane-doe`) — OR a legacy relationship-prefixed id
        #    (`customer:globex`, `partner:initech-jane-doe`) that
        #    resolves as a legacy alias of the new id, post-split.
        if ":" in s:
            head, _, tail = s.partition(":")
            candidate = f"{head.strip().lower()}:{tail.strip().lower()}"
            if candidate in self._by_canonical:
                return candidate
            if candidate in self._legacy_alias:
                return self._legacy_alias[candidate]

        # 2. Email address.
        if "@" in s:
            low = s.lower()
            if low in self._email_exact:
                return self._email_exact[low]
            return self._domain_lookup(low.rsplit("@", 1)[-1])

        # 3. Domain-shaped.
        if "." in s and " " not in s:
            # Some organizations use a domain-shaped legal/display name
            # (e.g. ``Acme.io``).  An exact registry name is identity evidence
            # even when its email-domain field is still awaiting review.
            # Prefer an attested domain match, then fall back to that exact
            # primary-name token; aliases remain provenance-gated.
            return (self._domain_lookup(s.lower())
                    or self._token_exact.get(_normalize_token(s)))

        # 4. Name / alias.
        key = _normalize_token(s)
        if key:
            return self._token_exact.get(key)
        return None

    def display_name(self, canonical_id: str) -> str:
        """`'person:jane-doe' -> 'Jane Doe'` (the registry's `name:`, not the
        slug). Best-effort (never raises) for an id we didn't build."""
        entry = self._by_canonical.get(canonical_id)
        if entry:
            return entry["name"]
        tail = canonical_id.split(":", 1)[-1] if canonical_id else ""
        return tail.replace("-", " ").replace("_", " ").strip().title()

    def is_self(self, s: str) -> bool:
        """True if `s` — a raw identifier OR an already-canonical id —
        resolves to a self entity: `self:*` in fallback mode, or an org whose
        `relationship` includes `self` post-split (the kind changes, the
        self-ness signal doesn't)."""
        cid = self.canonical_id(s)
        if not cid:
            return False
        if cid.startswith("self:"):
            return True
        entry = self._by_canonical.get(cid)
        return bool(entry) and "self" in (
            entry.get("declared_relationship") or entry.get("relationship") or []
        )

    def type_of(self, canonical_id: str) -> str:
        """`'person:jane-doe' -> 'person'`; `'org:globex' -> 'org'`.

        For an org this is `org_type[0]`, else `relationship[0]` **only when
        `field_provenance.relationship` is human-attested**, else the kind
        `org` (see `_load_from_split` and remembrancerlib/registry_trust.py). Routing
        on an unreviewed relationship is what the trust gate forbids, so an
        org with a merely-declared `relationship: [customer]` types as `org`,
        NOT `customer` — `TYPE_DOMAIN` has no `org` key, so such an entity
        contributes no relationship domain to `distill.engine._registry_domain`
        and dossier routing falls back to the page generator's default
        domain. Attesting the field (see
        `set_registry_field_attestation`) is what restores relationship typing.

        Falls back to the id's own prefix for an id we didn't build (never
        raises)."""
        entry = self._by_canonical.get(canonical_id)
        if entry:
            return entry["type"]
        return canonical_id.split(":", 1)[0] if ":" in canonical_id else ""

    def kind_of(self, s: str) -> str | None:
        """Identifier -> node kind (`person`/`org`/`system`/`component`/
        `initiative`). Accepts a raw identifier, a legacy
        relationship-prefixed id, or an already kind-prefixed id (including
        `component:`/`initiative:`, whose registries are not implemented yet —
        recognised syntactically so callers can route on them before
        those loaders exist). None if unresolvable.
        """
        if not s:
            return None
        s = _strip(str(s))
        if not s:
            return None
        if ":" in s:
            head = s.split(":", 1)[0].strip().lower()
            if head in _KIND_PREFIXES:
                return head
        cid = self.canonical_id(s)
        if not cid:
            return None
        t = self.type_of(cid)
        if t in _KIND_PREFIXES:
            return t
        return _RELATIONSHIP_TYPE_TO_KIND.get(t)


# Module-level convenience (builds a resolver per call; fine for ad-hoc use).
_DEFAULT: EntityResolver | None = None
_DEFAULT_TYPES: RegistryTypes | None = None
_DEFAULT_CANONICALIZER: Canonicalizer | None = None


def registry_types(registry_path: str = "registry/entities.yml") -> RegistryTypes:
    """Cached RegistryTypes singleton (rebuilt if the path changes)."""
    global _DEFAULT_TYPES
    if _DEFAULT_TYPES is None or _DEFAULT_TYPES.registry_path != registry_path:
        _DEFAULT_TYPES = RegistryTypes(registry_path)
    return _DEFAULT_TYPES


def canonicalize(identifier: str, registry_path: str = "registry/entities.yml") -> str | None:
    """Cached module-level singleton, mirroring `registry_types()`."""
    global _DEFAULT_CANONICALIZER
    if _DEFAULT_CANONICALIZER is None or _DEFAULT_CANONICALIZER.registry_path != registry_path:
        _DEFAULT_CANONICALIZER = Canonicalizer(registry_path)
    return _DEFAULT_CANONICALIZER.canonical_id(identifier)


def resolve(text: str = "", author: str | None = None,
            provenance: dict[str, Any] | None = None,
            registry_path: str = "registry/customers.yml") -> list[str]:
    """Stateless convenience wrapper around EntityResolver.resolve.

    The engine builds one EntityResolver and reuses it; this is for callers who
    just want a one-shot resolve without managing the object.
    """
    global _DEFAULT
    if _DEFAULT is None or _DEFAULT.registry_path != registry_path:
        _DEFAULT = EntityResolver(registry_path)
    return _DEFAULT.resolve(text=text, author=author, provenance=provenance)
