#!/usr/bin/env python3
"""Person identity resolution — the one join between raw actor strings and `person:` keys.

Every source names a person differently: a git commit carries
``Ana (Annie) Lima <annie.lima@mail.example>``, an artifact manifest carries
publisher ``ana`` and a source dir ``/home/alima/projects/...``, Slack
carries a display name. This module is the single place that maps any of those
onto a canonical ``person:<slug>`` from ``registry/people.yml``, using
``config.json``'s ``git_identities`` as the declared override for handles that
share no prefix with the person's name (``lima-a`` and ``ana``, which no
prefix match would ever connect).

Matching is two-tier:

  STRONG   an exact hit on a normalised email, GitHub handle, declared name
           variant, alias, or full name. Wins outright.
  FALLBACK a distinctive-token hit (``mei`` inside "Laura Mei Chen").
           **Ambiguous tokens matching >1 person are dropped**, never guessed —
           a wrong attribution is worse than no attribution.

``status`` on a record says whether the person is still here: absent or
``active`` is a current teammate; ``former`` is a past employee. Former people
keep resolving (old pages and commits still attribute correctly) but leave the
teammate lists (``internal_people``) unless a caller asks for them, so no
surface puts them level with the current team.

Nothing here writes. Ownership is derived at READ time from fields that already
exist on every artifact manifest, so a people page needs no backfill and no
change to the publish path to start working.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# Identities that name no person: the empty identity, the shared tier a caller
# gets by default (``internal``), ``unknown`` and the superuser. An artifact
# published under one of these is UNATTRIBUTED — it belongs to the commons, not
# to whoever happened to run the service. A deployment names its own service and
# deploy accounts in config.json ``identity.shared_identities``
# (``shared_identities``).
SHARED_IDENTITIES = frozenset({"", "internal", "unknown", "root"})

_WORD = re.compile(r"[a-z0-9]+")


# ─── normalisation ───────────────────────────────────────────────────────────

def _norm(s: Any) -> str:
    """Fold to a comparable key: lowercase, de-accent, drop every non-alnum.

    "Ana (Annie) Lima" -> "anaannielima"; "ana@acme.example" -> "anaacmeexample".
    Collapsing punctuation is what lets a unix home dir (``analima``) match a
    display name ("Ana Lima") without a hand-maintained alias for each host.
    """
    text = unicodedata.normalize("NFKD", str(s or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _tokens(s: Any) -> set[str]:
    """Distinctive lowercase word tokens, minus single letters and stopwords."""
    text = unicodedata.normalize("NFKD", str(s or "")).lower()
    text = "".join(c for c in text if not unicodedata.combining(c))
    return {t for t in _WORD.findall(text) if len(t) > 2}


def shared_identities(cfg: Any) -> frozenset[str]:
    """SHARED_IDENTITIES plus config.json ``identity.shared_identities``,
    lower-cased. A missing or malformed key adds nothing."""
    identity = cfg.get("identity") if isinstance(cfg, dict) else None
    raw = identity.get("shared_identities") if isinstance(identity, dict) else None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return SHARED_IDENTITIES
    return SHARED_IDENTITIES | {v.strip().lower() for v in raw if isinstance(v, str)}


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v if x]
    return [str(v)]


def _email_local(addr: str) -> str:
    return addr.split("@", 1)[0] if "@" in addr else addr


# ─── the person record ───────────────────────────────────────────────────────

@dataclass
class Person:
    key: str                       # canonical `person:<slug>`
    name: str
    internal: bool = False
    status: str = "active"         # "active" (default) or "former" — a past employee
    login: str = ""                # the person's server account — the `~handle`
    emails: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    github: list[str] = field(default_factory=list)
    exact: set[str] = field(default_factory=set)    # normalised STRONG keys
    tokens: set[str] = field(default_factory=set)   # FALLBACK tokens
    handle: str = ""               # the `~handle` in URLs

    @property
    def slug(self) -> str:
        return self.key.split(":", 1)[1] if ":" in self.key else self.key

    @property
    def former(self) -> bool:
        return self.status == "former"


# ─── the registry ────────────────────────────────────────────────────────────

class PeopleRegistry:
    """registry/people.yml + config.json's git_identities, indexed for lookup."""

    def __init__(self, people: Iterable[Person],
                 shared: Iterable[str] = SHARED_IDENTITIES):
        self.people: list[Person] = list(people)
        # Identities that name no person (``shared_identities``).
        self.shared: frozenset[str] = frozenset(shared)
        self._by_key = {p.key: p for p in self.people}
        self._by_exact: dict[str, list[Person]] = {}
        self._by_token: dict[str, list[Person]] = {}
        for p in self.people:
            for e in p.exact:
                self._by_exact.setdefault(e, []).append(p)
            for t in p.tokens:
                self._by_token.setdefault(t, []).append(p)
        self._by_handle = {p.handle: p for p in self.people if p.handle}

    # --- loading ---------------------------------------------------------

    @classmethod
    def load(cls, root: str = ".") -> "PeopleRegistry":
        people_yml = os.path.join(root, "registry", "people.yml")
        config_json = os.path.join(root, "config.json")
        return cls.from_files(people_yml, config_json)

    @classmethod
    def from_files(cls, people_yml: str, config_json: str = "") -> "PeopleRegistry":
        records = _read_people(people_yml)
        cfg = _read_json(config_json) if config_json else {}
        git_ids = cfg.get("git_identities") or {}
        identities = cfg.get("identities") or {}
        shared = shared_identities(cfg)
        return cls(_build(records, git_ids, identities, shared), shared)

    # --- lookup ----------------------------------------------------------

    def by_key(self, key: str) -> Optional[Person]:
        return self._by_key.get(key)

    def by_email(self, email: str) -> Optional[Person]:
        """The ONE person whose registry ``emails`` list holds ``email`` exactly
        (case-insensitive), or None when absent or shared by several records.
        No local-part or fuzzy fallback: callers use this for verified identity."""
        e = (email or "").strip().lower()
        if not e or "@" not in e:
            return None
        hits = [p for p in self.people if e in {x.strip().lower() for x in p.emails}]
        return hits[0] if len(hits) == 1 else None

    def by_handle(self, handle: str) -> Optional[Person]:
        """Resolve a URL handle (`~ana`). Falls back to the person slug."""
        h = (handle or "").strip().lstrip("~").lower()
        if not h:
            return None
        hit = self._by_handle.get(h)
        if hit is not None:
            return hit
        for p in self.people:
            if p.slug == h:
                return p
        return self.resolve_person(h)

    def internal_people(self, *, include_former: bool = False) -> list[Person]:
        """Teammates (``internal: true``), by name. Past employees
        (``status: former``) are left out unless ``include_former`` is set."""
        return sorted((p for p in self.people
                       if p.internal and (include_former or not p.former)),
                      key=lambda p: p.name)

    # --- resolution ------------------------------------------------------

    def resolve_person(self, raw: Any) -> Optional[Person]:
        """Map one actor string to a Person, or None when it is shared/ambiguous.

        Accepts a bare handle (``analima``), an email, a display name, or the
        git ``"Name <email>"`` form. Returns None rather than guessing whenever
        the string names a shared identity or matches more than one person.
        """
        text = str(raw or "").strip()
        if not text:
            return None

        candidates = [text]
        addr = re.search(r"<([^>]+)>", text)
        if addr:                                   # "Name <email>" — try both
            candidates.append(addr.group(1))
            candidates.append(text[:addr.start()])
        for c in list(candidates):
            if "@" in c:
                candidates.append(_email_local(c))

        for c in candidates:
            if _norm(c) in self.shared or (c or "").strip().lower() in self.shared:
                return None

        # STRONG — exact normalised hit, unique.
        for c in candidates:
            hits = self._by_exact.get(_norm(c)) or []
            uniq = {p.key: p for p in hits}
            if len(uniq) == 1:
                return next(iter(uniq.values()))

        # FALLBACK — every distinctive token must point at the same person.
        for c in candidates:
            owners: set[str] = set()
            person: Optional[Person] = None
            for tok in _tokens(c):
                hits = self._by_token.get(tok) or []
                for p in hits:
                    owners.add(p.key)
                    person = p
            if len(owners) == 1 and person is not None:
                return person
        return None

    def resolve_actor(self, raw: Any) -> Optional[str]:
        """resolve_person, returning the canonical `person:<slug>` key."""
        p = self.resolve_person(raw)
        return p.key if p else None

    # --- artifacts -------------------------------------------------------

    def owner_of_artifact(self, manifest: dict,
                          trust_build_dir: bool = True) -> tuple[Optional[str], str]:
        """Derive (person_key, how) for one artifact manifest, at read time.

        Three signals, all declarations rather than guesses:

        0. ``owner`` — the declared person the work belongs to, retained across
           edits by other publishers. Unknown explicit owners stay unresolved.

        1. ``publisher`` — the identity the publish call ran as, when it names a
           person rather than a shared account (see ``shared_identities``).
           Often absent where a launcher stamps most callers with one shared
           identity.
        2. ``source.dir``'s ``/home/<user>/`` — **the account the publishing
           session ran under**. On a multi-user host where each person works
           under their own login, this names the person operating the session.

        A NOTE ON WHY (2) IS HERE. A folder name inside the path (say
        ``..._audit/``) is no evidence that the pages are someone else's work
        merely being audited, and must not be read that way. Do not remove
        this signal on the strength of a folder name: a path is not proof of
        authorship, but the ACCOUNT is the one thing a shared box does know
        about who is working.

        ``trust_build_dir=False`` still asks for declared owner/publisher only, which is what a
        caller wants when it must distinguish a signed page from an inferred one.
        """
        if not isinstance(manifest, dict):
            return None, ""

        # New manifests separate the work's owner from its publishing account
        # and editor. A recorded owner wins even when someone else republishes.
        # Resolve only exact registry keys/names here: a fuzzy match must never
        # turn an unknown declared owner into the publisher or a similar name.
        owner = manifest.get("owner")
        if owner:
            if not isinstance(owner, str):
                return None, ""
            owner = owner.strip()
            if owner.startswith("person:"):
                person = self.by_key(owner)
            else:
                declared = owner.removeprefix("declared:")
                hits = {p.key: p for p in self._by_exact.get(_norm(declared), [])}
                person = next(iter(hits.values())) if len(hits) == 1 else None
            return (person.key, "owner") if person else (None, "")

        key = self.resolve_actor(manifest.get("publisher"))
        if key:
            return key, "publisher"

        if trust_build_dir:
            source = manifest.get("source")
            if isinstance(source, dict):
                home = re.match(r"/home/([^/]+)/", str(source.get("dir") or ""))
                if home:
                    key = self.resolve_actor(home.group(1))
                    if key:
                        return key, "account"
        return None, ""


# ─── construction helpers ────────────────────────────────────────────────────

def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


def _read_people(path: str) -> list[dict]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:                            # pragma: no cover - yaml is a dep
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
    except (OSError, ValueError):
        return []
    records = doc.get("people") if isinstance(doc, dict) else doc
    return [r for r in (records or []) if isinstance(r, dict) and r.get("id")]


def _build(records: list[dict], git_ids: dict, identities: dict,
           shared: Iterable[str] = SHARED_IDENTITIES) -> list[Person]:
    shared = frozenset(shared)
    people: list[Person] = []
    for rec in records:
        emails = _as_list(rec.get("emails"))
        aliases = _as_list(rec.get("aliases"))
        name = str(rec.get("name") or "")
        login = str(rec.get("login") or "").strip().lower()
        status = str(rec.get("status") or "active").strip().lower() or "active"

        # config.json's git_identities is keyed by the person's primary email.
        declared: dict = {}
        for e in emails:
            got = git_ids.get(e)
            if isinstance(got, dict):
                declared = got
                break
        github = _as_list(declared.get("github"))
        alt_emails = _as_list(declared.get("emails"))
        alt_names = _as_list(declared.get("names"))

        exact: set[str] = set()
        for value in [name, login, *aliases, *emails, *alt_emails, *github, *alt_names]:
            n = _norm(value)
            if n:
                exact.add(n)
            if "@" in str(value):
                loc = _norm(_email_local(str(value)))
                # Bare noreply locals ("12345678+alima") normalise into a
                # digit-prefixed blob; keep the handle part too.
                if loc:
                    exact.add(loc)

        tokens: set[str] = set()
        for value in [name, *aliases, *alt_names, *github]:
            tokens |= _tokens(value)

        person = Person(
            key=str(rec["id"]), name=name, internal=bool(rec.get("internal")),
            status=status, login=login, emails=emails, aliases=aliases, github=github,
            exact=exact, tokens=tokens,
        )
        people.append(person)

    # The URL handle, strongest claim first:
    #   1. `login` — the person's server account. This is what `~user` means
    #      in the original convention (a `/~user` web home is a unix login),
    #      it is the name teammates already know, and it is the SAME string
    #      that shows up in an artifact's source.dir — so the handle and the
    #      attribution key are one thing, with no mapping in between. Where
    #      accounts differ per host (`lchen` on one, `laura` on another), the
    #      host that serves the pages is canonical, and the other host's
    #      name stays an alias so old manifests still resolve.
    #   2. a remembrancer identity from config.json (`ana`, `dana`).
    #   3. the person slug — everyone gets a page, even with neither declared.
    reg = PeopleRegistry(people, shared)
    claimed: dict[str, str] = {}
    taken = {p.login for p in people if p.login}
    for ident in sorted(identities, key=len):
        if ident.startswith("_") or ident in shared or ident in taken:
            continue
        hit = reg.resolve_person(ident)
        if hit is not None and hit.key not in claimed:
            claimed[hit.key] = ident.lower()
    for p in people:
        p.handle = p.login or claimed.get(p.key, p.slug)
    return people


__all__ = ["Person", "PeopleRegistry", "SHARED_IDENTITIES", "shared_identities"]
