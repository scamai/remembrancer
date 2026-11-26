"""Stable metadata primitives shared by dossier producers and auditors.

This module deliberately contains no contributor-selection policy.  The
entity-page generator and lint independently select current source notes, then
use the same canonical serialization here so the auditor can prove that the
published machine region was built from exactly that source revision set.
"""
from __future__ import annotations

import hashlib
import json
import re

from .note import concise_description


DOSSIER_SELECTOR_VERSION = 1
DOSSIER_PROFILE_VERSION = 1
DOSSIER_SCHEMA_VERSION = 2
DOSSIER_METADATA_MARKER = f"dossier-schema:v{DOSSIER_SCHEMA_VERSION}"
DOSSIER_PROFILE_PLACEHOLDER = "_no summary yet_"
DOSSIER_PROFILE_FIELDS = ("who", "relationship", "status")
DOSSIER_PROFILE_QUOTE_MIN_CHARS = 16
DOSSIER_PROFILE_QUOTE_MIN_ALNUM = 10
DOSSIER_PROFILE_QUOTE_MAX_CHARS = 500

_PROFILE_FIELD_CUES = {
    "relationship": (
        "customer", "prospect", "partner", "vendor", "investor", "advisor",
        "relationship", "works with", "collaborates with", "client", "客户",
        "潜在客户", "合作伙伴", "供应商", "投资人", "顾问", "合作关系",
    ),
    "status": (
        "active", "inactive", "current", "pending", "blocked", "closed",
        "complete", "completed", "live", "paused", "renewed", "signed",
        "launched", "deprecated", "on track", "at risk", "status", "状态",
        "进行中", "已完成", "已暂停", "已签署", "已续约", "有风险",
    ),
    "open_item": (
        "follow up", "todo", "to do", "need to", "needs to", "should",
        "next step", "open item", "pending", "waiting", "investigate",
        "action item", "question", "跟进", "待办", "下一步", "需要", "等待",
    ),
}


def _semantic_text(value: object) -> str:
    return re.sub(r"[^\w]+", " ", str(value or "").casefold()).strip()


def dossier_profile_field_supported(
    field: str,
    quote: str,
    subject_labels: tuple[str, ...] | list[str] = (),
) -> bool:
    """Require the rendered label's meaning to exist in the quoted clause."""
    labels = {_semantic_text(value) for value in subject_labels if _semantic_text(value)}
    clauses = [_semantic_text(value) for value in re.split(r"[.!?;\n。！？；]+", quote)]
    for clause in clauses:
        if not clause:
            continue
        padded = f" {clause} "
        if labels and not any(f" {label} " in padded for label in labels):
            continue
        if field == "who":
            return bool(labels)
        cues = _PROFILE_FIELD_CUES.get("open_item" if field == "open_items" else field)
        if cues and any(f" {_semantic_text(cue)} " in padded for cue in cues):
            return True
    return False


def dossier_source_fingerprint(notes: list) -> str:
    """Hash the complete logical contents of a deterministic note collection.

    ``updated`` is date-granular in legacy notes, so hashing only name/date/ACL
    can miss a same-day correction.  ``Note.to_markdown`` captures the fields
    and body actually visible to the dossier producer without depending on a
    note's filesystem path.
    """
    rows = sorted(
        (str(note.name or ""), note.to_markdown())
        for note in notes
    )
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dossier_slug(value: str) -> str:
    """Match the generator's stable lowercase/hyphen slug contract."""
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-") or "x"


def dossier_base_name(kind: str, subject_id: str) -> str:
    """Return the fixed logical page name for a dossier target."""
    subject_id = str(subject_id or "").strip()
    if kind == "entity":
        tail = subject_id.split(":", 1)[1] if ":" in subject_id else subject_id
        return f"entity-{dossier_slug(tail)}"
    if kind == "initiative":
        return f"initiative-{dossier_slug(subject_id)}"
    raise ValueError(f"unsupported dossier kind: {kind!r}")


def dossier_profile_prompt_material(
    notes: list,
    *,
    limit_chars: int = 8000,
    per_note_chars: int = 1500,
    require_names: "Sequence[str]" = (),
) -> tuple[str, dict[str, str], bool]:
    """Return the exact profile prompt notes, source excerpts and completeness.

    Quotes are validated against ``excerpts`` rather than the full note so a
    model cannot cite tail content it never received. ``incomplete`` is true
    whenever a body or the overall note list was clipped; callers must not
    interpret an empty model result as proof that a relationship is absent.

    ``require_names`` is for RE-VERIFICATION, not generation. The cross-note
    ``limit_chars`` budget evicts whichever notes fall past the cap, and that
    subset changes as an entity accumulates notes. Re-deriving the window later
    therefore invalidates claims that were legitimately grounded when written —
    which can deadlock a scheduled re-verification. Listing the notes a stored claim
    CITES pulls them in first so the budget cannot evict them.

    This does not weaken the anti-hallucination control: ``per_note_chars``
    truncation still applies to required notes, so tail content the model never
    received remains uncitable, and generation (which passes no
    ``require_names``) still enforces the strict window at write time — a claim
    citing a never-shown note could not have been persisted to begin with.
    """
    required = {str(n).strip() for n in require_names if str(n).strip()}
    if required:
        # Stable partition: required notes first, everything else in original order.
        notes = ([n for n in notes if str(getattr(n, "name", "") or "").strip() in required]
                 + [n for n in notes if str(getattr(n, "name", "") or "").strip() not in required])

    blocks: list[str] = []
    excerpts: dict[str, str] = {}
    used = 0
    incomplete = False
    for index, note in enumerate(notes):
        name = str(getattr(note, "name", "") or "").strip()
        description = str(getattr(note, "description", "") or "")
        full_body = str(getattr(note, "body", "") or "")
        body = full_body[:per_note_chars]
        if len(full_body) > per_note_chars:
            incomplete = True
        excerpt = f"{description}\n{body}".strip()
        block = f"### {name}\n{excerpt}".strip()
        if used + len(block) > limit_chars:
            incomplete = True
            break
        if not name or name in excerpts:
            # Mutation paths already refuse duplicate logical names. Treat an
            # unexpected duplicate as incomplete and expose neither ambiguous
            # source to the evidence validator.
            incomplete = True
            excerpts.pop(name, None)
        else:
            excerpts[name] = excerpt
        blocks.append(block)
        used += len(block)
        if index + 1 < len(notes) and used >= limit_chars:
            incomplete = True
    return "\n\n".join(blocks), excerpts, incomplete


def substantive_exact_quote(quote: object, excerpt: str) -> str | None:
    """Return a canonical exact quote only when it is materially informative."""
    if not isinstance(quote, str):
        return None
    quote = quote.strip()
    if not quote or len(quote) > DOSSIER_PROFILE_QUOTE_MAX_CHARS:
        return None
    collapsed = re.sub(r"\s+", " ", quote).strip()
    alnum = sum(character.isalnum() for character in collapsed)
    if (len(collapsed) < DOSSIER_PROFILE_QUOTE_MIN_CHARS
            or alnum < DOSSIER_PROFILE_QUOTE_MIN_ALNUM):
        return None
    if quote not in excerpt:
        return None
    return quote


def normalize_dossier_profile_claims(
    payload: object,
    excerpts: dict[str, str],
    subject_labels: tuple[str, ...] | list[str] = (),
) -> dict[str, object]:
    """Drop every ungrounded model field and return a canonical claim object.

    The model does not get to author summary prose. Each accepted field is an
    exact source quote plus its source note; rendering below copies that quote
    rather than a free-form paraphrase, eliminating the hallucination channel.
    Partial validity is intentional: one bad field does not discard unrelated
    grounded fields.
    """
    if not isinstance(payload, dict):
        return {}

    def claim(field: str, value: object) -> dict[str, str] | None:
        if not isinstance(value, dict):
            return None
        source = value.get("source_note")
        if not isinstance(source, str) or source not in excerpts:
            return None
        quote = substantive_exact_quote(value.get("quote"), excerpts[source])
        if (quote is None
                or not dossier_profile_field_supported(field, quote, subject_labels)):
            return None
        return {"source_note": source, "quote": quote}

    normalized: dict[str, object] = {}
    for field in DOSSIER_PROFILE_FIELDS:
        accepted = claim(field, payload.get(field))
        if accepted is not None:
            normalized[field] = accepted

    items: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    raw_items = payload.get("open_items")
    if isinstance(raw_items, list):
        for raw in raw_items[:20]:
            accepted = claim("open_items", raw)
            if accepted is None:
                continue
            key = (accepted["source_note"], accepted["quote"])
            if key in seen:
                continue
            seen.add(key)
            items.append(accepted)
            if len(items) >= 8:
                break
    if items:
        normalized["open_items"] = items
    return normalized


def dossier_profile_claims_fingerprint(claims: object) -> str:
    payload = json.dumps(
        claims if isinstance(claims, dict) else {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_grounded_claim(claim: dict[str, str]) -> str:
    quote = concise_description(claim.get("quote", ""))
    # Quoted source prose is data. Neutralize copied wiki-link delimiters so it
    # cannot manufacture a cross-reference to a target the dossier audience
    # was never authorized to discover.
    quote = quote.replace("[[", "［［").replace("]]", "］］")
    return f"{quote} ([[{claim['source_note']}]])"


def render_dossier_profile(claims: object) -> str:
    if not isinstance(claims, dict):
        return ""
    parts: list[str] = []
    who = claims.get("who")
    if isinstance(who, dict):
        parts.append(_render_grounded_claim(who))
    relationship = claims.get("relationship")
    if isinstance(relationship, dict):
        parts.append(f"**Relationship:** {_render_grounded_claim(relationship)}")
    status = claims.get("status")
    if isinstance(status, dict):
        parts.append(f"**Status:** {_render_grounded_claim(status)}")
    raw_items = claims.get("open_items")
    if isinstance(raw_items, list) and raw_items:
        rendered = [
            f"- {_render_grounded_claim(item)}"
            for item in raw_items if isinstance(item, dict)
        ]
        if rendered:
            parts.append("**Open items:**\n" + "\n".join(rendered))
    return "\n\n".join(parts)


def extract_dossier_profile(body: str) -> str:
    """Extract the rendered ``## Profile`` payload from a dossier body."""
    marker = "## Profile"
    index = str(body or "").find(marker)
    if index == -1:
        return ""
    remainder = str(body or "")[index + len(marker):]
    end = remainder.find("\n## ")
    profile = remainder if end == -1 else remainder[:end]
    return profile.strip("\n")


def dossier_profile_description(
    profile_text: str,
    display: str,
    note_count: int,
    last_activity: str,
    *,
    max_chars: int = 160,
) -> str:
    normalized = concise_description(profile_text)
    first = normalized
    if ". " in normalized:
        first = normalized.split(". ", 1)[0].strip() + "."
    candidate = first or (
        f"Dossier: {display} — {note_count} notes, "
        f"last activity {last_activity or '—'}"
    )
    return concise_description(candidate, fallback="Dossier")[:max_chars].rstrip()
