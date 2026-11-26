"""Credibility scoring — a deterministic, stdlib-only confidence signal per note.

No LLM. The score is a pure function of data remembrancer already has: how many distinct source
events corroborate the note, whether verification (Part b) supported its claims, whether it sits
in an unresolved contradiction, and whether an approver has verified it (the approval contract). Because
scoring needs no model, `compute_confidence` can be run over the ENTIRE corpus offline —
verification just contributes its component when available (absent ⇒ `unverified`, no
penalty).

Frozen output shape (other modules depend on this shape — do not change without coordinating):

    {"score": 0.62, "level": "medium", "sources": 3,
     "verification": {"status": "verified", "checked": 4, "supported": 4, "unsupported": 0},
     "human": "none", "contradicted": false, "computed_at": "2026-07-03"}

Staleness is deliberately NOT folded into the stored score (age drifts every day, which would
make the persisted number lie). `is_stale()` is computed live at render time instead.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

# Provenance tokens that are human pseudo-provenance, not real source-event ids (see
# retrieve.query.remember: writes stamp ["by:<id>", "source:<x>", "at:<date>"]).
_HUMAN_PROV_PREFIXES = (
    "by:",
    "source:",
    "at:",
    "machine-update:",
    "generated-by:",
)

HIGH, MEDIUM, LOW = "high", "medium", "low"
DEFAULT_STALE_DAYS = 180


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def source_count(note) -> int:
    """Distinct real source-event ids in provenance (excludes by:/source:/at: pseudo-tokens)."""
    prov = getattr(note, "provenance", None) or []
    real = {
        p
        for p in prov
        if isinstance(p, str) and p and not p.startswith(_HUMAN_PROV_PREFIXES)
    }
    return len(real)


def _level(score: float) -> str:
    if score >= 0.7:
        return HIGH
    if score >= 0.4:
        return MEDIUM
    return LOW


def _base_from_sources(n: int) -> float:
    """Corroboration curve (Zep-style Fact Ratings): more independent sources ⇒ more trust."""
    if n >= 5:
        return 0.75
    if n >= 3:
        return 0.65
    if n == 2:
        return 0.50
    if n == 1:
        return 0.30
    return 0.20  # zero real sources (purely human-asserted or orphaned)


def compute_confidence(
    note,
    source_events=None,
    vres=None,
    contradicted: bool = False,
    human: str | None = None,
    stale_days: int = DEFAULT_STALE_DAYS,
) -> dict:
    """Deterministic confidence for a note. Components, then clamp to [0, 1]:

      base(sources):  0→0.20, 1→0.30, 2→0.50, 3–4→0.65, 5+→0.75   (corroboration)
      verification:   verified +0.20 ; unverified/skipped +0 ; flagged ⇒ cap score ≤ 0.35
      contradiction:  contradicted ⇒ cap score ≤ 0.30    (an unresolved conflict is not trusted)
      human:          human=='approved' ⇒ floor score = 0.90   (approval wins)

    `vres` is a distill.verify.VerificationResult (or None ⇒ unverified). `human`
    defaults to a strict read of the note: only an explicit `state=="verified"`
    means approved. A `by:` provenance token identifies an author but is not an
    approval, because human-authored agent/file-back notes still enter as proposed.
    Callers may pass an explicit verdict.
    """
    n = source_count(note)
    score = _base_from_sources(n)

    # verification component
    vstatus = getattr(vres, "status", None) if vres is not None else "unverified"
    checked = int(getattr(vres, "checked", 0) or 0) if vres is not None else 0
    supported = int(getattr(vres, "supported", 0) or 0) if vres is not None else 0
    unsupported = int(getattr(vres, "unsupported", 0) or 0) if vres is not None else 0
    if vstatus == "verified":
        score += 0.20
    verification = {
        "status": vstatus or "unverified",
        "checked": checked,
        "supported": supported,
        "unsupported": unsupported,
    }
    content_fingerprint = getattr(vres, "content_fingerprint", None)
    audit_id = getattr(vres, "audit_id", None)
    if isinstance(content_fingerprint, str) and content_fingerprint:
        verification["content_fingerprint"] = content_fingerprint
    if isinstance(audit_id, str) and audit_id:
        verification["audit_id"] = audit_id
    producer_verification = getattr(vres, "producer_verification", None)
    if isinstance(producer_verification, dict):
        # Metadata alone grants nothing: write_note still requires the sealed,
        # exact-content capability issued by the closed source validator.
        verification.update(producer_verification)

    # caps (applied after additive components; the strictest cap wins)
    if vstatus == "flagged":
        score = min(score, 0.35)
    if contradicted:
        score = min(score, 0.30)

    # Approval floor. Authorship alone never upgrades a proposed note.
    if human is None:
        from remembrancerlib.verification import human_trust_valid

        human = (
            "approved"
            if getattr(note, "state", None) == "verified" and human_trust_valid(note)
            else "none"
        )
    if human == "approved":
        score = max(score, 0.90)

    score = round(max(0.0, min(1.0, score)), 3)
    return {
        "score": score,
        "level": _level(score),
        "sources": n,
        "verification": verification,
        "human": human,
        "contradicted": bool(contradicted),
        "computed_at": _today(),
    }


def _parse_date(s: str):
    s = str(s or "")[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def is_stale(note, now=None, stale_days: int = DEFAULT_STALE_DAYS) -> dict:
    """Live staleness (never stored — age changes daily). Returns
    {updated, age_days, stale}. An unparseable/missing `updated` ⇒ not stale (age None)."""
    now = now or datetime.now(timezone.utc).date()
    if hasattr(now, "date"):  # tolerate a datetime
        now = now.date()
    updated = getattr(note, "updated", "") or ""
    d = _parse_date(updated)
    if d is None:
        return {"updated": updated, "age_days": None, "stale": False}
    age = (now - d).days
    return {"updated": updated, "age_days": age, "stale": age > stale_days}
