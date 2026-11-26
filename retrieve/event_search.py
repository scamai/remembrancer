"""L1 raw-event search — BM25 over raw events when L2 notes miss. Stdlib-only, ACL-gated.

Every hit carries its source event id so the answer is grounded/clickable. Gmail events
store only subject+snippet (no full body) — that lives at L0 (live fetch)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from retrieve.bm25 import rank  # noqa: E402
from remembrancerlib import iter_validated_events, event_text, FULL_ACCESS  # noqa: E402


def _event_text(e) -> str:
    parts = [event_text(e), " ".join(getattr(e, "entities", []) or []),
             str(getattr(e, "author", "") or "")]
    return " ".join(x for x in parts if x).strip()


def event_visible(acl, grants) -> bool:
    """Return whether an event ACL is visible to explicit caller grants.

    ``None`` and other non-set values deliberately do not mean full access.
    Keeping this as a public helper lets non-retrieval consumers apply exactly
    the same fail-closed rule when rendering raw event evidence.
    """
    if grants is FULL_ACCESS:
        return True
    if not acl:
        return True
    if not isinstance(grants, (set, frozenset)):
        return False
    return bool(set(acl) & set(grants))


# Backward-compatible private spelling used by older callers/tests.
_visible = event_visible


def search_events(query, grants, k=8, events_dir="events"):
    evs, docs = [], []
    for e in iter_validated_events(events_dir):
        if not event_visible(getattr(e, "acl", []) or [], grants):
            continue
        evs.append(e)
        docs.append(_event_text(e))
    out = []
    for idx, score in rank(query, docs)[:k]:
        if score <= 0:
            break
        e = evs[idx]
        snippet = event_text(e)
        out.append({"id": e.id, "date": (getattr(e, "occurred_at", "") or "")[:10],
                    "source": e.source, "snippet": snippet.strip(" —")[:240],
                    "score": round(score, 4), "acl": list(getattr(e, "acl", []) or [])})
    return out
