"""Credential redaction at the L1 boundary.

Slack threads, Drive docs and email bodies sometimes carry a live API key that a
teammate pasted to share access.  Events are immutable and the corpus is
committed to git, so a key that reaches ``events/`` is effectively permanent —
it travels into ``context/`` notes, derived indexes, LLM prompts and GitHub
history.  The only cheap place to stop it is before the first write.

:func:`redact_event` replaces credential-shaped strings inside an event's
``payload`` with a stable placeholder::

    [REDACTED:anthropic-key:1a2b3c4d]

The suffix is the first 8 hex chars of the SHA-256 of the removed value.  It
reveals nothing usable about a high-entropy key, but lets a responder match two
mentions of the same key (or a mention against a secret store) without anyone
handling the value.  ``id``, ``provenance``, ``acl`` and ``entities`` are never
touched: they are identifiers the pipeline depends on, and connectors do not
copy message bodies into them.

Detection is deliberately conservative: vendor-prefixed formats with a known
shape, plus a guarded generic ``sk-`` rule that requires key-like entropy so
prose such as ``sk-learn`` or ``sk-release-checklist`` survives.  Bare 32-hex
tokens (some vendors issue keys in that shape) are only redacted when an API-key/token word
appears just before them — a bare hex string is far more often a hash.

Existing events are NOT rewritten.  :func:`redact_payload` is also applied when
event fingerprints are compared, so re-fetching an already-stored message whose
stored copy still holds the raw key is an idempotent no-op rather than an id
conflict.

Stdlib only.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Any, Callable

# (label, compiled pattern, optional validator on the matched secret).  The
# secret is group "s" when present, else the whole match.  Order matters: the
# specific vendor rules run before the generic ``sk-`` fallback.
_Rule = tuple[str, "re.Pattern[str]", Callable[[str], bool] | None]

_B = r"(?<![A-Za-z0-9_\-])"   # left boundary: not inside a longer token
_E = r"(?![A-Za-z0-9_\-])"    # right boundary


def _key_like(value: str) -> bool:
    """High-entropy check for the generic ``sk-`` rule.

    Real keys mix upper, lower and digits over a long body; hyphenated prose
    identifiers (``sk-some-long-feature-branch-name``) do not.
    """
    body = value[3:]
    if len(body) < 32:
        return False
    classes = sum((any(c.islower() for c in body), any(c.isupper() for c in body),
                   any(c.isdigit() for c in body)))
    if classes < 3:
        return False
    # Long runs between separators are typical of keys, not of slugs.
    longest = max((len(part) for part in re.split(r"[-_]", body)), default=0)
    return longest >= 20


def _assigned_secret(value: str) -> bool:
    """Guard for ``NAME=value``: skip placeholders such as ``your-api-key-here``."""
    if value.startswith("[REDACTED:"):
        return False
    classes = sum((any(c.islower() for c in value), any(c.isupper() for c in value),
                   any(c.isdigit() for c in value)))
    if value.startswith("/") or value.startswith("."):
        return False                      # a path, not a secret
    return classes >= 2 and not re.search(r"(?i)your|example|placeholder|changeme|x{6,}", value)


_RULES: list[_Rule] = [
    ("private-key", re.compile(
        r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----.*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----",
        re.DOTALL), None),
    ("anthropic-key", re.compile(_B + r"sk-ant-[A-Za-z0-9_\-]{20,}" + _E), None),
    ("openai-key", re.compile(_B + r"sk-(?:proj|svcacct|admin)-[A-Za-z0-9_\-]{20,}" + _E), None),
    ("openai-key", re.compile(_B + r"sk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}" + _E), None),
    ("github-token", re.compile(_B + r"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})" + _E), None),
    ("slack-token", re.compile(_B + r"(?:xox[abposr]|xapp)-[0-9A-Za-z\-]{20,}" + _E), None),
    # Stripe-style live secrets; some products' own customer API keys share the shape.
    ("live-secret-key", re.compile(_B + r"(?:sk|rk)_live_[A-Za-z0-9]{16,}" + _E), None),
    ("twilio-key", re.compile(_B + r"SK[0-9a-f]{32}" + _E), None),
    ("resend-key", re.compile(_B + r"re_[A-Za-z0-9]{8,}_[A-Za-z0-9]{16,}" + _E), None),
    ("aws-access-key", re.compile(_B + r"(?:AKIA|ASIA)[0-9A-Z]{16}" + _E), None),
    ("google-api-key", re.compile(_B + r"AIza[0-9A-Za-z_\-]{35}" + _E), None),
    ("hf-token", re.compile(_B + r"hf_[A-Za-z0-9]{30,}" + _E), None),
    ("api-key", re.compile(_B + r"sk-[A-Za-z0-9_\-]{32,}" + _E), _key_like),
    # .env-style assignments pasted into chat/email: FOO_API_KEY=<value>.
    ("api-key", re.compile(
        r"\b(?:[A-Z][A-Z0-9_]*_)?(?:API_KEY|AUTH_TOKEN|ACCESS_TOKEN|SECRET_KEY|SECRET|PASSWORD|TOKEN)"
        r"[ \t]*[=:][ \t]*[\"'`]?(?P<s>[A-Za-z0-9_\-+/.=]{16,})"), _assigned_secret),
    # A bare 32-hex token is only a credential when the text says so just
    # before it ("API key: <hex>", "token is `<hex>`", "<vendor> key ... <hex>").
    ("api-key", re.compile(
        r"(?i:api[ _-]?key|access[ _-]?key|secret|token|fish[ .]?audio)"
        r"[\s\S]{0,80}?" + _B + r"(?P<s>[0-9a-f]{32})" + _E), None),
]

# Cheap pre-filter: most event payloads contain none of these, and the
# fingerprint path runs over the whole corpus.
_PREFILTER = re.compile(
    r"sk-|gh[pousr]_|github_pat_|xox|xapp-|_live_|AKIA|ASIA|AIza|hf_|re_|PRIVATE KEY|[0-9a-f]{32}"
    r"|(?:KEY|TOKEN|SECRET|PASSWORD)[ \t]*[=:]")

PLACEHOLDER_RE = re.compile(r"\[REDACTED:[a-z0-9-]+:[0-9a-f]{8}\]")


def _placeholder(label: str, secret: str) -> str:
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:8]
    return f"[REDACTED:{label}:{digest}]"


def redact_text(text: str) -> tuple[str, int]:
    """Return ``(redacted_text, count)`` for one string."""
    if not text or not _PREFILTER.search(text):
        return text, 0
    count = 0
    for label, pattern, valid in _RULES:
        def _sub(match: "re.Match[str]") -> str:
            nonlocal count
            secret = match.group("s") if "s" in pattern.groupindex else match.group(0)
            if valid is not None and not valid(secret):
                return match.group(0)
            count += 1
            whole = match.group(0)
            start = match.start("s") - match.start() if "s" in pattern.groupindex else 0
            return whole[:start] + _placeholder(label, secret) + whole[start + len(secret):]
        text = pattern.sub(_sub, text)
    return text, count


def redact_payload(value: Any) -> tuple[Any, int]:
    """Recursively redact every string in a JSON-like value.

    Returns the original object unchanged (same identity) when nothing matched,
    so callers can cheaply detect a no-op.
    """
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        total = 0
        out: dict[Any, Any] = {}
        for key, item in value.items():
            new, n = redact_payload(item)
            out[key] = new
            total += n
        return (out, total) if total else (value, 0)
    if isinstance(value, list):
        total = 0
        items = []
        for item in value:
            new, n = redact_payload(item)
            items.append(new)
            total += n
        return (items, total) if total else (value, 0)
    return value, 0


def redact_event(event: Any) -> tuple[Any, int]:
    """Return ``(event, count)`` with credentials removed from ``payload``.

    Accepts an :class:`~remembrancerlib.event.Event` (returns a new Event when
    anything changed) or a plain event dict (returns a shallow copy).
    """
    if isinstance(event, dict):
        payload, n = redact_payload(event.get("payload"))
        if not n:
            return event, 0
        copy = dict(event)
        copy["payload"] = payload
        return copy, n
    payload, n = redact_payload(getattr(event, "payload", None))
    if not n:
        return event, 0
    return replace(event, payload=payload), n
