"""Format repair, failure ledger and health counters for an LLM's tool calls.

Under a forced ``tool_choice`` a model sometimes still answers in prose. Replaying
such answers shows what that prose is: most often the tool payload
itself, written as a fenced ```json block (sometimes with an invented key, a
"thinking" preamble, or cut off at ``max_tokens``); occasionally a genuine
refusal in words. This module turns the recoverable cases into a valid tool
payload and makes every unrecoverable one visible:

* :func:`repair_tool_payload` — fenced JSON → first balanced ``{…}`` object →
  key-name-tolerant coercion against the tool's ``input_schema``. Truncated
  JSON is rejected, never guessed at.
* :func:`repair_list_payload` — the answer written as a plain numbered/bulleted
  list with no JSON at all, which in practice is the most common unrecoverable
  ``extract_claims`` reply. Applies ONLY to a
  *pure list schema* (:func:`list_payload_schema`): one required array property
  whose items are strings, or objects with exactly one required string field.
  Each line is carried across verbatim; nothing is split or inferred.
* :func:`classify_prose` — a closed vocabulary for WHY prose was not a tool
  call (``fenced_json`` / ``json_in_prose`` / ``truncated`` / ``refusal`` /
  ``preamble`` / ``empty``), safe to log.
* :func:`record_failure` — the append-only failures log (0600; where
  ``REMEMBRANCER_LLM_FAILURES_LOG`` says, see distill/llm_env.py):
  ``{ts, reason, prompt_hash, batch_size, attempt, tool, max_tokens}`` — never
  the prompt or the response text.
* :func:`bump_health` — the health file's (``REMEMBRANCER_LLM_HEALTH``) counters
  ``{window_started, calls, prose_failures, repaired, split_retries,
  exhausted, transport_failures, pushback_failures, updated_at}``, safe across the
  nightly's shard children
  (flock + atomic replace).
* :func:`set_health` — the same file's non-counter fields (the adaptive
  concurrency level and its ceiling), so the level the fleet actually runs at
  is observable instead of inferred.

Everything here is best-effort accounting around the decision path: no function
raises into the caller, and nothing reads a prompt back out of a file.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone

from distill.llm_env import (FAILURES_LOG_DEFAULT, HEALTH_PATH_DEFAULT, LLM_ENV,
                             llm_setting)

FAILURES_LOG_ENV = LLM_ENV["failures_log"]
HEALTH_PATH_ENV = LLM_ENV["health"]

HEALTH_COUNTERS = ("calls", "prose_failures", "repaired", "split_retries", "exhausted",
                   "transport_failures", "pushback_failures")
#: Non-counter fields ``set_health`` may replace outright.  Keeping this closed
#: means the health file stays a machine-owned snapshot with a known shape: the
#: metrics exporter and a dashboard can both read it without a schema guess.
HEALTH_FIELDS = ("concurrency", "concurrency_max", "concurrency_shares",
                 "concurrency_updated_at", "concurrency_reason")

_FENCE_RE = re.compile(r"```(?:json|JSON|javascript|js)?\s*\n?(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_REFUSAL_MARKERS = (
    "no factual claim", "does not contain", "doesn't contain", "contains no",
    "nothing to", "cannot ", "can't ", "unable to", "not possible", "no durable",
    "no relevant", "i'm sorry", "i am sorry", "insufficient", "not enough information",
    "no information", "none of the", "there are no", "there is no",
)
_PREAMBLE_MARKERS = ("i'll ", "i will ", "let me ", "looking at", "analyzing", "i need to",
                     "first,", "based on the")

_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Recovering a payload from prose                                             #
# --------------------------------------------------------------------------- #

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def prompt_hash(prompt: str) -> str:
    """Short stable digest of a prompt — enough to correlate, never to recover it."""
    return hashlib.sha256((prompt or "").encode("utf-8", "replace")).hexdigest()[:16]


def _balanced_objects(text: str) -> list[str]:
    """Every top-level ``{…}`` span with balanced braces (string-aware), in order."""
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(text[start:i + 1])
                    start = -1
    return spans


def looks_truncated(text: str) -> bool:
    """An opening brace with no balancing close, or an unterminated string."""
    if not text:
        return False
    if "{" not in text:
        return False
    if _balanced_objects(text):
        return False
    return True


def _loads_tolerant(raw: str):
    """``json.loads`` with the two harmless fixes model output needs (trailing commas,
    Python literals). Anything else is a rejection — never a guess."""
    try:
        return json.loads(raw)
    except ValueError:
        pass
    fixed = _TRAILING_COMMA_RE.sub(r"\1", raw)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    try:
        return json.loads(fixed)
    except ValueError:
        return None


def extract_candidates(text: str) -> list[dict]:
    """JSON objects found in prose: fenced blocks first, then balanced ``{…}`` spans."""
    if not text:
        return []
    found: list[dict] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        raw = raw.strip()
        if not raw or raw in seen:
            return
        seen.add(raw)
        obj = _loads_tolerant(raw)
        if isinstance(obj, dict):
            found.append(obj)
        elif isinstance(obj, list):
            # A bare list of the tool's one array property is not a payload;
            # let coercion decide later by wrapping it in nothing.
            return

    for m in _FENCE_RE.finditer(text):
        body = m.group(1)
        _add(body)
        for span in _balanced_objects(body):
            _add(span)
    for span in _balanced_objects(text):
        _add(span)
    return found


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _type_ok(value, types) -> bool:
    if types is None:
        return True
    if isinstance(types, str):
        types = [types]
    for t in types:
        if t == "null" and value is None:
            return True
        if t == "string" and isinstance(value, str):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "object" and isinstance(value, dict):
            return True
    return False


def _coerce_value(value, schema):
    """Validate/coerce one value against a (sub)schema. Returns (ok, value)."""
    if not isinstance(schema, dict):
        return True, value
    types = schema.get("type")
    if isinstance(types, str):
        types = [types]
    # Scalar-to-string leniency: a number where a string is wanted.
    if types and "string" in types and isinstance(value, (int, float)) \
            and not isinstance(value, bool):
        value = str(value)
    if types and not _type_ok(value, types):
        # A single object where an array of objects is wanted.
        if "array" in types and isinstance(value, dict):
            value = [value]
        elif "array" in types and isinstance(value, str) and value.strip() == "":
            value = []
        elif "null" in types and value == "":
            value = None
        else:
            return False, None
    enum = schema.get("enum")
    if enum is not None and value is not None:
        if value not in enum:
            # Case-insensitive enum match (a model writes "ignore" for "IGNORE").
            match = [e for e in enum if isinstance(e, str) and isinstance(value, str)
                     and e.lower() == value.strip().lower()]
            if not match:
                return False, None
            value = match[0]
    if isinstance(value, dict) and (schema.get("properties") or schema.get("required")):
        return _coerce_object(value, schema)
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        out = []
        for item in value:
            ok, coerced = _coerce_value(item, schema["items"])
            if not ok:
                return False, None
            out.append(coerced)
        return True, out
    return True, value


def _coerce_object(obj: dict, schema: dict):
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    by_norm = {_norm_key(k): k for k in props}
    out: dict = {}
    for key, value in obj.items():
        canon = props.get(key) is not None and key or by_norm.get(_norm_key(key))
        if canon is None:
            if props:
                continue            # unknown key under a typed schema: drop it
            out[key] = value        # untyped schema: keep as-is
            continue
        ok, coerced = _coerce_value(value, props[canon])
        if not ok:
            if canon in required:
                return False, None
            continue                # optional field with a bad shape: drop it
        out[canon] = coerced
    for key in required:
        if key not in out:
            return False, None
    return True, out


def coerce_to_schema(obj, schema: dict | None):
    """Key-name-tolerant validation of an untrusted object against a tool's
    ``input_schema``. Returns the coerced dict or ``None``."""
    if not isinstance(obj, dict):
        return None
    if not isinstance(schema, dict) or not (schema.get("properties") or schema.get("required")):
        return dict(obj)
    ok, coerced = _coerce_object(obj, schema)
    return coerced if ok else None


_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+•·]|\d{1,3}[.)])\s+(.*\S)\s*$")
_MARKDOWN_EMPHASIS_RE = re.compile(r"(?<!\w)[*_]{1,3}(?=\S)|(?<=\S)[*_]{1,3}(?!\w)")


def list_payload_schema(schema) -> tuple[str, str | None] | None:
    """``(property, item_field)`` when a schema is a *pure list* schema.

    A pure list schema has exactly one property, that property is required, it
    is an array, and its items are either plain strings (``item_field`` None) or
    objects with exactly ONE required property, itself a string.  Only such a
    schema can be filled from a written list without inventing anything: each
    line supplies the single required value and nothing else is guessed.

    ``extract_claims`` matches; ``verify_claims`` (claim + verdict + event ids +
    quote) and ``consolidate`` deliberately do not.
    """
    if not isinstance(schema, dict):
        return None
    props = schema.get("properties")
    required = schema.get("required")
    if not isinstance(props, dict) or len(props) != 1:
        return None
    if not isinstance(required, list) or len(required) != 1:
        return None
    prop = required[0]
    spec = props.get(prop)
    if not isinstance(spec, dict) or spec.get("type") != "array":
        return None
    items = spec.get("items")
    if not isinstance(items, dict):
        return None
    if items.get("type") == "string" and not items.get("enum"):
        return prop, None
    if items.get("type") != "object":
        return None
    item_required = items.get("required")
    item_props = items.get("properties")
    if not isinstance(item_required, list) or len(item_required) != 1:
        return None
    if not isinstance(item_props, dict):
        return None
    field = item_required[0]
    field_spec = item_props.get(field)
    if not isinstance(field_spec, dict) or field_spec.get("type") != "string" \
            or field_spec.get("enum"):
        return None
    return prop, field


_SENTENCE_END = (".", "!", "?", "。", "！", "？", '."', ".'", ".)", ".”", ".’")


def _drop_cut_off_tail(items: list[str]) -> list[str]:
    """Drop a final item that reads as cut off mid-sentence.

    A written list carries no syntactic sign of truncation the way unbalanced
    JSON does, so the shape of the answer has to supply it: when the model has
    been punctuating every item and the last one stops without punctuation, that
    last line is a fragment. Half a claim is not a claim.
    """
    if len(items) < 2:
        return items
    punctuated = sum(1 for item in items[:-1] if item.endswith(_SENTENCE_END))
    if punctuated * 5 >= len(items[:-1]) * 3 and not items[-1].endswith(_SENTENCE_END):
        return items[:-1]
    return items


def _written_list_items(text: str) -> list[str]:
    """Bullet/numbered lines of a written list, or ``[]``.

    Conservative on purpose: at least two items, and the items must be the bulk
    of the answer (half its non-blank lines), so a refusal paragraph that
    happens to contain one dash is never read as a payload.
    """
    lines = [line for line in (text or "").splitlines() if line.strip()]
    items: list[str] = []
    for line in lines:
        match = _LIST_ITEM_RE.match(line)
        if match is None:
            continue
        value = _MARKDOWN_EMPHASIS_RE.sub("", match.group(1)).strip()
        if value:
            items.append(value)
    if len(items) < 2 or len(items) * 2 < len(lines):
        return []
    items = _drop_cut_off_tail(items)
    return items if len(items) >= 2 else []


def _bare_arrays(text: str) -> list[list]:
    """Top-level JSON arrays in the text, fenced blocks first."""
    found: list[list] = []
    seen: set[str] = set()
    for candidate in _FENCE_RE.findall(text or "") + [text or ""]:
        raw = candidate.strip()
        if not raw.startswith("[") or raw in seen:
            continue
        seen.add(raw)
        obj = _loads_tolerant(raw)
        if isinstance(obj, list):
            found.append(obj)
    return found


def repair_list_payload(text: str, tool: dict) -> dict | None:
    """Recover a *pure list* tool payload from a written list, or ``None``.

    The model answered the question and only got the envelope wrong: under a
    forced ``tool_choice`` it writes the list of claims as text. Every line is
    carried across verbatim — nothing is parsed out of it, split, or inferred —
    and the recovered items face exactly the same downstream gates as a native
    tool call, so this recovers the envelope without touching what the content
    is allowed to prove. A truncated answer is refused: its last line may be a
    fragment, and half a claim is not a claim.
    """
    if not text or not isinstance(tool, dict):
        return None
    shape = list_payload_schema(tool.get("input_schema"))
    if shape is None:
        return None
    prop, field = shape
    schema = tool.get("input_schema")
    # The array itself, with no object around it: for a pure list schema there
    # is exactly one property it could belong to, so wrapping invents nothing.
    for array in _bare_arrays(text):
        payload = coerce_to_schema({prop: array}, schema)
        if payload:
            return payload
    if looks_truncated(text):
        return None
    items = _written_list_items(text)
    if not items:
        return None
    values = items if field is None else [{field: item} for item in items]
    return coerce_to_schema({prop: values}, schema)


# The provider's own tool-call encoding, arriving as a text block instead of a
# parsed ``tool_use``: an XML-ish tree whose element names are the schema's, cut
# up by a fixed separator token. It is the highest-fidelity recovery available —
# these are the model's structured fields, not a re-reading of its prose.
_PROVIDER_JOIN_RE = re.compile(r"\]<\]minimax\[>\[")
_ARRAY_ITEM_TAG = "item"


def _tag_block(text: str, tag: str, start: int = 0) -> tuple[str, int] | None:
    """Inner text of the first ``<tag>…</tag>`` at or after ``start``, and the
    offset just past it. Depth-aware, because the same element name nests
    (``<item>`` holds verdicts and, inside one, event ids)."""
    open_tag, close_tag = f"<{tag}>", f"</{tag}>"
    begin = text.find(open_tag, start)
    if begin < 0:
        return None
    cursor = begin + len(open_tag)
    depth = 1
    while depth:
        next_open = text.find(open_tag, cursor)
        next_close = text.find(close_tag, cursor)
        if next_close < 0:
            return None                       # unterminated: refuse, never guess
        if 0 <= next_open < next_close:
            depth += 1
            cursor = next_open + len(open_tag)
            continue
        depth -= 1
        cursor = next_close + len(close_tag)
    return text[begin + len(open_tag):cursor - len(close_tag)], cursor


def _xml_value(inner: str, schema):
    types = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(types, str):
        types = [types]
    types = types or []
    if "array" in types:
        items = []
        position = 0
        while True:
            found = _tag_block(inner, _ARRAY_ITEM_TAG, position)
            if found is None:
                break
            body, position = found
            items.append(_xml_value(body, schema.get("items") or {}))
        return items
    if "object" in types:
        out: dict = {}
        for prop, sub in (schema.get("properties") or {}).items():
            found = _tag_block(inner, prop)
            if found is not None:
                out[prop] = _xml_value(found[0], sub)
        return out
    return inner.strip()


def repair_provider_markup(text: str, tool: dict) -> dict | None:
    """Recover the provider's leaked tool-call markup, or ``None``.

    Element names must be the schema's own, and the result goes through
    :func:`coerce_to_schema` unchanged, so a missing required field takes the
    whole payload down rather than being quietly dropped — the same rule
    truncated JSON already gets.
    """
    if not text or not isinstance(tool, dict):
        return None
    schema = tool.get("input_schema")
    if not isinstance(schema, dict) or not schema.get("properties"):
        return None
    body = _PROVIDER_JOIN_RE.sub("", text)
    if "<" not in body:
        return None
    obj = _xml_value(body, {"type": "object", "properties": schema["properties"]})
    if not obj:
        return None
    return coerce_to_schema(obj, schema)


def repair_tool_payload(text: str, tool: dict) -> dict | None:
    """Recover a valid ``tool`` payload from prose, or ``None``.

    Order: fenced ```json blocks, then balanced ``{…}`` objects in reading order;
    the first candidate that validates against ``tool["input_schema"]`` wins.
    A candidate that merely WRAPS the payload (``{"input": {...}}``,
    ``{"arguments": {...}}``, ``{"<tool name>": {...}}``) is unwrapped once.
    Then the provider's own leaked tool-call markup
    (:func:`repair_provider_markup`), and last, for a *pure list* schema only, a
    written list read as the payload (:func:`repair_list_payload`).
    """
    if not text or not isinstance(tool, dict):
        return None
    schema = tool.get("input_schema")
    name = str(tool.get("name") or "")
    for cand in extract_candidates(text):
        payload = coerce_to_schema(cand, schema)
        if payload is not None and (payload or not schema):
            return payload
        for wrapper in ("input", "arguments", "parameters", "tool_input", name):
            inner = cand.get(wrapper) if wrapper else None
            if isinstance(inner, dict):
                payload = coerce_to_schema(inner, schema)
                if payload is not None and (payload or not schema):
                    return payload
    payload = repair_provider_markup(text, tool)
    if payload:
        return payload
    return repair_list_payload(text, tool)


def classify_prose(text: str, stop_reason: str | None = None) -> str:
    """Closed vocabulary for why a text response was not a tool call."""
    body = (text or "").strip()
    if not body:
        return "empty"
    if stop_reason == "max_tokens" or looks_truncated(body):
        return "truncated"
    if _FENCE_RE.search(body):
        return "fenced_json"
    if _balanced_objects(body):
        return "json_in_prose"
    low = body.lower()
    if any(m in low for m in _REFUSAL_MARKERS):
        return "refusal"
    if any(low.startswith(m) or f" {m}" in low[:200] for m in _PREAMBLE_MARKERS):
        return "preamble"
    return "prose"


# --------------------------------------------------------------------------- #
# Failure ledger + health counters                                            #
# --------------------------------------------------------------------------- #

def failures_log_path() -> str | None:
    """``None`` when accounting is switched off (``REMEMBRANCER_LLM_FAILURES_LOG=off``,
    the test suite's setting — the same convention as ``REMEMBRANCER_AUDIT_LOG``)."""
    path = llm_setting("failures_log", FAILURES_LOG_DEFAULT)
    return None if path.strip().lower() == "off" else path


def health_path() -> str | None:
    path = llm_setting("health", HEALTH_PATH_DEFAULT)
    return None if path.strip().lower() == "off" else path


def record_failure(reason: str, *, prompt: str = "", prompt_hash_value: str | None = None,
                   batch_size: int | None = None, attempt: int = 0,
                   tool: str | None = None, max_tokens: int | None = None) -> None:
    """Append one exhausted-job record. Never raises; never stores prompt text."""
    try:
        path = failures_log_path()
        if path is None:
            return
        rec = {
            "ts": _now(),
            "reason": str(reason)[:80],
            "prompt_hash": prompt_hash_value or prompt_hash(prompt),
            "batch_size": batch_size,
            "attempt": int(attempt),
        }
        if tool:
            rec["tool"] = str(tool)[:60]
        if max_tokens is not None:
            rec["max_tokens"] = int(max_tokens)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(rec, sort_keys=True) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


def _empty_health() -> dict:
    now = _now()
    doc = {"window_started": now, "updated_at": now}
    doc.update({k: 0 for k in HEALTH_COUNTERS})
    return doc


def load_health(path: str | None = None) -> dict | None:
    """The parsed health file (``health_path()``), or ``None`` when absent/malformed."""
    path = path or health_path()
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    for k in HEALTH_COUNTERS:
        try:
            doc[k] = int(doc.get(k) or 0)
        except (TypeError, ValueError):
            doc[k] = 0
    return doc


def bump_health(**counters: int) -> None:
    """Increment counters in the health file (created on first use).

    Read-modify-write under an exclusive ``flock`` on ``<path>.lock`` so the
    nightly's parallel shard children never lose each other's increments;
    the file itself is replaced atomically (0600). Never raises.
    """
    increments = {k: int(v) for k, v in counters.items() if k in HEALTH_COUNTERS and v}
    if not increments:
        return
    _update_health(increments, {})


def set_health(**fields) -> None:
    """Replace (not increment) the non-counter fields in ``HEALTH_FIELDS``.

    The adaptive concurrency level is a LEVEL, not a total: summing it across
    the nightly's shard children would be meaningless, and a level nobody can
    read back is a level nobody can operate. Written through the same
    flock + atomic-replace path as the counters, so a shard child publishing
    its level never truncates a sibling's increments. Never raises.
    """
    replacements = {k: v for k, v in fields.items() if k in HEALTH_FIELDS}
    if not replacements:
        return
    _update_health({}, replacements)


def _update_health(increments: dict, replacements: dict) -> None:
    try:
        path = health_path()
        if path is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        lock_path = path + ".lock"
        with _lock:
            lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                doc = load_health(path) or _empty_health()
                for k, v in increments.items():
                    doc[k] = int(doc.get(k) or 0) + v
                doc.update(replacements)
                doc["updated_at"] = _now()
                tmp = f"{path}.tmp.{os.getpid()}"
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    os.write(fd, json.dumps(doc, indent=2, sort_keys=True).encode("utf-8"))
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.replace(tmp, path)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
    except Exception:
        pass
