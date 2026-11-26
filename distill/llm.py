"""Pluggable LLM client for distillation, over the Anthropic SDK. No key => unavailable.

Each setting is one ``REMEMBRANCER_LLM_*`` environment variable (distill/llm_env.py).
A ``REMEMBRANCER_LLM_API_KEY`` gets the SDK's own endpoint unless
``REMEMBRANCER_LLM_BASE_URL`` names another, and has no default model:
``REMEMBRANCER_LLM_MODEL`` must be set.

Every completed logical call is appended to a usage log (``REMEMBRANCER_LLM_USAGE_LOG``;
its default is in distill/llm_env.py) so "how many LLM calls and tokens?" is
answerable at any time. Each row carries the CONSUMER that spent it
(``caller``), taken from ``REMEMBRANCER_LLM_CALLER`` (set per command by whatever
runs the pipeline) or from ``caller_scope`` for a process whose phases are
different consumers; unlabelled calls say ``unknown`` rather than guess.

Tool-call reliability. A model sometimes answers in prose instead of calling
the forced tool (about 1.5% of calls on the model this client was built
against); the prose is usually the payload itself in a ```json fence.
``decide`` therefore layers, in order:
  1. format repair  — recover a schema-valid payload from the text (no call);
  2. repair retry   — one re-send with a "respond only with the tool call"
                      system reminder, the prior text as context, and a larger
                      ``max_tokens`` when the first answer was cut off;
  3. never silent   — an exhausted job is appended to the failures log
                      (``REMEMBRANCER_LLM_FAILURES_LOG``) and counted in the health
                      file (``REMEMBRANCER_LLM_HEALTH``) before it raises.
``decide_many`` adds a split retry: the prose-failed subset of a batch is re-run
alone at halved concurrency (down to 1) in strict mode.

Throughput. ``decide_many``'s fan-out is no longer a
constant: ``AdaptiveConcurrency`` starts at ``REMEMBRANCER_LLM_CONCURRENCY`` (default 8,
divided by ``REMEMBRANCER_LLM_CONCURRENCY_SHARES`` so the knob describes account-wide
in-flight requests), adds one worker per quiet interval while the sliding-window
failure rate stays under 2%, and halves within seconds of a 429/5xx/timeout
burst — floor 4, ceiling ``REMEMBRANCER_LLM_CONCURRENCY_MAX`` (16). The live level is
published to the health file.
"""
import sys
import json
import os
import random
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone

from distill import llm_repair
from distill.llm_env import (LEGACY_KEY_DEFAULTS, LEGACY_LLM_ENV, LLM_ENV,
                             USAGE_LOG_DEFAULT, llm_float, llm_int, llm_setting)

# Markers that an error is a rate-limit / concurrency-limit / overload (back off long),
# vs an ordinary transient error (back off short). A provider may cap concurrency, so
# parallel callers WILL hit this — the back-off lets them self-throttle instead of failing.
_RATE_MARKERS = ("429", "529", "rate limit", "rate_limit", "too many requests",
                 "concurrency", "concurrent", "overloaded", "quota", "throttl")
MAX_ATTEMPTS = llm_int("max_attempts", 7)
# Extra sends after a prose answer that format repair could not recover (layer 2).
REPAIR_RETRIES = llm_int("repair_retries", 1)
# Split-retry rounds in decide_many for the prose-failed subset of a batch (layer 3).
SPLIT_RETRIES = llm_int("split_retries", 1)
# Ceiling for the enlarged max_tokens on a truncated first answer.
MAX_TOKENS_CEILING = llm_int("max_tokens_ceiling", 4000)

REPAIR_SYSTEM = (
    "Respond ONLY by calling the `{tool}` tool with a JSON object that matches its "
    "input schema exactly. Do not write prose, explanations, or markdown fences, and do "
    "not add keys the schema does not define. If the input supports nothing, still call "
    "the tool with its empty/negative option (empty arrays, null fields, or the "
    "IGNORE-style action) rather than explaining in words."
)
REPAIR_FOLLOWUP = (
    "That reply was text, not a tool call. Call the `{tool}` tool now with the same "
    "content as a valid JSON object matching its schema."
)

# decide_many tells decide how large the batch around a job is (for the ledger).
_batch_ctx = threading.local()


class LLMUnavailable(Exception):
    """Raised when no LLM is configured/available so callers fall back to the
    deterministic distiller (no key, missing SDK, or repeated API failure)."""


class LLMDeclined(LLMUnavailable):
    """The model answered, but declined to call the tool.

    Subclasses LLMUnavailable on purpose: every existing ``except
    LLMUnavailable`` keeps working unchanged, so this cannot alter any caller's
    behaviour. What it adds is the ability to stop retrying a deterministic
    refusal, and to tell the two apart wherever that matters.

    Under ``tool_choice`` a model sometimes returns a bare text block explaining why
    it produced no structured output ("the provided text contains no factual
    claims..."). That is a VERDICT, not a transport failure: it is deterministic,
    so retrying it seven times returns the same refusal seven times and then
    reports an outage. Callers must be able to tell "the model looked and found
    nothing" apart from "the model could not be reached" -- conflating them is
    how notes end up at status="skipped", checked=0, and are never verified.
    """


def _error_category(exc: BaseException) -> str:
    """A closed vocabulary for why a call failed.

    Deliberately carries no upstream text: categories are fixed strings, so they
    can be logged, counted and alerted on without leaking request ids, URLs or
    echoed note content into status files.
    """
    name = type(exc).__name__.lower()
    text = f"{name} {exc}".lower()
    if "no tool_use" in text:
        return "no_tool_use"
    if "timeout" in text or "timedout" in text:
        return "timeout"
    if "429" in text or "rate" in text and "limit" in text:
        return "rate_limit"
    if "connection" in text or "connect" in name:
        return "connection"
    for code in ("400", "401", "403", "404", "413", "422"):
        if code in text:
            return f"http_{code}"
    for code in ("500", "502", "503", "504"):
        if code in text:
            return f"http_{code}"
    return "other"


def _retry_plan(exc, attempt: int) -> tuple[bool, float]:
    """Given an exception and 0-based attempt #, return (is_rate_limited, delay_seconds).
    Rate-limit/overload => exponential backoff with jitter (honors Retry-After if present);
    other transient => short linear backoff."""
    txt = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    is_rate = status in (429, 529) or any(m in txt for m in _RATE_MARKERS)
    if is_rate:
        retry_after = None
        try:
            ra = getattr(exc, "response", None).headers.get("retry-after")
            retry_after = float(ra) if ra is not None else None
        except Exception:
            retry_after = None
        delay = retry_after if retry_after else min(60.0, (2.0 ** attempt) + random.random())
        return True, delay
    return False, 0.5 * (attempt + 1)


# --------------------------------------------------------------------------- #
# Adaptive concurrency                                                        #
# --------------------------------------------------------------------------- #
# The fixed default of 4 was chosen when nothing measured what the provider would take,
# and a sharded pipeline multiplies it by its shard count. A direct probe
# (identical tool calls, fixed fan-out, nothing else running) showed that
# throughput does not rise with fan-out — past ~8 in flight it FALLS, because
# the account queues and then refuses, and provider pushback (429/5xx/timeout)
# climbs with every step past that knee. Several shards at 4 each sit far past
# it, and transport failures then dominate: a backlog is not starved of
# concurrency, it is drowning in it.
#
# A constant cannot be right twice — 4 is below the knee on a quiet day, a
# sharded total is far above it, and the knee itself moves with whatever else
# shares the account. So the level is a
# control loop: additive increase while the observed failure rate stays under
# the threshold, multiplicative decrease the moment a 429/5xx/timeout burst
# appears. Back off fast, recover slowly. The defaults start just under the
# measured knee and let the loop find it.
DEFAULT_CONCURRENCY = llm_int("concurrency_default", 8)
DEFAULT_CONCURRENCY_MIN = 4
DEFAULT_CONCURRENCY_MAX = 16
#: Failure-rate ceiling for the sliding window: above it, back off.
DEFAULT_FAILURE_THRESHOLD = 0.02
#: Sliding-window horizon and cap. Long enough to be a rate, short enough that
#: a level set during an outage does not survive the outage.
WINDOW_SECONDS = 180.0
WINDOW_MAX_SAMPLES = 400
#: Ramp gate: how much evidence, and how much quiet time, one +1 step costs.
RAMP_MIN_SAMPLES = 20
RAMP_INTERVAL_SECONDS = 30.0
#: Burst rule: this many back-off-worthy failures inside this window halves the
#: level immediately, without waiting for the batch to finish.
BURST_FAILURES = 3
BURST_SECONDS = 20.0
#: Evidence a *rate* needs before it may halve the level at settle time. Without
#: it one failure in a three-call batch reads as a 33% failure rate; real storms
#: are caught by the burst rule above, which needs no window at all.
BACKOFF_MIN_SAMPLES = 10
#: Categories (from ``_error_category``) that mean "the provider pushed back",
#: as opposed to a schema refusal, which more concurrency neither causes nor cures.
BACKOFF_CATEGORIES = frozenset({
    "rate_limit", "timeout", "connection",
    "http_500", "http_502", "http_503", "http_504",
})


#: Why a client with a key but no model is off.
NO_MODEL = (f"no {LLM_ENV['model']}: this key has no default model, so LLM "
            "calls are off")


def llm_requirement() -> str:
    """The variables that turn the LLM on, in words, for messages and help."""
    text = f"{LLM_ENV['api_key']} and {LLM_ENV['model']}"
    older = LEGACY_LLM_ENV.get("api_key")
    if older and LEGACY_KEY_DEFAULTS.get("model"):
        text += f", or {older}, which brings its own endpoint and model"
    return text


_WARNED: set[str] = set()


def _warn_once(message: str) -> None:
    """A misconfiguration the pipeline survives (it runs without the LLM) must
    still be said once, or the run looks like a working one."""
    if message not in _WARNED:
        _WARNED.add(message)
        print(f"[llm] {message}", file=sys.stderr)


def concurrency_config() -> dict:
    """The configured window the controller may move inside — clock-free.

    Read from the environment on every call so a shard child that inherits a
    different budget needs no plumbing, and so callers (digest's startup line,
    the metrics exporter) can report the budget without disturbing the loop.
    """
    shares = max(1, llm_int("concurrency_shares", 1))
    start = llm_int("concurrency", DEFAULT_CONCURRENCY)
    floor = llm_int("concurrency_min", DEFAULT_CONCURRENCY_MIN)
    ceiling = llm_int("concurrency_max", DEFAULT_CONCURRENCY_MAX)
    threshold = llm_float("failure_threshold", DEFAULT_FAILURE_THRESHOLD)
    adaptive = llm_setting("adaptive_concurrency", "1") != "0"
    # Per-process shares of an account-wide budget.
    start = max(1, start // shares)
    floor = max(1, floor // shares)
    ceiling = max(1, ceiling // shares)
    # A deliberately small start must not be raised by the default floor, and a
    # ceiling below the start is a typo, not an instruction to drop.
    floor = min(floor, start)
    ceiling = max(ceiling, start)
    return {
        "start": start, "floor": floor, "ceiling": ceiling, "shares": shares,
        "threshold": max(0.0, threshold), "adaptive": adaptive,
    }


class AdaptiveConcurrency:
    """Self-tuning fan-out level for ``decide_many`` (AIMD).

    Configuration is read from the environment on every use, so a shard child
    that inherits a different budget than its parent needs no plumbing:

    ``REMEMBRANCER_LLM_CONCURRENCY``     start level (default 8)
    ``REMEMBRANCER_LLM_CONCURRENCY_MIN`` floor (default 4; never lower than an
                                  explicitly configured start, so a deliberate
                                  ``REMEMBRANCER_LLM_CONCURRENCY=2`` is still honoured)
    ``REMEMBRANCER_LLM_CONCURRENCY_MAX`` ceiling (default 16)
    ``REMEMBRANCER_LLM_CONCURRENCY_SHARES`` sibling processes sharing ONE LLM account
                                  (the sharded digest parent sets this to the
                                  shard count). Every level above is divided by
                                  it, so the knobs describe account-wide
                                  in-flight requests no matter how many
                                  processes the pipeline splits into — the
                                  provider's cap is per account, not per pid.
    ``REMEMBRANCER_LLM_ADAPTIVE_CONCURRENCY=0`` pins the level to the start value.

    The level is published to the health file whenever it changes, so
    "what concurrency is the fleet running at?" is a gauge, not an inference.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: deque = deque()
        self._level: int | None = None
        self._config: tuple | None = None
        self._last_change = 0.0
        self._reason = "initial"

    # -- configuration ----------------------------------------------------- #
    @staticmethod
    def _read_config() -> tuple[int, int, int, int, float, bool]:
        cfg = concurrency_config()
        return (cfg["start"], cfg["floor"], cfg["ceiling"], cfg["shares"],
                cfg["threshold"], cfg["adaptive"])

    def _ensure(self) -> tuple[int, int, int, int, float, bool]:
        config = self._read_config()
        if config != self._config:
            # New budget (or first use): reset the loop rather than carrying a
            # level that was tuned against different limits.
            self._config = config
            self._level = config[0]
            self._samples.clear()
            self._last_change = time.monotonic()
            self._reason = "initial"
        return config

    # -- observation ------------------------------------------------------- #
    def _prune(self, now: float) -> None:
        horizon = now - WINDOW_SECONDS
        while self._samples and self._samples[0][0] < horizon:
            self._samples.popleft()
        while len(self._samples) > WINDOW_MAX_SAMPLES:
            self._samples.popleft()

    def note_success(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._ensure()
            self._samples.append((now, False, ""))
            self._prune(now)

    def note_failure(self, category: str) -> None:
        """Record one failed provider attempt; halve immediately on a burst."""
        now = time.monotonic()
        with self._lock:
            _start, floor, _ceiling, _shares, _threshold, adaptive = self._ensure()
            self._samples.append((now, True, category))
            self._prune(now)
            if not adaptive or category not in BACKOFF_CATEGORIES:
                return
            recent = sum(
                1 for ts, failed, cat in self._samples
                if failed and cat in BACKOFF_CATEGORIES and ts >= now - BURST_SECONDS
            )
            if recent >= BURST_FAILURES:
                self._set_level(max(floor, (self._level or floor) // 2),
                                f"burst:{category}", now)

    # -- control ----------------------------------------------------------- #
    def _set_level(self, level: int, reason: str, now: float) -> None:
        level = max(1, int(level))
        if level == self._level:
            return
        self._level = level
        self._reason = reason
        self._last_change = now
        self._publish()

    def _publish(self) -> None:
        _start, _floor, ceiling, shares, _threshold, _adaptive = (
            self._config or self._read_config()
        )
        llm_repair.set_health(
            concurrency=int(self._level or 0),
            concurrency_max=int(ceiling),
            concurrency_shares=int(shares),
            concurrency_reason=str(self._reason),
            concurrency_updated_at=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        )

    def level(self) -> int:
        with self._lock:
            self._ensure()
            return max(1, int(self._level or 1))

    def settle(self) -> int:
        """Apply the AIMD step after a completed batch; return the new level."""
        now = time.monotonic()
        with self._lock:
            _start, floor, ceiling, _shares, threshold, adaptive = self._ensure()
            if not adaptive:
                return max(1, int(self._level or 1))
            self._prune(now)
            total = len(self._samples)
            failures = sum(1 for _ts, failed, _cat in self._samples if failed)
            rate = (failures / total) if total else 0.0
            if total >= BACKOFF_MIN_SAMPLES and rate > threshold:
                self._set_level(max(floor, (self._level or floor) // 2),
                                f"failure_rate:{rate:.3f}", now)
            elif (
                total >= RAMP_MIN_SAMPLES
                and rate <= threshold / 2
                and now - self._last_change >= RAMP_INTERVAL_SECONDS
                and (self._level or 0) < ceiling
            ):
                self._set_level((self._level or floor) + 1,
                                f"ramp:{rate:.3f}", now)
            return max(1, int(self._level or 1))

    def snapshot(self) -> dict:
        """Current level + window, for logs and tests (never raises)."""
        now = time.monotonic()
        with self._lock:
            _start, floor, ceiling, shares, threshold, adaptive = self._ensure()
            self._prune(now)
            total = len(self._samples)
            failures = sum(1 for _ts, failed, _cat in self._samples if failed)
            return {
                "level": max(1, int(self._level or 1)),
                "floor": floor,
                "ceiling": ceiling,
                "shares": shares,
                "threshold": threshold,
                "adaptive": adaptive,
                "samples": total,
                "failures": failures,
                "failure_rate": (failures / total) if total else 0.0,
                "reason": self._reason,
            }


#: One controller per process: every ``decide_many`` without an explicit
#: ``max_workers`` shares it, so the level reflects the whole process's
#: experience of the provider rather than one batch's.
CONCURRENCY = AdaptiveConcurrency()


def _usage_token_counts(usage) -> tuple[int | None, int | None]:
    """Tolerant (input, output) token extraction.

    An Anthropic-compatible endpoint does not always populate the SDK's
    ``Usage`` object attributes (rows of ``null`` in the usage log made
    throughput accounting unverifiable). Handle object attrs, plain dicts,
    and the OpenAI-style aliases the endpoint sometimes echoes instead.
    """
    if usage is None:
        return None, None

    def get(key):
        if isinstance(usage, dict):
            return usage.get(key)
        return getattr(usage, key, None)

    inp = get("input_tokens")
    if inp is None:
        inp = get("prompt_tokens")
    out = get("output_tokens")
    if out is None:
        out = get("completion_tokens")
    return inp, out


# --------------------------------------------------------------------------- #
# Caller attribution                                                          #
# --------------------------------------------------------------------------- #
# ``_log_usage`` is a module-level function with no caller context, and there
# can be many independent ``LLMClient()`` construction sites across distill/,
# tools/ and scripts/. Threading a ``caller=`` parameter from each of them down through
# ``decide`` / ``decide_many`` / ``_log_usage`` would be a wide, mechanical
# change to every one of those files and to the frozen-ish call contract, for a
# fact that is constant for the whole process anyway. So the label is PROCESS
# scope, read from the environment the same way the usage log's path
# (``REMEMBRANCER_LLM_USAGE_LOG``) already is, and set per command by whatever starts
# the process.
#
# The known weakness of a process-wide label is a process that spends LLM calls as two
# different consumers AT THE SAME TIME — its threads would share one label. That
# does not happen here, and the shape of the code is why:
#
#   * every parallel LLM path is homogeneous. ``decide_many`` fans one batch of
#     one consumer's jobs out over a pool, and a tool with a pool of its own runs
#     only its own jobs. A pool never mixes consumers, so a pool never needs two
#     labels.
#   * a process that spends as two consumers runs them in strictly sequential
#     phases: the first phase's pool returns before the second phase's pool is
#     created. So a scope entered and left between the phases is exact, not
#     approximate.
#
# ``caller_scope`` exists for that second case. It is a process-wide swap rather
# than a thread-local/ContextVar on purpose: worker threads start with a fresh
# context and would NOT inherit either, so a thread-local would mislabel exactly
# the fan-out it was meant to describe.
CALLER_ENV = LLM_ENV["caller"]
#: What an unlabelled call is called. Never a guess at which stage it was:
#: rows written before this existed, ad-hoc CLI runs and any stage nobody
#: labelled all roll up here, honestly.
UNKNOWN_CALLER = "unknown"
MAX_CALLER_LENGTH = 40
_CALLER_STRIP = re.compile(r"[^a-z0-9._-]+")
_caller_override: str | None = None
_caller_lock = threading.Lock()


def normalize_caller(value) -> str | None:
    """A label safe to use as a rollup key, or ``None`` when there isn't one.

    The value comes from the environment, so it is untrusted input to a counter
    keyed by it: bound the alphabet and the length so a typo (or a stray path,
    or a whole shell command) cannot become an unbounded set of dictionary keys
    in every day of the rollup.
    """
    if not isinstance(value, str):
        return None
    label = _CALLER_STRIP.sub("-", value.strip().lower()).strip("-")
    return label[:MAX_CALLER_LENGTH] or None


def caller() -> str:
    """Which consumer the calls being made right now belong to."""
    with _caller_lock:
        override = _caller_override
    return override or normalize_caller(llm_setting("caller")) or UNKNOWN_CALLER


def set_caller(label) -> str | None:
    """Pin the process-wide label; returns the previous override for restore."""
    global _caller_override
    with _caller_lock:
        previous = _caller_override
        _caller_override = normalize_caller(label)
    return previous


@contextmanager
def caller_scope(label):
    """Label every LLM call made inside the block (workers included).

    The safe form for a process that spends as two consumers: it restores the
    previous label on the way out, so a raise inside the block cannot leave the
    rest of the run mislabelled. Use it whenever the second consumer's work fits
    in one block.

    A digest whose two phases are the two halves of a loop body rather than a
    block uses the underlying ``set_caller`` instead — it re-asserts the
    decision label at the top of every wave, which self-corrects for the same
    reason this ``finally`` does.
    """
    previous = set_caller(label)
    try:
        yield
    finally:
        set_caller(previous)


# Per-call attribution. ``caller`` names the consumer for the whole
# process; ``cluster_key`` and ``phase`` name the one cluster a call was made
# for and which step of its decision spent it (``decision``, ``rejudge``,
# ``apply``, ``verification``). Unlike the caller, these differ between the
# concurrent calls of one ``decide_many`` batch, so they are THREAD-local, set
# around each ``decide`` in the thread that makes it -- a client wrapper can
# do that for a whole digest.
_usage_ctx = threading.local()
#: Cluster keys are ids (thread, channel, repo, entity slug, date); bound them
#: anyway, since they become values in a long-lived log.
MAX_USAGE_KEY_LENGTH = 200


@contextmanager
def usage_context(cluster_key=None, phase=None):
    """Attribute every usage row this thread writes inside the block."""
    previous = (getattr(_usage_ctx, "cluster_key", None),
                getattr(_usage_ctx, "phase", None))
    _usage_ctx.cluster_key = cluster_key
    _usage_ctx.phase = phase
    try:
        yield
    finally:
        _usage_ctx.cluster_key, _usage_ctx.phase = previous


def _log_usage(model: str, usage, ok: bool = True,
               duration_ms: int | None = None) -> None:
    """Best-effort append of one call's token usage + latency. Never raises —
    accounting must not break distillation.

    ``caller`` is a pure ADDITION to the row: consumers key off ``ts`` and skip
    unknown fields, so an existing long-lived usage log keeps folding
    incrementally across the change instead of being re-read or counted
    malformed. ``cluster_key`` and ``phase`` are additions of the same kind,
    written only when a ``usage_context`` is active, so rows from every other
    consumer keep their shape and size.
    """
    try:
        path = llm_setting("usage_log", USAGE_LOG_DEFAULT)
        input_tokens, output_tokens = _usage_token_counts(usage)
        rec = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": model, "ok": ok,
            "caller": caller(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        cluster_key = getattr(_usage_ctx, "cluster_key", None)
        if cluster_key is not None:
            rec["cluster_key"] = str(cluster_key)[:MAX_USAGE_KEY_LENGTH]
        phase = getattr(_usage_ctx, "phase", None)
        if phase is not None:
            rec["phase"] = str(phase)
        if duration_ms is not None:
            rec["duration_ms"] = int(duration_ms)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


class LLMClient:
    def __init__(self, model=None, api_key=None, base_url=None):
        self.api_key = api_key or llm_setting("api_key")
        # A key from REMEMBRANCER_LLM_API_KEY gets the SDK's own endpoint (None) and no
        # default model; any other key gets LEGACY_KEY_DEFAULTS, if there are any.
        neutral_key = not api_key and bool(os.environ.get(LLM_ENV["api_key"]))
        defaults = {} if neutral_key else LEGACY_KEY_DEFAULTS
        self.base_url = base_url or llm_setting("base_url", defaults.get("base_url"))
        self.model = model or llm_setting("model", defaults.get("model"))
        try:
            self.temperature = float(llm_setting("temperature", "0"))
        except ValueError:
            self.temperature = 0.0
        self.temperature = min(1.0, max(0.0, self.temperature))
        try:
            self.timeout = float(llm_setting("timeout_seconds", "120"))
        except ValueError:
            self.timeout = 120.0
        # Prevent a typo from turning an unattended pass into an effectively
        # unbounded socket wait (or an immediate retry storm).
        self.timeout = min(300.0, max(5.0, self.timeout))
        self.deadline_monotonic: float | None = None

    def set_deadline(self, deadline_monotonic: float | None) -> None:
        """Bound every request/retry by one caller-owned monotonic deadline.

        A bounded unattended batch must not let one slow model request (or its
        retry backoff) consume the outer process timeout and discard otherwise
        checkpointable work. ``None`` restores the ordinary per-request limit.
        """
        self.deadline_monotonic = (
            None if deadline_monotonic is None else float(deadline_monotonic)
        )

    def _remaining_seconds(self) -> float | None:
        if self.deadline_monotonic is None:
            return None
        return self.deadline_monotonic - time.monotonic()

    def available(self) -> bool:
        if not self.api_key:
            return False
        if not self.model:
            _warn_once(NO_MODEL)
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except Exception:
            return False


    # ------------------------------------------------------------------ #
    # one logical decision                                               #
    # ------------------------------------------------------------------ #
    def _messages_for(self, prompt: str, tool: dict, prior_text: str | None):
        """The user turn, plus — on a repair retry — the model's prose as an
        assistant turn and a short follow-up so it converts rather than restarts."""
        msgs = [{"role": "user", "content": prompt}]
        if prior_text:
            msgs.append({"role": "assistant", "content": prior_text[:6000]})
            msgs.append({"role": "user",
                         "content": REPAIR_FOLLOWUP.format(tool=tool.get("name", "tool"))})
        return msgs

    def decide(self, prompt: str, tool: dict, max_tokens: int = 1500,
               strict: bool = False) -> dict:
        """One decision: a dict matching ``tool["input_schema"]``.

        ``strict=True`` sends the format reminder from the first request (used by
        ``decide_many``'s split retry). Raises ``LLMDeclined`` when the model
        answered in prose that could not be repaired, ``LLMUnavailable`` on
        transport exhaustion or a missing key/SDK.
        """
        if not self.api_key:
            raise LLMUnavailable(f"no LLM key: set {llm_requirement()}")
        if not self.model:
            raise LLMUnavailable(NO_MODEL)
        try:
            import anthropic
        except Exception as e:
            raise LLMUnavailable(f"anthropic SDK unavailable: {e}")
        tool_name = str(tool.get("name") or "tool")
        batch_size = getattr(_batch_ctx, "size", None)
        prompt_hash = llm_repair.prompt_hash(prompt)
        last_category = "unknown"
        prior_text: str | None = None
        repair_round = 0           # how many reminder re-sends have been made
        attempt = 0                # transport attempts (rate limit / timeout / 5xx)
        calls_made = 0
        while attempt < MAX_ATTEMPTS:
            remaining = self._remaining_seconds()
            if remaining is not None and remaining <= 0:
                self._exhausted("deadline", prompt_hash, batch_size, calls_made,
                                tool_name, max_tokens)
                raise LLMUnavailable("run deadline exceeded")
            request_timeout = (
                self.timeout
                if remaining is None else min(self.timeout, max(0.1, remaining))
            )
            client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=request_timeout,
            )
            kwargs = dict(
                model=self.model, max_tokens=max_tokens,
                temperature=self.temperature,
                tools=[tool], tool_choice={"type": "tool", "name": tool_name},
                messages=self._messages_for(prompt, tool, prior_text),
            )
            if strict or prior_text is not None:
                kwargs["system"] = REPAIR_SYSTEM.format(tool=tool_name)
            try:
                t0 = time.monotonic()
                msg = client.messages.create(**kwargs)
                calls_made += 1
                duration_ms = int((time.monotonic() - t0) * 1000)
                usage = getattr(msg, "usage", None)
                for block in msg.content:
                    if getattr(block, "type", None) == "tool_use":
                        _log_usage(self.model, usage, ok=True, duration_ms=duration_ms)
                        llm_repair.bump_health(calls=1, repaired=1 if prior_text else 0)
                        CONCURRENCY.note_success()
                        return dict(block.input)
                text = "\n".join(
                    str(getattr(block, "text", "") or "")
                    for block in msg.content
                    if getattr(block, "type", None) == "text"
                ).strip()
                if not text:
                    # A genuinely empty response is transient: retry as transport.
                    raise RuntimeError("no tool_use block in response")
                # The model said something instead of calling the tool. Layer 1:
                # the prose is usually the payload in a ```json fence — recover it.
                _log_usage(self.model, usage, ok=True, duration_ms=duration_ms)
                stop_reason = getattr(msg, "stop_reason", None)
                kind = llm_repair.classify_prose(text, stop_reason)
                llm_repair.bump_health(calls=1, prose_failures=1)
                payload = llm_repair.repair_tool_payload(text, tool)
                if payload is not None:
                    llm_repair.bump_health(repaired=1)
                    # The request itself completed: prose is a schema problem,
                    # not provider pushback, so it must not shrink the fan-out.
                    CONCURRENCY.note_success()
                    return payload
                # Layer 2: one re-send with the reminder (and more room if cut off).
                if repair_round < max(0, REPAIR_RETRIES):
                    repair_round += 1
                    prior_text = text
                    if kind == "truncated":
                        max_tokens = min(MAX_TOKENS_CEILING, max(max_tokens * 2, max_tokens + 500))
                    continue
                self._exhausted(f"prose:{kind}", prompt_hash, batch_size, calls_made,
                                tool_name, max_tokens)
                raise LLMDeclined("model returned prose instead of a tool call")
            except (LLMUnavailable, LLMDeclined):
                raise
            except Exception as e:  # transient API / rate-limit error
                last_category = _error_category(e)
                # Two counters, because they answer different questions: every
                # failed attempt (retried or not), and the subset that was the
                # PROVIDER pushing back. An empty response and a 429 are both
                # transport failures; only one of them means "slow down".
                llm_repair.bump_health(
                    transport_failures=1,
                    pushback_failures=1 if last_category in BACKOFF_CATEGORIES else 0,
                )
                # Feed the control loop BEFORE the backoff sleep: a 429 storm
                # must shrink the next wave while this one is still retrying.
                CONCURRENCY.note_failure(last_category)
                attempt += 1
                if attempt >= MAX_ATTEMPTS:
                    break
                _, delay = _retry_plan(e, attempt - 1)
                remaining = self._remaining_seconds()
                if remaining is not None:
                    if remaining <= 0:
                        self._exhausted("deadline", prompt_hash, batch_size, calls_made,
                                        tool_name, max_tokens)
                        raise LLMUnavailable("run deadline exceeded")
                    delay = min(delay, remaining)
                time.sleep(delay)
        _log_usage(self.model, None, ok=False)
        self._exhausted(f"api:{last_category}", prompt_hash, batch_size, calls_made,
                        tool_name, max_tokens)
        # Upstream exception text can contain request ids, URLs, or echoed
        # payload fragments, so it is never surfaced. But a bare "it failed" is
        # undiagnosable: the same message covered rate limits, timeouts and
        # schema refusals alike, and nobody could tell which was happening.
        # The category is a closed vocabulary, so it carries no payload.
        raise LLMUnavailable(
            f"API failed after {MAX_ATTEMPTS} retries [{last_category}]"
        )

    @staticmethod
    def _exhausted(reason: str, prompt_hash: str, batch_size, attempt: int,
                   tool_name: str, max_tokens: int) -> None:
        """Layer 3: nothing fails silently — ledger row + health counter."""
        llm_repair.record_failure(reason, prompt_hash_value=prompt_hash,
                                  batch_size=batch_size, attempt=attempt,
                                  tool=tool_name, max_tokens=max_tokens)
        llm_repair.bump_health(exhausted=1)


def _is_prose_failure(exc: BaseException) -> bool:
    return isinstance(exc, LLMDeclined) or "prose instead of a tool call" in str(exc)


def decide_many(client, jobs, max_workers=None):
    """Bounded fan-out over ``LLMClient.decide`` — THE sanctioned parallel-LLM path.

    jobs: list of ``(prompt, tool)`` or ``(prompt, tool, max_tokens)`` tuples.
    Returns a list of dict|None, same order as ``jobs``. ``decide`` already retries a
    single call with rate-limit backoff (see ``_retry_plan``) and repairs/re-sends a
    prose answer; this adds bounded concurrency plus ONE more layer on top:

    Split retry. After the batch completes, the jobs that ended in a prose failure
    (``LLMDeclined``) are re-run by themselves as a smaller batch — half the
    concurrency, down to 1 — with the format reminder from the first request
    (``strict=True``), for ``REMEMBRANCER_LLM_SPLIT_RETRIES`` rounds (default 1). Each
    prompt is one cluster, so a bad answer never sinks its neighbours; the split
    keeps the retry cheap (only the failures) and isolated (serial at the end).

    A job whose retries are exhausted raises (normally ``LLMUnavailable``, but any
    exception is treated the same) — that job's slot is logged and set to None; it
    never aborts the batch. ``client.available()`` False short-circuits to
    ``[None] * len(jobs)`` without spawning any threads.

    max_workers: an explicit arg pins the level for this batch; ``None`` (the
    production path) hands the choice to the process-wide ``AdaptiveConcurrency``
    controller, which starts at ``REMEMBRANCER_LLM_CONCURRENCY`` (default 8, divided by
    ``REMEMBRANCER_LLM_CONCURRENCY_SHARES``), climbs one worker at a time while the
    observed failure rate stays under ``REMEMBRANCER_LLM_FAILURE_THRESHOLD``, and halves
    on a 429/5xx/timeout burst. A provider may cap concurrency per account; the
    controller finds the cap instead of assuming it, and the per-call backoff
    still self-throttles whatever gets through.
    """
    if not jobs:
        return []
    if not client.available():
        return [None] * len(jobs)

    adaptive = max_workers is None
    if adaptive:
        max_workers = CONCURRENCY.level()
    max_workers = max(1, max_workers)

    def _run_batch(indexed_jobs, workers, strict):
        """Run [(index, job)] → {index: (result|None, exception|None)}."""
        batch_size = len(indexed_jobs)

        def _run(item):
            idx, job = item
            prompt, tool = job[0], job[1]
            max_tokens = job[2] if len(job) > 2 else 1500
            _batch_ctx.size = batch_size
            try:
                if strict:
                    try:
                        return idx, client.decide(prompt, tool, max_tokens, strict=True), None
                    except TypeError:
                        # A fake/legacy client without the keyword: plain re-run.
                        return idx, client.decide(prompt, tool, max_tokens), None
                return idx, client.decide(prompt, tool, max_tokens), None
            except Exception as e:  # noqa: BLE001
                return idx, None, e
            finally:
                _batch_ctx.size = None

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return {idx: (res, err) for idx, res, err in pool.map(_run, indexed_jobs)}

    results: list = [None] * len(jobs)
    outcome = _run_batch(list(enumerate(jobs)), max_workers, strict=False)
    failed = []
    for idx, (res, err) in outcome.items():
        if err is None:
            results[idx] = res
        elif _is_prose_failure(err):
            failed.append(idx)
        else:
            print(f"[llm] decide_many: job exhausted retries, returning None: {err}", file=sys.stderr)

    workers = max_workers
    rounds = 0
    while failed and rounds < max(0, SPLIT_RETRIES):
        rounds += 1
        workers = max(1, workers // 2)
        llm_repair.bump_health(split_retries=1)
        print(f"[llm] decide_many: split retry {rounds}: re-running {len(failed)} "
              f"prose-failed job(s) at concurrency {workers}", file=sys.stderr)
        outcome = _run_batch([(idx, jobs[idx]) for idx in failed], workers, strict=True)
        still = []
        for idx, (res, err) in outcome.items():
            if err is None:
                results[idx] = res
            elif _is_prose_failure(err):
                still.append(idx)
            else:
                print(f"[llm] decide_many: job exhausted retries, returning None: {err}", file=sys.stderr)
        failed = still
    for idx in failed:
        print("[llm] decide_many: job exhausted retries, returning None: "
              "model returned prose instead of a tool call", file=sys.stderr)
    if adaptive:
        level = CONCURRENCY.settle()
        if level != max_workers:
            snapshot = CONCURRENCY.snapshot()
            print(f"[llm] concurrency {max_workers} -> {level} "
                  f"({snapshot['reason']}, failure_rate="
                  f"{snapshot['failure_rate']:.3f} over {snapshot['samples']} "
                  f"calls, ceiling {snapshot['ceiling']})", file=sys.stderr)
    return results
