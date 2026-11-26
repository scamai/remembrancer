"""The environment variables the LLM client reads, one per setting.

Every setting has one name, ``REMEMBRANCER_LLM_<SETTING>`` (``LLM_ENV``).
:func:`llm_setting` reads it, and falls back to an older name when a
deployment still sets one (``LEGACY_LLM_ENV``). The files the client writes
default to ``state/llm_usage.jsonl``, ``state/llm_failures.jsonl`` and
``state/llm_health.json``.
"""
from __future__ import annotations

import os

#: Each LLM setting's environment variable.
LLM_ENV = {
    # The endpoint.
    "api_key": "REMEMBRANCER_LLM_API_KEY",
    "base_url": "REMEMBRANCER_LLM_BASE_URL",
    "model": "REMEMBRANCER_LLM_MODEL",
    "temperature": "REMEMBRANCER_LLM_TEMPERATURE",
    "timeout_seconds": "REMEMBRANCER_LLM_TIMEOUT_SECONDS",
    # Fan-out (``concurrency_config`` in distill/llm.py): the start level, its
    # floor and ceiling, the processes that share one account, and the failure
    # rate above which the level backs off.
    "concurrency": "REMEMBRANCER_LLM_CONCURRENCY",
    "concurrency_default": "REMEMBRANCER_LLM_CONCURRENCY_DEFAULT",
    "concurrency_min": "REMEMBRANCER_LLM_CONCURRENCY_MIN",
    "concurrency_max": "REMEMBRANCER_LLM_CONCURRENCY_MAX",
    "concurrency_shares": "REMEMBRANCER_LLM_CONCURRENCY_SHARES",
    "adaptive_concurrency": "REMEMBRANCER_LLM_ADAPTIVE_CONCURRENCY",
    "failure_threshold": "REMEMBRANCER_LLM_FAILURE_THRESHOLD",
    # Retries of one call.
    "max_attempts": "REMEMBRANCER_LLM_MAX_ATTEMPTS",
    "repair_retries": "REMEMBRANCER_LLM_REPAIR_RETRIES",
    "split_retries": "REMEMBRANCER_LLM_SPLIT_RETRIES",
    "max_tokens_ceiling": "REMEMBRANCER_LLM_MAX_TOKENS_CEILING",
    # Accounting: who spent each call, and where usage, failures and health go
    # ("off" turns the failures log and the health file off).
    "caller": "REMEMBRANCER_LLM_CALLER",
    "usage_log": "REMEMBRANCER_LLM_USAGE_LOG",
    "failures_log": "REMEMBRANCER_LLM_FAILURES_LOG",
    "health": "REMEMBRANCER_LLM_HEALTH",
}

#: Where usage, failures and health go when their variable is unset.
USAGE_LOG_DEFAULT = os.path.join("state", "llm_usage.jsonl")
FAILURES_LOG_DEFAULT = os.path.join("state", "llm_failures.jsonl")
HEALTH_PATH_DEFAULT = os.path.join("state", "llm_health.json")

#: An older name for a setting, read only when its name in ``LLM_ENV`` is unset
#: or empty.
LEGACY_LLM_ENV: dict[str, str] = {}

#: Defaults for a key that did not come from ``REMEMBRANCER_LLM_API_KEY`` (one from an
#: older name, or passed in). A key from ``REMEMBRANCER_LLM_API_KEY`` gets the SDK's
#: own endpoint and no default model.
LEGACY_KEY_DEFAULTS: dict[str, str] = {}



def llm_setting(name: str, default: str | None = None) -> str | None:
    """One LLM setting: its ``LLM_ENV`` variable when that is set and not empty,
    else its older name when it has one and that is set, else ``default``."""
    value = os.environ.get(LLM_ENV[name])
    if value:
        return value
    legacy = LEGACY_LLM_ENV.get(name)
    return os.environ.get(legacy, default) if legacy else default


def llm_int(name: str, default: int) -> int:
    """An integer setting; ``default`` when it is unset or not an integer."""
    try:
        return int(llm_setting(name, str(default)))
    except (TypeError, ValueError):
        return default


def llm_float(name: str, default: float) -> float:
    """A number setting; ``default`` when it is unset or not a number."""
    try:
        return float(llm_setting(name, str(default)))
    except (TypeError, ValueError):
        return default
