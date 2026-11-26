"""LLMClient must back off LONG on a provider's rate/concurrency limits (so parallel
callers self-throttle) and SHORT on ordinary transient errors."""
from distill.llm import LLMDeclined, LLMUnavailable, _error_category, _retry_plan


class _Err(Exception):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status_code = status


def test_rate_limit_by_status_backs_off_exponentially():
    is_rate, d0 = _retry_plan(_Err("boom", status=429), 0)
    is_rate2, d3 = _retry_plan(_Err("boom", status=429), 3)
    assert is_rate and is_rate2
    assert d3 > d0           # grows with attempt
    assert d3 <= 60.0        # capped


def test_concurrency_message_is_rate_limited():
    for m in ("concurrency limit exceeded", "Too Many Requests", "overloaded", "rate_limit"):
        is_rate, d = _retry_plan(_Err(m), 1)
        assert is_rate, m
        assert d >= 1.0


def test_ordinary_error_short_linear_backoff():
    is_rate, d = _retry_plan(_Err("connection reset"), 2)
    assert not is_rate
    assert d == 0.5 * 3


def test_retry_after_header_is_honored():
    class _R(Exception):
        status_code = 429
        class response:
            headers = {"retry-after": "12"}
    is_rate, d = _retry_plan(_R(), 0)
    assert is_rate and d == 12.0


# --------------------------------------------------------------------------- #
# a refusal is a verdict, an outage is an outage
# --------------------------------------------------------------------------- #
def test_declined_is_catchable_as_unavailable():
    """Every existing `except LLMUnavailable` must keep working unchanged."""
    assert issubclass(LLMDeclined, LLMUnavailable)


def test_error_category_is_a_closed_vocabulary_that_leaks_nothing():
    """The category must be diagnosable without carrying upstream text.

    The old message was `API failed after 7 retries` for every cause alike --
    rate limits, timeouts and schema refusals were indistinguishable, so nobody
    could tell an outage from a deterministic refusal. The redaction was right;
    having nothing at all in its place was not.
    """
    secret = "https://api.example/v1/x?key=SEKRET note body fragment"
    cases = {
        "timeout": _Err(f"Request timeout {secret}"),
        "rate_limit": _Err(f"429 Too Many Requests {secret}", status=429),
        "connection": _Err(f"Connection reset by peer {secret}"),
        "no_tool_use": _Err(f"no tool_use block in response {secret}"),
        "http_400": _Err(f"400 Bad Request {secret}"),
        "http_503": _Err(f"503 Service Unavailable {secret}"),
        "other": _Err("something entirely new"),
    }
    for expected, exc in cases.items():
        got = _error_category(exc)
        assert got == expected, f"{exc} -> {got}, wanted {expected}"
        assert "SEKRET" not in got and "example" not in got, (
            "the category must never carry upstream text"
        )


def test_every_category_is_a_short_bare_token():
    """Categories become label values and status-file contents; keep them inert."""
    import re
    for exc in (_Err("boom"), _Err("429 rate limit", status=429), _Err("timeout")):
        category = _error_category(exc)
        assert re.fullmatch(r"[a-z0-9_]{1,20}", category), category

