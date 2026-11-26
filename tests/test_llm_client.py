# tests/test_llm_client.py
import json
import os
import sys
import types
import pytest
from distill.llm import LLMClient, LLMUnavailable
from distill.llm_env import LEGACY_LLM_ENV, LLM_ENV

MODEL = "test-model"


@pytest.fixture(autouse=True)
def _no_llm_env(monkeypatch):
    """No LLM variable from the caller's shell reaches these tests, and none of
    them writes the failures log or the health file."""
    for name in (*LLM_ENV.values(), *LEGACY_LLM_ENV.values()):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("REMEMBRANCER_LLM_FAILURES_LOG", "off")
    monkeypatch.setenv("REMEMBRANCER_LLM_HEALTH", "off")


def _configure(monkeypatch, key="k"):
    """The two settings that turn the LLM on: a key and a model."""
    monkeypatch.setenv("REMEMBRANCER_LLM_API_KEY", key)
    monkeypatch.setenv("REMEMBRANCER_LLM_MODEL", MODEL)


def test_available_false_without_key(monkeypatch):
    monkeypatch.delenv("REMEMBRANCER_LLM_API_KEY", raising=False)
    assert LLMClient().available() is False


def test_available_false_when_sdk_is_missing(monkeypatch):
    _configure(monkeypatch, "configured")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert LLMClient().available() is False

def test_decide_returns_tool_input(monkeypatch, tmp_path):
    _configure(monkeypatch)
    monkeypatch.setenv("REMEMBRANCER_LLM_TEMPERATURE", "0")
    monkeypatch.setenv("REMEMBRANCER_LLM_USAGE_LOG", str(tmp_path / "llm_usage.jsonl"))
    # fake anthropic module: messages.create returns a tool_use block
    fake = types.SimpleNamespace()
    calls = []
    class _Msg:
        content = [types.SimpleNamespace(type="tool_use", name="consolidate", input={"action": "IGNORE"})]
    def create(**kwargs):
        calls.append(kwargs)
        return _Msg()
    client_kwargs = []
    def anthropic_client(**kwargs):
        client_kwargs.append(kwargs)
        return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    fake.Anthropic = anthropic_client
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    out = LLMClient().decide("hi", {"name": "consolidate", "input_schema": {}})
    assert out == {"action": "IGNORE"}
    assert calls[0]["temperature"] == 0.0
    assert client_kwargs[0]["timeout"] == 120.0


def test_temperature_is_clamped_and_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_LLM_TEMPERATURE", "3")
    assert LLMClient().temperature == 1.0
    monkeypatch.setenv("REMEMBRANCER_LLM_TEMPERATURE", "invalid")
    assert LLMClient().temperature == 0.0


def test_timeout_is_bounded_and_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_LLM_TIMEOUT_SECONDS", "9999")
    assert LLMClient().timeout == 300.0
    monkeypatch.setenv("REMEMBRANCER_LLM_TIMEOUT_SECONDS", "0")
    assert LLMClient().timeout == 5.0
    monkeypatch.setenv("REMEMBRANCER_LLM_TIMEOUT_SECONDS", "invalid")
    assert LLMClient().timeout == 120.0


def test_run_deadline_stops_before_starting_api_request(monkeypatch):
    _configure(monkeypatch, "configured")
    created = []
    fake = types.SimpleNamespace(
        Anthropic=lambda **kwargs: created.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setattr("distill.llm.time.monotonic", lambda: 10.0)
    client = LLMClient()
    client.set_deadline(10.0)

    with pytest.raises(LLMUnavailable, match="run deadline exceeded"):
        client.decide("hi", {"name": "consolidate", "input_schema": {}})

    assert created == []


def test_run_deadline_caps_request_timeout(monkeypatch, tmp_path):
    _configure(monkeypatch, "configured")
    monkeypatch.setenv("REMEMBRANCER_LLM_USAGE_LOG", str(tmp_path / "llm_usage.jsonl"))
    client_kwargs = []

    class _Msg:
        content = [types.SimpleNamespace(
            type="tool_use", name="consolidate", input={"action": "IGNORE"},
        )]

    fake = types.SimpleNamespace(Anthropic=lambda **kwargs: (
        client_kwargs.append(kwargs)
        or types.SimpleNamespace(
            messages=types.SimpleNamespace(create=lambda **_kwargs: _Msg()),
        )
    ))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setattr("distill.llm.time.monotonic", lambda: 10.0)
    client = LLMClient()
    client.set_deadline(17.0)

    assert client.decide(
        "hi", {"name": "consolidate", "input_schema": {}},
    ) == {"action": "IGNORE"}
    assert client_kwargs[0]["timeout"] == 7.0

def test_decide_raises_without_key(monkeypatch):
    monkeypatch.delenv("REMEMBRANCER_LLM_API_KEY", raising=False)
    try:
        LLMClient().decide("hi", {"name": "consolidate"})
        assert False, "expected LLMUnavailable"
    except LLMUnavailable:
        pass


def test_exhausted_error_is_counted_without_upstream_detail(monkeypatch, tmp_path):
    from distill import llm

    _configure(monkeypatch, "configured")
    usage_log = tmp_path / "llm_usage.jsonl"
    monkeypatch.setenv("REMEMBRANCER_LLM_USAGE_LOG", str(usage_log))
    monkeypatch.setattr(llm, "MAX_ATTEMPTS", 1)
    secret = "sensitive-request-detail"

    class BrokenMessages:
        @staticmethod
        def create(**_kwargs):
            raise RuntimeError(secret)

    fake = types.SimpleNamespace(Anthropic=lambda **_kwargs: types.SimpleNamespace(
        messages=BrokenMessages()
    ))
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    try:
        LLMClient().decide("hi", {"name": "consolidate", "input_schema": {}})
        assert False, "expected LLMUnavailable"
    except LLMUnavailable as exc:
        assert secret not in str(exc)
        # The message now carries a failure CATEGORY as well. That is the point
        # of the change: `API failed after N retries` alone covered rate limits,
        # timeouts and schema refusals identically, so nobody could tell an
        # outage from a deterministic refusal. The category is a closed
        # vocabulary, so the redaction this test guards is unaffected.
        assert str(exc).startswith("API failed after 1 retries [")
        assert str(exc).endswith("]")
        import re
        category = str(exc).rsplit("[", 1)[1].rstrip("]")
        assert re.fullmatch(r"[a-z0-9_]{1,20}", category), category
    usage = json.loads(usage_log.read_text())
    assert usage["model"] == MODEL
    assert usage["ok"] is False
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None


def test_log_usage_reads_dict_object_and_alias_shapes(tmp_path):
    from distill import llm

    log = tmp_path / "llm_usage.jsonl"
    monkeypatch_target = "REMEMBRANCER_LLM_USAGE_LOG"
    os.environ[monkeypatch_target] = str(log)
    try:
        llm._log_usage(MODEL, {"input_tokens": 11, "output_tokens": 7})
        llm._log_usage(MODEL, types.SimpleNamespace(input_tokens=5, output_tokens=3))
        llm._log_usage(MODEL, {"prompt_tokens": 100, "completion_tokens": 40})
        llm._log_usage(MODEL, None, ok=False, duration_ms=1234)
    finally:
        del os.environ[monkeypatch_target]
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert (rows[0]["input_tokens"], rows[0]["output_tokens"]) == (11, 7)
    assert (rows[1]["input_tokens"], rows[1]["output_tokens"]) == (5, 3)
    assert (rows[2]["input_tokens"], rows[2]["output_tokens"]) == (100, 40)
    assert rows[3]["ok"] is False and rows[3]["duration_ms"] == 1234
    assert rows[3]["input_tokens"] is None


def test_successful_decide_logs_duration_and_tokens(monkeypatch, tmp_path):

    _configure(monkeypatch, "configured")
    usage_log = tmp_path / "llm_usage.jsonl"
    monkeypatch.setenv("REMEMBRANCER_LLM_USAGE_LOG", str(usage_log))

    class FakeMessages:
        @staticmethod
        def create(**_kwargs):
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="tool_use",
                                               input={"action": "IGNORE"})],
                usage={"input_tokens": 42, "output_tokens": 9},
            )

    fake = types.SimpleNamespace(Anthropic=lambda **_kwargs: types.SimpleNamespace(
        messages=FakeMessages()
    ))
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    out = LLMClient().decide("hi", {"name": "consolidate", "input_schema": {}})
    assert out == {"action": "IGNORE"}
    usage = json.loads(usage_log.read_text())
    assert usage["ok"] is True
    assert usage["input_tokens"] == 42 and usage["output_tokens"] == 9
    assert isinstance(usage["duration_ms"], int) and usage["duration_ms"] >= 0


def _fake_anthropic(monkeypatch, blocks, calls):
    fake = types.SimpleNamespace()

    class _Msg:
        content = blocks
        usage = None

    def create(**kwargs):
        calls.append(kwargs)
        return _Msg()

    fake.Anthropic = lambda **kw: types.SimpleNamespace(
        messages=types.SimpleNamespace(create=create)
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake)


def test_prose_instead_of_a_tool_call_gets_one_reminder_not_seven_retries(
        monkeypatch, tmp_path):
    """A refusal is deterministic; retrying it seven times just repeats it.

    Under tool_choice, a model sometimes answers in prose -- "the provided text
    contains no factual claims" -- rather than calling the tool. The client used
    to treat that as a transient error, burn all seven attempts, and raise
    "API failed after 7 retries", which downstream became status="skipped".
    That is how a whole backlog reached checked=0 with none verified.

    The reliability layer now allows exactly ONE
    repair re-send carrying the "respond only with the tool call" reminder and
    the prior text; a second prose answer is still a verdict, never an outage.
    """
    from distill.llm import LLMDeclined, REPAIR_SYSTEM

    _configure(monkeypatch)
    monkeypatch.setenv("REMEMBRANCER_LLM_USAGE_LOG", str(tmp_path / "llm_usage.jsonl"))
    calls = []
    _fake_anthropic(
        monkeypatch,
        [types.SimpleNamespace(type="text", text="No factual claims here.")],
        calls,
    )

    with pytest.raises(LLMDeclined):
        LLMClient().decide("hi", {"name": "extract", "input_schema": {}})
    assert len(calls) == 2, f"one reminder re-send, not seven; made {len(calls)} calls"
    assert "system" not in calls[0]
    assert calls[1]["system"] == REPAIR_SYSTEM.format(tool="extract")
    roles = [m["role"] for m in calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert calls[1]["messages"][1]["content"] == "No factual claims here."
    assert calls[1]["tool_choice"] == {"type": "tool", "name": "extract"}


def test_a_genuinely_empty_response_is_still_retried(monkeypatch, tmp_path):
    """The guard must not swallow real transport faults as verdicts."""
    from distill.llm import LLMUnavailable

    _configure(monkeypatch)
    monkeypatch.setenv("REMEMBRANCER_LLM_USAGE_LOG", str(tmp_path / "llm_usage.jsonl"))
    monkeypatch.setattr("distill.llm.time.sleep", lambda _s: None)
    calls = []
    _fake_anthropic(monkeypatch, [], calls)

    with pytest.raises(LLMUnavailable) as excinfo:
        LLMClient().decide("hi", {"name": "extract", "input_schema": {}})
    assert len(calls) > 1, "an empty response is transient and must be retried"
    assert "no_tool_use" in str(excinfo.value), (
        "the failure category must survive into the message"
    )

