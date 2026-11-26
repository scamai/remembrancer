"""LLM settings: each is one REMEMBRANCER_LLM_* variable (distill/llm_env.py). A key
gets the Anthropic SDK's own endpoint unless REMEMBRANCER_LLM_BASE_URL names another,
and has no default model: REMEMBRANCER_LLM_MODEL must be set. An empty variable counts
as unset."""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from distill import llm, llm_repair
from distill.llm import LLMClient, LLMUnavailable, concurrency_config
from distill.llm_env import (FAILURES_LOG_DEFAULT, HEALTH_PATH_DEFAULT,
                             LEGACY_LLM_ENV, LLM_ENV, USAGE_LOG_DEFAULT,
                             llm_float, llm_int)

TOOL = {"name": "t", "input_schema": {}}
ROOT = Path(__file__).resolve().parents[1]
ALL_NAMES = (*LLM_ENV.values(), *LEGACY_LLM_ENV.values())


@pytest.fixture(autouse=True)
def _no_llm_env(monkeypatch):
    """No LLM variable from the caller's shell reaches these tests, and none of
    them writes the failures log or the health file unless it asks to."""
    for name in ALL_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("REMEMBRANCER_LLM_FAILURES_LOG", "off")
    monkeypatch.setenv("REMEMBRANCER_LLM_HEALTH", "off")
    monkeypatch.setattr(llm, "_WARNED", set())


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    """A stand-in Anthropic SDK that records what the client hands it."""
    monkeypatch.setenv("REMEMBRANCER_LLM_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    seen = {"client": [], "create": []}

    class _Msg:
        content = [types.SimpleNamespace(type="tool_use", name="t", input={"ok": True})]
        usage = None

    def create(**kwargs):
        seen["create"].append(kwargs)
        return _Msg()

    def anthropic_client(**kwargs):
        seen["client"].append(kwargs)
        return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=anthropic_client))
    return seen


def _request(seen):
    """(base_url, model, api_key) of the request one decision sends."""
    assert LLMClient().decide("hi", TOOL) == {"ok": True}
    client = seen["client"][-1]
    return client["base_url"], seen["create"][-1]["model"], client["api_key"]


# ─── The endpoint ────────────────────────────────────────────────────────────


def test_a_key_and_a_model_use_the_sdk_endpoint(sdk, monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_LLM_API_KEY", "k2")
    monkeypatch.setenv("REMEMBRANCER_LLM_MODEL", "claude-x")
    assert _request(sdk) == (None, "claude-x", "k2")


def test_each_endpoint_setting_is_read_from_its_variable(sdk, monkeypatch):
    for suffix, value in (
        ("API_KEY", "k2"),
        ("BASE_URL", "https://new.example"),
        ("MODEL", "new"),
        ("TEMPERATURE", "0.7"),
        ("TIMEOUT_SECONDS", "60"),
    ):
        monkeypatch.setenv(f"REMEMBRANCER_LLM_{suffix}", value)
    assert _request(sdk) == ("https://new.example", "new", "k2")
    assert sdk["client"][-1]["timeout"] == 60.0
    assert sdk["create"][-1]["temperature"] == 0.7


def test_a_key_passed_in_reads_the_model_and_endpoint_variables(sdk, monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_LLM_BASE_URL", "https://llm.example/anthropic")
    monkeypatch.setenv("REMEMBRANCER_LLM_MODEL", "m")
    assert LLMClient(api_key="k3").decide("hi", TOOL) == {"ok": True}
    assert sdk["client"][-1]["base_url"] == "https://llm.example/anthropic"
    assert (sdk["create"][-1]["model"], sdk["client"][-1]["api_key"]) == ("m", "k3")


def test_a_key_without_a_model_is_off_and_says_so_once(sdk, monkeypatch, capsys):
    monkeypatch.setenv("REMEMBRANCER_LLM_API_KEY", "k2")
    client = LLMClient()
    assert client.available() is False
    assert client.available() is False
    assert capsys.readouterr().err.count("no REMEMBRANCER_LLM_MODEL") == 1
    with pytest.raises(LLMUnavailable, match="no REMEMBRANCER_LLM_MODEL"):
        client.decide("hi", TOOL)
    assert llm.decide_many(client, [("hi", TOOL)]) == [None]
    assert sdk["client"] == []


def test_no_llm_variables_means_no_llm(sdk):
    client = LLMClient()
    assert client.available() is False
    with pytest.raises(LLMUnavailable,
                       match="no LLM key: set REMEMBRANCER_LLM_API_KEY and REMEMBRANCER_LLM_MODEL"):
        client.decide("hi", TOOL)
    assert sdk["client"] == []


def test_an_empty_variable_counts_as_unset(sdk, monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_LLM_API_KEY", "k")
    monkeypatch.setenv("REMEMBRANCER_LLM_MODEL", "m")
    monkeypatch.setenv("REMEMBRANCER_LLM_BASE_URL", "")
    monkeypatch.setenv("REMEMBRANCER_LLM_TIMEOUT_SECONDS", "")
    assert _request(sdk) == (None, "m", "k")
    assert sdk["client"][-1]["timeout"] == 120.0
    monkeypatch.setenv("REMEMBRANCER_LLM_API_KEY", "")
    assert LLMClient().available() is False


def test_the_requirement_names_the_key_and_the_model():
    assert llm.llm_requirement().startswith("REMEMBRANCER_LLM_API_KEY and REMEMBRANCER_LLM_MODEL")


# ─── Fan-out ─────────────────────────────────────────────────────────────────


def test_concurrency_reads_its_variables(monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY", "6")
    monkeypatch.setenv("REMEMBRANCER_LLM_ADAPTIVE_CONCURRENCY", "0")
    assert llm.AdaptiveConcurrency().level() == 6
    assert concurrency_config()["adaptive"] is False

    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY", "5")
    monkeypatch.setenv("REMEMBRANCER_LLM_ADAPTIVE_CONCURRENCY", "1")
    assert llm.AdaptiveConcurrency().level() == 5
    assert concurrency_config()["adaptive"] is True


def test_the_fan_out_bounds_are_read_from_their_variables(monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY", "12")
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY_MIN", "3")
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY_MAX", "20")
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY_SHARES", "2")
    monkeypatch.setenv("REMEMBRANCER_LLM_FAILURE_THRESHOLD", "0.25")
    cfg = concurrency_config()
    assert (cfg["start"], cfg["floor"], cfg["ceiling"], cfg["shares"]) == (6, 1, 10, 2)
    assert cfg["threshold"] == 0.25


def test_an_unreadable_number_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY", "many")
    assert llm.AdaptiveConcurrency().level() == llm.DEFAULT_CONCURRENCY
    monkeypatch.setenv("REMEMBRANCER_LLM_FAILURE_THRESHOLD", "high")
    assert llm_float("failure_threshold", 0.02) == 0.02
    monkeypatch.setenv("REMEMBRANCER_LLM_MAX_ATTEMPTS", "")
    assert llm_int("max_attempts", 7) == 7


def test_the_retry_knobs_are_read_from_their_variables_at_import():
    """MAX_ATTEMPTS and the others are module constants, so a fresh process
    is the only honest way to read them."""
    env = {k: v for k, v in os.environ.items() if k not in ALL_NAMES}
    env.update({
        "REMEMBRANCER_LLM_MAX_ATTEMPTS": "3",
        "REMEMBRANCER_LLM_REPAIR_RETRIES": "2",
        "REMEMBRANCER_LLM_SPLIT_RETRIES": "4",
        "REMEMBRANCER_LLM_MAX_TOKENS_CEILING": "999",
        "REMEMBRANCER_LLM_CONCURRENCY_DEFAULT": "5",
    })
    code = ("from distill import llm; print(llm.MAX_ATTEMPTS, llm.REPAIR_RETRIES, "
            "llm.SPLIT_RETRIES, llm.MAX_TOKENS_CEILING, llm.DEFAULT_CONCURRENCY)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout
    assert out.split() == ["3", "2", "4", "999", "5"]


# ─── Accounting ──────────────────────────────────────────────────────────────


def test_the_caller_label_is_read_from_its_variable(monkeypatch):
    assert llm.caller() == llm.UNKNOWN_CALLER
    monkeypatch.setenv("REMEMBRANCER_LLM_CALLER", "nightly")
    assert llm.caller() == "nightly"


def test_usage_goes_to_its_variable_and_defaults_under_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    llm._log_usage("m", {"input_tokens": 1, "output_tokens": 1})
    assert json.loads((tmp_path / USAGE_LOG_DEFAULT).read_text())["model"] == "m"

    named = tmp_path / "elsewhere.jsonl"
    monkeypatch.setenv("REMEMBRANCER_LLM_USAGE_LOG", str(named))
    llm._log_usage("m2", {"input_tokens": 1, "output_tokens": 1})
    assert json.loads(named.read_text())["model"] == "m2"


def test_the_failures_log_and_health_file_follow_their_variables(monkeypatch, tmp_path):
    assert llm_repair.failures_log_path() is None      # "off", as the fixture set
    assert llm_repair.health_path() is None
    monkeypatch.setenv("REMEMBRANCER_LLM_FAILURES_LOG", str(tmp_path / "f.jsonl"))
    monkeypatch.setenv("REMEMBRANCER_LLM_HEALTH", str(tmp_path / "h.json"))
    assert llm_repair.failures_log_path() == str(tmp_path / "f.jsonl")
    assert llm_repair.health_path() == str(tmp_path / "h.json")
    monkeypatch.delenv("REMEMBRANCER_LLM_FAILURES_LOG")
    monkeypatch.delenv("REMEMBRANCER_LLM_HEALTH")
    assert llm_repair.failures_log_path() == FAILURES_LOG_DEFAULT
    assert llm_repair.health_path() == HEALTH_PATH_DEFAULT


def test_every_default_file_is_under_state():
    for path in (USAGE_LOG_DEFAULT, FAILURES_LOG_DEFAULT, HEALTH_PATH_DEFAULT):
        assert Path(path).parent == Path("state"), path
