"""distill/llm.py::AdaptiveConcurrency — the self-tuning fan-out level.

The fixed default of 4 could not be right twice: it can sit far below what an
account takes on a quiet day and far above what it takes during a provider
incident. So the level is a control loop, and these tests pin the loop's
contract:

* additive increase, multiplicative decrease — back off fast, recover slowly;
* a floor, a ceiling, and an account-wide budget divided by sibling processes;
* a prose answer is a schema problem, not provider pushback, so it never
  shrinks the fan-out;
* the level is published to the health file, because a level nobody can
  read back is a level nobody can operate.

Stubs only: no network, no key.
"""
import json
import threading

import pytest

from distill import llm
from distill.llm_env import LEGACY_LLM_ENV, LLM_ENV


class _Clock:
    """Monotonic clock the test advances by hand."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


#: The fan-out settings (distill/llm_env.py).
_FAN_OUT = ("concurrency", "concurrency_min", "concurrency_max", "concurrency_shares",
            "failure_threshold", "adaptive_concurrency")


@pytest.fixture(autouse=True)
def _clean_concurrency_env(monkeypatch):
    """Every test here starts from the production defaults.

    Without this, one leaked REMEMBRANCER_LLM_CONCURRENCY_SHARES (or an older name for
    it) from anywhere in the suite silently divides every level these tests
    assert on.
    """
    for setting in _FAN_OUT:
        for table in (LLM_ENV, LEGACY_LLM_ENV):
            if setting in table:
                monkeypatch.delenv(table[setting], raising=False)


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(llm.time, "monotonic", fake)
    return fake


@pytest.fixture
def controller():
    """A fresh controller with the production defaults and no shard division."""
    return llm.AdaptiveConcurrency()


def _successes(controller, count):
    for _ in range(count):
        controller.note_success()


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

def test_default_start_is_the_measured_level_not_the_old_constant(controller):
    """8 start / 16 ceiling / 4 floor: the loop starts just under the knee
    where a throughput probe stopped gaining and pushback began."""
    assert llm.DEFAULT_CONCURRENCY == 8
    assert controller.level() == 8
    assert controller.snapshot()["ceiling"] == 16
    assert controller.snapshot()["floor"] == 4


def test_shares_divide_one_account_budget_between_sibling_processes(monkeypatch):
    """The provider's cap is per ACCOUNT: N shard children must not each claim
    the whole budget (N times the per-process default in flight)."""
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY_SHARES", "4")
    controller = llm.AdaptiveConcurrency()
    assert controller.level() == 2            # 8 / 4
    assert controller.snapshot()["ceiling"] == 4   # 16 / 4


def test_an_explicitly_small_start_is_not_raised_by_the_default_floor(monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY", "2")
    controller = llm.AdaptiveConcurrency()
    assert controller.level() == 2
    assert controller.snapshot()["floor"] == 2


def test_changing_the_budget_resets_the_loop(monkeypatch, controller, clock):
    _successes(controller, llm.RAMP_MIN_SAMPLES)
    clock.advance(llm.RAMP_INTERVAL_SECONDS + 1)
    assert controller.settle() == 9
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY", "6")
    # A level tuned against a different budget must not survive the change.
    assert controller.level() == 6
    assert controller.snapshot()["samples"] == 0


# --------------------------------------------------------------------------- #
# Ramp up                                                                     #
# --------------------------------------------------------------------------- #

def test_ramp_is_additive_and_gated_on_evidence_and_time(controller, clock):
    _successes(controller, llm.RAMP_MIN_SAMPLES - 1)
    clock.advance(llm.RAMP_INTERVAL_SECONDS + 1)
    assert controller.settle() == 8, "too few samples to justify a step"

    controller.note_success()
    assert controller.settle() == 9
    # One step per interval: a second settle in the same instant changes nothing.
    assert controller.settle() == 9
    clock.advance(llm.RAMP_INTERVAL_SECONDS + 1)
    assert controller.settle() == 10


def test_ramp_stops_at_the_ceiling(monkeypatch, clock):
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY", "4")
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY_MAX", "6")
    controller = llm.AdaptiveConcurrency()
    for _ in range(10):
        _successes(controller, llm.RAMP_MIN_SAMPLES)
        clock.advance(llm.RAMP_INTERVAL_SECONDS + 1)
        controller.settle()
    assert controller.level() == 6


def test_a_failure_rate_over_the_threshold_blocks_the_ramp(controller, clock):
    _successes(controller, 90)
    for _ in range(10):                      # 10% — five times the threshold
        controller.note_failure("http_400")
    clock.advance(llm.RAMP_INTERVAL_SECONDS + 1)
    assert controller.settle() == 4, "over threshold backs off instead of ramping"


# --------------------------------------------------------------------------- #
# Back off                                                                    #
# --------------------------------------------------------------------------- #

def test_a_rate_limit_burst_halves_the_level_immediately(controller, clock):
    _successes(controller, 200)
    for _ in range(llm.BURST_FAILURES - 1):
        controller.note_failure("rate_limit")
    assert controller.level() == 8, "a couple of 429s is not a burst"
    controller.note_failure("rate_limit")
    assert controller.level() == 4, "a burst must not wait for the batch to end"


def test_backoff_is_multiplicative_and_the_floor_holds(controller, clock):
    for _ in range(6):
        for _ in range(llm.BURST_FAILURES):
            controller.note_failure("http_503")
        clock.advance(llm.BURST_SECONDS + 1)
    assert controller.level() == controller.snapshot()["floor"] == 4


def test_recovery_from_a_burst_is_slow(controller, clock):
    """Fast down, slow up: halving is one event, climbing back costs one step
    per quiet interval — the property that stops a burst from oscillating."""
    for _ in range(llm.BURST_FAILURES):
        controller.note_failure("timeout")
    assert controller.level() == 4
    clock.advance(llm.WINDOW_SECONDS + 1)      # the burst leaves the window
    _successes(controller, llm.RAMP_MIN_SAMPLES)
    clock.advance(llm.RAMP_INTERVAL_SECONDS + 1)
    assert controller.settle() == 5            # +1, not back to 8


def test_a_schema_refusal_never_shrinks_the_fan_out(controller, clock):
    """More concurrency neither causes nor cures prose instead of a tool call."""
    for _ in range(20):
        controller.note_failure("no_tool_use")
    assert controller.level() == 8


def test_adaptation_can_be_pinned_off(monkeypatch, clock):
    monkeypatch.setenv("REMEMBRANCER_LLM_ADAPTIVE_CONCURRENCY", "0")
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY", "8")
    controller = llm.AdaptiveConcurrency()
    for _ in range(10):
        controller.note_failure("rate_limit")
    _successes(controller, 100)
    clock.advance(llm.RAMP_INTERVAL_SECONDS + 1)
    assert controller.settle() == 8
    assert controller.level() == 8


# --------------------------------------------------------------------------- #
# Observability                                                               #
# --------------------------------------------------------------------------- #

def test_the_level_is_published_to_the_health_file(monkeypatch, tmp_path, clock):
    monkeypatch.setenv("REMEMBRANCER_LLM_HEALTH", str(tmp_path / "llm_health.json"))
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY", "8")
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY_MAX", "16")
    controller = llm.AdaptiveConcurrency()
    for _ in range(llm.BURST_FAILURES):
        controller.note_failure("rate_limit")
    doc = json.loads((tmp_path / "llm_health.json").read_text())
    assert doc["concurrency"] == 4
    assert doc["concurrency_max"] == 16
    assert doc["concurrency_shares"] == 1
    assert doc["concurrency_reason"].startswith("burst:")
    assert doc["concurrency_updated_at"]


def test_health_counters_still_add_up_beside_the_level(monkeypatch, tmp_path):
    from distill import llm_repair

    monkeypatch.setenv("REMEMBRANCER_LLM_HEALTH", str(tmp_path / "llm_health.json"))
    llm_repair.bump_health(calls=2, transport_failures=1)
    llm_repair.set_health(concurrency=9)
    llm_repair.bump_health(calls=3)
    doc = llm_repair.load_health()
    assert doc["calls"] == 5 and doc["transport_failures"] == 1
    assert doc["concurrency"] == 9


def test_set_health_ignores_unknown_fields(monkeypatch, tmp_path):
    from distill import llm_repair

    monkeypatch.setenv("REMEMBRANCER_LLM_HEALTH", str(tmp_path / "llm_health.json"))
    llm_repair.set_health(concurrency=5, prompt="secret text")
    doc = llm_repair.load_health()
    assert doc["concurrency"] == 5
    assert "prompt" not in doc


# --------------------------------------------------------------------------- #
# decide_many wiring                                                          #
# --------------------------------------------------------------------------- #

class _CountingClient:
    def __init__(self, fail_first=0):
        self.lock = threading.Lock()
        self.current = 0
        self.max_concurrent = 0
        self.calls = 0
        self.fail_first = fail_first

    def available(self):
        return True

    def decide(self, prompt, tool, max_tokens=1500, strict=False):
        with self.lock:
            self.calls += 1
            index = self.calls
            self.current += 1
            self.max_concurrent = max(self.max_concurrent, self.current)
        try:
            if index <= self.fail_first:
                raise llm.LLMUnavailable("API failed [rate_limit]")
            return {"ok": index}
        finally:
            with self.lock:
                self.current -= 1


def _jobs(n):
    return [("prompt", {"name": "t"}) for _ in range(n)]


def test_decide_many_without_max_workers_uses_the_controller(monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY", "3")
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY_SHARES", "1")
    llm.CONCURRENCY._config = None            # re-read the env for this test
    client = _CountingClient()
    results = llm.decide_many(client, _jobs(9))
    assert len(results) == 9
    assert client.max_concurrent <= 3
    assert all(result is not None for result in results)


def test_an_explicit_max_workers_still_pins_the_batch(monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY", "16")
    llm.CONCURRENCY._config = None
    client = _CountingClient()
    llm.decide_many(client, _jobs(8), max_workers=2)
    assert client.max_concurrent <= 2
