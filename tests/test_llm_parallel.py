"""distill/llm.py::decide_many — THE sanctioned bounded-parallel LLM path.

LLMClient.decide already does per-call rate-limit retry/backoff; decide_many adds ONLY
bounded fan-out (ThreadPoolExecutor sized off REMEMBRANCER_LLM_CONCURRENCY, default 4, floor 1).
Fake clients only — no network, no env key.
"""
import threading
import time

from distill.llm import decide_many, LLMUnavailable


class _CountingClient:
    """Fake LLMClient: tracks max concurrent in-flight `decide` calls and can be told
    to fail specific jobs (raising LLMUnavailable, exactly what a retry-exhausted
    real decide() would raise)."""

    def __init__(self, available=True, fail_idx=frozenset(), delay=0.02):
        self._available = available
        self.fail_idx = set(fail_idx)
        self.delay = delay
        self.lock = threading.Lock()
        self.current = 0
        self.max_concurrent = 0
        self.call_count = 0

    def available(self):
        return self._available

    def decide(self, prompt, tool, max_tokens=1500):
        with self.lock:
            self.call_count += 1
            self.current += 1
            self.max_concurrent = max(self.max_concurrent, self.current)
        try:
            idx = tool["_idx"]
            # Vary delay so completion order differs from submission order —
            # exercises order-preservation for real.
            time.sleep(self.delay * (1 if idx % 2 == 0 else 0.1))
            if idx in self.fail_idx:
                raise LLMUnavailable(f"retries exhausted for job {idx}")
            return {"idx": idx}
        finally:
            with self.lock:
                self.current -= 1


def _jobs(n):
    return [("prompt", {"name": "t", "_idx": i}) for i in range(n)]


def test_order_preserved_regardless_of_completion_order():
    client = _CountingClient(delay=0.02)
    results = decide_many(client, _jobs(8), max_workers=4)
    assert len(results) == 8
    for i, r in enumerate(results):
        assert r == {"idx": i}


def test_concurrency_bound_respected():
    client = _CountingClient(delay=0.03)
    results = decide_many(client, _jobs(12), max_workers=3)
    assert len(results) == 12
    assert client.max_concurrent <= 3
    assert client.max_concurrent > 1  # actually ran in parallel, not accidentally serial


def test_one_failing_job_yields_none_without_killing_others():
    client = _CountingClient(fail_idx={2, 5}, delay=0.01)
    results = decide_many(client, _jobs(7), max_workers=3)
    assert len(results) == 7
    for i, r in enumerate(results):
        if i in (2, 5):
            assert r is None
        else:
            assert r == {"idx": i}


def test_unavailable_client_short_circuits_without_calling_decide():
    client = _CountingClient(available=False)
    results = decide_many(client, _jobs(5), max_workers=4)
    assert results == [None] * 5
    assert client.call_count == 0


def test_empty_jobs_returns_empty_list():
    client = _CountingClient()
    assert decide_many(client, []) == []
    assert client.call_count == 0


def test_default_max_workers_from_env(monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY", "2")
    client = _CountingClient(delay=0.03)
    results = decide_many(client, _jobs(8))  # no max_workers arg
    assert len(results) == 8
    assert client.max_concurrent <= 2


def test_max_workers_floored_at_one(monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_LLM_CONCURRENCY", "0")
    client = _CountingClient(delay=0.01)
    results = decide_many(client, _jobs(4))
    assert len(results) == 4
    assert client.max_concurrent >= 1
