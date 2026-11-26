"""Domain routing: the LLM routes a note's domain by what the fact is ABOUT.

consolidate() must surface a validated `domain` (one of DOMAINS, else None), and the
engine must honor it over the cluster's coarse sender-derived domain.
"""
from distill.consolidate import consolidate, DOMAINS


class _FakeClient:
    def __init__(self, out):
        self._out = out
    def decide(self, prompt, tool, max_tokens=1500):
        return self._out


class _Cl:
    domain = "customers"
    entities = []
    class _Event:
        id = "git:domain-test"
        source = "git"
        occurred_at = "2026-01-01T00:00:00Z"
        author = "tester"
        payload = {"subject": "durable domain routing fact"}
    events = [_Event()]
    key = "k"


def _grounded(out):
    return {**out, "evidence_event_ids": ["git:domain-test"],
            "evidence_quotes": ["durable domain routing fact"]}


def test_valid_domain_passes_through():
    d = consolidate(_Cl(), [], _FakeClient(
        _grounded({"action": "CREATE", "name": "hire-x", "domain": "internal", "rationale": "r"})))
    assert d.domain == "internal"


def test_invalid_domain_falls_back_to_none():
    d = consolidate(_Cl(), [], _FakeClient(
        _grounded({"action": "CREATE", "name": "x", "domain": "not-a-domain", "rationale": "r"})))
    assert d is None


def test_missing_domain_is_none():
    d = consolidate(_Cl(), [], _FakeClient(
        _grounded({"action": "CREATE", "name": "x", "rationale": "r"})))
    assert d.domain is None


def test_all_declared_domains_validate():
    assert "company" in DOMAINS
    for dom in DOMAINS:
        d = consolidate(_Cl(), [], _FakeClient(
            _grounded({"action": "CREATE", "name": "x", "domain": dom, "rationale": "r"})))
        assert d.domain == dom
