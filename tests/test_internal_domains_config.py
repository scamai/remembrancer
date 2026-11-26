"""The org's own mail domains come from config, never from code.

Attendee resolution (``distill/entities.py``) reads ``entities.internal_domains``,
then ``governance.internal_domains``, from the ``config.json`` of the memory
root that holds the registry. A root that names none has none: every attendee
is resolved as an outsider. Governance (``govern.py``) reads only
``governance.internal_domains``.
"""
import json

import pytest

from distill.entities import (
    EntityResolver,
    configured_internal_domains,
    internal_domains_for_registry,
)
from tests.registry_trust_helpers import attest_yaml_text

_OURS = "ana@acme.example"


def _root(tmp_path, cfg=None):
    """A memory root with an empty registry, and ``cfg`` as its config.json."""
    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "people.yml").write_text(attest_yaml_text("people: []\n", "people"))
    (reg / "orgs.yml").write_text(attest_yaml_text("orgs: []\n", "orgs"))
    (reg / "customers.yml").write_text("customers: []\n")
    (reg / "noise.yml").write_text("free_email: [gmail.com]\ninfra: []\n")
    if cfg is not None:
        (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(reg / "customers.yml")


def _via(resolver, email):
    """How the resolver's attendee chain classified one attendee."""
    (rec,) = resolver.resolve_attendees_deep([("", email)])["trace"]
    return rec["via"]


# ─── configured_internal_domains ────────────────────────────────────────────

def test_entities_key_wins_over_governance():
    cfg = {"entities": {"internal_domains": ["acme.example"]},
           "governance": {"internal_domains": ["acme.example", "acme-mail.example"]}}
    assert configured_internal_domains(cfg) == ["acme.example"]


def test_governance_key_is_the_fallback():
    cfg = {"entities": {}, "governance": {"internal_domains": ["acme.example"]}}
    assert configured_internal_domains(cfg) == ["acme.example"]


def test_values_are_normalised_like_governance_reads_them():
    cfg = {"entities": {"internal_domains": ["@Acme.Example", " .acme.example ",
                                             "ACME.example", "", 7]}}
    assert configured_internal_domains(cfg) == ["acme.example"]
    assert configured_internal_domains(
        {"entities": {"internal_domains": "Acme.Example"}}) == ["acme.example"]


@pytest.mark.parametrize("cfg", [
    {}, None, [], "acme.example",
    {"entities": None}, {"governance": "acme.example"},
    {"entities": {"internal_domains": None}},
    {"entities": {"internal_domains": {"acme.example": True}}},
])
def test_no_usable_list_names_no_domain(cfg):
    assert configured_internal_domains(cfg) == []


def test_a_malformed_entities_list_does_not_fall_back_to_governance():
    """The key is present, so it is the answer, even when it names nothing."""
    cfg = {"entities": {"internal_domains": 7},
           "governance": {"internal_domains": ["acme.example"]}}
    assert configured_internal_domains(cfg) == []


# ─── internal_domains_for_registry ──────────────────────────────────────────

def test_the_registry_root_config_is_read(tmp_path):
    path = _root(tmp_path, {"entities": {"internal_domains": ["acme.example"]}})
    assert internal_domains_for_registry(path) == ["acme.example"]


def test_a_registry_root_without_config_names_no_domain(tmp_path):
    assert internal_domains_for_registry(_root(tmp_path)) == []


@pytest.mark.parametrize("text", ["{not json", "", "[]"])
def test_an_unusable_config_names_no_domain(tmp_path, text):
    path = _root(tmp_path)
    (tmp_path / "config.json").write_text(text, encoding="utf-8")
    assert internal_domains_for_registry(path) == []


# ─── EntityResolver, built the way callers build it ─────────────────────────

def test_the_resolver_reads_its_root_config(tmp_path):
    resolver = EntityResolver(_root(
        tmp_path, {"governance": {"internal_domains": ["acme.example"]}}))
    assert _via(resolver, _OURS) == "internal-domain"
    assert resolver.resolve_attendees_deep([("", _OURS)])["proposals"] == []


def test_without_config_every_attendee_is_an_outsider(tmp_path):
    resolver = EntityResolver(_root(tmp_path))
    assert _via(resolver, _OURS) == "proposed:map-domain"


def test_an_explicit_list_overrides_the_root_config(tmp_path):
    path = _root(tmp_path, {"entities": {"internal_domains": ["acme.example"]}})
    other = EntityResolver(path, internal_domains=["other.example"])
    assert _via(other, _OURS) == "proposed:map-domain"
    assert _via(other, "bo@other.example") == "internal-domain"
    assert _via(EntityResolver(path, internal_domains=[]), _OURS) \
        == "proposed:map-domain"


