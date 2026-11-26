"""Which identities name no person comes from config, past a short generic list.

``people_registry.SHARED_IDENTITIES`` holds only names that mean "no person" in
any deployment: the empty identity, the default shared tier ``internal``,
``unknown`` and ``root``. A deployment adds its own service, CI and deploy
accounts under ``config.json`` ``identity.shared_identities``. Each reader takes
the union from its root's config: the people registry and the MCP server's
identity checks.
"""
import asyncio
import json

import pytest
import yaml  # type: ignore[import-untyped]

import mcp.server as s
from people_registry import SHARED_IDENTITIES, PeopleRegistry, shared_identities

_PEOPLE = {"people": [
    {"id": "person:ana-lima", "name": "Ana Lima", "login": "analima",
     "aliases": ["ci-bot"], "emails": ["ana@acme.example"], "internal": True},
]}
_SHARED = {"identity": {"shared_identities": ["CI-Bot", " deploy "]}}


def _root(path, cfg=None, people=None):
    """A memory root: registry/people.yml, and ``cfg`` as its config.json."""
    reg = path / "registry"
    reg.mkdir(parents=True)
    (reg / "people.yml").write_text(yaml.safe_dump(people or _PEOPLE), encoding="utf-8")
    if cfg is not None:
        (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(path)


# ─── shared_identities ──────────────────────────────────────────────────────

def test_the_code_list_is_generic():
    assert SHARED_IDENTITIES == {"", "internal", "unknown", "root"}


def test_config_adds_its_names_lower_cased_and_stripped():
    assert shared_identities(_SHARED) == SHARED_IDENTITIES | {"ci-bot", "deploy"}
    assert shared_identities({"identity": {"shared_identities": "CI-Bot"}}) \
        == SHARED_IDENTITIES | {"ci-bot"}
    assert shared_identities({"identity": {"shared_identities": ["ci-bot", 7, None]}}) \
        == SHARED_IDENTITIES | {"ci-bot"}


@pytest.mark.parametrize("cfg", [
    None, [], "ci-bot", {}, {"identity": None}, {"identity": "ci-bot"},
    {"identity": {}}, {"identity": {"shared_identities": None}},
    {"identity": {"shared_identities": 7}},
    {"identity": {"shared_identities": {"ci-bot": True}}},
])
def test_no_usable_list_adds_nothing(cfg):
    assert shared_identities(cfg) == SHARED_IDENTITIES


# ─── the people registry, loaded the way callers load it ────────────────────

def test_a_configured_shared_identity_resolves_to_nobody(tmp_path):
    reg = PeopleRegistry.load(_root(tmp_path, _SHARED))
    assert reg.resolve_actor("ci-bot") is None
    assert reg.resolve_actor("analima") == "person:ana-lima"


def test_without_the_config_list_the_same_name_is_a_person(tmp_path):
    """Control: the alias resolves when the root's config does not list it."""
    reg = PeopleRegistry.load(_root(tmp_path, {}))
    assert reg.resolve_actor("ci-bot") == "person:ana-lima"


@pytest.mark.parametrize("ident", ["ana", "ANA"])
def test_a_configured_shared_identity_never_becomes_a_handle(tmp_path, ident):
    """A remembrancer identity that resolves to a person names their page, unless the
    config lists it as shared, in any case."""
    people = {"people": [{"id": "person:ana-lima", "name": "Ana Lima",
                          "emails": ["ana@acme.example"], "internal": True}]}
    cfg = {"identities": {ident: ["eng"]}}
    claimed = PeopleRegistry.load(_root(tmp_path / "a", cfg, people))
    assert claimed.by_key("person:ana-lima").handle == "ana"
    shared = PeopleRegistry.load(_root(
        tmp_path / "b", {**cfg, "identity": {"shared_identities": ["ana"]}}, people))
    assert shared.by_key("person:ana-lima").handle == "ana-lima"


# ─── the MCP server ─────────────────────────────────────────────────────────

def test_the_server_reads_the_list_from_the_root_it_is_given(tmp_path):
    listed = _root(tmp_path / "a", _SHARED)
    assert s._is_shared_identity("ci-bot", listed)
    assert s._is_shared_identity("Internal", listed)
    assert not s._is_shared_identity("analima", listed)
    assert not s._is_shared_identity("ci-bot", _root(tmp_path / "b", {}))


def test_a_private_delivery_to_a_configured_shared_tier_is_refused(tmp_path):
    ctx = {"identity": "ci-bot", "verified": True, "source": "ssh-peer",
           "host": "box", "session_id": "s1"}
    cfg = {"identities": {"ci-bot": ["eng"]}}
    listed = _root(tmp_path / "a", {**cfg, **_SHARED})
    assert "identity is a shared tier, not a person" in \
        s._private_delivery_blockers(ctx, root=listed)
    assert s._private_delivery_blockers(ctx, root=_root(tmp_path / "b", cfg)) == []


def test_only_a_configured_shared_tier_may_be_refined(tmp_path):
    identities = {"identities": {"ci-bot": ["eng"], "analima": ["eng"]}}
    shared = _root(tmp_path / "a", {**identities, **_SHARED})
    assert s._session_refined_identity("ci-bot", "analima", shared) == "analima"
    personal = _root(tmp_path / "b", identities)
    assert s._session_refined_identity("ci-bot", "analima", personal) == "ci-bot"


def test_the_edge_middleware_reports_a_configured_tier_as_shared(monkeypatch):
    """Through the real middleware: the request's identity source."""
    monkeypatch.setenv("REMEMBRANCER_MCP_PROXY_SECRET", "m" * 48)
    seen = {}

    async def _app(scope, receive, send):
        seen["source"] = s._REQUEST_SOURCE.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def _send(message):
        pass

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    def _source_for(cfg):
        monkeypatch.setattr(s, "_load_config", lambda root=None: cfg)
        middleware = s._build_tailscale_identity_middleware()(_app)
        headers = [(b"x-remembrancer-proxy-secret", b"m" * 48),
                   (b"tailscale-user-login", b"ci-bot")]
        asyncio.run(middleware({"type": "http", "method": "POST", "path": "/remembrancer",
                                "headers": headers}, _receive, _send))
        return seen["source"]

    assert _source_for(_SHARED) == "edge-shared"
    assert _source_for({}) == "edge-personal"


