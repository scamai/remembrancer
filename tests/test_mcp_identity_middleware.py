"""The MCP transport's ACL boundary, driven as ASGI.

`_proxy_proof_valid` and `_session_refined_identity` were unit-tested, but
nothing ever instantiated the middleware or sent a scope through it: coverage
reported every line of `TailscaleIdentityMiddleware.__call__` as never executed.
Deleting the proof check -- or the `add_middleware` call in `_run_http` -- left
the whole suite green while every MCP caller became an unauthenticated proxy
free to set `Tailscale-User-Login`.
"""
import asyncio
import json

import mcp.server as s

_SECRET = "m" * 48


def _headers(**kwargs):
    out = []
    for key, value in kwargs.items():
        if value is None:
            continue
        out.append((key.replace("_", "-").encode("ascii"), value.encode("latin-1")))
    return out


def _drive(scope_headers, monkeypatch, scope_type="http"):
    """Send one scope through the real middleware over a recording app."""
    monkeypatch.setenv("REMEMBRANCER_MCP_PROXY_SECRET", _SECRET)
    seen = {}
    sent = []

    async def _app(scope, receive, send):
        seen["identity"] = s._REQUEST_IDENTITY.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = s._build_tailscale_identity_middleware()(_app)

    async def _send(message):
        sent.append(message)

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(middleware(
        {"type": scope_type, "method": "POST", "path": "/remembrancer",
         "headers": scope_headers}, _receive, _send))
    return seen, sent


def test_a_request_without_the_proof_is_refused_and_never_reaches_the_app(monkeypatch):
    seen, sent = _drive(
        _headers(**{"tailscale_user_login": "jdoe"}), monkeypatch)
    assert sent[0]["status"] == 403
    assert json.loads(sent[1]["body"])["error"] == "untrusted reverse proxy"
    assert "identity" not in seen, "the app must not run for an unproven caller"


def test_a_wrong_proof_is_refused(monkeypatch):
    _, sent = _drive(
        _headers(x_remembrancer_proxy_secret="n" * 48, tailscale_user_login="jdoe"),
        monkeypatch)
    assert sent[0]["status"] == 403


def test_a_proven_request_binds_the_identity_it_carries(monkeypatch):
    seen, sent = _drive(
        _headers(x_remembrancer_proxy_secret=_SECRET, tailscale_user_login="jdoe"),
        monkeypatch)
    assert sent[0]["status"] == 200
    assert seen["identity"] == "jdoe"


def test_an_authorization_header_is_refused_when_no_token_sign_in_is_loaded(monkeypatch):
    """A token the server cannot check never falls back to the login header."""
    monkeypatch.setattr(s, "_owner_auth", None)
    seen, sent = _drive(
        _headers(x_remembrancer_proxy_secret=_SECRET, tailscale_user_login="jdoe",
                 authorization="Bearer t0ken"), monkeypatch)
    assert sent[0]["status"] == 401
    assert json.loads(sent[1]["body"]) == {
        "error": "token sign-in unavailable: send no Authorization header"}
    assert "identity" not in seen, "the app must not run for a refused caller"
    seen, sent = _drive(
        _headers(x_remembrancer_proxy_secret=_SECRET, tailscale_user_login="jdoe"), monkeypatch)
    assert sent[0]["status"] == 200 and seen["identity"] == "jdoe"


def test_the_identity_is_unset_again_after_the_request(monkeypatch):
    _drive(_headers(x_remembrancer_proxy_secret=_SECRET, tailscale_user_login="jdoe"),
           monkeypatch)
    assert s._REQUEST_IDENTITY.get() is None


def test_a_websocket_scope_cannot_skip_the_proof(monkeypatch):
    """The scope test used to be `!= "http"`, so a websocket handshake bypassed
    the only control protecting the MCP surface and ran with no identity bound.
    """
    _, sent = _drive(_headers(tailscale_user_login="jdoe"), monkeypatch,
                     scope_type="websocket")
    assert sent and sent[0]["type"] == "websocket.close"
    assert sent[0]["code"] == 1008


def test_a_lifespan_scope_still_passes_through(monkeypatch):
    monkeypatch.setenv("REMEMBRANCER_MCP_PROXY_SECRET", _SECRET)
    reached = []

    async def _app(scope, receive, send):
        reached.append(scope["type"])

    middleware = s._build_tailscale_identity_middleware()(_app)

    async def _noop(*_a, **_k):
        return {}

    asyncio.run(middleware({"type": "lifespan"}, _noop, _noop))
    assert reached == ["lifespan"]


def test_the_http_transport_actually_installs_the_middleware():
    """Dropping the add_middleware call leaves every unit test above passing."""
    import inspect

    source = inspect.getsource(s._run_http)
    assert "_build_tailscale_identity_middleware()" in source
    assert "add_middleware" in source
