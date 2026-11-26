"""End-to-end timing on the MCP audit rows of the three read tools.

`duration_ms` in state/queries.jsonl covered only retrieve.search(); the
workspace lane, trimming and the audit write ran after it, so the latency a
caller felt was in no log. These tests drive the real `_do_*` functions over a
real repository and read the row the audit log receives. The field list below
is a literal, not TIMING_FIELDS, so dropping a field from the server's tuple is
caught too (a list derived from the code under test could only shrink with it).

The wiring each FastMCP wrapper owns, handing `_read_call` the headers of its
own request, is pinned here with a stand-in FastMCP, so it is checked under any
interpreter, with or without the mcp SDK.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

import mcp.server as s
from remembrancerlib import Note, regen_index, write_note
from retrieve import contested as contested_module
from retrieve import query as retrieval
from retrieve import telemetry
from retrieve import verification as verification_module

COMMON = {
    "handler_start", "response_ts", "mcp_total_ms", "duration_ms", "path",
    "audit_loads", "audit_ms", "contested_scan_ms", "caller", "mcp_session",
    "request_id",
}
EXPECTED_FIELDS = {
    "remembrancer_search": COMMON | {"workspace_ms", "k"},
    "remembrancer_timeline": COMMON | {"k"},
    "remembrancer_get_note": COMMON,
}
HEADERS = {
    "x-request-id": "0123456789abcdef0123456789abcdef",
    "remembrancer-caller": "coverage",
    "mcp-session-id": "5f1c0e4b9a2d4c7e8b6a1f3d2e4c5b6a",
}
DELAY = 0.02   # seconds injected into each timed stage, so a zero cannot pass


def _note(root, name, description, *, entities=(), acl=("eng",)):
    write_note(
        Note(name=name, description=description, type="state", domain="eng",
             acl=list(acl), entities=list(entities), state="proposed", body=""),
        str(root / "context"),
    )
    result = retrieval.promote(name, identity="reviewer", root=str(root))
    assert result["ok"], result


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({
        "identities": {"engineer": ["eng"], "reviewer": ["eng"]},
        "retrieval": {"query_log": True},
        "trust": {"approver_identities": ["reviewer"]},
        "digest": {"review_policy": "settle"},
    }))
    _note(tmp_path, "orion-latency-fact", "orion latency fact",
          entities=["org:orion"])
    _note(tmp_path, "orion-second-fact", "orion second latency fact",
          entities=["org:orion"])
    regen_index(str(tmp_path / "context"))
    audit_log = tmp_path / "audit.jsonl"
    query_log = tmp_path / "queries.jsonl"
    monkeypatch.setenv("REMEMBRANCER_AUDIT_LOG", str(audit_log))
    monkeypatch.setenv("REMEMBRANCER_QUERY_LOG", str(query_log))
    monkeypatch.delenv("REMEMBRANCER_CALLER", raising=False)
    monkeypatch.delenv("REMEMBRANCER_IDENTITY", raising=False)
    monkeypatch.setattr(s, "ROOT", str(tmp_path))
    monkeypatch.setattr(s, "_retrieve_module", lambda: retrieval)
    # No event corpus here (it needs an attended corpus-guard checkpoint); the
    # timeline's note half is what loads the audit.
    monkeypatch.setattr(s, "_iter_events_safe", lambda *a, **k: [])
    verification_module.reset_audit_cache()
    contested_module.reset_cache()
    return tmp_path


def _slow(monkeypatch, module, attribute):
    real = getattr(module, attribute)

    def slow(*args, **kwargs):
        time.sleep(DELAY)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, attribute, slow)


@pytest.fixture
def slow_stages(monkeypatch):
    """Make every timed stage take measurable time on a tiny repository."""
    _slow(monkeypatch, verification_module, "_load_verification_audit")
    _slow(monkeypatch, contested_module, "_contested_index")
    _slow(monkeypatch, s, "_do_workspace_evidence")


def _rows(repo):
    path = repo / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _only_row(repo, tool):
    rows = [row for row in _rows(repo) if row["tool"] == tool]
    assert len(rows) == 1, rows
    return rows[0]


def _call(tool, fn, *args, **kwargs):
    """What each FastMCP tool wrapper does: open the call with its headers."""
    with s._read_call(tool, HEADERS):
        return fn(*args, **kwargs)


def _assert_fields(row, tool):
    expected = EXPECTED_FIELDS[tool]
    missing = sorted(field for field in expected if row.get(field) is None)
    assert not missing, f"{tool} row lacks {missing}"
    assert row["mcp_total_ms"] >= row["duration_ms"]
    assert row["caller"] == "coverage"
    assert row["request_id"] == HEADERS["x-request-id"]
    assert len(row["mcp_session"]) == 16
    assert HEADERS["mcp-session-id"] not in json.dumps(row)   # hashed, never raw
    assert row["handler_start"] <= row["response_ts"]
    assert row["handler_start"].endswith("Z") and "." in row["handler_start"]


# ─── every field, on each tool ───

def test_search_row_carries_every_timing_field(repo, slow_stages):
    _call("remembrancer_search", s._do_search, "orion latency", "engineer", 7, None,
          root=str(repo))
    row = _only_row(repo, "remembrancer_search")
    _assert_fields(row, "remembrancer_search")
    assert row["path"] == "index"
    assert row["k"] == 7
    # One search loads the verification audit exactly once.
    assert row["audit_loads"] == 1
    for stage in ("audit_ms", "contested_scan_ms", "workspace_ms"):
        assert row[stage] >= DELAY * 1000 * 0.9, (stage, row[stage])
    # duration_ms is search()'s own number: the same one the query log carries.
    logged = [json.loads(line) for line in
              (repo / "queries.jsonl").read_text().splitlines()]
    assert [entry["duration_ms"] for entry in logged] == [row["duration_ms"]]


def test_get_note_row_carries_every_timing_field(repo, slow_stages):
    _call("remembrancer_get_note", s._do_get_note, "orion-latency-fact", "engineer")
    row = _only_row(repo, "remembrancer_get_note")
    _assert_fields(row, "remembrancer_get_note")
    assert row["hit"] is True
    assert row["path"] == "full_load"
    assert row["audit_loads"] == 1
    assert row["audit_ms"] >= DELAY * 1000 * 0.9
    # The contested label is computed AFTER the hit is known; it must still be
    # inside the timed call, which is why the row is written at return.
    assert row["contested_scan_ms"] >= DELAY * 1000 * 0.9


def test_get_note_miss_row_is_timed_too(repo):
    _call("remembrancer_get_note", s._do_get_note, "no-such-note", "engineer")
    row = _only_row(repo, "remembrancer_get_note")
    assert row["hit"] is False
    assert row["mcp_total_ms"] >= row["duration_ms"] >= 0


def test_timeline_row_carries_every_timing_field(repo, slow_stages):
    _call("remembrancer_timeline", s._do_timeline, "org:orion", "engineer", 9,
          root=str(repo))
    row = _only_row(repo, "remembrancer_timeline")
    _assert_fields(row, "remembrancer_timeline")
    assert row["path"] == "full_load"
    assert row["k"] == 9
    # The timeline still loads the audit twice: the note half loads it inside
    # load_notes and again for the gate. This pins today's count, so a
    # change that removes one shows up here as a deliberate edit.
    assert row["audit_loads"] == 2
    assert row["contested_scan_ms"] == 0.0   # timeline never reads labels
    assert row["n_results"] == 2             # the note half really ran


# ─── what the end-to-end number adds ───

def test_end_to_end_includes_the_lane_search_duration_never_saw(repo, monkeypatch):
    lane = 0.15
    real = s._do_workspace_evidence

    def slow_lane(*args, **kwargs):
        time.sleep(lane)
        return real(*args, **kwargs)

    monkeypatch.setattr(s, "_do_workspace_evidence", slow_lane)
    _call("remembrancer_search", s._do_search, "orion latency", "engineer", 5, None,
          root=str(repo))
    row = _only_row(repo, "remembrancer_search")
    assert row["workspace_ms"] >= lane * 1000 * 0.9
    assert row["mcp_total_ms"] - row["duration_ms"] >= lane * 1000 * 0.9


def test_without_an_index_search_reports_the_fallback_path(repo):
    for index in (repo / "context").rglob("index.md"):
        index.unlink()
    _call("remembrancer_search", s._do_search, "orion latency", "engineer", 5, None,
          root=str(repo))
    assert _only_row(repo, "remembrancer_search")["path"] == "fallback"


def test_recovery_path_is_named_and_timed(repo, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("simulated index failure")

    monkeypatch.setattr(retrieval, "search", broken)
    _call("remembrancer_search", s._do_search, "orion latency", "engineer", 5, None,
          root=str(repo))
    row = _only_row(repo, "remembrancer_search")
    assert row["path"] == "recovery"
    assert row["duration_ms"] is not None
    assert row["mcp_total_ms"] >= row["duration_ms"]


def test_called_directly_a_do_function_times_itself(repo):
    """An in-process caller (a web UI, say) calls _do_timeline with no wrapper."""
    s._do_timeline("org:orion", "engineer", 5, root=str(repo))
    row = _only_row(repo, "remembrancer_timeline")
    assert row["mcp_total_ms"] >= row["duration_ms"]
    assert row["caller"] == "organic"
    assert row["request_id"] is None and row["mcp_session"] is None


def test_one_call_writes_one_row(repo):
    _call("remembrancer_search", s._do_search, "orion latency", "engineer", 5, None,
          root=str(repo))
    assert [row["tool"] for row in _rows(repo)] == ["remembrancer_search"]


def test_other_tools_audit_immediately_inside_a_timed_call(repo):
    with s._read_call("remembrancer_search", HEADERS):
        s._audit("remembrancer_graph", "engineer", "org:orion", 0, root=str(repo))
        assert [row["tool"] for row in _rows(repo)] == ["remembrancer_graph"]
    assert "mcp_total_ms" not in _rows(repo)[0]


def test_kept_mutation_an_immediate_audit_write_loses_the_end_to_end_time(
        repo, monkeypatch):
    """Kept mutation: write the row where _audit is called (the old behaviour)
    and the workspace lane, trimming and the label scan fall outside it."""
    monkeypatch.setattr(s, "_audit", s._write_audit)
    _call("remembrancer_search", s._do_search, "orion latency", "engineer", 5, None,
          root=str(repo))
    row = _only_row(repo, "remembrancer_search")
    assert "mcp_total_ms" not in row and "request_id" not in row


def test_no_record_leaks_out_of_a_call(repo):
    _call("remembrancer_search", s._do_search, "orion latency", "engineer", 5, None,
          root=str(repo))
    assert telemetry.current() is None


# ─── the FastMCP wrappers pass their request's headers ───

class _NoRequest(Exception):
    """What FastMCP's get_context().request_context raises outside a request."""


class _StandInFastMCP:
    """The two FastMCP surfaces the read wrappers touch: tool() and get_context().

    ``get_context().request_context.request`` is the HTTP request that carried
    the tool call (a starlette Request under mcp 1.26); _sdk_request_headers
    reads its ``headers``. Whether the real SDK puts the right request there
    needs an end-to-end test against the SDK; this checks that every wrapper asks.
    """

    def __init__(self, *_args, **_kwargs):
        self.tools: dict = {}
        self.headers = None

    def tool(self):
        def register(function):
            self.tools[function.__name__] = function
            return function
        return register

    def get_context(self):
        if self.headers is None:
            raise _NoRequest("no request: stdio")
        return SimpleNamespace(request_context=SimpleNamespace(
            request=SimpleNamespace(headers=self.headers)))


@pytest.fixture
def wrappers(repo, monkeypatch):
    monkeypatch.setattr(s, "_import_fastmcp", lambda: _StandInFastMCP)
    monkeypatch.delenv("REMEMBRANCER_TOOLS", raising=False)
    return s._build_server()


WRAPPER_CALLS = [
    ("remembrancer_search", {"query": "orion latency", "as_identity": "engineer"}),
    ("remembrancer_get_note", {"name": "orion-latency-fact", "as_identity": "engineer"}),
    ("remembrancer_timeline", {"entity": "org:orion", "as_identity": "engineer"}),
]


@pytest.mark.parametrize("tool, arguments", WRAPPER_CALLS,
                         ids=[tool for tool, _ in WRAPPER_CALLS])
def test_each_tool_wrapper_passes_its_request_headers(repo, wrappers, tool,
                                                      arguments):
    wrappers.headers = dict(HEADERS)
    wrappers.tools[tool](**arguments)
    _assert_fields(_only_row(repo, tool), tool)


@pytest.mark.parametrize("tool, arguments", WRAPPER_CALLS,
                         ids=[tool for tool, _ in WRAPPER_CALLS])
def test_a_wrapper_with_no_request_still_writes_its_row(repo, wrappers, tool,
                                                        arguments):
    """stdio: no HTTP request, so no request id or session, and no crash."""
    wrappers.headers = None
    wrappers.tools[tool](**arguments)
    row = _only_row(repo, tool)
    assert row["mcp_total_ms"] >= row["duration_ms"]
    assert row["caller"] == "organic"
    assert row["request_id"] is None and row["mcp_session"] is None


# ─── caller tags ───

@pytest.mark.parametrize("headers, env, expected", [
    ({}, None, "organic"),
    ({"remembrancer-caller": "  Meeting-Bot "}, None, "meeting-bot"),
    ({"remembrancer-caller": "bad tag; drop table"}, None, "invalid"),
    ({"remembrancer-caller": "x" * 41}, None, "invalid"),
    (None, "billing-sync", "billing-sync"),  # stdio / in-process: the env names it
    (None, None, "organic"),
    ({}, "billing-sync", "organic"),         # over HTTP the env never speaks
])
def test_caller_tag_resolution(monkeypatch, headers, env, expected):
    if env is None:
        monkeypatch.delenv("REMEMBRANCER_CALLER", raising=False)
    else:
        monkeypatch.setenv("REMEMBRANCER_CALLER", env)
    assert s._caller_tag(headers) == expected


def test_a_malformed_request_id_is_dropped_not_logged(repo):
    with s._read_call("remembrancer_search", {"x-request-id": "bad id\nforged"}):
        s._do_search("orion latency", "engineer", 5, None, root=str(repo))
    assert _only_row(repo, "remembrancer_search")["request_id"] is None


def test_the_probe_stays_out_of_the_query_log(repo):
    with s._read_call("remembrancer_search", {"remembrancer-caller": "mcp_smoke"}):
        s._do_search("orion latency", "engineer", 5, None, root=str(repo))
    assert not (repo / "queries.jsonl").exists()
    assert _only_row(repo, "remembrancer_search")["caller"] == "mcp_smoke"
    with s._read_call("remembrancer_search", {}):
        s._do_search("orion latency", "engineer", 5, None, root=str(repo))
    assert len((repo / "queries.jsonl").read_text().splitlines()) == 1


def test_every_timed_tool_has_a_field_list():
    assert set(s.TIMING_FIELDS) == set(EXPECTED_FIELDS) == set(s.TIMED_READ_TOOLS)
