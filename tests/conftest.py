"""Shared pytest fixtures for Remembrancer tests."""
import os
import base64
import subprocess
import sys
import tempfile

import pytest

from remembrancerlib import Note, write_note


def _test_approval_private_key() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    raw = Ed25519PrivateKey.generate().private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return base64.b64encode(raw).decode("ascii")


_APPROVAL_TEST_KEY = _test_approval_private_key()


@pytest.fixture(autouse=True)
def _default_full_access():
    """Run the suite in the local/CLI full-access context (the threat model is the serving
    edge, not tests). The fail-open regression tests explicitly `monkeypatch.delenv` this to
    assert deny-by-default. Serving paths (MCP/UI) never set the flag, so they default-deny."""
    prev = os.environ.get("REMEMBRANCER_ALLOW_FULL_ACCESS")
    previous_key = os.environ.get("REMEMBRANCER_APPROVAL_PRIVATE_KEY")
    previous_derivation = os.environ.get(
        "REMEMBRANCER_TEST_ALLOW_PRIVATE_APPROVAL_DERIVATION"
    )
    os.environ["REMEMBRANCER_ALLOW_FULL_ACCESS"] = "1"
    os.environ["REMEMBRANCER_APPROVAL_PRIVATE_KEY"] = _APPROVAL_TEST_KEY
    os.environ["REMEMBRANCER_TEST_ALLOW_PRIVATE_APPROVAL_DERIVATION"] = "1"
    # Never let a test write usage/audit rows into the checkout's real state/ —
    # residue there is easy to mistake for real query traffic.
    prev_audit = os.environ.get("REMEMBRANCER_AUDIT_LOG")
    prev_qlog = os.environ.get("REMEMBRANCER_QUERY_LOG")
    prev_llm_failures = os.environ.get("REMEMBRANCER_LLM_FAILURES_LOG")
    prev_llm_health = os.environ.get("REMEMBRANCER_LLM_HEALTH")
    prev_trace = os.environ.get("REMEMBRANCER_DECISION_TRACE_LOG")
    os.environ["REMEMBRANCER_AUDIT_LOG"] = "off"
    os.environ["REMEMBRANCER_QUERY_LOG"] = "off"
    os.environ.setdefault("REMEMBRANCER_WARM_AUDIT", "0")
    # Same rule for the LLM failures log and health file (distill/llm_repair):
    # tests that assert on them point the vars at tmp_path explicitly.
    os.environ["REMEMBRANCER_LLM_FAILURES_LOG"] = "off"
    os.environ["REMEMBRANCER_LLM_HEALTH"] = "off"
    # Same rule for the decision trace: a test must not append rows to the
    # checkout's state/decision_trace.jsonl. Tests that assert on the trace
    # point the var at tmp_path explicitly.
    os.environ["REMEMBRANCER_DECISION_TRACE_LOG"] = "off"
    yield
    if prev is None:
        os.environ.pop("REMEMBRANCER_ALLOW_FULL_ACCESS", None)
    else:
        os.environ["REMEMBRANCER_ALLOW_FULL_ACCESS"] = prev
    if previous_key is None:
        os.environ.pop("REMEMBRANCER_APPROVAL_PRIVATE_KEY", None)
    else:
        os.environ["REMEMBRANCER_APPROVAL_PRIVATE_KEY"] = previous_key
    if previous_derivation is None:
        os.environ.pop("REMEMBRANCER_TEST_ALLOW_PRIVATE_APPROVAL_DERIVATION", None)
    else:
        os.environ[
            "REMEMBRANCER_TEST_ALLOW_PRIVATE_APPROVAL_DERIVATION"
        ] = previous_derivation
    for name, prev_value in (("REMEMBRANCER_AUDIT_LOG", prev_audit),
                             ("REMEMBRANCER_QUERY_LOG", prev_qlog),
                             ("REMEMBRANCER_LLM_FAILURES_LOG", prev_llm_failures),
                             ("REMEMBRANCER_LLM_HEALTH", prev_llm_health),
                             ("REMEMBRANCER_DECISION_TRACE_LOG", prev_trace)):
        if prev_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev_value


@pytest.fixture(autouse=True)
def _repo_domains_from_the_code_folder():
    """run() binds the engine's repository-domain map to its root's config and
    leaves it bound. Reset it after each test, so a test that skips run() reads
    the code folder's config.json, whatever ran before it."""
    yield
    engine = sys.modules.get("distill.engine")
    if engine is not None:
        engine._REPO_DOMAIN = None


@pytest.fixture
def tmp_remembrancer():
    """Temp dir with a minimal Remembrancer layout: context/ with two notes.

    - public-note  (acl=[], domain=company) — public, mentions "detector"
    - ml-only-note (acl=["ml"], domain=ml)  — restricted, mentions "detector"
    """
    with tempfile.TemporaryDirectory() as tmp:
        context_dir = os.path.join(tmp, "context")
        os.makedirs(context_dir, exist_ok=True)

        public = Note(
            name="public-note",
            description="Detector evaluation baseline",
            domain="company",
            acl=[],
            body="Public benchmark for the detector pipeline.",
        )
        write_note(public, context_dir)

        restricted = Note(
            name="ml-only-note",
            description="ML-internal detector training metrics",
            domain="ml",
            acl=["ml"],
            body="Internal ML notes on the detector model.",
        )
        write_note(restricted, context_dir)

        yield tmp


@pytest.fixture
def tmp_git_repo():
    """Temp dir initialised as a git repo with one committed file."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "init", tmp], check=True, capture_output=True)
        readme = os.path.join(tmp, "README.md")
        with open(readme, "w") as f:
            f.write("# tmp repo\n")
        subprocess.run(
            ["git", "-C", tmp, "add", "README.md"], check=True, capture_output=True
        )
        subprocess.run(
            ["git", "-C", tmp, "commit", "-m", "init", "--allow-empty-message"],
            check=True,
            capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test",
                 "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test"},
        )
        yield tmp
