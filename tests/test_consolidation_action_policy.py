from types import SimpleNamespace

from distill.consolidate import build_prompt, consolidate
from distill.engine import Cluster, _apply_decision
from distill.consolidate import ConsolidationDecision
from remembrancerlib import load_notes, write_note
from remembrancerlib import Event, Note


def _cluster():
    return Cluster(key="policy", events=[Event(
        id="slack:policy",
        source="slack",
        author="lead",
        payload={"text": "Launch moved from July 20 to July 27."},
    )])


def test_prompt_distinguishes_state_evolution_from_true_supersession():
    prompt = build_prompt(_cluster(), [Note(
        name="launch-date", description="Launch is July 20", domain="product"
    )])
    assert "normal lifecycle progress is UPDATE" in prompt
    assert "old date/port/version" in prompt
    assert "separate replacement note" in prompt.lower()
    assert "merely repeats or confirms" in prompt


def test_update_requires_exact_candidate_target():
    client = SimpleNamespace(decide=lambda *_args, **_kwargs: {
        "action": "UPDATE",
        "target_note": "invented-note",
        "evidence_event_ids": ["slack:policy"],
        "evidence_quotes": ["Launch moved from July 20 to July 27."],
        "rationale": "same state",
    })
    result = consolidate(
        _cluster(), [Note(name="launch-date", description="old", domain="product")], client
    )
    assert result is None


def test_supersede_requires_distinct_name_and_real_candidate():
    candidate = Note(name="old-policy", description="old", domain="eng")
    base = {
        "action": "SUPERSEDE",
        "superseded_note": candidate.name,
        "evidence_event_ids": ["slack:policy"],
        "evidence_quotes": ["Launch moved from July 20 to July 27."],
        "rationale": "replacement",
    }
    client = SimpleNamespace(decide=lambda *_args, **_kwargs: dict(base))
    assert consolidate(_cluster(), [candidate], client) is None
    client.decide = lambda *_args, **_kwargs: {**base, "name": "new-policy"}
    assert consolidate(_cluster(), [candidate], client).action == "SUPERSEDE"


def test_unsafe_model_payload_shapes_fail_closed():
    base = {
        "action": "CREATE",
        "name": "safe-note",
        "description": "A grounded launch update",
        "evidence_event_ids": ["slack:policy"],
        "evidence_quotes": ["Launch moved from July 20 to July 27."],
        "rationale": "new state",
    }
    for patch in (
        {"name": "Unsafe_Name"},
        {"description": "line one\nline two"},
        {"evidence_event_ids": [{"bad": "shape"}]},
        {"links": ["not-an-object"]},
        {"entities": {"not": "a-list"}},
    ):
        client = SimpleNamespace(decide=lambda *_args, _patch=patch, **_kwargs: {
            **base, **_patch,
        })
        assert consolidate(_cluster(), [], client) is None


def test_link_requires_a_different_offered_candidate():
    target = Note(name="launch-date", description="old", domain="product")
    peer = Note(name="launch-plan", description="plan", domain="product")
    base = {
        "action": "LINK", "target_note": target.name,
        "evidence_event_ids": ["slack:policy"],
        "evidence_quotes": ["Launch moved from July 20 to July 27."],
        "rationale": "related",
    }
    client = SimpleNamespace(decide=lambda *_args, **_kwargs: {
        **base, "links": [{"relation": "relates", "name": target.name}],
    })
    assert consolidate(_cluster(), [target, peer], client) is None
    client.decide = lambda *_args, **_kwargs: {
        **base, "links": [{"relation": "relates", "name": peer.name}],
    }
    assert consolidate(_cluster(), [target, peer], client).action == "LINK"


def test_grounded_model_update_applies_deterministically_and_converges(tmp_path):
    context = str(tmp_path / "context")
    target = Note(
        name="launch-date", description="Launch moved from July 20 to July 27",
        domain="product", entities=["initiative:launch"], provenance=["slack:old"],
        body="Old launch state.", state="proposed",
    )
    write_note(target, context)
    cluster = _cluster()
    cluster.entities = ["initiative:launch"]
    decision = ConsolidationDecision(
        action="UPDATE", target_note=target.name,
        evidence_event_ids=["slack:policy"],
        evidence_quotes=["Launch moved from July 20 to July 27."],
    )
    result = _apply_decision(
        decision, cluster, {target.name: target}, context,
        allowed_targets={target.name},
    )
    assert result is not None and result.action == "UPDATE"
    assert "slack:policy" in load_notes(context)[0].provenance


def test_model_supersede_policy_fallback_is_quality_terminal(tmp_path):
    context = str(tmp_path / "context")
    target = Note(
        name="old-policy", description="Launch moved from July 20 to July 27",
        domain="product", entities=["initiative:launch"], body="Old policy.",
    )
    write_note(target, context)
    cluster = _cluster()
    cluster.entities = ["initiative:launch"]
    decision = ConsolidationDecision(
        action="SUPERSEDE", name="replacement-policy",
        superseded_note=target.name, description="Replacement policy",
        body="Replacement policy body.", evidence_event_ids=["slack:policy"],
        evidence_quotes=["Launch moved from July 20 to July 27."],
    )
    result = _apply_decision(
        decision, cluster, {target.name: target}, context,
        allowed_targets={target.name},
    )
    assert result is not None
    assert "supersede-policy-fallback" in result.detail
    assert load_notes(context)[0].superseded is False
