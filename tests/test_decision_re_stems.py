"""`DECISION_RE`'s four stems match their inflected forms.

`decid`, `migrat`, `adopt` and `deprecat` sat inside a group that closes with `\\b`,
so each matched only the literal stem: "we decided", "migrated the db", "deprecated v1"
and "adoption of uv" were not durable, while the future-tense "we will ship it" was.
Slack and gmail are not in `DURABLE_SOURCES`, so for them this regex alone decides
whether a cluster is durable, and an LLM CREATE needs a durable cluster.

The table tests read the module's `DECISION_RE`; the run test goes through
`engine.run`, the entry point that gates the LLM on durability. The kept mutations at the
bottom restore each bare stem (and drop the leading `\\b`) and require the matching
test inputs to flip, so every guard below is shown to be load-bearing.
"""
from __future__ import annotations

import json
import os
import re
import types

import pytest

from distill import engine
from distill.consolidate import ConsolidationDecision
from distill.engine import DECISION_RE, _is_durable
from remembrancerlib import Event, append_events
from tests.event_validation_helpers import guarded_run

# ─── the word table ───

# Required words, keyed by the stem that has to carry them.  "decision"
# has its own alternative, and "adopt" is the stem itself, so neither depends on `\w*`.
INFLECTED = {
    "decid": ("decide", "decided", "deciding"),
    "migrat": ("migrate", "migrated", "migration"),
    "adopt": ("adopted", "adoption"),
    "deprecat": ("deprecated", "deprecation"),
}
ALWAYS = ("decision", "adopt")
REQUIRED = tuple(w for words in INFLECTED.values() for w in words) + ALWAYS

# Phrases the old pattern missed, which match now.  None of these phrases
# carries any other DECISION_RE keyword, so each depends on one stem alone.
WERE_FALSE = (
    "we decided to go", "we decide today", "migrated the db", "db migration done",
    "deprecated v1", "deprecation notice", "adopted bun", "adoption of uv",
)

# Phrases the old regex already accepted, one or more per alternative, including
# the future-tense ones.  Each must stay True.
ACCEPTED_BEFORE = (
    "we will ship it next week", "plan to launch in Q4", "will approve",
    "decision", "we chose rust", "chosen", "agreed", "ship", "shipped", "ships",
    "shipping", "launch", "launched", "rolled out", "release", "released", "approve",
    "approved", "fix", "fixed", "fixes", "resolved", "resolves", "root cause",
    "root-cause", "rootcause", "postmortem", "outcome", "eval", "evaluation",
    "baseline", "threshold", "policy", "standard", "adopt",
)

# The pattern exactly as it was before the stems were widened, to prove
# ACCEPTED_BEFORE and WERE_FALSE describe the old behaviour rather than
# restating the new one.
OLD_DECISION_RE = re.compile(
    r"\b(decid|decision|chose|chosen|agreed|ship(?:ped|s|ping)?|launch(?:ed)?|"
    r"rolled out|released?|approved?|migrat|adopt|deprecat|fix(?:ed|es)?|"
    r"resolv(?:ed|es)?|root[- ]?cause|postmortem|outcome|eval(?:uation)?|"
    r"baseline|threshold|policy|standard)\b",
    re.IGNORECASE,
)

# The leading `\b` must survive the widening: a stem inside a longer word is not one.
STAY_FALSE = (
    "undecided", "indecision", "readopted", "remigrate", "thanks, see you tomorrow",
    "lol nice",
)


def _missed(pattern: re.Pattern, words) -> set[str]:
    return {w for w in words if not pattern.search(w)}


def _hits(pattern: re.Pattern, words) -> set[str]:
    return {w for w in words if pattern.search(w)}


def test_the_fixtures_describe_the_old_regex():
    """Control: the old pattern missed every WERE_FALSE phrase and every inflected
    form, and accepted every ACCEPTED_BEFORE phrase."""
    assert _hits(OLD_DECISION_RE, WERE_FALSE) == set()
    assert _hits(OLD_DECISION_RE, [w for ws in INFLECTED.values() for w in ws]) == set()
    assert _missed(OLD_DECISION_RE, ACCEPTED_BEFORE) == set()
    assert _hits(OLD_DECISION_RE, STAY_FALSE) == set()


@pytest.mark.parametrize("word", REQUIRED)
def test_each_required_word_matches(word):
    assert DECISION_RE.search(word)
    assert DECISION_RE.search(word.upper())          # IGNORECASE is kept
    assert DECISION_RE.search(f"ok so we {word} it")  # mid-sentence, not just whole text


@pytest.mark.parametrize("phrase", WERE_FALSE)
def test_the_formerly_missed_phrases_now_match(phrase):
    assert DECISION_RE.search(phrase)


def test_todays_accepted_phrases_stay_accepted():
    assert _missed(DECISION_RE, ACCEPTED_BEFORE) == set()


def test_a_stem_inside_a_longer_word_still_does_not_match():
    assert _hits(DECISION_RE, STAY_FALSE) == set()


# ─── through the product's entry points ───

@pytest.mark.parametrize("source,payload", [
    ("slack", {"text": "we decided to go with postgres"}),
    ("slack", {"text": "migrated the db last night"}),
    ("gmail", {"subject": "Deprecation notice for the v1 API", "snippet": "see below"}),
    ("gmail", {"subject": "adopted uv for the monorepo", "snippet": "details inside"}),
])
def test_a_slack_or_gmail_event_with_an_inflected_stem_is_durable(source, payload):
    ev = Event(id=f"{source}:x", source=source, author="a", payload=payload)
    assert source not in engine.DURABLE_SOURCES   # so the regex is what decides
    assert _is_durable(ev)


def test_slack_chatter_stays_non_durable():
    ev = Event(id="slack:c", source="slack", author="a",
               payload={"text": "thanks, see you tomorrow"})
    assert not _is_durable(ev)


def _seed(root, subject, snippet):
    append_events([Event(id="gmail:t1", source="gmail", author="x@acme.io",
                         payload={"subject": subject, "snippet": snippet},
                         acl=["customers"], entities=["acme.io"],
                         provenance={"threadId": "t1", "domain": "customers"})],
                  os.path.join(root, "events"))
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"distill": {"thorough": False}}, fh)   # durability gates the LLM


def _count_consolidate_calls(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(engine, "LLMClient",
                        lambda: types.SimpleNamespace(available=lambda: True))

    def fake(cl, cand, client):
        calls["n"] += 1
        return ConsolidationDecision(action="IGNORE")
    monkeypatch.setattr(engine, "consolidate", fake)
    return calls


def test_engine_run_sends_a_decided_gmail_cluster_to_the_llm(tmp_path, monkeypatch):
    root = str(tmp_path)
    _seed(root, "vendor choice", "we decided to go with the cheaper vendor")
    calls = _count_consolidate_calls(monkeypatch)
    guarded_run(engine, root=root, use_llm=True)
    assert calls["n"] == 1


def test_engine_run_control_a_chatty_gmail_cluster_skips_the_llm(tmp_path, monkeypatch):
    """Control for the test above: the harness can report 0, so the 1 is the regex."""
    root = str(tmp_path)
    _seed(root, "quick hello", "just saying hi")
    calls = _count_consolidate_calls(monkeypatch)
    guarded_run(engine, root=root, use_llm=True)
    assert calls["n"] == 0


# ─── kept mutations ───
# Each rebuilds DECISION_RE with one guard removed.  The anchor assertion makes a
# mutation that no longer applies a failure, not a silent no-op.

def _mutate(old: str, new: str) -> re.Pattern:
    src = DECISION_RE.pattern
    assert src.count(old) == 1, f"mutation anchor {old!r} not found once in DECISION_RE"
    return re.compile(src.replace(old, new), DECISION_RE.flags)


@pytest.mark.parametrize("stem", sorted(INFLECTED))
def test_kept_mutation_restoring_a_bare_stem_loses_its_inflections(stem):
    mutated = _mutate(stem + r"\w*", stem)
    assert _missed(mutated, REQUIRED) == set(INFLECTED[stem])
    # and the phrases that depend on that stem go False with it
    lost = _missed(mutated, WERE_FALSE)
    assert lost and all(stem in p.lower() for p in lost)


def test_kept_mutation_dropping_the_leading_boundary_admits_substrings():
    mutated = _mutate(r"\b(", "(")
    assert {"undecided", "indecision", "readopted", "remigrate"} <= _hits(mutated, STAY_FALSE)


def test_kept_mutation_engine_run_with_a_bare_decid_skips_the_llm(tmp_path, monkeypatch):
    """Kept mutation for the run test: with `decid` bare again, the same gmail
    cluster never reaches the LLM.  `_is_durable` reads the module global, so this goes
    through the real entry point."""
    monkeypatch.setattr(engine, "DECISION_RE", _mutate(r"decid\w*", "decid"))
    root = str(tmp_path)
    _seed(root, "vendor choice", "we decided to go with the cheaper vendor")
    calls = _count_consolidate_calls(monkeypatch)
    guarded_run(engine, root=root, use_llm=True)
    assert calls["n"] == 0
