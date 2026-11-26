"""Dossier re-verification must not decay as an entity accumulates notes.

Failure this pins: `dossier_profile_prompt_material` caps the
prompt window (limit_chars) across ALL contributor notes. A dossier's claim is
grounded against that window at GENERATION time, but lint re-derives the window
later — by which point the entity has more notes, so a *different* subset
survives the cap. If the cited source falls outside the new window, the claim is
ruled ungrounded, lint reports a dossier conflict, and the pipeline refuses to
publish. A dossier with many contributors can deadlock the whole distill
pipeline this way: every pass redoes the same work and discards it, and the
backlog stops draining.

The anti-hallucination property must be PRESERVED: per-note truncation still
applies, so tail content the model never received still cannot be cited. Only
the cross-note budget eviction is prevented from invalidating an already-
persisted claim — the generator enforces the strict window at write time, so a
claim citing a never-seen note could not have been persisted in the first place.
"""
from remembrancerlib.dossier import (dossier_profile_prompt_material,
                              normalize_dossier_profile_claims)


class _N:
    def __init__(self, name, description="", body=""):
        self.name = name
        self.description = description
        self.body = body


CITED = "onboarding-checklist-review"
# A "who" claim must name its subject to pass
# dossier_profile_field_supported(), so the subject label is part of the quote.
QUOTE = "Onboarding checklist review and hand-off (Jane Doe)"


def _crowd(n):
    """Filler notes big enough to exhaust the prompt budget."""
    return [_N(f"filler-{i}", body="x" * 1400) for i in range(n)]


def test_cited_source_survives_a_crowded_window():
    """The cited note must get an excerpt even when the budget is exhausted."""
    notes = _crowd(20) + [_N(CITED, body=f"Jane Doe. {QUOTE} continues here.")]

    _p, without, _i = dossier_profile_prompt_material(notes)
    assert CITED not in without, "precondition: budget should evict the cited note"

    _p, with_req, _i = dossier_profile_prompt_material(notes, require_names=[CITED])
    assert CITED in with_req, "required source must be present despite the cap"


def test_claim_reverifies_when_source_is_required():
    notes = _crowd(20) + [_N(CITED, body=f"Jane Doe. {QUOTE} continues here.")]
    raw = {"who": {"source_note": CITED, "quote": QUOTE}}

    _p, evicted, _i = dossier_profile_prompt_material(notes)
    assert normalize_dossier_profile_claims(raw, evicted, ["Jane Doe"]) != raw

    _p, kept, _i = dossier_profile_prompt_material(notes, require_names=[CITED])
    assert normalize_dossier_profile_claims(raw, kept, ["Jane Doe"]) == raw


def test_per_note_truncation_still_applies():
    """Anti-hallucination control preserved: tail content stays uncitable."""
    tail = "SECRET TAIL CONTENT THAT WAS NEVER SHOWN"
    notes = [_N(CITED, body=("y" * 1600) + tail)]
    _p, exc, incomplete = dossier_profile_prompt_material(notes, require_names=[CITED])
    assert incomplete is True
    assert tail not in exc[CITED], "required inclusion must NOT bypass per-note truncation"


def test_generation_path_is_unchanged():
    """Default (no require_names) must behave exactly as before."""
    notes = _crowd(20) + [_N(CITED, body="something")]
    a = dossier_profile_prompt_material(notes)
    b = dossier_profile_prompt_material(notes, require_names=[])
    assert a == b


def test_unknown_required_name_is_harmless():
    notes = [_N(CITED, body="a body long enough to matter here")]
    _p, exc, _i = dossier_profile_prompt_material(notes, require_names=["does-not-exist"])
    assert CITED in exc
