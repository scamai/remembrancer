"""Regression test: the human-curated `noise.review` bucket must actually drive
type resolution.

For a long time `RegistryTypes` loaded only `noise.free_email/infra/edu_suffix` and
IGNORED `noise.review`, the bucket that holds every human "this is noise" verdict.
The result: a reviewer could mark a cold-email
domain as noise and the distiller would keep minting a note from its next email. This test
locks in that `review` entries resolve to "noise", while real registered entities still win.
"""
import textwrap

from distill.entities import RegistryTypes


def _registry(tmp_path):
    p = tmp_path / "entities.yml"
    p.write_text(textwrap.dedent("""
        vendor:
          - name: Initech
            email_domains: [initech.example]
        noise:
          free_email: [gmail.com]
          infra: [google.com]
          edu_suffix: [.edu]
          review:
            - spammydomain.info
            - coldpitch@outreach.example
            - Some Cold Agency
            - initech.example          # accidental overlap with a REAL vendor -> must NOT win
    """))
    return RegistryTypes(str(p))


def test_review_domain_resolves_to_noise(tmp_path):
    rt = _registry(tmp_path)
    assert rt.type_for("spammydomain.info") == "noise"
    assert rt.type_for("anyone@spammydomain.info") == "noise"


def test_review_email_and_name_resolve_to_noise(tmp_path):
    rt = _registry(tmp_path)
    assert rt.type_for("coldpitch@outreach.example") == "noise"
    assert rt.type_for("Some Cold Agency") == "noise"


def test_real_entity_beats_review_overlap(tmp_path):
    # initech.example is both a registered vendor and (accidentally) in review — vendor wins.
    rt = _registry(tmp_path)
    assert rt.type_for("initech.example") == "vendor"
    assert rt.type_for("billing@initech.example") == "vendor"


def test_unknown_still_none(tmp_path):
    rt = _registry(tmp_path)
    assert rt.type_for("never-heard-of-this.com") is None
