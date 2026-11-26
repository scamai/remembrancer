"""Tests for resolve_links — OKF path link resolution."""
from remembrancerlib.note import resolve_links


def test_resolve_wiki_to_path():
    out = resolve_links("see [[other-note]] here", {"other-note": "/eng/other-note.md"})
    assert out == "see [other-note](/eng/other-note.md) here"


def test_resolve_unknown_stays_wiki():
    out = resolve_links("see [[unknown]] here", {"other-note": "/eng/other-note.md"})
    assert out == "see [[unknown]] here"


def test_resolve_multiple_links():
    name_to_path = {
        "alpha": "/eng/alpha.md",
        "beta": "/ml/beta.md",
    }
    out = resolve_links("[[alpha]] and [[beta]] and [[gamma]]", name_to_path)
    assert "[alpha](/eng/alpha.md)" in out
    assert "[beta](/ml/beta.md)" in out
    assert "[[gamma]]" in out


def test_resolve_typed_link_strips_rel_prefix():
    """[[rel:name]] in body should resolve to [name](path) stripping the rel: prefix."""
    out = resolve_links("see [[supersedes:old-note]] here", {"old-note": "/eng/old-note.md"})
    assert out == "see [old-note](/eng/old-note.md) here"


def test_resolve_typed_link_unknown():
    out = resolve_links("[[contradicts:missing]]", {})
    assert out == "[[contradicts:missing]]"


def test_resolve_empty_body():
    assert resolve_links("", {}) == ""


def test_resolve_no_links():
    assert resolve_links("plain text", {"a": "/a/a.md"}) == "plain text"
