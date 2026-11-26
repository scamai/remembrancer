"""Tests for Note.typed_links property."""
from remembrancerlib.note import Note


def test_typed_links_parse():
    n = Note(name="x", description="d", type="concept", domain="eng",
             body="Links: [[supersedes:old]] and [[plain]]")
    assert ("supersedes", "old") in n.typed_links
    assert ("relates", "plain") in n.typed_links


def test_typed_links_all_relations():
    n = Note(name="x", description="d", type="concept", domain="eng",
             body="[[supersedes:a]] [[contradicts:b]] [[extends:c]] [[relates:d]]")
    assert ("supersedes", "a") in n.typed_links
    assert ("contradicts", "b") in n.typed_links
    assert ("extends", "c") in n.typed_links
    assert ("relates", "d") in n.typed_links


def test_typed_links_plain_defaults_to_relates():
    n = Note(name="x", description="d", type="concept", domain="eng",
             body="[[some-note]]")
    assert n.typed_links == [("relates", "some-note")]


def test_typed_links_empty_body():
    n = Note(name="x", description="d", type="concept", domain="eng", body="")
    assert n.typed_links == []


def test_links_property_still_works():
    """The existing untyped links property must remain unchanged."""
    n = Note(name="x", description="d", type="concept", domain="eng",
             body="[[foo]] and [[supersedes:bar]]")
    # LINK_RE matches the whole [[...]] content including rel: prefix
    links = n.links
    assert "foo" in links
    # links property captures the raw inner content (including rel: prefix)
    assert any("bar" in lnk for lnk in links)


def test_typed_links_multiple_same_relation():
    n = Note(name="x", description="d", type="concept", domain="eng",
             body="[[supersedes:a]] [[supersedes:b]]")
    assert ("supersedes", "a") in n.typed_links
    assert ("supersedes", "b") in n.typed_links
    assert len(n.typed_links) == 2


def test_typed_links_accept_open_relation_vocabulary():
    n = Note(name="x", description="d", type="concept", domain="eng",
             body="[[supports:alpha]] [[related_to:beta]] [[relates-to:gamma]]")
    assert n.typed_links == [
        ("supports", "alpha"),
        ("related_to", "beta"),
        ("relates-to", "gamma"),
    ]
