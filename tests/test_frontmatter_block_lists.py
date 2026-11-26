"""Block-list frontmatter: `acl:` followed by `  - team-x` is YAML for
['team-x']. The parser used to skip the item lines, which left acl empty, and
an empty acl is public: a hand-written restricted note was readable by anyone."""
import json

from remembrancerlib import Note, load_notes, parse_frontmatter
from retrieve.query import search


def _fm(*lines):
    return parse_frontmatter("---\nname: x\n" + "\n".join(lines) + "\n---\nbody")[0]


def test_indented_block_list():
    assert _fm("acl:", "  - team-x")["acl"] == ["team-x"]


def test_unindented_block_list():
    # PyYAML's own dump style puts the items at the key's indentation.
    assert _fm("acl:", "- team-x", "- team-y")["acl"] == ["team-x", "team-y"]


def test_items_keep_colons_and_lose_quotes():
    fm = _fm("entities:", "  - customer:acme", "  - 'person:jane-doe'",
             '  - "team-x"', "  -", "  - ''")
    assert fm["entities"] == ["customer:acme", "person:jane-doe", "team-x"]


def test_blank_and_comment_lines_do_not_end_the_list():
    fm = _fm("acl:", "  # who may read this", "  - team-x", "", "  - team-y",
             "description: d")
    assert fm["acl"] == ["team-x", "team-y"]
    assert fm["description"] == "d"


def test_the_next_key_ends_the_list():
    fm = _fm("acl:", "  - team-x", "tags: [a]", "  - stray")
    assert fm["acl"] == ["team-x"]
    assert fm["tags"] == ["a"]


def test_forms_without_a_block_list_parse_as_before():
    fm = _fm("acl:", "superseded:", "description: d")
    assert fm["acl"] == "" and fm["superseded"] == ""
    # An item after a key that has a value is not a list item.
    assert _fm("acl: team-x", "  - team-y")["acl"] == "team-x"
    # A nested key opens no list: nested maps stay unsupported.
    fm = _fm("structured:", "  kind:", "    - a")
    assert fm["structured"] == "" and fm["kind"] == ""


def _block_acl_note(context_dir):
    folder = context_dir / "company"
    folder.mkdir(parents=True)
    (folder / "ledger-plan.md").write_text(
        "---\nname: ledger-plan\ndescription: Ledger migration plan\n"
        "type: project\nacl:\n  - team-x\nupdated: 2026-11-11\n---\n"
        "The ledger migration plan.\n")


def test_a_loaded_note_keeps_the_block_acl_and_writes_it_inline(tmp_path):
    _block_acl_note(tmp_path / "context")
    [note] = load_notes(str(tmp_path / "context"))
    assert note.acl == ["team-x"]
    assert "\nacl: [team-x]\n" in note.to_markdown()
    assert isinstance(Note.from_frontmatter(*parse_frontmatter(note.to_markdown())).acl, list)


def test_search_shows_a_block_acl_note_only_to_its_label(tmp_path, monkeypatch):
    monkeypatch.delenv("REMEMBRANCER_ALLOW_FULL_ACCESS", raising=False)
    (tmp_path / "config.json").write_text(json.dumps(
        {"identities": {"alice": ["team-x"], "bob": ["team-y"]}}))
    _block_acl_note(tmp_path / "context")

    def names(identity):
        return {hit["name"] for hit in search("ledger", identity=identity, root=str(tmp_path))}

    assert names("alice") == {"ledger-plan"}
    assert names("bob") == set()
    assert names(None) == set()
