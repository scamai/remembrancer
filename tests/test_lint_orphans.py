"""Orphan detection and dangling-link detection."""
from remembrancerlib import Note
from lint.engine import find_orphans, find_dangling


def make_note(name, body="", domain="eng", note_type="concept"):
    return Note(name=name, description="", type=note_type, domain=domain, body=body)


class TestFindOrphans:
    def test_orphan_detected_when_no_inbound_links(self):
        """c has no inbound links from anyone — it is an orphan."""
        a = make_note("a", body="[[b]]")
        b = make_note("b", body="")
        c = make_note("c", body="")  # orphan
        orphans = find_orphans([a, b, c])
        assert "c" in orphans

    def test_linked_note_not_orphan(self):
        """b is referenced by a — not an orphan."""
        a = make_note("a", body="[[b]]")
        b = make_note("b", body="")
        c = make_note("c", body="")
        orphans = find_orphans([a, b, c])
        assert "b" not in orphans

    def test_single_note_is_orphan(self):
        """A lone note with no links in or out is an orphan."""
        a = make_note("a", body="")
        assert "a" in find_orphans([a])

    def test_empty_notes_returns_empty(self):
        assert find_orphans([]) == []

    def test_typed_link_also_counts_as_inbound(self):
        """[[supersedes:old]] counts as a link to 'old', so old is not an orphan."""
        new = make_note("new", body="[[supersedes:old]]")
        old = make_note("old", body="")
        orphans = find_orphans([new, old])
        assert "old" not in orphans

    def test_orphan_from_typed_link_still_detected(self):
        """Note 'x' that is never targeted by anyone is still orphan even if it links outward."""
        x = make_note("x", body="[[supersedes:y]]")
        y = make_note("y", body="")
        orphans = find_orphans([x, y])
        assert "x" in orphans  # x has no inbound links

    def test_mutual_link_neither_orphan(self):
        a = make_note("a", body="[[b]]")
        b = make_note("b", body="[[a]]")
        orphans = find_orphans([a, b])
        assert "a" not in orphans
        assert "b" not in orphans


class TestFindDangling:
    def test_dangling_link_detected(self):
        """a links to nonexistent 'ghost' — dangling."""
        a = make_note("a", body="[[ghost]]")
        b = make_note("b", body="")
        dangling = find_dangling([a, b])
        assert ("a", "ghost") in dangling

    def test_valid_link_not_dangling(self):
        a = make_note("a", body="[[b]]")
        b = make_note("b", body="")
        dangling = find_dangling([a, b])
        assert ("a", "b") not in dangling

    def test_typed_dangling_link(self):
        """[[supersedes:missing]] where 'missing' doesn't exist."""
        a = make_note("a", body="[[supersedes:missing]]")
        dangling = find_dangling([a])
        assert ("a", "missing") in dangling

    def test_empty_notes_returns_empty(self):
        assert find_dangling([]) == []

    def test_multiple_dangling_links(self):
        a = make_note("a", body="[[x]] and [[y]]")
        dangling = find_dangling([a])
        assert ("a", "x") in dangling
        assert ("a", "y") in dangling

    def test_no_duplicate_pairs_when_same_missing_target_linked_twice(self):
        """A note that links the same missing target twice produces only one pair."""
        a = make_note("a", body="[[ghost]] and also [[ghost]]")
        dangling = find_dangling([a])
        assert dangling.count(("a", "ghost")) == 1

    def test_dangling_returned_sorted(self):
        """find_dangling returns pairs in sorted order."""
        a = make_note("a", body="[[z-missing]] and [[a-missing]]")
        dangling = find_dangling([a])
        assert dangling == sorted(dangling)


class TestFindOrphansContradictionExclusion:
    def test_contradiction_note_not_in_orphans(self):
        """Notes with type='contradiction' are excluded from orphan detection."""
        real = make_note("real-note")
        c_note = make_note("contradiction-a-vs-b", note_type="contradiction")
        orphans = find_orphans([real, c_note])
        assert "contradiction-a-vs-b" not in orphans
        assert "real-note" in orphans  # still orphan (nothing links to it)

    def test_contradiction_exclusion_does_not_affect_concept_orphans(self):
        """Regular notes that are orphans are still detected after filtering contradictions."""
        orphan = make_note("lonely")
        linked = make_note("linked", body="[[target]]")
        target = make_note("target")
        c_note = make_note("contradiction-x-vs-y", note_type="contradiction")
        orphans = find_orphans([orphan, linked, target, c_note])
        assert "lonely" in orphans
        assert "linked" in orphans  # linked has no inbound links
        assert "target" not in orphans  # target is referenced by linked
        assert "contradiction-x-vs-y" not in orphans
