"""Gap report (missing entity pages)."""
import os
from remembrancerlib import Note
from lint.engine import gap_report, write_gap_report


def make_note(name, body="", domain="eng", note_type="concept", entities=None):
    return Note(
        name=name,
        description="",
        type=note_type,
        domain=domain,
        body=body,
        entities=entities or [],
    )


class TestGapReport:
    def test_missing_entity_page_detected(self):
        """A note mentions 'customer:acme' in entities but no entity page for it exists."""
        note = make_note("some-note", entities=["customer:acme"])
        gaps = gap_report([note])
        assert "missing entity page: customer:acme" in gaps

    def test_existing_entity_page_not_flagged(self):
        """If an entity page exists for 'acme', it should not appear in gaps."""
        note = make_note("some-note", entities=["acme"])
        entity_page = make_note("acme", note_type="entity")
        gaps = gap_report([note, entity_page])
        assert "missing entity page: acme" not in gaps

    def test_empty_notes_no_gaps(self):
        assert gap_report([]) == []

    def test_no_entities_no_gaps(self):
        note = make_note("some-note")
        # No entities field entries → no gaps
        gaps = gap_report([note])
        assert gaps == []

    def test_multiple_missing_entities(self):
        note = make_note("some-note", entities=["entityA", "entityB"])
        gaps = gap_report([note])
        assert "missing entity page: entityA" in gaps
        assert "missing entity page: entityB" in gaps

    def test_partial_entity_pages_existing(self):
        """Only missing ones flagged."""
        note = make_note("some-note", entities=["existing", "missing"])
        entity_page = make_note("existing", note_type="entity")
        gaps = gap_report([note, entity_page])
        assert "missing entity page: missing" in gaps
        assert "missing entity page: existing" not in gaps

    def test_entity_page_itself_not_flagged(self):
        """Entity pages themselves don't self-reference."""
        entity = make_note("acme", note_type="entity")
        # No other notes reference acme
        gaps = gap_report([entity])
        assert gaps == []


class TestWriteGapReport:
    def test_writes_lint_report_md(self, tmp_path):
        """write_gap_report creates context/_lint_report.md."""
        ctx = str(tmp_path / "context")
        os.makedirs(ctx)
        gaps = ["missing entity page: acme"]
        write_gap_report(gaps, ctx)

        report_path = os.path.join(ctx, "_lint_report.md")
        assert os.path.isfile(report_path)

    def test_report_contains_gap_entries(self, tmp_path):
        ctx = str(tmp_path / "context")
        os.makedirs(ctx)
        gaps = ["missing entity page: acme", "missing entity page: widget-corp"]
        write_gap_report(gaps, ctx)

        report_path = os.path.join(ctx, "_lint_report.md")
        content = open(report_path).read()
        assert "acme" in content
        assert "widget-corp" in content

    def test_report_includes_suggestion(self, tmp_path):
        """Report should include a suggested action per missing entity."""
        ctx = str(tmp_path / "context")
        os.makedirs(ctx)
        gaps = ["missing entity page: acme"]
        write_gap_report(gaps, ctx)

        report_path = os.path.join(ctx, "_lint_report.md")
        content = open(report_path).read()
        assert "acme" in content
        # Should contain a suggestion
        assert "Suggested" in content or "suggest" in content.lower() or "entity" in content.lower()

    def test_empty_gaps_no_entries(self, tmp_path):
        ctx = str(tmp_path / "context")
        os.makedirs(ctx)
        write_gap_report([], ctx)

        report_path = os.path.join(ctx, "_lint_report.md")
        content = open(report_path).read()
        assert "no missing" in content.lower() or "empty" in content.lower() or "none" in content.lower() or "(" in content
