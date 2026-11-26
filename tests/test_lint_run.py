"""The run() orchestrator ties all lint checks together."""
import os
import subprocess
import sys
import json
from remembrancerlib import Note, write_note, load_notes
from lint.engine import run
from tests.trust_helpers import write_reviewed_note


def make_note(name, body="", domain="eng", note_type="concept", entities=None,
              superseded=False, state=None):
    return Note(
        name=name,
        description="test note",
        type=note_type,
        domain=domain,
        body=body,
        entities=entities or [],
        superseded=superseded,
        state=state,
    )


class TestRunOrchestrator:
    def test_returns_summary_dict_with_required_keys(self, tmp_path):
        """run() returns dict with orphans, superseded, contradictions, gaps keys."""
        a = make_note("a")
        write_note(a, os.path.join(str(tmp_path), "context"))

        summary = run(root=str(tmp_path))

        assert "orphans" in summary
        assert "superseded" in summary
        assert "contradictions" in summary
        assert "gaps" in summary

    def test_cli_writes_machine_readable_file_back_summary(self, tmp_path):
        ctx = str(tmp_path / "context")
        write_note(make_note("source", body="[[target]]"), ctx)
        write_note(make_note("target"), ctx)
        output = tmp_path / "state" / "lint.json"
        completed = subprocess.run(
            [sys.executable, "-m", "lint.engine", "--root", str(tmp_path),
             "--max-file-back-writes", "0", "--json-summary", str(output)],
            cwd=os.path.dirname(os.path.dirname(__file__)), text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(output.read_text())
        assert payload["file_back"]["deferred"] == 1
        assert payload["file_back"]["deferred_links"] == 1

    def _legacy_dossier_corpus(self, tmp_path, dossier_body=""):
        """A pre-partition dossier whose sources have since crossed audiences."""
        ctx = tmp_path / "context"
        notes = [
            Note(
                name="entity-acme", description="Acme dossier", type="entity",
                domain="customers", body=dossier_body, entities=["org:acme"],
                acl=["customers"],
            ),
            Note(
                name="thread-customer", description="Acme customer thread",
                type="reference", domain="customers", body="",
                entities=["org:acme"], acl=["customers"],
            ),
            Note(
                name="thread-eng", description="Acme engineering thread",
                type="reference", domain="eng", body="",
                entities=["org:acme"], acl=["eng"],
            ),
        ]
        if dossier_body:
            # write_note refuses an over-exposing link from either direction, so
            # this shape can only reach the corpus as pre-existing files.
            for note in notes:
                domain_dir = ctx / note.domain
                domain_dir.mkdir(parents=True, exist_ok=True)
                (domain_dir / f"{note.name}.md").write_text(note.to_markdown())
        else:
            for note in notes:
                write_note(note, str(ctx))
        return tmp_path / "state" / "lint.json"

    def _run_lint_cli(self, tmp_path, output):
        return subprocess.run(
            [sys.executable, "-m", "lint.engine", "--root", str(tmp_path),
             "--fail-on-unsafe-links", "--max-file-back-writes", "0",
             "--json-summary", str(output)],
            cwd=os.path.dirname(os.path.dirname(__file__)), text=True,
            capture_output=True,
        )

    def test_cli_defers_a_legacy_dossier_acl_conflict_instead_of_refusing(self, tmp_path):
        """One unfixable page must not freeze every unrelated memory update.

        Nothing in the pipeline rewrites an unmarked pre-partition page, so
        blocking publication on it blocked the whole corpus indefinitely.
        """
        output = self._legacy_dossier_corpus(tmp_path)

        completed = self._run_lint_cli(tmp_path, output)

        assert completed.returncode == 0, completed.stdout + completed.stderr
        payload = json.loads(output.read_text())
        assert [c["name"] for c in payload["dossier_acl_deferred"]] == ["entity-acme"]
        assert payload["dossier_acl_blocking"] == []
        assert "Dossier ACL deferred to attended repair (1)" in completed.stdout

    def test_cli_still_refuses_when_the_legacy_dossier_already_reveals_a_source(
            self, tmp_path):
        """Deferring the split does not defer over-exposure that already exists."""
        output = self._legacy_dossier_corpus(
            tmp_path, dossier_body="- 2026-01-15 · [[thread-eng]]\n",
        )

        completed = self._run_lint_cli(tmp_path, output)

        assert completed.returncode == 1, completed.stdout + completed.stderr
        payload = json.loads(output.read_text())
        assert payload["unsafe_forward_links"] == [
            ["entity-acme", "relates", "thread-eng"],
        ]

    def test_orphan_detected_in_run(self, tmp_path):
        """An isolated note is detected as an orphan by run()."""
        ctx = str(tmp_path / "context")
        a = make_note("a")
        b = make_note("b", body="[[a]]")  # a is referenced, b is orphan
        write_note(a, ctx)
        write_note(b, ctx)

        summary = run(root=str(tmp_path))

        assert "b" in summary["orphans"]
        assert "a" not in summary["orphans"]

    def test_active_supersession_reported_without_materialization(self, tmp_path):
        """run() reports a verified edge but does not rewrite its target."""
        ctx = str(tmp_path / "context")
        old = make_note("old-approach", state="verified")
        new = make_note(
            "new-approach", body="[[supersedes:old-approach]]", state="verified"
        )
        write_reviewed_note(tmp_path, old)
        write_reviewed_note(tmp_path, new)

        summary = run(root=str(tmp_path))

        assert "old-approach" in summary["superseded"]
        # The lifecycle is projected at retrieval time, never materialized.
        fresh = load_notes(ctx)
        old_note = next(n for n in fresh if n.name == "old-approach")
        assert old_note.superseded is False
        assert old_note.state == "verified"

    def test_contradiction_detected_in_run(self, tmp_path):
        """run() detects contradictions and creates contradiction notes."""
        ctx = str(tmp_path / "context")
        a = make_note("a", body="[[contradicts:b]]")
        b = make_note("b")
        write_note(a, ctx)
        write_note(b, ctx)

        summary = run(root=str(tmp_path))

        assert len(summary["contradictions"]) >= 1
        # Check contradiction note was written
        fresh = load_notes(ctx)
        c_notes = [n for n in fresh if n.type == "contradiction"]
        assert len(c_notes) >= 1

    def test_gap_detected_in_run(self, tmp_path):
        """run() detects missing entity pages and includes them in gaps."""
        ctx = str(tmp_path / "context")
        note = make_note("some-note", entities=["acme-corp"])
        write_note(note, ctx)

        summary = run(root=str(tmp_path))

        assert any("acme-corp" in g for g in summary["gaps"])

    def test_lint_report_written_by_run(self, tmp_path):
        """run() writes context/_lint_report.md."""
        ctx = str(tmp_path / "context")
        note = make_note("lone-note")
        write_note(note, ctx)

        run(root=str(tmp_path))

        report_path = os.path.join(ctx, "_lint_report.md")
        assert os.path.isfile(report_path)

    def test_index_regenerated_by_run(self, tmp_path):
        """run() regenerates INDEX.md."""
        ctx = str(tmp_path / "context")
        a = make_note("a")
        write_note(a, ctx)

        run(root=str(tmp_path))

        index_path = os.path.join(ctx, "INDEX.md")
        assert os.path.isfile(index_path)

    def test_run_idempotent(self, tmp_path):
        """Running run() twice on the same data produces stable contradiction + gap results.

        Orphans may grow on a second run if auto-generated notes (contradiction notes)
        are themselves orphans — that is expected.  Contradictions and gaps are stable.
        """
        ctx = str(tmp_path / "context")
        a = make_note("a", body="[[contradicts:b]]")
        b = make_note("b")
        c = make_note("c")  # orphan
        write_note(a, ctx)
        write_note(b, ctx)
        write_note(c, ctx)

        summary1 = run(root=str(tmp_path))
        summary2 = run(root=str(tmp_path))

        # Contradictions and gaps are fully stable across runs
        assert summary1["contradictions"] == summary2["contradictions"]
        assert summary1["gaps"] == summary2["gaps"]
        # The report is a stable projection (this fixture has no active edge).
        assert summary2["superseded"] == []

    def test_dangling_in_summary(self, tmp_path):
        """run() also returns dangling links in the summary."""
        ctx = str(tmp_path / "context")
        a = make_note("a", body="[[ghost-note]]")
        write_note(a, ctx)

        summary = run(root=str(tmp_path))

        # dangling should be present (may or may not be a top-level key)
        assert "dangling" in summary
        assert ("a", "ghost-note") in summary["dangling"]

    def test_second_run_orphans_exclude_contradiction_notes(self, tmp_path):
        """After run() creates a contradiction note, a second run() orphans list does NOT contain it."""
        ctx = str(tmp_path / "context")
        a = make_note("a", body="[[contradicts:b]]")
        b = make_note("b")
        write_note(a, ctx)
        write_note(b, ctx)

        # First run creates the contradiction note
        run(root=str(tmp_path))

        # Second run: contradiction note must NOT appear in orphans
        summary2 = run(root=str(tmp_path))
        contradiction_orphans = [name for name in summary2["orphans"] if "contradiction" in name]
        assert contradiction_orphans == [], (
            f"Second run orphans should not include contradiction notes, got: {contradiction_orphans}"
        )

    def test_cli_runs_without_error(self, tmp_path):
        """python3 -m lint.engine --root <root> exits 0."""
        import pathlib
        ctx = str(tmp_path / "context")
        a = make_note("a")
        write_note(a, ctx)

        # Locate the worktree root so the subprocess can import lint + remembrancerlib
        worktree = str(pathlib.Path(__file__).parent.parent)
        env = {**os.environ, "PYTHONPATH": worktree}

        result = subprocess.run(
            [sys.executable, "-m", "lint.engine", "--root", str(tmp_path)],
            capture_output=True,
            text=True,
            cwd=worktree,
            env=env,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "Lint Summary" in result.stdout
