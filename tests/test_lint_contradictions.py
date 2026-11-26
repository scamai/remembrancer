"""Contradiction detection and contradiction note creation."""
import os
import pytest
from remembrancerlib import ACLConflict, Note, write_note, load_notes
from lint.engine import find_contradictions, write_contradiction_notes, run
from retrieve import query as Q
from retrieve.verification import load_verification_audit
from tests.trust_helpers import write_reviewed_note


def make_note(name, body="", domain="eng", description="", entities=None, acl=None):
    return Note(name=name, description=description, type="concept", domain=domain,
                body=body, entities=entities or [], acl=acl or [])


def pair_notes(pairs, **kwargs):
    return [make_note(name, description=f"source {name}", **kwargs)
            for name in sorted({name for pair in pairs for name in pair})]


class TestFindContradictions:
    def test_contradicts_edge_returns_pair(self):
        """A [[contradicts:B]] → find_contradictions returns ("a","b")."""
        a = make_note("a", body="[[contradicts:b]]")
        b = make_note("b")
        pairs = find_contradictions([a, b])
        assert ("a", "b") in pairs

    def test_pair_normalised_lexicographically(self):
        """Pair is always (smaller, larger) lexicographically."""
        z = make_note("z", body="[[contradicts:a]]")
        a = make_note("a")
        pairs = find_contradictions([z, a])
        # "a" < "z", so pair should be ("a", "z")
        assert ("a", "z") in pairs

    def test_duplicate_edges_deduplicated(self):
        """If both A contradicts B and B contradicts A, only one pair returned."""
        a = make_note("a", body="[[contradicts:b]]")
        b = make_note("b", body="[[contradicts:a]]")
        pairs = find_contradictions([a, b])
        assert len(pairs) == 1
        assert ("a", "b") in pairs

    def test_no_contradiction_returns_empty(self):
        a = make_note("a", body="[[b]]")
        b = make_note("b")
        assert find_contradictions([a, b]) == []

    def test_empty_notes_returns_empty(self):
        assert find_contradictions([]) == []

    def test_non_contradicts_edge_ignored(self):
        a = make_note("a", body="[[supersedes:b]]")
        b = make_note("b")
        assert find_contradictions([a, b]) == []

    def test_same_entity_antonym_descriptions_are_detected(self):
        a = make_note("a", description="Acme API is enabled", entities=["org:acme"])
        b = make_note("b", description="Acme API is disabled", entities=["org:acme"])
        assert find_contradictions([a, b]) == [("a", "b")]

    def test_antonyms_without_same_entity_are_not_detected(self):
        a = make_note("a", description="Acme API is enabled", entities=["org:acme"])
        b = make_note("b", description="Beta API is disabled", entities=["org:beta"])
        assert find_contradictions([a, b]) == []

    def test_different_subjects_on_same_entity_are_not_detected(self):
        a = make_note("a", description="Acme API is enabled", entities=["org:acme"])
        b = make_note("b", description="Acme dashboard is disabled", entities=["org:acme"])
        assert find_contradictions([a, b]) == []

    def test_transition_description_with_both_polarities_is_not_detected(self):
        a = make_note("a", description="Acme API changed from disabled to enabled",
                      entities=["org:acme"])
        b = make_note("b", description="Acme API is disabled", entities=["org:acme"])
        assert find_contradictions([a, b]) == []


class TestWriteContradictionNotes:
    def test_closed_validator_replays_authenticated_sources_and_dual_binds_success(
        self, tmp_path,
    ):
        ctx = str(tmp_path / "context")
        alpha = make_note(
            "alpha", body="[[contradicts:beta]]", description="source alpha",
        )
        beta = make_note("beta", description="source beta")
        write_reviewed_note(tmp_path, alpha, state="verified")
        write_reviewed_note(tmp_path, beta, state="verified")
        sources = {
            note.name: note for note in load_notes(ctx)
            if note.name in {"alpha", "beta"}
        }

        written = write_contradiction_notes(
            [("alpha", "beta")], ctx,
            notes=[sources["alpha"], sources["beta"]],
        )

        contradiction = next(
            note for note in load_notes(ctx) if note.name == written[0]
        )
        verification = contradiction.confidence["verification"]
        audit = load_verification_audit(str(tmp_path))
        assert verification["status"] == "verified"
        assert verification["producer"] == "contradiction-lint"
        assert audit.statuses[contradiction.name] == "verified"
        assert audit.audit_ids[contradiction.name] == verification["audit_id"]
        assert audit.content_fingerprints[contradiction.name] == \
            verification["content_fingerprint"]
        from remembrancerlib.verification_audit import head_matches_audit, parse_heads
        audit_bytes = (tmp_path / "state" / "verification.jsonl").read_bytes()
        heads, trustworthy = parse_heads(
            (tmp_path / "state" / "verification.head.jsonl").read_bytes()
        )
        assert trustworthy and heads and head_matches_audit(heads[-1], audit_bytes)
        assert {row["name"] for row in Q.search(
            "contradiction", root=str(tmp_path),
        )} >= {contradiction.name}

        # The frontmatter hash alone is not authority. Removing the audit-side
        # binding immediately withdraws the machine grant.
        (tmp_path / "state" / "verification.jsonl").unlink()
        assert Q.get_note(contradiction.name, root=str(tmp_path)) is None

    def test_closed_success_is_invalidated_when_content_changes(self, tmp_path):
        ctx = str(tmp_path / "context")
        alpha = make_note(
            "alpha", body="[[contradicts:beta]]", description="source alpha",
        )
        beta = make_note("beta", description="source beta")
        write_reviewed_note(tmp_path, alpha, state="verified")
        write_reviewed_note(tmp_path, beta, state="verified")
        sources = {note.name: note for note in load_notes(ctx)}
        written = write_contradiction_notes(
            [("alpha", "beta")], ctx,
            notes=[sources["alpha"], sources["beta"]],
        )
        contradiction = next(
            note for note in load_notes(ctx) if note.name == written[0]
        )

        contradiction.body += "\n\nFabricated extra claim."
        write_note(contradiction, ctx)

        changed = next(
            note for note in load_notes(ctx) if note.name == contradiction.name
        )
        assert changed.confidence["verification"]["status"] == "skipped"
        assert Q.get_note(changed.name, root=str(tmp_path)) is None

    def test_closed_validator_refuses_unapproved_sources_without_success_audit(
        self, tmp_path,
    ):
        ctx = str(tmp_path / "context")
        alpha = make_note(
            "alpha", body="[[contradicts:beta]]", description="source alpha",
        )
        beta = make_note("beta", description="source beta")
        write_note(alpha, ctx)
        write_note(beta, ctx)
        sources = {note.name: note for note in load_notes(ctx)}

        written = write_contradiction_notes(
            [("alpha", "beta")], ctx,
            notes=[sources["alpha"], sources["beta"]],
        )

        contradiction = next(
            note for note in load_notes(ctx) if note.name == written[0]
        )
        assert contradiction.confidence["verification"]["status"] == "skipped"
        audit = load_verification_audit(str(tmp_path))
        assert contradiction.name not in audit.statuses

    def test_closed_validator_rolls_back_partial_authority_audit(
        self, tmp_path, monkeypatch,
    ):
        ctx = str(tmp_path / "context")
        alpha = make_note(
            "alpha", body="[[contradicts:beta]]", description="source alpha",
        )
        beta = make_note("beta", description="source beta")
        write_reviewed_note(tmp_path, alpha, state="verified")
        write_reviewed_note(tmp_path, beta, state="verified")
        sources = {note.name: note for note in load_notes(ctx)}
        from distill.verify import _append_audit
        audit_path = tmp_path / "state" / "verification.jsonl"
        assert _append_audit(
            {"note": "seed", "status": "unverified"},
            path=str(audit_path),
        )
        before_audit = audit_path.read_bytes()
        head_path = tmp_path / "state" / "verification.head.jsonl"
        before_head = head_path.read_bytes()
        import remembrancerlib.verification as verification_module
        real_write = verification_module.os.write
        writes = 0

        def interrupted_write(descriptor, payload):
            nonlocal writes
            writes += 1
            if writes == 1:
                return real_write(descriptor, payload[:max(1, len(payload) // 2)])
            raise OSError("simulated producer authority crash")

        monkeypatch.setattr(verification_module.os, "write", interrupted_write)
        with pytest.raises(OSError, match="audit/head append failed"):
            write_contradiction_notes(
                [("alpha", "beta")], ctx,
                notes=[sources["alpha"], sources["beta"]],
            )

        assert audit_path.read_bytes() == before_audit
        assert head_path.read_bytes() == before_head
        assert load_verification_audit(str(tmp_path)).trustworthy is True
        assert not any(
            note.type == "contradiction" for note in load_notes(ctx)
        )

    def test_writes_contradiction_note_for_pair(self, tmp_path):
        """A contradiction note file is created when policy=flag."""
        ctx = str(tmp_path / "context")
        pairs = [("alpha", "beta")]
        written = write_contradiction_notes(pairs, ctx, notes=pair_notes(pairs))

        assert len(written) == 1
        # The note file should exist somewhere under context/
        # It'll be in company/ domain
        note_name = written[0]
        from remembrancerlib import slugify
        note_path = os.path.join(ctx, "company", slugify(note_name) + ".md")
        assert os.path.isfile(note_path), f"Expected file at {note_path}"

    def test_contradiction_note_has_correct_type(self, tmp_path):
        """Written contradiction note has type: contradiction in frontmatter."""
        ctx = str(tmp_path / "context")
        pairs = [("alpha", "beta")]
        write_contradiction_notes(pairs, ctx, notes=pair_notes(pairs))

        notes = load_notes(ctx)
        contradiction_notes = [n for n in notes if n.type == "contradiction"]
        assert len(contradiction_notes) >= 1

    def test_contradiction_note_body_references_both(self, tmp_path):
        """Contradiction note body contains links to both notes."""
        ctx = str(tmp_path / "context")
        pairs = [("alpha", "beta")]
        write_contradiction_notes(pairs, ctx, notes=pair_notes(pairs))

        notes = load_notes(ctx)
        c_note = next(n for n in notes if n.type == "contradiction")
        assert "alpha" in c_note.body
        assert "beta" in c_note.body

    def test_idempotent_rewrite(self, tmp_path):
        """Calling twice with same pairs doesn't create duplicate files."""
        ctx = str(tmp_path / "context")
        pairs = [("alpha", "beta")]
        sources = pair_notes(pairs)
        sources[0].body = "[[contradicts:beta]]"
        for source in sources:
            write_note(source, ctx)
        current = {note.name: note for note in load_notes(ctx)}
        sources = [current[source.name] for source in sources]
        write_contradiction_notes(pairs, ctx, notes=sources)
        write_contradiction_notes(pairs, ctx, notes=sources)

        notes = load_notes(ctx)
        contradiction_notes = [n for n in notes if n.type == "contradiction"]
        assert len(contradiction_notes) == 1

    def test_reconcile_policy_falls_back_to_flag_without_evidence(self, tmp_path):
        """Uncertain reconcile pairs remain visible as contradiction notes."""
        # Write a config.json with reconcile policy in the parent of context_dir
        cfg_path = tmp_path / "config.json"
        import json
        cfg_path.write_text(json.dumps({"lint": {"contradiction_policy": "reconcile"}}))

        ctx = str(tmp_path / "context")
        pairs = [("alpha", "beta")]
        written = write_contradiction_notes(pairs, ctx, notes=pair_notes(pairs))
        assert written == ["contradiction-alpha-vs-beta"]

    def test_empty_pairs_writes_nothing(self, tmp_path):
        ctx = str(tmp_path / "context")
        written = write_contradiction_notes([], ctx)
        assert written == []

    def test_policy_param_overrides_config(self, tmp_path):
        """policy kwarg overrides the internal config.json lookup entirely."""
        # No config.json — default would be 'flag' (writes notes).
        ctx = str(tmp_path / "context")
        pairs = [("alpha", "beta")]
        # Explicit reconcile still falls back to a flag when no source evidence exists.
        written = write_contradiction_notes(
            pairs, ctx, policy="reconcile", notes=pair_notes(pairs)
        )
        assert written == ["contradiction-alpha-vs-beta"]

    def test_policy_flag_param_forces_write(self, tmp_path):
        """policy='flag' kwarg forces contradiction note creation."""
        ctx = str(tmp_path / "context")
        pairs = [("alpha", "beta")]
        written = write_contradiction_notes(
            pairs, ctx, policy="flag", notes=pair_notes(pairs)
        )
        assert len(written) == 1

    def test_suppress_is_the_only_silent_policy(self, tmp_path):
        ctx = str(tmp_path / "context")
        assert write_contradiction_notes(
            [("alpha", "beta")], ctx, policy="suppress"
        ) == []

    def test_reconcile_uses_explicit_supersession(self, tmp_path):
        ctx = str(tmp_path / "context")
        newer = make_note(
            "newer", body="[[contradicts:older]] [[supersedes:older]]",
            description="Acme API is enabled", entities=["org:acme"],
        )
        newer.state = "verified"
        older = make_note(
            "older", description="Acme API is disabled", entities=["org:acme"]
        )
        write_note(older, ctx)
        write_reviewed_note(tmp_path, newer)
        current = {note.name: note for note in load_notes(ctx)}
        newer, older = current["newer"], current["older"]

        written = write_contradiction_notes(
            [("newer", "older")], ctx, policy="reconcile", notes=[newer, older]
        )
        assert written == []
        fresh = {note.name: note for note in load_notes(ctx)}
        assert fresh["older"].superseded is False

    def test_reconcile_does_not_activate_proposed_supersession(self, tmp_path):
        ctx = str(tmp_path / "context")
        newer = make_note(
            "newer", body="[[supersedes:older]]", description="new",
        )
        newer.state = "proposed"
        older = make_note("older", description="old")
        written = write_contradiction_notes(
            [("newer", "older")], ctx, policy="reconcile", notes=[newer, older]
        )
        assert written == ["contradiction-newer-vs-older"]
        assert older.superseded is False

    def test_contradiction_note_acl_cannot_publish_restricted_name(self, tmp_path):
        ctx = str(tmp_path / "context")
        secret = make_note("secret-note", acl=["secrets"])
        public = make_note("public-note")
        written = write_contradiction_notes(
            [("public-note", "secret-note")], ctx, policy="flag",
            notes=[public, secret],
        )
        assert len(written) == 1
        contradiction = next(note for note in load_notes(ctx) if note.type == "contradiction")
        assert contradiction.acl == ["secrets"]

    def test_missing_source_metadata_defaults_flag_to_secrets(self, tmp_path):
        ctx = str(tmp_path / "context")
        with pytest.raises(ACLConflict, match="metadata is required"):
            write_contradiction_notes(
                [("unknown-a", "unknown-b")], ctx, policy="flag"
            )

    def test_long_pair_names_do_not_collide_after_slug_truncation(self, tmp_path):
        ctx = str(tmp_path / "context")
        common = "a" * 100
        pairs = [(common + "-one", "b" * 100), (common + "-two", "b" * 100)]
        written = write_contradiction_notes(
            pairs, ctx, policy="flag", notes=pair_notes(pairs)
        )
        assert len(set(written)) == 2
        assert all(len(name) <= 80 for name in written)
        assert len([note for note in load_notes(ctx) if note.type == "contradiction"]) == 2

    def test_refuses_to_overwrite_user_owned_deterministic_name(self, tmp_path):
        ctx = str(tmp_path / "context")
        pair = [("alpha", "beta")]
        protected = make_note(
            "contradiction-alpha-vs-beta", description="human-owned note"
        )
        write_note(protected, ctx)
        with pytest.raises(FileExistsError, match="non-owned"):
            write_contradiction_notes(
                pair, ctx, policy="flag", notes=pair_notes(pair)
            )
        stored = next(note for note in load_notes(ctx)
                      if note.name == protected.name)
        assert stored.description == "human-owned note"

    def test_generated_flag_resolves_and_no_longer_surfaces(self, tmp_path):
        from retrieve.query import search

        ctx = str(tmp_path / "context")
        pair = [("alpha", "beta")]
        sources = pair_notes(pair)
        sources[0].body = "[[contradicts:beta]]"
        for source in sources:
            write_reviewed_note(tmp_path, source, state="verified")
        current = {note.name: note for note in load_notes(ctx)}
        sources = [current[source.name] for source in sources]
        write_contradiction_notes(pair, ctx, policy="flag", notes=sources)
        assert "contradiction-alpha-vs-beta" in {
            row["name"]
            for row in search("contradiction alpha beta", root=str(tmp_path))
        }

        corrected = make_note(
            "alpha-corrected",
            description="Corrected alpha state",
            body="[[supersedes:alpha]]",
        )
        write_reviewed_note(tmp_path, corrected, state="verified")
        changed = write_contradiction_notes([], ctx, policy="flag", notes=sources)
        assert changed == ["contradiction-alpha-vs-beta"]
        stored = next(note for note in load_notes(ctx)
                      if note.name == "contradiction-alpha-vs-beta")
        assert stored.state == "superseded"
        assert stored.superseded is True
        assert "contradiction-alpha-vs-beta" not in {
            row["name"]
            for row in search("contradiction alpha beta", root=str(tmp_path))
        }

    def test_pending_sources_cannot_be_laundered_into_visible_flag(self, tmp_path):
        from retrieve.query import search

        ctx = str(tmp_path / "context")
        pair = [("alpha", "beta")]
        sources = pair_notes(pair)
        sources[0].body = "[[contradicts:beta]]"
        for source in sources:
            write_note(source, ctx)

        write_contradiction_notes(pair, ctx, policy="flag", notes=sources)

        flag = next(note for note in load_notes(ctx) if note.type == "contradiction")
        verification = flag.confidence["verification"]
        assert verification["status"] == "skipped"
        assert verification["grounded"] is False
        assert verification["review_required"] is True
        assert "contradiction-alpha-vs-beta" not in {
            row["name"]
            for row in search("contradiction alpha beta", root=str(tmp_path))
        }

    def test_caller_cannot_attest_an_unreproducible_pair(self, tmp_path):
        ctx = str(tmp_path / "context")
        pair = [("alpha", "beta")]
        sources = pair_notes(pair)
        for source in sources:
            write_reviewed_note(tmp_path, source, state="verified")
        current = {note.name: note for note in load_notes(ctx)}
        sources = [current[source.name] for source in sources]

        write_contradiction_notes(pair, ctx, policy="flag", notes=sources)

        flag = next(note for note in load_notes(ctx) if note.type == "contradiction")
        verification = flag.confidence["verification"]
        assert verification["status"] == "skipped"
        assert "not reproducible" in verification["reason"]

    def test_human_resolved_flag_is_never_rewritten_by_lint(self, tmp_path):
        ctx = str(tmp_path / "context")
        pair = [("alpha", "beta")]
        sources = pair_notes(pair)
        write_contradiction_notes(pair, ctx, policy="flag", notes=sources)
        flag = next(note for note in load_notes(ctx) if note.type == "contradiction")
        flag.provenance.append("by:alice")
        flag.body = "HUMAN RESOLUTION: both statements apply to different environments."
        write_note(flag, ctx)
        path, flag = write_reviewed_note(tmp_path, flag, state="verified")
        before = open(path, "rb").read()

        written = write_contradiction_notes(pair, ctx, policy="flag", notes=sources)

        assert open(path, "rb").read() == before
        proposals = [note for note in load_notes(ctx)
                     if ("supersedes", flag.name) in note.typed_links]
        assert len(proposals) == 1
        assert proposals[0].state == "proposed"
        assert proposals[0].name in written


# ---------------------------------------------------------------------------
# AGENTS.md contradiction_policy wired into lint.engine.run()
# ---------------------------------------------------------------------------

class TestAgentsMdPolicyInRun:
    """lint.engine.run() must honour contradiction_policy from AGENTS.md."""

    def _make_note(self, name, body=""):
        return Note(name=name, description="desc", type="concept", domain="eng", body=body)

    def test_agents_md_reconcile_falls_back_to_flag_when_uncertain(self, tmp_path):
        """AGENTS reconcile keeps uncertain contradictions visible for review."""
        # Write AGENTS.md with reconcile policy at project root
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text(
            "# AGENTS\n"
            "```conventions\n"
            "contradiction_policy: reconcile\n"
            "```\n"
        )

        ctx = tmp_path / "context"
        ctx.mkdir()
        a = self._make_note("note-a", body="[[contradicts:note-b]]")
        b = self._make_note("note-b")
        write_note(a, str(ctx))
        write_note(b, str(ctx))

        summary = run(root=str(tmp_path))

        # Contradictions are still detected
        assert ("note-a", "note-b") in summary["contradictions"] or \
               ("note-b", "note-a") in summary["contradictions"]

        # No safe supersession evidence exists, so reconcile falls back to flag.
        notes = load_notes(str(ctx))
        c_notes = [n for n in notes if n.type == "contradiction"]
        assert len(c_notes) == 1
        assert summary["reconciled"] == []

    def test_agents_md_flag_writes_contradiction_notes(self, tmp_path):
        """AGENTS.md with contradiction_policy: flag creates contradiction notes."""
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text(
            "# AGENTS\n"
            "```conventions\n"
            "contradiction_policy: flag\n"
            "```\n"
        )

        ctx = tmp_path / "context"
        ctx.mkdir()
        a = self._make_note("note-x", body="[[contradicts:note-y]]")
        b = self._make_note("note-y")
        write_note(a, str(ctx))
        write_note(b, str(ctx))

        run(root=str(tmp_path))

        notes = load_notes(str(ctx))
        c_notes = [n for n in notes if n.type == "contradiction"]
        assert len(c_notes) >= 1, (
            "flag policy must cause contradiction note creation"
        )

    def test_run_never_flags_a_historical_supersession_target(self, tmp_path):
        ctx = tmp_path / "context"
        old = self._make_note("old-claim", body="[[contradicts:other-claim]]")
        other = self._make_note("other-claim")
        replacement = self._make_note(
            "current-claim", body="[[supersedes:old-claim]]",
        )
        for note in (old, other, replacement):
            write_reviewed_note(tmp_path, note, state="verified")

        summary = run(root=str(tmp_path))

        assert summary["contradictions"] == []
        assert not [
            note for note in load_notes(str(ctx))
            if note.type == "contradiction"
        ]
