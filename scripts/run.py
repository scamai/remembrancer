#!/usr/bin/env python3
"""Local pipeline orchestrator — the runtime (no CI/CD). ingest -> guard -> distill."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.git_events import GitConnector
from distill import engine as distiller
from remembrancerlib import Cursors, iter_events, pipeline_mutation_lock, regen_index
from remembrancerlib.verification_audit import rotate_verification_audit_if_full
from tools.corpus_guard import inspect as guard_corpus

#: The audit distill appends each verdict to (``distill/engine.py``).
VERIFICATION_AUDIT = os.path.join("state", "verification.jsonl")


def _rotate_full_audit(root="."):
    """Seal the verification audit once it nears its size cap.

    An append never rotates the audit: a full one refuses every new verdict.
    Sealing belongs here, under the pipeline lock and before this run appends
    anything. A failure is a warning, not an abort: the run goes on, and new
    verdicts are refused until a later run seals the file.
    """
    try:
        sealed = rotate_verification_audit_if_full(os.path.join(root, VERIFICATION_AUDIT))
    except OSError as exc:
        print(f"[run] verification audit not rotated: {exc}", file=sys.stderr)
        return 0
    if sealed:
        print(f"[run] sealed verification audit segment {sealed}", file=sys.stderr)
    return sealed


def _ensure_empty_checkpoint(root="."):
    checkpoint = os.path.join(root, "state", "event_validation_checkpoint.json")
    if os.path.exists(checkpoint):
        return
    events = os.path.join(root, "events")
    os.makedirs(events, exist_ok=True)
    if any(name.endswith(".jsonl") and os.path.getsize(os.path.join(events, name))
           for name in os.listdir(events)):
        raise RuntimeError(
            "non-empty event store has no validation checkpoint; attended bootstrap required"
        )
    result = guard_corpus(root, bootstrap_checkpoint=True)
    if not result["ok"]:
        raise RuntimeError("empty event checkpoint bootstrap failed: " + "; ".join(result["errors"]))

def _event_ids(root="."):
    events = os.path.join(root, "events")
    return {ev.id for ev in iter_events(events)} if os.path.isdir(events) else set()


def _behind_watermark(root, before):
    """The events this run appended that the distill watermark has already passed.

    The store is ordered by day-file, each named from its events' own dates,
    and the watermark is a position in that order. A repository ingested after
    an earlier run files its older commits into past day-files, behind the
    watermark, where the normal pass never reads them.
    """
    events = os.path.join(root, "events")
    new = [ev for ev in iter_events(events) if ev.id not in before]
    watermark = Cursors(os.path.join(root, "state", "cursors.json")).get("distill")
    if not new or watermark is None:
        return []
    ahead = {ev.id for ev in iter_events(events, after_id=watermark)}
    return [ev for ev in new if ev.id not in ahead]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest-git", nargs="*", action="append", default=None,
                    help="Repo paths to ingest. Bare flag (no paths) ingests the "
                         "config.json `monorepos` via auto-maintained mirrors.")
    ap.add_argument("--no-distill", action="store_true")
    ap.add_argument("--lint", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="Ignore distill watermark and reprocess all events.")
    args = ap.parse_args()
    with pipeline_mutation_lock(".", blocking=True):
        _rotate_full_audit(".")
        _ensure_empty_checkpoint(".")
        # A rebuild reprocesses every event, so it needs no catch-up pass.
        catch_up = args.ingest_git is not None and not args.no_distill and not args.rebuild
        before = _event_ids(".") if catch_up else set()
        if args.ingest_git is not None:
            repos = [repo for group in args.ingest_git for repo in group]
            if repos:
                for repo in repos:
                    # GitConnector expects config with "repos" (list of dicts with
                    # "path"), not a bare "repo" string key — adapt accordingly.
                    GitConnector({"repos": [{"path": repo}]}, events_dir="events").run()
            else:
                # Config-driven: ingest the config.json `monorepos` (the nightly path;
                # git_events.py maintains bare mirror clones under state/git_mirrors/).
                import json
                config = {}
                if os.path.exists("config.json"):
                    with open("config.json") as f:
                        config = json.load(f)
                if config.get("repos") or config.get("monorepos"):
                    GitConnector(config, events_dir="events").run()
                else:
                    print("[run] --ingest-git given but no monorepos in config.json",
                          file=sys.stderr)
        guarded = guard_corpus(".")
        if not guarded["ok"]:
            raise RuntimeError("event corpus guard failed: " + "; ".join(guarded["errors"]))
        if not args.no_distill:
            # Found before the pass below, which moves the watermark past them.
            behind = _behind_watermark(".", before) if catch_up else []
            distiller.run(root=".", rebuild=args.rebuild)
            if behind:
                print(f"[run] distilling {len(behind)} ingested events behind the "
                      "distill watermark", file=sys.stderr)
                distiller.run(root=".", only_events=behind)
        if args.lint:
            from lint import engine as linter
            linter.run(root=".")
        regen_index("context")
    print("[run] pipeline complete")

if __name__ == "__main__":
    main()
