"""The connector contract — every source implements the SAME four stages, so
adding a source never changes anything downstream (the extensibility property).

    fetch(cursor)  -> raw items          # pull new items from the source API
    filter(raw)    -> kept               # THE anti-blowup valve: drop noise here
    normalize(kept)-> list[Event]        # map to the standard event shape
    + a watermark cursor so runs are incremental & idempotent

Concrete connectors subclass Connector and implement fetch/filter/normalize and
the `name`/`cursor_key` attributes. `run()` ties them together: pull from the
watermark, filter, normalize, label (domain+acl via governance), resolve entities
later in distillation, append to the event log, advance the cursor.

Credentials come from environment variables at runtime — NEVER committed.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remembrancerlib import Event, append_events, Cursors, Labeler  # noqa: E402


def strict_connector_health() -> bool:
    """Whether an unavailable configured source must fail its process status.

    Connector CLIs remain convenient no-ops in credential-free developer/CI
    environments.  Governed unattended runs set ``REMEMBRANCER_CONNECTORS_STRICT=1``
    so missing auth, incomplete pagination, and corrupt continuation state are
    visible to per-source health monitoring instead of looking like zero deltas.
    """
    return os.environ.get("REMEMBRANCER_CONNECTORS_STRICT", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


class Connector:
    name: str = "base"            # source label, e.g. "slack"
    cursor_key: str = "base"      # key in state/cursors.json

    def __init__(self, config: dict[str, Any], events_dir: str = "events",
                 cursors: Cursors | None = None, labeler: Labeler | None = None):
        self.config = config
        self.events_dir = events_dir
        self.cursors = cursors or Cursors()
        self.labeler = labeler or Labeler()

    # --- the three stages a subclass MUST implement -------------------------
    def fetch(self, cursor: str | None) -> Iterable[Any]:
        """Pull raw items newer than `cursor` from the source. Yield raw objects."""
        raise NotImplementedError

    def filter(self, raw: Any) -> bool:
        """Return True to keep this raw item. THE anti-blowup valve — be strict."""
        return True

    def normalize(self, raw: Any) -> Event:
        """Map one kept raw item to a standard Event. Use self.labeler.label(...)
        to set domain+acl; set provenance + a stable id (source:natural-id)."""
        raise NotImplementedError

    # --- the orchestration (subclasses usually don't override) --------------
    def new_cursor(self, raw_items: list[Any]) -> str | None:
        """Compute the new watermark from the items just pulled. Override if the
        source's cursor isn't simply 'the last item's id'."""
        return None

    def run(self) -> int:
        cursor = self.cursors.get(self.cursor_key)
        raw_items = list(self.fetch(cursor))
        kept = [r for r in raw_items if self.filter(r)]
        events = [self.normalize(r) for r in kept]
        written = append_events(events, self.events_dir)
        nc = self.new_cursor(raw_items)
        if nc:
            self.cursors.set(self.cursor_key, nc)
        print(f"[{self.name}] pulled={len(raw_items)} kept={len(kept)} "
              f"written={written} cursor={nc}", file=sys.stderr)
        return written
