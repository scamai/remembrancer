"""Test helpers that exercise engine calls through the real corpus guard."""
from __future__ import annotations

import os

from tools.corpus_guard import inspect


def guard_events(root: str) -> None:
    checkpoint = os.path.join(root, "state", "event_validation_checkpoint.json")
    result = inspect(root, bootstrap_checkpoint=not os.path.exists(checkpoint))
    assert result["ok"], result


def guarded_run(engine_module, *args, **kwargs):
    root = os.path.abspath(kwargs.get("root", args[0] if args else "."))
    guard_events(root)
    return engine_module.run(*args, **kwargs)
