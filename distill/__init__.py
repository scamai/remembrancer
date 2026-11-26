"""distill — the distillation engine.

Maps the flood of Layer-1 events into a small Layer-2 tree of durable truth.
Reads new events past the `distill` watermark, clusters related ones, resolves
customer entities against the registry, and per cluster picks exactly one of
CREATE / UPDATE / LINK / IGNORE on context/ notes. Distillation happens at
WRITE time only (an optional LLM may sharpen a description), never read time.

Stdlib-first; PyYAML used if present (registry parsing falls back to a tiny
line parser otherwise).
"""
from .entities import EntityResolver, resolve
from .engine import run, cluster_events

__all__ = ["EntityResolver", "resolve", "run", "cluster_events"]
