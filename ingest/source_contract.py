"""Shared scaffolding for per-connector source contracts.

Each connector (slack / gcal / gdrive / …) defines its OWN field set and validation
rules, but three pieces were byte-identical in every ``*_contract.py``: the
field-access helper, the error base, and the ``validate → bool`` wrapper. They live
here so a new connector's contract is *(fields + rules)*, not another copy of the
scaffolding. Connector-specific logic (``snapshot_revision``, ``validate_*_event``,
lineage, source-time) stays in each connector module — it genuinely differs.
"""
from __future__ import annotations

from typing import Any, Callable


class ContractError(ValueError):
    """Base for per-source contract violations. Each connector's error subclasses
    this (which is still a ``ValueError``), so existing callers catching either the
    connector error or ``ValueError`` are unaffected."""


def field(event: Any, name: str) -> Any:
    """Read a field from an event that may be a dict or an object."""
    return event.get(name) if isinstance(event, dict) else getattr(event, name, None)


def event_valid(validate: Callable[[Any], None], event: Any) -> bool:
    """True iff ``validate(event)`` raises no contract/type/value error — the wrapper
    every connector spelled out identically. Catches ``ContractError`` (hence every
    connector's error subclass), ``TypeError`` and ``ValueError``."""
    try:
        validate(event)
    except (ContractError, TypeError, ValueError):
        return False
    return True
