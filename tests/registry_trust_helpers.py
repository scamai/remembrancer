"""Synthetic registry fixtures with explicit attended field provenance."""
from __future__ import annotations

import yaml

from remembrancerlib.registry_trust import HIGH_IMPACT_FIELDS, set_registry_field_attestation


def attest_records(records: list[dict]) -> list[dict]:
    for record in records:
        for field in sorted(HIGH_IMPACT_FIELDS):
            value = record.get(field)
            meaningful = value is True if field == "internal" else value not in (None, [], "")
            if meaningful:
                set_registry_field_attestation(
                    record, field, evidence=["test:registry-fixture"],
                    reviewed_by="test-suite",
                )
    return records


def attest_yaml_text(raw: str, key: str) -> str:
    document = yaml.safe_load(raw) or {}
    attest_records(document.get(key) or [])
    return yaml.safe_dump(document, sort_keys=False)
