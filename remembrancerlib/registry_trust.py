"""Field-level trust and semantic quarantine for the split entity registry.

The registry is identity input, not self-authenticating truth.  High-impact
fields influence domain routing, graph edges and ACL-visible generated pages;
they become authoritative only with an explicit human-reviewed provenance
record.  Obviously kind-confused rows are quarantined in memory and sent to a
bounded review queue rather than guessed into a different kind.
"""
from __future__ import annotations

import re
from typing import Any


HIGH_IMPACT_FIELDS = frozenset({
    "relationship", "org_type", "aliases", "email_domains", "emails",
    "works_at", "role", "owner", "introduced_by", "champions", "acl",
    "internal",
})
AUTHORITY_REVIEW_FIELDS = frozenset({
    "relationship", "org_type", "works_at", "role", "owner",
    "introduced_by", "champions",
})
_ORG_WORDS = frozenset({
    "ai", "labs", "lab", "ventures", "venture", "capital", "catalyst",
    "company", "corp", "corporation", "inc", "llc", "ltd",
    "foundation", "university", "studio", "partners", "group", "network",
    "fund", "systems", "solutions", "security", "audio", "technologies",
    "technology", "collective", "accelerator", "services", "enterprise",
})


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def registry_field_attestation(record: dict[str, Any], field: str) -> dict[str, Any] | None:
    """Return a canonical human-reviewed field proof, else ``None``."""
    provenance = record.get("field_provenance")
    proof = provenance.get(field) if isinstance(provenance, dict) else None
    if not isinstance(proof, dict) or proof.get("status") != "verified":
        return None
    evidence = proof.get("evidence")
    reviewed_by = proof.get("reviewed_by")
    if (not isinstance(evidence, list) or not evidence
            or any(not isinstance(item, str) or not item.strip() for item in evidence)
            or not isinstance(reviewed_by, str) or not reviewed_by.strip()):
        return None
    return {
        "field": field,
        "status": "verified",
        "evidence": sorted(set(item.strip() for item in evidence)),
        "reviewed_by": reviewed_by.strip(),
    }


def registry_field_verified(record: dict[str, Any], field: str) -> bool:
    return registry_field_attestation(record, field) is not None


def set_registry_field_attestation(
    record: dict[str, Any],
    field: str,
    *,
    evidence: list[str],
    reviewed_by: str,
) -> dict[str, Any]:
    """Attach a normalized proof for a field a governed reviewer just checked.

    ``reviewed_by`` is an accountable actor id.  It is normally a human; a
    deterministic producer may use its stable ``producer:<name>`` id only when
    its evidence contract independently proves the exact field (the meeting
    reconciler does this for observed email/domain identity, never for inferred
    relationship type or employment).
    """
    if field not in HIGH_IMPACT_FIELDS:
        raise ValueError(f"unsupported registry trust field: {field}")
    clean_evidence = sorted({str(item).strip() for item in evidence if str(item).strip()})
    actor = str(reviewed_by or "").strip()
    if not clean_evidence or not actor:
        raise ValueError("registry field attestation requires evidence and reviewer")
    proof = {
        "status": "verified",
        "evidence": clean_evidence,
        "reviewed_by": actor,
    }
    record.setdefault("field_provenance", {})[field] = proof
    return proof


def registry_proofs_valid(value: Any, required_fields: set[str] | None = None) -> bool:
    """Validate the durable attestation envelope carried by a registry edge."""
    if not isinstance(value, list) or not value:
        return False
    fields: set[str] = set()
    for proof in value:
        if not isinstance(proof, dict):
            return False
        field = proof.get("field")
        if (not isinstance(field, str) or field not in HIGH_IMPACT_FIELDS
                or field in fields or proof.get("status") != "verified"
                or not isinstance(proof.get("reviewed_by"), str)
                or not proof["reviewed_by"].strip()
                or not isinstance(proof.get("evidence"), list)
                or not proof["evidence"]
                or any(not isinstance(item, str) or not item.strip()
                       for item in proof["evidence"])):
            return False
        fields.add(field)
    return required_fields is None or required_fields <= fields


def registry_edge_proofs_valid(edge: dict[str, Any]) -> bool:
    """Require the field proof appropriate for this exact registry edge."""
    source = edge.get("source")
    edge_type = edge.get("type")
    if source == "curated":
        required = {
            "works_at": {"works_at"},
            "owner_of": {"owner"},
            "introduced": {"introduced_by"},
            "champion_of": {"champions"},
        }.get(edge_type)
    elif source == "derived":
        required = ({"emails", "email_domains"} if edge_type == "works_at"
                    else {"relationship"} if str(edge_type or "").endswith("_of")
                    else None)
    else:
        required = None
    return required is not None and registry_proofs_valid(
        edge.get("registry_proof"), required,
    )


def _name_tokens(name: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", str(name or ""))


def looks_like_person_name(name: str) -> bool:
    tokens = _name_tokens(name)
    if not 2 <= len(tokens) <= 4:
        return False
    if any(token.casefold() in _ORG_WORDS for token in tokens):
        return False
    return all(token[:1].isupper() for token in tokens)


def looks_like_org_name(name: str) -> bool:
    text = str(name or "").casefold()
    tokens = _name_tokens(name)
    return (any(token.casefold() in _ORG_WORDS for token in tokens)
            or any(marker in text for marker in (".com", ".ai", ".org", "&", "/")))


def registry_record_quarantine_reasons(record: dict[str, Any], kind: str) -> list[str]:
    """Fail-closed checks for kind pollution and sensitive unreviewed flags."""
    identifier = str(record.get("id") or "").strip().lower()
    name = str(record.get("name") or "").strip()
    reasons: list[str] = []
    if not identifier or not name:
        reasons.append("missing-id-or-name")
    elif not identifier.startswith(f"{kind}:"):
        reasons.append("id-kind-mismatch")
    if kind == "person" and looks_like_org_name(name):
        reasons.append("person-record-looks-like-organization")
    # A human-looking organization name can be legitimate (law firms and
    # founder-named companies are common).  It becomes an obvious kind error
    # only in the relationship buckets that historically represented people.
    # The raw relationship is used only to *quarantine for review* here; it is
    # never promoted to authoritative routing without an attestation.
    # ``advisor`` and ``recruiting`` are the buckets that tend to collect
    # individual contacts.  ``friend``/``champion`` usually mix people and
    # organizations, so a name-only heuristic is not strong enough to isolate
    # those rows; their unreviewed relationship is still ignored and queued.
    person_relationships = {"advisor", "recruiting"}
    raw_relationships = {
        value.casefold() for value in _values(record.get("relationship"))
    }
    if (kind == "org" and looks_like_person_name(name)
            and raw_relationships & person_relationships):
        reasons.append("org-record-looks-like-person")
    return reasons


def audit_registry_records(split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Return deterministic quarantine and attended-review work items."""
    blocking: list[dict[str, str]] = []
    review: list[dict[str, str]] = []
    quarantined: set[str] = set()
    seen_ids: set[str] = set()
    token_owners: dict[str, set[str]] = {}

    for bucket, kind in (("people", "person"), ("orgs", "org"), ("systems", "system")):
        for record in split.get(bucket) or []:
            identifier = str(record.get("id") or "").strip().lower()
            for reason in registry_record_quarantine_reasons(record, kind):
                quarantined.add(identifier)
                blocking.append({"entity_id": identifier, "field": "kind", "reason": reason})
            if identifier in seen_ids:
                quarantined.add(identifier)
                blocking.append({"entity_id": identifier, "field": "id",
                                 "reason": "duplicate-registry-id"})
            seen_ids.add(identifier)

            name = str(record.get("name") or "").strip().casefold()
            if name:
                token_owners.setdefault(name, set()).add(identifier)
            for alias in _values(record.get("aliases")):
                token_owners.setdefault(alias.casefold(), set()).add(identifier)

            for field in sorted(AUTHORITY_REVIEW_FIELDS):
                value = record.get(field)
                meaningful = bool(_values(value))
                if field in record and meaningful \
                        and not registry_field_verified(record, field):
                    review.append({"entity_id": identifier, "field": field,
                                   "reason": "field-is-not-human-reviewed"})

    for token, owners in sorted(token_owners.items()):
        if len(owners) > 1:
            for identifier in sorted(owners):
                review.append({"entity_id": identifier, "field": "aliases",
                               "reason": f"ambiguous-token:{token}"})

    def dedup(items: list[dict[str, str]]) -> list[dict[str, str]]:
        keys = sorted({
            (item["entity_id"], item["field"], item["reason"])
            for item in items
        })
        return [
            {"entity_id": entity_id, "field": field, "reason": reason}
            for entity_id, field, reason in keys
        ]

    return {
        "quarantined_ids": sorted(value for value in quarantined if value),
        "blocking": dedup(blocking),
        "review": dedup(review),
    }
