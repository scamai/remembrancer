"""The ACL / domain labeler — governance is the single labeling authority.

At ingest, every event must be assigned a `domain` (which context/ folder it
distills into) and an `acl` (who may see it). Rather than hand-config per source,
we read the org's own policy/governance.yml so Remembrancer's security boundary == the
org's governance boundary, defined once:

    never_public  -> acl == ['secrets']                   (sensitive material)
    public_allowlist -> acl == []                         (world-readable)
    otherwise     -> per-source default from config.json acl_defaults

If governance.yml is absent (e.g. local dev), fall back to config defaults so the
pipeline still runs. PyYAML is used if available; otherwise a tiny line parser.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any


_SENSITIVE_CONTENT = re.compile(
    r"\b(?:salary|compensation|offer[ -]?letter|term[ -]?sheet|data[ -]?room|"
    r"attorney[ -]?client|legally privileged|privileged legal|severance|termination|"
    r"h-?1b|immigration|visa|payroll|bank account|routing number|social security)\b",
    re.IGNORECASE,
)


class ACLConflict(ValueError):
    """Raised when OR-grant ACLs cannot safely represent combined content."""


def combine_acl(*acls: list[str]) -> list[str]:
    """Return an OR-grant ACL safe for every input, or refuse the merge.

    Public inputs add no restriction.  For restricted inputs, every output
    label must occur in every source label set; otherwise a holder of just one
    output label could read source content they were never granted.  Remembrancer has
    no AND expression, so an empty intersection is unrepresentable and must be
    split or rejected—not relabeled as ``secrets``.
    """
    restricted = [
        {str(label).strip() for label in (acl or []) if str(label).strip()}
        for acl in acls if acl
    ]
    if not restricted:
        return []
    common = set.intersection(*restricted)
    if not common:
        raise ACLConflict("incompatible OR-grant ACL scopes cannot be combined")
    return sorted(common)


def acl_can_reveal(container_acl: list[str], source_acl: list[str]) -> bool:
    """Whether every reader of ``container_acl`` may see ``source_acl``.

    This is used before placing even a source note's name/link in another note.
    A public source is always safe; a public container cannot reveal a private
    source; otherwise each container label must also grant the source.
    """
    container = {str(label).strip() for label in (container_acl or []) if str(label).strip()}
    source = {str(label).strip() for label in (source_acl or []) if str(label).strip()}
    if not source:
        return True
    if not container:
        return False
    return container.issubset(source)


def restrict_sensitive_acl(text: str, acl: list[str]) -> list[str]:
    """Fail closed for obvious HR/fundraising/legal/financial source text."""
    return ["secrets"] if _SENSITIVE_CONTENT.search(text or "") else list(acl)


def restricted_labels(cfg: Any) -> frozenset[str]:
    """Labels whose notes always wait for a person.

    ``secrets`` always counts, and ``governance.restricted_labels`` in
    config.json may add more. Governance holds a note with one of them for a
    person, no machine approves it, and no visibility valve shows it while it
    is held. Config can add labels, never remove ``secrets``.
    """
    governance = cfg.get("governance") if isinstance(cfg, dict) else None
    extra = governance.get("restricted_labels") if isinstance(governance, dict) else None
    labels = {"secrets"}
    if isinstance(extra, list):
        labels.update(label.strip() for label in extra if isinstance(label, str) and label.strip())
    return frozenset(labels)

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


def _load_yaml_lists(path: str) -> dict[str, list[str]]:
    """Load a governance.yml that contains top-level lists like
    `never_public:` / `public_allowlist:` with `- name` items. Returns name->list."""
    if not os.path.exists(path):
        return {}
    if _HAVE_YAML:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return {k: list(v) for k, v in data.items() if isinstance(v, list)}
    # minimal fallback parser for `key:` then `  - item` blocks
    out: dict[str, list[str]] = {}
    cur = None
    with open(path) as f:
        for line in f:
            if re_top := (line.rstrip().endswith(":") and not line.startswith((" ", "\t"))):
                cur = line.strip()[:-1]
                out[cur] = []
            elif cur and line.lstrip().startswith("- "):
                out[cur].append(line.lstrip()[2:].strip().strip('"\''))
    return out


class Labeler:
    def __init__(self, config_path: str = "config.json", governance_path: str = "policy/governance.yml"):
        self.cfg: dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.cfg = json.load(f)
        self.acl_defaults: dict[str, list[str]] = self.cfg.get("acl_defaults", {})
        gov = _load_yaml_lists(governance_path)
        self.never_public = set(gov.get("never_public", []))
        self.public_allowlist = set(gov.get("public_allowlist", []))

    def label(self, source: str, key: str | None = None, domain_hint: str | None = None) -> tuple[str, list[str]]:
        """Return (domain, acl) for an event.

        source: the connector ('git', 'slack', ...).
        key:    the governance-relevant identifier (repo, repo/folder, channel).
        domain_hint: connector's suggested context/ folder (e.g. top-level monorepo folder).
        """
        domain = domain_hint or self.acl_defaults.get(f"{source}.domain", source) or "company"
        # governance overrides win
        if key:
            head = key.split("/")[0]
            if key in self.never_public or head in self.never_public:
                return domain, ["secrets"]
            if key in self.public_allowlist or head in self.public_allowlist:
                return domain, []
        acl = list(self.acl_defaults.get(source, ["eng"]))
        return domain, acl
