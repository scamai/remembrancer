"""Test-only constructors for governed and deliberately legacy trust states."""
from __future__ import annotations

import json
from pathlib import Path

from remembrancerlib import load_notes, write_note
from retrieve.query import promote, reject


_ALL_TEST_GRANTS = [
    "eng", "ml", "customers", "infra", "secrets", "founders",
]


def configure_approver(root, identity: str = "reviewer") -> None:
    root_path = Path(root)
    config_path = root_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        config = {}
    assert isinstance(config, dict)
    identities = config.setdefault("identities", {})
    assert isinstance(identities, dict)
    identities[identity] = list(_ALL_TEST_GRANTS)
    trust = config.setdefault("trust", {})
    assert isinstance(trust, dict)
    approvers = trust.setdefault("approver_identities", [])
    assert isinstance(approvers, list)
    if identity not in approvers:
        approvers.append(identity)
    config_path.write_text(json.dumps(config), encoding="utf-8")


def review_note(root, name: str, *, state: str = "verified",
                identity: str = "reviewer"):
    configure_approver(root, identity)
    transition = promote if state == "verified" else reject
    result = transition(name, identity=identity, root=str(root))
    assert result["ok"], result
    return next(
        note for note in load_notes(str(Path(root) / "context"))
        if note.name == name
    )


def write_reviewed_note(root, note, *, state: str | None = None,
                        identity: str = "reviewer"):
    target = state or note.state
    assert target in {"verified", "superseded"}
    configure_approver(root, identity)
    note.state = "proposed"
    note.superseded = False
    note.approval = None
    path = write_note(note, str(Path(root) / "context"))
    reviewed = review_note(root, note.name, state=target, identity=identity)
    return path, reviewed


def write_raw_legacy_note(root, note) -> str:
    """Bypass the write door only to exercise fail-closed imported legacy bytes."""
    path = Path(root) / "context" / note.domain / f"{note.name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note.to_markdown(), encoding="utf-8")
    return str(path)
