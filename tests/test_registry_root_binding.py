"""run() reads the registries of the root it runs on, and only those.

The engine's module-level registry paths name the code folder's own
``registry/``, for callers that use the engine without run(). run() used to
keep those paths whenever its root lacked a registry file, so a run on any
other root silently read the code folder's registry. Here a stand-in folder
plays the code folder: its registry names an investor, a component and an
initiative.
"""
import os

import pytest

from distill import engine
from remembrancerlib import Event, append_events, load_notes
from tests.event_validation_helpers import guarded_run

_TYPES = """\
investor:
- name: Initech Ventures
  email_domains: [initechventure.example]
"""
_COMPONENTS = "- {repo: billing-engine, entities: [billing engine]}\n"
_INITIATIVES = """\
initiatives:
- id: faster-invoicing
  title: Faster invoicing
"""
_IDS = {"investor:initech-ventures", "component:billing-engine",
        "initiative:faster-invoicing"}


def _write_registry(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "entities.yml").write_text(_TYPES)
    (folder / "eng_components.yml").write_text(_COMPONENTS)
    (folder / "initiatives.yml").write_text(_INITIATIVES)


@pytest.fixture
def code_registry(tmp_path, monkeypatch):
    """Point the module defaults at a stand-in for the code folder's registry."""
    folder = tmp_path / "code" / "registry"
    _write_registry(folder)
    monkeypatch.setattr(engine, "_REGISTRY_TYPES_PATH", str(folder / "entities.yml"))
    monkeypatch.setattr(engine, "_ENG_COMPONENTS_PATH",
                        str(folder / "eng_components.yml"))
    monkeypatch.setattr(engine, "_INITIATIVES_PATH", str(folder / "initiatives.yml"))
    monkeypatch.setattr(engine, "_ENTITY_UNRESOLVED_PATH",
                        str(tmp_path / "code" / "state" / "entity_unresolved.jsonl"))
    return folder


def _run(root):
    """One deterministic run over a git event naming all three registry entries.
    Returns the notes' entity ids and component ids, and the notes."""
    os.makedirs(root, exist_ok=True)
    append_events([Event(
        id="git:lv1", source="git", author="Ana <ana@acme.example>",
        payload={"subject": "signed the Initech Ventures term sheet",
                 "files": ["billing/terms.md"]},
        occurred_at="2026-07-01T00:00:00Z",
        entities=["Initech Ventures", "billing engine", "Faster invoicing"],
        provenance={"repo": "acme/app"},
    )], os.path.join(root, "events"))
    guarded_run(engine, root=root, use_llm=False, rebuild=True)
    notes = load_notes(os.path.join(root, "context"))
    assert notes
    ids = {i for n in notes for i in [*n.entities, *n.components]}
    return ids, notes


def test_a_root_without_a_registry_does_not_read_the_code_folders(
        tmp_path, code_registry):
    ids, notes = _run(str(tmp_path / "root"))
    assert not ids & _IDS
    assert all(n.domain != "investors" for n in notes)


def test_a_root_with_a_registry_reads_its_own(tmp_path, code_registry):
    """Control: the same run reads the registry when the root has it."""
    root = tmp_path / "root"
    _write_registry(root / "registry")
    ids, notes = _run(str(root))
    assert _IDS <= ids
    assert all(n.domain == "investors" for n in notes
               if "investor:initech-ventures" in n.entities)
