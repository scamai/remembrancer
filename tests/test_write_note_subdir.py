"""Project subdirectory write support.

Tests the distiller-side changes:
  - Note.project field round-trips through frontmatter
  - write_note uses 4-segment join when project set
  - _infer_project picks up an existing peer's project (same primary entity,
    same domain)
  - flat layout (project="") is the safe default
"""
import os
from remembrancerlib import Note, write_note
from distill.engine import _infer_project, Cluster, _apply_create
from remembrancerlib import Event


def test_note_project_roundtrip(tmp_path):
    """Note.project serializes as frontmatter and reads back."""
    ctx = str(tmp_path / "context")
    n = Note(name="x", description="d", type="project",
             domain="ml", project="search-ranking")
    write_note(n, ctx)
    raw = open(os.path.join(ctx, "ml", "search-ranking", "x.md")).read()
    assert "project: search-ranking" in raw
    # Reload via regen_index -> load_notes -> from_frontmatter
    notes = []
    from remembrancerlib import load_notes
    for n in load_notes(ctx):
        if n.name == "x":
            notes.append(n)
    assert len(notes) == 1
    assert notes[0].project == "search-ranking"


def test_write_note_no_project_uses_2_segment(tmp_path):
    """Empty project keeps the existing 2-segment flat layout."""
    ctx = str(tmp_path / "context")
    n = Note(name="x", description="d", type="project", domain="ml")
    write_note(n, ctx)
    assert os.path.exists(os.path.join(ctx, "ml", "x.md"))
    # Flat note lives directly at context/ml/x.md; no project subfolder
    # was created when project="" (os.path.join collapses empty segments,
    # so verify the directory contents directly instead).
    files_in_ml = os.listdir(os.path.join(ctx, "ml"))
    assert "x.md" in files_in_ml
    assert not os.path.isdir(os.path.join(ctx, "ml", "search-ranking"))


def test_write_note_with_project_uses_4_segment(tmp_path):
    ctx = str(tmp_path / "context")
    n = Note(name="x", description="d", type="project",
             domain="ml", project="search-ranking")
    write_note(n, ctx)
    assert os.path.exists(os.path.join(ctx, "ml", "search-ranking", "x.md"))
    # 2-segment path does NOT exist.
    assert not os.path.exists(os.path.join(ctx, "ml", "x.md"))


def test_infer_project_returns_empty_for_no_peers(tmp_path):
    """No same-entity-same-domain peers → empty (flat layout)."""
    cl = Cluster(key="k1", events=[], entities=["person:jane-doe"])
    notes_by_name = {}  # empty
    assert _infer_project(cl, "ml", notes_by_name) == ""


def test_infer_project_picks_up_peer_project(tmp_path):
    """Existing peer in same domain sharing primary entity → inherit its project."""
    cl = Cluster(key="k1", events=[], entities=["person:jane-doe"])
    notes_by_name = {
        "ml-search-ranking-v3": Note(
            name="ml-search-ranking-v3", description="v3", type="project",
            domain="ml", entities=["person:jane-doe"],
            project="search-ranking",
        ),
    }
    assert _infer_project(cl, "ml", notes_by_name) == "search-ranking"


def test_infer_project_no_cross_domain(tmp_path):
    """Same owner in different domain → no project inheritance."""
    cl = Cluster(key="k1", events=[], entities=["person:jane-doe"])
    notes_by_name = {
        "vend-1": Note(
            name="vend-1", description="v", type="project",
            domain="customers", entities=["person:jane-doe"],
            project="acme-renewal",
        ),
    }
    assert _infer_project(cl, "ml", notes_by_name) == ""


def test_infer_project_different_owner_no_match(tmp_path):
    """Same domain but different primary entity → no project inheritance."""
    cl = Cluster(key="k1", events=[], entities=["person:sam-roe"])
    notes_by_name = {
        "ml-jane-v3": Note(
            name="ml-jane-v3", description="v3", type="project",
            domain="ml", entities=["person:jane-doe"],
            project="search-ranking",
        ),
    }
    assert _infer_project(cl, "ml", notes_by_name) == ""


def test_apply_create_writes_to_project_subdir(tmp_path):
    """_apply_create, when a same-owner-same-domain peer has project set, writes
    the new note into that project's subfolder."""
    ctx = str(tmp_path / "context")
    os.makedirs(ctx, exist_ok=True)
    # Pre-seed a peer note in ml/search-ranking/ so the new note inherits it.
    write_note(Note(
        name="ml-search-ranking-v3", description="v3 progress",
        type="project", domain="ml", entities=["person:jane-doe"],
        project="search-ranking",
    ), ctx)
    notes_by_name = {
        "ml-search-ranking-v3": Note(
            name="ml-search-ranking-v3", description="v3 progress",
            type="project", domain="ml", entities=["person:jane-doe"],
            project="search-ranking",
        ),
    }
    # Use source="training" so the cluster's domain is ml/ (matching the
    # peer's domain). git source would route to eng/ and miss the peer.
    ev = Event(id="training:abc", source="training", author="janedoe",
               payload={"subject": "ranking model v4 update"},
               occurred_at="2026-07-02T00:00:00Z",
               entities=["person:jane-doe"], provenance={})
    cl = Cluster(key="k1", events=[ev], entities=["person:jane-doe"])
    _apply_create(cl, ctx, notes_by_name, [])
    # New note should land in the search-ranking subfolder.
    project_dir = os.path.join(ctx, "ml", "search-ranking")
    files = os.listdir(project_dir)
    # Both notes should be in there (the pre-seeded one + the new one).
    assert len(files) >= 2, f"expected ≥2 files in {project_dir}, got {files}"
    # The new note should NOT be at the flat path.
    flat_files = [f for f in os.listdir(os.path.join(ctx, "ml")) if f.endswith(".md")]
    assert flat_files == [], (
        f"new note must NOT also live at the flat ml/ path; got {flat_files}"
    )


# ---------------------------------------------------------------------------
# Layer 2 — project-namespace fallback for multi-contributor projects
# ---------------------------------------------------------------------------
def test_infer_project_namespace_fallback_different_owner(tmp_path):
    """Layer 2: when layer 1 (same-owner) misses, fall back to namespace-token
    match across all same-domain peers with project set. Lets a Sam search-ranking
    event inherit Jane's search-ranking project even though they don't share a
    primary entity."""
    # Cluster needs events with content so _name_for_cluster produces a slug
    # whose project-namespace tokens overlap the peer's.
    ev = Event(id="git:abc", source="training", author="sroe",
               payload={"subject": "search ranking v4 update"},
               occurred_at="2026-07-02T00:00:00Z",
               entities=["person:sam-roe"], provenance={})
    cl = Cluster(key="k1", events=[ev], entities=["person:sam-roe"])
    notes_by_name = {
        "ml-search-ranking-v3": Note(
            name="ml-search-ranking-v3", description="v3", type="project",
            domain="ml", entities=["person:jane-doe"],
            project="search-ranking",
        ),
    }
    # Different primary entity, but predicted name has 'search' / 'ranking'
    # namespace tokens matching the peer's namespace.
    assert _infer_project(cl, "ml", notes_by_name) == "search-ranking"


def test_infer_project_namespace_no_match_returns_empty(tmp_path):
    """Layer 2 returns "" when no peer's namespace tokens overlap."""
    cl = Cluster(key="k1", events=[], entities=["person:sam-roe"])
    notes_by_name = {
        "ml-face-detection-v3": Note(
            name="ml-face-detection-v3", description="v3", type="project",
            domain="ml", entities=["person:jane-doe"],
            project="face-detection",
        ),
    }
    assert _infer_project(cl, "ml", notes_by_name) == ""


# ---------------------------------------------------------------------------
# Layer 3 — LLM-proposed namespaces (opt-in)
# ---------------------------------------------------------------------------
class _FakeLLMClient:
    """Minimal stub for testing _propose_project_via_llm."""
    def __init__(self, available=True, response=None, raise_exc=False):
        self._available = available
        self._response = response
        self._raise = raise_exc
    def available(self): return self._available
    def decide(self, prompt, tool, max_tokens=400):
        if self._raise:
            raise RuntimeError("simulated LLM failure")
        return self._response or {}


def test_propose_project_normalize_filters_invalid_slugs():
    """_normalize_project_slug rejects empty, too-long, weird-char inputs;
    _looks_like_project rejects too-generic single tokens."""
    from distill.engine import _normalize_project_slug, _looks_like_project
    # Valid forms.
    assert _normalize_project_slug("search-ranking") == "search-ranking"
    assert _normalize_project_slug("  SEARCH Ranking  ") == "search-ranking"
    assert _normalize_project_slug("face-detection") == "face-detection"
    assert _normalize_project_slug("kernel-security") == "kernel-security"
    # Whitespace -> hyphen.
    assert _normalize_project_slug("kernel security") == "kernel-security"
    # Too many segments.
    assert _normalize_project_slug("a-b-c-d-e") == ""
    # Each segment must be 2-30 chars. Single-char segments rejected.
    assert _normalize_project_slug("a-b") == ""
    assert _normalize_project_slug("a-bb") == ""
    assert _normalize_project_slug("ab-cd") == "ab-cd"
    # Empty / None / weird.
    assert _normalize_project_slug("") == ""
    assert _normalize_project_slug(None) == ""
    assert _normalize_project_slug("a-") == ""
    # _looks_like_project: ≥ 2 segments OR single segment of length ≥ 5.
    # Single-token "model" (length 5) passes the threshold; multi-segment
    # names always pass.
    assert _looks_like_project("model") is True
    assert _looks_like_project("modelling") is True
    assert _looks_like_project("search-ranking") is True
    assert _looks_like_project("kernel-security") is True
    # Truly short single tokens (< 5 chars) are rejected as too generic.
    assert _looks_like_project("ab") is False
    assert _looks_like_project("data") is False


def test_propose_project_returns_first_valid_candidate():
    from distill.engine import _propose_project_via_llm
    cl = Cluster(key="k1", events=[], entities=[])
    client = _FakeLLMClient(response={"candidates": ["search-ranking", "too short"],
                                       "reasoning": "it's search ranking"})
    ns = _propose_project_via_llm(cl, "ml", set(), client)
    # First valid candidate returned.
    assert ns == "search-ranking"


def test_propose_project_skips_collisions():
    from distill.engine import _propose_project_via_llm
    cl = Cluster(key="k1", events=[], entities=[])
    client = _FakeLLMClient(response={"candidates": ["existing", "new-one"],
                                       "reasoning": "..."})
    ns = _propose_project_via_llm(cl, "ml", {"existing"}, client)
    assert ns == "new-one"


def test_propose_project_unavailable_returns_empty():
    from distill.engine import _propose_project_via_llm
    cl = Cluster(key="k1", events=[], entities=[])
    client = _FakeLLMClient(available=False)
    assert _propose_project_via_llm(cl, "ml", set(), client) == ""
    client = _FakeLLMClient(raise_exc=True)
    assert _propose_project_via_llm(cl, "ml", set(), client) == ""


def test_infer_project_layer3_only_with_auto_create(tmp_path):
    """Layer 3 (LLM-proposed) only fires when auto_create=True AND client given."""
    cl = Cluster(key="k1", events=[], entities=["person:newbie"])
    notes_by_name = {}  # no peers
    # Without client/auto_create: returns "".
    assert _infer_project(cl, "ml", notes_by_name) == ""
    assert _infer_project(cl, "ml", notes_by_name, client=None, auto_create=False) == ""
    # With client + auto_create: the LLM is asked.
    fake = _FakeLLMClient(response={"candidates": ["partner-pilot"], "reasoning": "..."})
    assert _infer_project(cl, "ml", notes_by_name, client=fake, auto_create=True) == "partner-pilot"
