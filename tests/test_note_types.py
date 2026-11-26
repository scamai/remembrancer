"""Test for enriched note type vocabulary (OKF page typology)."""
from remembrancerlib.note import Note, VALID_TYPES


def test_entity_type_valid():
    """Test that new page types (entity, concept, etc.) are in VALID_TYPES."""
    assert "entity" in VALID_TYPES
    assert "concept" in VALID_TYPES
    assert "source-summary" in VALID_TYPES
    assert "synthesis" in VALID_TYPES
    assert "contradiction" in VALID_TYPES
    assert "artifact" in VALID_TYPES
    assert "initiative" in VALID_TYPES
    assert "correction" in VALID_TYPES


def test_entity_note_serialization():
    """Test that a Note with type='entity' serializes correctly to markdown."""
    n = Note(name="acme", description="customer", type="entity", domain="customers")
    markdown = n.to_markdown()
    assert "type: entity" in markdown
