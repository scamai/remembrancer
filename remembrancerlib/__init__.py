"""remembrancerlib — the shared core contract every Remembrancer module imports.

Defines the two record types (Event, Note), the append-only event store,
the note/frontmatter/index helpers, the per-source watermark cursor, and the
governance-driven ACL labeler. Stdlib-only (PyYAML optional for governance.yml).

Modules build on THIS; they never re-implement the schema.
"""
from .event import (
    Event, EventConflictError, EventSchemaError, EventValidationError, append_events, iter_events,
    iter_validated_events, iter_validated_event_records, existing_ids, event_text, event_store_lock,
    event_validation_checkpoint_path, event_validation_status,
    load_event_validation_checkpoint,
)
from .note import (Note, parse_frontmatter, load_notes, index_notes_by_name, write_note, regen_index,
                   render_index_artifacts, compute_index_keys,
                   slugify, resolve_links, load_suppressed, is_suppressed,
                   suppression_lock, effective_state,
                   supersession_head_active, active_supersession_targets,
                   current_note_names, concise_description,
                   forward_link_validation_error,
                   MAX_DESCRIPTION_CHARS, VALID_STATES)
from .cursor import Cursors
from .redact import redact_event, redact_payload, redact_text
from .timeutil import clamp_iso_to_now
from .acl import (ACLConflict, Labeler, acl_can_reveal, combine_acl,
                  restrict_sensitive_acl)
from .access import FULL_ACCESS, DENY
from .locks import (PipelineBusyError, pipeline_mutation_lock,
                    artifact_publish_lock, serialized_pipeline_mutation)
from .atomic import durable_replace_bytes, durable_replace_json
from .verification import (
    HumanTrustAudit,
    PRODUCER_GROUNDING_PREFIX,
    auto_approval_valid,
    bind_loaded_auto_approvals,
    bind_loaded_human_trust,
    ensure_machine_verification_binding,
    human_trust_valid,
    human_trust_write_authorized,
    invalidate_verification,
    load_human_trust_audit,
    machine_authored,
    producer_verification_valid,
    verification_content_fingerprint,
)
from .source_lifecycle import (
    SourceLifecycleAudit,
    bind_loaded_source_retirements,
    load_source_lifecycle_audit,
    source_retirement_valid,
)
from .producer_lifecycle import (
    ProducerLifecycleAudit,
    bind_loaded_producer_lifecycles,
    load_producer_lifecycle_audit,
    producer_lifecycle_valid,
)
# Aliased on import: ``fold_relations`` / ``load_relation_decisions`` keep the
# names distinct from another ``load_decisions`` that the artifact graph's
# builder still imports under its own name.
from .artifact_relations import (
    LEDGER_REL_PATH,
    RELATION_TYPES,
    append_decision,
    current_relations,
    decision_id,
    decisions_for,
)
from .artifact_relations import fold as fold_relations
from .artifact_relations import load_decisions as load_relation_decisions

__all__ = [
    "Event", "EventConflictError", "EventSchemaError", "EventValidationError", "append_events", "iter_events",
    "iter_validated_events", "iter_validated_event_records", "existing_ids", "event_text", "event_store_lock",
    "event_validation_checkpoint_path", "event_validation_status",
    "load_event_validation_checkpoint",
    "Note", "parse_frontmatter", "load_notes", "index_notes_by_name", "write_note", "regen_index", "render_index_artifacts", "compute_index_keys", "slugify", "resolve_links", "load_suppressed", "is_suppressed", "suppression_lock",
    "effective_state", "supersession_head_active", "active_supersession_targets",
    "current_note_names", "VALID_STATES",
    "concise_description", "forward_link_validation_error", "MAX_DESCRIPTION_CHARS",
    "Cursors", "clamp_iso_to_now", "redact_event", "redact_payload", "redact_text", "ACLConflict", "Labeler", "acl_can_reveal",
    "combine_acl", "restrict_sensitive_acl",
    "FULL_ACCESS", "DENY", "PipelineBusyError", "pipeline_mutation_lock",
    "artifact_publish_lock",
    "serialized_pipeline_mutation", "durable_replace_bytes", "durable_replace_json",
    "invalidate_verification", "verification_content_fingerprint",
    "ensure_machine_verification_binding",
    "machine_authored",
    "PRODUCER_GROUNDING_PREFIX", "producer_verification_valid",
    "HumanTrustAudit", "load_human_trust_audit", "human_trust_valid",
    "bind_loaded_human_trust",
    "human_trust_write_authorized",
    "auto_approval_valid", "bind_loaded_auto_approvals",
    "SourceLifecycleAudit", "load_source_lifecycle_audit",
    "source_retirement_valid", "bind_loaded_source_retirements",
    "ProducerLifecycleAudit", "load_producer_lifecycle_audit",
    "producer_lifecycle_valid", "bind_loaded_producer_lifecycles",
    "RELATION_TYPES", "LEDGER_REL_PATH", "decision_id", "append_decision",
    "load_relation_decisions", "current_relations", "fold_relations",
    "decisions_for",
]
