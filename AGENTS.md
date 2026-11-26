# Remembrancer operating spec

The operating spec for agents and contributors who maintain a Remembrancer
deployment's memory. The distiller and lint read the `conventions` block at the
end of this file through `distill.engine.load_spec`.

## Layers, commands and configuration

Remembrancer is event-sourced. L1 events are append-only JSON lines in
`events/YYYY-MM-DD.jsonl`. L2 notes are typed markdown files in `context/<domain>/`,
indexed by `context/INDEX.md` and a per-domain `index.md`. `state/` holds cursors,
checkpoints, queues and audit logs; `registry/` holds entity records. These and
`config.json` are deployment data that `.gitignore` keeps out of the code history.
Config keys below live in `config.json`; start from `config.example.json`.

- `make run` ingests this repository's git history, then guards, distills, lints
  and reindexes (`scripts/run.py`). `python3 scripts/run.py --ingest-git --lint`
  ingests the repositories in the `repos` and `monorepos` keys instead.
- `make query Q="release checklist" IDENT=dana` runs ACL-filtered retrieval (an
  empty `IDENT` is the local operator, who reads every note). `make lint`,
  `make mcp` (MCP over stdio) and `make test` (hermetic end-to-end) also exist.

v0.1 runs without an LLM: distillation is deterministic, and notes distilled from
git carry no people or component tags. An optional LLM at any Anthropic-compatible
endpoint can judge consolidation and draft writes; distill uses it whenever
`REMEMBRANCER_LLM_API_KEY` and `REMEMBRANCER_LLM_MODEL` are set
(`REMEMBRANCER_LLM_BASE_URL` names an endpoint other than the SDK's own). Every
other LLM setting is one `REMEMBRANCER_LLM_*` variable too: temperature, timeout,
the fan-out, retries, and where the client logs usage, failures and its health.
`distill/llm_env.py` lists them; the logs default to `state/llm_usage.jsonl`,
`state/llm_failures.jsonl` and `state/llm_health.json`.

With no `config.json`, search shows no machine-written note: the verification gate
(`trust.hide_verification_failures`, on by default) hides every note without a
passing verification, and without an LLM nothing is verified.
`config.example.json` turns the gate off, so distilled notes appear as
`proposed`. The notes `govern.py` holds, decisions among them, still wait in
`state/approval_queue.jsonl`; v0.1 has no command to approve them (see "Approval
and governance"), and `governance.admission_mode: automatic` shows them as
proposed instead.

## Note types

Distill writes one of four atomic types, picked by keywords in the cluster
(`distill.engine._note_type`):

| Type | Picked for |
|---|---|
| `project` | Bounded work with a goal and a status; the default. |
| `decision` | decided, chose, agreed, adopted, deprecated, policy, standard. |
| `state` | state, status, current, now, baseline, version. |
| `reference` | doc, reference, guide, how-to, spec. |

Other producers use the rest of `remembrancerlib.note.VALID_TYPES`: `entity`,
`concept`, `source-summary`, `synthesis` (file-back answers), `contradiction`
(lint findings), `page`, `artifact`, `initiative`, `correction` (MCP edits). The
write door (`remembrancerlib.note.write_note`) requires a unique `name` matching
`[a-z0-9][a-z0-9-]*`, a one-line `description` of at most 280 characters (full
text goes in the body), valid `type`, `domain` and `state`, single-line values
without commas in list items, and `state: superseded` for `superseded: true`.

## Ingest

- An event has `id`, `source`, `author`, `occurred_at`, `ingested_at`, `payload`,
  `acl`, `entities` and `provenance` (`remembrancerlib/event.py`); the date that
  starts `occurred_at` picks the day file. Sources: `git`, `slack`, `gmail`,
  `gcal`, `gdrive`, `crm`, `ci`, `training`, `session`, `meeting`, `artifact`,
  `okf`, `upload`, `decision`. v0.1 ships one connector, `ingest/git_events.py`:
  one event per non-merge commit, id `git:<full sha>`, author `Name <email>`.
- `id` is `<source>:<natural-id>`, the corpus-wide dedup key. Mutable objects add
  a stable revision, so a replay is a no-op and a correction is a new event.
  Events are write-once: a reused id with different data is refused.
- Connectors follow `ingest/base.py` (fetch, filter, normalize), keep their
  watermark in `state/cursors.json` and read credentials only from the
  environment. `acl_defaults` labels each source; a source with no entry gets
  `["eng"]`, and an empty list means public.
- The corpus guard (`tools/corpus_guard.py`) proves each pass that every byte it
  validated before is unchanged, checks the new tail (schema, id conflicts,
  `--max-new-events`, `--future-days`), then records each file's size, line count
  and SHA-256 in `state/event_validation_checkpoint.json`. Distill reads only
  validated prefixes, and a missing or modified checkpoint fails closed. `make run`
  bootstraps it only for an empty store; otherwise run the attended
  `python3 tools/corpus_guard.py --bootstrap-checkpoint` once.

## Entity registry

- Split mode reads `registry/orgs.yml`, `registry/people.yml`,
  `registry/systems.yml` and `registry/noise.yml` (drops and review hints, not
  parties) once `registry/orgs.yml` exists and PyYAML is installed; otherwise it
  reads `registry/entities.yml`. Without PyYAML the registry reads as empty.
- Ids are canonical (`person:dana-okafor`, `org:<slug>`, `system:<slug>`).
  Ambiguous names, emails and domains stay unattributed, never guessed. A record
  whose kind looks wrong is quarantined in memory and queued, never retyped.
- An organization's `relationship` steers routing, and fields like `works_at` or
  `champions` become graph edges, only when `field_provenance.<field>` has
  `status: verified` plus non-empty `evidence` and `reviewed_by`. A deterministic
  producer may sign (`producer:<name>`) only what its evidence proves exactly.
  Lint publishes the review queue at `state/registry_review_queue.json`.
- `status: former` keeps a person resolving for old history, but
  `people_registry.internal_people()` omits them unless `include_former=True`.

## Distill

- Distillation runs at write time only; retrieval never triggers it. Intake
  valves such as `distill.git_max_age_days` skip events without deleting them.
- Each cluster gets one action per pass (`CREATE`, `UPDATE`, `LINK`, `SUPERSEDE`,
  `IGNORE`), idempotent on replay. Without the LLM, distill ignores clusters
  already filed, noise or bulk, updates a matching note, links a non-durable
  cluster related to two or more notes, and otherwise creates one. With it,
  durable clusters go to the model with up to `distill.candidate_k` candidates,
  and a missing or rejected answer falls back to the deterministic path.
- `UPDATE` appends provenance and a dated `Update YYYY-MM-DD:` entry to the
  append-only body; the description becomes the newest truth unless the note is
  human-authored (`by:` provenance). New and changed notes are `proposed`.
- Distill never edits a `verified` note, or a human-authored one it does not own:
  it writes a proposed `<name>-update` with `[[supersedes:<name>]]` and a
  `machine-update:<name>` marker instead.
- `SUPERSEDE` writes a proposed replacement with `[[supersedes:<old-name>]]` and
  leaves the old note untouched. The edge counts only once an approver verifies
  the replacement, and hides the old note only from callers who can see the
  replacement; demoting or tombstoning it brings the old note back.
- The watermark lives in `state/cursors.json` (`make distill-rebuild` replays
  everything). `distill.verify: true` checks each written note against its source
  events; it needs the LLM, and distill refuses to run without one.
- Routing: `domain` follows what a note is about, not who sent it. Values come
  from `distill.consolidate.DOMAINS`, plus `people`, `design` and `synthesis` for
  producer-owned notes. A registered external party's type decides first
  (`distill.engine.TYPE_DOMAIN`), and a noise-only cluster is ignored; otherwise
  the source does (`git` goes to `eng`) unless a repository sets its `domain`.
- Generated pages write through `tools/generated_notes.py` with a
  `generated-by:<producer>` marker and may rewrite only a `proposed` note with
  exactly that marker and no `by:` marker. The marker is ownership, not evidence.

## Lint

- `make lint` (`lint/engine.py`) reports schema problems (missing `name`,
  `description` or `type`; an overlong description; a bad slug, type, state or
  domain) without fixing them. `--quarantine` moves notes missing required fields
  to `state/quarantine/context/` with an append-only manifest, within a cap.
- It also reports orphans, dangling and unsafe links, supersession edges and
  contradictions, and writes ACL-safe backlinks, the registry review queue and
  fresh indexes. `--fail-on-schema` and `--fail-on-unsafe-links` fail the run.
- A tombstone in `state/suppressed.json` soft-deletes a note: reads skip it, the
  write door refuses to recreate it, and a malformed file fails closed.

## Retrieval

- Reads are ACL-gated: `identities` maps each identity to labels, and a note is
  visible when its `acl` is empty or shares a label. The query CLI's operator (no
  `--as`) reads everything; MCP denies by default, so no identity or an unknown
  one reads only public notes. A denied note looks exactly like a missing one.
  `SECURITY.md` covers how the MCP server learns its caller's identity.
- Search never writes `events/`, `context/` or `registry/` or runs distillation;
  it logs to `state/queries.jsonl` only if `retrieval.query_log` is `true`.
- Ranking is BM25 over `name`, `description`, `tags` and up to 16 index keys
  (distinctive body terms), read from the per-domain indexes first, with a body
  re-rank of the top hits; ties go to the newer `updated`. Superseded notes are
  hidden, and proposed ones are scored down by `trust.proposed_rank_penalty`.
- One time phrase ("last week", "in July", "since <ISO date>", "recently") is
  parsed without an LLM (`retrieve/timeparse.py`) and filters on `updated`.
- Memory is background and history: for a codebase's present state, the checkout
  beats a stale note. `retrieval.workspace_evidence` adds read-only snippets from
  configured Git workspaces, each with its own `acl`, to MCP search.
- When a user insists something is missing, `--insist` (MCP `remembrancer_dig`)
  returns grounded, unverified L1 candidates. `remembrancer_confirm_answer`
  queues a confirmed one with citations; `tools/fallback_queue.py
  --promote-confirmed --apply` writes it (`make promote-confirmed` is a dry run).
- `remembrancer_file_answer` and `remembrancer_remember` write `synthesis` notes,
  or a `correction` with a supersedes edge, gated by `retrieval.file_back`. A
  filed answer gets only the labels every restricted source shares, so no one
  reads it who could not read each source; with none in common it is refused. A
  remembered note gets the labels of its domain: `authored_acl` in `config.json`
  adds to or overrides the defaults in `retrieve/query.py`, and a domain neither
  names gets `secrets`. The writer needs one of the note's labels; an empty list
  lets any mapped identity write it, and makes the note public.

## Approval and governance

- Trust states are `proposed` (also a note with no state), `verified` and
  `superseded`. `promote`, `demote` and `reject` in `retrieve/query.py` make the
  moves; v0.1 has no command-line or MCP door for them.
- The approver must be in `trust.approver_identities` and `identities` and hold a
  grant on the note's ACL. The signer's Ed25519 key comes from
  `REMEMBRANCER_APPROVAL_PRIVATE_KEY`, never the repository; readers verify with
  `trust.approval_public_key` (optionally pinned by `trust.approval_key_id`).
  Each transition appends a signed row to `state/trust_audit.jsonl`.
- A signature binds the note's current content, `updated` included. A bare
  `state: verified` without one is denied, and a model's verdict alone never
  makes a note answerable.
- `governance.admission_mode` defaults to approval. `govern.py` then holds, and
  queues in `state/approval_queue.jsonl`, a note that carries a restricted label
  (`secrets`, plus any listed in `governance.restricted_labels`), has an email
  author outside `governance.internal_domains` (all are outside when none is
  set), is a decision, or is a customer state or deal note.
- Machine-written notes start with verification `skipped`, which
  `trust.hide_verification_failures` (on by default) hides until approval.
  Turning that gate off shows notes without a hold; held notes stay hidden.
- `automatic` holds nothing: every new note is admitted at once when ACLs, signed
  lifecycle and the trust audit allow. Admission is not verification: notes stay
  `proposed` with the rank penalty, and nothing is signed, rewritten or deleted.
  The queue's existing entries are kept, so switching back to approval restores
  their holds.
- `trust.auto_approve: true` lets deterministic checks or a current verifier
  result make a journaled machine approval, never for a note with a restricted
  label.
- `trust.verification_audit_required: true` makes missing verification journals
  fail closed. Journals catch corruption, not one account rolling back a journal
  and its anchors together; copy them to append-only storage on another host.
- Verdicts are appended to `state/verification.jsonl`, which holds at most
  128 MiB. Once it is within 16 MiB of that, `scripts/run.py` seals it into a
  numbered segment before the run appends anything, and readers fold the
  segments back in order. A distill started any other way (`make distill`) never
  seals it, and a full journal records no new verdict.

## Contradictions

- A contradiction is two current notes making mutually exclusive claims about one
  entity or concept. Lint finds them from an explicit `[[contradicts:<name>]]`
  link, or heuristically: a shared entity plus opposite sides of a known antonym
  pair in the descriptions.
- `contradiction_policy` (below) decides the outcome; unknown values act as `flag`.
  - `flag`: write a `type: contradiction` note `contradiction-<a>-vs-<b>` with
    the common ACL of both notes; both originals stay intact.
  - `reconcile`: resolve a pair only when one note already supersedes the other
    through an active, verified replacement the ACLs allow; flag every other
    pair. It never merges notes.
  - `suppress`: report nothing. For tests only.

## Cross-references

- Links are wiki-links in bodies: `[[note-name]]`, or typed
  `[[relation:note-name]]`. A plain link means `relates`; the vocabulary is open.
- A link reveals its target's name, so A may link to B only when every reader of
  A may read B (`acl_can_reveal(A.acl, B.acl)`). The write door refuses an unsafe
  link but allows one to a missing note; lint reports dangling links, never
  removing them.
- Lint adds an untyped backlink `[[source]]` only when the reverse direction
  passes the same check, a bounded number of notes per pass; a verified or
  human-authored target gets a replacement proposal instead.
  `file_back_enabled: false` turns backlinks off, for read-only or test use only.
- A page built from several notes takes their common grants (`combine_acl`).
  Labels are OR grants, so a union would leak; with no common grant
  `combine_acl` raises `ACLConflict`. Split such a page by audience, and never
  substitute `secrets`, which is a different audience, not an AND.

## Machine-readable conventions

`distill.engine.load_spec` reads this file from the repository root. It starts at
the first line that is exactly ```` ```conventions ````, stops at the next line
that is exactly ```` ``` ````, and parses each `key: value` line between them;
`true`/`false` (also `yes`/`no`, `on`/`off`, `1`/`0`) become booleans. A key
missing here, a missing block or a missing file falls back to the defaults
`contradiction_policy: flag` and `file_back_enabled: true`.

Keep this strict `key: value` format: one key per line, no nested structures and
no comments inside the block. Lines that do not match are ignored, and an inline
comment would become part of the value.

Config split: behavioural policies (`contradiction_policy`, `file_back_enabled`)
are read from this file; numbers and operational settings (retrieval pool sizes,
`retrieval.file_back.min_chars`, identities and ACLs, distill settings) live in
`config.json`.

```conventions
contradiction_policy: flag
file_back_enabled: true
```
