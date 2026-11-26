# Remembrancer — the company context fabric. Developer entrypoints.
# ---------------------------------------------------------------------------
# The pipeline mirrors the layers:  ingest (L0->L1)  ->  distill (L1->L2)  ->
# index (regenerate L2 INDEX)  ->  query (L3 retrieval).  `make test` runs the
# whole chain against a temporary synthetic git source with NO external creds;
# it never mutates the checked-in memory corpus.
#
# Override identity/query at the CLI:   make query Q="release checklist"
# (IDENT, not AS — `AS` is reserved by make for the assembler and gets clobbered
# by the environment.)
# ---------------------------------------------------------------------------

PYTHON  ?= python3
# Empty IDENT = the query CLI's local operator, who reads every note.
IDENT   ?=
Q       ?= context
SINCE   ?= 30 days ago
CONTEXT ?= context

.DEFAULT_GOAL := help
.PHONY: help install ingest distill distill-rebuild index query mcp test clean run lint promote-confirmed

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (stdlib-first; these are the optional deps)
	$(PYTHON) -m pip install -r requirements.txt

ingest: ## L0->L1: pull this repo's git history into events/
	$(PYTHON) ingest/git_events.py --repo . --since "$(SINCE)"

distill: ## L1->L2: cluster events into context/ notes (+ rewrites INDEX.md)
	$(PYTHON) distill/engine.py --root .

distill-rebuild: ## L1->L2: re-distill ALL events ignoring stored watermark
	$(PYTHON) distill/engine.py --root . --rebuild

index: ## Regenerate context/INDEX.md from note frontmatter
	$(PYTHON) -c "from remembrancerlib import regen_index; print(regen_index('context'))"

query: ## L3: retrieve an ACL-filtered slice.  IDENT=<identity> Q="..."
	PYTHONPATH=. $(PYTHON) retrieve/query.py --as "$(IDENT)" --q "$(Q)"

mcp: ## Run the Remembrancer MCP server (stdio) for Claude Code / Cursor
	$(PYTHON) mcp/server.py


test: ## Hermetic smoke: temp git -> ingest -> guard -> distill -> lint/index/query
	PYTHONPATH=. $(PYTHON) -m pytest -q tests/test_integration_v2.py

run: ## Run full local pipeline: ingest git + distill + lint + index
	$(PYTHON) scripts/run.py --ingest-git . --lint

lint: ## Lint context notes
	$(PYTHON) -m lint.engine --root .

clean: ## Remove derived/cache artifacts (events + context stay — they're data)
	find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	rm -rf _dispatch 2>/dev/null || true


promote-confirmed: ## DRY-RUN: attended confirmed-fallback curation (--apply is never run automatically)
	$(PYTHON) tools/fallback_queue.py --promote-confirmed

