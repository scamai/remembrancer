#!/usr/bin/env bash
# Builds a small fictional git repository to try the pipeline on: one team at
# "Acme", ten commits over three weeks. Every name and address is made up.
#
#   examples/demo-repo.sh [dir]     # default: ./demo-repo
set -euo pipefail
R=${1:-demo-repo}
if [ -e "$R" ]; then echo "$R already exists" >&2; exit 1; fi
mkdir -p "$R"
git -C "$R" init -q -b main

commit() {  # name email date message file line
  mkdir -p "$R/$(dirname "$5")"
  printf '%s\n' "$6" >> "$R/$5"
  git -C "$R" add "$5"
  GIT_AUTHOR_NAME="$1" GIT_AUTHOR_EMAIL="$2" GIT_AUTHOR_DATE="$3" \
  GIT_COMMITTER_NAME="$1" GIT_COMMITTER_EMAIL="$2" GIT_COMMITTER_DATE="$3" \
    git -C "$R" -c commit.gpgsign=false commit -q -m "$4"
}

DANA=("Dana Okafor" "dana@acme.example")
LEE=("Lee Park" "lee@acme.example")
SAM=("Sam Rivera" "sam@acme.example")

commit "${DANA[@]}" 2026-10-20T09:00:00Z "chore: initialize ledger-service skeleton" \
  ledger-service/README.md "# ledger-service: Acme's double-entry ledger"
commit "${LEE[@]}" 2026-10-21T10:30:00Z "feat(ledger-service): add double-entry posting API" \
  ledger-service/api.py "def post(entry): ..."
commit "${DANA[@]}" 2026-10-22T14:00:00Z "decision: adopt PostgreSQL 16 as the ledger-service store" \
  docs/decisions/0001-postgres.md "We adopt PostgreSQL 16 for ledger-service. Owner: Dana Okafor."
commit "${SAM[@]}" 2026-10-24T08:15:00Z "docs: Sam Rivera becomes on-call owner for ledger-service" \
  OWNERS.md "ledger-service: Sam Rivera (on-call), Dana Okafor (tech lead)"
commit "${LEE[@]}" 2026-10-27T16:45:00Z "fix(ledger-service): resolve rounding error in currency conversion" \
  ledger-service/fx.py "ROUNDING = 'bankers'"
commit "${DANA[@]}" 2026-10-29T11:00:00Z "decision: deprecate the v1 webhook API by 2027-01-31" \
  docs/decisions/0002-webhook-v1.md "The v1 webhook API is deprecated; removal on 2027-01-31."
commit "${LEE[@]}" 2026-11-02T13:20:00Z "feat(ledger-service): idempotency keys on POST /postings" \
  ledger-service/api.py "IDEMPOTENCY_HEADER = 'Idempotency-Key'"
commit "${SAM[@]}" 2026-11-04T09:40:00Z "eval: reconciliation baseline at 99.2% match on the March sample" \
  evals/reconciliation.md "baseline: 99.2% match"
commit "${DANA[@]}" 2026-11-06T17:00:00Z "release: ledger-service v0.3.0 shipped to staging" \
  CHANGELOG.md "v0.3.0: idempotency keys, rounding fix"
commit "${SAM[@]}" 2026-11-07T10:05:00Z "policy: every schema migration needs two reviewers" \
  docs/policies/migrations.md "Two reviewers for every schema migration."

echo "built $R: $(git -C "$R" rev-list --count HEAD) commits"
