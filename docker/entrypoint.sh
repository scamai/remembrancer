#!/bin/sh
# Commands of the Remembrancer image. compose.yaml mounts the memory folders
# and your config.json into /app, where the code reads them.
set -eu
cd /app

usage() {
  cat <<'EOF'
Usage: docker compose run --rm remembrancer <command>

  demo                   build a fictional repository, index it, and ask it a question
  ingest [PATH ...]      ingest, distill, lint and index git repositories mounted into
                         the container; with no PATH, the repositories config.json
                         lists under "monorepos" (a URL or an owner/name each)
  query [ARGS ...]       ask the memory: query --as dana --q "who owns ledger-service"
  mcp                    the MCP server over stdio (docker compose run --rm -T ...)
  mcp-http               the MCP server over streamable HTTP on port 8790, for a team
                         behind a reverse proxy (compose's http profile; SECURITY.md)
  help                   this text

Any other command runs as given, inside /app.
EOF
}

cmd=${1:-help}
[ $# -gt 0 ] && shift
case "$cmd" in
  help|-h|--help)
    usage; exit 0 ;;
esac

if [ ! -f config.json ]; then
  echo "No config.json in /app. Create one next to compose.yaml first:" >&2
  echo "  cp config.example.json config.json" >&2
  exit 2
fi

case "$cmd" in
  demo)
    [ -d /tmp/demo-repo ] || examples/demo-repo.sh /tmp/demo-repo
    python scripts/run.py --ingest-git /tmp/demo-repo --lint
    exec python retrieve/query.py --as dana --q "who is on call for ledger-service" ;;
  ingest)
    exec python scripts/run.py --ingest-git "$@" --lint ;;
  query)
    exec python retrieve/query.py "$@" ;;
  mcp)
    exec python mcp/server.py ;;
  mcp-http)
    export REMEMBRANCER_MCP_TRANSPORT=http REMEMBRANCER_MCP_HOST=0.0.0.0
    exec python mcp/server.py ;;
  *)
    exec "$cmd" "$@" ;;
esac
