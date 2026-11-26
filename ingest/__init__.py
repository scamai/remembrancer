"""ingest — one connector per source (L0 -> L1).

Git, Slack, Gmail, Calendar, Drive, CRM, OKF, meeting write-back, and artifact
publishing each implement the same four-stage contract (see base.py) and emit
the same Event shape into events/YYYY-MM-DD.jsonl. See remembrancer/CLAUDE.md.

Was an implicit namespace package (no __init__.py) until this file was added
so setuptools package discovery in pyproject.toml has a real package to find.
Purely additive: `import ingest.xxx` behaved identically before and after —
Python treats a directory with a matching name as importable either way, and
nothing here changes what that import resolves to.
"""
