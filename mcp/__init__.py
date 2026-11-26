"""Remembrancer MCP server package.

Exposes Remembrancer's distilled memory (L3 serving) to coding agents (Claude Code,
Cursor) and a Slack bot through one standard MCP tool surface:
remembrancer_search · remembrancer_get_note · remembrancer_timeline. ACL is enforced per call via
the acting `as_identity`. See server.py and README.md.
"""
