"""Remembrancer retrieval layer — index-first, ACL-gated, BM25-ranked (stdlib only).

The read path of the 3-layer memory: load distilled L2 notes, filter by the acting
identity's ACL grants, BM25-rank a recall-wide candidate pool, and deliver an
index-first slice (progressive disclosure — metadata before bodies). No embeddings,
no vector DB, no LLM call at read time.

Public surface (reused by the MCP server, mcp/server.py):
    from retrieve import search, get_note, grants_for, visible
    from retrieve import BM25, rank
"""
from retrieve.bm25 import BM25, rank, tokenize
from retrieve.query import search, get_note, grants_for, visible

__all__ = [
    "BM25", "rank", "tokenize",
    "search", "get_note", "grants_for", "visible",
]
