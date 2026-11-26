"""Shared ACL sentinels.

Only identity resolution may return ``FULL_ACCESS``.  Consumers must compare by
object identity; ``None`` and malformed grant values always fail closed.
"""

FULL_ACCESS = object()
DENY = frozenset()
