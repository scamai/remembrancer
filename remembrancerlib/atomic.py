"""Durable atomic replacement helpers for small machine-readable state files."""
from __future__ import annotations

import json
import os
import stat
import tempfile
from typing import Any


def durable_replace_bytes(
    path: str,
    payload: bytes,
    *,
    mode: int | None = None,
    prefix: str = ".remembrancer-state-",
) -> None:
    """Durably replace *path* without exposing a partial file.

    The complete payload is written and fsynced in the destination directory,
    then promoted with one ``os.replace`` and followed by a directory fsync.
    If serialization, writing, fsync, or the replace itself fails before that
    commit point, an existing destination remains byte-for-byte untouched.
    """
    if not isinstance(payload, bytes):
        raise TypeError("atomic replacement payload must be bytes")
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        if mode is None:
            mode = 0o644
    else:
        if not stat.S_ISREG(current.st_mode):
            raise ValueError(f"atomic replacement target is not a regular file: {path}")
        if mode is None:
            mode = stat.S_IMODE(current.st_mode)
    if not isinstance(mode, int) or mode < 0 or mode > 0o7777:
        raise ValueError(f"invalid atomic replacement mode: {mode!r}")

    descriptor, temporary = tempfile.mkstemp(
        prefix=prefix, suffix=".tmp", dir=directory,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), mode)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def durable_replace_json(
    path: str,
    value: Any,
    *,
    mode: int | None = None,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """Serialize JSON completely, then durably replace *path* atomically."""
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=indent,
            sort_keys=sort_keys,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    durable_replace_bytes(path, encoded, mode=mode, prefix=".remembrancer-json-")
