"""Per-source watermark cursors — make every connector & the distiller
incremental and idempotent. Stored as one JSON file (state/cursors.json).

A connector records "last item I pulled" (e.g. a Slack ts, a Gmail historyId,
a git sha); the distiller records "last event id I processed". Re-running then
resumes from the watermark instead of reprocessing everything.
"""
from __future__ import annotations

import json
import os
import fcntl
import tempfile


class Cursors:
    def __init__(self, path: str = "state/cursors.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._data = self._read()

    @property
    def _lock_path(self) -> str:
        return self.path + ".lock"

    def _read(self) -> dict[str, str]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            raise RuntimeError(f"cursor store is corrupt: {self.path}") from exc
        if not isinstance(data, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                             for k, v in data.items()):
            raise RuntimeError(f"cursor store has invalid schema: {self.path}")
        return data

    def get(self, key: str, default: str | None = None) -> str | None:
        with open(self._lock_path, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            self._data = self._read()
            return self._data.get(key, default)

    def set(self, key: str, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("cursor values must be strings")
        directory = os.path.dirname(self.path) or "."
        with open(self._lock_path, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            current = self._read()
            current[key] = value
            fd, temp_path = tempfile.mkstemp(prefix=".cursors-", suffix=".tmp", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(current, handle, indent=2, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, self.path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            self._data = current
