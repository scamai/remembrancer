"""OKF bundle validator for Remembrancer context directories.

OKF (Open Knowledge Format) hard rule: every concept file must have a
non-empty `type` frontmatter field.

Usage (CLI):
    python -m tools.okf_validate            # validates context/ in cwd
    python tools/okf_validate.py [dir]      # validates arbitrary dir

Exit code: 0 if bundle is valid, 1 if violations were found.
"""
from __future__ import annotations

import os
import sys

# Ensure repo root is importable when run as a plain script from any cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from remembrancerlib import parse_frontmatter  # noqa: E402  (after sys.path fixup)

_INDEX_NAMES = {"index.md", "INDEX.md", "_lint_report.md"}


def validate_bundle(context_dir: str) -> list[str]:
    """Walk all .md files under *context_dir*, skipping index.md/INDEX.md.

    For each file, parse frontmatter and check that the ``type`` field is
    present and non-empty.  Return a list of violation strings of the form::

        "<path>: missing required 'type'"

    An empty return value means the bundle is a valid OKF bundle.
    """
    violations: list[str] = []

    if not os.path.isdir(context_dir):
        return violations

    for dirpath, _dirs, files in os.walk(context_dir):
        for filename in files:
            if not filename.endswith(".md"):
                continue
            if filename in _INDEX_NAMES:
                continue

            path = os.path.join(dirpath, filename)
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                # Unreadable file counts as a violation.
                violations.append(f"{path}: missing required 'type'")
                continue

            fm, _body = parse_frontmatter(text)
            note_type = fm.get("type", "")
            if not note_type or not str(note_type).strip():
                violations.append(f"{path}: missing required 'type'")

    return violations


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "context"
    issues = validate_bundle(target)
    if issues:
        for v in issues:
            print(v)
        sys.exit(1)
    else:
        print(f"OK — {target} is a valid OKF bundle.")
        sys.exit(0)
