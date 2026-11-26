#!/usr/bin/env python3
"""Deterministic repository profiler used by the governed engineering-wiki generator.

``profile_repo`` reads a local checkout without executing repository code and returns a
bounded structural profile.  A weekly engineering-wiki projection consumes that
data and writes through the shared generated-note ownership and grounding gate.

The older GitHub+LLM org-map helpers remain importable for read-only compatibility, but
their direct L2 note writer and this module's legacy CLI write mode are deliberately refused.
Durable facts must enter through L1 connectors and digest; generated repo pages must use the
governed engineering-wiki projection.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.llm import LLMClient, LLMUnavailable      # noqa: E402


class LegacyRepoProfileWriteDisabled(RuntimeError):
    """Raised when obsolete code attempts to bypass the governed memory write path."""


_WRITE_DISABLED_MESSAGE = (
    "direct repo-profile L2 writes are retired; ingest durable facts through L1 connectors "
    "and digest"
)

PROFILE_TOOL = {
    "name": "repo_profile",
    "description": "Profile what a software repository is and does.",
    "input_schema": {
        "type": "object",
        "properties": {
            "one_liner": {"type": "string", "description": "<=14 words: what this repo is"},
            "what_it_does": {"type": "string", "description": "2-4 sentences: purpose and what it produces"},
            "stack": {"type": "string", "description": "languages/frameworks if evident"},
            "status": {"type": "string", "description": "active/prototype/deprecated + maturity if evident"},
        },
        "required": ["one_liner", "what_it_does"],
    },
}


def _gh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["gh"] + args, capture_output=True, text=True)


def list_active_repos(org: str) -> list[dict]:
    out = _gh(["repo", "list", org, "--limit", "300", "--json",
               "name,description,isArchived,primaryLanguage,pushedAt"]).stdout
    repos = json.loads(out or "[]")
    active = [r for r in repos if not r.get("isArchived")]
    return sorted(active, key=lambda r: r.get("pushedAt", ""), reverse=True)


def fetch_readme(org: str, name: str) -> str:
    rd = _gh(["api", f"repos/{org}/{name}/readme", "-H", "Accept: application/vnd.github.raw"])
    return rd.stdout[:6000] if rd.returncode == 0 else ""


def profile_repo_gh(org: str, repo: dict, client: LLMClient) -> dict:
    """Return {one_liner, what_it_does, stack, status}. Deterministic fallback w/o key.

    (The org-MAP profiler: uses `gh` + an LLM to summarize what a repo IS. The DEEP
    structural profiler that reads a local clone is ``profile_repo`` below.)
    """
    name = repo["name"]
    desc = repo.get("description") or ""
    lang = (repo.get("primaryLanguage") or {}).get("name", "")
    readme = fetch_readme(org, name) or "(no README)"
    if not client.available():
        return {"one_liner": desc or f"{name} ({lang})",
                "what_it_does": desc or "(no description / README available)",
                "stack": lang, "status": ""}
    prompt = (f"Repository: {org}/{name}\nGitHub description: {desc}\nPrimary language: {lang}\n"
              f"README (may be truncated):\n{readme}\n\nUsing the repo_profile tool, profile what THIS "
              f"repository is and does. Assert only what the description/README supports; if thin, say so.")
    # the client retries transient errors internally; add one extra slow retry for batch rate-limits
    for attempt in range(2):
        try:
            return client.decide(prompt, PROFILE_TOOL, max_tokens=700)
        except LLMUnavailable:
            if attempt == 0:
                time.sleep(6.0)
                continue
            return {"one_liner": desc or name,
                    "what_it_does": desc or "(profile unavailable; README ingest only)",
                    "stack": lang, "status": ""}


def write_profile(org: str, repo: dict, prof: dict, context_dir: str, domain: str) -> str:
    """Refuse the retired direct L2 materialization API.

    The signature is retained so an old caller fails explicitly instead of silently
    overwriting a verified ``repo-<name>`` note.  No argument is inspected and no file is
    created.
    """
    raise LegacyRepoProfileWriteDisabled(_WRITE_DISABLED_MESSAGE)


# ---------------------------------------------------------------------------
# DEEP structural profiler — read a LOCAL cloned repo dir and emit a bounded,
# deterministic structured profile for the engineering-wiki generator.
# Pure/stdlib-only, NO network, NEVER executes repo code, every field
# capped so the downstream LLM prompt stays small. Safe on any missing file.
# ---------------------------------------------------------------------------

# Bounds (keep the LLM prompt small; large repos summarized top-down).
_MAX_TREE = 120           # tree entries (dirs + notable files)
_MAX_TREE_DEPTH = 4       # relative depth for tree listing
_MAX_LANGS = 10           # distinct file extensions reported
_MAX_ROUTES = 60
_MAX_CLI = 40
_MAX_MCP_TOOLS = 40
_MAX_MODELS = 12
_MAX_TRAINING = 12
_MAX_DEPS = 40
_MAX_DEPLOY_LIST = 30
_MAX_HOSTS = 12
_MAX_SOURCE_SCAN = 400    # source files scanned for routes/interfaces/models
_MAX_FILE_BYTES = 200_000  # cap bytes read per source file
_README_CHARS = 2400
_ARCH_CHARS = 2400
_DOC_CHARS = 900
_CARD_CHARS = 700
_MAX_OTHER_DOCS = 12

# Directories never walked (huge / vendored / build noise). `.github` kept.
_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build",
    ".idea", ".vscode", "site-packages", ".next", ".cache", "coverage",
    "vendor", ".terraform", ".gradle", "target",
}

_SOURCE_EXTS = {".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}
_WEIGHT_EXTS = {".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".pb", ".tflite", ".gguf"}
_DOC_EXTS = {".md", ".rst", ".txt"}

# Route decorators: FastAPI/Flask `@app.get(...)` / APIRouter `@router.post(...)`
# and Express `app.get(...)` / `router.post(...)`. Method + first string arg.
_ROUTE_RE = re.compile(
    r"""@?\b(?:app|router|api|bp|blueprint)\.(get|post|put|delete|patch|route)\s*\(\s*["'`]([^"'`]+)["'`]""",
    re.IGNORECASE,
)
# argparse subcommands: subparsers.add_parser("name")
_ADDPARSER_RE = re.compile(r"""add_parser\(\s*["']([^"']+)["']""")
# click commands: @cli.command("name") / @click.command(name="name") / bare @command
_CLICK_NAMED_RE = re.compile(r"""@\w+\.command\(\s*["']([^"']+)["']""")
_CLICK_KW_RE = re.compile(r"""@\w+\.command\([^)]*name\s*=\s*["']([^"']+)["']""")
_CLICK_BARE_RE = re.compile(r"""@(?:\w+\.)?command\s*\(?\s*\)?\s*$""")
# MCP tools: @mcp.tool / @tool / @server.tool / register_tool("name")
_MCP_NAMED_RE = re.compile(r"""(?:@\w+\.tool|@tool|register_tool)\([^)]*name\s*=\s*["']([^"']+)["']""")
_REGISTER_TOOL_RE = re.compile(r"""register_tool\(\s*["']([^"']+)["']""")
_MCP_DECOR_RE = re.compile(r"""@(?:\w+\.)?(?:mcp\.)?tool\b""")
_DEF_RE = re.compile(r"""^\s*(?:async\s+)?def\s+(\w+)""")
_NNMODULE_RE = re.compile(r"""^\s*class\s+(\w+)\s*\(([^)]*)\)""")


def _safe_read(path: str, max_bytes: int = _MAX_FILE_BYTES) -> str:
    """Read a text file safely (never raises). Bounded. NEVER executes anything."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(max_bytes)
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _git(repo_path: str, args: list[str]) -> str:
    """Run a read-only `git -C <path>` query. Returns stripped stdout or ''.

    git is a version-control tool, not the repo's own code — this never executes
    anything from the cloned repo.
    """
    try:
        r = subprocess.run(
            ["git", "-C", repo_path] + args,
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _git_meta(repo_path: str) -> dict:
    sha = _git(repo_path, ["rev-parse", "--short", "HEAD"])
    if not sha:
        return {"sha": "", "branch": "", "last_commit_date": ""}
    branch = _git(repo_path, ["symbolic-ref", "--short", "HEAD"]) \
        or _git(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"])
    last = _git(repo_path, ["log", "-1", "--format=%cI"])
    return {"sha": sha, "branch": branch, "last_commit_date": last}


def _walk(repo_path: str):
    """Yield (relpath, is_dir) for the repo, skipping vendored/build dirs.

    Bounded via caller. Deterministic order (sorted). Follows no symlinks.
    """
    for root, dirs, files in os.walk(repo_path, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.endswith(".egg-info"))
        rel_root = os.path.relpath(root, repo_path)
        if rel_root == ".":
            rel_root = ""
        for d in dirs:
            yield (os.path.join(rel_root, d) if rel_root else d), True
        for f in sorted(files):
            yield (os.path.join(rel_root, f) if rel_root else f), False


def _collect(repo_path: str):
    """Single pass: gather dir rels, file rels, and languages. All bounded."""
    dirs: list[str] = []
    files: list[str] = []
    langs: dict[str, int] = {}
    for rel, is_dir in _walk(repo_path):
        if is_dir:
            if len(dirs) < 4000:
                dirs.append(rel)
            continue
        if len(files) < 8000:
            files.append(rel)
        ext = os.path.splitext(rel)[1].lower()
        if ext:
            langs[ext] = langs.get(ext, 0) + 1
    top_langs = dict(sorted(langs.items(), key=lambda kv: (-kv[1], kv[0]))[:_MAX_LANGS])
    return dirs, files, top_langs


def _depth(rel: str) -> int:
    return rel.count(os.sep)


def _build_tree(dirs: list[str], files: list[str]) -> list[str]:
    """Notable dirs + files, depth-limited and count-capped, deterministic."""
    notable_names = {
        "README.md", "ARCHITECTURE.md", "Makefile", "Dockerfile", "pyproject.toml",
        "requirements.txt", "package.json", "docker-compose.yml", "docker-compose.yaml",
        "cloudbuild.yaml", "main.py", "app.py", "__main__.py", "setup.py", "go.mod",
    }
    out: list[str] = []
    for d in dirs:
        if _depth(d) < _MAX_TREE_DEPTH:
            out.append(d + "/")
    for f in files:
        base = os.path.basename(f)
        if base in notable_names or _depth(f) < _MAX_TREE_DEPTH:
            out.append(f)
    # De-dup preserving order, then cap.
    seen: set[str] = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:_MAX_TREE]


def _parse_makefile_targets(repo_path: str) -> list[str]:
    text = _safe_read(os.path.join(repo_path, "Makefile"))
    targets = []
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*:(?!=)", line)
        if m and not line.startswith("\t"):
            t = m.group(1)
            if t not in (".PHONY",) and t not in targets:
                targets.append(t)
    return targets


def _load_package_json(repo_path: str, files: list[str]) -> dict:
    for f in files:
        if os.path.basename(f) == "package.json" and _depth(f) == 0:
            try:
                return json.loads(_safe_read(os.path.join(repo_path, f)))
            except Exception:
                return {}
    return {}


def _entrypoints(repo_path: str, files: list[str], pkg: dict) -> list[str]:
    eps: list[str] = []
    wanted = {"main.py", "app.py", "fastapi_app.py", "__main__.py", "manage.py", "server.py"}
    for f in files:
        if os.path.basename(f) in wanted:
            eps.append(f)
    for t in _parse_makefile_targets(repo_path):
        eps.append(f"make:{t}")
    for name in (pkg.get("scripts") or {}):
        eps.append(f"npm:{name}")
    # De-dup, cap.
    out, seen = [], set()
    for e in eps:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out[:_MAX_CLI]


def _scan_sources(repo_path: str, files: list[str]) -> dict:
    """One bounded read-pass over source files → routes + cli + mcp tools + models."""
    routes: list[dict] = []
    cli: list[str] = []
    mcp_tools: list[str] = []
    models: list[dict] = []
    scanned = 0
    src_files = [f for f in files if os.path.splitext(f)[1].lower() in _SOURCE_EXTS]
    for f in src_files:
        if scanned >= _MAX_SOURCE_SCAN:
            break
        text = _safe_read(os.path.join(repo_path, f))
        if not text:
            continue
        scanned += 1
        lines = text.splitlines()

        for m in _ROUTE_RE.finditer(text):
            if len(routes) < _MAX_ROUTES:
                routes.append({"method": m.group(1).lower(), "path": m.group(2), "file": f})

        for m in _ADDPARSER_RE.finditer(text):
            if m.group(1) not in cli and len(cli) < _MAX_CLI:
                cli.append(m.group(1))
        for m in _CLICK_NAMED_RE.finditer(text):
            if m.group(1) not in cli and len(cli) < _MAX_CLI:
                cli.append(m.group(1))
        for m in _CLICK_KW_RE.finditer(text):
            if m.group(1) not in cli and len(cli) < _MAX_CLI:
                cli.append(m.group(1))

        for m in _MCP_NAMED_RE.finditer(text):
            if m.group(1) not in mcp_tools and len(mcp_tools) < _MAX_MCP_TOOLS:
                mcp_tools.append(m.group(1))
        for m in _REGISTER_TOOL_RE.finditer(text):
            if m.group(1) not in mcp_tools and len(mcp_tools) < _MAX_MCP_TOOLS:
                mcp_tools.append(m.group(1))
        # Bare @tool / @mcp.tool → name = the decorated function's name.
        for i, line in enumerate(lines):
            if _MCP_DECOR_RE.search(line) and "register_tool" not in line:
                for j in range(i + 1, min(i + 4, len(lines))):
                    dm = _DEF_RE.match(lines[j])
                    if dm:
                        if dm.group(1) not in mcp_tools and len(mcp_tools) < _MAX_MCP_TOOLS:
                            mcp_tools.append(dm.group(1))
                        break
            if _CLICK_BARE_RE.search(line):
                for j in range(i + 1, min(i + 4, len(lines))):
                    dm = _DEF_RE.match(lines[j])
                    if dm:
                        if dm.group(1) not in cli and len(cli) < _MAX_CLI:
                            cli.append(dm.group(1))
                        break

        # Models: python files defining nn.Module subclasses.
        if f.endswith(".py") and "nn.Module" in text and len(models) < _MAX_MODELS:
            for line in lines:
                cm = _NNMODULE_RE.match(line)
                if cm and "nn.Module" in cm.group(2):
                    models.append({"name": cm.group(1), "file": f, "weights": [], "card": ""})
                    if len(models) >= _MAX_MODELS:
                        break

    return {"routes": routes, "cli": cli, "mcp_tools": mcp_tools, "models": models}


def _models(repo_path: str, files: list[str], scanned_models: list[dict]) -> list[dict]:
    """Merge nn.Module classes with weight files + files under models/, attach cards."""
    models = list(scanned_models)
    known = {(m["name"], m["file"]) for m in models}
    weight_files = [f for f in files if os.path.splitext(f)[1].lower() in _WEIGHT_EXTS]

    # Files living under a models/ directory (any depth) that we haven't captured.
    under_models = [
        f for f in files
        if any(part.lower() in ("models", "model") for part in f.split(os.sep)[:-1])
        and os.path.splitext(f)[1].lower() in (_SOURCE_EXTS | _WEIGHT_EXTS)
    ]
    for f in under_models:
        key = (os.path.basename(f), f)
        if key not in known and len(models) < _MAX_MODELS:
            models.append({"name": os.path.basename(f), "file": f, "weights": [], "card": ""})
            known.add(key)

    # A model card excerpt: a README/MODEL_CARD near a models/ dir, else repo README.
    card_text = ""
    for f in files:
        base = os.path.basename(f).lower()
        if base in ("model_card.md", "modelcard.md") or (
            base == "readme.md" and "model" in f.lower() and os.sep in f
        ):
            card_text = _safe_read(os.path.join(repo_path, f))[:_CARD_CHARS]
            break

    for m in models:
        # Attach weight files sitting in the same directory as the model file.
        mdir = os.path.dirname(m["file"])
        m["weights"] = [w for w in weight_files if os.path.dirname(w) == mdir][:8]
        if not m["weights"] and weight_files:
            m["weights"] = weight_files[:8]
        m["card"] = card_text
    return models[:_MAX_MODELS]


def _training(repo_path: str, files: list[str]) -> list[dict]:
    out: list[dict] = []
    cfg_files = [f for f in files if os.path.splitext(f)[1].lower() in (".yaml", ".yml", ".json")]
    for f in files:
        base = os.path.basename(f)
        if not base.endswith(".py"):
            continue
        if base.startswith("train") or base.endswith("_train.py") or "training" in base:
            fdir = os.path.dirname(f)
            near = next((c for c in cfg_files if os.path.dirname(c) == fdir), "")
            out.append({"script": f, "config": near})
            if len(out) >= _MAX_TRAINING:
                break
    return out


def _deploy(repo_path: str, files: list[str]) -> dict:
    dockerfiles, compose, ci = [], [], []
    for f in files:
        base = os.path.basename(f).lower()
        if base.startswith("dockerfile"):
            dockerfiles.append(f)
        elif base.startswith("docker-compose") and base.endswith((".yml", ".yaml")):
            compose.append(f)
        elif f.startswith(os.path.join(".github", "workflows") + os.sep):
            ci.append(f)
        elif base in ("cloudbuild.yaml", "cloudbuild.yml"):
            ci.append(f)
        elif base == "skaffold.yaml" or (base.endswith((".yml", ".yaml")) and os.sep in f
                                         and f.split(os.sep)[0] in ("k8s", "deploy", "kubernetes")):
            ci.append(f)
    dockerfiles, compose, ci = dockerfiles[:_MAX_DEPLOY_LIST], compose[:_MAX_DEPLOY_LIST], ci[:_MAX_DEPLOY_LIST]

    gpu = False
    hosts: list[str] = []
    host_re = re.compile(r"\b([a-z0-9][a-z0-9.-]*\.(?:com|ai|io|internal|local))\b", re.I)
    for f in dockerfiles + compose + ci:
        text = _safe_read(os.path.join(repo_path, f), max_bytes=60_000)
        low = text.lower()
        if "--gpus" in low or "cuda" in low or "nvidia" in low or "gpu:" in low:
            gpu = True
        for m in host_re.finditer(text):
            h = m.group(1).lower()
            if h not in hosts and len(hosts) < _MAX_HOSTS:
                hosts.append(h)
    # Trim registry/image noise like docker.io / ghcr.io if they slipped in.
    hosts = [h for h in hosts if h not in ("docker.io", "ghcr.io", "index.docker.io", "pypi.org")][:_MAX_HOSTS]
    return {"dockerfiles": dockerfiles, "compose": compose, "ci": ci, "gpu": gpu, "hosts": hosts}


def _req_name(line: str) -> str:
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("-"):
        return ""
    # Strip version / markers / extras.
    return re.split(r"[<>=!~;\[ ]", line, 1)[0].strip()


def _deps(repo_path: str, files: list[str], pkg: dict) -> dict:
    py: list[str] = []
    for f in files:
        base = os.path.basename(f)
        if base.startswith("requirements") and base.endswith(".txt"):
            for line in _safe_read(os.path.join(repo_path, f)).splitlines():
                name = _req_name(line)
                if name and name not in py and len(py) < _MAX_DEPS:
                    py.append(name)
    # pyproject.toml dependencies (stdlib line scan, no toml parser needed).
    for f in files:
        if os.path.basename(f) == "pyproject.toml" and _depth(f) == 0:
            text = _safe_read(os.path.join(repo_path, f))
            for m in re.finditer(r'["\']([A-Za-z0-9][A-Za-z0-9._-]+)\s*(?:[<>=!~ ]|["\'])', text):
                name = m.group(1)
                if name and name not in py and len(py) < _MAX_DEPS and name not in ("python",):
                    # only accept from a dependencies-ish context: cheap heuristic
                    py.append(name)
            break
    py = py[:_MAX_DEPS]

    node: list[str] = []
    for section in ("dependencies", "devDependencies"):
        for name in (pkg.get(section) or {}):
            if name not in node and len(node) < _MAX_DEPS:
                node.append(name)
    return {"python": py, "node": node[:_MAX_DEPS]}


def _docs(repo_path: str, files: list[str]) -> dict:
    readme, architecture = "", ""
    other: dict[str, str] = {}
    for f in files:
        base = os.path.basename(f).lower()
        if base == "readme.md" and _depth(f) == 0 and not readme:
            readme = _safe_read(os.path.join(repo_path, f))[:_README_CHARS]
        elif base.startswith("architecture") and base.endswith(".md") and not architecture:
            architecture = _safe_read(os.path.join(repo_path, f))[:_ARCH_CHARS]
    for f in files:
        if f.split(os.sep)[0] == "docs" and os.path.splitext(f)[1].lower() in _DOC_EXTS:
            if len(other) >= _MAX_OTHER_DOCS:
                break
            other[f] = _safe_read(os.path.join(repo_path, f))[:_DOC_CHARS]
    return {"readme": readme, "architecture": architecture, "other": other}


def profile_repo(repo_path: str) -> dict:
    """Deterministic, stdlib-only, bounded structural profile of a LOCAL clone.

    Reads files only (never executes repo code, never touches the network) and
    returns the fixed shape consumed by the engineering-wiki generator.
    Safe/empty on any missing file or non-repo directory.
    """
    repo_path = os.path.abspath(repo_path)
    name = os.path.basename(repo_path.rstrip(os.sep))
    meta = _git_meta(repo_path)

    empty = {
        "repo": name, "sha": meta["sha"], "branch": meta["branch"],
        "last_commit_date": meta["last_commit_date"],
        "languages": {}, "tree": [], "entrypoints": [], "routes": [],
        "interfaces": {"cli": [], "mcp_tools": []}, "models": [], "training": [],
        "deploy": {"dockerfiles": [], "compose": [], "ci": [], "gpu": False, "hosts": []},
        "deps": {"python": [], "node": []},
        "docs": {"readme": "", "architecture": "", "other": {}},
    }
    if not os.path.isdir(repo_path):
        return empty

    try:
        dirs, files, langs = _collect(repo_path)
        pkg = _load_package_json(repo_path, files)
        scan = _scan_sources(repo_path, files)
        return {
            "repo": name,
            "sha": meta["sha"],
            "branch": meta["branch"],
            "last_commit_date": meta["last_commit_date"],
            "languages": langs,
            "tree": _build_tree(dirs, files),
            "entrypoints": _entrypoints(repo_path, files, pkg),
            "routes": scan["routes"],
            "interfaces": {"cli": scan["cli"][:_MAX_CLI], "mcp_tools": scan["mcp_tools"][:_MAX_MCP_TOOLS]},
            "models": _models(repo_path, files, scan["models"]),
            "training": _training(repo_path, files),
            "deploy": _deploy(repo_path, files),
            "deps": _deps(repo_path, files, pkg),
            "docs": _docs(repo_path, files),
        }
    except Exception:
        # Never raise on a malformed repo dir — the generator must degrade, not crash.
        return empty


# ---------------------------------------------------------------------------
# Component registry loader — registry/eng_components.yml.
# Stdlib-only: tolerates the flat `- {k: v, list: [a, b]}` flow-mapping form so
# the engine never hard-depends on PyYAML.
# ---------------------------------------------------------------------------

_REGISTRY_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "registry", "eng_components.yml",
)


def _split_flow_pairs(inner: str) -> list[str]:
    """Split a flow-mapping body on top-level commas (respect [] and quotes)."""
    parts, buf, depth = [], [], 0
    quote = ""
    for ch in inner:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "[{":
            depth += 1
            buf.append(ch)
        elif ch in "]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _coerce(val: str):
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        return [x.strip().strip("\"'") for x in _split_flow_pairs(val[1:-1])]
    return val.strip().strip("\"'")


def _parse_flow_mapping(inner: str) -> dict:
    out: dict = {}
    for pair in _split_flow_pairs(inner):
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        out[k.strip().strip("\"'")] = _coerce(v)
    return out


def load_components(path: str | None = None) -> list[dict]:
    """Load registry/eng_components.yml → list of component mappings.

    Stdlib-only. Uses PyYAML when present, else a tiny parser for the documented
    flat ``- {repo: ..., entities: [a, b], ...}`` form. Returns [] if missing.
    """
    path = path or _REGISTRY_DEFAULT
    if not os.path.exists(path):
        return []
    # Happy path: PyYAML if available (the flow-mapping form is valid YAML too).
    try:
        import yaml  # type: ignore
        with open(path) as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except Exception:
        pass
    # Fallback: line parser for the flat `- {...}` form.
    out: list[dict] = []
    for raw in _safe_read(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or not line.startswith("-"):
            continue
        body = line[1:].strip()
        if body.startswith("{") and body.endswith("}"):
            out.append(_parse_flow_mapping(body[1:-1]))
        elif ":" in body:
            out.append(_parse_flow_mapping(body))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", help="GitHub organization whose repositories to profile")
    ap.add_argument("--context-dir", default="context")
    ap.add_argument("--domain", default="repos")
    ap.add_argument("--limit", type=int, default=0, help="cap repos (0 = all active)")
    ap.add_argument("--sleep", type=float, default=1.2, help="seconds between calls (rate-limit guard)")
    ap.parse_args()
    ap.error(_WRITE_DISABLED_MESSAGE)


if __name__ == "__main__":
    main()
