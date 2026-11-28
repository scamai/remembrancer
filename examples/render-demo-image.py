"""Render the README figure from the released CLI and its fictional Git fixture."""
import argparse
from html import escape
from io import BytesIO
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from textwrap import wrap

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "who is on call for ledger-service"

PALETTES = {
    "light": ("#ffffff", "#0a0a0a", "#55555e", "#f3f3f6", "#2b50d8"),
    "dark": ("#040406", "#ffffff", "#adadb0", "#18181e", "#6b8cff"),
}

PAGE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 508" role="img"
  aria-labelledby="title description">
  <title id="title">Remembrancer fictional Git demo</title>
  <desc id="description">{question} {answer}. Source {source}, commit {commit}. Guest {empty}.</desc>
  <path fill="{page}" d="M0 0h760v508H0z"/>
  <path fill="{panel}" d="M32 212h696v216H32z"/>
  <path fill="{accent}" d="M32 212h4v216h-4z"/>
  <g font-family="system-ui, sans-serif" font-weight="700" fill="{ink}">
    <text x="32" y="126" font-size="48">{question_lines}</text>
    <text x="60" y="312" font-size="40">{answer_lines}</text>
  </g>
  <g font-family="ui-monospace, monospace" font-size="25" fill="{muted}">
    <text x="32" y="53">Fictional Git demo</text>
    <text x="60" y="259" fill="{accent}">dana · eng</text>
    <text x="60" y="398">Git · {source} · {commit}</text>
    <text x="60" y="476">guest</text>
    <text x="704" y="476" text-anchor="end">{empty}</text>
  </g>
</svg>
"""


def run(cwd, *args, env=None):
    return subprocess.run(args, cwd=cwd, env=env, check=True, capture_output=True).stdout


def capture(revision):
    with TemporaryDirectory(prefix="remembrancer-demo-") as directory:
        checkout = Path(directory)
        commit = run(ROOT, "git", "rev-parse", "--verify", "--end-of-options",
                     f"{revision}^{{commit}}").decode().strip()
        archive = run(ROOT, "git", "archive", commit)
        with tarfile.open(fileobj=BytesIO(archive)) as source:
            source.extractall(checkout, filter="data")
        (checkout / "config.json").write_bytes((checkout / "config.example.json").read_bytes())
        env = {"PATH": os.environ["PATH"], "HOME": directory, "GIT_CONFIG_NOSYSTEM": "1",
               "LANG": "C.UTF-8"}
        run(checkout, "bash", "examples/demo-repo.sh", env=env)
        run(checkout, sys.executable, "scripts/run.py", "--ingest-git", "demo-repo", "--lint", env=env)
        command = (sys.executable, "retrieve/query.py", "--q", QUESTION, "--as")
        engineer = run(checkout, *command, "dana", env=env).decode()
        guest = run(checkout, *command, "guest", env=env).decode()

        blocks = engineer.split("\n\n")
        best = next(block for block in blocks if re.match(r"★ DELIVER +\[ *1\]", block))
        citation = re.search(r"provenance:\['git:([0-9a-f]{40})'\]", best)
        if not citation or "acl:['eng']" not in best or "(no matches)" not in guest.splitlines():
            raise ValueError("The demo did not return the expected citation and access results")
        commit = citation[1]
        repo = checkout / "demo-repo"
        subject = run(repo, "git", "show", "-s", "--format=%s", commit, env=env).decode().strip()
        [source_file] = run(
            repo, "git", "show", "--format=", "--name-only", commit, env=env
        ).decode().strip().splitlines()
        if subject not in {line.strip() for line in best.splitlines()}:
            raise ValueError("The result does not show the cited commit's subject")
        return {
            "question": QUESTION.capitalize() + "?",
            "answer": subject.removeprefix("docs: "),
            "source": source_file,
            "commit": commit[:7],
            "empty": "(no matches)",
        }


def lines(value, width, x, step):
    rows = wrap(value, width, break_long_words=False, break_on_hyphens=False)
    if len(rows) != 2:
        raise ValueError("The demo text no longer fits the two-line image layout")
    first, second = map(escape, rows)
    return f'<tspan x="{x}">{first}</tspan><tspan x="{x}" dy="{step}">{second}</tspan>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="v0.1.0-alpha", help="public release commit or tag")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "media")
    args = parser.parse_args()

    data = capture(args.revision)
    labels = {key: escape(value) for key, value in data.items()}
    labels["question_lines"] = lines(data["question"], width=25, x=32, step=52)
    labels["answer_lines"] = lines(data["answer"], width=31, x=60, step=44)
    args.output.mkdir(parents=True, exist_ok=True)
    for theme, (page, ink, muted, panel, accent) in PALETTES.items():
        (args.output / f"demo-{theme}.svg").write_text(
            PAGE.format(**labels, page=page, ink=ink, muted=muted, panel=panel, accent=accent),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
