# Contributing to Remembrancer

Thank you for helping. This page says how to propose a change and what a change needs before it merges.

## Before you start

- **Bugs and small fixes:** open a pull request.
- **Larger changes**, such as a new connector, a new MCP tool or a storage change: open an issue first, so we agree on the shape before you write it.
- **Security problems:** do not open an issue. Follow [SECURITY.md](SECURITY.md).

## Set up

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest -q
```

You need Python 3.12 or newer. Ingest, distill, lint, index and the query CLI use only the standard library. `requirements.txt` lists what the MCP server, the registry reader and signed approvals need.

## What a pull request needs

- **Tests.** A change in behaviour comes with a test that fails without it. Run the whole suite before you push.
- **One change per pull request:** a fix, a feature or a refactor, with its tests and docs.
- **No real data.** Fixtures use made-up people and reserved example domains (`example.com`, `*.example`). Never commit the `events/`, `context/` or `state/` folder of a real deployment; `.gitignore` keeps them out.
- **A stdlib-only core.** A new third-party import belongs in the MCP server or behind an optional import, and in `requirements.txt`.

## Sign off your commits (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/) instead of a contributor licence agreement. By signing off a commit, you certify that you wrote it, or otherwise have the right to submit it under the project's licence.

Add the sign-off with `-s`:

```bash
git commit -s -m "fix: ..."
```

Git appends a line with your name and email:

```text
Signed-off-by: Your Name <you@example.com>
```

Every commit in a pull request needs one, and a CI check refuses a pull request with an unsigned commit. To sign off commits you already made, run `git rebase --signoff main`.

## Licence

By contributing, you agree that your contributions are licensed under the Apache License 2.0, as in [LICENSE](LICENSE).
