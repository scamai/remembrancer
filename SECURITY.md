# Security policy

## Report a vulnerability

Report security problems privately, never in a public issue or pull request.

Use GitHub's private vulnerability reporting: on this repository's **Security** tab, choose **Report a vulnerability**. Only the maintainers see the report.

Include what you can: the version or commit, the steps to reproduce, and what an attacker gains. We will reply in the report and agree a disclosure date with you.

## Supported versions

Remembrancer is in alpha. Fixes land on `main` and in the next release; older releases get no backports.

## What v0.1 protects, and what it leaves to you

- **No login of its own.** The stdio server, the default, reads with the identity the operator gives it: `REMEMBRANCER_IDENTITY` in the MCP client's server settings. A caller's own `as_identity` claim is ignored unless `config.json` sets `identity.trust_caller_claims` to `true`. A caller with no identity is anonymous and reads only notes with no access label.
- **The HTTP transport is closed by default.** It binds to `127.0.0.1` and refuses every request until `REMEMBRANCER_MCP_PROXY_SECRET` is set (32 characters or more) and a reverse proxy sends the same value in the `X-Remembrancer-Proxy-Secret` header. The proxy is what authenticates callers and sets their identity, so do not expose the port directly. There is no token sign-in: a request that carries an `Authorization` header gets 401 and never falls back to the identity header, so the proxy must drop that header.
- **Identity is per request because HTTP is stateless by default.** In stateful mode (`mcp.http.stateless: false` in `config.json`, or `REMEMBRANCER_MCP_STATELESS=0`), every call on a session runs with the identity of the request that opened the session, so a caller who sends another caller's session id reads with that caller's access. Use stateful mode only behind a proxy that serves one person.
- **Your memory is plain files** in `events/`, `context/` and `state/`, readable by anyone who can read the folder. Protect it with file permissions, and keep it out of version control (`.gitignore` already lists it).
- **Access labels decide what each identity reads.** A note's `acl` labels are matched against the grants that `config.json` → `identities` gives each identity. Check those grants before you connect a shared agent.
