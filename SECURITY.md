# Security Policy

## Supported versions

Security fixes are applied to the latest minor release.

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Use the repository's
**Security** tab on GitHub (*Report a vulnerability*, GitHub's private vulnerability
reporting) and include the version affected, a description of the issue and its impact,
and steps to reproduce. You will receive an acknowledgement within 5 working days, and a
fixed release with credit to the reporter (unless you prefer anonymity).

## Scope notes

- Recorded episodes contain prompts, tool arguments and outputs. The episode store
  (`./.ma-trace/episodes` by default) should be treated as sensitive data; do not
  commit it. Set `record_outputs=False` or `encode=` on `mt.llm`/`mt.tool` to redact.
- `ma-trace replay --entrypoint module:function` imports and executes code from the
  current working directory by design. Only run it in directories you trust.
