# Contributing to multi-agent-observability

Thanks for your interest. This document explains how to set up a development
environment, what we expect in a pull request, and where help is most useful.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/). Everything is installed into a
project-local virtual environment (`./.venv`); nothing touches your global Python.

```bash
git clone https://github.com/nunar-nexus-forge/multi-agent-observability
cd multi-agent-observability
make sync             # uv sync --group dev --group frameworks  -> ./.venv
make check            # ruff + mypy + pytest
```

Without `make`: `uv sync --group dev --group frameworks`, then `uv run pytest`.

## What makes a good pull request

- **Tests.** Every behaviour change comes with a test in `tests/`. Adapter tests must
  not require network access or API keys (use fakes, as the existing tests do).
- **Type hints and docstrings** on public API.
- **Lint clean**: `uv run ruff check . && uv run ruff format --check .`.
- **A CHANGELOG entry** under "Unreleased".
- **No copied text or figures** from other projects or publications; write it in your own
  words and credit your sources.

## Where help is most welcome

- Adapters for other frameworks (Semantic Kernel, OpenAI Agents SDK, smolagents, Google ADK).
- Backend-specific guides (Langfuse, Phoenix, Datadog dashboards for the SLO metrics).
- Feedback on the proposed `gen_ai.*` attribute names in `docs/semantic-conventions.md`.
- Replay ergonomics: codecs for rich output objects, partial replays, time-travel debugging.

## Reporting bugs and requesting features

Use the issue templates. For security problems see [SECURITY.md](SECURITY.md).

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## Release process (maintainers)

1. Bump `version` in `pyproject.toml` and `src/ma_trace/_version.py`, update `CHANGELOG.md` and `CITATION.cff`.
2. Tag: `git tag v0.2.0 && git push --tags`.
3. The `release.yml` workflow builds and publishes to PyPI via trusted publishing.
