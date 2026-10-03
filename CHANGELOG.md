# Changelog

All notable changes to `multi-agent-observability` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- `ma-trace show --events` now prints meaningful `False` flags (`hit=false` for memory misses,
  `verified=false`, `output_recorded=false`) and zero-valued fields such as `index=0`; only the
  `noop` and `replayed` flags are hidden when false.
- The bundled demo and the README explain that `ma-trace replay` must run from the directory
  that contains the entrypoint module (`examples/`), and the demo's `sys.path` handling is portable.

### Changed
- README, `NOTICE`, `CITATION.cff` and the package metadata now describe the software only;
  the SLO thresholds and the sampler default are documented as library defaults to calibrate.

### Removed
- Unused `ReplaySession.note_contract_result` (contract pass counts come from the replayed episode).

## [0.1.0] - 2026-09-27

### Added
- Episodes with seeded randomness, an ordered event log and JSON persistence (`EpisodeStore`).
- Coordination graphs (typed, confidence-scored edges; bottlenecks, cycles, delegation depth; DOT/Mermaid/JSON export).
- Memory operation traces (`mt.memory`, `TracedMemory`; hit ratio, eviction reasons, thrashing detection).
- Environment contracts with per-action seeds and pre/post-state fingerprints.
- SLO metrics: coordination overhead, redundant action rate, oscillation rate, memory hit ratio; thresholds and violations.
- OpenTelemetry spans (standard `gen_ai.*` attributes plus `ma_trace.*` extensions), OTel metrics and Prometheus gauges.
- `AdaptiveSampler`: baseline ratio with anomaly escalation, implemented as an OpenTelemetry `Sampler`.
- Deterministic replay engine (`mt.replay`, `ma-trace replay`) with divergence reporting.
- CLI: `list`, `show`, `graph`, `metrics`, `memory`, `replay`, `diff`, `export`, `import`, `delete`.
- Adapters: LangChain/LangGraph callback handler and helpers, CrewAI event-bus listener, AutoGen AgentChat message tracer and AG2 hooks.
