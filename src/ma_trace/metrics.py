# SPDX-License-Identifier: Apache-2.0
"""Multi-agent service level objectives (SLOs).

Three coordination-quality metrics are computed for every episode:

* **Coordination overhead** ``CO = (T_wall - sum(T_llm + T_tool)) / T_wall`` -
  the share of wall-clock time not spent doing productive inference or tool work.
* **Redundant action rate** ``RAR`` - duplicate or no-op actions relative to all
  actions (``redundant_actions_per_second`` gives the time-normalised variant).
* **Oscillation rate** ``OR = repeated state transitions / total state transitions`` -
  how often agents flip back and forth between states they have already visited.

plus the **memory hit ratio** ``MHR = successful retrievals / total memory accesses``.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ._version import __version__
from .events import ActionEvent, CallEvent, CallType, ContractEvent, MemoryOp, StateEvent

if TYPE_CHECKING:  # pragma: no cover
    from .episode import Episode


@dataclass
class SLOMetrics:
    episode_id: str
    episode_name: str
    wall_time_s: float
    llm_time_s: float
    tool_time_s: float
    coordination_overhead: float
    coordination_overhead_raw: float
    total_actions: int
    redundant_actions: int
    redundant_action_rate: float
    redundant_actions_per_second: float
    total_transitions: int
    repeated_transitions: int
    oscillation_rate: float
    memory_accesses: int
    memory_hits: int
    memory_hit_ratio: float
    coordination_events: int
    agents: int
    llm_calls: int
    tool_calls: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        rows = [
            ("coordination overhead (CO)", f"{self.coordination_overhead:.3f}"),
            (
                "redundant action rate (RAR)",
                f"{self.redundant_action_rate:.3f}  ({self.redundant_actions}/{self.total_actions})",
            ),
            (
                "oscillation rate (OR)",
                f"{self.oscillation_rate:.3f}  ({self.repeated_transitions}/{self.total_transitions})",
            ),
            (
                "memory hit ratio (MHR)",
                f"{self.memory_hit_ratio:.3f}  ({self.memory_hits}/{self.memory_accesses})",
            ),
            ("wall time", f"{self.wall_time_s:.3f}s"),
            ("llm / tool time", f"{self.llm_time_s:.3f}s / {self.tool_time_s:.3f}s"),
            ("llm / tool calls", f"{self.llm_calls} / {self.tool_calls}"),
            ("coordination events", str(self.coordination_events)),
            ("agents", str(self.agents)),
        ]
        width = max(len(r[0]) for r in rows)
        return "\n".join(f"{k.ljust(width)}  {v}" for k, v in rows)


@dataclass
class SLOThresholds:
    """Alert thresholds. The defaults are conservative levels typical of an un-optimised
    multi-agent system: crossing one means the system is doing no better than such a baseline.
    Calibrate them against your own recorded episodes."""

    coordination_overhead_max: float = 0.42
    redundant_action_rate_max: float = 0.18
    oscillation_rate_max: float = 0.23
    memory_hit_ratio_min: float | None = None


@dataclass
class SLOViolation:
    metric: str
    value: float
    threshold: float
    comparison: str  # ">" or "<"

    def __str__(self) -> str:
        return f"{self.metric}={self.value:.3f} {self.comparison} {self.threshold:.3f}"


def evaluate(metrics: SLOMetrics, thresholds: SLOThresholds | None = None) -> list[SLOViolation]:
    t = thresholds or SLOThresholds()
    out: list[SLOViolation] = []
    if metrics.coordination_overhead > t.coordination_overhead_max:
        out.append(
            SLOViolation(
                "coordination_overhead", metrics.coordination_overhead, t.coordination_overhead_max, ">"
            )
        )
    if metrics.redundant_action_rate > t.redundant_action_rate_max:
        out.append(
            SLOViolation(
                "redundant_action_rate", metrics.redundant_action_rate, t.redundant_action_rate_max, ">"
            )
        )
    if metrics.oscillation_rate > t.oscillation_rate_max:
        out.append(SLOViolation("oscillation_rate", metrics.oscillation_rate, t.oscillation_rate_max, ">"))
    if (
        t.memory_hit_ratio_min is not None
        and metrics.memory_accesses
        and metrics.memory_hit_ratio < t.memory_hit_ratio_min
    ):
        out.append(SLOViolation("memory_hit_ratio", metrics.memory_hit_ratio, t.memory_hit_ratio_min, "<"))
    return out


# --- building blocks ----------------------------------------------------------


def find_redundant_actions(actions: list[ActionEvent]) -> list[ActionEvent]:
    """Return the actions that are duplicates of an earlier action or explicit no-ops."""
    seen: set[tuple[str | None, str, str]] = set()
    redundant: list[ActionEvent] = []
    for a in sorted(actions, key=lambda e: e.seq):
        key = (a.agent, a.name, a.args_fingerprint)
        is_noop = a.noop or (
            a.state_before is not None and a.state_after is not None and a.state_before == a.state_after
        )
        if key in seen or is_noop:
            redundant.append(a)
        seen.add(key)
    return redundant


def state_sequences(episode: Episode) -> dict[str, list[str]]:
    """Ordered state-hash sequence per agent, with consecutive duplicates collapsed."""
    seqs: dict[str, list[str]] = {}

    def push(agent: str | None, h: str | None) -> None:
        if not h:
            return
        key = agent or "?"
        seq = seqs.setdefault(key, [])
        if not seq or seq[-1] != h:
            seq.append(h)

    for e in sorted(episode.events, key=lambda ev: ev.seq):
        if isinstance(e, StateEvent):
            push(e.agent, e.state_hash)
        elif isinstance(e, ContractEvent):
            push(e.agent, e.pre_hash)
            push(e.agent, e.post_hash)
        elif isinstance(e, ActionEvent):
            push(e.agent, e.state_before)
            push(e.agent, e.state_after)
    return seqs


def count_transitions(seqs: dict[str, list[str]]) -> tuple[int, int]:
    """``(total_transitions, repeated_transitions)`` over all agents."""
    total = repeated = 0
    for seq in seqs.values():
        seen: set[tuple[str, str]] = set()
        for a, b in itertools.pairwise(seq):
            total += 1
            if (a, b) in seen:
                repeated += 1
            seen.add((a, b))
    return total, repeated


def compute_metrics(episode: Episode) -> SLOMetrics:
    wall = episode.duration_s
    calls: list[CallEvent] = episode.calls
    llm_time = sum(c.duration_s for c in calls if c.call_type == CallType.LLM)
    tool_time = sum(c.duration_s for c in calls if c.call_type == CallType.TOOL)
    co_raw = (wall - (llm_time + tool_time)) / wall if wall > 0 else 0.0
    co = min(1.0, max(0.0, co_raw))

    actions = episode.actions
    redundant = find_redundant_actions(actions)
    rar = len(redundant) / len(actions) if actions else 0.0
    rar_per_s = len(redundant) / wall if wall > 0 else 0.0

    total_tr, repeated_tr = count_transitions(state_sequences(episode))
    osc = repeated_tr / total_tr if total_tr else 0.0

    mem = episode.memory_events
    reads = [m for m in mem if m.op == MemoryOp.READ]
    writes = [m for m in mem if m.op == MemoryOp.WRITE]
    hits = sum(1 for m in reads if m.hit)
    accesses = len(reads) + len(writes)
    mhr = hits / accesses if accesses else 0.0

    return SLOMetrics(
        episode_id=episode.id,
        episode_name=episode.name,
        wall_time_s=wall,
        llm_time_s=llm_time,
        tool_time_s=tool_time,
        coordination_overhead=co,
        coordination_overhead_raw=co_raw,
        total_actions=len(actions),
        redundant_actions=len(redundant),
        redundant_action_rate=rar,
        redundant_actions_per_second=rar_per_s,
        total_transitions=total_tr,
        repeated_transitions=repeated_tr,
        oscillation_rate=osc,
        memory_accesses=accesses,
        memory_hits=hits,
        memory_hit_ratio=mhr,
        coordination_events=len(episode.coordination_events),
        agents=len(episode.agents),
        llm_calls=sum(1 for c in calls if c.call_type == CallType.LLM),
        tool_calls=sum(1 for c in calls if c.call_type == CallType.TOOL),
    )


# --- exporters ----------------------------------------------------------------


class OTelMetricsExporter:
    """Publishes SLO metrics through the OpenTelemetry metrics API.

    Instruments (all unit ``1`` unless noted): ``<ns>.coordination_overhead``,
    ``<ns>.redundant_action_rate``, ``<ns>.oscillation_rate``, ``<ns>.memory_hit_ratio``
    (histograms), ``<ns>.episodes`` and ``<ns>.redundant_actions`` (counters) and
    ``<ns>.episode.duration`` (histogram, seconds).
    """

    def __init__(self, meter_provider: Any = None, namespace: str = "ma_trace") -> None:
        from opentelemetry import metrics as otel_metrics

        meter = otel_metrics.get_meter("ma_trace", __version__, meter_provider=meter_provider)
        ns = namespace
        self._co = meter.create_histogram(
            f"{ns}.coordination_overhead", unit="1", description="Coordination overhead per episode"
        )
        self._rar = meter.create_histogram(
            f"{ns}.redundant_action_rate", unit="1", description="Redundant action rate per episode"
        )
        self._or = meter.create_histogram(
            f"{ns}.oscillation_rate", unit="1", description="Oscillation rate per episode"
        )
        self._mhr = meter.create_histogram(
            f"{ns}.memory_hit_ratio", unit="1", description="Memory hit ratio per episode"
        )
        self._dur = meter.create_histogram(
            f"{ns}.episode.duration", unit="s", description="Episode wall-clock duration"
        )
        self._episodes = meter.create_counter(f"{ns}.episodes", unit="1", description="Episodes completed")
        self._redundant = meter.create_counter(
            f"{ns}.redundant_actions", unit="1", description="Redundant actions observed"
        )

    def export(self, m: SLOMetrics, attributes: dict[str, Any] | None = None) -> None:
        attrs = {"episode.name": m.episode_name}
        if attributes:
            attrs.update(attributes)
        self._co.record(m.coordination_overhead, attrs)
        self._rar.record(m.redundant_action_rate, attrs)
        self._or.record(m.oscillation_rate, attrs)
        self._mhr.record(m.memory_hit_ratio, attrs)
        self._dur.record(m.wall_time_s, attrs)
        self._episodes.add(1, attrs)
        if m.redundant_actions:
            self._redundant.add(m.redundant_actions, attrs)


class PrometheusExporter:
    """Exposes the SLO metrics as Prometheus gauges (requires ``prometheus-client``).

    >>> exporter = PrometheusExporter()          # uses the default registry
    >>> tracer = MATrace(metrics_exporters=[exporter])
    >>> # then serve with prometheus_client.start_http_server(9464)
    """

    def __init__(self, registry: Any = None, prefix: str = "ma_trace") -> None:
        try:
            from prometheus_client import REGISTRY, Counter, Gauge
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "PrometheusExporter requires prometheus-client: "
                "pip install 'multi-agent-observability[prometheus]'"
            ) from e
        reg = registry or REGISTRY
        labels = ["episode_name"]
        self.coordination_overhead = Gauge(
            f"{prefix}_coordination_overhead",
            "Coordination overhead of the last episode",
            labels,
            registry=reg,
        )
        self.redundant_action_rate = Gauge(
            f"{prefix}_redundant_action_rate",
            "Redundant action rate of the last episode",
            labels,
            registry=reg,
        )
        self.oscillation_rate = Gauge(
            f"{prefix}_oscillation_rate", "Oscillation rate of the last episode", labels, registry=reg
        )
        self.memory_hit_ratio = Gauge(
            f"{prefix}_memory_hit_ratio", "Memory hit ratio of the last episode", labels, registry=reg
        )
        self.episode_duration = Gauge(
            f"{prefix}_episode_duration_seconds",
            "Wall-clock duration of the last episode",
            labels,
            registry=reg,
        )
        self.episodes_total = Counter(f"{prefix}_episodes", "Episodes completed", labels, registry=reg)
        self.redundant_actions_total = Counter(
            f"{prefix}_redundant_actions", "Redundant actions observed", labels, registry=reg
        )

    def export(self, m: SLOMetrics, attributes: dict[str, Any] | None = None) -> None:
        lbl = {"episode_name": m.episode_name}
        self.coordination_overhead.labels(**lbl).set(m.coordination_overhead)
        self.redundant_action_rate.labels(**lbl).set(m.redundant_action_rate)
        self.oscillation_rate.labels(**lbl).set(m.oscillation_rate)
        self.memory_hit_ratio.labels(**lbl).set(m.memory_hit_ratio)
        self.episode_duration.labels(**lbl).set(m.wall_time_s)
        self.episodes_total.labels(**lbl).inc()
        if m.redundant_actions:
            self.redundant_actions_total.labels(**lbl).inc(m.redundant_actions)
