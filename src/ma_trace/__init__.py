# SPDX-License-Identifier: Apache-2.0
"""MA-Trace (pip install multi-agent-observability): causal tracing, coordination SLOs and
deterministic replay for multi-agent LLM systems.

Quick start::

    import ma_trace as mt

    mt.configure(exporter="console")           # optional: OpenTelemetry export

    with mt.episode("checkout", seed=42):
        @mt.agent("planner")
        def plan(goal): ...

        mt.coordination("planner", "executor", "do step 1", kind="delegation", confidence=0.9)
        mt.memory("executor", "read", "episodic", "step:1", hit=True, confidence=0.8)
        with mt.contract("executor", "apply", pre_state=state) as c:
            state = step(state); c.set_post_state(state)

    report = mt.replay("latest", entrypoint="my_app:run")
"""

from __future__ import annotations

from ._version import __version__
from .context import current_agent, current_episode
from .contracts import Contract, ContractCheck, state_fingerprint
from .episode import Episode
from .events import (
    ActionEvent,
    AgentEvent,
    CallEvent,
    CallType,
    ContractEvent,
    CoordinationEvent,
    CoordinationKind,
    Event,
    EvictionReason,
    LogEvent,
    MemoryEvent,
    MemoryOp,
    MemoryType,
    StateEvent,
    fingerprint,
)
from .graph import CoordinationGraph
from .memory import MemoryTrace, TracedMemory
from .metrics import (
    OTelMetricsExporter,
    PrometheusExporter,
    SLOMetrics,
    SLOThresholds,
    SLOViolation,
    compute_metrics,
    evaluate,
)
from .replay import Divergence, ReplayDivergenceError, ReplayError, ReplayReport, run_replay
from .sampling import AdaptiveSampler
from .semconv import SemConv
from .store import EpisodeStore, EpisodeSummary, InMemoryEpisodeStore
from .tracer import CallHandle, MATrace, configure, get_tracer, set_tracer

__all__ = [
    "ActionEvent",
    "AdaptiveSampler",
    "AgentEvent",
    "CallEvent",
    "CallHandle",
    "CallType",
    "Contract",
    "ContractCheck",
    "ContractEvent",
    "CoordinationEvent",
    "CoordinationGraph",
    "CoordinationKind",
    "Divergence",
    "Episode",
    "EpisodeStore",
    "EpisodeSummary",
    "Event",
    "EvictionReason",
    "InMemoryEpisodeStore",
    "LogEvent",
    "MATrace",
    "MemoryEvent",
    "MemoryOp",
    "MemoryTrace",
    "MemoryType",
    "OTelMetricsExporter",
    "PrometheusExporter",
    "ReplayDivergenceError",
    "ReplayError",
    "ReplayReport",
    "SLOMetrics",
    "SLOThresholds",
    "SLOViolation",
    "SemConv",
    "StateEvent",
    "TracedMemory",
    "__version__",
    "action",
    "agent",
    "agent_span",
    "compute_metrics",
    "configure",
    "contract",
    "coordination",
    "current_agent",
    "current_episode",
    "episode",
    "evaluate",
    "fingerprint",
    "flag_anomaly",
    "get_tracer",
    "llm",
    "llm_call",
    "log",
    "memory",
    "replay",
    "run_replay",
    "set_tracer",
    "state",
    "state_fingerprint",
    "tool",
    "tool_call",
]


# --- module-level convenience functions delegating to the default tracer -------


def episode(name, **kwargs):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.episode`."""
    return get_tracer().episode(name, **kwargs)


def agent(name, **kwargs):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.agent`."""
    return get_tracer().agent(name, **kwargs)


def agent_span(name, **kwargs):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.agent_span`."""
    return get_tracer().agent_span(name, **kwargs)


def llm(agent=None, **kwargs):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.llm`."""
    return get_tracer().llm(agent, **kwargs)


def tool(name=None, agent=None, **kwargs):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.tool`."""
    return get_tracer().tool(name, agent, **kwargs)


def llm_call(agent=None, **kwargs):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.llm_call`."""
    return get_tracer().llm_call(agent, **kwargs)


def tool_call(name, agent=None, **kwargs):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.tool_call`."""
    return get_tracer().tool_call(name, agent, **kwargs)


def coordination(source, target, message="", **kwargs):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.coordination`."""
    return get_tracer().coordination(source, target, message, **kwargs)


def memory(agent, op, memory_type, key, **kwargs):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.memory`."""
    return get_tracer().memory(agent, op, memory_type, key, **kwargs)


def contract(agent, action, pre_state, **kwargs):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.contract`."""
    return get_tracer().contract(agent, action, pre_state, **kwargs)


def action(agent, name, args=None, **kwargs):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.action`."""
    return get_tracer().action(agent, name, args, **kwargs)


def state(agent, state_value, label=None):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.state`."""
    return get_tracer().state(agent, state_value, label)


def log(message, **kwargs):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.log`."""
    return get_tracer().log(message, **kwargs)


def flag_anomaly(reason=None):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.flag_anomaly`."""
    return get_tracer().flag_anomaly(reason)


def replay(episode_ref, entrypoint=None, **kwargs):  # type: ignore[no-untyped-def]
    """See :meth:`MATrace.replay`."""
    return get_tracer().replay(episode_ref, entrypoint, **kwargs)
