# SPDX-License-Identifier: Apache-2.0
"""CrewAI adapter built on the CrewAI event bus.

Usage::

    from ma_trace.adapters.crewai import create_listener
    listener = create_listener()          # registers itself on crewai_event_bus
    crew.kickoff()                        # one ma-trace episode per kickoff

Mapping: crew kickoff -> episode; task start/completion -> delegation/response edges
between the crew and the executing agent; "delegate work"/"ask question" tool calls ->
delegation/negotiation edges between agents; agent execution -> agent spans; tool usage
and LLM calls -> recorded calls; memory query/save events -> memory operations.

:class:`CrewTraceCore` is framework-independent (it only relies on event attribute
names), so it can be tested - and reused - without CrewAI installed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

from ..events import CoordinationKind, MemoryOp, MemoryType, truncate
from ..tracer import CallHandle, MATrace, get_tracer

DELEGATION_TOOLS = {
    "delegate work to coworker": CoordinationKind.DELEGATION,
    "ask question to coworker": CoordinationKind.NEGOTIATION,
}


def _agent_name(event: Any) -> str:
    role = getattr(event, "agent_role", None)
    if role:
        return str(role)
    for attr in ("agent", "from_agent"):
        obj = getattr(event, attr, None)
        if obj is not None:
            r = getattr(obj, "role", None) or getattr(obj, "name", None)
            if r:
                return str(r)
    return "unknown"


def _task_label(event: Any) -> str:
    task = getattr(event, "task", None)
    for attr in ("name", "description"):
        val = getattr(task, attr, None) if task is not None else None
        if val:
            return str(val)
    return str(getattr(event, "task_name", None) or "task")


def _text(value: Any) -> Any:
    if value is None:
        return ""
    raw = getattr(value, "raw", None)
    return raw if raw is not None else value


class CrewTraceCore:
    """Translates CrewAI-shaped events into ma-trace records (duck-typed on attributes)."""

    def __init__(
        self,
        tracer: MATrace | None = None,
        *,
        episode_per_kickoff: bool = True,
        crew_node: str = "crew",
        record_outputs: bool = True,
        episode_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._tracer = tracer
        self.episode_per_kickoff = episode_per_kickoff
        self.crew_node = crew_node
        self.record_outputs = record_outputs
        self.episode_kwargs = dict(episode_kwargs or {})
        self._episode_stack: ExitStack | None = None
        self.episode: Any = None
        self._agent_stacks: dict[str, list[ExitStack]] = {}
        self._tool_runs: dict[Any, tuple[ExitStack, CallHandle]] = {}
        self._llm_runs: dict[Any, tuple[ExitStack, CallHandle]] = {}
        self._handlers: dict[str, Callable[[Any], None]] = {
            "CrewKickoffStartedEvent": self.on_kickoff_started,
            "CrewKickoffCompletedEvent": self.on_kickoff_finished,
            "CrewKickoffFailedEvent": self.on_kickoff_finished,
            "TaskStartedEvent": self.on_task_started,
            "TaskCompletedEvent": self.on_task_completed,
            "TaskFailedEvent": self.on_task_completed,
            "AgentExecutionStartedEvent": self.on_agent_started,
            "AgentExecutionCompletedEvent": self.on_agent_finished,
            "AgentExecutionErrorEvent": self.on_agent_finished,
            "ToolUsageStartedEvent": self.on_tool_started,
            "ToolUsageFinishedEvent": self.on_tool_finished,
            "ToolUsageErrorEvent": self.on_tool_finished,
            "ToolExecutionErrorEvent": self.on_tool_finished,
            "LLMCallStartedEvent": self.on_llm_started,
            "LLMCallCompletedEvent": self.on_llm_finished,
            "LLMCallFailedEvent": self.on_llm_finished,
            "MemoryQueryCompletedEvent": self.on_memory_query,
            "MemoryQueryFailedEvent": self.on_memory_query,
            "MemoryRetrievalCompletedEvent": self.on_memory_retrieval,
            "MemorySaveCompletedEvent": self.on_memory_save,
        }

    @property
    def tracer(self) -> MATrace:
        return self._tracer or get_tracer()

    @property
    def handled_event_names(self) -> list[str]:
        return list(self._handlers)

    def handle(self, event: Any) -> None:
        handler = self._handlers.get(type(event).__name__)
        if handler is not None:
            handler(event)

    # -- crew / episode -------------------------------------------------------

    def on_kickoff_started(self, event: Any) -> None:
        if not self.episode_per_kickoff or self._episode_stack is not None:
            return
        name = str(getattr(event, "crew_name", None) or "crew")
        self._episode_stack = ExitStack()
        self.episode = self._episode_stack.enter_context(self.tracer.episode(name, **self.episode_kwargs))
        self.tracer.bind_episode(self.episode)

    def on_kickoff_finished(self, event: Any) -> None:
        if self._episode_stack is None:
            return
        error = getattr(event, "error", None)
        if error is not None and self.episode is not None:
            self.episode.attributes["error"] = str(error)
        self.tracer.bind_episode(None)
        stack, self._episode_stack = self._episode_stack, None
        stack.close()

    # -- tasks ----------------------------------------------------------------

    def on_task_started(self, event: Any) -> None:
        self.tracer.coordination(
            self.crew_node, _agent_name(event), _task_label(event), kind=CoordinationKind.DELEGATION
        )

    def on_task_completed(self, event: Any) -> None:
        output = _text(getattr(event, "output", None))
        error = getattr(event, "error", None)
        self.tracer.coordination(
            _agent_name(event),
            self.crew_node,
            output if error is None else f"error: {error}",
            kind=CoordinationKind.RESPONSE,
            task=_task_label(event),
        )

    # -- agents ---------------------------------------------------------------

    def on_agent_started(self, event: Any) -> None:
        agent = _agent_name(event)
        stack = ExitStack()
        stack.enter_context(self.tracer.agent_span(agent))
        self._agent_stacks.setdefault(agent, []).append(stack)

    def on_agent_finished(self, event: Any) -> None:
        agent = _agent_name(event)
        stacks = self._agent_stacks.get(agent)
        if not stacks:
            return
        stack = stacks.pop()
        error = getattr(event, "error", None)
        if error is None:
            stack.close()
        else:
            exc = error if isinstance(error, BaseException) else RuntimeError(str(error))
            with contextlib.suppress(BaseException):
                stack.__exit__(type(exc), exc, exc.__traceback__)

    # -- tools ----------------------------------------------------------------

    @staticmethod
    def _tool_key(event: Any) -> Any:
        return getattr(event, "started_event_id", None) or (
            _agent_name(event),
            getattr(event, "tool_name", None),
        )

    def on_tool_started(self, event: Any) -> None:
        agent = _agent_name(event)
        tool = str(getattr(event, "tool_name", None) or "tool")
        args = getattr(event, "tool_args", None)
        kind = DELEGATION_TOOLS.get(tool.lower())
        if kind is not None and isinstance(args, dict) and args.get("coworker"):
            self.tracer.coordination(
                agent, str(args["coworker"]), args.get("task") or args.get("question") or tool, kind=kind
            )
        stack = ExitStack()
        handle = stack.enter_context(
            self.tracer.call("tool", tool, agent=agent, inputs=args, record_output=self.record_outputs)
        )
        key = getattr(event, "event_id", None) or (agent, tool)
        self._tool_runs[key] = (stack, handle)

    def on_tool_finished(self, event: Any) -> None:
        key = self._tool_key(event)
        run = self._tool_runs.pop(key, None)
        if run is None:  # fall back to the most recent run for this agent/tool
            fallback = (_agent_name(event), getattr(event, "tool_name", None))
            for k in list(self._tool_runs):
                if k == fallback or (isinstance(k, tuple) and k == fallback):
                    run = self._tool_runs.pop(k)
                    break
        if run is None:
            return
        stack, handle = run
        error = getattr(event, "error", None)
        if error is None:
            handle.set_output(_text(getattr(event, "output", None)))
            stack.close()
        else:
            exc = error if isinstance(error, BaseException) else RuntimeError(str(error))
            with contextlib.suppress(BaseException):
                stack.__exit__(type(exc), exc, exc.__traceback__)

    # -- LLM calls ------------------------------------------------------------

    def on_llm_started(self, event: Any) -> None:
        agent = _agent_name(event)
        model = getattr(event, "model", None)
        provider = str(model).split("/", 1)[0] if model and "/" in str(model) else "crewai"
        stack = ExitStack()
        handle = stack.enter_context(
            self.tracer.call(
                "llm",
                "chat",
                agent=agent,
                provider=provider,
                model=str(model) if model else None,
                inputs=getattr(event, "messages", None),
                record_output=self.record_outputs,
            )
        )
        key = getattr(event, "call_id", None) or (agent, "llm")
        self._llm_runs[key] = (stack, handle)

    def on_llm_finished(self, event: Any) -> None:
        key = getattr(event, "call_id", None) or (_agent_name(event), "llm")
        run = self._llm_runs.pop(key, None)
        if run is None:
            return
        stack, handle = run
        error = getattr(event, "error", None)
        if error is None:
            handle.set_output(_text(getattr(event, "response", None)))
            stack.close()
        else:
            exc = error if isinstance(error, BaseException) else RuntimeError(str(error))
            with contextlib.suppress(BaseException):
                stack.__exit__(type(exc), exc, exc.__traceback__)

    # -- memory ---------------------------------------------------------------

    def on_memory_query(self, event: Any) -> None:
        results = getattr(event, "results", None)
        failed = getattr(event, "error", None) is not None
        hit = bool(results) and not failed
        self.tracer.memory(
            _agent_name(event),
            MemoryOp.READ,
            MemoryType.SEMANTIC,
            truncate(getattr(event, "query", "") or "query", 120),
            hit=hit,
            results=len(results) if isinstance(results, (list, tuple, dict)) else None,
        )

    def on_memory_retrieval(self, event: Any) -> None:
        content = getattr(event, "memory_content", None)
        self.tracer.memory(
            _agent_name(event), MemoryOp.READ, MemoryType.EPISODIC, _task_label(event), hit=bool(content)
        )

    def on_memory_save(self, event: Any) -> None:
        self.tracer.memory(
            _agent_name(event), MemoryOp.WRITE, MemoryType.SEMANTIC, _task_label(event), confidence=None
        )


def create_listener(tracer: MATrace | None = None, **core_kwargs: Any) -> Any:
    """Create and register an event-bus listener (requires ``crewai``).

    Returns the listener; its ``core`` attribute is the :class:`CrewTraceCore`.
    """
    try:
        from crewai.events import BaseEventListener
        from crewai.events.types import (
            agent_events,
            crew_events,
            llm_events,
            memory_events,
            task_events,
            tool_usage_events,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "the CrewAI adapter needs crewai: pip install 'multi-agent-observability[crewai]'"
        ) from e

    core = CrewTraceCore(tracer, **core_kwargs)
    modules = (agent_events, crew_events, llm_events, memory_events, task_events, tool_usage_events)
    event_types: list[type] = []
    for name in core.handled_event_names:
        for module in modules:
            et = getattr(module, name, None)
            if et is not None:
                event_types.append(et)
                break

    class MATraceCrewListener(BaseEventListener):  # type: ignore[misc,valid-type]
        core: CrewTraceCore

        def setup_listeners(self, crewai_event_bus: Any) -> None:
            for et in event_types:

                def _make(_core: CrewTraceCore) -> Callable[[Any, Any], None]:
                    def _handler(source: Any, event: Any) -> None:
                        _core.handle(event)

                    return _handler

                crewai_event_bus.on(et)(_make(core))

    listener = MATraceCrewListener()
    listener.core = core
    return listener
