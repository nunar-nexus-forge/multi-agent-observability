# SPDX-License-Identifier: Apache-2.0
"""AutoGen adapter.

AutoGen AgentChat (``autogen_agentchat`` 0.4+) exposes a run as a stream of typed
messages. :class:`AutoGenMessageTracer` turns them into ma-trace records:

* speaker changes -> coordination edges (``response``; ``handoff`` for HandoffMessage)
* ``ToolCallRequestEvent`` / ``ToolCallExecutionEvent`` -> actions and recorded tool calls
* ``MemoryQueryEvent`` -> semantic-memory reads
* ``ThoughtEvent`` -> log events

Use :func:`trace_task_result` after ``team.run(...)`` or wrap ``team.run_stream(...)`` with
:func:`traced_run_stream`. For the older AG2 / pyautogen ``ConversableAgent`` API use
:func:`register_ag2_hooks`. Everything is duck-typed, so no AutoGen import is required.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any

from ..episode import Episode
from ..events import CoordinationKind, MemoryOp, MemoryType
from ..tracer import MATrace, get_tracer

TEXT_MESSAGES = {
    "TextMessage",
    "MultiModalMessage",
    "StopMessage",
    "ToolCallSummaryMessage",
    "StructuredMessage",
}


class AutoGenMessageTracer:
    def __init__(
        self,
        tracer: MATrace | None = None,
        *,
        team_name: str = "team",
        user_name: str = "user",
        record_outputs: bool = True,
    ) -> None:
        self._tracer = tracer
        self.team_name = team_name
        self.user_name = user_name
        self.record_outputs = record_outputs
        self._last_source: str | None = None
        self._pending: dict[str, tuple[str, Any, str]] = {}
        self.processed = 0

    @property
    def tracer(self) -> MATrace:
        return self._tracer or get_tracer()

    def process_all(self, messages: Iterable[Any]) -> int:
        for m in messages:
            self.process(m)
        return self.processed

    def process(self, message: Any) -> None:
        cls = type(message).__name__
        source = str(getattr(message, "source", None) or "unknown")
        t = self.tracer
        self.processed += 1
        if cls in TEXT_MESSAGES:
            content = getattr(message, "content", "")
            if self._last_source and self._last_source != source:
                t.coordination(self._last_source, source, content, kind=CoordinationKind.RESPONSE)
            self._last_source = source
        elif cls == "HandoffMessage":
            target = str(getattr(message, "target", "unknown"))
            t.coordination(source, target, getattr(message, "content", ""), kind=CoordinationKind.HANDOFF)
            self._last_source = target
        elif cls == "ToolCallRequestEvent":
            for fc in getattr(message, "content", None) or []:
                name = str(getattr(fc, "name", "tool"))
                args = getattr(fc, "arguments", None)
                call_id = str(getattr(fc, "id", ""))
                self._pending[call_id] = (name, args, source)
                t.action(source, name, args)
        elif cls == "ToolCallExecutionEvent":
            for res in getattr(message, "content", None) or []:
                call_id = str(getattr(res, "call_id", ""))
                name, args, agent = self._pending.pop(
                    call_id, (str(getattr(res, "name", None) or "tool"), None, source)
                )
                with t.call(
                    "tool",
                    name,
                    agent=agent,
                    inputs=args,
                    call_id=call_id or None,
                    record_output=self.record_outputs,
                ) as h:
                    h.set_output(getattr(res, "content", None))
                    if getattr(res, "is_error", False):
                        h.event.error = "tool reported is_error=True"
        elif cls == "MemoryQueryEvent":
            results = getattr(message, "content", None) or []
            t.memory(
                source,
                MemoryOp.READ,
                MemoryType.SEMANTIC,
                "memory_query",
                hit=len(results) > 0,
                results=len(results),
            )
        elif cls == "ThoughtEvent":
            t.log(str(getattr(message, "content", "")), level="debug", agent=source)
        elif cls == "UserInputRequestedEvent":
            t.coordination(source, self.user_name, "input requested", kind=CoordinationKind.NEGOTIATION)
        elif cls == "SelectSpeakerEvent":
            for name in getattr(message, "content", None) or []:
                t.coordination(self.team_name, str(name), "select speaker", kind=CoordinationKind.DELEGATION)
        # streaming chunks and unknown events are ignored


def trace_task_result(
    result: Any,
    tracer: MATrace | None = None,
    *,
    episode_name: str = "autogen-task",
    team_name: str = "team",
    **episode_kwargs: Any,
) -> Episode:
    """Record the messages of a finished ``TaskResult`` as one episode and return it."""
    t = tracer or get_tracer()
    with t.episode(episode_name, **episode_kwargs) as ep:
        AutoGenMessageTracer(t, team_name=team_name).process_all(getattr(result, "messages", result))
    return ep


async def traced_run_stream(
    stream: AsyncIterator[Any], tracer: MATrace | None = None, **kwargs: Any
) -> AsyncIterator[Any]:
    """Wrap ``team.run_stream(...)``; every yielded message is recorded as it arrives."""
    proc = AutoGenMessageTracer(tracer, **kwargs)
    async for item in stream:
        if type(item).__name__ != "TaskResult":
            proc.process(item)
        yield item


def register_ag2_hooks(agent: Any, tracer: MATrace | None = None) -> None:
    """Register a ``process_message_before_send`` hook on an AG2 / pyautogen ``ConversableAgent``."""
    t = tracer or get_tracer()

    def _hook(sender: Any, message: Any, recipient: Any, silent: bool) -> Any:
        content = message.get("content", "") if isinstance(message, dict) else message
        t.coordination(
            str(getattr(sender, "name", sender)),
            str(getattr(recipient, "name", recipient)),
            content,
            kind=CoordinationKind.MESSAGE,
        )
        return message

    agent.register_hook("process_message_before_send", _hook)
