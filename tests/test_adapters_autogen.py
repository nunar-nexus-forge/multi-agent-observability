from __future__ import annotations

import asyncio
from types import SimpleNamespace

import ma_trace as mt
from ma_trace.adapters.autogen import (
    AutoGenMessageTracer,
    register_ag2_hooks,
    trace_task_result,
    traced_run_stream,
)


def msg(name, **fields):
    cls = type(name, (), {})
    obj = cls()
    for k, v in fields.items():
        setattr(obj, k, v)
    return obj


def sample_messages():
    return [
        msg("TextMessage", source="user", content="Write a haiku"),
        msg("TextMessage", source="writer", content="draft 1"),
        msg(
            "ToolCallRequestEvent",
            source="writer",
            content=[SimpleNamespace(id="call1", name="lookup", arguments='{"q": "syllables"}')],
        ),
        msg(
            "ToolCallExecutionEvent",
            source="writer",
            content=[SimpleNamespace(call_id="call1", name="lookup", content="5-7-5", is_error=False)],
        ),
        msg("MemoryQueryEvent", source="writer", content=[]),
        msg("ThoughtEvent", source="writer", content="thinking"),
        msg("HandoffMessage", source="writer", target="critic", content="please review"),
        msg("TextMessage", source="critic", content="looks good"),
        msg("SelectSpeakerEvent", source="team", content=["writer"]),
        msg("ModelClientStreamingChunkEvent", source="writer", content="x"),
        msg("StopMessage", source="writer", content="done"),
    ]


def test_trace_task_result(tracer):
    result = SimpleNamespace(messages=sample_messages(), stop_reason="done")
    ep = trace_task_result(result, tracer, episode_name="haiku", seed=1)
    assert ep.finished and ep.name == "haiku"
    edges = [(e.source, e.target, e.kind.value) for e in ep.coordination_events]
    assert edges == [
        ("user", "writer", "response"),
        ("writer", "critic", "handoff"),
        ("team", "writer", "delegation"),
        ("critic", "writer", "response"),
    ]
    call = ep.calls[0]
    assert (
        call.name == "lookup"
        and call.agent == "writer"
        and call.output == "5-7-5"
        and call.call_type is mt.CallType.TOOL
    )
    assert ep.actions[0].name == "lookup"
    assert ep.memory_events[0].hit is False
    assert ep.events_of(mt.LogEvent)[0].message == "thinking"


def test_traced_run_stream(tracer):
    async def stream():
        for m in sample_messages():
            yield m
        yield msg("TaskResult", messages=[])

    async def consume():
        seen = []
        async for item in traced_run_stream(stream(), tracer):
            seen.append(type(item).__name__)
        return seen

    with mt.episode("stream") as ep:
        seen = asyncio.run(consume())
    assert seen[-1] == "TaskResult" and len(ep.coordination_events) == 4


def test_tool_error_flag(tracer):
    proc = AutoGenMessageTracer(tracer)
    with mt.episode("err") as ep:
        proc.process(
            msg(
                "ToolCallExecutionEvent",
                source="a",
                content=[SimpleNamespace(call_id="x", name="t", content="bad", is_error=True)],
            )
        )
    assert ep.calls[0].error and proc.processed == 1


def test_register_ag2_hooks(tracer):
    hooks = {}

    class FakeAgent:
        name = "assistant"

        def register_hook(self, name, fn):
            hooks[name] = fn

    agent = FakeAgent()
    register_ag2_hooks(agent, tracer)
    hook = hooks["process_message_before_send"]
    with mt.episode("ag2") as ep:
        out = hook(agent, {"content": "hello"}, SimpleNamespace(name="user_proxy"), False)
    assert out == {"content": "hello"}
    e = ep.coordination_events[0]
    assert (e.source, e.target, e.message) == ("assistant", "user_proxy", "hello")
