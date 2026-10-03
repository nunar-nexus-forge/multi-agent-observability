from __future__ import annotations

from types import SimpleNamespace

import ma_trace as mt
from ma_trace.adapters.crewai import CrewTraceCore


def make(name, **fields):
    """Create an event object whose class name matches a CrewAI event type."""
    cls = type(name, (), {})
    obj = cls()
    for k, v in fields.items():
        setattr(obj, k, v)
    return obj


def test_core_maps_crewai_events(tracer):
    core = CrewTraceCore(tracer)
    agent = SimpleNamespace(role="Researcher")
    task = SimpleNamespace(name="find facts", description="Find facts about X")
    core.handle(make("CrewKickoffStartedEvent", crew_name="demo-crew"))
    assert core.episode is not None and mt.current_episode() is core.episode
    core.handle(make("TaskStartedEvent", agent_role="Researcher", task=task))
    core.handle(make("AgentExecutionStartedEvent", agent=agent, task=task))
    core.handle(
        make(
            "LLMCallStartedEvent",
            agent_role="Researcher",
            model="openai/gpt-4o",
            call_id="c1",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    core.handle(
        make(
            "LLMCallCompletedEvent",
            agent_role="Researcher",
            model="openai/gpt-4o",
            call_id="c1",
            response="plan",
        )
    )
    core.handle(
        make(
            "ToolUsageStartedEvent",
            agent_role="Researcher",
            tool_name="Delegate work to coworker",
            tool_args={"coworker": "Writer", "task": "draft"},
            event_id="t1",
        )
    )
    core.handle(
        make(
            "ToolUsageFinishedEvent",
            agent_role="Researcher",
            tool_name="Delegate work to coworker",
            started_event_id="t1",
            output="done",
        )
    )
    core.handle(
        make(
            "ToolUsageStartedEvent",
            agent_role="Researcher",
            tool_name="search",
            tool_args={"q": "x"},
            event_id="t2",
        )
    )
    core.handle(
        make(
            "ToolUsageErrorEvent",
            agent_role="Researcher",
            tool_name="search",
            started_event_id="t2",
            error=RuntimeError("timeout"),
        )
    )
    core.handle(
        make("MemoryQueryCompletedEvent", agent_role="Researcher", query="facts about X", results=[1, 2])
    )
    core.handle(make("MemoryQueryFailedEvent", agent_role="Researcher", query="other", error="nope"))
    core.handle(make("MemorySaveCompletedEvent", agent_role="Researcher", task=task))
    core.handle(make("AgentExecutionCompletedEvent", agent=agent, task=task, output="result"))
    core.handle(
        make(
            "TaskCompletedEvent",
            agent_role="Researcher",
            task=task,
            output=SimpleNamespace(raw="final answer"),
        )
    )
    core.handle(make("UnknownEvent"))
    ep = core.episode
    core.handle(make("CrewKickoffCompletedEvent", crew_name="demo-crew"))
    assert ep.finished and mt.current_episode() is None and core.episode is ep

    edges = [(e.source, e.target, e.kind.value) for e in ep.coordination_events]
    assert edges == [
        ("crew", "Researcher", "delegation"),
        ("Researcher", "Writer", "delegation"),
        ("Researcher", "crew", "response"),
    ]
    assert ep.agents[:2] == ["crew", "Researcher"]
    calls = ep.calls
    assert (
        calls[0].call_type is mt.CallType.LLM
        and calls[0].model == "openai/gpt-4o"
        and calls[0].provider == "openai"
        and calls[0].output == "plan"
    )
    assert calls[1].name == "Delegate work to coworker" and calls[1].output == "done"
    assert calls[2].name == "search" and calls[2].error == "RuntimeError: timeout"
    mem = ep.memory_trace()
    assert [m.hit for m in mem.reads] == [True, False] and len(mem.writes) == 1
    agent_events = ep.events_of(mt.AgentEvent)
    assert [(e.agent, e.phase) for e in agent_events] == [("Researcher", "start"), ("Researcher", "end")]
    assert tracer.store.load(ep.id).name == "demo-crew"


def test_core_without_episode_per_kickoff(tracer):
    core = CrewTraceCore(tracer, episode_per_kickoff=False)
    with mt.episode("outer") as ep:
        core.handle(make("CrewKickoffStartedEvent", crew_name="c"))
        core.handle(make("TaskStartedEvent", agent_role="A", task_name="t"))
        core.handle(make("CrewKickoffFailedEvent", crew_name="c", error="x"))
    assert core.episode is None and len(ep.coordination_events) == 1
    assert "TaskStartedEvent" in core.handled_event_names
