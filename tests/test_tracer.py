from __future__ import annotations

import asyncio

import pytest

import ma_trace as mt
from ma_trace import semconv as sc
from ma_trace.events import (
    AgentEvent,
    CallEvent,
    ContractEvent,
    CoordinationEvent,
    LogEvent,
    MemoryEvent,
    StateEvent,
)


def test_episode_records_and_persists(tracer):
    with mt.episode("demo", seed=3, attributes={"env": "test"}) as ep:
        assert mt.current_episode() is ep
        assert ep.rng is not None and ep.seed == 3
        mt.coordination("a", "b", "hello", kind="delegation", confidence=0.9, message_id="m1")
        mt.memory("b", "read", "episodic", "k", hit=True, confidence=0.7)
        mt.action("b", "do", {"x": 1})
        mt.state("b", {"s": 1})
        mt.log("note", agent="b", extra=1)
    assert mt.current_episode() is None
    assert ep.finished and ep.duration_s >= 0
    assert [type(e) for e in ep.events] == [
        CoordinationEvent,
        MemoryEvent,
        mt.ActionEvent,
        StateEvent,
        LogEvent,
    ]
    assert [e.seq for e in ep.events] == [0, 1, 2, 3, 4]
    stored = tracer.store.load(ep.id)
    assert stored.name == "demo" and stored.attributes == {"env": "test"} and len(stored.events) == 5
    assert tracer.last_metrics is not None and tracer.last_metrics.coordination_events == 1
    assert ep.agents == ["a", "b"]


def test_unsampled_episode_is_not_persisted(tracer):
    with mt.episode("skip", record=False) as ep:
        mt.coordination("a", "b")
    assert not ep.sampled and ep.id not in tracer.store
    assert len(ep.events) == 1  # still counted for metrics


def test_agent_decorator_sync_and_async(tracer):
    @mt.agent("planner")
    def plan():
        assert mt.current_agent() == "planner"
        return "p"

    @mt.agent("worker")
    async def work():
        assert mt.current_agent() == "worker"
        await asyncio.sleep(0)
        return "w"

    with mt.episode("agents") as ep:
        assert plan() == "p"
        assert asyncio.run(work()) == "w"
        assert mt.current_agent() is None
    agent_events = ep.events_of(AgentEvent)
    assert [(e.agent, e.phase) for e in agent_events] == [
        ("planner", "start"),
        ("planner", "end"),
        ("worker", "start"),
        ("worker", "end"),
    ]
    assert all(e.duration_s is not None for e in agent_events if e.phase == "end")
    assert ep.agent_durations().keys() == {"planner", "worker"}


def test_agent_error_is_recorded_and_reraised(tracer):
    @mt.agent("bad")
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError), mt.episode("err") as ep:
        boom()
    end = next(e for e in ep.events_of(AgentEvent) if e.phase == "end")
    assert end.error == "ValueError: nope"
    assert ep.attributes["error"] == "ValueError: nope"


def test_llm_and_tool_decorators_record_calls(tracer):
    calls = []

    @mt.llm("planner", provider="fake", model="fake-1")
    def ask(prompt):
        calls.append(prompt)
        return {"text": prompt.upper()}

    @mt.tool("search")
    def search(q, limit=1):
        return [q] * limit

    with mt.episode("calls") as ep:
        assert ask("hi") == {"text": "HI"}
        with mt.agent_span("worker"):
            assert search("x", limit=2) == ["x", "x"]
    c = ep.calls
    assert len(c) == 2
    assert (
        c[0].agent == "planner"
        and c[0].call_type is mt.CallType.LLM
        and c[0].model == "fake-1"
        and c[0].provider == "fake"
    )
    assert c[0].output == {"text": "HI"} and c[0].output_recorded and c[0].index == 0 and c[0].duration_s >= 0
    assert (
        c[1].agent == "worker"
        and c[1].name == "search"
        and c[1].call_type is mt.CallType.TOOL
        and c[1].index == 1
    )
    assert c[1].output == ["x", "x"] and c[1].input_fingerprint


def test_call_context_managers_and_unserialisable_output(tracer):
    class Obj:
        pass

    with mt.episode("cm") as ep, mt.llm_call("a", provider="p", model="m", inputs={"p": 1}) as h:
        assert h.should_execute
        h.set_output(Obj())
    ev = ep.calls[0]
    assert ev.output_recorded is False and "__repr__" in ev.output and ev.output_fingerprint

    with mt.episode("cm2") as ep2, mt.tool_call("t", "agentx", inputs=[1]) as h2:
        h2.set_output(42)
    assert ep2.calls[0].output == 42 and ep2.calls[0].agent == "agentx"


def test_call_error_recorded(tracer):
    @mt.tool("fail")
    def fail():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError), mt.episode("e") as ep:
        fail()
    assert ep.calls[0].error == "RuntimeError: x"


def test_contract_derives_seed_from_episode(tracer):
    with mt.episode("c", seed=5) as ep:
        with mt.contract("env", "step", {"pos": 0}) as c:
            c.set_post_state({"pos": 1})
            c.set_result("moved")
        with mt.contract("env", "step", {"pos": 1}, seed=99) as c2:
            c2.set_post_state({"pos": 2})
    evs = ep.contract_events
    assert isinstance(evs[0], ContractEvent) and evs[0].seed is not None and evs[0].result == "moved"
    assert evs[1].seed == 99 and evs[0].post_hash == evs[1].pre_hash
    # same seed -> same derived contract seed
    with mt.episode("c", seed=5) as ep2, mt.contract("env", "step", {"pos": 0}):
        pass
    assert ep2.contract_events[0].seed == evs[0].seed


def test_events_outside_episode_are_dropped(tracer):
    ev = mt.coordination("a", "b")
    assert ev.seq == 0 and mt.current_episode() is None


def test_bind_episode_fallback(tracer):
    with mt.episode("bound") as ep:
        pass
    ep.ended_at = None
    tracer.bind_episode(ep)
    try:
        mt.coordination("x", "y")
        assert len(ep.coordination_events) == 1
    finally:
        tracer.bind_episode(None)


def test_otel_spans_and_attributes(spans):
    exporter, t = spans

    @t.agent("planner")
    def plan():
        with t.tool_call("lookup", inputs={"q": 1}) as h:
            h.set_output("r")
        t.coordination("planner", "executor", "go", kind="delegation")
        t.memory("planner", "write", "semantic", "fact")
        with t.contract("planner", "apply", {"s": 0}) as c:
            c.set_post_state({"s": 1})

    with t.episode("otel", seed=1):
        plan()
    finished = {s.name: s for s in exporter.get_finished_spans()}
    assert "invoke_workflow otel" in finished and "invoke_agent planner" in finished
    assert "execute_tool lookup" in finished and "coordinate planner->executor" in finished
    assert "memory_access write fact" in finished and "environment_contract apply" in finished
    wf = finished["invoke_workflow otel"]
    assert wf.attributes[sc.GEN_AI_OPERATION_NAME] == "invoke_workflow"
    assert wf.attributes["ma_trace.episode.seed"] == 1
    assert "ma_trace.slo.coordination_overhead" in wf.attributes
    agent = finished["invoke_agent planner"]
    assert agent.attributes[sc.GEN_AI_AGENT_NAME] == "planner" and agent.parent.span_id == wf.context.span_id
    tool = finished["execute_tool lookup"]
    assert (
        tool.attributes[sc.GEN_AI_TOOL_NAME] == "lookup"
        and tool.attributes["ma_trace.call.output_fingerprint"]
    )
    coord = finished["coordinate planner->executor"]
    assert (
        coord.attributes["ma_trace.coordination.kind"] == "delegation"
        and coord.attributes["ma_trace.coordination.target"] == "executor"
    )
    contract = finished["environment_contract apply"]
    assert contract.attributes["ma_trace.contract.post_state_hash"]


def test_gen_ai_namespace(spans):
    exporter, _ = spans
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    t = mt.MATrace(store="memory", tracer_provider=provider, namespace="gen_ai", otel_metrics=False)
    with t.episode("ns"):
        t.coordination("a", "b")
    names = {k for s in exporter.get_finished_spans() for k in s.attributes}
    assert "gen_ai.coordination.source" in names and "gen_ai.episode.id" in names
    assert t.semconv.COORD_SOURCE == "gen_ai.coordination.source"


def test_configure_console_and_prometheus(tmp_path):
    pytest.importorskip("prometheus_client")
    from prometheus_client import CollectorRegistry

    from ma_trace.metrics import PrometheusExporter

    registry = CollectorRegistry()
    exp = PrometheusExporter(registry=registry)
    t = mt.configure("svc", exporter="none", store=str(tmp_path), metrics_exporters=[exp], otel_metrics=False)
    assert mt.get_tracer() is t
    with t.episode("p"):
        t.action("a", "x")
    assert registry.get_sample_value("ma_trace_episodes_total", {"episode_name": "p"}) == 1.0
    assert (tmp_path / f"{t.last_episode.id}.json").exists()
    mt.set_tracer(None)


def test_flag_anomaly_escalates_sampler(tracer):
    tracer.sampler = mt.AdaptiveSampler(base_ratio=0.0, escalation_episodes=2)
    with mt.episode("a") as e1:
        pass
    assert not e1.sampled
    with mt.episode("b") as e2:
        mt.flag_anomaly("weird")
    assert not e2.sampled and tracer.sampler.in_anomaly_mode
    with mt.episode("c") as e3:
        pass
    with mt.episode("d") as e4:
        pass
    with mt.episode("e") as e5:
        pass
    assert e3.sampled and e4.sampled and not e5.sampled
    assert isinstance(e2.events[0], LogEvent) and e2.events[0].level == "warning"


def test_episode_json_roundtrip(tracer):
    with (
        mt.episode("rt", seed=2, entrypoint="mod:fn", entrypoint_args={"n": 1}) as ep,
        mt.llm_call("a", provider="p", model="m", inputs="x") as h,
    ):
        h.set_output("y")
    back = mt.Episode.from_json(ep.to_json())
    assert back.id == ep.id and back.entrypoint == "mod:fn" and back.entrypoint_args == {"n": 1}
    assert isinstance(back.calls[0], CallEvent) and back.calls[0].output == "y" and back.rng is not None
    assert back.summary().startswith(ep.id)
