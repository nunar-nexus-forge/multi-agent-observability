from __future__ import annotations

from typing import TypedDict

import pytest

pytest.importorskip("langgraph")

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.graph import END, START, StateGraph

import ma_trace as mt
from ma_trace.adapters.langgraph import trace_config, traced_node


class State(TypedDict):
    n: int
    log: list[str]


def build_graph(llm):
    def planner(state: State, config):
        reply = llm.invoke("plan", config=config)
        return {"n": state["n"] + 1, "log": state["log"] + [reply.content]}

    @traced_node("executor")
    def executor(state: State):
        return {"n": state["n"] * 2, "log": state["log"] + ["exec"]}

    def critic(state: State):
        return {"n": state["n"], "log": state["log"] + ["ok"]}

    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("critic", critic)
    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "critic")
    g.add_edge("critic", END)
    return g.compile()


def test_langgraph_zero_code_tracing(tracer):
    graph = build_graph(FakeListChatModel(responses=["step one"]))
    with mt.episode("graph", seed=1) as ep:
        out = graph.invoke({"n": 1, "log": []}, config=trace_config(tracer))
    assert out["n"] == 4 and out["log"] == ["step one", "exec", "ok"]
    starts = [e.agent for e in ep.events_of(mt.AgentEvent) if e.phase == "start"]
    assert starts == ["planner", "executor", "critic"]  # traced_node did not double-count executor
    edges = [(e.source, e.target, e.kind.value) for e in ep.coordination_events]
    assert edges == [("planner", "executor", "handoff"), ("executor", "critic", "handoff")]
    llm_calls = [c for c in ep.calls if c.call_type is mt.CallType.LLM]
    assert len(llm_calls) == 1 and llm_calls[0].agent == "planner" and llm_calls[0].output == "step one"
    states = ep.states
    assert len(states) == 1 and states[0].agent == "executor"
    assert ep.coordination_graph().is_dag


def test_trace_config_merges_existing_callbacks(tracer):
    handler_before = object()
    cfg = trace_config(tracer, config={"callbacks": [handler_before], "tags": ["t"]})
    assert cfg["callbacks"][0] is handler_before and len(cfg["callbacks"]) == 2 and cfg["tags"] == ["t"]


def test_traced_node_standalone(tracer):
    @traced_node()
    def worker(state):
        return {"n": state["n"] + 1}

    with mt.episode("standalone") as ep:
        assert worker({"n": 1}) == {"n": 2}
    assert [e.agent for e in ep.events_of(mt.AgentEvent)] == ["worker", "worker"]
    assert ep.states[0].label == "node_output"
