from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

import ma_trace as mt
from ma_trace.adapters.langchain import MATraceCallbackHandler


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return f"result for {q}"


def test_callback_handler_records_llm_and_tool(tracer):
    handler = MATraceCallbackHandler(tracer, agent_chains=("researcher",))
    llm = FakeListChatModel(responses=["forty-two"])

    def research(inputs, config):
        answer = llm.invoke("question?", config=config)
        found = lookup.invoke({"q": "life"}, config=config)
        return {"answer": answer.content, "found": found}

    chain = RunnableLambda(research).with_config(run_name="researcher")
    with mt.episode("lc") as ep:
        out = chain.invoke({"x": 1}, config={"callbacks": [handler]})
    assert out == {"answer": "forty-two", "found": "result for life"}
    agent_events = [(e.agent, e.phase) for e in ep.events_of(mt.AgentEvent)]
    assert agent_events == [("researcher", "start"), ("researcher", "end")]
    calls = ep.calls
    assert len(calls) == 2
    assert (
        calls[0].call_type is mt.CallType.LLM
        and calls[0].agent == "researcher"
        and calls[0].output == "forty-two"
    )
    assert (
        calls[1].call_type is mt.CallType.TOOL
        and calls[1].name == "lookup"
        and calls[1].agent == "researcher"
    )
    assert calls[1].output == "result for life"


def test_agent_from_metadata_and_handoffs(tracer):
    handler = MATraceCallbackHandler(tracer)
    first = RunnableLambda(lambda x: x + 1).with_config(metadata={"ma_trace_agent": "planner"})
    second = RunnableLambda(lambda x: x * 2).with_config(metadata={"ma_trace_agent": "executor"})
    with mt.episode("meta") as ep:
        assert first.invoke(1, config={"callbacks": [handler]}) == 2
        assert second.invoke(2, config={"callbacks": [handler]}) == 4
    assert ep.agents == ["planner", "executor"]
    edges = [(e.source, e.target, e.kind.value) for e in ep.coordination_events]
    assert edges == [("planner", "executor", "handoff")]


def test_chain_error_closes_span(tracer):
    handler = MATraceCallbackHandler(tracer)

    def boom(x):
        raise ValueError("bad")

    chain = RunnableLambda(boom).with_config(metadata={"ma_trace_agent": "bad"})
    with pytest.raises(ValueError), mt.episode("err") as ep:
        chain.invoke(1, config={"callbacks": [handler]})
    end = next(e for e in ep.events_of(mt.AgentEvent) if e.phase == "end")
    assert end.error == "ValueError: bad"
