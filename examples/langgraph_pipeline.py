# SPDX-License-Identifier: Apache-2.0
"""Zero-code LangGraph tracing (requires ``pip install 'multi-agent-observability[langgraph]'``).

Every node becomes an agent span, node transitions become handoff edges, and the fake
chat model's calls are recorded, all through ``config=trace_config()``.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.graph import END, START, StateGraph

import ma_trace as mt
from ma_trace.adapters.langgraph import trace_config


class State(TypedDict):
    question: str
    draft: str
    verdict: str


llm = FakeListChatModel(responses=["The mug goes on the shelf.", "approved"])


def writer(state: State, config):
    reply = llm.invoke(state["question"], config=config)
    return {"draft": reply.content}


def reviewer(state: State, config):
    reply = llm.invoke("review: " + state["draft"], config=config)
    return {"verdict": reply.content}


def main() -> None:
    g = StateGraph(State)
    g.add_node("writer", writer)
    g.add_node("reviewer", reviewer)
    g.add_edge(START, "writer")
    g.add_edge("writer", "reviewer")
    g.add_edge("reviewer", END)
    graph = g.compile()

    mt.configure("langgraph-pipeline", exporter="console", store="memory")
    with mt.episode("langgraph-pipeline", seed=1) as ep:
        out = graph.invoke(
            {"question": "Where does the mug go?", "draft": "", "verdict": ""}, config=trace_config()
        )
    print(out)
    print(ep.metrics().format())
    print(ep.coordination_graph().to_mermaid())


if __name__ == "__main__":
    main()
