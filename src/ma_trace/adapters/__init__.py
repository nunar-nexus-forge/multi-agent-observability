# SPDX-License-Identifier: Apache-2.0
"""Framework adapters. Each module imports its framework lazily so that ``ma_trace``
itself has no framework dependency:

* :mod:`ma_trace.adapters.langchain` - ``MATraceCallbackHandler`` for LangChain and LangGraph
* :mod:`ma_trace.adapters.langgraph` - ``trace_config`` / ``traced_node`` helpers for LangGraph
* :mod:`ma_trace.adapters.crewai` - ``create_listener`` for the CrewAI event bus
* :mod:`ma_trace.adapters.autogen` - message tracing for AutoGen AgentChat (and AG2 hooks)
"""
