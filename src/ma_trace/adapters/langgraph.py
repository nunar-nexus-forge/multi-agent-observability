# SPDX-License-Identifier: Apache-2.0
"""LangGraph helpers.

Zero-code path::

    from ma_trace.adapters.langgraph import trace_config
    with mt.episode("pipeline", seed=1):
        graph.invoke(inputs, config=trace_config())

Every node becomes an agent span (via the ``langgraph_node`` run metadata), node-to-node
transitions become ``handoff`` coordination edges, and LLM/tool calls made inside nodes
are recorded when the node passes ``config`` on to them.

Explicit path: decorate node functions with :func:`traced_node` to open the agent span
yourself and snapshot the node's output state for oscillation detection.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from ..context import current_agent
from ..contracts import state_fingerprint
from ..tracer import MATrace, get_tracer
from .langchain import MATraceCallbackHandler

__all__ = ["MATraceCallbackHandler", "state_fingerprint", "trace_config", "traced_node"]


def trace_config(
    tracer: MATrace | None = None, *, config: dict[str, Any] | None = None, **handler_kwargs: Any
) -> dict[str, Any]:
    """Return a ``RunnableConfig`` (dict) carrying an :class:`MATraceCallbackHandler`.

    Existing ``config`` entries are preserved; the handler is appended to its callbacks.
    """
    handler = MATraceCallbackHandler(tracer, **handler_kwargs)
    cfg: dict[str, Any] = dict(config or {})
    existing = cfg.get("callbacks")
    if existing is None:
        callbacks: list[Any] = []
    elif isinstance(existing, list):
        callbacks = list(existing)
    else:  # a CallbackManager instance
        callbacks = list(getattr(existing, "handlers", []))
    callbacks.append(handler)
    cfg["callbacks"] = callbacks
    return cfg


def traced_node(
    name: str | None = None, *, tracer: MATrace | None = None, snapshot: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for LangGraph node functions.

    Opens an agent span named after the node (unless the callback handler already did)
    and, when ``snapshot`` is true, records the node's returned state update as a
    :class:`~ma_trace.events.StateEvent` so that oscillation between states is measurable.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        node_name: str = name or str(getattr(fn, "__name__", "node"))

        def _t() -> MATrace:
            return tracer or get_tracer()

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
                t = _t()
                if current_agent() == node_name:
                    out = await fn(state, *args, **kwargs)
                else:
                    with t.agent_span(node_name):
                        out = await fn(state, *args, **kwargs)
                if snapshot:
                    t.state(node_name, out, label="node_output")
                return out

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            t = _t()
            if current_agent() == node_name:
                out = fn(state, *args, **kwargs)
            else:
                with t.agent_span(node_name):
                    out = fn(state, *args, **kwargs)
            if snapshot:
                t.state(node_name, out, label="node_output")
            return out

        return wrapper

    return decorate
