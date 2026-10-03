# SPDX-License-Identifier: Apache-2.0
"""LangChain / LangGraph callback handler.

Attach :class:`MATraceCallbackHandler` through ``config={"callbacks": [handler]}`` (or use
:func:`ma_trace.adapters.langgraph.trace_config`). Runs are mapped as follows:

* chain runs whose metadata carries ``langgraph_node`` (LangGraph nodes), ``agent_name``
  or ``ma_trace_agent``, or whose name is listed in ``agent_chains`` -> agent spans;
* consecutive agent runs -> ``handoff`` coordination edges (``infer_handoffs``);
* LLM / chat-model runs -> recorded LLM calls (model taken from LangSmith metadata);
* tool runs -> recorded tool calls;
* retriever runs -> semantic-memory reads (hit when at least one document came back).
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "the LangChain adapter needs langchain-core: pip install 'multi-agent-observability[langgraph]'"
    ) from e

from ..context import current_agent
from ..events import CoordinationKind, MemoryOp, MemoryType, truncate
from ..tracer import CallHandle, MATrace, get_tracer

AGENT_METADATA_KEYS: tuple[str, ...] = ("ma_trace_agent", "langgraph_node", "agent_name")


class _Run:
    __slots__ = ("agent", "handle", "is_agent", "kind", "name", "parent", "stack")

    def __init__(self, kind: str, name: str, agent: str | None, parent: UUID | None) -> None:
        self.kind = kind
        self.name = name
        self.agent = agent
        self.parent = parent
        self.stack: ExitStack | None = None
        self.handle: CallHandle | None = None
        self.is_agent = False


def _serialized_name(serialized: Any) -> str | None:
    if not isinstance(serialized, dict):
        return None
    name = serialized.get("name")
    if name:
        return str(name)
    ident = serialized.get("id")
    if isinstance(ident, list) and ident:
        return str(ident[-1])
    return None


def _serialized_kwargs(serialized: Any) -> dict[str, Any]:
    if isinstance(serialized, dict) and isinstance(serialized.get("kwargs"), dict):
        return serialized["kwargs"]
    return {}


def _llm_inputs(payload: Any) -> Any:
    if isinstance(payload, list):
        out: list[Any] = []
        for item in payload:
            if isinstance(item, list):
                out.append([_message_dict(m) for m in item])
            else:
                out.append(_message_dict(item))
        return out
    return payload


def _message_dict(m: Any) -> Any:
    if isinstance(m, str):
        return m
    return {"type": getattr(m, "type", type(m).__name__), "content": getattr(m, "content", str(m))}


def _llm_output(response: Any) -> Any:
    try:
        gen = response.generations[0][0]
    except (AttributeError, IndexError, TypeError):
        return str(response)
    message = getattr(gen, "message", None)
    if message is not None and getattr(message, "content", None) is not None:
        return message.content
    return getattr(gen, "text", str(gen))


def _tool_output(output: Any) -> Any:
    if isinstance(output, (str, int, float, bool, list, dict)) or output is None:
        return output
    content = getattr(output, "content", None)
    return content if content is not None else str(output)


class MATraceCallbackHandler(BaseCallbackHandler):
    """Records LangChain / LangGraph callbacks into the current ma-trace episode."""

    raise_error = False
    run_inline = True

    def __init__(
        self,
        tracer: MATrace | None = None,
        *,
        infer_handoffs: bool = True,
        agent_chains: tuple[str, ...] | list[str] = (),
        record_outputs: bool = True,
        agent_metadata_keys: tuple[str, ...] = AGENT_METADATA_KEYS,
        default_provider: str = "langchain",
    ) -> None:
        super().__init__()
        self._tracer = tracer
        self.infer_handoffs = infer_handoffs
        self.agent_chains = set(agent_chains)
        self.record_outputs = record_outputs
        self.agent_metadata_keys = agent_metadata_keys
        self.default_provider = default_provider
        self._runs: dict[UUID, _Run] = {}
        self._last_agent: str | None = None

    @property
    def tracer(self) -> MATrace:
        return self._tracer or get_tracer()

    # -- helpers ------------------------------------------------------------

    def _agent_for(
        self,
        parent_run_id: UUID | None,
        metadata: dict[str, Any] | None,
        serialized: Any,
        name_hint: str | None,
    ) -> tuple[str | None, bool]:
        agent: str | None = None
        if metadata:
            for key in self.agent_metadata_keys:
                if metadata.get(key):
                    agent = str(metadata[key])
                    break
        if agent is None:
            sname = _serialized_name(serialized) or name_hint
            if sname and sname in self.agent_chains:
                agent = sname
        if agent is None:
            return None, False
        parent = parent_run_id
        for _ in range(64):
            if parent is None:
                break
            run = self._runs.get(parent)
            if run is None:
                break
            if run.is_agent and run.agent == agent:
                return agent, False
            parent = run.parent
        return agent, True

    def _enclosing_agent(self, parent_run_id: UUID | None) -> str | None:
        parent = parent_run_id
        for _ in range(64):
            if parent is None:
                break
            run = self._runs.get(parent)
            if run is None:
                break
            if run.agent:
                return run.agent
            parent = run.parent
        return current_agent()

    def _close(self, run_id: UUID, error: BaseException | None = None) -> None:
        run = self._runs.pop(run_id, None)
        if run is None or run.stack is None:
            return
        try:
            if error is None:
                run.stack.close()
            else:
                run.stack.__exit__(type(error), error, error.__traceback__)
        except BaseException:  # the agent/call span re-raises the error; never propagate from a callback
            pass

    # -- chains / LangGraph nodes ------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        name_hint = kwargs.get("name")
        agent, open_span = self._agent_for(parent_run_id, metadata, serialized, name_hint)
        run = _Run("chain", _serialized_name(serialized) or name_hint or "chain", agent, parent_run_id)
        if agent and open_span:
            if self.infer_handoffs and self._last_agent and self._last_agent != agent:
                self.tracer.coordination(self._last_agent, agent, "handoff", kind=CoordinationKind.HANDOFF)
            self._last_agent = agent
            run.stack = ExitStack()
            run.stack.enter_context(self.tracer.agent_span(agent))
            run.is_agent = True
        self._runs[run_id] = run

    def on_chain_end(
        self, outputs: dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        self._close(run_id)

    def on_chain_error(
        self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        self._close(run_id, error)

    # -- LLMs ---------------------------------------------------------------

    def _start_llm(
        self,
        serialized: Any,
        payload: Any,
        run_id: UUID,
        parent_run_id: UUID | None,
        metadata: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> None:
        md = metadata or {}
        params = kwargs.get("invocation_params") or {}
        skw = _serialized_kwargs(serialized)
        model = (
            md.get("ls_model_name")
            or params.get("model")
            or params.get("model_name")
            or skw.get("model")
            or skw.get("model_name")
        )
        provider = md.get("ls_provider") or self.default_provider
        agent = self._enclosing_agent(parent_run_id)
        run = _Run("llm", _serialized_name(serialized) or "llm", agent, parent_run_id)
        run.stack = ExitStack()
        run.handle = run.stack.enter_context(
            self.tracer.call(
                "llm",
                "chat",
                agent=agent,
                provider=provider,
                model=model,
                inputs=_llm_inputs(payload),
                record_output=self.record_outputs,
            )
        )
        self._runs[run_id] = run

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._start_llm(serialized, messages, run_id, parent_run_id, metadata, kwargs)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._start_llm(serialized, prompts, run_id, parent_run_id, metadata, kwargs)

    def on_llm_end(
        self, response: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        run = self._runs.get(run_id)
        if run is not None and run.handle is not None:
            run.handle.set_output(_llm_output(response))
        self._close(run_id)

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        self._close(run_id, error)

    # -- tools --------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        name = _serialized_name(serialized) or kwargs.get("name") or "tool"
        agent = self._enclosing_agent(parent_run_id)
        run = _Run("tool", name, agent, parent_run_id)
        run.stack = ExitStack()
        run.handle = run.stack.enter_context(
            self.tracer.call(
                "tool",
                name,
                agent=agent,
                inputs=inputs if inputs is not None else input_str,
                record_output=self.record_outputs,
            )
        )
        self._runs[run_id] = run

    def on_tool_end(
        self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        run = self._runs.get(run_id)
        if run is not None and run.handle is not None:
            run.handle.set_output(_tool_output(output))
        self._close(run_id)

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        self._close(run_id, error)

    # -- retrievers (memory) ------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._runs[run_id] = _Run("retriever", query, self._enclosing_agent(parent_run_id), parent_run_id)

    def on_retriever_end(
        self, documents: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        count = len(documents) if documents is not None else 0
        self.tracer.memory(
            run.agent or "unknown",
            MemoryOp.READ,
            MemoryType.SEMANTIC,
            truncate(run.name, 120),
            hit=count > 0,
            documents=count,
        )

    def on_retriever_error(
        self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        run = self._runs.pop(run_id, None)
        if run is not None:
            self.tracer.memory(
                run.agent or "unknown",
                MemoryOp.READ,
                MemoryType.SEMANTIC,
                truncate(run.name, 120),
                hit=False,
                error=str(error),
            )

    # -- legacy agent executors --------------------------------------------

    def on_agent_action(
        self, action: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> Any:
        agent = self._enclosing_agent(parent_run_id) or "agent"
        self.tracer.action(agent, getattr(action, "tool", "tool"), getattr(action, "tool_input", None))
