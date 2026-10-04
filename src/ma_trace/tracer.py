# SPDX-License-Identifier: Apache-2.0
"""The :class:`MATrace` facade: episodes, agent spans, recorded calls, coordination edges,
memory operations, environment contracts and SLO export."""

from __future__ import annotations

import functools
import inspect
import itertools
import logging
import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from . import semconv as sc
from ._version import __version__
from .context import _current_agent, _current_episode, current_agent, current_episode, current_replay
from .contracts import Contract
from .episode import Episode, _wall_time
from .events import (
    ActionEvent,
    AgentEvent,
    CallEvent,
    CallType,
    CoordinationEvent,
    CoordinationKind,
    Event,
    EvictionReason,
    LogEvent,
    MemoryEvent,
    MemoryOp,
    MemoryType,
    StateEvent,
    fingerprint,
    json_safe,
    truncate,
)
from .metrics import OTelMetricsExporter, SLOMetrics, compute_metrics
from .sampling import AdaptiveSampler
from .semconv import SemConv
from .store import EpisodeStore, InMemoryEpisodeStore

log = logging.getLogger("ma_trace")

F = Callable[..., Any]


class CallHandle:
    """Handle yielded by :meth:`MATrace.llm_call` / :meth:`MATrace.tool_call`.

    Inside the ``with`` block check :attr:`should_execute`; when it is ``False`` the
    call is being replayed and :attr:`recorded_output` holds the recorded result.
    Call :meth:`set_output` with whatever the call produced (or the recorded output).
    """

    def __init__(self, event: CallEvent) -> None:
        self.event = event
        self.should_execute = True
        self.recorded_output: Any = None
        self.matched: CallEvent | None = None
        self.output: Any = None
        self.output_set = False

    def set_output(self, output: Any) -> None:
        self.output = output
        self.output_set = True

    @property
    def replayed(self) -> bool:
        return not self.should_execute


class MATrace:
    """Central object of MA-Trace. One instance per process is typical (see :func:`configure`).

    Parameters
    ----------
    service_name:
        Reported as the OpenTelemetry ``service.name`` when this object builds providers.
    tracer_provider / meter_provider:
        Explicit OpenTelemetry providers. Defaults to the global ones (no-op until configured).
    store:
        Where sampled episodes are persisted: an :class:`EpisodeStore`, an
        :class:`InMemoryEpisodeStore`, a directory path, ``"memory"`` or ``None``
        (default: :class:`EpisodeStore` under ``$MA_TRACE_STORE`` or ``./.ma-trace/episodes``).
    sampler:
        An :class:`AdaptiveSampler` deciding which episodes are captured in full.
    namespace:
        Attribute namespace for the extension attributes (``"ma_trace"`` or ``"gen_ai"``).
    record_outputs:
        Store LLM/tool outputs inside episodes so they can be replayed.
    max_message_chars:
        Coordination messages are truncated to this length before being stored.
    metrics_exporters:
        Objects with an ``export(SLOMetrics)`` method, called at the end of every episode
        (for example :class:`ma_trace.metrics.PrometheusExporter`).
    """

    def __init__(
        self,
        service_name: str = "ma-trace",
        *,
        tracer_provider: Any = None,
        meter_provider: Any = None,
        store: EpisodeStore | InMemoryEpisodeStore | str | None = None,
        sampler: AdaptiveSampler | None = None,
        namespace: str = sc.DEFAULT_NAMESPACE,
        record_outputs: bool = True,
        max_message_chars: int = 512,
        metrics_exporters: list[Any] | None = None,
        otel_metrics: bool = True,
    ) -> None:
        self.service_name = service_name
        self.semconv = SemConv(namespace)
        self._otel = trace.get_tracer("ma_trace", __version__, tracer_provider=tracer_provider)
        self.record_outputs = record_outputs
        self.max_message_chars = max_message_chars
        self.sampler = sampler or AdaptiveSampler(
            base_ratio=1.0, sampled_attribute=self.semconv.EPISODE_SAMPLED
        )
        if isinstance(self.sampler, AdaptiveSampler):
            self.sampler.sampled_attribute = self.semconv.EPISODE_SAMPLED
        if store is None:
            self.store: EpisodeStore | InMemoryEpisodeStore | None = EpisodeStore()
        elif isinstance(store, str):
            self.store = InMemoryEpisodeStore() if store == "memory" else EpisodeStore(store)
        else:
            self.store = store
        self.metrics_exporters: list[Any] = list(metrics_exporters or [])
        if otel_metrics:
            self.metrics_exporters.append(OTelMetricsExporter(meter_provider, namespace=namespace))
        self.on_episode_end: list[Callable[[Episode, SLOMetrics], None]] = []
        self.last_metrics: SLOMetrics | None = None
        self.last_episode: Episode | None = None
        self._fallback_episode: Episode | None = None
        self._seq: dict[str, itertools.count[int]] = {}
        self._call_index: dict[str, itertools.count[int]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ episodes

    @contextmanager
    def episode(
        self,
        name: str,
        *,
        seed: int | None = None,
        entrypoint: str | Callable[..., Any] | None = None,
        entrypoint_args: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        record: bool | None = None,
        replay_of: str | None = None,
    ) -> Iterator[Episode]:
        """Open an episode. Everything recorded inside the block belongs to it.

        ``seed`` seeds :mod:`random` and ``episode.rng`` so that seeded randomness is
        reproducible under replay. ``entrypoint`` (``"pkg.module:function"`` or a
        callable) plus ``entrypoint_args`` let ``ma-trace replay <id>`` re-execute the
        episode without further arguments.
        """
        sampled = self.sampler.decide_episode() if record is None else bool(record)
        ep = Episode(
            id=Episode.new_id(),
            name=name,
            seed=seed,
            entrypoint=_entrypoint_name(entrypoint),
            entrypoint_args=dict(entrypoint_args or {}),
            attributes=dict(attributes or {}),
            sampled=sampled,
            replay_of=replay_of,
        )
        if seed is not None:
            ep.rng = random.Random(seed)
            random.seed(seed)
        s = self.semconv
        attrs: dict[str, Any] = {
            sc.GEN_AI_OPERATION_NAME: sc.OP_INVOKE_WORKFLOW,
            s.EPISODE_ID: ep.id,
            s.EPISODE_NAME: name,
            s.EPISODE_SAMPLED: sampled,
        }
        if seed is not None:
            attrs[s.EPISODE_SEED] = seed
        if replay_of:
            attrs[s.EPISODE_REPLAY_OF] = replay_of
        span = self._otel.start_span(f"{sc.OP_INVOKE_WORKFLOW} {name}", attributes=attrs)
        ctx_token = otel_context.attach(trace.set_span_in_context(span))
        ep_token = _current_episode.set(ep)
        try:
            yield ep
        except BaseException as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            ep.attributes["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ep.ended_at = _wall_time()
            try:
                self._finish_episode(ep, span)
            finally:
                _current_episode.reset(ep_token)
                otel_context.detach(ctx_token)
                span.end()
                self._seq.pop(ep.id, None)
                self._call_index.pop(ep.id, None)

    def _finish_episode(self, ep: Episode, span: trace.Span) -> None:
        m = compute_metrics(ep)
        self.last_metrics = m
        self.last_episode = ep
        prefix = f"{self.semconv.namespace}.slo."
        span.set_attribute(prefix + "coordination_overhead", m.coordination_overhead)
        span.set_attribute(prefix + "redundant_action_rate", m.redundant_action_rate)
        span.set_attribute(prefix + "oscillation_rate", m.oscillation_rate)
        span.set_attribute(prefix + "memory_hit_ratio", m.memory_hit_ratio)
        for exporter in self.metrics_exporters:
            try:
                exporter.export(m)
            except Exception as exc:  # pragma: no cover - never let telemetry break the app
                log.warning("metrics exporter %r failed: %s", exporter, exc)
        if ep.sampled and self.store is not None:
            try:
                self.store.save(ep)
            except Exception as exc:  # pragma: no cover
                log.warning("could not persist episode %s: %s", ep.id, exc)
        for cb in self.on_episode_end:
            try:
                cb(ep, m)
            except Exception as exc:  # pragma: no cover
                log.warning("on_episode_end callback failed: %s", exc)

    def bind_episode(self, episode: Episode | None) -> None:
        """Set a process-wide fallback episode for callbacks that run outside the
        context in which the episode was opened (multi-threaded frameworks)."""
        self._fallback_episode = episode

    def current_episode(self) -> Episode | None:
        return current_episode() or self._fallback_episode

    # ------------------------------------------------------------------ recording

    def _record(self, event: Event, episode: Episode | None = None) -> Event:
        ep = episode or self.current_episode()
        if ep is None:
            return event
        with self._lock:
            counter = self._seq.setdefault(ep.id, itertools.count())
            event.seq = next(counter)
            event.t = _wall_time()
            ep.events.append(event)
        return event

    def _next_call_index(self, ep: Episode | None) -> int:
        if ep is None:
            return 0
        with self._lock:
            return next(self._call_index.setdefault(ep.id, itertools.count()))

    # ------------------------------------------------------------------ agents

    @contextmanager
    def agent_span(
        self, name: str, *, agent_id: str | None = None, attributes: dict[str, Any] | None = None
    ) -> Iterator[trace.Span]:
        """Context manager marking an agent invocation (``invoke_agent`` span)."""
        token = _current_agent.set(name)
        start = time.perf_counter()
        self._record(AgentEvent(agent=name, phase="start"))
        attrs: dict[str, Any] = {sc.GEN_AI_OPERATION_NAME: sc.OP_INVOKE_AGENT, sc.GEN_AI_AGENT_NAME: name}
        if agent_id:
            attrs[sc.GEN_AI_AGENT_ID] = agent_id
        if attributes:
            attrs.update(attributes)
        error: str | None = None
        with self._otel.start_as_current_span(
            f"{sc.OP_INVOKE_AGENT} {name}", attributes=attrs, kind=SpanKind.CLIENT
        ) as span:
            try:
                yield span
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.set_attribute(sc.ERROR_TYPE, type(exc).__name__)
                raise
            finally:
                self._record(
                    AgentEvent(agent=name, phase="end", duration_s=time.perf_counter() - start, error=error)
                )
                _current_agent.reset(token)

    def agent(self, name: str, *, agent_id: str | None = None) -> Callable[[F], F]:
        """Decorator: run the function inside an agent span."""

        def decorate(fn: F) -> F:
            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.agent_span(name, agent_id=agent_id):
                        return await fn(*args, **kwargs)

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.agent_span(name, agent_id=agent_id):
                    return fn(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorate

    # ------------------------------------------------------------------ calls

    @contextmanager
    def call(
        self,
        call_type: CallType | str,
        name: str,
        *,
        agent: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        inputs: Any = None,
        record_output: bool | None = None,
        encode: Callable[[Any], Any] | None = None,
        decode: Callable[[Any], Any] | None = None,
        call_id: str | None = None,
    ) -> Iterator[CallHandle]:
        """Record one non-deterministic call (LLM or tool). Replay-aware."""
        ctype = CallType(call_type)
        agent_name = agent or current_agent() or "unknown"
        ep = self.current_episode()
        ev = CallEvent(
            agent=agent_name,
            call_type=ctype,
            name=name,
            provider=provider,
            model=model,
            index=self._next_call_index(ep),
            input_fingerprint=fingerprint(inputs) if inputs is not None else "",
        )
        handle = CallHandle(ev)
        session = current_replay()
        if session is not None:
            matched, divergence = session.consume_call(ev)
            if divergence is not None:
                session.report(divergence)
            if matched is not None:
                if matched.output_recorded:
                    handle.should_execute = False
                    handle.matched = matched
                    handle.recorded_output = decode(matched.output) if decode else matched.output
                    ev.replayed = True
                else:
                    session.report_output_unavailable(matched)
        s = self.semconv
        attrs: dict[str, Any] = {
            sc.GEN_AI_AGENT_NAME: agent_name,
            sc.GEN_AI_PROVIDER_NAME: provider or ("unknown" if ctype == CallType.LLM else "tool"),
            s.CALL_INDEX: ev.index,
            s.CALL_REPLAYED: ev.replayed,
        }
        if ev.input_fingerprint:
            attrs[s.CALL_INPUT_FINGERPRINT] = ev.input_fingerprint
        if ctype == CallType.LLM:
            attrs[sc.GEN_AI_OPERATION_NAME] = sc.OP_CHAT
            if model:
                attrs[sc.GEN_AI_REQUEST_MODEL] = model
            span_name = f"{sc.OP_CHAT} {model}" if model else sc.OP_CHAT
        else:
            attrs[sc.GEN_AI_OPERATION_NAME] = sc.OP_EXECUTE_TOOL
            attrs[sc.GEN_AI_TOOL_NAME] = name
            if call_id:
                attrs[sc.GEN_AI_TOOL_CALL_ID] = call_id
            span_name = f"{sc.OP_EXECUTE_TOOL} {name}"
        start = time.perf_counter()
        with self._otel.start_as_current_span(span_name, attributes=attrs, kind=SpanKind.CLIENT) as span:
            try:
                yield handle
            except BaseException as exc:
                ev.error = f"{type(exc).__name__}: {exc}"
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.set_attribute(sc.ERROR_TYPE, type(exc).__name__)
                raise
            finally:
                ev.duration_s = time.perf_counter() - start
                if handle.output_set:
                    should_record = self.record_outputs if record_output is None else record_output
                    if should_record and ep is not None and ep.sampled:
                        stored = encode(handle.output) if encode else handle.output
                        value, ok = json_safe(stored)
                        ev.output = value
                        ev.output_recorded = ok
                    ev.output_fingerprint = fingerprint(handle.output)
                    span.set_attribute(s.CALL_OUTPUT_FINGERPRINT, ev.output_fingerprint)
                self._record(ev, ep)

    def llm_call(
        self,
        agent: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        name: str = "chat",
        **kw: Any,
    ):  # type: ignore[no-untyped-def]
        """Context-manager form of :meth:`llm` for inline calls."""
        return self.call(CallType.LLM, name, agent=agent, provider=provider, model=model, **kw)

    def tool_call(self, name: str, agent: str | None = None, **kw: Any):  # type: ignore[no-untyped-def]
        """Context-manager form of :meth:`tool` for inline calls."""
        return self.call(CallType.TOOL, name, agent=agent, **kw)

    def _wrap_call(
        self,
        call_type: CallType,
        name: str,
        agent: str | None,
        provider: str | None,
        model: str | None,
        inputs: Callable[..., Any] | None,
        kw: dict[str, Any],
    ) -> Callable[[F], F]:
        def decorate(fn: F) -> F:
            call_name = name or getattr(fn, "__name__", "call")

            def _inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
                if inputs is not None:
                    return inputs(*args, **kwargs)
                return {"args": list(args), "kwargs": kwargs}

            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.call(
                        call_type,
                        call_name,
                        agent=agent,
                        provider=provider,
                        model=model,
                        inputs=_inputs(args, kwargs),
                        **kw,
                    ) as h:
                        out = await fn(*args, **kwargs) if h.should_execute else h.recorded_output
                        h.set_output(out)
                        return out

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.call(
                    call_type,
                    call_name,
                    agent=agent,
                    provider=provider,
                    model=model,
                    inputs=_inputs(args, kwargs),
                    **kw,
                ) as h:
                    out = fn(*args, **kwargs) if h.should_execute else h.recorded_output
                    h.set_output(out)
                    return out

            return wrapper  # type: ignore[return-value]

        return decorate

    def llm(
        self,
        agent: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        name: str = "chat",
        inputs: Callable[..., Any] | None = None,
        **kw: Any,
    ) -> Callable[[F], F]:
        """Decorator for a function that performs an LLM call.

        The wrapped function's arguments are fingerprinted and its return value is
        recorded. Under replay the function is *not* executed when a recorded output
        with a matching fingerprint exists; the recorded output is returned instead.
        ``inputs`` may be a function that maps the call arguments to the value that
        should be fingerprinted (use it to exclude clients, callbacks, timestamps...).
        """
        return self._wrap_call(CallType.LLM, name, agent, provider, model, inputs, kw)

    def tool(
        self,
        name: str | None = None,
        agent: str | None = None,
        *,
        inputs: Callable[..., Any] | None = None,
        **kw: Any,
    ) -> Callable[[F], F]:
        """Decorator for a tool function (same replay semantics as :meth:`llm`)."""
        return self._wrap_call(CallType.TOOL, name or "", agent, None, None, inputs, kw)

    # ------------------------------------------------------------------ coordination / memory / contracts

    def coordination(
        self,
        source: str,
        target: str,
        message: Any = "",
        *,
        kind: CoordinationKind | str = CoordinationKind.MESSAGE,
        confidence: float | None = None,
        message_id: str | None = None,
        in_reply_to: str | None = None,
        **metadata: Any,
    ) -> CoordinationEvent:
        """Record a coordination edge ``source -> target``."""
        k = CoordinationKind(kind)
        ev = CoordinationEvent(
            agent=source,
            source=source,
            target=target,
            message=truncate(message, self.max_message_chars),
            kind=k,
            confidence=confidence,
            message_id=message_id,
            in_reply_to=in_reply_to,
            metadata=dict(metadata),
        )
        self._record(ev)
        s = self.semconv
        attrs: dict[str, Any] = {
            sc.GEN_AI_OPERATION_NAME: sc.OP_COORDINATE,
            sc.GEN_AI_AGENT_NAME: source,
            s.COORD_SOURCE: source,
            s.COORD_TARGET: target,
            s.COORD_KIND: k.value,
            s.COORD_MESSAGE: ev.message,
        }
        if confidence is not None:
            attrs[s.COORD_CONFIDENCE] = confidence
        if message_id:
            attrs[s.COORD_MESSAGE_ID] = message_id
        if in_reply_to:
            attrs[s.COORD_IN_REPLY_TO] = in_reply_to
        span = self._otel.start_span(
            f"{sc.OP_COORDINATE} {source}->{target}", attributes=attrs, kind=SpanKind.PRODUCER
        )
        span.end()
        return ev

    def memory(
        self,
        agent: str,
        op: MemoryOp | str,
        memory_type: MemoryType | str,
        key: str,
        *,
        confidence: float | None = None,
        hit: bool | None = None,
        eviction_reason: EvictionReason | str | None = None,
        **metadata: Any,
    ) -> MemoryEvent:
        """Record a memory operation (read / write / evict / delete)."""
        ev = MemoryEvent(
            agent=agent,
            op=MemoryOp(op),
            memory_type=MemoryType(memory_type),
            key=key,
            confidence=confidence,
            hit=hit,
            eviction_reason=EvictionReason(eviction_reason) if eviction_reason is not None else None,
            metadata=dict(metadata),
        )
        self._record(ev)
        s = self.semconv
        attrs: dict[str, Any] = {
            sc.GEN_AI_OPERATION_NAME: sc.OP_MEMORY_ACCESS,
            sc.GEN_AI_AGENT_NAME: agent,
            s.MEMORY_AGENT: agent,
            s.MEMORY_OPERATION: ev.op.value,
            s.MEMORY_TYPE: ev.memory_type.value,
            s.MEMORY_KEY: key,
        }
        if confidence is not None:
            attrs[s.MEMORY_CONFIDENCE] = confidence
        if hit is not None:
            attrs[s.MEMORY_HIT] = hit
        if ev.eviction_reason is not None:
            attrs[s.MEMORY_EVICTION_REASON] = ev.eviction_reason.value
        span = self._otel.start_span(f"{sc.OP_MEMORY_ACCESS} {ev.op.value} {key}", attributes=attrs)
        span.end()
        return ev

    @contextmanager
    def contract(
        self, agent: str, action: str, pre_state: Any, *, seed: int | None = None, **metadata: Any
    ) -> Iterator[Contract]:
        """Record an environment contract around an action.

        If ``seed`` is omitted and the episode has a seed, a per-action seed is derived
        from the episode's RNG so that replays reproduce the same sequence of seeds.
        """
        ep = self.current_episode()
        if seed is None and ep is not None and ep.rng is not None:
            seed = ep.rng.randrange(1 << 31)
        c = Contract.begin(agent, action, pre_state, seed=seed, **metadata)
        session = current_replay()
        if session is not None:
            recorded, divergence = session.consume_contract(c)
            c.recorded = recorded
            if divergence is not None:
                session.report(divergence)
        s = self.semconv
        attrs: dict[str, Any] = {
            sc.GEN_AI_OPERATION_NAME: sc.OP_ENVIRONMENT_CONTRACT,
            sc.GEN_AI_AGENT_NAME: agent,
            s.CONTRACT_ACTION: action,
            s.CONTRACT_PRE_STATE_HASH: c.pre_hash,
        }
        if seed is not None:
            attrs[s.CONTRACT_SEED] = seed
        with self._otel.start_as_current_span(
            f"{sc.OP_ENVIRONMENT_CONTRACT} {action}", attributes=attrs
        ) as span:
            try:
                yield c
            finally:
                if c.recorded is not None:
                    check = c.verify()
                    if check is not None and not check.ok and session is not None:
                        session.report_contract_mismatch(c, check)
                ev = c.to_event()
                if ev.post_hash:
                    span.set_attribute(s.CONTRACT_POST_STATE_HASH, ev.post_hash)
                if ev.result is not None:
                    span.set_attribute(s.CONTRACT_RESULT, ev.result)
                if ev.verified is not None:
                    span.set_attribute(s.CONTRACT_VERIFIED, ev.verified)
                self._record(ev, ep)

    def action(
        self,
        agent: str,
        name: str,
        args: Any = None,
        *,
        noop: bool = False,
        state_before: Any = None,
        state_after: Any = None,
    ) -> ActionEvent:
        """Record a discrete action (feeds the redundant-action and oscillation SLOs)."""
        ev = ActionEvent(
            agent=agent,
            name=name,
            args_fingerprint=fingerprint(args),
            noop=noop,
            state_before=fingerprint(state_before) if state_before is not None else None,
            state_after=fingerprint(state_after) if state_after is not None else None,
        )
        self._record(ev)
        return ev

    def state(self, agent: str, state: Any, label: str | None = None) -> StateEvent:
        """Record a snapshot of an agent's state (feeds the oscillation SLO)."""
        ev = StateEvent(agent=agent, state_hash=fingerprint(state), label=label)
        self._record(ev)
        return ev

    def log(self, message: str, *, level: str = "info", agent: str | None = None, **data: Any) -> LogEvent:
        ev = LogEvent(agent=agent or current_agent(), level=level, message=message, data=dict(data))
        self._record(ev)
        return ev

    # ------------------------------------------------------------------ anomalies / replay

    def flag_anomaly(self, reason: str | None = None) -> None:
        """Escalate the adaptive sampler to full capture for the next episodes."""
        if isinstance(self.sampler, AdaptiveSampler):
            self.sampler.escalate(reason)
        self.log(f"anomaly flagged: {reason}", level="warning", reason=reason)

    def replay(
        self,
        episode: Episode | str,
        entrypoint: str | Callable[..., Any] | None = None,
        *,
        strict: bool = False,
        entrypoint_args: dict[str, Any] | None = None,
    ):  # type: ignore[no-untyped-def]
        """Deterministically re-execute a recorded episode. See :func:`ma_trace.replay.run_replay`."""
        from .replay import run_replay

        return run_replay(self, episode, entrypoint, strict=strict, entrypoint_args=entrypoint_args)


def _entrypoint_name(entrypoint: str | Callable[..., Any] | None) -> str | None:
    if entrypoint is None:
        return None
    if isinstance(entrypoint, str):
        return entrypoint
    module = getattr(entrypoint, "__module__", None)
    qualname = getattr(entrypoint, "__qualname__", getattr(entrypoint, "__name__", None))
    if module and qualname:
        return f"{module}:{qualname}"
    return repr(entrypoint)


# ---------------------------------------------------------------------- module-level API

_default: MATrace | None = None
_default_lock = threading.Lock()


def get_tracer() -> MATrace:
    """The process-wide default :class:`MATrace` (created lazily)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = MATrace()
        return _default


def set_tracer(tracer: MATrace | None) -> None:
    global _default
    with _default_lock:
        _default = tracer


def configure(
    service_name: str = "ma-trace",
    *,
    exporter: str | Any | None = None,
    endpoint: str | None = None,
    metrics_exporter: str | Any | None = None,
    store: EpisodeStore | InMemoryEpisodeStore | str | None = None,
    sampler: AdaptiveSampler | None = None,
    namespace: str = sc.DEFAULT_NAMESPACE,
    **kwargs: Any,
) -> MATrace:
    """Create the default tracer, optionally wiring OpenTelemetry export.

    ``exporter``: ``"console"``, ``"otlp"``, a ``SpanExporter`` or ``None`` (no export).
    ``metrics_exporter``: ``"console"``, ``"otlp"``, ``"prometheus"``, a ``MetricExporter``
    or ``None``.
    """
    from .exporters import build_meter_provider, build_tracer_provider

    tracer_provider = None
    meter_provider = None
    metrics_exporters: list[Any] = list(kwargs.pop("metrics_exporters", []) or [])
    if exporter is not None:
        tracer_provider = build_tracer_provider(service_name, exporter, endpoint=endpoint, sampler=sampler)
    if metrics_exporter == "prometheus":
        from .metrics import PrometheusExporter

        metrics_exporters.append(PrometheusExporter())
    elif metrics_exporter is not None:
        meter_provider = build_meter_provider(service_name, metrics_exporter, endpoint=endpoint)
    tracer = MATrace(
        service_name,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        store=store,
        sampler=sampler,
        namespace=namespace,
        metrics_exporters=metrics_exporters,
        **kwargs,
    )
    set_tracer(tracer)
    return tracer
