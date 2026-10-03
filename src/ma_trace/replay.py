# SPDX-License-Identifier: Apache-2.0
"""Deterministic replay of recorded episodes.

Replay re-runs the workflow's entrypoint with the episode's seed while every
recorded LLM/tool call is served from the episode instead of being executed.
Environment contracts are verified against their recorded pre/post-state hashes.
Any difference between the live run and the recording is reported as a
:class:`Divergence`; a replay with no divergences is *deterministic*.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .context import _current_replay
from .contracts import Contract, ContractCheck
from .episode import Episode
from .events import CallEvent, ContractEvent

if TYPE_CHECKING:  # pragma: no cover
    from .tracer import MATrace


class ReplayError(RuntimeError):
    """Replay could not be started (missing entrypoint, unknown episode, ...)."""


@dataclass
class Divergence:
    """One difference between the live run and the recording.

    ``kind`` is one of ``input_mismatch``, ``unexpected_call``, ``missing_call``,
    ``output_unavailable``, ``unexpected_contract``, ``missing_contract``,
    ``contract_pre_mismatch``, ``contract_post_mismatch``, ``action_sequence_mismatch``,
    ``coordination_mismatch`` or ``error``.
    """

    kind: str
    agent: str | None = None
    name: str | None = None
    index: int | None = None
    expected: str | None = None
    actual: str | None = None
    detail: str | None = None

    def describe(self) -> str:
        where = " ".join(p for p in (self.agent, self.name) if p)
        parts = [self.kind, where]
        if self.expected or self.actual:
            parts.append(f"expected={_short(self.expected)} actual={_short(self.actual)}")
        if self.detail:
            parts.append(self.detail)
        return "  ".join(p for p in parts if p)


class ReplayDivergenceError(ReplayError):
    def __init__(self, divergence: Divergence) -> None:
        super().__init__(divergence.describe())
        self.divergence = divergence


def _short(h: str | None) -> str:
    if not h:
        return "-"
    return h[:12] if len(h) > 16 else h


class ReplaySession:
    """Serves recorded calls and contracts to the tracer during a replay."""

    def __init__(self, recorded: Episode, *, strict: bool = False) -> None:
        self.recorded = recorded
        self.strict = strict
        self.divergences: list[Divergence] = []
        self.matched_calls = 0
        self.executed_live = 0
        self.contracts_checked = 0
        self.contracts_passed = 0
        self._calls: dict[tuple[str | None, str, str], deque[CallEvent]] = {}
        for c in sorted(recorded.calls, key=lambda e: e.seq):
            self._calls.setdefault((c.agent, c.call_type.value, c.name), deque()).append(c)
        self._contracts: dict[tuple[str | None, str], deque[ContractEvent]] = {}
        for ce in sorted(recorded.contract_events, key=lambda e: e.seq):
            self._contracts.setdefault((ce.agent, ce.action), deque()).append(ce)

    # -- calls ----------------------------------------------------------------

    def consume_call(self, live: CallEvent) -> tuple[CallEvent | None, Divergence | None]:
        key = (live.agent, live.call_type.value, live.name)
        queue = self._calls.get(key)
        if not queue:
            self.executed_live += 1
            return None, Divergence(
                "unexpected_call",
                live.agent,
                live.name,
                live.index,
                detail="no recorded call left for this agent/name",
            )
        rec = queue.popleft()
        self.matched_calls += 1
        if (
            rec.input_fingerprint
            and live.input_fingerprint
            and rec.input_fingerprint != live.input_fingerprint
        ):
            return rec, Divergence(
                "input_mismatch",
                live.agent,
                live.name,
                rec.index,
                rec.input_fingerprint,
                live.input_fingerprint,
            )
        return rec, None

    def report_output_unavailable(self, rec: CallEvent) -> None:
        self.executed_live += 1
        self.report(
            Divergence(
                "output_unavailable",
                rec.agent,
                rec.name,
                rec.index,
                detail="recorded output was not JSON-serialisable; executing live",
            )
        )

    # -- contracts ------------------------------------------------------------

    def consume_contract(self, live: Contract) -> tuple[ContractEvent | None, Divergence | None]:
        key = (live.agent, live.action)
        queue = self._contracts.get(key)
        if not queue:
            return None, Divergence(
                "unexpected_contract",
                live.agent,
                live.action,
                detail="no recorded contract left for this agent/action",
            )
        rec = queue.popleft()
        self.contracts_checked += 1
        if rec.pre_hash != live.pre_hash:
            return rec, Divergence(
                "contract_pre_mismatch", live.agent, live.action, None, rec.pre_hash, live.pre_hash
            )
        return rec, None

    def report_contract_mismatch(self, live: Contract, check: ContractCheck) -> None:
        if check.post_match is False:
            self.report(
                Divergence(
                    "contract_post_mismatch",
                    live.agent,
                    live.action,
                    None,
                    check.recorded_post,
                    check.actual_post,
                )
            )

    # -- bookkeeping ----------------------------------------------------------

    def report(self, divergence: Divergence) -> None:
        self.divergences.append(divergence)
        if self.strict:
            raise ReplayDivergenceError(divergence)

    def finish(self) -> None:
        for (agent, _ctype, name), queue in self._calls.items():
            for rec in queue:
                self.divergences.append(
                    Divergence(
                        "missing_call",
                        agent,
                        name,
                        rec.index,
                        detail="recorded call was never made during replay",
                    )
                )
        for (agent, action), cqueue in self._contracts.items():
            for _rec in cqueue:
                self.divergences.append(
                    Divergence(
                        "missing_contract",
                        agent,
                        action,
                        detail="recorded contract was never opened during replay",
                    )
                )
        if self.strict and self.divergences:
            raise ReplayDivergenceError(self.divergences[0])

    @property
    def deterministic(self) -> bool:
        return not self.divergences


@dataclass
class ReplayReport:
    recorded_episode_id: str
    replay_episode_id: str | None
    entrypoint: str | None
    deterministic: bool
    matched_calls: int
    executed_live: int
    contracts_checked: int
    contracts_passed: int
    divergences: list[Divergence] = field(default_factory=list)
    recorded_metrics: dict[str, Any] = field(default_factory=dict)
    replayed_metrics: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    error: str | None = None
    result: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("result", None)
        return d

    def summary(self) -> str:
        status = "DETERMINISTIC" if self.deterministic else f"{len(self.divergences)} DIVERGENCE(S)"
        lines = [
            f"replay of {self.recorded_episode_id} -> {self.replay_episode_id}: {status}",
            f"  entrypoint        : {self.entrypoint}",
            f"  calls replayed    : {self.matched_calls}  (executed live: {self.executed_live})",
            f"  contracts checked : {self.contracts_checked}  (passed: {self.contracts_passed})",
            f"  duration          : {self.duration_s:.3f}s",
        ]
        if self.error:
            lines.append(f"  error             : {self.error}")
        for key in ("coordination_overhead", "redundant_action_rate", "oscillation_rate", "memory_hit_ratio"):
            a = self.recorded_metrics.get(key)
            b = self.replayed_metrics.get(key)
            if a is not None and b is not None:
                lines.append(f"  {key:<18}: recorded={a:.3f} replayed={b:.3f}")
        for d in self.divergences:
            lines.append(f"  ! {d.describe()}")
        return "\n".join(lines)


def structural_divergences(recorded: Episode, replayed: Episode) -> list[Divergence]:
    """Compare the action sequence and the coordination-edge sequence of two episodes."""
    out: list[Divergence] = []
    rec_actions = [(a.agent, a.name, a.args_fingerprint) for a in recorded.actions]
    rep_actions = [(a.agent, a.name, a.args_fingerprint) for a in replayed.actions]
    if rec_actions != rep_actions:
        idx = _first_difference(rec_actions, rep_actions)
        at = rec_actions[idx] if idx < len(rec_actions) else rep_actions[idx]
        out.append(
            Divergence(
                "action_sequence_mismatch",
                agent=at[0],
                name=at[1],
                index=idx,
                expected=f"{len(rec_actions)} actions",
                actual=f"{len(rep_actions)} actions",
                detail=f"first difference at action #{idx}",
            )
        )
    rec_edges = [(e.source, e.target, e.kind.value) for e in recorded.coordination_events]
    rep_edges = [(e.source, e.target, e.kind.value) for e in replayed.coordination_events]
    if rec_edges != rep_edges:
        idx = _first_difference(rec_edges, rep_edges)
        at = rec_edges[idx] if idx < len(rec_edges) else rep_edges[idx]
        out.append(
            Divergence(
                "coordination_mismatch",
                agent=at[0],
                name=f"{at[0]}->{at[1]}",
                index=idx,
                expected=f"{len(rec_edges)} edges",
                actual=f"{len(rep_edges)} edges",
                detail=f"first difference at edge #{idx}",
            )
        )
    return out


def _first_difference(a: list[Any], b: list[Any]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def resolve_entrypoint(spec: str | Callable[..., Any] | None) -> Callable[..., Any]:
    if spec is None:
        raise ReplayError("no entrypoint: pass one to replay() or record the episode with entrypoint=...")
    if callable(spec):
        return spec
    if ":" not in spec:
        raise ReplayError(f"entrypoint must look like 'package.module:function', got {spec!r}")
    module_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ReplayError(f"cannot import {module_name!r}: {e}") from e
    target: Any = module
    for part in attr.split("."):
        try:
            target = getattr(target, part)
        except AttributeError as e:
            raise ReplayError(f"{spec!r} not found") from e
    if not callable(target):
        raise ReplayError(f"{spec!r} is not callable")
    return target


def run_replay(
    tracer: MATrace,
    episode: Episode | str,
    entrypoint: str | Callable[..., Any] | None = None,
    *,
    strict: bool = False,
    entrypoint_args: dict[str, Any] | None = None,
) -> ReplayReport:
    """Replay ``episode`` (an :class:`Episode` or a store reference) with ``tracer``.

    The entrypoint is called with ``entrypoint_args`` (default: the arguments recorded
    with the episode). With ``strict=True`` the first divergence raises
    :class:`ReplayDivergenceError`; otherwise all divergences are collected in the report.
    """
    if isinstance(episode, str):
        if tracer.store is None:
            raise ReplayError("tracer has no episode store; pass an Episode object")
        try:
            recorded = tracer.store.load(episode)
        except KeyError as e:
            raise ReplayError(str(e)) from e
    else:
        recorded = episode
    fn = resolve_entrypoint(entrypoint or recorded.entrypoint)
    args = dict(recorded.entrypoint_args if entrypoint_args is None else entrypoint_args)
    session = ReplaySession(recorded, strict=strict)
    token = _current_replay.set(session)
    start = time.perf_counter()
    error: str | None = None
    result: Any = None
    replay_ep: Episode | None = None
    try:
        with tracer.episode(
            f"replay:{recorded.name}",
            seed=recorded.seed,
            entrypoint=recorded.entrypoint,
            entrypoint_args=args,
            attributes={"replay_of": recorded.id},
            record=True,
            replay_of=recorded.id,
        ) as ep:
            replay_ep = ep
            try:
                if inspect.iscoroutinefunction(fn):
                    result = asyncio.run(fn(**args))
                else:
                    result = fn(**args)
            except ReplayDivergenceError:
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                session.report(Divergence("error", detail=error))
        if replay_ep is not None:
            session.divergences.extend(structural_divergences(recorded, replay_ep))
        session.finish()
    finally:
        _current_replay.reset(token)
    for ce in replay_ep.contract_events if replay_ep else []:
        if ce.verified:
            session.contracts_passed += 1
    return ReplayReport(
        recorded_episode_id=recorded.id,
        replay_episode_id=replay_ep.id if replay_ep else None,
        entrypoint=recorded.entrypoint
        if entrypoint is None
        else (
            entrypoint
            if isinstance(entrypoint, str)
            else getattr(entrypoint, "__qualname__", repr(entrypoint))
        ),
        deterministic=session.deterministic and error is None,
        matched_calls=session.matched_calls,
        executed_live=session.executed_live,
        contracts_checked=session.contracts_checked,
        contracts_passed=session.contracts_passed,
        divergences=list(session.divergences),
        recorded_metrics=recorded.metrics().to_dict(),
        replayed_metrics=replay_ep.metrics().to_dict() if replay_ep else {},
        duration_s=time.perf_counter() - start,
        error=error,
        result=result,
    )
