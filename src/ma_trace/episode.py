# SPDX-License-Identifier: Apache-2.0
"""The :class:`Episode` container: one traced run of a multi-agent workflow."""

from __future__ import annotations

import json
import random
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TypeVar

from .events import (
    ActionEvent,
    AgentEvent,
    CallEvent,
    ContractEvent,
    CoordinationEvent,
    Event,
    MemoryEvent,
    StateEvent,
    event_from_dict,
)

if TYPE_CHECKING:  # pragma: no cover
    from .graph import CoordinationGraph
    from .memory import MemoryTrace
    from .metrics import SLOMetrics

SCHEMA_VERSION = 1
E = TypeVar("E", bound=Event)

_clock_lock = threading.Lock()
_last_wall_time = 0.0


def _wall_time() -> float:
    """``time.time()``, but strictly increasing within the process.

    Wall clocks can advance in coarse steps (about 16 ms on Windows before Python 3.13), so
    episodes and events recorded back to back would otherwise share a timestamp, and the order
    of ``latest`` and ``ma-trace list`` would depend on the random part of the episode id.
    """
    global _last_wall_time
    with _clock_lock:
        now = time.time()
        _last_wall_time = now if now > _last_wall_time else _last_wall_time + 1e-6
        return _last_wall_time


@dataclass
class Episode:
    """An ordered record of everything that happened during one workflow run."""

    id: str
    name: str
    seed: int | None = None
    started_at: float = field(default_factory=_wall_time)
    ended_at: float | None = None
    entrypoint: str | None = None
    entrypoint_args: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    sampled: bool = True
    replay_of: str | None = None
    schema_version: int = SCHEMA_VERSION
    rng: random.Random | None = field(default=None, repr=False, compare=False)

    # -- construction -------------------------------------------------------

    @staticmethod
    def new_id(now: float | None = None) -> str:
        stamp = datetime.fromtimestamp(now or time.time(), tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"ep-{stamp}-{secrets.token_hex(3)}"

    # -- accessors ----------------------------------------------------------

    @property
    def duration_s(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, end - self.started_at)

    @property
    def finished(self) -> bool:
        return self.ended_at is not None

    def events_of(self, cls: type[E]) -> list[E]:
        return [e for e in self.events if isinstance(e, cls)]

    @property
    def coordination_events(self) -> list[CoordinationEvent]:
        return self.events_of(CoordinationEvent)

    @property
    def memory_events(self) -> list[MemoryEvent]:
        return self.events_of(MemoryEvent)

    @property
    def contract_events(self) -> list[ContractEvent]:
        return self.events_of(ContractEvent)

    @property
    def calls(self) -> list[CallEvent]:
        return self.events_of(CallEvent)

    @property
    def actions(self) -> list[ActionEvent]:
        return self.events_of(ActionEvent)

    @property
    def states(self) -> list[StateEvent]:
        return self.events_of(StateEvent)

    @property
    def agents(self) -> list[str]:
        """Agents that appear anywhere in the episode, in first-seen order."""
        seen: dict[str, None] = {}
        for e in self.events:
            if e.agent:
                seen.setdefault(e.agent, None)
            if isinstance(e, CoordinationEvent):
                seen.setdefault(e.source, None)
                seen.setdefault(e.target, None)
        return list(seen)

    def agent_durations(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for e in self.events_of(AgentEvent):
            if e.phase == "end" and e.agent and e.duration_s is not None:
                totals[e.agent] = totals.get(e.agent, 0.0) + e.duration_s
        return totals

    # -- derived views ------------------------------------------------------

    def coordination_graph(self) -> CoordinationGraph:
        from .graph import CoordinationGraph

        return CoordinationGraph.from_events(self.coordination_events)

    def memory_trace(self) -> MemoryTrace:
        from .memory import MemoryTrace

        return MemoryTrace(self.memory_events)

    def metrics(self) -> SLOMetrics:
        from .metrics import compute_metrics

        return compute_metrics(self)

    # -- (de)serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "seed": self.seed,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "entrypoint": self.entrypoint,
            "entrypoint_args": self.entrypoint_args,
            "attributes": self.attributes,
            "sampled": self.sampled,
            "replay_of": self.replay_of,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        ep = cls(
            id=data["id"],
            name=data.get("name", ""),
            seed=data.get("seed"),
            started_at=data.get("started_at", 0.0),
            ended_at=data.get("ended_at"),
            entrypoint=data.get("entrypoint"),
            entrypoint_args=dict(data.get("entrypoint_args") or {}),
            attributes=dict(data.get("attributes") or {}),
            sampled=bool(data.get("sampled", True)),
            replay_of=data.get("replay_of"),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )
        ep.events = [event_from_dict(e) for e in data.get("events", [])]
        if ep.seed is not None:
            ep.rng = random.Random(ep.seed)
        return ep

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Episode:
        return cls.from_dict(json.loads(text))

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.type] = counts.get(e.type, 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        started = datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(timespec="seconds")
        seed = f" seed={self.seed}" if self.seed is not None else ""
        replay = f" replay_of={self.replay_of}" if self.replay_of else ""
        return (
            f"{self.id}  {self.name!r}  started={started}  duration={self.duration_s:.3f}s"
            f"{seed}{replay}  agents={len(self.agents)}  events[{parts}]"
        )
