# SPDX-License-Identifier: Apache-2.0
"""Memory operation traces and a drop-in wrapper for dict-like memory stores."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, MutableMapping
from typing import TYPE_CHECKING, Any

from .events import EvictionReason, MemoryEvent, MemoryOp, MemoryType

if TYPE_CHECKING:  # pragma: no cover
    from .tracer import MATrace


class MemoryTrace:
    """Analysis helpers over a sequence of :class:`~ma_trace.events.MemoryEvent`."""

    def __init__(self, events: Iterable[MemoryEvent]) -> None:
        self.events = list(events)

    @property
    def reads(self) -> list[MemoryEvent]:
        return [e for e in self.events if e.op == MemoryOp.READ]

    @property
    def writes(self) -> list[MemoryEvent]:
        return [e for e in self.events if e.op == MemoryOp.WRITE]

    @property
    def evictions(self) -> list[MemoryEvent]:
        return [e for e in self.events if e.op == MemoryOp.EVICT]

    def hit_ratio(self, reads_only: bool = False) -> float:
        """Memory hit ratio.

        MA-Trace defines it as successful retrievals divided by *total memory
        accesses* (reads and writes). Pass ``reads_only=True`` for the classic
        cache-hit-ratio denominator (reads only).
        """
        hits = sum(1 for e in self.reads if e.hit)
        denom = len(self.reads) if reads_only else len(self.reads) + len(self.writes)
        return hits / denom if denom else 0.0

    def by_agent(self) -> dict[str, MemoryTrace]:
        groups: dict[str, list[MemoryEvent]] = defaultdict(list)
        for e in self.events:
            groups[e.agent or "?"].append(e)
        return {a: MemoryTrace(evs) for a, evs in groups.items()}

    def by_type(self) -> dict[str, MemoryTrace]:
        groups: dict[str, list[MemoryEvent]] = defaultdict(list)
        for e in self.events:
            groups[e.memory_type.value].append(e)
        return {t: MemoryTrace(evs) for t, evs in groups.items()}

    def eviction_reasons(self) -> dict[str, int]:
        return dict(Counter((e.eviction_reason or EvictionReason.UNKNOWN).value for e in self.evictions))

    def thrashing_keys(self, min_cycles: int = 2) -> list[tuple[str, int]]:
        """Keys that were written and evicted repeatedly (write->evict cycles)."""
        cycles: Counter[str] = Counter()
        last_written: set[str] = set()
        for e in self.events:
            if e.op == MemoryOp.WRITE:
                last_written.add(e.key)
            elif e.op in (MemoryOp.EVICT, MemoryOp.DELETE) and e.key in last_written:
                cycles[e.key] += 1
                last_written.discard(e.key)
        return sorted(((k, n) for k, n in cycles.items() if n >= min_cycles), key=lambda x: -x[1])

    def low_confidence_reads(self, threshold: float = 0.5) -> list[MemoryEvent]:
        return [e for e in self.reads if e.confidence is not None and e.confidence < threshold]

    def misses(self) -> list[MemoryEvent]:
        return [e for e in self.reads if e.hit is False]

    def summary(self) -> dict[str, Any]:
        return {
            "accesses": len(self.events),
            "reads": len(self.reads),
            "writes": len(self.writes),
            "evictions": len(self.evictions),
            "hit_ratio": round(self.hit_ratio(), 4),
            "hit_ratio_reads_only": round(self.hit_ratio(reads_only=True), 4),
            "eviction_reasons": self.eviction_reasons(),
            "thrashing_keys": self.thrashing_keys(),
        }


class TracedMemory(MutableMapping[str, Any]):
    """Wrap any dict-like store so every access is recorded as a memory operation.

    >>> mem = TracedMemory({}, agent="planner", memory_type=MemoryType.EPISODIC, tracer=tracer)
    >>> mem["goal"] = "buy milk"          # WRITE
    >>> mem.get("goal")                    # READ (hit)
    >>> mem.evict("goal", EvictionReason.CAPACITY)
    """

    def __init__(
        self,
        store: MutableMapping[str, Any] | None,
        *,
        agent: str,
        memory_type: MemoryType | str = MemoryType.EPISODIC,
        tracer: MATrace | None = None,
        capacity: int | None = None,
    ) -> None:
        self._store: MutableMapping[str, Any] = store if store is not None else {}
        self.agent = agent
        self.memory_type = MemoryType(memory_type)
        self._tracer = tracer
        self.capacity = capacity
        self._order: list[str] = list(self._store.keys())

    def _t(self) -> MATrace:
        if self._tracer is None:
            from .tracer import get_tracer

            self._tracer = get_tracer()
        return self._tracer

    # MutableMapping protocol ------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        hit = key in self._store
        self._t().memory(self.agent, MemoryOp.READ, self.memory_type, key, hit=hit)
        return self._store[key]

    def get(self, key: str, default: Any = None, *, confidence: float | None = None) -> Any:  # type: ignore[override]
        hit = key in self._store
        self._t().memory(self.agent, MemoryOp.READ, self.memory_type, key, hit=hit, confidence=confidence)
        return self._store.get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def set(self, key: str, value: Any, *, confidence: float | None = None) -> None:
        if self.capacity is not None and key not in self._store and len(self._store) >= self.capacity:
            victim = self._order.pop(0)
            self._store.pop(victim, None)
            self._t().memory(
                self.agent, MemoryOp.EVICT, self.memory_type, victim, eviction_reason=EvictionReason.CAPACITY
            )
        if key not in self._store:
            self._order.append(key)
        self._store[key] = value
        self._t().memory(self.agent, MemoryOp.WRITE, self.memory_type, key, confidence=confidence)

    def __delitem__(self, key: str) -> None:
        del self._store[key]
        if key in self._order:
            self._order.remove(key)
        self._t().memory(self.agent, MemoryOp.DELETE, self.memory_type, key)

    def evict(self, key: str, reason: EvictionReason | str = EvictionReason.POLICY) -> None:
        self._store.pop(key, None)
        if key in self._order:
            self._order.remove(key)
        self._t().memory(
            self.agent, MemoryOp.EVICT, self.memory_type, key, eviction_reason=EvictionReason(reason)
        )

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def raw(self) -> MutableMapping[str, Any]:
        return self._store
