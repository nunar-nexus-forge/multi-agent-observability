from __future__ import annotations

import ma_trace as mt
from ma_trace.events import EvictionReason, MemoryEvent, MemoryOp, MemoryType
from ma_trace.memory import MemoryTrace, TracedMemory


def mem(op, key, hit=None, reason=None, conf=None, seq=0):
    return MemoryEvent(
        seq=seq,
        agent="a",
        op=op,
        memory_type=MemoryType.EPISODIC,
        key=key,
        hit=hit,
        eviction_reason=reason,
        confidence=conf,
    )


def test_hit_ratio_both_denominators():
    t = MemoryTrace(
        [mem(MemoryOp.READ, "k", hit=True), mem(MemoryOp.READ, "k", hit=False), mem(MemoryOp.WRITE, "k")]
    )
    assert t.hit_ratio() == 1 / 3
    assert t.hit_ratio(reads_only=True) == 0.5
    assert MemoryTrace([]).hit_ratio() == 0.0


def test_eviction_summary_and_thrashing():
    events = [
        mem(MemoryOp.WRITE, "k", seq=1),
        mem(MemoryOp.EVICT, "k", reason=EvictionReason.CAPACITY, seq=2),
        mem(MemoryOp.WRITE, "k", seq=3),
        mem(MemoryOp.EVICT, "k", reason=EvictionReason.CAPACITY, seq=4),
        mem(MemoryOp.EVICT, "other", reason=EvictionReason.POLICY, seq=5),
    ]
    t = MemoryTrace(events)
    assert t.eviction_reasons() == {"capacity": 2, "policy": 1}
    assert t.thrashing_keys() == [("k", 2)]
    assert t.summary()["evictions"] == 3


def test_traced_memory_records_into_episode(tracer):
    with mt.episode("m") as ep:
        m = TracedMemory({}, agent="planner", memory_type="episodic", tracer=tracer, capacity=2)
        m["a"] = 1
        m["b"] = 2
        m["c"] = 3  # evicts "a" (capacity)
        assert m.get("b") == 2
        assert m.get("zzz") is None
        m.evict("b", EvictionReason.POLICY)
        assert "c" in m and len(m) == 1
    trace = ep.memory_trace()
    assert len(trace.writes) == 3
    assert len(trace.evictions) == 2
    assert trace.eviction_reasons() == {"capacity": 1, "policy": 1}
    assert [r.hit for r in trace.reads] == [True, False]
    assert trace.hit_ratio(reads_only=True) == 0.5
