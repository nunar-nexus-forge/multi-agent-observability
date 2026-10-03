from __future__ import annotations

from ma_trace.events import (
    CallEvent,
    CallType,
    CoordinationEvent,
    CoordinationKind,
    MemoryEvent,
    MemoryOp,
    MemoryType,
    canonical_json,
    event_from_dict,
    fingerprint,
    json_safe,
    truncate,
)


class Opaque:
    pass


def test_canonical_json_is_sorted_and_compact():
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'


def test_fingerprint_is_stable_and_strips_addresses():
    a = fingerprint({"x": Opaque()})
    b = fingerprint({"x": Opaque()})
    assert a == b  # different objects, same repr after address stripping
    assert fingerprint({"x": 1}) != fingerprint({"x": 2})
    assert len(fingerprint("s")) == 64


def test_json_safe_paths():
    assert json_safe({"a": 1}) == ({"a": 1}, True)
    value, ok = json_safe(Opaque())
    assert ok is False and "__repr__" in value
    value, ok = json_safe({"k": {1, 2}})
    assert ok is True and value == {"k": [1, 2]}


def test_truncate():
    assert truncate("abcdef", 4) == "abc…"
    assert truncate({"a": 1}, 100) == '{"a":1}'


def test_event_roundtrip_with_enums():
    e = CoordinationEvent(
        seq=3,
        t=1.0,
        agent="a",
        source="a",
        target="b",
        message="m",
        kind=CoordinationKind.DELEGATION,
        confidence=0.5,
    )
    d = e.to_dict()
    assert d["type"] == "coordination" and d["kind"] == "delegation"
    back = event_from_dict(d)
    assert isinstance(back, CoordinationEvent) and back.kind is CoordinationKind.DELEGATION and back.seq == 3

    m = MemoryEvent(agent="a", op=MemoryOp.WRITE, memory_type=MemoryType.SEMANTIC, key="k")
    back_m = event_from_dict(m.to_dict())
    assert (
        isinstance(back_m, MemoryEvent)
        and back_m.op is MemoryOp.WRITE
        and back_m.memory_type is MemoryType.SEMANTIC
    )

    c = CallEvent(agent="a", call_type=CallType.TOOL, name="search", output={"r": 1}, output_recorded=True)
    back_c = event_from_dict(c.to_dict())
    assert isinstance(back_c, CallEvent) and back_c.call_type is CallType.TOOL and back_c.output == {"r": 1}


def test_unknown_keys_are_ignored():
    back = event_from_dict({"type": "coordination", "source": "a", "target": "b", "future_field": 1})
    assert isinstance(back, CoordinationEvent) and back.source == "a"
