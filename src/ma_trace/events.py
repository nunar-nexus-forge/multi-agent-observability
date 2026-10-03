# SPDX-License-Identifier: Apache-2.0
"""Event model recorded inside an episode.

Every observable thing that happens while an episode is open is appended to the
episode as one of the dataclasses below. They are plain data: JSON-serialisable
through :meth:`Event.to_dict` and reconstructable through :func:`event_from_dict`.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar


class CoordinationKind(str, enum.Enum):
    """Semantics of a coordination edge between two agents."""

    MESSAGE = "message"
    DELEGATION = "delegation"
    NEGOTIATION = "negotiation"
    CONSENSUS = "consensus"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    HANDOFF = "handoff"


class MemoryOp(str, enum.Enum):
    READ = "read"
    WRITE = "write"
    EVICT = "evict"
    DELETE = "delete"


class MemoryType(str, enum.Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"


class EvictionReason(str, enum.Enum):
    CAPACITY = "capacity"
    POLICY = "policy"
    TTL = "ttl"
    STALE = "stale"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class CallType(str, enum.Enum):
    LLM = "llm"
    TOOL = "tool"


_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, enum.Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:  # pragma: no cover - defensive
            pass
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # pragma: no cover - defensive
            pass
    return repr(obj)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding (sorted keys, compact separators, repr fallback)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default, ensure_ascii=False)


def fingerprint(obj: Any) -> str:
    """SHA-256 fingerprint of ``obj`` based on :func:`canonical_json`.

    Python object addresses (``<Foo object at 0x...>``) are stripped before hashing so
    that the fingerprint of an un-serialisable object is stable across processes.
    """
    text = _ADDRESS_RE.sub("", canonical_json(obj))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_safe(obj: Any) -> tuple[Any, bool]:
    """Return ``(value, serialisable)`` where ``value`` can be stored as JSON.

    If ``obj`` (or a natural conversion of it - pydantic ``model_dump``, dataclass,
    ``to_dict``) is JSON serialisable it is returned with ``True``. Otherwise a
    ``{"__repr__": ...}`` placeholder is returned with ``False`` so that the episode
    can still be persisted, while replay knows that the output is not available.
    """
    try:
        json.dumps(obj)
        return obj, True
    except (TypeError, ValueError):
        pass
    try:
        converted = json.loads(canonical_json(obj))
    except (TypeError, ValueError):  # pragma: no cover - canonical_json never raises in practice
        return {"__repr__": repr(obj)}, False
    # canonical_json falls back to ``repr`` for unknown objects; treat a bare
    # string result for a non-string input as "not really serialisable".
    if isinstance(converted, str) and not isinstance(obj, str):
        return {"__repr__": converted}, False
    return converted, True


def truncate(text: Any, limit: int) -> str:
    s = text if isinstance(text, str) else canonical_json(text)
    if limit and len(s) > limit:
        return s[: limit - 1] + "…"
    return s


# --- Events -----------------------------------------------------------------


@dataclass
class Event:
    """Base class for everything recorded in an episode."""

    type: ClassVar[str] = "event"
    seq: int = 0
    t: float = 0.0
    agent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = json.loads(canonical_json(dataclasses.asdict(self)))
        data["type"] = self.type
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue
            enum_cls = _ENUM_FIELDS.get(key)
            if enum_cls is not None and isinstance(value, str):
                value = enum_cls(value)
            kwargs[key] = value
        return cls(**kwargs)


@dataclass
class AgentEvent(Event):
    """Start/end of an agent invocation."""

    type: ClassVar[str] = "agent"
    phase: str = "start"  # "start" | "end"
    duration_s: float | None = None
    error: str | None = None


@dataclass
class CoordinationEvent(Event):
    """A directed coordination edge ``source -> target`` carrying a message."""

    type: ClassVar[str] = "coordination"
    source: str = ""
    target: str = ""
    message: str = ""
    kind: CoordinationKind = CoordinationKind.MESSAGE
    confidence: float | None = None
    message_id: str | None = None
    in_reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEvent(Event):
    """One access to an agent memory system."""

    type: ClassVar[str] = "memory"
    op: MemoryOp = MemoryOp.READ
    memory_type: MemoryType = MemoryType.EPISODIC
    key: str = ""
    confidence: float | None = None
    hit: bool | None = None
    eviction_reason: EvictionReason | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractEvent(Event):
    """An environment contract: pre-state, post-state, seed and result of an action."""

    type: ClassVar[str] = "contract"
    action: str = ""
    pre_hash: str = ""
    post_hash: str | None = None
    seed: int | None = None
    result: str | None = None
    verified: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallEvent(Event):
    """A recorded non-deterministic call (LLM inference or tool execution)."""

    type: ClassVar[str] = "call"
    call_type: CallType = CallType.LLM
    name: str = ""
    provider: str | None = None
    model: str | None = None
    index: int = 0
    input_fingerprint: str = ""
    output: Any = None
    output_fingerprint: str | None = None
    output_recorded: bool = False
    duration_s: float = 0.0
    error: str | None = None
    replayed: bool = False


@dataclass
class ActionEvent(Event):
    """A discrete action taken by an agent (used for the redundant-action SLO)."""

    type: ClassVar[str] = "action"
    name: str = ""
    args_fingerprint: str = ""
    noop: bool = False
    state_before: str | None = None
    state_after: str | None = None


@dataclass
class StateEvent(Event):
    """A snapshot of an agent's (hashed) state, used for oscillation detection."""

    type: ClassVar[str] = "state"
    state_hash: str = ""
    label: str | None = None


@dataclass
class LogEvent(Event):
    type: ClassVar[str] = "log"
    level: str = "info"
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


_ENUM_FIELDS: dict[str, type[enum.Enum]] = {
    "kind": CoordinationKind,
    "op": MemoryOp,
    "memory_type": MemoryType,
    "eviction_reason": EvictionReason,
    "call_type": CallType,
}

EVENT_TYPES: dict[str, type[Event]] = {
    cls.type: cls
    for cls in (
        AgentEvent,
        CoordinationEvent,
        MemoryEvent,
        ContractEvent,
        CallEvent,
        ActionEvent,
        StateEvent,
        LogEvent,
    )
}


def event_from_dict(data: dict[str, Any]) -> Event:
    """Rebuild an event from its :meth:`Event.to_dict` form."""
    cls = EVENT_TYPES.get(data.get("type", ""), Event)
    return cls.from_dict(data)
