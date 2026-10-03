# SPDX-License-Identifier: Apache-2.0
"""Span attribute names and well-known values.

MA-Trace emits two families of attributes:

* the standard OpenTelemetry GenAI attributes that already exist in the
  semantic conventions (operation name, agent name, tool name, provider,
  model), and
* MA-Trace *extension* attributes for coordination edges, memory operations,
  environment contracts, recorded calls and episodes.

Extension attributes are emitted under the ``ma_trace.`` namespace by default
so that they never collide with future official names. ``docs/semantic-conventions.md``
in the repository documents the proposed ``gen_ai.*`` spelling of every
extension attribute; pass ``namespace="gen_ai"`` to :class:`ma_trace.MATrace`
to emit those instead.
"""

from __future__ import annotations

# --- Standard OpenTelemetry GenAI attribute names -----------------------------
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
ERROR_TYPE = "error.type"

# --- Well-known ``gen_ai.operation.name`` values -------------------------------
OP_INVOKE_AGENT = "invoke_agent"
OP_INVOKE_WORKFLOW = "invoke_workflow"
OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"
# Extension operation values introduced by MA-Trace.
OP_COORDINATE = "coordinate"
OP_MEMORY_ACCESS = "memory_access"
OP_ENVIRONMENT_CONTRACT = "environment_contract"

DEFAULT_NAMESPACE = "ma_trace"
PROPOSED_NAMESPACE = "gen_ai"

# Relative names of the extension attributes (joined with the namespace).
_EXTENSION_KEYS: dict[str, str] = {
    "EPISODE_ID": "episode.id",
    "EPISODE_NAME": "episode.name",
    "EPISODE_SEED": "episode.seed",
    "EPISODE_SAMPLED": "episode.sampled",
    "EPISODE_REPLAY_OF": "episode.replay_of",
    "COORD_SOURCE": "coordination.source",
    "COORD_TARGET": "coordination.target",
    "COORD_KIND": "coordination.kind",
    "COORD_MESSAGE": "coordination.message",
    "COORD_CONFIDENCE": "coordination.confidence",
    "COORD_MESSAGE_ID": "coordination.message_id",
    "COORD_IN_REPLY_TO": "coordination.in_reply_to",
    "MEMORY_AGENT": "memory.agent",
    "MEMORY_OPERATION": "memory.operation",
    "MEMORY_TYPE": "memory.type",
    "MEMORY_KEY": "memory.key",
    "MEMORY_CONFIDENCE": "memory.confidence",
    "MEMORY_HIT": "memory.hit",
    "MEMORY_EVICTION_REASON": "memory.eviction_reason",
    "CONTRACT_ACTION": "contract.action",
    "CONTRACT_PRE_STATE_HASH": "contract.pre_state_hash",
    "CONTRACT_POST_STATE_HASH": "contract.post_state_hash",
    "CONTRACT_SEED": "contract.seed",
    "CONTRACT_RESULT": "contract.result",
    "CONTRACT_VERIFIED": "contract.verified",
    "CALL_INDEX": "call.index",
    "CALL_INPUT_FINGERPRINT": "call.input_fingerprint",
    "CALL_OUTPUT_FINGERPRINT": "call.output_fingerprint",
    "CALL_REPLAYED": "call.replayed",
    "ACTION_NAME": "action.name",
    "ACTION_FINGERPRINT": "action.fingerprint",
    "ACTION_NOOP": "action.noop",
    "STATE_HASH": "state.hash",
    "STATE_LABEL": "state.label",
}


class SemConv:
    """Resolved attribute names for one namespace.

    >>> SemConv().COORD_SOURCE
    'ma_trace.coordination.source'
    >>> SemConv("gen_ai").COORD_SOURCE
    'gen_ai.coordination.source'
    """

    EPISODE_ID: str
    EPISODE_NAME: str
    EPISODE_SEED: str
    EPISODE_SAMPLED: str
    EPISODE_REPLAY_OF: str
    COORD_SOURCE: str
    COORD_TARGET: str
    COORD_KIND: str
    COORD_MESSAGE: str
    COORD_CONFIDENCE: str
    COORD_MESSAGE_ID: str
    COORD_IN_REPLY_TO: str
    MEMORY_AGENT: str
    MEMORY_OPERATION: str
    MEMORY_TYPE: str
    MEMORY_KEY: str
    MEMORY_CONFIDENCE: str
    MEMORY_HIT: str
    MEMORY_EVICTION_REASON: str
    CONTRACT_ACTION: str
    CONTRACT_PRE_STATE_HASH: str
    CONTRACT_POST_STATE_HASH: str
    CONTRACT_SEED: str
    CONTRACT_RESULT: str
    CONTRACT_VERIFIED: str
    CALL_INDEX: str
    CALL_INPUT_FINGERPRINT: str
    CALL_OUTPUT_FINGERPRINT: str
    CALL_REPLAYED: str
    ACTION_NAME: str
    ACTION_FINGERPRINT: str
    ACTION_NOOP: str
    STATE_HASH: str
    STATE_LABEL: str

    def __init__(self, namespace: str = DEFAULT_NAMESPACE) -> None:
        if not namespace or not namespace.replace("_", "").replace(".", "").isalnum():
            raise ValueError(f"invalid attribute namespace: {namespace!r}")
        self.namespace = namespace
        for const, rel in _EXTENSION_KEYS.items():
            setattr(self, const, f"{namespace}.{rel}")

    def all_attributes(self) -> dict[str, str]:
        """Return ``{CONSTANT_NAME: fully.qualified.name}`` for every extension attribute."""
        return {k: getattr(self, k) for k in _EXTENSION_KEYS}

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SemConv(namespace={self.namespace!r})"
