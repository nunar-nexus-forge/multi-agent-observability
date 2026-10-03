# Proposed OpenTelemetry GenAI semantic conventions for multi-agent coordination, memory and replay

*Status: draft proposal maintained in the `multi-agent-observability` repository. Feedback welcome via issues.*

## Motivation

The OpenTelemetry GenAI semantic conventions define spans for model calls (`chat`,
`text_completion`, `embeddings`), agent lifecycle (`create_agent`, `invoke_agent`,
`invoke_workflow`, `plan`) and tool execution (`execute_tool`), together with the
`gen_ai.agent.*`, `gen_ai.tool.*`, `gen_ai.provider.name` and `gen_ai.conversation.id`
attributes. They say nothing about three things every multi-agent system produces:

1. **inter-agent messages** - who delegated what to whom, with what confidence, and in
   reply to which message;
2. **memory operations** - reads, writes and evictions on episodic / semantic memory, with
   hit/miss and the *reason* an entry was evicted;
3. **reproducibility** - seeds and pre/post-state fingerprints that make an episode
   replayable, and the link between a replay and the run it reproduces.

Without them, a backend can show a tree of `invoke_agent` spans but cannot draw the
coordination graph, cannot explain why an agent retrieved stale information, and cannot
tell a replay from the original run. The attributes below are implemented by
[`multi-agent-observability`](../README.md) under the `ma_trace.*` namespace (or under `gen_ai.*` when
constructed with `namespace="gen_ai"`) and are proposed here for standardisation.

## New `gen_ai.operation.name` values

| Value | Span name | Kind | Description |
|---|---|---|---|
| `coordinate` | `coordinate {source}->{target}` | `PRODUCER` | one directed coordination event between two agents |
| `memory_access` | `memory_access {operation} {key}` | `INTERNAL` | one access to an agent memory system |
| `environment_contract` | `environment_contract {action}` | `INTERNAL` | one action executed under a verifiable pre/post-condition contract |

A `coordinate` span is zero-duration and carries the message metadata; when the target
agent's `invoke_agent` span is known, implementations SHOULD add a span link to it.

## Attributes

Requirement levels follow the OpenTelemetry vocabulary: **Required**, **Conditionally
Required** (with condition), **Recommended**, **Opt-In**.

### Episode (on `invoke_workflow` spans)

| Attribute | Type | Level | Description |
|---|---|---|---|
| `gen_ai.episode.id` | string | Recommended | stable identifier of the recorded episode |
| `gen_ai.episode.name` | string | Recommended | human-readable workflow name |
| `gen_ai.episode.seed` | int | Conditionally Required if the run was seeded | seed controlling all stochastic choices of the run |
| `gen_ai.episode.sampled` | boolean | Recommended | whether the episode was captured in full (detail sampling decision) |
| `gen_ai.episode.replay_of` | string | Conditionally Required if this run is a replay | `gen_ai.episode.id` of the run being reproduced |

### Coordination (on `coordinate` spans)

| Attribute | Type | Level | Description |
|---|---|---|---|
| `gen_ai.coordination.source` | string | Required | name of the sending agent (`gen_ai.agent.name` of the source) |
| `gen_ai.coordination.target` | string | Required | name of the receiving agent |
| `gen_ai.coordination.kind` | string (enum) | Required | `message`, `delegation`, `negotiation`, `consensus`, `response`, `broadcast`, `handoff` |
| `gen_ai.coordination.message` | string | Opt-In | the message content, truncated; may contain sensitive data |
| `gen_ai.coordination.confidence` | double | Recommended | sender's confidence in the message, `0.0`-`1.0` |
| `gen_ai.coordination.message_id` | string | Recommended | identifier of this message |
| `gen_ai.coordination.in_reply_to` | string | Conditionally Required if replying | `message_id` of the message being answered |

### Memory (on `memory_access` spans)

| Attribute | Type | Level | Description |
|---|---|---|---|
| `gen_ai.memory.agent` | string | Required | agent performing the access |
| `gen_ai.memory.operation` | string (enum) | Required | `read`, `write`, `evict`, `delete` |
| `gen_ai.memory.type` | string (enum) | Required | `episodic`, `semantic`, `working`, `procedural` |
| `gen_ai.memory.key` | string | Recommended | key or query identifying the entry |
| `gen_ai.memory.confidence` | double | Recommended | retrieval confidence / similarity score |
| `gen_ai.memory.hit` | boolean | Conditionally Required for reads | whether the read returned an entry |
| `gen_ai.memory.eviction_reason` | string (enum) | Conditionally Required for evictions | `capacity`, `policy`, `ttl`, `stale`, `manual`, `unknown` |

### Environment contracts (on `environment_contract` spans)

| Attribute | Type | Level | Description |
|---|---|---|---|
| `gen_ai.contract.action` | string | Required | the action name |
| `gen_ai.contract.pre_state_hash` | string | Required | fingerprint (SHA-256 of a canonical serialisation) of the state before the action |
| `gen_ai.contract.post_state_hash` | string | Recommended | fingerprint of the state after the action |
| `gen_ai.contract.seed` | int | Recommended | seed used for the action's stochastic choices |
| `gen_ai.contract.result` | string | Opt-In | outcome label |
| `gen_ai.contract.verified` | boolean | Conditionally Required under replay | whether the pre/post hashes matched the recording |

### Recorded calls (on `chat` / `execute_tool` spans)

| Attribute | Type | Level | Description |
|---|---|---|---|
| `gen_ai.call.index` | int | Recommended | position of the call within the episode |
| `gen_ai.call.input_fingerprint` | string | Recommended | fingerprint of the call inputs (never the inputs themselves) |
| `gen_ai.call.output_fingerprint` | string | Recommended | fingerprint of the call output |
| `gen_ai.call.replayed` | boolean | Conditionally Required under replay | the output was served from a recording instead of executing the call |

## Metrics

| Instrument | Type | Unit | Description |
|---|---|---|---|
| `gen_ai.coordination_overhead` | histogram | `1` | `(T_wall - Σ T_llm+tool) / T_wall` per episode |
| `gen_ai.redundant_action_rate` | histogram | `1` | duplicate or no-op actions / all actions per episode |
| `gen_ai.oscillation_rate` | histogram | `1` | repeated state transitions / all transitions per episode |
| `gen_ai.memory_hit_ratio` | histogram | `1` | successful retrievals / memory accesses per episode |
| `gen_ai.episodes` | counter | `1` | episodes completed |
| `gen_ai.redundant_actions` | counter | `1` | redundant actions observed |

Recommended attribute on all of them: `episode.name` (or `gen_ai.episode.name`).

## Example

```text
invoke_workflow checkout        gen_ai.operation.name=invoke_workflow  gen_ai.episode.id=ep-…  gen_ai.episode.seed=42
├─ invoke_agent planner         gen_ai.agent.name=planner
│  ├─ chat gpt-4o               gen_ai.call.index=0  gen_ai.call.input_fingerprint=3f9…
│  └─ coordinate planner->executor  gen_ai.coordination.kind=delegation  gen_ai.coordination.confidence=0.9
└─ invoke_agent executor
   ├─ memory_access read order:A1     gen_ai.memory.type=episodic  gen_ai.memory.hit=false
   ├─ environment_contract charge     gen_ai.contract.pre_state_hash=…  gen_ai.contract.seed=1839
   └─ execute_tool payments.charge    gen_ai.tool.name=payments.charge
```

## Privacy

`gen_ai.coordination.message` and `gen_ai.contract.result` may carry user data and are
Opt-In. Fingerprints are one-way hashes; they let a backend correlate identical inputs
across runs without storing the inputs.
