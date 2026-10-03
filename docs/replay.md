# Deterministic replay

Replay re-executes a workflow while serving every recorded non-deterministic call from the
episode and verifying every environment contract. It answers "does my code, with the same
seed and the same model outputs, do the same thing?" and, when it does not, "where exactly
does it diverge?".

## What gets recorded

| Source | Recorded as | Used during replay |
|---|---|---|
| `@mt.llm(...)` / `mt.llm_call(...)` | `CallEvent(call_type="llm")` with input fingerprint, output (JSON), duration | output is returned instead of executing the call |
| `@mt.tool(...)` / `mt.tool_call(...)` | `CallEvent(call_type="tool")` | same |
| `mt.contract(agent, action, pre_state)` | `ContractEvent(pre_hash, post_hash, seed, result)` | pre/post hashes are compared, seed is re-derived from the episode RNG |
| `mt.action(...)` | `ActionEvent` | the action sequence must match |
| `mt.coordination(...)` | `CoordinationEvent` | the edge sequence (source, target, kind) must match |
| `mt.episode(..., seed=N)` | `seed`, `entrypoint`, `entrypoint_args` | `random.seed(N)`; `episode.rng` is re-created; entrypoint is re-run with the same args |

## Matching rules

* Calls are matched **FIFO per (agent, call type, call name)**. This tolerates changes in
  unrelated parts of the workflow: adding a new tool to one agent does not shift the other
  agents' calls.
* A matched call whose **input fingerprint differs** yields an `input_mismatch` divergence;
  the recorded output is still returned (so the rest of the run stays comparable) unless
  `strict=True`.
* Fingerprints are SHA-256 hashes of a canonical JSON encoding of the call arguments.
  Objects that are not JSON-serialisable are encoded with `repr()` with memory addresses
  stripped. To control what is fingerprinted, pass `inputs=` (a function of the call
  arguments) to the decorator - for example to exclude clients, callbacks or timestamps.

## Divergence kinds

| Kind | Meaning |
|---|---|
| `input_mismatch` | the call happened, but with different inputs |
| `unexpected_call` | no recorded call left for this agent/name (the code made an extra call; it is executed live) |
| `missing_call` | a recorded call was never made |
| `output_unavailable` | the recorded output was not JSON-serialisable, so the call is executed live |
| `contract_pre_mismatch` / `contract_post_mismatch` | the state before / after an action differs from the recording |
| `unexpected_contract` / `missing_contract` | contract opened that was not recorded / recorded contract never opened |
| `action_sequence_mismatch` | the sequence of `mt.action` calls differs |
| `coordination_mismatch` | the sequence of coordination edges differs |
| `error` | the entrypoint raised |

A report with no divergences is **deterministic**.

## Usage

```python
report = mt.replay("latest", entrypoint="myapp:run")  # or a callable
report = mt.replay(episode, run, entrypoint_args={"order_id": "A1"}, strict=True)
print(report.summary())
report.to_dict()
```

```bash
ma-trace replay latest                              # uses the recorded entrypoint + args
ma-trace replay ep-2026… --entrypoint myapp:run --arg order_id=A2 --strict --json
```

The CLI adds the current working directory to `sys.path` so that `myapp:run` resolves.

## Rich outputs

Recorded outputs are JSON values. If your wrapped function returns a rich object
(a pydantic model, a LangChain message), give the decorator a codec:

```python
@mt.llm(
    "planner",
    provider="openai",
    model="gpt-4o",
    encode=lambda r: r.model_dump(),
    decode=lambda d: Reply.model_validate(d),
)
def ask(prompt): ...
```

Without a codec the output is stored via `model_dump()` / `to_dict()` / `dataclasses.asdict`
when available, and replayed as that JSON value.

## Limitations

* **Timing metrics differ under replay.** Calls served from the recording take microseconds,
  so the replayed coordination overhead is much higher than the recorded one. Use replay to
  verify structure and outputs; compare timing between *recorded* episodes.
* **Concurrency.** Matching is per agent/name queue, so two agents running in parallel
  replay fine as long as each agent's own call order is stable. Calls that race for the
  same agent/name may be served in a different order.
* **Side effects.** Replay does not execute recorded tool calls, so external side effects
  (payments, emails) do not happen again - which is the point. Unrecorded side effects in
  plain code do.
* **Unsampled episodes** store no outputs and cannot be replayed (`record=True` forces
  full capture).
