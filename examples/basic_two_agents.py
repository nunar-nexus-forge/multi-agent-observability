# SPDX-License-Identifier: Apache-2.0
"""A self-contained MA-Trace demo: planner / executor / critic with a fake LLM.

Run it to see deterministic replay in action::

    python examples/basic_two_agents.py    # records an episode, prints metrics + graph, replays it in-process

The CLI can replay the recorded episode as well. ``ma-trace replay`` imports the recorded
entrypoint (``basic_two_agents:run``) from the current working directory, and the store is
``./.ma-trace/episodes`` relative to where the demo ran, so run everything from this folder::

    cd examples
    python basic_two_agents.py
    ma-trace list
    ma-trace replay latest

(From the repository root use ``cd examples && ma-trace --store ../.ma-trace/episodes replay latest``.)

No API keys or network access are needed - the "LLM" is a seeded random generator.
"""

from __future__ import annotations

import os
import random
import sys
import time

import ma_trace as mt
from ma_trace import CoordinationKind, EvictionReason, MemoryType, TracedMemory

WORDS = ["fetch", "inspect", "clean", "heat", "place", "verify"]


def fake_llm(prompt: str, rng: random.Random) -> str:
    """Stands in for a model call: slow-ish, non-deterministic unless seeded."""
    time.sleep(0.005)
    return " ".join(rng.choice(WORDS) for _ in range(3)) + f" (re: {prompt[:20]})"


def run(task: str = "clean the mug and put it on the shelf", steps: int = 4) -> dict:
    ep = mt.current_episode()
    assert ep is not None and ep.rng is not None, "call run() inside mt.episode(...)"
    rng = ep.rng

    @mt.llm("planner", provider="fake", model="seeded-rng")
    def plan(prompt: str) -> str:
        return fake_llm(prompt, rng)

    @mt.tool("env_step")
    def env_step(action: str, state: dict) -> dict:
        time.sleep(0.002)
        new = dict(state)
        new["history"] = state["history"] + [action]
        new["pos"] = (state["pos"] + (1 if action.startswith("fetch") else -1)) % 3
        return new

    memory = TracedMemory({}, agent="executor", memory_type=MemoryType.EPISODIC, capacity=3)
    state = {"pos": 0, "history": []}

    with mt.agent_span("planner"):
        plan_text = plan(task)
        mt.coordination(
            "planner",
            "executor",
            plan_text,
            kind=CoordinationKind.DELEGATION,
            confidence=0.9,
            message_id="m1",
        )

    with mt.agent_span("executor"):
        actions = plan_text.split(" (")[0].split()
        for i in range(steps):
            action = actions[i % len(actions)]
            memory.get(f"step:{i}")  # miss: nothing stored for this step yet
            if i > 0:
                memory.get(f"step:{i - 1}", confidence=0.9)  # hit: the previous step's result
            with mt.contract("executor", action, state) as c:
                state = env_step(action, state)
                c.set_post_state(state)
                c.set_result("ok")
            memory.set(f"step:{i}", state["pos"], confidence=0.8)
            mt.action("executor", "env_step", {"action": action})  # repeated actions are redundant
            mt.state("executor", {"pos": state["pos"]})  # positions cycle -> oscillation
        memory.evict("step:0", EvictionReason.POLICY)
        mt.coordination(
            "executor",
            "critic",
            f"done after {steps} steps",
            kind=CoordinationKind.RESPONSE,
            in_reply_to="m1",
        )

    with mt.agent_span("critic"):
        verdict = plan("review: " + str(state["history"]))
        mt.coordination("critic", "planner", verdict, kind=CoordinationKind.NEGOTIATION, confidence=0.6)
    return {"state": state, "verdict": verdict}


def main() -> None:
    mt.configure("basic-two-agents", exporter=None)  # set exporter="console" to see the OTel spans
    with mt.episode(
        "basic-two-agents", seed=7, entrypoint="basic_two_agents:run", entrypoint_args={"steps": 4}
    ) as ep:
        result = run(steps=4)

    print(f"episode {ep.id} recorded ({len(ep.events)} events)\n")
    print(ep.metrics().format())
    print("\ncoordination graph (mermaid):")
    print(ep.coordination_graph().to_mermaid())
    print("\nmemory:", ep.memory_trace().summary())
    print("\nreplaying ...")
    report = mt.replay(ep.id, run)
    print(report.summary())
    assert report.result == result, "replay produced a different result"


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
