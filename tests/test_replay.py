from __future__ import annotations

import random

import pytest

import ma_trace as mt
from ma_trace.replay import ReplayDivergenceError, ReplayError, resolve_entrypoint

LLM_CALLS = {"n": 0}


def fake_llm(prompt: str, rng: random.Random) -> str:
    LLM_CALLS["n"] += 1
    return f"{prompt}:{rng.randint(0, 10**6)}"


def workflow(steps: int = 3, prompt: str = "plan") -> list[str]:
    """A toy planner/executor workflow with seeded randomness and recorded calls."""
    ep = mt.current_episode()
    assert ep is not None and ep.rng is not None
    rng = ep.rng
    outputs = []

    @mt.llm("planner", provider="fake", model="fake-1")
    def ask(p: str) -> str:
        return fake_llm(p, rng)

    @mt.tool("act")
    def act(step: int) -> dict:
        return {"step": step, "noise": rng.random()}

    state = {"pos": 0}
    with mt.agent_span("planner"):
        plan = ask(prompt)
        outputs.append(plan)
        mt.coordination("planner", "executor", plan, kind="delegation", confidence=0.8)
    with mt.agent_span("executor"):
        for i in range(steps):
            with mt.contract("executor", "step", state) as c:
                r = act(i)
                state = {"pos": state["pos"] + 1, "noise": r["noise"]}
                c.set_post_state(state)
            mt.action("executor", "act", {"i": i})
    return outputs


def test_record_then_deterministic_replay(tracer):
    LLM_CALLS["n"] = 0
    with mt.episode("wf", seed=11, entrypoint=workflow, entrypoint_args={"steps": 2}) as ep:
        out = workflow(steps=2)
    assert LLM_CALLS["n"] == 1
    assert ep.entrypoint.endswith(":workflow") and ep.entrypoint_args == {"steps": 2}

    report = mt.replay(ep.id, workflow)
    assert report.deterministic, report.summary()
    assert LLM_CALLS["n"] == 1  # the LLM was NOT executed again
    assert report.matched_calls == 3 and report.executed_live == 0
    assert report.contracts_checked == 2 and report.contracts_passed == 2
    assert report.result == out
    assert report.replay_episode_id != ep.id
    replayed = tracer.store.load(report.replay_episode_id)
    assert replayed.replay_of == ep.id and replayed.name == "replay:wf"
    assert all(c.replayed for c in replayed.calls)
    assert "DETERMINISTIC" in report.summary()
    assert report.to_dict()["deterministic"] is True


def test_replay_detects_input_divergence(tracer):
    with mt.episode("wf", seed=11, entrypoint_args={"steps": 1}) as ep:
        workflow(steps=1)
    report = mt.replay(ep, workflow, entrypoint_args={"steps": 1, "prompt": "different"})
    kinds = [d.kind for d in report.divergences]
    assert not report.deterministic and kinds == ["input_mismatch"]
    assert report.divergences[0].agent == "planner" and report.divergences[0].describe().startswith(
        "input_mismatch"
    )


def test_replay_reports_missing_and_unexpected_calls(tracer):
    with mt.episode("wf", seed=11) as ep:
        workflow(steps=3)
    report = mt.replay(ep, workflow, entrypoint_args={"steps": 1})
    kinds = sorted(d.kind for d in report.divergences)
    assert kinds == [
        "action_sequence_mismatch",
        "missing_call",
        "missing_call",
        "missing_contract",
        "missing_contract",
    ]
    report2 = mt.replay(ep, workflow, entrypoint_args={"steps": 5})
    kinds2 = sorted({d.kind for d in report2.divergences})
    assert "unexpected_call" in kinds2 and "unexpected_contract" in kinds2
    assert report2.executed_live == 2


def test_strict_replay_raises(tracer):
    with mt.episode("wf", seed=11) as ep:
        workflow(steps=1)
    with pytest.raises(ReplayDivergenceError):
        mt.replay(ep, workflow, strict=True, entrypoint_args={"steps": 1, "prompt": "x"})


def test_replay_captures_entrypoint_errors(tracer):
    def broken():
        raise RuntimeError("boom")

    with mt.episode("b", seed=1) as ep:
        pass
    report = mt.replay(ep, broken)
    assert (
        not report.deterministic
        and report.error == "RuntimeError: boom"
        and report.divergences[0].kind == "error"
    )


def test_contract_post_mismatch_detected(tracer):
    def env_run(bump: int = 1):
        state = {"pos": 0}
        with mt.contract("env", "step", state) as c:
            state = {"pos": state["pos"] + bump}
            c.set_post_state(state)

    with mt.episode("env", seed=1, entrypoint_args={"bump": 1}) as ep:
        env_run()
    ok = mt.replay(ep, env_run)
    assert ok.deterministic
    bad = mt.replay(ep, env_run, entrypoint_args={"bump": 2})
    assert [d.kind for d in bad.divergences] == ["contract_post_mismatch"]
    assert bad.contracts_checked == 1 and bad.contracts_passed == 0


def test_replay_seeds_random_module(tracer):
    def rnd():
        return random.random()

    with mt.episode("r", seed=99) as ep:
        first = rnd()
    report = mt.replay(ep, rnd)
    assert report.result == first


def test_resolve_entrypoint_errors():
    assert resolve_entrypoint("json:dumps")({"a": 1}) == '{"a": 1}'
    with pytest.raises(ReplayError):
        resolve_entrypoint(None)
    with pytest.raises(ReplayError):
        resolve_entrypoint("nonexistent_module_xyz:fn")
    with pytest.raises(ReplayError):
        resolve_entrypoint("json:nope")
    with pytest.raises(ReplayError):
        resolve_entrypoint("json")


def test_replay_unknown_episode(tracer):
    with pytest.raises(ReplayError):
        mt.replay("ep-missing", workflow)


def test_async_entrypoint(tracer):
    async def run(n: int = 2):
        @mt.llm("a", provider="p", model="m")
        async def ask(x):
            return x * 2

        return [await ask(i) for i in range(n)]

    with mt.episode("async", seed=1, entrypoint_args={"n": 2}) as ep:
        import asyncio

        asyncio.run(run(2))
    report = mt.replay(ep, run)
    assert report.deterministic and report.result == [0, 2] and report.matched_calls == 2
