from __future__ import annotations

from ma_trace.episode import Episode
from ma_trace.events import (
    ActionEvent,
    CallEvent,
    CallType,
    ContractEvent,
    MemoryEvent,
    MemoryOp,
    MemoryType,
    StateEvent,
)
from ma_trace.metrics import SLOThresholds, compute_metrics, evaluate, find_redundant_actions


def make_episode():
    ep = Episode(id="ep-test", name="t", started_at=100.0, ended_at=110.0)
    seq = iter(range(1000))

    def add(e):
        e.seq = next(seq)
        ep.events.append(e)

    add(CallEvent(agent="a", call_type=CallType.LLM, name="chat", duration_s=2.0))
    add(CallEvent(agent="b", call_type=CallType.TOOL, name="search", duration_s=1.0))
    add(ActionEvent(agent="a", name="search", args_fingerprint="q1"))
    add(ActionEvent(agent="a", name="search", args_fingerprint="q1"))  # duplicate
    add(ActionEvent(agent="a", name="noop", args_fingerprint="x", noop=True))  # explicit no-op
    add(
        ActionEvent(agent="b", name="move", args_fingerprint="m", state_before="s1", state_after="s1")
    )  # implicit no-op
    add(ActionEvent(agent="b", name="move", args_fingerprint="m2", state_before="s1", state_after="s2"))
    # oscillation for agent c: s1->s2->s1->s2 : transitions (s1,s2),(s2,s1),(s1,s2) -> 1 repeated of 3
    for h in ("s1", "s2", "s1", "s2"):
        add(StateEvent(agent="c", state_hash=h))
    add(ContractEvent(agent="d", action="act", pre_hash="p", post_hash="q"))
    add(MemoryEvent(agent="a", op=MemoryOp.READ, memory_type=MemoryType.EPISODIC, key="k", hit=True))
    add(MemoryEvent(agent="a", op=MemoryOp.READ, memory_type=MemoryType.EPISODIC, key="k", hit=False))
    add(MemoryEvent(agent="a", op=MemoryOp.WRITE, memory_type=MemoryType.EPISODIC, key="k"))
    add(MemoryEvent(agent="a", op=MemoryOp.EVICT, memory_type=MemoryType.EPISODIC, key="k"))
    return ep


def test_compute_metrics():
    ep = make_episode()
    m = compute_metrics(ep)
    assert m.wall_time_s == 10.0
    assert m.llm_time_s == 2.0 and m.tool_time_s == 1.0
    assert abs(m.coordination_overhead - 0.7) < 1e-9
    assert m.total_actions == 5 and m.redundant_actions == 3
    assert abs(m.redundant_action_rate - 0.6) < 1e-9
    assert abs(m.redundant_actions_per_second - 0.3) < 1e-9
    # transitions: agent b s1->s2 (1), agent c 3 transitions (1 repeated), agent d p->q (1)
    assert m.total_transitions == 5 and m.repeated_transitions == 1
    assert abs(m.oscillation_rate - 0.2) < 1e-9
    assert m.memory_accesses == 3 and m.memory_hits == 1
    assert abs(m.memory_hit_ratio - 1 / 3) < 1e-9
    assert m.llm_calls == 1 and m.tool_calls == 1
    assert "coordination overhead" in m.format()


def test_find_redundant_actions_marks_only_later_duplicates():
    a1 = ActionEvent(seq=1, agent="a", name="x", args_fingerprint="1")
    a2 = ActionEvent(seq=2, agent="a", name="x", args_fingerprint="1")
    a3 = ActionEvent(seq=3, agent="b", name="x", args_fingerprint="1")  # different agent -> not redundant
    assert find_redundant_actions([a3, a2, a1]) == [a2]


def test_negative_raw_overhead_is_clamped():
    ep = Episode(id="e", name="n", started_at=0.0, ended_at=1.0)
    ep.events.append(CallEvent(agent="a", call_type=CallType.LLM, name="chat", duration_s=1.5))
    m = compute_metrics(ep)
    assert m.coordination_overhead == 0.0 and m.coordination_overhead_raw < 0


def test_evaluate_thresholds():
    m = compute_metrics(make_episode())
    violations = evaluate(m, SLOThresholds())
    names = {v.metric for v in violations}
    assert names == {"coordination_overhead", "redundant_action_rate"}
    assert (
        evaluate(
            m,
            SLOThresholds(
                coordination_overhead_max=1,
                redundant_action_rate_max=1,
                oscillation_rate_max=1,
                memory_hit_ratio_min=0.9,
            ),
        )[0].metric
        == "memory_hit_ratio"
    )
    assert str(violations[0]).startswith("coordination_overhead=0.700 > 0.420")
