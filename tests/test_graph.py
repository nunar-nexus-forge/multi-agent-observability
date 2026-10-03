from __future__ import annotations

from ma_trace.events import CoordinationEvent, CoordinationKind
from ma_trace.graph import CoordinationGraph


def ev(src, tgt, kind=CoordinationKind.MESSAGE, seq=0, conf=None):
    return CoordinationEvent(
        seq=seq, t=float(seq), agent=src, source=src, target=tgt, kind=kind, confidence=conf
    )


def test_degrees_and_bottlenecks():
    g = CoordinationGraph.from_events(
        [ev("a", "b", seq=1), ev("a", "b", seq=2), ev("c", "b", seq=3), ev("b", "a", seq=4)]
    )
    assert g.agents == ["a", "b", "c"]
    assert g.in_degree("b") == 3 and g.out_degree("a") == 2 and g.edge_count("a", "b") == 2
    assert g.bottlenecks(1) == [("b", 3)]
    assert g.successors("a") == ["b"] and set(g.predecessors("b")) == {"a", "c"}


def test_cycles_and_dag():
    g = CoordinationGraph.from_events([ev("a", "b"), ev("b", "c")])
    assert g.is_dag and g.cycles() == []
    g.add(ev("c", "a"))
    assert not g.is_dag and g.cycles() == [["a", "b", "c"]]


def test_delegation_depth():
    g = CoordinationGraph.from_events(
        [
            ev("manager", "planner", CoordinationKind.DELEGATION),
            ev("planner", "worker", CoordinationKind.DELEGATION),
            ev("worker", "planner", CoordinationKind.RESPONSE),
        ]
    )
    assert g.delegation_depth() == {"manager": 0, "planner": 1, "worker": 2}


def test_exports():
    g = CoordinationGraph.from_events(
        [ev("a", "b", CoordinationKind.DELEGATION, conf=0.9), ev("b", "a", CoordinationKind.RESPONSE)]
    )
    dot = g.to_dot()
    assert "digraph" in dot and '"a" -> "b"' in dot and "c=0.90" in dot
    mermaid = g.to_mermaid()
    assert mermaid.startswith("flowchart LR") and "-->|delegation x1|" in mermaid
    d = g.to_dict()
    assert d["agents"] == ["a", "b"] and d["edges"][0]["count"] == 1
    summaries = g.edge_summaries()
    assert summaries[0].mean_confidence == 0.9 and summaries[1].mean_confidence is None
