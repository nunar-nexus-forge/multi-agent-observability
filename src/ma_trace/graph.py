# SPDX-License-Identifier: Apache-2.0
"""Coordination graphs: agents as nodes, coordination events as directed edges."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .events import CoordinationEvent, CoordinationKind


@dataclass(frozen=True)
class EdgeSummary:
    source: str
    target: str
    count: int
    kinds: tuple[str, ...]
    mean_confidence: float | None
    first_t: float
    last_t: float


class CoordinationGraph:
    """A directed multigraph of agent-to-agent coordination events.

    Nodes are agent names; every :class:`~ma_trace.events.CoordinationEvent` is one
    directed edge from ``source`` to ``target`` carrying the message, its kind, a
    confidence and a timestamp.
    """

    def __init__(self) -> None:
        self._agents: dict[str, None] = {}
        self._edges: list[CoordinationEvent] = []
        self._adj: dict[str, dict[str, list[CoordinationEvent]]] = defaultdict(lambda: defaultdict(list))

    @classmethod
    def from_events(cls, events: Iterable[CoordinationEvent]) -> CoordinationGraph:
        g = cls()
        for e in events:
            g.add(e)
        return g

    # -- mutation -----------------------------------------------------------

    def add_agent(self, name: str) -> None:
        self._agents.setdefault(name, None)

    def add(self, event: CoordinationEvent) -> None:
        self.add_agent(event.source)
        self.add_agent(event.target)
        self._edges.append(event)
        self._adj[event.source][event.target].append(event)

    # -- queries ------------------------------------------------------------

    @property
    def agents(self) -> list[str]:
        return list(self._agents)

    @property
    def edges(self) -> list[CoordinationEvent]:
        return list(self._edges)

    def successors(self, agent: str) -> list[str]:
        return list(self._adj.get(agent, {}))

    def predecessors(self, agent: str) -> list[str]:
        return [src for src, targets in self._adj.items() if agent in targets]

    def out_degree(self, agent: str) -> int:
        return sum(len(v) for v in self._adj.get(agent, {}).values())

    def in_degree(self, agent: str) -> int:
        return sum(len(targets[agent]) for targets in self._adj.values() if agent in targets)

    def edge_count(self, source: str, target: str) -> int:
        return len(self._adj.get(source, {}).get(target, []))

    def edge_summaries(self) -> list[EdgeSummary]:
        out: list[EdgeSummary] = []
        for src, targets in self._adj.items():
            for tgt, evs in targets.items():
                confs = [e.confidence for e in evs if e.confidence is not None]
                kinds = tuple(sorted({e.kind.value for e in evs}))
                out.append(
                    EdgeSummary(
                        source=src,
                        target=tgt,
                        count=len(evs),
                        kinds=kinds,
                        mean_confidence=(sum(confs) / len(confs)) if confs else None,
                        first_t=min(e.t for e in evs),
                        last_t=max(e.t for e in evs),
                    )
                )
        out.sort(key=lambda s: (s.first_t, s.source, s.target))
        return out

    def bottlenecks(self, top: int = 3) -> list[tuple[str, int]]:
        """Agents ranked by inbound message volume (the busiest receivers)."""
        ranked = sorted(((a, self.in_degree(a)) for a in self._agents), key=lambda x: (-x[1], x[0]))
        return [(a, d) for a, d in ranked[:top] if d > 0]

    def cycles(self) -> list[list[str]]:
        """Simple cycles in the aggregated digraph (each returned once, rotated to start at min)."""
        found: set[tuple[str, ...]] = set()
        result: list[list[str]] = []
        nodes = list(self._agents)

        def dfs(start: str, node: str, path: list[str], visited: set[str]) -> None:
            for nxt in self.successors(node):
                if nxt == start and len(path) >= 1:
                    cyc = path[:]
                    # rotate so the lexicographically smallest node comes first
                    i = cyc.index(min(cyc))
                    key = tuple(cyc[i:] + cyc[:i])
                    if key not in found:
                        found.add(key)
                        result.append(list(key))
                elif nxt not in visited and nxt > start:
                    visited.add(nxt)
                    dfs(start, nxt, path + [nxt], visited)
                    visited.discard(nxt)

        for n in nodes:
            dfs(n, n, [n], {n})
        return result

    @property
    def is_dag(self) -> bool:
        return not self.cycles()

    def delegation_depth(self) -> dict[str, int]:
        """Depth of every agent in the delegation tree (roots have depth 0)."""
        deleg: dict[str, set[str]] = defaultdict(set)
        parents: dict[str, set[str]] = defaultdict(set)
        for e in self._edges:
            if e.kind in (CoordinationKind.DELEGATION, CoordinationKind.HANDOFF):
                deleg[e.source].add(e.target)
                parents[e.target].add(e.source)
        depth: dict[str, int] = {}
        roots = [a for a in self._agents if not parents.get(a)]
        frontier = [(r, 0) for r in roots]
        while frontier:
            node, d = frontier.pop(0)
            if node in depth and depth[node] <= d:
                continue
            depth[node] = d
            for child in deleg.get(node, ()):
                if depth.get(child, 10**9) > d + 1:
                    frontier.append((child, d + 1))
        return depth

    def timeline(self) -> list[tuple[float, str, str, str, str]]:
        return [
            (e.t, e.source, e.target, e.kind.value, e.message)
            for e in sorted(self._edges, key=lambda e: e.seq)
        ]

    # -- export -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": self.agents,
            "edges": [
                {
                    "source": s.source,
                    "target": s.target,
                    "count": s.count,
                    "kinds": list(s.kinds),
                    "mean_confidence": s.mean_confidence,
                }
                for s in self.edge_summaries()
            ],
            "bottlenecks": self.bottlenecks(),
            "cycles": self.cycles(),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_dot(self, name: str = "coordination") -> str:
        lines = [f"digraph {json.dumps(name)} {{", "  rankdir=LR;", "  node [shape=box, style=rounded];"]
        for a in self._agents:
            lines.append(f"  {json.dumps(a)};")
        for s in self.edge_summaries():
            label = f"{'/'.join(s.kinds)} x{s.count}"
            if s.mean_confidence is not None:
                label += f" c={s.mean_confidence:.2f}"
            lines.append(f"  {json.dumps(s.source)} -> {json.dumps(s.target)} [label={json.dumps(label)}];")
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        ids = {a: f"a{i}" for i, a in enumerate(self._agents)}
        lines = ["flowchart LR"]
        for a, i in ids.items():
            lines.append(f'  {i}["{a}"]')
        for s in self.edge_summaries():
            label = f"{'/'.join(s.kinds)} x{s.count}"
            lines.append(f"  {ids[s.source]} -->|{label}| {ids[s.target]}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._edges)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"CoordinationGraph(agents={len(self._agents)}, edges={len(self._edges)})"
