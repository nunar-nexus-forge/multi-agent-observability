# SPDX-License-Identifier: Apache-2.0
"""``ma-trace`` command line interface (distribution ``multi-agent-observability``)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

from ._version import __version__
from .episode import Episode
from .metrics import SLOThresholds, evaluate
from .store import EpisodeStore


def _fmt_time(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _parse_kv(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--arg expects key=value, got {item!r}")
        k, _, v = item.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def cmd_list(store: EpisodeStore, args: argparse.Namespace) -> int:
    rows = store.list(limit=args.limit)
    if args.json:
        print(json.dumps([r.__dict__ for r in rows], indent=2))
        return 0
    if not rows:
        print(f"no episodes in {store.root}")
        return 0
    print(f"{'EPISODE':<28} {'STARTED (UTC)':<20} {'DURATION':>9} {'EVENTS':>7} {'AGENTS':>6}  NAME")
    for r in rows:
        flag = "" if r.sampled else " (unsampled)"
        replay = f"  replay_of={r.replay_of}" if r.replay_of else ""
        print(
            f"{r.id:<28} {_fmt_time(r.started_at):<20} {r.duration_s:>8.3f}s "
            f"{r.events:>7} {r.agents:>6}  {r.name}{flag}{replay}"
        )
    return 0


def cmd_show(store: EpisodeStore, args: argparse.Namespace) -> int:
    ep = store.load(args.episode)
    if args.json:
        print(ep.to_json(indent=2))
        return 0
    print(ep.summary())
    print(f"  entrypoint: {ep.entrypoint or '-'}  args={json.dumps(ep.entrypoint_args)}")
    if ep.attributes:
        print(f"  attributes: {json.dumps(ep.attributes)}")
    print()
    print(ep.metrics().format())
    if args.events:
        print()
        t0 = ep.started_at
        for e in ep.events:
            d = e.to_dict()
            d.pop("seq", None)
            t = d.pop("t", 0.0)
            typ = d.pop("type")
            agent = d.pop("agent", None)
            fields = ", ".join(f"{k}={_short(v)}" for k, v in d.items() if _visible(k, v))
            print(f"  {t - t0:8.3f}s  #{e.seq:<4} {typ:<12} {agent or '-':<14} {fields}")
    return 0


def _short(v: Any) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= 60 else s[:57] + "..."


_HIDDEN_WHEN_FALSE = frozenset({"noop", "replayed"})


def _visible(key: str, value: Any) -> bool:
    """Hide empty values in the event timeline but keep meaningful ``False`` flags (``hit``,
    ``verified``, ``output_recorded``) and zero-valued numbers such as ``index=0``."""
    if value is None or value == "" or value == {} or value == []:
        return False
    return not (value is False and key in _HIDDEN_WHEN_FALSE)


def cmd_graph(store: EpisodeStore, args: argparse.Namespace) -> int:
    g = store.load(args.episode).coordination_graph()
    if args.format == "dot":
        print(g.to_dot())
    elif args.format == "json":
        print(g.to_json())
    else:
        print(g.to_mermaid())
    return 0


def cmd_metrics(store: EpisodeStore, args: argparse.Namespace) -> int:
    ep = store.load(args.episode)
    m = ep.metrics()
    thresholds = SLOThresholds(
        coordination_overhead_max=args.co_max,
        redundant_action_rate_max=args.rar_max,
        oscillation_rate_max=args.or_max,
        memory_hit_ratio_min=args.mhr_min,
    )
    violations = evaluate(m, thresholds)
    if args.json:
        print(json.dumps({"metrics": m.to_dict(), "violations": [v.__dict__ for v in violations]}, indent=2))
    else:
        print(m.format())
        if violations:
            print("\nSLO violations:")
            for v in violations:
                print(f"  ! {v}")
        else:
            print("\nall SLOs within thresholds")
    return 1 if violations and args.fail_on_violation else 0


def cmd_memory(store: EpisodeStore, args: argparse.Namespace) -> int:
    trace = store.load(args.episode).memory_trace()
    summary = trace.summary()
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    for k, v in summary.items():
        print(f"{k:<22} {v}")
    per_agent = trace.by_agent()
    if per_agent:
        print("\nper agent:")
        for agent, t in per_agent.items():
            s = t.summary()
            print(
                f"  {agent:<16} accesses={s['accesses']:<4} "
                f"hit_ratio={s['hit_ratio']:.3f} evictions={s['evictions']}"
            )
    return 0


def cmd_replay(store: EpisodeStore, args: argparse.Namespace) -> int:
    from .tracer import MATrace

    sys.path.insert(0, os.getcwd())
    tracer = MATrace(store=store)
    if args.exporter and args.exporter != "none":
        from .exporters import build_tracer_provider

        tracer = MATrace(
            store=store,
            tracer_provider=build_tracer_provider(
                "ma-trace-replay", args.exporter, simple=True, set_global=False
            ),
        )
    overrides = _parse_kv(args.arg)
    ep = store.load(args.episode)
    entry_args = dict(ep.entrypoint_args)
    entry_args.update(overrides)
    try:
        report = tracer.replay(ep, args.entrypoint, strict=args.strict, entrypoint_args=entry_args)
    except Exception as exc:
        print(f"replay failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())
    return 0 if report.deterministic else 1


def cmd_diff(store: EpisodeStore, args: argparse.Namespace) -> int:
    a = store.load(args.a)
    b = store.load(args.b)
    ma, mb = a.metrics().to_dict(), b.metrics().to_dict()
    print(f"{'metric':<32} {a.id:<28} {b.id:<28}")
    for key in (
        "coordination_overhead",
        "redundant_action_rate",
        "oscillation_rate",
        "memory_hit_ratio",
        "wall_time_s",
        "llm_calls",
        "tool_calls",
        "coordination_events",
        "agents",
    ):
        va, vb = ma[key], mb[key]
        mark = "" if va == vb else "  <-- differs"
        fa = f"{va:.3f}" if isinstance(va, float) else str(va)
        fb = f"{vb:.3f}" if isinstance(vb, float) else str(vb)
        print(f"{key:<32} {fa:<28} {fb:<28}{mark}")
    ea = {(s.source, s.target): s.count for s in a.coordination_graph().edge_summaries()}
    eb = {(s.source, s.target): s.count for s in b.coordination_graph().edge_summaries()}
    print("\ncoordination edges:")
    for edge in sorted(set(ea) | set(eb)):
        ca, cb = ea.get(edge, 0), eb.get(edge, 0)
        mark = "" if ca == cb else "  <-- differs"
        print(f"  {edge[0]} -> {edge[1]:<20} {ca:<6} {cb:<6}{mark}")
    print("\nfirst differing recorded call:")
    for ca_, cb_ in zip(a.calls, b.calls):
        if (ca_.agent, ca_.name, ca_.input_fingerprint, ca_.output_fingerprint) != (
            cb_.agent,
            cb_.name,
            cb_.input_fingerprint,
            cb_.output_fingerprint,
        ):
            print(
                f"  #{ca_.index} {ca_.agent}/{ca_.name}: "
                f"in {ca_.input_fingerprint[:12]}->{cb_.input_fingerprint[:12]}  "
                f"out {str(ca_.output_fingerprint)[:12]}->{str(cb_.output_fingerprint)[:12]}"
            )
            break
    else:
        if len(a.calls) != len(b.calls):
            print(f"  call counts differ: {len(a.calls)} vs {len(b.calls)}")
        else:
            print("  none")
    return 0


def cmd_export(store: EpisodeStore, args: argparse.Namespace) -> int:
    ep = store.load(args.episode)
    text = ep.to_json(indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_import(store: EpisodeStore, args: argparse.Namespace) -> int:
    with open(args.file, encoding="utf-8") as fh:
        ep = Episode.from_json(fh.read())
    path = store.save(ep)
    print(f"imported {ep.id} -> {path}")
    return 0


def cmd_delete(store: EpisodeStore, args: argparse.Namespace) -> int:
    episode_id = store.resolve(args.episode)
    if not args.yes:
        answer = input(f"delete {episode_id}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted")
            return 1
    store.delete(episode_id)
    print(f"deleted {episode_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ma-trace", description="Inspect, analyse and replay MA-Trace episodes.")
    p.add_argument(
        "--store", help="episode store directory (default: $MA_TRACE_STORE or ./.ma-trace/episodes)"
    )
    p.add_argument("--version", action="version", version=f"ma-trace {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list", help="list recorded episodes")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="show an episode's summary, metrics and (optionally) events")
    s.add_argument("episode", help="episode id, unique prefix, or 'latest'")
    s.add_argument("--events", action="store_true", help="print the event timeline")
    s.add_argument("--json", action="store_true", help="dump the full episode as JSON")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("graph", help="print the coordination graph")
    s.add_argument("episode")
    s.add_argument("--format", choices=["mermaid", "dot", "json"], default="mermaid")
    s.set_defaults(func=cmd_graph)

    s = sub.add_parser("metrics", help="compute the SLO metrics and check thresholds")
    s.add_argument("episode")
    s.add_argument("--json", action="store_true")
    s.add_argument("--co-max", type=float, default=SLOThresholds.coordination_overhead_max)
    s.add_argument("--rar-max", type=float, default=SLOThresholds.redundant_action_rate_max)
    s.add_argument("--or-max", type=float, default=SLOThresholds.oscillation_rate_max)
    s.add_argument("--mhr-min", type=float, default=None)
    s.add_argument("--fail-on-violation", action="store_true", help="exit 1 if any SLO is violated")
    s.set_defaults(func=cmd_metrics)

    s = sub.add_parser("memory", help="summarise memory operations")
    s.add_argument("episode")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_memory)

    s = sub.add_parser("replay", help="deterministically re-execute an episode")
    s.add_argument("episode")
    s.add_argument(
        "--entrypoint", help="'package.module:function' (default: the episode's recorded entrypoint)"
    )
    s.add_argument(
        "--arg", action="append", help="override an entrypoint argument: key=value (JSON values accepted)"
    )
    s.add_argument("--strict", action="store_true", help="stop at the first divergence")
    s.add_argument(
        "--exporter", choices=["none", "console"], default="none", help="also export the replay's spans"
    )
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("diff", help="compare two episodes")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(func=cmd_diff)

    s = sub.add_parser("export", help="write an episode as a JSON file")
    s.add_argument("episode")
    s.add_argument("-o", "--out")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("import", help="import an exported episode JSON file into the store")
    s.add_argument("file")
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("delete", help="delete an episode")
    s.add_argument("episode")
    s.add_argument("--yes", "-y", action="store_true")
    s.set_defaults(func=cmd_delete)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = EpisodeStore(args.store)
    try:
        return int(args.func(store, args))
    except KeyError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
