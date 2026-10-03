from __future__ import annotations

import json
import textwrap

import pytest

import ma_trace as mt
from ma_trace.cli import main


@pytest.fixture
def store_dir(tmp_path):
    return tmp_path / "episodes"


@pytest.fixture
def recorded(store_dir, monkeypatch):
    """Record an episode from a workflow module living in a temporary cwd."""
    monkeypatch.chdir(store_dir.parent)
    (store_dir.parent / "wf.py").write_text(
        textwrap.dedent(
            """
            import ma_trace as mt

            def run(n=2):
                @mt.llm("planner", provider="fake", model="m")
                def ask(p):
                    return p + "!"

                out = []
                with mt.agent_span("planner"):
                    out.append(ask("go"))
                    mt.coordination("planner", "worker", "task", kind="delegation")
                for i in range(n):
                    mt.action("worker", "act", {"i": i})
                mt.action("worker", "act", {"i": 0})
                mt.memory("worker", "read", "episodic", "k", hit=True)
                mt.memory("worker", "read", "episodic", "missing", hit=False)
                return out
            """
        )
    )
    t = mt.MATrace(store=str(store_dir), otel_metrics=False)
    mt.set_tracer(t)
    import importlib
    import sys

    sys.path.insert(0, str(store_dir.parent))
    wf = importlib.import_module("wf")
    with t.episode("cli-demo", seed=1, entrypoint="wf:run", entrypoint_args={"n": 2}) as ep:
        wf.run(2)
    with t.episode("cli-demo-2", seed=1, entrypoint="wf:run", entrypoint_args={"n": 3}) as ep2:
        wf.run(3)
    mt.set_tracer(None)
    yield ep, ep2
    sys.modules.pop("wf", None)
    sys.path.remove(str(store_dir.parent))


def test_list_show_graph_metrics_memory(recorded, store_dir, capsys):
    ep, ep2 = recorded
    assert main(["--store", str(store_dir), "list"]) == 0
    out = capsys.readouterr().out
    assert ep.id in out and "cli-demo" in out
    assert main(["--store", str(store_dir), "list", "--json", "--limit", "1"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["id"] == ep2.id

    assert main(["--store", str(store_dir), "show", ep.id, "--events"]) == 0
    out = capsys.readouterr().out
    assert "coordination overhead" in out and "entrypoint: wf:run" in out and "coordination" in out
    assert "hit=true" in out and "hit=false" in out and "index=0" in out  # misses and index 0 are shown
    assert main(["--store", str(store_dir), "show", "latest", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == ep2.id

    for fmt, marker in (("mermaid", "flowchart LR"), ("dot", "digraph"), ("json", '"agents"')):
        assert main(["--store", str(store_dir), "graph", ep.id, "--format", fmt]) == 0
        assert marker in capsys.readouterr().out

    assert main(["--store", str(store_dir), "metrics", ep.id]) == 0
    assert "redundant action rate" in capsys.readouterr().out
    assert main(["--store", str(store_dir), "metrics", ep.id, "--rar-max", "0.1", "--fail-on-violation"]) == 1
    assert "SLO violations" in capsys.readouterr().out
    assert main(["--store", str(store_dir), "metrics", ep.id, "--json"]) == 0
    assert "metrics" in json.loads(capsys.readouterr().out)

    assert main(["--store", str(store_dir), "memory", ep.id]) == 0
    assert "hit_ratio" in capsys.readouterr().out
    assert main(["--store", str(store_dir), "memory", ep.id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["reads"] == 2


def test_replay_diff_export_import_delete(recorded, store_dir, tmp_path, capsys):
    ep, ep2 = recorded
    assert main(["--store", str(store_dir), "replay", ep.id]) == 0
    assert "DETERMINISTIC" in capsys.readouterr().out
    assert main(["--store", str(store_dir), "replay", ep.id, "--arg", "n=3", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["deterministic"] is False
    assert main(["--store", str(store_dir), "replay", ep.id, "--entrypoint", "wf:missing"]) == 2

    assert main(["--store", str(store_dir), "diff", ep.id, ep2.id]) == 0
    out = capsys.readouterr().out
    assert "<-- differs" in out and "coordination edges" in out

    target = tmp_path / "exported.json"
    assert main(["--store", str(store_dir), "export", ep.id, "-o", str(target)]) == 0
    assert json.loads(target.read_text())["id"] == ep.id
    other = tmp_path / "other-store"
    assert main(["--store", str(other), "import", str(target)]) == 0
    assert main(["--store", str(other), "list"]) == 0
    assert ep.id in capsys.readouterr().out

    assert main(["--store", str(store_dir), "delete", ep.id, "--yes"]) == 0
    assert main(["--store", str(store_dir), "show", ep.id]) == 2
    assert "error" in capsys.readouterr().err


def test_empty_store(tmp_path, capsys):
    assert main(["--store", str(tmp_path), "list"]) == 0
    assert "no episodes" in capsys.readouterr().out
