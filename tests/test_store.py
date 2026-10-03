from __future__ import annotations

import os
from pathlib import Path

import pytest

import ma_trace as mt
from ma_trace.store import EpisodeStore, InMemoryEpisodeStore


def _episode(name, started):
    return mt.Episode(id=f"ep-2026-{name}", name=name, started_at=started, ended_at=started + 1)


@pytest.mark.parametrize("factory", [lambda p: EpisodeStore(p), lambda p: InMemoryEpisodeStore()])
def test_store_roundtrip(tmp_path, factory):
    store = factory(tmp_path)
    assert len(store) == 0 and store.latest() is None
    a = _episode("aaa", 10)
    b = _episode("bbb", 20)
    store.save(a)
    store.save(b)
    assert len(store) == 2 and "ep-2026-aaa" in store
    assert store.load("ep-2026-aaa").name == "aaa"
    assert store.load("ep-2026-b").name == "bbb"  # prefix
    assert store.resolve("latest") == b.id
    assert [s.id for s in store.list()] == [b.id, a.id]
    assert store.list(limit=1)[0].name == "bbb"
    with pytest.raises(KeyError):
        store.load("ep-2026-")  # ambiguous
    with pytest.raises(KeyError):
        store.load("nope")
    store.delete("ep-2026-aaa")
    assert len(store) == 1
    store.delete("latest")
    assert store.latest() is None


def test_env_and_default_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MA_TRACE_STORE", str(tmp_path / "custom"))
    assert EpisodeStore().root == tmp_path / "custom"
    monkeypatch.delenv("MA_TRACE_STORE")
    assert EpisodeStore().root == Path(".ma-trace/episodes")


def test_latest_prefers_start_time_over_file_time(tmp_path):
    store = EpisodeStore(tmp_path)
    store.save(_episode("bbb", 20))
    store.save(_episode("aaa", 10))
    same = 1_700_000_000
    for episode_id in store.ids():  # identical file times, as on coarse-timestamp file systems
        os.utime(store._path(episode_id), (same, same))
    assert store.resolve("latest") == "ep-2026-bbb"
    memory = InMemoryEpisodeStore()
    memory.save(_episode("ccc", 30))
    memory.save(_episode("ddd", 30))  # equal start times: the one saved last wins
    assert memory.resolve("latest") == "ep-2026-ddd"


def test_summary_fields(tmp_path):
    store = EpisodeStore(tmp_path)
    ep = _episode("x", 1)
    ep.seed = 4
    store.save(ep)
    s = store.list()[0]
    assert s.seed == 4 and s.duration_s == 1 and s.events == 0 and s.sampled
