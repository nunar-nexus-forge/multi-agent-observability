# SPDX-License-Identifier: Apache-2.0
"""Episode persistence: one JSON document per episode."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .episode import Episode

DEFAULT_STORE_DIR = ".ma-trace/episodes"
ENV_STORE = "MA_TRACE_STORE"


@dataclass
class EpisodeSummary:
    id: str
    name: str
    started_at: float
    duration_s: float
    events: int
    agents: int
    sampled: bool
    seed: int | None
    replay_of: str | None

    @classmethod
    def of(cls, ep: Episode) -> EpisodeSummary:
        return cls(
            id=ep.id,
            name=ep.name,
            started_at=ep.started_at,
            duration_s=ep.duration_s,
            events=len(ep.events),
            agents=len(ep.agents),
            sampled=ep.sampled,
            seed=ep.seed,
            replay_of=ep.replay_of,
        )


class EpisodeStore:
    """File-backed store: ``<root>/<episode-id>.json``.

    ``root`` defaults to ``$MA_TRACE_STORE`` or ``./.ma-trace/episodes``. References
    accepted by :meth:`load`, :meth:`delete` and :meth:`resolve` are a full id, a
    unique id prefix, or ``"latest"``.
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root or os.environ.get(ENV_STORE) or DEFAULT_STORE_DIR).expanduser()

    def _path(self, episode_id: str) -> Path:
        return self.root / f"{episode_id}.json"

    def save(self, episode: Episode) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(episode.id)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(episode.to_json())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):  # pragma: no cover - only on failure
                os.unlink(tmp)
        return target

    def ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("ep-*.json"))

    def _recency_key(self, episode_id: str) -> tuple[float, int, str]:
        """Sort key for ``latest``: the episode's recorded start time first, then the file
        modification time and the id as tie-breakers (file times coincide on file systems with
        coarse timestamps, so they are not reliable on their own)."""
        path = self._path(episode_id)
        try:
            with path.open(encoding="utf-8") as fh:
                started = float(json.load(fh).get("started_at", 0.0))
        except (OSError, ValueError, TypeError):
            started = 0.0
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = 0
        return (started, mtime, episode_id)

    def resolve(self, ref: str) -> str:
        ids = self.ids()
        if ref == "latest":
            if not ids:
                raise KeyError("no episodes recorded yet")
            return max(ids, key=self._recency_key)
        if ref in ids:
            return ref
        matches = [i for i in ids if i.startswith(ref)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"no episode matches {ref!r}")
        raise KeyError(f"ambiguous episode reference {ref!r}: {', '.join(matches)}")

    def load(self, ref: str) -> Episode:
        path = self._path(self.resolve(ref))
        with path.open(encoding="utf-8") as fh:
            return Episode.from_dict(json.load(fh))

    def list(self, limit: int | None = None) -> list[EpisodeSummary]:
        summaries = [EpisodeSummary.of(self.load(i)) for i in self.ids()]
        summaries.sort(key=lambda s: s.started_at, reverse=True)
        return summaries[:limit] if limit else summaries

    def delete(self, ref: str) -> None:
        self._path(self.resolve(ref)).unlink()

    def latest(self) -> Episode | None:
        try:
            return self.load("latest")
        except KeyError:
            return None

    def __contains__(self, episode_id: object) -> bool:
        return isinstance(episode_id, str) and self._path(episode_id).exists()

    def __len__(self) -> int:
        return len(self.ids())


class InMemoryEpisodeStore:
    """Dictionary-backed store with the same interface as :class:`EpisodeStore` (tests, notebooks)."""

    def __init__(self) -> None:
        self._episodes: dict[str, Episode] = {}

    def save(self, episode: Episode) -> Any:
        self._episodes[episode.id] = Episode.from_dict(episode.to_dict())
        return episode.id

    def ids(self) -> list[str]:
        return sorted(self._episodes)

    def resolve(self, ref: str) -> str:
        if ref == "latest":
            if not self._episodes:
                raise KeyError("no episodes recorded yet")
            # most recently started; among equal start times the one saved last
            return max(enumerate(self._episodes.values()), key=lambda ie: (ie[1].started_at, ie[0]))[1].id
        if ref in self._episodes:
            return ref
        matches = [i for i in self._episodes if i.startswith(ref)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"no episode matches {ref!r}")
        raise KeyError(f"ambiguous episode reference {ref!r}")

    def load(self, ref: str) -> Episode:
        return Episode.from_dict(self._episodes[self.resolve(ref)].to_dict())

    def list(self, limit: int | None = None) -> list[EpisodeSummary]:
        summaries = sorted(
            (EpisodeSummary.of(e) for e in self._episodes.values()), key=lambda s: s.started_at, reverse=True
        )
        return summaries[:limit] if limit else summaries

    def delete(self, ref: str) -> None:
        del self._episodes[self.resolve(ref)]

    def latest(self) -> Episode | None:
        try:
            return self.load("latest")
        except KeyError:
            return None

    def __contains__(self, episode_id: object) -> bool:
        return episode_id in self._episodes

    def __len__(self) -> int:
        return len(self._episodes)
