# SPDX-License-Identifier: Apache-2.0
"""Context variables shared between the tracer, decorators and replay engine."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .episode import Episode
    from .replay import ReplaySession

_current_episode: ContextVar[Episode | None] = ContextVar("ma_trace_episode", default=None)
_current_agent: ContextVar[str | None] = ContextVar("ma_trace_agent", default=None)
_current_replay: ContextVar[ReplaySession | None] = ContextVar("ma_trace_replay", default=None)


def current_episode() -> Episode | None:
    """The episode open in the current context (thread / task), if any."""
    return _current_episode.get()


def current_agent() -> str | None:
    """The innermost agent whose span is active in the current context."""
    return _current_agent.get()


def current_replay() -> ReplaySession | None:
    return _current_replay.get()
