# SPDX-License-Identifier: Apache-2.0
"""Adaptive sampling: a low baseline sample rate that escalates to full capture after anomalies."""

from __future__ import annotations

import random
import threading
from collections.abc import Sequence
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from opentelemetry.trace import Link, SpanKind, TraceState

from .semconv import SemConv

_MAX_TRACE_ID = (1 << 64) - 1


class AdaptiveSampler(Sampler):
    """OpenTelemetry sampler with an episode-level decision and anomaly escalation.

    * ``base_ratio`` - fraction of episodes captured in full during normal operation
      (a typical production setting is 0.10; the library default is 1.0 so that a
      developer sees everything until they opt in to sampling).
    * ``anomaly_ratio`` - fraction captured while in anomaly mode (default 1.0).
    * ``escalation_episodes`` - how many subsequent episodes stay in anomaly mode after
      :meth:`escalate` is called.

    The tracer calls :meth:`decide_episode` when an episode starts and stamps the
    decision on the root span as the ``<ns>.episode.sampled`` attribute; the sampler
    honours that attribute for root spans and follows the parent for child spans, so
    OTel export and episode recording always agree.
    """

    def __init__(
        self,
        base_ratio: float = 1.0,
        anomaly_ratio: float = 1.0,
        escalation_episodes: int = 20,
        *,
        rng: random.Random | None = None,
        sampled_attribute: str | None = None,
    ) -> None:
        if not 0.0 <= base_ratio <= 1.0 or not 0.0 <= anomaly_ratio <= 1.0:
            raise ValueError("ratios must be within [0, 1]")
        self.base_ratio = base_ratio
        self.anomaly_ratio = anomaly_ratio
        self.escalation_episodes = escalation_episodes
        self.sampled_attribute = sampled_attribute or SemConv().EPISODE_SAMPLED
        self._rng = rng or random.Random()
        self._remaining = 0
        self._lock = threading.Lock()
        self.escalations: list[str | None] = []
        self.decisions: int = 0
        self.sampled_episodes: int = 0

    # -- episode level ------------------------------------------------------

    @property
    def in_anomaly_mode(self) -> bool:
        return self._remaining > 0

    @property
    def current_ratio(self) -> float:
        return self.anomaly_ratio if self.in_anomaly_mode else self.base_ratio

    def escalate(self, reason: str | None = None) -> None:
        """Switch to ``anomaly_ratio`` for the next ``escalation_episodes`` episodes."""
        with self._lock:
            self._remaining = self.escalation_episodes
            self.escalations.append(reason)

    def decide_episode(self) -> bool:
        with self._lock:
            ratio = self.current_ratio
            if self._remaining > 0:
                self._remaining -= 1
            self.decisions += 1
            sampled = ratio >= 1.0 or self._rng.random() < ratio
            if sampled:
                self.sampled_episodes += 1
            return sampled

    # -- OpenTelemetry Sampler API -----------------------------------------

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Any = None,
        links: Sequence[Link] | None = None,
        trace_state: TraceState | None = None,
    ) -> SamplingResult:
        parent_span = trace.get_current_span(parent_context)
        psc = parent_span.get_span_context()
        if psc.is_valid:
            decision = Decision.RECORD_AND_SAMPLE if psc.trace_flags.sampled else Decision.DROP
            return SamplingResult(
                decision, attributes if decision != Decision.DROP else None, psc.trace_state
            )
        if attributes and self.sampled_attribute in attributes:
            flag = bool(attributes[self.sampled_attribute])
            decision = Decision.RECORD_AND_SAMPLE if flag else Decision.DROP
        else:
            bound = round(self.current_ratio * _MAX_TRACE_ID)
            decision = Decision.RECORD_AND_SAMPLE if (trace_id & _MAX_TRACE_ID) < bound else Decision.DROP
        return SamplingResult(decision, attributes if decision != Decision.DROP else None, trace_state)

    def get_description(self) -> str:
        return (
            f"AdaptiveSampler(base={self.base_ratio}, anomaly={self.anomaly_ratio}, "
            f"escalation={self.escalation_episodes})"
        )
