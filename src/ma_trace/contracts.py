# SPDX-License-Identifier: Apache-2.0
"""Environment contracts: seed-controlled, verifiable pre/post-conditions for actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .events import ContractEvent, fingerprint


def state_fingerprint(state: Any) -> str:
    """Stable SHA-256 fingerprint of an environment/agent state (see :func:`ma_trace.events.fingerprint`)."""
    return fingerprint(state)


@dataclass
class ContractCheck:
    """Outcome of verifying a live contract against a recorded one during replay."""

    agent: str
    action: str
    pre_match: bool
    post_match: bool | None
    recorded_pre: str
    recorded_post: str | None
    actual_pre: str
    actual_post: str | None

    @property
    def ok(self) -> bool:
        return self.pre_match and (self.post_match is None or self.post_match)


@dataclass
class Contract:
    """A live contract handle yielded by ``tracer.contract(...)``.

    Capture the state before the action, run the action, then call
    :meth:`set_post_state` (and optionally :meth:`set_result`). The tracer records a
    :class:`~ma_trace.events.ContractEvent` on exit. Under replay the pre- and
    post-state hashes are compared with the recorded ones.
    """

    agent: str
    action: str
    pre_hash: str
    seed: int | None = None
    post_hash: str | None = None
    result: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    recorded: ContractEvent | None = None  # the recorded counterpart under replay
    check: ContractCheck | None = None

    @classmethod
    def begin(
        cls, agent: str, action: str, pre_state: Any, *, seed: int | None = None, **metadata: Any
    ) -> Contract:
        return cls(
            agent=agent, action=action, pre_hash=state_fingerprint(pre_state), seed=seed, metadata=metadata
        )

    def set_post_state(self, state: Any) -> None:
        self.post_hash = state_fingerprint(state)

    def set_result(self, result: Any) -> None:
        self.result = result if isinstance(result, str) else repr(result)

    @property
    def expected_post_hash(self) -> str | None:
        """Under replay: the post-state hash the recorded run produced."""
        return self.recorded.post_hash if self.recorded is not None else None

    def verify(self) -> ContractCheck | None:
        if self.recorded is None:
            return None
        post_match: bool | None
        if self.post_hash is None or self.recorded.post_hash is None:
            post_match = None
        else:
            post_match = self.post_hash == self.recorded.post_hash
        self.check = ContractCheck(
            agent=self.agent,
            action=self.action,
            pre_match=self.pre_hash == self.recorded.pre_hash,
            post_match=post_match,
            recorded_pre=self.recorded.pre_hash,
            recorded_post=self.recorded.post_hash,
            actual_pre=self.pre_hash,
            actual_post=self.post_hash,
        )
        return self.check

    def to_event(self) -> ContractEvent:
        verified: bool | None = None
        if self.check is not None:
            verified = self.check.ok
        return ContractEvent(
            agent=self.agent,
            action=self.action,
            pre_hash=self.pre_hash,
            post_hash=self.post_hash,
            seed=self.seed,
            result=self.result,
            verified=verified,
            metadata=dict(self.metadata),
        )
