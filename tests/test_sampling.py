from __future__ import annotations

import random

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import Decision

import ma_trace as mt
from ma_trace.sampling import AdaptiveSampler


def test_ratio_validation():
    with pytest.raises(ValueError):
        AdaptiveSampler(base_ratio=1.5)


def test_decide_episode_and_escalation():
    s = AdaptiveSampler(base_ratio=0.0, anomaly_ratio=1.0, escalation_episodes=3, rng=random.Random(1))
    assert [s.decide_episode() for _ in range(5)] == [False] * 5
    s.escalate("spike")
    assert s.in_anomaly_mode and s.escalations == ["spike"]
    assert [s.decide_episode() for _ in range(4)] == [True, True, True, False]
    assert s.sampled_episodes == 3 and s.decisions == 9
    assert "AdaptiveSampler" in s.get_description()


def test_base_ratio_statistics():
    s = AdaptiveSampler(base_ratio=0.1, rng=random.Random(42))
    n = sum(s.decide_episode() for _ in range(2000))
    assert 150 < n < 250


def test_should_sample_honours_episode_attribute_and_parent():
    s = AdaptiveSampler(base_ratio=0.0)
    res = s.should_sample(None, 123, "root", attributes={s.sampled_attribute: True})
    assert res.decision == Decision.RECORD_AND_SAMPLE
    res = s.should_sample(None, 123, "root", attributes={s.sampled_attribute: False})
    assert res.decision == Decision.DROP
    res = s.should_sample(None, 123, "root", attributes={})
    assert res.decision == Decision.DROP
    full = AdaptiveSampler(base_ratio=1.0)
    assert full.should_sample(None, 2**63, "root", attributes={}).decision == Decision.RECORD_AND_SAMPLE


def test_unsampled_episode_exports_no_spans():
    exporter = InMemorySpanExporter()
    sampler = AdaptiveSampler(base_ratio=0.0)
    provider = TracerProvider(sampler=sampler)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    t = mt.MATrace(store="memory", tracer_provider=provider, sampler=sampler, otel_metrics=False)
    with t.episode("silent"), t.agent_span("a"):
        assert not trace.get_current_span().get_span_context().trace_flags.sampled
    assert exporter.get_finished_spans() == ()
    with t.episode("loud", record=True), t.agent_span("a"):
        pass
    assert {s.name for s in exporter.get_finished_spans()} == {"invoke_workflow loud", "invoke_agent a"}
