from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import ma_trace as mt
from ma_trace.sampling import AdaptiveSampler


@pytest.fixture
def tracer():
    t = mt.MATrace(store="memory", otel_metrics=False)
    mt.set_tracer(t)
    yield t
    mt.set_tracer(None)


@pytest.fixture
def spans():
    """An in-memory OTel exporter plus a tracer wired to it."""
    exporter = InMemorySpanExporter()
    sampler = AdaptiveSampler(base_ratio=1.0)
    provider = TracerProvider(sampler=sampler)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    t = mt.MATrace(store="memory", tracer_provider=provider, sampler=sampler, otel_metrics=False)
    mt.set_tracer(t)
    yield exporter, t
    mt.set_tracer(None)
