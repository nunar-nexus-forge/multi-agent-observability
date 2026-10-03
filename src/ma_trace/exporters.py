# SPDX-License-Identifier: Apache-2.0
"""Helpers that build OpenTelemetry providers wired for MA-Trace.

The defaults suit a collector-based deployment: traces are buffered locally and exported
in batches every 10 seconds.
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import metrics as otel_metrics
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.sampling import Sampler

log = logging.getLogger("ma_trace")

BATCH_DELAY_MS = 10_000  # batched export every 10 s


def build_tracer_provider(
    service_name: str = "ma-trace",
    exporter: str | SpanExporter | None = "console",
    *,
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
    sampler: Sampler | None = None,
    batch_delay_ms: int = BATCH_DELAY_MS,
    max_queue_size: int = 8192,
    max_export_batch_size: int = 512,
    simple: bool = False,
    resource_attributes: dict[str, Any] | None = None,
    set_global: bool = True,
) -> TracerProvider:
    """Create a :class:`TracerProvider`.

    ``exporter`` may be ``"console"``, ``"otlp"`` (HTTP/protobuf; needs the ``otlp``
    extra), ``"none"``/``None`` (no export) or any :class:`SpanExporter` instance.
    """
    attrs: dict[str, Any] = {SERVICE_NAME: service_name, "ma_trace.version": _version()}
    if resource_attributes:
        attrs.update(resource_attributes)
    provider = TracerProvider(resource=Resource.create(attrs), sampler=sampler)
    span_exporter = _resolve_span_exporter(exporter, endpoint, headers)
    if span_exporter is not None:
        if simple:
            provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        else:
            provider.add_span_processor(
                BatchSpanProcessor(
                    span_exporter,
                    max_queue_size=max_queue_size,
                    schedule_delay_millis=batch_delay_ms,
                    max_export_batch_size=max_export_batch_size,
                )
            )
    if set_global:
        _set_global_tracer_provider(provider)
    return provider


def build_meter_provider(
    service_name: str = "ma-trace",
    exporter: str | Any | None = "console",
    *,
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
    interval_ms: int = BATCH_DELAY_MS,
    readers: list[MetricReader] | None = None,
    set_global: bool = True,
) -> MeterProvider:
    resource = Resource.create({SERVICE_NAME: service_name})
    all_readers: list[MetricReader] = list(readers or [])
    metric_exporter = _resolve_metric_exporter(exporter, endpoint, headers)
    if metric_exporter is not None:
        all_readers.append(PeriodicExportingMetricReader(metric_exporter, export_interval_millis=interval_ms))
    provider = MeterProvider(resource=resource, metric_readers=all_readers)
    if set_global:
        try:
            otel_metrics.set_meter_provider(provider)
        except Exception:  # pragma: no cover - OTel logs a warning when re-set
            log.debug("global meter provider already set")
    return provider


def _version() -> str:
    from ._version import __version__

    return __version__


def _set_global_tracer_provider(provider: TracerProvider) -> None:
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        log.debug("global tracer provider already set; not overriding")
        return
    trace.set_tracer_provider(provider)


def _resolve_span_exporter(
    exporter: str | SpanExporter | None, endpoint: str | None, headers: dict[str, str] | None
) -> SpanExporter | None:
    if exporter is None or exporter == "none":
        return None
    if isinstance(exporter, str):
        if exporter == "console":
            return ConsoleSpanExporter()
        if exporter == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            except ImportError as e:
                raise ImportError(
                    "OTLP export requires the otlp extra: pip install 'multi-agent-observability[otlp]'"
                ) from e
            return OTLPSpanExporter(endpoint=endpoint, headers=headers)
        raise ValueError(
            f"unknown span exporter {exporter!r}; use 'console', 'otlp', 'none' or a SpanExporter"
        )
    return exporter


def _resolve_metric_exporter(
    exporter: str | Any | None, endpoint: str | None, headers: dict[str, str] | None
) -> Any:
    if exporter is None or exporter == "none":
        return None
    if isinstance(exporter, str):
        if exporter == "console":
            return ConsoleMetricExporter()
        if exporter == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            except ImportError as e:
                raise ImportError(
                    "OTLP export requires the otlp extra: pip install 'multi-agent-observability[otlp]'"
                ) from e
            return OTLPMetricExporter(endpoint=endpoint, headers=headers)
        raise ValueError(f"unknown metric exporter {exporter!r}")
    return exporter
