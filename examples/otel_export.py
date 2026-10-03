# SPDX-License-Identifier: Apache-2.0
"""Export MA-Trace spans and SLO metrics to an OpenTelemetry collector / Jaeger / Phoenix.

    pip install 'multi-agent-observability[otlp]'
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318      # e.g. Jaeger's OTLP HTTP port
    python examples/otel_export.py

With ``exporter="console"`` instead, spans are printed to stdout.
"""

from __future__ import annotations

import ma_trace as mt


def main() -> None:
    mt.configure(
        "ma-trace-demo",
        exporter="otlp",  # spans -> OTLP/HTTP (uses OTEL_EXPORTER_OTLP_ENDPOINT)
        metrics_exporter="otlp",  # SLO metrics -> OTLP/HTTP
        sampler=mt.AdaptiveSampler(base_ratio=0.1, anomaly_ratio=1.0, escalation_episodes=10),
    )
    for i in range(20):
        with mt.episode("checkout", seed=i, record=(i == 0) or None) as ep:
            with mt.agent_span("planner"):
                mt.coordination("planner", "executor", f"order {i}", kind="delegation")
            with mt.agent_span("executor"):
                mt.action("executor", "charge", {"order": i})
                mt.action("executor", "charge", {"order": i})  # a redundant retry
        m = ep.metrics()
        if m.redundant_action_rate > 0.4:
            mt.flag_anomaly("redundant action rate above 0.4")  # next 10 episodes are captured in full
        print(ep.id, "sampled" if ep.sampled else "sampled=no", f"RAR={m.redundant_action_rate:.2f}")


if __name__ == "__main__":
    main()
