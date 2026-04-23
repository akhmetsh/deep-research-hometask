"""Latency metric.

Reports total wall time and the latency of each assistant turn (each
representing one LLM API call). Informational only.
"""

from __future__ import annotations

from eval.case_schema import TestCase
from eval.eval_trace import EvalTrace
from eval.metrics.base import MetricPlugin, MetricResult


def _mean(values: list[int | float]) -> float:
    return sum(values) / len(values) if values else 0.0


class LatencyMetric(MetricPlugin):
    @property
    def name(self) -> str:
        return "latency"

    def score(self, case: TestCase, trace: EvalTrace) -> MetricResult:
        step_latencies = [
            msg["latency_ms"]
            for msg in trace.messages
            if msg.get("role") == "assistant" and "latency_ms" in msg
        ]

        wall_ms = trace.wall_time_ms
        steps = len(step_latencies)
        mean_step = round(_mean(step_latencies))

        return MetricResult(
            metric_name=self.name,
            value={
                "wall_time_ms": wall_ms,
                "steps": steps,
                "step_latencies_ms": step_latencies,
                "mean_step_ms": mean_step,
            },
            passed=None,
            reason=(
                f"wall={wall_ms} ms, "
                f"{steps} LLM step(s), "
                f"mean step={mean_step} ms"
            ),
        )
