"""Cost metric.

Reports cost_usd, input tokens, and output tokens directly from the trace.
Informational only (no pass/fail threshold at this stage).
"""

from __future__ import annotations

from eval.case_schema import TestCase
from eval.eval_trace import EvalTrace
from eval.metrics.base import MetricPlugin, MetricResult


class CostMetric(MetricPlugin):
    @property
    def name(self) -> str:
        return "cost"

    def score(self, case: TestCase, trace: EvalTrace) -> MetricResult:
        tokens = trace.total_tokens
        in_tok = tokens.get("input", 0)
        out_tok = tokens.get("output", 0)
        cost = trace.cost_usd

        return MetricResult(
            metric_name=self.name,
            value={
                "cost_usd": cost,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            },
            passed=None,
            reason=f"${cost:.6f} ({in_tok} in + {out_tok} out tokens)",
        )
