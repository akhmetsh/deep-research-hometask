"""Tool efficiency metric.

Reports a breakdown of tool usage and flags the most common protocol
violation in this agent: answering from a search snippet without ever
calling fetch_url ("answered_from_snippet").

passed = False when answered_from_snippet is True, because the system
prompt explicitly requires "Fetch before you answer."
"""

from __future__ import annotations

from eval.case_schema import TestCase
from eval.eval_trace import EvalTrace
from eval.metrics.base import MetricPlugin, MetricResult


def _all_tool_calls(trace: EvalTrace) -> list[dict]:
    calls = []
    for msg in trace.messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                calls.append(tc)
    return calls


class ToolEfficiencyMetric(MetricPlugin):
    @property
    def name(self) -> str:
        return "tool_efficiency"

    def score(self, case: TestCase, trace: EvalTrace) -> MetricResult:
        calls = _all_tool_calls(trace)
        total = len(calls)

        counts: dict[str, int] = {}
        for tc in calls:
            n = tc.get("name", "unknown")
            counts[n] = counts.get(n, 0) + 1

        fetch_count = counts.get("fetch_url", 0)
        finish_called = trace.stopped_reason == "finish"

        # Core violation: agent reached finish without fetching any page.
        answered_from_snippet = finish_called and fetch_count == 0

        # Secondary signal: extract_quotes called without any preceding fetch
        # (quotes would have nothing real to extract from).
        eq_count = counts.get("extract_quotes", 0)
        quotes_without_fetch = eq_count > 0 and fetch_count == 0

        passed = not answered_from_snippet

        if answered_from_snippet:
            reason = (
                f"VIOLATION: finish() called without any fetch_url — "
                f"answer grounded in search snippet only (total calls: {total})"
            )
        else:
            reason = (
                f"{total} tool call(s): "
                + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            )

        return MetricResult(
            metric_name=self.name,
            value={
                "total_tool_calls": total,
                "tool_counts": counts,
                "answered_from_snippet": answered_from_snippet,
                "quotes_without_fetch": quotes_without_fetch,
            },
            passed=passed,
            reason=reason,
        )
