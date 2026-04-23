"""Metric plugin registry.

To add a new metric:
  1. Create eval/metrics/your_metric.py implementing MetricPlugin.
  2. Import it and add an instance to REGISTERED_METRICS below.
  scorer.py and eval_trace.py need no changes.
"""

from eval.metrics.base import MetricPlugin, MetricResult
from eval.metrics.cost import CostMetric
from eval.metrics.latency import LatencyMetric
from eval.metrics.safety_format import SafetyFormatMetric
from eval.metrics.tool_efficiency import ToolEfficiencyMetric

REGISTERED_METRICS: list[MetricPlugin] = [
    ToolEfficiencyMetric(),
    CostMetric(),
    LatencyMetric(),
    SafetyFormatMetric(),
]

__all__ = ["MetricPlugin", "MetricResult", "REGISTERED_METRICS"]
