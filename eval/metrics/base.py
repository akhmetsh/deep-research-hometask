"""Base types for the metric plugin system.

A MetricPlugin is a stateless object that inspects a scored EvalTrace and
returns a MetricResult. Plugins are registered in eval/metrics/__init__.py.

Adding a new metric:
  1. Create eval/metrics/my_metric.py implementing MetricPlugin.
  2. Add one line to REGISTERED_METRICS in eval/metrics/__init__.py.
  That is the full contract — scorer.py and eval_trace.py need no changes.

MetricResult.passed semantics:
  True / False  — the metric has a clear pass/fail threshold
  None          — informational; reported but does not gate case_passed
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    metric_name: str
    value: Any          # scalar, dict, or list — must be JSON-serialisable
    passed: bool | None # None = informational only
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "passed": self.passed,
            "reason": self.reason,
            "details": self.details,
        }


class MetricPlugin(ABC):
    """Stateless scorer plugin. Implement `name` and `score` only."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique snake_case identifier shown in reports."""

    @abstractmethod
    def score(self, case: "TestCase", trace: "EvalTrace") -> MetricResult:  # type: ignore[name-defined]
        """Compute and return a MetricResult for the given trace."""
