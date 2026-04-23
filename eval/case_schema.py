"""Schema types for eval test cases and assertion results.

Everything downstream (loader, assertions, scorer, reporter) imports from
here so the shape of a test case is defined in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HardAssertion:
    """A single deterministic check to run against a trace.

    `type` is the assertion name (e.g. "stopped_reason_is").
    `params` holds every other key from the YAML/JSON definition.
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCase:
    id: str
    description: str
    input: str
    hard_assertions: list[HardAssertion] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    repeats: int = 1
    rubric: str | None = None  # name of rubric file in tests/rubrics/ (None = skip judge)


@dataclass
class AssertionResult:
    assertion_type: str
    passed: bool
    reason: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assertion_type": self.assertion_type,
            "passed": self.passed,
            "reason": self.reason,
            "params": self.params,
        }
