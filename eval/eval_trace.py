"""EvalTrace: the central data model for a single agent run + its scores.

Design goals:
  - Completely serialisable to JSON on disk (no live objects).
  - Loadable without importing agent.py, so the scorer can rescore an old
    trace without re-running the agent.
  - run_result is stored as a raw dict (RunResult.to_dict() output) so we
    stay decoupled from agent internals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Bump this string whenever the scoring logic changes so old cached scores
# can be identified as stale and rescored.
SCORER_VERSION = "0.1.0"

EVAL_TRACES_DIR = Path(__file__).parent.parent / "eval_traces"


@dataclass
class EvalTrace:
    case_id: str
    repeat_index: int              # 0-based; >0 when --repeats N is used
    run_result: dict[str, Any]     # RunResult.to_dict() from agent.py

    # Populated by scorer.score(); None means "not yet scored".
    assertion_results: list[dict[str, Any]] = field(default_factory=list)
    metric_results: list[dict[str, Any]] = field(default_factory=list)
    judge_verdict: dict[str, Any] | None = None   # JudgeVerdict.to_dict() or None
    case_passed: bool | None = None
    scorer_version: str = SCORER_VERSION
    scored_at: str | None = None

    # Populated by run_case_with_retry(); 0 when no retries were needed.
    retry_count: int = 0
    retry_errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors — read from run_result without coupling callers
    # to the agent's RunResult class.
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self.run_result["run_id"]

    @property
    def final_answer(self) -> str | None:
        return self.run_result.get("final_answer")

    @property
    def citations(self) -> list[str]:
        return self.run_result.get("citations", [])

    @property
    def stopped_reason(self) -> str:
        return self.run_result.get("stopped_reason", "")

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.run_result.get("messages", [])

    @property
    def cost_usd(self) -> float:
        return float(self.run_result.get("cost_usd", 0.0))

    @property
    def wall_time_ms(self) -> int:
        return int(self.run_result.get("wall_time_ms", 0))

    @property
    def total_tokens(self) -> dict[str, int]:
        return self.run_result.get("total_tokens", {"input": 0, "output": 0})

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "repeat_index": self.repeat_index,
            "scorer_version": self.scorer_version,
            "scored_at": self.scored_at,
            "case_passed": self.case_passed,
            "assertion_results": self.assertion_results,
            "metric_results": self.metric_results,
            "judge_verdict": self.judge_verdict,
            "retry_count": self.retry_count,
            "retry_errors": self.retry_errors,
            "run_result": self.run_result,
        }

    def save(self, directory: Path | None = None) -> Path:
        """Write the trace to <directory>/<case_id>/<run_id>.json."""
        out_dir = (directory or EVAL_TRACES_DIR) / self.case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "EvalTrace":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            case_id=data["case_id"],
            repeat_index=data.get("repeat_index", 0),
            run_result=data["run_result"],
            assertion_results=data.get("assertion_results", []),
            metric_results=data.get("metric_results", []),
            judge_verdict=data.get("judge_verdict"),           # None for old traces
            case_passed=data.get("case_passed"),
            scorer_version=data.get("scorer_version", SCORER_VERSION),
            scored_at=data.get("scored_at"),
            retry_count=data.get("retry_count", 0),
            retry_errors=data.get("retry_errors", []),
        )

    @classmethod
    def from_run_result(
        cls,
        run_result_dict: dict[str, Any],
        case_id: str,
        repeat_index: int = 0,
    ) -> "EvalTrace":
        return cls(
            case_id=case_id,
            repeat_index=repeat_index,
            run_result=run_result_dict,
        )
