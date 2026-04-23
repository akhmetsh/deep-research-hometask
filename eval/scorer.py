"""Scorer: orchestrates all checks against a trace and stamps results.

Execution order:
  1. Hard assertions  (deterministic; primary gate for case_passed)
  2. Metric plugins   (informational + warnings; stored separately)
  3. LLM judge        (soft quality gate; only when case.rubric is set)

case_passed logic:
  - No rubric: case_passed = all hard assertions passed
  - With rubric: case_passed = hard_passed AND judge_passed
  - Judge error: treated as non-blocking (case_passed = hard_passed)

Usage (live run):
    trace = EvalTrace.from_run_result(run_result.to_dict(), case.id)
    trace = score(case, trace)

Usage (rescore from disk — no agent call needed):
    trace = EvalTrace.load(path)
    trace = score(case, trace)
    trace.save()
"""

from __future__ import annotations

from datetime import datetime, timezone

from eval.assertions import HardAssertionEngine
from eval.case_schema import TestCase
from eval.eval_trace import EvalTrace
from eval.judge import LLMJudge
from eval.metrics import REGISTERED_METRICS

_engine = HardAssertionEngine()
_judge = LLMJudge()


def score(case: TestCase, trace: EvalTrace) -> EvalTrace:
    """Run all checks and stamp results onto *trace*. Returns trace."""

    # 1. Hard assertions
    assertion_results = _engine.check_all(case.hard_assertions, trace)
    trace.assertion_results = [r.to_dict() for r in assertion_results]
    hard_passed = all(r.passed for r in assertion_results)

    # 2. Metric plugins
    trace.metric_results = [m.score(case, trace).to_dict() for m in REGISTERED_METRICS]

    # 3. LLM judge (only when the case declares a rubric)
    if case.rubric:
        verdict = _judge.judge(case, trace)
        trace.judge_verdict = verdict.to_dict()
        # Judge errors are non-blocking so a transient API failure doesn't
        # fail every case in a run.
        if verdict.verdict == "error":
            judge_gate = True
        else:
            judge_gate = verdict.passed
    else:
        trace.judge_verdict = None
        judge_gate = True

    trace.case_passed = hard_passed and judge_gate
    trace.scored_at = datetime.now(timezone.utc).isoformat()
    return trace
