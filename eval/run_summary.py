"""Persist and diff run-all summaries.

Each run-all execution saves a RunSummary JSON to run_summaries/.
The diff command compares two summaries and highlights regressions.
Supports N repeats per case; spread stats (min/mean/max) are stored
per-case so the diff and terminal display can surface flakiness.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.eval_trace import EvalTrace

_ROOT = Path(__file__).parent.parent
SUMMARIES_DIR = _ROOT / "run_summaries"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pct(values: list[int | float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    k = (len(sv) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (k - lo)


def _tool_calls_from_trace(trace: EvalTrace) -> int | None:
    for m in trace.metric_results:
        if m.get("metric_name") == "tool_efficiency":
            val = m.get("value")
            if isinstance(val, dict):
                return val.get("total_tool_calls")
    return None


# ---------------------------------------------------------------------------
# CaseSummary
# ---------------------------------------------------------------------------

@dataclass
class CaseSummary:
    case_id: str

    # Pass/fail across repeats
    passed: bool | None      # True only if ALL repeats passed
    pass_count: int          # number of repeats that passed
    repeats: int             # N (total repeats run)

    # Representative / aggregate scalars (means for N>1 — used by diff delta)
    stopped_reason: str
    cost_usd: float          # mean cost
    wall_time_ms: int        # mean latency
    n_assertions_failed: int # total across all repeats
    judge_score: float | None    # mean judge score, or None
    judge_verdict: str | None    # majority verdict, or None
    tool_calls: int | None       # mean tool calls rounded, or None

    # Per-repeat raw values for spread display
    latency_ms_values: list[int] = field(default_factory=list)
    cost_usd_values: list[float] = field(default_factory=list)
    tool_calls_values: list[int | None] = field(default_factory=list)
    judge_score_values: list[float | None] = field(default_factory=list)

    # Trace file paths (one per repeat)
    trace_paths: list[str] = field(default_factory=list)

    @property
    def trace_path(self) -> str:
        """First trace path — backward compat."""
        return self.trace_paths[0] if self.trace_paths else ""

    @property
    def pass_rate(self) -> float:
        return self.pass_count / self.repeats if self.repeats else 0.0

    def spread_str(self, values: list[int | float], unit: str = "") -> str:
        """'min–max' when N>1, single value when N=1."""
        clean = [v for v in values if v is not None]
        if not clean:
            return "n/a"
        if len(clean) == 1:
            v = clean[0]
            return f"{v:.0f}{unit}" if isinstance(v, float) else f"{v}{unit}"
        lo, hi = min(clean), max(clean)
        if isinstance(lo, float):
            return f"{lo:.0f}–{hi:.0f}{unit}"
        return f"{lo}–{hi}{unit}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "pass_count": self.pass_count,
            "repeats": self.repeats,
            "stopped_reason": self.stopped_reason,
            "cost_usd": self.cost_usd,
            "wall_time_ms": self.wall_time_ms,
            "n_assertions_failed": self.n_assertions_failed,
            "judge_score": self.judge_score,
            "judge_verdict": self.judge_verdict,
            "tool_calls": self.tool_calls,
            "latency_ms_values": self.latency_ms_values,
            "cost_usd_values": self.cost_usd_values,
            "tool_calls_values": self.tool_calls_values,
            "judge_score_values": self.judge_score_values,
            "trace_paths": self.trace_paths,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CaseSummary":
        passed = d.get("passed")
        repeats = d.get("repeats", 1)
        pass_count = d.get("pass_count", (1 if passed else 0) if passed is not None else 0)
        wt = d.get("wall_time_ms", 0)
        cu = d.get("cost_usd", 0.0)
        tc = d.get("tool_calls")
        js = d.get("judge_score")
        tp = d.get("trace_path", "")
        tps = d.get("trace_paths", [tp] if tp else [])
        return cls(
            case_id=d["case_id"],
            passed=passed,
            pass_count=pass_count,
            repeats=repeats,
            stopped_reason=d.get("stopped_reason", ""),
            cost_usd=cu,
            wall_time_ms=wt,
            n_assertions_failed=d.get("n_assertions_failed", 0),
            judge_score=js,
            judge_verdict=d.get("judge_verdict"),
            tool_calls=tc,
            latency_ms_values=d.get("latency_ms_values", [wt] if wt else []),
            cost_usd_values=d.get("cost_usd_values", [cu] if cu else []),
            tool_calls_values=d.get("tool_calls_values", [tc]),
            judge_score_values=d.get("judge_score_values", [js]),
            trace_paths=tps,
        )


# ---------------------------------------------------------------------------
# RunSummary
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    summary_id: str
    run_at: str
    cases: dict[str, CaseSummary] = field(default_factory=dict)
    aggregate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "run_at": self.run_at,
            "cases": {k: v.to_dict() for k, v in self.cases.items()},
            "aggregate": self.aggregate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunSummary":
        rs = cls(
            summary_id=d["summary_id"],
            run_at=d["run_at"],
            aggregate=d.get("aggregate", {}),
        )
        for case_id, cd in d.get("cases", {}).items():
            rs.cases[case_id] = CaseSummary.from_dict(cd)
        return rs


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build_summary(rows: list[tuple[str, list[EvalTrace]]]) -> RunSummary:
    """Build a RunSummary from (case_id, [traces]) rows."""
    ts = datetime.now(timezone.utc)
    summary_id = ts.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_at = ts.isoformat()

    cases: dict[str, CaseSummary] = {}
    all_latencies: list[int] = []
    total_cost = 0.0
    all_tool_calls: list[int] = []
    total_passed_cases = 0
    n_repeats_global = 1

    for case_id, traces in rows:
        n = len(traces)
        n_repeats_global = max(n_repeats_global, n)
        pass_count = sum(1 for t in traces if t.case_passed)
        all_passed = pass_count == n

        latencies = [t.wall_time_ms for t in traces]
        costs = [t.cost_usd for t in traces]
        tool_calls_list = [_tool_calls_from_trace(t) for t in traces]
        judge_scores = []
        judge_verdicts = []
        n_assertions_total = 0
        trace_paths = []

        for t in traces:
            jv = t.judge_verdict
            if jv:
                score = jv.get("score")
                if score is not None:
                    judge_scores.append(float(score))
                judge_verdicts.append(jv.get("verdict"))
            n_assertions_total += sum(1 for r in t.assertion_results if not r["passed"])
            trace_paths.append(str(t.save()))

        # Aggregate scalars
        mean_cost = _mean(costs)
        mean_lat = int(round(_mean(latencies)))
        mean_tc = (
            int(round(_mean([v for v in tool_calls_list if v is not None])))
            if any(v is not None for v in tool_calls_list) else None
        )
        mean_js = round(_mean(judge_scores), 3) if judge_scores else None

        # Representative judge verdict: last non-None, or None
        jv_repr = next(
            (v for v in reversed(judge_verdicts) if v is not None), None
        )

        stopped = traces[-1].stopped_reason if traces else ""

        cases[case_id] = CaseSummary(
            case_id=case_id,
            passed=all_passed,
            pass_count=pass_count,
            repeats=n,
            stopped_reason=stopped,
            cost_usd=mean_cost,
            wall_time_ms=mean_lat,
            n_assertions_failed=n_assertions_total,
            judge_score=mean_js,
            judge_verdict=jv_repr,
            tool_calls=mean_tc,
            latency_ms_values=latencies,
            cost_usd_values=costs,
            tool_calls_values=tool_calls_list,
            judge_score_values=judge_scores if judge_scores else [None] * n,
            trace_paths=trace_paths,
        )

        all_latencies.extend(latencies)
        total_cost += sum(costs)
        all_tool_calls.extend(v for v in tool_calls_list if v is not None)
        if all_passed:
            total_passed_cases += 1

    n_cases = len(rows)
    aggregate = {
        "total": n_cases,
        "passed": total_passed_cases,
        "repeats": n_repeats_global,
        "pass_rate": round(total_passed_cases / n_cases, 4) if n_cases else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "p50_ms": int(round(_pct(all_latencies, 50))),
        "p95_ms": int(round(_pct(all_latencies, 95))),
        "mean_tool_calls": (
            round(_mean(all_tool_calls), 1) if all_tool_calls else None
        ),
    }

    return RunSummary(summary_id=summary_id, run_at=run_at, cases=cases, aggregate=aggregate)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_summary(summary: RunSummary, out_dir: Path | None = None) -> Path:
    d = out_dir or SUMMARIES_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{summary.summary_id}.json"
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return path


def load_summary(path: Path) -> RunSummary:
    return RunSummary.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

@dataclass
class CaseDiff:
    case_id: str
    status: str   # "regression" | "improvement" | "stable_pass" | "stable_fail" | "new" | "removed"
    a: CaseSummary | None
    b: CaseSummary | None

    # Deltas (B - A), None if not computable
    cost_delta: float | None = None
    latency_delta: int | None = None
    judge_delta: float | None = None
    tool_calls_delta: int | None = None
    pass_rate_delta: float | None = None   # for N>1 flakiness changes


def diff_summaries(a: RunSummary, b: RunSummary) -> list[CaseDiff]:
    all_ids = sorted(set(a.cases) | set(b.cases))
    diffs: list[CaseDiff] = []

    for case_id in all_ids:
        ca = a.cases.get(case_id)
        cb = b.cases.get(case_id)

        if ca is None:
            diffs.append(CaseDiff(case_id=case_id, status="new", a=None, b=cb))
            continue
        if cb is None:
            diffs.append(CaseDiff(case_id=case_id, status="removed", a=ca, b=None))
            continue

        if ca.passed and not cb.passed:
            status = "regression"
        elif not ca.passed and cb.passed:
            status = "improvement"
        elif cb.passed:
            status = "stable_pass"
        else:
            status = "stable_fail"

        cost_delta = round(cb.cost_usd - ca.cost_usd, 6)
        latency_delta = cb.wall_time_ms - ca.wall_time_ms
        judge_delta = (
            round(cb.judge_score - ca.judge_score, 3)
            if ca.judge_score is not None and cb.judge_score is not None else None
        )
        tool_calls_delta = (
            cb.tool_calls - ca.tool_calls
            if ca.tool_calls is not None and cb.tool_calls is not None else None
        )
        pass_rate_delta = round(cb.pass_rate - ca.pass_rate, 3)

        diffs.append(CaseDiff(
            case_id=case_id, status=status, a=ca, b=cb,
            cost_delta=cost_delta,
            latency_delta=latency_delta,
            judge_delta=judge_delta,
            tool_calls_delta=tool_calls_delta,
            pass_rate_delta=pass_rate_delta,
        ))

    return diffs
