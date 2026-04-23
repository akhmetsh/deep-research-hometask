"""Judge validation utility.

Scans saved EvalTraces for LLM judge verdicts, reports per-case consistency,
and supports manual spot-checking of rationales.

Usage
-----
    python -m eval.judge_validate                      # summary table
    python -m eval.judge_validate --case tc01          # full rationale per run
    python -m eval.judge_validate --sample 5           # 5 random rationales
    python -m eval.judge_validate --threshold 0.20     # custom inconsistency flag
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from eval.eval_trace import EvalTrace, EVAL_TRACES_DIR

_SEP = "=" * 70


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _load_judged_traces() -> list[EvalTrace]:
    """Return every saved EvalTrace that has a judge_verdict."""
    traces: list[EvalTrace] = []
    if not EVAL_TRACES_DIR.exists():
        return traces
    for f in sorted(EVAL_TRACES_DIR.rglob("*.json")):
        try:
            t = EvalTrace.load(f)
            if t.judge_verdict is not None:
                traces.append(t)
        except Exception:
            pass
    return traces


def _group_by_case(
    traces: list[EvalTrace],
) -> dict[str, list[EvalTrace]]:
    groups: dict[str, list[EvalTrace]] = {}
    for t in traces:
        groups.setdefault(t.case_id, []).append(t)
    return groups


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _summary_table(
    groups: dict[str, list[EvalTrace]],
    threshold: float,
) -> None:
    total_traces = sum(len(v) for v in groups.values())
    print(_SEP)
    print(f"Judge Validation — {total_traces} trace(s) with verdicts across {len(groups)} case(s)")
    print(_SEP)

    col = f"{'Case':<42} {'Rubric':<18} {'N':>3}  {'Mean':>5}  {'Range':>11}  Flag"
    print(col)
    print("-" * len(col))

    inconsistent = 0
    all_scores: list[float] = []
    rows = []

    for case_id in sorted(groups):
        ts = groups[case_id]
        scores = [
            float(t.judge_verdict["score"])
            for t in ts
            if t.judge_verdict and t.judge_verdict.get("score") is not None
        ]
        rubric = next(
            (t.judge_verdict.get("rubric_id", "?") for t in ts if t.judge_verdict), "?"
        )
        if not scores:
            continue
        mn = sum(scores) / len(scores)
        lo, hi = min(scores), max(scores)
        spread = hi - lo
        flag = "  ** inconsistent **" if spread > threshold else ""
        if spread > threshold:
            inconsistent += 1
        all_scores.extend(scores)
        rows.append(
            f"  {case_id:<40} {rubric:<18} {len(scores):>3}  {mn:>5.2f}"
            f"  {lo:.2f}–{hi:.2f}  {flag}"
        )

    for r in rows:
        print(r)

    print(_SEP)
    global_mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"Inconsistency threshold : {threshold}")
    print(f"Inconsistent cases      : {inconsistent}/{len(groups)}")
    print(f"Mean score (all traces) : {global_mean:.3f}")


def _print_rationale(trace: EvalTrace) -> None:
    jv = trace.judge_verdict
    if not jv:
        return
    verdict = jv.get("verdict", "?").upper()
    score = jv.get("score", 0.0)
    passed = jv.get("passed")
    flag = "PASS" if passed else "FAIL"
    rubric = jv.get("rubric_id", "?")
    rationale = jv.get("rationale", "(none)")
    flags = jv.get("flags", [])

    print(f"  Run     : {trace.run_id}")
    print(f"  Repeat  : {trace.repeat_index}")
    print(f"  Verdict : {verdict}  [{flag}]  score={score:.2f}  rubric={rubric}")
    if flags:
        print(f"  Dims    : {', '.join(flags)}")
    print(f"  Rationale:")
    for line in rationale.splitlines():
        print(f"    {line}")
    print()


def _case_detail(case_id: str, groups: dict[str, list[EvalTrace]]) -> None:
    ts = groups.get(case_id)
    if not ts:
        print(f"No judged traces found for case: {case_id}")
        return
    print(_SEP)
    print(f"Detail: {case_id}  ({len(ts)} run(s))")
    print(_SEP)
    for t in sorted(ts, key=lambda x: x.repeat_index):
        _print_rationale(t)


def _random_sample(
    traces: list[EvalTrace],
    n: int,
) -> None:
    sample = random.sample(traces, min(n, len(traces)))
    print(_SEP)
    print(f"Random sample — {len(sample)} trace(s)")
    print(_SEP)
    for t in sample:
        print(f"[{t.case_id}]")
        _print_rationale(t)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="judge_validate",
        description="Spot-check LLM judge verdicts across saved EvalTraces",
    )
    parser.add_argument(
        "--case", metavar="CASE_ID",
        help="Show full rationale for all runs of a specific case",
    )
    parser.add_argument(
        "--sample", type=int, metavar="N",
        help="Print N randomly selected rationales for manual review",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15, metavar="F",
        help="Score-range threshold above which a case is flagged inconsistent (default: 0.15)",
    )
    args = parser.parse_args()

    traces = _load_judged_traces()
    if not traces:
        print("No judged traces found in eval_traces/. Run 'eval run-all' first.")
        sys.exit(1)

    groups = _group_by_case(traces)

    _summary_table(groups, args.threshold)

    if args.case:
        print()
        _case_detail(args.case, groups)

    if args.sample:
        print()
        _random_sample(traces, args.sample)


if __name__ == "__main__":
    main()
