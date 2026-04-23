"""CLI entry point for the Deep Research Lite eval framework.

Usage
-----
    python -m eval.eval_cli run <case_id>
    python -m eval.eval_cli run-all
    python -m eval.eval_cli rescore <trace_path>

Exit codes: 0 = all assertions passed, 1 = one or more failed.

Commands to add later:
    diff <run_a> <run_b>   -- show regressions between two run summaries
    report <run_dir>       -- generate HTML report for a run directory
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reconfigure stdout/stderr to UTF-8 so judge rationale (which may contain
# Unicode from the LLM) prints correctly on Windows consoles using narrow
# codepages like cp1251.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", encoding="utf-8-sig")
except ImportError:
    pass

from eval.eval_trace import EvalTrace
from eval.loader import load_all_cases, load_case_by_id
from eval.runner import make_error_trace, run_case, run_case_with_retry
from eval.run_summary import build_summary, diff_summaries, load_summary, save_summary
from eval.scorer import score

_SEP = "=" * 66


# ---------------------------------------------------------------------------
# Shared display helpers
# ---------------------------------------------------------------------------

def _tool_call_sequence(trace: EvalTrace) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for msg in trace.messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                calls.append(tc)
    return calls


def _args_preview(args: dict[str, Any], max_len: int = 55) -> str:
    if not args:
        return ""
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


def _wrap_text(text: str, width: int = 68, indent: str = "  ") -> str:
    wrapped = textwrap.fill(text.replace("\n", " "), width=width)
    return "\n".join(indent + line for line in wrapped.splitlines())


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile over a sorted list."""
    if not values:
        return 0.0
    sv = sorted(values)
    k = (len(sv) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (k - lo)


def _get_metric(trace: EvalTrace, name: str) -> dict[str, Any] | None:
    for m in trace.metric_results:
        if m.get("metric_name") == name:
            return m
    return None


# ---------------------------------------------------------------------------
# Diagnostic per-trace printer
# ---------------------------------------------------------------------------

def _print_trace(trace: EvalTrace) -> None:
    status = "PASS" if trace.case_passed else "FAIL"
    failed = [r for r in trace.assertion_results if not r["passed"]]
    n_fail = len(failed)
    n_total = len(trace.assertion_results)

    print(_SEP)
    print(f"Case:      {trace.case_id}  [{status}]")
    print(f"Run ID:    {trace.run_id}")

    stopped_note = ""
    for r in failed:
        if r["assertion_type"] == "stopped_reason_is":
            stopped_note = f"  (expected '{r['params'].get('value', '?')}')"
            break
    print(f"Stopped:   {trace.stopped_reason}{stopped_note}")
    print(
        f"Cost:      ${trace.cost_usd:.6f}  |  "
        f"Time: {trace.wall_time_ms} ms  |  "
        f"Tokens: in={trace.total_tokens.get('input', 0)} "
        f"out={trace.total_tokens.get('output', 0)}"
    )

    # ---- Tool call sequence ------------------------------------------------
    calls = _tool_call_sequence(trace)
    print()
    if calls:
        print(f"Tool sequence ({len(calls)} call{'s' if len(calls) != 1 else ''}):")
        for i, tc in enumerate(calls, 1):
            preview = _args_preview(tc.get("args") or {})
            print(f"  {i:>2}.  {tc.get('name', '?'):<20} {preview}")
        if trace.stopped_reason != "finish":
            print("        (finish() was never called)")
    else:
        print("Tool sequence: (none)")

    # ---- Answer ------------------------------------------------------------
    answer_text = trace.final_answer or "(none)"
    word_count = len((trace.final_answer or "").split())
    print()
    print(f"Answer ({word_count} words):")
    print(_wrap_text(answer_text))

    # ---- Citations ---------------------------------------------------------
    if trace.citations:
        print()
        print("Citations:")
        for url in trace.citations:
            print(f"  - {url}")
    else:
        print("\nCitations: (none)")

    # ---- Assertions: failures first ----------------------------------------
    print()
    if n_fail == 0:
        print(f"Assertions: all {n_total} passed")
    else:
        print(f"Assertions: {n_fail} failed / {n_total} total")
    for r in sorted(trace.assertion_results, key=lambda r: (r["passed"], r["assertion_type"])):
        icon = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"  {icon}  {r['assertion_type']}")
        print(f"         {r['reason']}")

    # ---- Metric plugin results ---------------------------------------------
    if trace.metric_results:
        print()
        print("Metrics:")
        for m in trace.metric_results:
            mname = m.get("metric_name", "?")
            reason = m.get("reason", "")
            mp = m.get("passed")
            flag = " [WARN]" if mp is False else " [ok]" if mp is True else ""
            print(f"  {mname}{flag}")
            print(f"    {reason}")

    # ---- LLM judge verdict -------------------------------------------------
    jv = trace.judge_verdict
    if jv is not None:
        print()
        rubric_id = jv.get("rubric_id", "?")
        verdict = jv.get("verdict", "?").upper()
        score_val = jv.get("score", 0.0)
        j_passed = jv.get("passed")
        j_flag = "[PASS]" if j_passed else "[FAIL]"
        if jv.get("verdict") == "error":
            j_flag = "[ERR]"
        print(f"Judge ({rubric_id}):  {verdict}  {j_flag}  score={score_val:.2f}")
        rationale = jv.get("rationale", "")
        if rationale:
            print(_wrap_text(rationale))
        flags = jv.get("flags", [])
        if flags:
            print(f"  Failed dimensions: {flags}")
        if jv.get("error"):
            print(f"  Error: {jv['error']}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    case = load_case_by_id(args.case_id)
    n = args.repeats
    max_retries = args.max_retries
    print(f"Running: {case.id}  (repeats={n}, max-retries={max_retries})")
    print(f"Input:   {case.input}")
    print()

    traces: list[EvalTrace] = []
    for i in range(n):
        if n > 1:
            print(f"--- repeat {i + 1}/{n} ---")
        trace = run_case_with_retry(case, repeat_index=i, max_retries=max_retries)
        saved = trace.save()
        _print_trace(trace)
        if trace.retry_count > 0:
            print(f"Retried: {trace.retry_count} time(s)")
            for err in trace.retry_errors:
                print(f"  - {err}")
        print()
        print(f"Trace saved: {saved}")
        traces.append(trace)

    if n > 1:
        pass_count = sum(1 for t in traces if t.case_passed)
        print()
        print(_SEP)
        status = "PASS" if pass_count == n else "FAIL"
        print(f"[{status}]  {case.id}  {pass_count}/{n} repeats passed")
        return 0 if pass_count == n else 1

    return 0 if traces[0].case_passed else 1


def cmd_run_all(args: argparse.Namespace) -> int:
    cases = load_all_cases()
    if not cases:
        print("No test cases found in tests/cases/")
        return 1

    n = args.repeats
    workers = args.workers
    max_retries = args.max_retries
    total_tasks = len(cases) * n
    print(
        f"Running {len(cases)} case(s) x {n} repeat(s)"
        f"  [{total_tasks} task(s), workers={workers}, max-retries={max_retries}]\n"
    )

    print_lock = Lock()
    results: dict[tuple[str, int], EvalTrace] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_meta = {
            executor.submit(run_case_with_retry, case, i, max_retries): (case.id, i, case)
            for case in cases
            for i in range(n)
        }

        for future in as_completed(future_to_meta):
            case_id, rep_idx, case = future_to_meta[future]
            try:
                trace = future.result()
            except Exception as exc:
                trace = make_error_trace(case, rep_idx, str(exc))

            results[(case_id, rep_idx)] = trace

            with print_lock:
                rlabel = f"  r{rep_idx + 1}/{n}" if n > 1 else ""
                status = "PASS" if trace.case_passed else "FAIL"
                first_fail = next(
                    (r["reason"] for r in trace.assertion_results if not r["passed"]),
                    None,
                )
                retry_note = (
                    f"  (retried {trace.retry_count}x)" if trace.retry_count > 0 else ""
                )
                fail_note = (
                    f"  -- {first_fail[:50]}" if first_fail and not trace.case_passed else ""
                )
                print(
                    f"  done  {case_id:<42}{rlabel}"
                    f"  {status}  ${trace.cost_usd:.4f}  {trace.wall_time_ms}ms"
                    f"{retry_note}{fail_note}"
                )

    # Reconstruct rows in original case order
    rows: list[tuple[str, list[EvalTrace]]] = [
        (case.id, [results[(case.id, i)] for i in range(n)])
        for case in cases
    ]

    # ---- Per-case results table --------------------------------------------
    total_cases = len(rows)
    fully_passed = sum(1 for _, traces in rows if all(t.case_passed for t in traces))
    print(_SEP)
    if n > 1:
        print(f"RESULTS  {fully_passed}/{total_cases} cases fully passed  (repeats={n})")
    else:
        print(f"RESULTS  {fully_passed}/{total_cases} passed")
    print(_SEP)

    for case_id, traces in rows:
        pass_count = sum(1 for t in traces if t.case_passed)
        all_ok = pass_count == n
        icon = "PASS" if all_ok else "FAIL"
        label = f"{icon} {pass_count}/{n}" if n > 1 else icon

        # Gather notes from all traces (de-duped)
        notes: list[str] = []
        total_assert_fail = sum(
            sum(1 for r in t.assertion_results if not r["passed"]) for t in traces
        )
        if not all_ok and total_assert_fail:
            notes.append(f"{total_assert_fail} assertion fail(s)")

        metric_warns: set[str] = set()
        for t in traces:
            for m in t.metric_results:
                if m.get("passed") is False:
                    metric_warns.add(m["metric_name"])
        if metric_warns:
            notes.append(f"metric warn: {', '.join(sorted(metric_warns))}")

        # Judge: show score range across repeats
        judge_scores = []
        for t in traces:
            jv = t.judge_verdict
            if jv and jv.get("verdict") != "error":
                judge_scores.append(jv.get("score", 0.0))
        if judge_scores:
            if len(judge_scores) > 1:
                notes.append(f"judge={min(judge_scores):.2f}-{max(judge_scores):.2f}")
            else:
                jv = traces[0].judge_verdict
                j_icon = "ok" if jv.get("passed") else "FAIL"
                notes.append(f"judge={j_icon}({judge_scores[0]:.2f})")

        note_str = f"  [{';  '.join(notes)}]" if notes else ""
        print(f"  [{label}]  {case_id}{note_str}")

    # ---- Aggregate metrics -------------------------------------------------
    all_traces = [t for _, traces in rows for t in traces]
    wall_times = [t.wall_time_ms for t in all_traces]
    total_cost = sum(t.cost_usd for t in all_traces)
    total_in = sum(t.total_tokens.get("input", 0) for t in all_traces)
    total_out = sum(t.total_tokens.get("output", 0) for t in all_traces)

    tool_call_counts: list[int] = []
    for t in all_traces:
        eff = _get_metric(t, "tool_efficiency")
        if eff and isinstance(eff.get("value"), dict):
            v = eff["value"].get("total_tool_calls")
            if v is not None:
                tool_call_counts.append(v)

    p50 = round(_percentile(wall_times, 50))
    p95 = round(_percentile(wall_times, 95))
    mean_calls = (
        round(sum(tool_call_counts) / len(tool_call_counts), 1)
        if tool_call_counts else "n/a"
    )

    print()
    print("Aggregate metrics:")
    pct_label = f"{100 * fully_passed // total_cases if total_cases else 0}%"
    print(f"  Pass rate:       {fully_passed}/{total_cases}  ({pct_label})")
    if n > 1:
        # Flakiness: cases that are not 0% and not 100%
        flaky = [
            (cid, sum(1 for t in ts if t.case_passed), len(ts))
            for cid, ts in rows
            if 0 < sum(1 for t in ts if t.case_passed) < len(ts)
        ]
        if flaky:
            print(f"  Flaky cases ({len(flaky)}):")
            for cid, pc, nc in flaky:
                print(f"    {cid}: {pc}/{nc} passed")
    print(f"  Total cost:      ${total_cost:.4f}  (tokens in={total_in} out={total_out})")
    print(f"  p50 latency:     {p50} ms")
    print(f"  p95 latency:     {p95} ms")
    print(f"  Mean tool calls: {mean_calls}")

    summary = build_summary(rows)
    summary_path = save_summary(summary)
    print()
    print(f"Summary saved: {summary_path}")

    return 0 if fully_passed == total_cases else 1


def _delta_str(val: float | int | None, unit: str = "", precision: int = 4) -> str:
    if val is None:
        return ""
    sign = "+" if val > 0 else ""
    if isinstance(val, float):
        return f"  {sign}{val:.{precision}f}{unit}"
    return f"  {sign}{val}{unit}"


def cmd_diff(args: argparse.Namespace) -> int:
    """Diff two run-all summaries, highlighting regressions and improvements."""
    path_a = Path(args.run_a)
    path_b = Path(args.run_b)
    for p in (path_a, path_b):
        if not p.exists():
            print(f"Error: file not found: {p}")
            return 1

    a = load_summary(path_a)
    b = load_summary(path_b)

    diffs = diff_summaries(a, b)

    regressions = [d for d in diffs if d.status == "regression"]
    improvements = [d for d in diffs if d.status == "improvement"]
    stable_fail = [d for d in diffs if d.status == "stable_fail"]
    stable_pass = [d for d in diffs if d.status == "stable_pass"]
    new_cases = [d for d in diffs if d.status == "new"]
    removed = [d for d in diffs if d.status == "removed"]

    print(_SEP)
    print(f"DIFF  A: {path_a.name}  vs  B: {path_b.name}")
    print(f"      A run at: {a.run_at[:19].replace('T', ' ')} UTC")
    print(f"      B run at: {b.run_at[:19].replace('T', ' ')} UTC")
    print(_SEP)

    repeats_b = b.aggregate.get("repeats", 1)

    def _print_case(d: "CaseDiff", label: str) -> None:  # noqa: F821
        # Pass-rate suffix for repeat-aware runs
        rate_str = ""
        if d.b and d.b.repeats > 1:
            rate_str = f"  ({d.b.pass_count}/{d.b.repeats} repeats passed)"
        elif d.a and d.a.repeats > 1 and d.b is None:
            rate_str = f"  ({d.a.pass_count}/{d.a.repeats})"
        print(f"  [{label}]  {d.case_id}{rate_str}")

        deltas: list[str] = []
        if d.latency_delta is not None:
            deltas.append(f"latency{_delta_str(d.latency_delta, ' ms', 0)}")
        if d.cost_delta is not None:
            deltas.append(f"cost{_delta_str(d.cost_delta, '$', 6)}")
        if d.judge_delta is not None:
            deltas.append(f"judge{_delta_str(d.judge_delta, '', 3)}")
        if d.tool_calls_delta is not None:
            deltas.append(f"tools{_delta_str(d.tool_calls_delta, '', 0)}")
        if d.pass_rate_delta is not None and (
            d.a and d.a.repeats > 1 or d.b and d.b.repeats > 1
        ):
            deltas.append(f"pass_rate{_delta_str(d.pass_rate_delta, '', 3)}")
        if deltas:
            print(f"           " + "  ".join(deltas))
        if d.b and d.b.n_assertions_failed:
            print(f"           {d.b.n_assertions_failed} assertion(s) failed")
        if d.b and d.b.stopped_reason not in ("finish", None, ""):
            print(f"           stopped: {d.b.stopped_reason}")

    # ---- Regressions (most important — shown first) ------------------------
    if regressions:
        print(f"\n*** REGRESSIONS ({len(regressions)}) — pass in A, FAIL in B ***")
        for d in regressions:
            _print_case(d, "REGR")
    else:
        print("\nRegressions: none")

    # ---- Improvements -------------------------------------------------------
    if improvements:
        print(f"\nImprovements ({len(improvements)}) — fail in A, PASS in B:")
        for d in improvements:
            _print_case(d, "IMPR")

    # ---- Stable failures ----------------------------------------------------
    if stable_fail:
        print(f"\nStable failures ({len(stable_fail)}):")
        for d in stable_fail:
            _print_case(d, "FAIL")

    # ---- Stable passes (brief) ----------------------------------------------
    if stable_pass:
        ids = ", ".join(d.case_id for d in stable_pass)
        print(f"\nStable passes ({len(stable_pass)}): {ids}")

    # ---- New / removed cases ------------------------------------------------
    if new_cases:
        print(f"\nNew cases in B ({len(new_cases)}): " + ", ".join(d.case_id for d in new_cases))
    if removed:
        print(f"\nRemoved cases from B ({len(removed)}): " + ", ".join(d.case_id for d in removed))

    # ---- Aggregate delta ----------------------------------------------------
    ag_a = a.aggregate
    ag_b = b.aggregate

    def _agg_delta(key: str, unit: str = "", precision: int = 4) -> str:
        va, vb = ag_a.get(key), ag_b.get(key)
        if va is None or vb is None:
            return "n/a"
        delta = vb - va
        sign = "+" if delta > 0 else ""
        if isinstance(delta, float):
            return f"{vb:.{precision}f} ({sign}{delta:.{precision}f})"
        return f"{vb} ({sign}{delta})"

    print()
    print(_SEP)
    print("Aggregate delta  (B vs A):")
    pa = ag_a.get("passed", 0)
    pb = ag_b.get("passed", 0)
    ta = ag_a.get("total", 0)
    tb = ag_b.get("total", 0)
    ra = ag_a.get("repeats", 1)
    rb = ag_b.get("repeats", 1)
    delta_pass = pb - pa
    sign = "+" if delta_pass > 0 else ""
    rep_note = f"  (A repeats={ra}, B repeats={rb})" if ra != 1 or rb != 1 else ""
    print(f"  Pass rate:       {pb}/{tb}  ({sign}{delta_pass} cases){rep_note}")
    print(f"  Total cost:      {_agg_delta('total_cost_usd', '$', 4)}")
    print(f"  p50 latency ms:  {_agg_delta('p50_ms', ' ms', 0)}")
    print(f"  p95 latency ms:  {_agg_delta('p95_ms', ' ms', 0)}")
    print(f"  Mean tool calls: {_agg_delta('mean_tool_calls', '', 1)}")

    return 0 if not regressions else 1


def cmd_view(args: argparse.Namespace) -> int:
    """Generate a self-contained HTML viewer for a saved EvalTrace."""
    from eval.viewer import save_viewer
    path = Path(args.trace_path)
    if not path.exists():
        print(f"Error: file not found: {path}")
        return 1
    trace = EvalTrace.load(path)
    out = save_viewer(trace)
    print(f"Viewer saved: {out}")
    print(f"Open in browser: file:///{out.resolve().as_posix()}")
    return 0


def cmd_rescore(args: argparse.Namespace) -> int:
    """Reload a saved EvalTrace from disk and re-run the scorer.

    The agent is NOT called. Useful for updating scores after assertion or
    metric logic changes (bump SCORER_VERSION in eval_trace.py).
    The trace file is updated in place.
    """
    path = Path(args.trace_path)
    if not path.exists():
        print(f"Error: file not found: {path}")
        return 1

    trace = EvalTrace.load(path)
    print(f"Loaded trace:   {trace.run_id}")
    print(f"Case:           {trace.case_id}")
    print(f"Previously scored at: {trace.scored_at or '(never)'}")
    old_status = (
        "PASS" if trace.case_passed is True
        else "FAIL" if trace.case_passed is False
        else "unscored"
    )
    print(f"Old result:     {old_status}")
    print()

    try:
        case = load_case_by_id(trace.case_id)
    except ValueError as e:
        print(f"Error loading case: {e}")
        return 1

    trace = score(case, trace)
    path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")

    _print_trace(trace)
    print()
    print(f"Trace updated in place: {path}")
    return 0 if trace.case_passed else 1


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eval",
        description="Deep Research Lite evaluation framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a single test case by ID")
    run_p.add_argument("case_id", help="Test case ID to run")
    run_p.add_argument(
        "--repeats", type=int, default=1, metavar="N",
        help="Number of times to run the case (default: 1)",
    )
    run_p.add_argument(
        "--max-retries", type=int, default=3, metavar="N",
        help="Max retries on transient API/network errors (default: 3)",
    )

    run_all_p = sub.add_parser("run-all", help="Run every test case and print a summary")
    run_all_p.add_argument(
        "--repeats", type=int, default=1, metavar="N",
        help="Number of times to run each case (default: 1)",
    )
    run_all_p.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help="Parallel worker threads (default: 4; use 1 for serial)",
    )
    run_all_p.add_argument(
        "--max-retries", type=int, default=3, metavar="N",
        help="Max retries per task on transient API/network errors (default: 3)",
    )

    rescore_p = sub.add_parser(
        "rescore",
        help="Rescore a saved EvalTrace without re-running the agent",
    )
    rescore_p.add_argument("trace_path", help="Path to a saved .json EvalTrace file")

    view_p = sub.add_parser("view", help="Generate an HTML viewer for a saved EvalTrace")
    view_p.add_argument("trace_path", help="Path to a saved .json EvalTrace file")

    diff_p = sub.add_parser("diff", help="Diff two run-all summary files")
    diff_p.add_argument("run_a", help="Path to the baseline summary JSON (run A)")
    diff_p.add_argument("run_b", help="Path to the comparison summary JSON (run B)")

    args = parser.parse_args()

    if args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "run-all":
        sys.exit(cmd_run_all(args))
    elif args.command == "rescore":
        sys.exit(cmd_rescore(args))
    elif args.command == "view":
        sys.exit(cmd_view(args))
    elif args.command == "diff":
        sys.exit(cmd_diff(args))


if __name__ == "__main__":
    main()
