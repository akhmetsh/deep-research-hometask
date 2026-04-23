"""Hard assertion engine for the Deep Research Lite eval framework.

Each assertion is a pure function:
    (trace: EvalTrace, params: dict) -> AssertionResult

The engine dispatches by the assertion's `type` string.

Implemented assertions
----------------------
stopped_reason_is    -- trace.stopped_reason matches expected value
tool_called          -- a specific tool appears at least once in the trace
tool_sequence        -- a list of tools appears in order (as a subsequence)
citation_was_fetched -- every cited URL was actually passed to fetch_url
answer_contains      -- final_answer contains a substring
answer_not_contains  -- final_answer does NOT contain a substring
answer_word_count_le -- word count of final_answer ≤ max_words
answer_not_empty     -- final_answer is non-None and non-blank

Adding a new assertion: write a function matching the signature above and
register it in _ASSERTION_HANDLERS at the bottom of this file.
"""

from __future__ import annotations

from typing import Any

from eval.case_schema import AssertionResult, HardAssertion
from eval.eval_trace import EvalTrace


# ---------------------------------------------------------------------------
# Helpers for reading the trace
# ---------------------------------------------------------------------------


def _all_tool_calls(trace: EvalTrace) -> list[dict[str, Any]]:
    """Return every tool-call record in chronological order."""
    calls: list[dict[str, Any]] = []
    for msg in trace.messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                calls.append(tc)
    return calls


def _fetched_urls(trace: EvalTrace) -> list[str]:
    """Return all URL strings passed to fetch_url, in call order."""
    return [
        tc["args"]["url"]
        for tc in _all_tool_calls(trace)
        if tc.get("name") == "fetch_url" and isinstance(tc.get("args"), dict) and "url" in tc["args"]
    ]


# ---------------------------------------------------------------------------
# Assertion handlers
# ---------------------------------------------------------------------------


def _stopped_reason_is(trace: EvalTrace, params: dict) -> AssertionResult:
    expected = str(params.get("value", "finish"))
    actual = trace.stopped_reason
    passed = actual == expected
    return AssertionResult(
        assertion_type="stopped_reason_is",
        passed=passed,
        reason=f"stopped_reason={actual!r}, expected={expected!r}",
        params=params,
    )


def _tool_called(trace: EvalTrace, params: dict) -> AssertionResult:
    tool = str(params.get("tool", ""))
    names = [tc.get("name") for tc in _all_tool_calls(trace)]
    passed = tool in names
    return AssertionResult(
        assertion_type="tool_called",
        passed=passed,
        reason=f"tool={tool!r} {'found' if passed else 'not found'} in call sequence {names}",
        params=params,
    )


def _tool_sequence(trace: EvalTrace, params: dict) -> AssertionResult:
    """Check that `sequence` appears as an ordered subsequence of tool calls.

    Example: sequence=[web_search, fetch_url, finish] passes if those three
    tools appear in that order, regardless of what else is called in between.
    """
    sequence: list[str] = [str(s) for s in params.get("sequence", [])]
    names = [tc.get("name", "") for tc in _all_tool_calls(trace)]

    if not sequence:
        return AssertionResult(
            assertion_type="tool_sequence",
            passed=True,
            reason="empty sequence — vacuously satisfied",
            params=params,
        )

    seq_idx = 0
    for name in names:
        if seq_idx < len(sequence) and name == sequence[seq_idx]:
            seq_idx += 1

    passed = seq_idx == len(sequence)
    if passed:
        reason = f"sequence {sequence} satisfied as subsequence of {names}"
    else:
        reason = (
            f"sequence {sequence} not satisfied; "
            f"matched {seq_idx}/{len(sequence)} step(s) in {names}"
        )
    return AssertionResult(
        assertion_type="tool_sequence",
        passed=passed,
        reason=reason,
        params=params,
    )


def _citation_was_fetched(trace: EvalTrace, params: dict) -> AssertionResult:
    """Every URL in citations must have been passed to fetch_url.

    Vacuously true when citations is empty (pair with a citation_count_ge
    assertion if you also need to enforce that citations exist).
    """
    citations = trace.citations
    fetched = set(_fetched_urls(trace))

    if not citations:
        return AssertionResult(
            assertion_type="citation_was_fetched",
            passed=True,
            reason="no citations present — vacuously satisfied",
            params=params,
        )

    unfetched = [c for c in citations if c not in fetched]
    passed = not unfetched
    if passed:
        reason = f"all {len(citations)} citation(s) were fetched: {citations}"
    else:
        reason = f"citation(s) not fetched: {unfetched} (fetched: {sorted(fetched)})"
    return AssertionResult(
        assertion_type="citation_was_fetched",
        passed=passed,
        reason=reason,
        params=params,
    )


def _answer_contains(trace: EvalTrace, params: dict) -> AssertionResult:
    substring = str(params.get("substring", ""))
    ci = bool(params.get("case_insensitive", False))
    answer = trace.final_answer or ""
    haystack = answer.lower() if ci else answer
    needle = substring.lower() if ci else substring
    passed = needle in haystack
    return AssertionResult(
        assertion_type="answer_contains",
        passed=passed,
        reason=f"{'found' if passed else 'not found'} {substring!r} in answer (case_insensitive={ci})",
        params=params,
    )


def _answer_not_contains(trace: EvalTrace, params: dict) -> AssertionResult:
    substring = str(params.get("substring", ""))
    ci = bool(params.get("case_insensitive", False))
    answer = trace.final_answer or ""
    haystack = answer.lower() if ci else answer
    needle = substring.lower() if ci else substring
    passed = needle not in haystack
    return AssertionResult(
        assertion_type="answer_not_contains",
        passed=passed,
        reason=f"{substring!r} {'absent' if passed else 'PRESENT'} in answer (case_insensitive={ci})",
        params=params,
    )


def _answer_word_count_le(trace: EvalTrace, params: dict) -> AssertionResult:
    max_words = int(params.get("max_words", 120))
    answer = trace.final_answer or ""
    count = len(answer.split())
    passed = count <= max_words
    return AssertionResult(
        assertion_type="answer_word_count_le",
        passed=passed,
        reason=f"word_count={count}, max_words={max_words}",
        params=params,
    )


def _answer_not_empty(trace: EvalTrace, params: dict) -> AssertionResult:
    answer = trace.final_answer
    passed = answer is not None and answer.strip() != ""
    return AssertionResult(
        assertion_type="answer_not_empty",
        passed=passed,
        reason="answer is non-empty" if passed else "answer is None or blank",
        params=params,
    )


def _citation_not_contains(trace: EvalTrace, params: dict) -> AssertionResult:
    """Assert that a specific URL does NOT appear in citations.

    Complementary to citation_was_fetched: use this when a page is known to
    be useless (stub, broken, confidential) and should never be cited even if
    the agent fetched it while exploring.
    """
    url = str(params.get("url", ""))
    passed = url not in trace.citations
    if passed:
        reason = f"{url!r} not present in citations"
    else:
        reason = f"{url!r} PRESENT in citations: {trace.citations}"
    return AssertionResult(
        assertion_type="citation_not_contains",
        passed=passed,
        reason=reason,
        params=params,
    )


# ---------------------------------------------------------------------------
# Registry and engine
# ---------------------------------------------------------------------------

_ASSERTION_HANDLERS: dict[str, Any] = {
    "stopped_reason_is": _stopped_reason_is,
    "tool_called": _tool_called,
    "tool_sequence": _tool_sequence,
    "citation_was_fetched": _citation_was_fetched,
    "answer_contains": _answer_contains,
    "answer_not_contains": _answer_not_contains,
    "answer_word_count_le": _answer_word_count_le,
    "answer_not_empty": _answer_not_empty,
    "citation_not_contains": _citation_not_contains,
}


class HardAssertionEngine:
    def check(self, assertion: HardAssertion, trace: EvalTrace) -> AssertionResult:
        handler = _ASSERTION_HANDLERS.get(assertion.type)
        if handler is None:
            return AssertionResult(
                assertion_type=assertion.type,
                passed=False,
                reason=f"Unknown assertion type: {assertion.type!r}. "
                       f"Known types: {sorted(_ASSERTION_HANDLERS)}",
                params=assertion.params,
            )
        return handler(trace, assertion.params)

    def check_all(
        self, assertions: list[HardAssertion], trace: EvalTrace
    ) -> list[AssertionResult]:
        return [self.check(a, trace) for a in assertions]
