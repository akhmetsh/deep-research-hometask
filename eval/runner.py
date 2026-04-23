"""Runner: invoke the agent and return a scored EvalTrace.

This is the only module that imports agent.py. Keeping the dependency
isolated here means the rest of the eval framework (loader, scorer,
reporter) can be used without an Anthropic API key — e.g. for rescoring
saved traces or generating HTML reports.
"""

from __future__ import annotations

import time
import uuid
import sys
from pathlib import Path

# agent.py lives in the project root; add it to sys.path if needed.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent import run_agent  # noqa: E402  (import after sys.path mutation)

from eval.case_schema import TestCase
from eval.eval_trace import EvalTrace
from eval.scorer import score


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

try:
    import anthropic as _anthropic

    def _is_retryable(exc: Exception) -> bool:
        """True for transient API/network failures that are safe to retry."""
        if isinstance(exc, (
            _anthropic.RateLimitError,      # 429
            _anthropic.APIConnectionError,  # network failure
            _anthropic.APITimeoutError,     # request timed out
        )):
            return True
        # 5xx server errors
        if isinstance(exc, _anthropic.APIStatusError) and exc.status_code >= 500:
            return True
        return False

except ImportError:  # shouldn't happen in practice, but keeps the module importable
    def _is_retryable(exc: Exception) -> bool:  # type: ignore[misc]
        return False

# Patterns that appear in RunResult.error when the agent's inner loop
# catches a transient Anthropic API exception and sets stopped_reason="error".
# We detect these so the outer retry wrapper can re-run the whole case.
_RETRYABLE_AGENT_ERROR_PATTERNS = (
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ServiceUnavailableError",
    "OverloadedError",
)


def _agent_error_is_retryable(trace: EvalTrace) -> bool:
    """True when the agent set stopped_reason='error' due to a transient failure.

    The agent catches all API exceptions internally (cannot be modified) and
    surfaces them via RunResult.error. This function inspects that field so
    the outer retry wrapper can recognise them.
    """
    if trace.stopped_reason != "error":
        return False
    error_str = trace.run_result.get("error") or ""
    return any(pat in error_str for pat in _RETRYABLE_AGENT_ERROR_PATTERNS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_case(case: TestCase, repeat_index: int = 0) -> EvalTrace:
    """Run the agent on *case.input* and return a fully scored EvalTrace.

    The agent is treated as a black box; its source is never modified.
    """
    run_result = run_agent(case.input)

    trace = EvalTrace.from_run_result(
        run_result_dict=run_result.to_dict(),
        case_id=case.id,
        repeat_index=repeat_index,
    )
    trace = score(case, trace)
    return trace


def run_case_with_retry(
    case: TestCase,
    repeat_index: int = 0,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> EvalTrace:
    """Like run_case but retries on transient failures with exponential back-off.

    Retryable: RateLimitError (429), APIConnectionError, APITimeoutError, 5xx.
    Not retryable: AuthenticationError, BadRequestError, assertion failures,
    or any non-API exception.

    On success after retries, trace.retry_count > 0 and trace.retry_errors
    holds one message per failed attempt.

    Raises the last exception if all retries are exhausted.
    """
    retry_errors: list[str] = []

    for attempt in range(max_retries + 1):
        try:
            trace = run_case(case, repeat_index)
        except Exception as exc:
            if attempt < max_retries and _is_retryable(exc):
                delay = base_delay * (2 ** attempt)   # 2s, 4s, 8s, …
                retry_errors.append(
                    f"attempt {attempt + 1}: {type(exc).__name__}: {exc}"
                )
                time.sleep(delay)
                continue
            raise

        # The agent may catch its own API exceptions and return a trace with
        # stopped_reason="error". Detect and retry those the same way.
        if attempt < max_retries and _agent_error_is_retryable(trace):
            delay = base_delay * (2 ** attempt)
            error_msg = trace.run_result.get("error", "unknown agent error")
            retry_errors.append(
                f"attempt {attempt + 1}: agent error: {error_msg}"
            )
            time.sleep(delay)
            continue

        trace.retry_count = len(retry_errors)
        trace.retry_errors = retry_errors
        return trace

    # Should not reach here but satisfy type checker
    raise RuntimeError("run_case_with_retry: exceeded max_retries")


def make_error_trace(case: TestCase, repeat_index: int, error_msg: str) -> EvalTrace:
    """Construct a minimal failed EvalTrace for a run that raised fatally.

    Used so a single task failure does not abort the entire run-all suite.
    The trace is marked case_passed=False with the error in retry_errors.
    """
    synthetic_run_result = {
        "run_id": str(uuid.uuid4()),
        "stopped_reason": "error",
        "final_answer": None,
        "citations": [],
        "messages": [],
        "cost_usd": 0.0,
        "wall_time_ms": 0,
        "total_tokens": {"input": 0, "output": 0},
    }
    trace = EvalTrace.from_run_result(synthetic_run_result, case.id, repeat_index)
    trace.case_passed = False
    trace.retry_errors = [error_msg]
    return trace
