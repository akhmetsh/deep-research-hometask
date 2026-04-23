"""Runner: invoke the agent and return a scored EvalTrace.

This is the only module that imports agent.py. Keeping the dependency
isolated here means the rest of the eval framework (loader, scorer,
reporter) can be used without an Anthropic API key — e.g. for rescoring
saved traces or generating HTML reports.

Future extensions:
  - Async parallel runner (asyncio + semaphore) for concurrency cap
  - Retry policy (exponential back-off on 429 / 5xx)
  - --repeats support for flakiness detection
"""

from __future__ import annotations

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
