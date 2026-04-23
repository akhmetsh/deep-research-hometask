"""Populate fixtures/traces/ with one representative EvalTrace per test case.

Run from the project root:
    python fixtures/make_fixtures.py

Selection criteria per case (in priority order):
  1. Most recently saved trace that matches the case's "expected" pass/fail
     (i.e. the characterised behaviour we want to document).
  2. If no match, use the most recently saved trace regardless.

The resulting fixtures are small enough to commit (~20–60 KB each) and
demonstrate: happy-path pass, safety refusal, out-of-corpus refusal,
conflicting-sources, citation integrity, prompt injection, broken page.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.eval_trace import EvalTrace, EVAL_TRACES_DIR

FIXTURES_DIR = Path(__file__).parent / "traces"

# Expected pass/fail for each case (characterised behaviour).
# True  = we want a passing trace as the fixture.
# False = we want a failing trace as the fixture (documents a known bug/failure).
EXPECTED: dict[str, bool] = {
    "tc01_happy_path_voyager":       True,
    "tc02_confidential_refusal":     False,   # known protocol bug
    "tc03_citation_integrity":       True,
    "tc04_prompt_injection_meiosis": True,
    "tc05_broken_page_citation":     False,   # known word-limit failure
    "tc06_out_of_corpus":            False,   # known protocol bug
    "tc07_conflicting_sources":      True,
}


def _pick_trace(case_id: str, want_pass: bool) -> Path | None:
    case_dir = EVAL_TRACES_DIR / case_id
    if not case_dir.exists():
        return None
    candidates = sorted(case_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    # Prefer traces matching want_pass; fall back to newest regardless.
    for p in reversed(candidates):
        try:
            t = EvalTrace.load(p)
            if bool(t.case_passed) == want_pass:
                return p
        except Exception:
            pass
    return candidates[-1]  # newest as fallback


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for case_id, want_pass in sorted(EXPECTED.items()):
        src = _pick_trace(case_id, want_pass)
        if src is None:
            print(f"  SKIP  {case_id}  (no saved traces found — run eval run-all first)")
            continue
        dst = FIXTURES_DIR / f"{case_id}.json"
        shutil.copy2(src, dst)
        t = EvalTrace.load(dst)
        status = "PASS" if t.case_passed else "FAIL"
        kb = dst.stat().st_size // 1024
        print(f"  {status}  {case_id:<42}  {kb:>3} KB  ->  {dst.name}")
        copied += 1
    print(f"\n{copied}/{len(EXPECTED)} fixtures written to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
