"""Load and validate test cases from YAML or JSON files.

Convention:
  - Files live in tests/cases/
  - Files whose name starts with '_' are skipped (drafts, placeholders, disabled)
  - Every file must supply: id, description, input
  - hard_assertions, tags, repeats are optional
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from eval.case_schema import HardAssertion, TestCase

CASES_DIR = Path(__file__).parent.parent / "tests" / "cases"

_REQUIRED_FIELDS = {"id", "description", "input"}


def _parse_assertion(raw: dict[str, Any]) -> HardAssertion:
    raw = dict(raw)
    try:
        assertion_type = raw.pop("type")
    except KeyError:
        raise ValueError(f"Assertion is missing required key 'type': {raw}")
    return HardAssertion(type=assertion_type, params=raw)


def _parse_case(data: dict[str, Any], source: Path) -> TestCase:
    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"{source}: missing required field(s): {sorted(missing)}")

    assertions = [_parse_assertion(a) for a in data.get("hard_assertions", [])]

    rubric_raw = data.get("rubric")
    rubric = str(rubric_raw) if rubric_raw is not None else None

    return TestCase(
        id=str(data["id"]),
        description=str(data["description"]),
        input=str(data["input"]),
        hard_assertions=assertions,
        tags=list(data.get("tags", [])),
        repeats=int(data.get("repeats", 1)),
        rubric=rubric,
    )


def load_case(path: Path) -> TestCase:
    with path.open(encoding="utf-8") as fh:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(fh)
        elif path.suffix == ".json":
            data = json.load(fh)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")
    return _parse_case(data, path)


def load_all_cases(directory: Path = CASES_DIR) -> list[TestCase]:
    """Return all non-skipped cases from *directory*, sorted by filename."""
    exts = {".yaml", ".yml", ".json"}
    paths = sorted(
        p for p in directory.iterdir()
        if p.suffix in exts and not p.name.startswith("_")
    )
    return [load_case(p) for p in paths]


def load_case_by_id(case_id: str, directory: Path = CASES_DIR) -> TestCase:
    for case in load_all_cases(directory):
        if case.id == case_id:
            return case
    raise ValueError(
        f"No test case with id={case_id!r} found in {directory}. "
        f"Available IDs: {[c.id for c in load_all_cases(directory)]}"
    )
