"""Broken results and nightlies, each built from a valid fixture by one change.

Every case starts from a file in fixtures/valid/, changes one field, and names
the single problem validate.py must report for it. test_validate.py runs every
case through validate.py; the reporter's tests build the nightly cases the same
way. Standard library only.

    for case in CASES:
        text = invalid_text(case)      # the broken file's JSON
        validate.validate_text(text)   # -> [Problem(case.code, case.at, ...)]
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, NamedTuple, Tuple, Union

VALID = Path(__file__).resolve().parent / "fixtures" / "valid"


class Delete:
    """Remove the field instead of changing it."""


DELETE = Delete()


class Case(NamedTuple):
    name: str
    start: str  # a file in fixtures/valid/
    path: Tuple[Union[str, int], ...]  # the field to change
    value: Any  # its new value, or DELETE
    code: str  # the problem validate.py reports
    at: str  # where it reports it


MICRO_PASS = "result-micro-pass.json"
MICRO_REGRESSION = "result-micro-regression.json"
POKEC_ERROR = "result-pokec-error.json"
GREEN = "nightly-green.json"
RED = "nightly-red.json"
MISSING = "nightly-missing-result.json"
SKIPPED = "nightly-skipped.json"

CASES = [
    # result.json
    Case("image tag instead of digest", MICRO_PASS, ("baseline", "image"), "falkordb/falkordb-server:6.2.0", "image-not-digest", "$.baseline.image"),
    Case("commit not 40 hex", MICRO_PASS, ("candidate", "commit"), "4c1088949", "bad-commit", "$.candidate.commit"),
    Case("regression not confirmed", MICRO_REGRESSION, ("confirmed",), False, "regression-unconfirmed", "$.confirmed"),
    Case("regression with no regression finding", MICRO_REGRESSION, ("findings",), [], "regression-without-finding", "$.findings"),
    Case("pass carrying a regression finding", MICRO_PASS, ("findings", 0, "status"), "regression", "pass-with-regression", "$.findings[0].status"),
    Case("error verdict without a reason", POKEC_ERROR, ("error",), "", "error-without-message", "$.error"),
    Case("error text on a pass", MICRO_PASS, ("error",), "callgrind exited 1 on one row; retried", "error-on-non-error", "$.error"),
    Case("metric the contract does not list", MICRO_PASS, ("findings", 0, "metric"), "latency_ms", "unknown-metric", "$.findings[0].metric"),
    Case("required field missing", MICRO_PASS, ("machine",), DELETE, "missing-field", "$.machine"),
    Case("Infinity ratio (a 0 baseline divided)", MICRO_PASS, ("findings", 0, "ratio"), float("inf"), "non-finite-number", "$.findings[0].ratio"),
    # nightly.json
    Case("green while a check regressed", RED, ("status",), "green", "nightly-status-mismatch", "$.status"),
    Case("green while a check is missing", MISSING, ("status",), "green", "nightly-status-mismatch", "$.status"),
    Case("no checks on a night that ran", GREEN, ("checks",), [], "empty-checks", "$.checks"),
    Case("stub claiming pass", MISSING, ("checks", 2, "verdict"), "pass", "stub-not-error", "$.checks[2].verdict"),
    Case("two entries for one check", GREEN, ("checks", 1, "check"), "micro", "duplicate-check", "$.checks[1].check"),
    Case("skipped without a reason", SKIPPED, ("skip_reason",), None, "skip-without-reason", "$.skip_reason"),
    Case("tag inside an embedded result", GREEN, ("checks", 0, "candidate", "image"), "ghcr.io/falkordb/falkordb-server:edge", "image-not-digest", "$.checks[0].candidate.image"),
]


def kind(case: Case) -> str:
    return "nightly" if case.start.startswith("nightly-") else "result"


def invalid_document(case: Case) -> Any:
    """The start fixture with the case's one change applied."""
    document = json.loads((VALID / case.start).read_text(encoding="utf-8"))
    target = document
    for key in case.path[:-1]:
        target = target[key]
    if isinstance(case.value, Delete):
        del target[case.path[-1]]
    else:
        target[case.path[-1]] = copy.deepcopy(case.value)
    return document


def invalid_text(case: Case) -> str:
    """The broken file's JSON text (an infinite float is written as Infinity)."""
    return json.dumps(invalid_document(case), indent=2)
