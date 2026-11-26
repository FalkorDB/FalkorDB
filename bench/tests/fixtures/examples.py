"""Example results and nightlies for the tests, and broken variants of them.

`example(name)` returns a valid document: a file in valid/ (beside this file), or a nightly
built from the result files there (only nightly-red.json is kept as a literal).
Each `Case` changes one field of an example and names the one problem
the contract (falkorbench.contract) must report. Standard library only.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from typing import NamedTuple

VALID = Path(__file__).resolve().parent / "valid"


def _file(name: str) -> Any:
    return json.loads((VALID / name).read_text(encoding="utf-8"))


def nightly(status: str, checks: list, skip_reason: str | None = None) -> dict:
    """A nightly with nightly-red.json's header (date, images, run link) and these checks."""
    night = _file("nightly-red.json")
    night.update(status=status, skip_reason=skip_reason, checks=copy.deepcopy(checks))
    return night


def _stub(check: str, error: str) -> dict:
    return {"check": check, "verdict": "error", "error": error}


def _nightlies() -> dict:
    passed, error = _file("result-micro-pass.json"), _file("result-pokec-error.json")
    soak = _file("result-soak-both-broken.json")  # pass, one invariant broken on both sides
    return {
        "nightly-green.json": nightly("green", [passed, soak]),
        "nightly-error.json": nightly("error", [passed, error, soak]),
        "nightly-missing-result.json": nightly("error", [
            passed, soak, _stub("pokec", "no result.json"),
            _stub("micro-hw", "no C4 VM with performance-monitoring-unit=standard came up; no counterless fallback"),
        ]),
        "nightly-skipped.json": nightly("skipped", [], "main has not moved since the last run (main@4c1088949)"),
    }


BUILT = tuple(_nightlies())
NAMES = tuple(sorted({p.name for p in VALID.glob("*.json")} | set(BUILT)))


def example(name: str) -> Any:
    return _nightlies()[name] if name in BUILT else _file(name)


def kind(name: str) -> str:
    return "nightly" if name.startswith("nightly-") else "result"


class Delete:
    """Remove the field instead of changing it."""


DELETE = Delete()


class Case(NamedTuple):
    name: str
    start: str  # an example name
    path: tuple[str | int, ...]  # the field to change
    value: Any  # its new value, or DELETE
    code: str  # the problem the contract reports
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


def changed(case: Case) -> Any:
    """The case's start example with its one change applied."""
    document = example(case.start)
    target = document
    for key in case.path[:-1]:
        target = target[key]
    if isinstance(case.value, Delete):
        del target[case.path[-1]]
    else:
        target[case.path[-1]] = copy.deepcopy(case.value)
    return document
