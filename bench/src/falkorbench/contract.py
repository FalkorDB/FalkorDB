"""The result contract: the format of a perf check's result.json and of the
night's nightly.json, and the checker that runs on them before anything is
reported (bench/README.md, "Result contract").

    bench validate [--kind result|nightly] <file>...

Code that writes or reads these files imports it:
validate_file(path, kind) -> (kind, [Problem(code, path, message)]).
Standard library only, so the reporting path never depends on a package.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from typing import NamedTuple

VERDICTS = ("pass", "regression", "error")
METRICS = (
    "instructions", "cycles", "branch_misses", "cache_misses", "l2_misses", "alloc_bytes",
    "p50_ms", "p95_ms", "p99_ms", "throughput", "ping_max_ms", "rows", "growth_ratio",
    "completed", "invariant",
)
FINDING_STATUSES = ("regression", "improvement", "ok", "info")
DIRECTIONS = ("higher_is_worse", "lower_is_worse", "mismatch")
NIGHTLY_STATUSES = ("green", "red", "error", "skipped")
MAX_FINDINGS = 50

RESULT_FIELDS = ("schema", "check", "verdict", "confirmed", "summary", "candidate", "baseline",
                 "machine", "started", "finished", "findings", "error", "artifacts")
SIDE_FIELDS = ("image", "commit", "label")
MACHINE_FIELDS = ("runner", "type", "cpu", "cores")
FINDING_FIELDS = ("name", "metric", "candidate", "baseline", "ratio", "threshold", "direction", "status")
NIGHTLY_FIELDS = ("schema", "date", "run_url", "candidate", "baseline", "previous_green_commit",
                  "status", "skip_reason", "checks")
STUB_FIELDS = ("check", "verdict", "error")

# Character sets are spelled out: Python's \d and \s also match non-ASCII.
_PATH_PART = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_HOST_PART = r"(?:[A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9-]*[A-Za-z0-9])"
# Docker's image reference with the digest required: [registry/]path[:tag]@sha256:<64 hex>.
IMAGE_RE = re.compile(
    rf"(?:(?:localhost|{_HOST_PART}(?:\.{_HOST_PART})+)(?::[0-9]+)?/|{_HOST_PART}:[0-9]+/)?"
    rf"{_PATH_PART}(?:/{_PATH_PART})*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{{0,127}})?@sha256:[0-9a-f]{{64}}"
)
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?Z")
DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
URL_RE = re.compile(r"https://[!-~]+")
# Whitespace in Python's sense or JavaScript's: a string of only these is blank.
NON_BLANK_RE = re.compile("[^\t\n\x0b\x0c\r\x1c-\x1f \x85\xa0  -     　﻿]")
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

CODES = {
    "unreadable": "the file could not be read",
    "invalid-json": "the file is not JSON",
    "unknown-kind": "neither a result nor a nightly",
    "non-finite-number": "NaN or Infinity, which JSON does not allow",
    "missing-field": "a required field is absent",
    "wrong-type": "a field holds the wrong JSON type",
    "unsupported-schema": "schema is not 1",
    "empty-string": "a name, label or reason that is blank",
    "unknown-verdict": "verdict is not pass, regression or error",
    "unknown-metric": "a metric the contract does not list",
    "unknown-direction": "a direction the contract does not list",
    "unknown-status": "a status the contract does not list",
    "image-not-digest": "an image not pinned as name[:tag]@sha256:<64 hex>",
    "bad-commit": "a commit that is not 40 lowercase hex characters",
    "bad-timestamp": "not YYYY-MM-DDTHH:MM:SS[.fff]Z",
    "bad-date": "not YYYY-MM-DD",
    "bad-url": "not an https:// URL",
    "too-many-findings": f"more than {MAX_FINDINGS} findings",
    "regression-unconfirmed": "a regression verdict without confirmed: true",
    "regression-without-finding": "a regression verdict with no regression finding",
    "pass-with-regression": "a pass verdict with a regression finding",
    "error-with-regression": "an error verdict with a regression finding (regression outranks error)",
    "error-without-message": "an error verdict or stub without a reason",
    "error-on-non-error": "error text on a verdict that is not error",
    "correctness-not-regression": "a run that did not finish, a broken invariant or a row mismatch "
    "that is not marked regression",
    "both-broken-not-info": "completed/invariant 0 on both sides that is not marked info",
    "stub-not-error": "a check without a result that does not say error",
    "duplicate-check": "two entries for one check",
    "empty-checks": "no checks on a night that was not skipped",
    "skipped-with-checks": "a skipped night that lists checks",
    "skip-without-reason": "a skipped night without a reason",
    "nightly-status-mismatch": "status is not the one its checks imply",
}


class Problem(NamedTuple):
    code: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.path}: {self.message}"


class _NonFinite:
    """Stands in for NaN/Infinity while parsing, so the check can say where it was."""

    def __init__(self, text: str) -> None:
        self.text = text


def _parse_float(text: str) -> Any:
    value = float(text)
    return value if math.isfinite(value) else _NonFinite(text)


def load_json(text: str) -> Any:
    """json.loads, except that NaN, Infinity and overflowing numbers become
    reportable instead of passing as floats."""
    return json.loads(text, parse_constant=_NonFinite, parse_float=_parse_float)


def _non_finite(value: Any) -> bool:
    return isinstance(value, _NonFinite) or (isinstance(value, float) and not math.isfinite(value))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not _non_finite(value)


def _is_int(value: Any) -> bool:
    # JSON has one number type: 16.0 is the integer 16. A Python int of any size
    # is an integer as it is; float() of a huge one would raise OverflowError.
    if isinstance(value, bool):
        return False
    return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value) and value.is_integer())


def _child(path: str, key: Any) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if IDENTIFIER_RE.fullmatch(key) else f"{path}[{json.dumps(key)}]"


def _type_name(value: Any) -> str:
    for kind, test in (("null", lambda v: v is None), ("boolean", lambda v: isinstance(v, bool)),
                       ("number", _is_number), ("string", lambda v: isinstance(v, str)),
                       ("array", lambda v: isinstance(v, list)), ("object", lambda v: isinstance(v, dict))):
        if test(value):
            return kind
    return type(value).__name__


def _blank(text: str) -> bool:
    return not NON_BLANK_RE.search(text)


class _Checker:
    def __init__(self) -> None:
        self.problems: list[Problem] = []

    def add(self, code: str, path: str, message: str) -> None:
        assert code in CODES, code
        self.problems.append(Problem(code, path, message))

    # -- building blocks ---------------------------------------------------

    def scan(self, value: Any, path: str) -> None:
        """Report NaN/Infinity anywhere. Iterative: deep nesting cannot exhaust the stack."""
        pending = [(value, path)]
        while pending:
            value, path = pending.pop()
            if _non_finite(value):
                text = value.text if isinstance(value, _NonFinite) else repr(value)
                self.add("non-finite-number", path, f"{text} is not a JSON number")
            elif isinstance(value, dict):
                pending.extend(reversed([(v, _child(path, k)) for k, v in value.items()]))
            elif isinstance(value, list):
                pending.extend(reversed([(v, _child(path, i)) for i, v in enumerate(value)]))

    def type_ok(self, value: Any, path: str, expected: str, ok: bool) -> bool:
        if not ok and not _non_finite(value):  # NaN is reported once, by scan()
            self.add("wrong-type", path, f"expected {expected}, got {_type_name(value)}")
        return ok

    def fields(self, obj: Any, path: str, names: tuple, checks: dict) -> bool:
        """Require each of `names` and check it; other keys are ignored."""
        if not self.type_ok(obj, path, "object", isinstance(obj, dict)):
            return False
        for name in names:
            where = _child(path, name)
            if name not in obj:
                self.add("missing-field", where, "required field is absent")
            elif name in checks:
                checks[name](obj[name], where)
        return True

    def text(self, value: Any, path: str) -> bool:
        if not self.type_ok(value, path, "string", isinstance(value, str)):
            return False
        if _blank(value):
            self.add("empty-string", path, "must not be blank")
            return False
        return True

    def enum(self, code: str, allowed: tuple) -> Callable[[Any, str], None]:
        def check(value: Any, path: str) -> None:
            if self.type_ok(value, path, "string", isinstance(value, str)) and value not in allowed:
                self.add(code, path, f"{json.dumps(value)} is not one of {', '.join(allowed)}")
        return check

    def pattern(self, code: str, regex: re.Pattern, what: str, nullable: bool = False) -> Callable[[Any, str], None]:
        def check(value: Any, path: str) -> None:
            if nullable and value is None:
                return
            expected = "string or null" if nullable else "string"
            if self.type_ok(value, path, expected, isinstance(value, str)) and not regex.fullmatch(value):
                self.add(code, path, f"{json.dumps(value)} is not {what}")
        return check

    def number_or_null(self, value: Any, path: str) -> None:
        self.type_ok(value, path, "number or null", value is None or _is_number(value))

    def string_or_null(self, value: Any, path: str) -> None:
        self.type_ok(value, path, "string or null", value is None or isinstance(value, str))

    def schema_version(self, value: Any, path: str) -> None:
        if self.type_ok(value, path, "integer", _is_int(value)) and value != 1:
            self.add("unsupported-schema", path, f"schema {value} is not 1")

    # -- result.json ---------------------------------------------------------

    def side(self, value: Any, path: str) -> None:
        self.fields(value, path, SIDE_FIELDS, {
            "image": self.pattern("image-not-digest", IMAGE_RE, "pinned as name[:tag]@sha256:<64 lowercase hex>"),
            "commit": self.pattern("bad-commit", COMMIT_RE, "40 lowercase hex characters", nullable=True),
            "label": self.text,
        })

    def machine(self, value: Any, path: str) -> None:
        self.fields(value, path, MACHINE_FIELDS, {
            "runner": self.text, "type": self.text, "cpu": self.text,
            "cores": lambda v, p: self.type_ok(v, p, "integer", _is_int(v)),
        })

    def finding(self, value: Any, path: str) -> None:
        is_object = self.fields(value, path, FINDING_FIELDS, {
            "name": self.text,
            "metric": self.enum("unknown-metric", METRICS),
            "candidate": self.number_or_null, "baseline": self.number_or_null,
            "ratio": self.number_or_null, "threshold": self.number_or_null,
            "direction": self.enum("unknown-direction", DIRECTIONS),
            "status": self.enum("unknown-status", FINDING_STATUSES),
        })
        if not is_object or value.get("status") not in FINDING_STATUSES:
            return
        metric, status = value.get("metric"), value["status"]
        candidate, baseline = value.get("candidate"), value.get("baseline")
        where = _child(path, "status")
        # A run that did not finish or a broken invariant (1 held, 0 not) is a
        # regression at once, unless the baseline broke too: then it is info.
        if metric in ("completed", "invariant") and _is_number(candidate) and candidate == 0:
            if _is_number(baseline) and baseline == 0:
                if status != "info":
                    self.add("both-broken-not-info", where, f"{metric} is 0 on both sides: info, not {status}")
            elif status != "regression":
                self.add("correctness-not-regression", where,
                         f"{metric} 0 on the candidate only is a regression, not {status}")
        # A row count that differs (one side null counts as different) is a regression.
        if (metric == "rows" and value.get("direction") == "mismatch" and status != "regression"
                and all(v is None or _is_number(v) for v in (candidate, baseline)) and candidate != baseline):
            self.add("correctness-not-regression", where,
                     f"{candidate} vs {baseline} rows is a regression, not {status}")

    def findings(self, value: Any, path: str) -> None:
        if not self.type_ok(value, path, "array", isinstance(value, list)):
            return
        if len(value) > MAX_FINDINGS:
            self.add("too-many-findings", path, f"{len(value)} findings; at most {MAX_FINDINGS}")
        for index, item in enumerate(value):
            self.finding(item, _child(path, index))

    def artifacts(self, value: Any, path: str) -> None:
        if self.type_ok(value, path, "array", isinstance(value, list)):
            for index, item in enumerate(value):
                self.text(item, _child(path, index))

    def result(self, value: Any, path: str) -> None:
        timestamp = self.pattern("bad-timestamp", TIMESTAMP_RE, "YYYY-MM-DDTHH:MM:SS[.fff]Z")
        if not self.fields(value, path, RESULT_FIELDS, {
            "schema": self.schema_version, "check": self.text,
            "verdict": self.enum("unknown-verdict", VERDICTS),
            "confirmed": lambda v, p: self.type_ok(v, p, "boolean", isinstance(v, bool)),
            "summary": self.text, "candidate": self.side, "baseline": self.side,
            "machine": self.machine, "started": timestamp, "finished": timestamp,
            "findings": self.findings, "error": self.string_or_null, "artifacts": self.artifacts,
        }):
            return
        verdict = value.get("verdict")
        if verdict not in VERDICTS:
            return
        findings = value.get("findings") if isinstance(value.get("findings"), list) else []
        regressions = [i for i, f in enumerate(findings) if isinstance(f, dict) and f.get("status") == "regression"]
        error = value.get("error")
        if "incomplete" in value:  # optional: only its type is checked
            self.string_or_null(value["incomplete"], _child(path, "incomplete"))
        if verdict == "regression":
            if "confirmed" in value and value["confirmed"] is not True:
                self.add("regression-unconfirmed", _child(path, "confirmed"),
                         "a regression needs confirmed: true (a re-run repeated it, or it is a "
                         "correctness failure, crash or run that did not finish)")
            if "findings" in value and isinstance(value["findings"], list) and not regressions:
                self.add("regression-without-finding", _child(path, "findings"),
                         "a regression verdict needs a finding with status regression")
        else:
            for index in regressions:
                self.add(f"{verdict}-with-regression", _child(_child(_child(path, "findings"), index), "status"),
                         f"a {verdict} verdict cannot carry a regression finding")
        if verdict == "error":
            if "error" in value and (error is None or (isinstance(error, str) and _blank(error))):
                self.add("error-without-message", _child(path, "error"), "an error verdict needs a reason")
        elif error is not None:
            self.add("error-on-non-error", _child(path, "error"), f"error must be null when verdict is {verdict}")

    # -- nightly.json --------------------------------------------------------

    def stub(self, value: dict, path: str) -> None:
        def verdict(v: Any, p: str) -> None:
            self.enum("unknown-verdict", VERDICTS)(v, p)
            if v in VERDICTS and v != "error":
                self.add("stub-not-error", p, f"a check without a result can only say error, not {v}")

        def error(v: Any, p: str) -> None:
            if v is None or (isinstance(v, str) and _blank(v)):
                self.add("error-without-message", p, "a stub needs a reason")
            else:
                self.type_ok(v, p, "string", isinstance(v, str))

        self.fields(value, path, STUB_FIELDS, {"check": self.text, "verdict": verdict, "error": error})

    def checks(self, value: Any, path: str) -> None:
        if not self.type_ok(value, path, "array", isinstance(value, list)):
            return
        seen = set()
        for index, entry in enumerate(value):
            where = _child(path, index)
            # An entry with "schema" is a full result; otherwise the merge step's stub.
            if isinstance(entry, dict) and "schema" not in entry:
                self.stub(entry, where)
            else:
                self.result(entry, where)
            name = entry.get("check") if isinstance(entry, dict) else None
            if isinstance(name, str):
                if name in seen:
                    self.add("duplicate-check", _child(where, "check"), f"check {json.dumps(name)} appears twice")
                seen.add(name)

    def nightly(self, value: Any, path: str) -> None:
        if not self.fields(value, path, NIGHTLY_FIELDS, {
            "schema": self.schema_version,
            "date": self.pattern("bad-date", DATE_RE, "YYYY-MM-DD"),
            "run_url": self.pattern("bad-url", URL_RE, "an https:// URL"),
            "candidate": self.side, "baseline": self.side,
            "previous_green_commit": self.pattern("bad-commit", COMMIT_RE, "40 lowercase hex characters", nullable=True),
            "status": self.enum("unknown-status", NIGHTLY_STATUSES),
            "skip_reason": self.string_or_null, "checks": self.checks,
        }):
            return
        status, checks, reason = value.get("status"), value.get("checks"), value.get("skip_reason")
        if status not in NIGHTLY_STATUSES or not isinstance(checks, list):
            return
        if status == "skipped":
            if "skip_reason" in value and (reason is None or (isinstance(reason, str) and _blank(reason))):
                self.add("skip-without-reason", _child(path, "skip_reason"), "a skipped night needs a reason")
            if checks:
                self.add("skipped-with-checks", _child(path, "checks"), f"a skipped night lists {len(checks)} check(s)")
            return
        if not checks:
            self.add("empty-checks", _child(path, "checks"),
                     "a night that ran lists every planned check (a missing one as an error stub)")
            return
        verdicts = [entry.get("verdict") if isinstance(entry, dict) else None for entry in checks]
        if any(v not in VERDICTS for v in verdicts):
            return
        derived = "red" if "regression" in verdicts else "error" if "error" in verdicts else "green"
        if status != derived:
            self.add("nightly-status-mismatch", _child(path, "status"),
                     f"status is {status} but its checks make it {derived}")


def detect_kind(document: Any) -> str | None:
    if not isinstance(document, dict):
        return None
    if "checks" in document:
        return "nightly"
    if "check" in document:
        return "result"
    return None


def validate_result(document: Any) -> list[Problem]:
    checker = _Checker()
    checker.scan(document, "$")
    checker.result(document, "$")
    return checker.problems


def validate_nightly(document: Any) -> list[Problem]:
    checker = _Checker()
    checker.scan(document, "$")
    checker.nightly(document, "$")
    return checker.problems


def validate_document(document: Any, kind: str | None = None) -> tuple[str | None, list[Problem]]:
    kind = kind or detect_kind(document)
    if kind == "result":
        return kind, validate_result(document)
    if kind == "nightly":
        return kind, validate_nightly(document)
    return None, [Problem("unknown-kind", "$", 'neither a result (has "check") nor a nightly (has "checks")')]


def validate_text(text: str, kind: str | None = None) -> tuple[str | None, list[Problem]]:
    try:
        document = load_json(text)
    except ValueError as error:  # json.JSONDecodeError is a ValueError
        return kind, [Problem("invalid-json", "$", str(error))]
    except RecursionError:
        return kind, [Problem("invalid-json", "$", "nested too deeply to parse")]
    return validate_document(document, kind)


def validate_file(path: str | Path, kind: str | None = None) -> tuple[str | None, list[Problem]]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return kind, [Problem("unreadable", "$", str(error))]
    return validate_text(text, kind)


def check_files(files: Iterable[str | Path], kind: str | None = None,
                echo: Callable[[str], Any] = print) -> int:
    """`bench validate`: one line per problem, `<file>: <code>: <path>: <message>`,
    or `<file>: ok (<kind>)`. Returns 1 if any file has a problem, else 0."""
    failed = False
    for name in files:
        found, problems = validate_file(name, kind)
        failed |= bool(problems)
        for problem in problems:
            echo(f"{name}: {problem}")
        if not problems:
            echo(f"{name}: ok ({found})")
    return 1 if failed else 0
