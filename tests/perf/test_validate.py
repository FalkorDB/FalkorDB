"""Tests for tests/perf/validate.py and tests/perf/acceptance.yaml.

    uv run --no-project --with pytest --with pyyaml python -I -m pytest tests/perf

Every file in fixtures/valid/ must pass. Every case in invalid_cases.py, and in
the table below, changes one field of a valid file and must give exactly its
expected problems. acceptance.yaml must hold exactly the acceptance pairs
pinned here.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
VALID = HERE / "fixtures" / "valid"
VALIDATE_PY = HERE / "validate.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate = _load("perf_validate", VALIDATE_PY)
invalid_cases = _load("perf_invalid_cases", HERE / "invalid_cases.py")
Case, DELETE = invalid_cases.Case, invalid_cases.DELETE

VALID_FILES = sorted(p.name for p in VALID.glob("*.json"))


def _fixture(name: str):
    return json.loads((VALID / name).read_text())


def _problems(case) -> list:
    _, problems = validate.validate_text(invalid_cases.invalid_text(case), invalid_cases.kind(case))
    return [(p.code, p.path) for p in problems]


# -- the valid fixtures ----------------------------------------------------------


@pytest.mark.parametrize("name", VALID_FILES)
def test_valid_fixture_passes(name):
    kind, problems = validate.validate_file(VALID / name)
    assert kind == ("nightly" if name.startswith("nightly-") else "result")
    assert problems == []


def test_valid_fixtures_cover_every_verdict_status_and_correctness_case():
    results = [_fixture(n) for n in VALID_FILES if n.startswith("result-")]
    nights = [_fixture(n) for n in VALID_FILES if n.startswith("nightly-")]
    findings = [f for r in results for f in r["findings"]]
    assert {r["verdict"] for r in results} == {"pass", "regression", "error"}
    assert {n["status"] for n in nights} == {"green", "red", "error", "skipped"}
    assert any(f["metric"] == "completed" and f["status"] == "regression" for f in findings)  # did not finish
    assert any(f["metric"] == "invariant" and f["status"] == "regression" for f in findings)  # broke
    assert any(f["metric"] == "rows" and f["status"] == "regression" for f in findings)  # row mismatch
    assert any(f["candidate"] == 0 and f["baseline"] == 0 and f["status"] == "info" for f in findings)  # both broke
    assert any("schema" not in c for n in nights for c in n["checks"])  # a missing check's stub


# -- one-field breaks ------------------------------------------------------------


@pytest.mark.parametrize("case", invalid_cases.CASES, ids=[c.name for c in invalid_cases.CASES])
def test_invalid_case_gives_exactly_its_problem(case):
    assert _problems(case) == [(case.code, case.at)]


MICRO_PASS = invalid_cases.MICRO_PASS
MICRO_REGRESSION = invalid_cases.MICRO_REGRESSION
POKEC_ERROR = invalid_cases.POKEC_ERROR
DID_NOT_FINISH = "result-pokec-did-not-finish.json"  # findings[0]: completed 0 vs 1
CORRECTNESS = "result-soak-correctness.json"  # [0] invariant 0 vs 1, [1] rows 99412 vs 100000
BOTH_BROKEN = "result-soak-both-broken.json"  # [0] invariant 0 vs 0, info
GREEN, RED, SKIPPED = invalid_cases.GREEN, invalid_cases.RED, invalid_cases.SKIPPED
DIGEST = "a" * 64
HANG = {"name": "3-hop read", "metric": "completed", "candidate": 0, "baseline": 1, "ratio": 0.0,
        "threshold": None, "direction": "lower_is_worse", "status": "improvement"}

# Changes that must stay valid: guards against a checker stricter than the contract.
ACCEPTED = [
    ("tag and digest", MICRO_PASS, ("candidate", "image"), f"falkordb/falkordb-server:6.2.0@sha256:{DIGEST}"),
    ("registry with port", MICRO_PASS, ("candidate", "image"), f"localhost:5000/falkordb/falkordb-server@sha256:{DIGEST}"),
    ("nanosecond timestamp", MICRO_PASS, ("finished",), "2026-11-24T00:31:47.123456789Z"),
    ("null commit", MICRO_PASS, ("candidate", "commit"), None),
    ("null ratio", MICRO_PASS, ("findings", 0, "ratio"), None),
    ("16.0 cores", MICRO_PASS, ("machine", "cores"), 16.0),
    ("401-digit core count (an int, however large)", MICRO_PASS, ("machine", "cores"), 10**400),
    ("extra key ignored", MICRO_PASS, ("notes",), "pinned to cores 0-4"),
    ("hardware-counter metric", MICRO_PASS, ("findings", 1, "metric"), "l2_misses"),
    ("error verdict with ordinary findings", POKEC_ERROR, ("findings",), _fixture(MICRO_PASS)["findings"]),
]

# Breaks beyond invalid_cases.py: one field each, with every problem expected.
REJECTED = [
    ("schema 2", MICRO_PASS, ("schema",), 2, [("unsupported-schema", "$.schema")]),
    # A huge integer is reported, never raised: merge.py and post.py call validate_file.
    ("401-digit schema", MICRO_PASS, ("schema",), 10**400, [("unsupported-schema", "$.schema")]),
    ('confirmed "false" (truthy)', MICRO_PASS, ("confirmed",), "false", [("wrong-type", "$.confirmed")]),
    ("ratio as a string", MICRO_PASS, ("findings", 0, "ratio"), "0.03", [("wrong-type", "$.findings[0].ratio")]),
    ("blank label", MICRO_PASS, ("baseline", "label"), "　", [("empty-string", "$.baseline.label")]),
    ("verdict fail", MICRO_PASS, ("verdict",), "fail", [("unknown-verdict", "$.verdict")]),
    ("direction up", MICRO_PASS, ("findings", 0, "direction"), "up_is_bad", [("unknown-direction", "$.findings[0].direction")]),
    ("status warn", MICRO_PASS, ("findings", 0, "status"), "warn", [("unknown-status", "$.findings[0].status")]),
    ("time with offset", MICRO_PASS, ("started",), "2026-11-24T00:06:12+00:00", [("bad-timestamp", "$.started")]),
    ("51 findings", MICRO_PASS, ("findings",), _fixture(MICRO_PASS)["findings"][:1] * 51, [("too-many-findings", "$.findings")]),
    ("hang written as improvement", MICRO_PASS, ("findings", 0), HANG, [("correctness-not-regression", "$.findings[0].status")]),
    ("did-not-finish marked info", DID_NOT_FINISH, ("findings", 0, "status"), "info", [("correctness-not-regression", "$.findings[0].status")]),
    ("broken invariant marked ok", CORRECTNESS, ("findings", 0, "status"), "ok", [("correctness-not-regression", "$.findings[0].status")]),
    ("row mismatch marked ok", CORRECTNESS, ("findings", 1, "status"), "ok", [("correctness-not-regression", "$.findings[1].status")]),
    ("broken on both sides marked ok", BOTH_BROKEN, ("findings", 0, "status"), "ok", [("both-broken-not-info", "$.findings[0].status")]),
    ("error carrying regressions", POKEC_ERROR, ("findings",), _fixture(DID_NOT_FINISH)["findings"],
     [("error-with-regression", "$.findings[0].status"), ("error-with-regression", "$.findings[1].status")]),
    ("date as 24/11/2026", GREEN, ("date",), "24/11/2026", [("bad-date", "$.date")]),
    ("http run link", GREEN, ("run_url",), "http://github.com/FalkorDB/FalkorDB/actions/runs/1", [("bad-url", "$.run_url")]),
    ("skipped night with checks", SKIPPED, ("checks",), _fixture(GREEN)["checks"], [("skipped-with-checks", "$.checks")]),
    ("red night said error", RED, ("status",), "error", [("nightly-status-mismatch", "$.status")]),
    ("embedded unconfirmed regression", RED, ("checks", 0, "confirmed"), False, [("regression-unconfirmed", "$.checks[0].confirmed")]),
    ("check entry not an object", GREEN, ("checks", 1), "size-probe", [("wrong-type", "$.checks[1]")]),
]


@pytest.mark.parametrize("start, path, value", [r[1:] for r in ACCEPTED], ids=[r[0] for r in ACCEPTED])
def test_accepted_change_stays_valid(start, path, value):
    assert _problems(Case("", start, path, value, "", "")) == []


@pytest.mark.parametrize("start, path, value, expected", [r[1:] for r in REJECTED], ids=[r[0] for r in REJECTED])
def test_rejected_change_gives_exactly_its_problems(start, path, value, expected):
    assert _problems(Case("", start, path, value, "", "")) == expected


# -- whole-file and API cases ------------------------------------------------------

TEXT_CASES = [
    ('{"schema": 1, "check": "micro",}', "invalid-json"),
    ("", "invalid-json"),
    ("[]", "unknown-kind"),
    ('{"schema": 1}', "unknown-kind"),
]


@pytest.mark.parametrize("text, code", TEXT_CASES)
def test_unparseable_or_shapeless_text(text, code):
    _, problems = validate.validate_text(text)
    assert [(p.code, p.path) for p in problems] == [(code, "$")]


@pytest.mark.parametrize("literal", ["NaN", "-Infinity", "1e999"])
def test_nan_and_infinity_in_a_file_are_reported_where_they_are(literal):
    document = _fixture(MICRO_PASS)
    text = json.dumps(document).replace(f'"ratio": {json.dumps(document["findings"][0]["ratio"])}', f'"ratio": {literal}', 1)
    _, problems = validate.validate_text(text)
    assert [(p.code, p.path) for p in problems] == [("non-finite-number", "$.findings[0].ratio")]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_in_a_document_built_in_memory(value):
    # merge.py validates parsed results, where NaN is a float, not the parser's stand-in.
    document = _fixture(GREEN)
    document["checks"][0]["machine"]["cores"] = value
    assert [(p.code, p.path) for p in validate.validate_nightly(document)] == [("non-finite-number", "$.checks[0].machine.cores")]


def test_nesting_deeper_than_python_recursion_is_a_problem_not_a_crash():
    depth = 200_000
    _, problems = validate.validate_text('{"check": "micro", "artifacts": ' + "[" * depth + "]" * depth + "}")
    assert [p.code for p in problems] == ["invalid-json"]
    nested: list = []
    for _ in range(sys.getrecursionlimit() * 5):
        nested = [nested]
    document = dict(_fixture(MICRO_PASS), artifacts=nested)
    assert ("wrong-type", "$.artifacts[0]") in [(p.code, p.path) for p in validate.validate_result(document)]


def test_unreadable_file(tmp_path):
    _, problems = validate.validate_file(tmp_path / "absent.json")
    assert [p.code for p in problems] == ["unreadable"]


def test_every_problem_code_is_produced_by_some_test():
    produced = {c.code for c in invalid_cases.CASES} | {code for _, code in TEXT_CASES}
    produced |= {code for row in REJECTED for code, _ in row[4]} | {"non-finite-number", "unreadable"}
    assert produced == set(validate.CODES)


# -- the command line ---------------------------------------------------------------


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(VALIDATE_PY), *args], capture_output=True, text=True, check=False)


def test_cli_accepts_every_valid_fixture():
    files = [str(VALID / n) for n in VALID_FILES]
    run = _run(*files)
    assert run.returncode == 0, run.stdout + run.stderr
    assert len(run.stdout.splitlines()) == len(files)


def test_cli_prints_every_problem_and_fails(tmp_path):
    bad = tmp_path / "result.json"
    bad.write_text(invalid_cases.invalid_text(invalid_cases.CASES[0]))
    run = _run(str(VALID / MICRO_PASS), str(bad))
    assert run.returncode == 1
    lines = run.stdout.splitlines()
    assert lines[0].endswith(": ok (result)")
    assert re.fullmatch(re.escape(str(bad)) + r": image-not-digest: \$\.baseline\.image: .+", lines[1])
    assert _run().returncode == 2


# -- acceptance.yaml -----------------------------------------------------------------

ACCEPTANCE = yaml.safe_load((HERE / "acceptance.yaml").read_text())
CHECKS = ("micro", "micro-hw", "size-probe", "pokec", "soak-a", "soak-b", "fuzz", "ldbc")
MAIN = "main@4c1088949"
# The pinned images (2026-11-24) and their commits (release tags, and main's merge commit for #3963).
PINNED = {
    "6.0.1": ("falkordb/falkordb-server@sha256:f4f60c62532b67f8651d0f4f50bbb4db7548717b502bd05abd80e668f2d974c6", "32aacf47f23f9df7f7da216f74eeb07f537feb94"),
    "6.0.4": ("falkordb/falkordb-server@sha256:698a6e748b42699c1f63af01f8902652b764582e363731de56c7d566c322c209", "6ca3adc9d0418f76c5a15f820552e0603a65e553"),
    "6.2.0": ("falkordb/falkordb-server@sha256:44eb4a6d48ef6f2b06aeefea46c5f4c17cd2310d3832135be18fd7fc8ba30923", "d8c329b777c177fc7343d959157d6acdf7398fa5"),
    "6.2.1": ("falkordb/falkordb-server@sha256:a8fb264e6b90db5c1e6e3a42af5128efcc13590ce2ca8e65d870b456806f24de", "617955cf660770f33e69fb9765e8b75df03636c1"),
    MAIN: ("ghcr.io/falkordb/falkordb-server@sha256:95b610a600682533d854b6ece6bddbea80b49d0b89e2cd0b2cf19bd0a8142aed", "4c1088949a5852e3949c05bfb3d92a603c4cd4ca"),
    "C v4.22.2": ("falkordb/falkordb-server@sha256:3c4e618ed169893bc8346e4d8e51ed2c270b484c91d93e3da5c82745db9117c2", "7b61f511de44613e4f3d26e629ac0143f6e594b2"),
    "C v4.22.3": ("falkordb/falkordb-server@sha256:ae49d4f04f72f71e0a85b005a3a04f1917f51c527b59d1560ed6074ad3b0e884", "3dbf13e9efec3f340226d3994cba4296e7c3b7bc"),
    "C v4.22.4": ("falkordb/falkordb-server@sha256:4e651dac5dbabf3169d91af06db4be13ee7679ffc392f14bac5fa730349b5dbf", "958fbead7d6e08d50d48d4776f0f310ea1dfb9a4"),
}
AA = {"verdict": "pass", "no_regression_findings": True}
LATENCY = ["p50_ms", "p95_ms", "p99_ms"]
# The rows #3963 added. Three must regress; UNWIND index MATCH edge 100 rows is at
# parity at bench sizes and only reported. micro names a finding "<row> [<class>]",
# so each row is matched by an anchored pattern.
MICRO_REQUIRED_ROWS = ("UNWIND index MATCH edge 1 row", "UNWIND id MATCH edge 1 row", "UNWIND id MATCH edge 100 rows")
MICRO_REPORTED_ROWS = ("UNWIND index MATCH edge 100 rows",)


def _micro_row_pattern(row: str) -> str:
    return f"^{row}( \\[(reads|writes)\\])?$"


# (check, candidate, baseline) -> expectation. Every check has main against itself.
TABLE = {
    **{(check, MAIN, MAIN): AA for check in CHECKS},
    ("micro", "6.2.0", MAIN): {
        "verdict": "regression",
        "findings": [{"name_regex": _micro_row_pattern(row), "status": "regression"} for row in MICRO_REQUIRED_ROWS],
    },
    ("micro-hw", "6.2.0", "6.2.0"): AA,
    ("micro-hw", "6.2.0", MAIN): {
        "verdict": "regression",
        "findings": [{"name_regex": _micro_row_pattern(row), "metric": "cycles", "status": "regression"} for row in MICRO_REQUIRED_ROWS],
    },
    ("micro-hw", "C v4.22.4", "C v4.22.4"): {},
    ("size-probe", "6.2.0", "6.0.4"): {"verdict": "regression", "findings": [{"name_regex": "unwind", "status": "regression"}]},
    ("size-probe", MAIN, "6.2.0"): {"findings": [{"name_regex": "unwind", "status": "improvement"}], "no_regression_findings_matching": "unwind"},
    ("size-probe", "C v4.22.4", "C v4.22.4"): AA,
    ("pokec", "C v4.22.4", "C v4.22.4"): AA,
    ("pokec", "6.0.4", "6.0.1"): {"verdict": "regression", "findings": [{"name_regex": "3-hop", "status": "regression"}]},
    ("soak-b", "6.2.0", "6.2.1"): {"verdict": "regression", "findings": [{"name_regex": "s12", "status": "regression"}]},
    ("soak-b", "6.2.0", "6.0.4"): {"findings": [{"name_regex": "s12", "status": "improvement"}], "no_regression_findings_matching": "s12"},
    ("fuzz", "C v4.22.2", "C v4.22.2"): AA,
    ("fuzz", "6.2.0", "6.0.4"): {"verdict": "regression", "findings": [{"name_regex": "build", "status": "regression"}]},
    ("fuzz", "C v4.22.3", "C v4.22.2"): {"verdict": "regression", "findings": [{"name_regex": "control", "status": "regression"}]},
    ("ldbc", "C v4.22.4", "C v4.22.4"): AA,
    ("ldbc", "6.0.4", "6.2.0"): {
        "verdict": "regression",
        "findings": [
            *({"name_regex": rf"\b{ic}\b", "metric": LATENCY, "status": "regression"} for ic in ("ic1", "ic2")),
            {"name_regex": r"\bic12\b", "status": "regression"},
        ],
    },
}


def _pairs():
    for check, pairs in ACCEPTANCE["checks"].items():
        for pair in pairs:
            yield check, pair


def test_acceptance_pairs_are_exactly_the_table():
    assert ACCEPTANCE["schema"] == 1
    assert tuple(ACCEPTANCE["checks"]) == CHECKS
    found = {}
    for check, pair in _pairs():
        key = (check, pair["candidate"]["label"], pair["baseline"]["label"])
        assert key not in found, f"two pairs for {key}"
        found[key] = pair["expect"]
    assert found == TABLE


def test_acceptance_pairs_are_well_formed():
    ids = [pair["id"] for _, pair in _pairs()]
    assert len(ids) == len(set(ids)), "duplicate pair id"
    for _, pair in _pairs():
        assert set(pair) <= {"id", "candidate", "baseline", "mode", "expect", "judge"}, pair["id"]
        assert pair["mode"] == "nightly"
        for side in (pair["candidate"], pair["baseline"]):
            assert (side["image"], side["commit"]) == PINNED[side["label"]], side["label"]
            assert validate.IMAGE_RE.fullmatch(side["image"])
        for matcher in pair["expect"].get("findings", []):
            assert set(matcher) <= {"name", "name_regex", "metric", "status"}, pair["id"]
            metrics = matcher.get("metric", [])
            assert set(metrics if isinstance(metrics, list) else [metrics]) <= set(validate.METRICS)
            re.compile(matcher.get("name_regex", ""))


def test_micro_acceptance_rows_exist_in_bench_and_the_reported_one_is_named():
    queries = (HERE.parent.parent / "bench" / "src" / "falkorbench" / "queries.py").read_text()
    for row in MICRO_REQUIRED_ROWS + MICRO_REPORTED_ROWS:
        assert f'Q("{row}"' in queries, f"bench has no row named {row!r}"
    pair = next(p for p in ACCEPTANCE["checks"]["micro"] if p["id"] == "micro-6.2.0-vs-main")
    for row in MICRO_REPORTED_ROWS:
        assert row in " ".join(pair["judge"].split())


@pytest.mark.parametrize("name, matches", [
    ("UNWIND id MATCH edge 1 row", True),
    ("UNWIND id MATCH edge 1 row [writes]", True),
    ("unwind id match edge 1 row [Writes]", True),
    ("UNWIND id MATCH edge 100 rows [writes]", False),
    ("UNWIND id MATCH edge 1 row [writes] (re-run)", False),
])
def test_micro_row_pattern_matches_only_its_row(name, matches):
    assert bool(re.search(_micro_row_pattern("UNWIND id MATCH edge 1 row"), name, re.IGNORECASE)) == matches


@pytest.mark.parametrize("name, matches", [("IC1 p95", True), ("ic1", True), ("IC12 p95", False), ("IC10", False)])
def test_ldbc_ic1_pattern_matches_only_ic1(name, matches):
    assert bool(re.search(r"\bic1\b", name, re.IGNORECASE)) == matches
