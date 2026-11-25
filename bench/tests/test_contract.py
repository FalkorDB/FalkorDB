"""Tests for falkorbench.contract and bench/acceptance.yaml (bench/README.md,
"Result contract", says what they define)."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from falkorbench import contract
from falkorbench.cli import cli
from falkorbench.queries import QUERIES
from fixtures import examples

BENCH = Path(__file__).resolve().parents[1]
Case, DELETE, example = examples.Case, examples.DELETE, examples.example


def _problems(case) -> list:
    # Through the strict parser, as a file would go (NaN/Infinity included).
    _, problems = contract.validate_text(json.dumps(examples.changed(case)), examples.kind(case.start))
    return [(p.code, p.path) for p in problems]


def _unchanged(name: str) -> Case:
    return Case(name, name, (), None, "", "")


# -- the valid examples ------------------------------------------------------------


@pytest.mark.parametrize("name", examples.NAMES)
def test_valid_example_passes(name):
    _, problems = contract.validate_text(json.dumps(example(name)), examples.kind(name))
    assert problems == []


def test_valid_examples_cover_every_verdict_status_and_correctness_case():
    results = [example(n) for n in examples.NAMES if n.startswith("result-")]
    nights = [example(n) for n in examples.NAMES if n.startswith("nightly-")]
    findings = [f for r in results for f in r["findings"]]
    assert {r["verdict"] for r in results} == {"pass", "regression", "error"}
    assert {n["status"] for n in nights} == {"green", "red", "error", "skipped"}
    assert any(f["metric"] == "completed" and f["status"] == "regression" for f in findings)  # did not finish
    assert any(f["metric"] == "invariant" and f["status"] == "regression" for f in findings)  # broke
    assert any(f["metric"] == "rows" and f["status"] == "regression" for f in findings)  # row mismatch
    assert any(f["candidate"] == 0 and f["baseline"] == 0 and f["status"] == "info" for f in findings)  # both broke
    assert any("schema" not in c for n in nights for c in n["checks"])  # a missing check's stub


# -- one-field changes --------------------------------------------------------------

MICRO_PASS, MICRO_REGRESSION, POKEC_ERROR = examples.MICRO_PASS, examples.MICRO_REGRESSION, examples.POKEC_ERROR
GREEN, RED, SKIPPED = examples.GREEN, examples.RED, examples.SKIPPED
DID_NOT_FINISH = "result-pokec-did-not-finish.json"  # findings[0]: completed 0 vs 1
CORRECTNESS = "result-soak-correctness.json"  # [0] invariant 0 vs 1, [1] rows 99412 vs 100000
BOTH_BROKEN = "result-soak-both-broken.json"  # [0] invariant 0 vs 0, info
DIGEST = "a" * 64
HANG = {"name": "3-hop read", "metric": "completed", "candidate": 0, "baseline": 1, "ratio": 0.0,
        "threshold": None, "direction": "lower_is_worse", "status": "improvement"}

# (name, start, path, value, expected problems): [] means the change must stay valid.
CHANGES = [(c.name, c.start, c.path, c.value, [(c.code, c.at)]) for c in examples.CASES] + [
    # Must stay valid: guards against a checker stricter than the contract.
    ("tag and digest", MICRO_PASS, ("candidate", "image"), f"falkordb/falkordb-server:6.2.0@sha256:{DIGEST}", []),
    ("registry with port", MICRO_PASS, ("candidate", "image"), f"localhost:5000/falkordb/falkordb-server@sha256:{DIGEST}", []),
    ("nanosecond timestamp", MICRO_PASS, ("finished",), "2026-11-24T00:31:47.123456789Z", []),
    ("null commit", MICRO_PASS, ("candidate", "commit"), None, []),
    ("null ratio", MICRO_PASS, ("findings", 0, "ratio"), None, []),
    ("16.0 cores", MICRO_PASS, ("machine", "cores"), 16.0, []),
    ("401-digit core count (an int, however large)", MICRO_PASS, ("machine", "cores"), 10**400, []),
    ("extra key ignored", MICRO_PASS, ("notes",), "pinned to cores 0-4", []),
    ("hardware-counter metric", MICRO_PASS, ("findings", 1, "metric"), "l2_misses", []),
    ("error verdict with ordinary findings", POKEC_ERROR, ("findings",), example(MICRO_PASS)["findings"], []),
    # Must be rejected, with exactly these problems.
    ("schema 2", MICRO_PASS, ("schema",), 2, [("unsupported-schema", "$.schema")]),
    # A huge integer is reported, never raised: merge.py calls validate_file.
    ("401-digit schema", MICRO_PASS, ("schema",), 10**400, [("unsupported-schema", "$.schema")]),
    ('confirmed "false" (truthy)', MICRO_PASS, ("confirmed",), "false", [("wrong-type", "$.confirmed")]),
    ("ratio as a string", MICRO_PASS, ("findings", 0, "ratio"), "0.03", [("wrong-type", "$.findings[0].ratio")]),
    ("blank label", MICRO_PASS, ("baseline", "label"), "　", [("empty-string", "$.baseline.label")]),
    ("verdict fail", MICRO_PASS, ("verdict",), "fail", [("unknown-verdict", "$.verdict")]),
    ("direction up", MICRO_PASS, ("findings", 0, "direction"), "up_is_bad", [("unknown-direction", "$.findings[0].direction")]),
    ("status warn", MICRO_PASS, ("findings", 0, "status"), "warn", [("unknown-status", "$.findings[0].status")]),
    ("time with offset", MICRO_PASS, ("started",), "2026-11-24T00:06:12+00:00", [("bad-timestamp", "$.started")]),
    ("51 findings", MICRO_PASS, ("findings",), example(MICRO_PASS)["findings"][:1] * 51, [("too-many-findings", "$.findings")]),
    ("hang written as improvement", MICRO_PASS, ("findings", 0), HANG, [("correctness-not-regression", "$.findings[0].status")]),
    ("did-not-finish marked info", DID_NOT_FINISH, ("findings", 0, "status"), "info", [("correctness-not-regression", "$.findings[0].status")]),
    ("broken invariant marked ok", CORRECTNESS, ("findings", 0, "status"), "ok", [("correctness-not-regression", "$.findings[0].status")]),
    ("row mismatch marked ok", CORRECTNESS, ("findings", 1, "status"), "ok", [("correctness-not-regression", "$.findings[1].status")]),
    ("broken on both sides marked ok", BOTH_BROKEN, ("findings", 0, "status"), "ok", [("both-broken-not-info", "$.findings[0].status")]),
    ("error carrying regressions", POKEC_ERROR, ("findings",), example(DID_NOT_FINISH)["findings"],
     [("error-with-regression", "$.findings[0].status"), ("error-with-regression", "$.findings[1].status")]),
    ("date as 24/11/2026", GREEN, ("date",), "24/11/2026", [("bad-date", "$.date")]),
    ("http run link", GREEN, ("run_url",), "http://github.com/FalkorDB/FalkorDB/actions/runs/1", [("bad-url", "$.run_url")]),
    ("skipped night with checks", SKIPPED, ("checks",), example(GREEN)["checks"], [("skipped-with-checks", "$.checks")]),
    ("red night said error", RED, ("status",), "error", [("nightly-status-mismatch", "$.status")]),
    ("embedded unconfirmed regression", RED, ("checks", 0, "confirmed"), False, [("regression-unconfirmed", "$.checks[0].confirmed")]),
    ("check entry not an object", GREEN, ("checks", 1), "pokec", [("wrong-type", "$.checks[1]")]),
]


@pytest.mark.parametrize("start, path, value, expected", [c[1:] for c in CHANGES], ids=[c[0] for c in CHANGES])
def test_one_field_change_gives_exactly_its_problems(start, path, value, expected):
    assert _problems(Case("", start, path, value, "", "")) == expected


# -- whole-text and API cases -----------------------------------------------------

_RATIO = json.dumps(example(MICRO_PASS)["findings"][0]["ratio"])
TEXT_CASES = [
    ('{"schema": 1, "check": "micro",}', "invalid-json", "$"),
    ("", "invalid-json", "$"),
    ("[]", "unknown-kind", "$"),
    ('{"schema": 1}', "unknown-kind", "$"),
    *((json.dumps(example(MICRO_PASS)).replace(f'"ratio": {_RATIO}', f'"ratio": {literal}', 1), "non-finite-number", "$.findings[0].ratio")
      for literal in ("NaN", "-Infinity", "1e999")),
]


@pytest.mark.parametrize("text, code, path", TEXT_CASES)
def test_text_gives_exactly_its_problem(text, code, path):
    _, problems = contract.validate_text(text)
    assert [(p.code, p.path) for p in problems] == [(code, path)]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_in_a_document_built_in_memory(value):
    # merge.py validates parsed results, where NaN is a float, not the parser's stand-in.
    document = example(GREEN)
    document["checks"][0]["machine"]["cores"] = value
    assert [(p.code, p.path) for p in contract.validate_nightly(document)] == [("non-finite-number", "$.checks[0].machine.cores")]


def test_nesting_deeper_than_python_recursion_is_a_problem_not_a_crash():
    depth = 200_000
    _, problems = contract.validate_text('{"check": "micro", "artifacts": ' + "[" * depth + "]" * depth + "}")
    assert [p.code for p in problems] == ["invalid-json"]
    nested: list = []
    for _ in range(sys.getrecursionlimit() * 5):
        nested = [nested]
    document = dict(example(MICRO_PASS), artifacts=nested)
    assert ("wrong-type", "$.artifacts[0]") in [(p.code, p.path) for p in contract.validate_result(document)]


def test_unreadable_file(tmp_path):
    _, problems = contract.validate_file(tmp_path / "absent.json")
    assert [p.code for p in problems] == ["unreadable"]


def test_every_problem_code_is_produced_by_some_test():
    produced = {code for c in CHANGES for code, _ in c[4]} | {code for _, code, _ in TEXT_CASES} | {"unreadable"}
    assert produced == set(contract.CODES)


# -- standard library only ------------------------------------------------------------


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            # A relative import would pull in the rest of the package and its dependencies.
            roots.add(node.module.split(".")[0] if node.level == 0 and node.module else ".")
    return roots


@pytest.mark.parametrize("name", ["contract.py", "__init__.py"])
def test_contract_imports_only_the_standard_library(name):
    # The reporting path must never depend on a package: importing falkorbench.contract
    # runs falkorbench/__init__.py first, so both stay standard library only.
    roots = _imported_roots(Path(contract.__file__).with_name(name))
    assert roots <= sys.stdlib_module_names, sorted(roots - sys.stdlib_module_names)


# -- bench validate -------------------------------------------------------------------


def _bench_validate(*args: str):
    return CliRunner().invoke(cli, ["validate", *args])


def test_bench_validate(tmp_path):
    files = sorted(str(p) for p in examples.VALID.glob("*.json"))
    run = _bench_validate(*files)
    assert run.exit_code == 0, run.output
    assert run.output.splitlines() == [f"{f}: ok ({examples.kind(Path(f).name)})" for f in files]
    # A tag in place of a digest is refused.
    assert examples.CASES[0].code == "image-not-digest"
    bad = tmp_path / "result.json"
    bad.write_text(json.dumps(examples.changed(examples.CASES[0])))
    run = _bench_validate(files[0], str(bad))
    assert run.exit_code == 1
    assert run.output.splitlines()[0].endswith(": ok (nightly)")
    assert re.fullmatch(re.escape(str(bad)) + r": image-not-digest: \$\.baseline\.image: .+", run.output.splitlines()[1])
    assert _bench_validate().exit_code == 2
    assert _bench_validate("--kind", "nightlies", files[0]).exit_code == 2


def test_bench_validate_kind_skips_detection():
    result = str(examples.VALID / examples.MICRO_PASS)
    run = _bench_validate("--kind", "nightly", result)
    assert run.exit_code == 1
    assert ": missing-field: $.date: " in run.output


# -- acceptance.yaml -----------------------------------------------------------------

ACCEPTANCE = yaml.safe_load((BENCH / "acceptance.yaml").read_text())
PAIRS = [(check, pair) for check, pairs in ACCEPTANCE["checks"].items() for pair in pairs]


def _pair(pair_id: str) -> dict:
    return next(pair for _, pair in PAIRS if pair["id"] == pair_id)


def test_acceptance_pairs_are_well_formed():
    assert ACCEPTANCE["schema"] == 1
    ids = [pair["id"] for _, pair in PAIRS]
    assert len(ids) == len(set(ids)), "duplicate pair id"
    for _, pair in PAIRS:
        assert set(pair) <= {"id", "candidate", "baseline", "mode", "expect", "judge"}, pair["id"]
        assert pair["mode"] == "nightly"
        for side in (pair["candidate"], pair["baseline"]):
            assert contract.IMAGE_RE.fullmatch(side["image"]), side
            assert contract.COMMIT_RE.fullmatch(side["commit"]), side
        if pair["candidate"]["image"] == pair["baseline"]["image"] and pair["expect"]:
            assert pair["expect"] == {"verdict": "pass", "no_regression_findings": True}, pair["id"]
        for matcher in pair["expect"].get("findings", []):
            assert set(matcher) <= {"name", "name_regex", "metric", "status"}, pair["id"]
            metrics = matcher.get("metric", [])
            assert set(metrics if isinstance(metrics, list) else [metrics]) <= set(contract.METRICS), pair["id"]
            assert matcher["status"] in contract.FINDING_STATUSES, pair["id"]
            re.compile(matcher.get("name_regex", ""))


def test_micro_hw_acceptance_rows_exist_in_bench():
    names = {q.name for q in QUERIES}
    rows = [re.match(r"\^(.+?)\(", m["name_regex"]).group(1) for m in _pair("micro-hw-6.2.0-vs-main")["expect"]["findings"]]
    assert rows
    for row in rows:
        assert row in names, f"bench has no row named {row!r}"


# A pattern from acceptance.yaml must match its own finding names and no neighbour's.
PATTERN_CASES = [
    ("micro-hw-6.2.0-vs-main", "UNWIND id MATCH edge 1 row", "UNWIND id MATCH edge 1 row [writes]", True),
    ("micro-hw-6.2.0-vs-main", "UNWIND id MATCH edge 1 row", "unwind id match edge 1 row", True),
    ("micro-hw-6.2.0-vs-main", "UNWIND id MATCH edge 1 row", "UNWIND id MATCH edge 100 rows [writes]", False),
    ("micro-hw-6.2.0-vs-main", "UNWIND id MATCH edge 1 row", "UNWIND id MATCH edge 1 row [writes] (re-run)", False),
    ("ldbc-6.0.4-vs-6.2.0", "ic1", "IC1 p95", True),
    ("ldbc-6.0.4-vs-6.2.0", "ic1", "IC12 p95", False),
    ("ldbc-6.0.4-vs-6.2.0", "ic1", "IC10", False),
]


@pytest.mark.parametrize("pair_id, row, name, matches", PATTERN_CASES)
def test_acceptance_pattern_matches_only_its_row(pair_id, row, name, matches):
    patterns = [m["name_regex"] for m in _pair(pair_id)["expect"]["findings"] if row.lower() in m["name_regex"].lower()]
    pattern = min(patterns, key=len)  # "ic1" is in both \bic1\b and \bic12\b: take the shorter
    assert bool(re.search(pattern, name, re.IGNORECASE)) == matches
