"""Tests for the nightly perf reporter (post.py) and the missed-run alarm (missed_run.py).

Standard library only: `python -m pytest tests/perf/report` or
`python -m unittest discover tests/perf/report`.

Expected messages come from the contract (nightly.json schema 1, issue #3974) and
from perf-judge's fixtures in tests/perf/fixtures/, never from what merge.py happens
to write. The reporter runs perf-judge's tests/perf/validate.py; PERF_JUDGE_DIR points
both at another checkout's tests/perf (development before they are on this branch).
"""

from __future__ import annotations

import contextlib
import http.client
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
import urllib.error
import urllib.parse
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import missed_run
import post

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
JUDGE = Path(os.environ.get("PERF_JUDGE_DIR") or REPO_ROOT / "tests" / "perf")
FIXTURES = JUDGE / "fixtures"
VALIDATOR = JUDGE / "validate.py"
INVALID_CASES = JUDGE / "invalid_cases.py"


def load_invalid_cases():
    """perf-judge's table of broken documents (tests/perf/invalid_cases.py), loaded by path."""
    spec = importlib.util.spec_from_file_location("perf_invalid_cases", INVALID_CASES)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

CANDIDATE_IMAGE = ("ghcr.io/falkordb/falkordb-server@sha256:"
                   "95b610a600682533d854b6ece6bddbea80b49d0b89e2cd0b2cf19bd0a8142aed")
BASELINE_IMAGE = "falkordb/falkordb-server@sha256:44eb4a6d48ef6f2b06aeefea46c5f4c17cd2310d3832135be18fd7fc8ba30923"
CANDIDATE_COMMIT = "4c1088949a5852e3949c05bfb3d92a603c4cd4ca"
PREVIOUS_GREEN = "41d2f20ce9047eb178378fe6440241c73278e6f0"
RUN_URL = "https://github.com/FalkorDB/FalkorDB/actions/runs/1000"
WEBHOOK_SECRET = "key=SECRETKEY123&token=SECRETTOKEN456"

HEADLINE_WORD = {"green": "green", "skipped": "skipped", "error": "ERROR", "red": "REGRESSION"}


def result(check: str, verdict: str, findings: list | None = None, summary: str = "one line",
           confirmed: bool | None = None, error: str | None = None) -> dict:
    """A schema-1 result.json, as in the contract's §2 example."""
    return {
        "schema": 1,
        "check": check,
        "verdict": verdict,
        "confirmed": verdict == "regression" if confirmed is None else confirmed,
        "summary": summary,
        "candidate": {"image": CANDIDATE_IMAGE, "commit": CANDIDATE_COMMIT, "label": "main@4c10889"},
        "baseline": {"image": BASELINE_IMAGE, "commit": None, "label": "6.2.0"},
        "machine": {"runner": "gce", "type": "n4-standard-16", "cpu": "Intel Xeon", "cores": 16},
        "started": "2026-11-24T00:05:00Z",
        "finished": "2026-11-24T00:41:00Z",
        "findings": findings or [],
        "error": error,
        "artifacts": [],
    }


def finding(name: str, metric: str, candidate, baseline, ratio, threshold,
            direction: str = "higher_is_worse", status: str = "regression") -> dict:
    return {"name": name, "metric": metric, "candidate": candidate, "baseline": baseline, "ratio": ratio,
            "threshold": threshold, "direction": direction, "status": status}


UNWIND = finding("UNWIND id seek @200k", "p50_ms", 13.3, 0.4, 33.25, 1.5)


def nightly(status: str, checks: list, skip_reason: str | None = None,
            previous_green: str | None = PREVIOUS_GREEN, candidate_commit: str | None = CANDIDATE_COMMIT) -> dict:
    """A schema-1 nightly.json, as in the contract's §3 example."""
    return {
        "schema": 1,
        "date": "2026-11-24",
        "run_url": RUN_URL,
        "candidate": {"image": CANDIDATE_IMAGE, "commit": candidate_commit, "label": "main@4c10889"},
        "baseline": {"image": BASELINE_IMAGE, "commit": None, "label": "6.2.0"},
        "previous_green_commit": previous_green,
        "status": status,
        "skip_reason": skip_reason,
        "checks": checks,
    }


class Workspace(unittest.TestCase):
    """Writes nightly.json files to a temporary directory and runs the reporter on them."""

    def setUp(self):
        self.directory = Path(self.enter(tempfile.TemporaryDirectory()))
        self.environ = {}
        self.enter(unittest.mock.patch.object(post, "POST_RETRY_DELAYS", (0.0, 0.0)))
        self.enter(unittest.mock.patch.object(post, "VALIDATOR", VALIDATOR))

    def enter(self, context):
        """TestCase.enterContext, which needs Python 3.11."""
        value = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return value

    def write(self, content, name: str = "nightly.json") -> Path:
        path = self.directory / name
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return path

    def run_post(self, content, environ: dict | None = None) -> tuple[int, str, str]:
        path = content if isinstance(content, Path) else self.write(content)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = post.main(["--nightly", str(path)], dict(self.environ if environ is None else environ))
        return code, stdout.getvalue(), stderr.getvalue()

    def chat(self, content) -> str:
        code, stdout, _ = self.run_post(content)
        self.assertEqual(code, 0)
        return stdout


# ---------------------------------------------------------------------------
# The four kinds of night, with messages written from the contract.


class GreenSkippedErrorMessages(Workspace):
    def test_green_is_one_line_with_date_labels_count_and_run(self):
        night = nightly("green", [result("micro", "pass"), result("size-probe", "pass")])
        self.assertEqual(self.chat(night), (
            f"*Perf nightly 2026-11-24: green* — main@4c10889 vs 6.2.0, all 2 checks passed. <{RUN_URL}|run>\n"))

    def test_skipped_is_one_line_with_the_reason(self):
        night = nightly("skipped", [], skip_reason="main has not moved since 4c10889")
        self.assertEqual(self.chat(night), (
            "*Perf nightly 2026-11-24: skipped* — main@4c10889 vs 6.2.0: main has not moved since 4c10889. "
            f"<{RUN_URL}|run>\n"))

    def test_error_names_each_check_that_could_not_measure_and_never_says_pass(self):
        night = nightly("error", [
            result("micro", "pass"),
            {"check": "pokec", "verdict": "error", "error": "no result.json"},
            result("soak", "error", error="server did not start: port in use"),
        ])
        self.assertEqual(self.chat(night), (
            "*Perf nightly 2026-11-24: ERROR* — main@4c10889 vs 6.2.0: 2 of 3 checks could not measure; "
            f"tonight is not green. <{RUN_URL}|run>\n"
            "*Could not measure* (no verdict, not a pass):\n"
            "- *pokec*: no result.json\n"
            "- *soak*: server did not start: port in use\n"
            "Passed: micro.\n"))

    def test_an_error_result_without_text_still_says_why_not(self):
        night = nightly("error", [{"check": "pokec", "verdict": "error", "error": ""}])
        self.assertIn("- *pokec*: no reason given", self.chat(night))


class RedMessage(Workspace):
    def setUp(self):
        super().setUp()
        # A real range on main, read with git when the checkout has it (CI checks out
        # with history), else from GitHub's compare API: either way these four PRs.
        self.night = nightly("red", [
            result("micro", "regression", confirmed=True, summary="4 rows regressed, worst 33x", findings=[
                finding("bulk create", "instructions", 47_190_000, 39_250_000, 1.2023, 1.05),
                UNWIND,
                finding("steady", "p99_ms", 1.0, 1.0, 1.0, 1.5, status="ok"),
            ]),
            {"check": "pokec", "verdict": "error", "error": "no result.json"},
            result("size-probe", "pass"),
        ])

    def test_red_lists_regressions_worst_first_errors_commits_and_links(self):
        commits = post.CommitRange([
            post.Commit("4c1088949", "fix(optimizer): per-row id seek", 3963),
            post.Commit("0ecba8bc9", "fix(optimizer): per-row index seek (#3887)", 3889),
            post.Commit("abcdef123", "direct push without a PR", None),
        ], 3)
        with unittest.mock.patch.object(post, "commits_since_green", return_value=commits):
            text = self.chat(self.night)
        lines = text.splitlines()
        self.assertEqual(lines[0], "*Perf nightly 2026-11-24: REGRESSION* — main@4c10889 vs 6.2.0: "
                                   "1 of 3 checks regressed.")
        self.assertEqual(lines[1], "*micro* (confirmed): 4 rows regressed, worst 33x")
        # Worst ratio first, whatever order the check wrote them in; the `ok` row is not shown.
        self.assertEqual(lines[2], "- UNWIND id seek @200k: p50_ms 13.3 vs 0.4 (33.25x, threshold 1.50x)")
        self.assertEqual(lines[3], "- bulk create: instructions 47,190,000 vs 39,250,000 (1.20x, threshold 1.05x)")
        self.assertNotIn("steady", text)
        self.assertEqual(lines[4:7], ["*Could not measure* (no verdict, not a pass):",
                                      "- *pokec*: no result.json", "Passed: size-probe."])
        compare = f"https://github.com/FalkorDB/FalkorDB/compare/{PREVIOUS_GREEN}...{CANDIDATE_COMMIT}"
        self.assertIn(f"Commits since the last green night (3, <{compare}|41d2f20...4c10889>):", lines)
        self.assertIn("- <https://github.com/FalkorDB/FalkorDB/pull/3963|#3963> fix(optimizer): per-row id seek",
                      lines)
        self.assertIn("- <https://github.com/FalkorDB/FalkorDB/pull/3889|#3889> fix(optimizer): per-row index seek "
                      "(#3887)", lines)
        self.assertIn("- `abcdef1` direct push without a PR (no PR number in the title)", lines)
        self.assertTrue(lines[-1].startswith(f"<{RUN_URL}|run> · <https://github.com/FalkorDB/FalkorDB/issues/new?"))

    def test_the_issue_link_is_prefilled_and_labelled(self):
        with unittest.mock.patch.object(post, "commits_since_green", return_value=post.CommitRange([], 0, "x")):
            text = self.chat(self.night)
        url = text.splitlines()[-1].split(" · <", 1)[1].rsplit("|open an issue>", 1)[0]
        parts = urllib.parse.urlsplit(url)
        self.assertEqual((parts.netloc, parts.path), ("github.com", "/FalkorDB/FalkorDB/issues/new"))
        query = urllib.parse.parse_qs(parts.query)
        self.assertEqual(query["labels"], ["performance,perf-regression"])
        self.assertIn("&labels=performance,perf-regression&", url, "a literal comma, as GitHub's docs write it")
        self.assertEqual(query["title"], ["perf: nightly 2026-11-24 regression in micro (main@4c10889 vs 6.2.0)"])
        body = query["body"][0]
        for expected in (RUN_URL, CANDIDATE_IMAGE, BASELINE_IMAGE,
                         f"compare/{PREVIOUS_GREEN}...{CANDIDATE_COMMIT}",
                         "- micro: UNWIND id seek @200k: p50_ms 13.3 vs 0.4 (33.25x, threshold 1.50x)"):
            self.assertIn(expected, body)
        self.assertNotIn(" ", url, "the link must be URL-encoded")

    def test_an_unconfirmed_regression_says_so(self):
        self.night["checks"][0]["confirmed"] = False
        with unittest.mock.patch.object(post, "commits_since_green", return_value=post.CommitRange([], 0, "x")):
            self.assertIn("*micro* (NOT confirmed by a re-run)", self.chat(self.night))

    def test_a_regression_on_the_same_commit_as_the_last_green_night_points_at_the_run(self):
        self.night["previous_green_commit"] = CANDIDATE_COMMIT
        self.assertIn("Same candidate commit as the last green night (`4c1088949a`): nothing new was merged on main",
                      self.chat(self.night))

    def test_the_compare_link_stays_when_the_commit_list_is_unavailable(self):
        unavailable = post.CommitRange([], 0, "git: no; API: no")
        with unittest.mock.patch.object(post, "commits_since_green", return_value=unavailable):
            text = self.chat(self.night)
        compare = f"https://github.com/FalkorDB/FalkorDB/compare/{PREVIOUS_GREEN}...{CANDIDATE_COMMIT}"
        self.assertIn(f"Commits since the last green night: git: no; API: no. <{compare}|compare>", text)

    def test_no_previous_green_night(self):
        self.night["previous_green_commit"] = None
        self.assertIn("Commits since the last green night: no earlier green night is recorded.",
                      self.chat(self.night))

    def test_unknown_candidate_commit(self):
        self.night["candidate"]["commit"] = None
        self.assertIn("Commits since the last green night: the candidate's commit is unknown.",
                      self.chat(self.night))

    def test_more_regression_rows_than_shown_are_counted(self):
        self.night["checks"][0]["findings"] = [
            finding(f"query {index}", "cycles", 2.0 + index, 1.0, 2.0 + index, 1.1) for index in range(7)]
        with unittest.mock.patch.object(post, "commits_since_green", return_value=post.CommitRange([], 0, "x")):
            text = self.chat(self.night)
        self.assertIn("- query 6: cycles 8 vs 1 (8.00x, threshold 1.10x)", text)
        self.assertIn("- 4 more regression rows in the run's result.json", text)


# ---------------------------------------------------------------------------
# Never green unless every check passed and the file agrees with itself.


class StatusIsNeverBetterThanTheEvidence(Workspace):
    CASES = [
        # (declared status, verdicts, shown status)
        ("green", ["pass", "pass"], "green"),
        ("red", ["regression", "error", "pass"], "red"),
        ("error", ["error", "pass"], "error"),
        ("green", ["pass", "error"], "error"),
        ("green", ["pass", "regression"], "red"),
        ("error", ["pass", "regression"], "red"),
        ("red", ["pass", "pass"], "red"),
        ("error", ["pass"], "error"),
        ("green", [], "error"),
        ("red", [], "error"),
        ("skipped", ["pass"], "error"),
        ("skipped", ["regression"], "red"),
        ("bogus", ["pass"], "error"),
        ("green", ["passed"], "error"),  # an unknown verdict is not a pass
        ("green", [None], "error"),  # neither is a missing one
    ]

    def test_shown_status(self):
        for declared, verdicts, shown in self.CASES:
            with self.subTest(declared=declared, verdicts=verdicts):
                checks = [result(f"check{index}", verdict, [UNWIND] if verdict == "regression" else None)
                          if verdict in ("pass", "regression") else
                          {"check": f"check{index}", "verdict": verdict, "error": "x"}
                          for index, verdict in enumerate(verdicts)]
                night = post.read_night(self.write(nightly(declared, checks)), {})
                self.assertEqual(night.status, shown)
                with unittest.mock.patch.object(post, "commits_since_green", return_value=post.CommitRange([], 0, "x")):
                    text = self.chat(nightly(declared, checks))
                self.assertTrue(text.startswith(f"*Perf nightly 2026-11-24: {HEADLINE_WORD[shown]}*"), text)
                if shown != declared:
                    self.assertIn("*Problems with nightly.json:*", text)

    def test_guards_that_hold_without_the_validator(self):
        with unittest.mock.patch.object(post, "contract_problems", return_value=[]):
            duplicate = self.chat(nightly("green", [result("micro", "pass"), result("micro", "pass")]))
            stub = self.chat(nightly("green", [result("micro", "pass"),
                                               {"check": "pokec", "verdict": "pass", "error": None}]))
        self.assertTrue(duplicate.startswith("*Perf nightly 2026-11-24: ERROR*"), duplicate)
        self.assertIn("check 'micro' appears 2 times", duplicate)
        self.assertIn(": nightly.json has problems, so tonight is not green.", duplicate.splitlines()[0])
        self.assertTrue(stub.startswith("*Perf nightly 2026-11-24: ERROR*"), stub)
        self.assertIn("- *pokec*: says pass but carries no result", stub)

    def test_a_validator_that_cannot_run_is_an_error_not_a_pass(self):
        for validator in (self.directory / "absent.py", self.write("raise RuntimeError('boom')", "broken.py")):
            with self.subTest(validator=validator.name), unittest.mock.patch.object(post, "VALIDATOR", validator):
                text = self.chat(nightly("green", [result("micro", "pass")]))
                self.assertTrue(text.startswith("*Perf nightly 2026-11-24: ERROR*"), text)
                self.assertIn(f"could not run the contract validator {validator.name}", text)

    def test_contract_problems_are_listed_and_capped(self):
        # Eight findings with a metric the contract does not list: eight unknown-metric problems.
        rows = [finding(f"row {index}", "latency_ms", 1.0, 1.0, 1.0, 1.1, status="ok") for index in range(8)]
        text = self.chat(nightly("green", [result("micro", "pass", rows)]))
        self.assertTrue(text.startswith("*Perf nightly 2026-11-24: ERROR*"), text)
        self.assertIn("contract: unknown-metric at $.checks[0].findings[0].metric", text)
        self.assertIn("contract: unknown-metric at $.checks[0].findings[4].metric", text)
        self.assertNotIn("findings[5].metric", text)
        self.assertIn("contract: 3 more (run tests/perf/validate.py)", text)

    def test_a_check_entry_that_is_not_an_object_is_an_error(self):
        text = self.chat(nightly("green", [result("micro", "pass"), "pokec"]))
        self.assertIn("- *check #2*: the entry is not a JSON object", text)
        self.assertTrue(text.startswith("*Perf nightly 2026-11-24: ERROR*"))

    def test_unreadable_files_are_errors_with_exit_zero(self):
        for content in ("", "{", "[]", "{}", "null", "1", json.dumps({"schema": 2, "status": "green", "checks": []}),
                        json.dumps({"schema": 1, "status": "green", "checks": "micro"})):
            with self.subTest(content=content):
                text = self.chat(content)
                self.assertIn(": ERROR*", text.splitlines()[0])
                self.assertIn("*Problems with nightly.json:*", text)
                self.assertNotIn("green*", text)

    def test_a_crash_inside_the_reporter_still_reports_and_exits_non_zero(self):
        environ = {"GITHUB_RUN_ID": "77"}
        with unittest.mock.patch.object(post, "build_message", side_effect=RuntimeError("boom <users/all>")):
            code, stdout, stderr = self.run_post(nightly("green", [result("micro", "pass")]), environ)
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "*Perf nightly: the reporter crashed* (RuntimeError: boom ‹users/all>), so tonight's "
                                 "result was not reported. <https://github.com/FalkorDB/FalkorDB/actions/runs/77|run>\n")
        self.assertIn("Traceback", stderr)

    def test_a_missing_file_is_an_error_and_links_the_current_run(self):
        environ = {"GITHUB_RUN_ID": "77", "GITHUB_REPOSITORY": "FalkorDB/FalkorDB"}
        code, stdout, _ = self.run_post(self.directory / "absent.json", environ)
        self.assertEqual(code, 0)
        self.assertIn("could not read absent.json", stdout)
        self.assertIn("<https://github.com/FalkorDB/FalkorDB/actions/runs/77|run>", stdout)


# ---------------------------------------------------------------------------
# How findings read (contract v1.1: nullable values, correctness metrics).


class FindingWords(unittest.TestCase):
    def test_values_and_ratios(self):
        cases = [
            (UNWIND, "UNWIND id seek @200k: p50_ms 13.3 vs 0.4 (33.25x, threshold 1.50x)"),
            (finding("q", "throughput", 50, 100, 0.5, 0.8, "lower_is_worse"),
             "q: throughput 50 vs 100 (0.50x, threshold 0.80x)"),
            (finding("q", "alloc_bytes", 1_234_567, 1000, 1234.567, 1.1),
             "q: alloc_bytes 1,234,567 vs 1,000 (1,235x, threshold 1.10x)"),
            (finding("q", "p99_ms", 0.00123, 0.001, 1.23, None), "q: p99_ms 0.00123 vs 0.001 (1.23x)"),
            # baseline 0: the check leaves the ratio null and so does the message.
            (finding("q", "p99_ms", 5, 0, None, 1.5), "q: p99_ms 5 vs 0 (threshold 1.50x)"),
            (finding("q", "p99_ms", None, None, None, None), "q: p99_ms n/a vs n/a"),
            (finding("3-hop read", "completed", 0, 1, None, None, "lower_is_worse"),
             "3-hop read: did not finish (baseline finished)"),
            (finding("3-hop read", "completed", 0, 0, None, None, "lower_is_worse", "info"),
             "3-hop read: did not finish on either side"),
            (finding("edge count matches", "invariant", 0, 0, None, None, "lower_is_worse", "info"),
             "invariant broken on both sides: edge count matches"),
            # Values the validator would reject must still not crash the reporter.
            (finding("q", "completed", 0, [1], None, None, "lower_is_worse"), "q: did not finish"),
            (finding("q", "p50_ms", 2, 1, 10 ** 400, 1.5), "q: p50_ms 2 vs 1 (threshold 1.50x)"),
            (finding("q", "throughput", 1, 300, 1 / 300, 0.8, "lower_is_worse"),
             "q: throughput 1 vs 300 (0.0033x, threshold 0.80x)"),
            (finding("3-hop read", "completed", 0, None, None, None, "lower_is_worse"), "3-hop read: did not finish"),
            (finding("edge count matches", "invariant", 0, 1, None, None, "lower_is_worse"),
             "invariant broken: edge count matches (holds on baseline)"),
            (finding("IC3", "rows", 12, 14, None, None, "mismatch"), "IC3: row count differs: 12 vs 14"),
        ]
        for row, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(post.describe_finding(row), expected)
                self.assertNotIn("None", post.describe_finding(row))

    def test_worst_first_across_directions(self):
        rows = [
            finding("slower 2x", "p50_ms", 2, 1, 2.0, 1.5),
            finding("throughput a third", "throughput", 1, 3, 1 / 3, 0.8, "lower_is_worse"),
            finding("unknown ratio", "p50_ms", 1, 0, None, 1.5),
            finding("rows", "rows", 1, 2, None, None, "mismatch"),
            finding("slower 5x", "p50_ms", 5, 1, 5.0, 1.5),
            finding("crash", "completed", 0, 1, None, None, "lower_is_worse"),
        ]
        names = [row["name"] for row in sorted(rows, key=post.severity_key)]
        self.assertEqual(names, ["rows", "crash", "unknown ratio", "slower 5x", "throughput a third", "slower 2x"])


# ---------------------------------------------------------------------------
# Commits since the last green night.


def git(directory: Path, *arguments: str) -> str:
    environ = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    return subprocess.run(["git", "-C", str(directory), *arguments], check=True, capture_output=True,
                          text=True, env=environ).stdout.strip()


class CommitList(Workspace):
    def setUp(self):
        super().setUp()
        self.repository = self.directory / "repository"
        self.repository.mkdir()
        git(self.repository, "init", "-q")
        self.shas = []
        for subject in ("base (#1)", "fix: a < b (#3887) (#3889)", "direct push", "perf: x (#3963)"):
            git(self.repository, "commit", "-q", "--allow-empty", "-m", subject)
            self.shas.append(git(self.repository, "rev-parse", "HEAD"))
        self.enter(unittest.mock.patch.object(post, "CHECKOUT", self.repository))

    def night(self, previous: str, candidate: str) -> post.Night:
        return post.read_night(self.write(nightly("red", [result("micro", "regression")], previous_green=previous,
                                                  candidate_commit=candidate)), {})

    def test_from_git_newest_first_with_the_last_pr_number_of_each_title(self):
        commits = post.commits_since_green(self.night(self.shas[0], self.shas[3]), {})
        self.assertEqual(commits.unavailable, "")
        self.assertEqual(commits.total, 3)
        self.assertEqual([(c.sha, c.title, c.pull_request) for c in commits.commits], [
            (self.shas[3], "perf: x", 3963),
            (self.shas[2], "direct push", None),
            (self.shas[1], "fix: a < b (#3887)", 3889),
        ])

    def test_from_the_compare_api_when_git_does_not_have_the_commits(self):
        previous, candidate = "a" * 40, "b" * 40
        compare = {"total_commits": 2, "commits": [
            {"sha": "1" * 40, "commit": {"message": "older (#10)\n\nbody"}},
            {"sha": "2" * 40, "commit": {"message": "newer (#11)"}},
        ]}
        with StubServer({f"/repos/FalkorDB/FalkorDB/compare/{previous}...{candidate}": (200, compare)}) as server:
            environ = {"GITHUB_API_URL": server.url, "GH_TOKEN": "t0ken"}
            commits = post.commits_since_green(self.night(previous, candidate), environ)
        self.assertEqual([(c.title, c.pull_request) for c in commits.commits], [("newer", 11), ("older", 10)])
        self.assertEqual(server.requests[0]["headers"].get("Authorization"), "Bearer t0ken")

    def test_unavailable_says_why_from_both_sources(self):
        previous, candidate = "a" * 40, "b" * 40
        with StubServer({}) as server:
            commits = post.commits_since_green(self.night(previous, candidate), {"GITHUB_API_URL": server.url})
        self.assertIn("aaaaaaaaaa is not in this checkout", commits.unavailable)
        self.assertIn("compare API: HTTP 404", commits.unavailable)

    def test_a_value_that_is_not_a_hash_never_reaches_git(self):
        night = self.night("--output=/tmp/x", self.shas[3])
        self.assertIsNone(night.previous_green_commit)
        self.assertEqual(post.commits_since_green(night, {}).unavailable, "no earlier green night is recorded")

    @unittest.skipUnless(
        subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{PREVIOUS_GREEN}^{{commit}}"],
                       capture_output=True, check=False).returncode == 0
        and subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{CANDIDATE_COMMIT}^{{commit}}"],
                           capture_output=True, check=False).returncode == 0,
        "this checkout lacks the history of main around 4c1088949")
    def test_a_real_range_on_main(self):
        with unittest.mock.patch.object(post, "CHECKOUT", REPO_ROOT):
            commits = post.commits_since_green(self.night(PREVIOUS_GREEN, CANDIDATE_COMMIT), {})
        self.assertEqual([c.pull_request for c in commits.commits], [3963, 3017, 3889, 3930])


# ---------------------------------------------------------------------------
# Delivery: stdout, the job summary, the webhook.


class StubServer:
    """A local HTTP server answering from `routes`: path -> (status, JSON) or a list of them."""

    def __init__(self, routes: dict):
        self.routes = {path: list(answer) if isinstance(answer, list) else answer for path, answer in routes.items()}
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                parts = urllib.parse.urlsplit(self.path)
                stub.requests.append({"method": self.command, "path": parts.path, "query": parts.query,
                                      "headers": dict(self.headers), "body": self.rfile.read(length)})
                answer = stub.routes.get(parts.path, (404, {"message": "Not Found"}))
                if isinstance(answer, list):
                    answer = answer.pop(0) if len(answer) > 1 else answer[0]
                status, body, *delay = answer
                if delay:
                    time.sleep(delay[0])
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = answer

            def log_message(self, *arguments):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        return self

    def __exit__(self, *exception):
        self.server.shutdown()
        self.server.server_close()


class Delivery(Workspace):
    GREEN = nightly("green", [result("micro", "pass")])

    def test_posts_the_printed_text_as_chat_json(self):
        with StubServer({"/v1/spaces/X/messages": (200, {"name": "spaces/X/messages/1"})}) as server:
            code, stdout, stderr = self.run_post(self.GREEN, {
                "PERF_CHAT_WEBHOOK": f"{server.url}/v1/spaces/X/messages?{WEBHOOK_SECRET}"})
        self.assertEqual(code, 0)
        self.assertEqual(len(server.requests), 1)
        request = server.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["headers"]["Content-Type"], "application/json; charset=UTF-8")
        self.assertEqual(json.loads(request["body"]), {"text": stdout.rstrip("\n")})
        self.assertIn("Posted to Chat.", stderr)

    def test_a_failed_post_is_a_warning_retried_and_never_shows_the_webhook(self):
        with StubServer({"/hook": (500, {"error": "boom"})}) as server:
            code, stdout, stderr = self.run_post(self.GREEN, {
                "PERF_CHAT_WEBHOOK": f"{server.url}/hook?{WEBHOOK_SECRET}", "GITHUB_ACTIONS": "true"})
        self.assertEqual(code, 0)
        self.assertEqual(len(server.requests), 3)
        self.assertIn("::warning::posting to Chat failed: HTTP 500", stderr)
        for secret in ("SECRETKEY123", "SECRETTOKEN456", "/hook"):
            self.assertNotIn(secret, stdout + stderr)

    def test_a_timeout_is_not_retried_because_the_post_may_have_arrived(self):
        slow = StubServer({"/hook": (200, {}, 1.0)})
        with unittest.mock.patch.object(post, "HTTP_TIMEOUT_SECONDS", 0.2), slow as server:
            code, _, stderr = self.run_post(self.GREEN, {"PERF_CHAT_WEBHOOK": f"{server.url}/hook"})
        self.assertEqual(code, 0)
        self.assertEqual(len(server.requests), 1)
        self.assertIn("timed out waiting for Chat (not retried: the message may have arrived)", stderr)

    def test_a_connect_timeout_is_retried_because_nothing_was_sent(self):
        timeout = urllib.error.URLError(TimeoutError("timed out"))
        with unittest.mock.patch.object(post.urllib.request, "urlopen", side_effect=timeout) as urlopen:
            code, _, stderr = self.run_post(self.GREEN, {"PERF_CHAT_WEBHOOK": "https://chat.example/hook"})
        self.assertEqual((code, urlopen.call_count), (0, 3))
        self.assertIn("posting to Chat failed: connection failed: TimeoutError", stderr)

    def test_a_malformed_webhook_is_a_warning_not_a_crash(self):
        code, stdout, stderr = self.run_post(self.GREEN, {"PERF_CHAT_WEBHOOK": "https://[SECRETKEY/v1/x"})
        self.assertEqual(code, 0)
        self.assertIn("PERF_CHAT_WEBHOOK is not a valid URL", stderr)
        self.assertNotIn("SECRETKEY", stdout + stderr)

    def test_a_cut_short_answer_after_a_2xx_is_still_a_delivered_post(self):
        class Answer:
            def __enter__(self):
                return self

            def __exit__(self, *exception):
                return False

            def read(self, *arguments):
                raise http.client.IncompleteRead(b"")

        with unittest.mock.patch.object(post.urllib.request, "urlopen", return_value=Answer()) as urlopen:
            code, _, stderr = self.run_post(self.GREEN, {"PERF_CHAT_WEBHOOK": "https://chat.example/hook"})
        self.assertEqual((code, urlopen.call_count), (0, 1))
        self.assertIn("Posted to Chat.", stderr)

    def test_a_malformed_answer_is_a_warning_and_not_retried(self):
        with unittest.mock.patch.object(post.urllib.request, "urlopen",
                                        side_effect=http.client.BadStatusLine("x")) as urlopen:
            code, _, stderr = self.run_post(self.GREEN, {"PERF_CHAT_WEBHOOK": "https://chat.example/hook"})
        self.assertEqual((code, urlopen.call_count), (0, 1))
        self.assertIn("posting to Chat failed: bad answer from Chat: BadStatusLine (not retried", stderr)

    def test_any_other_failure_while_posting_is_a_warning(self):
        with unittest.mock.patch.object(post, "post_to_chat", side_effect=RuntimeError("boom")):
            code, stdout, stderr = self.run_post(self.GREEN, {"PERF_CHAT_WEBHOOK": "https://chat.example/hook"})
        self.assertEqual(code, 0)
        self.assertTrue(stdout.startswith("*Perf nightly 2026-11-24: green*"))
        self.assertIn("posting to Chat failed: unexpected RuntimeError", stderr)

    def test_a_github_body_cut_short_is_a_github_error(self):
        class Answer:
            def __enter__(self):
                return self

            def __exit__(self, *exception):
                return False

            def read(self, *arguments):
                raise http.client.IncompleteRead(b"{")

        with unittest.mock.patch.object(post.urllib.request, "urlopen", return_value=Answer()), \
                self.assertRaises(post.GitHubError) as raised:
            post.github_get("https://api.github.com/repos/x/y", "")
        self.assertIn("IncompleteRead", str(raised.exception))

    def test_a_client_error_is_not_retried(self):
        with StubServer({"/hook": (400, {"error": "bad"})}) as server:
            _, _, stderr = self.run_post(self.GREEN, {"PERF_CHAT_WEBHOOK": f"{server.url}/hook"})
        self.assertEqual(len(server.requests), 1)
        self.assertIn("warning: posting to Chat failed: HTTP 400", stderr)

    def test_an_unreachable_webhook_is_a_warning(self):
        with StubServer({}) as server:
            url = f"{server.url}/hook?{WEBHOOK_SECRET}"
        code, _, stderr = self.run_post(self.GREEN, {"PERF_CHAT_WEBHOOK": url})  # server is gone
        self.assertEqual(code, 0)
        self.assertIn("posting to Chat failed: connection failed", stderr)
        self.assertNotIn("SECRET", stderr)

    def test_only_https_webhooks(self):
        for url in ("http://chat.googleapis.com/v1/x", f"file:///etc/passwd?{WEBHOOK_SECRET}", "not a url"):
            with self.subTest(url=url):
                code, _, stderr = self.run_post(self.GREEN, {"PERF_CHAT_WEBHOOK": url})
                self.assertEqual(code, 0)
                self.assertIn("PERF_CHAT_WEBHOOK is not an https:// URL", stderr)
                self.assertNotIn("SECRET", stderr)

    def test_without_a_webhook_it_prints_and_writes_the_summary(self):
        summary = self.directory / "summary.md"
        summary.write_text("earlier step\n")
        code, stdout, stderr = self.run_post(self.GREEN, {"GITHUB_STEP_SUMMARY": str(summary)})
        self.assertEqual(code, 0)
        self.assertIn("PERF_CHAT_WEBHOOK is not set: not posted to Chat.", stderr)
        self.assertTrue(stdout.startswith("*Perf nightly 2026-11-24: green*"))
        self.assertEqual(summary.read_text(), (
            "earlier step\n**Perf nightly 2026-11-24: green** — main@4c10889 vs 6.2.0, all 1 check passed. "
            f"[run]({RUN_URL})  \n\n"))

    def test_text_from_results_and_commits_cannot_mention_or_link_in_chat(self):
        night = nightly("error", [{"check": "<users/all>", "verdict": "error", "error": "<https://evil|click>"}])
        text = self.chat(night)
        self.assertNotIn("<users/all>", text)
        self.assertNotIn("<https://evil", text)
        self.assertIn("‹users/all>", text)

    def test_a_run_url_that_could_inject_markup_is_not_used(self):
        night = nightly("green", [result("micro", "pass")])
        night["run_url"] = "https://github.com/FalkorDB/FalkorDB/actions/runs/1|click><users/all><https://evil.example|x"
        with unittest.mock.patch.object(post, "contract_problems", return_value=[]):
            code, stdout, _ = self.run_post(night, {"GITHUB_RUN_ID": "5"})
        self.assertEqual(code, 0)
        self.assertNotIn("users/all", stdout)
        self.assertNotIn("evil.example", stdout)
        self.assertIn("<https://github.com/FalkorDB/FalkorDB/actions/runs/5|run>", stdout)

    def test_an_oversized_run_url_is_replaced_and_every_kind_of_night_fits(self):
        huge = "https://github.com/FalkorDB/FalkorDB/actions/runs/1?" + "a" * 40_000
        nights = {
            "green": nightly("green", [result("micro", "pass")]),
            "error": nightly("error", [{"check": "pokec", "verdict": "error", "error": "no result.json"}]),
            "red": nightly("red", [result("micro", "regression", [UNWIND])]),
            "skipped": nightly("skipped", [], skip_reason="main has not moved"),
        }
        for status, night in nights.items():
            for environ, link in (({"GITHUB_RUN_ID": "5"}, "<https://github.com/FalkorDB/FalkorDB/actions/runs/5|run>"),
                                  ({}, "run link unknown")):
                with self.subTest(status=status, environ=environ):
                    night["run_url"] = huge
                    with unittest.mock.patch.object(post, "contract_problems", return_value=[]), \
                            unittest.mock.patch.object(post, "commits_since_green",
                                                       return_value=post.CommitRange([], 0, "x")):
                        code, stdout, _ = self.run_post(night, environ)
                    self.assertEqual(code, 0)
                    self.assertLessEqual(len(stdout.rstrip("\n")), post.CHAT_CHARACTER_BUDGET)
                    self.assertTrue(stdout.startswith("*Perf nightly 2026-11-24: "), stdout[:100])
                    self.assertIn(link, stdout)
                    self.assertNotIn("a" * 100, stdout)

    def test_commit_titles_cannot_mention_or_link_in_chat(self):
        night = nightly("red", [result("micro", "regression", [UNWIND])])
        commits = post.CommitRange([post.Commit("1" * 40, "fix <users/all> <https://evil.example|x>", 12)], 1)
        with unittest.mock.patch.object(post, "commits_since_green", return_value=commits):
            text = self.chat(night)
        self.assertNotIn("<users/all>", text)
        self.assertNotIn("<https://evil.example", text)

    def test_a_link_whose_url_could_break_out_renders_as_plain_text(self):
        for url in ("https://x.example/a|b", "https://x.example/<users/all>", "https://x.example/a b",
                    "javascript:alert(1)", "https://x.example/a)b"):
            with self.subTest(url=url):
                lines = [post.Line([post.Link(url, "run")])]
                self.assertEqual(post.render_chat(lines), "run")
                self.assertEqual(post.render_markdown(lines), "run  \n")

    def test_markdown_escapes_result_text(self):
        lines = [post.Line(["a *b* _c_ <script> [x](y) | #1"])]
        self.assertEqual(post.render_markdown(lines), r"a \*b\* \_c\_ &lt;script\> \[x\](y) \| \#1  " + "\n")


class BrokenOnBothSides(Workspace):
    """Contract v1.4: a correctness row at 0 on both sides is `info`, not a regression."""

    SOAK = result("soak", "pass", findings=[
        finding("edge count matches", "invariant", 0, 0, None, None, "lower_is_worse", "info"),
        finding("3-hop read", "completed", 0, 0, None, None, "lower_is_worse", "info"),
        finding("ingest rate", "throughput", 190, 191, 0.99, 0.8, "lower_is_worse", "ok")])

    def test_green_stays_one_line_in_chat_and_the_summary_names_them(self):
        summary = self.directory / "summary.md"
        code, stdout, _ = self.run_post(nightly("green", [result("micro", "pass"), self.SOAK]),
                                        {"GITHUB_STEP_SUMMARY": str(summary)})
        self.assertEqual(code, 0)
        self.assertEqual(stdout, (
            "*Perf nightly 2026-11-24: green* — main@4c10889 vs 6.2.0, all 2 checks passed; 2 rows broken on both "
            f"sides (not new). <{RUN_URL}|run>\n"))
        self.assertEqual(summary.read_text(), (
            "**Perf nightly 2026-11-24: green** — main@4c10889 vs 6.2.0, all 2 checks passed; 2 rows broken on both "
            f"sides (not new). [run]({RUN_URL})  \n"
            "**Broken on both sides** (already broken on the baseline, not new):  \n\n"
            "- soak: invariant broken on both sides: edge count matches\n"
            "- soak: 3-hop read: did not finish on either side\n\n"))

    def test_a_red_night_counts_them_in_chat_and_never_as_regressions(self):
        night = nightly("red", [result("micro", "regression", [UNWIND]), self.SOAK])
        with unittest.mock.patch.object(post, "commits_since_green", return_value=post.CommitRange([], 0, "x")):
            text = self.chat(night)
        self.assertIn("2 rows broken on both sides (not new); named in the run's job summary.", text.splitlines())
        self.assertIn("1 of 2 checks regressed.", text.splitlines()[0])
        self.assertNotIn("edge count matches", text)


class ChatBudget(Workspace):
    def test_a_huge_red_night_fits_and_keeps_what_matters(self):
        checks = [result(f"check{c}", "regression", confirmed=True, findings=[
            finding(f"a long query name number {i} " + "x" * 60, "instructions", 10 + i, 1, 10.0 + i, 1.05)
            for i in range(50)]) for c in range(6)]
        commits = post.CommitRange([post.Commit(f"{i:040x}", "a very long commit title " + "y" * 120, 4000 + i)
                                    for i in range(300)], 300)
        with unittest.mock.patch.object(post, "commits_since_green", return_value=commits):
            summary = self.directory / "summary.md"
            code, stdout, _ = self.run_post(nightly("red", checks), {"GITHUB_STEP_SUMMARY": str(summary)})
        self.assertEqual(code, 0)
        self.assertLessEqual(len(stdout.rstrip("\n")), post.CHAT_CHARACTER_BUDGET)
        for c in range(6):
            self.assertIn(f"*check{c}* (confirmed)", stdout)
        self.assertIn("|open an issue>", stdout)
        self.assertIn("Commits since the last green night (300,", stdout)
        self.assertGreater(len(summary.read_text()), len(stdout), "the job summary keeps more than Chat")

    def test_a_trimmed_red_night_keeps_the_run_commit_and_issue_links_and_the_budget(self):
        checks = [result(f"check-{c}", "regression", [UNWIND], summary="s" * 290) for c in range(4)]
        checks += [{"check": f"stub-{c}", "verdict": "error", "error": "e" * 400} for c in range(14)]
        with unittest.mock.patch.object(post, "commits_since_green", return_value=post.CommitRange([], 0, "x")):
            text = self.chat(nightly("red", checks)).rstrip("\n")
        self.assertLessEqual(len(text), post.CHAT_CHARACTER_BUDGET)
        last = text.splitlines()[-1]
        compare = f"https://github.com/FalkorDB/FalkorDB/compare/{PREVIOUS_GREEN}...{CANDIDATE_COMMIT}"
        self.assertTrue(last.startswith(f"(trimmed: the full report is in the run's job summary) <{RUN_URL}|run> · "
                                        f"<{compare}|commits since the last green night> · "
                                        "<https://github.com/FalkorDB/FalkorDB/issues/new?"), last)
        self.assertTrue(last.endswith("|open an issue>"), last)

    def test_a_trimmed_red_night_without_a_commit_range_has_no_compare_link(self):
        checks = [result(f"check-{c}", "regression", [UNWIND], summary="s" * 290) for c in range(4)]
        checks += [{"check": f"stub-{c}", "verdict": "error", "error": "e" * 400} for c in range(14)]
        for previous in (None, CANDIDATE_COMMIT):
            with self.subTest(previous=previous):
                night = nightly("red", checks, previous_green=previous)
                with unittest.mock.patch.object(post, "commits_since_green",
                                                return_value=post.CommitRange([], 0, "x")):
                    last = self.chat(night).rstrip("\n").splitlines()[-1]
                self.assertNotIn("/compare/", last)
                self.assertIn("|open an issue>", last)

    def test_the_cap_holds_for_many_shapes_of_night(self):
        for regressed in range(7):
            for errored in range(0, 40, 3):
                for summary in (0, 150, 300):
                    checks = [result(f"r-{c}", "regression", [UNWIND] * 5, summary="s" * max(1, summary))
                              for c in range(regressed)]
                    checks += [{"check": f"e-{c}", "verdict": "error", "error": "e" * 350} for c in range(errored)]
                    checks += [result("p", "pass")]
                    status = "red" if regressed else "error"
                    night = post.read_night(self.write(nightly(status, checks)), {})
                    commits = post.CommitRange([post.Commit("1" * 40, "t" * 120, 1)] * 30, 30)
                    text = post.chat_text(night, commits, {})
                    self.assertLessEqual(len(text), post.CHAT_CHARACTER_BUDGET, (regressed, errored, summary))

    def test_fifty_long_check_names_keep_the_headline_and_a_short_issue_link(self):
        # Contract v1.4 allows check names of 43 characters; a title listing 50 of them
        # used to make the issue link alone longer than the whole budget.
        names = [f"check-{index:02d}-" + "x" * 34 for index in range(50)]
        self.assertEqual({len(name) for name in names}, {43})
        night = nightly("red", [result(name, "regression", [UNWIND]) for name in names])
        with unittest.mock.patch.object(post, "commits_since_green", return_value=post.CommitRange([], 0, "x")):
            text = self.chat(night).rstrip("\n")
        self.assertLessEqual(len(text), post.CHAT_CHARACTER_BUDGET)
        self.assertTrue(text.startswith("*Perf nightly 2026-11-24: REGRESSION* — main@4c10889 vs 6.2.0: "
                                        "50 of 50 checks regressed."), text[:200])
        last = text.splitlines()[-1]
        self.assertTrue(last.endswith("|open an issue>"), last[-200:])
        url = last.split(" · <", 1)[1].rsplit("|open an issue>", 1)[0]
        self.assertLessEqual(len(url), post.CHAT_LIMITS[-1].issue_url_characters)
        title = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["title"][0]
        self.assertLessEqual(len(title), 220)
        self.assertRegex(title, r"regression in (check-\d\d-x+(, )?)+ and \d+ more|regression in 50 checks")

    def test_the_issue_title_summarizes_long_check_names_even_when_the_url_has_room(self):
        names = [f"check-{index:02d}-" + "x" * 34 for index in range(4)]
        night = post.read_night(self.write(nightly("red", [result(name, "regression", [UNWIND]) for name in names])), {})
        url = post.issue_link(night, {}, post.SUMMARY_LIMITS)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        self.assertEqual(query["title"][0], f"perf: nightly 2026-11-24 regression in {names[0]}, {names[1]} and 2 more "
                                            "(main@4c10889 vs 6.2.0)")
        self.assertIn(f"found a regression in {names[0]}, {names[1]} and 2 more.", query["body"][0])
        self.assertIn(names[3], query["body"][0], "the findings in the body still name every check")

    def test_an_issue_link_too_long_for_half_the_budget_is_left_out_of_the_trimmed_tail(self):
        checks = [result(f"check-{c}", "regression", [UNWIND], summary="s" * 290) for c in range(4)]
        checks += [{"check": f"stub-{c}", "verdict": "error", "error": "e" * 400} for c in range(14)]
        night = post.read_night(self.write(nightly("red", checks)), {})
        huge = "https://github.com/FalkorDB/FalkorDB/issues/new?title=" + "x" * 3000
        with unittest.mock.patch.object(post, "issue_link", return_value=huge):
            text = post.chat_text(night, post.CommitRange([], 0, "x"), {})
        self.assertLessEqual(len(text), post.CHAT_CHARACTER_BUDGET)
        self.assertTrue(text.startswith("*Perf nightly 2026-11-24: REGRESSION*"), text[:100])
        compare = f"https://github.com/FalkorDB/FalkorDB/compare/{PREVIOUS_GREEN}...{CANDIDATE_COMMIT}"
        self.assertTrue(text.endswith(f"(trimmed: the full report is in the run's job summary) <{RUN_URL}|run> · "
                                      f"<{compare}|commits since the last green night>"), text[-300:])

    def test_summarize_names(self):
        self.assertEqual(post.summarize_names(["micro", "pokec"], 100), "micro, pokec")
        self.assertEqual(post.summarize_names(["a" * 40, "b" * 40, "c" * 40], 100), f"{'a' * 40}, {'b' * 40} and 1 more")
        self.assertEqual(post.summarize_names(["a" * 200, "b"], 100), "2 checks")
        self.assertEqual(post.summarize_names([], 100), "")

    def test_the_cap_counts_a_kept_blank_line(self):
        # A blank line just before the cut used to be stripped when measuring but kept
        # in the final text: one character over. Sweep the first line across the edge.
        night = post.read_night(self.write(nightly("error", [{"check": "e", "verdict": "error", "error": "x"}])), {})
        for length in range(post.CHAT_CHARACTER_BUDGET - 150, post.CHAT_CHARACTER_BUDGET + 1):
            lines = [post.Line(["a" * length]), post.BLANK, post.Line(["b" * 5000])]
            with unittest.mock.patch.object(post, "build_message", return_value=lines):
                text = post.chat_text(night, None, {})
            self.assertLessEqual(len(text), post.CHAT_CHARACTER_BUDGET, length)

    def test_long_free_text_is_shortened_and_the_message_still_fits(self):
        checks = [{"check": f"check-{index}", "verdict": "error", "error": "x" * 40_000} for index in range(30)]
        text = self.chat(nightly("error", checks))
        self.assertLessEqual(len(text.rstrip("\n")), post.CHAT_CHARACTER_BUDGET)
        self.assertTrue(text.startswith("*Perf nightly 2026-11-24: ERROR* — main@4c10889 vs 6.2.0: 30 of 30 checks"))
        self.assertTrue(text.rstrip("\n").endswith(
            f"(trimmed: the full report is in the run's job summary) <{RUN_URL}|run>"), text[-300:])


# ---------------------------------------------------------------------------
# perf-judge's fixtures (tests/perf/fixtures/).


def fixture_files(kind: str) -> list[Path]:
    return sorted((FIXTURES / kind).glob("nightly*.json"))


class JudgeFixtures(Workspace):
    def setUp(self):
        super().setUp()
        if not FIXTURES.is_dir():
            self.fail(f"perf-judge's fixtures are missing: {FIXTURES} (set PERF_JUDGE_DIR)")
        self.enter(unittest.mock.patch.object(post, "commits_since_green",
                                            return_value=post.CommitRange([], 0, "not looked up in tests")))

    def test_valid_nightlies_show_their_status_and_every_check_that_did_not_pass(self):
        files = fixture_files("valid")
        self.assertTrue(files, f"no nightly*.json in {FIXTURES / 'valid'}")
        for path in files:
            with self.subTest(fixture=path.name):
                data = json.loads(path.read_text())
                text = self.chat(path)
                first = text.splitlines()[0]
                self.assertTrue(first.startswith(f"*Perf nightly {data['date']}: {HEADLINE_WORD[data['status']]}*"),
                                first)
                self.assertNotIn("Problems with nightly.json", text)
                self.assertIn(data["candidate"]["label"], first)
                self.assertIn(data["baseline"]["label"], first)
                self.assertIn(f"<{data['run_url']}|run>", text)
                for check in data["checks"]:
                    if check["verdict"] == "error":
                        self.assertIn(f"- *{check['check']}*: {' '.join(check['error'].split())}", text)
                    if check["verdict"] == "regression":
                        self.assertIn(f"*{check['check']}* (", text)
                if data["status"] in ("green", "skipped"):
                    self.assertEqual(len(text.splitlines()), 1)
                if data["status"] == "green":
                    self.assertIn(f"all {len(data['checks'])} check", first)
                if data["status"] == "skipped":
                    self.assertIn(data["skip_reason"], first)

    def test_invalid_nightlies_are_never_green_and_name_every_contract_problem(self):
        cases = load_invalid_cases()
        nightlies = [case for case in cases.CASES if cases.kind(case) == "nightly"]
        self.assertTrue(nightlies, f"no nightly case in {INVALID_CASES}")
        for case in nightlies:
            with self.subTest(case=case.name):
                text = self.chat(self.write(cases.invalid_text(case), "invalid-nightly.json"))
                self.assertNotIn(": green*", text.splitlines()[0])
                self.assertNotIn(": skipped*", text.splitlines()[0])
                self.assertIn("*Problems with nightly.json:*", text)
                self.assertIn(f"contract: {case.code} at {case.at}", text)

    def test_each_valid_result_reads_right_as_a_one_check_night(self):
        files = sorted((FIXTURES / "valid").glob("result*.json"))
        self.assertTrue(files, f"no result*.json in {FIXTURES / 'valid'}")
        for path in files:
            with self.subTest(fixture=path.name):
                check = json.loads(path.read_text())
                status = {"pass": "green", "regression": "red", "error": "error"}[check["verdict"]]
                night = {"schema": 1, "date": "2026-11-24", "run_url": RUN_URL, "candidate": check["candidate"],
                         "baseline": check["baseline"], "previous_green_commit": None, "status": status,
                         "skip_reason": None, "checks": [check]}
                text = self.chat(night)
                self.assertTrue(text.startswith(f"*Perf nightly 2026-11-24: {HEADLINE_WORD[status]}*"), text)
                self.assertNotIn("Problems with nightly.json", text)
                if check["verdict"] == "error":
                    self.assertIn(f"- *{check['check']}*: {' '.join(check['error'].split())}", text)
                if check["verdict"] == "regression":
                    worst = post.regression_findings(post.read_check(0, check))[0]
                    self.assertIn(f"- {post.describe_finding(worst)}", text)
                    self.assertNotIn("None", text)
                both = [f for f in check["findings"] if f["status"] == "info" and f["candidate"] == 0
                        and f["baseline"] == 0 and f["metric"] in ("completed", "invariant")]
                if both:
                    noun = "row" if len(both) == 1 else "rows"
                    self.assertIn(f"{len(both)} {noun} broken on both sides (not new)", text)
                    self.assertNotIn("REGRESSION", text)


# ---------------------------------------------------------------------------
# The missed-run alarm.


NIGHT = date(2026, 11, 24)
NOW = datetime(2026, 11, 24, 5, 0, 7, tzinfo=timezone.utc)
RUNS_PATH = "/repos/FalkorDB/FalkorDB/actions/workflows/perf-nightly.yml/runs"


def run(identifier: int, created: str, status: str = "completed", conclusion: str | None = "success",
        branch: str = "main", event: str = "schedule") -> dict:
    return {"id": identifier, "created_at": created, "run_started_at": created, "status": status,
            "conclusion": conclusion, "head_branch": branch, "event": event,
            "html_url": f"https://github.com/FalkorDB/FalkorDB/actions/runs/{identifier}"}


def artifacts(*names: str, expired: bool = False) -> dict:
    return {"total_count": len(names), "artifacts": [{"name": name, "expired": expired} for name in names]}


class MissedRun(Workspace):
    def alarm(self, routes: dict, arguments: list | None = None, environ: dict | None = None):
        with StubServer(routes) as server:
            environ = {"GITHUB_API_URL": server.url, **(environ or {})}
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = missed_run.main(["--date", NIGHT.isoformat(), *(arguments or [])], environ, now=NOW)
        return code, stdout.getvalue(), stderr.getvalue(), server.requests

    def test_quiet_when_tonights_run_has_the_artifact(self):
        code, stdout, stderr, requests = self.alarm({
            RUNS_PATH: (200, {"workflow_runs": [run(7, "2026-11-24T00:03:11Z")]}),
            "/repos/FalkorDB/FalkorDB/actions/runs/7/artifacts": (200, artifacts("perf-nightly")),
        }, environ={"PERF_CHAT_WEBHOOK": "http://127.0.0.1:9/never"})
        self.assertEqual(code, 0)
        self.assertNotIn("Chat", stderr)
        self.assertIn("has the perf-nightly artifact; no alarm.", stdout)
        query = urllib.parse.parse_qs(requests[0]["query"])
        self.assertEqual(query["branch"], ["main"])
        self.assertEqual(query["event"], ["schedule"])
        self.assertEqual(query["created"], ["2026-11-23T05:00:00Z..2026-11-24T05:00:00Z"])
        self.assertEqual(urllib.parse.parse_qs(requests[1]["query"])["name"], ["perf-nightly"])

    def test_quiet_on_a_skipped_night_too(self):
        # A skipped night still uploads perf-nightly (status "skipped"); that is a result.
        code, _, _, _ = self.alarm({
            RUNS_PATH: (200, {"workflow_runs": [run(8, "2026-11-24T00:01:00Z")]}),
            "/repos/FalkorDB/FalkorDB/actions/runs/8/artifacts": (200, artifacts("perf-nightly")),
        })
        self.assertEqual(code, 0)

    def test_no_run(self):
        code, stdout, _, _ = self.alarm({RUNS_PATH: (200, {"workflow_runs": []})})
        self.assertEqual(code, 1)
        self.assertIn("*Perf nightly alarm 2026-11-24*: no nightly result. No schedule run of perf-nightly.yml on main "
                      "between 2026-11-23 05:00 and 2026-11-24 05:00 UTC.", stdout)

    def test_runs_outside_the_window_or_off_main_do_not_count(self):
        code, stdout, _, _ = self.alarm({RUNS_PATH: (200, {"workflow_runs": [
            run(1, "2026-11-24T05:00:00Z"),  # the window ends at 05:00, exclusive
            run(2, "2026-11-23T04:59:59Z"),
            run(3, "2026-11-24T00:00:05Z", branch="ci/perf-nightly"),
        ]})})
        self.assertEqual(code, 1)
        self.assertIn("No schedule run of perf-nightly.yml on main", stdout)

    def test_a_hand_started_run_with_the_artifact_does_not_stand_in_for_the_night(self):
        # A dispatch given its own images uploads nightly.json but posts nothing.
        code, stdout, _, _ = self.alarm({
            RUNS_PATH: (200, {"workflow_runs": [run(4, "2026-11-23T14:00:00Z", event="workflow_dispatch")]}),
            "/repos/FalkorDB/FalkorDB/actions/runs/4/artifacts": (200, artifacts("perf-nightly")),
        })
        self.assertEqual(code, 1)
        self.assertIn("No schedule run of perf-nightly.yml on main", stdout)

    def test_a_run_still_going(self):
        code, stdout, _, _ = self.alarm({
            RUNS_PATH: (200, {"workflow_runs": [run(9, "2026-11-23T05:00:00Z", "in_progress", None)]}),
            "/repos/FalkorDB/FalkorDB/actions/runs/9/artifacts": (200, artifacts()),
        })
        self.assertEqual(code, 1)
        self.assertIn("The run (started 2026-11-23 05:00 UTC) is still in_progress at 2026-11-24 05:00 UTC and has "
                      "not uploaded perf-nightly. <https://github.com/FalkorDB/FalkorDB/actions/runs/9|run>", stdout)

    def test_a_failed_run_without_the_artifact(self):
        code, stdout, _, _ = self.alarm({
            RUNS_PATH: (200, {"workflow_runs": [run(10, "2026-11-24T00:00:09Z", conclusion="failure")]}),
            "/repos/FalkorDB/FalkorDB/actions/runs/10/artifacts": (200, artifacts("perf-result-micro")),
        })
        self.assertEqual(code, 1)
        self.assertIn("The run finished `failure` without uploading perf-nightly.", stdout)

    def test_an_expired_artifact_does_not_count(self):
        code, _, _, _ = self.alarm({
            RUNS_PATH: (200, {"workflow_runs": [run(11, "2026-11-24T00:00:09Z")]}),
            "/repos/FalkorDB/FalkorDB/actions/runs/11/artifacts": (200, artifacts("perf-nightly", expired=True)),
        })
        self.assertEqual(code, 1)

    def test_any_run_in_the_window_with_the_artifact_is_enough(self):
        code, _, _, _ = self.alarm({
            RUNS_PATH: (200, {"workflow_runs": [run(12, "2026-11-24T02:00:00Z", conclusion="cancelled"),
                                                run(13, "2026-11-24T00:00:00Z")]}),
            "/repos/FalkorDB/FalkorDB/actions/runs/12/artifacts": (200, artifacts()),
            "/repos/FalkorDB/FalkorDB/actions/runs/13/artifacts": (200, artifacts("perf-nightly")),
        })
        self.assertEqual(code, 0)

    def test_several_runs_none_with_the_artifact_reports_the_newest(self):
        code, stdout, _, _ = self.alarm({
            RUNS_PATH: (200, {"workflow_runs": [run(14, "2026-11-24T00:00:00Z", conclusion="failure"),
                                                run(15, "2026-11-24T02:00:00Z", conclusion="cancelled")]}),
            "/repos/FalkorDB/FalkorDB/actions/runs/14/artifacts": (200, artifacts()),
            "/repos/FalkorDB/FalkorDB/actions/runs/15/artifacts": (200, artifacts()),
        })
        self.assertEqual(code, 1)
        self.assertIn("finished `cancelled` without uploading perf-nightly. (2 runs between", stdout)
        self.assertIn("runs/15|run>", stdout)

    def test_a_missing_workflow_is_loud(self):
        code, stdout, _, _ = self.alarm({})
        self.assertEqual(code, 1)
        self.assertIn("perf-nightly.yml was not found in FalkorDB/FalkorDB.", stdout)

    def test_not_being_able_to_ask_is_an_alarm(self):
        code, stdout, _, _ = self.alarm({RUNS_PATH: (503, {"message": "unavailable"})})
        self.assertEqual(code, 1)
        self.assertIn("Could not ask GitHub for perf-nightly.yml runs (HTTP 503", stdout)

    def test_the_alarm_is_posted_and_quiet_is_not(self):
        with StubServer({"/hook": (200, {})}) as chat:
            hook = {"PERF_CHAT_WEBHOOK": f"{chat.url}/hook"}
            fired, _, _, _ = self.alarm({RUNS_PATH: (200, {"workflow_runs": []})}, environ=hook)
            quiet, _, _, _ = self.alarm({
                RUNS_PATH: (200, {"workflow_runs": [run(7, "2026-11-24T00:03:11Z")]}),
                "/repos/FalkorDB/FalkorDB/actions/runs/7/artifacts": (200, artifacts("perf-nightly")),
            }, environ=hook)
        self.assertEqual((fired, quiet), (1, 0))
        self.assertEqual(len(chat.requests), 1)
        self.assertIn("Perf nightly alarm 2026-11-24", json.loads(chat.requests[0]["body"])["text"])

    def test_workflow_branch_and_artifact_are_injectable(self):
        code, _, _, requests = self.alarm({
            "/repos/FalkorDB/FalkorDB/actions/workflows/rust-push.yml/runs": (
                200, {"workflow_runs": [run(16, "2026-11-24T01:00:00Z", branch="dev", event="push")]}),
            "/repos/FalkorDB/FalkorDB/actions/runs/16/artifacts": (200, artifacts("coverage-binary")),
        }, ["--workflow", "rust-push.yml", "--branch", "dev", "--event", "push", "--artifact", "coverage-binary"])
        self.assertEqual(code, 0)
        query = urllib.parse.parse_qs(requests[0]["query"])
        self.assertEqual((query["branch"], query["event"]), (["dev"], ["push"]))

    def test_a_rerun_of_a_scheduled_check_checks_the_night_it_was_created_for(self):
        routes = {RUNS_PATH: (200, {"workflow_runs": []}),
                  "/repos/FalkorDB/FalkorDB/actions/runs/99": (200, {"id": 99, "created_at": "2026-11-24T05:03:00Z"})}
        with StubServer(routes) as server:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                missed_run.main([], {"GITHUB_API_URL": server.url, "GITHUB_EVENT_NAME": "schedule",
                                     "GITHUB_RUN_ID": "99"},
                                now=datetime(2026, 11, 26, 9, 30, 0, tzinfo=timezone.utc))
        runs = [request for request in server.requests if request["path"] == RUNS_PATH]
        self.assertEqual(urllib.parse.parse_qs(runs[0]["query"])["created"],
                         ["2026-11-23T05:00:00Z..2026-11-24T05:00:00Z"])

    def test_today_when_the_run_cannot_be_looked_up_or_was_not_scheduled(self):
        for environ in ({"GITHUB_EVENT_NAME": "schedule", "GITHUB_RUN_ID": "98"},  # lookup answers 404
                        {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_RUN_ID": "99"},
                        # Not a number: never put in the URL (here it would reach runs/99).
                        {"GITHUB_EVENT_NAME": "schedule", "GITHUB_RUN_ID": "99?x=1"}):
            with self.subTest(environ=environ):
                routes = {RUNS_PATH: (200, {"workflow_runs": []}),
                          "/repos/FalkorDB/FalkorDB/actions/runs/99": (200, {"created_at": "2026-11-20T05:03:00Z"})}
                with StubServer(routes) as server:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        missed_run.main([], {"GITHUB_API_URL": server.url, **environ},
                                        now=datetime(2026, 11, 26, 9, 30, 0, tzinfo=timezone.utc))
                runs = [request for request in server.requests if request["path"] == RUNS_PATH]
                self.assertEqual(urllib.parse.parse_qs(runs[0]["query"])["created"],
                                 ["2026-11-25T05:00:00Z..2026-11-26T05:00:00Z"])

    def test_an_explicit_date_wins_over_the_scheduled_night(self):
        routes = {RUNS_PATH: (200, {"workflow_runs": []}),
                  "/repos/FalkorDB/FalkorDB/actions/runs/99": (200, {"created_at": "2026-11-24T05:03:00Z"})}
        with StubServer(routes) as server:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                missed_run.main(["--date", "2026-11-10"], {"GITHUB_API_URL": server.url,
                                                           "GITHUB_EVENT_NAME": "schedule", "GITHUB_RUN_ID": "99"},
                                now=datetime(2026, 11, 26, 9, 30, 0, tzinfo=timezone.utc))
        self.assertEqual([request["path"] for request in server.requests], [RUNS_PATH])
        self.assertEqual(urllib.parse.parse_qs(server.requests[0]["query"])["created"],
                         ["2026-11-09T05:00:00Z..2026-11-10T05:00:00Z"])

    def test_the_default_night_is_today_in_utc(self):
        with StubServer({RUNS_PATH: (200, {"workflow_runs": []})}) as server:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                missed_run.main([], {"GITHUB_API_URL": server.url},
                                now=datetime(2026, 11, 25, 5, 0, 3, tzinfo=timezone.utc))
        self.assertEqual(urllib.parse.parse_qs(server.requests[0]["query"])["created"],
                         ["2026-11-24T05:00:00Z..2026-11-25T05:00:00Z"])


class CommandLine(unittest.TestCase):
    """The scripts run as the workflows run them: `python -I <path>`."""

    def test_post_runs_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nightly.json"
            path.write_text(json.dumps(nightly("green", [result("micro", "pass")])))
            completed = subprocess.run(
                [sys.executable, "-I", str(HERE / "post.py"), "--nightly", str(path)],
                capture_output=True, text=True, check=False, cwd=directory, env={"PATH": os.environ.get("PATH", "")})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(completed.stdout.startswith("*Perf nightly 2026-11-24: green*"))

    def test_missed_run_imports_post_when_isolated(self):
        completed = subprocess.run(
            [sys.executable, "-I", str(HERE / "missed_run.py"), "--help"],
            capture_output=True, text=True, check=False, env={"PATH": os.environ.get("PATH", "")})
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
