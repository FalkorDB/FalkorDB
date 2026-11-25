"""Tests for falkorbench.nightly.report and .missed_run: one test per rule, each failing when its rule breaks.

`uv run --project bench --group dev python -I -B -m pytest bench/tests/test_nightly.py`; see bench/README.md.
"""

from __future__ import annotations

import ast
import contextlib
import http.client
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
from datetime import UTC
from datetime import date
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

from click.testing import CliRunner

from falkorbench import nightly as nightly_package
from falkorbench.cli import cli
from falkorbench.nightly import missed_run
from falkorbench.nightly import report

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FIXTURES = HERE / "fixtures"

CANDIDATE_IMAGE = (
    "ghcr.io/falkordb/falkordb-server@sha256:"
    "95b610a600682533d854b6ece6bddbea80b49d0b89e2cd0b2cf19bd0a8142aed"
)
BASELINE_IMAGE = "falkordb/falkordb-server@sha256:44eb4a6d48ef6f2b06aeefea46c5f4c17cd2310d3832135be18fd7fc8ba30923"
CANDIDATE_COMMIT = "4c1088949a5852e3949c05bfb3d92a603c4cd4ca"
PREVIOUS_GREEN = "41d2f20ce9047eb178378fe6440241c73278e6f0"
RUN_URL = "https://github.com/FalkorDB/FalkorDB/actions/runs/1000"
WEBHOOK_SECRET = "key=SECRETKEY123&token=SECRETTOKEN456"


def result(
    check: str,
    verdict: str,
    findings: list | None = None,
    summary: str = "one line",
    error: str | None = None,
) -> dict:
    """A schema-1 result.json, as in the contract's §2 example."""
    return {
        "schema": 1,
        "check": check,
        "verdict": verdict,
        "confirmed": verdict == "regression",
        "summary": summary,
        "candidate": {
            "image": CANDIDATE_IMAGE,
            "commit": CANDIDATE_COMMIT,
            "label": "main@4c10889",
        },
        "baseline": {"image": BASELINE_IMAGE, "commit": None, "label": "6.2.0"},
        "machine": {"runner": "gce", "type": "n4-standard-16", "cpu": "Intel Xeon", "cores": 16},
        "started": "2026-11-24T00:05:00Z",
        "finished": "2026-11-24T00:41:00Z",
        "findings": findings or [],
        "error": error,
        "artifacts": [],
    }


def finding(
    name: str, metric: str, candidate, baseline, ratio, threshold, status: str = "regression"
) -> dict:
    return {
        "name": name,
        "metric": metric,
        "candidate": candidate,
        "baseline": baseline,
        "ratio": ratio,
        "threshold": threshold,
        "direction": "higher_is_worse",
        "status": status,
    }


UNWIND = finding("UNWIND id seek @200k", "p50_ms", 13.3, 0.4, 33.25, 1.5)
STUB = {"check": "pokec", "verdict": "error", "error": "no result.json"}


def nightly(
    status: str,
    checks: list,
    skip_reason: str | None = None,
    previous_green: str | None = PREVIOUS_GREEN,
    candidate_commit: str | None = CANDIDATE_COMMIT,
) -> dict:
    """A schema-1 nightly.json, as in the contract's §3 example."""
    return {
        "schema": 1,
        "date": "2026-11-24",
        "run_url": RUN_URL,
        "candidate": {
            "image": CANDIDATE_IMAGE,
            "commit": candidate_commit,
            "label": "main@4c10889",
        },
        "baseline": {"image": BASELINE_IMAGE, "commit": None, "label": "6.2.0"},
        "previous_green_commit": previous_green,
        "status": status,
        "skip_reason": skip_reason,
        "checks": checks,
    }


GREEN = nightly("green", [result("micro-hw", "pass"), result("size-probe", "pass")])
RED = nightly(
    "red", [result("micro-hw", "regression", [UNWIND]), STUB, result("size-probe", "pass")]
)
ERROR = nightly(
    "error",
    [result("micro-hw", "pass"), STUB, result("soak-a", "error", error="server did not start")],
)
SKIPPED = nightly("skipped", [], skip_reason="main has not moved since 4c10889")


class Workspace(unittest.TestCase):
    """Writes nightly.json files to a temporary directory and runs the reporter on them."""

    def setUp(self):
        self.directory = Path(self.enter(tempfile.TemporaryDirectory()))
        self.enter(unittest.mock.patch.object(report, "POST_RETRY_DELAYS", (0.0, 0.0)))
        # No commit lookups unless a test asks for them.
        self.enter(unittest.mock.patch.object(report, "CHECKOUT", self.directory))

    def enter(self, context):
        """TestCase.enterContext, which needs Python 3.11."""
        value = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return value

    def write(self, content, name: str = "nightly.json") -> Path:
        path = self.directory / name
        path.write_text(
            content if isinstance(content, str) else json.dumps(content), encoding="utf-8"
        )
        return path

    def run_post(self, content, environ: dict | None = None) -> tuple[int, str, str]:
        path = content if isinstance(content, Path) else self.write(content)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = report.run(path, dict(environ or {}))
        return code, stdout.getvalue(), stderr.getvalue()

    def summary(self, content, environ: dict | None = None) -> tuple[int, str, str]:
        """Exit code, Chat line and step summary for one nightly."""
        path = self.directory / "summary.md"
        path.write_text("", encoding="utf-8")
        code, stdout, _ = self.run_post(
            content, {**(environ or {}), "GITHUB_STEP_SUMMARY": str(path)}
        )
        return code, stdout.rstrip("\n"), path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# One line in Chat, and the exit code.


class ChatLineAndExitCode(Workspace):
    """One line per night, every night; exit 0 whenever the night was reported."""

    def test_green_is_one_line_with_a_details_link(self):
        code, line, _ = self.summary(GREEN)
        self.assertEqual(
            (code, line),
            (
                0,
                (
                    f"Perf nightly 2026-11-24: GREEN (2 checks), main@4c10889 vs 6.2.0 "
                    f"· <{RUN_URL}|details>"
                ),
            ),
        )

    def test_skipped_says_why(self):
        code, line, _ = self.summary(SKIPPED)
        self.assertEqual(
            (code, line),
            (
                0,
                (
                    "Perf nightly 2026-11-24: SKIPPED (main has not moved since 4c10889), "
                    f"main@4c10889 vs 6.2.0 · <{RUN_URL}|details>"
                ),
            ),
        )

    def test_red_names_the_regressed_checks_and_adds_a_new_issue_link(self):
        code, line, _ = self.summary(RED)
        self.assertEqual(code, 0)
        self.assertTrue(
            line.startswith(
                f"Perf nightly 2026-11-24: RED (micro-hw), main@4c10889 vs 6.2.0 "
                f"· <{RUN_URL}|details> · <https://github.com/FalkorDB/FalkorDB/issues/new?"
            ),
            line,
        )
        self.assertTrue(line.endswith("|new issue>"), line)

    def test_error_names_the_checks_that_could_not_measure(self):
        code, line, _ = self.summary(ERROR)
        self.assertEqual(
            (code, line),
            (
                0,
                (
                    "Perf nightly 2026-11-24: ERROR (pokec, soak-a), main@4c10889 vs 6.2.0 "
                    f"· <{RUN_URL}|details>"
                ),
            ),
        )

    def test_only_a_red_night_gets_a_new_issue_link(self):
        for night in (GREEN, SKIPPED, ERROR):
            with self.subTest(status=night["status"]):
                _, line, _ = self.summary(night)
                self.assertNotIn("new issue", line)
                self.assertNotIn("/issues/new", line)
                self.assertTrue(line.endswith(f"· <{RUN_URL}|details>"), line)

    def test_the_status_is_trusted_as_merge_py_wrote_it(self):
        # merge.py validated nightly.json and set the status; the reporter does not re-derive it.
        code, line, _ = self.summary(nightly("green", [result("micro-hw", "pass"), STUB]))
        self.assertEqual(code, 0)
        self.assertTrue(line.startswith("Perf nightly 2026-11-24: GREEN (2 checks)"), line)


class NewIssueLink(Workspace):
    def issue(self, night: dict) -> dict:
        _, line, _ = self.summary(night)
        url = line.split(" · <")[-1].removesuffix("|new issue>")
        self.assertLessEqual(len(url), report.ISSUE_URL_CHARACTERS)
        return {
            key: values[0]
            for key, values in urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).items()
        }

    def test_it_is_prefilled_from_nightly_json_only(self):
        night = nightly(
            "red",
            [
                result("micro-hw", "regression", [UNWIND], summary="2 rows slower, worst 33x"),
                result("size-probe", "regression", [UNWIND], summary="UNWIND 1.6x at 500k"),
                STUB,
                result("soak-a", "pass", summary="steady"),
            ],
        )
        issue = self.issue(night)
        self.assertEqual(issue["title"], "Perf regression 2026-11-24: micro-hw, size-probe")
        self.assertEqual(issue["labels"], "performance,perf-regression")
        self.assertEqual(
            issue["body"],
            (
                f"main@4c10889 vs 6.2.0\nRun: {RUN_URL}\n\n"
                "- micro-hw: 2 rows slower, worst 33x\n- size-probe: UNWIND 1.6x at 500k"
            ),
        )

    def test_a_long_body_is_cut_and_the_link_kept(self):
        checks = [
            result(f"check-{index}", "regression", [UNWIND], summary="s" * 290)
            for index in range(30)
        ]
        issue = self.issue(nightly("red", checks))
        self.assertTrue(
            issue["body"].startswith(f"main@4c10889 vs 6.2.0\nRun: {RUN_URL}\n\n- check-0: sss")
        )
        self.assertTrue(issue["body"].endswith("(more on the run page)"), issue["body"][-80:])
        self.assertLessEqual(len(issue["title"]), 150)

    def test_the_run_page_offers_the_same_issue(self):
        _, line, summary = self.summary(RED)
        url = line.split(" · <")[-1].removesuffix("|new issue>")
        self.assertIn(f"[Open an issue for this regression]({url})", summary)


class GuardsThatNeedNoValidator(Workspace):
    """The only cases where the report exits 1: it could not report the night as written."""

    def test_a_file_that_cannot_be_read_is_an_error(self):
        cases = {
            "": "could not read nightly.json",
            "{": "could not read nightly.json",
            "[]": "nightly.json is not a JSON object",
            "null": "nightly.json is not a JSON object",
            "1": "nightly.json is not a JSON object",
        }
        for content, why in cases.items():
            with self.subTest(content=content):
                code, line, summary = self.summary(content)
                self.assertEqual(code, 1)
                self.assertTrue(line.startswith(f"Perf nightly date unknown: ERROR ({why}"), line)
                self.assertIn("**Not green:**", summary)
        code, stdout, _ = self.run_post(self.directory / "absent.json", {"GITHUB_RUN_ID": "77"})
        self.assertEqual(code, 1)
        self.assertIn("ERROR (could not read absent.json", stdout)
        self.assertIn("· <https://github.com/FalkorDB/FalkorDB/actions/runs/77|details>", stdout)

    def test_an_unknown_status_is_an_error(self):
        for status in ("GREEN", "bogus", None, 1):
            with self.subTest(status=status):
                code, line, _ = self.summary(nightly(status, [result("micro-hw", "pass")]))
                self.assertEqual(code, 1)
                self.assertTrue(
                    line.startswith(f"Perf nightly 2026-11-24: ERROR (unknown status {status!r}"),
                    line,
                )

    def test_a_crash_inside_the_reporter_still_reports_and_exits_1(self):
        with unittest.mock.patch.object(
            report, "summary_markdown", side_effect=RuntimeError("boom <users/all>")
        ):
            code, stdout, stderr = self.run_post(GREEN, {"GITHUB_RUN_ID": "77"})
        self.assertEqual(code, 1)
        self.assertEqual(
            stdout,
            "Perf nightly: ERROR (the reporter crashed: RuntimeError: boom ‹users/all>) "
            "· <https://github.com/FalkorDB/FalkorDB/actions/runs/77|details>\n",
        )
        self.assertIn("Traceback", stderr)


# ---------------------------------------------------------------------------
# The run page: $GITHUB_STEP_SUMMARY.


class RunPage(Workspace):
    def test_every_check_has_a_row_with_its_verdict_and_its_summary_or_error(self):
        _, _, summary = self.summary(ERROR)
        self.assertIn("## Perf nightly 2026-11-24: ERROR", summary)
        self.assertIn(f"main@4c10889 vs 6.2.0 · [run]({RUN_URL})", summary)
        self.assertIn("| micro-hw | pass | one line |", summary)
        self.assertIn("| pokec | error | no result.json |", summary)
        self.assertIn("| soak-a | error | server did not start |", summary)

    def test_a_red_check_lists_its_worst_regression_rows(self):
        rows = [
            finding(f"query {index}", "instructions", 47_190_000, 39_250_000, 1.2023, 1.05)
            for index in range(6)
        ]
        rows += [
            finding("steady", "p99_ms", 1.0, 1.0, 1.0, 1.5, status="ok"),
            finding("no baseline", "p50_ms", 5, 0, None, 1.5),
        ]
        _, _, summary = self.summary(nightly("red", [result("micro-hw", "regression", rows)]))
        self.assertIn("**micro-hw**, worst regression rows (in the check's order):", summary)
        self.assertIn(
            "- query 0: instructions 47,190,000 vs 39,250,000 (1.20x, threshold 1.05x)", summary
        )
        self.assertIn("- query 4:", summary)
        self.assertNotIn("query 5", summary)
        self.assertIn("- 2 more in the check's result.json", summary)
        self.assertNotIn("steady", summary)
        self.assertNotIn("None", summary)

    def test_only_a_red_night_ends_with_an_issue_link(self):
        _, _, summary = self.summary(RED)
        last = summary.rstrip("\n").splitlines()[-1]
        self.assertTrue(
            last.startswith(
                "[Open an issue for this regression](https://github.com/FalkorDB/FalkorDB/"
                "issues/new?"
            ),
            last,
        )
        self.assertNotIn("Open an issue", self.summary(ERROR)[2])

    def test_a_skipped_night_shows_why(self):
        _, _, summary = self.summary(SKIPPED)
        self.assertIn("Skipped: main has not moved since 4c10889", summary)

    def test_the_summary_is_appended_not_overwritten(self):
        path = self.directory / "summary.md"
        path.write_text("earlier step\n", encoding="utf-8")
        self.run_post(GREEN, {"GITHUB_STEP_SUMMARY": str(path)})
        self.assertTrue(
            path.read_text().startswith("earlier step\n## Perf nightly 2026-11-24: GREEN")
        )


class TextSafety(Workspace):
    def test_text_from_the_file_cannot_mention_or_link_in_chat(self):
        night = nightly("red", [result("<users/all>", "regression", [UNWIND])])
        night["candidate"]["label"] = "<https://evil.example|click>"
        _, line, _ = self.summary(night)
        self.assertNotIn("<users/all>", line)
        self.assertNotIn("<https://evil", line)
        self.assertIn("‹users/all>", line)

    def test_an_unsafe_or_oversized_run_url_is_replaced_by_the_current_run(self):
        for bad in (
            "https://github.com/FalkorDB/FalkorDB/actions/runs/1|x><users/all>",
            "https://github.com/FalkorDB/FalkorDB/actions/runs/1?" + "a" * 400,
            "http://plain.example/run",
        ):
            for environ, link in (
                (
                    {"GITHUB_RUN_ID": "5"},
                    "· <https://github.com/FalkorDB/FalkorDB/actions/runs/5|details>",
                ),
                ({}, ""),
            ):
                with self.subTest(url=bad[:60], environ=environ):
                    night = dict(GREEN, run_url=bad)
                    _, line, _ = self.summary(night, environ)
                    self.assertTrue(line.endswith(link or "main@4c10889 vs 6.2.0"), line[-120:])
                    self.assertNotIn("users/all", line)
                    self.assertNotIn("a" * 100, line)

    def test_markdown_escapes_text_from_the_file(self):
        night = nightly(
            "error", [{"check": "a*b_[x](y)|<script>", "verdict": "error", "error": "e"}]
        )
        _, _, summary = self.summary(night)
        self.assertIn(r"| a\*b\_\[x\](y)\|&lt;script\> | error | e |", summary)


# ---------------------------------------------------------------------------
# Commits since the last green night (red nights only).


def git(directory: Path, *arguments: str) -> str:
    environ = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    return subprocess.run(
        ["git", "-C", str(directory), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=environ,
    ).stdout.strip()


class CommitList(Workspace):
    def setUp(self):
        super().setUp()
        self.repository = self.directory / "repository"
        self.repository.mkdir()
        git(self.repository, "init", "-q")
        self.shas = []
        for subject in (
            "base (#1)",
            "fix: a < b (#3887) (#3889)",
            "direct push",
            "perf: x (#3963)",
        ):
            git(self.repository, "commit", "-q", "--allow-empty", "-m", subject)
            self.shas.append(git(self.repository, "rev-parse", "HEAD"))
        self.enter(unittest.mock.patch.object(report, "CHECKOUT", self.repository))

    def red(self, previous, candidate) -> dict:
        return nightly(
            "red",
            [result("micro-hw", "regression", [UNWIND])],
            previous_green=previous,
            candidate_commit=candidate,
        )

    def test_newest_first_with_the_last_pr_number_of_each_title(self):
        commits, why = report.commits_since_green(self.red(self.shas[0], self.shas[3]))
        self.assertEqual(why, "")
        self.assertEqual(
            commits,
            [
                (self.shas[3], "perf: x", 3963),
                (self.shas[2], "direct push", None),
                (self.shas[1], "fix: a < b (#3887)", 3889),
            ],
        )

    def test_the_red_summary_lists_them_with_pr_links_and_a_compare_link(self):
        _, _, summary = self.summary(self.red(self.shas[0], self.shas[3]))
        compare = f"https://github.com/FalkorDB/FalkorDB/compare/{self.shas[0]}...{self.shas[3]}"
        self.assertIn(f"Commits since the last green night (3, [compare]({compare})):", summary)
        self.assertIn(
            "- [\\#3963](https://github.com/FalkorDB/FalkorDB/pull/3963) perf: x", summary
        )
        self.assertIn(f"- `{self.shas[2][:7]}` direct push", summary)

    def test_says_why_when_there_is_no_list(self):
        cases = [
            (None, self.shas[3], "no earlier green night is recorded."),
            (self.shas[0], None, "the candidate's commit is unknown."),
            (self.shas[3], self.shas[3], "same candidate commit as the last green night"),
            ("a" * 40, "b" * 40, "the commits are not in this checkout. [compare]("),
        ]
        for previous, candidate, why in cases:
            with self.subTest(why=why):
                _, _, summary = self.summary(self.red(previous, candidate))
                self.assertIn(f"Commits since the last green night: {why}", summary)

    def test_a_value_that_is_not_a_hash_never_reaches_git(self):
        commits, why = report.commits_since_green(self.red("--output=/tmp/x", self.shas[3]))
        self.assertEqual((commits, why), ([], "no earlier green night is recorded"))

    @unittest.skipUnless(
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{PREVIOUS_GREEN}^{{commit}}"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
        and subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{CANDIDATE_COMMIT}^{{commit}}"],
            capture_output=True,
            check=False,
        ).returncode
        == 0,
        "this checkout lacks the history of main around 4c1088949",
    )
    def test_a_real_range_on_main(self):
        with unittest.mock.patch.object(report, "CHECKOUT", REPO_ROOT):
            commits, _ = report.commits_since_green(self.red(PREVIOUS_GREEN, CANDIDATE_COMMIT))
        self.assertEqual([pull for _, _, pull in commits], [3963, 3017, 3889, 3930])


# ---------------------------------------------------------------------------
# Delivery to Chat.


class StubServer:
    """A local HTTP server answering from `routes`: path -> (status, JSON) or a list of them."""

    def __init__(self, routes: dict):
        self.routes = {
            path: list(answer) if isinstance(answer, list) else answer
            for path, answer in routes.items()
        }
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                parts = urllib.parse.urlsplit(self.path)
                stub.requests.append(
                    {
                        "method": self.command,
                        "path": parts.path,
                        "query": parts.query,
                        "headers": dict(self.headers),
                        "body": self.rfile.read(length),
                    }
                )
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
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()
        return self

    def __exit__(self, *exception):
        self.server.shutdown()
        self.server.server_close()


class Delivery(Workspace):
    def test_posts_the_printed_line_as_chat_json(self):
        with StubServer(
            {"/v1/spaces/X/messages": (200, {"name": "spaces/X/messages/1"})}
        ) as server:
            code, stdout, stderr = self.run_post(
                GREEN, {"PERF_CHAT_WEBHOOK": f"{server.url}/v1/spaces/X/messages?{WEBHOOK_SECRET}"}
            )
        self.assertEqual(code, 0)
        self.assertEqual(len(server.requests), 1)
        request = server.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["headers"]["Content-Type"], "application/json; charset=UTF-8")
        self.assertEqual(json.loads(request["body"]), {"text": stdout.rstrip("\n")})
        self.assertIn("Posted to Chat.", stderr)

    def test_without_a_webhook_it_prints_and_says_so(self):
        code, stdout, stderr = self.run_post(GREEN)
        self.assertEqual(code, 0)
        self.assertIn("PERF_CHAT_WEBHOOK is not set: not posted to Chat.", stderr)
        self.assertTrue(stdout.startswith("Perf nightly 2026-11-24: GREEN"))

    def test_text_utf8_cannot_encode_is_replaced_and_the_night_still_posted(self):
        # JSON may escape a lone surrogate ("\\ud800"); writing it as UTF-8 would raise.
        night = nightly("red", [result("micro-hw", "regression", [UNWIND], summary="out: \ud800")])
        night["checks"][0]["check"] = "micro-hw\udfff"
        summary = self.directory / "summary.md"
        with StubServer({"/hook": (200, {})}) as server:
            code, stdout, stderr = self.run_post(
                night,
                {"PERF_CHAT_WEBHOOK": f"{server.url}/hook", "GITHUB_STEP_SUMMARY": str(summary)},
            )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(server.requests), 1)
        self.assertIn("Posted to Chat.", stderr)
        self.assertIn("RED (micro-hw?)", stdout)
        self.assertIn("| micro-hw? | regression | out: ? |", summary.read_text(encoding="utf-8"))

    def test_a_failed_post_is_a_warning_retried_and_never_shows_the_webhook(self):
        with StubServer({"/hook": (500, {"error": "boom"})}) as server:
            code, stdout, stderr = self.run_post(
                GREEN,
                {
                    "PERF_CHAT_WEBHOOK": f"{server.url}/hook?{WEBHOOK_SECRET}",
                    "GITHUB_ACTIONS": "true",
                },
            )
        self.assertEqual(code, 0)
        self.assertEqual(len(server.requests), 3)
        self.assertIn("::warning::posting to Chat failed: HTTP 500", stderr)
        for secret in ("SECRETKEY123", "SECRETTOKEN456", "/hook"):
            self.assertNotIn(secret, stdout + stderr)

    def test_a_failed_post_does_not_change_the_exit_code(self):
        with StubServer({"/hook": (400, {})}) as server:
            reported, _, _ = self.run_post(RED, {"PERF_CHAT_WEBHOOK": f"{server.url}/hook"})
            unreadable, _, _ = self.run_post("{", {"PERF_CHAT_WEBHOOK": f"{server.url}/hook"})
        self.assertEqual((reported, unreadable), (0, 1))

    def test_a_client_error_is_not_retried(self):
        with StubServer({"/hook": (400, {"error": "bad"})}) as server:
            _, _, stderr = self.run_post(GREEN, {"PERF_CHAT_WEBHOOK": f"{server.url}/hook"})
        self.assertEqual(len(server.requests), 1)
        self.assertIn("warning: posting to Chat failed: HTTP 400", stderr)

    def test_an_unreachable_webhook_is_retried_then_a_warning(self):
        with StubServer({}) as server:
            url = f"{server.url}/hook?{WEBHOOK_SECRET}"
        code, _, stderr = self.run_post(GREEN, {"PERF_CHAT_WEBHOOK": url})  # the server is gone
        self.assertEqual(code, 0)
        self.assertIn("posting to Chat failed: connection failed", stderr)
        self.assertNotIn("SECRET", stderr)

    def test_a_connect_timeout_is_retried_because_nothing_was_sent(self):
        timeout = urllib.error.URLError(TimeoutError("timed out"))
        with unittest.mock.patch.object(
            report.urllib.request, "urlopen", side_effect=timeout
        ) as urlopen:
            code, _, stderr = self.run_post(
                GREEN, {"PERF_CHAT_WEBHOOK": "https://chat.example/hook"}
            )
        self.assertEqual((code, urlopen.call_count), (0, 3))
        self.assertIn("posting to Chat failed: connection failed: TimeoutError", stderr)

    def test_a_failure_after_sending_is_not_retried_because_the_post_may_have_arrived(self):
        # urllib raises these bare (not as URLError) only once the request went out.
        for error in (
            TimeoutError("timed out"),
            ConnectionResetError(54, "reset"),
            http.client.BadStatusLine("x"),
        ):
            with self.subTest(error=type(error).__name__):
                with unittest.mock.patch.object(
                    report.urllib.request, "urlopen", side_effect=error
                ) as urlopen:
                    code, _, stderr = self.run_post(
                        GREEN, {"PERF_CHAT_WEBHOOK": "https://chat.example/hook"}
                    )
                self.assertEqual((code, urlopen.call_count), (0, 1))
                self.assertIn(
                    f"no answer from Chat: {type(error).__name__} (not retried: the message may have arrived)",
                    stderr,
                )

    def test_a_cut_short_answer_after_a_2xx_is_still_a_delivered_post(self):
        class Answer:
            def __enter__(self):
                return self

            def __exit__(self, *exception):
                return False

            def read(self, *arguments):
                raise http.client.IncompleteRead(b"")

        with unittest.mock.patch.object(
            report.urllib.request, "urlopen", return_value=Answer()
        ) as urlopen:
            code, _, stderr = self.run_post(
                GREEN, {"PERF_CHAT_WEBHOOK": "https://chat.example/hook"}
            )
        self.assertEqual((code, urlopen.call_count), (0, 1))
        self.assertIn("Posted to Chat.", stderr)

    def test_only_https_webhooks(self):
        for url in (
            "http://chat.googleapis.com/v1/x",
            f"file:///etc/passwd?{WEBHOOK_SECRET}",
            "not a url",
            "https://[SECRETKEY/v1/x",
        ):
            with self.subTest(url=url):
                code, _, stderr = self.run_post(GREEN, {"PERF_CHAT_WEBHOOK": url})
                self.assertEqual(code, 0)
                self.assertRegex(stderr, r"PERF_CHAT_WEBHOOK is not (an https:// URL|a valid URL)")
                self.assertNotIn("SECRET", stderr)

    def test_any_other_failure_while_posting_is_a_warning(self):
        with unittest.mock.patch.object(report, "post_to_chat", side_effect=RuntimeError("boom")):
            code, stdout, stderr = self.run_post(
                GREEN, {"PERF_CHAT_WEBHOOK": "https://chat.example/hook"}
            )
        self.assertEqual(code, 0)
        self.assertTrue(stdout.startswith("Perf nightly 2026-11-24: GREEN"))
        self.assertIn("posting to Chat failed: unexpected RuntimeError", stderr)

    def test_a_github_body_cut_short_is_a_github_error(self):
        class Answer:
            def __enter__(self):
                return self

            def __exit__(self, *exception):
                return False

            def read(self, *arguments):
                raise http.client.IncompleteRead(b"{")

        with (
            unittest.mock.patch.object(report.urllib.request, "urlopen", return_value=Answer()),
            self.assertRaises(report.GitHubError) as raised,
        ):
            report.github_get("https://api.github.com/repos/x/y", "")
        self.assertIn("IncompleteRead", str(raised.exception))


# ---------------------------------------------------------------------------
# perf-judge's valid nightly fixtures (tests/perf/fixtures/valid/).


class JudgeFixtures(Workspace):
    def test_each_valid_nightly_gets_its_status_word_names_and_exit_code(self):
        files = sorted((FIXTURES / "valid").glob("nightly*.json"))
        self.assertTrue(files, f"no nightly*.json in {FIXTURES / 'valid'}")
        for path in files:
            with self.subTest(fixture=path.name):
                data = json.loads(path.read_text())
                code, line, summary = self.summary(path)
                self.assertEqual(code, 0)
                self.assertTrue(
                    line.startswith(f"Perf nightly {data['date']}: {data['status'].upper()} ("),
                    line,
                )
                self.assertIn(f"· <{data['run_url']}|details>", line)
                self.assertEqual("|new issue>" in line, data["status"] == "red")
                verdict = {"red": "regression", "error": "error"}.get(data["status"])
                for check in data["checks"]:
                    if check["verdict"] == verdict:
                        self.assertIn(check["check"], line)
                    self.assertIn(
                        f"| {report.markdown_plain(check['check'])} | {check['verdict']} |", summary
                    )


# ---------------------------------------------------------------------------
# The missed-run alarm.

NIGHT = date(2026, 11, 24)
NOW = datetime(2026, 11, 24, 5, 0, 7, tzinfo=UTC)
RUNS_PATH = "/repos/FalkorDB/FalkorDB/actions/workflows/perf-nightly.yml/runs"


def run(
    identifier: int,
    created: str,
    status: str = "completed",
    conclusion: str | None = "success",
    branch: str = "main",
    event: str = "schedule",
) -> dict:
    return {
        "id": identifier,
        "created_at": created,
        "run_started_at": created,
        "status": status,
        "conclusion": conclusion,
        "head_branch": branch,
        "event": event,
        "html_url": f"https://github.com/FalkorDB/FalkorDB/actions/runs/{identifier}",
    }


def artifacts(*names: str, expired: bool = False) -> dict:
    return {
        "total_count": len(names),
        "artifacts": [{"name": name, "expired": expired} for name in names],
    }


class MissedRun(Workspace):
    def alarm(self, routes: dict, options: dict | None = None, environ: dict | None = None):
        with StubServer(routes) as server:
            environ = {"GITHUB_API_URL": server.url, **(environ or {})}
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = missed_run.run(environ, NIGHT, now=NOW, **(options or {}))
        return code, stdout.getvalue(), stderr.getvalue(), server.requests

    def test_quiet_when_tonights_run_has_the_artifact(self):
        code, stdout, stderr, requests = self.alarm(
            {
                RUNS_PATH: (200, {"workflow_runs": [run(7, "2026-11-24T00:03:11Z")]}),
                "/repos/FalkorDB/FalkorDB/actions/runs/7/artifacts": (
                    200,
                    artifacts("perf-nightly"),
                ),
            },
            environ={"PERF_CHAT_WEBHOOK": "http://127.0.0.1:9/never"},
        )
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
        code, _, _, _ = self.alarm(
            {
                RUNS_PATH: (200, {"workflow_runs": [run(8, "2026-11-24T00:01:00Z")]}),
                "/repos/FalkorDB/FalkorDB/actions/runs/8/artifacts": (
                    200,
                    artifacts("perf-nightly"),
                ),
            }
        )
        self.assertEqual(code, 0)

    def test_no_run(self):
        code, stdout, _, _ = self.alarm({RUNS_PATH: (200, {"workflow_runs": []})})
        self.assertEqual(code, 1)
        self.assertIn(
            "*Perf nightly alarm 2026-11-24*: no nightly result. No schedule run of perf-nightly.yml on main "
            "between 2026-11-23 05:00 and 2026-11-24 05:00 UTC.",
            stdout,
        )

    def test_runs_outside_the_window_or_off_main_do_not_count(self):
        code, stdout, _, _ = self.alarm(
            {
                RUNS_PATH: (
                    200,
                    {
                        "workflow_runs": [
                            run(1, "2026-11-24T05:00:00Z"),  # the window ends at 05:00, exclusive
                            run(2, "2026-11-23T04:59:59Z"),
                            run(3, "2026-11-24T00:00:05Z", branch="ci/perf-nightly"),
                        ]
                    },
                )
            }
        )
        self.assertEqual(code, 1)
        self.assertIn("No schedule run of perf-nightly.yml on main", stdout)

    def test_a_hand_started_run_with_the_artifact_does_not_stand_in_for_the_night(self):
        # A dispatch given its own images uploads nightly.json but posts nothing.
        code, stdout, _, _ = self.alarm(
            {
                RUNS_PATH: (
                    200,
                    {"workflow_runs": [run(4, "2026-11-23T14:00:00Z", event="workflow_dispatch")]},
                ),
                "/repos/FalkorDB/FalkorDB/actions/runs/4/artifacts": (
                    200,
                    artifacts("perf-nightly"),
                ),
            }
        )
        self.assertEqual(code, 1)
        self.assertIn("No schedule run of perf-nightly.yml on main", stdout)

    def test_a_run_still_going(self):
        code, stdout, _, _ = self.alarm(
            {
                RUNS_PATH: (
                    200,
                    {"workflow_runs": [run(9, "2026-11-23T05:00:00Z", "in_progress", None)]},
                ),
                "/repos/FalkorDB/FalkorDB/actions/runs/9/artifacts": (200, artifacts()),
            }
        )
        self.assertEqual(code, 1)
        self.assertIn(
            "The run (started 2026-11-23 05:00 UTC) is still in_progress at 2026-11-24 05:00 UTC and has "
            "not uploaded perf-nightly. <https://github.com/FalkorDB/FalkorDB/actions/runs/9|run>",
            stdout,
        )

    def test_a_failed_run_without_the_artifact(self):
        code, stdout, _, _ = self.alarm(
            {
                RUNS_PATH: (
                    200,
                    {"workflow_runs": [run(10, "2026-11-24T00:00:09Z", conclusion="failure")]},
                ),
                "/repos/FalkorDB/FalkorDB/actions/runs/10/artifacts": (
                    200,
                    artifacts("perf-result-micro"),
                ),
            }
        )
        self.assertEqual(code, 1)
        self.assertIn("The run finished `failure` without uploading perf-nightly.", stdout)

    def test_an_expired_artifact_does_not_count(self):
        code, _, _, _ = self.alarm(
            {
                RUNS_PATH: (200, {"workflow_runs": [run(11, "2026-11-24T00:00:09Z")]}),
                "/repos/FalkorDB/FalkorDB/actions/runs/11/artifacts": (
                    200,
                    artifacts("perf-nightly", expired=True),
                ),
            }
        )
        self.assertEqual(code, 1)

    def test_any_run_in_the_window_with_the_artifact_is_enough(self):
        code, _, _, _ = self.alarm(
            {
                RUNS_PATH: (
                    200,
                    {
                        "workflow_runs": [
                            run(12, "2026-11-24T02:00:00Z", conclusion="cancelled"),
                            run(13, "2026-11-24T00:00:00Z"),
                        ]
                    },
                ),
                "/repos/FalkorDB/FalkorDB/actions/runs/12/artifacts": (200, artifacts()),
                "/repos/FalkorDB/FalkorDB/actions/runs/13/artifacts": (
                    200,
                    artifacts("perf-nightly"),
                ),
            }
        )
        self.assertEqual(code, 0)

    def test_several_runs_none_with_the_artifact_reports_the_newest(self):
        code, stdout, _, _ = self.alarm(
            {
                RUNS_PATH: (
                    200,
                    {
                        "workflow_runs": [
                            run(14, "2026-11-24T00:00:00Z", conclusion="failure"),
                            run(15, "2026-11-24T02:00:00Z", conclusion="cancelled"),
                        ]
                    },
                ),
                "/repos/FalkorDB/FalkorDB/actions/runs/14/artifacts": (200, artifacts()),
                "/repos/FalkorDB/FalkorDB/actions/runs/15/artifacts": (200, artifacts()),
            }
        )
        self.assertEqual(code, 1)
        self.assertIn(
            "finished `cancelled` without uploading perf-nightly. (2 runs between", stdout
        )
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
            quiet, _, _, _ = self.alarm(
                {
                    RUNS_PATH: (200, {"workflow_runs": [run(7, "2026-11-24T00:03:11Z")]}),
                    "/repos/FalkorDB/FalkorDB/actions/runs/7/artifacts": (
                        200,
                        artifacts("perf-nightly"),
                    ),
                },
                environ=hook,
            )
        self.assertEqual((fired, quiet), (1, 0))
        self.assertEqual(len(chat.requests), 1)
        self.assertIn("Perf nightly alarm 2026-11-24", json.loads(chat.requests[0]["body"])["text"])

    def test_workflow_branch_and_artifact_are_injectable(self):
        code, _, _, requests = self.alarm(
            {
                "/repos/FalkorDB/FalkorDB/actions/workflows/rust-push.yml/runs": (
                    200,
                    {
                        "workflow_runs": [
                            run(16, "2026-11-24T01:00:00Z", branch="dev", event="push")
                        ]
                    },
                ),
                "/repos/FalkorDB/FalkorDB/actions/runs/16/artifacts": (
                    200,
                    artifacts("coverage-binary"),
                ),
            },
            {
                "workflow": "rust-push.yml",
                "branch": "dev",
                "event": "push",
                "artifact": "coverage-binary",
            },
        )
        self.assertEqual(code, 0)
        query = urllib.parse.parse_qs(requests[0]["query"])
        self.assertEqual((query["branch"], query["event"]), (["dev"], ["push"]))

    def test_a_rerun_of_a_scheduled_check_checks_the_night_it_was_created_for(self):
        routes = {
            RUNS_PATH: (200, {"workflow_runs": []}),
            "/repos/FalkorDB/FalkorDB/actions/runs/99": (
                200,
                {"id": 99, "created_at": "2026-11-24T05:03:00Z"},
            ),
        }
        with (
            StubServer(routes) as server,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            missed_run.run(
                {
                    "GITHUB_API_URL": server.url,
                    "GITHUB_EVENT_NAME": "schedule",
                    "GITHUB_RUN_ID": "99",
                },
                now=datetime(2026, 11, 26, 9, 30, 0, tzinfo=UTC),
            )
        runs = [request for request in server.requests if request["path"] == RUNS_PATH]
        self.assertEqual(
            urllib.parse.parse_qs(runs[0]["query"])["created"],
            ["2026-11-23T05:00:00Z..2026-11-24T05:00:00Z"],
        )

    def test_today_when_the_run_cannot_be_looked_up_or_was_not_scheduled(self):
        for environ in (
            {"GITHUB_EVENT_NAME": "schedule", "GITHUB_RUN_ID": "98"},  # lookup answers 404
            {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_RUN_ID": "99"},
            # Not a number: never put in the URL (here it would reach runs/99).
            {"GITHUB_EVENT_NAME": "schedule", "GITHUB_RUN_ID": "99?x=1"},
        ):
            with self.subTest(environ=environ):
                routes = {
                    RUNS_PATH: (200, {"workflow_runs": []}),
                    "/repos/FalkorDB/FalkorDB/actions/runs/99": (
                        200,
                        {"created_at": "2026-11-20T05:03:00Z"},
                    ),
                }
                with (
                    StubServer(routes) as server,
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    missed_run.run(
                        {"GITHUB_API_URL": server.url, **environ},
                        now=datetime(2026, 11, 26, 9, 30, 0, tzinfo=UTC),
                    )
                runs = [request for request in server.requests if request["path"] == RUNS_PATH]
                self.assertEqual(
                    urllib.parse.parse_qs(runs[0]["query"])["created"],
                    ["2026-11-25T05:00:00Z..2026-11-26T05:00:00Z"],
                )

    def test_an_explicit_date_wins_over_the_scheduled_night(self):
        routes = {
            RUNS_PATH: (200, {"workflow_runs": []}),
            "/repos/FalkorDB/FalkorDB/actions/runs/99": (
                200,
                {"created_at": "2026-11-24T05:03:00Z"},
            ),
        }
        with (
            StubServer(routes) as server,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            missed_run.run(
                {
                    "GITHUB_API_URL": server.url,
                    "GITHUB_EVENT_NAME": "schedule",
                    "GITHUB_RUN_ID": "99",
                },
                date(2026, 11, 10),
                now=datetime(2026, 11, 26, 9, 30, 0, tzinfo=UTC),
            )
        self.assertEqual([request["path"] for request in server.requests], [RUNS_PATH])
        self.assertEqual(
            urllib.parse.parse_qs(server.requests[0]["query"])["created"],
            ["2026-11-09T05:00:00Z..2026-11-10T05:00:00Z"],
        )

    def test_the_default_night_is_today_in_utc(self):
        with (
            StubServer({RUNS_PATH: (200, {"workflow_runs": []})}) as server,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            missed_run.run(
                {"GITHUB_API_URL": server.url}, now=datetime(2026, 11, 25, 5, 0, 3, tzinfo=UTC)
            )
        self.assertEqual(
            urllib.parse.parse_qs(server.requests[0]["query"])["created"],
            ["2026-11-24T05:00:00Z..2026-11-25T05:00:00Z"],
        )


class AlarmWorkflow(unittest.TestCase):
    def test_the_alarm_runs_one_concurrency_group_per_run(self):
        # A group shared by scheduled runs keeps one pending run: a later one would cancel it.
        workflow = (REPO_ROOT / ".github" / "workflows" / "perf-missed-run.yml").read_text()
        groups = [
            line.strip() for line in workflow.splitlines() if line.strip().startswith("group:")
        ]
        self.assertEqual(groups, ["group: perf-missed-run-${{ github.run_id }}"])


class CommandLine(unittest.TestCase):
    """`bench nightly report` and `bench nightly missed-run`, as the workflows run them."""

    # Never let a test write to a real job summary or post to a real webhook.
    QUIET: ClassVar[dict[str, None]] = {"GITHUB_STEP_SUMMARY": None, "PERF_CHAT_WEBHOOK": None}

    def bench(self, *arguments: str, env: dict | None = None):
        return CliRunner().invoke(
            cli, list(arguments), env={**self.QUIET, **(env or {})}, catch_exceptions=False
        )

    def test_bench_nightly_report_exits_1_only_when_it_could_not_report(self):
        lines, codes = [], []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nightly.json"
            for document in (GREEN, RED, nightly("x", [])):
                path.write_text(json.dumps(document))
                run = self.bench("nightly", "report", "--nightly", str(path))
                codes.append(run.exit_code)
                lines.append(run.stdout)
        self.assertEqual(codes, [0, 0, 1])
        self.assertTrue(lines[0].startswith("Perf nightly 2026-11-24: GREEN"), lines[0])
        self.assertTrue(lines[1].startswith("Perf nightly 2026-11-24: RED (micro-hw)"), lines[1])
        self.assertTrue(
            lines[2].startswith("Perf nightly 2026-11-24: ERROR (unknown status 'x'"), lines[2]
        )

    def test_bench_nightly_missed_run_passes_its_options_through(self):
        path = "/repos/FalkorDB/FalkorDB/actions/workflows/rust-push.yml/runs"
        with StubServer({path: (200, {"workflow_runs": []})}) as server:
            run = self.bench(
                "nightly",
                "missed-run",
                "--date",
                "2026-11-10",
                "--workflow",
                "rust-push.yml",
                "--branch",
                "dev",
                "--event",
                "push",
                "--artifact",
                "coverage-binary",
                env={"GITHUB_API_URL": server.url},
            )
        self.assertEqual(run.exit_code, 1)
        self.assertIn(
            "No push run of rust-push.yml on dev between 2026-11-09 05:00 and 2026-11-10 05:00 UTC.",
            run.stdout,
        )
        query = urllib.parse.parse_qs(server.requests[0]["query"])
        self.assertEqual((query["branch"], query["event"]), (["dev"], ["push"]))


class StandardLibraryOnly(unittest.TestCase):
    """The reporting path never depends on a package (bench/README.md, the toolset rule)."""

    # The package's own modules these may import, each standard library only itself.
    ALLOWED = frozenset(
        {"falkorbench", "falkorbench.contract", "falkorbench.nightly", "falkorbench.nightly.report"}
    )

    def test_the_nightly_modules_import_only_the_standard_library(self):
        for module in (nightly_package, report, missed_run):
            with self.subTest(module=module.__name__):
                for node in ast.walk(ast.parse(Path(module.__file__).read_text(encoding="utf-8"))):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        self.assertEqual(
                            node.level, 0, "a relative import pulls in the rest of the package"
                        )
                        names = [
                            f"{node.module}.{alias.name}"
                            if node.module in self.ALLOWED
                            else node.module
                            for alias in node.names
                        ]
                    else:
                        continue
                    for name in names:
                        if name.split(".")[0] == "falkorbench":
                            self.assertIn(name, self.ALLOWED)
                        else:
                            self.assertIn(name.split(".")[0], sys.stdlib_module_names, name)


if __name__ == "__main__":
    unittest.main()
