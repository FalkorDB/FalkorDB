#!/usr/bin/env python3
"""Turn one night's `nightly.json` into one message for the team's Chat space.

    python -I tests/perf/report/post.py --nightly <path/to/nightly.json>

Every verdict comes from `nightly.json` alone (schema 1, see README.md), checked
first with perf-judge's validator (`tests/perf/validate.py`): a file that breaks the
contract is shown as ERROR at best, never green. The only other thing read is the
list of commits since the last green night, from `git log` in this checkout or,
failing that, GitHub's compare API.

The message is printed to stdout, appended as Markdown to `$GITHUB_STEP_SUMMARY`
when that is set, and posted to the Google Chat incoming webhook in
`$PERF_CHAT_WEBHOOK` when that is set. A failed post is a warning. The exit code is
0 unless this script itself crashed: a red night is reported, not raised.

Standard library only, so it runs under `python -I` on any runner.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REPOSITORY = "FalkorDB/FalkorDB"
DEFAULT_SERVER_URL = "https://github.com"
DEFAULT_API_URL = "https://api.github.com"
ISSUE_LABEL = "performance"
USER_AGENT = "falkordb-perf-report"

# "The maximum message size (including any text or cards) is 32,000 bytes"
# (developers.google.com/workspace/chat/create-messages, written for the API; the
# webhook page states no limit). A night's message stays far below that: fewer rows,
# free text shortened, and as a last resort whole lines dropped from the end.
CHAT_CHARACTER_BUDGET = 4000
FREE_TEXT_CHARACTERS = 300
LABEL_CHARACTERS = 80

VERDICTS = ("pass", "regression", "error")
STATUS_RANK = {"green": 0, "error": 1, "red": 2}
CORRECTNESS_METRICS = ("completed", "invariant")
COMMIT_HASH = re.compile(r"^[0-9a-f]{7,40}$")
TRAILING_PULL_REQUEST = re.compile(r"^(?P<title>.*?)\s*\(#(?P<number>\d+)\)\s*$")
# A URL that can sit inside Chat's `<url|text>` and Markdown's `[text](url)` intact.
SAFE_URL = re.compile(r"https?://[^\s<>|\"'`()\[\]]+")

# perf-judge's validator for the results contract; this reporter never copies its rules.
VALIDATOR = Path(__file__).resolve().parents[1] / "validate.py"
CONTRACT_PROBLEMS_SHOWN = 5

# The checkout whose `git log` lists the commits since the last green night.
CHECKOUT = Path(__file__).resolve().parent

# Seconds to wait before each retry of a webhook post. Tests set it to zeros.
POST_RETRY_DELAYS = (1.0, 3.0)
HTTP_TIMEOUT_SECONDS = 20


# ---------------------------------------------------------------------------
# A message: lines of segments, rendered once for Chat and once for Markdown.


@dataclass(frozen=True)
class Bold:
    text: str


@dataclass(frozen=True)
class Code:
    text: str


@dataclass(frozen=True)
class Link:
    url: str
    text: str


Segment = str | Bold | Code | Link


@dataclass
class Line:
    segments: list[Segment] = field(default_factory=list)
    bullet: bool = False


BLANK = Line()


def chat_plain(text: str) -> str:
    # In Chat markup `<` opens a link or a mention (`<users/all>` pings the whole
    # space). Text from commits and results must never do either.
    return text.replace("<", "‹")


def chat_link_text(text: str) -> str:
    return chat_plain(text).replace("|", "/").replace(">", "›")


def is_safe_url(url: str) -> bool:
    return bool(SAFE_URL.fullmatch(url))


def render_chat(lines: list[Line]) -> str:
    rendered = []
    for line in lines:
        parts = []
        for segment in line.segments:
            if isinstance(segment, Bold):
                parts.append(f"*{chat_plain(segment.text)}*")
            elif isinstance(segment, Code):
                parts.append(f"`{chat_plain(segment.text).replace('`', chr(39))}`")
            elif isinstance(segment, Link) and is_safe_url(segment.url):
                parts.append(f"<{segment.url}|{chat_link_text(segment.text)}>")
            elif isinstance(segment, Link):
                parts.append(chat_plain(segment.text))
            else:
                parts.append(chat_plain(segment))
        text = "".join(parts)
        rendered.append(f"- {text}" if line.bullet else text)
    return "\n".join(rendered).strip("\n")


MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]~|#>])")


def markdown_plain(text: str) -> str:
    return MARKDOWN_SPECIAL.sub(r"\\\1", text.replace("&", "&amp;").replace("<", "&lt;"))


def render_markdown(lines: list[Line]) -> str:
    rendered = []
    previous_was_bullet = False
    for line in lines:
        parts = []
        for segment in line.segments:
            if isinstance(segment, Bold):
                parts.append(f"**{markdown_plain(segment.text)}**")
            elif isinstance(segment, Code):
                parts.append(f"`` {segment.text} ``" if "`" in segment.text else f"`{segment.text}`")
            elif isinstance(segment, Link) and is_safe_url(segment.url):
                parts.append(f"[{markdown_plain(segment.text)}]({segment.url})")
            elif isinstance(segment, Link):
                parts.append(markdown_plain(segment.text))
            else:
                parts.append(markdown_plain(segment))
        text = "".join(parts)
        if not line.segments:
            rendered.append("")
        elif line.bullet:
            if rendered and not previous_was_bullet and rendered[-1] != "":
                rendered.append("")
            rendered.append(f"- {text}")
        else:
            if previous_was_bullet:
                rendered.append("")
            # Two trailing spaces: a hard line break, so consecutive lines do not
            # run together into one paragraph.
            rendered.append(f"{text}  ")
        previous_was_bullet = line.bullet and bool(line.segments)
    return "\n".join(rendered).strip("\n") + "\n"


# ---------------------------------------------------------------------------
# Reading nightly.json without trusting it.


@dataclass
class Check:
    name: str
    verdict: str  # pass | regression | error, after normalisation
    summary: str
    confirmed: bool
    findings: list[dict]
    error: str


@dataclass
class Night:
    date: str
    run_url: str
    candidate_label: str
    baseline_label: str
    candidate_image: str
    baseline_image: str
    candidate_commit: str | None
    previous_green_commit: str | None
    declared_status: str
    skip_reason: str
    checks: list[Check]
    problems: list[str]  # anything that stops the file from being trusted as written
    readable: bool = True
    status: str = "error"  # what the message shows: green | skipped | error | red


def text_or(value: object, fallback: str, limit: int = FREE_TEXT_CHARACTERS) -> str:
    """One line of at most `limit` characters, or `fallback` when there is no text."""
    if isinstance(value, str) and value.strip():
        return shorten(" ".join(value.split()), limit)
    return fallback


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def read_check(index: int, entry: object) -> Check:
    if not isinstance(entry, dict):
        return Check(f"check #{index + 1}", "error", "", False, [], "the entry is not a JSON object")
    name = text_or(entry.get("check"), f"check #{index + 1}")
    verdict = entry.get("verdict")
    error = text_or(entry.get("error"), "")
    if verdict not in VERDICTS:
        error = f"unknown verdict {verdict!r}" + (f": {error}" if error else "")
        verdict = "error"
    elif verdict != "error" and "schema" not in entry:
        # Only the merge step's stub lacks `schema`, and a stub can only say error.
        error = f"says {verdict} but carries no result"
        verdict = "error"
    elif verdict == "error" and not error:
        error = "no reason given"
    findings = entry.get("findings")
    findings = [f for f in findings if isinstance(f, dict)] if isinstance(findings, list) else []
    return Check(
        name=name,
        verdict=verdict,
        summary=text_or(entry.get("summary"), ""),
        confirmed=entry.get("confirmed") is True,
        findings=findings,
        error=error,
    )


def status_from_checks(checks: list[Check]) -> str:
    if any(check.verdict == "regression" for check in checks):
        return "red"
    if any(check.verdict == "error" for check in checks):
        return "error"
    return "green"


def read_night(path: Path, environ: dict) -> Night:
    problems: list[str] = []
    data: object = None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        problems.append(shorten(f"could not read {path.name}: {exc}", FREE_TEXT_CHARACTERS))
    else:
        if not isinstance(data, dict):
            problems.append(f"{path.name} is not a JSON object")
    readable = isinstance(data, dict)
    raw = data if isinstance(data, dict) else {}

    if readable and raw.get("schema") != 1:
        problems.append(shorten(f"unsupported schema {raw.get('schema')!r} (this reporter reads schema 1)",
                                FREE_TEXT_CHARACTERS))

    raw_checks = raw.get("checks")
    if readable and not isinstance(raw_checks, list):
        problems.append("`checks` is not a list")
        raw_checks = []
    checks = [read_check(index, entry) for index, entry in enumerate(raw_checks or [])]
    names = [check.name for check in checks]
    for name in sorted({name for name in names if names.count(name) > 1}):
        problems.append(f"check {name!r} appears {names.count(name)} times")
    if readable:
        problems.extend(contract_problems(path))

    candidate = raw.get("candidate") if isinstance(raw.get("candidate"), dict) else {}
    baseline = raw.get("baseline") if isinstance(raw.get("baseline"), dict) else {}
    declared = raw.get("status") if isinstance(raw.get("status"), str) else repr(raw.get("status"))
    declared = shorten(declared, LABEL_CHARACTERS)

    run_url = raw.get("run_url")
    if not (isinstance(run_url, str) and run_url.startswith("https://") and is_safe_url(run_url)):
        run_url = run_url_from_environment(environ)

    unknown_candidate = text_or(candidate.get("image"), "candidate unknown", LABEL_CHARACTERS)
    unknown_baseline = text_or(baseline.get("image"), "baseline unknown", LABEL_CHARACTERS)
    night = Night(
        date=text_or(raw.get("date"), "date unknown", LABEL_CHARACTERS),
        run_url=run_url,
        candidate_label=text_or(candidate.get("label"), unknown_candidate, LABEL_CHARACTERS),
        baseline_label=text_or(baseline.get("label"), unknown_baseline, LABEL_CHARACTERS),
        candidate_image=text_or(candidate.get("image"), ""),
        baseline_image=text_or(baseline.get("image"), ""),
        candidate_commit=commit_or_none(candidate.get("commit")),
        previous_green_commit=commit_or_none(raw.get("previous_green_commit")),
        declared_status=declared,
        skip_reason=text_or(raw.get("skip_reason"), ""),
        checks=checks,
        problems=problems,
        readable=readable,
    )
    night.status = decide_status(night)
    return night


def contract_problems(path: Path) -> list[str]:
    """What perf-judge's validator finds wrong with the file, in short form.

    A validator that is missing or crashes is itself a problem: an unchecked file
    must not read as a checked one.
    """
    try:
        spec = importlib.util.spec_from_file_location("perf_contract_validator", VALIDATOR)
        if spec is None or spec.loader is None:
            raise ImportError("no loader")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            _, problems = module.validate_file(path, "nightly")
        finally:
            sys.modules.pop(spec.name, None)
        found = [f"{problem.code} at {problem.path}" for problem in problems]
    except Exception as exc:  # noqa: BLE001  (any failure to validate must be reported, not raised)
        return [f"could not run the contract validator {VALIDATOR.name} ({type(exc).__name__}: {str(exc)[:160]})"]
    if len(found) > CONTRACT_PROBLEMS_SHOWN:
        hidden = len(found) - CONTRACT_PROBLEMS_SHOWN
        found = found[:CONTRACT_PROBLEMS_SHOWN] + [f"{hidden} more (run tests/perf/validate.py)"]
    return [f"contract: {problem}" for problem in found]


def commit_or_none(value: object) -> str | None:
    if isinstance(value, str) and COMMIT_HASH.match(value.strip().lower()):
        return value.strip().lower()
    return None


def decide_status(night: Night) -> str:
    """The status the message shows: never better than the file or its checks say.

    Red when any check regressed. Otherwise any problem (the validator's, a declared
    status that disagrees with the checks, checks on a skipped night, no checks on
    another) makes it `error`, so a broken nightly.json can never read as green.
    """
    derived = status_from_checks(night.checks)
    declared = night.declared_status
    shown = derived
    if not night.readable:
        return "error"  # read_night recorded why; there are no checks to go on
    if declared == "skipped" and not night.checks:
        shown = "skipped"
    elif not night.checks:
        night.problems.append(f"no checks on a night whose status is {declared!r}")
    elif declared == "skipped":
        night.problems.append(f"status 'skipped' but {len(night.checks)} checks are present")
    elif declared not in STATUS_RANK:
        night.problems.append(f"status {declared!r} is not green, red, error or skipped")
    elif declared != derived:
        night.problems.append(f"nightly.json says {declared!r} but its checks say {derived!r}")
        shown = max(declared, derived, key=STATUS_RANK.__getitem__)
    if night.problems and shown in ("green", "skipped"):
        return "error"
    return shown


def run_url_from_environment(environ: dict) -> str:
    run_id = environ.get("GITHUB_RUN_ID")
    if not run_id:
        return ""
    url = f"{server_url(environ)}/{repository(environ)}/actions/runs/{run_id}"
    return url if is_safe_url(url) else ""


def server_url(environ: dict) -> str:
    return (environ.get("GITHUB_SERVER_URL") or DEFAULT_SERVER_URL).rstrip("/")


def api_url(environ: dict) -> str:
    return (environ.get("GITHUB_API_URL") or DEFAULT_API_URL).rstrip("/")


def repository(environ: dict) -> str:
    return environ.get("GITHUB_REPOSITORY") or DEFAULT_REPOSITORY


def github_token(environ: dict) -> str:
    return environ.get("GH_TOKEN") or environ.get("GITHUB_TOKEN") or ""


# ---------------------------------------------------------------------------
# Findings.


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def finite(value: object) -> float | None:
    """`value` as a finite float; None when it is not a number, NaN, infinite or too big."""
    if not is_number(value):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def format_value(value: object) -> str:
    if value is None:
        return "n/a"
    number = finite(value)
    if number is None:
        return text_or(str(value), "n/a", 40)
    if isinstance(value, int) and abs(value) >= 1000:
        return shorten(f"{value:,}", 40)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) >= 1 or number == 0:
        return f"{number:.2f}".rstrip("0").rstrip(".")
    return f"{number:.3g}"


def format_ratio(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}x"
    if 0 < abs(value) < 0.01:
        return f"{value:.2g}x"
    return f"{value:.2f}x"


def is_correctness(finding: dict) -> bool:
    return finding.get("direction") == "mismatch" or finding.get("metric") in CORRECTNESS_METRICS


def severity_key(finding: dict) -> tuple:
    """Sort key, worst first: correctness, then an unknown ratio, then the ratio."""
    if is_correctness(finding):
        return (0, 0.0)
    ratio = finite(finding.get("ratio"))
    if ratio is None:
        return (1, 0.0)
    if finding.get("direction") == "lower_is_worse":
        badness = 1 / ratio if ratio > 0 else math.inf
    else:
        badness = ratio
    return (2, -badness)


def is_zero(value: object) -> bool:
    return is_number(value) and value == 0


def is_one(value: object) -> bool:
    return is_number(value) and value == 1


def describe_finding(finding: dict) -> str:
    name = text_or(finding.get("name"), "unnamed", 120)
    metric = text_or(finding.get("metric"), "metric unknown", 40)
    candidate = finding.get("candidate")
    baseline = finding.get("baseline")
    if finding.get("direction") == "mismatch":
        what = "row count" if metric == "rows" else metric
        return f"{name}: {what} differs: {format_value(candidate)} vs {format_value(baseline)}"
    if metric == "completed" and is_zero(candidate):
        if is_zero(baseline):
            return f"{name}: did not finish on either side"
        return f"{name}: did not finish" + (" (baseline finished)" if is_one(baseline) else "")
    if metric == "invariant" and is_zero(candidate):
        if is_zero(baseline):
            return f"invariant broken on both sides: {name}"
        return f"invariant broken: {name}" + (" (holds on baseline)" if is_one(baseline) else "")
    text = f"{name}: {metric} {format_value(candidate)} vs {format_value(baseline)}"
    extras = []
    ratio = finite(finding.get("ratio"))
    threshold = finite(finding.get("threshold"))
    if ratio is not None:
        extras.append(format_ratio(ratio))
    if threshold is not None:
        extras.append(f"threshold {format_ratio(threshold)}")
    return f"{text} ({', '.join(extras)})" if extras else text


def regression_findings(check: Check) -> list[dict]:
    rows = [finding for finding in check.findings if finding.get("status") == "regression"]
    return sorted(rows, key=severity_key)


def broken_on_both_sides(night: Night) -> list[tuple[str, dict]]:
    """Correctness rows that fail on the baseline too (contract v1.4: status `info`)."""
    return [(check.name, finding) for check in night.checks for finding in check.findings
            if finding.get("status") == "info" and finding.get("metric") in CORRECTNESS_METRICS
            and is_zero(finding.get("candidate")) and is_zero(finding.get("baseline"))]


# ---------------------------------------------------------------------------
# Commits since the last green night.


@dataclass
class Commit:
    sha: str
    title: str
    pull_request: int | None


@dataclass
class CommitRange:
    commits: list[Commit]  # newest first
    total: int
    unavailable: str = ""  # why there is no list, when there is none


def split_pull_request(subject: str) -> tuple[str, int | None]:
    match = TRAILING_PULL_REQUEST.match(subject)
    if not match:
        return subject, None
    return match.group("title"), int(match.group("number"))


def make_commit(sha: str, subject: str) -> Commit:
    title, number = split_pull_request(" ".join(subject.split()))
    return Commit(sha=sha, title=title, pull_request=number)


def commits_from_git(previous: str, candidate: str) -> CommitRange | str:
    def git(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(CHECKOUT), *arguments],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    try:
        for commit in (previous, candidate):
            if git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
                return f"{commit[:10]} is not in this checkout"
        log = git("log", "--format=%H%x1f%s", f"{previous}..{candidate}")
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git failed: {exc}"
    if log.returncode != 0:
        return f"git log failed: {log.stderr.strip()[:200]}"
    commits = []
    for row in log.stdout.splitlines():
        sha, _, subject = row.partition("\x1f")
        if sha:
            commits.append(make_commit(sha, subject))
    return CommitRange(commits=commits, total=len(commits))


class GitHubError(Exception):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


def github_get(url: str, token: str) -> object:
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    })
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise GitHubError(exc.code, f"HTTP {exc.code} {exc.reason}") from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise GitHubError(None, f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}") from None


def commits_from_api(previous: str, candidate: str, environ: dict) -> CommitRange | str:
    url = f"{api_url(environ)}/repos/{repository(environ)}/compare/{previous}...{candidate}"
    try:
        data = github_get(url, github_token(environ))
    except GitHubError as exc:
        return f"compare API: {exc}"
    if not isinstance(data, dict) or not isinstance(data.get("commits"), list):
        return "compare API: unexpected response"
    commits = []
    for item in data["commits"]:
        if not isinstance(item, dict):
            continue
        message = (item.get("commit") or {}).get("message") or ""
        commits.append(make_commit(str(item.get("sha") or ""), message.split("\n", 1)[0]))
    commits.reverse()  # the API lists oldest first
    total = data.get("total_commits")
    return CommitRange(commits=commits, total=total if isinstance(total, int) else len(commits))


def commits_since_green(night: Night, environ: dict) -> CommitRange:
    previous, candidate = night.previous_green_commit, night.candidate_commit
    if previous is None:
        return CommitRange([], 0, "no earlier green night is recorded")
    if candidate is None:
        return CommitRange([], 0, "the candidate's commit is unknown")
    if previous == candidate:
        return CommitRange([], 0, "")
    from_git = commits_from_git(previous, candidate)
    if isinstance(from_git, CommitRange):
        return from_git
    from_api = commits_from_api(previous, candidate, environ)
    if isinstance(from_api, CommitRange):
        return from_api
    return CommitRange([], 0, f"commit list unavailable ({from_git}; {from_api})")


# ---------------------------------------------------------------------------
# Building the message.


@dataclass(frozen=True)
class Limits:
    findings_per_check: int
    findings_total: int
    commits: int
    issue_url_characters: int
    # Rows broken on both sides recur every night, so Chat only counts them; the
    # job summary names them.
    name_pre_existing: bool = False


# Tried in order until the Chat text fits CHAT_CHARACTER_BUDGET.
CHAT_LIMITS = (
    Limits(3, 10, 15, 1800),
    Limits(2, 6, 8, 1200),
    Limits(1, 3, 4, 800),
    Limits(1, 1, 0, 500),
)
SUMMARY_LIMITS = Limits(5, 25, 60, 6000, name_pre_existing=True)


def headline(night: Night, word: str) -> list[Segment]:
    return [Bold(f"Perf nightly {night.date}: {word}"), f" — {night.candidate_label} vs {night.baseline_label}"]


def run_link(night: Night) -> Segment:
    return Link(night.run_url, "run") if night.run_url else "run link unknown"


def problem_lines(night: Night) -> list[Line]:
    if not night.problems:
        return []
    return [Line([Bold("Problems with nightly.json:"), " " + "; ".join(night.problems)])]


def error_lines(night: Night) -> list[Line]:
    failed = [check for check in night.checks if check.verdict == "error"]
    if not failed:
        return []
    lines = [Line([Bold("Could not measure"), " (no verdict, not a pass):"])]
    lines += [Line([Bold(check.name), f": {check.error}"], bullet=True) for check in failed]
    return lines


def pre_existing_count(night: Night) -> str:
    """"2 rows broken on both sides (not new)", or "" when there are none."""
    count = len(broken_on_both_sides(night))
    if not count:
        return ""
    return f"{count} {'row' if count == 1 else 'rows'} broken on both sides (not new)"


def pre_existing_lines(night: Night, limits: Limits) -> list[Line]:
    rows = broken_on_both_sides(night)
    if not rows:
        return []
    if not limits.name_pre_existing:
        return [Line([f"{pre_existing_count(night).capitalize()}; named in the run's job summary."])]
    names = [Line([f"{name}: {describe_finding(finding)}"], bullet=True) for name, finding in rows]
    return [Line([Bold("Broken on both sides"), " (already broken on the baseline, not new):"]), *names]


def passed_line(night: Night) -> list[Line]:
    passed = [check.name for check in night.checks if check.verdict == "pass"]
    if not passed:
        return []
    return [Line([f"Passed: {', '.join(passed)}."])]


def green_message(night: Night, limits: Limits) -> list[Line]:
    count = len(night.checks)
    noun = "check" if count == 1 else "checks"
    pre_existing = pre_existing_count(night)
    detail = f", all {count} {noun} passed" + (f"; {pre_existing}" if pre_existing else "") + ". "
    lines = [Line([*headline(night, "green"), detail, run_link(night)])]
    if limits.name_pre_existing:
        lines += pre_existing_lines(night, limits)
    return lines


def skipped_message(night: Night) -> list[Line]:
    reason = night.skip_reason or "no reason given"
    return [Line(headline(night, "skipped") + [f": {reason}. ", run_link(night)])]


def error_message(night: Night, limits: Limits) -> list[Line]:
    failed = sum(check.verdict == "error" for check in night.checks)
    if failed:
        detail = f": {failed} of {len(night.checks)} checks could not measure; tonight is not green. "
    elif night.checks:
        detail = ": nightly.json has problems, so tonight is not green. "
    else:
        detail = ": no checks were measured; tonight is not green. "
    lines = [Line(headline(night, "ERROR") + [detail, run_link(night)])]
    lines += problem_lines(night) + error_lines(night) + passed_line(night) + pre_existing_lines(night, limits)
    return lines


def regressed_check_lines(night: Night, limits: Limits) -> list[Line]:
    regressed = [check for check in night.checks if check.verdict == "regression"]
    per_check = max(1, min(limits.findings_per_check, limits.findings_total // max(1, len(regressed))))
    lines = []
    for check in regressed:
        confirmed = "confirmed" if check.confirmed else "NOT confirmed by a re-run"
        summary = f": {check.summary}" if check.summary else ""
        lines.append(Line([Bold(check.name), f" ({confirmed}){summary}"]))
        rows = regression_findings(check)
        for finding in rows[:per_check]:
            lines.append(Line([describe_finding(finding)], bullet=True))
        if len(rows) > per_check:
            more = len(rows) - per_check
            noun = "row" if more == 1 else "rows"
            lines.append(Line([f"{more} more regression {noun} in the run's result.json"], bullet=True))
    return lines


def commit_lines(night: Night, commits: CommitRange, environ: dict, limits: Limits) -> list[Line]:
    previous, candidate = night.previous_green_commit, night.candidate_commit
    if previous and candidate and previous == candidate:
        return [Line(["Same candidate commit as the last green night (", Code(candidate[:10]),
                      "): nothing new was merged on main, so look at the baseline, the run or the machine."])]
    compare = f"{server_url(environ)}/{repository(environ)}/compare/{previous}...{candidate}"
    if commits.unavailable:
        line = Line([f"Commits since the last green night: {commits.unavailable}."])
        if previous and candidate:
            line.segments += [" ", Link(compare, "compare")]
        return [line]
    lines = [Line([f"Commits since the last green night ({commits.total}, ",
                   Link(compare, f"{previous[:7]}...{candidate[:7]}"), "):"])]
    if limits.commits == 0:
        return lines
    for commit in commits.commits[:limits.commits]:
        if commit.pull_request is not None:
            pull = f"{server_url(environ)}/{repository(environ)}/pull/{commit.pull_request}"
            lines.append(Line([Link(pull, f"#{commit.pull_request}"), f" {shorten(commit.title, 90)}"], bullet=True))
        else:
            lines.append(Line([Code(commit.sha[:7]), f" {shorten(commit.title, 90)} (no PR number in the title)"],
                              bullet=True))
    hidden = commits.total - min(len(commits.commits), limits.commits)
    if hidden > 0:
        lines.append(Line([f"{hidden} more, see the compare link"], bullet=True))
    return lines


def issue_link(night: Night, environ: dict, limits: Limits) -> str:
    regressed = [check for check in night.checks if check.verdict == "regression"]
    names = ", ".join(check.name for check in regressed) or "unknown check"
    title = f"perf: nightly {night.date} regression in {names} ({night.candidate_label} vs {night.baseline_label})"
    body = [
        f"The nightly perf run of {night.date} found a regression in {names}.",
        "",
        f"- Run: {night.run_url or 'unknown'}",
        f"- Candidate: {night.candidate_label} `{night.candidate_image or 'image unknown'}`",
        f"- Baseline: {night.baseline_label} `{night.baseline_image or 'image unknown'}`",
    ]
    if night.previous_green_commit and night.candidate_commit and night.previous_green_commit != night.candidate_commit:
        compare = (f"{server_url(environ)}/{repository(environ)}/compare/"
                   f"{night.previous_green_commit}...{night.candidate_commit}")
        body.append(f"- Commits since the last green night: {compare}")
    body += ["", "Worst findings:", ""]
    for check in regressed:
        for finding in regression_findings(check)[:3]:
            body.append(f"- {check.name}: {describe_finding(finding)}")
    base = f"{server_url(environ)}/{repository(environ)}/issues/new?"

    def build(lines: list[str]) -> str:
        query = {"title": title, "labels": ISSUE_LABEL, "body": "\n".join(lines)}
        return base + urllib.parse.urlencode(query, quote_via=urllib.parse.quote)

    url = build(body)
    while len(url) > limits.issue_url_characters and len(body) > 1:
        body = body[:-1]
        url = build(body + ["", "(trimmed: see the run)"])
    return url


def red_message(night: Night, commits: CommitRange, environ: dict, limits: Limits) -> list[Line]:
    regressed = sum(check.verdict == "regression" for check in night.checks)
    lines = [Line(headline(night, "REGRESSION") + [f": {regressed} of {len(night.checks)} checks regressed."])]
    lines += problem_lines(night)
    lines += regressed_check_lines(night, limits)
    lines += error_lines(night)
    lines += passed_line(night)
    lines += pre_existing_lines(night, limits)
    lines.append(BLANK)
    lines += commit_lines(night, commits, environ, limits)
    lines.append(BLANK)
    lines.append(Line([run_link(night), " · ", Link(issue_link(night, environ, limits), "open an issue")]))
    return lines


def build_message(night: Night, commits: CommitRange | None, environ: dict, limits: Limits) -> list[Line]:
    if night.status == "green":
        return green_message(night, limits)
    if night.status == "skipped":
        return skipped_message(night)
    if night.status == "red":
        return red_message(night, commits or CommitRange([], 0, "not looked up"), environ, limits)
    return error_message(night, limits)


def chat_text(night: Night, commits: CommitRange | None, environ: dict) -> str:
    lines: list[Line] = []
    for limits in CHAT_LIMITS:
        lines = build_message(night, commits, environ, limits)
        text = render_chat(lines)
        if len(text) <= CHAT_CHARACTER_BUDGET:
            return text
    # Still too long: keep whole lines from the top and say where the rest is.
    tail = Line(["(trimmed: the full report is in the run's job summary) ", run_link(night)])
    if night.status == "red":
        tail.segments += [" · ", Link(issue_link(night, environ, CHAT_LIMITS[-1]), "open an issue")]
    kept: list[Line] = []
    for line in lines:
        if len(render_chat([*kept, line, tail])) > CHAT_CHARACTER_BUDGET:
            break
        kept.append(line)
    return render_chat([*kept, tail])


# ---------------------------------------------------------------------------
# Delivery.


def warn(message: str, environ: dict) -> None:
    if environ.get("GITHUB_ACTIONS") == "true":
        print(f"::warning::{message}", file=sys.stderr)
    else:
        print(f"warning: {message}", file=sys.stderr)


def post_to_chat(webhook: str, text: str) -> str:
    """POST `text` to a Chat incoming webhook. Returns "" on success, else why not.

    Never puts the webhook URL in what it returns: the URL is the credential.
    """
    try:
        parts = urllib.parse.urlsplit(webhook)
        loopback = parts.scheme == "http" and parts.hostname in ("127.0.0.1", "localhost")  # the tests' stub
    except ValueError:
        return "PERF_CHAT_WEBHOOK is not a valid URL"
    if parts.scheme != "https" and not loopback:
        return "PERF_CHAT_WEBHOOK is not an https:// URL"
    body = json.dumps({"text": text}).encode("utf-8")
    failure = ""
    for attempt in range(len(POST_RETRY_DELAYS) + 1):
        if attempt:
            time.sleep(POST_RETRY_DELAYS[attempt - 1])
        request = urllib.request.Request(webhook, data=body, method="POST", headers={
            "Content-Type": "application/json; charset=UTF-8",
            "User-Agent": USER_AGENT,
        })
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                response.read()
            return ""
        except urllib.error.HTTPError as exc:
            failure = f"HTTP {exc.code} {exc.reason}"
            if exc.code != 429 and exc.code < 500:
                return failure
        except urllib.error.URLError as exc:
            # Raised while connecting or sending, before any response: retrying is safe.
            failure = f"connection failed: {type(exc.reason).__name__}"
        except TimeoutError:
            # Raised while waiting for the response: the message may have arrived.
            return "timed out waiting for Chat (not retried: the message may have arrived)"
        except (OSError, ValueError) as exc:
            failure = f"request failed: {type(exc).__name__}"
    return failure


def deliver(chat: str, markdown: str, environ: dict, post: bool = True) -> None:
    print(chat)
    summary = environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(markdown + "\n")
        except OSError as exc:
            warn(f"could not write the job summary: {exc}", environ)
    if not post:
        return
    webhook = (environ.get("PERF_CHAT_WEBHOOK") or "").strip()
    if not webhook:
        print("PERF_CHAT_WEBHOOK is not set: not posted to Chat.", file=sys.stderr)
        return
    failure = post_to_chat(webhook, chat)
    if failure:
        warn(f"posting to Chat failed: {failure}", environ)
    else:
        print("Posted to Chat.", file=sys.stderr)


def report(path: Path, environ: dict) -> tuple[str, str]:
    """The Chat text and the Markdown summary for one nightly.json."""
    night = read_night(path, environ)
    commits = commits_since_green(night, environ) if night.status == "red" else None
    chat = chat_text(night, commits, environ)
    markdown = render_markdown(build_message(night, commits, environ, SUMMARY_LIMITS))
    return chat, markdown


def crash_message(exc: BaseException, environ: dict) -> list[Line]:
    run_url = run_url_from_environment(environ)
    reason = text_or(f"{type(exc).__name__}: {exc}", type(exc).__name__)
    return [Line([Bold("Perf nightly: the reporter crashed"), f" ({reason}), so tonight's result was not reported. ",
                  Link(run_url, "run") if run_url else "run link unknown"])]


def main(argv: list[str] | None = None, environ: dict | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--nightly", required=True, type=Path, help="path to nightly.json")
    arguments = parser.parse_args(argv)
    environ = dict(os.environ) if environ is None else environ
    try:
        chat, markdown = report(arguments.nightly, environ)
    except Exception as exc:  # noqa: BLE001  (a broken reporter must still say so, not go silent)
        traceback.print_exc()
        lines = crash_message(exc, environ)
        deliver(render_chat(lines), render_markdown(lines), environ)
        return 1
    deliver(chat, markdown, environ)
    return 0


if __name__ == "__main__":
    sys.exit(main())
