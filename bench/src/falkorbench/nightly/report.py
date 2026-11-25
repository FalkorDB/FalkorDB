"""Report one perf nightly: one line to Chat, the details on the run page.

`bench nightly report --nightly <nightly.json>`; see bench/README.md, "Nightly report and
missed-run alarm". Standard library only (and falkorbench.contract, which is too).
"""

from __future__ import annotations

import http.client
import json
import math
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from falkorbench import contract

DEFAULT_REPOSITORY = "FalkorDB/FalkorDB"
DEFAULT_SERVER_URL = "https://github.com"
DEFAULT_API_URL = "https://api.github.com"
USER_AGENT = "falkordb-perf-report"
# Comma-separated, as GitHub's issues/new `labels` parameter takes them.
ISSUE_LABELS = "performance,perf-regression"
# GitHub refuses very long URLs (414); the body is cut to stay well under that.
ISSUE_URL_CHARACTERS = 2000

TEXT_CHARACTERS = 300
NAME_CHARACTERS = 80
RUN_URL_CHARACTERS = 300  # a run link is about 70
WORST_ROWS = 5
COMMITS_SHOWN = 30
TRAILING_PULL_REQUEST = re.compile(r"^(?P<title>.*?)\s*\(#(?P<number>\d+)\)\s*$")
# A URL that can sit inside Chat's `<url|text>` and Markdown's `[text](url)` intact.
SAFE_URL = re.compile(r"https?://[^\s<>|\"'`()\[\]]+")

# The checkout whose `git log` lists the commits since the last green night.
CHECKOUT = Path(__file__).resolve().parent
# Seconds to wait before each retry of a webhook post. Tests set it to zeros.
POST_RETRY_DELAYS = (1.0, 3.0)
HTTP_TIMEOUT_SECONDS = 20


# ---------------------------------------------------------------------------
# Text that came from a file or a commit, made safe for Chat and Markdown.


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def one_line(value: object, fallback: str, limit: int = TEXT_CHARACTERS) -> str:
    if isinstance(value, str) and value.strip():
        return shorten(" ".join(value.split()), limit)
    return fallback


def chat_plain(text: str) -> str:
    # In Chat markup `<` opens a link or a mention (`<users/all>` pings the space).
    return text.replace("<", "‹")


def chat_link(url: str, text: str) -> str:
    if not SAFE_URL.fullmatch(url):
        return chat_plain(text)
    return f"<{url}|{chat_plain(text).replace('|', '/').replace('>', '›')}>"


def markdown_plain(text: str) -> str:
    return re.sub(r"([\\`*_\[\]~|#>])", r"\\\1", text.replace("&", "&amp;").replace("<", "&lt;"))


def markdown_link(url: str, text: str) -> str:
    return f"[{markdown_plain(text)}]({url})" if SAFE_URL.fullmatch(url) else markdown_plain(text)


# ---------------------------------------------------------------------------
# The environment: which repository, which run, which API.


def server_url(environ: Mapping[str, str]) -> str:
    return (environ.get("GITHUB_SERVER_URL") or DEFAULT_SERVER_URL).rstrip("/")


def api_url(environ: Mapping[str, str]) -> str:
    return (environ.get("GITHUB_API_URL") or DEFAULT_API_URL).rstrip("/")


def repository(environ: Mapping[str, str]) -> str:
    return environ.get("GITHUB_REPOSITORY") or DEFAULT_REPOSITORY


def github_token(environ: Mapping[str, str]) -> str:
    return environ.get("GH_TOKEN") or environ.get("GITHUB_TOKEN") or ""


def current_run_url(environ: Mapping[str, str]) -> str:
    run_id = environ.get("GITHUB_RUN_ID") or ""
    url = f"{server_url(environ)}/{repository(environ)}/actions/runs/{run_id}"
    return (
        url
        if run_id.isdigit() and SAFE_URL.fullmatch(url) and len(url) <= RUN_URL_CHARACTERS
        else ""
    )


class GitHubError(Exception):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


def github_get(url: str, token: str) -> object:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        },
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise GitHubError(exc.code, f"HTTP {exc.code} {exc.reason}") from None
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
        # HTTPException covers a body cut short (IncompleteRead) after the headers arrived.
        raise GitHubError(None, f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}") from None


# ---------------------------------------------------------------------------
# The night.


def read_night(path: Path) -> tuple[dict, str, str]:
    """The nightly document, the status to report, and why it is ERROR if it had to be."""
    try:
        night = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {}, "error", shorten(f"could not read {path.name}: {exc}", TEXT_CHARACTERS)
    if not isinstance(night, dict):
        return {}, "error", f"{path.name} is not a JSON object"
    status = night.get("status")
    if status not in contract.NIGHTLY_STATUSES:
        return night, "error", shorten(f"unknown status {status!r} in {path.name}", TEXT_CHARACTERS)
    return night, status, ""


def checks_of(night: dict) -> list[dict]:
    checks = night.get("checks")
    return (
        [check for check in checks if isinstance(check, dict)] if isinstance(checks, list) else []
    )


def name_of(check: dict) -> str:
    return one_line(check.get("check"), "unnamed check", NAME_CHARACTERS)


def label(side: object) -> str:
    side = side if isinstance(side, dict) else {}
    return one_line(
        side.get("label"), one_line(side.get("image"), "unknown", NAME_CHARACTERS), NAME_CHARACTERS
    )


def run_url_of(night: dict, environ: Mapping[str, str]) -> str:
    url = night.get("run_url")
    if (
        isinstance(url, str)
        and url.startswith("https://")
        and SAFE_URL.fullmatch(url)
        and len(url) <= RUN_URL_CHARACTERS
    ):
        return url
    return current_run_url(environ)


def regressed_checks(night: dict) -> list[dict]:
    return [check for check in checks_of(night) if check.get("verdict") == "regression"]


def issue_url(night: dict, environ: Mapping[str, str]) -> str:
    """A pre-filled new issue for a red night, built only from nightly.json, in plain text.

    The body is cut line by line (and then shortened) so the URL stays under
    ISSUE_URL_CHARACTERS; the link itself is never dropped.
    """
    date = one_line(night.get("date"), "date unknown", 20)
    names = ", ".join(name_of(check) for check in regressed_checks(night)) or "unknown check"
    title = shorten(f"Perf regression {date}: {names}", 150)
    url = run_url_of(night, environ)
    body = [
        f"{label(night.get('candidate'))} vs {label(night.get('baseline'))}",
        f"Run: {url or 'unknown'}",
        "",
    ]
    body += [
        f"- {name_of(check)}: {one_line(check.get('summary'), 'no summary')}"
        for check in regressed_checks(night)
    ]
    base = f"{server_url(environ)}/{repository(environ)}/issues/new?"

    def build(text: str) -> str:
        query = {"title": title, "labels": ISSUE_LABELS, "body": text}
        return base + urllib.parse.urlencode(query, safe=",", quote_via=urllib.parse.quote)

    text = "\n".join(body)
    while len(build(text)) > ISSUE_URL_CHARACTERS and len(body) > 3:
        body = body[:-1]
        text = "\n".join([*body, "(more on the run page)"])
    while len(build(text)) > ISSUE_URL_CHARACTERS and len(text) > 1:
        text = shorten(text, len(text) * 3 // 4)
    return build(text)


def chat_line(night: dict, status: str, reason: str, environ: Mapping[str, str]) -> str:
    """Perf nightly 2026-11-24: RED (micro-hw, pokec), main@4c10889 vs 6.2.0 · details · new issue"""
    date = one_line(night.get("date"), "date unknown", 20)
    checks = checks_of(night)
    if reason:
        detail = reason
    elif status == "red":
        detail = ", ".join(
            name_of(check) for check in checks if check.get("verdict") == "regression"
        )
    elif status == "error":
        detail = ", ".join(name_of(check) for check in checks if check.get("verdict") == "error")
    elif status == "skipped":
        detail = one_line(night.get("skip_reason"), "no reason given")
    else:
        detail = f"{len(checks)} {'check' if len(checks) == 1 else 'checks'}"
    versus = f"{label(night.get('candidate'))} vs {label(night.get('baseline'))}" if night else ""
    text = chat_plain(f"Perf nightly {date}: {status.upper()} ({shorten(detail, TEXT_CHARACTERS)})")
    text += chat_plain(f", {versus}") if versus else ""
    url = run_url_of(night, environ)
    text += f" · {chat_link(url, 'details')}" if url else ""
    if status == "red" and not reason:
        text += f" · {chat_link(issue_url(night, environ), 'new issue')}"
    return text


# ---------------------------------------------------------------------------
# The run page.


def number(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return one_line(str(value), "n/a", 40)
    try:
        as_float = float(value)
    except OverflowError:
        return shorten(f"{value:,}", 40)
    if not math.isfinite(as_float):
        return str(as_float)
    if abs(as_float) >= 1000:
        return f"{as_float:,.0f}"
    return str(value) if isinstance(value, int) else f"{as_float:.3g}"


def ratio(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    try:
        as_float = float(value)
    except OverflowError:
        return ""
    return f"{as_float:.2f}x" if math.isfinite(as_float) else ""


def candidate_commit(night: dict) -> object:
    return night["candidate"].get("commit") if isinstance(night.get("candidate"), dict) else None


def commits_since_green(night: dict) -> tuple[list[tuple[str, str, int | None]], str]:
    """(sha, title, PR number) newest first from this checkout's git log, or why there are none."""
    previous = night.get("previous_green_commit")
    candidate = candidate_commit(night)
    if not (isinstance(previous, str) and contract.COMMIT_RE.fullmatch(previous)):
        return [], "no earlier green night is recorded"
    if not (isinstance(candidate, str) and contract.COMMIT_RE.fullmatch(candidate)):
        return [], "the candidate's commit is unknown"
    if previous == candidate:
        return [], "same candidate commit as the last green night: nothing new was merged on main"
    try:
        log = subprocess.run(
            ["git", "-C", str(CHECKOUT), "log", "--format=%H%x1f%s", f"{previous}..{candidate}"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"git failed: {type(exc).__name__}"
    if log.returncode != 0:
        return [], "the commits are not in this checkout"
    commits = []
    for row in log.stdout.splitlines():
        sha, _, subject = row.partition("\x1f")
        match = TRAILING_PULL_REQUEST.match(subject)
        title, pull = (
            (match.group("title"), int(match.group("number"))) if match else (subject, None)
        )
        commits.append((sha, " ".join(title.split()), pull))
    return commits, ""


def regression_lines(check: dict) -> list[str]:
    rows = [
        row
        for row in check.get("findings") or []
        if isinstance(row, dict) and row.get("status") == "regression"
    ]
    lines = [
        f"**{markdown_plain(name_of(check))}**, worst regression rows (in the check's order):",
        "",
    ]
    for row in rows[:WORST_ROWS]:
        threshold = ratio(row.get("threshold"))
        extras = ", ".join(
            part
            for part in (ratio(row.get("ratio")), threshold and f"threshold {threshold}")
            if part
        )
        lines.append(
            f"- {markdown_plain(one_line(row.get('name'), 'unnamed'))}: "
            f"{markdown_plain(one_line(row.get('metric'), '?', 40))} {number(row.get('candidate'))} vs "
            f"{number(row.get('baseline'))}" + (f" ({extras})" if extras else "")
        )
    if len(rows) > WORST_ROWS:
        lines.append(f"- {len(rows) - WORST_ROWS} more in the check's result.json")
    return [*lines, ""]


def commit_lines(night: dict, environ: Mapping[str, str]) -> list[str]:
    server, repo = server_url(environ), repository(environ)
    commits, why = commits_since_green(night)
    compare = (
        f"{server}/{repo}/compare/{night.get('previous_green_commit')}...{candidate_commit(night)}"
    )
    if why:
        link = (
            f" {markdown_link(compare, 'compare')}"
            if why == "the commits are not in this checkout"
            else ""
        )
        return [f"Commits since the last green night: {markdown_plain(why)}.{link}", ""]
    lines = [
        f"Commits since the last green night ({len(commits)}, {markdown_link(compare, 'compare')}):",
        "",
    ]
    for sha, title, pull in commits[:COMMITS_SHOWN]:
        what = markdown_link(f"{server}/{repo}/pull/{pull}", f"#{pull}") if pull else f"`{sha[:7]}`"
        lines.append(f"- {what} {markdown_plain(shorten(title, 100))}")
    if len(commits) > COMMITS_SHOWN:
        lines.append(f"- {len(commits) - COMMITS_SHOWN} more")
    return [*lines, ""]


def summary_markdown(night: dict, status: str, reason: str, environ: Mapping[str, str]) -> str:
    checks = checks_of(night)
    url = run_url_of(night, environ)
    date = one_line(night.get("date"), "date unknown", 20)
    lines = [f"## Perf nightly {markdown_plain(date)}: {status.upper()}", ""]
    if reason:
        lines += [f"**Not green:** {markdown_plain(reason)}", ""]
    if night:
        versus = f"{markdown_plain(label(night.get('candidate')))} vs {markdown_plain(label(night.get('baseline')))}"
        lines += [versus + (f" · {markdown_link(url, 'run')}" if url else ""), ""]
    if status == "skipped":
        lines += [
            f"Skipped: {markdown_plain(one_line(night.get('skip_reason'), 'no reason given'))}",
            "",
        ]
    if checks:
        lines += ["| check | verdict | summary |", "| --- | --- | --- |"]
        for check in checks:
            text = check.get("error") if check.get("verdict") == "error" else check.get("summary")
            lines.append(
                f"| {markdown_plain(name_of(check))} | {markdown_plain(one_line(check.get('verdict'), '?'))} "
                f"| {markdown_plain(one_line(text, ''))} |"
            )
        lines.append("")
    for check in regressed_checks(night):
        lines += regression_lines(check)
    if status == "red":
        lines += commit_lines(night, environ)
        lines += [markdown_link(issue_url(night, environ), "Open an issue for this regression"), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Delivery.


def warn(message: str, environ: Mapping[str, str]) -> None:
    prefix = "::warning::" if environ.get("GITHUB_ACTIONS") == "true" else "warning: "
    print(f"{prefix}{message}", file=sys.stderr)


def post_to_chat(webhook: str, text: str) -> str:
    """POST `text` to a Chat incoming webhook. Returns "" on success, else why not.

    Never puts the webhook URL in what it returns: the URL is the credential.
    """
    try:
        parts = urllib.parse.urlsplit(webhook)
        loopback = parts.scheme == "http" and parts.hostname in (
            "127.0.0.1",
            "localhost",
        )  # the tests' stub
    except ValueError:
        return "PERF_CHAT_WEBHOOK is not a valid URL"
    if parts.scheme != "https" and not loopback:
        return "PERF_CHAT_WEBHOOK is not an https:// URL"
    body = json.dumps({"text": text}).encode("utf-8")
    failure = ""
    for attempt in range(len(POST_RETRY_DELAYS) + 1):
        if attempt:
            time.sleep(POST_RETRY_DELAYS[attempt - 1])
        request = urllib.request.Request(
            webhook,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=UTF-8",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            # The answer's body is not used, so it is not read: a body cut short after
            # a 2xx must not turn a delivered message into a failure.
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS):
                pass
            return ""
        except urllib.error.HTTPError as exc:
            failure = f"HTTP {exc.code} {exc.reason}"
            if exc.code != 429 and exc.code < 500:
                return failure
        except urllib.error.URLError as exc:
            # urllib raises URLError for errors while connecting or sending: nothing
            # arrived yet, so retrying is safe.
            failure = f"connection failed: {type(exc.reason).__name__}"
        except (TimeoutError, http.client.HTTPException, OSError, ValueError) as exc:
            # Anything else came while waiting for the answer: the message may have arrived.
            return f"no answer from Chat: {type(exc).__name__} (not retried: the message may have arrived)"
    return failure


def deliver(chat: str, markdown: str, environ: Mapping[str, str], post: bool = True) -> None:
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
    try:
        failure = post_to_chat(webhook, chat)
    except Exception as exc:
        failure = f"unexpected {type(exc).__name__}"
    if failure:
        warn(f"posting to Chat failed: {failure}", environ)
    else:
        print("Posted to Chat.", file=sys.stderr)


def run(path: Path, environ: Mapping[str, str]) -> int:
    """Report the night in `path`. 0 when it was reported, 1 when it could not be."""
    try:
        night, status, reason = read_night(path)
        chat, markdown = (
            chat_line(night, status, reason, environ),
            summary_markdown(night, status, reason, environ),
        )
    except Exception as exc:
        traceback.print_exc()
        reason = one_line(
            f"the reporter crashed: {type(exc).__name__}: {exc}", "the reporter crashed"
        )
        status, url = "error", current_run_url(environ)
        chat = chat_plain(f"Perf nightly: ERROR ({reason})") + (
            f" · {chat_link(url, 'details')}" if url else ""
        )
        markdown = f"## Perf nightly: ERROR\n\n**Not green:** {markdown_plain(reason)}\n"
    deliver(chat, markdown, environ)
    # `reason` is set exactly when the night could not be reported as written.
    return 1 if reason else 0
