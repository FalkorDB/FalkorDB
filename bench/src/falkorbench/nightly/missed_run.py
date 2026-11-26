"""The 08:30 UTC alarm: did the night's perf-nightly run leave its result?

`bench nightly missed-run [--date YYYY-MM-DD]`; see bench/README.md, "Nightly report and
missed-run alarm". Standard library only (and falkorbench.nightly.report, which is too).
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import time

from falkorbench.nightly import report

# perf-nightly.yml's cron and perf-missed-run.yml's. A night's runs are the scheduled ones
# created from the nightly's start to the alarm, so a late run from the night before cannot
# stand in for this one; the longest night fits before the alarm (tests pin both).
NIGHTLY_START_UTC = time(0, 0)
ALARM_TIME_UTC = time(8, 30)


@dataclass
class Outcome:
    alarm: bool
    chat: str  # the line for Chat and stdout
    markdown: str  # the same line for the job summary


def window(night: date) -> tuple[datetime, datetime]:
    start = datetime.combine(night, NIGHTLY_START_UTC, tzinfo=UTC)
    return start, datetime.combine(night, ALARM_TIME_UTC, tzinfo=UTC)


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text: object) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def runs_in_window(
    environ: Mapping[str, str],
    workflow: str,
    branch: str,
    event: str,
    start: datetime,
    end: datetime,
) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "branch": branch,
            "event": event,
            "created": f"{iso(start)}..{iso(end)}",
            "per_page": 100,
            "exclude_pull_requests": "true",
        }
    )
    url = (
        f"{report.api_url(environ)}/repos/{report.repository(environ)}/actions/workflows/"
        f"{urllib.parse.quote(workflow)}/runs?{query}"
    )
    data = report.github_get(url, report.github_token(environ))
    runs = data.get("workflow_runs") if isinstance(data, dict) else None
    if not isinstance(runs, list):
        raise report.GitHubError(None, "unexpected response listing workflow runs")
    # The `created` filter's edges are inclusive; the window is [start, end).
    kept = []
    for run in runs:
        created = parse_time(run.get("created_at")) if isinstance(run, dict) else None
        if (
            created is not None
            and start <= created < end
            and run.get("head_branch") == branch
            and run.get("event") == event
        ):
            kept.append(run)
    return sorted(kept, key=lambda run: run["created_at"], reverse=True)


def has_artifact(environ: Mapping[str, str], run: dict, artifact: str) -> bool:
    query = urllib.parse.urlencode({"name": artifact, "per_page": 100})
    url = f"{report.api_url(environ)}/repos/{report.repository(environ)}/actions/runs/{run['id']}/artifacts?{query}"
    data = report.github_get(url, report.github_token(environ))
    artifacts = data.get("artifacts") if isinstance(data, dict) else None
    if not isinstance(artifacts, list):
        raise report.GitHubError(None, "unexpected response listing artifacts")
    return any(
        isinstance(item, dict) and item.get("name") == artifact and not item.get("expired")
        for item in artifacts
    )


def runs_page(environ: Mapping[str, str], workflow: str, branch: str) -> str:
    query = urllib.parse.urlencode({"query": f"branch:{branch}"})
    return f"{report.server_url(environ)}/{report.repository(environ)}/actions/workflows/{workflow}?{query}"


def alarm(night: date, reason: str, link: str, link_text: str) -> Outcome:
    title, text = f"Perf nightly alarm {night.isoformat()}", f": no nightly result. {reason} "
    return Outcome(
        True,
        f"*{report.chat_plain(title)}*{report.chat_plain(text)}{report.chat_link(link, link_text)}",
        f"**{report.markdown_plain(title)}**{report.markdown_plain(text)}{report.markdown_link(link, link_text)}",
    )


def check(
    environ: Mapping[str, str],
    night: date,
    workflow: str,
    branch: str,
    event: str,
    artifact: str,
    now: datetime,
) -> Outcome:
    start, end = window(night)
    span = f"between {start:%Y-%m-%d %H:%M} and {end:%Y-%m-%d %H:%M} UTC"
    listing = runs_page(environ, workflow, branch)
    try:
        runs = runs_in_window(environ, workflow, branch, event, start, end)
    except report.GitHubError as exc:
        if exc.status == 404:
            return alarm(
                night,
                f"{workflow} was not found in {report.repository(environ)}.",
                listing,
                "workflow",
            )
        return alarm(
            night,
            f"Could not ask GitHub for {workflow} runs ({exc}), so it is unknown whether one ran.",
            listing,
            "workflow runs",
        )
    if not runs:
        return alarm(
            night, f"No {event} run of {workflow} on {branch} {span}.", listing, "workflow runs"
        )

    try:
        for run in runs:
            if has_artifact(environ, run, artifact):
                url, run_id = str(run.get("html_url") or listing), str(run.get("id"))
                before, after = (
                    f"Perf nightly {night.isoformat()}: run ",
                    f" has the {artifact} artifact; no alarm.",
                )
                chat = (
                    report.chat_plain(before)
                    + report.chat_link(url, run_id)
                    + report.chat_plain(after)
                )
                markdown = (
                    report.markdown_plain(before)
                    + report.markdown_link(url, run_id)
                    + report.markdown_plain(after)
                )
                return Outcome(False, chat, markdown)
    except report.GitHubError as exc:
        return alarm(
            night,
            f"Could not list the artifacts of tonight's run ({exc}).",
            str(runs[0].get("html_url") or listing),
            "run",
        )

    newest = runs[0]
    link = str(newest.get("html_url") or listing)
    others = f" ({len(runs)} runs {span}; none has it.)" if len(runs) > 1 else ""
    started = parse_time(newest.get("run_started_at")) or parse_time(newest.get("created_at"))
    started_text = f" (started {started:%Y-%m-%d %H:%M} UTC)" if started else ""
    if newest.get("status") != "completed":
        # Not yet, not missed: a running night still writes and posts its result when it
        # ends (the merge and report jobs run after any check fails or times out).
        before = f"Perf nightly {night.isoformat()}: not yet. "
        after = (
            f"The run{started_text} is still {newest.get('status')} at {now:%Y-%m-%d %H:%M} UTC; "
            f"its report follows when it ends."
        )
        return Outcome(
            False,
            report.chat_plain(before)
            + report.chat_link(link, "run")
            + " "
            + report.chat_plain(after),
            report.markdown_plain(before)
            + report.markdown_link(link, "run")
            + " "
            + report.markdown_plain(after),
        )
    else:
        reason = (
            f"The run finished `{newest.get('conclusion')}` without uploading {artifact}.{others}"
        )
    return alarm(night, reason, link, "run")


def scheduled_night(environ: Mapping[str, str]) -> date | None:
    """The night a scheduled run was created for, so a re-run days later checks that night.

    None outside a scheduled Actions run, or when GitHub cannot say; the caller then
    uses today.
    """
    run_id = environ.get("GITHUB_RUN_ID") or ""
    if environ.get("GITHUB_EVENT_NAME") != "schedule" or not run_id.isdigit():
        return None
    url = f"{report.api_url(environ)}/repos/{report.repository(environ)}/actions/runs/{run_id}"
    try:
        data = report.github_get(url, report.github_token(environ))
    except report.GitHubError:
        return None
    created = parse_time(data.get("created_at")) if isinstance(data, dict) else None
    return created.date() if created else None


def run(
    environ: Mapping[str, str],
    night: date | None = None,
    *,
    workflow: str = "perf-nightly.yml",
    branch: str = "main",
    event: str = "schedule",
    artifact: str = "perf-nightly",
    now: datetime | None = None,
) -> int:
    """Check one night (default: the night a scheduled run was created for, else today).

    0 when the night has its result, 1 when the alarm went out.
    """
    now = now or datetime.now(UTC)
    night = night or scheduled_night(environ) or now.date()
    outcome = check(environ, night, workflow, branch, event, artifact, now)
    report.deliver(outcome.chat, outcome.markdown, environ, post=outcome.alarm)
    return 1 if outcome.alarm else 0
