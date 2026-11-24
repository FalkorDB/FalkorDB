#!/usr/bin/env python3
"""The 05:00 UTC alarm: did tonight's perf-nightly run leave its result?

    python -I tests/perf/report/missed_run.py [--date YYYY-MM-DD]

Looks at the scheduled runs of `perf-nightly.yml` on `main` created in the 24 hours
before <date> 05:00 UTC. It stays quiet when one of them uploaded the `perf-nightly`
artifact (nightly.json: the reporter already spoke for that night). A hand-started
run does not count: one given its own images uploads nightly.json without posting
it, so it must not stand in for the night's scheduled run. Otherwise it
raises the alarm and says which case it is: no run, a run still going, or a run
that finished without the artifact. The alarm goes out the same way as the nightly
report (stdout, `$GITHUB_STEP_SUMMARY`, `$PERF_CHAT_WEBHOOK`) and the exit code is
1, so the scheduled run itself fails too. A check that cannot reach GitHub is an
alarm as well: not knowing is not quiet.

`--workflow`, `--branch`, `--event` and `--artifact` exist so the alarm can be
accepted against other runs; the defaults are the nightly's.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

# `python -I` leaves the script's own directory off sys.path; post.py sits next to it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import post  # needs the path above

ALARM_HOUR_UTC = 5


@dataclass
class Outcome:
    alarm: bool
    chat: str  # the line for Chat and stdout
    markdown: str  # the same line for the job summary


def window(night: date) -> tuple[datetime, datetime]:
    end = datetime.combine(night, time(ALARM_HOUR_UTC), tzinfo=timezone.utc)
    return end - timedelta(days=1), end


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text: object) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def runs_in_window(environ: dict, workflow: str, branch: str, event: str, start: datetime,
                   end: datetime) -> list[dict]:
    query = urllib.parse.urlencode({
        "branch": branch,
        "event": event,
        "created": f"{iso(start)}..{iso(end)}",
        "per_page": 100,
        "exclude_pull_requests": "true",
    })
    url = (f"{post.api_url(environ)}/repos/{post.repository(environ)}/actions/workflows/"
           f"{urllib.parse.quote(workflow)}/runs?{query}")
    data = post.github_get(url, post.github_token(environ))
    runs = data.get("workflow_runs") if isinstance(data, dict) else None
    if not isinstance(runs, list):
        raise post.GitHubError(None, "unexpected response listing workflow runs")
    # The `created` filter's edges are inclusive; the window is [start, end).
    kept = []
    for run in runs:
        created = parse_time(run.get("created_at")) if isinstance(run, dict) else None
        if (created is not None and start <= created < end and run.get("head_branch") == branch
                and run.get("event") == event):
            kept.append(run)
    return sorted(kept, key=lambda run: run["created_at"], reverse=True)


def has_artifact(environ: dict, run: dict, artifact: str) -> bool:
    query = urllib.parse.urlencode({"name": artifact, "per_page": 100})
    url = f"{post.api_url(environ)}/repos/{post.repository(environ)}/actions/runs/{run['id']}/artifacts?{query}"
    data = post.github_get(url, post.github_token(environ))
    artifacts = data.get("artifacts") if isinstance(data, dict) else None
    if not isinstance(artifacts, list):
        raise post.GitHubError(None, "unexpected response listing artifacts")
    return any(isinstance(item, dict) and item.get("name") == artifact and not item.get("expired")
               for item in artifacts)


def runs_page(environ: dict, workflow: str, branch: str) -> str:
    query = urllib.parse.urlencode({"query": f"branch:{branch}"})
    return f"{post.server_url(environ)}/{post.repository(environ)}/actions/workflows/{workflow}?{query}"


def alarm(night: date, reason: str, link: str, link_text: str) -> Outcome:
    title, text = f"Perf nightly alarm {night.isoformat()}", f": no nightly result. {reason} "
    return Outcome(True,
                   f"*{post.chat_plain(title)}*{post.chat_plain(text)}{post.chat_link(link, link_text)}",
                   f"**{post.markdown_plain(title)}**{post.markdown_plain(text)}{post.markdown_link(link, link_text)}")


def check(environ: dict, night: date, workflow: str, branch: str, event: str, artifact: str,
          now: datetime) -> Outcome:
    start, end = window(night)
    span = f"between {start:%Y-%m-%d %H:%M} and {end:%Y-%m-%d %H:%M} UTC"
    listing = runs_page(environ, workflow, branch)
    try:
        runs = runs_in_window(environ, workflow, branch, event, start, end)
    except post.GitHubError as exc:
        if exc.status == 404:
            return alarm(night, f"{workflow} was not found in {post.repository(environ)}.", listing, "workflow")
        return alarm(night, f"Could not ask GitHub for {workflow} runs ({exc}), so it is unknown whether one ran.",
                     listing, "workflow runs")
    if not runs:
        return alarm(night, f"No {event} run of {workflow} on {branch} {span}.", listing, "workflow runs")

    try:
        for run in runs:
            if has_artifact(environ, run, artifact):
                url, run_id = str(run.get("html_url") or listing), str(run.get("id"))
                before, after = f"Perf nightly {night.isoformat()}: run ", f" has the {artifact} artifact; no alarm."
                chat = post.chat_plain(before) + post.chat_link(url, run_id) + post.chat_plain(after)
                markdown = post.markdown_plain(before) + post.markdown_link(url, run_id) + post.markdown_plain(after)
                return Outcome(False, chat, markdown)
    except post.GitHubError as exc:
        return alarm(night, f"Could not list the artifacts of tonight's run ({exc}).",
                     str(runs[0].get("html_url") or listing), "run")

    newest = runs[0]
    link = str(newest.get("html_url") or listing)
    others = f" ({len(runs)} runs {span}; none has it.)" if len(runs) > 1 else ""
    started = parse_time(newest.get("run_started_at")) or parse_time(newest.get("created_at"))
    started_text = f" (started {started:%Y-%m-%d %H:%M} UTC)" if started else ""
    if newest.get("status") != "completed":
        reason = (f"The run{started_text} is still {newest.get('status')} at {now:%Y-%m-%d %H:%M} UTC and has not "
                  f"uploaded {artifact}.{others}")
    else:
        reason = f"The run finished `{newest.get('conclusion')}` without uploading {artifact}.{others}"
    return alarm(night, reason, link, "run")


def scheduled_night(environ: dict) -> date | None:
    """The night a scheduled run was created for, so a re-run days later checks that night.

    None outside a scheduled Actions run, or when GitHub cannot say; the caller then
    uses today.
    """
    run_id = environ.get("GITHUB_RUN_ID") or ""
    if environ.get("GITHUB_EVENT_NAME") != "schedule" or not run_id.isdigit():
        return None
    url = f"{post.api_url(environ)}/repos/{post.repository(environ)}/actions/runs/{run_id}"
    try:
        data = post.github_get(url, post.github_token(environ))
    except post.GitHubError:
        return None
    created = parse_time(data.get("created_at")) if isinstance(data, dict) else None
    return created.date() if created else None


def main(argv: list[str] | None = None, environ: dict | None = None, now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--date", type=date.fromisoformat, default=None,
                        help="the night to check, YYYY-MM-DD (UTC); default: the night a scheduled run "
                             "was created for, else today")
    parser.add_argument("--workflow", default="perf-nightly.yml")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--event", default="schedule", help="the run's trigger (default: schedule)")
    parser.add_argument("--artifact", default="perf-nightly")
    arguments = parser.parse_args(argv)
    environ = dict(os.environ) if environ is None else environ

    night = arguments.date or scheduled_night(environ) or now.date()
    outcome = check(environ, night, arguments.workflow, arguments.branch, arguments.event, arguments.artifact, now)
    post.deliver(outcome.chat, outcome.markdown, environ, post=outcome.alarm)
    return 1 if outcome.alarm else 0


if __name__ == "__main__":
    sys.exit(main())
