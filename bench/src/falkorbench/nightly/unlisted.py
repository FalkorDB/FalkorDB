"""The dashboard's unlisted pages: one per run that is not a night, kept 30 days.

`bench nightly unlisted --out <site> [--days 30]` writes, into the site perf-dashboard.yml is
about to deploy:

  manual/<date>-<run id>/index.html   a Perf nightly dispatch that named its own images
  pr/<N>/<date>-<run id>/index.html   a `/perf` run on pull request <N>

Each page is the run's page itself (noindex, and no link back to the runs list, which does not
list it either), over the run as `bench nightly store` lays it out: only the files the page reads
or links, of types a browser shows as text (store.PAGE_FILES, store.SHOWN), since a `/perf`
candidate is pull request code. Nothing here is stored on the gh-pages
branch: every deploy rebuilds these pages from the runs created in the last <days> days, so a
page is gone at the first deploy after its run is older than that.

`bench nightly unlisted url --kind manual|pr ...` prints a run's page URL (the `/perf` answer and
a manual run's summary link it). Standard library only (with nightly.store and dashboard).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from falkorbench.nightly import dashboard
from falkorbench.nightly import store

DAYS = 30
PR_TITLE = re.compile(r"/perf on #([1-9][0-9]{0,6})")
PAGES = 50
NIGHTLY = "perf-nightly.yml"
COMMENT = "perf-comment.yml"


@dataclass(frozen=True)
class Candidate:
    run_id: str
    kind: str  # manual | pr
    pr: int | None
    created: datetime


def pages_url(repo: str, environ: Mapping[str, str]) -> str:
    """The Pages site's root, ending in `/`: $PERF_PAGES_URL (a repository variable, for a custom
    domain), else GitHub's default for a project site."""
    given = (environ.get("PERF_PAGES_URL") or "").strip()
    if given:
        return given.rstrip("/") + "/"
    owner, _, name = repo.partition("/")
    return f"https://{owner.lower()}.github.io/{name}/"


def page_path(kind: str, folder: str, pr: int | None = None) -> str:
    if kind == "pr":
        return f"pr/{pr}/{folder}/"
    return f"manual/{folder}/"


def _created(text: object) -> datetime | None:
    try:
        return datetime.strptime(str(text), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def _get(repo: str, path: str, run: store.Runner) -> object:
    done = store._gh(
        ["gh", "api", f"repos/{repo}/{path}"],
        run,
        store.LIST_TIMEOUT,
        capture_output=True,
        text=True,
    )
    if done is None or done.returncode != 0:
        return None
    try:
        return json.loads(done.stdout)
    except ValueError:
        return None


def candidates(repo: str, now: datetime, days: int, run: store.Runner) -> list[Candidate]:
    """The runs that get an unlisted page, created in the last `days` days, newest first.

    Found through their `perf-nightly` artifacts (every run that measured leaves one), not the
    runs listing: perf-comment.yml runs on every comment in the repository, and the API returns
    at most 1,000 runs of a filtered listing, far fewer than a month of comments.
    """
    since = now - timedelta(days=days)
    seen: set[int] = set()
    out = []
    for page in range(1, PAGES + 1):
        body = _get(repo, f"actions/artifacts?name=perf-nightly&per_page=100&page={page}", run)
        if not isinstance(body, dict):
            store.warn(f"could not list the perf-nightly artifacts (page {page})")
            break
        artifacts = [a for a in body.get("artifacts") or [] if isinstance(a, dict)]
        for artifact in artifacts:
            made = _created(artifact.get("created_at"))
            run_id = (artifact.get("workflow_run") or {}).get("id")
            # A run's artifact comes after the run starts: an older artifact is an older run.
            if (
                made is None
                or made < since
                or artifact.get("expired")
                or not isinstance(run_id, int)
            ):
                continue
            if run_id in seen:
                continue
            seen.add(run_id)
            entry = _get(repo, f"actions/runs/{run_id}", run)
            if not isinstance(entry, dict):
                store.warn(f"run {run_id}: could not read it")
                continue
            created = _created(entry.get("created_at"))
            if created is None or created < since:
                continue
            path, event = entry.get("path"), entry.get("event")
            title = PR_TITLE.fullmatch(str(entry.get("display_title", "")))
            if path == f".github/workflows/{COMMENT}" and event == "issue_comment" and title:
                out.append(Candidate(str(run_id), "pr", int(title.group(1)), created))
            elif path == f".github/workflows/{NIGHTLY}" and event == "workflow_dispatch":
                out.append(Candidate(str(run_id), "manual", None, created))
        if len(artifacts) < 100:
            break
    return sorted(out, key=lambda c: c.created, reverse=True)


def named_images(repo: str, run_id: str, run: store.Runner) -> bool:
    """Whether a dispatch named its own images: its perf-plan's dispatch.json says so, as true.
    A dispatch that named none is a night on the public dashboard; one with no record is neither."""
    with tempfile.TemporaryDirectory() as scratch:
        if store.download(repo, run_id, "perf-plan", Path(scratch) / "plan", run) != "ok":
            return False
        try:
            record = json.loads(
                (Path(scratch) / "plan" / "dispatch.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError):
            return False
    return isinstance(record, dict) and record.get("image_overrides") is True


def build_one(candidate: Candidate, repo: str, out: Path, run: store.Runner) -> Path | None:
    """Store one run (the page's files only) and write its page under <out>; None when it could
    not be read."""
    trigger, scope = (
        ("pr-comment", f"pr-{candidate.pr}")
        if candidate.kind == "pr"
        else ("manual", "named images")
    )
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        got = store.download(repo, candidate.run_id, "perf-nightly", scratch / "night", run)
        if got != "ok":
            store.warn(f"run {candidate.run_id}: no nightly.json ({got}): no page")
            return None
        try:
            folder = store.store(
                scratch / "night" / "nightly.json",
                candidate.run_id,
                scratch / "data",
                trigger,
                scope,
                lambda name, dest: store.fetch(repo, candidate.run_id, name, dest, run),
            )
        except store.StoreError as err:
            store.warn(f"run {candidate.run_id}: {err}: no page")
            return None
        where = out / page_path(candidate.kind, folder, candidate.pr)
        if where.exists():
            shutil.rmtree(where)
        (where / "data").mkdir(parents=True)
        shutil.move(scratch / "data" / folder, where / "data" / folder)
    root = "../" * page_path(candidate.kind, folder, candidate.pr).count("/")
    return dashboard.build_unlisted(where / "data", folder, where, root)


def build(
    out: Path, repo: str, days: int, now: datetime, run: store.Runner = subprocess.run
) -> list[Path]:
    """Every unlisted page of the last `days` days, under <out>/manual and <out>/pr (both
    replaced). A run that cannot be read is left out with a warning, never a failure."""
    for area in ("manual", "pr"):
        shutil.rmtree(out / area, ignore_errors=True)
    pages = []
    for candidate in candidates(repo, now, days, run):
        try:
            if candidate.kind == "manual" and not named_images(repo, candidate.run_id, run):
                continue
            made = build_one(candidate, repo, out, run)
        except Exception as err:  # one run's page must never fail every deploy for 30 days
            store.warn(f"run {candidate.run_id}: {type(err).__name__}: {err}: no page")
            continue
        if made is not None:
            pages.append(made)
            print(f"unlisted: {made.parent.relative_to(out).as_posix()}/")
    return pages


def main(
    argv: list[str] | None = None,
    environ: Mapping[str, str] | None = None,
    run: store.Runner = subprocess.run,
) -> int:
    environ = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(
        prog="bench nightly unlisted", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--repo", default=environ.get("GITHUB_REPOSITORY", "FalkorDB/FalkorDB"))
    sub = parser.add_subparsers(dest="command")
    url = sub.add_parser("url", help="a run's page URL")
    url.add_argument("--kind", required=True, choices=("manual", "pr"))
    url.add_argument(
        "--nightly", required=True, type=Path, help="the run's nightly.json (its date)"
    )
    url.add_argument("--run-id", required=True)
    url.add_argument("--pr", type=int)
    parser.add_argument("--out", type=Path, help="the site to add the pages to")
    parser.add_argument("--days", type=int, default=DAYS, help="keep the runs of this many days")
    args = parser.parse_args(argv)
    if args.command == "url":
        try:
            date = json.loads(args.nightly.read_text(encoding="utf-8")).get("date")
        except (OSError, UnicodeError, ValueError, AttributeError):
            date = None
        folder = f"{date}-{args.run_id}"
        if not dashboard.FOLDER_RE.fullmatch(folder) or (args.kind == "pr" and not args.pr):
            print(
                f"::error::no page for run {args.run_id} (date {date!r}, pr {args.pr})",
                file=sys.stderr,
            )
            return 1
        print(pages_url(args.repo, environ) + page_path(args.kind, folder, args.pr))
        return 0
    if args.out is None:
        parser.error("--out is required")
    build(args.out, args.repo, args.days, datetime.now(UTC), run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
