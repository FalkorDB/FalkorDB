"""A `/perf` comment on a pull request runs the perf checks on its image (#4037).

.github/workflows/perf-comment.yml calls two subcommands, `bench nightly pr-comment ...`:

  resolve  the comment (env COMMENT_BODY) -> go and the two images to measure, or a reply that
           says why not. Who may ask, which pull request, which commit, which baseline.
  answer   the run's nightly.json -> the one comment that answers the request.

The command, on the comment's first line (the rest of the comment is not read):

  /perf <checks> [baseline <main | X.Y.Z>]

<checks> is `all` or a comma list of micro-hw and the names in bench/checks.yaml. The baseline is
main's newest merge-queue image (`main`, the default), a Rust release X.Y.Z (X >= 5,
falkordb/falkordb-server:X.Y.Z) or a C engine release 4.Y.Z (falkordb/falkordb-server:v4.Y.Z).

Standard library only (with contract, nightly.merge and nightly.report, which are too). Every
GitHub call goes through `gh`, so the tests swap in a fake.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from falkorbench import contract
from falkorbench.nightly import merge
from falkorbench.nightly import report

# The first line of every reply and answer: a reader (or a later tool) can tell them apart.
MARKER = "<!-- perf-comment -->"
COMMAND = "/perf"
ROLES = ("admin", "maintain", "write")
# The collaborator API's `permission`, the base a custom role is built on (maintain shows as write).
BASE_PERMISSIONS = ("admin", "write")
CANDIDATE_REPO = "ghcr.io/falkordb/falkordb-server"
RELEASE_REPO = "falkordb/falkordb-server"
RUST_PR = ".github/workflows/rust-pr.yml"
# benchmark-cov.yml's rule for "the PR image was built from this commit": the Valkey leg's
# build-rc-image jobs, matched as path segments so a deeper or shallower nesting still matches,
# and neither build-rc-asan-image nor Redis's leg (whose images are not measured).
BUILD_JOB = re.compile(r"(^|/ )image-valkey / build-rc-image / ")
ACTIVE = ("queued", "in_progress", "waiting", "requested", "pending")
# perf-comment.yml's job that calls perf-nightly.yml: its jobs show as "measure / <job>".
MEASURE_JOB = re.compile(r"measure( / |$)")
MAIN = "main"
VERSION = re.compile(r"(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})")
# A name the reply may quote back: anything else is described, not echoed.
QUOTABLE = re.compile(r"[A-Za-z0-9._-]{1,40}")
RELEASES_SHOWN = 6
PAGES = 10
# Seconds before each retry of a GET that failed without an answer. Tests pass ().
GET_RETRY_DELAYS = (2.0, 5.0)


class Refusal(Exception):
    """The request does not run; the message is the reply, in Markdown."""


class GitHubError(Exception):
    pass


@dataclass(frozen=True)
class Request:
    checks: tuple[str, ...]  # empty: all
    baseline: str  # MAIN or X.Y.Z

    @property
    def text(self) -> str:
        """The request as the answer quotes it, normalised."""
        return f"{COMMAND} {','.join(self.checks) or 'all'} baseline {self.baseline}"

    @property
    def engine(self) -> str:
        """`main`, `rust` (a v5+ release) or `c` (a 4.x release)."""
        if self.baseline == MAIN:
            return MAIN
        return "c" if self.baseline.split(".")[0] == "4" else "rust"

    @property
    def baseline_ref(self) -> str:
        """The image tag to pin for a release baseline (MAIN is resolved by the nightly's walk)."""
        if self.baseline == MAIN:
            return MAIN
        tag = f"v{self.baseline}" if self.engine == "c" else self.baseline
        return f"docker.io/{RELEASE_REPO}:{tag}"

    @property
    def baseline_label(self) -> str:
        return (
            "" if self.baseline == MAIN else ("C v" if self.engine == "c" else "") + self.baseline
        )


# --- the command ----------------------------------------------------------------------


def command_text(body: str | None) -> str | None:
    """What follows `/perf` on the comment's first line, or None when the comment is not the
    command. A first line starting `/perf` and then anything but a blank (`/performance`) is not."""
    lines = (body or "").splitlines()
    first = lines[0].rstrip() if lines else ""
    if first != COMMAND and not re.match(rf"{re.escape(COMMAND)}\s", first):
        return None
    return first[len(COMMAND) :].strip()


def quoted(name: str) -> str:
    return f"`{name}`" if QUOTABLE.fullmatch(name) else "a name"


def usage(valid: list[str]) -> str:
    checks = ", ".join(f"`{name}`" for name in valid)
    return (
        f"Usage: `{COMMAND} <checks> [baseline <main | X.Y.Z>]`, on the comment's first line. "
        f"Checks: `all`, or a comma list of {checks}. Baseline: `main` (main's newest merge-queue "
        "image, the default), a release `X.Y.Z` (5.0.0 and later), or a C engine release `4.Y.Z`. "
        f"Example: `{COMMAND} micro-hw,pokec baseline 6.2.1`."
    )


def parse(text: str, valid: list[str]) -> Request:
    """The request in `text` (what follows /perf), or Refusal with a reply that lists what is valid."""
    words = text.split()
    if not words:
        raise Refusal(f"Say which checks to run. {usage(valid)}")
    if len(words) not in (1, 3) or (len(words) == 3 and words[1] != "baseline"):
        raise Refusal(f"Could not read the request. {usage(valid)}")
    names = words[0].split(",")
    if any(not name for name in names):
        raise Refusal(f"The check list has an empty name. {usage(valid)}")
    if "all" in names:
        if len(names) > 1:
            raise Refusal(f"`all` stands alone. {usage(valid)}")
        checks: tuple[str, ...] = ()
    else:
        unknown = [name for name in dict.fromkeys(names) if name not in valid]
        if unknown:
            listed = ", ".join(quoted(name) for name in unknown)
            raise Refusal(f"Unknown check {listed}. {usage(valid)}")
        checks = tuple(dict.fromkeys(names))
    baseline = words[2] if len(words) == 3 else MAIN
    if baseline != MAIN:
        version = VERSION.fullmatch(baseline)
        if not version or int(version.group(1)) < 4:
            raise Refusal(f"Unknown baseline {quoted(baseline)}. {usage(valid)}")
    return Request(checks, baseline)


def valid_checks(registry_json: str | None) -> list[str]:
    """micro-hw and every usable bench/checks.yaml entry, in the nightly's plan order."""
    registry = json.loads(registry_json) if registry_json and registry_json.strip() else None
    plan = merge.build_plan(registry, "")
    return [row["check"] for row in plan if row.get("runner") and not row.get("error")]


# --- GitHub and the registry ----------------------------------------------------------


Runner = Callable[..., subprocess.CompletedProcess]


class GitHub:
    """`gh api` for one repository. `run` is subprocess.run; the tests pass a fake."""

    def __init__(self, repo: str, run: Runner = subprocess.run, delays=GET_RETRY_DELAYS):
        self.repo = repo
        self.run = run
        self.delays = delays

    def get(self, path: str) -> object:
        """One GET. A failure that is not the API's own 4xx answer (a dropped connection, a
        5xx) is tried again: a GET changes nothing."""
        for delay in (*self.delays, None):
            done = self.run(["gh", "api", path], capture_output=True, text=True, check=False)
            if done.returncode == 0:
                return json.loads(done.stdout) if done.stdout.strip() else None
            said = (done.stderr or "").strip()
            if delay is None or re.search(r"HTTP 4\d\d", said):
                raise GitHubError(f"gh api {path}: {said[:300]}")
            time.sleep(delay)
        raise AssertionError("unreachable")

    def get_all(self, path: str, key: str | None = None) -> list:
        """Every item of a listing, 100 a page (at most PAGES pages)."""
        items: list = []
        join = "&" if "?" in path else "?"
        for page in range(1, PAGES + 1):
            body = self.get(f"{path}{join}per_page=100&page={page}")
            batch = body.get(key, []) if key and isinstance(body, dict) else body
            batch = batch if isinstance(batch, list) else []
            items += batch
            if len(batch) < 100:
                break
        return items

    def post(self, path: str, payload: dict) -> bool:
        done = self.run(
            ["gh", "api", "-X", "POST", path, "--input", "-"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            print(f"::warning::gh api POST {path}: {(done.stderr or '').strip()[:300]}")
        return done.returncode == 0


def image_digest(ref: str, run: Runner = subprocess.run) -> str | None:
    """REF's manifest digest; None when the registry says REF does not exist. Any other failure
    raises, so a network error never reads as "no image"."""
    done = run(
        ["docker", "buildx", "imagetools", "inspect", ref, "--format", "{{json .Manifest}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode == 0:
        try:
            digest = json.loads(done.stdout).get("digest")
        except (ValueError, AttributeError):
            digest = None
        if isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            return digest
        raise GitHubError(f"{ref}: the registry's answer has no digest")
    if re.search(r"not found|manifest unknown", done.stderr or "", re.IGNORECASE):
        return None
    raise GitHubError(f"{ref}: {(done.stderr or '').strip()[:300]}")


# --- resolve --------------------------------------------------------------------------


@dataclass
class Context:
    repo: str
    pr: int
    actor: str
    actor_type: str
    body: str
    run_id: int
    server: str = report.DEFAULT_SERVER_URL
    valid: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    go: bool
    reply: str = ""  # Markdown; empty: say nothing
    outputs: dict[str, str] = field(default_factory=dict)
    why: str = ""  # one line for the log


def run_url(ctx: Context, run_id: int) -> str:
    return f"{ctx.server}/{ctx.repo}/actions/runs/{run_id}"


def role_of(gh: GitHub, actor: str) -> tuple[str, bool]:
    """(the commenter's role, whether it may ask). A custom role counts by the base
    permission it is built on (`permission`: admin, write, read or none). Only GitHub's
    404 (no such user) reads as no access; any other failure raises, so a rate limit is
    never reported as missing access."""
    try:
        perm = gh.get(f"repos/{gh.repo}/collaborators/{actor}/permission")
    except GitHubError as err:
        if "HTTP 404" not in str(err):
            raise
        print(err, file=sys.stderr)
        perm = None
    perm = perm if isinstance(perm, dict) else {}
    role = str(perm.get("role_name") or perm.get("permission") or "none")
    return role, role in ROLES or perm.get("permission") in BASE_PERMISSIONS


def busy_with(gh: GitHub, run_id: int) -> list[int]:
    """Earlier runs of this workflow for the same pull request that are measuring.

    The workflow's run-name is "/perf on #<N>" for anything that starts like the command, so
    the same title is the same pull request; an earlier run id is the one that came first.
    Only a run that reached its measure job counts: a comment that is refused, or is not the
    command at all, never blocks a real request. Two requests that both pass resolve at once
    queue on the measure job's concurrency group instead.
    """
    me = gh.get(f"repos/{gh.repo}/actions/runs/{run_id}")
    me = me if isinstance(me, dict) else {}
    workflow, title = me.get("workflow_id"), me.get("display_title")
    if not workflow or not title:
        raise GitHubError(f"run {run_id} has no workflow id or title")
    earlier = set()
    for status in ACTIVE:
        runs = gh.get_all(
            f"repos/{gh.repo}/actions/workflows/{workflow}/runs?status={status}", "workflow_runs"
        )
        for run in runs:
            if (
                isinstance(run, dict)
                and run.get("display_title") == title
                and isinstance(run.get("id"), int)
                and run["id"] < run_id
            ):
                earlier.add(run["id"])
    measuring = []
    for other in sorted(earlier):
        jobs = gh.get_all(f"repos/{gh.repo}/actions/runs/{other}/jobs", "jobs")
        if any(
            isinstance(job, dict)
            and MEASURE_JOB.match(str(job.get("name", "")))
            and job.get("conclusion") != "skipped"
            for job in jobs
        ):
            measuring.append(other)
    return measuring


@dataclass(frozen=True)
class Build:
    state: str  # built | building | missing | failed
    detail: str
    run_id: int | None = None
    finished: str = ""  # when its last build-rc-image job finished (ISO 8601, UTC)


def _build_jobs(gh: GitHub, run_id: object) -> list[dict]:
    jobs = gh.get_all(f"repos/{gh.repo}/actions/runs/{run_id}/jobs", "jobs")
    return [j for j in jobs if isinstance(j, dict) and BUILD_JOB.search(str(j.get("name", "")))]


def build_state(gh: GitHub, pr: int, sha: str) -> Build:
    """The state of rust-pr.yml's Valkey build-rc-image jobs on `sha`, in the newest rust-pr.yml
    run of this pull request there."""
    runs = gh.get_all(
        f"repos/{gh.repo}/actions/runs?head_sha={sha}&event=pull_request", "workflow_runs"
    )
    ours = [
        run
        for run in runs
        if isinstance(run, dict)
        and run.get("path") == RUST_PR
        and run.get("head_sha") == sha
        and any(
            p.get("number") == pr for p in run.get("pull_requests") or [] if isinstance(p, dict)
        )
    ]
    if not ours:
        return Build("missing", "no rust-pr.yml run for this commit yet")
    newest = max(ours, key=lambda run: (run.get("run_number") or 0, run.get("id") or 0))
    jobs = _build_jobs(gh, newest["id"])
    states = [job.get("conclusion") or "pending" for job in jobs]
    detail = (
        f"rust-pr.yml run {newest['id']}: build-rc-image {', '.join(states) or 'not scheduled'}"
    )
    if not states:
        return Build("missing", detail)
    if "pending" in states:
        return Build("building", detail)
    if all(state == "success" for state in states):
        finished = max(str(job.get("completed_at") or "") for job in jobs)
        return Build("built", detail, newest["id"], finished)
    return Build("failed", detail)


def retagged_after(gh: GitHub, pr: int, sha: str, branch: str, build: Build) -> str:
    """Another commit's rust-pr.yml run of this pull request whose build-rc-image finished after
    `build`'s, and so pushed rc-pr-<N>-valkey last (a re-run of an older commit, say), as
    "<run id> on <sha>"; "" when `build` was the last to push it. The tag carries no commit, so
    this is what binds the digest about to be pinned to the build that passed."""
    runs = gh.get_all(
        f"repos/{gh.repo}/actions/runs?branch={urllib.parse.quote(branch, safe='')}&event=pull_request",
        "workflow_runs",
    )
    for run in runs:
        if not (
            isinstance(run, dict)
            and run.get("path") == RUST_PR
            and run.get("id") != build.run_id
            and run.get("head_sha") != sha
            # A run that finished a job later was updated later too.
            and str(run.get("updated_at") or "") > build.finished
            and any(
                p.get("number") == pr for p in run.get("pull_requests") or [] if isinstance(p, dict)
            )
        ):
            continue
        for job in _build_jobs(gh, run["id"]):
            if (
                job.get("conclusion") == "success"
                and str(job.get("completed_at") or "") > build.finished
            ):
                return f"{run['id']} on {str(run.get('head_sha'))[:9]}"
    return ""


def release_list(gh: GitHub) -> str:
    """The newest stable releases of each engine, as a reply lists them."""
    try:
        tags = [
            r.get("tag_name")
            for r in gh.get_all(f"repos/{gh.repo}/releases")
            if isinstance(r, dict) and not r.get("draft") and not r.get("prerelease")
        ]
    except GitHubError:
        return ""
    versions = []
    for tag in tags:
        match = VERSION.fullmatch(tag[1:]) if isinstance(tag, str) and tag.startswith("v") else None
        if match:
            versions.append(tuple(int(part) for part in match.groups()))
    versions.sort(reverse=True)
    rust = [v for v in versions if v[0] >= 5][:RELEASES_SHOWN]
    c = [v for v in versions if v[0] == 4][:RELEASES_SHOWN]
    shown = ", ".join(f"`{'.'.join(map(str, v))}`" for v in rust + c)
    return f" Newest releases: {shown}." if shown else ""


def resolve(ctx: Context, gh: GitHub, digest_of: Callable[[str], str | None]) -> Outcome:
    """Who asked, for what, on which commit; the images to measure, or why not."""
    text = command_text(ctx.body)
    if text is None:
        return Outcome(False, why="not a /perf command")
    if ctx.actor_type == "Bot":
        return Outcome(False, why="a bot's comment")
    role, allowed = role_of(gh, ctx.actor)
    if not allowed:
        return Outcome(
            False,
            f"`{COMMAND}` runs only for people with write, maintain or admin access to the "
            f"repository ({quoted(ctx.actor)} has `{role if QUOTABLE.fullmatch(role) else '?'}`).",
            why=f"{ctx.actor} has {role}",
        )
    try:
        request = parse(text, ctx.valid)
    except Refusal as refusal:
        return Outcome(False, str(refusal), why="the request does not parse")

    pull = gh.get(f"repos/{gh.repo}/pulls/{ctx.pr}")
    pull = pull if isinstance(pull, dict) else {}
    head = pull.get("head") if isinstance(pull.get("head"), dict) else {}
    head_repo = (
        (head.get("repo") or {}).get("full_name") if isinstance(head.get("repo"), dict) else None
    )
    if head_repo != gh.repo:
        return Outcome(
            False,
            f"`{COMMAND}` does not run on pull requests from forks: they have no image of their own.",
            why=f"head repository {head_repo}",
        )
    if pull.get("state") != "open":
        return Outcome(False, "This pull request is not open.", why="not open")
    sha = head.get("sha")
    if not (isinstance(sha, str) and contract.COMMIT_RE.fullmatch(sha)):
        raise GitHubError(f"pull request {ctx.pr} has no head commit")

    earlier = busy_with(gh, ctx.run_id)
    if earlier:
        return Outcome(
            False,
            f"Not started: a `{COMMAND}` run on this pull request is still going "
            f"([run]({run_url(ctx, earlier[0])})). One runs at a time; ask again when it answers.",
            why=f"busy with run {earlier[0]}",
        )

    build = build_state(gh, ctx.pr, sha)
    if build.state != "built":
        why = {
            "missing": "is not built yet",
            "building": "is still being built",
            "failed": "did not build",
        }[build.state]
        return Outcome(
            False,
            f"Not started: the image for `{sha[:9]}` (the head of this pull request) {why} "
            f"({build.detail}). Ask again once rust-pr.yml's build-rc-image succeeded for it.",
            why=f"{sha}: {build.detail}",
        )
    later = retagged_after(gh, ctx.pr, sha, str(head.get("ref") or ""), build)
    if later:
        return Outcome(
            False,
            f"Not started: the image tag of this pull request was pushed again after `{sha[:9]}` "
            f"was built, by rust-pr.yml run {later}, so it may not be this commit's image. Re-run "
            f"rust-pr.yml on the head (or push), then ask again once it is built.",
            why=f"{sha}: retagged by {later}",
        )
    tag = f"{CANDIDATE_REPO}:rc-pr-{ctx.pr}-valkey"
    digest = digest_of(tag)
    if digest is None:
        return Outcome(
            False,
            f"Not started: `{tag}` does not exist, though build-rc-image succeeded for `{sha[:9]}` "
            "(cleanup-rc-images.yml may have removed it). Push a commit to rebuild it.",
            why=f"{tag} not found",
        )
    # The tag carries no commit: it is this commit's only if the head did not move while it was
    # read (a push starts a new build, which retags it).
    again = gh.get(f"repos/{gh.repo}/pulls/{ctx.pr}")
    again_sha = (again.get("head") or {}).get("sha") if isinstance(again, dict) else None
    if again_sha != sha:
        return Outcome(
            False,
            f"Not started: the head of this pull request moved from `{sha[:9]}` while its image "
            "was being pinned. Ask again once the new head's image is built.",
            why=f"head moved {sha} -> {again_sha}",
        )

    baseline_image, baseline_commit = MAIN, ""
    if request.baseline != MAIN:
        found = digest_of(request.baseline_ref)
        if found is None:
            return Outcome(
                False,
                f"Unknown baseline {quoted(request.baseline)}: `{request.baseline_ref}` does not "
                f"exist.{release_list(gh)} {usage(ctx.valid)}",
                why=f"{request.baseline_ref} not found",
            )
        baseline_image = f"{RELEASE_REPO}@{found}"
        try:
            tagged = gh.get(f"repos/{gh.repo}/commits/v{request.baseline}")
            baseline_commit = tagged.get("sha", "") if isinstance(tagged, dict) else ""
        except GitHubError:
            baseline_commit = ""  # an image with no tag here: the answer says commit unknown
        if not contract.COMMIT_RE.fullmatch(baseline_commit or ""):
            baseline_commit = ""

    outputs = {
        "pr": str(ctx.pr),
        "request": request.text,
        "checks": ",".join(request.checks),
        "head_sha": sha,
        "candidate_image": f"{CANDIDATE_REPO}@{digest}",
        "candidate_label": f"pr-{ctx.pr}@{sha[:7]}",
        "candidate_commit": sha,
        "baseline_image": baseline_image,
        "baseline_label": request.baseline_label,
        "baseline_commit": baseline_commit,
    }
    return Outcome(True, outputs=outputs, why=f"{request.text} on {sha}")


# --- answer ---------------------------------------------------------------------------


def commit_link(server: str, repo: str, commit: object) -> str:
    if isinstance(commit, str) and contract.COMMIT_RE.fullmatch(commit):
        return f"[`{commit[:12]}`]({server}/{repo}/commit/{commit})"
    return "unknown"


def side_row(night: dict, which: str, server: str, repo: str, note: str = "") -> str:
    side = night.get(which) if isinstance(night.get(which), dict) else {}
    image = side.get("image")
    shown = (
        f"`{image}`" if isinstance(image, str) and contract.IMAGE_RE.fullmatch(image) else "unknown"
    )
    label = report.markdown_plain(report.label(side))
    commit = commit_link(server, repo, side.get("commit"))
    return f"| {which} | {label}{note} | {shown} | {commit} |"


def answer(
    night_path: Path | None,
    request: str,
    *,
    run: str,
    dashboard: str = "",
    page: str = "",
    measured: str = "",
    server: str = report.DEFAULT_SERVER_URL,
    repo: str = report.DEFAULT_REPOSITORY,
) -> str:
    """The answer to one /perf request, built from the run's nightly.json alone."""
    asked = (
        f"`{request}`"
        if re.fullmatch(r"/perf [a-z0-9,.-]+ baseline [a-z0-9.]+", request)
        else "`/perf`"
    )
    links = [report.markdown_link(run, "run")]
    if page:
        links.append(
            report.markdown_link(page, "page")
            + " (appears within minutes of the run's end; kept 30 days)"
        )
    if dashboard:
        links.append(
            report.markdown_link(dashboard, "the page's files")
            + " (download, open `index.html`, if the page is not up)"
        )
    if night_path is None or not night_path.is_file():
        return "\n".join(
            [
                MARKER,
                "### Perf: ERROR",
                "",
                f"{asked}: the run left no nightly.json (measure job: {measured or 'unknown'}), "
                "so nothing was judged. Read the run.",
                "",
                " · ".join(links),
                "",
            ]
        )
    night, status, reason = report.read_night(night_path)
    lines = [MARKER, f"### Perf: {status.upper()}", "", asked, ""]
    if reason:
        lines += [f"**Not green:** {report.markdown_plain(reason)}", ""]
    note = (
        " (main's newest merge-queue image, resolved by the run)"
        if request.endswith(f"baseline {MAIN}")
        else ""
    )
    lines += [
        "| side | label | image | commit |",
        "| --- | --- | --- | --- |",
        side_row(night, "candidate", server, repo),
        side_row(night, "baseline", server, repo, note),
        "",
    ]
    checks = report.checks_of(night)
    if checks:
        lines += ["| check | verdict | summary |", "| --- | --- | --- |"]
        for check in checks:
            verdict = report.one_line(check.get("verdict"), "?", 20)
            if verdict == "regression" and check.get("confirmed"):
                verdict += " (confirmed)"
            said = check.get("error") if check.get("verdict") == "error" else check.get("summary")
            lines.append(
                f"| {report.markdown_plain(report.name_of(check))} | {report.markdown_plain(verdict)} "
                f"| {report.markdown_plain(report.one_line(said, ''))} |"
            )
        lines.append("")
    for check in report.regressed_checks(night):
        lines += report.regression_lines(check)
    if status == "red" and measured == "success":
        lines += [
            "The run passed: bench/waivers.yaml waives every regression above (the run page names the waivers).",
            "",
        ]
    if status == "green" and measured not in ("", "success"):
        lines += [
            f"The checks passed, but the measure job ended `{report.markdown_plain(measured)}`: read the run.",
            "",
        ]
    lines += [" · ".join(links), ""]
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------------------


def write_outputs(path: str, outputs: Mapping[str, str]) -> None:
    for key, value in outputs.items():
        if "\n" in value or "\r" in value:
            raise GitHubError(f"output {key} spans lines")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.writelines(f"{key}={value}\n" for key, value in outputs.items())
    for key, value in outputs.items():
        print(f"{key}={value}")


def summarise(text: str, environ: Mapping[str, str]) -> None:
    summary = environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def say(gh: GitHub, ctx: Context, text: str, dry_run: bool, environ: Mapping[str, str]) -> bool:
    """One comment on the pull request; a dry run writes it to the job summary instead.
    False when the comment could not be posted."""
    if dry_run:
        summarise(f"**Dry run: this comment would be posted on #{ctx.pr}:**\n\n{text}", environ)
        print(text)
        return True
    return gh.post(f"repos/{gh.repo}/issues/{ctx.pr}/comments", {"body": text})


def cmd_resolve(
    args: argparse.Namespace, environ: Mapping[str, str], run: Runner = subprocess.run
) -> int:
    pr = environ.get("PR", "")
    run_id = environ.get("GITHUB_RUN_ID", "")
    if not (pr.isdigit() and run_id.isdigit()):
        print(f"::error::PR {pr!r} or GITHUB_RUN_ID {run_id!r} is not a number", file=sys.stderr)
        return 1
    registry = Path(args.registry).read_text(encoding="utf-8") if args.registry else ""
    ctx = Context(
        repo=environ.get("GITHUB_REPOSITORY") or report.DEFAULT_REPOSITORY,
        pr=int(pr),
        actor=environ.get("ACTOR", ""),
        actor_type=environ.get("ACTOR_TYPE", ""),
        body=environ.get("COMMENT_BODY", ""),
        run_id=int(run_id),
        server=report.server_url(environ),
        valid=valid_checks(registry),
    )
    gh = GitHub(ctx.repo, run)
    write_outputs(args.github_output, {"go": "false"})
    try:
        outcome = resolve(ctx, gh, lambda ref: image_digest(ref, run))
    except (GitHubError, ValueError) as err:
        print(f"::error::{err}", file=sys.stderr)
        say(
            gh,
            ctx,
            f"{MARKER}\nNot started: the request could not be checked ({report.markdown_plain(report.one_line(str(err), 'error'))}). "
            f"[Run]({run_url(ctx, ctx.run_id)}).",
            args.dry_run,
            environ,
        )
        return 1
    print(f"{'go' if outcome.go else 'no'}: {outcome.why}")
    if outcome.reply and not say(gh, ctx, f"{MARKER}\n{outcome.reply}", args.dry_run, environ):
        print("::error::the reply could not be posted", file=sys.stderr)
        return 1
    if not outcome.go:
        return 0
    write_outputs(args.github_output, {**outcome.outputs, "go": "true"})
    comment = environ.get("COMMENT_ID", "")
    if args.dry_run:
        summarise(
            f"Dry run: no reaction. Measuring `{outcome.outputs['request']}` on `{outcome.outputs['head_sha']}`.",
            environ,
        )
    elif comment.isdigit():
        gh.post(f"repos/{ctx.repo}/issues/comments/{comment}/reactions", {"content": "rocket"})
    return 0


def cmd_answer(
    args: argparse.Namespace, environ: Mapping[str, str], run: Runner = subprocess.run
) -> int:
    repo = environ.get("GITHUB_REPOSITORY") or report.DEFAULT_REPOSITORY
    run_link = report.current_run_url(environ) or f"{report.server_url(environ)}/{repo}/actions"
    text = answer(
        Path(args.nightly) if args.nightly else None,
        args.request,
        run=run_link,
        dashboard=args.dashboard_url,
        page=args.page_url,
        measured=args.measured,
        server=report.server_url(environ),
        repo=repo,
    )
    ctx = Context(repo, int(args.pr), "", "", "", 0, report.server_url(environ))
    if not say(GitHub(repo, run), ctx, text, args.dry_run, environ):
        print("::error::the answer could not be posted", file=sys.stderr)
        return 1
    return 0


def main(
    argv: list[str] | None = None,
    environ: Mapping[str, str] | None = None,
    run: Runner = subprocess.run,
) -> int:
    environ = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(
        prog="bench nightly pr-comment", description=__doc__.split("\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("resolve", help="the comment (env) -> go and the images, or a reply")
    p.add_argument("--registry", default="", help="bench/checks.yaml converted to JSON")
    p.add_argument("--github-output", default="")
    p.add_argument("--dry-run", action="store_true", help="reply to the job summary, no reaction")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("answer", help="the run's nightly.json -> the answer on the pull request")
    p.add_argument("--nightly", default="", help="nightly.json; absent when the run left none")
    p.add_argument("--request", required=True, help="the request, as resolve normalised it")
    p.add_argument("--pr", required=True, type=int)
    p.add_argument("--dashboard-url", default="")
    p.add_argument("--page-url", default="", help="the run's unlisted page")
    p.add_argument("--measured", default="", help="the measure job's result")
    p.add_argument("--dry-run", action="store_true", help="write the answer to the job summary")
    p.set_defaults(func=cmd_answer)

    args = parser.parse_args(argv)
    return args.func(args, environ, run)


if __name__ == "__main__":
    sys.exit(main())
