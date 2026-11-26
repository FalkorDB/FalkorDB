"""`bench nightly pr-comment`: a `/perf` comment on a pull request (falkorbench.nightly.pr_comment).

GitHub and the registry are faked: `Fake` answers `gh api` and `docker buildx imagetools inspect`
from tables, and records every POST. One test per rule, each failing when its rule breaks.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from falkorbench.nightly import pr_comment as pc
from fixtures import examples

BENCH = Path(__file__).resolve().parents[1]
REPO = "FalkorDB/FalkorDB"
PR = 3967
RUN_ID = 5000
SHA = "727e385d70cdb4bd7a326053314cceeec7b12cd4"
NEW_SHA = "1" * 40
TAG_SHA = "a" * 40
PR_DIGEST = "sha256:" + "f" * 64
C_DIGEST = "sha256:" + "c" * 64
RUST_DIGEST = "sha256:" + "6" * 64
VALID = ["micro-hw", "pokec", "ldbc", "soak-a", "soak-b", "fuzz"]
TITLE = f"/perf on #{PR}"
BUILD_JOBS = [
    f"ci-pipeline / image-valkey / build-rc-image / build ({arch})" for arch in ("x64", "arm64v8")
]


def registry_json() -> str:
    return json.dumps(yaml.safe_load((BENCH / "checks.yaml").read_text(encoding="utf-8")))


class Fake:
    """`gh api` and `docker buildx imagetools inspect`, answered from tables.

    `api` maps a path (query string included) to its JSON answer, or to an int exit status with
    `stderr`. Listings are found by the path before `per_page=`. `images` maps a ref to its digest,
    None (the registry says not found) or an Exception (any other failure).
    """

    def __init__(self):
        self.api: dict[str, object] = {}
        self.images: dict[str, object] = {}
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[str] = []
        self.pull_answers: list[dict] = []
        self.failures: dict[str, list[str]] = {}
        self.post_rc = 0

    def __call__(self, argv, **kwargs):
        if argv[:4] == ["docker", "buildx", "imagetools", "inspect"]:
            found = self.images.get(argv[4])
            if isinstance(found, Exception):
                return subprocess.CompletedProcess(argv, 1, "", f"ERROR: {found}")
            if found is None:
                return subprocess.CompletedProcess(argv, 1, "", f"ERROR: {argv[4]}: not found")
            return subprocess.CompletedProcess(argv, 0, json.dumps({"digest": found}), "")
        assert argv[:2] == ["gh", "api"], argv
        if "-X" in argv:
            path = argv[4]
            self.posts.append((path, json.loads(kwargs["input"])))
            return subprocess.CompletedProcess(argv, self.post_rc, "{}", "HTTP 403")
        path = argv[2]
        self.gets.append(path)
        if self.failures.get(path):
            return subprocess.CompletedProcess(argv, 1, "", self.failures[path].pop(0))
        if path == f"repos/{REPO}/pulls/{PR}" and self.pull_answers:
            answer = self.pull_answers.pop(0)
        else:
            key = path.split("per_page=")[0].rstrip("?&")
            answer = self.api.get(path, self.api.get(key))
        if answer is None:
            return subprocess.CompletedProcess(argv, 1, "", f"gh: Not Found (HTTP 404) {path}")
        return subprocess.CompletedProcess(argv, 0, json.dumps(answer), "")


def pull(sha=SHA, repo=REPO, state="open") -> dict:
    return {"state": state, "head": {"sha": sha, "ref": "perf/x", "repo": {"full_name": repo}}}


def rust_pr_run(run_id=900, sha=SHA, pr=PR, number=10) -> dict:
    return {
        "id": run_id,
        "run_number": number,
        "path": pc.RUST_PR,
        "head_sha": sha,
        "pull_requests": [{"number": pr}],
    }


@pytest.fixture
def fake() -> Fake:
    """A request that runs: write access, an open same-repo PR whose head's image was built."""
    f = Fake()
    f.api[f"repos/{REPO}/collaborators/dvir/permission"] = {"role_name": "write"}
    f.api[f"repos/{REPO}/pulls/{PR}"] = pull()
    f.api[f"repos/{REPO}/actions/runs/{RUN_ID}"] = {"workflow_id": 77, "display_title": TITLE}
    for status in pc.ACTIVE:
        f.api[f"repos/{REPO}/actions/workflows/77/runs?status={status}"] = {"workflow_runs": []}
    f.api[f"repos/{REPO}/actions/runs?head_sha={SHA}&event=pull_request"] = {
        "workflow_runs": [rust_pr_run()]
    }
    f.api[f"repos/{REPO}/actions/runs/900/jobs"] = {
        "jobs": [{"name": n, "conclusion": "success"} for n in BUILD_JOBS]
        + [{"name": "ci-pipeline / image-redis / build-rc-image / build (x64)", "conclusion": None}]
    }
    f.api[f"repos/{REPO}/commits/v4.22.4"] = {"sha": TAG_SHA}
    f.api[f"repos/{REPO}/releases"] = [
        {"tag_name": tag, "draft": False, "prerelease": False}
        for tag in ("v6.2.1", "v6.2.0", "v4.22.4", "v4.22.3", "v6.0.4")
    ] + [{"tag_name": "v6.3.0-rc1", "draft": False, "prerelease": True}]
    f.images[f"{pc.CANDIDATE_REPO}:rc-pr-{PR}-valkey"] = PR_DIGEST
    f.images[f"docker.io/{pc.RELEASE_REPO}:v4.22.4"] = C_DIGEST
    f.images[f"docker.io/{pc.RELEASE_REPO}:6.2.1"] = RUST_DIGEST
    return f


def context(body: str, actor="dvir", actor_type="User") -> pc.Context:
    return pc.Context(REPO, PR, actor, actor_type, body, RUN_ID, valid=VALID)


def resolve(fake: Fake, body: str, **kw) -> pc.Outcome:
    gh = pc.GitHub(REPO, fake, delays=(0.0, 0.0))
    return pc.resolve(context(body, **kw), gh, lambda ref: pc.image_digest(ref, fake))


# --- the command ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "text"),
    [
        ("/perf pokec", "pokec"),
        ("/perf", ""),
        ("/perf   micro-hw,pokec  baseline 6.2.1  \r\nthanks", "micro-hw,pokec  baseline 6.2.1"),
        ("/perf\tpokec", "pokec"),
        ("/performance is worse", None),
        (" /perf pokec", None),  # the workflow's startsWith filter skips it too
        ("please /perf pokec", None),
        ("thanks\n/perf pokec", None),  # the first line only
        ("", None),
    ],
)
def test_only_a_first_line_starting_with_the_command_is_one(body, text):
    assert pc.command_text(body) == text


def test_a_request_names_checks_and_a_baseline():
    request = pc.parse("micro-hw,pokec,pokec baseline 4.22.4", VALID)
    assert request.checks == ("micro-hw", "pokec")
    assert request.text == "/perf micro-hw,pokec baseline 4.22.4"
    # A C engine release keeps its `v`; a Rust release has none.
    assert request.baseline_ref == "docker.io/falkordb/falkordb-server:v4.22.4"
    assert request.baseline_label == "C v4.22.4"
    rust = pc.parse("pokec baseline 6.2.1", VALID)
    assert (rust.baseline_ref, rust.baseline_label) == (
        "docker.io/falkordb/falkordb-server:6.2.1",
        "6.2.1",
    )
    assert pc.parse("pokec baseline 5.0.0", VALID).engine == "rust"


def test_all_and_a_left_out_baseline():
    request = pc.parse("all", VALID)
    # Empty checks: the nightly's plan runs every registered check.
    assert (request.checks, request.baseline, request.text) == (
        (),
        "main",
        "/perf all baseline main",
    )
    assert request.baseline_ref == "main" and request.baseline_label == ""


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("", "Say which checks"),
        ("pokec baseline", "Could not read"),
        ("pokec base 6.2.1", "Could not read"),
        ("pokec baseline 6.2.1 now", "Could not read"),
        ("pokec,,ldbc", "empty name"),
        ("all,pokec", "`all` stands alone"),
        ("pokecc", "Unknown check `pokecc`"),
        ("micro", "Unknown check `micro`"),  # removed in v1.11
        ("checks-yaml", "Unknown check `checks-yaml`"),
        ("pokec baseline 3.9.9", "Unknown baseline `3.9.9`"),
        ("pokec baseline 6.2", "Unknown baseline `6.2`"),
        ("pokec baseline v6.2.1", "Unknown baseline `v6.2.1`"),
        ("pokec baseline edge", "Unknown baseline `edge`"),
        ("`x`<b>", "Unknown check a name"),
    ],
)
def test_a_request_that_cannot_run_is_answered_with_what_is_valid(text, said):
    with pytest.raises(pc.Refusal) as refused:
        pc.parse(text, VALID)
    reply = str(refused.value)
    assert said in reply
    assert "<b>" not in reply
    for name in VALID:
        assert f"`{name}`" in reply
    assert "`main`" in reply and "`4.Y.Z`" in reply


def test_the_valid_checks_are_micro_hw_and_the_registry():
    assert pc.valid_checks(registry_json()) == VALID
    # A registry entry the nightly would stub (runner unknown) is not offered.
    broken = json.dumps([{"name": "x", "runner": "moon", "command": "c", "timeout_minutes": 1}])
    assert pc.valid_checks(broken) == ["micro-hw"]
    assert pc.valid_checks("") == ["micro-hw"]


# --- resolve --------------------------------------------------------------------------


def test_a_request_resolves_to_the_pinned_head_image_and_the_release(fake):
    outcome = resolve(fake, "/perf micro-hw,pokec baseline 4.22.4")
    assert outcome.go and not outcome.reply, outcome
    assert outcome.outputs == {
        "pr": str(PR),
        "request": "/perf micro-hw,pokec baseline 4.22.4",
        "checks": "micro-hw,pokec",
        "head_sha": SHA,
        "candidate_image": f"ghcr.io/falkordb/falkordb-server@{PR_DIGEST}",
        "candidate_label": f"pr-{PR}@{SHA[:7]}",
        "candidate_commit": SHA,
        "baseline_image": f"falkordb/falkordb-server@{C_DIGEST}",
        "baseline_label": "C v4.22.4",
        "baseline_commit": TAG_SHA,
    }
    assert fake.posts == []


def test_main_is_left_to_the_nightly_walk(fake):
    outcome = resolve(fake, "/perf all")
    assert outcome.go
    assert (outcome.outputs["checks"], outcome.outputs["baseline_image"]) == ("", "main")
    assert outcome.outputs["baseline_label"] == outcome.outputs["baseline_commit"] == ""


def test_a_release_with_no_tag_here_has_no_commit(fake):
    outcome = resolve(fake, "/perf pokec baseline 6.2.1")
    assert outcome.go
    assert outcome.outputs["baseline_image"] == f"falkordb/falkordb-server@{RUST_DIGEST}"
    assert outcome.outputs["baseline_commit"] == ""


@pytest.mark.parametrize(("body", "kw"), [("thanks", {}), ("/perf pokec", {"actor_type": "Bot"})])
def test_other_comments_and_bots_get_no_reply(fake, body, kw):
    outcome = resolve(fake, body, **kw)
    assert not outcome.go and not outcome.reply
    assert fake.gets == []


@pytest.mark.parametrize(
    "answer",
    [
        {"role_name": "read", "permission": "read"},
        {"role_name": "triage", "permission": "read"},
        {"role_name": "reviewer", "permission": "read"},  # a custom role built on read
        None,  # GitHub's 404: no such user
    ],
)
def test_only_write_maintain_or_admin_may_ask(fake, answer):
    fake.api[f"repos/{REPO}/collaborators/dvir/permission"] = answer
    outcome = resolve(fake, "/perf pokec")
    assert not outcome.go
    assert "write, maintain or admin" in outcome.reply
    assert f"repos/{REPO}/pulls/{PR}" not in fake.gets  # checked before anything else


@pytest.mark.parametrize(
    "answer",
    [
        {"role_name": "admin", "permission": "admin"},
        {"role_name": "maintain", "permission": "write"},
        {"role_name": "write", "permission": "write"},
        {"role_name": "release-manager", "permission": "write"},  # a custom role built on write
    ],
)
def test_each_writing_role_may_ask(fake, answer):
    fake.api[f"repos/{REPO}/collaborators/dvir/permission"] = answer
    assert resolve(fake, "/perf pokec").go


def test_a_permission_lookup_that_fails_is_never_read_as_no_access(fake):
    path = f"repos/{REPO}/collaborators/dvir/permission"
    fake.failures[path] = ["gh: API rate limit exceeded (HTTP 403)"]
    with pytest.raises(pc.GitHubError, match="rate limit"):
        resolve(fake, "/perf pokec")


def test_the_permission_comes_before_the_usage_reply(fake):
    fake.api[f"repos/{REPO}/collaborators/dvir/permission"] = {"role_name": "read"}
    assert "write, maintain or admin" in resolve(fake, "/perf nonsense words here").reply


def test_a_fork_or_a_closed_pull_request_is_refused(fake):
    fake.api[f"repos/{REPO}/pulls/{PR}"] = pull(repo="someone/FalkorDB")
    assert "forks" in resolve(fake, "/perf pokec").reply
    fake.api[f"repos/{REPO}/pulls/{PR}"] = pull(state="closed")
    assert "not open" in resolve(fake, "/perf pokec").reply


def jobs_of(fake: Fake, run_id: int, *jobs: tuple[str, str | None]) -> None:
    fake.api[f"repos/{REPO}/actions/runs/{run_id}/jobs"] = {
        "jobs": [{"name": name, "conclusion": conclusion} for name, conclusion in jobs]
    }


def test_a_second_request_while_one_measures_is_refused_with_its_link(fake):
    earlier = {"id": RUN_ID - 3, "display_title": TITLE}
    fake.api[f"repos/{REPO}/actions/workflows/77/runs?status=in_progress"] = {
        "workflow_runs": [earlier, {"id": RUN_ID + 1, "display_title": TITLE}]
    }
    jobs_of(fake, RUN_ID - 3, ("resolve", "success"), ("measure / micro / preflight", None))
    outcome = resolve(fake, "/perf pokec")
    assert not outcome.go
    assert f"/actions/runs/{RUN_ID - 3})" in outcome.reply
    assert "still going" in outcome.reply


@pytest.mark.parametrize(
    "jobs",
    [
        [("resolve", None)],  # still resolving: the measure job's concurrency group queues them
        [("resolve", "success"), ("measure", "skipped"), ("answer", "skipped")],  # refused
        [("resolve", "success")],  # not the command, or a refusal
        [("measurement", None)],  # a job merely named alike
    ],
)
def test_a_run_that_does_not_measure_never_blocks_a_request(fake, jobs):
    """Anyone's comment that starts like the command gets the same title; only a run that
    reached its measure job counts as a /perf run that is going."""
    fake.api[f"repos/{REPO}/actions/workflows/77/runs?status=in_progress"] = {
        "workflow_runs": [{"id": RUN_ID - 3, "display_title": TITLE}]
    }
    jobs_of(fake, RUN_ID - 3, *jobs)
    assert resolve(fake, "/perf pokec").go


@pytest.mark.parametrize(
    "others",
    [
        [{"id": RUN_ID + 1, "display_title": TITLE}],  # the later one refuses, not this one
        [{"id": RUN_ID - 1, "display_title": "/perf on #1"}],  # another pull request
        [{"id": RUN_ID - 1, "display_title": "Comment, not /perf"}],
        [{"id": RUN_ID, "display_title": TITLE}],  # this run itself
    ],
)
def test_other_runs_do_not_make_a_request_wait(fake, others):
    fake.api[f"repos/{REPO}/actions/workflows/77/runs?status=queued"] = {"workflow_runs": others}
    assert resolve(fake, "/perf pokec").go


@pytest.mark.parametrize(
    ("change", "said"),
    [
        ("no-run", "is not built yet"),
        ("other-pr", "is not built yet"),
        ("pending", "is still being built"),
        ("failure", "did not build"),
        ("skipped", "did not build"),
        ("no-jobs", "is not built yet"),
    ],
)
def test_the_head_image_is_used_only_when_built_for_that_commit(fake, change, said):
    runs = f"repos/{REPO}/actions/runs?head_sha={SHA}&event=pull_request"
    jobs = f"repos/{REPO}/actions/runs/900/jobs"
    if change == "no-run":
        fake.api[runs] = {"workflow_runs": []}
    elif change == "other-pr":
        fake.api[runs] = {"workflow_runs": [rust_pr_run(pr=PR + 1)]}
    elif change == "no-jobs":
        fake.api[jobs] = {"jobs": [{"name": "lint", "conclusion": "success"}]}
    else:
        fake.api[jobs]["jobs"][1]["conclusion"] = None if change == "pending" else change
    outcome = resolve(fake, "/perf pokec")
    assert not outcome.go
    assert f"`{SHA[:9]}`" in outcome.reply and said in outcome.reply
    assert not any("imagetools" in g for g in fake.gets)


def test_the_newest_rust_pr_run_of_the_commit_decides(fake):
    runs = f"repos/{REPO}/actions/runs?head_sha={SHA}&event=pull_request"
    fake.api[runs] = {"workflow_runs": [rust_pr_run(), rust_pr_run(run_id=901, number=11)]}
    fake.api[f"repos/{REPO}/actions/runs/901/jobs"] = {
        "jobs": [{"name": BUILD_JOBS[0], "conclusion": "failure"}]
    }
    assert "did not build" in resolve(fake, "/perf pokec").reply


def test_a_head_that_moves_while_the_image_is_pinned_is_refused(fake):
    fake.pull_answers = [pull(), pull(sha=NEW_SHA)]
    outcome = resolve(fake, "/perf pokec")
    assert not outcome.go
    assert "moved" in outcome.reply and f"`{SHA[:9]}`" in outcome.reply
    # The head is read again after the digest: two reads of the pull request.
    assert fake.gets.count(f"repos/{REPO}/pulls/{PR}") == 2


def test_a_missing_image_is_refused(fake):
    fake.images[f"{pc.CANDIDATE_REPO}:rc-pr-{PR}-valkey"] = None
    assert "does not exist" in resolve(fake, "/perf pokec").reply


def test_an_unknown_release_is_answered_with_the_releases(fake):
    outcome = resolve(fake, "/perf pokec baseline 6.9.9")
    assert not outcome.go
    assert "`docker.io/falkordb/falkordb-server:6.9.9` does not exist" in outcome.reply
    assert "`6.2.1`, `6.2.0`, `6.0.4`, `4.22.4`, `4.22.3`" in outcome.reply
    assert "6.3.0" not in outcome.reply


def test_a_registry_failure_is_never_read_as_no_image(fake):
    fake.images[f"{pc.CANDIDATE_REPO}:rc-pr-{PR}-valkey"] = OSError("connection reset")
    with pytest.raises(pc.GitHubError):
        resolve(fake, "/perf pokec")


def test_a_get_that_fails_without_an_answer_is_tried_again(fake):
    path = f"repos/{REPO}/pulls/{PR}"
    fake.failures[path] = ["stream error: stream ID 1; CANCEL", "HTTP 502"]
    assert resolve(fake, "/perf pokec").go
    fake.failures[path] = ["a", "b", "c"]
    with pytest.raises(pc.GitHubError):
        resolve(fake, "/perf pokec")


def test_a_4xx_answer_is_not_tried_again(fake):
    path = f"repos/{REPO}/pulls/{PR}"
    fake.failures[path] = ["gh: Server Error (HTTP 404)", "never read"]
    with pytest.raises(pc.GitHubError):
        resolve(fake, "/perf pokec")
    assert fake.failures[path] == ["never read"]


# --- the command line -----------------------------------------------------------------


def environ(tmp_path: Path, body: str, **extra) -> dict:
    return {
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_RUN_ID": str(RUN_ID),
        "GITHUB_OUTPUT": str(tmp_path / "out"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "PR": str(PR),
        "ACTOR": "dvir",
        "ACTOR_TYPE": "User",
        "COMMENT_BODY": body,
        "COMMENT_ID": "42",
        **extra,
    }


def outputs(tmp_path: Path) -> dict:
    lines = (tmp_path / "out").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines)


def test_a_request_that_runs_gets_a_reaction_and_its_outputs(fake, tmp_path):
    registry = tmp_path / "checks.json"
    registry.write_text(registry_json())
    env = environ(tmp_path, "/perf micro-hw baseline 4.22.4")
    assert (
        pc.main(
            ["resolve", "--registry", str(registry), "--github-output", env["GITHUB_OUTPUT"]],
            env,
            run=fake,
        )
        == 0
    )
    got = outputs(tmp_path)
    assert got["go"] == "true" and got["checks"] == "micro-hw"
    assert fake.posts == [(f"repos/{REPO}/issues/comments/42/reactions", {"content": "rocket"})]


def test_a_refusal_is_one_reply_on_the_pull_request(fake, tmp_path):
    env = environ(tmp_path, "/perf nope")
    assert pc.main(["resolve", "--github-output", env["GITHUB_OUTPUT"]], env, run=fake) == 0
    assert outputs(tmp_path)["go"] == "false"
    [(path, body)] = fake.posts
    assert path == f"repos/{REPO}/issues/{PR}/comments"
    assert body["body"].startswith(pc.MARKER + "\n") and "Unknown check `nope`" in body["body"]


def test_a_dry_run_posts_nothing_and_writes_the_summary(fake, tmp_path):
    env = environ(tmp_path, "/perf nope")
    pc.main(["resolve", "--github-output", env["GITHUB_OUTPUT"], "--dry-run"], env, run=fake)
    env = environ(tmp_path, "/perf micro-hw")
    pc.main(["resolve", "--github-output", env["GITHUB_OUTPUT"], "--dry-run"], env, run=fake)
    assert fake.posts == []
    summary = (tmp_path / "summary").read_text()
    assert f"would be posted on #{PR}" in summary and "Unknown check `nope`" in summary
    assert "Dry run: no reaction" in summary


def test_a_reply_that_could_not_be_posted_fails_the_step(fake, tmp_path):
    fake.post_rc = 1
    env = environ(tmp_path, "/perf nope")
    assert pc.main(["resolve", "--github-output", env["GITHUB_OUTPUT"]], env, run=fake) == 1
    assert outputs(tmp_path)["go"] == "false"


def test_a_check_that_fails_says_so_and_fails_the_step(fake, tmp_path):
    fake.images[f"{pc.CANDIDATE_REPO}:rc-pr-{PR}-valkey"] = OSError("registry down")
    env = environ(tmp_path, "/perf micro-hw")
    assert pc.main(["resolve", "--github-output", env["GITHUB_OUTPUT"]], env, run=fake) == 1
    assert outputs(tmp_path)["go"] == "false"
    [(_, body)] = fake.posts
    assert "could not be checked" in body["body"] and "registry down" in body["body"]


# --- answer ---------------------------------------------------------------------------


def write(tmp_path: Path, night: dict) -> Path:
    path = tmp_path / "nightly.json"
    path.write_text(json.dumps(night), encoding="utf-8")
    return path


RUN = "https://github.com/FalkorDB/FalkorDB/actions/runs/5000"
SITE = "https://github.com/FalkorDB/FalkorDB/actions/runs/5000/artifacts/77"


def test_the_answer_has_every_verdict_both_images_and_commits_and_both_links(tmp_path):
    night = examples.example("nightly-red.json")
    text = pc.answer(
        write(tmp_path, night),
        "/perf all baseline 6.2.0",
        run=RUN,
        dashboard=SITE,
        measured="failure",
    )
    assert text.startswith(pc.MARKER + "\n### Perf: RED\n")
    assert "`/perf all baseline 6.2.0`" in text
    for side in ("candidate", "baseline"):
        image, commit = night[side]["image"], night[side]["commit"]
        assert f"`{image}`" in text
        assert f"[`{commit[:12]}`](https://github.com/{REPO}/commit/{commit})" in text
    for check in night["checks"]:
        assert f"| {check['check']} | {check['verdict']}" in text
    assert "| micro | regression (confirmed) |" in text
    assert "worst regression rows" in text
    assert f"[run]({RUN})" in text and f"[dashboard]({SITE})" in text
    assert "resolved by the run" not in text


def test_a_main_baseline_says_the_run_resolved_it(tmp_path):
    text = pc.answer(
        write(tmp_path, examples.example("nightly-green.json")),
        "/perf pokec baseline main",
        run=RUN,
    )
    assert "### Perf: GREEN" in text
    assert "(main's newest merge-queue image, resolved by the run)" in text


def test_a_run_that_left_no_nightly_is_an_error_never_green(tmp_path):
    for path in (None, tmp_path / "absent.json"):
        text = pc.answer(path, "/perf pokec baseline main", run=RUN, measured="cancelled")
        assert "### Perf: ERROR" in text and "GREEN" not in text
        assert "left no nightly.json (measure job: cancelled)" in text


def test_an_unreadable_nightly_is_an_error(tmp_path):
    path = tmp_path / "nightly.json"
    path.write_text("{not json")
    assert "### Perf: ERROR" in pc.answer(path, "/perf pokec baseline main", run=RUN)
    night = examples.example("nightly-green.json")
    night["status"] = "purple"
    path = write(tmp_path, night)
    assert "### Perf: ERROR" in pc.answer(path, "/perf pokec baseline main", run=RUN)


def test_a_green_night_from_a_job_that_failed_says_so(tmp_path):
    path = write(tmp_path, examples.example("nightly-green.json"))
    assert "measure job ended `failure`" in pc.answer(
        path, "/perf all baseline main", run=RUN, measured="failure"
    )
    assert "measure job ended" not in pc.answer(
        path, "/perf all baseline main", run=RUN, measured="success"
    )


def test_text_from_the_night_cannot_inject_markdown(tmp_path):
    night = examples.example("nightly-green.json")
    night["checks"][0]["summary"] = "<img src=x> | [click](https://evil) `code`"
    night["candidate"]["label"] = "<b>pr</b>"
    path = write(tmp_path, night)
    text = pc.answer(path, "/perf all baseline main", run=RUN)
    assert "<img" not in text and "<b>" not in text
    # Escaped, so it reads as the text it is, not a link.
    assert "\\[click\\](https://evil)" in text and "\\`code\\`" in text
    # A request that is not the normalised form is not quoted back.
    hostile = pc.answer(path, "`/perf` <x>", run=RUN)
    assert "<x>" not in hostile


def test_answer_dry_run_writes_the_summary(tmp_path):
    fake = Fake()
    env = environ(tmp_path, "")
    path = write(tmp_path, examples.example("nightly-green.json"))
    argv = [
        "answer",
        "--nightly",
        str(path),
        "--request",
        "/perf all baseline main",
        "--pr",
        str(PR),
        "--dry-run",
    ]
    assert pc.main(argv, env, run=fake) == 0
    assert fake.posts == []
    assert "### Perf: GREEN" in (tmp_path / "summary").read_text()
    argv.remove("--dry-run")
    assert pc.main(argv, env, run=fake) == 0
    [(path, body)] = fake.posts
    assert path == f"repos/{REPO}/issues/{PR}/comments" and "### Perf: GREEN" in body["body"]
    # An answer that could not be posted fails the step: the run must not end green.
    fake.post_rc = 1
    assert pc.main(argv, env, run=fake) == 1


def test_a_red_night_whose_regressions_are_all_waived_says_so(tmp_path):
    path = write(tmp_path, examples.example("nightly-red.json"))
    waived = pc.answer(path, "/perf all baseline main", run=RUN, measured="success")
    assert "### Perf: RED" in waived and "bench/waivers.yaml waives every regression" in waived
    assert "waives" not in pc.answer(path, "/perf all baseline main", run=RUN, measured="failure")


# --- the workflow ---------------------------------------------------------------------


def comment_workflow() -> dict:
    doc = yaml.safe_load((BENCH.parent / ".github/workflows/perf-comment.yml").read_text())
    doc["on"] = doc.pop(True, doc.get("on"))  # PyYAML reads the key `on` as True
    return doc


def test_the_comment_reaches_a_script_only_through_env():
    """issue_comment runs carry secrets: an expression in a `run:` block is pasted into the
    script before bash reads it, so a comment there could run as code."""
    for name, job in comment_workflow()["jobs"].items():
        for step in job.get("steps", []):
            assert "${{" not in step.get("run", ""), (name, step.get("name"))
    text = (BENCH.parent / ".github/workflows/perf-comment.yml").read_text()
    for line in text.splitlines():
        if "github.event.comment.body" in line:
            assert line.strip().startswith(("COMMENT_BODY:", "&&", "${{", "github.event")), line


def test_every_request_that_runs_is_answered():
    jobs = comment_workflow()["jobs"]
    assert jobs["answer"]["needs"] == ["resolve", "measure"]
    assert jobs["answer"]["if"] == "always() && needs.resolve.outputs.go == 'true'"
    [step] = [s for s in jobs["answer"]["steps"] if s.get("name") == "Answer on the pull request"]
    # Whatever the dashboard step did: a failed dashboard must not leave the request unanswered.
    assert step["if"] == "${{ !cancelled() }}"
    assert jobs["measure"]["concurrency"] == {
        "group": "perf-comment-${{ needs.resolve.outputs.pr }}",
        "cancel-in-progress": False,
    }
    # The measure job's jobs show as "measure / <job>": what busy_with looks for.
    assert "measure" in jobs and pc.MEASURE_JOB.match("measure / resolve")
