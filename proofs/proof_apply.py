#!/usr/bin/env python3
"""Commit the proof-impact bot's verified patch to a PR branch, on request.

Run by .github/workflows/proof-apply.yml from the default branch (never from the PR).
Everything about the PR (comment text, branch name, login) arrives through the
environment, never through a shell command line.

  resolve   Who asked, and may the bot act? Replies and stops when not. Finds the
            latest successful proof-impact run for the PR's current head.
  apply     Reads that run's `proof-gate` artifact (downloaded by the workflow into
            GATE_DIR), checks it again, and only when its gate status is `pass`
            commits the patch on top of the PR head and pushes it as a plain
            fast-forward. Never forces; a moved head changes nothing.

Both talk to GitHub through the `gh` CLI (GH_TOKEN). Tests put a fake `gh` first on PATH.
"""
import base64
import json
import os
import re
import subprocess
import sys

MARKER = "<!-- proof-apply -->"
# Who may ask: the role names the collaborators API reports.
ROLES = {"admin", "maintain", "write"}
# What a PR must not change for its proof-impact run to be trusted: the gate is that
# workflow's own code, and a `pull_request` run uses the PR's copy of it.
GATE_CODE = re.compile(r"^(\.github/workflows/proof-(impact|apply)\.yml|proofs/(proof_apply|pr_impact)\.py|"
                       r"proofs/(lean_ci|coverage)\.sh)$")
# Identity of the bot's commits. proof-impact.yml recognises them by this email and
# the `Proof-Bot-Run:` trailer (keep both in sync), so it does not re-prove them.
BOT_NAME = "falkordb-proof-bot"
BOT_EMAIL = "falkordb-proof-bot@users.noreply.github.com"
TRAILER = "Proof-Bot-Run"
LABEL = "proofs:apply"
COMMAND = re.compile(r"\A[ \t]*/proofs[ \t]+apply[ \t]*\Z")
APP_NOTE = (
    "CI does not run on this commit: it was pushed with `GITHUB_TOKEN`, which GitHub never "
    "lets start workflows. Push an empty commit (`git commit --allow-empty`) or close and reopen the "
    "PR to run it on the new head (re-running an old run would check the old head). "
    "Configure the repository secrets `PROOF_BOT_APP_ID` and `PROOF_BOT_APP_KEY` (a GitHub App "
    "with contents and pull-requests write) so the bot's pushes start CI."
)


def env(name, default=""):
    return os.environ.get(name, default)


def gh(*args, data=None):
    """Run `gh`; return (returncode, stdout, stderr)."""
    p = subprocess.run(["gh", *args], capture_output=True, text=True,
                       input=None if data is None else json.dumps(data))
    return p.returncode, p.stdout, p.stderr


def api(path, *extra):
    rc, out, err = gh("api", path, *extra)
    if rc != 0:
        raise RuntimeError(f"gh api {path}: {err.strip()[:300]}")
    return json.loads(out) if out.strip() else None


def reply(repo, pr, text):
    """One comment on the PR. Markdown only; no mention of the commenter."""
    rc, _, err = gh("api", "-X", "POST", f"repos/{repo}/issues/{pr}/comments", "--input", "-",
                    data={"body": f"{MARKER}\n{text}"})
    if rc != 0:
        print(f"could not comment: {err.strip()[:300]}", file=sys.stderr)


def output(**kv):
    path = env("GITHUB_OUTPUT")
    lines = "".join(f"{k}={v}\n" for k, v in kv.items())
    if path:
        with open(path, "a") as f:
            f.write(lines)
    else:
        sys.stdout.write(lines)


def git(cwd, *args, check=True):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])}: {(p.stderr or p.stdout).strip()[:400]}")
    return p


def allowlist(workflow):
    """PROOF_PATHS and PROOF_DENY, read from the default branch's proof-impact.yml, so
    the gate and this job cannot drift apart. Fails closed."""
    with open(workflow) as f:
        text = f.read()
    found = {}
    for name in ("PROOF_PATHS", "PROOF_DENY"):
        m = re.search(rf"^[ \t]*{name}:[ \t]*'((?:[^']|'')*)'[ \t]*$", text, re.M)
        if not m:
            raise RuntimeError(f"{name} not found in {workflow}")
        found[name] = re.compile(m.group(1).replace("''", "'"))
    return found["PROOF_PATHS"], found["PROOF_DENY"]


# ---------------------------------------------------------------- resolve

def resolve():
    repo, pr, actor = env("REPO"), env("PR"), env("ACTOR")
    event = env("EVENT_NAME")
    output(go="false")

    if event == "issue_comment":
        first = (env("COMMENT_BODY").splitlines() or [""])[0]
        if not COMMAND.match(first):
            print("not an apply command"); return
    elif event == "pull_request_target":
        if env("LABEL_NAME") != LABEL:
            print("another label"); return
        # Re-adding the label must ask again: drop it now, whoever added it.
        gh("api", "-X", "DELETE", f"repos/{repo}/issues/{pr}/labels/{LABEL.replace(':', '%3A')}")
    else:
        print(f"unexpected event {event}"); return

    # The commenter's own right on this repo, not the PR author's.
    try:
        perm = api(f"repos/{repo}/collaborators/{actor}/permission")
    except RuntimeError as e:
        print(e, file=sys.stderr); perm = {}
    role = perm.get("role_name") or perm.get("permission") or "none"
    if role not in ROLES:
        reply(repo, pr, f"`/proofs apply` is ignored: only people with write, maintain or admin "
                        f"access to the repository can commit the bot's patch (`{actor}` has `{role}`).")
        return

    pull = api(f"repos/{repo}/pulls/{pr}")
    head = pull["head"]
    head_repo = (head.get("repo") or {}).get("full_name")
    if head_repo != repo:
        reply(repo, pr, "Apply is not available for pull requests from forks: the bot has no "
                        "right to push to a fork's branch. Apply the patch from the "
                        "re-prove comment's `proof-gate` artifact yourself.")
        return
    if pull.get("state") != "open":
        reply(repo, pr, "This pull request is not open; nothing to apply.")
        return
    sha = head["sha"]

    # The gate is proof-impact.yml's own code, and a `pull_request` run uses the PR's
    # copy of it: a PR that edits the gate (or this bot) cannot vouch for its own patch.
    touched = []
    for page in range(1, 31):
        files = api(f"repos/{repo}/pulls/{pr}/files?per_page=100&page={page}") or []
        touched += [f["filename"] for f in files if GATE_CODE.match(f.get("filename", ""))]
        if len(files) < 100:
            break
    if touched:
        listing = ", ".join(f"`{t}`" for t in touched)
        reply(repo, pr, f"Not applied: this PR changes the code that checks the bot's patches ({listing}), "
                        f"so the bot's run cannot vouch for itself. Apply `proofs.patch` from the run's "
                        f"`proof-gate` artifact yourself.")
        return

    runs = api(f"repos/{repo}/actions/workflows/proof-impact.yml/runs"
               f"?head_sha={sha}&event=pull_request&per_page=100").get("workflow_runs", [])
    # Only this PR's runs: two PRs can share a head commit and differ in base.
    runs = [r for r in runs if r.get("head_sha") == sha
            and (r.get("head_repository") or {}).get("full_name") == repo
            and any(str(p.get("number")) == str(pr) for p in r.get("pull_requests") or [])]
    ok = sorted((r for r in runs if r.get("conclusion") == "success"),
                key=lambda r: (r.get("created_at") or "", r["id"]))
    if not ok:
        busy = any(r.get("status") != "completed" for r in runs)
        why = ("the bot's run on the current head is still in progress" if busy
               else "there is no successful proof-impact run for the current head")
        reply(repo, pr, f"Not applied: {why} (`{sha[:9]}`). Wait for the bot's run on the "
                        f"current head, then comment `/proofs apply` again.")
        return
    # A title/body edit makes a newer run whose jobs are all skipped: take the newest
    # one that has a live gate artifact.
    run = None
    for r in reversed(ok):
        arts = api(f"repos/{repo}/actions/runs/{r['id']}/artifacts?per_page=100").get("artifacts", [])
        if any(a.get("name") == "proof-gate" and not a.get("expired") for a in arts):
            run = r
            break
    if run is None:
        reply(repo, pr, f"Nothing to apply for `{sha[:9]}`: no successful proof-impact run on it has a "
                        f"`proof-gate` artifact (no verified function changed, or the artifact expired).")
        return
    print(f"actor {actor} ({role}); head {sha}; run {run['id']}")
    output(go="true", head_sha=sha, head_ref=head["ref"], run_id=run["id"])


# ---------------------------------------------------------------- apply

def refuse(repo, pr, text):
    reply(repo, pr, text)
    output(committed="false")
    sys.exit(0)


def apply():
    repo, pr, actor = env("REPO"), env("PR"), env("ACTOR")
    sha, ref, run_id = env("HEAD_SHA"), env("HEAD_REF"), env("RUN_ID")
    gate, work = env("GATE_DIR"), env("WORKTREE")
    run_url = f"{env('GITHUB_SERVER_URL', 'https://github.com')}/{repo}/actions/runs/{run_id}"

    def read(name):
        try:
            with open(os.path.join(gate, name)) as f:
                return f.read()
        except OSError:
            return None

    status = (read("gate_status") or "").strip()
    if not status:
        refuse(repo, pr, f"Not applied: the [run]({run_url}) has no gate result (no `proof-gate` "
                         f"artifact), so no verified patch exists for `{sha[:9]}`. This happens when no "
                         f"verified function changed or the gate did not finish.")
    # The artifact must be about the head we were asked about.
    if (read("head_sha") or "").strip() != sha:
        refuse(repo, pr, f"Not applied: the [run]({run_url})'s artifact is not for `{sha[:9]}`.")
    if status != "pass":
        reason = (read("gates.md") or "").strip()[:2000]
        refuse(repo, pr, f"Not applied: the gate result is **{status}**, and a patch that is not "
                         f"`pass` is never committed by the bot. See the [run]({run_url}).\n\n{reason}")
    patch = os.path.join(gate, "proofs.patch")
    if not os.path.exists(patch) or os.path.getsize(patch) == 0:
        refuse(repo, pr, f"Nothing to apply: the [run]({run_url}) needed no proof change.")

    # Hold the head still. The push below is the real guard; this saves the work.
    if api(f"repos/{repo}/pulls/{pr}")["head"]["sha"] != sha:
        refuse(repo, pr, f"The PR head moved since the bot's run (`{sha[:9]}`); nothing changed. "
                         f"Wait for the bot's run on the new head and comment `/proofs apply` again.")
    if git(work, "rev-parse", "HEAD").stdout.strip() != sha:
        raise RuntimeError("the checkout is not the PR head")

    paths_re, deny_re = allowlist(env("ALLOWLIST_WORKFLOW"))
    p = git(work, "apply", "--index", os.path.abspath(patch), check=False)
    if p.returncode != 0:
        refuse(repo, pr, f"Not applied: the patch does not apply to `{sha[:9]}`:\n```\n{p.stderr.strip()[:1500]}\n```")
    # Same checks as the gate, on what git really staged: every path allowed, regular
    # non-executable files (or deletions) only, and each inside a Lake project.
    names = [n for n in git(work, "diff", "--cached", "--name-only", "--no-renames", "-z").stdout.split("\0") if n]
    raw = git(work, "diff", "--cached", "--raw", "--no-renames").stdout.splitlines()
    bad = [n for n in names if not paths_re.search(n) or deny_re.search(n)]
    bad += [f"{ln.split()[-1]} (mode {ln.split()[1]})" for ln in raw if ln.split()[1] not in ("100644", "000000")]
    for n in names:
        if n == "proofs/FINDINGS.md":
            continue
        proj = "/".join(n.split("/")[:2])
        if not any(git(work, "cat-file", "-e", f"HEAD:{proj}/{lk}", check=False).returncode == 0
                   for lk in ("lakefile.toml", "lakefile.lean")):
            bad.append(f"{n} (no Lake project at {proj})")
    if not names:
        refuse(repo, pr, f"Nothing to apply: the patch from the [run]({run_url}) changes nothing.")
    if bad:
        listing = "\n".join(f"- `{b}`" for b in bad)
        refuse(repo, pr, f"Not applied: the patch touches files outside the proof allowlist; nothing was committed.\n\n{listing}")

    msg = (f"proofs: update the Lean proofs for this change (proof-impact bot)\n\n"
           f"Verified patch of the proof-impact run {run_url}\n"
           f"at {sha}, applied on request of {actor}.\n\n"
           f"{TRAILER}: {run_url}\n")
    git(work, "-c", f"user.name={BOT_NAME}", "-c", f"user.email={BOT_EMAIL}", "commit", "-q", "-m", msg)
    new = git(work, "rev-parse", "HEAD").stdout.strip()

    remote = env("REMOTE") or f"https://github.com/{repo}.git"
    cfg = {}
    token = env("PUSH_TOKEN")
    if token:
        b64 = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        print(f"::add-mask::{b64}")
        cfg = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
               "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {b64}"}
    # A plain push: a head the author moved meanwhile is rejected, never overwritten.
    # The lease pins the remote tip to the head we built on: a branch the author reset
    # to an ancestor or deleted meanwhile is not restored. The update is still a
    # fast-forward, so the lease never overwrites anything.
    push = subprocess.run(["git", "push", "-q", f"--force-with-lease=refs/heads/{ref}:{sha}", remote,
                           f"HEAD:refs/heads/{ref}"], cwd=work,
                          capture_output=True, text=True, env={**os.environ, **cfg})
    if push.returncode != 0:
        now = subprocess.run(["git", "ls-remote", remote, f"refs/heads/{ref}"], cwd=work,
                             capture_output=True, text=True, env={**os.environ, **cfg}).stdout.split()
        if now and now[0] != sha:
            refuse(repo, pr, f"The PR head moved while the bot was working (`{sha[:9]}` is now "
                             f"`{now[0][:9]}`); nothing was pushed. Wait for the bot's run on the new "
                             f"head and comment `/proofs apply` again.")
        err = push.stderr.strip()[:600]
        refuse(repo, pr, f"The push was rejected and nothing changed (a branch rule may need the "
                         f"bot to be allowed to push):\n```\n{err}\n```")

    note = "CI runs on this commit." if env("HAS_APP") == "true" else APP_NOTE
    reply(repo, pr, f"Committed the bot's verified proof patch as `{new}` on `{ref}` "
                    f"(from the [run]({run_url}), requested by {actor}).\n\n{note}")
    output(committed="true", commit=new)
    print(f"committed {new}")


if __name__ == "__main__":
    cmds = {"resolve": resolve, "apply": apply}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit("usage: proof_apply.py resolve|apply")
    try:
        cmds[sys.argv[1]]()
    except Exception as e:  # never leave the requester without an answer
        print(f"error: {e}", file=sys.stderr)
        if env("REPO") and env("PR"):
            reply(env("REPO"), env("PR"), "Not applied: the bot hit an unexpected error and changed "
                  f"nothing; comment `/proofs apply` again to retry. See the [workflow run]({env('GITHUB_SERVER_URL', 'https://github.com')}/"
                  f"{env('REPO')}/actions/runs/{env('GITHUB_RUN_ID')}).")
        sys.exit(1)
