#!/usr/bin/env python3
"""Work out, and optionally push, the public image tags of one publish.

Every publishing workflow gets its tags from here, so the tags a workflow
pushes are the tags this file's tests check:

  edge     rust-push.yml     the queue's rc-mg-<sha>-<server> → the rolling tags
  release  release-tag.yml   the queue's rc-mg-<sha>-<server> → <version>[, latest]
  rebuild  release-image.yml rc-rebuild-<sha>-<server> → <tag>[, latest]
  per-os   release-tag.yml   per-OS server image archives → <version>-<os>-<server>

The server is always the last part of a tag, and a tag with no server means
Valkey. Every public tag is a multi-arch index, so none names an architecture.

Without --execute nothing is pushed: the commands are printed, which is the
dry run. With --execute each source is checked first (its platforms and the
server its label names), and nothing is pushed unless every source passes.
After the push every tag must hold the digest it was given, on GHCR and on
Docker Hub alike: publishing retags, it never rebuilds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

GHCR = "ghcr.io/falkordb"
DOCKER_HUB = "docker.io/falkordb"
REGISTRIES = (GHCR, DOCKER_HUB)

REPOS = ("falkordb", "falkordb-server")
PER_OS_REPO = "falkordb-server"

SERVERS = ("valkey", "redis")
# The server an unsuffixed tag (edge, edge-rs, latest, <version>) points at.
DEFAULT_SERVER = "valkey"

# CI's architecture names and the platform each is published as.
PLATFORMS = {"x64": "linux/amd64", "arm64v8": "linux/arm64"}

# The per-OS images: (os, server). Release only, falkordb-server only.
PER_OS = (
    ("alpine", "valkey"),
    ("amazonlinux2023", "valkey"),
    ("rhel9", "redis"),
    ("amazonlinux2023", "redis"),
)

SERVER_LABEL = "com.falkordb.server.name"

# Whole tags other workflows own; a rebuild may not publish over them.
RESERVED_TAGS = {"edge", "edge-rs", "latest"}
# Parts of a tag that say what it holds. A version or an ad-hoc tag carrying
# one would collide with, or impersonate, another tag: 6.1.0-alpine's Valkey
# tag would be 6.1.0's Alpine tag.
RESERVED_PARTS = (
    set(SERVERS)
    | set(PLATFORMS)
    | {p.split("/")[1] for p in PLATFORMS.values()}
    | {os_name for os_name, _ in PER_OS}
)

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
ARCHIVE_RE = re.compile(
    r"^falkordb-server-(?P<os>[a-z0-9]+)-(?P<server>[a-z]+)-(?P<arch>[a-z0-9]+)\.tar$"
)


class PlanError(Exception):
    """The inputs cannot be published as asked."""


@dataclass(frozen=True)
class Retag:
    """Point every tag in `tags` at the index `source` already holds."""

    source: str
    server: str
    tags: tuple


@dataclass(frozen=True)
class Assemble:
    """Push per-arch archives as one multi-arch index tagged `name`."""

    name: str
    server: str
    archives: tuple  # ((arch, path), ...)
    tags: tuple  # (ghcr tag, docker hub tag)


# --- versions -----------------------------------------------------------------


def parse_version(text: str):
    """(major, minor, patch, prerelease-or-None) of X.Y.Z[-pre], else None."""
    m = VERSION_RE.match(text)
    if not m:
        return None
    return int(m[1]), int(m[2]), int(m[3]), m[4]


def check_parts(tag: str, what: str) -> None:
    for part in re.split(r"[.-]", tag):
        if part in RESERVED_PARTS:
            raise PlanError(
                f"{what} '{tag}' contains '{part}', which tags use to name a "
                "server, an OS or an architecture"
            )


def check_version(version: str) -> None:
    v = parse_version(version)
    if v is None:
        raise PlanError(f"version '{version}' is not X.Y.Z[-prerelease] (no leading v)")
    if v[3] is not None:
        check_parts(v[3], "prerelease")


def check_sha(sha: str) -> None:
    if not SHA_RE.match(sha):
        raise PlanError(f"commit '{sha}' is not a full 40-character SHA")


def is_prerelease(version: str) -> bool:
    v = parse_version(version)
    return v is not None and v[3] is not None


def moves_latest(version: str, releases: Iterable[dict]) -> bool:
    """Whether `version` is where latest* points once it is published.

    latest* points at the highest stable version that has a published,
    non-draft GitHub release. A prerelease never moves it, whatever its
    GitHub release says, and nor does a version without a published release
    (a tag pushed over a draft, say).
    """
    if is_prerelease(version):
        return False
    stable = set()
    for release in releases:
        if release.get("draft") or release.get("prerelease"):
            continue
        name = str(release.get("tag_name", ""))
        v = parse_version(name[1:] if name.startswith("v") else name)
        if v is not None and v[3] is None:
            stable.add(v[:3])
    mine = parse_version(version)[:3]
    return mine in stable and mine == max(stable)


# --- plans --------------------------------------------------------------------


def server_tags(name: str, server: str) -> list:
    """`name-<server>`, and bare `name` too for the default server."""
    tags = [f"{name}-{server}"]
    if server == DEFAULT_SERVER:
        tags.append(name)
    return tags


def everywhere(repo: str, tags: Iterable[str]) -> tuple:
    return tuple(f"{registry}/{repo}:{tag}" for tag in tags for registry in REGISTRIES)


def edge_plan(sha: str) -> list:
    """The rolling tags, from the merge queue's images of `sha`."""
    check_sha(sha)
    plan = []
    for repo in REPOS:
        for server in SERVERS:
            tags = server_tags("edge-rs", server)
            if server == DEFAULT_SERVER:
                tags.append("edge")
            plan.append(Retag(f"{GHCR}/{repo}:rc-mg-{sha}-{server}", server, everywhere(repo, tags)))
    return plan


def release_plan(version: str, sha: str, releases: Iterable[dict]) -> list:
    """A release's tags, from the merge queue's images of its commit."""
    check_version(version)
    check_sha(sha)
    latest = moves_latest(version, releases)
    return _named_plan(version, lambda repo, server: f"{GHCR}/{repo}:rc-mg-{sha}-{server}", latest)


def rebuild_plan(tag: str, sha: str, latest: bool) -> list:
    """release-image.yml's tags, from the images it rebuilt from source."""
    check_sha(sha)
    if parse_version(tag) is not None:
        check_version(tag)
    else:
        if not TAG_RE.match(tag):
            raise PlanError(f"tag '{tag}' is not a valid image tag")
        if tag in RESERVED_TAGS or tag.startswith("rc-"):
            raise PlanError(f"tag '{tag}' belongs to another workflow")
        check_parts(tag, "tag")
    if latest and is_prerelease(tag):
        raise PlanError(f"a prerelease never moves latest; '{tag}' is one")
    return _named_plan(tag, lambda repo, server: f"{GHCR}/{repo}:rc-rebuild-{sha}-{server}", latest)


def _named_plan(name: str, source: Callable, latest: bool) -> list:
    plan = []
    for repo in REPOS:
        for server in SERVERS:
            tags = server_tags(name, server)
            if latest:
                tags += server_tags("latest", server)
            plan.append(Retag(source(repo, server), server, everywhere(repo, tags)))
    return plan


def find_archives(images_dir: Path) -> dict:
    """{(os, server, arch): path} of the per-OS archives under images_dir.

    Fails on any archive whose name says something this file does not
    publish, rather than leave it out unnoticed.
    """
    found = {}
    for path in sorted(images_dir.rglob("*.tar")):
        m = ARCHIVE_RE.match(path.name)
        key = m and (m["os"], m["server"], m["arch"])
        if not m or (key[0], key[1]) not in PER_OS or key[2] not in PLATFORMS:
            raise PlanError(f"{path}: not a per-OS image archive this release publishes")
        if key in found:
            raise PlanError(f"{path}: a second archive for {'-'.join(key)}")
        found[key] = path
    return found


def per_os_plan(version: str, archives: Optional[dict] = None) -> list:
    """The per-OS images: every (os, server) in PER_OS, on every platform.

    `archives` is find_archives()'s result; None plans with placeholder paths
    (the dry run). Nothing is planned unless every archive is there.
    """
    check_version(version)
    if archives is not None:
        missing = [
            f"falkordb-server-{os_name}-{server}-{arch}.tar"
            for os_name, server in PER_OS
            for arch in PLATFORMS
            if (os_name, server, arch) not in archives
        ]
        if missing:
            raise PlanError("missing per-OS image archives: " + " ".join(missing))
    plan = []
    for os_name, server in PER_OS:
        name = f"{os_name}-{server}"
        paths = tuple(
            (
                arch,
                str(archives[(os_name, server, arch)])
                if archives is not None
                else f"<images>/falkordb-server-{name}-{arch}.tar",
            )
            for arch in PLATFORMS
        )
        plan.append(Assemble(name, server, paths, everywhere(PER_OS_REPO, [f"{version}-{name}"])))
    return plan


# --- printing -----------------------------------------------------------------


def retag_command(step: Retag, source: Optional[str] = None) -> list:
    cmd = ["docker", "buildx", "imagetools", "create"]
    for tag in step.tags:
        cmd += ["-t", tag]
    return cmd + [source or step.source]


def repository(ref: str) -> str:
    """ghcr.io/falkordb/x of ghcr.io/falkordb/x:tag."""
    return ref.rsplit(":", 1)[0]


def podman_commands(step: Assemble, digest_file: str) -> list:
    """Assemble the archives into one index and push it to GHCR.

    The per-arch manifests are pushed by digest, with no tag, so no tag ever
    names an architecture.
    """
    ghcr_tag, _ = step.tags
    local = f"localhost/publish-{step.name}"
    cmds = [["podman", "manifest", "create", local]]
    for _, path in step.archives:
        cmds.append(["podman", "manifest", "add", local, f"docker-archive:{path}"])
    cmds.append(
        ["podman", "manifest", "push", "--all", "--format", "oci",
         "--digestfile", digest_file, local, f"docker://{ghcr_tag}"]
    )
    return cmds


def hub_copy_command(step: Assemble, digest: str) -> list:
    """Copy the index GHCR now holds to Docker Hub, as is: same digest."""
    ghcr_tag, hub_tag = step.tags
    return ["docker", "buildx", "imagetools", "create", "-t", hub_tag, f"{repository(ghcr_tag)}@{digest}"]


def plan_json(plan: Sequence) -> list:
    out = []
    for step in plan:
        if isinstance(step, Retag):
            out.append({"source": step.source, "server": step.server, "tags": list(step.tags)})
        else:
            out.append(
                {
                    "name": step.name,
                    "server": step.server,
                    "archives": {arch: path for arch, path in step.archives},
                    "tags": list(step.tags),
                }
            )
    return out


def print_plan(plan: Sequence, out=sys.stdout) -> None:
    for step in plan:
        if isinstance(step, Retag):
            print(f"# {step.source} ({step.server})", file=out)
            for tag in step.tags:
                print(f"#   -> {tag}", file=out)
            print(shlex.join(retag_command(step)), file=out)
        else:
            print(f"# {step.name}: {', '.join(a for a, _ in step.archives)} ({step.server})", file=out)
            for tag in step.tags:
                print(f"#   -> {tag}", file=out)
            for cmd in podman_commands(step, "<digest-file>") + [hub_copy_command(step, "<digest>")]:
                print(shlex.join(cmd), file=out)


# --- pushing ------------------------------------------------------------------


def run(cmd: Sequence[str]) -> str:
    print("+ " + shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout


def digest_of(ref: str, runner: Callable = run) -> str:
    raw = runner(["docker", "buildx", "imagetools", "inspect", "--raw", ref])
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def check_source(step: Retag, runner: Callable = run) -> str:
    """The digest of step.source, once it holds the server and platforms it should."""
    out = runner(["docker", "buildx", "imagetools", "inspect", "--format", "{{json .Image}}", step.source])
    images = {k: v for k, v in json.loads(out).items() if k != "unknown/unknown"}
    want = set(PLATFORMS.values())
    if set(images) != want:
        raise PlanError(f"{step.source} holds {sorted(images)}, expected {sorted(want)}")
    for platform, image in sorted(images.items()):
        server = ((image.get("config") or {}).get("Labels") or {}).get(SERVER_LABEL)
        if server != step.server:
            raise PlanError(f"{step.source} {platform}: {SERVER_LABEL} is {server!r}, expected {step.server!r}")
    return digest_of(step.source, runner)


def check_archive(step: Assemble, arch: str, path: str, runner: Callable = run) -> None:
    config = json.loads(runner(["skopeo", "inspect", "--config", f"docker-archive:{path}"]))
    platform = f"{config.get('os')}/{config.get('architecture')}"
    if platform != PLATFORMS[arch]:
        raise PlanError(f"{path} is {platform}, expected {PLATFORMS[arch]}")
    server = ((config.get("config") or {}).get("Labels") or {}).get(SERVER_LABEL)
    if server != step.server:
        raise PlanError(f"{path}: {SERVER_LABEL} is {server!r}, expected {step.server!r}")


def expect_digest(tags: Iterable[str], digest: str, runner: Callable = run) -> None:
    for tag in tags:
        got = digest_of(tag, runner)
        if got != digest:
            raise PlanError(f"{tag} holds {got}, expected {digest}")


def execute(plan: Sequence, runner: Callable = run, workdir: Optional[Path] = None) -> None:
    # Check every source before pushing anything: a bad source stops the
    # whole publish, not the half after it.
    digests = {}
    for step in plan:
        if isinstance(step, Retag):
            digests[step] = check_source(step, runner)
        else:
            for arch, path in step.archives:
                check_archive(step, arch, path, runner)
    for step in plan:
        if isinstance(step, Retag):
            # From the digest just checked, not the tag, which could move.
            runner(retag_command(step, f"{repository(step.source)}@{digests[step]}"))
            expect_digest(step.tags, digests[step], runner)
        else:
            digest_file = (workdir or Path(".")) / f"publish-{step.name}.digest"
            for cmd in podman_commands(step, str(digest_file)):
                runner(cmd)
            digest = digest_file.read_text().strip()
            runner(hub_copy_command(step, digest))
            expect_digest(step.tags, digest, runner)


# --- command line -------------------------------------------------------------


def read_releases(path: str) -> list:
    """JSON objects, one per line: `gh api --paginate .../releases --jq
    '.[] | {tag_name, draft, prerelease}'`."""
    text = sys.stdin.read() if path == "-" else Path(path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="channel", required=True)

    p = sub.add_parser("edge", help="rolling tags from the queue's rc-mg-<sha> images")
    p.add_argument("--sha", required=True)

    p = sub.add_parser("release", help="release tags from the queue's rc-mg-<sha> images")
    p.add_argument("--version", required=True, help="X.Y.Z or X.Y.Z-pre, no leading v")
    p.add_argument("--sha", required=True)
    p.add_argument("--releases", required=True,
                   help="file of the repository's GitHub releases, one {tag_name, draft, prerelease} per line; - for stdin")

    p = sub.add_parser("rebuild", help="release-image.yml's tags from its rc-rebuild-<sha> images")
    p.add_argument("--tag", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--latest", action="store_true", help="also move latest*")

    p = sub.add_parser("per-os", help="per-OS falkordb-server images from their archives")
    p.add_argument("--version", required=True)
    p.add_argument("--images", help="directory holding the archives (required with --execute)")

    for p in sub.choices.values():
        p.add_argument("--execute", action="store_true", help="push; without it, only print the plan")
        p.add_argument("--json", action="store_true", help="print the plan as JSON")

    args = parser.parse_args(argv)
    try:
        if args.channel == "edge":
            plan = edge_plan(args.sha)
        elif args.channel == "release":
            plan = release_plan(args.version, args.sha, read_releases(args.releases))
        elif args.channel == "rebuild":
            plan = rebuild_plan(args.tag, args.sha, args.latest)
        else:
            if args.execute and not args.images:
                raise PlanError("--execute needs --images")
            archives = find_archives(Path(args.images)) if args.images else None
            plan = per_os_plan(args.version, archives)

        if args.json:
            print(json.dumps(plan_json(plan), indent=2))
        else:
            print_plan(plan)
        if args.execute:
            execute(plan, workdir=Path(args.images) if getattr(args, "images", None) else None)
            print(f"published {sum(len(s.tags) for s in plan)} tags")
    except PlanError as e:
        print(f"::error::{e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
