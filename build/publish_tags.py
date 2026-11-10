#!/usr/bin/env python3
"""Work out, and optionally push, the public image tags of one publish.

Every publishing workflow gets its tags from here, so the tags a workflow
pushes are the tags this file's tests check:

  edge     rust-push.yml     the queue's rc-mg-<sha>-<server> → the rolling tags
  release  release-tag.yml   the queue's rc-mg-<sha>-<server> → <version>[, latest]
  rebuild  release-image.yml rc-rebuild-<sha>-<run>-<server> → <tag>[, latest]
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
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

GHCR = "ghcr.io/falkordb"
DOCKER_HUB = "docker.io/falkordb"
# tag-plan.yml's rehearsal points both at local registries, as
# "<ghcr stand-in> <docker hub stand-in>"; nothing else sets it.
if os.environ.get("PUBLISH_TAGS_REHEARSAL_REGISTRIES"):
    GHCR, DOCKER_HUB = os.environ["PUBLISH_TAGS_REHEARSAL_REGISTRIES"].split()
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

# The registries refuse a tag longer than this. A version is judged against the
# longest suffix any channel adds to it (a per-OS tag's -<os>-<server>), so a
# release is refused before anything is pushed, never half published.
MAX_TAG_LENGTH = 128
LONGEST_VERSION_SUFFIX = max(
    [len(f"-{server}") for server in SERVERS]
    + [len(f"-{os_name}-{server}") for os_name, server in PER_OS]
)
MAX_VERSION_LENGTH = MAX_TAG_LENGTH - LONGEST_VERSION_SUFFIX

# Whole tags other workflows own; a rebuild may not publish over them. The C
# engine publishes the edge-c family from master.
RESERVED_TAGS = {"edge", "edge-rs", "edge-c", "latest"}
RESERVED_PREFIXES = ("rc-", "edge-c-")
# The C engine owns every version below 6.0.0.
FIRST_MAJOR = 6
# Parts of a tag that say what it holds. A version or an ad-hoc tag carrying
# one would collide with, or impersonate, another tag: 6.1.0-alpine's Valkey
# tag would be 6.1.0's Alpine tag. Compared lower-cased, after splitting on
# `.`, `-` and `_` (so x86_64 is refused through `x86`).
RESERVED_PARTS = (
    set(PLATFORMS)
    | {p.split("/")[1] for p in PLATFORMS.values()}
    | {"aarch64", "x86", "i386", "i686", "arm", "armv6", "armv7", "armv8", "armhf", "armel",
       "ppc64le", "s390x", "riscv64"}
)
# Server and OS names, with or without a version: redis8 reads as Redis 8,
# rhel8 as an OS, whatever the tag holds.
RESERVED_PART_RE = re.compile(
    r"(" + "|".join(sorted(set(SERVERS) | {"alpine", "rhel", "amazonlinux", "ubuntu", "debian"})) + r")\d*"
)

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[0-9]{1,20}$")
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
    m = VERSION_RE.fullmatch(text)
    if not m:
        return None
    return int(m[1]), int(m[2]), int(m[3]), m[4]


def check_parts(tag: str, what: str) -> None:
    # Case-blind, and split on `_` too: X64 and test_x64 name an architecture.
    for part in re.split(r"[._-]", tag.lower()):
        if part in RESERVED_PARTS or RESERVED_PART_RE.fullmatch(part):
            raise PlanError(
                f"{what} '{tag}' contains '{part}', which tags use to name a "
                "server, an OS or an architecture"
            )


def check_version(version: str) -> None:
    v = parse_version(version)
    if v is None:
        raise PlanError(f"version '{version}' is not X.Y.Z[-prerelease] (no leading v)")
    if len(version) > MAX_VERSION_LENGTH:
        raise PlanError(
            f"version '{version}' is {len(version)} characters; the longest is {MAX_VERSION_LENGTH}, "
            f"so its tags fit the registries' {MAX_TAG_LENGTH}"
        )
    if v[0] < FIRST_MAJOR:
        raise PlanError(f"version '{version}' is a C engine version (below {FIRST_MAJOR}.0.0)")
    if v[3] is not None:
        check_parts(v[3], "prerelease")


def check_sha(sha: str) -> None:
    if not SHA_RE.fullmatch(sha):
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


def rebuild_key(sha: str, run_id: Optional[str] = None) -> str:
    """The CI key release-image.yml stages a rebuild under: rebuild-<sha>,
    plus the workflow run's id, so two rebuilds of one commit (for different
    tags, which do not share a concurrency group) never share staging tags."""
    check_sha(sha)
    if run_id is None:
        return f"rebuild-{sha}"
    if not RUN_ID_RE.fullmatch(run_id):
        raise PlanError(f"run id '{run_id}' is not a workflow run id (digits only)")
    return f"rebuild-{sha}-{run_id}"


def rebuild_plan(tag: str, sha: str, latest: bool, releases: Optional[Iterable[dict]] = None,
                 run_id: Optional[str] = None) -> list:
    """release-image.yml's tags, from the images it rebuilt from source under
    rc-<rebuild_key(sha, run_id)>-<server>.

    latest* moves only where a release would move it: `tag` must be the
    highest stable version with a published GitHub release, in `releases`.
    """
    key = rebuild_key(sha, run_id)
    if parse_version(tag) is not None:
        check_version(tag)
    else:
        if not TAG_RE.fullmatch(tag):
            raise PlanError(f"tag '{tag}' is not a valid image tag")
        if re.match(r"v\d", tag, re.IGNORECASE):
            raise PlanError(f"tag '{tag}': image versions have no leading v")
        # Judged on what would be pushed: `rc` alone would publish rc-valkey.
        for name in server_tags(tag, DEFAULT_SERVER) + server_tags(tag, "redis"):
            # Case-blind: Latest or RC-x would read as the real channels.
            if name.lower() in RESERVED_TAGS or name.lower().startswith(RESERVED_PREFIXES):
                raise PlanError(f"tag '{tag}' would publish '{name}', which belongs to another workflow")
        check_parts(tag, "tag")
    if latest:
        if parse_version(tag) is None or is_prerelease(tag):
            raise PlanError(f"latest only moves to a stable X.Y.Z version; '{tag}' is not one")
        if releases is None or not moves_latest(tag, releases):
            raise PlanError(
                f"latest stays where it is: '{tag}' is not the highest stable version "
                "with a published GitHub release"
            )
    return _named_plan(tag, lambda repo, server: f"{GHCR}/{repo}:rc-{key}-{server}", latest)


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
        m = ARCHIVE_RE.fullmatch(path.name)
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
    """The digest of step.source, once it holds the server and platforms it should.

    The digest is read first and the checks run on it, so the digest that is
    published is the one that was checked, even if the tag moves meanwhile.
    """
    try:
        digest = digest_of(step.source, runner)
    except subprocess.CalledProcessError as e:
        hint = ""
        if ":rc-mg-" in step.source:
            hint = (
                ". The merge queue builds these only for commits after the switch to "
                "server-named tags, and cleanup-rc-images.yml sweeps them after its TTL. A "
                "commit from after the switch whose images were swept can be rebuilt from "
                "source with release-image.yml; a commit from before it has no Valkey image "
                "and cannot be published under these tags"
            )
        raise PlanError(f"cannot read {step.source}{hint}") from e
    pinned = f"{repository(step.source)}@{digest}"
    out = runner(["docker", "buildx", "imagetools", "inspect", "--format", "{{json .Image}}", pinned])
    images = {k: v for k, v in json.loads(out).items() if k != "unknown/unknown"}
    want = set(PLATFORMS.values())
    if set(images) != want:
        raise PlanError(f"{step.source} holds {sorted(images)}, expected {sorted(want)}")
    for platform, image in sorted(images.items()):
        server = ((image.get("config") or {}).get("Labels") or {}).get(SERVER_LABEL)
        if server != step.server:
            raise PlanError(f"{step.source} {platform}: {SERVER_LABEL} is {server!r}, expected {step.server!r}")
    return digest


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

    p = sub.add_parser("rebuild", help="release-image.yml's tags from its rc-rebuild-<sha>-<run> images")
    p.add_argument("--tag", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--run-id", required=True,
                   help="the release-image.yml run whose images to publish (github.run_id)")
    p.add_argument("--latest", action="store_true",
                   help="also move latest*; refused unless the tag is the highest stable version with a published release")
    p.add_argument("--releases", help="with --latest: the GitHub releases, as for `release`")

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
            releases = read_releases(args.releases) if args.releases else None
            plan = rebuild_plan(args.tag, args.sha, args.latest, releases, run_id=args.run_id)
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
    except subprocess.CalledProcessError as e:
        print(f"::error::{shlex.join(e.cmd)} failed with exit code {e.returncode}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
