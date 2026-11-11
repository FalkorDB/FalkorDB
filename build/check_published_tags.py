#!/usr/bin/env python3
"""Check that every published FalkorDB image tag holds what its name says.

For one target (edge, latest, a release version, or a CI key) this reads the
expected tags from build/published-tags.json and checks, on every registry the
channel publishes to:

  * each tag exists, and each multi-arch tag lists linux/amd64 and linux/arm64;
  * on each platform, the image holds the server its tag names: by the
    com.falkordb.server.name/.version labels, by the FALKORDB_SERVER and
    FALKORDB_SERVER_VERSION env that run.sh starts the server from, and by
    running `<server>-server --version` inside the image; and no other server
    name in the image answers as a different server;
  * /etc/os-release names the expected base OS;
  * a falkordb image carries the browser (BROWSER=1 and server.js under
    FALKORDB_BROWSER_PATH), and a falkordb-server image does not (BROWSER=0);
  * an alias (edge, edge-rs, X.Y.Z, latest) has the digest of the tag it stands
    for, a tag has the same digest on every registry, and a CI per-arch tag is,
    by its own digest, that platform's manifest of its multi-arch tag;
  * latest* point at the newest stable GitHub release, whatever was published
    (while that release has no images yet, they cannot be checked, and fail);
  * no tag exists where the contract says it must not: a per-OS tag on
    falkordb or without its server suffix, a CI tag on Docker Hub, or a public
    tag naming an architecture; the few left from before the switch are
    either gone or still hold the digest they had.

Every failure is reported, not just the first, and the exit status is 1 if
there is any. Needs docker with buildx, and gh unless --latest names a version.

    build/check_published_tags.py edge
    build/check_published_tags.py 6.1.0
    build/check_published_tags.py rc-pr-1234 --servers valkey
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_TAGS_FILE = Path(__file__).with_name("published-tags.json")
SERVER_NAME_LABEL = "com.falkordb.server.name"
SERVER_VERSION_LABEL = "com.falkordb.server.version"
SERVER_ENV = "FALKORDB_SERVER"
SERVER_VERSION_ENV = "FALKORDB_SERVER_VERSION"
BROWSER_ENV = "BROWSER"
BROWSER_PATH_ENV = "FALKORDB_BROWSER_PATH"

# `valkey-server --version` prints "Valkey server v=9.1.2 sha=...", and
# redis-server prints the same with "Redis".
VERSION_LINE = re.compile(r"^(?P<product>\S+) server v=(?P<version>\S+)", re.MULTILINE)
RELEASE_VERSION = re.compile(r"^v?(?P<core>\d+\.\d+\.\d+)(?P<pre>-[0-9A-Za-z.-]+)?$")
PIN_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
CI_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REGISTRY_WORKERS = 6


class NotFound(Exception):
    """The registry has no such tag."""


class InspectError(Exception):
    """The registry or docker failed for a reason other than a missing tag."""


# --------------------------------------------------------------------------
# Docker: the only part that talks to registries and runs containers.
# --------------------------------------------------------------------------


class Docker:
    def __init__(self, keep_images: bool = False, sleep: Callable[[float], None] = time.sleep) -> None:
        self.keep_images = keep_images
        self.sleep = sleep

    def _docker(self, args: list[str], timeout: int) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(args, 124, "", f"docker {args[0]} timed out after {timeout} s")

    @staticmethod
    def is_not_found(error: str) -> bool:
        """`imagetools inspect` ends with "<ref>: not found"; `docker pull`
        says "manifest unknown". Nothing else counts, so a credential helper's
        "executable file not found" is an error, not a missing tag."""
        lowered = error.strip().lower()
        return lowered.endswith(": not found") or "manifest unknown" in lowered

    def _registry(self, args: list[str], timeout: int) -> str:
        """Run a docker command that talks to a registry; return its stdout.
        Retries, waiting longer when the registry rate-limits us."""
        error = ""
        for attempt in range(4):
            proc = self._docker(args, timeout)
            if proc.returncode == 0:
                return proc.stdout
            error = (proc.stderr or proc.stdout).strip()
            if self.is_not_found(error):
                raise NotFound(error)
            if attempt < 3:
                rate_limited = "429" in error or "toomanyrequests" in error.lower()
                self.sleep(30 * (attempt + 1) if rate_limited else 2)
        raise InspectError(error)

    def _imagetools(self, ref: str, template: str) -> dict:
        out = self._registry(["buildx", "imagetools", "inspect", ref, "--format", template], 300)
        try:
            return json.loads(out)
        except json.JSONDecodeError as error:
            raise InspectError(f"unreadable `imagetools inspect` output for {ref}: {error}") from error

    def manifest(self, ref: str) -> dict:
        """The tag's manifest or index, as `imagetools inspect` reports it."""
        return self._imagetools(ref, "{{json .Manifest}}")

    def config(self, ref: str) -> dict:
        """The image config of one platform manifest (`ref` is repo@digest)."""
        return self._imagetools(ref, "{{json .Image}}")

    def has_image(self, ref: str) -> bool:
        return self._docker(["image", "inspect", ref], 60).returncode == 0

    def pull(self, ref: str, platform: str) -> None:
        self._registry(["pull", "--quiet", "--platform", platform, ref], 900)

    def run(self, ref: str, platform: str, entrypoint: str, args: list[str]) -> tuple[int, str]:
        proc = self._docker(
            ["run", "--rm", "--pull", "never", "--network", "none", "--platform", platform,
             "--entrypoint", entrypoint, ref, *args],
            300,
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()

    def remove(self, ref: str) -> None:
        if not self.keep_images:
            self._docker(["image", "rm", "--force", ref], 120)


# --------------------------------------------------------------------------
# What to check
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    kind: str  # "edge", "ci", "release" or "latest"
    version: str | None = None
    ci_key: str | None = None


@dataclass(frozen=True)
class Expect:
    server: str
    os: str
    browser: bool = False


@dataclass
class TagSpec:
    tag: str
    repos: list[str]
    expect: Expect
    alias_of: str | None = None
    follows: str | None = None  # the tag this one must have the digest of
    per_arch: str | None = None  # CI: the per-arch tag, with {arch} still in it
    without_server: str | None = None  # the tag with its server suffix removed


@dataclass
class Plan:
    title: str
    kind: str  # the Target kind: "edge", "ci", "release" or "latest"
    registries: list[str]  # where this channel publishes: "ghcr.io/falkordb", ...
    all_registries: list[str]
    repos: list[str]
    platforms: list[str]
    arch_suffixes: dict[str, str]
    specs: list[TagSpec]
    absent: dict[str, str] = field(default_factory=dict)  # ref -> why it must not exist
    frozen: dict[str, str] = field(default_factory=dict)  # ref -> the only digest it may hold
    # Where latest* must point when this target doesn't move them; checked by
    # digest only. Empty when the target moves latest (then the latest tags
    # are in `specs`) or doesn't concern it (CI).
    latest_specs: list[TagSpec] = field(default_factory=list)
    latest_registries: list[str] = field(default_factory=list)
    # The release tags of latest_version, every server and OS (the whole release
    # channel): any one existing means that release is from after the switch.
    latest_release_specs: list[TagSpec] = field(default_factory=list)
    latest_version: str | None = None
    latest_error: str | None = None


def parse_target(text: str, data: dict) -> Target:
    if text in ("edge", "latest"):
        return Target(text)
    if text.startswith("rc-"):
        ci_key = text[len("rc-"):]
        suffixes = [f"-{name}" for name in data["servers"]] + [f"-{s}" for s in data["arch_suffixes"].values()]
        if not CI_KEY.match(ci_key) or any(ci_key.endswith(suffix) for suffix in suffixes):
            raise ValueError(f"{text!r}: want rc-<ci_key>, without a server or arch suffix")
        return Target("ci", ci_key=ci_key)
    match = RELEASE_VERSION.match(text)
    if not match:
        raise ValueError(f"unknown target {text!r}: want edge, latest, X.Y.Z[-prerelease] or rc-<ci_key>")
    # release-tag.yml takes any prerelease suffix, so v6.1.0-rc1 and
    # v6.1.0-preview.1 are real releases; a trailing server or arch suffix is a
    # mistake for the tag that carries it, not a prerelease.
    suffixes = [f"-{name}" for name in data["servers"]] + [f"-{s}" for s in data["arch_suffixes"].values()]
    if any(text.endswith(suffix) for suffix in suffixes):
        raise ValueError(f"unknown target {text!r}: a release version takes no server or arch suffix")
    return Target("release", version=text.lstrip("v"))


def _fill(text: str, **values: str) -> str:
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def _channel_specs(channel: dict, servers: set[str], **values: str) -> list[TagSpec]:
    specs: list[TagSpec] = []
    by_tag: dict[str, TagSpec] = {}
    for image in channel["images"]:
        if image["server"] not in servers:
            continue
        tag = _fill(image["tag"], **values)
        suffix = f"-{image['server']}"
        spec = TagSpec(
            tag=tag,
            repos=list(image["repos"]),
            expect=Expect(image["server"], image["os"]),
            follows=_fill(image["follows"], **values) if "follows" in image else None,
            per_arch=_fill(image["per_arch"], **values) if "per_arch" in image else None,
            without_server=tag[: -len(suffix)] if tag.endswith(suffix) else None,
        )
        specs.append(spec)
        by_tag[spec.tag] = spec
    for alias in channel.get("aliases", []):
        target = by_tag.get(_fill(alias["of"], **values))
        if target is None:
            continue
        specs.append(
            TagSpec(
                tag=_fill(alias["tag"], **values),
                repos=target.repos,
                expect=target.expect,
                alias_of=target.tag,
                follows=_fill(alias["follows"], **values) if "follows" in alias else None,
            )
        )
    return specs


def _absent_refs(plan: Plan, public: bool, left_over: set[str], fresh_ci_key: bool = False) -> dict[str, str]:
    """Where the plan's tags must not exist: on repos and registries not
    listed for them; for public tags, under an architecture suffix (bar the
    tags left from before the switch); a per-OS tag without its server
    suffix; and CI tags without their server suffix, anywhere but the CI
    registries, and there too for a key that cannot predate the switch.
    Every name the plan expects is taken back out."""
    def tags_of(spec: TagSpec) -> list[str]:
        per_arch = [_fill(spec.per_arch, arch=s) for s in plan.arch_suffixes.values()] if spec.per_arch else []
        return [spec.tag, *per_arch]

    expected = {f"{reg}/{repo}:{tag}" for spec in plan.specs for tag in tags_of(spec)
                for reg in plan.registries for repo in spec.repos}
    expected |= {f"{reg}/{repo}:{spec.tag}" for spec in plan.latest_specs
                 for reg in plan.latest_registries for repo in spec.repos}
    absent: dict[str, str] = {}

    def forbid(name: str, why: str, registries: list[str] | None = None) -> None:
        for registry in plan.all_registries if registries is None else registries:
            for repo in plan.repos:
                ref = f"{registry}/{repo}:{name}"
                if ref not in left_over:
                    absent.setdefault(ref, why)

    for spec in plan.specs:
        for tag in tags_of(spec):
            for registry in plan.all_registries:
                for repo in plan.repos:
                    ref = f"{registry}/{repo}:{tag}"
                    if ref not in expected:
                        absent[ref] = (f"{tag} belongs on {', '.join(spec.repos)} "
                                       f"in {', '.join(plan.registries)} only")
    if not public:
        # A pull-request key may keep names from before the switch on the CI
        # registries; no CI name was ever published anywhere else.
        elsewhere = [registry for registry in plan.all_registries if registry not in plan.registries]
        for spec in plan.specs:
            if not spec.without_server:
                continue
            unsuffixed = [spec.without_server, *(f"{spec.without_server}-{s}" for s in plan.arch_suffixes.values())]
            for name in unsuffixed:
                if fresh_ci_key:
                    forbid(name, f"{spec.tag} has no name without its server")
                else:
                    forbid(name, f"CI tags belong in {', '.join(plan.registries)} only", elsewhere)
    if public:
        for spec in plan.specs + plan.latest_specs:
            if spec.without_server:
                # edge-rs, latest and X.Y.Z are aliases and come back out
                # below; X.Y.Z-rhel9 and the like must not exist.
                forbid(spec.without_server, f"{spec.tag} has no name without its server")
            for name in filter(None, (spec.tag, spec.without_server)):
                for suffix in plan.arch_suffixes.values():
                    forbid(f"{name}-{suffix}", "no public tag names an architecture")
    for ref in expected:
        absent.pop(ref, None)
    return absent


def make_plan(
    data: dict,
    target: Target,
    *,
    servers: set[str],
    registries: dict[str, str],
    latest_version: str | None = None,
    latest_error: str | None = None,
) -> Plan:
    """The tags to check for a target. `latest_version` is the release latest*
    must point at (None for CI targets, or when it could not be worked out:
    then `latest_error` says why)."""
    channels = data["channels"]
    left_over = {
        f"{registries[name]}/{repo_tag}": digest
        for name, tags in data.get("left_from_before_the_switch", {}).items() if not name.startswith("_")
        for repo_tag, digest in tags.items()
    }
    if target.kind != "ci" and latest_version is None and latest_error is None:
        raise ValueError("a public target needs the release latest* points at, or why it is unknown")

    def plan_for(title: str, channel: dict, specs: list[TagSpec]) -> Plan:
        return Plan(
            title=title,
            kind=target.kind,
            registries=[registries[name] for name in channel["registries"]],
            all_registries=list(registries.values()),
            repos=list(data["repos"]),
            platforms=list(data["platforms"]),
            arch_suffixes=dict(data["arch_suffixes"]),
            specs=specs,
        )

    if target.kind == "ci":
        channel = channels["ci"]
        plan = plan_for(f"CI images rc-{target.ci_key}", channel,
                        _channel_specs(channel, servers, ci_key=target.ci_key or ""))
        # A merge-queue key is new for every entry; a pull-request key is
        # reused and may hold names from before the switch (see the data).
        fresh = any((target.ci_key or "").startswith(p) for p in channel.get("fresh_key_prefixes", []))
        plan.absent = _absent_refs(plan, False, set(left_over), fresh_ci_key=fresh)
        # Servers left out of the check still have no CI names off the CI
        # registries; on them, a left-out server's images may well exist.
        elsewhere = [registry for registry in plan.all_registries if registry not in plan.registries]
        for spec in _channel_specs(channel, set(data["servers"]) - servers, ci_key=target.ci_key or ""):
            per_arch = [_fill(spec.per_arch, arch=a) for a in plan.arch_suffixes.values()] if spec.per_arch else []
            names = [spec.tag, *per_arch]
            for name in names:
                for registry in elsewhere:
                    for repo in plan.repos:
                        plan.absent.setdefault(f"{registry}/{repo}:{name}",
                                               f"CI tags belong in {', '.join(plan.registries)} only")
        return plan

    version = target.version
    if target.kind == "latest":
        if latest_version is None:
            raise ValueError(f"cannot tell which release latest is: {latest_error}")
        version = latest_version

    if target.kind == "edge":
        channel = channels["edge"]
        plan = plan_for("edge", channel, _channel_specs(channel, servers))
    else:
        channel = channels["release"]
        plan = plan_for(f"release {version}", channel, _channel_specs(channel, servers, version=version or ""))

    latest = channels["latest"]
    plan.latest_registries = [registries[name] for name in latest["registries"]]
    plan.latest_version = latest_version
    plan.latest_error = latest_error
    if latest_version is not None and latest_version == version:
        # This release moves latest: check the latest tags in full.
        plan.specs += _channel_specs(latest, servers, version=version)
    elif latest_version is not None:
        plan.latest_specs = _channel_specs(latest, servers, version=latest_version)
        plan.latest_release_specs = _channel_specs(channels["release"], set(data["servers"]),
                                                   version=latest_version)
    plan.absent = _absent_refs(plan, True, set(left_over))
    plan.frozen = left_over
    return plan


def newest_stable(releases: list[dict]) -> str | None:
    """The highest X.Y.Z among published, non-prerelease GitHub releases."""
    best: tuple[int, ...] | None = None
    for release in releases:
        if release.get("draft") or release.get("prerelease"):
            continue
        match = RELEASE_VERSION.match(release.get("tag_name", ""))
        if match and not match.group("pre"):
            core = tuple(int(part) for part in match.group("core").split("."))
            best = max(best, core) if best else core
    return ".".join(map(str, best)) if best else None


def latest_rule(releases: list[dict], candidate: str | None, repo: str) -> tuple[str | None, str]:
    """Which release latest* must point at, and why: the highest stable
    version with a published (non-draft, non-prerelease) GitHub release, as
    release-tag.yml decides it. `candidate`, the release being checked, moves
    latest only if it is that release; without a published GitHub release (a
    draft, which a read-only token cannot even see, or none yet) it does not."""
    reason = ""
    if candidate:
        match = RELEASE_VERSION.match(candidate)
        on_github = [r for r in releases if r.get("tag_name", "").lstrip("v") == candidate and not r.get("draft")]
        if match and match.group("pre"):
            reason = f"{candidate} is a prerelease"
        elif not on_github:
            reason = f"{candidate} has no published GitHub release, so it does not move latest"
        elif all(r.get("prerelease") for r in on_github):
            reason = f"the GitHub release of {candidate} is marked prerelease"
    newest = newest_stable(releases)
    if newest is None:
        return None, f"no stable release of {repo} on GitHub"
    note = f"latest* must point at {newest}, the newest stable release of {repo}"
    return newest, f"{note} ({reason})" if reason else note


def github_releases(repo: str) -> list[dict]:
    try:
        proc = subprocess.run(
            ["gh", "api", f"repos/{repo}/releases", "--paginate",
             "--jq", ".[] | {tag_name, draft, prerelease}"],
            capture_output=True, text=True, timeout=120, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise InspectError(f"gh api repos/{repo}/releases: {error}") from error
    if proc.returncode != 0:
        raise InspectError(f"gh api repos/{repo}/releases: {proc.stderr.strip()}")
    try:
        return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise InspectError(f"gh api repos/{repo}/releases: unreadable output: {error}") from error


# --------------------------------------------------------------------------
# Checking
# --------------------------------------------------------------------------

TAGS, IMAGES, RELATIONS = "tags", "inside the images", "between tags"
SECTIONS = (TAGS, IMAGES, RELATIONS)


@dataclass
class Result:
    ok: bool
    section: str
    subject: str
    check: str
    detail: str


@dataclass
class Manifest:
    digest: str
    is_index: bool
    platforms: dict[str, str]  # "linux/amd64" -> that platform's manifest digest
    duplicates: list[str] = field(default_factory=list)  # platforms listed more than once


@dataclass
class Facts:
    """What one platform image holds; gathered once per digest and platform."""

    config_error: str | None = None
    run_error: str | None = None
    platform: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    os_release: dict[str, str] = field(default_factory=dict)
    os_release_error: str | None = None
    version_runs: dict[str, tuple[int, str]] = field(default_factory=dict)
    browser_files: dict[str, tuple[int, str]] = field(default_factory=dict)  # path -> `test -f` result


def parse_os_release(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            values[key] = value.strip().strip("\"'")
    return values


def describe_os(values: dict[str, str]) -> str:
    keys = ("ID", "VERSION_ID", "VERSION_CODENAME")
    return " ".join(f"{key}={values[key]}" for key in keys if key in values) or "(empty)"


def _describe_missing(result: Manifest | Exception) -> str:
    return "not found" if isinstance(result, NotFound) else f"could not inspect: {result}"


class Checker:
    def __init__(self, data: dict, docker: Docker, pins: dict[str, str]) -> None:
        self.data = data
        self.docker = docker
        self.pins = pins
        self.results: list[Result] = []
        self._release_target = False
        self._manifests: dict[str, Manifest | Exception] = {}
        self._digests: dict[str, str | Exception] = {}
        # (platform digest, platform the index lists it as) ->
        #     (ref to pull it by, [(registry, repo:tag, expect)])
        self._content: dict[tuple[str, str], tuple[str, list[tuple[str, str, Expect]]]] = {}

    # -- recording ---------------------------------------------------------

    def record(self, ok: bool, section: str, subject: str, check: str, detail: str) -> None:
        self.results.append(Result(ok, section, subject, check, detail))

    @property
    def failures(self) -> list[Result]:
        return [result for result in self.results if not result.ok]

    # -- registry ----------------------------------------------------------

    def manifest(self, ref: str) -> Manifest | Exception:
        if ref not in self._manifests:
            self._manifests[ref] = self._load_manifest(ref)
        return self._manifests[ref]

    def prefetch(self, refs: list[str], digest_only: list[str]) -> None:
        """Inspect many tags at once; the checks then read the caches. For
        `digest_only` refs (names that must be absent or unchanged) only the
        digest is read, one registry call each."""
        todo = [ref for ref in dict.fromkeys(refs) if ref not in self._manifests]
        todo_digests = [ref for ref in dict.fromkeys(digest_only) if ref not in self._digests]
        with ThreadPoolExecutor(REGISTRY_WORKERS) as pool:
            for ref, manifest in zip(todo, pool.map(self._load_manifest, todo), strict=True):
                self._manifests[ref] = manifest
            for ref, digest in zip(todo_digests, pool.map(self._load_digest, todo_digests), strict=True):
                self._digests[ref] = digest

    def digest_of(self, ref: str) -> str | Exception:
        """The tag's digest, or why it has none (NotFound when it is absent)."""
        if ref not in self._digests:
            self._digests[ref] = self._load_digest(ref)
        return self._digests[ref]

    def _load_digest(self, ref: str) -> str | Exception:
        try:
            return self.docker.manifest(ref)["digest"]
        except (NotFound, InspectError) as error:
            return error

    def _load_manifest(self, ref: str) -> Manifest | Exception:
        try:
            raw = self.docker.manifest(ref)
        except (NotFound, InspectError) as error:
            return error
        repo = ref.rsplit(":", 1)[0]
        if "manifests" in raw:
            platforms: dict[str, str] = {}
            duplicates: list[str] = []
            for entry in raw["manifests"]:
                platform = entry.get("platform") or {}
                annotations = entry.get("annotations") or {}
                if annotations.get("vnd.docker.reference.type") == "attestation-manifest":
                    continue
                # Any other entry counts, a missing or unknown platform included:
                # it is listed under that name and fails the exact-platform check
                # instead of slipping by as an attestation.
                name = f"{platform.get('os') or '?'}/{platform.get('architecture') or '?'}"
                if name in platforms:
                    duplicates.append(name)
                else:
                    platforms[name] = entry["digest"]  # the first, as a pull takes
            return Manifest(raw["digest"], True, platforms, duplicates)
        try:
            config = self.docker.config(f"{repo}@{raw['digest']}")
        except (NotFound, InspectError) as error:
            # The tag exists; only its config is unreadable. Never "absent".
            return InspectError(f"{ref} exists ({raw['digest']}), but its config cannot be read: {error}")
        platform = f"{config.get('os')}/{config.get('architecture')}"
        return Manifest(raw["digest"], False, {platform: raw["digest"]})

    def digest(self, ref: str) -> str | None:
        manifest = self.manifest(ref)
        return manifest.digest if isinstance(manifest, Manifest) else None

    # -- the checks ----------------------------------------------------------

    def check(self, plan: Plan) -> None:
        self._release_target = plan.kind in ("release", "latest")
        self.prefetch(self._refs(plan), list(plan.absent) + list(plan.frozen))
        for spec in plan.specs:
            for registry in plan.registries:
                for repo in spec.repos:
                    self._check_tag(plan, registry, repo, spec)
        self._check_content()
        self._check_same_digest_everywhere(plan)
        self._check_latest(plan)
        self._check_absent(plan)
        self._check_frozen(plan)

    @staticmethod
    def _refs(plan: Plan) -> list[str]:
        refs = []
        for spec in plan.specs:
            for registry in plan.registries:
                for repo in spec.repos:
                    refs.append(f"{registry}/{repo}:{spec.tag}")
                    for other in (spec.alias_of, spec.follows):
                        if other:
                            refs.append(f"{registry}/{repo}:{other}")
                    if spec.per_arch:
                        refs += [f"{registry}/{repo}:{_fill(spec.per_arch, arch=suffix)}"
                                 for suffix in plan.arch_suffixes.values()]
        for spec in plan.latest_specs:
            for registry in plan.latest_registries:
                for repo in spec.repos:
                    refs += [f"{registry}/{repo}:{spec.tag}", f"{registry}/{repo}:{spec.follows}"]
        return refs

    def _check_tag(self, plan: Plan, registry: str, repo: str, spec: TagSpec) -> None:
        ref = f"{registry}/{repo}:{spec.tag}"
        manifest = self.manifest(ref)
        if not isinstance(manifest, Manifest):
            self.record(False, TAGS, ref, "exists", _describe_missing(manifest))
            if spec.per_arch:
                self._check_per_arch(plan, registry, repo, spec, None)
            return
        self.record(True, TAGS, ref, "exists", manifest.digest)

        listed = sorted(manifest.platforms)
        problems = []
        if missing := [p for p in plan.platforms if p not in manifest.platforms]:
            problems.append(f"missing {', '.join(missing)}")
        if extra := [p for p in listed if p not in plan.platforms]:
            problems.append(f"also lists {', '.join(extra)}")
        if manifest.duplicates:
            problems.append(f"lists {', '.join(manifest.duplicates)} more than once")
        if not manifest.is_index:
            self.record(False, TAGS, ref, "architectures",
                        f"a single-platform manifest ({', '.join(listed)}), "
                        f"not a multi-arch index of {', '.join(plan.platforms)}")
        elif problems:
            self.record(False, TAGS, ref, "architectures",
                        f"lists {', '.join(listed) or 'nothing'}; {'; '.join(problems)}")
        else:
            self.record(True, TAGS, ref, "architectures", ", ".join(plan.platforms))

        for platform in plan.platforms:
            if platform in manifest.platforms:
                self._want_content(registry, repo, spec.tag, manifest.platforms[platform],
                                   platform, spec.expect)

        if spec.alias_of:
            self._check_same_digest(ref, manifest, f"{registry}/{repo}:{spec.alias_of}", "alias of")
        if spec.follows:
            self._check_same_digest(ref, manifest, f"{registry}/{repo}:{spec.follows}", "follows")
        if spec.per_arch:
            self._check_per_arch(plan, registry, repo, spec, manifest)

    def _check_same_digest(self, ref: str, manifest: Manifest, other_ref: str, relation: str) -> None:
        other = self.manifest(other_ref)
        if not isinstance(other, Manifest):
            self.record(False, RELATIONS, ref, relation,
                        f"{other_ref} cannot be compared: {_describe_missing(other)}")
        elif other.digest != manifest.digest:
            self.record(False, RELATIONS, ref, relation,
                        f"digest {manifest.digest} differs from {other_ref} {other.digest}")
        else:
            self.record(True, RELATIONS, ref, relation, f"{other_ref} (same digest)")

    def _check_per_arch(self, plan: Plan, registry: str, repo: str, spec: TagSpec,
                        index: Manifest | None) -> None:
        """Each per-arch tag is that platform's manifest of the multi-arch tag
        itself, not an index around it (when the multi-arch tag exists; its
        absence is its own failure)."""
        for platform in plan.platforms:
            arch_tag = _fill(spec.per_arch or "", arch=plan.arch_suffixes[platform])
            ref = f"{registry}/{repo}:{arch_tag}"
            manifest = self.manifest(ref)
            if not isinstance(manifest, Manifest):
                self.record(False, TAGS, ref, "exists", _describe_missing(manifest))
                continue
            self.record(True, TAGS, ref, "exists", manifest.digest)
            if list(manifest.platforms) != [platform] or manifest.duplicates:
                listed = ", ".join(list(manifest.platforms) + manifest.duplicates) or "nothing"
                self.record(False, TAGS, ref, "architectures", f"holds {listed}, expected only {platform}")
                continue
            self.record(True, TAGS, ref, "architectures", platform)
            self._want_content(registry, repo, arch_tag, manifest.platforms[platform], platform, spec.expect)
            if index is None:
                continue
            expected = index.platforms.get(platform)
            if manifest.digest == expected:
                detail = "same digest"
            elif manifest.is_index and manifest.platforms[platform] == expected:
                detail = (f"an index ({manifest.digest}) around the multi-arch tag's {platform} manifest, "
                          f"not that manifest")
            else:
                kind = "an index" if manifest.is_index else "a manifest"
                detail = (f"{kind} with digest {manifest.digest}, not the multi-arch tag's {platform} manifest "
                          f"{expected}" if expected else f"the multi-arch tag lists no {platform} to compare with")
            self.record(manifest.digest == expected, RELATIONS, ref, f"{platform} of {spec.tag}", detail)

    def _check_same_digest_everywhere(self, plan: Plan) -> None:
        """Every tag this target checks holds one digest on every registry it is
        published to. latest* is checked here too when the target does not own
        it: edge and older-release runs have it in `latest_specs`, on
        `latest_registries`."""
        groups = [(plan.specs, plan.registries), (plan.latest_specs, plan.latest_registries)]
        seen = set()
        for specs, registries in groups:
            if len(registries) < 2:
                continue
            for spec in specs:
                for repo in spec.repos:
                    if (repo, spec.tag) in seen:
                        continue
                    seen.add((repo, spec.tag))
                    found = {}
                    for registry in registries:
                        digest = self.digest(f"{registry}/{repo}:{spec.tag}")
                        if digest:
                            found[registry] = digest
                    if len(found) < 2:
                        continue  # a missing tag is already a failure of its own
                    listing = ", ".join(f"{registry} {digest}" for registry, digest in found.items())
                    self.record(len(set(found.values())) == 1, RELATIONS, f"{repo}:{spec.tag}",
                                "same digest on every registry", listing)

    def _check_latest(self, plan: Plan) -> None:
        """latest* must point at the newest stable release, whatever this
        target published. A target that moves latest checks them in full
        through `follows` instead."""
        if plan.latest_error:
            self.record(False, RELATIONS, "latest*", "points at the newest stable release", plan.latest_error)
            return
        version = plan.latest_version or ""
        repos = sorted({repo for spec in plan.latest_specs for repo in spec.repos})
        # A release with a server-suffixed tag anywhere is from after the
        # switch, and has all of them everywhere; one from before has only
        # the bare version. Decided once, so a partial push can't pass here
        # as "from before the switch" because it reached another registry.
        suffixed = [self.manifest(f"{registry}/{repo}:{spec.follows}")
                    for registry in plan.latest_registries for spec in plan.latest_specs if not spec.alias_of
                    for repo in spec.repos]
        # The per-OS tags (alpine, amazonlinux2023, rhel9) count too, so a
        # release with only those is not taken for one from before the switch.
        suffixed += [self.manifest(f"{registry}/{repo}:{spec.tag}")
                     for registry in plan.latest_registries for spec in plan.latest_release_specs
                     if not spec.alias_of for repo in spec.repos]
        after_switch: bool | None = any(isinstance(result, Manifest) for result in suffixed)
        if not after_switch and any(not isinstance(result, (Manifest, NotFound)) for result in suffixed):
            after_switch = None  # one could not be read: can't tell
        for registry in plan.latest_registries:
            for repo in repos:
                specs = [spec for spec in plan.latest_specs if repo in spec.repos]
                self._check_latest_in(registry, repo, version, specs, after_switch)

    def _check_latest_in(self, registry: str, repo: str, version: str, specs: list[TagSpec],
                         after_switch: bool | None) -> None:
        subject = f"{registry}/{repo}:latest*"
        check = f"points at {version}"
        releases = {spec.follows: self.manifest(f"{registry}/{repo}:{spec.follows}") for spec in specs}
        for tag, result in releases.items():
            if isinstance(result, Exception) and not isinstance(result, NotFound):
                self.record(False, RELATIONS, subject, check, f"could not inspect {registry}/{repo}:{tag}: {result}")
                return
        if all(isinstance(result, NotFound) for result in releases.values()):
            # Its GitHub release is published before its images: in that
            # window, or if its images never came, latest* cannot be checked.
            self.record(False, RELATIONS, subject, check,
                        f"{version} has no images here yet, so latest* cannot be checked; "
                        f"re-run this check once {version}'s image jobs have finished")
            return
        for ok, ref, verdict_check, detail in self._latest_verdicts(registry, repo, version, specs, releases,
                                                                    after_switch):
            self.record(ok, RELATIONS, ref, verdict_check, detail)

    def _latest_verdicts(self, registry: str, repo: str, version: str, specs: list[TagSpec],
                         releases: dict[str, Manifest | Exception],
                         after_switch: bool | None) -> list[tuple[bool, str, str, str]]:
        """Each latest* tag against `version`'s tag, and after the switch the
        bare latest against `version`'s -valkey tag too (latest stands for
        latest-valkey), as (ok, ref, check, detail)."""
        verdicts = []
        for spec in specs:
            ref = f"{registry}/{repo}:{spec.tag}"
            other_ref = f"{registry}/{repo}:{spec.follows}"
            latest = self.manifest(ref)
            if isinstance(latest, Exception) and not isinstance(latest, NotFound):
                ok, detail = False, f"could not inspect {ref}: {latest}"
            else:
                ok, detail = _judge_latest(latest, other_ref, releases[spec.follows], version, after_switch)
            verdicts.append((ok, ref, f"points at {version}", detail))
            # After the switch the bare alias also stands for its server tag
            # (latest = latest-valkey, so N = N-valkey); before it, N-valkey
            # doesn't exist and there is nothing to compare. Only judged when
            # latest = N held, so a single mistake reports once: this fails
            # only when N and N-valkey differ.
            target = next((s for s in specs if s.tag == spec.alias_of), None)
            if ok and spec.alias_of and target and isinstance(latest, Manifest):
                server_ref = f"{registry}/{repo}:{target.follows}"
                server = releases.get(target.follows)
                if isinstance(server, Manifest):
                    same = latest.digest == server.digest
                    verdicts.append((same, ref, f"same image as {target.follows}",
                                     f"{server_ref} (same digest)" if same
                                     else f"digest {latest.digest} is {other_ref}'s, not {server_ref}'s "
                                          f"{server.digest}: {spec.follows} and {target.follows} differ"))
        return verdicts

    def _check_absent(self, plan: Plan) -> None:
        present = 0
        for ref, why in plan.absent.items():
            result = self.digest_of(ref)
            if isinstance(result, NotFound):
                continue
            present += 1
            detail = (f"exists ({result}); {why}" if isinstance(result, str)
                      else f"could not inspect: {result}")
            self.record(False, TAGS, ref, "must not exist", detail)
        if not present:
            self.record(True, TAGS, plan.title, "must not exist",
                        f"none of the {len(plan.absent)} forbidden names exists")

    def _check_frozen(self, plan: Plan) -> None:
        """Tags from before the switch are never touched: gone, or as they were."""
        for ref, digest in plan.frozen.items():
            result = self.digest_of(ref)
            if isinstance(result, NotFound):
                self.record(True, TAGS, ref, "left from before the switch", "deleted")
            elif not isinstance(result, str):
                self.record(False, TAGS, ref, "left from before the switch", f"could not inspect: {result}")
            elif result != digest:
                self.record(False, TAGS, ref, "left from before the switch",
                            f"now {result}, was {digest}; a tag from before the switch is never touched")
            else:
                self.record(True, TAGS, ref, "left from before the switch", "unchanged")

    # -- inside the image ------------------------------------------------

    def _want_content(self, registry: str, repo: str, tag: str, digest: str, platform: str,
                      expect: Expect) -> None:
        expect = replace(expect, browser=bool(self.data["repos"][repo]["browser"]))
        entry = self._content.setdefault((digest, platform), (f"{registry}/{repo}@{digest}", []))
        entry[1].append((registry, f"{repo}:{tag}", expect))

    def _check_content(self) -> None:
        """Look inside each platform image once, however many tags name it,
        and judge it once per expectation, naming every tag that holds it."""
        for (_, platform), (ref, wanted) in self._content.items():
            facts = self._gather(ref, platform)
            by_expect: dict[Expect, list[tuple[str, str]]] = {}
            for registry, repo_tag, expect in wanted:
                by_expect.setdefault(expect, []).append((registry, repo_tag))
            for expect, names in by_expect.items():
                self._judge(f"{_name_tags(names)} [{platform}]", platform, expect, facts)

    def _gather(self, ref: str, platform: str) -> Facts:
        facts = Facts()
        try:
            config = self.docker.config(ref)
        except (NotFound, InspectError) as error:
            facts.config_error = f"could not read the image config: {error}"
        else:
            facts.platform = f"{config.get('os')}/{config.get('architecture')}"
            inner = config.get("config") or {}
            facts.labels = inner.get("Labels") or {}
            facts.env = dict(item.partition("=")[::2] for item in inner.get("Env") or [])

        had_image = self.docker.has_image(ref)
        try:
            self.docker.pull(ref, platform)
        except (NotFound, InspectError) as error:
            facts.run_error = f"could not pull {platform}: {error}"
            return facts
        try:
            code, out = self.docker.run(ref, platform, "cat", ["/etc/os-release"])
            if code == 0:
                facts.os_release = parse_os_release(out)
            else:
                facts.os_release_error = out or f"exit {code}"
            # Every known server name, not only the expected one: a wrong
            # image then says what it holds, and a second server is caught.
            for server in sorted(self.data["servers"]):
                facts.version_runs[server] = self.docker.run(
                    ref, platform, f"{server}-server", ["--version"])
            # Where a browser would be: its configured path, and the usual one.
            browser = self.data["browser"]
            paths = {facts.env.get(BROWSER_PATH_ENV) or "", browser["default_path"]} - {""}
            for path in sorted(paths):
                entry = f"{path.rstrip('/')}/{browser['entry']}"
                facts.browser_files[entry] = self.docker.run(ref, platform, "sh", ["-c", 'test -f "$1"', "sh", entry])
        finally:
            if not had_image:
                self.docker.remove(ref)
        return facts

    def _judge(self, subject: str, platform: str, expect: Expect, facts: Facts) -> None:
        server = expect.server
        pin = self.pins[server]
        product = self.data["servers"][server]["product"]

        if facts.config_error:
            self.record(False, IMAGES, subject, "image config", facts.config_error)
        else:
            self.record(facts.platform == platform, IMAGES, subject, "config platform",
                        facts.platform if facts.platform == platform
                        else f"config says {facts.platform}, the index says {platform}")
            for kind, values, pairs in (
                ("label", facts.labels, ((SERVER_NAME_LABEL, server), (SERVER_VERSION_LABEL, pin))),
                ("env", facts.env, ((SERVER_ENV, server), (SERVER_VERSION_ENV, pin))),
            ):
                (name_key, _), _ = pairs
                for key, want in pairs:
                    have = values.get(key)
                    # Only a version of the right server can be a stale pin.
                    hint = self._pins_hint(server, have) if want == pin and values.get(name_key) == server else ""
                    self.record(have == want, IMAGES, subject, f"{kind} {key}",
                                have if have == want else f"{_show(have)}, expected {want}{hint}")
            want_browser = "1" if expect.browser else "0"
            have = facts.env.get(BROWSER_ENV)
            self.record(have == want_browser, IMAGES, subject, f"env {BROWSER_ENV}",
                        have if have == want_browser else f"{_show(have)}, expected {want_browser}")
            if expect.browser:
                path = facts.env.get(BROWSER_PATH_ENV)
                self.record(bool(path), IMAGES, subject, f"env {BROWSER_PATH_ENV}",
                            path or f"{_show(path)}, expected the browser's path")

        if facts.run_error:
            self.record(False, IMAGES, subject, "run the image", facts.run_error)
            return

        code, out = facts.version_runs.get(server, (127, "not run"))
        match = VERSION_LINE.search(out) if code == 0 else None
        check = f"{server}-server --version"
        if match and match.group("product") == product and match.group("version") == pin:
            self.record(True, IMAGES, subject, check, match.group(0))
        else:
            got = match.group(0) if match else f"exit {code}: {_short_error(out)}"
            same_product = match is not None and match.group("product") == product
            hint = self._pins_hint(server, match.group("version")) if same_product else ""
            self.record(False, IMAGES, subject, check,
                        f"{got}; expected {product} server v={pin}{_holds(facts, server)}{hint}")

        # Valkey images ship redis-* names that answer as Valkey; that is fine.
        # A name answering as another product means a second server inside.
        others = []
        for name, (other_code, other_out) in sorted(facts.version_runs.items()):
            other_match = VERSION_LINE.search(other_out) if other_code == 0 else None
            if name != server and other_match and other_match.group("product") != product:
                others.append(f"{name}-server is {other_match.group(0)}")
        self.record(not others, IMAGES, subject, "no other server",
                    "; ".join(others) or f"every server name answers as {product} or is absent")

        self._judge_browser(subject, expect, facts)

        expected_os = self.data["base_os"][expect.os]
        if facts.os_release_error is not None:
            self.record(False, IMAGES, subject, "/etc/os-release",
                        f"could not read it: {_short_error(facts.os_release_error)}")
        else:
            ok = all(fnmatch.fnmatchcase(facts.os_release.get(key, ""), want)
                     for key, want in expected_os.items())
            detail = describe_os(facts.os_release)
            if not ok:
                detail += f"; expected {expect.os} ({describe_os(expected_os)})"
            self.record(ok, IMAGES, subject, "/etc/os-release", detail)


    def _pins_hint(self, server: str, have: str | None) -> str:
        """A release re-run after main moved its pins fails here; say how to
        check it by hand."""
        if not (self._release_target and have and PIN_VERSION.match(have)):
            return ""
        if _version_key(have) >= _version_key(self.pins[server]):
            return ""  # an older release cannot hold a newer server
        return (f" (if this is a re-run of an older release, check it by hand with the "
                f"workflow's pins input {server}={have}, or --pin {server}={have})")

    def _judge_browser(self, subject: str, expect: Expect, facts: Facts) -> None:
        """falkordb runs the browser from $FALKORDB_BROWSER_PATH/server.js;
        falkordb-server has none, at that path or the usual one."""
        if facts.config_error:
            return  # its env is unknown; the unreadable config is already a failure
        found = {path for path, (code, _) in facts.browser_files.items() if code == 0}
        errors = {path: out for path, (code, out) in facts.browser_files.items() if code not in (0, 1)}
        if errors:
            path, out = next(iter(errors.items()))
            self.record(False, IMAGES, subject, "browser", f"could not look for {path}: {_short_error(out)}")
        elif expect.browser:
            configured = facts.env.get(BROWSER_PATH_ENV)
            entry = f"{configured.rstrip('/')}/{self.data['browser']['entry']}" if configured else None
            if entry is None:
                detail = f"no {BROWSER_PATH_ENV}, so no browser to start; expected the browser"
            else:
                detail = f"{entry} is there" if entry in found else f"{entry} is missing; expected the browser"
            self.record(entry in found, IMAGES, subject, "browser", detail)
        else:
            self.record(not found, IMAGES, subject, "browser",
                        f"{', '.join(sorted(found))} is there; expected no browser" if found else "none, as expected")


def _judge_latest(latest: Manifest | Exception, other_ref: str, other: Manifest | Exception,
                  version: str, after_switch: bool | None) -> tuple[bool, str]:
    """One latest* tag against the release's tag it must equal. `after_switch`
    is None when that could not be told: a server-suffixed tag was unreadable."""
    if isinstance(latest, NotFound) and isinstance(other, NotFound):
        if after_switch is None:
            return False, (f"missing, and so is {other_ref}, but whether {version} is from before the switch "
                           f"can't be told: one of its server-suffixed tags could not be inspected")
        if after_switch:
            return False, (f"missing, and so is {other_ref}: {version} has server-suffixed tags, "
                           f"so it must have this one too")
        return True, f"neither it nor {other_ref} exists: {version} is from before the switch"
    if isinstance(latest, NotFound):
        return False, f"missing, but {other_ref} exists"
    if isinstance(other, NotFound):
        return False, f"{latest.digest}, but {other_ref} does not exist"
    if latest.digest != other.digest:
        return False, f"digest {latest.digest} is not {other_ref}'s {other.digest}"
    return True, f"{other_ref} (same digest)"


def _holds(facts: Facts, expected: str) -> str:
    """`; the image holds Redis 8.10.2 (as redis-server)` when another server's
    name answers."""
    found = []
    for server, (code, out) in facts.version_runs.items():
        match = VERSION_LINE.search(out) if code == 0 else None
        if server != expected and match:
            found.append(f"{match.group('product')} {match.group('version')} (as {server}-server)")
    return f"; the image holds {', '.join(found)}" if found else ""


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _show(value: str | None) -> str:
    return "missing" if value is None else repr(value)


def _short_error(text: str) -> str:
    """The informative part of a failed `docker run`: for a missing binary,
    `exec: "valkey-server": executable file not found in $PATH`."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "(no output)"
    line = lines[0]
    for marker in ("error during container init: ", "Error response from daemon: "):
        if marker in line:
            line = line.split(marker, 1)[1]
            break
    return line[:300]


def _name_tags(names: list[tuple[str, str]]) -> str:
    """`falkordb:edge, falkordb:edge-rs on ghcr.io/falkordb, docker.io/falkordb`
    when every registry has the same tags, otherwise every full reference."""
    by_registry: dict[str, list[str]] = {}
    for registry, repo_tag in names:
        tags = by_registry.setdefault(registry, [])
        if repo_tag not in tags:
            tags.append(repo_tag)
    tag_sets = {tuple(sorted(tags)) for tags in by_registry.values()}
    if len(tag_sets) == 1:
        return f"{', '.join(next(iter(tag_sets)))} on {', '.join(by_registry)}"
    return ", ".join(f"{registry}/{tag}" for registry, tags in by_registry.items() for tag in tags)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def report(plan: Plan, checker: Checker, quiet: bool, note: str | None) -> int:
    print(f"== {plan.title} on {', '.join(plan.registries)}")
    if note:
        print(f"   {note}")
    for section in SECTIONS:
        results = sorted((r for r in checker.results if r.section == section), key=lambda r: r.subject)
        shown = [r for r in results if not (quiet and r.ok)]
        if shown:
            print(f"-- {section}")
        for result in shown:
            print(f"{'PASS' if result.ok else 'FAIL'}  {result.subject}  {result.check}: {result.detail}")
    failures = checker.failures
    passed = len(checker.results) - len(failures)
    print()
    if failures:
        print(f"== {len(failures)} of {len(checker.results)} checks FAILED for {plan.title}:")
        for section in SECTIONS:
            for result in sorted((r for r in failures if r.section == section), key=lambda r: r.subject):
                print(f"  - {result.subject}  {result.check}: {result.detail}")
    else:
        print(f"== all {passed} checks passed for {plan.title}")

    lines = [f"### Published tags: {plan.title}", ""]
    if note:
        lines += [note, ""]
    if failures:
        lines += [f"**{len(failures)} of {len(checker.results)} checks failed.**", ""]
        lines += [f"- `{result.subject}` {result.check}: {result.detail}" for result in failures]
    else:
        lines.append(f"All {passed} checks passed.")
    _step_summary(lines)
    return 1 if failures else 0


def _step_summary(lines: list[str]) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    try:
        with open(summary, "a", encoding="utf-8") as out:
            out.write("\n".join(lines) + "\n")
    except OSError as error:
        print(f"warning: cannot write the step summary {summary}: {error}", file=sys.stderr)


def _fatal(message: str) -> int:
    """An error that stops the check before it starts: no result is known."""
    print(f"error: {message}", file=sys.stderr)
    _step_summary(["### Published tags", "", f"**Not checked:** {message}"])
    return 2


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        _step_summary(["### Published tags", "", f"**Not checked:** {message}"])
        super().error(message)


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="edge, latest, a release version (6.1.0, 6.1.0-rc.1), or rc-<ci_key>")
    parser.add_argument("--tags-file", type=Path, default=DEFAULT_TAGS_FILE)
    parser.add_argument("--servers", default=None,
                        help="CI targets only: comma-separated servers to check (default: all)")
    parser.add_argument("--pin", action="append", default=[], metavar="SERVER=X.Y.Z",
                        help="expect this server version instead of the pin in the tags file "
                             "(for an older release); repeatable")
    parser.add_argument("--latest", default="auto", metavar="X.Y.Z|auto",
                        help="the release latest* must point at; auto (the default) takes the "
                             "newest stable GitHub release of --github-repo")
    parser.add_argument("--github-repo", default=os.environ.get("GITHUB_REPOSITORY", "FalkorDB/FalkorDB"))
    parser.add_argument("--registry", action="append", default=[], metavar="NAME=PREFIX",
                        help="check another registry in place of one in the tags file, "
                             "e.g. ghcr=localhost:5000/falkordb; repeatable")
    parser.add_argument("--keep-images", action="store_true", help="don't remove the images it pulls")
    parser.add_argument("--quiet", action="store_true", help="print only failures")
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except FileNotFoundError as error:
        if error.filename in ("docker", "gh"):
            return _fatal(f"{error.filename} is not installed")
        traceback.print_exc()
        return _fatal(f"the check crashed: {error!r}")
    except Exception as error:  # a bug in the check or its data: say so where the run is read
        traceback.print_exc()
        return _fatal(f"the check crashed: {error!r}")


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    data = json.loads(args.tags_file.read_text(encoding="utf-8"))
    try:
        target = parse_target(args.target, data)
    except ValueError as error:
        parser.error(str(error))

    all_servers = set(data["servers"])
    servers = all_servers
    if args.servers:
        if target.kind != "ci":
            parser.error("--servers applies to CI targets only; every public tag is required")
        servers = {name.strip() for name in args.servers.split(",") if name.strip()}
        if not servers or not servers <= all_servers:
            parser.error(f"--servers must name some of: {', '.join(sorted(all_servers))}")
        every = data["channels"]["ci"].get("every_server_key_prefixes", [])
        if servers != all_servers and any((target.ci_key or "").startswith(prefix) for prefix in every):
            parser.error(f"--servers: rc-{target.ci_key} has every server's images; omit --servers or name them all")

    pins = {name: spec["version"] for name, spec in data["servers"].items()}
    for item in args.pin:
        name, sep, version = item.partition("=")
        if not sep or name not in pins or not PIN_VERSION.match(version):
            parser.error(f"--pin wants SERVER=X.Y.Z with SERVER one of {', '.join(sorted(pins))}, got {item!r}")
        pins[name] = version

    registries = dict(data["registries"])
    for item in args.registry:
        name, sep, prefix = item.partition("=")
        if not sep or name not in registries or not prefix:
            parser.error(f"--registry wants NAME=PREFIX with NAME one of {', '.join(sorted(registries))}")
        registries[name] = prefix.rstrip("/")

    note = None
    latest_version = latest_error = None
    if target.kind != "ci":
        if args.latest != "auto":
            match = RELEASE_VERSION.match(args.latest)
            if not match or match.group("pre"):
                parser.error("--latest wants auto or a stable X.Y.Z: a prerelease never moves latest")
            latest_version = args.latest.lstrip("v")
            note = f"latest* must point at {latest_version} (--latest)"
        else:
            try:
                releases = github_releases(args.github_repo)
            except InspectError as error:
                latest_error = f"cannot tell where latest* must point: {error}"
            else:
                latest_version, note = latest_rule(releases, target.version, args.github_repo)
                if latest_version is None:
                    latest_error = note

    try:
        plan = make_plan(data, target, servers=servers, registries=registries,
                         latest_version=latest_version, latest_error=latest_error)
    except ValueError as error:
        return _fatal(str(error))

    checker = Checker(data, Docker(keep_images=args.keep_images), pins)
    checker.check(plan)
    return report(plan, checker, args.quiet, note)


if __name__ == "__main__":
    sys.exit(main())
