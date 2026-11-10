#!/usr/bin/env python3
"""Tests for check_published_tags.py, against a fake registry.

    python3 build/test_check_published_tags.py
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_published_tags as judge  # noqa: E402

# The workflow runs these tests in the job that then runs the check: they must
# not write to its step summary. Tests that read a summary set their own.
os.environ.pop("GITHUB_STEP_SUMMARY", None)

DATA = json.loads(judge.DEFAULT_TAGS_FILE.read_text(encoding="utf-8"))
REGISTRIES = dict(DATA["registries"])
GHCR, HUB = REGISTRIES["ghcr"], REGISTRIES["dockerhub"]
PLATFORMS = DATA["platforms"]
REPOS = list(DATA["repos"])
SUFFIX = DATA["arch_suffixes"]
BROWSER_PATH = DATA["browser"]["default_path"]
BROWSER_ENTRY = f"{BROWSER_PATH}/{DATA['browser']['entry']}"
LEFT_OVER = {f"{REGISTRIES[name]}/{repo_tag}": digest
             for name, tags in DATA["left_from_before_the_switch"].items() if not name.startswith("_")
             for repo_tag, digest in tags.items()}
PER_OS = (("alpine", "valkey"), ("amazonlinux2023", "valkey"), ("rhel9", "redis"), ("amazonlinux2023", "redis"))

OS_RELEASE = {
    "ubuntu-26.04": 'PRETTY_NAME="Ubuntu 26.04 LTS"\nID=ubuntu\nVERSION_ID="26.04"\n',
    "debian-trixie": 'PRETTY_NAME="Debian GNU/Linux 13 (trixie)"\nID=debian\nVERSION_ID="13"\n'
                     "VERSION_CODENAME=trixie\n",
    "alpine": "ID=alpine\nVERSION_ID=3.22.1\n",
    "amazonlinux2023": 'ID="amzn"\nVERSION_ID="2023"\n',
    "rhel9": 'ID="rhel"\nVERSION_ID="9.6"\n',
}
VERSION_OUTPUT = {
    "valkey": "Valkey server v={version} sha=00000000:0 malloc=jemalloc-5.3.0 bits=64 build=1",
    "redis": "Redis server v={version} sha=00000000:0 malloc=jemalloc-5.3.0 bits=64 build=2",
}
PIN = {name: spec["version"] for name, spec in DATA["servers"].items()}
OS_OF = {"valkey": "ubuntu-26.04", "redis": "debian-trixie"}
SAME = "same"


def _digest(*parts: object) -> str:
    return "sha256:" + hashlib.sha256(repr(parts).encode()).hexdigest()


def browser_for(repo: str) -> bool:
    return bool(DATA["repos"][repo]["browser"])


class FakeDocker:
    """A registry and a container runtime, as dictionaries."""

    def __init__(self) -> None:
        self.tags: dict[str, str] = {}  # "registry/repo:tag" -> manifest digest
        self.manifests: dict[str, dict] = {}  # digest -> imagetools .Manifest
        self.images: dict[str, dict] = {}  # platform digest -> what is inside
        self.errors: dict[str, Exception] = {}  # ref -> what inspecting it raises
        self.pull_errors: dict[str, Exception] = {}  # platform digest -> what pulling it raises
        self.config_errors: set[str] = set()  # platform digests whose config cannot be read
        self.pulls: list[str] = []
        self.removed: list[str] = []
        self.inspected: list[str] = []  # every ref whose manifest was asked for

    # -- building the world --------------------------------------------------

    def image(self, server: str, base_os: str, platform: str, *, label: str | None = SAME,
              env: str | None = SAME, binary: str | None = SAME, version: str | None = None,
              binary_version: str | None = None, extra_bins: dict[str, str] | None = None,
              browser: bool = False, browser_env: str | None = SAME, browser_path_env: bool | None = None,
              browser_files: bool | None = None, browser_path: str = BROWSER_PATH, salt: str = "") -> str:
        """One platform image of `server` at `version` (default: its pin).
        `label`/`env`/`binary` name the server each claims (None leaves it
        out); `extra_bins` adds `<name>-server` binaries answering as a server;
        `browser` puts the browser in (env and files); `browser_env` overrides
        BROWSER, `browser_path_env` whether FALKORDB_BROWSER_PATH is set,
        `browser_files` whether server.js is there, and `browser_path` where."""
        label = server if label == SAME else label
        env = server if env == SAME else env
        binary = server if binary == SAME else binary
        browser_env = ("1" if browser else "0") if browser_env == SAME else browser_env
        browser_path_env = browser if browser_path_env is None else browser_path_env
        browser_files = browser if browser_files is None else browser_files

        def version_of(name: str) -> str:
            return version if (version and name == server) else PIN[name]

        os_name, arch = platform.split("/")
        labels = {}
        if label:
            labels = {judge.SERVER_NAME_LABEL: label, judge.SERVER_VERSION_LABEL: version_of(label)}
        env_list = ["PATH=/usr/bin"]
        if env:
            env_list += [f"{judge.SERVER_ENV}={env}", f"{judge.SERVER_VERSION_ENV}={version_of(env)}"]
        if browser_env is not None:
            env_list.append(f"{judge.BROWSER_ENV}={browser_env}")
        if browser_path_env:
            env_list.append(f"{judge.BROWSER_PATH_ENV}={browser_path}")
        bins = {}
        if binary:
            bins[f"{server}-server"] = VERSION_OUTPUT[binary].format(
                version=binary_version or version_of(binary))
        for name, answers_as in (extra_bins or {}).items():
            bins[f"{name}-server"] = VERSION_OUTPUT[answers_as].format(version=PIN[answers_as])
        digest = _digest(server, base_os, platform, label, env, binary, version, binary_version,
                         sorted((extra_bins or {}).items()), browser_env, browser_path_env, browser_files,
                         browser_path, salt)
        self.images[digest] = {
            "config": {"os": os_name, "architecture": arch,
                       "config": {"Labels": labels or None, "Env": env_list}},
            "os_release": OS_RELEASE[base_os],
            "bins": bins,
            "files": {f"{browser_path}/{DATA['browser']['entry']}"} if browser_files else set(),
        }
        self.manifests[digest] = {"mediaType": "application/vnd.oci.image.manifest.v1+json",
                                  "digest": digest}
        return digest

    def index(self, *platform_digests: str) -> str:
        entries = []
        for digest in platform_digests:
            config = self.images[digest]["config"]
            entries.append({"digest": digest,
                            "platform": {"os": config["os"], "architecture": config["architecture"]}})
        # An attestation manifest, as buildx pushes them; never a platform.
        entries.append({"digest": _digest("attestation", platform_digests),
                        "platform": {"os": "unknown", "architecture": "unknown"},
                        "annotations": {"vnd.docker.reference.type": "attestation-manifest"}})
        digest = _digest("index", platform_digests)
        self.manifests[digest] = {"mediaType": "application/vnd.oci.image.index.v1+json",
                                  "digest": digest, "manifests": entries}
        return digest

    def multi(self, server: str, base_os: str, **kwargs) -> str:
        return self.index(*(self.image(server, base_os, platform, **kwargs) for platform in PLATFORMS))

    def keep(self, ref: str, digest: str) -> None:
        """Publish a single-platform image under `ref` with a given digest, as
        the tags left from before the switch are."""
        like = self.images[self.image("redis", "debian-trixie", "linux/amd64", salt=digest)]
        self.images[digest] = like
        self.manifests[digest] = {"mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": digest}
        self.tags[ref] = digest

    def push(self, digest: str, repo: str, *tags: str, registries=(GHCR, HUB)) -> None:
        for registry in registries:
            for tag in tags:
                self.tags[f"{registry}/{repo}:{tag}"] = digest

    # -- the Docker interface ------------------------------------------------

    def manifest(self, ref: str) -> dict:
        self.inspected.append(ref)
        if ref in self.errors:
            raise self.errors[ref]
        if ref not in self.tags:
            raise judge.NotFound(f"{ref}: not found")
        return self.manifests[self.tags[ref]]

    def config(self, ref: str) -> dict:
        digest = ref.split("@", 1)[1]
        if digest in self.config_errors:
            # The worst case: the tag is there, but its config lookup says "not found".
            raise judge.NotFound(f"{ref}: not found")
        return self.images[digest]["config"]

    def has_image(self, ref: str) -> bool:
        return False

    def pull(self, ref: str, platform: str) -> None:
        digest = ref.split("@", 1)[1]
        if digest in self.pull_errors:
            raise self.pull_errors[digest]
        self.pulls.append(ref)

    def run(self, ref: str, platform: str, entrypoint: str, args: list[str]) -> tuple[int, str]:
        image = self.images[ref.split("@", 1)[1]]
        if entrypoint == "cat":
            return 0, image["os_release"]
        if entrypoint == "sh" and args[:2] == ["-c", 'test -f "$1"']:
            return (0 if args[3] in image["files"] else 1), ""
        if entrypoint in image["bins"]:
            return 0, image["bins"][entrypoint]
        return 127, (f'docker: Error response from daemon: failed to create task: error during '
                     f'container init: exec: "{entrypoint}": executable file not found in $PATH\n\n'
                     "Run 'docker run --help' for more information")

    def remove(self, ref: str) -> None:
        self.removed.append(ref)


def publish_edge(docker: FakeDocker) -> dict[str, str]:
    """Every edge tag, as the contract says. Returns repo -> valkey index."""
    valkey = {}
    for repo in REPOS:
        valkey[repo] = docker.multi("valkey", "ubuntu-26.04", browser=browser_for(repo))
        docker.push(valkey[repo], repo, "edge-rs-valkey", "edge", "edge-rs")
        docker.push(docker.multi("redis", "debian-trixie", browser=browser_for(repo)), repo, "edge-rs-redis")
    return valkey


def publish_release(docker: FakeDocker, version: str, *, latest: bool, valkey_version: str | None = None) -> None:
    for repo in REPOS:
        valkey = docker.multi("valkey", "ubuntu-26.04", version=valkey_version, browser=browser_for(repo),
                              salt=version)
        redis = docker.multi("redis", "debian-trixie", browser=browser_for(repo), salt=version)
        docker.push(valkey, repo, f"{version}-valkey", version)
        docker.push(redis, repo, f"{version}-redis")
        if latest:
            docker.push(valkey, repo, "latest-valkey", "latest")
            docker.push(redis, repo, "latest-redis")
    for base_os, server in PER_OS:
        image = docker.multi(server, base_os, version=valkey_version if server == "valkey" else None, salt=version)
        docker.push(image, "falkordb-server", f"{version}-{base_os}-{server}")


def publish_ci(docker: FakeDocker, key: str, servers=("valkey", "redis")) -> None:
    for repo in REPOS:
        for server in servers:
            images = [docker.image(server, OS_OF[server], platform, browser=browser_for(repo))
                      for platform in PLATFORMS]
            docker.push(docker.index(*images), repo, f"rc-{key}-{server}", registries=(GHCR,))
            for platform, image in zip(PLATFORMS, images, strict=True):
                docker.push(image, repo, f"rc-{key}-{server}-{SUFFIX[platform]}", registries=(GHCR,))


def publish_pre_switch_latest(docker: FakeDocker, version: str = "6.0.4") -> None:
    """latest as it is today: a bare release from before the switch, with no
    -valkey/-redis tags."""
    for repo in REPOS:
        old = docker.multi("redis", "debian-trixie", label=None, env=None, browser=browser_for(repo),
                           salt=version + repo)
        docker.push(old, repo, version, "latest")


def check(docker: FakeDocker, target: str, *, servers=None, latest: str | None = None,
          latest_error: str | None = None, pins=None) -> judge.Checker:
    """Check a target the way main() does. A public target always knows
    where latest* points: by default at today's pre-switch 6.0.4, published
    here unless the test published 6.0.4 itself."""
    parsed = judge.parse_target(target, DATA)
    if parsed.kind != "ci" and latest is None and latest_error is None:
        latest = "6.0.4"
        if f"{GHCR}/falkordb:6.0.4" not in docker.tags:
            publish_pre_switch_latest(docker)
    plan = judge.make_plan(DATA, parsed, servers=set(servers or DATA["servers"]),
                           registries=REGISTRIES, latest_version=latest, latest_error=latest_error)
    checker = judge.Checker(DATA, docker, dict(PIN, **(pins or {})))
    checker.check(plan)
    return checker


def failed(checker: judge.Checker) -> list[tuple[str, str]]:
    return [(result.subject, result.check) for result in checker.failures]


def checks_failed(checker: judge.Checker) -> set[str]:
    return {result.check for result in checker.failures}


class EdgeTest(unittest.TestCase):
    def test_the_contract_passes(self):
        docker = FakeDocker()
        publish_edge(docker)
        publish_release(docker, "6.0.4", latest=True)
        checker = check(docker, "edge", latest="6.0.4")
        self.assertEqual(failed(checker), [])
        # Prove each kind of check ran, so "no failures" means something.
        ran = {result.check for result in checker.results}
        for name in ("exists", "architectures", "alias of", "same digest on every registry",
                     "config platform", f"label {judge.SERVER_NAME_LABEL}", f"label {judge.SERVER_VERSION_LABEL}",
                     f"env {judge.SERVER_ENV}", f"env {judge.SERVER_VERSION_ENV}", f"env {judge.BROWSER_ENV}",
                     f"env {judge.BROWSER_PATH_ENV}", "browser", "valkey-server --version",
                     "redis-server --version", "no other server", "/etc/os-release", "points at 6.0.4",
                     "must not exist"):
            self.assertIn(name, ran)
        # 2 repos x 2 servers x 2 platforms: each image is pulled once, however
        # many tags and registries name it, and removed after.
        self.assertEqual(len(docker.pulls), 8)
        self.assertEqual(sorted(docker.pulls), sorted(docker.removed))

    def test_redis_under_a_valkey_tag_fails_by_version_even_with_valkey_label_and_env(self):
        docker = FakeDocker()
        publish_edge(docker)
        # Valkey's label, env and OS, Redis's binary under the valkey-server name.
        lying = docker.multi("valkey", "ubuntu-26.04", binary="redis")
        docker.push(lying, "falkordb-server", "edge-rs-valkey", "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {"valkey-server --version"})
        details = [result.detail for result in checker.failures]
        self.assertEqual(len(details), 2)  # one per platform
        self.assertTrue(all("Redis server v=8.10.2; expected Valkey server v=9.1.2" in d for d in details))

    def test_another_server_with_the_pinned_version_fails(self):
        """The product name is checked, not just the version number."""
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.multi("valkey", "ubuntu-26.04", binary="redis", binary_version=PIN["valkey"],
                                 browser=True), "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {"valkey-server --version"})
        self.assertIn(f"Redis server v={PIN['valkey']}; expected Valkey", checker.failures[0].detail)

    def test_redis_image_under_a_valkey_tag_fails_by_label_env_version_and_os(self):
        docker = FakeDocker()
        publish_edge(docker)
        redis = docker.multi("redis", "debian-trixie")
        docker.push(redis, "falkordb-server", "edge-rs-valkey", "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertEqual(
            checks_failed(checker),
            {f"label {judge.SERVER_NAME_LABEL}", f"label {judge.SERVER_VERSION_LABEL}",
             f"env {judge.SERVER_ENV}", f"env {judge.SERVER_VERSION_ENV}",
             "valkey-server --version", "no other server", "/etc/os-release"},
        )
        version_failure = next(r for r in checker.failures if r.check == "valkey-server --version")
        self.assertIn('exec: "valkey-server": executable file not found', version_failure.detail)
        self.assertIn("the image holds Redis 8.10.2 (as redis-server)", version_failure.detail)

    def test_a_second_server_inside_a_valkey_image_fails(self):
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.multi("valkey", "ubuntu-26.04", extra_bins={"redis": "redis"}, browser=True),
                    "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {"no other server"})
        self.assertIn("redis-server is Redis server v=8.10.2", checker.failures[0].detail)

    def test_redis_names_answering_as_valkey_pass(self):
        """Valkey images ship redis-* symlinks to Valkey."""
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.multi("valkey", "ubuntu-26.04", extra_bins={"redis": "valkey"}, browser=True),
                    "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        self.assertEqual(failed(check(docker, "edge")), [])

    def test_a_wrong_label_fails_on_its_own(self):
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.multi("valkey", "ubuntu-26.04", label="redis", browser=True),
                    "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker),
                         {f"label {judge.SERVER_NAME_LABEL}", f"label {judge.SERVER_VERSION_LABEL}"})

    def test_a_missing_label_and_env_fail(self):
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.multi("valkey", "ubuntu-26.04", label=None, env=None, browser=True),
                    "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertEqual(
            checks_failed(checker),
            {f"label {judge.SERVER_NAME_LABEL}", f"label {judge.SERVER_VERSION_LABEL}",
             f"env {judge.SERVER_ENV}", f"env {judge.SERVER_VERSION_ENV}"},
        )
        self.assertTrue(all(r.detail.startswith("missing, expected") for r in checker.failures))

    def test_the_wrong_server_version_fails(self):
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.multi("valkey", "ubuntu-26.04", binary_version="9.0.4", browser=True),
                    "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {"valkey-server --version"})

    def test_falkordb_without_the_browser_fails(self):
        """The server-only image pushed to falkordb as well."""
        docker = FakeDocker()
        publish_edge(docker)
        server_only = docker.tags[f"{GHCR}/falkordb-server:edge-rs-redis"]
        docker.push(server_only, "falkordb", "edge-rs-redis")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker),
                         {f"env {judge.BROWSER_ENV}", f"env {judge.BROWSER_PATH_ENV}", "browser"})
        browser = next(r for r in checker.failures if r.check == "browser")
        self.assertIn("no FALKORDB_BROWSER_PATH, so no browser to start", browser.detail)

    def test_falkordb_with_browser_env_but_no_browser_files_fails(self):
        docker = FakeDocker()
        publish_edge(docker)
        empty = docker.multi("valkey", "ubuntu-26.04", browser_env="1")
        docker.push(empty, "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {f"env {judge.BROWSER_PATH_ENV}", "browser"})

    def test_falkordb_server_with_the_browser_fails(self):
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.multi("redis", "debian-trixie", browser=True, salt="x"), "falkordb-server",
                    "edge-rs-redis")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {f"env {judge.BROWSER_ENV}", "browser"})
        browser = next(r for r in checker.failures if r.check == "browser")
        self.assertIn(f"{BROWSER_ENTRY} is there; expected no browser", browser.detail)

    def test_a_browser_in_falkordb_server_is_found_even_with_browser_0(self):
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.multi("redis", "debian-trixie", browser=True, browser_env="0", salt="y"),
                    "falkordb-server", "edge-rs-redis")
        self.assertEqual(checks_failed(check(docker, "edge")), {"browser"})

    def test_a_browser_at_the_usual_path_without_its_env_is_found(self):
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.multi("redis", "debian-trixie", browser=True, browser_env="0", browser_path_env=False),
                    "falkordb-server", "edge-rs-redis")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {"browser"})
        self.assertIn(f"{BROWSER_ENTRY} is there", checker.failures[0].detail)

    def test_falkordb_with_its_browser_path_but_no_server_js_fails(self):
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.multi("valkey", "ubuntu-26.04", browser=True, browser_files=False),
                    "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {"browser"})
        self.assertIn(f"{BROWSER_ENTRY} is missing; expected the browser", checker.failures[0].detail)

    def test_the_browser_is_looked_for_at_its_configured_path(self):
        docker = FakeDocker()
        publish_edge(docker)
        elsewhere = docker.multi("valkey", "ubuntu-26.04", browser=True, browser_path="/opt/browser")
        docker.push(elsewhere, "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        self.assertEqual(failed(check(docker, "edge")), [])
        hidden = docker.multi("redis", "debian-trixie", browser_env="0", browser_path_env=True,
                              browser_files=True, browser_path="/opt/browser")
        docker.push(hidden, "falkordb-server", "edge-rs-redis")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {"browser"})
        self.assertIn("/opt/browser/server.js is there; expected no browser", checker.failures[0].detail)

    def test_an_unreadable_config_is_one_failure_not_a_browser_one(self):
        docker = FakeDocker()
        valkey = publish_edge(docker)
        amd64 = docker.manifests[valkey["falkordb"]]["manifests"][0]["digest"]
        docker.config_errors.add(amd64)
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {"image config"})

    def test_a_missing_architecture_fails(self):
        docker = FakeDocker()
        valkey = publish_edge(docker)
        amd64_only = docker.index(docker.manifests[valkey["falkordb"]]["manifests"][0]["digest"])
        docker.push(amd64_only, "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {"architectures"})
        self.assertEqual(len(checker.failures), 6)  # 3 tags x 2 registries
        self.assertIn("lists linux/amd64; missing linux/arm64", checker.failures[0].detail)

    def test_a_platform_listed_twice_fails(self):
        """A pull takes the first linux/amd64; the judge must not pass the second."""
        docker = FakeDocker()
        valkey = publish_edge(docker)
        entries = docker.manifests[valkey["falkordb-server"]]["manifests"]
        redis_amd64 = docker.image("redis", "debian-trixie", "linux/amd64")
        twice = docker.index(redis_amd64, *(entry["digest"] for entry in entries[:2]))
        docker.push(twice, "falkordb-server", "edge-rs-valkey", "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertIn("architectures", checks_failed(checker))
        self.assertIn("valkey-server --version", checks_failed(checker))
        arch = next(r for r in checker.failures if r.check == "architectures")
        self.assertIn("lists linux/amd64 more than once", arch.detail)

    def test_an_index_entry_without_an_architecture_fails_without_crashing(self):
        docker = FakeDocker()
        valkey = publish_edge(docker)
        entries = docker.manifests[valkey["falkordb-server"]]["manifests"]
        del entries[1]["platform"]["architecture"]
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {"architectures"})
        self.assertIn("missing linux/arm64", checker.failures[0].detail)

    def test_an_extra_platform_fails(self):
        docker = FakeDocker()
        valkey = publish_edge(docker)
        entries = docker.manifests[valkey["falkordb-server"]]["manifests"]
        s390x = docker.image("redis", "debian-trixie", "linux/s390x")
        extra = docker.index(*(entry["digest"] for entry in entries[:2]), s390x)
        docker.push(extra, "falkordb-server", "edge-rs-valkey", "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {"architectures"})
        self.assertIn("also lists linux/s390x", checker.failures[0].detail)

    def test_a_single_platform_manifest_under_a_public_tag_fails(self):
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.image("valkey", "ubuntu-26.04", "linux/amd64", browser=True),
                    "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {"architectures"})
        self.assertIn("a single-platform manifest (linux/amd64), not a multi-arch index",
                      checker.failures[0].detail)

    def test_an_index_whose_platform_lies_about_its_image_fails(self):
        docker = FakeDocker()
        publish_edge(docker)
        amd64 = docker.image("valkey", "ubuntu-26.04", "linux/amd64", browser=True)
        crossed = docker.index(amd64, docker.image("valkey", "ubuntu-26.04", "linux/arm64", browser=True))
        # Claim the amd64 image is the arm64 one.
        docker.manifests[crossed]["manifests"][1]["digest"] = amd64
        docker.push(crossed, "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        self.assertEqual(checks_failed(check(docker, "edge")), {"config platform"})

    def test_a_failed_pull_still_reports_what_the_config_says(self):
        docker = FakeDocker()
        publish_edge(docker)
        wrong = docker.image("valkey", "ubuntu-26.04", "linux/arm64", label="redis", browser=True)
        docker.push(docker.index(docker.image("valkey", "ubuntu-26.04", "linux/amd64", browser=True), wrong),
                    "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        docker.pull_errors[wrong] = judge.InspectError("toomanyrequests")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {f"label {judge.SERVER_NAME_LABEL}",
                                                  f"label {judge.SERVER_VERSION_LABEL}", "run the image"})

    def test_edge_with_another_digest_than_edge_rs_valkey_fails(self):
        docker = FakeDocker()
        publish_edge(docker)
        # A correct Valkey image, but not the one edge-rs-valkey holds.
        docker.push(docker.multi("valkey", "ubuntu-26.04", browser=True, salt="rebuilt"), "falkordb", "edge")
        checker = check(docker, "edge")
        self.assertEqual(failed(checker), [(f"{GHCR}/falkordb:edge", "alias of"),
                                           (f"{HUB}/falkordb:edge", "alias of")])

    def test_another_digest_on_docker_hub_fails(self):
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.multi("redis", "debian-trixie", salt="rebuilt"), "falkordb-server",
                    "edge-rs-redis", registries=(HUB,))
        checker = check(docker, "edge")
        self.assertEqual(failed(checker), [("falkordb-server:edge-rs-redis", "same digest on every registry")])

    def test_the_wrong_base_os_fails(self):
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.multi("valkey", "debian-trixie", browser=True), "falkordb", "edge-rs-valkey",
                    "edge", "edge-rs")
        checker = check(docker, "edge")
        self.assertEqual(checks_failed(checker), {"/etc/os-release"})
        self.assertIn("expected ubuntu-26.04 (ID=ubuntu VERSION_ID=26.04)", checker.failures[0].detail)

    def test_an_edge_publish_that_moved_latest_fails(self):
        docker = FakeDocker()
        publish_release(docker, "6.0.4", latest=True)
        valkey = publish_edge(docker)
        docker.push(valkey["falkordb"], "falkordb", "latest", "latest-valkey")
        checker = check(docker, "edge", latest="6.0.4")
        self.assertEqual(checks_failed(checker), {"points at 6.0.4"})
        self.assertEqual({r.subject for r in checker.failures},
                         {f"{registry}/falkordb:{tag}" for registry in (GHCR, HUB)
                          for tag in ("latest", "latest-valkey")})

    def test_latest_from_before_the_switch_passes(self):
        """Until the first new release, latest is a bare 6.0.4 with no -valkey/-redis tags."""
        docker = FakeDocker()
        publish_edge(docker)
        for repo in REPOS:
            old = docker.multi("redis", "debian-trixie", label=None, env=None, salt="6.0.4" + repo)
            docker.push(old, repo, "6.0.4", "latest")
        checker = check(docker, "edge", latest="6.0.4")
        self.assertEqual(failed(checker), [])
        details = [r.detail for r in checker.results if r.check == "points at 6.0.4"]
        self.assertEqual(len(details), 12)
        self.assertEqual(sum("is from before the switch" in d for d in details), 8)

    def test_a_stray_latest_valkey_before_the_first_new_release_fails(self):
        docker = FakeDocker()
        valkey = publish_edge(docker)
        for repo in REPOS:
            old = docker.multi("redis", "debian-trixie", label=None, env=None, salt="6.0.4" + repo)
            docker.push(old, repo, "6.0.4", "latest")
        docker.push(valkey["falkordb"], "falkordb", "latest-valkey")
        checker = check(docker, "edge", latest="6.0.4")
        self.assertEqual(failed(checker), [(f"{GHCR}/falkordb:latest-valkey", "points at 6.0.4"),
                                           (f"{HUB}/falkordb:latest-valkey", "points at 6.0.4")])
        self.assertIn("but", checker.failures[0].detail)

    def test_latest_left_on_an_older_release_fails_on_an_edge_run(self):
        docker = FakeDocker()
        publish_edge(docker)
        publish_release(docker, "6.0.4", latest=True)
        publish_release(docker, "6.1.0", latest=False)
        checker = check(docker, "edge", latest="6.1.0")
        self.assertEqual(checks_failed(checker), {"points at 6.1.0"})
        self.assertEqual(len(checker.failures), 12)

    def test_latest_cannot_be_checked_while_the_newest_release_has_no_images(self):
        """Its GitHub release is published before its images. Passing in that
        window would let an edge publish that moved latest go unseen."""
        docker = FakeDocker()
        valkey = publish_edge(docker)
        publish_release(docker, "6.0.4", latest=True)
        for moved in (False, True):
            if moved:
                docker.push(valkey["falkordb"], "falkordb", "latest", "latest-valkey")
            with self.subTest(latest_moved_to_edge=moved):
                checker = check(docker, "edge", latest="6.1.0")
                self.assertEqual(failed(checker), [(f"{registry}/{repo}:latest*", "points at 6.1.0")
                                                   for registry in (GHCR, HUB) for repo in REPOS])
                self.assertIn("6.1.0 has no images here yet, so latest* cannot be checked; "
                              "re-run this check once 6.1.0's image jobs have finished",
                              checker.failures[0].detail)
        # No older release is looked at: every lookup counts against
        # Docker Hub's rate limit.
        self.assertFalse([ref for ref in docker.inspected if ":6.0.4" in ref])

    def test_a_release_from_after_the_switch_must_have_both_server_tags(self):
        """latest-redis and 6.1.0-redis both missing passes only for a release
        from before the switch."""
        docker = FakeDocker()
        publish_edge(docker)
        publish_release(docker, "6.1.0", latest=True)
        for registry in (GHCR, HUB):
            for repo in REPOS:
                del docker.tags[f"{registry}/{repo}:6.1.0-redis"]
                del docker.tags[f"{registry}/{repo}:latest-redis"]
        checker = check(docker, "edge", latest="6.1.0")
        self.assertEqual(checks_failed(checker), {"points at 6.1.0"})
        self.assertEqual({r.subject for r in checker.failures},
                         {f"{registry}/{repo}:latest-redis" for registry in (GHCR, HUB) for repo in REPOS})
        self.assertIn("6.1.0 has server-suffixed tags, so it must have this one too", checker.failures[0].detail)

    def test_a_release_after_the_switch_on_one_registry_is_after_it_on_every_registry(self):
        """A partial push: Docker Hub got the bare 6.1.0 only."""
        docker = FakeDocker()
        publish_edge(docker)
        publish_release(docker, "6.1.0", latest=True)
        for repo in REPOS:
            for tag in ("6.1.0-valkey", "6.1.0-redis", "latest-valkey", "latest-redis"):
                del docker.tags[f"{HUB}/{repo}:{tag}"]
        checker = check(docker, "edge", latest="6.1.0")
        self.assertEqual({r.subject for r in checker.failures},
                         {f"{HUB}/{repo}:{tag}" for repo in REPOS for tag in ("latest-valkey", "latest-redis")})
        self.assertIn("6.1.0 has server-suffixed tags", checker.failures[0].detail)

    def test_a_release_after_the_switch_in_one_repo_is_after_it_in_every_repo(self):
        """A partial push: falkordb-server got the bare 6.1.0 only."""
        docker = FakeDocker()
        publish_edge(docker)
        publish_release(docker, "6.1.0", latest=True)
        for registry in (GHCR, HUB):
            for tag in ("6.1.0-valkey", "6.1.0-redis", "latest-valkey", "latest-redis"):
                del docker.tags[f"{registry}/falkordb-server:{tag}"]
        checker = check(docker, "edge", latest="6.1.0")
        self.assertEqual({r.subject for r in checker.failures},
                         {f"{registry}/falkordb-server:{tag}" for registry in (GHCR, HUB)
                          for tag in ("latest-valkey", "latest-redis")})

    def test_an_unreadable_server_tag_means_the_switch_cannot_be_told(self):
        docker = FakeDocker()
        publish_edge(docker)
        publish_pre_switch_latest(docker)
        for repo in REPOS:
            docker.errors[f"{GHCR}/{repo}:6.0.4-valkey"] = judge.InspectError("toomanyrequests")
        checker = check(docker, "edge", latest="6.0.4")
        hub = [r for r in checker.failures if r.subject.startswith(HUB)]
        self.assertEqual({r.subject for r in hub},
                         {f"{HUB}/{repo}:{tag}" for repo in REPOS for tag in ("latest-valkey", "latest-redis")})
        self.assertTrue(all("can't be told" in r.detail for r in hub))

    def test_a_release_with_only_its_redis_tag_must_have_the_valkey_pair_too(self):
        docker = FakeDocker()
        publish_edge(docker)
        publish_release(docker, "6.1.0", latest=True)
        for registry in (GHCR, HUB):
            for repo in REPOS:
                for tag in ("6.1.0-valkey", "latest-valkey", "6.1.0", "latest"):
                    del docker.tags[f"{registry}/{repo}:{tag}"]
        checker = check(docker, "edge", latest="6.1.0")
        self.assertEqual({r.subject.rsplit(":", 1)[1] for r in checker.failures}, {"latest-valkey", "latest"})
        self.assertEqual(len(checker.failures), 8)  # 2 tags x 2 repos x 2 registries

    def test_an_uninspectable_latest_fails_instead_of_passing(self):
        docker = FakeDocker()
        publish_release(docker, "6.0.4", latest=True)
        publish_edge(docker)
        docker.errors[f"{HUB}/falkordb:latest"] = judge.InspectError("toomanyrequests")
        checker = check(docker, "edge", latest="6.0.4")
        self.assertEqual(failed(checker), [(f"{HUB}/falkordb:latest", "points at 6.0.4")])
        self.assertIn("could not inspect", checker.failures[0].detail)

    def test_an_uninspectable_release_tag_fails_the_latest_check(self):
        docker = FakeDocker()
        publish_release(docker, "6.0.4", latest=True)
        publish_edge(docker)
        docker.errors[f"{HUB}/falkordb:6.0.4-redis"] = judge.InspectError("toomanyrequests")
        checker = check(docker, "edge", latest="6.0.4")
        self.assertEqual(failed(checker), [(f"{HUB}/falkordb:latest*", "points at 6.0.4")])
        self.assertIn(f"could not inspect {HUB}/falkordb:6.0.4-redis", checker.failures[0].detail)

    def test_not_knowing_where_latest_points_fails(self):
        docker = FakeDocker()
        publish_edge(docker)
        checker = check(docker, "edge", latest_error="cannot tell where latest* must point: gh failed")
        self.assertEqual(failed(checker), [("latest*", "points at the newest stable release")])

    def test_a_server_suffixed_tag_with_an_arch_suffix_fails(self):
        docker = FakeDocker()
        valkey = publish_edge(docker)
        amd64 = docker.manifests[valkey["falkordb"]]["manifests"][0]["digest"]
        docker.push(amd64, "falkordb", "edge-rs-valkey-x64", registries=(HUB,))
        checker = check(docker, "edge")
        self.assertEqual(failed(checker), [(f"{HUB}/falkordb:edge-rs-valkey-x64", "must not exist")])
        self.assertIn("no public tag names an architecture", checker.failures[0].detail)

    def test_an_arch_named_latest_tag_on_an_edge_run_fails(self):
        docker = FakeDocker()
        valkey = publish_edge(docker)
        amd64 = docker.manifests[valkey["falkordb"]]["manifests"][0]["digest"]
        docker.push(amd64, "falkordb", "latest-valkey-x64", registries=(HUB,))
        self.assertEqual(failed(check(docker, "edge")), [(f"{HUB}/falkordb:latest-valkey-x64", "must not exist")])

    def test_an_uninspectable_forbidden_name_fails(self):
        docker = FakeDocker()
        publish_edge(docker)
        docker.errors[f"{HUB}/falkordb:edge-rs-valkey-x64"] = judge.InspectError("toomanyrequests")
        checker = check(docker, "edge")
        self.assertEqual(failed(checker), [(f"{HUB}/falkordb:edge-rs-valkey-x64", "must not exist")])

    def test_only_the_arch_tags_left_from_before_the_switch_are_allowed(self):
        self.assertEqual(len(LEFT_OVER), 8)
        docker = FakeDocker()
        valkey = publish_edge(docker)
        for ref, digest in LEFT_OVER.items():
            docker.keep(ref, digest)
        checker = check(docker, "edge")
        self.assertEqual(failed(checker), [])
        unchanged = [r for r in checker.results if r.check == "left from before the switch"]
        self.assertEqual(sorted(r.subject for r in unchanged), sorted(LEFT_OVER))
        self.assertTrue(all(r.detail == "unchanged" for r in unchanged))
        amd64 = docker.manifests[valkey["falkordb"]]["manifests"][0]["digest"]
        docker.push(amd64, "falkordb", "edge-x64", registries=(GHCR,))
        docker.push(amd64, "falkordb-server", "edge-rs-arm64v8", registries=(HUB,))
        self.assertEqual(sorted(failed(check(docker, "edge"))),
                         [(f"{HUB}/falkordb-server:edge-rs-arm64v8", "must not exist"),
                          (f"{GHCR}/falkordb:edge-x64", "must not exist")])

    def test_a_left_over_arch_tag_that_was_pushed_again_fails(self):
        docker = FakeDocker()
        valkey = publish_edge(docker)
        for ref, digest in LEFT_OVER.items():
            docker.keep(ref, digest)
        amd64 = docker.manifests[valkey["falkordb"]]["manifests"][0]["digest"]
        docker.push(amd64, "falkordb", "edge-x64", registries=(HUB,))
        checker = check(docker, "edge")
        self.assertEqual(failed(checker), [(f"{HUB}/falkordb:edge-x64", "left from before the switch")])
        self.assertIn(f"now {amd64}, was {LEFT_OVER[f'{HUB}/falkordb:edge-x64']}", checker.failures[0].detail)

    def test_left_over_arch_tags_may_be_deleted_but_must_be_inspectable(self):
        docker = FakeDocker()
        publish_edge(docker)
        checker = check(docker, "edge")
        self.assertEqual(failed(checker), [])
        deleted = [r for r in checker.results if r.check == "left from before the switch"]
        self.assertEqual(sorted(r.subject for r in deleted), sorted(LEFT_OVER))
        self.assertTrue(all(r.detail == "deleted" for r in deleted))
        docker.errors[f"{HUB}/falkordb:latest-x64"] = judge.InspectError("toomanyrequests")
        self.assertEqual(failed(check(docker, "edge")),
                         [(f"{HUB}/falkordb:latest-x64", "left from before the switch")])

    def test_left_over_tags_are_not_checked_on_ci_runs(self):
        plan = judge.make_plan(DATA, judge.parse_target("rc-pr-1", DATA), servers=set(DATA["servers"]),
                               registries=REGISTRIES)
        self.assertEqual(plan.frozen, {})

    def test_todays_shape_reports_every_failure(self):
        """edge and edge-rs hold Redis on Debian with no labels, and there are
        no -valkey/-redis tags: every one of those is reported."""
        docker = FakeDocker()
        for repo in REPOS:
            today = docker.multi("redis", "debian-trixie", label=None, env=None, browser=browser_for(repo))
            docker.push(today, repo, "edge", "edge-rs")
        checker = check(docker, "edge")
        missing = {r.subject for r in checker.failures if r.check == "exists"}
        self.assertEqual(missing, {f"{registry}/{repo}:{tag}" for registry in (GHCR, HUB)
                                   for repo in REPOS for tag in ("edge-rs-valkey", "edge-rs-redis")})
        holds_redis = [r for r in checker.failures if r.check == "valkey-server --version"]
        self.assertEqual(len(holds_redis), 4)  # 2 repos x 2 platforms, each naming every tag
        self.assertTrue(all("the image holds Redis" in r.detail for r in holds_redis))
        self.assertTrue(all("edge, " in r.subject and "edge-rs on" in r.subject for r in holds_redis))
        self.assertIn("/etc/os-release", checks_failed(checker))
        self.assertIn("alias of", checks_failed(checker))


class ReleaseTest(unittest.TestCase):
    def test_the_contract_passes_with_and_without_moving_latest(self):
        docker = FakeDocker()
        publish_release(docker, "6.0.9", latest=False)
        publish_release(docker, "6.1.0", latest=True)
        publish_release(docker, "6.2.0-rc.1", latest=False)
        moving = check(docker, "6.1.0", latest="6.1.0")
        self.assertEqual(failed(moving), [])
        self.assertIn("follows", {r.check for r in moving.results})
        self.assertEqual(failed(check(docker, "6.0.9", latest="6.1.0")), [])
        self.assertEqual(failed(check(docker, "6.2.0-rc.1", latest="6.1.0")), [])
        self.assertEqual(failed(check(docker, "latest", latest="6.1.0")), [])

    def test_per_os_tags_are_checked_for_server_and_os(self):
        docker = FakeDocker()
        publish_release(docker, "6.1.0", latest=True)
        docker.push(docker.multi("redis", "amazonlinux2023", salt="x"), "falkordb-server",
                    "6.1.0-amazonlinux2023-valkey")
        docker.push(docker.multi("redis", "debian-trixie", salt="y"), "falkordb-server", "6.1.0-rhel9-redis")
        checker = check(docker, "6.1.0", latest="6.1.0")
        by_tag = {}
        for result in checker.failures:
            by_tag.setdefault(result.subject.split(" on ")[0], set()).add(result.check)
        self.assertIn("valkey-server --version", by_tag["falkordb-server:6.1.0-amazonlinux2023-valkey"])
        self.assertEqual(by_tag["falkordb-server:6.1.0-rhel9-redis"], {"/etc/os-release"})

    def test_per_os_tags_only_on_falkordb_server(self):
        plan = judge.make_plan(DATA, judge.parse_target("6.1.0", DATA), servers=set(DATA["servers"]),
                               registries=REGISTRIES, latest_version="6.0.4")
        per_os = [spec for spec in plan.specs if spec.tag.count("-") > 1]
        self.assertEqual({spec.tag for spec in per_os},
                         {"6.1.0-alpine-valkey", "6.1.0-amazonlinux2023-valkey",
                          "6.1.0-rhel9-redis", "6.1.0-amazonlinux2023-redis"})
        self.assertTrue(all(spec.repos == ["falkordb-server"] for spec in per_os))

    def test_a_per_os_tag_on_falkordb_fails(self):
        docker = FakeDocker()
        publish_release(docker, "6.1.0", latest=True)
        docker.push(docker.multi("valkey", "alpine"), "falkordb", "6.1.0-alpine-valkey", registries=(GHCR,))
        checker = check(docker, "6.1.0", latest="6.1.0")
        self.assertEqual(failed(checker), [(f"{GHCR}/falkordb:6.1.0-alpine-valkey", "must not exist")])

    def test_a_per_os_tag_without_its_server_fails(self):
        docker = FakeDocker()
        publish_release(docker, "6.1.0", latest=True)
        image = docker.tags[f"{GHCR}/falkordb-server:6.1.0-amazonlinux2023-redis"]
        docker.push(image, "falkordb-server", "6.1.0-amazonlinux2023", "6.1.0-rhel9", registries=(GHCR,))
        checker = check(docker, "6.1.0", latest="6.1.0")
        self.assertEqual(sorted(failed(checker)),
                         [(f"{GHCR}/falkordb-server:6.1.0-amazonlinux2023", "must not exist"),
                          (f"{GHCR}/falkordb-server:6.1.0-rhel9", "must not exist")])
        self.assertIn("has no name without its server", checker.failures[0].detail)

    def test_old_style_arch_tags_of_a_new_release_fail(self):
        docker = FakeDocker()
        publish_release(docker, "6.1.0", latest=True)
        image = docker.image("redis", "rhel9", "linux/amd64")
        docker.push(image, "falkordb-server", "6.1.0-rhel9-x64", "6.1.0-x64", registries=(HUB,))
        checker = check(docker, "6.1.0", latest="6.1.0")
        self.assertEqual(sorted(failed(checker)), [(f"{HUB}/falkordb-server:6.1.0-rhel9-x64", "must not exist"),
                                                   (f"{HUB}/falkordb-server:6.1.0-x64", "must not exist")])

    def test_a_release_that_pushes_a_left_over_latest_arch_tag_again_fails(self):
        docker = FakeDocker()
        publish_release(docker, "6.1.0", latest=True)
        for ref, digest in LEFT_OVER.items():
            docker.keep(ref, digest)
        amd64 = docker.manifests[docker.tags[f"{HUB}/falkordb-server:6.1.0-valkey"]]["manifests"][0]["digest"]
        docker.push(amd64, "falkordb-server", "latest-x64", registries=(HUB,))
        for target, latest in (("6.1.0", "6.1.0"), ("latest", "6.1.0"), ("6.0.9", "6.1.0")):
            with self.subTest(target=target):
                checker = check(docker, target, latest=latest)
                self.assertIn((f"{HUB}/falkordb-server:latest-x64", "left from before the switch"),
                              failed(checker))

    def test_an_existing_tag_with_an_unreadable_config_is_not_absent(self):
        """A single-platform tag whose config can't be read exists all the same."""
        docker = FakeDocker()
        publish_release(docker, "6.1.0", latest=True)
        single = docker.image("redis", "rhel9", "linux/amd64")
        docker.push(single, "falkordb-server", "6.1.0-rhel9-x64", registries=(HUB,))
        docker.config_errors.add(single)
        checker = check(docker, "6.1.0", latest="6.1.0")
        self.assertIn((f"{HUB}/falkordb-server:6.1.0-rhel9-x64", "must not exist"), failed(checker))
        loaded = checker._load_manifest(f"{HUB}/falkordb-server:6.1.0-rhel9-x64")
        self.assertIsInstance(loaded, judge.InspectError)
        self.assertNotIsInstance(loaded, judge.NotFound)

    def test_a_newest_release_that_did_not_move_latest_fails(self):
        docker = FakeDocker()
        publish_release(docker, "6.0.9", latest=True)
        publish_release(docker, "6.1.0", latest=False)
        checker = check(docker, "6.1.0", latest="6.1.0")
        self.assertEqual(checks_failed(checker), {"follows"})
        self.assertEqual(len(checker.failures), 12)  # 3 latest tags x 2 repos x 2 registries

    def test_a_prerelease_that_moved_latest_fails(self):
        docker = FakeDocker()
        publish_release(docker, "6.1.0", latest=True)
        publish_release(docker, "6.2.0-rc.1", latest=True)
        checker = check(docker, "6.2.0-rc.1", latest="6.1.0")
        self.assertEqual(checks_failed(checker), {"points at 6.1.0"})
        self.assertEqual(len(checker.failures), 12)

    def test_a_prerelease_that_rebuilt_latest_fails(self):
        """latest* with new digests, neither the rc's nor the newest release's."""
        docker = FakeDocker()
        publish_release(docker, "6.1.0", latest=True)
        publish_release(docker, "6.2.0-rc.1", latest=False)
        for repo in REPOS:
            rebuilt = docker.multi("valkey", "ubuntu-26.04", browser=browser_for(repo), salt="rebuilt")
            docker.push(rebuilt, repo, "latest-valkey", "latest")
        checker = check(docker, "6.2.0-rc.1", latest="6.1.0")
        self.assertEqual(checks_failed(checker), {"points at 6.1.0"})
        self.assertEqual(len(checker.failures), 8)  # latest, latest-valkey x 2 repos x 2 registries

    def test_a_prerelease_gets_per_os_tags(self):
        plan = judge.make_plan(DATA, judge.parse_target("6.1.0-rc.1", DATA), servers=set(DATA["servers"]),
                               registries=REGISTRIES, latest_version="6.0.4")
        tags = [spec.tag for spec in plan.specs]
        self.assertIn("6.1.0-rc.1-alpine-valkey", tags)
        self.assertIn("6.1.0-rc.1", tags)
        self.assertNotIn("latest-valkey", tags)

    def test_pins_can_be_overridden_for_an_older_release(self):
        docker = FakeDocker()
        publish_release(docker, "6.0.9", latest=True, valkey_version="9.1.1")
        self.assertEqual(failed(check(docker, "6.0.9", latest="6.0.9", pins={"valkey": "9.1.1"})), [])
        self.assertEqual(checks_failed(check(docker, "6.0.9", latest="6.0.9")),
                         {f"label {judge.SERVER_VERSION_LABEL}", f"env {judge.SERVER_VERSION_ENV}",
                          "valkey-server --version"})

    def test_a_version_mismatch_on_a_release_says_how_to_check_a_re_run(self):
        docker = FakeDocker()
        publish_release(docker, "6.0.9", latest=True, valkey_version="9.1.1")
        checker = check(docker, "6.0.9", latest="6.0.9")
        hinted = [r for r in checker.failures if "pins input valkey=9.1.1, or --pin valkey=9.1.1" in r.detail]
        self.assertEqual({r.check for r in hinted}, {f"label {judge.SERVER_VERSION_LABEL}",
                                                     f"env {judge.SERVER_VERSION_ENV}", "valkey-server --version"})
        self.assertEqual(len(hinted), len(checker.failures))

    def test_no_pins_hint_outside_a_release_or_for_another_server(self):
        docker = FakeDocker()
        publish_edge(docker)
        docker.push(docker.multi("valkey", "ubuntu-26.04", version="9.1.1", browser=True),
                    "falkordb", "edge-rs-valkey", "edge", "edge-rs")
        self.assertFalse(any("pins input" in r.detail for r in check(docker, "edge").failures))
        docker = FakeDocker()
        publish_release(docker, "6.1.0", latest=True)
        docker.push(docker.multi("redis", "debian-trixie", salt="z"), "falkordb-server", "6.1.0-alpine-valkey")
        self.assertFalse(any("pins input" in r.detail for r in check(docker, "6.1.0", latest="6.1.0").failures))

    def test_no_pins_hint_for_another_product_or_a_newer_server(self):
        docker = FakeDocker()
        publish_release(docker, "6.1.0", latest=True)
        valkey_as_redis = docker.multi("redis", "debian-trixie", binary="valkey", binary_version="8.2.0",
                                       browser=False, salt="w")
        docker.push(valkey_as_redis, "falkordb-server", "6.1.0-redis")
        newer = docker.multi("valkey", "alpine", version="9.1.3", salt="n")
        docker.push(newer, "falkordb-server", "6.1.0-alpine-valkey")
        checker = check(docker, "6.1.0", latest="6.1.0")
        self.assertIn("redis-server --version", checks_failed(checker))
        self.assertIn("valkey-server --version", checks_failed(checker))
        self.assertFalse(any("pins input" in r.detail for r in checker.failures))

    def test_a_public_target_needs_to_know_where_latest_points(self):
        with self.assertRaises(ValueError):
            judge.make_plan(DATA, judge.parse_target("edge", DATA), servers=set(DATA["servers"]),
                            registries=REGISTRIES)


class LatestRuleTest(unittest.TestCase):
    RELEASES = (
        {"tag_name": "v4.22.3", "draft": False, "prerelease": False},
        {"tag_name": "v6.0.4", "draft": False, "prerelease": False},
        {"tag_name": "v6.1.0-rc.1", "draft": False, "prerelease": True},
        {"tag_name": "v7.0.0", "draft": True, "prerelease": False},
        {"tag_name": "v6.0.5", "draft": False, "prerelease": True},
    )

    def test_newest_stable_ignores_drafts_and_prereleases(self):
        self.assertEqual(judge.newest_stable(list(self.RELEASES)), "6.0.4")

    def test_where_latest_points_while_publishing_a_release(self):
        cases = {
            None: "6.0.4",          # edge
            "6.0.4": "6.0.4",       # the newest stable release itself
            "6.0.10": "6.0.4",      # newer, but no published GitHub release
            "4.22.4": "6.0.4",      # an older line
            "6.1.0-rc.1": "6.0.4",  # a prerelease by its version
            "6.0.5": "6.0.4",       # marked prerelease on GitHub
            "7.0.0": "6.0.4",       # a draft is not a published release
        }
        for candidate, expected in cases.items():
            with self.subTest(candidate=candidate):
                self.assertEqual(judge.latest_rule(list(self.RELEASES), candidate, "o/r")[0], expected)

    def test_the_reason_is_given(self):
        cases = ("6.1.0-rc.1", "6.0.5", "6.0.10", "7.0.0")
        why = {c: judge.latest_rule(list(self.RELEASES), c, "o/r")[1] for c in cases}
        self.assertIn("6.1.0-rc.1 is a prerelease", why["6.1.0-rc.1"])
        self.assertIn("the GitHub release of 6.0.5 is marked prerelease", why["6.0.5"])
        self.assertIn("6.0.10 has no published GitHub release, so it does not move latest", why["6.0.10"])
        self.assertIn("7.0.0 has no published GitHub release", why["7.0.0"])

    def test_a_published_release_counts_even_beside_a_draft_of_the_same_tag(self):
        releases = [{"tag_name": "v6.1.0", "draft": True, "prerelease": False},
                    {"tag_name": "v6.1.0", "draft": False, "prerelease": False}]
        self.assertEqual(judge.latest_rule(releases, "6.1.0", "o/r"),
                         ("6.1.0", "latest* must point at 6.1.0, the newest stable release of o/r"))

    def test_unreadable_gh_output_is_an_error(self):
        with mock.patch.object(judge.subprocess, "run",
                               return_value=subprocess.CompletedProcess([], 0, "not json\n", "")), \
                self.assertRaises(judge.InspectError):
            judge.github_releases("o/r")

    def test_no_stable_release_is_an_error(self):
        version, why = judge.latest_rule([{"tag_name": "v1.0.0-rc.1", "prerelease": True}], None, "o/r")
        self.assertIsNone(version)
        self.assertIn("no stable release", why)

    def test_github_releases_reads_one_object_per_line(self):
        out = ('{"tag_name":"v6.0.4","draft":false,"prerelease":false}\n'
               '{"tag_name":"v6.1.0-rc.1","draft":false,"prerelease":true}\n')
        with mock.patch.object(judge.subprocess, "run",
                               return_value=subprocess.CompletedProcess([], 0, out, "")) as run:
            releases = judge.github_releases("o/r")
        self.assertEqual([r["tag_name"] for r in releases], ["v6.0.4", "v6.1.0-rc.1"])
        self.assertIn("repos/o/r/releases", run.call_args.args[0])


class CiTest(unittest.TestCase):
    def test_the_contract_passes(self):
        docker = FakeDocker()
        publish_ci(docker, "pr-1")
        self.assertEqual(failed(check(docker, "rc-pr-1")), [])

    def test_a_valkey_only_run_does_not_need_redis(self):
        docker = FakeDocker()
        publish_ci(docker, "pr-1", servers=("valkey",))
        self.assertEqual(failed(check(docker, "rc-pr-1", servers={"valkey"})), [])
        self.assertIn("exists", checks_failed(check(docker, "rc-pr-1")))

    def test_ci_tags_are_checked_on_ghcr_only(self):
        plan = judge.make_plan(DATA, judge.parse_target("rc-pr-1", DATA), servers=set(DATA["servers"]),
                               registries=REGISTRIES)
        self.assertEqual(plan.registries, [GHCR])

    def test_a_ci_tag_on_docker_hub_fails(self):
        docker = FakeDocker()
        publish_ci(docker, "pr-1")
        docker.tags[f"{HUB}/falkordb:rc-pr-1-valkey-x64"] = docker.tags[f"{GHCR}/falkordb:rc-pr-1-valkey-x64"]
        checker = check(docker, "rc-pr-1")
        self.assertEqual(failed(checker), [(f"{HUB}/falkordb:rc-pr-1-valkey-x64", "must not exist")])

    def test_unsuffixed_names_of_a_merge_queue_key_fail(self):
        docker = FakeDocker()
        publish_ci(docker, "mg-abc")
        index = docker.tags[f"{GHCR}/falkordb:rc-mg-abc-valkey"]
        docker.push(index, "falkordb", "rc-mg-abc", registries=(GHCR,))
        docker.push(docker.tags[f"{GHCR}/falkordb:rc-mg-abc-valkey-x64"], "falkordb", "rc-mg-abc-x64",
                    registries=(GHCR,))
        self.assertEqual(sorted(failed(check(docker, "rc-mg-abc"))),
                         [(f"{GHCR}/falkordb:rc-mg-abc", "must not exist"),
                          (f"{GHCR}/falkordb:rc-mg-abc-x64", "must not exist")])

    def test_unsuffixed_names_of_a_merge_queue_key_on_docker_hub_fail(self):
        docker = FakeDocker()
        publish_ci(docker, "mg-abc")
        docker.push(docker.tags[f"{GHCR}/falkordb-server:rc-mg-abc-valkey"], "falkordb-server", "rc-mg-abc",
                    registries=(HUB,))
        self.assertEqual(failed(check(docker, "rc-mg-abc")),
                         [(f"{HUB}/falkordb-server:rc-mg-abc", "must not exist")])

    def test_a_left_out_servers_ci_tags_on_docker_hub_fail(self):
        """--servers valkey for an unlabelled PR still forbids rc-pr-N-redis off GHCR."""
        docker = FakeDocker()
        publish_ci(docker, "pr-1")
        stray = [f"{repo}:rc-pr-1-redis{arch}" for repo in REPOS for arch in ("", "-x64", "-arm64v8")]
        for repo_tag in stray:
            docker.tags[f"{HUB}/{repo_tag}"] = docker.tags[f"{GHCR}/{repo_tag}"]
        self.assertEqual(sorted(failed(check(docker, "rc-pr-1", servers={"valkey"}))),
                         sorted((f"{HUB}/{repo_tag}", "must not exist") for repo_tag in stray))

    def test_unsuffixed_names_of_a_release_rebuild_key_fail(self):
        docker = FakeDocker()
        publish_ci(docker, "rebuild-abc")
        docker.push(docker.tags[f"{GHCR}/falkordb:rc-rebuild-abc-valkey"], "falkordb", "rc-rebuild-abc",
                    registries=(GHCR,))
        self.assertEqual(failed(check(docker, "rc-rebuild-abc")),
                         [(f"{GHCR}/falkordb:rc-rebuild-abc", "must not exist")])

    def test_unsuffixed_names_of_a_pull_request_key_are_allowed(self):
        """A PR whose CI ran before the switch keeps rc-pr-<N> for a while."""
        docker = FakeDocker()
        publish_ci(docker, "pr-1")
        docker.push(docker.tags[f"{GHCR}/falkordb:rc-pr-1-valkey"], "falkordb", "rc-pr-1", registries=(GHCR,))
        docker.push(docker.tags[f"{GHCR}/falkordb:rc-pr-1-valkey-x64"], "falkordb", "rc-pr-1-x64",
                    registries=(GHCR,))
        self.assertEqual(failed(check(docker, "rc-pr-1")), [])

    def test_unsuffixed_names_of_a_pull_request_key_on_docker_hub_fail(self):
        """Only GHCR ever held CI tags; the allowance for old names stops there."""
        docker = FakeDocker()
        publish_ci(docker, "pr-1")
        docker.push(docker.tags[f"{GHCR}/falkordb:rc-pr-1-valkey"], "falkordb", "rc-pr-1", registries=(HUB,))
        docker.push(docker.tags[f"{GHCR}/falkordb:rc-pr-1-valkey-x64"], "falkordb", "rc-pr-1-x64",
                    registries=(HUB,))
        checker = check(docker, "rc-pr-1")
        self.assertEqual(sorted(failed(checker)), [(f"{HUB}/falkordb:rc-pr-1", "must not exist"),
                                                   (f"{HUB}/falkordb:rc-pr-1-x64", "must not exist")])
        self.assertIn(f"CI tags belong in {GHCR} only", checker.failures[0].detail)

    def test_a_per_arch_tag_that_is_not_the_multi_arch_tags_platform_fails(self):
        docker = FakeDocker()
        publish_ci(docker, "pr-1")
        docker.push(docker.image("valkey", "ubuntu-26.04", "linux/arm64", browser=True, salt="rebuilt"),
                    "falkordb", "rc-pr-1-valkey-arm64v8", registries=(GHCR,))
        checker = check(docker, "rc-pr-1")
        self.assertEqual(failed(checker), [(f"{GHCR}/falkordb:rc-pr-1-valkey-arm64v8",
                                            "linux/arm64 of rc-pr-1-valkey")])

    def test_a_per_arch_tag_holding_the_other_arch_fails(self):
        docker = FakeDocker()
        publish_ci(docker, "pr-1")
        docker.push(docker.image("valkey", "ubuntu-26.04", "linux/amd64", browser=True),
                    "falkordb", "rc-pr-1-valkey-arm64v8", registries=(GHCR,))
        checker = check(docker, "rc-pr-1")
        self.assertEqual(failed(checker), [(f"{GHCR}/falkordb:rc-pr-1-valkey-arm64v8", "architectures")])

    def test_per_arch_tags_are_checked_when_the_multi_arch_tag_is_missing(self):
        docker = FakeDocker()
        publish_ci(docker, "pr-1")
        del docker.tags[f"{GHCR}/falkordb:rc-pr-1-valkey"]
        docker.push(docker.image("redis", "debian-trixie", "linux/amd64", browser=True),
                    "falkordb", "rc-pr-1-valkey-x64", registries=(GHCR,))
        checker = check(docker, "rc-pr-1")
        self.assertIn((f"{GHCR}/falkordb:rc-pr-1-valkey", "exists"), failed(checker))
        self.assertIn("valkey-server --version", checks_failed(checker))


class TargetTest(unittest.TestCase):
    def test_targets(self):
        self.assertEqual(judge.parse_target("edge", DATA), judge.Target("edge"))
        self.assertEqual(judge.parse_target("v6.1.0", DATA), judge.Target("release", version="6.1.0"))
        self.assertEqual(judge.parse_target("6.1.0-rc.2", DATA), judge.Target("release", version="6.1.0-rc.2"))
        self.assertEqual(judge.parse_target("rc-mg-abc123", DATA), judge.Target("ci", ci_key="mg-abc123"))

    def test_mistyped_targets_are_refused(self):
        for text in ("6.1.0-valkey", "6.1", "edge-rs", "rc-pr-1-valkey", "rc-pr-1-x64", "rc-", "rc--x"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                judge.parse_target(text, DATA)


class CommandLineTest(unittest.TestCase):
    def refused(self, *argv: str) -> str:
        stderr = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(stderr), \
                mock.patch.object(judge, "github_releases", side_effect=AssertionError("gh called")):
            judge.main(list(argv))
        return stderr.getvalue()

    def run_main(self, docker: FakeDocker, releases, *argv: str) -> tuple[int, str, str]:
        """main() end to end, on the fake registry; returns (exit, stdout, step summary)."""
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.md"
            with mock.patch.object(judge, "Docker", return_value=docker), \
                    mock.patch.object(judge, "github_releases", side_effect=releases), \
                    mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                code = judge.main(list(argv))
            return code, stdout.getvalue(), summary.read_text() if summary.exists() else ""

    def test_main_passes_a_correct_release_and_fails_a_broken_one(self):
        docker = FakeDocker()
        publish_release(docker, "6.1.0", latest=True)
        releases = [[{"tag_name": "v6.1.0", "draft": False, "prerelease": False}]]
        code, out, summary = self.run_main(docker, releases, "6.1.0")
        self.assertEqual(code, 0, out)
        self.assertIn("latest* must point at 6.1.0", out)
        self.assertIn("checks passed", summary)
        del docker.tags[f"{HUB}/falkordb:6.1.0-redis"]
        code, out, summary = self.run_main(docker, releases, "6.1.0")
        self.assertEqual(code, 1)
        self.assertIn(f"FAIL  {HUB}/falkordb:6.1.0-redis  exists: not found", out)
        self.assertIn(f"`{HUB}/falkordb:6.1.0-redis` exists: not found", summary)

    def test_main_fails_a_release_that_moved_latest_without_a_published_github_release(self):
        docker = FakeDocker()
        publish_release(docker, "6.0.4", latest=False)
        publish_release(docker, "6.1.0", latest=True)
        releases = [[{"tag_name": "v6.0.4", "draft": False, "prerelease": False}]]
        code, out, _ = self.run_main(docker, releases, "6.1.0")
        self.assertEqual(code, 1)
        self.assertIn("6.1.0 has no published GitHub release, so it does not move latest", out)
        self.assertIn(f"FAIL  {GHCR}/falkordb:latest-valkey  points at 6.0.4", out)

    def test_main_reports_a_refused_argument_in_the_step_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.md"
            with mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                judge.main(["rc-pr-1-valkey"])
            self.assertIn("Not checked", summary.read_text())

    def test_main_reports_a_bad_tags_file_in_the_step_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "tags.json"
            bad.write_text("{ not json")
            code, _, summary = self.run_main(FakeDocker(), [], "edge", "--latest", "6.0.4", "--tags-file", str(bad))
        self.assertEqual(code, 2)
        self.assertIn("the check crashed: JSONDecodeError", summary)

    def test_main_reports_a_crash_in_the_step_summary(self):
        with mock.patch.object(judge.Checker, "check", side_effect=RuntimeError("boom")):
            code, _, summary = self.run_main(FakeDocker(), [], "rc-pr-1")
        self.assertEqual(code, 2)
        self.assertIn("the check crashed: RuntimeError('boom')", summary)

    def test_main_fails_latest_while_the_newest_release_has_no_images(self):
        docker = FakeDocker()
        publish_edge(docker)
        publish_release(docker, "6.0.4", latest=True)
        releases = [[{"tag_name": "v6.1.0", "draft": False, "prerelease": False},
                     {"tag_name": "v6.0.4", "draft": False, "prerelease": False}]]
        code, out, summary = self.run_main(docker, releases, "edge")
        self.assertEqual(code, 1)
        self.assertIn("6.1.0 has no images here yet, so latest* cannot be checked; "
                      "re-run this check once 6.1.0's image jobs have finished", out)
        self.assertIn("re-run this check once 6.1.0's image jobs have finished", summary)

    def test_main_reports_an_unknown_latest_in_the_step_summary(self):
        code, _, summary = self.run_main(FakeDocker(), judge.InspectError("gh: no token"), "latest")
        self.assertEqual(code, 2)
        self.assertIn("Not checked", summary)
        self.assertIn("gh: no token", summary)

    def test_servers_is_refused_for_public_tags(self):
        self.assertIn("--servers applies to CI targets only", self.refused("edge", "--servers", "valkey"))

    def test_a_merge_queue_key_is_checked_for_every_server(self):
        self.assertIn("rc-mg-abc has every server's images; omit --servers or name them all",
                      self.refused("rc-mg-abc", "--servers", "valkey"))
        docker = FakeDocker()
        publish_ci(docker, "mg-abc")
        code, out, _ = self.run_main(docker, [], "rc-mg-abc", "--servers", "redis,valkey")
        self.assertEqual(code, 0, out)

    def test_a_pull_request_key_may_be_checked_for_valkey_alone(self):
        docker = FakeDocker()
        publish_ci(docker, "pr-1", servers=("valkey",))
        code, out, _ = self.run_main(docker, [], "rc-pr-1", "--servers", "valkey")
        self.assertEqual(code, 0, out)

    def test_a_prerelease_cannot_be_named_latest(self):
        self.assertIn("a prerelease never moves latest", self.refused("6.1.0-rc.1", "--latest", "6.1.0-rc.1"))

    def test_a_malformed_pin_is_refused(self):
        self.assertIn("--pin wants SERVER=X.Y.Z", self.refused("edge", "--pin", "valkey=9.1.1,redis=8.10.1"))

    def test_a_bad_target_is_refused_before_asking_github(self):
        self.assertIn("unknown target", self.refused("6.1.0-valkey"))

    def test_short_error_names_the_missing_binary(self):
        docker = FakeDocker()
        image = docker.image("redis", "debian-trixie", "linux/amd64")
        code, out = docker.run(f"repo@{image}", "linux/amd64", "valkey-server", ["--version"])
        self.assertEqual(code, 127)
        self.assertEqual(judge._short_error(out),
                         'exec: "valkey-server": executable file not found in $PATH')


class DockerTest(unittest.TestCase):
    """The real Docker class, with the docker CLI replaced by scripted replies."""

    def scripted(self, *replies: tuple[int, str, str]) -> tuple[judge.Docker, list[float], list]:
        sleeps: list[float] = []
        calls: list = []
        docker = judge.Docker(sleep=sleeps.append)

        def fake(args, timeout):
            calls.append(args)
            code, out, err = replies[len(calls) - 1]
            return subprocess.CompletedProcess(args, code, out, err)

        docker._docker = fake  # type: ignore[method-assign]
        return docker, sleeps, calls

    def test_a_missing_tag_is_not_found_at_once(self):
        docker, sleeps, calls = self.scripted((1, "", "ERROR: ghcr.io/falkordb/falkordb:x: not found"))
        with self.assertRaises(judge.NotFound):
            docker.manifest("ghcr.io/falkordb/falkordb:x")
        self.assertEqual((len(calls), sleeps), (1, []))

    def test_a_missing_credential_helper_is_an_error_not_a_missing_tag(self):
        error = 'error getting credentials - err: exec: "docker-credential-x": executable file not found in $PATH'
        docker, sleeps, _ = self.scripted(*[(1, "", error)] * 4)
        with self.assertRaises(judge.InspectError):
            docker.manifest("docker.io/falkordb/falkordb:edge")
        self.assertEqual(sleeps, [2, 2, 2])

    def test_a_rate_limit_is_waited_out(self):
        docker, sleeps, _ = self.scripted((1, "", "429 Too Many Requests"), (0, '{"digest": "sha256:1"}', ""))
        self.assertEqual(docker.manifest("docker.io/falkordb/falkordb:edge"), {"digest": "sha256:1"})
        self.assertEqual(sleeps, [30])

    def test_unreadable_output_is_an_error(self):
        docker, _, _ = self.scripted((0, "not json", ""))
        with self.assertRaises(judge.InspectError):
            docker.manifest("ghcr.io/falkordb/falkordb:edge")

    def test_a_timeout_is_a_failed_command_not_a_crash(self):
        docker = judge.Docker(sleep=lambda _: None)
        with mock.patch.object(judge.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired(["docker"], 300)):
            code, out = docker.run("repo@sha256:1", "linux/arm64", "valkey-server", ["--version"])
            self.assertEqual(code, 124)
            self.assertIn("timed out", out)
            with self.assertRaises(judge.InspectError):
                docker.manifest("ghcr.io/falkordb/falkordb:edge")


if __name__ == "__main__":
    unittest.main()
