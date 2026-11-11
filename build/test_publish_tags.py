#!/usr/bin/env python3
"""Tests for publish_tags.py: python3 build/test_publish_tags.py

The expected tags are written out by hand from the contract's table of
public tags, not computed, so a change to the generator cannot change what
it is checked against. tag-plan.yml's rehearse job runs the --execute path
against real registries; the Execute tests here cover only its refusals.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import publish_tags as pt  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"


def release(tag, draft=False, prerelease=False):
    return {"tag_name": tag, "draft": draft, "prerelease": prerelease}


def both(repo, *tags):
    """Each tag on GHCR and Docker Hub."""
    return {f"{reg}/{repo}:{t}" for t in tags for reg in ("ghcr.io/falkordb", "docker.io/falkordb")}


def by_source(plan):
    return {step.source: set(step.tags) for step in plan}


def all_tags(plan):
    return {tag for step in plan for tag in step.tags}


# Published GitHub releases a 6.1.0 release could see: C releases share the
# repository, and 6.0.x is the previous Rust line.
RELEASES = [
    release("v4.22.3"),
    release("v6.0.3"),
    release("v6.0.4"),
    release("v4.20.5", prerelease=True),
]

REPOS = ("falkordb", "falkordb-server")


class Tables(unittest.TestCase):
    def test_edge(self):
        # Rolling: edge-rs-valkey, edge-rs-redis; edge and edge-rs -> Valkey.
        want = {}
        for repo in REPOS:
            want[f"ghcr.io/falkordb/{repo}:rc-mg-{SHA}-valkey"] = both(repo, "edge-rs-valkey", "edge-rs", "edge")
            want[f"ghcr.io/falkordb/{repo}:rc-mg-{SHA}-redis"] = both(repo, "edge-rs-redis")
        self.assertEqual(by_source(pt.edge_plan(SHA)), want)

    def test_release_that_moves_latest(self):
        want = {}
        for repo in REPOS:
            want[f"ghcr.io/falkordb/{repo}:rc-mg-{SHA}-valkey"] = both(
                repo, "6.1.0-valkey", "6.1.0", "latest-valkey", "latest"
            )
            want[f"ghcr.io/falkordb/{repo}:rc-mg-{SHA}-redis"] = both(repo, "6.1.0-redis", "latest-redis")
        self.assertEqual(by_source(pt.release_plan("6.1.0", SHA, RELEASES + [release("v6.1.0")])), want)

    def test_prerelease(self):
        # Never latest, even when its GitHub release was not marked prerelease.
        for listed in (release("v6.1.0-rc.1", prerelease=True), release("v6.1.0-rc.1")):
            want = {}
            for repo in REPOS:
                want[f"ghcr.io/falkordb/{repo}:rc-mg-{SHA}-valkey"] = both(repo, "6.1.0-rc.1-valkey", "6.1.0-rc.1")
                want[f"ghcr.io/falkordb/{repo}:rc-mg-{SHA}-redis"] = both(repo, "6.1.0-rc.1-redis")
            self.assertEqual(by_source(pt.release_plan("6.1.0-rc.1", SHA, RELEASES + [listed])), want)

    def test_rebuild(self):
        want = {}
        for repo in REPOS:
            want[f"ghcr.io/falkordb/{repo}:rc-rebuild-{SHA}-valkey"] = both(repo, "edge-test-valkey", "edge-test")
            want[f"ghcr.io/falkordb/{repo}:rc-rebuild-{SHA}-redis"] = both(repo, "edge-test-redis")
        self.assertEqual(by_source(pt.rebuild_plan("edge-test", SHA, latest=False)), want)

    def test_rebuild_with_latest(self):
        tags = all_tags(pt.rebuild_plan("6.1.0", SHA, latest=True, releases=RELEASES + [release("v6.1.0")]))
        for repo in REPOS:
            self.assertLessEqual(both(repo, "6.1.0", "latest", "latest-valkey", "latest-redis"), tags)

    def test_per_os(self):
        # Release only: Valkey on Alpine and AL2023, Redis on RHEL 9 and AL2023.
        for version in ("6.1.0", "6.1.0-rc.1"):
            want = set()
            for name in ("alpine-valkey", "amazonlinux2023-valkey", "rhel9-redis", "amazonlinux2023-redis"):
                want |= both("falkordb-server", f"{version}-{name}")
            plan = pt.per_os_plan(version)
            self.assertEqual(all_tags(plan), want)
            for step in plan:
                self.assertEqual([arch for arch, _ in step.archives], ["x64", "arm64v8"])


class MovesLatest(unittest.TestCase):
    def test_only_the_highest_published_stable_release(self):
        cases = (
            # (version, releases, moves)
            ("6.1.0-rc.2", RELEASES + [release("v6.1.0"), release("v6.1.0-rc.2")], False),  # prerelease
            ("6.0.5", RELEASES + [release("v6.1.0"), release("v6.0.5")], False),  # backport
            ("6.1.0", RELEASES + [release("v6.1.0", draft=True)], False),  # draft only
            ("6.1.0", RELEASES, False),  # no release at all
            ("6.1.0", RELEASES + [release("v6.2.0", draft=True), release("v6.2.0-rc.1", prerelease=True),
                                  release("v6.1.0")], True),  # higher draft/prerelease don't count
            ("6.10.0", [release("v6.9.0"), release("v6.10.0")], True),  # numbers, not strings
            ("6.1.0", [release("nightly"), release("v7"), release("6.1.0")], True),  # odd names ignored
        )
        for version, releases, moves in cases:
            with self.subTest(version=version, releases=releases):
                self.assertEqual(pt.moves_latest(version, releases), moves)
                plan = pt.release_plan(version, SHA, releases)
                self.assertEqual(any(":latest" in t for t in all_tags(plan)), moves)

    def test_rebuild_refuses_latest_where_a_release_would_not_move_it(self):
        rels = RELEASES + [release("v6.1.0"), release("v6.0.5"), release("v6.2.0", draft=True)]
        for tag in ("6.0.5", "6.2.0", "6.1.0-rc.1", "edge-test"):
            with self.subTest(tag=tag), self.assertRaises(pt.PlanError):
                pt.rebuild_plan(tag, SHA, latest=True, releases=rels)

    def test_rebuild_refuses_other_workflows_tags(self):
        for tag in ("edge", "edge-rs", "latest", "Latest", "edge-c", "edge-c-x64", "rc", "rc-mg-abc", "v6.1.0"):
            with self.subTest(tag=tag), self.assertRaises(pt.PlanError):
                pt.rebuild_plan(tag, SHA, latest=False)


class Archives(unittest.TestCase):
    def _archives(self, root, names):
        for name in names:
            path = root / f"image-{name}" / f"{name}.tar"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")

    def _complete(self):
        return [f"falkordb-server-{o}-{s}-{a}" for o, s in pt.PER_OS for a in pt.PLATFORMS]

    def test_a_missing_archive_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            self._archives(Path(d), [n for n in self._complete() if n != "falkordb-server-rhel9-redis-arm64v8"])
            with self.assertRaisesRegex(pt.PlanError, "rhel9-redis-arm64v8"):
                pt.per_os_plan("6.1.0", pt.find_archives(Path(d)))

    def test_an_unknown_archive_publishes_nothing(self):
        for extra in ("falkordb-server-rhel9-valkey-x64", "falkordb-server-alpine-valkey-riscv64", "something"):
            with tempfile.TemporaryDirectory() as d:
                self._archives(Path(d), self._complete() + [extra])
                with self.subTest(extra=extra), self.assertRaises(pt.PlanError):
                    pt.find_archives(Path(d))


class BuildCells(unittest.TestCase):
    """_build-artifacts.yml builds every per-OS archive a release needs, under
    the names find_archives() reads, and publishes no other."""

    def test_cells_build_exactly_the_published_archives(self):
        text = (Path(__file__).resolve().parent.parent / ".github/workflows/_build-artifacts.yml").read_text()
        m = re.search(r"ALL='(\[.*?\])'", text, re.S)
        self.assertIsNotNone(m, "ALL='[...]' not found in _build-artifacts.yml")
        built = set()
        for cell in json.loads(m[1]):
            self.assertEqual(cell["platform"], pt.PLATFORMS[cell["arch"]])
            if cell["image"] and cell["publish"]:
                for server in cell["servers"]:
                    built.add((cell["os"], server, cell["arch"]))
        self.assertEqual(built, {(o, s, a) for o, s in pt.PER_OS for a in pt.PLATFORMS})
        self.assertIn("dest=${{ runner.temp }}/falkordb-server-${{ matrix.os }}-valkey-${{ matrix.arch }}.tar", text)


class FakeRegistry:
    """Answers the imagetools commands a retag runs, and records them."""

    def __init__(self, labels=None, platforms=("linux/amd64", "linux/arm64"), moved=()):
        self.labels = labels or {}
        self.platforms = platforms
        self.moved = set(moved)
        self.by_digest = {}
        self.tags = {}
        self.calls = []

    @staticmethod
    def digest(raw):
        return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        ref = cmd[-1]
        if "inspect" in cmd and "--raw" in cmd:
            raw = "something else" if ref in self.moved else self.tags.get(ref, f"manifest of {ref}")
            self.by_digest[self.digest(raw)] = raw
            return raw
        if "inspect" in cmd:
            source = self.by_digest[ref.split("@", 1)[1]].removeprefix("manifest of ")
            server = self.labels.get(source, source.rsplit("-", 1)[-1])
            return json.dumps({p: {"config": {"Labels": {pt.SERVER_LABEL: server}}} for p in self.platforms})
        if "create" in cmd:
            raw = self.by_digest[ref.split("@", 1)[1]]
            for i, arg in enumerate(cmd):
                if arg == "-t":
                    self.tags[cmd[i + 1]] = raw
        return ""

    def creates(self):
        return [c for c in self.calls if "create" in c]


class Execute(unittest.TestCase):
    def test_a_bad_source_pushes_nothing(self):
        plan = pt.release_plan("6.1.0", SHA, [])
        # The last source checked holds the wrong server, or one platform.
        for kw in ({"labels": {plan[-1].source: "valkey"}}, {"platforms": ("linux/amd64",)}):
            fake = FakeRegistry(**kw)
            with self.subTest(kw=kw), self.assertRaises(pt.PlanError):
                pt.execute(plan, fake)
            self.assertEqual(fake.creates(), [], kw)

    def test_a_tag_that_does_not_hold_the_digest_fails(self):
        plan = pt.edge_plan(SHA)
        fake = FakeRegistry(moved={"docker.io/falkordb/falkordb:edge"})
        with self.assertRaisesRegex(pt.PlanError, "docker.io/falkordb/falkordb:edge "):
            pt.execute(plan, fake)


if __name__ == "__main__":
    unittest.main(verbosity=2)
