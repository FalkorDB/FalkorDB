#!/usr/bin/env python3
"""Tests for publish_tags.py: python3 build/test_publish_tags.py

The expected tags are written out by hand from the contract's table of
public tags, not computed, so a change to the generator cannot change what
it is checked against.
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
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


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


class EdgeTags(unittest.TestCase):
    def test_table(self):
        # Rolling: edge-rs-valkey, edge-rs-redis; edge and edge-rs -> Valkey.
        got = by_source(pt.edge_plan(SHA))
        want = {}
        for repo in ("falkordb", "falkordb-server"):
            want[f"ghcr.io/falkordb/{repo}:rc-mg-{SHA}-valkey"] = both(repo, "edge-rs-valkey", "edge-rs", "edge")
            want[f"ghcr.io/falkordb/{repo}:rc-mg-{SHA}-redis"] = both(repo, "edge-rs-redis")
        self.assertEqual(got, want)

    def test_never_reads_the_unsuffixed_ci_tag(self):
        for step in pt.edge_plan(SHA):
            self.assertRegex(step.source, r":rc-mg-[0-9a-f]{40}-(valkey|redis)$")

    def test_needs_a_full_sha(self):
        for bad in ("abc123", SHA.upper(), SHA + "0", ""):
            with self.assertRaises(pt.PlanError):
                pt.edge_plan(bad)


class ReleaseTags(unittest.TestCase):
    def test_highest_stable_moves_latest(self):
        got = by_source(pt.release_plan("6.1.0", SHA, RELEASES + [release("v6.1.0")]))
        want = {}
        for repo in ("falkordb", "falkordb-server"):
            want[f"ghcr.io/falkordb/{repo}:rc-mg-{SHA}-valkey"] = both(
                repo, "6.1.0-valkey", "6.1.0", "latest-valkey", "latest"
            )
            want[f"ghcr.io/falkordb/{repo}:rc-mg-{SHA}-redis"] = both(repo, "6.1.0-redis", "latest-redis")
        self.assertEqual(got, want)

    def test_prerelease_gets_release_tags_but_never_latest(self):
        # Even when its GitHub release was not marked prerelease, and is the
        # highest there is.
        for listed in (release("v6.1.0-rc.1", prerelease=True), release("v6.1.0-rc.1")):
            got = by_source(pt.release_plan("6.1.0-rc.1", SHA, RELEASES + [listed]))
            want = {}
            for repo in ("falkordb", "falkordb-server"):
                want[f"ghcr.io/falkordb/{repo}:rc-mg-{SHA}-valkey"] = both(repo, "6.1.0-rc.1-valkey", "6.1.0-rc.1")
                want[f"ghcr.io/falkordb/{repo}:rc-mg-{SHA}-redis"] = both(repo, "6.1.0-rc.1-redis")
            self.assertEqual(got, want)

    def test_prerelease_of_the_latest_version_does_not_move_latest(self):
        # 6.1.0-rc.2 cut after 6.1.0 shipped shares 6.1.0's X.Y.Z.
        rels = RELEASES + [release("v6.1.0"), release("v6.1.0-rc.2")]
        self.assertFalse(pt.moves_latest("6.1.0-rc.2", rels))
        plan = pt.release_plan("6.1.0-rc.2", SHA, rels)
        self.assertFalse(any(":latest" in t for t in all_tags(plan)))

    def test_backport_does_not_move_latest(self):
        plan = pt.release_plan("6.0.5", SHA, RELEASES + [release("v6.1.0"), release("v6.0.5")])
        self.assertFalse(any(":latest" in t for t in all_tags(plan)))
        self.assertIn("docker.io/falkordb/falkordb:6.0.5", all_tags(plan))

    def test_no_published_release_does_not_move_latest(self):
        # Pushed over an existing draft: the release stays a draft.
        for listed in ([release("v6.1.0", draft=True)], []):
            plan = pt.release_plan("6.1.0", SHA, RELEASES + listed)
            self.assertFalse(any(":latest" in t for t in all_tags(plan)), listed)

    def test_a_higher_draft_or_prerelease_does_not_block_latest(self):
        rels = RELEASES + [release("v6.2.0", draft=True), release("v6.2.0-rc.1", prerelease=True), release("v6.1.0")]
        self.assertTrue(pt.moves_latest("6.1.0", rels))

    def test_versions_compare_as_numbers(self):
        rels = [release("v6.9.0"), release("v6.10.0")]
        self.assertTrue(pt.moves_latest("6.10.0", rels))
        self.assertFalse(pt.moves_latest("6.9.0", rels))

    def test_malformed_release_names_are_ignored(self):
        rels = [release("nightly"), release("v7"), release("6.1.0")]
        self.assertTrue(pt.moves_latest("6.1.0", rels))

    def test_version_shape(self):
        for bad in ("v6.1.0", "6.1", "6.1.0.1", "6.1.0-", "latest"):
            with self.assertRaises(pt.PlanError, msg=bad):
                pt.release_plan(bad, SHA, [])

    def test_prerelease_cannot_name_a_server_os_or_arch(self):
        # 6.1.0-alpine's Valkey tag would be 6.1.0's Alpine image tag.
        for bad in ("6.1.0-alpine", "6.1.0-redis", "6.1.0-rc.1-valkey", "6.1.0-x64",
                    "6.1.0-arm64", "6.1.0-rhel9", "6.1.0-amazonlinux2023.1"):
            with self.assertRaises(pt.PlanError, msg=bad):
                pt.release_plan(bad, SHA, [])


class RebuildTags(unittest.TestCase):
    def test_table(self):
        got = by_source(pt.rebuild_plan("6.1.0", SHA, latest=False))
        want = {}
        for repo in ("falkordb", "falkordb-server"):
            want[f"ghcr.io/falkordb/{repo}:rc-rebuild-{SHA}-valkey"] = both(repo, "6.1.0-valkey", "6.1.0")
            want[f"ghcr.io/falkordb/{repo}:rc-rebuild-{SHA}-redis"] = both(repo, "6.1.0-redis")
        self.assertEqual(got, want)

    def test_latest_when_asked(self):
        tags = all_tags(pt.rebuild_plan("6.1.0", SHA, latest=True))
        for repo in ("falkordb", "falkordb-server"):
            self.assertLessEqual(both(repo, "latest", "latest-valkey", "latest-redis"), tags)

    def test_ad_hoc_tag(self):
        got = by_source(pt.rebuild_plan("edge-test", SHA, latest=False))
        self.assertEqual(
            got[f"ghcr.io/falkordb/falkordb-server:rc-rebuild-{SHA}-valkey"],
            both("falkordb-server", "edge-test-valkey", "edge-test"),
        )
        self.assertEqual(
            got[f"ghcr.io/falkordb/falkordb-server:rc-rebuild-{SHA}-redis"],
            both("falkordb-server", "edge-test-redis"),
        )

    def test_refusals(self):
        for tag, latest in (
            ("6.1.0-rc.1", True),  # a prerelease never moves latest
            ("edge", False), ("edge-rs", False), ("latest", False),  # other workflows' channels
            ("rc-mg-abc", False),  # CI's namespace
            ("test-x64", False), ("foo-valkey", False), ("alpine", False),
            ("-x", False), ("a:b", False), ("", False),
        ):
            with self.assertRaises(pt.PlanError, msg=tag):
                pt.rebuild_plan(tag, SHA, latest)


PER_OS_TABLE = {
    # Per-OS, release only: Valkey on Alpine and AL2023, Redis on RHEL 9 and AL2023.
    "alpine-valkey", "amazonlinux2023-valkey", "rhel9-redis", "amazonlinux2023-redis",
}


class PerOsTags(unittest.TestCase):
    def test_table(self):
        for version in ("6.1.0", "6.1.0-rc.1"):
            plan = pt.per_os_plan(version)
            want = set()
            for name in PER_OS_TABLE:
                want |= both("falkordb-server", f"{version}-{name}")
            self.assertEqual(all_tags(plan), want)
            self.assertFalse(any(":latest" in t for t in all_tags(plan)))

    def test_each_image_is_every_platform(self):
        for step in pt.per_os_plan("6.1.0"):
            self.assertEqual([arch for arch, _ in step.archives], ["x64", "arm64v8"])

    def _archives(self, root, names):
        for name in names:
            path = root / f"image-{name}" / f"{name}.tar"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")

    def _complete(self):
        return [f"falkordb-server-{n}-{a}" for n in sorted(PER_OS_TABLE) for a in ("x64", "arm64v8")]

    def test_complete_archives(self):
        with tempfile.TemporaryDirectory() as d:
            self._archives(Path(d), self._complete())
            plan = pt.per_os_plan("6.1.0", pt.find_archives(Path(d)))
            for step in plan:
                for arch, path in step.archives:
                    self.assertTrue(path.endswith(f"falkordb-server-{step.name}-{arch}.tar"), path)

    def test_a_missing_archive_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            self._archives(Path(d), [n for n in self._complete() if n != "falkordb-server-rhel9-redis-arm64v8"])
            with self.assertRaisesRegex(pt.PlanError, "rhel9-redis-arm64v8"):
                pt.per_os_plan("6.1.0", pt.find_archives(Path(d)))

    def test_an_archive_it_does_not_know_publishes_nothing(self):
        for extra in ("falkordb-server-rhel9-valkey-x64", "falkordb-server-alpine-valkey-riscv64", "something"):
            with tempfile.TemporaryDirectory() as d:
                self._archives(Path(d), self._complete() + [extra])
                with self.assertRaises(pt.PlanError, msg=extra):
                    pt.find_archives(Path(d))


class ContractRules(unittest.TestCase):
    """Rules that hold for every tag any plan pushes."""

    def plans(self):
        yield pt.edge_plan(SHA)
        yield pt.release_plan("6.1.0", SHA, [release("v6.1.0")])
        yield pt.release_plan("6.1.0-rc.1", SHA, [release("v6.1.0-rc.1")])
        yield pt.rebuild_plan("6.1.0", SHA, latest=True)
        yield pt.rebuild_plan("edge-test", SHA, latest=False)
        yield pt.per_os_plan("6.1.0")
        yield pt.per_os_plan("6.1.0-rc.1")

    def test_no_public_tag_names_an_architecture(self):
        for plan in self.plans():
            for tag in all_tags(plan):
                parts = re.split(r"[.-]", tag.rsplit(":", 1)[1])
                for arch in ("x64", "arm64v8", "amd64", "arm64", "aarch64", "x86_64"):
                    self.assertNotIn(arch, parts, tag)

    def test_server_is_the_last_part_and_no_suffix_means_valkey(self):
        for plan in self.plans():
            for step in plan:
                for tag in step.tags:
                    name = tag.rsplit(":", 1)[1]
                    last = name.rsplit("-", 1)[-1]
                    if last in ("valkey", "redis"):
                        self.assertEqual(last, step.server, tag)
                    else:
                        self.assertEqual(step.server, "valkey", tag)

    def test_every_tag_goes_to_both_registries(self):
        for plan in self.plans():
            for step in plan:
                names = {t.rsplit(":", 1)[1] for t in step.tags}
                repos = {t.rsplit(":", 1)[0].rsplit("/", 1)[1] for t in step.tags}
                self.assertEqual(len(repos), 1, step)
                self.assertEqual(set(step.tags), both(repos.pop(), *names))

    def test_no_tag_is_pushed_twice(self):
        for plan in self.plans():
            tags = [t for step in plan for t in step.tags]
            self.assertEqual(len(tags), len(set(tags)))


class FakeRegistry:
    """Answers the commands execute() runs, and records them."""

    def __init__(self, labels=None, platforms=("linux/amd64", "linux/arm64"), configs=None, moved=()):
        self.labels = labels or {}
        self.platforms = platforms
        self.configs = configs or {}
        self.moved = set(moved)
        self.calls = []
        self.tags = {}

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        if cmd[:4] == ["docker", "buildx", "imagetools", "inspect"]:
            ref = cmd[-1]
            if "--raw" in cmd:
                if ref in self.moved:
                    return "something else"
                return self.tags.get(ref, f"manifest of {ref}")
            server = self.labels.get(ref, ref.rsplit("-", 1)[-1])
            return json.dumps({p: {"config": {"Labels": {pt.SERVER_LABEL: server}}} for p in self.platforms})
        if cmd[:4] == ["docker", "buildx", "imagetools", "create"]:
            digest = cmd[-1].split("@", 1)[1]
            for i, arg in enumerate(cmd):
                if arg == "-t":
                    self.tags[cmd[i + 1]] = self.by_digest[digest]
            return ""
        if cmd[:2] == ["skopeo", "inspect"]:
            return json.dumps(self.configs[cmd[-1].split(":", 1)[1]])
        if cmd[:3] == ["podman", "manifest", "push"]:
            digest_file = Path(cmd[cmd.index("--digestfile") + 1])
            raw = f"index of {cmd[-1]}"
            digest = "sha256:" + hashlib.sha256(raw.encode()).hexdigest()
            self.by_digest[digest] = raw
            self.tags[cmd[-1].removeprefix("docker://")] = raw
            digest_file.write_text(digest + "\n")
            return ""
        if cmd[:2] == ["podman", "manifest"]:
            return ""
        raise AssertionError(f"unexpected command {cmd}")

    @property
    def by_digest(self):
        if not hasattr(self, "_by_digest"):
            self._by_digest = {}
        return self._by_digest

    def digest_of_source(self, ref):
        raw = f"manifest of {ref}"
        digest = "sha256:" + hashlib.sha256(raw.encode()).hexdigest()
        self.by_digest[digest] = raw
        return digest

    def creates(self):
        return [c for c in self.calls if c[:4] == ["docker", "buildx", "imagetools", "create"] or c[:3] == ["podman", "manifest", "push"]]


class Execute(unittest.TestCase):
    def fake(self, plan, **kw):
        fake = FakeRegistry(**kw)
        for step in plan:
            if isinstance(step, pt.Retag):
                fake.digest_of_source(step.source)
        return fake

    def test_retags_from_the_checked_digest_and_checks_every_tag(self):
        plan = pt.edge_plan(SHA)
        fake = self.fake(plan)
        pt.execute(plan, fake)
        creates = fake.creates()
        self.assertEqual(len(creates), len(plan))
        for step, cmd in zip(plan, creates):
            self.assertEqual(cmd[-1], f"{pt.repository(step.source)}@{fake.digest_of_source(step.source)}")
        inspected = {c[-1] for c in fake.calls if "--raw" in c}
        self.assertLessEqual(all_tags(plan), inspected)

    def test_a_bad_source_pushes_nothing(self):
        plan = pt.release_plan("6.1.0", SHA, [])
        # The last source checked holds the wrong server.
        bad = plan[-1].source
        for kw in ({"labels": {bad: "valkey"}}, {"platforms": ("linux/amd64",)}):
            fake = self.fake(plan, **kw)
            with self.assertRaises(pt.PlanError):
                pt.execute(plan, fake)
            self.assertEqual(fake.creates(), [], kw)

    def test_a_tag_that_does_not_hold_the_digest_fails(self):
        plan = pt.edge_plan(SHA)
        fake = self.fake(plan, moved={"docker.io/falkordb/falkordb:edge"})
        with self.assertRaisesRegex(pt.PlanError, "docker.io/falkordb/falkordb:edge "):
            pt.execute(plan, fake)

    def _per_os(self, d, wrong=None):
        root = Path(d)
        configs = {}
        for os_name, server in pt.PER_OS:
            for arch, platform in pt.PLATFORMS.items():
                path = root / f"falkordb-server-{os_name}-{server}-{arch}.tar"
                path.write_bytes(b"")
                os_part, arch_part = platform.split("/")
                configs[str(path)] = {"os": os_part, "architecture": arch_part,
                                      "config": {"Labels": {pt.SERVER_LABEL: server}}}
        if wrong:
            configs[str(root / wrong[0])].update(wrong[1])
        return pt.per_os_plan("6.1.0", pt.find_archives(root)), configs

    def test_per_os_pushes_ghcr_then_copies_its_digest_to_docker_hub(self):
        with tempfile.TemporaryDirectory() as d:
            plan, configs = self._per_os(d)
            fake = FakeRegistry(configs=configs)
            pt.execute(plan, fake, workdir=Path(d))
            pushes = [c for c in fake.calls if c[:3] == ["podman", "manifest", "push"]]
            self.assertEqual(len(pushes), 4)
            for step in plan:
                ghcr, hub = step.tags
                self.assertTrue(ghcr.startswith("ghcr.io/") and hub.startswith("docker.io/"))
                self.assertEqual(fake.tags[ghcr], fake.tags[hub], step.name)

    def test_a_wrong_archive_pushes_nothing(self):
        for wrong in (
            ("falkordb-server-alpine-valkey-arm64v8.tar", {"architecture": "amd64"}),
            ("falkordb-server-rhel9-redis-x64.tar", {"config": {"Labels": {pt.SERVER_LABEL: "valkey"}}}),
        ):
            with tempfile.TemporaryDirectory() as d:
                plan, configs = self._per_os(d, wrong)
                fake = FakeRegistry(configs=configs)
                with self.assertRaises(pt.PlanError, msg=wrong[0]):
                    pt.execute(plan, fake, workdir=Path(d))
                self.assertEqual(fake.creates(), [], wrong[0])


class CommandLine(unittest.TestCase):
    def test_dry_run_prints_and_pushes_nothing(self):
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(pt.main(["edge", "--sha", SHA, "--json"]), 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(len(plan), 4)

    def test_errors_exit_nonzero(self):
        import contextlib
        import io

        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(pt.main(["edge", "--sha", "nope"]), 1)
            self.assertEqual(pt.main(["per-os", "--version", "6.1.0", "--execute"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
