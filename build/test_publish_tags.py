#!/usr/bin/env python3
"""Regression tests for the image promotion SBOM gate."""

import copy
import configparser
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import native_sbom
import publish_tags as publish


class SbomChecks(unittest.TestCase):
    def setUp(self):
        self.repo = "docker.io/falkordb/falkordb"
        self.ref = f"{self.repo}:edge"
        self.index = {"manifests": []}
        self.manifests = {}
        self.statements = {}
        for arch in ("amd64", "arm64"):
            image = f"sha256:{arch}"
            attestation = f"sha256:{arch}-attestation"
            layer = f"sha256:{arch}-sbom"
            self.index["manifests"].extend([
                {"digest": image, "platform": {"os": "linux", "architecture": arch}},
                {"digest": attestation, "platform": {"os": "unknown", "architecture": "unknown"},
                 "annotations": {"vnd.docker.reference.type": "attestation-manifest",
                                 "vnd.docker.reference.digest": image}},
            ])
            self.manifests[attestation] = {
                "layers": [{"digest": layer, "annotations": {
                    "in-toto.io/predicate-type": publish.SPDX_PREDICATE,
                }}],
            }
            self.statements[layer] = {
                "predicateType": publish.SPDX_PREDICATE,
                "subject": [{"digest": {"sha256": arch}}],
                "predicate": {"SPDXID": "SPDXRef-DOCUMENT", "spdxVersion": "SPDX-2.2",
                              "packages": [{"name": name} for name in
                                           ("falkordb-rs", "thiserror", "graphblas",
                                            "lagraph", "redisearch", "libomp")]},
            }

    def runner(self, cmd):
        ref = cmd[-1]
        if ref == self.ref or ref == f"{self.repo}@sha256:index":
            result = self.index
        elif cmd[0] == "crane":
            result = self.statements[ref.split("@", 1)[1]]
        else:
            result = self.manifests[ref.split("@", 1)[1]]
        return json.dumps(result)

    def test_both_platforms_and_digest_reference(self):
        publish.check_sboms(self.ref, self.runner)
        publish.check_sboms(f"{self.repo}@sha256:index", self.runner)

    def test_missing_platform_attestation(self):
        self.index["manifests"].pop()
        with self.assertRaisesRegex(publish.PlanError, "missing SPDX attestation"):
            publish.check_sboms(self.ref, self.runner)

    def test_subject_cannot_disagree_with_annotation(self):
        self.statements["sha256:arm64-sbom"]["subject"][0]["digest"]["sha256"] = "amd64"
        with self.assertRaisesRegex(publish.PlanError, "wrong subject"):
            publish.check_sboms(self.ref, self.runner)

    def test_missing_module_or_rust_dependencies(self):
        self.statements["sha256:amd64-sbom"]["predicate"]["packages"].pop()
        with self.assertRaisesRegex(publish.PlanError, "lacks FalkorDB"):
            publish.check_sboms(self.ref, self.runner)

    def test_legacy_sbom_still_requires_a_valid_digest_bound_statement(self):
        self.statements["sha256:amd64-sbom"]["predicate"]["packages"] = [{"name": "legacy"}]
        publish.check_sboms(self.ref, self.runner, "valkey", require_complete=False)

    def test_missing_server_is_rejected(self):
        with self.assertRaisesRegex(publish.PlanError, "lacks the valkey server"):
            publish.check_sboms(self.ref, self.runner, "valkey")

    def test_duplicate_platform_is_rejected(self):
        self.index["manifests"].append(copy.deepcopy(self.index["manifests"][0]))
        with self.assertRaisesRegex(publish.PlanError, "expected one"):
            publish.check_sboms(self.ref, self.runner)

    def test_manifest_subject_cannot_disagree(self):
        self.manifests["sha256:amd64-attestation"]["subject"] = {"digest": "sha256:arm64"}
        with self.assertRaisesRegex(publish.PlanError, "manifest has the wrong subject"):
            publish.check_sboms(self.ref, self.runner)

    def test_promoted_tag_must_match_checked_index(self):
        step = publish.Retag("ghcr.io/falkordb/falkordb:rc", "valkey", (self.ref,))
        with patch.object(publish, "check_source", return_value="sha256:source"), \
             patch.object(publish, "digest_of", return_value="sha256:other"):
            with self.assertRaisesRegex(publish.PlanError, "differs from the checked source"):
                publish.execute([step], runner=lambda _: "")


class NativeMetadata(unittest.TestCase):
    def test_native_entries_follow_lock_and_resolved_libomp_source(self):
        lock = Path(__file__).resolve().parent.parent / "deps/native-deps.lock"
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "libomp-source-version"
            version_file.write_text("23.1.2\n", encoding="utf-8")
            document = native_sbom.inventory(lock, version_file, "valkey", "9.2.0")
        parsed = configparser.ConfigParser()
        parsed.read(lock)
        identifiers = [p["SPDXID"] for p in document["packages"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        packages = {p["name"]: p for p in document["packages"]}
        self.assertEqual(packages["libomp"]["versionInfo"], "23.1.2")
        self.assertEqual(packages["graphblas"]["versionInfo"], parsed["graphblas"]["rev"])
        self.assertEqual(packages["valkey"]["versionInfo"], "9.2.0")
        self.assertTrue(all(p["externalRefs"][0]["referenceLocator"].startswith("pkg:github/")
                            for p in packages.values()))


if __name__ == "__main__":
    unittest.main()
