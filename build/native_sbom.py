#!/usr/bin/env python3
"""Generate SPDX metadata for the native libraries linked into FalkorDB."""

import configparser
import datetime
import json
import subprocess
import sys
import urllib.parse
import uuid


def inventory(lock_path, compiler, server, server_version):
    lock = configparser.ConfigParser(interpolation=None)
    with open(lock_path, encoding="utf-8") as source:
        lock.read_file(source)
    packages = []
    for section in ("graphblas", "lagraph", "redisearch"):
        dep = lock[section]
        source = urllib.parse.urlparse(dep["url"])
        if source.scheme != "https" or source.netloc != "github.com" or not source.path.endswith(".git"):
            raise ValueError(f"unsupported native dependency source: {dep['url']}")
        source_path = source.path.removeprefix("/").removesuffix(".git")
        packages.append({
            "name": section,
            "SPDXID": f"SPDXRef-{section}",
            "versionInfo": dep["rev"],
            "downloadLocation": f'git+{dep["url"]}@{dep["rev"]}',
            "sourceInfo": f'Pinned source: {dep["pin"]}',
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "externalRefs": [{
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": f'pkg:github/{source_path}@{dep["rev"]}',
            }],
        })
        if server not in ("redis", "valkey") or not server_version:
            raise ValueError(f"invalid server or version: {server} {server_version}")
        owner = "redis" if server == "redis" else "valkey-io"
        server_purl = f"pkg:github/{owner}/{server}@{urllib.parse.quote(server_version, safe='')}"
        packages.append({
            "name": server,
            "SPDXID": "SPDXRef-server",
            "versionInfo": server_version,
            "downloadLocation": f"https://github.com/{owner}/{server}/releases/tag/{urllib.parse.quote(server_version, safe='')}",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "externalRefs": [{
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": server_purl,
            }],
        })
    # The toolchain builds libomp from the version printed by its compiler.
    version = subprocess.run([compiler, "-dumpversion"], check=True, capture_output=True, text=True).stdout.strip()
    if not version:
        raise ValueError(f"{compiler} returned an empty version")
    packages.append({
        "name": "libomp",
        "SPDXID": "SPDXRef-libomp",
        "versionInfo": version,
        "downloadLocation": f"https://github.com/llvm/llvm-project/releases/tag/llvmorg-{version}",
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": "NOASSERTION",
        "copyrightText": "NOASSERTION",
        "externalRefs": [{
            "referenceCategory": "PACKAGE-MANAGER",
            "referenceType": "purl",
            "referenceLocator": f"pkg:github/llvm/llvm-project@llvmorg-{version}?subpath=openmp",
        }],
    })
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "falkordb-native-dependencies",
        "documentNamespace": f"https://falkordb.com/spdx/native-dependencies/{uuid.uuid4()}",
        "creationInfo": {
            "creators": ["Tool: falkordb-native-sbom"],
            "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "packages": packages,
        "relationships": [
            {"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES",
             "relatedSpdxElement": p["SPDXID"]}
            for p in packages
        ],
    }


if __name__ == "__main__":
    json.dump(inventory(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]), sys.stdout, indent=2)
    print()
