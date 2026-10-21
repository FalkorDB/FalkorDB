# VEX for FalkorDB's container images

Container scanners report CVEs in the operating system packages of our
images, including many that have no fixed package version yet. These
[OpenVEX](https://openvex.dev) documents say, for each of those, whether it
affects FalkorDB as we ship it, and why.

| Document | Images |
|---|---|
| [`falkordb.openvex.json`](falkordb.openvex.json) | `falkordb/falkordb`, `ghcr.io/falkordb/falkordb` |
| [`falkordb-server.openvex.json`](falkordb-server.openvex.json) | `falkordb/falkordb-server`, `ghcr.io/falkordb/falkordb-server`: the Debian image (the default tags), RHEL 9 (`-rhel9-*`) and Amazon Linux 2023 (`-amazonlinux2023-*`) |

## Using a document

Download the document from the release the image comes from (image `6.1.0`
is release `v6.1.0`), and give it to Trivy. Trivy reads `--vex` from a local
file, not a URL:

```sh
VERSION=6.1.0
curl -fsSLO "https://raw.githubusercontent.com/FalkorDB/FalkorDB/v$VERSION/.vex/falkordb.openvex.json"
trivy image --vex falkordb.openvex.json "falkordb/falkordb:$VERSION"
```

Findings a statement covers are left out of the report. Add
`--show-suppressed` to list them with the reason.

Use a document only for the image it is written for. Its statements name the
affected packages (for example `pkg:deb/debian/libc6`), not the image, so
another image that contains the same package would have them applied too.

Each statement has one of these statuses:

- `not_affected`: the CVE does not affect the image. The statement gives the
  reason (`justification`) and explains it (`impact_statement`).
- `affected`: it does. The statement says what to do (`action_statement`).
- `under_investigation`: we have not finished assessing it.

We test the documents with Trivy 0.69.

## Maintaining the documents

[`statements.toml`](statements.toml) is the source. Edit it, then regenerate:

```sh
python3 build/vex/vex.py generate
```

Bump `version` and `timestamp` in `[document]` whenever a statement changes.

### What CI checks

Every image build scans the image with Trivy three times: twice without the
VEX, for the full report and for the gate that fails on a fixable HIGH or
CRITICAL vulnerability, and once with the image's document applied.
`build/vex/vex.py check` then reads the last scan and the image's filesystem. The Debian images are checked in
`_ci-pipeline.yml`'s `security-scan`, the RHEL 9 and Amazon Linux 2023 images
in `_build-artifacts.yml`. The check fails when:

| Failure | What to do |
|---|---|
| A HIGH or CRITICAL finding with no fixed version has no statement | Assess it and add a statement, or remove the package from the image. Lower severities are listed as warnings. |
| A statement does not name every package Trivy reports the CVE in | Add the package to `packages`. |
| A statement names a package Trivy does not report the CVE in | Drop the package from `packages`. If the package is in only some of the images, split the statement by image. |
| A finding a statement hides now has a fixed version | Rebuild so the image gets the fix, then delete the statement. |
| A statement hid nothing | The CVE is gone from the image (fixed, or its package removed): delete the statement. |
| A statement's evidence no longer holds | The claim may now be false. Re-assess it before changing anything. |
| A program listed in `[runtime]` is missing from the image | Update `[runtime]` to match `build/runtime/run.sh`. |

### Writing a statement

```toml
[[statement]]
cve = "CVE-2026-6368"
os = "debian"                       # debian, rhel or amazon
images = ["falkordb", "falkordb-server"]
packages = ["libc-bin", "libc6"]    # every package Trivy reports it in
status = "not_affected"
justification = "vulnerable_code_not_in_execute_path"
impact = """The flaw is in wordexp(), which expands shell-style words. \
No program or library in this image calls it."""
evidence = [{ check = "no-import", symbols = ["wordexp"] }]
notes = "How this was established, and the sources. Not published."
```

`impact` (and `action`, `status_notes`) is published. Write it for a customer:
what the flaw needs, and why the image does not give it.

`evidence` is re-checked against every image the statement covers, on every
build. Use it wherever the claim can be checked mechanically:

| Check | Holds when |
|---|---|
| `{ check = "absent", paths = [...] }` | none of the paths exists |
| `{ check = "contains", path = "...", text = "..." }` | the file exists and contains the text: a setting the claim relies on, say |
| `{ check = "lacks", path = "...", text = "..." }` | the file exists and does not contain the text |
| `{ check = "no-import", symbols = [...] }` | no ELF file in the image has an undefined dynamic reference to any of the symbols. Calls inside one library (inside glibc, say) are invisible to it. `except = [paths]` leaves out files the claim accounts for, such as the `openssl` tool that gen-certs.sh runs. |
| `{ check = "not-run", packages = [...] }` | no file of these packages is a program in `[runtime]`, a library in its `dlopen` list, or what either runs through a script's `#!` line or loads through `DT_NEEDED`. Any other library loaded by name is not seen; account for it in `notes`. |
| `{ check = "no-setid" }` | no file in the image is setuid, setgid or carries file capabilities, so nothing runs with more privileges than its caller |
| `{ check = "version", package = "...", is = "..." }` | the package's installed version is exactly this. Use it when the claim depends on the version, so a package update forces a re-assessment. |

A statement with no `evidence` relies on its `notes` alone, and CI checks only
that it still matches a finding.

`[runtime]` lists what the containers run: `build/runtime/run.sh`, what it
starts, and the browser's entrypoint in the `falkordb` image. When a script
starts a new program, add it there, or `not-run` evidence can pass while the
package runs.
