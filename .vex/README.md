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

A statement is `not_affected` when the image as we ship it, including its
documented settings such as `TLS=1`, cannot reach the flaw. When a choice of
yours could, the statement's `impact_statement` names it.

We test the documents with Trivy 0.69.

### If you supply your own TLS key

With `TLS=1` the image generates RSA keys, and these statements assume them.
If you put your own key in the TLS directory, or point redis at one through
`REDIS_ARGS`, two of them depend on its type, both in the RHEL 9 image:

- CVE-2026-54872: use an RSA key, or an EC key on P-256, P-384 or P-521, not
  one on Brainpool, secp256k1 or any other curve.
- CVE-2024-13176: on a host in FIPS mode, do not use a P-521 key.

## Maintaining the documents

[`statements.toml`](statements.toml) is the source. Edit it, then regenerate:

```sh
python3 build/vex/vex.py generate
```

Bump `version` and `timestamp` in `[document]` whenever a statement changes.

### What CI checks

Pull requests and the merge queue check only that the documents match
`statements.toml` (`vex.py generate --check`, in `rust-pr.yml`'s `lint`). Their
image builds still fail on a fixable HIGH or CRITICAL vulnerability, a gate
that never reads the VEX.

The statements themselves are checked against the images every night and at
release. Trivy scans each image with its document applied, and
`build/vex/vex.py check` reads that scan and the image's filesystem. The same
runs also apply the fixable HIGH or CRITICAL gate, since a fix can appear
after an image was built. Pull
requests skip this because an upstream change alone, such as a new CVE or a
package update, can fail it, and would then block every one of them.

- **Nightly** (`vex-nightly.yml`): the published Debian `:edge` images, and the
  RHEL 9 and Amazon Linux 2023 images built from `main`. A failing scheduled
  run opens an issue titled "Nightly VEX check failed", or comments on the open
  one, and the next passing scheduled run closes it.
- **At release** (`release-tag.yml`): the images being released. A failure
  stops them from being published.

To check a branch before merging it, run `vex-nightly.yml` on the branch
(`gh workflow run vex-nightly.yml --ref <branch>`), with `-f tag=rc-pr-<N>` to
check the pull request's Debian images instead of `:edge`. A run started this
way never opens or closes the issue.

The check fails when:

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
