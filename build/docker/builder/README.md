# Builder toolchain Dockerfiles

Per-OS toolchain images used as `FROM` bases by the FalkorDB compiler image
(`build/docker/Dockerfile.compiler`). Absorbed from
[FalkorDB/build-image](https://github.com/FalkorDB/build-image) and extended with
the dependencies RediSearch 8.6 added on top of the previous (`1.2.0`)
generation:

| Prerequisite | Why | Affected OS bases |
|---|---|---|
| CMake `3.31.6` | RediSearch's `boost.cmake` uses `FetchContent`'s `DOWNLOAD_EXTRACT_TIMESTAMP` (added in 3.24). Pinned identically across all OSes via pip so `cmake --version` is byte-identical across the matrix. | all bases |
| libclang / clang dev headers (LLVM **≥ 15**) | `redisearch_rs`'s `bindgen` / `clang-sys` crates need to load `libclang.so` to generate FFI bindings at build time. bindgen accepts any libclang ≥ 9, but we enforce ≥ 15 to keep C++23-era parsing consistent across the matrix. | all bases |
| OpenSSL 3 headers on default compiler include path | RediSearch's `coord/rmr/conn.c` does `#include <openssl/ssl.h>` without going through cmake's `find_package` | `rhel8` only (no OpenSSL 3 package in UBI8 — built from source under `/usr/local/openssl3`) |

## libclang version per OS (first-party, floor ≥ 15)

We pick each OS's newest first-party LLVM major rather than chasing a single
integer across all 7 (the cost of forcing a specific major on UBI8 / AL2023
outweighs the symmetry gain — see commit history for the research). The
contract is: **libclang ≥ 15, from the distro's official repo**. No
`apt.llvm.org`, no EPEL, no COPR, no source builds.

| Image | Base | clang/libclang | Source |
|---|---|---|---|
| `ubuntu` | `ubuntu:22.04` | LLVM 15 | jammy universe (explicit `clang-15 libclang-15-dev`) |
| `debian` | `debian:trixie-slim` | LLVM 19 | trixie main (default) |
| `alpine` | `alpine:3.22` | LLVM 20 | alpine community (explicit `clang20-dev/-libs/-static` + `llvm20-dev/-static`; *-static needed for clang-sys's musl-only static link) |
| `amazonlinux2023` | `amazonlinux:2023` | LLVM 15 | AL2023 default (`clang clang-devel`) |
| `rhel` / `rhel9` | `redhat/ubi9` | LLVM 21 | UBI9 default (`clang clang-devel`) |
| `rhel8` | `redhat/ubi8` | LLVM 21 | UBI8 default + `ubi-8-codeready-builder-rpms` for `clang-devel` |

If you bump a base image (e.g. `alpine:3.22` → `alpine:3.23`, or `ubuntu:22.04` → `ubuntu:24.04`), check that the new default clang still meets the floor and update this table.

## Release workflow

These images are released to **GHCR only** as `ghcr.io/falkordb/falkordb-build:<os>`.

| Event | Tag(s) produced |
|---|---|
| PR opened/updated, touching `build/docker/builder/Dockerfile.<os>` | `:<os>-pr-<N>` (per-arch `:<os>-pr-<N>-x64`, `:<os>-pr-<N>-arm64v8`, plus multi-arch manifest) |
| PR merged to `master` | `imagetools create` retags `:<os>-pr-<N>` → `:<os>` — no rebuild |
| 30 days after last access | swept by `.github/workflows/cleanup-rc-images.yml` daily cron |

## Consumer pattern (`.github/workflows/build.yml`)

Downstream build jobs select between the PR's RC image and the stable image
via a `Compute toolchain image tag` step that consults the `changed_oses`
JSON list emitted by `check-builder-changes`:

```yaml
- name: Compute toolchain image tag
  id: toolchain_tag
  env:
    OS: ${{ matrix.platform.os }}
    CHANGED_OSES: ${{ needs.check-builder-changes.outputs.changed_oses }}
    PR_NUMBER: ${{ github.event.pull_request.number || needs.check-builder-changes.outputs.merged_pr }}
  run: |
    set -euo pipefail
    changed=$(echo "$CHANGED_OSES" | jq -r '.[]' | grep -Fx "$OS" || true)
    if [ -n "$changed" ] && [ -n "$PR_NUMBER" ]; then
      echo "image=ghcr.io/falkordb/falkordb-build:${OS}-pr-${PR_NUMBER}" >> "$GITHUB_OUTPUT"
    else
      echo "image=ghcr.io/falkordb/falkordb-build:${OS}" >> "$GITHUB_OUTPUT"
    fi
```

The resolved tag is then passed into the compiler image build as
`BUILD_IMAGE=${{ steps.toolchain_tag.outputs.image }}`.

This means a PR that modifies `Dockerfile.ubuntu` validates the runtime build
against the new image before merge, and the same image becomes the stable
`falkordb-build:ubuntu` automatically on merge.
