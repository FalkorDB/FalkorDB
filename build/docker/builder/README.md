# Builder toolchain Dockerfiles

Per-OS toolchain images used as `FROM` bases by the FalkorDB compiler image
(`build/docker/Dockerfile.compiler`). Absorbed from
[FalkorDB/build-image](https://github.com/FalkorDB/build-image) and extended with
the dependencies RediSearch 8.6 added on top of the previous (`1.2.0`)
generation:

| Prerequisite | Why | Affected OS bases |
|---|---|---|
| CMake ≥ 3.24 | RediSearch's `boost.cmake` uses `FetchContent`'s `DOWNLOAD_EXTRACT_TIMESTAMP` (added in 3.24) | `ubuntu`, `amazonlinux2023` (others already shipped ≥ 3.25) |
| libclang / clang dev headers | `redisearch_rs`'s `bindgen` / `clang-sys` crates need to load `libclang.so` to generate FFI bindings at build time | all bases |
| OpenSSL 3 headers on default compiler include path | RediSearch's `coord/rmr/conn.c` does `#include <openssl/ssl.h>` without going through cmake's `find_package` | `rhel8` only (no OpenSSL 3 package in UBI8 — built from source under `/usr/local/openssl3`) |

## Release workflow

These images are released to **GHCR only** as `ghcr.io/falkordb/falkordb-build:<os>`.

| Event | Tag(s) produced |
|---|---|
| PR opened/updated, touching `build/docker/builder/Dockerfile.<os>` | `:<os>-pr-<N>` (per-arch `:<os>-pr-<N>-x64`, `:<os>-pr-<N>-arm64v8`, plus multi-arch manifest) |
| PR merged to `master` | `imagetools create` retags `:<os>-pr-<N>` → `:<os>` — no rebuild |
| 30 days after last access | swept by `.github/workflows/cleanup-rc-images.yml` daily cron |

## Consumer pattern (`.github/workflows/build.yml`)

Downstream build jobs select between the PR's RC image and the stable image:

```yaml
container: >-
  ${{ needs.check-builder-changes.outputs.<os>_changed == 'true'
      && format('ghcr.io/falkordb/falkordb-build:{0}-pr-{1}', matrix.os, github.event.pull_request.number)
      || format('ghcr.io/falkordb/falkordb-build:{0}', matrix.os) }}
```

This means a PR that modifies `Dockerfile.ubuntu` validates the runtime build
against the new image before merge, and the same image becomes the stable
`falkordb-build:ubuntu` automatically on merge.
