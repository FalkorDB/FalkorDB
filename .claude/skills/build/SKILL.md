---
name: build
description: Build FalkorDB (Rust code plus its native GraphBLAS/RediSearch dependencies), and run formatting/lint checks. Use when asked to build or compile the project, set up native dependencies for the first time, fix a build/compile/lint error, or run cargo fmt/clippy.
allowed-tools: Bash
---

# Build

FalkorDB-rs is a Redis module: the Rust crates link against two native C
libraries (GraphBLAS and RediSearch) that must be compiled and installed
*before* `cargo build` will succeed.

## 1. First-time native dependency setup

Only needed once per machine/container (skip if `cargo build` already works).

```bash
./graphblas.sh    # clones, builds (static, PIC) and installs GraphBLAS v10.5.0-beta.1 + LAGraph
./redisearch.sh   # clones and builds RediSearch (static) into redisearch/RediSearch/bin
```

If a script fails, read it before retrying by hand — `graphblas.sh` documents
the exact `cmake` flags it uses (`-DGRAPHBLAS_COMPACT=OFF`,
`-DCMAKE_POSITION_INDEPENDENT_CODE=ON`, static build, shared install prefix)
and has a `--skip-graphblas` flag to reuse an already-installed
`libgraphblas.a` while iterating on the LAGraph step.

## 2. Build

```bash
cargo build            # debug build -> target/debug/libfalkordb.{so,dylib}
cargo build --release  # release build -> target/release/libfalkordb.{so,dylib}
```

## 3. Format & lint

```bash
cargo fmt --all -- --check
CXX=clang++ cargo clippy --all-targets
```

- `CXX=clang++` is required for clippy (and any build invoking the GraphBLAS
  FFI bindings under `graph/src/graph/graphblas/`, whose bindgen output is
  `graphblas/mod.rs`) — without it, clippy fails trying to compile the C shims.
- Format check uses the project's `rustfmt.toml` (vertical function-parameter
  layout). Run `cargo fmt --all` (no `--check`) to auto-fix.
- Run format + build + clippy together, in that order, for a full check —
  fail fast on formatting before spending time on a full compile.
- For a stricter, opt-in clippy pass (pedantic/nursery/cargo lints, plus an
  auto-fix mode) see the `lint` skill.

## Notes

- CI (`.github/workflows/rust-pr.yml`) runs the same `cargo fmt --all --check`
  and `CXX=clang++ cargo clippy --all-targets` (pinned to a specific
  `clang++` version in CI containers; any reasonably recent local `clang++`
  works).
- Cross-platform release *artifacts* (Linux `.so`, macOS `.dylib`) and
  multi-arch Docker image publishing are handled entirely in CI — these need
  GH-hosted secrets/registries and are not meant to be reproduced locally.

If any step fails, report the errors clearly and help fix them.
