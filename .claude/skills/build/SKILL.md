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

On macOS, point the build at Homebrew clang first: the system clang has no
OpenMP, and GraphBLAS would build single-threaded without saying so.

```bash
export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
```

```bash
git submodule update --init --recursive   # populates deps/{GraphBLAS,LAGraph,RediSearch}
cargo build                               # graph/build.rs builds them via native-deps
```

`cargo build` builds whatever is missing, so this is usually all you need. To
build them ahead of time, or to see what is happening:

```bash
cargo run --manifest-path native-deps/Cargo.toml
```

Results are cached at
`${XDG_CACHE_HOME:-$HOME/.cache}/falkordb/native-deps/<dep>/<key>/`
(`$FALKORDB_DEPS_CACHE` overrides the root), keyed on the submodule revision,
the GB_control patch, the vendored PreJIT kernels, the recipe sources,
`$CC`/`$CXX --version`, the target triple and the OpenMP/sanitizer flavour.

If a build fails, `native-deps/src/recipes/<dep>.rs` documents the exact cmake
flags (`-DGRAPHBLAS_COMPACT=OFF`, `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`, static
build). To build one dep only, name it, e.g. `... -- graphblas`; add
`FALKORDB_NATIVE_DEPS_FORCE=1` to rebuild it even on a cache hit. A cache miss
is diffable: each entry's
`.stamp` records the full key manifest after a `--- manifest ---` marker.

## 2. Build

```bash
cargo build            # debug build -> target/debug/libfalkordb.{so,dylib}
cargo build --release  # release build -> target/release/libfalkordb.{so,dylib}
```

## 3. Format & lint

```bash
cargo fmt --all -- --check
CXX=clang++ cargo clippy -p falkordb-rs -p graph --all-targets -- -D warnings
CXX=clang++ cargo clippy -p falkordb-rs -p graph --all-targets --features index-falkordb -- -D warnings
cargo clippy --all-targets --manifest-path native-deps/Cargo.toml -- -D warnings
```

- The lint set is strict: pedantic, nursery and cargo, from `[workspace.lints]`
  in `Cargo.toml` (and `[lints]` in `native-deps/Cargo.toml`), so a plain
  `cargo clippy` shows the same warnings CI fails on. A lint a crate does not
  follow is allowed in its `lib.rs` with the reason, or locally on the item.
- Name both packages: with none, clippy lints only the root package's targets,
  not graph's tests or build script. The `index-falkordb` run lints the native
  index, which the default build does not compile.
- In `native-deps`, `src/recipes/`, `src/toolchain.rs` and `src/util.rs` feed
  the dependency cache key: a style edit there rebuilds all three native deps
  everywhere. Allow their lints on the `mod` line in `src/lib.rs` instead.

- `CXX=clang++` is required for clippy (and any build invoking the GraphBLAS
  FFI bindings under `graph/src/graph/graphblas/`, whose bindgen output is
  `graphblas/mod.rs`) — without it, clippy fails trying to compile the C shims.
- Format check uses the project's `rustfmt.toml` (vertical function-parameter
  layout). Run `cargo fmt --all` (no `--check`) to auto-fix.
- Run format + build + clippy together, in that order, for a full check —
  fail fast on formatting before spending time on a full compile.
- To auto-fix clippy warnings, see the `lint` skill.

## Notes

- CI (`.github/workflows/rust-pr.yml`) runs the same `cargo fmt --all --check`
  and the clippy commands above (pinned to a specific
  `clang++` version in CI containers; any reasonably recent local `clang++`
  works).
- Cross-platform release *artifacts* (Linux `.so`, macOS `.dylib`) and
  multi-arch Docker image publishing are handled entirely in CI — these need
  GH-hosted secrets/registries and are not meant to be reproduced locally.

If any step fails, report the errors clearly and help fix them.
