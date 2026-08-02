---
name: fuzz
description: Run and manage FalkorDB's cargo-fuzz pipeline for the query runtime (fuzz_target_runtime) - build the fuzz target, run/minimize the corpus, generate coverage, and reproduce a crash artifact. Use when asked to fuzz test the runtime, investigate a fuzzing crash/artifact, or update the fuzz corpus.
allowed-tools: Bash
---

# Fuzz

The fuzz target lives in `fuzz/` (crate `graph-fuzz`, binary
`fuzz_target_runtime`, fuzzing `graph`'s query runtime). `fuzz.sh` runs the
full local sequence below end-to-end.

## 1. Toolchain setup (one-time / per environment)

`cargo-fuzz` requires the nightly toolchain plus LLVM tools:

```bash
rustup default nightly
rustup component add llvm-tools-preview
cargo install cargo-fuzz --version <CARGO_FUZZ_VERSION> --locked
```
Check the currently-pinned `CARGO_FUZZ_VERSION` in `.github/workflows/rust-pr.yml`
so your local `cargo-fuzz` matches CI.

## 2. Build

```bash
cargo build --features fuzz
```

## 3. Seed the corpus

The corpus is seeded from the TCK test run (it generates a wide variety of
valid Cypher queries/inputs):

```bash
mkdir -p fuzz/corpus/fuzz_target_runtime/
source venv/bin/activate 2>/dev/null || source /data/venv/bin/activate   # pytest deps (/data/venv in devcontainer/CI)
TCK_DONE=tck_done.txt pytest tests/tck/test_tck.py -s
```

## 4. Run / minimize / cover

On Linux, first export the RediSearch link workaround CI uses, so the embedded
static libs link without duplicate-symbol errors (not needed / not a valid
flag with macOS's linker):
```bash
export RUSTFLAGS="-C link-arg=-Wl,--allow-multiple-definition"   # Linux/CI only

CXX=clang++ CARGO_PROFILE_RELEASE_LTO=false cargo fuzz run fuzz_target_runtime -- -max_total_time=<seconds>
CXX=clang++ CARGO_PROFILE_RELEASE_LTO=false cargo fuzz cmin fuzz_target_runtime      # minimize the corpus
CXX=clang++ CARGO_PROFILE_RELEASE_LTO=false cargo fuzz coverage fuzz_target_runtime  # generate coverage data
```
Then render an HTML coverage report. Resolve `llvm-cov` from the active Rust
toolchain (via `llvm-tools-preview`) rather than a system one, exactly as CI
does, to avoid version skew; the target triple differs by OS:
```bash
LLVM_COV=$(find "$(rustc --print sysroot)" -name llvm-cov -type f | head -1)

# Linux
"$LLVM_COV" show -format=html -instr-profile=fuzz/coverage/fuzz_target_runtime/coverage.profdata \
  -ignore-filename-regex=\.cargo/registry \
  target/x86_64-unknown-linux-gnu/coverage/x86_64-unknown-linux-gnu/release/fuzz_target_runtime > cov.html

# macOS
"$LLVM_COV" show -format=html -instr-profile=fuzz/coverage/fuzz_target_runtime/coverage.profdata \
  -ignore-filename-regex=\.cargo/registry \
  target/aarch64-apple-darwin/coverage/aarch64-apple-darwin/release/fuzz_target_runtime > cov.html
```

## 5. Reproduce a crash

A failing run writes a repro case to `fuzz/artifacts/fuzz_target_runtime/`.
Reproduce and debug it directly:
```bash
cargo fuzz run fuzz_target_runtime fuzz/artifacts/fuzz_target_runtime/<crash-file>
```
Combine with the `debug` skill (lldb/gdb, panic behavior) once you can
reproduce it — cargo-fuzz binaries are plain executables you can attach a
debugger to directly (no need to load a Redis module for this).

## Notes

- CI runs this on a time budget as its own job and uploads the corpus as a
  build artifact (`fuzzing-corpus`) and any crash artifacts on failure — grab
  those from a failed run's artifacts instead of re-fuzzing from scratch when
  triaging a CI-only failure.
- `CARGO_PROFILE_RELEASE_LTO=false` avoids excessively long fuzz build times;
  `CXX=clang++` is needed for the same GraphBLAS FFI reason as in the `build`
  skill.
