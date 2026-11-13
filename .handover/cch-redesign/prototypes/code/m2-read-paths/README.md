# m2-read-paths: reading the CCH topology through GraphBLAS

**Backs:** `../../m2-read-paths.md`.

**Came from:** the scratch git worktree `m2/wt` and the logs `m2/run1.log` and `m2/run2.log`. `m2/m2_bench.rs` was a byte-identical copy of the bench.

## What it measures

Three ignored tests in one file:
- **`m2_read_paths` (part 1).** Customization and 10,000 queries on SNAP roadNet-CA, reading `up(r)` four ways:
  - today's `customize()` and `Cch::query`;
  - (a) a copy of today's code over the flat arrays;
  - (b1) GraphBLAS row iterators through `matrix.rs`'s `Iter<Uint64Extract>`;
  - (b2) raw GxB iterator calls.

  It checks every metric slot and every query answer against today's code.
- **`m2_readonly_load` (part 2).** Loads Rust-owned CSR arrays into a `GrB_Matrix` read-only (`GxB_Vector_load` + `GxB_IS_READONLY` + `GxB_load_Matrix_from_Container`), runs a transpose and a masked `mxm` on it, and checks the arrays were neither copied nor freed by GraphBLAS. It includes the deliberate break: the same arrays loaded with `GrB_DEFAULT` handling.
- **`m2_readonly_peak` (part 2).** The same, started from a dump of the u32 arrays, so that the peak footprint is the arrays' own.

## Files

| Path | What it is |
|---|---|
| `m2.patch` | Against base `31ca1b802e4f876f1e4dc32b5b7b58918ead3190` (`origin/cch/08c-build-catch-up`; also contained in `origin/cch/09-procedure` and `origin/cch/10-rdb-v20`). It adds `graph/src/index/falkordb/cch/m2_bench.rs` and the `#[cfg(test)] mod m2_bench;` line in `cch/mod.rs`. |
| `results/run1.log`, `run2.log` | The two part-1 runs the report's tables come from, with `/usr/bin/time -l` output at the end. |

## Rebuild

```bash
git fetch origin cch/08c-build-catch-up
git worktree add --detach ../m2 31ca1b802e4f876f1e4dc32b5b7b58918ead3190
cd ../m2
git apply /path/to/prototypes/code/m2-read-paths/m2.patch
git submodule update --init --recursive   # unless the native-deps cache already has these pins
export CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++   # macOS
export CARGO_TARGET_DIR=$PWD/target-m2
cargo test --profile bench -p graph --features index-falkordb --lib --no-run
```

- The `bench` profile at this base inherits `release` with `lto = false` and full debug info. These are not LTO numbers.
- `--list --ignored` shows `m2_read_paths`, `m2_readonly_load` and `m2_readonly_peak` under `index::falkordb::cch::m2_bench::`.
- The patch applies cleanly and reproduces the scratch files byte for byte (checked on 2026-11-13). It was not rebuilt for the handoff.

## Data

`M2_ROAD` is the decompressed SNAP `roadNet-CA.txt`; see `../README.md`. The bench's loader gives 1,971,281 nodes, isolated ones included, and 5,533,214 directed entries. The metric is integer weights 1..100, hashed per direction.

## Run

```bash
M2_ROAD=/data/roadNet-CA.txt /usr/bin/time -l "$BIN" --ignored --exact \
    index::falkordb::cch::m2_bench::m2_read_paths --nocapture --test-threads=1 > run1.log 2>&1

M2_ROAD=/data/roadNet-CA.txt M2_DUMP=/tmp/up_u32.bin "$BIN" --ignored --exact \
    index::falkordb::cch::m2_bench::m2_readonly_load --nocapture
M2_DUMP=/tmp/up_u32.bin "$BIN" --ignored --exact \
    index::falkordb::cch::m2_bench::m2_readonly_peak --nocapture
```

| Variable | Meaning |
|---|---|
| `M2_ROAD` | Required for `m2_read_paths` and `m2_readonly_load`. |
| `M2_REPS` | Customization and query reps. Default 3. |
| `M2_QUERIES` | Query pairs. Default 10,000. |
| `M2_GB_THREADS` | GraphBLAS threads in part 1. Default 1. Part 2 sets none, so GraphBLAS uses its default (10 on the M5). |
| `M2_BREAK_IDS=1` | Break: swap two arc ids in every 1,000th row of the matrix. The report ran it with 2,000 queries, and it gave 3,166 differing slots and 224 distance mismatches. |
| `M2_BREAK=<any>` | Break: order by node id instead of feral's ND. |
| `M2_DUMP` | `m2_readonly_load` writes the u32 arrays there; `m2_readonly_peak` reads them. Delete the file afterwards. |
| `M2_BURBLE=1` | GraphBLAS burble in `m2_readonly_peak`. |

## Expected output

Part 1 ends with:
- `CUSTOMIZE median of 3: today … ms; A … ms (relax …); B1 …; B2 …`;
- the instructions line;
- `distance mismatches vs today 0, path mismatches 0` for A, B1 and B2;
- `QUERY best-of-3 per query, <variant>: p50 … p95 … p99 … max …`.

In the two runs, on the M5:
- today's `customize()` took 721 and 753 ms;
- query p99 was 524 µs for today and 524 µs for A, against 1,274 µs for B1 and 904 µs for B2.

Part 2 prints a `[<case>]` block per case:
- the load, with `IS_READONLY` and `memoryUsage(A)`;
- the transpose;
- the masked `mxm`, whose sum must equal the triangle count (`OK`);
- `memoryUsage(A)` after the operations;
- whether the arrays stayed unchanged, and `malloc_size` before and after `GrB_Matrix_free`.

Its numbers are in the report only, because its logs were not kept.

## Left out

- The part-2 logs, which were not saved.
- The decompressed `roadNet-CA.txt` (`m2/data/`).
- The target directory and the u32 dump, both deleted at the time.
