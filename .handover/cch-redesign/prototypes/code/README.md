# Prototype code for the CCH redesign

The CCH prototypes ran in temporary scratch folders on Dvir's laptop, which are wiped on reboot. Treat this directory as the only copy of their code. It holds the code, scripts and small results behind every report in `../`, enough to rebuild each bench and re-run each measurement.

Nothing here was committed upstream:
- code that lived in a FalkorDB worktree is a `.patch` against a base commit;
- code that lived in a plain folder is copied as source.

Every subfolder has a README with:
- what it measures and which report it backs;
- the exact rebuild and run steps;
- the environment switches;
- the data it reads;
- the expected output;
- what was left out.

## Index

| Folder | Backs | What it is | Base | Form |
|---|---|---|---|---|
| [`m1-optimize/`](m1-optimize/README.md) | `m1-optimize.md`, with its Verification | The optimized GraphBLAS coarsening bench (capstop: 7.22 → 2.30 s on CA), and the independent verifier's two extra switches | `7df8e5887` | `m1opt.patch`, plus `verifier-on-m1opt.patch` on top |
| [`m1-graphblas-cost/`](m1-graphblas-cost/README.md) | `m1-graphblas-cost.md` | The first GraphBLAS coarsening bench, design §3 as written | `7df8e5887` | `m1gb.patch` |
| [`m1-reuse-cost/`](m1-reuse-cost/README.md) | `m1-reuse-cost.md` (body: harness A; addendum: harness B), `d5-b-verdict.md` | The GraphBLAS cost of reusing depth 0's hierarchy: R, B1, B2 and capstop | `7df8e5887` | `m1rcA.patch`, `m1rcB.patch` |
| [`m1-matching/`](m1-matching/README.md) | `m1-matching-quality.md`, `m1-reuse-quality.md` | The plain-Rust ordering-quality harness: feral's crates with mutual-max matching (stage 1), then reuse and capstop (stage 2) | FalkorDB/feral `f127189` | source tree, plus diffs against upstream |
| [`cmetis/`](cmetis/README.md) | the C METIS columns of `m1-optimize.md` and `m1-matching-quality.md` | The `METIS_NodeND` driver, its build and the input converter | KarypisLab METIS `272d4a9`, GKlib `3b7d61b` | sources and `build.sh` |
| [`m2-read-paths/`](m2-read-paths/README.md) | `m2-read-paths.md` | Read paths for the topology: flat arrays against GxB iterators, and read-only loads | `31ca1b802` | `m2.patch` |
| [`m1-audit/`](m1-audit/README.md) | none of its own; ran before m1-optimize | Probes of cheaper GraphBLAS formulations on harness A | `7df8e5887` | `m1audit.patch` |
| [`m1-profile/`](m1-profile/README.md) | none of its own; ran before m1-optimize | rusage, samply and burble profile of harness A | `7df8e5887` | `m1prof.patch` |
| [`d5-judge/`](d5-judge/README.md) | `d5-b-verdict.md` (its source J) | The recount of stage 2's logs | none | scripts |

**How the GraphBLAS benches descend from each other.** Each is a full copy, so every patch applies on its own:

```
m1-graphblas-cost (m1gb)
├── m1-reuse-cost harness A (m1rcA) ── m1-audit, m1-profile   (probes added to A)
└── m1-reuse-cost harness B (m1rcB) ── m1-optimize (m1opt) ── verifier (m1verify)
m1-matching stage 1 ── stage 2 ── d5-judge reads stage 2's logs
```

## Base commits

- **`7df8e588727f47619d9567d111c7fc26a2208caf`:** `origin/main` on 2026-11-12, "build: Alpine (musl) falkordb-server image running Valkey (#3725)". It is an ancestor of `main`.
- **`31ca1b802e4f876f1e4dc32b5b7b58918ead3190`:** `origin/cch/08c-build-catch-up`, "CCH catch-up: a rebase error restarts the build; five overtaken builds fail it". It is also in `origin/cch/09-procedure` and `origin/cch/10-rdb-v20`, but **not in `main`**. If all three branches are deleted, m2's base goes with them.

Apply a patch with `git apply <patch>` in a worktree at its base; each patch creates its new files. GraphBLAS, LAGraph and RediSearch are submodules built by `graph/build.rs`. Run `git submodule update --init --recursive` unless the native-deps cache already holds the base's pins. On macOS, set `CC`/`CXX` to Homebrew clang (see `CLAUDE.md`).

## Data

Nothing here includes graph data. The three graphs are SNAP's road networks:

| Graph | URL | `.gz` bytes | sha256 of the `.gz` | sha256 of the decompressed `.txt` |
|---|---|---|---|---|
| CA | https://snap.stanford.edu/data/roadNet-CA.txt.gz | 17,892,860 | `383f7b14424530a9e25c392969a9c19dace32ebcfd141f0967a1433498c2acdd` | `04b991d4d5285bda35c699244ccdac9f4b35818c0d0bfb037b1f852be103b540` |
| PA | https://snap.stanford.edu/data/roadNet-PA.txt.gz | 9,945,340 | `450b8733635d887466a2b96b26411f6e62caf7006f8a264f59cf9b5d75cdf549` | `e7e9abdd06b512e47a6e63d056c1d1e0ad2cda14f993bf77c6217526f00579b1` |
| TX | https://snap.stanford.edu/data/roadNet-TX.txt.gz | 12,442,024 | `1bc0ce058546ae66101d57af7c16de37225e37778ad2e3781b4f7bf3a28cb41f` | `b97dac76f713ff7061e4582009a12d13f57c8a4a727338d94726dd5961c33966` |

**What each bench reads, and how to make it:**

| Form | Made by | Read by |
|---|---|---|
| `roadNet-<G>.txt.gz`, as downloaded | `curl -O <URL>` | `m1-matching` (`m1bench` decompresses it itself) |
| `roadNet-<G>.txt` | `gunzip -k roadNet-<G>.txt.gz` | `M1_GRAPH` (m1-graphblas-cost, m1-reuse-cost, m1-optimize, m1-audit, m1-profile) and `M2_ROAD` (m2-read-paths) |
| `<G>.bin`, int64 CSR | `python3 cmetis/to_nd_bin.py roadNet-<G>.txt <G>.bin`. `m1bench export` and m1-profile's `M1_DUMP_ND` write the same bytes. | `cmetis/nd`. Checksums are in `cmetis/README.md`. |
| feral's order | written by the m1-optimize bench at `M1_PERM_CACHE` on its first run | m1-optimize, later runs |

The loaders treat the graphs differently:
- **GraphBLAS benches and `to_nd_bin.py`:** drop self-loops, make the graph symmetric, remove duplicates and isolated nodes, and renumber. CA: n = 1,965,206 with 5,533,214 entries; PA: 1,088,092 and 3,083,796; TX: 1,379,917 and 3,843,320.
- **`m1bench` and m2's loader:** keep the isolated nodes. CA: n = 1,971,281, of which 6,075 are isolated; PA: 1,090,920, of which 2,828 are.

None of these is the stack's 1.54M-node "California" that the design's gates came from; see `../00-results.md`.

## Machine

The reports' runs used an Apple M5 laptop (10 cores: 4 performance, 6 efficiency; 32 GB) under macOS (Darwin 25.6.0), with other sessions running. On this machine on 2026-11-13 the tools were:
- rustc and cargo 1.96.1;
- Homebrew clang 23.1.2 as `CC`/`CXX` for the native dependencies;
- Apple clang 17.0.0 for the C METIS rebuild check.

## Conventions in these copies

- **Scripts are verbatim.** They hard-code the old scratch root, `/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad`, in variables such as `S`, `W`, `D`, `A`, `B` and `P`. Each README names them. Edit them before running.
  - The one edit made: m1-matching's two `Cargo.toml` path dependencies now point inside the folder, so it builds as is. Its README gives the original lines.
- **Raw logs are in `raw-logs.tar.xz`.** Extract with `tar -xJf`, outside the repo: the repo's `.gitignore` ignores `logs/`, `out` and `data`, which is why the folders here are named `results/` and so on.
- **About 1 MB of results per prototype.** Parsed tables and CSVs are plain text; raw logs are compressed. Each README's "Left out" says what was dropped and why.

## What was checked for this copy (2026-11-13)

- **Patches.** Every patch passes `git apply --check` on a fresh worktree at its base. Applying it reproduces the scratch sources byte for byte, and `verifier-on-m1opt.patch` does the same on top of `m1opt.patch`.
- **m1-optimize, rebuilt:**
  - built from a fresh worktree in 114 s, and `--list` shows `graph::graphblas::m1_coarsen_bench::reuse::m1_reuse`;
  - a PA check run (depths 0–3, final configuration, 1 and 4 threads) matched the saved `pa8_base.hash` 13/13, with 0 `HASHDIFF`.
- **m1-matching, rebuilt:** it builds as copied, and on PA `m1bench same` and `m1bench samebase … mm 1` both print true.
- **cmetis:**
  - `build.sh` builds a driver that writes the PA permutation byte-identical to the original binary's;
  - `to_nd_bin.py` reproduces `PA.bin`;
  - `m1bench export` matches it on PA, and `M1_DUMP_ND` matches it on CA, PA and TX.
- **Data.** The decompressed texts the benches read had the sha256 of `gunzip` of the `.gz` files above.
- **d5-judge.** The recount reproduces the verdict's 12/15, 1/15 and 13/15, and the break gives TX 4/5.
- **Not rebuilt:** m1-graphblas-cost, m1-reuse-cost, m1-audit, m1-profile and m2-read-paths. Only their patches were checked.

## What could not be saved

- **Never saved at the time:**
  - m1-graphblas-cost's break-run logs;
  - m2-read-paths' part-2 logs (read-only load and peak);
  - the command that built the C METIS driver. `cmetis/build.sh` is a reconstruction, checked to give identical output.
- **Lost by later edits:** the exact source of the `m1bench-main` binary that ran m1-matching's stage-2 main sweep. It was built 6 minutes before the final edits; only the final sources survive.
- **Left out on purpose:**
  - binaries, target directories, graph data, permutation caches and samply profiles;
  - raw logs over the per-prototype budget;
  - two abandoned folders: `m1rc`, whose runs the report discards, and `m1reuse-cost-instance2-unrun`, which never ran.

  The READMEs list each item.
