# m1-audit: the code and burble audit that planned m1-optimize

**Backs:** no report of its own. It was a read-only audit of harness A's bench (`../m1-reuse-cost`), run just before m1-optimize. Its probes try formulations that `../../m1-optimize.md` later adopted:
- in-place B2 levels and fine split;
- matches carried in the gather vector;
- a packed `C`;
- the row/column cap mask.

That report cites the audit once ("the audit estimated about 27 ms per depth 0"). The audit also made the C METIS inputs, with `to_nd_bin.py` (now in `../cmetis`).

**Came from:** the scratch folder `m1audit`. Its `wt/` is an rsync of a worktree at the base with harness A's bench (`m1rcA`), plus audit probes.

## What it measures

The probes run beside the official timers and do not change them. Each probe compares a cheaper formulation with the official call and checks that the results are equal:
- **`M1_AUDIT=match`:** matching with `assign`-with-accumulate, with `emult`+`add`, and the unmatched pattern;
- **`M1_AUDIT=contract`:** the contraction with `C⟨¬I⟩`, without the `C·A` wait, and with a packed `C`;
- **`M1_AUDIT=inplace`:** B2 levels and the fine split as in-place delete and insert;
- **`M1_AUDIT=zmatch`:** matches carried in the gather vector;
- **`M1_AUDIT=cap`:** the cap mask by row and column scales.

Other switches:
- `M1_AUDIT_BREAK=<probe>`: breaks a probe, so its equality check must fail;
- `M1_BURBLE_DEPTHS=0,1,13`: burble at chosen depths only;
- `M1_NAMED_OPS=1` with `M1_JIT=on M1_JIT_CACHE=<dir>`: user operators created with names and C definitions, so the JIT can compile them;
- per-call instruction counts on the `AUDITL` lines.

## Files

| Path | What it is |
|---|---|
| `m1audit.patch` | Against base `7df8e588727f47619d9567d111c7fc26a2208caf`. It holds the audit's `m1_coarsen_bench.rs` and `m1_reuse_bench.rs` in full, and the `mod.rs` line. It applies on its own, without `m1rcA.patch`. |
| `scripts/patch_coarsen.py`, `patch_reuse.py`, `patch3.py`, `patch4.py`, `patch5.py` | The edits that turned harness A's sources into the audit's, as exact string replacements, in that order. The patch above already contains their result. They are kept to show what was added. |
| `scripts/probes.py LOG`, `percall.py LOG [LOG2] [threads]`, `parse_burble.py LOG` | The parsers: probe totals, per-call tables (two logs side by side, with a hierarchy-identity check), and burble attributed to (depth, path, call). |
| `scripts/run.sh`, `run2.sh` | `run.sh LOG TEST VAR=VAL…`, with a 12 GB RSS watchdog. `run2.sh` also takes `BIN=`. |
| `results/*.txt` | The parsed summaries: `probes_ca.txt`, `instr_ca_summary.txt`, `burble_ca{,2}_t{1,4}.txt`, `jitcmp_pa.txt` (named operators with JIT on, against JIT RUN), `breaks_summary.txt`, and `cmetis.log`. |
| `results/raw-logs.tar.xz` | The raw logs: `probes{,2}_ca.log`, `instr_ca.log`, `burble_ca{,2}.log`, `burble_pa_probes.log`, `book_ca.log`, `jit_warm_pa.log`, `jitcmp_{on,run}_pa.log`, the smoke logs and `breaks/`. |

## Rebuild and run

The build is the same as `../m1-reuse-cost`, with `m1audit.patch`. The test is `graph::graphblas::m1_coarsen_bench::reuse::m1_reuse`. The settings from the logs' `ENV` lines:

```bash
# probes, CA, 1 and 4 threads
M1_GRAPH=/data/roadNet-CA.txt M1_THREADS=1,4 M1_RUNS=3 M1_FERAL=1 M1_CT=1 M1_JIT=run \
  M1_AUDIT=match,contract,inplace "$BIN" --ignored --exact graph::graphblas::m1_coarsen_bench::reuse::m1_reuse --nocapture
# instructions per probe group, CA, 1 thread
M1_THREADS=1 M1_RUNS=3 M1_FERAL=0 M1_CT=1 M1_JIT=run M1_AUDIT=match,contract,inplace,zmatch,cap …
# burble at depths 0, 1 and 13
M1_THREADS=1,4 M1_RUNS=1 M1_FERAL=0 M1_CT=1 M1_JIT=run M1_BURBLE=1 M1_BURBLE_DEPTHS=0,1,13 …
# named operators under the JIT, PA, depths 0-2
KMP_DUPLICATE_LIB_OK=TRUE M1_THREADS=1 M1_RUNS=3 M1_FERAL=0 M1_CT=1 M1_JIT=on M1_JIT_CACHE=/tmp/jitcache M1_NAMED_OPS=1 M1_DEPTHS=2 …
```

Expected output:
- `AUDIT…` lines beside the bench's own;
- `breaks_summary.txt` lists, for each probe break, the `AUDIT CHECK FAILED` message it must produce, and `none: EXIT 0` for the control.

The patch applies cleanly and reproduces the scratch files byte for byte (checked on 2026-11-13). It was not rebuilt for the handoff.

The scripts hard-code the scratch path, `/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1audit`, as `A`.

## Left out

- **The frozen binary** `graph-test`.
- **`cmetis/`:** the C METIS inputs and permutations (163 MB of data). See `../cmetis` for the converter and checksums.
- **`build*.log`, `OWNER.txt`, and the `jitcache` directory.**
