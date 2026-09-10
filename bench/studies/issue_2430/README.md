# Issue #2430: the two-state multi-edge point read — cause found

A bound multi-edge point read lands in one of two cost states, selected
non-monotonically by graph size.

**Cause: `me`'s delta-plus is non-empty.** When the multi-edge id matrix carries
any pending entries, every multi-edge row read must consult the delta layer
instead of reading the committed base alone. That costs a flat **~1,200
instructions per read**, and the cost does not depend on how much the delta
holds — one pending id costs the same as eight thousand.

That accounts for all three of the issue's puzzles:

| observation | explanation |
| --- | --- |
| two states, not a curve | the delta is empty or it isn't — binary |
| selected by graph size | size decides where the fold policy last fired relative to the final write |
| *non-monotonically* | which is not a function of size, so neither is the state |

## The evidence, in order

| measurement | what it showed |
| --- | --- |
| `engine_two_states.py` | the two states reproduce on `main`: 8,725 → 9,943 instr/pair, a step of ~1,150 |
| `stage_isolation.py` | the entire step is inside `ExpandInto` (base is flat at ~3,940); and it vanishes when the filler edges use a *different* relationship type, so it is a property of the tensor being read |
| `issue_2430_build_path` | the same logical tensor built incrementally vs in one batch differs by ~1,300 — non-monotonically — and the `me.dp` column is non-zero in exactly the high rows |
| `issue_2430_one_pending_id` | **causal**: adding one pending id to `me`, on a pair the probes never touch, costs ~1,200 per read at every size |

Ruled out along the way: storage format (identical, `m` sparse / `me` hypersparse
at every size), hyper-hash (forcing `GrB_Matrix_wait(MATERIALIZE)` does not move
it), index widths (identical), and the forward deltas (`dp`/`dm` are empty
throughout).

## Why the issue's third hypothesis looked refuted

The issue records "a lingering delta that a read-path fold decision latches but
never flushes" as refuted, because *an unrelated write did not move the number*.
That test is too weak. The fold policy is size-based: a delta of one entry never
meets the threshold, so a write is not obliged to fold `me` and generally will
not. The hypothesis was right; the experiment could not see it.

## What a fix looks like

`eff_get` already short-circuits the `dm` probe when `dm` is empty — the same
trick is missing one level down, where a multi-edge row read consults `me`'s
delta. Two candidates, in increasing ambition:

1. **Skip the delta per row.** `me.dp` is hypersparse and keyed by
   `(compound_key, id)`, so "does the delta touch this row" is one cheap probe
   against building a merged iterator. Rows the delta does not touch — nearly all
   of them, since a delta is small by construction — then read at base cost.
2. **Fold `me` when its delta is trivially small.** The square-root policy is
   tuned for *write* amortisation and has no term for what a resident delta costs
   *readers*. A tiny delta is nearly free to fold and, as measured here,
   expensive to keep.

(1) is the targeted fix; (2) is a question about the fold policy's cost model,
which currently prices only the write side.

## Running these

```sh
python3 bench/studies/issue_2430/engine_two_states.py target/release/libfalkordb.dylib 6650
python3 bench/studies/issue_2430/stage_isolation.py  target/release/libfalkordb.dylib 6660
cargo test --release -p graph issue_2430 -- --ignored --nocapture --test-threads=1
```
