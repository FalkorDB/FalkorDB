# id_space: how node and relationship ids are handed out, recycled and checked

This project proves what `IdSpace` (`graph/src/graph/id_space.rs`) promises. Since #2846, `IdSpace` is the single authority for ids. The graph keeps two of them, one for nodes and one for relationships. Each owns:

- the live count;
- the set of recycled (free) ids;
- the open batch: the boundary where the current write started, what the write has taken, and, since #3022, what it has released.

User-facing behaviour that depends on it:

- `CREATE` reusing the ids of deleted entities, and never handing one id out twice;
- `DELETE` refusing an id that is already free or was never created;
- commit-time validation (`Graph::validate`), which refuses a write whose ids do not add up;
- `GRAPH.EFFECT` replay on a replica, which must refuse records that create an id that is already live, or act on an entity that is not live (#3022).

The full Lean↔Rust mapping (every Lean name next to the `id_space.rs` line it models) is in the header of [`FalkorIdSpace.lean`](FalkorIdSpace.lean). This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does `verify` accept exactly a consistent batch?**

Yes. `verify_ok_iff` proves `verify` passes exactly when two things hold:

- the count equation holds (`checked`);
- the ids the batch took at or above its entry boundary are precisely `[entry, entry + created)`, with no gaps.

`verify_hole` proves a refusal names the highest id taken. `verify_no_underflow` proves the order of the `||` in the Rust keeps its subtractions from wrapping.

**2. Are `create` and `release` all or nothing?**

Yes.

- `create_spec` proves `create` either passes all its refusals and applies the whole batch, or changes nothing and returns `AlreadyLive` or `IdOutOfRange`.
- `create_out_of_range` proves an id `≥ ID_LIMIT` (`GrB_INDEX_MAX`) is refused before anything is sized; this is the #2892 fix in #2911.
- `release_spec` proves `release` is all or nothing in the same way, judged over every *requested* id. A free id gives `AlreadyRecycled` and a never-created one gives `NeverCreated`.

**3. Can the live count underflow?**

No, in any reachable state. `live_eq` proves `live` is exactly the number of live ids. So `release_no_underflow` follows: once `release`'s refusals pass, `live -= freed.len()` cannot wrap.

**4. Does every reachable state stay consistent, including across versions?**

Yes. `Inv` has four parts:

- the count equation;
- every free id is below the entry boundary or was taken by the batch;
- `u64::MAX` is never taken;
- every released id was live in the batch.

`inv_new` and `inv_restored` prove it holds at the start. `create_succeeds`, `cancel_succeeds` and `release_succeeds` prove every successful mutator keeps it. `roll_inv` proves that rolling a verified batch (`verify` then `open_batch`, which is `Graph::roll_id_batches`) opens a batch equal to `new_version` that satisfies `Inv` again.

**5. Does `reserve` hand out ids nobody else holds?**

`reserve_fresh` proves the reserved ids have three properties, under the precondition documented in Rust:

- they are duplicate-free;
- they are disjoint from what the batch took or already issued;
- each is either a recycled id below the boundary or a fresh one at or above it.

`reserve_then_cancel` proves every reserved id can be cancelled. `lifecycle_fresh` runs the whole path, restore then reserve *n* then create then verify, and proves every step succeeds.

**6. Can a replicated effect act on a deleted entity?**

No (#3022). `refuseNotLive_live` proves `refuse_not_live` accepts exactly the live ids. `refuseNotLive_minus_released` proves the effects `CreateEdge` check, `(src ∪ dst) − released`, admits only ids live now or freed by this same batch. `freed_earlier_refused` proves an id freed by an earlier batch is refused after the roll.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `verify_ok_iff` | `verify` accepts exactly a consistent, gap-free batch | `Verify.lean` |
| `create_spec`, `create_out_of_range` | `create` is all or nothing; ids past `ID_LIMIT` are refused before anything is sized | `Record.lean` |
| `release_spec` | `release` is all or nothing over every requested id | `Record.lean` |
| `live_eq`, `release_no_underflow` | `live` counts the live ids, so releasing never wraps it | `Inv.lean`, `Lifecycle.lean` |
| `create_succeeds`, `cancel_succeeds`, `release_succeeds` | Each successful mutator keeps the invariant | `Inv.lean`, `Lifecycle.lean` |
| `roll_inv` | Rolling a verified batch opens a consistent new one, equal to `new_version` | `Lifecycle.lean` |
| `reserve_fresh`, `reserve_then_cancel` | Reserved ids are fresh and can always be cancelled | `Reserve.lean`, `Inv.lean` |
| `lifecycle_fresh` | Restore → reserve → create → verify succeeds end to end | `Lifecycle.lean` |
| `refuseNotLive_live`, `refuseNotLive_minus_released`, `freed_earlier_refused` | An effect can only act on ids live now or freed by its own batch (#3022) | `Released.lean` |

### How `roll_inv` is proved

1. **Model the sets.** A `RoaringTreemap` is an `IdSet`, a membership function `Nat → Bool` read over the universe `[0, N)`. In the engine `N = 2^64`. In the `#guard` tests at the end of `FalkorIdSpace.lean`, `N = 32`, which lets the Rust unit tests be replayed at small size. The roaring operations are modelled by what they compute on sets:
   - `cnt n s` is `|s ∩ [0, n)|`, which is roaring's `rank(n-1)`;
   - `asc` is `iter()`;
   - `select(k)` is the *k*-th element of `asc`.
2. **Model the state.** `IdSpace` has the five Rust fields: `live`, `recycled`, `eb` (the entry boundary), `taken` and `released`.
3. **Model the methods.** Each method is modelled line by line and returns the state after the call together with its result. An `Inconsistent` from the trailing `checked()` leaves the mutated state behind, as in Rust (the caller discards the MVCC version). Every other refusal returns the state unchanged.
4. **The invariant.** `Inv` combines `checked` with the three structural facts the Rust relies on without checking (see question 4). `live_eq` follows from the count equation by splitting the ids into "below the boundary" and "taken at or above it".
5. **The roll.** A passing `verify` gives the count equation and a gap-free batch (`verify_ok_iff`). `open_batch` then moves the boundary to `bound` and empties `taken` and `released`. Every free id is now below the new boundary, so `Inv` holds again.
6. **Hypotheses.**
   - `cancel_succeeds` and `reserve_then_cancel` assume the id is not `u64::MAX`; `cancel` itself does not check this.
   - `reserve_fresh` takes the Rust doc's precondition about what the batch already holds.
   - The callers (`Graph`, `Pending`, bulk import) are assumed to settle every reservation before `open_batch`. That is not proven here.

## Bugs found

No confirmed bugs in the current `id_space.rs`.

Historical: #2892 (fixed by #2911). Before the fix, `create` refused only `u64::MAX`. A `GRAPH.EFFECT` with id 2^61 asserted in `GrB_Matrix_new`, and `u64::MAX - 1` made `grow_cap` hang. Now `create` refuses every id `≥ ID_LIMIT = GrB_INDEX_MAX` before anything is sized (`create_out_of_range`). `grow_cap`'s side is proven in `proofs/graph_queries` (`growCap_le_max`).

There are no CONDITIONAL rows.

## Files

| File | What's in it |
|---|---|
| `Card.lean` | Finite id sets over `[0, N)` and counting them (`cnt`, `asc`) |
| `Model.lean` | `IdSpace`, `IdSpaceError`, and every method of `id_space.rs` |
| `Sets.lean` | Lemmas about `min`/`max`/`rank`/`len` on the set model |
| `Reserve.lean` | `new`, `restored`, `new_version`, `reclaim_ids`, `reserve` |
| `Record.lean` | `checked`, `cancel`, `refuse_recycled`, `create`, `refuse_undeletable`, `release` |
| `Verify.lean` | `verify`'s hole test, `open_batch` |
| `Inv.lean` | The invariant `Inv`, `live_eq`, and preservation by `create`, `cancel`, `reserve` |
| `Lifecycle.lean` | `release` in a reachable state, `roll_inv`, `lifecycle_fresh` |
| `Released.lean` | `released`, `taken()`, `released()`, `refuse_not_live` (#3022) |
| `FalkorIdSpace.lean` | The report header and `#guard` replays of the Rust unit tests at `N = 32` |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The lakefile has no path dependencies. `proofs/graph_queries` depends on this project's results without importing it. Its `IdRust.lean` copies this model into list form to prove the `IdSpaceContract` that `graph.rs` relies on (`rustIdOps_contract`).

## Coverage

`COVERAGE.tsv` has 24 rows, all in `id_space.rs`:

| Bucket | Rows |
|---|---|
| PROVEN | 24 |

Every function is proven.

## What the model does not cover

- `u64` arithmetic outside `checked_add` and `live -=` is `Nat` here: `bound()`, `restored`'s sum, `reserve`'s run end, and `live += nodes.len()`. In a reachable state `checked` keeps the boundary below 2^64. Only `reserve`'s run end could exceed it, and that needs a `count` near 2^64, which `try_reserve_exact` refuses first.
- That callers settle every reservation before `open_batch` is not proven; `roll_inv` covers the verified path.
- Error messages are modelled only as the variant and its fields.
- The header's title still cites `e8f8a3017`. `id_space.rs` has had one small change since (#3307), and the `COVERAGE.tsv` line numbers match main.

## Build and check

```bash
cd proofs/id_space && lake build    # this project
bash proofs/lean_ci.sh id_space     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh             # repo-wide coverage report
```

**When you change `id_space.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model in `Model.lean` to match.
3. Update the list-form copy in `proofs/graph_queries/GraphQueries/IdRust.lean`.
4. Re-run both builds.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
