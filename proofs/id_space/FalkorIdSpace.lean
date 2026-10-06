/-
# IdSpace (`graph/src/graph/id_space.rs`, main @ fe619ac5f): the state machine, proven

Re-targeted to #2846 (`157d42ec1`, "make IdSpace the id allocation authority"):
`IdSpace` now *owns* `live` and `recycled` (the entity count and free set that
used to be `Graph::node_count` / `deleted_nodes`), plus the open batch
`entry_bound` / `taken`. `IdSpace::at`, `record_created` and `Miscounted` are
gone; `new`, `restored`, `new_version`, `checked`, `open_batch`, `cancel`,
`create`, `release` are new.

A `RoaringTreemap` is an `IdSet` read over `[0, N)`, `N = 2^64` in the engine
(`Card.lean`: `cnt n s = |s ∩ [0,n)|` = roaring's `rank(n-1)`, `asc` = `iter()`,
`select(k)` = `(asc N s)[k]?`). Roaring is a pure-Rust crate, not FFI; its
operations are modelled by their set semantics (`Model.lean`). Mutators return
`(state after, result)`: an `Inconsistent` from the trailing `checked()` leaves
the mutated state behind, every other refusal returns before any mutation.

| here (`IdSpaceModel`) | there (`graph/src/graph/id_space.rs`) |
| --- | --- |
| `above`                    | `above` (:220) — `len - rank(bound-1)` |
| `freeOf`, `reclaimIds`     | `reclaim_ids` (:228) — `pool - taken - issued`, lowest `count` |
| `IdSpace.new`              | `IdSpace::new` (:250), `Default::default` (:242) |
| `IdSpace.restored`         | `IdSpace::restored` (:263) |
| `IdSpace.live/recycled`    | `live` (:278), `recycled` (:284) — field reads |
| `IdSpace.recycledCount`    | `recycled_count` (:290) |
| `IdSpace.isFree`           | `is_free` (:296) |
| `IdSpace.bound`            | `bound` (:310) — `live + recycled.len()` |
| `IdSpace.maxId`            | `max_id` (:320) |
| `IdSpace.newVersion`       | `new_version` (:333) |
| `IdSpace.checked`          | `checked` (:368) — `checked_add`, `unwrap_or(u64::MAX)` |
| `IdSpace.openBatch`        | `open_batch` (:411) |
| `IdSpace.reserve`          | `reserve` (:446); `allocOk` = `try_reserve_exact` result |
| `IdSpace.cancel`           | `cancel` (:492) |
| `IdSpace.refuseRecycled`   | `refuse_recycled` (:513) |
| `idLimit`, `IdSpace.create`| `ID_LIMIT` (:95), `create` (:545) — `nodes.max() >= ID_LIMIT` refused first |
| `IdSpace.refuseUndeletable`| `refuse_undeletable` (:601) |
| `IdSpace.release`          | `release` (:642) |
| `IdSpace.holeCheck/verify` | `verify` (:676) — `select`, `max`, the `||` order |

## Theorems (all proven, no sorry)
* `above_spec`, `select_above`, `sMax_sFrom` — the rank/select arithmetic is "members ≥ bound",
  "lowest member ≥ bound", and the max is the max of that part.
* `new_spec`, `restored_spec`, `newVersion_spec`, `maxId_spec` (no underflow).
* `freeOf_spec`, `reclaimIds_spec`, `reclaimed_free`, `reclaimed_sorted`; `reserve_spec`,
  `reserve_length`, `reserve_alloc_fail`.
* `reserve_fresh` — under the Rust-doc precondition (what the batch took/issued at or above the
  boundary lies below `start`) and `binOk`: duplicate-free, disjoint from `taken` and `issued`;
  reclaimed ids are free and below the boundary, fresh ones at or above `start`.
* `checked_ok_iff`; `cancel_taken`, `cancel_fresh`; `refuseRecycled_ok/err`,
  `refuseUndeletable_ok/err` (name the lowest / highest offender).
* `create_spec` — all or nothing: refusals pass iff `CreateOk` (every id `< ID_LIMIT`; every non-free
  id ≥ entry and not taken), then the state moves in full; else it is untouched.
  `create_out_of_range` — a batch whose max is `≥ ID_LIMIT` is refused as `IdOutOfRange(max)`, untouched.
  `create_expect_unreachable` — the `.expect` cannot panic.
* `release_spec` — all or nothing, judged over *requested*; `release_refused`.
* `noHole_iff`, `holeCheck_eq`, `verify_ok_iff` — **`verify` accepts exactly a consistent,
  dense batch**: `checked` holds and `taken ∩ [entry, ∞) = [entry, entry + created)`.
  `verify_hole`, `verify_no_underflow` (the `||` order argument).
* `openBatch_spec`, `openBatch_ok`, `openBatch_eq_newVersion`.
* **Invariant** `Inv` (`Inv.lean`): `checked` ∧ every free id is below the entry boundary or
  taken ∧ `u64::MAX` never taken. `inv_new`, `inv_restored`, `create_succeeds`/`create_preserves`
  (a create its refusals accept always passes `checked`; needs only `ID_LIMIT < 2^64`), `cancel_succeeds` (iff free ↔ below the
  boundary), `reserve_then_cancel` (every reserved id can be cancelled), `release_succeeds`.
* `live_eq` — **`live` is exactly the number of live ids**, hence `release_no_underflow`:
  `live -= freed.len()` never wraps in a reachable state.
* `roll_inv` — `Graph::roll_id_batches` (`verify` then `open_batch`) on a reachable state opens a
  reachable batch equal to `new_version`.
* `lifecycle_fresh` — `restored` → `reserve n` = `[bound, bound+n)` → `create` ok → `verify` ok.

## Gaps
* `u64` arithmetic outside `checked_add`/`live -=` is `Nat` here: `bound()`, `restored`'s sum,
  `reserve`'s `start + (count - reclaimed)`, `live += nodes.len()`. In a reachable state
  `bound = eb + |taken ≥ eb| < 2^64` (from `checked`), so only `reserve`'s run end could exceed it,
  which needs `count` near 2^64 (refused first by `try_reserve_exact`).
* `cancel_succeeds`/`reserve_then_cancel` assume the id is not `u64::MAX` (`cancel` itself does not
  check; `Inv.noMax` would otherwise fail).
* That callers (`Graph`, `Pending`, bulk) settle every reservation before `open_batch` (the
  documented precondition) is not proven here; `roll_inv` covers the verified path.
* Error message strings are not modelled beyond the variant and its fields.
No confirmed bugs in this file.
Historical (#2892, fixed by #2911 `fe619ac5f`): at 2c874022a `create` refused only `u64::MAX`, so it
accepted 2^61 (or `u64::MAX - 1`) as a fresh id and `Graph::grow_for_nodes` then sized the matrices to
it (assert in `GrB_Matrix_new` / `grow_cap` hang). Now `create` refuses every id `≥ ID_LIMIT =
GrB_INDEX_MAX` before anything is sized (`create_out_of_range`, `create_spec`), and every id it accepts
has a matrix row (`< 2^60 - 1`). `grow_cap`'s side is proven in `graph_queries` (`Schema.lean`).
-/
import FalkorIdSpace.Lifecycle

namespace IdSpaceModel.Sanity
open IdSpaceModel

def set (l : List Nat) : IdSet := fun i => l.contains i
def R {α} : IdSpace × Except IdSpaceError α → Option IdSpaceError
  | (_, .error e) => some e | (_, .ok _) => none
def ver (sp : IdSpace) : Option IdSpaceError :=
  match sp.verify 32 with | .error e => some e | .ok () => none
def resv (r : Except String (List Nat)) : Option (List Nat) := r.toOption
def ob (sp : IdSpace) : IdSpace := match sp.openBatch 32 with | .ok s => s | .error _ => sp

-- Mirrors the Rust unit tests (`id_space.rs:727-1506`), universe N = 32 (`u64::MAX` = 31), ID_LIMIT = 28.
def g0 : IdSpace := ob IdSpace.new
-- ids_arriving_out_of_order_still_fill_the_range (scaled: 5..10 then 0..5)
#guard ver ((g0.create 32 28 (set [5,6,7,8,9])).1.create 32 28 (set [0,1,2,3,4])).1 == none
-- a_hole_left_at_the_end_is_reported
#guard ver (g0.create 32 28 (set [5,6,7,8,9])).1 == some (.hole 0 9 5)
-- a_node_the_batch_never_recorded_is_caught_by_the_counter (`wedged(4, {}, 0, 0..3)`)
#guard ver ⟨4, set [], 0, set [0,1,2]⟩ == some (.inconsistent 4 0 4 0 3 3)
-- a_range_that_starts_above_the_boundary_is_a_hole
#guard ver (g0.create 32 28 (set [1,2,3])).1 == some (.hole 0 3 3)
-- an_id_claimed_twice_is_refused_at_the_record
#guard R ((g0.create 32 28 (set [0,1,2,3,4,5])).1.create 32 28 (set [5])) == some (.alreadyLive 5 0)
-- creating_deleting_and_recreating_one_id_in_a_batch
#guard let s1 := (g0.create 32 28 (set [0])).1
       let s2 := (s1.release 32 (set [0]) (set [0])).1
       let s3 := s2.create 32 28 (set [0])
       R s3 == none && ver s3.1 == none && s3.1.live == 1
-- an_id_the_batch_already_took_cannot_be_cancelled
#guard let s1 := (IdSpace.new.cancel 32 0).1
       R (s1.cancel 32 0) == some (.alreadyTaken 0)
-- a_cancelled_reclaimed_id_is_not_reissued_in_the_same_batch
#guard let s := ob (IdSpace.restored 32 0 (set [0]))
       resv (s.reserve 32 true 1 (set [])) == some [0] &&
       resv ((s.cancel 32 0).1.reserve 32 true 1 (set [])) == some [1] && R (s.cancel 32 0) == none
-- deleting_the_same_id_twice_inside_one_batch_is_refused
#guard let s1 := ((g0.create 32 28 (set [0,1])).1.release 32 (set [1]) (set [1])).1
       R (s1.release 32 (set [1]) (set [1])) == some (.alreadyRecycled 1)
-- deleting_an_id_never_created_is_refused
#guard R ((g0.create 32 28 (set [0,1])).1.release 32 (set [7]) (set [7])) == some (.neverCreated 7)
-- an_id_no_matrix_can_hold_is_refused_before_anything_is_sized (#2911; scaled: ID_LIMIT = 28,
-- ids [0, id] for id in ID_LIMIT, ID_LIMIT + 1, u64::MAX - 1): refused naming the id, state untouched
#guard [28, 29, 30, 31].all fun id =>
  R (g0.create 32 28 (set [0, id])) == some (.idOutOfRange id) && (g0.create 32 28 (set [0, id])).1.live == 0
-- the_highest_creatable_id_reaches_verify
#guard let s := g0.create 32 28 (set [27])
       R s == none && ver s.1 == some (.hole 0 27 1)
-- a_reservation_that_outlives_its_batch_is_refused
#guard resv (IdSpace.new.reserve 32 true 3 (set [])) == some [0,1,2]
#guard R (ob (IdSpace.new.create 32 28 (set [1,2])).1 |>.create 32 28 (set [0])) == some (.alreadyLive 0 2)
-- release_frees_the_resolved_and_leaves_the_rest
#guard let s := (IdSpace.restored 32 4 (set [])).release 32 (set [0,1,2,3]) (set [1,2])
       R s == none && s.1.live == 2 && s.1.bound 32 == 4 && s.1.recycled 1 && !s.1.recycled 0
-- release_refuses_a_free_id_it_would_not_have_freed
#guard R ((IdSpace.restored 32 3 (set [3])).release 32 (set [2,3]) (set [2])) == some (.alreadyRecycled 3)

end IdSpaceModel.Sanity

#print axioms IdSpaceModel.verify_ok_iff
#print axioms IdSpaceModel.reserve_fresh
#print axioms IdSpaceModel.create_succeeds
#print axioms IdSpaceModel.release_no_underflow
#print axioms IdSpaceModel.roll_inv
#print axioms IdSpaceModel.lifecycle_fresh
