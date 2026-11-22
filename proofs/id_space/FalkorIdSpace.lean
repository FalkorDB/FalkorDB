/-
# IdSpace (`graph/src/graph/id_space.rs`, main @ 256ab4d3f): the state machine, proven

`IdSpace` *owns* `live` and `recycled` (the entity count and free set; `Graph` keeps
neither), plus the open batch `entry_bound` / `taken`. A fifth field `released` is what the open
batch freed (set by `release`, cleared by `open_batch`, not carried by `new_version`), read through
`taken()`/`released()`; `refuse_not_live` (= `refuse_recycled` then `refuse_undeletable`) is what
`release` calls, and `Inv` carries `relOk`. A sixth field `quarantined` is every id earlier batches
of the version freed or cancelled, accumulated by `open_batch` (`rolledQuar`), subtracted by
`reclaim_ids`/`reserve`, cleared by `end_quarantine`, empty in `new`/`restored`/`new_version`.
`Quarantine.lean`: `openBatch_quarantines`, `reserve_not_quarantined`, `roll_quarOk`,
**`rolled_not_reissued`**, and the `pre3641_*` examples (the model without the quarantine, kept as a
historical counterexample). `openBatch_eq_rolled` (new_version plus the quarantine); `roll_inv` is
stated on `rolled`. `Callers.lean` discharges `release`'s `freed ⊆ requested` premise for its three
engine call sites (`release_call_sites`).

A `RoaringTreemap` is an `IdSet` read over `[0, N)`, `N = 2^64` in the engine
(`Card.lean`: `cnt n s = |s ∩ [0,n)|` = roaring's `rank(n-1)`, `asc` = `iter()`,
`select(k)` = `(asc N s)[k]?`). Roaring is a pure-Rust crate, not FFI; its
operations are modelled by their set semantics (`Model.lean`). Mutators return
`(state after, result)`: an `Inconsistent` from the trailing `checked()` leaves
the mutated state behind, every other refusal returns before any mutation.

| here (`IdSpaceModel`) | there (`graph/src/graph/id_space.rs`) |
| --- | --- |
| `above`                    | `above` (:248) — `len - rank(bound-1)` |
| `freeOf`, `reclaimIds`     | `reclaim_ids` (:256) — `pool - taken - issued`, lowest `count` |
| `IdSpace.new`              | `IdSpace::new` (:280), `Default::default` (:272) |
| `IdSpace.restored`         | `IdSpace::restored` (:295) |
| `IdSpace.live/recycled`    | `live` (:312), `recycled` (:318) — field reads |
| `IdSpace.takenSet/releasedSet` | `taken` (:327), `released` (:333) — field reads |
| `IdSpace.refuseNotLive`    | `refuse_not_live` (:715) |
| `IdSpace.recycledCount`    | `recycled_count` (:339) |
| `IdSpace.isFree`           | `is_free` (:345) |
| `IdSpace.bound`            | `bound` (:359) — `live + recycled.len()` |
| `IdSpace.maxId`            | `max_id` (:369) |
| `IdSpace.newVersion`       | `new_version` (:389) |
| `IdSpace.checked`          | `checked` (:426) — `checked_add`, `unwrap_or(u64::MAX)` |
| `IdSpace.openBatch`        | `open_batch` (:470) — also `quarantined`; `end_quarantine` (:378) = `IdSpace.endQuarantine` |
| `IdSpace.reserve`          | `reserve` (:516); `allocOk` = `try_reserve_exact` result |
| `IdSpace.cancel`           | `cancel` (:569) |
| `IdSpace.refuseRecycled`   | `refuse_recycled` (:590) |
| `idLimit`, `IdSpace.create`| `ID_LIMIT` (:100), `create` (:621) — `nodes.max() >= ID_LIMIT` refused first |
| `IdSpace.refuseUndeletable`| `refuse_undeletable` (:677) |
| `IdSpace.release`          | `release` (:739) |
| `IdSpace.holeCheck/verify` | `verify` (:773) — `select`, `max`, the `||` order |

## Theorems (all proven, no sorry)
* `above_spec`, `select_above`, `sMax_sFrom` — the rank/select arithmetic is "members ≥ bound",
  "lowest member ≥ bound", and the max is the max of that part.
* `new_spec`, `restored_spec`, `newVersion_spec`, `maxId_spec` (no underflow).
* `freeOf_spec`, `reclaimIds_spec`, `reclaimed_free`, `reclaimed_sorted`; `reserve_spec`
  (exact, the run's `u64` end included: release wraps), `reserve_spec_nowrap`, `reserve_length` (needs
  the end to fit in a `u64`), `reserve_wraps` / `reserve_wraps_example` (the overflow case: the run is empty
  and fewer than `count` ids come back), `reserve_alloc_fail`.
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
* `openBatch_spec`, `openBatch_ok`, `openBatch_eq_rolled`.
* **Invariant** `Inv` (`Inv.lean`): `checked` ∧ every free id is below the entry boundary or
  taken ∧ `u64::MAX` never taken. `inv_new`, `inv_restored`, `create_succeeds`/`create_preserves`
  (a create its refusals accept always passes `checked`; needs only `ID_LIMIT < 2^64`), `cancel_succeeds` (iff free ↔ below the
  boundary), `reserve_then_cancel` (every reserved id can be cancelled), `release_succeeds`.
* **The quarantine (`Quarantine.lean`)**: `endQuarantine_spec`, `quarantined_new/restored/newVersion`, `openBatch_quarantines`
  (adds exactly `released ∪ (taken ∩ recycled)`), `QuarOk` (quarantine below the boundary) with `roll_quarOk`,
  `reserve_not_quarantined`, **`rolled_not_reissued`** (an id the previous batch freed or cancelled is not handed out by
  the next batch of the version), `pre3641_reissues` / `pre3641_cancelled_reissues` (without the quarantine the id comes
  back; kept as historical counterexamples) with `post3641_*`. `Callers.lean`: `release_call_sites`, `hsub_same_set`, `hsub_resolved`.
* **`released` (`Released.lean`)**: `refuseNotLive_ok`/`refuseNotLive_err` (passes iff `ReleaseOk`; names
  the lowest free id, else the highest never-allocated one), **`refuseNotLive_live` — it accepts
  exactly live ids**, `live_eq_accepted`; `released_new/restored/newVersion`, `openBatch_clears`
  (empty after `open_batch`), `create_keeps_released`, `cancel_keeps_released`, `release_records`
  (grows by exactly `freed`); `Inv.relOk` (every released id is below the entry boundary or taken),
  preserved by `create_succeeds`/`cancel_succeeds`/`release_succeeds`/`roll_inv`;
  **`refuseNotLive_minus_released`** — the effects `CreateEdge` check `(src ∪ dst) − released` admits
  only ids live now or released by this batch (hence live in it), and **`freed_earlier_refused`** —
  an id freed by an earlier batch is refused after the roll.
* `live_eq` — **`live` is exactly the number of live ids**, hence `release_no_underflow`:
  `live -= freed.len()` never wraps in a reachable state.
* `roll_inv` — `Graph::roll_id_batches` (`verify` then `open_batch`) on a reachable state opens a
  reachable batch equal to `rolled` (`new_version` plus the quarantine).
* `lifecycle_fresh` — `restored` → `reserve n` = `[bound, bound+n)` → `create` ok → `verify` ok.

## Gaps
* `u64` arithmetic outside `checked_add`/`live -=` is `Nat` here: `bound()`, `restored`'s sum,
  `start`'s two adds, `live += nodes.len()`. `reserve`'s run end `start + (count - reclaimed)` is modelled
  (release build: it wraps at `N`, `reserve_spec`; a debug build panics). `checked` keeps `eb + |taken ≥ eb|`
  below 2^64 but says nothing about `issued` or `count`, and `restored` takes any `live` (the RDB's
  number), so the end can wrap for a boundary within `count - reclaimed` of `2^64`
  (`restored(u64::MAX, ∅)` and one id: `reserve_wraps_example`). `try_reserve_exact` caps `count` at 2^60,
  and `create` refuses ids `≥ ID_LIMIT` (2^60 - 1), so a space built by this API stays far below; whether an
  RDB can restore a boundary that high is not proven here. `reserve_length` therefore takes the end-fits
  hypothesis; `reserve_fresh` does not need it.
* `cancel_succeeds`/`reserve_then_cancel` assume the id is not `u64::MAX` (`cancel` itself does not
  check; `Inv.noMax` would otherwise fail).
* That callers (`Graph`, `Pending`, bulk) settle every reservation before `open_batch` (the
  documented precondition) is not proven here; `roll_inv` covers the verified path.
* Error message strings are not modelled beyond the variant and its fields.
No open bugs. Fixed ones are in this file's git history.
`create` refuses every id `≥ ID_LIMIT = GrB_INDEX_MAX` before anything is sized (`create_out_of_range`,
`create_spec`), and every id it accepts has a matrix row (`< 2^60 - 1`). `grow_cap`'s side is proven in
`graph_queries` (`Schema.lean`).
-/
import FalkorIdSpace.Callers

namespace IdSpaceModel.Sanity
open IdSpaceModel

def set (l : List Nat) : IdSet := fun i => l.contains i
def R {α} : IdSpace × Except IdSpaceError α → Option IdSpaceError
  | (_, .error e) => some e | (_, .ok _) => none
def ver (sp : IdSpace) : Option IdSpaceError :=
  match sp.verify 32 with | .error e => some e | .ok () => none
def resv (r : Except String (List Nat)) : Option (List Nat) := r.toOption
def ob (sp : IdSpace) : IdSpace := match sp.openBatch 32 with | .ok s => s | .error _ => sp

-- Mirrors the Rust unit tests (`id_space.rs:788-1567`), universe N = 32 (`u64::MAX` = 31), ID_LIMIT = 28.
def g0 : IdSpace := ob IdSpace.new
-- ids_arriving_out_of_order_still_fill_the_range (scaled: 5..10 then 0..5)
#guard ver ((g0.create 32 28 (set [5,6,7,8,9])).1.create 32 28 (set [0,1,2,3,4])).1 == none
-- a_hole_left_at_the_end_is_reported
#guard ver (g0.create 32 28 (set [5,6,7,8,9])).1 == some (.hole 0 9 5)
-- a_node_the_batch_never_recorded_is_caught_by_the_counter (`wedged(4, {}, 0, 0..3)`)
#guard ver ⟨4, set [], 0, set [0,1,2], set [], set []⟩ == some (.inconsistent 4 0 4 0 3 3)
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
-- an_id_no_matrix_can_hold_is_refused_before_anything_is_sized (scaled: ID_LIMIT = 28,
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

-- `released` records what the batch freed; `refuse_not_live` refuses a free id
-- (`AlreadyRecycled`) and a never-allocated one (`NeverCreated`); `ids − released` lets
-- through what this batch freed; `open_batch` forgets it.
#guard let s1 := (g0.create 32 28 (set [0,1])).1
       let s2 := (s1.release 32 (set [1]) (set [1])).1
       s2.released 1 && !s2.released 0 &&
       (match s2.refuseNotLive 32 (set [1]) with | .error (.alreadyRecycled 1) => true | _ => false) &&
       (match s2.refuseNotLive 32 (set [7]) with | .error (.neverCreated 7) => true | _ => false) &&
       (match s2.refuseNotLive 32 (sDiff (set [0,1]) s2.released) with | .ok () => true | _ => false) &&
       !(ob s2).released 1 &&
       (match (ob s2).refuseNotLive 32 (sDiff (set [1]) (ob s2).released) with
        | .error (.alreadyRecycled 1) => true | _ => false)
-- a reclaimed id stays in `released` (created, deleted, recreated in one batch)
#guard let s1 := (g0.create 32 28 (set [0])).1
       let s3 := (((s1.release 32 (set [0]) (set [0])).1).create 32 28 (set [0])).1
       s3.released 0 && (match s3.refuseNotLive 32 (set [0]) with | .ok () => true | _ => false)

end IdSpaceModel.Sanity

#print axioms IdSpaceModel.verify_ok_iff
#print axioms IdSpaceModel.reserve_fresh
#print axioms IdSpaceModel.create_succeeds
#print axioms IdSpaceModel.release_no_underflow
#print axioms IdSpaceModel.roll_inv
#print axioms IdSpaceModel.lifecycle_fresh
#print axioms IdSpaceModel.refuseNotLive_live
#print axioms IdSpaceModel.refuseNotLive_minus_released
#print axioms IdSpaceModel.freed_earlier_refused
#print axioms IdSpaceModel.rolled_not_reissued
#print axioms IdSpaceModel.release_call_sites
