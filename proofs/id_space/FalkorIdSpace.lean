/-
# IdSpace (`graph/src/graph/id_space.rs`): the id allocator state machine

Models `live`, `recycled`, the open batch, the `released` set and the quarantine. The root file also holds `#guard` replays of the Rust unit tests at N = 32.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
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

-- Mirrors the Rust unit tests (`id_space.rs`), universe N = 32 (`u64::MAX` = 31), ID_LIMIT = 28.
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
