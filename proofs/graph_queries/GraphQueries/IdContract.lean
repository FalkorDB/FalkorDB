import GraphQueries.Attrs
/-
# The `IdSpace` contract graph.rs relies on (since #2846)

`IdSpace` (graph/src/graph/id_space.rs) is proved in proofs/id_space, which
this project does not own. What graph.rs needs from it is stated here as the
hypothesis structure `IdSpaceContract`: every theorem that depends on an id
space *mutation* takes `(hC : IdSpaceContract O)` as an argument. Nothing here
is an axiom.

The readers (`live`, `recycled`, `bound`, `max_id`, `is_free`,
`recycled_count`, `restored`, `new_version`) are one-line bodies and are
restated verbatim, with their id_space.rs line.

`RoaringTreemap` = a duplicate-free `List Nat` read with membership
(`insert` = `ins`, `|=` = `union`, `-=` = `diff`, `len` = `length`).
-/
namespace GQ

def ins (l : List Nat) (x : Nat) : List Nat := if x ∈ l then l else l ++ [x]
def union (a b : List Nat) : List Nat := a ++ b.filter (fun x => x ∉ a)
def diff (a b : List Nat) : List Nat := a.filter (fun x => x ∉ b)

/-- `struct IdSpace` (id_space.rs:159). -/
structure IdS where
  live : Nat
  recycled : List Nat
  entryBound : Nat
  taken : List Nat

namespace IdS
/-- `bound` (id_space.rs:310): `live + recycled.len()`. -/
def bound (s : IdS) : Nat := s.live + s.recycled.length
/-- `max_id` (id_space.rs:320). -/
def maxId (s : IdS) : Nat := if s.live = 0 then 0 else s.bound - 1
/-- `is_free` (id_space.rs:296). -/
def isFree (s : IdS) (id : Nat) : Bool := decide (id ∈ s.recycled)
/-- `recycled_count` (id_space.rs:290). -/
def recycledCount (s : IdS) : Nat := s.recycled.length
/-- `restored` (id_space.rs:263): a fresh batch at the restored boundary. -/
def restored (live : Nat) (recycled : List Nat) : IdS := ⟨live, recycled, live + recycled.length, []⟩
/-- `new` (id_space.rs:250). -/
def new : IdS := ⟨0, [], 0, []⟩
/-- `new_version` (id_space.rs:333). -/
def newVersion (s : IdS) : IdS := restored s.live s.recycled
end IdS

/-- The mutating half of `IdSpace`; `none` = `Err(IdSpaceError)`. -/
structure IdSpaceOps where
  /-- `create` (id_space.rs:545). -/
  create : IdS → List Nat → Option IdS
  /-- `release(requested, freed)` (id_space.rs:642). -/
  release : IdS → List Nat → List Nat → Option IdS
  /-- `cancel` (id_space.rs:492). -/
  cancel : IdS → Nat → Option IdS
  /-- `open_batch` (id_space.rs:411). -/
  openBatch : IdS → Option IdS
  /-- `verify` (id_space.rs:676): `true` = `Ok(())`. -/
  verify : IdS → Bool

/-- **Hypothesis** (owned by proofs/id_space): what each mutator does to the
two halves of the boundary when it succeeds, and the refusals graph.rs
documents. Each clause is the literal effect of the Rust body:
`create`: `recycled -= nodes; live += nodes.len()` (:583-584);
`release`: `refuse_recycled(requested)?` then `recycled |= freed;
live -= freed.len()` (:653-656); `cancel`: `recycled.insert(id)` (:499);
`open_batch` moves only `entry_bound`/`taken` (:413-414). -/
structure IdSpaceContract (O : IdSpaceOps) : Prop where
  create_ok : ∀ s ns s', O.create s ns = some s' →
    s'.live = s.live + ns.length ∧ s'.recycled = diff s.recycled ns
  create_refuses_live : ∀ s ns n, n ∈ ns → n ∉ s.recycled → n < s.entryBound →
    O.create s ns = none
  /-- #2911 (`fe619ac5f`): `nodes.max().filter(|&id| id >= ID_LIMIT)` (id_space.rs:562);
  `ID_LIMIT = GrB_INDEX_MAX = 2^60 - 1` (id_space.rs:107). Proven in proofs/id_space as
  `create_out_of_range`. -/
  create_refuses_limit : ∀ s ns n, n ∈ ns → 2 ^ 60 - 1 ≤ n → O.create s ns = none
  release_ok : ∀ s req fr s', O.release s req fr = some s' →
    s'.recycled = union s.recycled fr ∧ s'.live = s.live - fr.length
  release_refuses_recycled : ∀ s req fr n, n ∈ req → n ∈ s.recycled →
    O.release s req fr = none
  cancel_ok : ∀ s id s', O.cancel s id = some s' →
    s'.recycled = ins s.recycled id ∧ s'.live = s.live
  open_ok : ∀ s s', O.openBatch s = some s' →
    s'.live = s.live ∧ s'.recycled = s.recycled ∧ s'.entryBound = s.bound ∧ s'.taken = []

theorem IdS.restored_bound (live : Nat) (r : List Nat) :
    (IdS.restored live r).bound = live + r.length ∧ (IdS.restored live r).entryBound = live + r.length := ⟨rfl, rfl⟩

theorem IdS.newVersion_spec (s : IdS) :
    (IdS.newVersion s).live = s.live ∧ (IdS.newVersion s).recycled = s.recycled ∧
    (IdS.newVersion s).entryBound = s.bound ∧ (IdS.newVersion s).taken = [] := ⟨rfl, rfl, rfl, rfl⟩

end GQ
