/-
# Faithful copies of the engine steps the abstract model abstracts

From proofs/effects_emit_apply (`ApplyHelpers.lean`: `intern`, `verifyId`,
`applyAddSchema`) and proofs/id_space (`IdSpace::reserve` on a fresh batch, and
since #2846 the `live`/`recycled` half of `IdSpace` that `Graph` reads its
boundary from), restated over lists so this project stays standalone.
`FalkorReplication.lean` proves the abstract `applyRec`/`idsOf`/`createOne`/
`deleteOne`/`GState.bound` equal to these.
-/
namespace FalkorFaithful

/-- `findIdx?` lookup, as `get_label_id` does over the dictionary. -/
def idxOfF (d : List String) (n : String) : Option Nat := d.findIdx? (· == n)

/-- `get_label_id_mut` / `get_type_id_mut` / `add_node_attribute_name`. -/
def internF (d : List String) (n : String) : Nat × List String :=
  match idxOfF d n with
  | some i => (i, d)
  | none => (d.length, d ++ [n])

/-- `verify_id` (`apply.rs:580`), success only. -/
def verifyIdF (expected assigned : Nat) : Bool := assigned == expected

/-- `apply_add_schema` (`apply.rs:560`): intern, then `verify_id`. -/
def applyAddSchemaF (d : List String) (id : Nat) (n : String) : Option (List String) :=
  let (assigned, d') := internF d n
  if verifyIdF id assigned then some d' else none

/-- `IdSpace::reserve` (`id_space.rs:483`) on a batch opened at `bound` that has
taken and been issued nothing: `reclaim_ids` (`:244`) takes the lowest `count`
ids of `recycled - ∅ - ∅`, then fresh ids from `entry_bound + above(∅) +
above(∅) = bound + 0`. `bin` is the bin's ascending listing. -/
def reserveFresh (bound : Nat) (bin : List Nat) (count : Nat) : List Nat :=
  let reclaimed := bin.take count
  reclaimed ++ List.range' (bound + 0) (count - reclaimed.length)

/-- The half of the node `IdSpace` (`id_space.rs:164`) that `Graph` reads since
#2846: `live` and `recycled` (a set, here a duplicate-free list). -/
structure Ids where
  live : Nat
  bin  : List Nat

/-- `IdSpace::bound` (`:343`) = `Graph::node_id_bound` (`graph.rs:1585`). -/
def Ids.bound (s : Ids) : Nat := s.live + s.bin.length

/-- The state `IdSpace::create` (`:582`) moves to on `{id}` once its refusals
pass: `recycled -= nodes; live += nodes.len()` — what `Graph::create_nodes`
(`graph.rs:1624`) does to the id space. -/
def Ids.create1 (s : Ids) (id : Nat) : Ids := ⟨s.live + 1, s.bin.erase id⟩

/-- The state `IdSpace::release` (`:703`) moves to on `requested = freed = {id}`
once its refusals pass (`id` not free): `recycled |= freed; live -= freed.len()`
— `Graph::delete_nodes` (`graph.rs:2131`). -/
def Ids.release1 (s : Ids) (id : Nat) : Ids := ⟨s.live - 1, id :: s.bin⟩

end FalkorFaithful
