import PendingCommit.PendingCancel
import GraphQueries.IdRust
/-
# Discharging `CancelSpec`: `Graph::cancel_node_id` / `cancel_relationship_id`

`PendingCancel.lean` takes the graph's cancel as the hypothesis `CancelSpec`.
Here the graph's two id spaces are proofs/graph_queries' `GQ.IdS` / `GQ.rCancel`
(`IdRust.lean`, imported via a Lake path dependency) and the hypothesis is
proved for the Rust bodies:

* `cancel_node_id` (graph.rs) = `self.node_ids.cancel(id)?`,
* `cancel_relationship_id` (graph.rs) = `self.relationship_ids.cancel(id)?`,
* `IdSpace::cancel` (id_space.rs) = `if !taken.insert(id) {Err}; recycled.insert(id); checked()`.

`freeN g` / `freeR g` are the recycled sets of the node and relationship id
space; `CancelSpec` is stated per space (`cancelN` makes its id free in
`freeN` and may not shrink `freeR`, and conversely), so an id recycled only in
the other space does not count.
-/
namespace PendingCommit.PS.CancelRust

open GQ (IdS ins rCancel)

/-- The graph's two id spaces and its relationship-type list (names as `Nat`s):
`(node_ids, relationship_ids, schema.relationship_types)`. -/
abbrev Gr := IdS × IdS × List Nat

def freeN (g : Gr) : List Nat := g.1.recycled
def freeR (g : Gr) : List Nat := g.2.1.recycled
def types (g : Gr) : List Nat := g.2.2

/-- `cancel_node_id` (graph.rs). -/
def cancelN (g : Gr) (id : Nat) : Option Gr := (rCancel g.1 id).map (fun s => (s, g.2))
/-- `cancel_relationship_id` (graph.rs). -/
def cancelR (g : Gr) (id : Nat) : Option Gr := (rCancel g.2.1 id).map (fun s => (g.1, s, g.2.2))
/-- `get_type_id_mut(name)` (graph.rs, #3641's call in `remove_pending_relationships_for_node`):
a known name is left alone, a new one is appended to `relationship_types`
(graph_queries `Schema.getTypeIdMut`; its tensor and resize touch no id space). -/
def regT (g : Gr) (t : Nat) : Gr := (g.1, g.2.1, if t ∈ g.2.2 then g.2.2 else g.2.2 ++ [t])

theorem rCancel_ok (s s' : IdS) (id : Nat) (h : rCancel s id = some s') :
    s'.recycled = ins s.recycled id := by
  unfold rCancel at h
  split at h
  · simp at h
  · dsimp only at h
    split at h
    · simp at h; subst h; rfl
    · simp at h

theorem mem_ins (l : List Nat) (x y : Nat) : y ∈ ins l x ↔ y ∈ l ∨ y = x := by
  unfold ins
  by_cases hx : x ∈ l
  · simp only [hx, ite_true]
    constructor
    · exact Or.inl
    · rintro (h | rfl) <;> assumption
  · simp [hx]

theorem cancelN_spec : CancelSpec freeN freeR cancelN := ⟨by
  intro g id g' h
  simp only [cancelN, Option.map_eq_some_iff] at h
  obtain ⟨s', hs, rfl⟩ := h
  have := rCancel_ok _ _ _ hs
  refine ⟨?_, fun x hx => ?_, fun x hx => hx⟩
  · simp [freeN, this, mem_ins]
  · simp only [freeN] at hx ⊢
    rw [this, mem_ins]; exact Or.inl hx⟩

theorem cancelR_spec : CancelSpec freeR freeN cancelR := ⟨by
  intro g id g' h
  simp only [cancelR, Option.map_eq_some_iff] at h
  obtain ⟨s', hs, rfl⟩ := h
  have := rCancel_ok _ _ _ hs
  refine ⟨?_, fun x hx => ?_, fun x hx => hx⟩
  · simp [freeR, this, mem_ins]
  · simp only [freeR] at hx ⊢
    rw [this, mem_ins]; exact Or.inl hx⟩

/-- **`get_type_id_mut` meets `RegSpec`**: the name is registered, nothing registered is lost,
no id space changes. -/
theorem regT_spec : RegSpec types freeN freeR regT := ⟨by
  intro g t
  refine ⟨?_, fun y hy => ?_, fun x hx => hx, fun x hx => hx⟩
  · simp only [types, regT]; by_cases h : t ∈ g.2.2 <;> simp [h]
  · have hy' : y ∈ g.2.2 := hy
    simp only [types, regT]; by_cases h : t ∈ g.2.2 <;> simp [h, hy']⟩

/-- A node cancel touches only the node id space: no type is lost. -/
theorem cancelN_types : KeepsTypes types cancelN := by
  intro g id g' h
  simp only [cancelN, Option.map_eq_some_iff] at h
  obtain ⟨s', _, rfl⟩ := h
  intro y hy; exact hy

/-- A relationship cancel touches only the relationship id space: no type is lost. -/
theorem cancelR_types : KeepsTypes types cancelR := by
  intro g id g' h
  simp only [cancelR, Option.map_eq_some_iff] at h
  obtain ⟨s', _, rfl⟩ := h
  intro y hy; exact hy

/-- `deletePendingNode_spec` for the Rust graph: `CancelSpec` and `RegSpec` discharged. -/
theorem deletePendingNode_spec_rust :
    type_of% (@deletePendingNode_spec Gr types freeN freeR cancelN cancelR regT cancelN_spec cancelR_spec
      regT_spec cancelR_types cancelN_types) :=
  @deletePendingNode_spec Gr types freeN freeR cancelN cancelR regT cancelN_spec cancelR_spec
    regT_spec cancelR_types cancelN_types

/-- `removeRelsG_spec` for the Rust graph: `CancelSpec` and `RegSpec` discharged. -/
theorem removeRelsG_spec_rust :
    type_of% (@removeRelsG_spec Gr types freeN freeR cancelR regT cancelR_spec regT_spec cancelR_types) :=
  @removeRelsG_spec Gr types freeN freeR cancelR regT cancelR_spec regT_spec cancelR_types

#print axioms deletePendingNode_spec_rust

end PendingCommit.PS.CancelRust
