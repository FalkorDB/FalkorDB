import PendingCommit.PendingCancel
import GraphQueries.IdRust
/-
# Discharging `CancelSpec`: `Graph::cancel_node_id` / `cancel_relationship_id`

`PendingCancel.lean` takes the graph's cancel as the hypothesis `CancelSpec`.
Here the graph's two id spaces are proofs/graph_queries' `GQ.IdS` / `GQ.rCancel`
(`IdRust.lean`, imported via a Lake path dependency) and the hypothesis is
proved for the Rust bodies:

* `cancel_node_id` (graph.rs:1549) = `self.node_ids.cancel(id)?`,
* `cancel_relationship_id` (graph.rs:1562) = `self.relationship_ids.cancel(id)?`,
* `IdSpace::cancel` (id_space.rs:529) = `if !taken.insert(id) {Err}; recycled.insert(id); checked()`.

`freeN g` / `freeR g` are the recycled sets of the node and relationship id
space; `CancelSpec` is stated per space (`cancelN` makes its id free in
`freeN` and may not shrink `freeR`, and conversely), so an id recycled only in
the other space does not count.
-/
namespace PendingCommit.PS.CancelRust

open GQ (IdS ins rCancel)

/-- The graph's two id spaces: `(node_ids, relationship_ids)`. -/
abbrev Gr := IdS × IdS

def freeN (g : Gr) : List Nat := g.1.recycled
def freeR (g : Gr) : List Nat := g.2.recycled

/-- `cancel_node_id` (graph.rs:1549). -/
def cancelN (g : Gr) (id : Nat) : Option Gr := (rCancel g.1 id).map (·, g.2)
/-- `cancel_relationship_id` (graph.rs:1562). -/
def cancelR (g : Gr) (id : Nat) : Option Gr := (rCancel g.2 id).map (g.1, ·)

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

/-- `deletePendingNode_spec` for the Rust graph: `CancelSpec` discharged. -/
theorem deletePendingNode_spec_rust :
    type_of% (@deletePendingNode_spec Gr freeN freeR cancelN cancelR cancelN_spec cancelR_spec) :=
  @deletePendingNode_spec Gr freeN freeR cancelN cancelR cancelN_spec cancelR_spec

/-- `removeRelsG_spec` for the Rust graph: `CancelSpec` discharged. -/
theorem removeRelsG_spec_rust :
    type_of% (@removeRelsG_spec Gr freeN freeR cancelR cancelR_spec) :=
  @removeRelsG_spec Gr freeN freeR cancelR cancelR_spec

#print axioms deletePendingNode_spec_rust

end PendingCommit.PS.CancelRust
