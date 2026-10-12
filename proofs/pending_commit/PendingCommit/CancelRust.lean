import PendingCommit.PendingCancel
/-
# Discharging `CancelSpec`: `Graph::cancel_node_id` / `cancel_relationship_id`

`PendingCancel.lean` takes the graph's cancel as the hypothesis `CancelSpec`.
Here the graph's two id spaces are modelled (copied from proofs/graph_queries
`IdRust.lean`, which copies proofs/id_space `Model.lean`) and the hypothesis is
proved for the Rust bodies:

* `cancel_node_id` (graph.rs:1449) = `self.node_ids.cancel(id)?`,
* `cancel_relationship_id` (graph.rs:1462) = `self.relationship_ids.cancel(id)?`,
* `IdSpace::cancel` (id_space.rs:529) = `if !taken.insert(id) {Err}; recycled.insert(id); checked()`.

`free g` is "free in either space" (the theorems in PendingCancel use one
`free` for node and relationship ids).
-/
namespace PendingCommit.PS.CancelRust

/-- `struct IdSpace` (id_space.rs:164), list form (graph_queries `GQ.IdS`). -/
structure IdS where
  live : Nat
  recycled : List Nat
  entryBound : Nat
  taken : List Nat

/-- `RoaringTreemap::insert`. -/
def ins (l : List Nat) (x : Nat) : List Nat := if x ∈ l then l else l ++ [x]

def bound (s : IdS) : Nat := s.live + s.recycled.length
def above (t : List Nat) (eb : Nat) : Nat := (t.filter (fun x => decide (eb ≤ x))).length

/-- `checked` (id_space.rs:403). -/
def rChecked (s : IdS) : Bool :=
  let e := s.entryBound + above s.taken s.entryBound
  decide (e < 2 ^ 64) && e == bound s

/-- `cancel` (id_space.rs:529). -/
def rCancel (s : IdS) (id : Nat) : Option IdS :=
  if id ∈ s.taken then none
  else
    let s' : IdS := { s with taken := ins s.taken id, recycled := ins s.recycled id }
    if rChecked s' then some s' else none

/-- The graph's two id spaces: `(node_ids, relationship_ids)`. -/
abbrev Gr := IdS × IdS

def free (g : Gr) : List Nat := g.1.recycled ++ g.2.recycled

/-- `cancel_node_id` (graph.rs:1449). -/
def cancelN (g : Gr) (id : Nat) : Option Gr := (rCancel g.1 id).map (·, g.2)
/-- `cancel_relationship_id` (graph.rs:1462). -/
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

theorem cancelN_spec : CancelSpec free cancelN := ⟨by
  intro g id g' h
  simp only [cancelN, Option.map_eq_some_iff] at h
  obtain ⟨s', hs, rfl⟩ := h
  have := rCancel_ok _ _ _ hs
  refine ⟨?_, fun x hx => ?_⟩
  · simp [free, this, mem_ins]
  · simp only [free, List.mem_append] at hx ⊢
    rw [this, mem_ins]; rcases hx with hx | hx
    · exact Or.inl (Or.inl hx)
    · exact Or.inr hx⟩

theorem cancelR_spec : CancelSpec free cancelR := ⟨by
  intro g id g' h
  simp only [cancelR, Option.map_eq_some_iff] at h
  obtain ⟨s', hs, rfl⟩ := h
  have := rCancel_ok _ _ _ hs
  refine ⟨?_, fun x hx => ?_⟩
  · simp [free, this, mem_ins]
  · simp only [free, List.mem_append] at hx ⊢
    rw [this, mem_ins]; rcases hx with hx | hx
    · exact Or.inl hx
    · exact Or.inr (Or.inl hx)⟩

/-- `deletePendingNode_spec` for the Rust graph: `CancelSpec` discharged. -/
theorem deletePendingNode_spec_rust :
    type_of% (@deletePendingNode_spec Gr free cancelN cancelR cancelN_spec cancelR_spec) :=
  @deletePendingNode_spec Gr free cancelN cancelR cancelN_spec cancelR_spec

/-- `removeRelsG_spec` for the Rust graph: `CancelSpec` discharged. -/
theorem removeRelsG_spec_rust :
    type_of% (@removeRelsG_spec Gr free cancelR cancelR_spec) :=
  @removeRelsG_spec Gr free cancelR cancelR_spec

#print axioms deletePendingNode_spec_rust

end PendingCommit.PS.CancelRust
