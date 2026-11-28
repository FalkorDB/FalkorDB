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

/-- `relationship_types_fit` (graph.rs, #3967) on this graph: a registered name is checked at its
index, a new one at `next` (then `next + 1`), a repeated new one skipped; an index fits when it is
at most `MAX_TYPE_ID = 65534` (graph_queries `Schema.relTypesFitAux`, names as `Nat`s). -/
def fitAux (tys : List Nat) : List Nat → Nat → List Nat → Bool
  | [], _, _ => true
  | n :: ns, next, new =>
    match tys.idxOf? n with
    | some t => if t ≤ GQ.maxTypeId then fitAux tys ns next new else false
    | none =>
      if n ∈ new then fitAux tys ns next new
      else if next ≤ GQ.maxTypeId then fitAux tys ns (next + 1) (new ++ [n]) else false

def fit (g : Gr) (ns : List Nat) : Bool := fitAux g.2.2 ns g.2.2.length []

/-- With `MAX_TYPE_ID + 1` types registered, a cascade carrying an unregistered type is refused. -/
theorem fitAux_full (tys : List Nat) (hfull : GQ.maxTypeId < tys.length) :
    ∀ (ns : List Nat) (next : Nat) (new : List Nat), tys.length ≤ next →
    ∀ n ∈ ns, n ∉ tys → n ∉ new → fitAux tys ns next new = false := by
  intro ns
  induction ns with
  | nil => intro _ _ _ n hn; cases hn
  | cons m ms ih =>
    intro next new hnext n hn hnt hnew
    simp only [fitAux]
    cases hm : tys.idxOf? m with
    | some t =>
      simp only
      split
      · rcases List.mem_cons.1 hn with rfl | hn
        · exact absurd (List.idxOf?_eq_none_iff.2 hnt) (by rw [hm]; simp)
        · exact ih _ _ hnext n hn hnt hnew
      · rfl
    | none =>
      simp only
      split
      · rcases List.mem_cons.1 hn with rfl | hn
        · contradiction
        · exact ih _ _ hnext n hn hnt hnew
      · rw [if_neg (by omega)]

theorem fit_full (g : Gr) (hfull : GQ.maxTypeId < (types g).length) (ns : List Nat) (n : Nat)
    (hn : n ∈ ns) (hnt : n ∉ types g) : fit g ns = false :=
  fitAux_full g.2.2 hfull ns _ [] (Nat.le_refl _) n hn hnt (by simp)

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
    type_of% (@deletePendingNode_spec Gr types freeN freeR fit cancelN cancelR regT cancelN_spec cancelR_spec
      regT_spec cancelR_types cancelN_types) :=
  @deletePendingNode_spec Gr types freeN freeR fit cancelN cancelR regT cancelN_spec cancelR_spec
    regT_spec cancelR_types cancelN_types

/-- `removeRelsG_spec` for the Rust graph: `CancelSpec` and `RegSpec` discharged. -/
theorem removeRelsG_spec_rust :
    type_of% (@removeRelsG_spec Gr types freeN freeR fit cancelR regT cancelR_spec regT_spec cancelR_types) :=
  @removeRelsG_spec Gr types freeN freeR fit cancelR regT cancelR_spec regT_spec cancelR_types

/-- **#3967 on the Rust graph**: with the type column full, a node delete whose cascade carries a
new type is refused before anything is undone. -/
theorem removeRelsG_type_refused_rust (p : P) (g : Gr) (id : Nat) (hfull : GQ.maxTypeId < (types g).length)
    (n : Nat) (hn : n ∈ Tys (collect p id)) (hnt : n ∉ types g) :
    removeRelsG fit cancelR regT p g id = none :=
  removeRelsG_type_refused fit cancelR regT p g id (fit_full g hfull _ n hn hnt)

#print axioms deletePendingNode_spec_rust

end PendingCommit.PS.CancelRust
