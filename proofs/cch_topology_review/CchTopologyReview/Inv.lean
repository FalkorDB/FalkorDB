import CchTopologyReview.Model

/-!
# The invariant, split into ranks, arcs and the link between them

`Inv t = RInv t.ranks ∧ AInv t.arcs ∧ Held t.ranks t.arcs`.

* `RInv` is I1′ and P's first half (03 §2 "Contract changes"): `rank_of` and `node_of` are partial
  inverses inside the rank space `0 … nr-1`, and `free_ranks` is exactly the holderless ranks, ascending.
* `AInv` is I3′'s premises, the id rules of "Arc identity", I7 and the bookkeeping that ties
  `added_since_fold` and the metric to the deltas.
* `Held` is P's second half: an arc's two ends are held ranks, so a free rank has `up = down = ∅`.

Each operation touches one side only, and `Cch.arcs` / `Cch.ranks` say which.
-/

namespace CchTopo

variable {α : Type}

/-- The arc side of a version. -/
structure Arcs where
  ver : Nat
  upB : Rows
  downB : Rows
  upD : List Ent
  downD : List Ent
  cap : Nat
  added : List (Nat × List Ent)
  metLen : Nat

/-- The rank side of a version. -/
structure Ranks where
  nr : Nat
  nodeOf : Nat → Option Nat
  rankOf : Nat → Option Nat
  free : List Nat

def Cch.arcs (t : Cch α) : Arcs := ⟨t.ver, t.upB, t.downB, t.upD, t.downD, t.cap, t.added, t.met.length⟩
def Cch.ranks (t : Cch α) : Ranks := ⟨t.nr, t.nodeOf, t.rankOf, t.free⟩

@[simp] theorem Cch.ranks_nr (t : Cch α) : t.ranks.nr = t.nr := rfl
@[simp] theorem Cch.ranks_nodeOf (t : Cch α) : t.ranks.nodeOf = t.nodeOf := rfl
@[simp] theorem Cch.ranks_rankOf (t : Cch α) : t.ranks.rankOf = t.rankOf := rfl
@[simp] theorem Cch.ranks_free (t : Cch α) : t.ranks.free = t.free := rfl
@[simp] theorem Cch.arcs_ver (t : Cch α) : t.arcs.ver = t.ver := rfl
@[simp] theorem Cch.arcs_upB (t : Cch α) : t.arcs.upB = t.upB := rfl
@[simp] theorem Cch.arcs_downB (t : Cch α) : t.arcs.downB = t.downB := rfl
@[simp] theorem Cch.arcs_upD (t : Cch α) : t.arcs.upD = t.upD := rfl
@[simp] theorem Cch.arcs_downD (t : Cch α) : t.arcs.downD = t.downD := rfl
@[simp] theorem Cch.arcs_cap (t : Cch α) : t.arcs.cap = t.cap := rfl
@[simp] theorem Cch.arcs_added (t : Cch α) : t.arcs.added = t.added := rfl
@[simp] theorem Cch.arcs_metLen (t : Cch α) : t.arcs.metLen = t.met.length := rfl

def Arcs.upE (a : Arcs) (e : Ent) : Prop := inB a.upB e ∨ e ∈ a.upD
def Arcs.downE (a : Arcs) (e : Ent) : Prop := inB a.downB e ∨ e ∈ a.downD

theorem upE_arcs (t : Cch α) (e : Ent) : upE t e ↔ t.arcs.upE e := Iff.rfl
theorem downE_arcs (t : Cch α) (e : Ent) : downE t e ↔ t.arcs.downE e := Iff.rfl

/-- I1′ and P (first half). -/
structure RInv (k : Ranks) : Prop where
  node_lt : ∀ r x, k.nodeOf r = some x → r < k.nr
  rank_node : ∀ x r, k.rankOf x = some r ↔ k.nodeOf r = some x
  free_iff : ∀ r, r ∈ k.free ↔ r < k.nr ∧ k.nodeOf r = none
  free_asc : k.free.Pairwise (· < ·)

/-- The arc-side invariant. -/
structure AInv (a : Arcs) : Prop where
  up_lt : ∀ e, a.upE e → e.row < e.col
  upB_strict : ∀ r, StrictCol (bRow a.upB r)
  downB_strict : ∀ r, StrictCol (bRow a.downB r)
  upD_sorted : KSorted a.upD
  downD_sorted : KSorted a.downD
  up_disj : ∀ e ∈ a.upD, ∀ i, ¬ inB a.upB ⟨e.row, e.col, i⟩
  down_disj : ∀ e ∈ a.downD, ∀ i, ¬ inB a.downB ⟨e.row, e.col, i⟩
  i7B : ∀ e, inB a.upB e ↔ inB a.downB (tr e)
  i7D : ∀ e, e ∈ a.upD ↔ tr e ∈ a.downD
  id_lt : ∀ e, a.upE e → e.id < a.cap
  id_inj : ∀ e e', a.upE e → a.upE e' → e.id = e'.id → e = e'
  met_len : a.metLen = a.cap
  added_eq : ∀ e, e ∈ a.upD ↔ e ∈ ents a.added
  added_nodup : (ents a.added).Nodup
  added_seq : ∀ p ∈ a.added, p.1 ≤ a.ver

/-- P (second half): every arc's ends are held ranks. -/
def Held (k : Ranks) (a : Arcs) : Prop := ∀ e, a.upE e → (k.nodeOf e.row).isSome ∧ (k.nodeOf e.col).isSome

/-- **The invariant** every version of a lineage keeps. -/
def Inv (t : Cch α) : Prop := RInv t.ranks ∧ AInv t.arcs ∧ Held t.ranks t.arcs

/-! ## Basic consequences -/

theorem mem_ents (a : List (Nat × List Ent)) (e : Ent) : e ∈ ents a ↔ ∃ p ∈ a, e ∈ p.2 := by
  simp [ents, List.mem_flatMap]

theorem ents_append (a b : List (Nat × List Ent)) : ents (a ++ b) = ents a ++ ents b := by
  simp [ents, List.flatMap_append]

/-- With I7 an arc is in `Down` exactly when its mirror is in `Up`. -/
theorem AInv.downE_iff {a : Arcs} (h : AInv a) (e : Ent) : a.downE e ↔ a.upE (tr e) := by
  unfold Arcs.downE Arcs.upE
  rw [h.i7B (tr e), h.i7D (tr e), tr_tr]

theorem AInv.delta_key_unique {a : Arcs} (h : AInv a) : ∀ x ∈ a.upD, ∀ y ∈ a.upD, sameKey x y → x = y :=
  ksorted_key_unique h.upD_sorted

/-- An `Up` arc's key determines its id: base and delta never both hold a key, and each holds a key once. -/
theorem AInv.up_key_unique {a : Arcs} (h : AInv a) (x y : Ent) (hx : a.upE x) (hy : a.upE y) (hs : sameKey x y) :
    x = y := by
  rcases hx with hx | hx <;> rcases hy with hy | hy
  · unfold inB at hx hy
    rw [hs.1] at hx
    have := strictCol_col_unique (h.upB_strict y.row) _ hx _ hy hs.2
    simp only [Prod.mk.injEq] at this
    cases x; cases y; simp_all [sameKey]
  · exact absurd (by have := hx; unfold inB at this ⊢; rw [hs.1, hs.2] at this; exact this) (h.up_disj y hy x.id)
  · exact absurd (by have := hy; unfold inB at this ⊢; rw [← hs.1, ← hs.2] at this; exact this) (h.up_disj x hx y.id)
  · exact h.delta_key_unique x hx y hy hs

end CchTopo
