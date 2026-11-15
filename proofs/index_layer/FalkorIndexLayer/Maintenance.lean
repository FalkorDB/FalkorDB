import FalkorIndexLayer.Model

/-! # Index maintenance (`Indexer::commit`) and population tickets -/

namespace IndexLayer

theorem foldl_add_lookup (build : Nat → Doc) : ∀ (adds : List Nat) (t : Table) (j : Nat),
    (adds.foldl (fun t id => t.add id (build id)) t) j = if j ∈ adds then some (build j) else t j
  | [], t, j => by simp
  | a :: as, t, j => by
    rw [List.foldl_cons, foldl_add_lookup build as]
    by_cases h1 : j ∈ as
    · simp [h1]
    · by_cases h2 : j = a
      · subst h2; simp [Table.add, h1]
      · simp [Table.add, h1, h2]

theorem foldl_del_lookup : ∀ (removes : List Nat) (t : Table) (j : Nat),
    (removes.foldl Table.del t) j = if j ∈ removes then none else t j
  | [], t, j => by simp
  | a :: as, t, j => by
    rw [List.foldl_cons, foldl_del_lookup as]
    by_cases h1 : j ∈ as
    · simp [h1]
    · by_cases h2 : j = a
      · subst h2; simp [Table.del, h1]
      · simp [Table.del, h1, h2]

/-- What `Indexer::commit` leaves for one id: removed ids have no document
(removes run after adds, `indexer.rs:728`), otherwise an added id has the
freshly built document (`ADD_REPLACE`), otherwise the old one. -/
theorem commit_lookup (t : Table) (build : Nat → Doc) (adds removes : List Nat) (j : Nat) :
    commit t build adds removes j =
      if j ∈ removes then none else if j ∈ adds then some (build j) else t j := by
  simp [commit, foldl_del_lookup, foldl_add_lookup]

/-- The index invariant for one label: an id has a document iff the entity
carries the label, and that document is the one built from its current
properties. -/
def Inv (t : Table) (has : Nat → Bool) (build : Nat → Doc) : Prop :=
  ∀ j, t j = if has j then some (build j) else none

/-- Index-maintenance consistency. If the transaction's bookkeeping
satisfies (1) every removed id no longer carries the label, (2) an id that
lost the label is removed, and (3) an id that has the label but whose
document is not rebuilt kept both the label and its document, then after
`commit` the RS table again reflects the committed graph exactly. The live
Cypher scenarios in `repro.py`/`diff2.py` (SET/REMOVE property, SET n = map,
REMOVE/SET label in the same query, DELETE, MERGE, edge SET/DELETE) all
satisfy these side conditions. -/
theorem commit_preserves_inv (t : Table) (has has' : Nat → Bool) (build build' : Nat → Doc)
    (adds removes : List Nat) (hinv : Inv t has build)
    (h1 : ∀ j ∈ removes, has' j = false)
    (h2 : ∀ j, has' j = false → j ∉ removes → has j = false ∧ j ∉ adds)
    (h3 : ∀ j, has' j = true → j ∉ adds → has j = true ∧ build' j = build j) :
    Inv (commit t build' adds removes) has' build' := by
  intro j
  rw [commit_lookup]
  by_cases hr : j ∈ removes
  · simp [hr, h1 j hr]
  · simp only [hr, ite_false]
    cases hh : has' j
    · obtain ⟨ho, ha⟩ := h2 j hh hr
      simp [ha, hinv j, ho]
    · by_cases ha : j ∈ adds
      · simp [ha]
      · obtain ⟨ho, hb⟩ := h3 j hh ha
        simp [ha, hinv j, ho, hb]

/-- Hazard: the add-then-remove order means an id present in both sets ends
with no document, even if it still carries the label. -/
theorem commit_both_removes (t : Table) (build : Nat → Doc) (adds removes : List Nat) (j : Nat)
    (hr : j ∈ removes) : commit t build adds removes j = none := by
  simp [commit_lookup, hr]

/-! ## Population tickets -/

def Slots.ok (s : Slots) : Prop := 0 ≤ s.cur ∧ 0 ≤ s.stale

/-- Acquire + release of a ticket is a no-op (current generation). -/
theorem inc_dec_cur (s : Slots) (h : s.ok) : (s.inc s.gen).dec s.gen = s := by
  unfold Slots.ok at h
  have : s.cur + 1 > 0 := by omega
  simp [Slots.inc, Slots.dec, this]

/-- Acquire + release of a ticket is a no-op (stale generation). -/
theorem inc_dec_stale (s : Slots) (g : Nat) (hg : g ≠ s.gen) (h : s.ok) : (s.inc g).dec g = s := by
  unfold Slots.ok at h
  have : s.stale + 1 > 0 := by omega
  simp [Slots.inc, Slots.dec, hg, this]

theorem ok_inc (s : Slots) (g : Nat) (h : s.ok) : (s.inc g).ok := by
  unfold Slots.inc Slots.ok at *; split <;> simp <;> omega

theorem ok_dec (s : Slots) (g : Nat) (h : s.ok) : (s.dec g).ok := by
  unfold Slots.dec Slots.ok at *
  by_cases h1 : g = s.gen <;> by_cases h2 : s.cur > 0 <;> by_cases h3 : s.stale > 0 <;>
    simp [h1, h2, h3] <;> omega

theorem ok_bump (s : Slots) (g : Nat) (h : s.ok) : (s.bump g).ok := by
  unfold Slots.bump Slots.ok at *; simp; omega

/-- `bump_id` never loses a ticket: total pending is conserved. -/
theorem bump_total (s : Slots) (g : Nat) : (s.bump g).cur + (s.bump g).stale = s.cur + s.stale := by
  simp [Slots.bump]; omega

/-- After `bump_id` the new generation starts operational (pending 0), so a
recreated index is not blocked by workers of the dropped spec. -/
theorem bump_fresh (s : Slots) (g : Nat) : (s.bump g).countFor g = 0 := by
  simp [Slots.bump, Slots.countFor]

/-- Releasing a stale ticket never touches the current generation's counter
(so `is_operational` of the fresh index is unaffected). -/
theorem dec_stale_keeps_current (s : Slots) (g : Nat) (hg : g ≠ s.gen) :
    (s.dec g).countFor s.gen = s.countFor s.gen := by
  simp [Slots.dec, Slots.countFor, hg]; split <;> simp

theorem inc_stale_keeps_current (s : Slots) (g : Nat) (hg : g ≠ s.gen) :
    (s.inc g).countFor s.gen = s.countFor s.gen := by
  simp [Slots.inc, Slots.countFor, hg]

end IndexLayer
