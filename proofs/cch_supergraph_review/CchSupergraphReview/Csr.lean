import CchSupergraphReview.Order
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-!
# Row arrays: offsets, positions and the ids they give (03 §2 "Arc identity", §4 step 4)

Facts about any CSR `(offsOf R n, (rowsOf R n).flatten)` built from rows `R 0 .. R (n-1)`: entry `i` of row `r`
sits at position `offsets[r] + i` (`flatten_getElem?`), the rows tile the array (`locate`), and positions are
unique (`pos_inj`). With `R = gameRow` these are the arc slots of `Topology::up_range` (`order.rs:262-268`), and
with 03 §4 step 4's row-major ids (`id = Ap[lo] + i`) they are the arc ids.
-/
namespace CchSG

variable (R : Nat → List Nat)

theorem offsOf_length (n : Nat) : (offsOf R n).length = n + 1 := by simp [offsOf]

theorem offsOf_getD_succ {n h : Nat} (hh : h < n) :
    (offsOf R n).getD (h + 1) 0 = (offsOf R n).getD h 0 + (R h).length := by
  rw [offsOf_getD R (by omega), offsOf_getD R (by omega), rowsOf_succ]; simp

theorem offsOf_mono {n a b : Nat} (hab : a ≤ b) (hb : b ≤ n) : (offsOf R n).getD a 0 ≤ (offsOf R n).getD b 0 := by
  induction b with
  | zero => have : a = 0 := by omega
            subst this; exact Nat.le_refl _
  | succ b ih =>
    by_cases h : a = b + 1
    · subst h; exact Nat.le_refl _
    · rw [offsOf_getD_succ R (by omega)]; exact Nat.le_trans (ih (by omega) (by omega)) (Nat.le_add_right _ _)

theorem flatten_length (n : Nat) : (rowsOf R n).flatten.length = (offsOf R n).getD n 0 := by
  rw [offsOf_getD R (Nat.le_refl n)]

/-- Entry `j` of row `h` is at position `offsets[h] + j`. -/
theorem flatten_getElem? {n h j : Nat} (hh : h < n) (hj : j < (R h).length) :
    (rowsOf R n).flatten[(offsOf R n).getD h 0 + j]? = (R h)[j]? := by
  have hs := slice_rowOf R hh
  have e1 := offsOf_getD_succ R hh
  unfold slice rowSlots at hs
  simp only at hs
  rw [e1, Nat.add_sub_cancel_left] at hs
  rw [← hs, List.getElem?_take, ite_eq_left hj, List.getElem?_drop]

/-- Every position below the last offset lies in exactly one row. -/
theorem locate {n q : Nat} (hq : q < (offsOf R n).getD n 0) :
    ∃ h, h < n ∧ (offsOf R n).getD h 0 ≤ q ∧ q < (offsOf R n).getD h 0 + (R h).length := by
  induction n with
  | zero => simp [offsOf, rowsOf] at hq
  | succ n ih =>
    have e : ∀ i, i ≤ n → (offsOf R (n + 1)).getD i 0 = (offsOf R n).getD i 0 := by
      intro i hi; rw [offsOf_getD R hi, offsOf_getD R (by omega)]
    by_cases hlt : q < (offsOf R n).getD n 0
    · obtain ⟨h, hh, h1, h2⟩ := ih hlt
      exact ⟨h, by omega, by rw [e h (by omega)]; exact h1, by rw [e h (by omega)]; exact h2⟩
    · refine ⟨n, by omega, by rw [e n (Nat.le_refl _)]; omega, ?_⟩
      rw [e n (Nat.le_refl _)]
      rw [offsOf_getD_succ R (Nat.lt_succ_self n), e n (Nat.le_refl _)] at hq; exact hq

/-- Positions are unique: `(row, index)` is determined by the position. -/
theorem pos_inj {n h h' j j' : Nat} (hh : h < n) (hh' : h' < n) (hj : j < (R h).length)
    (hj' : j' < (R h').length) (he : (offsOf R n).getD h 0 + j = (offsOf R n).getD h' 0 + j') :
    h = h' ∧ j = j' := by
  rcases Nat.lt_trichotomy h h' with hlt | heq | hgt
  · have a := offsOf_getD_succ R hh
    have b := offsOf_mono R (by omega : h + 1 ≤ h') (Nat.le_of_lt hh')
    omega
  · subst heq; exact ⟨rfl, by omega⟩
  · have a := offsOf_getD_succ R hh'
    have b := offsOf_mono R (by omega : h' + 1 ≤ h) (Nat.le_of_lt hh)
    omega

/-- Two lists that agree on every row's positions and have the array's length are equal. -/
theorem eq_flatten_of_rows {n : Nat} (L : List Nat) (hl : L.length = (offsOf R n).getD n 0)
    (hrow : ∀ h, h < n → ∀ j, j < (R h).length → L[(offsOf R n).getD h 0 + j]? = (R h)[j]?) :
    L = (rowsOf R n).flatten := by
  apply List.ext_getElem?
  intro q
  by_cases hq : q < (offsOf R n).getD n 0
  · obtain ⟨h, hh, h1, h2⟩ := locate R hq
    have := hrow h hh (q - (offsOf R n).getD h 0) (by omega)
    rw [Nat.add_sub_cancel' h1] at this
    rw [this, ← flatten_getElem? R hh (by omega), Nat.add_sub_cancel' h1]
  · rw [List.getElem?_eq_none (by omega), List.getElem?_eq_none (by rw [flatten_length]; omega)]

end CchSG
