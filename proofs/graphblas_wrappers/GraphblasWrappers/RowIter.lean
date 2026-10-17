/-
# `matrix::Iter`: row-range iteration is exactly the entries in range, once each

Model of `Matrix::Iter` (matrix.rs:1604-1756) — `seek`, `next`, and the two
`GxB_rowIterator_*` drive loops.

The underlying `GxB_rowIterator` is axiomatised (it is C). Its documented
contract (GraphBLAS.h:7796-7990) is: attached to a matrix, `seekRow(r)` /
`nextRow()` move to the first *stored* entry on row `≥ r` (empty rows may be
skipped; on sparse storage `seekRow` lands on an empty row and returns
`GrB_NO_VALUE`, which the Rust loop then walks past — matrix.rs:1712-1716,
1715-1719), and `nextCol()` walks the stored entries within a row in ascending
column order, returning `GrB_NO_VALUE` at the row's end. `getRowIndex()` returns
the current row, or `nrows` when exhausted.

The net effect of the two drive loops is therefore: **visit every stored entry
with `min_row ≤ row ≤ max_row`, in row-major (row, then column) order, once.**
That is what this file models — the cursor is the sorted list of stored entries —
and proves. The `NO_VALUE` empty-row skipping folds into "advance to the next
stored entry", justified by the header spec cited above.
-/

namespace GBW

/-- A stored entry `(row, col)`. -/
abbrev Entry := Nat × Nat

/-- `depleted` is set (matrix.rs:1718-1719, 1750-1751) when there is no current
entry or the current row exceeds `max_row`. -/
def depletedFor : List Entry → Nat → Bool
  | [], _ => true
  | e :: _, maxRow => decide (e.1 > maxRow)

/-- Drive loop over the remaining stored entries: `next()` yields the head
unless depleted, then advances and recomputes `depleted`. Structural on the
list. Mirrors matrix.rs:1732-1755. -/
def drainL : List Entry → Nat → Bool → List Entry
  | [], _, _ => []
  | e :: rest, maxRow, depleted =>
      if depleted then [] else e :: drainL rest maxRow (depletedFor rest maxRow)

/-- `seek(min_row, max_row)` (matrix.rs:1699): skip stored entries on rows below
`min_row`, then drive to exhaustion. -/
def drainSeek (entries : List Entry) (minRow maxRow : Nat) : List Entry :=
  let rem := entries.filter (fun e => decide (e.1 ≥ minRow))
  drainL rem maxRow (depletedFor rem maxRow)

/-- The specification: the stored entries whose row lies in `[minRow, maxRow]`. -/
def spec (entries : List Entry) (minRow maxRow : Nat) : List Entry :=
  entries.filter (fun e => decide (minRow ≤ e.1 ∧ e.1 ≤ maxRow))

/-- Sortedness by row: each row `≤` the next. -/
def rowSorted : List Entry → Prop
  | [] => True
  | [_] => True
  | a :: b :: t => a.1 ≤ b.1 ∧ rowSorted (b :: t)

theorem rowSorted_tail {a : Entry} {t : List Entry} (h : rowSorted (a :: t)) : rowSorted t := by
  cases t with
  | nil => trivial
  | cons b t => exact h.2

theorem rowSorted_head_le {a b : Entry} {t : List Entry} (h : rowSorted (a :: b :: t)) :
    a.1 ≤ b.1 := h.1

/-- Head row is `≤` the row of every element of a row-sorted list. -/
theorem rowSorted_head_min {a : Entry} :
    ∀ {t : List Entry}, rowSorted (a :: t) → ∀ e ∈ t, a.1 ≤ e.1 := by
  intro t
  induction t generalizing a with
  | nil => intro _ e he; simp at he
  | cons b t ih =>
    intro h e he
    have hab : a.1 ≤ b.1 := rowSorted_head_le h
    rcases List.mem_cons.mp he with h1 | h1
    · subst h1; exact hab
    · have := ih (rowSorted_tail h) e h1; omega

/-- Once a row exceeds `maxRow`, the whole (sorted) tail filters to nothing. -/
theorem filter_over_max (maxRow : Nat) :
    ∀ {l : List Entry}, rowSorted l → (∀ e ∈ l, e.1 > maxRow) →
      l.filter (fun e => decide (e.1 ≤ maxRow)) = [] := by
  intro l
  induction l with
  | nil => intro _ _; rfl
  | cons a t ih =>
    intro hs hall
    have ha : a.1 > maxRow := hall a (by simp)
    have hbad : (decide (a.1 ≤ maxRow)) = false := by rw [decide_eq_false_iff_not]; omega
    simp only [List.filter_cons, hbad, if_false]
    exact ih (rowSorted_tail hs) (fun e he => hall e (List.mem_cons.mpr (Or.inr he)))

/-- The drive loop over a row-sorted remaining list yields exactly its entries
with row `≤ maxRow`: it stops at the first over-`maxRow` entry, sound because all
later rows are also over it. -/
theorem drainL_eq_filter :
    ∀ (rem : List Entry) (maxRow : Nat), rowSorted rem →
      drainL rem maxRow (depletedFor rem maxRow)
        = rem.filter (fun e => decide (e.1 ≤ maxRow)) := by
  intro rem
  induction rem with
  | nil => intro maxRow _; simp [drainL]
  | cons e rest ih =>
    intro maxRow hs
    have key := ih maxRow (rowSorted_tail hs)
    by_cases he : e.1 ≤ maxRow
    · have h1 : depletedFor (e :: rest) maxRow = false := by
        simp only [depletedFor, decide_eq_false_iff_not]; omega
      have h2 : (e :: rest).filter (fun e => decide (e.1 ≤ maxRow))
              = e :: rest.filter (fun e => decide (e.1 ≤ maxRow)) := by
        simp [List.filter_cons, he]
      rw [drainL, h1, if_neg (by simp), h2, key]
    · have h1 : depletedFor (e :: rest) maxRow = true := by
        simp only [depletedFor, decide_eq_true_eq]; omega
      have h2 : (e :: rest).filter (fun e => decide (e.1 ≤ maxRow)) = [] := by
        apply filter_over_max maxRow hs
        intro f hf
        rcases List.mem_cons.mp hf with h | h
        · subst h; omega
        · have := rowSorted_head_min hs f h; omega
      rw [drainL, h1, if_pos (by simp), h2]

/-- `filter` preserves the `rowSorted` shape. -/
theorem rowSorted_filter (p : Entry → Bool) :
    ∀ {l : List Entry}, rowSorted l → rowSorted (l.filter p) := by
  intro l
  induction l with
  | nil => intro _; simp [List.filter]; trivial
  | cons a t ih =>
    intro h
    rw [List.filter_cons]
    by_cases hp : p a
    · rw [if_pos hp]
      have htail := ih (rowSorted_tail h)
      -- prepend a: a.1 ≤ every remaining row
      cases hfl : t.filter p with
      | nil => exact trivial
      | cons c cr =>
        refine ⟨?_, by rw [← hfl]; exact htail⟩
        have hc : c ∈ t.filter p := by rw [hfl]; simp
        have hcmem : c ∈ t := (List.mem_filter.mp hc).1
        exact rowSorted_head_min h c hcmem
    · rw [if_neg hp]; exact ih (rowSorted_tail h)

/-- **Row iteration is its specification.** For a row-sorted stored-entry list,
`seek` then drain yields exactly the entries with `minRow ≤ row ≤ maxRow`, in
order and once each. -/
theorem drain_seek_eq_spec (entries : List Entry) (minRow maxRow : Nat)
    (hsorted : rowSorted entries) :
    drainSeek entries minRow maxRow = spec entries minRow maxRow := by
  unfold drainSeek spec
  rw [drainL_eq_filter _ _ (rowSorted_filter _ hsorted)]
  rw [List.filter_filter]
  apply List.filter_congr
  intro e _
  by_cases h1 : minRow ≤ e.1 <;> by_cases h2 : e.1 ≤ maxRow <;>
    simp [h1, h2, Bool.and_eq_true, decide_eq_true_eq] <;> omega

/-- Corollary: **exactly once**. No entry in range is dropped or repeated —
`drain` is the filtered sublist, so per-entry counts match one-for-one. -/
theorem drain_no_duplication (entries : List Entry) (minRow maxRow : Nat)
    (hsorted : rowSorted entries) (e : Entry) :
    (drainSeek entries minRow maxRow).count e
      = (spec entries minRow maxRow).count e := by
  rw [drain_seek_eq_spec entries minRow maxRow hsorted]

/-- Empty matrix yields nothing — the `detached`/depleted fast paths
(matrix.rs:1663). -/
theorem drain_empty_matrix (minRow maxRow : Nat) :
    drainSeek [] minRow maxRow = [] := by
  simp [drainSeek, drainL, depletedFor]

end GBW
