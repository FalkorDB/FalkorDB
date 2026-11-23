/-
# `replace_cartesian_with_index_seek` (#3889): the `UNWIND range(..)` row estimate

`unwound_len` (`replace_cartesian_with_index_seek.rs`) estimates how many rows
`UNWIND range(from, to[, step])` yields, from integer constants, as
`(to - from) / step + 1` (i128, truncating division) clamped at `0`; a zero step
gives no estimate. `rangeLen` is the length of the list the `range` function
(`runtime/functions/math.rs`) returns: empty when the step points away from
`to`, else `|to - from| / |step| + 1`.

* `rangeEst_exact`: when the step points towards `to` (or `from = to`) the
  estimate is the exact length.
* `rangeEst_upper`: every estimate is an upper bound on the length (the pass
  only uses it to bound the rows fed to an index seek).
* `rangeEst_overcount`: it is not always exact: `range(0, -1, 2)` is empty but
  is estimated at one row (harmless: an upper bound).

Abstraction: `usize::try_from` cannot fail on a 64-bit target for these values
(`|to - from| < 2^65`), so the clamp is `Int.toNat`.
-/

namespace Falkor.Opt.IndexSeek

/-- The step points away from `to`: `range` returns `[]` (math.rs). -/
def wrongDir (start stop step : Int) : Bool :=
  (decide (start > stop) && decide (step > 0)) || (decide (start < stop) && decide (step < 0))

/-- Length of `range(start, stop, step)`, `step ≠ 0` (math.rs `range`). -/
def rangeLen (start stop step : Int) : Nat :=
  if wrongDir start stop step then 0 else (stop - start).natAbs / step.natAbs + 1

/-- `unwound_len`, `range` arm: `None` on a zero step, else
`max((to - from) / step + 1, 0)` with truncating division. -/
def rangeEst (start stop step : Int) : Option Nat :=
  if step = 0 then none else some ((stop - start).tdiv step + 1).toNat

private theorem tdiv_eq_of_nonneg {d s : Int} (h : 0 ≤ d.tdiv s) :
    d.tdiv s = ((d.natAbs / s.natAbs : Nat) : Int) := by
  have h1 : ((d.tdiv s).natAbs : Int) = d.tdiv s := Int.natAbs_of_nonneg h
  rw [Int.natAbs_tdiv] at h1
  exact h1.symm

private theorem toNat_succ (n : Nat) : (((n : Int)) + 1).toNat = n + 1 := by
  have : ((n : Int) + 1) = ((n + 1 : Nat) : Int) := by simp
  rw [this, Int.toNat_natCast]

/-- **PROVEN**: when the step points towards `to`, the estimate is the exact length. -/
theorem rangeEst_exact (start stop step : Int) (h0 : step ≠ 0)
    (hd : wrongDir start stop step = false) :
    rangeEst start stop step = some (rangeLen start stop step) := by
  simp only [rangeEst, rangeLen, h0, hd, ite_false]
  simp only [Bool.false_eq_true, ite_false]
  have hnn : 0 ≤ (stop - start).tdiv step := by
    simp [wrongDir] at hd
    rcases Int.lt_or_gt_of_ne h0 with hs | hs
    · have : start ≥ stop := by
        by_cases hc : start < stop
        · exact absurd hs (Int.not_lt.mpr (hd.2 hc))
        · exact Int.not_lt.mp hc
      exact Int.tdiv_nonneg_of_nonpos_of_nonpos (by omega) (by omega)
    · have : start ≤ stop := by
        by_cases hc : start > stop
        · exact absurd hs (Int.not_lt.mpr (hd.1 hc))
        · exact Int.not_lt.mp hc
      exact Int.tdiv_nonneg (by omega) (by omega)
  rw [tdiv_eq_of_nonneg hnn, toNat_succ]

/-- **PROVEN**: every estimate bounds the length of the unwound list from above. -/
theorem rangeEst_upper (start stop step : Int) (k : Nat)
    (h : rangeEst start stop step = some k) : rangeLen start stop step ≤ k := by
  have h0 : step ≠ 0 := by
    intro hz; simp [rangeEst, hz] at h
  cases hd : wrongDir start stop step
  · rw [rangeEst_exact start stop step h0 hd] at h
    cases h; exact Nat.le_refl _
  · simp [rangeLen, hd]

/-- The estimate is not always exact: `range(0, -1, 2)` is `[]`, estimated at 1. -/
theorem rangeEst_overcount : rangeEst 0 (-1) 2 = some 1 ∧ rangeLen 0 (-1) 2 = 0 := by
  decide

/-- A zero step (a `range` that raises) gives no estimate. -/
theorem rangeEst_zero (start stop : Int) : rangeEst start stop 0 = none := by
  simp [rangeEst]

end Falkor.Opt.IndexSeek
