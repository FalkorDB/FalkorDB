import Columnar.Lanes

/-!
# Every lane returns a column of the batch length (`EC.WF`)

The agreement theorems (`compareColumns_agree`, `arithmetic_agree`, …) take well-formed
operand columns; these lemmas close the loop for nested expressions: `compare_columns`,
`arithmetic`, `negate_bools` and `eval_variable` return columns with exactly `len` rows.
-/

namespace Columnar
open FalkorCore (V)

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

theorem maskOut_len (res : List Bool) (nulls : Bits) : (maskOut res nulls).length = res.length :=
  maskOut_length res nulls

theorem compareColumns_wf (other : V F → V F → Ordering × Flag) (lhs rhs : EC F) (op : CmpOp)
    (len : Nat) (hl : lhs.WF len) (hr : rhs.WF len) : (compareColumns other lhs rhs op len).WF len := by
  have hn := unionNulls_length lhs rhs len hl hr
  unfold compareColumns
  cases ht : typedLane lhs rhs op (unionNulls lhs rhs len) with
  | some mask =>
    refine ⟨?_, hn⟩
    unfold typedLane at ht
    split at ht <;> simp at ht <;> subst ht <;> simp_all [EC.WF, kernelI, kernelF, maskOut_len]
  | none =>
    simp only
    unfold genericLane
    split
    · rename_i c d
      simp only [EC.WF] at hl
      simp [EC.WF, kernelV, hl]
    · simp [EC.WF]

theorem negate_wf (child : EC F) (len : Nat) (col : EC F) (h : negate child len = .ok col) : col.WF len := by
  unfold negate at h
  simp only [Except.map] at h
  split at h
  · simp at h
  · rename_i ps hm
    simp only [Except.ok.injEq] at h; subst h
    have := mapM_length hm
    simp only [List.length_range] at this
    simp [EC.WF, this]

theorem intLane_len {c : EC F} {len : Nat} {a : List Int} (hw : c.WF len) (h : intLane c len = some a) :
    a.length = len := by
  cases c with
  | ints d n => simp only [intLane, Option.some.injEq] at h; subst h; exact hw.1
  | floats d n => simp [intLane] at h
  | bools d n => simp [intLane] at h
  | values d => simp [intLane] at h
  | scalar v => cases v <;> simp [intLane] at h <;> (subst h; simp)

theorem floatLane_len {c : EC F} {len : Nat} {a : List F} (hw : c.WF len) (h : floatLane c len = some a) :
    a.length = len := by
  cases c with
  | ints d n => simp only [floatLane, Option.some.injEq] at h; subst h; simp [hw.1]
  | floats d n => simp only [floatLane, Option.some.injEq] at h; subst h; exact hw.1
  | bools d n => simp [floatLane] at h
  | values d => simp [floatLane] at h
  | scalar v => cases v <;> simp [floatLane] at h <;> (subst h; simp)

theorem arithmetic_wf (other : AOp → V F → V F → Except String (V F)) (op : AOp) (lhs rhs : EC F)
    (len : Nat) (hl : lhs.WF len) (hr : rhs.WF len) (col : EC F)
    (h : arithmetic other op lhs rhs len = .ok col) : col.WF len := by
  have hn := unionNulls_length lhs rhs len hl hr
  unfold arithmetic at h
  cases hi : intStage op lhs rhs len with
  | some c =>
    simp only [hi, Except.ok.injEq] at h; subst h
    unfold intStage at hi
    cases ha : intLane lhs len with
    | none => simp [ha] at hi
    | some a =>
      cases hb : intLane rhs len with
      | none => simp [ha, hb] at hi
      | some b =>
        have la := intLane_len hl ha; have lb := intLane_len hr hb
        simp only [ha, hb] at hi
        split at hi
        · split at hi
          · simp only [Option.some.injEq] at hi; subst hi; simp [EC.WF, hn]
          · simp at hi
        · simp only [Option.some.injEq] at hi; subst hi; simp [EC.WF, hn, la, lb]
  | none =>
    simp only [hi] at h
    cases hf : floatStage op lhs rhs len with
    | some c =>
      simp only [hf, Except.ok.injEq] at h; subst h
      unfold floatStage at hf
      split at hf
      · cases ha : floatLane lhs len with
        | none => simp [ha] at hf
        | some a =>
          cases hb : floatLane rhs len with
          | none => simp [ha, hb] at hf
          | some b =>
            have la := floatLane_len hl ha; have lb := floatLane_len hr hb
            simp only [ha, hb, Option.some.injEq] at hf; subst hf; simp [EC.WF, hn, la, lb]
      · simp at hf
    | none =>
      simp only [hf] at h
      unfold genericStage at h
      simp only [Except.map] at h
      split at h
      · simp at h
      · rename_i vs hm
        simp only [Except.ok.injEq] at h; subst h
        have := mapM_length hm; simpa [EC.WF] using this

theorem evalVariable_wf (b : Batch F) (i : Nat) (rows : List Nat) (c : EC F) (h : evalVariable b i rows = some c) :
    c.WF rows.length := by
  unfold evalVariable at h
  cases hc : b.column i with
  | ints d =>
    simp only [hc, Option.map_eq_some_iff] at h
    obtain ⟨g, hg, rfl⟩ := h
    simp [EC.WF, gatherL_length hg, Bits.none]
  | floats d =>
    simp only [hc, Option.map_eq_some_iff] at h
    obtain ⟨g, hg, rfl⟩ := h
    simp [EC.WF, gatherL_length hg, Bits.none]
  | nodeIds d | relIds d | values d | unbound =>
    simp only [hc, Option.map_eq_some_iff] at h
    obtain ⟨vs, hvs, rfl⟩ := h
    simp [EC.WF, mapO_length hvs]

end Columnar
