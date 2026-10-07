import FalkorIndexLayer.Proofs

/-! # Temporals are not indexed (#3076, e20300436)

`Document::set` now skips `Datetime`/`Date`/`Time`/`Duration` (`mod.rs:796-803`). So a stored temporal
contributes nothing to its attribute's RediSearch document, and every numeric or scalar index query
over that attribute is exact for it: the index does not return it, and Cypher's comparison of a
temporal with a number or string is null (not true). This turns the old counterexample
`pre3076_temporal_in_numeric_range` (W2-index-5) into theorems. -/

namespace IndexLayer

/-- A stored temporal leaves its attribute's document empty, in every sub-field. -/
theorem temporal_not_indexed (fields : Attr → Bool) (p : Props) (k : Attr) (kd : Nat) (ts : Int)
    (hpk : p k = some (.temporal kd ts)) (s : Sub) : docOf fields p k s = [] := by
  unfold docOf
  split
  · rw [hpk]; cases s <;> rfl
  · rfl

/-- **Numeric ranges are exact on a stored temporal** (both sides are false). -/
theorem temporal_numRange_exact (fields : Attr → Bool) (p : Props) (k : Attr)
    (mn mx : Option Val) (imn imx : Bool) (hmn : ∀ v, mn = some v → NumConst v)
    (hmx : ∀ v, mx = some v → NumConst v) (hb : mn.isSome ∨ mx.isSome)
    (kd : Nat) (ts : Int) (hpk : p k = some (.temporal kd ts)) :
    indexHit fields p (.range k mn mx imn imx) = false ∧ holds p (.range k mn mx imn imx) = false := by
  have hdoc := temporal_not_indexed fields p k kd ts hpk
  have key : ∀ (o : Option Val), (∀ v, o = some v → NumConst v) →
      o = none ∨ (∃ i, o = some (.int i)) ∨ (∃ f, o = some (.flt f)) := by
    intro o ho
    rcases o with _ | v
    · exact Or.inl rfl
    · cases ho v rfl with
      | int i => exact Or.inr (Or.inl ⟨i, rfl⟩)
      | flt f => exact Or.inr (Or.inr ⟨f, rfl⟩)
  rcases key mn hmn with rfl | ⟨a, rfl⟩ | ⟨a, rfl⟩ <;>
  rcases key mx hmx with rfl | ⟨b, rfl⟩ | ⟨b, rfl⟩ <;>
  (try simp at hb) <;>
  cases imn <;> cases imx <;> cases hfk : fields k <;>
  simp [indexHit, buildQ, isStrVal, buildNumRange, valueToNumeric, rsMatch, hdoc, holds, propOr, hpk,
    cyLt, cyLe, cyEqScalar, numOf, isTrue, hfk]

/-- **Scalar equality is exact on a stored temporal** (both sides are false). -/
theorem temporal_equal_exact (fields : Attr → Bool) (p : Props) (k : Attr) (c : Val) (hc : Const c)
    (kd : Nat) (ts : Int) (hpk : p k = some (.temporal kd ts)) :
    indexHit fields p (.equal k c) = false ∧ holds p (.equal k c) = false := by
  have hdoc := temporal_not_indexed fields p k kd ts hpk
  cases hc <;> cases hfk : fields k <;>
    simp [indexHit, buildQ, buildEq, valueToNumeric, holds, propOr, hpk, cyEqScalar, isTrue, rsMatch, hdoc, hfk, numOf]

end IndexLayer
