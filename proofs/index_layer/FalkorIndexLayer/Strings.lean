import FalkorIndexLayer.Proofs

/-! # String ranges are exact (#2975: `tag_encode_lower` is order-preserving) -/

namespace IndexLayer

theorem beq_bytes (a b : Bytes) : (a == b) = decide (a = b) := by
  by_cases h : a = b <;> simp [h]

/-- String ranges are exact. Since #2975 the TAG key is order-preserving
(`tagEnc_lexLt`) and injective (`tagEnc_beq`), so the bounds need no side
condition (before, every string had to avoid the bytes `tag_encode_lower`
escaped: `plain`; see the removed `bug_string_range_order`). The only premise
on a stored string is that it is a Rust `String` (UTF-8, `Utf8`), so its key
sorts below the open upper end `STRING_RANGE_END` (`mod.rs:1606-1614`). The
optimizer's equal-bounds case needs no side condition since #3072
(1c9994e37). -/
theorem strRange_exact (fields : Attr → Bool) (p : Props) (k : Attr)
    (lo hi : Option Bytes) (imn imx : Bool) (hf : fields k = true)
    (hb : lo.isSome ∨ hi.isSome)
    (hp : ∀ v, p k = some v → Pure v)
    (hps : ∀ t, p k = some (.str t) → Utf8 t) :
    indexHit fields p (.range k (lo.map .str) (hi.map .str) imn imx) =
      holds p (.range k (lo.map .str) (hi.map .str) imn imx) := by
  have common : ∀ (q : QN), (∀ d : Doc, (∀ s, d k .main = s → ∀ x ∈ s, ∃ u, x = .tag u) →
      rsMatch d q = false ∨ True) → True := fun _ _ => trivial
  clear common
  rcases lo with _ | a <;> rcases hi with _ | b
  · simp at hb
  · -- only an upper bound
    simp only [indexHit, buildQ, Option.map, isStrVal, strBound, Bool.or_true, Bool.true_or,
      ite_true, buildStrRange, hf]
    cases hpk : p k with
    | none => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
    | some v =>
      cases hp v hpk with
      | int i => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | flt f => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | str t ht htb =>
        have hu : Utf8 t := hps t hpk
        have ht' : (t = []) = False := by simp [ht]
        cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange, hf, tagEnc_lexLt, tagEnc_beq, tagEnc_inj, tagEnc_below_end t hu, ht', inLex, beq_bytes]
  · -- only a lower bound
    simp only [indexHit, buildQ, Option.map, isStrVal, strBound, Bool.or_true, Bool.true_or,
      ite_true, buildStrRange, hf]
    cases hpk : p k with
    | none => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
    | some v =>
      cases hp v hpk with
      | int i => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | flt f => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | str t ht htb =>
        have hu : Utf8 t := hps t hpk
        have ht' : (t = []) = False := by simp [ht]
        cases imn <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange, hf, tagEnc_lexLt, tagEnc_beq, tagEnc_inj, tagEnc_below_end t hu, ht', inLex, beq_bytes]
  · simp only [indexHit, buildQ, Option.map, isStrVal, strBound, Bool.or_true, Bool.true_or,
      ite_true, buildStrRange, hf]
    by_cases hab : a = b
    · subst hab
      by_cases hboth : ¬ (imn = true ∧ imx = true)
      · -- #3072: an exclusive side with equal bounds is the empty node, and selects nothing
        have hne : (!(imn && imx)) = true := by cases imn <;> cases imx <;> simp_all
        simp only [ite_true, hne, rsMatch]
        cases hpk : p k with
        | none => cases imn <;> cases imx <;> simp_all [holds, propOr, cyLt, cyLe, cyEqScalar, numOf, isTrue]
        | some v =>
          cases hp v hpk with
          | int i => cases imn <;> cases imx <;> simp_all [holds, propOr, cyLt, cyLe, cyEqScalar, numOf, isTrue]
          | flt f => cases imn <;> cases imx <;> simp_all [holds, propOr, cyLt, cyLe, cyEqScalar, numOf, isTrue]
          | str t ht htb =>
            simp only [holds, propOr, hpk, Option.getD_some, cyLe, cyLt, cyEqScalar, beq_bytes]
            by_cases hta : t = a
            · subst hta; cases imn <;> cases imx <;> simp_all [lexLt_irrefl]
            · rcases lexLt_total a t (Ne.symm hta) with h | h
              · cases imn <;> cases imx <;> simp_all [lexLt_asymm a t h, Ne.symm hta]
              · cases imn <;> cases imx <;> simp_all [lexLt_asymm t a h, Ne.symm hta]
      obtain ⟨rfl, rfl⟩ := Classical.not_not.1 hboth
      simp only [ite_true, Bool.and_self, Bool.not_true, Bool.false_eq_true, ite_false]
      cases hpk : p k with
      | none => simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | some v =>
        cases hp v hpk with
        | int i => simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
        | flt f => simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
        | str t ht htb =>
          have hu : Utf8 t := hps t hpk
          have ht' : (t = []) = False := by simp [ht]
          simp only [rsMatch, docOf, hf, ite_true, hpk, setRange, List.flatMap_cons,
            List.flatMap_nil, rsStore_tag, tagEnc_lexLt, tagEnc_beq, tagEnc_inj, tagEnc_below_end t hu, ht', ite_false, List.append_nil,
            List.any_cons, List.any_nil, Bool.or_false, holds, propOr, Option.getD_some,
            cyLe, cyLt, cyEqScalar, isTrue_some, beq_bytes]
          by_cases hta : t = a
          · subst hta; simp [lexLt_irrefl]
          · rcases lexLt_total a t (Ne.symm hta) with h | h
            · simp [lexLt_asymm a t h, h, hta, Ne.symm hta, tagEnc_inj]
            · simp [lexLt_asymm t a h, h, hta, Ne.symm hta, tagEnc_inj]
    · simp only [hab, ite_false]
      cases hpk : p k with
      | none => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | some v =>
        cases hp v hpk with
        | int i => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
        | flt f => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
        | str t ht htb =>
          have hu : Utf8 t := hps t hpk
          have ht' : (t = []) = False := by simp [ht]
          cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange, hf, tagEnc_lexLt, tagEnc_beq, tagEnc_inj, ht', inLex, beq_bytes]

end IndexLayer
