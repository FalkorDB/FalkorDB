import FalkorIndexLayer.Proofs

/-! # String ranges: exact on the bytes `tag_encode_lower` leaves alone -/

namespace IndexLayer

theorem tagEnc_id : ∀ s : Bytes, (∀ b ∈ s, escaped b = false) → tagEnc s = s
  | [], _ => rfl
  | b :: bs, h => by
    have hb := h b (by simp)
    simp only [tagEnc, hb, Bool.false_eq_true, ite_false]
    rw [tagEnc_id bs (fun x hx => h x (by simp [hx]))]

theorem lexLt_irrefl : ∀ a : Bytes, lexLt a a = false
  | [] => rfl
  | x :: xs => by simp [lexLt, lexLt_irrefl xs]

theorem lexLt_asymm : ∀ a b : Bytes, lexLt a b = true → lexLt b a = false
  | [], [], h => by simp [lexLt] at h
  | [], _ :: _, _ => rfl
  | _ :: _, [], h => by simp [lexLt] at h
  | x :: xs, y :: ys, h => by
    simp only [lexLt] at h ⊢
    by_cases hxy : x < y
    · have : ¬ y < x := by omega
      have : y ≠ x := by omega
      simp [*]
    · by_cases he : x = y
      · subst he; simp at h ⊢; exact lexLt_asymm xs ys h
      · simp [hxy, he] at h

theorem lexLt_total : ∀ a b : Bytes, a ≠ b → lexLt a b = true ∨ lexLt b a = true
  | [], [], h => absurd rfl h
  | [], _ :: _, _ => Or.inl rfl
  | _ :: _, [], _ => Or.inr rfl
  | x :: xs, y :: ys, h => by
    simp only [lexLt]
    by_cases hxy : x < y
    · simp [hxy]
    · by_cases he : x = y
      · subst he
        have : xs ≠ ys := fun e => h (by rw [e])
        simpa using lexLt_total xs ys this
      · have : y < x := by omega
        simp [this]

def plain (s : Bytes) : Prop := ∀ b ∈ s, escaped b = false

theorem beq_bytes (a b : Bytes) : (a == b) = decide (a = b) := by
  by_cases h : a = b <;> simp [h]

/-- String ranges are exact when no string involved contains a byte that
`tag_encode_lower` escapes (bytes `<= 0x20`, `\`, `_`) and every stored
string is non-empty. Both side conditions are necessary: see
`bug_string_range_order` and `bug_empty_string_not_indexed`. The optimizer's
equal-bounds shortcut is exact only when both bounds are inclusive; see
`bug_exclusive_equal_bounds`. -/
theorem strRange_exact (fields : Attr → Bool) (p : Props) (k : Attr)
    (lo hi : Option Bytes) (imn imx : Bool) (hf : fields k = true)
    (hb : lo.isSome ∨ hi.isSome)
    (hlo : ∀ s, lo = some s → plain s) (hhi : ∀ s, hi = some s → plain s)
    (heq : ∀ s, lo = some s → hi = some s → imn = true ∧ imx = true)
    (hp : ∀ v, p k = some v → Pure v)
    (hps : ∀ t, p k = some (.str t) → plain t) :
    indexHit fields p (.range k (lo.map .str) (hi.map .str) imn imx) =
      holds p (.range k (lo.map .str) (hi.map .str) imn imx) := by
  have common : ∀ (q : QN), (∀ d : Doc, (∀ s, d k .main = s → ∀ x ∈ s, ∃ u, x = .tag u) →
      rsMatch d q = false ∨ True) → True := fun _ _ => trivial
  clear common
  rcases lo with _ | a <;> rcases hi with _ | b
  · simp at hb
  · -- only an upper bound
    have eb : tagEnc b = b := tagEnc_id b (hhi b rfl)
    simp only [indexHit, buildQ, Option.map, isStrVal, strBound, Bool.or_true, Bool.true_or,
      ite_true, buildStrRange, hf]
    cases hpk : p k with
    | none => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
    | some v =>
      cases hp v hpk with
      | int i => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | flt f => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | str t ht htb =>
        have et : tagEnc t = t := tagEnc_id t (hps t hpk)
        have ht' : (t = []) = False := by simp [ht]
        cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange, hf, et, eb, ht', inLex, beq_bytes]
  · -- only a lower bound
    have ea : tagEnc a = a := tagEnc_id a (hlo a rfl)
    simp only [indexHit, buildQ, Option.map, isStrVal, strBound, Bool.or_true, Bool.true_or,
      ite_true, buildStrRange, hf]
    cases hpk : p k with
    | none => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
    | some v =>
      cases hp v hpk with
      | int i => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | flt f => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | str t ht htb =>
        have et : tagEnc t = t := tagEnc_id t (hps t hpk)
        have ht' : (t = []) = False := by simp [ht]
        cases imn <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange, hf, et, ea, ht', inLex, beq_bytes]
  · have ea : tagEnc a = a := tagEnc_id a (hlo a rfl)
    have eb : tagEnc b = b := tagEnc_id b (hhi b rfl)
    simp only [indexHit, buildQ, Option.map, isStrVal, strBound, Bool.or_true, Bool.true_or,
      ite_true, buildStrRange, hf]
    by_cases hab : a = b
    · subst hab
      obtain ⟨rfl, rfl⟩ := heq a rfl rfl
      simp only [ite_true]
      cases hpk : p k with
      | none => simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | some v =>
        cases hp v hpk with
        | int i => simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
        | flt f => simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
        | str t ht htb =>
          have et : tagEnc t = t := tagEnc_id t (hps t hpk)
          have ht' : (t = []) = False := by simp [ht]
          simp only [rsMatch, docOf, hf, ite_true, hpk, setRange, List.flatMap_cons,
            List.flatMap_nil, rsStore_tag, et, ea, ht', ite_false, List.append_nil,
            List.any_cons, List.any_nil, Bool.or_false, holds, propOr, Option.getD_some,
            cyLe, cyLt, cyEqScalar, isTrue_some, beq_bytes]
          by_cases hta : t = a
          · subst hta; simp [lexLt_irrefl]
          · rcases lexLt_total a t (Ne.symm hta) with h | h
            · simp [lexLt_asymm a t h, h, hta, Ne.symm hta]
            · simp [lexLt_asymm t a h, h, hta, Ne.symm hta]
    · simp only [hab, ite_false]
      cases hpk : p k with
      | none => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | some v =>
        cases hp v hpk with
        | int i => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
        | flt f => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
        | str t ht htb =>
          have et : tagEnc t = t := tagEnc_id t (hps t hpk)
          have ht' : (t = []) = False := by simp [ht]
          cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange, hf, et, ea, eb, ht', inLex, beq_bytes]

end IndexLayer
