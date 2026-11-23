import FalkorIndexLayer.Proofs

/-! # String ranges: exact since #2975, `tag_encode_lower` preserves byte order -/

namespace IndexLayer

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

theorem encByte_cases (x : Nat) :
    (x ≤ 0x21 ∧ encByte x = [0x21, 0x22 + x]) ∨ (x = 0x5b ∧ encByte x = [0x5b, 0x5b]) ∨
    (x = 0x5c ∧ encByte x = [0x5b, 0x5d]) ∨
    (0x21 < x ∧ x ≠ 0x5b ∧ x ≠ 0x5c ∧ encByte x = [x]) := by
  unfold encByte
  by_cases h1 : x ≤ 0x21
  · simp [h1]
  · by_cases h2 : x = 0x5b
    · subst h2; simp
    · by_cases h3 : x = 0x5c
      · subst h3; simp
      · simp only [h1, h2, h3, ite_false]; exact Or.inr (Or.inr (Or.inr ⟨by omega, h2, h3, trivial⟩))

theorem lexLt_encByte_app (x y : Nat) (hxy : x ≠ y) (r s : Bytes) :
    lexLt (encByte x ++ r) (encByte y ++ s) = decide (x < y) := by
  rcases encByte_cases x with ⟨hx, ex⟩ | ⟨hx, ex⟩ | ⟨hx, ex⟩ | ⟨hx, hx2, hx3, ex⟩ <;>
  rcases encByte_cases y with ⟨hy, ey⟩ | ⟨hy, ey⟩ | ⟨hy, ey⟩ | ⟨hy, hy2, hy3, ey⟩ <;>
  rw [ex, ey] <;> simp only [List.cons_append, List.nil_append, lexLt] <;>
  (repeat' split) <;> simp_all <;> omega

theorem tagEncB_mono : ∀ a b : Bytes, lexLt (tagEncB a) (tagEncB b) = lexLt a b
  | [], [] => rfl
  | [], y :: ys => by
    simp only [tagEncB, lexLt]
    cases h : encByte y ++ tagEncB ys with
    | nil => simp [encByte_ne_nil] at h
    | cons _ _ => rfl
  | x :: xs, [] => by
    simp only [tagEncB, lexLt]
    cases h : encByte x ++ tagEncB xs with
    | nil => simp [encByte_ne_nil] at h
    | cons _ _ => rfl
  | x :: xs, y :: ys => by
    simp only [tagEncB]
    by_cases hxy : x = y
    · subst hxy
      have : ∀ (u r s : Bytes), lexLt (u ++ r) (u ++ s) = lexLt r s := by
        intro u r s; induction u with
        | nil => rfl
        | cons a u ih => simp [lexLt, ih]
      rw [this, tagEncB_mono xs ys]; simp [lexLt]
    · rw [lexLt_encByte_app x y hxy]
      simp only [lexLt]
      by_cases h : x < y
      · simp [h]
      · simp [h, hxy]


/-- #2975 (bug 7 fixed): `tag_encode_lower` preserves byte order: every escape is two bytes whose
first (`0x21`, `[`) sorts in the gap the escaped bytes leave, and the empty string's key `0x21`
is a prefix of, or below, every other key. -/
theorem tagEnc_mono (a b : Bytes) : lexLt (tagEnc a) (tagEnc b) = lexLt a b := by
  unfold tagEnc
  cases a with
  | nil =>
    cases b with
    | nil => rfl
    | cons y ys =>
      simp only [reduceCtorEq, ite_true, ite_false, tagEncB, lexLt]
      rcases encByte_cases y with ⟨_, e⟩ | ⟨_, e⟩ | ⟨_, e⟩ | ⟨h, _, _, e⟩ <;>
        rw [e] <;> simp [lexLt] <;> omega
  | cons x xs =>
    cases b with
    | nil =>
      simp only [reduceCtorEq, ite_true, ite_false, tagEncB, lexLt]
      rcases encByte_cases x with ⟨_, e⟩ | ⟨_, e⟩ | ⟨_, e⟩ | ⟨h, _, _, e⟩ <;>
        rw [e] <;> simp [lexLt] <;> omega
    | cons y ys =>
      simp only [reduceCtorEq, ite_false]
      exact tagEncB_mono _ _

theorem tagEnc_eq_iff (a b : Bytes) : tagEnc a = tagEnc b ↔ a = b := by
  constructor
  · intro e
    by_cases h : a = b
    · exact h
    · exfalso
      rcases lexLt_total a b h with l | l
      · have := tagEnc_mono a b; rw [e, lexLt_irrefl, l] at this; exact absurd this (by decide)
      · have := tagEnc_mono b a; rw [e, lexLt_irrefl, l] at this; exact absurd this (by decide)
  · intro h; rw [h]

theorem tagEnc_beq (a b : Bytes) : (tagEnc a == tagEnc b) = (a == b) := by
  by_cases h : a = b
  · subst h; simp
  · have : tagEnc a ≠ tagEnc b := fun e => h ((tagEnc_eq_iff a b).1 e)
    rw [beq_false_of_ne this, beq_false_of_ne h]

/-- A Rust string is UTF-8: it never has the bytes `0xFE`/`0xFF`. -/
def utf8 (s : Bytes) : Prop := ∀ b ∈ s, b < 0xFE

/-- Every string key sorts below `STRING_RANGE_END` (so an open upper end still finds every
string, and stops below the Bool keys). -/
theorem tagEnc_lt_end (t : Bytes) (ht : utf8 t) : lexLt (tagEnc t) STRING_RANGE_END = true := by
  unfold tagEnc STRING_RANGE_END
  cases t with
  | nil => simp [lexLt]
  | cons x xs =>
    have hx := ht x (by simp)
    simp only [reduceCtorEq, ite_false, tagEncB]
    rcases encByte_cases x with ⟨_, e⟩ | ⟨_, e⟩ | ⟨_, e⟩ | ⟨h, _, _, e⟩ <;>
      rw [e] <;> simp [lexLt] <;> omega

theorem beq_bytes (a b : Bytes) : (a == b) = decide (a = b) := by
  by_cases h : a = b <;> simp [h]

/-- String ranges are exact for a stored value that is `Pure` (Int, Float or a non-empty
string; #2975). `hp : Pure` excludes a stored empty string and a stored Bool, so this theorem
does not cover them: a stored empty string's key (`0x21`, see `fixed2975_empty_string`) and the
Bool keys are shown only by the equality lemmas and `fixed2975_string_range_skips_bool`, not
by a range-exactness theorem. Stored strings must be UTF-8 (`hps`); the bounds `lo`/`hi` may be
any bytes, the empty string included. Before #2975 the bounds and the stored strings had to
avoid the bytes the old encoding escaped (see `fixed2975_string_range_order`). The optimizer's
equal-bounds case needs no side condition since #3072 (1c9994e37): with an exclusive side the
node is empty. -/
theorem strRange_exact (fields : Attr → Bool) (p : Props) (k : Attr)
    (lo hi : Option Bytes) (imn imx : Bool) (hf : fields k = true)
    (hb : lo.isSome ∨ hi.isSome)
    (hp : ∀ v, p k = some v → Pure v)
    (hps : ∀ t, p k = some (.str t) → utf8 t) :
    indexHit fields p (.range k (lo.map .str) (hi.map .str) imn imx) =
      holds p (.range k (lo.map .str) (hi.map .str) imn imx) := by
  have stored : ∀ (f : Bytes → Bool) (g : Option Bool),
      (∀ t, p k = some (.str t) → f (tagEnc t) = isTrue g) → True := fun _ _ _ => trivial
  clear stored
  rcases lo with _ | a <;> rcases hi with _ | b
  · simp at hb
  · -- only an upper bound
    simp only [indexHit, buildQ, Option.map, isStrVal, strBound, Bool.or_true,
      ite_true, buildStrRange, hf]
    cases hpk : p k with
    | none => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf]
    | some v =>
      cases hp v hpk with
      | int i => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | flt f => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | str t ht htb =>
        cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, setRange, hf,
          tagEnc_ne_nil, inLex, tagEnc_mono, tagEnc_beq]
  · -- only a lower bound: the upper end is `STRING_RANGE_END`, inclusive
    simp only [indexHit, buildQ, Option.map, isStrVal, strBound, Bool.true_or,
      ite_true, buildStrRange, hf]
    cases hpk : p k with
    | none => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf]
    | some v =>
      cases hp v hpk with
      | int i => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | flt f => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
      | str t ht htb =>
        have he := tagEnc_lt_end t (hps t hpk)
        cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, setRange, hf,
          tagEnc_ne_nil, inLex, tagEnc_mono, tagEnc_beq, he]
  · simp only [indexHit, buildQ, Option.map, isStrVal, strBound, Bool.or_true,
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
      | none => simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf]
      | some v =>
        cases hp v hpk with
        | int i => simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
        | flt f => simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
        | str t ht htb =>
          simp only [rsMatch, docOf, hf, ite_true, hpk, setRange, List.flatMap_cons,
            List.flatMap_nil, rsStore_tag, tagEnc_ne_nil, ite_false, List.append_nil,
            List.any_cons, List.any_nil, Bool.or_false, holds, propOr, Option.getD_some,
            cyLe, cyLt, cyEqScalar, isTrue_some, tagEnc_beq, beq_bytes, tagEnc_eq_iff]
          by_cases hta : t = a
          · subst hta; simp [lexLt_irrefl]
          · rcases lexLt_total a t (Ne.symm hta) with h | h
            · simp [lexLt_asymm a t h, h, hta, Ne.symm hta]
            · simp [lexLt_asymm t a h, h, hta, Ne.symm hta]
    · simp only [hab, ite_false]
      cases hpk : p k with
      | none => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf]
      | some v =>
        cases hp v hpk with
        | int i => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
        | flt f => cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, setRange]
        | str t ht htb =>
          cases imn <;> cases imx <;> simp [rsMatch, docOf, hpk, holds, propOr, cyLt, cyLe, cyEqScalar, setRange, hf,
            tagEnc_ne_nil, inLex, tagEnc_mono, tagEnc_beq]

end IndexLayer
