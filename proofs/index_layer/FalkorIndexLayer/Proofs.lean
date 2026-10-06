import FalkorIndexLayer.Model

/-! # Theorems about the index layer model -/

namespace IndexLayer

/-! ## Hex digits -/

theorem nib_digit : ∀ n, n < 16 → hexNibble (hexDigit n) = n := by decide

theorem hex_roundtrip_byte (b : Nat) (hb : b < 256) :
    hexNibble (hexDigit (b / 16)) * 16 + hexNibble (hexDigit (b % 16)) = b := by
  rw [nib_digit _ (by omega), nib_digit _ (Nat.mod_lt _ (by decide))]; omega

theorem tagDec_cons_ne (b : Nat) (rest : Bytes) (h : b ≠ 0x5f) : tagDec (b :: rest) = b :: tagDec rest := by
  rw [tagDec.eq_def]; split <;> simp_all

theorem escaped_le {b : Nat} (h : escaped b = true) : b ≤ 0x5f := by
  simp [escaped] at h; omega

@[simp] theorem isTrue_some (b : Bool) : isTrue (some b) = b := by cases b <;> rfl
@[simp] theorem isTrue_none : isTrue none = false := rfl
@[simp] theorem rsStore_num (v : ENum) : rsStore (.num v) = [.num v] := rfl
@[simp] theorem rsStore_geo (a b : Int) : rsStore (.geo a b) = [.geo a b] := rfl
@[simp] theorem rsStore_tag (t : Bytes) : rsStore (.tag t) = if t = [] then [] else [.tag t] := by
  cases t <;> rfl

theorem hexDigit_gt_space : ∀ n, n < 16 → 0x20 < hexDigit n := by decide

theorem hexDigit_ne_us : ∀ n, n < 16 → hexDigit n ≠ 0x5f := by decide

/-! ## `tag_encode_lower` -/

/-- Decoding inverts `tag_encode_lower` on byte strings. -/
theorem tagDec_tagEnc : ∀ s : Bytes, (∀ b ∈ s, b < 256) → tagDec (tagEnc s) = s
  | [], _ => rfl
  | b :: bs, h => by
    have hb : b < 256 := h b (by simp)
    have ih := tagDec_tagEnc bs (fun x hx => h x (by simp [hx]))
    by_cases he : escaped b = true
    · simp only [tagEnc, he, ite_true, tagDec]
      rw [hex_roundtrip_byte b hb, ih]
    · have hne : b ≠ 0x5f := by
        intro h5; apply he; simp [escaped, h5]
      simp only [tagEnc, he, Bool.false_eq_true, ite_false]
      rw [tagDec_cons_ne b _ hne, ih]

/-- `tag_encode_lower` is injective: distinct strings never share an index
key, so a TAG exact-match query cannot return a different string. -/
theorem tagEnc_injective (s t : Bytes) (hs : ∀ b ∈ s, b < 256) (ht : ∀ b ∈ t, b < 256)
    (h : tagEnc s = tagEnc t) : s = t := by
  rw [← tagDec_tagEnc s hs, ← tagDec_tagEnc t ht, h]

/-- Every byte of the encoding is `> 0x20`: no `\x01` separator, no NUL
(so `CString::new` never fails), no whitespace for the tag tokenizer to trim. -/
theorem tagEnc_printable : ∀ (s : Bytes), ∀ c ∈ tagEnc s, 0x20 < c
  | [], c, h => by simp [tagEnc] at h
  | b :: bs, c, h => by
    unfold tagEnc at h
    split at h
    · rename_i he
      have := escaped_le he
      simp only [List.mem_cons] at h
      rcases h with h | h | h | h
      · omega
      · subst h; exact hexDigit_gt_space _ (by omega)
      · subst h; exact hexDigit_gt_space _ (Nat.mod_lt _ (by decide))
      · exact tagEnc_printable bs c h
    · rename_i hn
      simp only [List.mem_cons] at h
      rcases h with h | h
      · subst h; simp [escaped] at hn; omega
      · exact tagEnc_printable bs c h

theorem hexDigit_ne_bs : ∀ n, n < 16 → hexDigit n ≠ 0x5c := by decide

/-- No backslash in the encoding either (the query-side strip target). -/
theorem tagEnc_no_backslash : ∀ (s : Bytes), 0x5c ∉ tagEnc s
  | [] => by simp [tagEnc]
  | b :: bs => by
    unfold tagEnc
    split
    · rename_i he
      have := escaped_le he
      simp only [List.mem_cons, not_or]
      refine ⟨by decide, fun h => hexDigit_ne_bs _ (by omega) h.symm,
        fun h => hexDigit_ne_bs _ (Nat.mod_lt _ (by decide)) h.symm, tagEnc_no_backslash bs⟩
    · rename_i hn
      simp only [List.mem_cons, not_or]
      refine ⟨?_, tagEnc_no_backslash bs⟩
      intro h; subst h; simp [escaped] at hn

theorem tagEnc_nil_iff (s : Bytes) : tagEnc s = [] ↔ s = [] := by
  cases s with
  | nil => simp [tagEnc]
  | cons b bs => unfold tagEnc; split <;> simp

/-! ## Document keys -/

theorem hexDecode_hexEncode : ∀ bs : List Nat, (∀ b ∈ bs, b < 256) → hexDecode (hexEncode bs) = bs
  | [], _ => rfl
  | b :: bs, h => by
    simp only [hexEncode, hexDecode]
    rw [hex_roundtrip_byte b (h b (by simp)), hexDecode_hexEncode bs (fun x hx => h x (by simp [hx]))]

theorem leBytesN_lt : ∀ n id, ∀ b ∈ leBytesN n id, b < 256
  | 0, _, b, h => by simp [leBytesN] at h
  | n + 1, id, b, h => by
    simp only [leBytesN, List.mem_cons] at h
    rcases h with h | h
    · subst h; exact Nat.mod_lt _ (by decide)
    · exact leBytesN_lt n _ b h

theorem fromLe_leBytesN : ∀ n id, fromLe (leBytesN n id) = id % 256 ^ n
  | 0, id => by simp [leBytesN, fromLe, Nat.mod_one]
  | n + 1, id => by
    simp only [leBytesN, fromLe, fromLe_leBytesN n]
    rw [Nat.pow_succ, Nat.mul_comm (256^n) 256, Nat.mod_mul]

theorem hexEncode_length : ∀ bs : List Nat, (hexEncode bs).length = 2 * bs.length
  | [] => rfl
  | _ :: bs => by simp [hexEncode, hexEncode_length bs]; omega

theorem leBytesN_length : ∀ n id, (leBytesN n id).length = n
  | 0, _ => rfl
  | n + 1, id => by simp [leBytesN, leBytesN_length n]

/-- `decode_id ∘ hex_encode_into ∘ to_le_bytes = id` on `u64`. -/
theorem decodeId_nodeKey (id : Nat) (h : id < 2 ^ 64) : decodeId (nodeKey id) = id := by
  unfold decodeId nodeKey leBytes
  rw [hexDecode_hexEncode _ (leBytesN_lt 8 id), fromLe_leBytesN]
  exact Nat.mod_eq_of_lt (by simpa using h)

theorem nodeKey_length (id : Nat) : (nodeKey id).length = 16 := by
  simp [nodeKey, leBytes, hexEncode_length, leBytesN_length]

/-- `decode_triple` recovers `(src, dst, edge_id)` from the 48-char edge key. -/
theorem decode_edgeKey (s d e : Nat) (hs : s < 2 ^ 64) (hd : d < 2 ^ 64) (he : e < 2 ^ 64) :
    decodeId ((edgeKey s d e).take 16) = s ∧
    decodeId (((edgeKey s d e).drop 16).take 16) = d ∧
    decodeId ((edgeKey s d e).drop 32) = e := by
  unfold edgeKey
  have l := nodeKey_length
  refine ⟨?_, ?_, ?_⟩
  · rw [List.append_assoc, List.take_left' (l s)]; exact decodeId_nodeKey s hs
  · rw [List.append_assoc, List.drop_left' (l s), List.take_left' (l d)]; exact decodeId_nodeKey d hd
  · rw [show (32 : Nat) = 16 + 16 from rfl, ← List.drop_drop, List.append_assoc,
        List.drop_left' (l s), List.drop_left' (l d)]
    exact decodeId_nodeKey e he

/-- Distinct ids get distinct keys (no two entities share an RS document). -/
theorem nodeKey_injective (a b : Nat) (ha : a < 2 ^ 64) (hb : b < 2 ^ 64)
    (h : nodeKey a = nodeKey b) : a = b := by
  rw [← decodeId_nodeKey a ha, ← decodeId_nodeKey b hb, h]

/-! ## `int_loses_f64_precision` -/

theorem mask_bits : ∀ i, i < 63 → 52 ≤ i → (MASK).testBit i = true := by decide

/-- If `int_loses_f64_precision` says "exact", `|i| < 2^52` (exactly
representable, with a spare bit) or `|i| ≥ 2^63` (only `i64::MIN`, which is
`-2^63`, a power of two, also exact). -/
theorem intLosesPrecision_sound (i : Int) (hm : intLosesPrecision i = false) :
    i.natAbs < 2 ^ 52 ∨ 2 ^ 63 ≤ i.natAbs := by
  simp only [intLosesPrecision, bne_eq_false_iff_eq] at hm
  by_cases h63 : 2 ^ 63 ≤ i.natAbs
  · exact Or.inr h63
  · left
    apply Nat.lt_pow_two_of_testBit
    intro j hj
    by_cases hj2 : j < 63
    · have := congrArg (fun x => x.testBit j) hm
      simp only [Nat.testBit_and, Nat.zero_testBit, mask_bits j hj2 hj, Bool.and_true] at this
      exact this
    · apply Nat.testBit_lt_two_pow
      have : 2 ^ 63 ≤ 2 ^ j := Nat.pow_le_pow_right (by decide) (by omega)
      omega

/-! ## RediSearch semantics facts -/

theorem ENum.le_le_iff_eq (a b : ENum) : (ENum.le a b && ENum.le b a) = ENum.eq b a := by
  cases a <;> cases b <;> simp [ENum.le, ENum.lt, ENum.eq]
  rename_i x y
  rcases Int.lt_trichotomy x y with h | h | h
  · simp [h, Int.ne_of_lt h, Int.not_lt.mpr (Int.le_of_lt h)]
  · subst h; simp
  · simp [h, Int.ne_of_gt h, Int.not_lt.mpr (Int.le_of_lt h)]; omega

theorem inNum_point (d v : ENum) : inNum d d true true v = ENum.eq v d := by
  simp [inNum, ENum.le_le_iff_eq]

theorem rsMatch_union (d : Doc) (cs : List QN) : rsMatch d (.union cs) = cs.any (rsMatch d) := by
  simp [rsMatch]

theorem rsMatch_inter (d : Doc) (cs : List QN) : rsMatch d (.inter cs) = cs.all (rsMatch d) := by
  simp [rsMatch]

/-! ## Exactness on the well-behaved domain

"Pure" stored values are Int, Float and non-empty String. For them the
index answers exactly the Cypher predicate (equality, `IN` over scalar
literals, numeric ranges over finite values). The counterexamples in
`Bugs.lean` show each hypothesis is needed. -/

def byteStr (s : Bytes) : Prop := ∀ b ∈ s, b < 256

inductive Pure : Val → Prop where
  | int (i : Int) : Pure (.int i)
  | flt (f : ENum) : Pure (.flt f)
  | str (s : Bytes) : s ≠ [] → byteStr s → Pure (.str s)

inductive Const : Val → Prop where
  | int (i : Int) : Const (.int i)
  | flt (f : ENum) : Const (.flt f)
  | str (s : Bytes) : byteStr s → Const (.str s)

/-- Equality: index scan = full scan + filter, for a pure stored value and a
scalar literal. -/
theorem equal_exact (fields : Attr → Bool) (p : Props) (k : Attr) (c : Val)
    (hf : fields k = true) (hc : Const c) (hp : ∀ v, p k = some v → Pure v) :
    indexHit fields p (.equal k c) = holds p (.equal k c) := by
  cases hpk : p k with
  | none =>
    cases hc <;> simp [indexHit, buildQ, buildEq, valueToNumeric, hf, rsMatch, docOf, hpk,
      holds, propOr, cyEqScalar]
  | some v =>
    have hv := hp v hpk
    cases hc with
    | int i =>
      cases hv <;> simp [indexHit, buildQ, buildEq, valueToNumeric, hf, rsMatch, docOf, hpk,
        setRange, inNum_point, holds, propOr, cyEqScalar, numOf]
    | flt f =>
      cases hv <;> simp [indexHit, buildQ, buildEq, valueToNumeric, hf, rsMatch, docOf, hpk,
        setRange, inNum_point, holds, propOr, cyEqScalar, numOf]
    | str s hs =>
      cases hv with
      | int _ | flt _ => simp [indexHit, buildQ, buildEq, valueToNumeric, hf, rsMatch, docOf, hpk,
          setRange, holds, propOr, cyEqScalar, numOf]
      | str t ht htb =>
        have hne : tagEnc t ≠ [] := by rw [Ne, tagEnc_nil_iff]; exact ht
        have hstore : rsStore (.tag (tagEnc t)) = [.tag (tagEnc t)] := by
          cases h : tagEnc t with
          | nil => exact absurd h hne
          | cons _ _ => rfl
        simp only [indexHit, buildQ, buildEq, valueToNumeric, hf, ite_true, rsMatch, docOf, hpk,
          setRange, List.flatMap_cons, List.flatMap_nil, hstore, List.append_nil, List.any_cons,
          List.any_nil, Bool.or_false, holds, propOr, Option.getD_some, cyEqScalar, isTrue_some]
        by_cases he : t = s
        · subst he; simp only [beq_self_eq_true]
        · have : tagEnc t ≠ tagEnc s := fun h => he (tagEnc_injective t s htb hs h)
          rw [beq_false_of_ne this, beq_false_of_ne he]

/-- `IN` over a list of scalar literals: exact. -/
theorem inList_exact (fields : Attr → Bool) (p : Props) (k : Attr) (xs : List Val)
    (hf : fields k = true) (hne : xs ≠ []) (hc : ∀ x ∈ xs, Const x)
    (hp : ∀ v, p k = some v → Pure v) :
    indexHit fields p (.inList k (.list xs)) = holds p (.inList k (.list xs)) := by
  have hsome : ∀ x ∈ xs, ∃ n, buildEq fields k x = some n := by
    intro x hx
    cases hc x hx <;> simp [buildEq, valueToNumeric, hf]
  have hxs : xs.isEmpty = false := by cases xs <;> simp_all
  simp only [indexHit, buildQ, hxs, Bool.false_eq_true, ite_false, rsMatch_union, holds]
  have key : ∀ x ∈ xs, ∀ n, buildEq fields k x = some n →
      rsMatch (docOf fields p) n = isTrue (cyEqScalar (propOr p k) x) := by
    intro x hx n hn
    have := equal_exact fields p k x hf (hc x hx) hp
    simp only [indexHit, buildQ, hn, holds] at this
    exact this
  clear hne hxs
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨n, hn⟩ := hsome x (by simp)
    have kx := key x (by simp) n hn
    simp only [List.filterMap_cons, hn, List.any_cons]
    rw [kx, ih (fun y hy => hc y (by simp [hy])) (fun y hy => hsome y (by simp [hy]))
      (fun y hy => key y (by simp [hy]))]

/-- Finite numeric stored value. -/
inductive FinNum : Val → Prop where
  | int (i : Int) : FinNum (.int i)
  | flt (i : Int) : FinNum (.flt (.fin i))
  | nan : FinNum (.flt .nan)

inductive NumConst : Val → Prop where
  | int (i : Int) : NumConst (.int i)
  | flt (f : ENum) : NumConst (.flt f)

theorem enum_le_eq (a b : ENum) : (ENum.lt a b || ENum.eq a b) = ENum.le a b := rfl

set_option maxHeartbeats 4000000 in
/-- Numeric ranges (any combination of open/closed/absent bounds, with at
least one bound, as the optimizer builds them) are exact for finite (or NaN)
stored numbers. `±inf` stored values are excluded: see
`bug_open_bound_excludes_inf`. -/
theorem numRange_exact (fields : Attr → Bool) (p : Props) (k : Attr)
    (mn mx : Option Val) (imn imx : Bool) (hf : fields k = true)
    (hb : mn.isSome ∨ mx.isSome)
    (hmn : ∀ v, mn = some v → NumConst v) (hmx : ∀ v, mx = some v → NumConst v)
    (hp : ∀ v, p k = some v → FinNum v) :
    indexHit fields p (.range k mn mx imn imx) = holds p (.range k mn mx imn imx) := by
  have key : ∀ (o : Option Val), (∀ v, o = some v → NumConst v) →
      o = none ∨ (∃ i, o = some (.int i)) ∨ (∃ f, o = some (.flt f)) := by
    intro o ho
    rcases o with _ | v
    · exact Or.inl rfl
    · cases ho v rfl with
      | int i => exact Or.inr (Or.inl ⟨i, rfl⟩)
      | flt f => exact Or.inr (Or.inr ⟨f, rfl⟩)
  have hv : p k = none ∨ (∃ i, p k = some (.int i)) ∨ (∃ i, p k = some (.flt (.fin i))) ∨
      p k = some (.flt .nan) := by
    rcases h : p k with _ | v
    · exact Or.inl rfl
    · cases hp v h with
      | int i => exact Or.inr (Or.inl ⟨i, rfl⟩)
      | flt i => exact Or.inr (Or.inr (Or.inl ⟨i, rfl⟩))
      | nan => exact Or.inr (Or.inr (Or.inr rfl))
  rcases key mn hmn with rfl | ⟨a, rfl⟩ | ⟨a, rfl⟩ <;>
  rcases key mx hmx with rfl | ⟨b, rfl⟩ | ⟨b, rfl⟩ <;>
  rcases hv with h | ⟨c, h⟩ | ⟨c, h⟩ | h <;>
  (try simp at hb) <;>
  (try (rcases a with _ | a | _ | _)) <;> (try (rcases b with _ | b | _ | _)) <;>
  cases imn <;> cases imx <;>
  simp [indexHit, buildQ, isStrVal, buildNumRange, valueToNumeric, hf, rsMatch, docOf, h,
    setRange, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, inNum, ENum.le, ENum.lt, ENum.eq] <;>
  omega

end IndexLayer
