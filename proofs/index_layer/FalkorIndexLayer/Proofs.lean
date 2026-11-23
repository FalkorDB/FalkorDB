import FalkorIndexLayer.Model

/-! # Theorems about the index layer model -/

namespace IndexLayer

/-! ## Hex digits -/

theorem nib_digit : ∀ n, n < 16 → hexNibble (hexDigit n) = n := by decide

theorem hex_roundtrip_byte (b : Nat) (hb : b < 256) :
    hexNibble (hexDigit (b / 16)) * 16 + hexNibble (hexDigit (b % 16)) = b := by
  rw [nib_digit _ (by omega), nib_digit _ (Nat.mod_lt _ (by decide))]; omega

@[simp] theorem isTrue_some (b : Bool) : isTrue (some b) = b := by cases b <;> rfl
@[simp] theorem isTrue_none : isTrue none = false := rfl
@[simp] theorem rsStore_num (v : ENum) : rsStore (.num v) = [.num v] := rfl
@[simp] theorem rsStore_geo (a b : Int) : rsStore (.geo a b) = [.geo a b] := rfl
@[simp] theorem rsStore_tag (t : Bytes) : rsStore (.tag t) = if t = [] then [] else [.tag t] := by
  cases t <;> rfl

theorem hexDigit_gt_space : ∀ n, n < 16 → 0x20 < hexDigit n := by decide

/-! ## `tag_encode_lower` -/

theorem tagDecB_encByte (b : Nat) (r : Bytes) : tagDecB (encByte b ++ r) = b :: tagDecB r := by
  unfold encByte
  by_cases h1 : b ≤ 0x21
  · simp only [h1, ite_true, List.cons_append, List.nil_append, tagDecB]; congr 1; omega
  · by_cases h2 : b = 0x5b
    · subst h2; simp [tagDecB]
    · by_cases h3 : b = 0x5c
      · subst h3; simp [tagDecB]
      · simp only [h1, h2, h3, ite_false, List.cons_append, List.nil_append]
        rw [tagDecB.eq_def]; split <;> simp_all <;> omega

theorem tagDecB_tagEncB : ∀ s : Bytes, tagDecB (tagEncB s) = s
  | [] => by simp [tagEncB, tagDecB]
  | b :: bs => by simp only [tagEncB, tagDecB_encByte, tagDecB_tagEncB bs]

theorem encByte_ne_nil (b : Nat) : encByte b ≠ [] := by
  unfold encByte; split <;> (try split) <;> (try split) <;> simp

/-- A non-empty string never encodes to the empty string's key `0x21`: an escape is two bytes,
and a passed-through byte is above `0x21`. -/
theorem tagEncB_ne_empty_key (b : Nat) (bs : Bytes) : tagEncB (b :: bs) ≠ [0x21] := by
  simp only [tagEncB, encByte]
  by_cases h1 : b ≤ 0x21
  · simp [h1]
  · by_cases h2 : b = 0x5b
    · simp [h2]
    · by_cases h3 : b = 0x5c
      · simp [h3]
      · simp only [h1, h2, h3, ite_false, List.cons_append, List.nil_append, ne_eq,
          List.cons.injEq, not_and]
        intro h; omega

/-- Decoding inverts `tag_encode_lower` on byte strings. -/
theorem tagDec_tagEnc (s : Bytes) (_h : ∀ b ∈ s, b < 256) : tagDec (tagEnc s) = s := by
  cases s with
  | nil => rfl
  | cons b bs =>
    simp only [tagEnc, tagDec, reduceCtorEq, ite_false, tagEncB_ne_empty_key, tagDecB_tagEncB]

/-- `tag_encode_lower` is injective: distinct strings never share an index
key, so a TAG exact-match query cannot return a different string. -/
theorem tagEnc_injective (s t : Bytes) (hs : ∀ b ∈ s, b < 256) (ht : ∀ b ∈ t, b < 256)
    (h : tagEnc s = tagEnc t) : s = t := by
  rw [← tagDec_tagEnc s hs, ← tagDec_tagEnc t ht, h]

theorem encByte_printable (b : Nat) : ∀ c ∈ encByte b, 0x20 < c := by
  unfold encByte
  by_cases h1 : b ≤ 0x21
  · simp only [h1, ite_true, List.mem_cons, List.not_mem_nil, or_false]; omega
  · by_cases h2 : b = 0x5b
    · simp [h2]
    · by_cases h3 : b = 0x5c
      · simp [h3]
      · simp only [h1, h2, h3, ite_false, List.mem_cons, List.not_mem_nil, or_false]; omega

theorem tagEncB_printable : ∀ (s : Bytes), ∀ c ∈ tagEncB s, 0x20 < c
  | [], c, h => by simp [tagEncB] at h
  | b :: bs, c, h => by
    simp only [tagEncB, List.mem_append] at h
    rcases h with h | h
    · exact encByte_printable b c h
    · exact tagEncB_printable bs c h

/-- Every byte of the encoding is `> 0x20`: no `\x01` separator, no NUL
(so `CString::new` never fails), no whitespace for the tag tokenizer to trim. -/
theorem tagEnc_printable (s : Bytes) : ∀ c ∈ tagEnc s, 0x20 < c := by
  unfold tagEnc
  split
  · simp
  · exact tagEncB_printable s

theorem encByte_no_backslash (b : Nat) : 0x5c ∉ encByte b := by
  unfold encByte
  by_cases h1 : b ≤ 0x21
  · simp only [h1, ite_true, List.mem_cons, List.not_mem_nil, or_false]; omega
  · by_cases h2 : b = 0x5b
    · simp [h2]
    · by_cases h3 : b = 0x5c
      · simp [h3]
      · simp only [h1, h2, h3, ite_false, List.mem_cons, List.not_mem_nil, or_false]; omega

theorem tagEncB_no_backslash : ∀ (s : Bytes), 0x5c ∉ tagEncB s
  | [] => by simp [tagEncB]
  | b :: bs => by
    simp only [tagEncB, List.mem_append, not_or]
    exact ⟨encByte_no_backslash b, tagEncB_no_backslash bs⟩

/-- No backslash in the encoding either (the query-side strip target). -/
theorem tagEnc_no_backslash (s : Bytes) : 0x5c ∉ tagEnc s := by
  unfold tagEnc
  split
  · simp
  · exact tagEncB_no_backslash s

/-- #2975 (bug 10 fixed): no string, the empty one included, has an empty key, so every string
is indexed (RediSearch skips only an empty TAG value). -/
theorem tagEnc_ne_nil (s : Bytes) : tagEnc s ≠ [] := by
  unfold tagEnc
  split
  · simp
  · cases s with
    | nil => contradiction
    | cons b bs => simp [tagEncB, encByte_ne_nil]

/-- A Bool key starts with `0xFF`, which no string key does when the string has no `0xFF` byte
(a UTF-8 string never does): a Bool and a string never share a key. -/
theorem tagEnc_ne_boolKey (s : Bytes) (hs : ∀ c ∈ s, c < 0xFF) (b : Bool) :
    tagEnc s ≠ boolKey b := by
  unfold tagEnc boolKey
  split
  · simp
  · cases s with
    | nil => contradiction
    | cons c cs =>
      have hc := hs c (by simp)
      simp only [tagEncB, encByte]
      by_cases h1 : c ≤ 0x21
      · simp [h1]
      · by_cases h2 : c = 0x5b
        · simp [h2]
        · by_cases h3 : c = 0x5c
          · simp [h3]
          · simp only [h1, h2, h3, ite_false, List.cons_append, List.nil_append, ne_eq,
              List.cons.injEq, not_and]
            intro h; omega

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
    cases hc <;> simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf, hpk,
      holds, propOr, cyEqScalar]
  | some v =>
    have hv := hp v hpk
    cases hc with
    | int i =>
      cases hv <;> simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf, hpk,
        setRange, inNum_point, holds, propOr, cyEqScalar, numOf]
    | flt f =>
      cases hv <;> simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf, hpk,
        setRange, inNum_point, holds, propOr, cyEqScalar, numOf]
    | str s hs =>
      cases hv with
      | int _ | flt _ => simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf, hpk,
          setRange, holds, propOr, cyEqScalar, numOf]
      | str t ht htb =>
        have hne : tagEnc t ≠ [] := tagEnc_ne_nil t
        have hstore : rsStore (.tag (tagEnc t)) = [.tag (tagEnc t)] := by
          cases h : tagEnc t with
          | nil => exact absurd h hne
          | cons _ _ => rfl
        simp only [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, ite_true, rsMatch, docOf, hpk,
          setRange, List.flatMap_cons, List.flatMap_nil, hstore, List.append_nil, List.any_cons,
          List.any_nil, Bool.or_false, holds, propOr, Option.getD_some, cyEqScalar, isTrue_some]
        by_cases he : t = s
        · subst he; simp only [beq_self_eq_true]
        · have : tagEnc t ≠ tagEnc s := fun h => he (tagEnc_injective t s htb hs h)
          rw [beq_false_of_ne this, beq_false_of_ne he]

/-- #2975 (bug 8 fixed): a Bool literal is exact against a stored Bool, number or string (a
string key never starts with `0xFF`, as no UTF-8 string has that byte): `n.v = true` no longer
returns `v = 1`. -/
theorem boolEqual_exact (fields : Attr → Bool) (p : Props) (k : Attr) (b : Bool)
    (hf : fields k = true) (hp : ∀ v, p k = some v → Pure v ∨ ∃ c, v = .bool c)
    (hps : ∀ t, p k = some (.str t) → ∀ c ∈ t, c < 0xFF) :
    indexHit fields p (.equal k (.bool b)) = holds p (.equal k (.bool b)) := by
  cases hpk : p k with
  | none => simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf, hpk,
      holds, propOr, cyEqScalar]
  | some v =>
    rcases hp v hpk with hv | ⟨c, rfl⟩
    · cases hv with
      | int _ | flt _ => simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch,
          docOf, hpk, setRange, holds, propOr, cyEqScalar, numOf]
      | str t _ _ =>
        have hne := beq_false_of_ne (tagEnc_ne_boolKey t (hps t hpk) b)
        simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf, hpk,
          setRange, holds, propOr, cyEqScalar, tagEnc_ne_nil, hne, numOf]
    · cases b <;> cases c <;> simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch,
        docOf, hpk, setRange, holds, propOr, cyEqScalar, boolKey]

/-- #2975 (bug 8 fixed): a stored Bool is never returned by a number or string literal (`n.v = 1`
no longer returns `v = true`). -/
theorem equal_exact_storedBool (fields : Attr → Bool) (p : Props) (k : Attr) (c : Val) (b : Bool)
    (hf : fields k = true) (hc : Const c) (hcs : ∀ t, c = .str t → ∀ x ∈ t, x < 0xFF)
    (hpk : p k = some (.bool b)) :
    indexHit fields p (.equal k c) = holds p (.equal k c) := by
  cases hc with
  | int _ | flt _ => simp [indexHit, buildQ, buildEq, valueToNumeric, hf, rsMatch, docOf,
      hpk, setRange, holds, propOr, cyEqScalar, numOf]
  | str s _ =>
    have hne := Ne.symm (tagEnc_ne_boolKey s (hcs s rfl) b)
    simp only [boolKey] at hne
    simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf, hpk, setRange,
      holds, propOr, cyEqScalar, hne, boolKey, numOf]

/-- `IN` over a list of scalar literals: exact. -/
theorem inList_exact (fields : Attr → Bool) (p : Props) (k : Attr) (xs : List Val)
    (hf : fields k = true) (hne : xs ≠ []) (hc : ∀ x ∈ xs, Const x)
    (hp : ∀ v, p k = some v → Pure v) :
    indexHit fields p (.inList k (.list xs)) = holds p (.inList k (.list xs)) := by
  have hsome : ∀ x ∈ xs, ∃ n, buildEq fields k x = some n := by
    intro x hx
    cases hc x hx <;> simp [buildEq, valueToNumeric, tagKey, hf]
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

/-- Numeric ranges (any combination of open/closed/absent bounds, with at
least one bound, as the optimizer builds them) are exact for every stored
number: finite, NaN, and — since #3081 (`ff3d24ba7`) made an absent bound
inclusive — `±inf` too (before it they were missed). -/
theorem numRange_exact (fields : Attr → Bool) (p : Props) (k : Attr)
    (mn mx : Option Val) (imn imx : Bool) (hf : fields k = true)
    (hb : mn.isSome ∨ mx.isSome)
    (hmn : ∀ v, mn = some v → NumConst v) (hmx : ∀ v, mx = some v → NumConst v)
    (hp : ∀ v, p k = some v → NumConst v) :
    indexHit fields p (.range k mn mx imn imx) = holds p (.range k mn mx imn imx) := by
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
  rcases key (p k) hp with h | ⟨c, h⟩ | ⟨c, h⟩ <;>
  (try simp at hb) <;>
  cases imn <;> cases imx <;>
  simp [indexHit, buildQ, isStrVal, buildNumRange, valueToNumeric, hf, rsMatch, docOf, h,
    setRange, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, inNum] <;>
  (try (rcases a with _ | a | _ | _)) <;> (try (rcases b with _ | b | _ | _)) <;>
  (try (rcases c with _ | c | _ | _)) <;>
  simp [ENum.le, ENum.lt, ENum.eq] <;>
  omega

end IndexLayer
