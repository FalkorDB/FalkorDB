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

/-! ## Byte-wise lexicographic order -/

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

theorem lexLt_append_left : ∀ (p x y : Bytes), lexLt (p ++ x) (p ++ y) = lexLt x y
  | [], _, _ => rfl
  | a :: p, x, y => by simp [lexLt, lexLt_append_left p x y]

/-! ## `tag_encode_lower` (#2975: unit encoding) -/

theorem tagUnit_ne_nil (b : Nat) : tagUnit b ≠ [] := by
  unfold tagUnit; split <;> (try split) <;> (try split) <;> simp

/-- Distinct units decide the order at a position inside both units (they are
prefix-free and ordered as the bytes), whatever follows them. -/
theorem tagUnit_lt (a b : Nat) (h : a < b) (x y : Bytes) :
    lexLt (tagUnit a ++ x) (tagUnit b ++ y) = true := by
  unfold tagUnit
  by_cases a1 : a ≤ 0x21 <;> by_cases a2 : a = 0x5b <;> by_cases a3 : a = 0x5c <;>
  by_cases b1 : b ≤ 0x21 <;> by_cases b2 : b = 0x5b <;> by_cases b3 : b = 0x5c <;>
  simp [a1, a2, a3, b1, b2, b3, lexLt] <;> omega

theorem tagEncRaw_ne_nil (b : Nat) (bs : Bytes) : tagEncRaw (b :: bs) ≠ [] := by
  simp [tagEncRaw, tagUnit_ne_nil]

/-- The raw loop is order-preserving (on all byte values). -/
theorem tagEncRaw_lexLt : ∀ s t : Bytes, lexLt (tagEncRaw s) (tagEncRaw t) = lexLt s t
  | [], [] => rfl
  | [], b :: bs => by
    have := tagEncRaw_ne_nil b bs
    cases h : tagEncRaw (b :: bs) with
    | nil => exact absurd h this
    | cons _ _ => simp [tagEncRaw, lexLt]
  | b :: bs, [] => by
    have := tagEncRaw_ne_nil b bs
    cases h : tagEncRaw (b :: bs) with
    | nil => exact absurd h this
    | cons _ _ => simp [tagEncRaw, lexLt]
  | a :: as, b :: bs => by
    simp only [tagEncRaw]
    by_cases hab : a < b
    · rw [tagUnit_lt a b hab]; simp [lexLt, hab]
    · by_cases he : a = b
      · subst he; rw [lexLt_append_left, tagEncRaw_lexLt as bs]; simp [lexLt]
      · have hba : b < a := by omega
        have h1 := tagUnit_lt b a hba (tagEncRaw bs) (tagEncRaw as)
        rw [lexLt_asymm _ _ h1]; simp [lexLt, hab, he]

/-- `[0x21]` (the key of `''`) sorts before every other key. -/
theorem lexLt_empty_key (b : Nat) (bs : Bytes) : lexLt [0x21] (tagEncRaw (b :: bs)) = true := by
  simp only [tagEncRaw]
  unfold tagUnit
  by_cases b1 : b ≤ 0x21 <;> by_cases b2 : b = 0x5b <;> by_cases b3 : b = 0x5c <;>
  simp [b1, b2, b3, lexLt] <;> omega

/-- **#2975: `tag_encode_lower` is order-preserving**: comparing keys is
comparing the strings, so a TAG lex range is a string range. -/
theorem tagEnc_lexLt (s t : Bytes) : lexLt (tagEnc s) (tagEnc t) = lexLt s t := by
  cases s with
  | nil => cases t with
    | nil => rfl
    | cons b bs => simp only [tagEnc]; rw [lexLt_empty_key]; rfl
  | cons a as => cases t with
    | nil =>
      simp only [tagEnc]
      rw [lexLt_asymm _ _ (lexLt_empty_key a as)]; rfl
    | cons b bs => exact tagEncRaw_lexLt _ _

theorem tagEnc_inj (s t : Bytes) : tagEnc s = tagEnc t ↔ s = t := by
  refine ⟨fun h => ?_, fun h => h ▸ rfl⟩
  apply Classical.byContradiction
  intro hne
  rcases lexLt_total s t hne with h1 | h1
  · rw [← tagEnc_lexLt, h, lexLt_irrefl] at h1; exact Bool.false_ne_true h1
  · rw [← tagEnc_lexLt, h, lexLt_irrefl] at h1; exact Bool.false_ne_true h1

/-- `tag_encode_lower` is injective: distinct strings never share an index
key, so a TAG exact-match query cannot return a different string. (Since
#2975 this holds for every byte value, `tagEnc_inj`; the byte bounds of the
old statement are unused.) -/
theorem tagEnc_injective (s t : Bytes) (_hs : ∀ b ∈ s, b < 256) (_ht : ∀ b ∈ t, b < 256)
    (h : tagEnc s = tagEnc t) : s = t := (tagEnc_inj s t).1 h

@[simp] theorem tagEnc_beq (s t : Bytes) : (tagEnc s == tagEnc t) = (s == t) := by
  by_cases h : s = t
  · subst h; simp only [beq_self_eq_true]
  · have : tagEnc s ≠ tagEnc t := fun e => h ((tagEnc_inj s t).1 e)
    rw [beq_false_of_ne this, beq_false_of_ne h]

theorem tagUnit_bytes (b c : Nat) (h : c ∈ tagUnit b) :
    0x20 < c ∧ c ≠ 0x5c ∧ (c = b ∨ c ≤ 0x5d) := by
  unfold tagUnit at h
  by_cases b1 : b ≤ 0x21 <;> by_cases b2 : b = 0x5b <;> by_cases b3 : b = 0x5c <;>
  simp [b1, b2, b3] at h <;> omega

theorem tagEncRaw_bytes : ∀ (s : Bytes) (c : Nat), c ∈ tagEncRaw s →
    0x20 < c ∧ c ≠ 0x5c ∧ (c ∈ s ∨ c ≤ 0x5d)
  | [], c, h => by simp [tagEncRaw] at h
  | b :: bs, c, h => by
    simp only [tagEncRaw, List.mem_append] at h
    rcases h with h | h
    · have := tagUnit_bytes b c h
      refine ⟨this.1, this.2.1, ?_⟩
      rcases this.2.2 with e | e
      · exact Or.inl (by simp [e])
      · exact Or.inr e
    · have := tagEncRaw_bytes bs c h
      refine ⟨this.1, this.2.1, ?_⟩
      rcases this.2.2 with e | e
      · exact Or.inl (by simp [e])
      · exact Or.inr e

theorem tagEnc_bytes (s : Bytes) (c : Nat) (h : c ∈ tagEnc s) :
    0x20 < c ∧ c ≠ 0x5c ∧ (c ∈ s ∨ c ≤ 0x5d) := by
  cases s with
  | nil => simp [tagEnc] at h; subst h; decide
  | cons b bs => exact tagEncRaw_bytes _ c h

/-- Every byte of the encoding is `> 0x20`: no `\x01` separator, no NUL
(so `CString::new` never fails), no whitespace for the tag tokenizer to trim. -/
theorem tagEnc_printable (s : Bytes) : ∀ c ∈ tagEnc s, 0x20 < c :=
  fun c h => (tagEnc_bytes s c h).1

/-- No backslash in the encoding either (the query-side strip target). -/
theorem tagEnc_no_backslash (s : Bytes) : 0x5c ∉ tagEnc s :=
  fun h => (tagEnc_bytes s _ h).2.1 rfl

/-- **#2975: every key is non-empty** (`''` is `[0x21]`), so RediSearch indexes it.
This replaces `tagEnc_nil_iff` (`tagEnc s = [] ↔ s = []`), whose `←` half was
the empty-string bug; its `→` half holds vacuously. -/
theorem tagEnc_ne_nil (s : Bytes) : tagEnc s ≠ [] := by
  cases s with
  | nil => simp [tagEnc]
  | cons b bs => exact tagEncRaw_ne_nil b bs

@[simp] theorem tagEnc_eq_nil (s : Bytes) : (tagEnc s = []) = False :=
  propext ⟨tagEnc_ne_nil s, False.elim⟩

@[simp] theorem boolKey_eq_nil (b : Bool) : (boolKey b = []) = False :=
  propext ⟨fun h => by cases b <;> simp [boolKey] at h, False.elim⟩

theorem tagEnc_nil_imp (s : Bytes) (h : tagEnc s = []) : s = [] := absurd h (tagEnc_ne_nil s)

/-- A Rust `String` is UTF-8, which never contains the bytes `0xFE`/`0xFF`. -/
def Utf8 (s : Bytes) : Prop := ∀ b ∈ s, b < 0xFE

/-- The key of a UTF-8 string has no `0xFE`/`0xFF` byte: it is never a Bool key
and sorts below `STRING_RANGE_END`. -/
theorem tagEnc_lt_FE (s : Bytes) (hs : Utf8 s) : ∀ c ∈ tagEnc s, c < 0xFE := by
  intro c hc
  rcases (tagEnc_bytes s c hc).2.2 with h | h
  · exact hs c h
  · omega

theorem tagEnc_below_end (s : Bytes) (hs : Utf8 s) : lexLt (tagEnc s) STRING_RANGE_END = true := by
  have hlt := tagEnc_lt_FE s hs
  have hne := tagEnc_ne_nil s
  cases h : tagEnc s with
  | nil => exact absurd h hne
  | cons c cs =>
    have : c < 0xFE := hlt c (by simp [h])
    simp [STRING_RANGE_END, lexLt, this]

theorem tagEnc_ne_boolKey (s : Bytes) (hs : Utf8 s) (b : Bool) : tagEnc s ≠ boolKey b := by
  intro h
  have := tagEnc_lt_FE s hs 0xFF (by rw [h]; simp [boolKey])
  omega

@[simp] theorem boolKey_beq (a b : Bool) : (boolKey a == boolKey b) = (a == b) := by
  cases a <;> cases b <;> decide

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
literals, numeric ranges over finite values). Since #2975 the domain is
wider (`Stored`: also `''` and Bool; `Lit`: also Bool literals), see
`equal_exact_gen`, `inList_exact_gen`. -/

def byteStr (s : Bytes) : Prop := ∀ b ∈ s, b < 256

inductive Pure : Val → Prop where
  | int (i : Int) : Pure (.int i)
  | flt (f : ENum) : Pure (.flt f)
  | str (s : Bytes) : s ≠ [] → byteStr s → Pure (.str s)

inductive Const : Val → Prop where
  | int (i : Int) : Const (.int i)
  | flt (f : ENum) : Const (.flt f)
  | str (s : Bytes) : byteStr s → Const (.str s)

/-- Stored values the index answers exactly since #2975: every number, every
string (also `''`) and Bool. -/
inductive Stored : Val → Prop where
  | int (i : Int) : Stored (.int i)
  | flt (f : ENum) : Stored (.flt f)
  | str (s : Bytes) : byteStr s → Stored (.str s)
  | bool (b : Bool) : Stored (.bool b)

/-- Literals since #2975: the scalar `Const`s and Bool. -/
inductive Lit : Val → Prop where
  | int (i : Int) : Lit (.int i)
  | flt (f : ENum) : Lit (.flt f)
  | str (s : Bytes) : byteStr s → Lit (.str s)
  | bool (b : Bool) : Lit (.bool b)

theorem Pure.stored {v : Val} (h : Pure v) : Stored v := by
  cases h with
  | int i => exact .int i
  | flt f => exact .flt f
  | str s _ hs => exact .str s hs

theorem Const.lit {v : Val} (h : Const v) : Lit v := by
  cases h with
  | int i => exact .int i
  | flt f => exact .flt f
  | str s hs => exact .str s hs

theorem boolKey_ne_tagEnc (s : Bytes) (hs : Utf8 s) (b : Bool) :
    (boolKey b == tagEnc s) = false := by
  have := tagEnc_ne_boolKey s hs b
  exact beq_false_of_ne (Ne.symm this)

theorem tagEnc_ne_boolKey' (s : Bytes) (hs : Utf8 s) (b : Bool) :
    (tagEnc s == boolKey b) = false := beq_false_of_ne (tagEnc_ne_boolKey s hs b)

theorem tagEnc_store (t : Bytes) : rsStore (.tag (tagEnc t)) = [.tag (tagEnc t)] := by
  cases h : tagEnc t with
  | nil => exact absurd h (tagEnc_ne_nil t)
  | cons _ _ => rfl

/-- **Equality is exact (#2975)** for every number, string (also `''`) and Bool,
stored or looked up: Bool is its own TAG key, never `= 1`. A string is a Rust
`String`, so it never contains `0xFE`/`0xFF` (`Utf8`); this is needed only when a
Bool meets a string (the Bool keys start with `0xFF`). -/
theorem equal_exact_gen (fields : Attr → Bool) (p : Props) (k : Attr) (c : Val)
    (hf : fields k = true) (hc : Lit c) (hp : ∀ v, p k = some v → Stored v)
    (hsep : ∀ b t, (c = .bool b ∧ p k = some (.str t)) ∨ (p k = some (.bool b) ∧ c = .str t) →
      Utf8 t) :
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
        setRange, inNum_point, holds, propOr, cyEqScalar, numOf, tagEnc_store]
    | flt f =>
      cases hv <;> simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf, hpk,
        setRange, inNum_point, holds, propOr, cyEqScalar, numOf, tagEnc_store]
    | str s hs =>
      cases hv with
      | int _ | flt _ => simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf,
          hpk, setRange, holds, propOr, cyEqScalar, numOf]
      | str t ht =>
        simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf, hpk,
          setRange, holds, propOr, cyEqScalar, tagEnc_store]
      | bool b =>
        have hu := hsep b s (Or.inr ⟨hpk, rfl⟩)
        simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf, hpk,
          setRange, holds, propOr, cyEqScalar, numOf, boolKey_ne_tagEnc s hu]
    | bool b =>
      cases hv with
      | int _ | flt _ => simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf,
          hpk, setRange, holds, propOr, cyEqScalar, numOf]
      | str t ht =>
        have hu := hsep b t (Or.inl ⟨rfl, hpk⟩)
        simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf, hpk,
          setRange, holds, propOr, cyEqScalar, numOf, tagEnc_ne_boolKey' t hu]
      | bool a =>
        simp [indexHit, buildQ, buildEq, valueToNumeric, tagKey, hf, rsMatch, docOf, hpk,
          setRange, holds, propOr, cyEqScalar]

/-- Equality: index scan = full scan + filter, for a pure stored value and a
scalar literal (an instance of `equal_exact_gen`). -/
theorem equal_exact (fields : Attr → Bool) (p : Props) (k : Attr) (c : Val)
    (hf : fields k = true) (hc : Const c) (hp : ∀ v, p k = some v → Pure v) :
    indexHit fields p (.equal k c) = holds p (.equal k c) := by
  apply equal_exact_gen fields p k c hf hc.lit (fun v hv => (hp v hv).stored)
  intro b t h
  rcases h with ⟨rfl, _⟩ | ⟨h1, _⟩
  · cases hc
  · cases hp _ h1

/-- `IN` over a list whose every member is answered exactly by `Equal`. -/
theorem inList_of_equal (fields : Attr → Bool) (p : Props) (k : Attr) (xs : List Val)
    (hf : fields k = true) (hne : xs ≠ [])
    (hsome : ∀ x ∈ xs, ∃ n, buildEq fields k x = some n)
    (heq : ∀ x ∈ xs, indexHit fields p (.equal k x) = holds p (.equal k x)) :
    indexHit fields p (.inList k (.list xs)) = holds p (.inList k (.list xs)) := by
  have hxs : xs.isEmpty = false := by cases xs <;> simp_all
  simp only [indexHit, buildQ, hxs, Bool.false_eq_true, ite_false, rsMatch_union, holds]
  have key : ∀ x ∈ xs, ∀ n, buildEq fields k x = some n →
      rsMatch (docOf fields p) n = isTrue (cyEqScalar (propOr p k) x) := by
    intro x hx n hn
    have := heq x hx
    simp only [indexHit, buildQ, hn, holds] at this
    exact this
  clear hne hxs heq
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨n, hn⟩ := hsome x (by simp)
    have kx := key x (by simp) n hn
    simp only [List.filterMap_cons, hn, List.any_cons]
    rw [kx, ih (fun y hy => hsome y (by simp [hy])) (fun y hy => key y (by simp [hy]))]

/-- `IN` over a list of scalar literals: exact. -/
theorem inList_exact (fields : Attr → Bool) (p : Props) (k : Attr) (xs : List Val)
    (hf : fields k = true) (hne : xs ≠ []) (hc : ∀ x ∈ xs, Const x)
    (hp : ∀ v, p k = some v → Pure v) :
    indexHit fields p (.inList k (.list xs)) = holds p (.inList k (.list xs)) :=
  inList_of_equal fields p k xs hf hne
    (fun x hx => by cases hc x hx <;> simp [buildEq, valueToNumeric, tagKey, hf])
    (fun x hx => equal_exact fields p k x hf (hc x hx) hp)

/-- **#2975: `IN` with Bool members (and stored Bools / `''`) is exact too.** -/
theorem inList_exact_gen (fields : Attr → Bool) (p : Props) (k : Attr) (xs : List Val)
    (hf : fields k = true) (hne : xs ≠ []) (hc : ∀ x ∈ xs, Lit x)
    (hp : ∀ v, p k = some v → Stored v)
    (hu : ∀ x ∈ xs, ∀ t, x = .str t → Utf8 t) (hpu : ∀ t, p k = some (.str t) → Utf8 t) :
    indexHit fields p (.inList k (.list xs)) = holds p (.inList k (.list xs)) :=
  inList_of_equal fields p k xs hf hne
    (fun x hx => by cases hc x hx <;> simp [buildEq, valueToNumeric, tagKey, hf])
    (fun x hx => equal_exact_gen fields p k x hf (hc x hx) hp (by
      intro b t h
      rcases h with ⟨_, h⟩ | ⟨_, h⟩
      · exact hpu t h
      · exact hu x hx t h))

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
  (try (rcases a with _ | a | _ | _)) <;> (try (rcases b with _ | b | _ | _)) <;>
  (try (rcases c with _ | c | _ | _)) <;>
  cases imn <;> cases imx <;>
  simp [indexHit, buildQ, isStrVal, buildNumRange, valueToNumeric, hf, rsMatch, docOf, h,
    setRange, holds, propOr, cyLt, cyLe, cyEqScalar, numOf, inNum, ENum.le, ENum.lt, ENum.eq] <;>
  omega

/-- The pre-#3081 statement (finite or NaN stored numbers) is an instance. -/
theorem numRange_exact_fin (fields : Attr → Bool) (p : Props) (k : Attr)
    (mn mx : Option Val) (imn imx : Bool) (hf : fields k = true)
    (hb : mn.isSome ∨ mx.isSome)
    (hmn : ∀ v, mn = some v → NumConst v) (hmx : ∀ v, mx = some v → NumConst v)
    (hp : ∀ v, p k = some v → FinNum v) :
    indexHit fields p (.range k mn mx imn imx) = holds p (.range k mn mx imn imx) :=
  numRange_exact fields p k mn mx imn imx hf hb hmn hmx (fun v hv => by
    cases hp v hv with
    | int i => exact .int i
    | flt i => exact .flt _
    | nan => exact .flt _)

end IndexLayer
