import FalkorEffectsCodec.Prim
/-
# The `SIValue` codec: `graph/src/effects/v3/value.rs`

| here | there |
| --- | --- |
| `Val`              | `runtime::value::Value` (`graph/src/runtime/value.rs:180`) — floats as their bit patterns, `i64` as its two's-complement bits |
| `T_*`              | `serialization::si_type` (`graph/src/graph/graphblas/serialization.rs:108-121`) |
| `encV` / `encL`    | `impl EffectEncode<3> for Value` (`value.rs:33-125`) |
| `Step`             | `read_one`'s `Option<Value>`: `None` = a frame was pushed (`value.rs:195`) |
| `readOne`          | `read_one` (`value.rs:197-265`) |
| `Frame`            | `enum Frame::List { items, left }` (`value.rs:132`) |
| `deliver`          | the inner `loop` of `decode` (`value.rs:173-188`) |
| `loop`             | the outer `loop` of `decode` (`value.rs:164-189`), with fuel |
| `decodeV`          | `impl EffectDecode<3> for Value::decode` (`value.rs:161`) |
| `propValid`        | `pending::is_valid_property` (`graph/src/runtime/pending.rs:63`) |

Floats: `f64::to_le_bytes`/`from_le_bytes` are the identity on the bit pattern
(`to_bits`/`from_bits`), so a float *is* its `BitVec 64` here and "roundtrips"
means bit-for-bit — NaN payloads and `-0.0` included. Lean's own `Float` is not
used: its runtime canonicalises NaN (`(Float.ofBits 0x7ff8000000000001).toBits
= 0x7ff8000000000000`), which Rust does not, so it would be the wrong model.

Fuel: the Rust `loop` has none. Every turn reads a 4-byte tag, so it runs at
most `remaining/4` times; `loop_fuel_enough` proves `decodeV`'s fuel
(`r.length + 1`) can never run out, so the fuel is not a semantic change.
-/

namespace FalkorCodec

/-- `si_type` constants (`serialization.rs:108-121`). -/
def T_MAP : Nat := 1
def T_ARRAY : Nat := 8
def T_DATETIME : Nat := 32
def T_DATE : Nat := 128
def T_TIME : Nat := 256
def T_DURATION : Nat := 1024
def T_STRING : Nat := 2048
def T_BOOL : Nat := 4096
def T_INT64 : Nat := 8192
def T_DOUBLE : Nat := 16384
def T_NULL : Nat := 32768
def T_POINT : Nat := 131072
def T_VECTOR_F32 : Nat := 262144
def T_INTERN : Nat := 524288
/-- `T_INTERN | T_STRING`. -/
def T_ISTRING : Nat := 526336

theorem T_ISTRING_eq : T_ISTRING = T_INTERN ||| T_STRING := by decide

/-- Tags are unambiguous: the fifteen values `read_one` dispatches on are
pairwise distinct, and each fits the 4-byte wire (`mod.rs:486` checks the
same thing on the Rust side). -/
def valueTags : List Nat :=
  [T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_ISTRING, T_ARRAY, T_MAP, T_POINT,
   T_VECTOR_F32, T_DATETIME, T_DATE, T_TIME, T_DURATION]

theorem valueTags_nodup : valueTags.Nodup := by decide
theorem valueTags_fit : ∀ t ∈ valueTags, t < 2 ^ 32 := by decide

inductive Val where
  | null
  | bool (b : Bool)
  | int (x : BitVec 64)
  | float (bits : BitVec 64)
  | str (s : Bytes)
  | list (xs : List Val)
  | map (vals : List Val)
  | node (id : Nat)
  | rel (id : Nat)
  | path (xs : List Val)
  | vecf32 (xs : List (BitVec 32))
  | point (lat lon : BitVec 32)
  | datetime (x : BitVec 64)
  | date (x : BitVec 64)
  | time (x : BitVec 64)
  | duration (x : BitVec 64)
deriving Repr, Inhabited

/-- Encode errors: `EncodeError::BlockCountTooLarge` (`error.rs:99`) and the
`panic!` of `value.rs:121`. -/
inductive EErr where
  | blockCountTooLarge
  | headerShapeMismatch
  | endpointMisaligned
  | optionsFieldTypeMismatch
  | badFieldType
  | emptyIndexSchemaList
  | idList
  | panic
  /-- `EncodeError::RowShapeMismatch` (`error.rs:46`, #2916). -/
  | rowShapeMismatch
  /-- `EncodeError::EmptyRecord` (`error.rs:57`, #2916). -/
  | emptyRecord
deriving Repr, DecidableEq

def f32s (xs : List (BitVec 32)) : Bytes := xs.flatMap (fun f => w32 f.toNat)

-- `EffectEncode<3> for Value` (`value.rs:34-124`). `interned` is
-- `string_pool::global().is_interned` (`value.rs:59`), which only chooses the tag.
mutual
def encV (interned : Bytes → Bool) : Val → Except EErr Bytes
  | .null => .ok (w32 T_NULL)
  | .bool b => .ok (w32 T_BOOL ++ w8 (if b then 1 else 0))
  | .int x => .ok (w32 T_INT64 ++ w64 x.toNat)
  | .float f => .ok (w32 T_DOUBLE ++ w64 f.toNat)
  | .str s => .ok (w32 (if interned s then T_ISTRING else T_STRING) ++ wString s)
  | .list xs =>
    -- `u32::try_from(items.len())` (`value.rs:70`)
    if xs.length < 2 ^ 32 then
      match encL interned xs with
      | .ok b => .ok (w32 T_ARRAY ++ w32 xs.length ++ b)
      | .error e => .error e
    else .error .blockCountTooLarge
  | .point la lo => .ok (w32 T_POINT ++ w32 la.toNat ++ w32 lo.toNat)
  | .vecf32 xs =>
    if xs.length < 2 ^ 32 then .ok (w32 T_VECTOR_F32 ++ w32 xs.length ++ f32s xs)
    else .error .blockCountTooLarge
  | .datetime x => .ok (w32 T_DATETIME ++ w64 x.toNat)
  | .date x => .ok (w32 T_DATE ++ w64 x.toNat)
  | .time x => .ok (w32 T_TIME ++ w64 x.toNat)
  | .duration x => .ok (w32 T_DURATION ++ w64 x.toNat)
  -- `other => panic!("value cannot appear in an effect")` (`value.rs:121`)
  | .map _ | .node _ | .rel _ | .path _ => .error .panic
def encL (interned : Bytes → Bool) : List Val → Except EErr Bytes
  | [] => .ok []
  | v :: vs =>
    match encV interned v with
    | .error e => .error e
    | .ok a => match encL interned vs with
      | .error e => .error e
      | .ok b => .ok (a ++ b)
end

/-! ## Decoder -/

inductive Step where
  | value (v : Val)
  | opened (n : Nat)

def f32 : R (BitVec 32) := do let n ← u32; pure (BitVec.ofNat 32 n)
def i64 : R (BitVec 64) := do let n ← u64; pure (BitVec.ofNat 64 n)

/-- `for _ in 0..n { v.push(r.f32()?) }` (`value.rs:251`). -/
def readN : Nat → R α → R (List α)
  | 0, _ => pure []
  | n + 1, m => do let a ← m; let as ← readN n m; pure (a :: as)

/-- The smallest a value can encode to (`value.rs:17`). -/
def MIN_VALUE_BYTES : Nat := 4

/-- `read_one` (`value.rs:197-265`), arm for arm and in the same order. -/
def readOne (utf8 : Bytes → Bool) : R Step := do
  let t ← u32
  if t = T_NULL then pure (.value .null)
  else if t = T_BOOL then do let b ← u8; pure (.value (.bool (b ≠ 0)))
  else if t = T_INT64 then do let x ← i64; pure (.value (.int x))
  else if t = T_DOUBLE then do let x ← i64; pure (.value (.float x))
  else if t = T_STRING ∨ t = T_ISTRING then do
    let s ← rString utf8; pure (.value (.str s))
  else if t = T_ARRAY then do
    let n ← u32
    let n ← guardCount n MIN_VALUE_BYTES
    if n = 0 then pure (.value (.list []))
    -- `ThinVec::with_capacity(n)`, `Frame::List { left: n }`
    else pure (.opened n)
  else if t = T_MAP then fail (.badValueType T_MAP)
  else if t = T_POINT then do
    let la ← f32; let lo ← f32; pure (.value (.point la lo))
  else if t = T_VECTOR_F32 then do
    let n ← u32
    let n ← guardCount n 4
    let xs ← readN n f32
    pure (.value (.vecf32 xs))
  else if t = T_DATETIME then do let x ← i64; pure (.value (.datetime x))
  else if t = T_DATE then do let x ← i64; pure (.value (.date x))
  else if t = T_TIME then do let x ← i64; pure (.value (.time x))
  else if t = T_DURATION then do let x ← i64; pure (.value (.duration x))
  else fail (.badValueType t)

/-- `Frame::List { items, left }`. `items` is pushed at the end, as `ThinVec::push`. -/
structure Frame where
  items : List Val
  left : Nat

inductive Deliv where
  | done (v : Val)
  | more (st : List Frame)
  | panic

/-- The inner loop (`value.rs:173-188`). `*left -= 1` on `left = 0` is an
overflow panic in debug (and a wrap in release); it is modelled as `panic`
and `loop_noPanic` proves it cannot happen. -/
def deliver : Val → List Frame → Deliv
  | v, [] => .done v
  | v, f :: st =>
    if f.left = 0 then .panic
    else if f.left - 1 > 0 then .more (⟨f.items ++ [v], f.left - 1⟩ :: st)
    else deliver (.list (f.items ++ [v])) st

/-- The outer loop (`value.rs:164-189`). -/
def loop (utf8 : Bytes → Bool) : Nat → List Frame → R Val
  | 0, _ => fun _ => .error .outOfFuel
  | fuel + 1, st => fun r =>
    match readOne utf8 r with
    | .error e => .error e
    | .ok (.opened n, r') => loop utf8 fuel (⟨[], n⟩ :: st) r'
    | .ok (.value v, r') =>
      match deliver v st with
      | .done v => .ok (v, r')
      | .more st' => loop utf8 fuel st' r'
      | .panic => .error .panic

def decodeV (utf8 : Bytes → Bool) : R Val := fun r => loop utf8 (r.length + 1) [] r

/-! ## Encoder facts -/

mutual
def cnt : Val → Nat
  | .list xs => 1 + cntL xs
  | _ => 1
def cntL : List Val → Nat
  | [] => 0
  | v :: vs => cnt v + cntL vs
end

-- What a roundtrip needs that `encV` does not check: string lengths fit
-- `u64` (they do: they are in memory) and bodies are UTF-8 (they are: they are
-- Rust `String`s).
mutual
def WF (utf8 : Bytes → Bool) : Val → Prop
  | .str s => s.length + 1 < 2 ^ 64 ∧ utf8 s = true
  | .list xs => WFL utf8 xs
  | _ => True
def WFL (utf8 : Bytes → Bool) : List Val → Prop
  | [] => True
  | v :: vs => WF utf8 v ∧ WFL utf8 vs
end

theorem f32s_length (xs : List (BitVec 32)) : (f32s xs).length = 4 * xs.length := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    unfold f32s at *
    rw [List.flatMap_cons, List.length_append, ih]
    simp [w32]; omega

mutual
theorem encV_len (I : Bytes → Bool) : ∀ v bs, encV I v = .ok bs → 4 * cnt v ≤ bs.length
  | .null, bs, h => by simp [encV] at h; subst h; simp [cnt, w32]
  | .bool b, bs, h => by simp [encV] at h; subst h; simp [cnt, w32]
  | .int x, bs, h => by simp [encV] at h; subst h; simp [cnt, w32]
  | .float x, bs, h => by simp [encV] at h; subst h; simp [cnt, w32]
  | .str s, bs, h => by simp [encV] at h; subst h; simp [cnt, w32]
  | .list xs, bs, h => by
    simp only [encV] at h
    split at h
    · split at h
      · rename_i b hb
        simp at h; subst h
        have := encL_len I xs b hb
        simp [cnt, w32]; omega
      · cases h
    · cases h
  | .map _, bs, h => by simp [encV] at h
  | .node _, bs, h => by simp [encV] at h
  | .rel _, bs, h => by simp [encV] at h
  | .path _, bs, h => by simp [encV] at h
  | .vecf32 xs, bs, h => by
    simp only [encV] at h; split at h
    · simp at h; subst h; simp [cnt, w32]
    · cases h
  | .point a b, bs, h => by simp [encV] at h; subst h; simp [cnt, w32]
  | .datetime x, bs, h => by simp [encV] at h; subst h; simp [cnt, w32]
  | .date x, bs, h => by simp [encV] at h; subst h; simp [cnt, w32]
  | .time x, bs, h => by simp [encV] at h; subst h; simp [cnt, w32]
  | .duration x, bs, h => by simp [encV] at h; subst h; simp [cnt, w32]
theorem encL_len (I : Bytes → Bool) : ∀ xs bs, encL I xs = .ok bs →
    4 * cntL xs ≤ bs.length ∧ 4 * xs.length ≤ bs.length
  | [], bs, h => by simp [encL] at h; subst h; simp [cntL]
  | v :: vs, bs, h => by
    simp only [encL] at h
    split at h
    · cases h
    · rename_i a ha
      split at h
      · cases h
      · rename_i b hb
        cases h
        have h1 := encV_len I v a ha
        have h2 := encL_len I vs b hb
        have h3 : 1 ≤ cnt v := by cases v <;> simp [cnt] <;> omega
        simp [cntL]; omega
end

/-! ## Decoder facts -/

theorem f32_w32 (x : BitVec 32) (rest : Bytes) : f32 (w32 x.toNat ++ rest) = .ok (x, rest) := by
  simp [f32, u32_w32 x.isLt]

theorem i64_w64 (x : BitVec 64) (rest : Bytes) : i64 (w64 x.toNat ++ rest) = .ok (x, rest) := by
  simp [i64, u64_w64 x.isLt]

theorem readN_f32s (xs : List (BitVec 32)) (rest : Bytes) :
    readN xs.length f32 (f32s xs ++ rest) = .ok (xs, rest) := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.length_cons, readN, f32s, List.flatMap_cons, List.append_assoc, bind_apply]
    rw [f32_w32]
    simp only [bind_apply]
    rw [show List.flatMap (fun f => w32 f.toNat) xs = f32s xs from rfl, ih]
    rfl

/-- What the outer loop does with a finished value, by the inner loop's verdict. -/
def cont (utf8 : Bytes → Bool) : Deliv → Nat → R Val
  | .done v, _ => fun r => .ok (v, r)
  | .more st, fuel => loop utf8 fuel st
  | .panic, _ => fun _ => .error .panic

theorem loop_value {utf8 fuel st r v r'} (h : readOne utf8 r = .ok (.value v, r')) :
    loop utf8 (fuel + 1) st r = cont utf8 (deliver v st) fuel r' := by
  simp only [loop, h]
  cases deliver v st <;> rfl

theorem loop_opened {utf8 fuel st r n r'} (h : readOne utf8 r = .ok (.opened n, r')) :
    loop utf8 (fuel + 1) st r = loop utf8 fuel (⟨[], n⟩ :: st) r' := by
  simp only [loop, h]

attribute [local simp] u32_w32 u8_w8 i64_w64 f32_w32

/-- Every scalar arm of `readOne` reads back exactly what `encV` wrote. -/
theorem readOne_scalar (utf8 I : Bytes → Bool) (v : Val) (bs rest : Bytes)
    (h : encV I v = .ok bs) (hw : WF utf8 v) (hl : ∀ xs, v ≠ .list xs) :
    readOne utf8 (bs ++ rest) = .ok (.value v, rest) := by
  cases v with
  | list xs => exact absurd rfl (hl xs)
  | null => simp [encV] at h; subst h; simp [readOne, T_NULL]
  | bool b =>
    simp [encV] at h; subst h
    cases b <;> simp [readOne, T_NULL, T_BOOL]
  | int x => simp [encV] at h; subst h; simp [readOne, T_NULL, T_BOOL, T_INT64]
  | float x =>
    simp [encV] at h; subst h; simp [readOne, T_NULL, T_BOOL, T_INT64, T_DOUBLE]
  | str s =>
    simp [encV] at h; subst h
    simp only [WF] at hw
    have hr : rString utf8 (w64 (s.length + 1) ++ (s ++ 0 :: rest)) = .ok (s, rest) := by
      simpa [wString] using rString_wString utf8 s rest hw.1 hw.2
    split <;>
      simp [readOne, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_ISTRING, wString, hr]
  | map _ => simp [encV] at h
  | node _ => simp [encV] at h
  | rel _ => simp [encV] at h
  | path _ => simp [encV] at h
  | vecf32 xs =>
    simp only [encV] at h; split at h
    · rename_i hlt
      simp at h; subst h
      have hr := readN_f32s xs rest
      have hg : guardCount xs.length 4 (f32s xs ++ rest) = .ok (xs.length, f32s xs ++ rest) := by
        simp [guardCount, f32s_length]; omega
      simp [readOne, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_ISTRING, T_ARRAY, T_MAP,
        T_POINT, T_VECTOR_F32, hlt, hg, hr]
    · cases h
  | point a b =>
    simp [encV] at h; subst h
    simp [readOne, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_ISTRING, T_ARRAY, T_MAP, T_POINT]
  | datetime x =>
    simp [encV] at h; subst h
    simp [readOne, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_ISTRING, T_ARRAY, T_MAP,
      T_POINT, T_VECTOR_F32, T_DATETIME]
  | date x =>
    simp [encV] at h; subst h
    simp [readOne, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_ISTRING, T_ARRAY, T_MAP,
      T_POINT, T_VECTOR_F32, T_DATETIME, T_DATE]
  | time x =>
    simp [encV] at h; subst h
    simp [readOne, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_ISTRING, T_ARRAY, T_MAP,
      T_POINT, T_VECTOR_F32, T_DATETIME, T_DATE, T_TIME]
  | duration x =>
    simp [encV] at h; subst h
    simp [readOne, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_ISTRING, T_ARRAY, T_MAP,
      T_POINT, T_VECTOR_F32, T_DATETIME, T_DATE, T_TIME, T_DURATION]

/-- The list arm: an empty list is a value; a non-empty one opens a frame of
exactly its length and leaves the elements unread. -/
theorem readOne_list (utf8 : Bytes → Bool) (xs : List Val) (b rest : Bytes)
    (hlt : xs.length < 2 ^ 32) (hb : 4 * xs.length ≤ b.length) :
    readOne utf8 (w32 T_ARRAY ++ w32 xs.length ++ b ++ rest) =
      if xs.length = 0 then .ok (.value (.list []), b ++ rest)
      else .ok (.opened xs.length, b ++ rest) := by
  have hg : guardCount xs.length MIN_VALUE_BYTES (b ++ rest) = .ok (xs.length, b ++ rest) := by
    simp [guardCount, MIN_VALUE_BYTES]; omega
  simp only [List.append_assoc]
  simp [readOne, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_ISTRING, T_ARRAY, hlt, hg]
  split <;> rfl

theorem scalar_step (utf8 I : Bytes → Bool) (v : Val) (bs : Bytes) (h : encV I v = .ok bs)
    (hw : WF utf8 v) (hl : ∀ xs, v ≠ .list xs) (fuel : Nat) (st : List Frame) (rest : Bytes)
    (hf : cnt v ≤ fuel) :
    loop utf8 fuel st (bs ++ rest) = cont utf8 (deliver v st) (fuel - cnt v) rest := by
  have hc : cnt v = 1 := by cases v <;> simp [cnt] <;> exact absurd rfl (hl _)
  obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
  rw [loop_value (readOne_scalar utf8 I v bs rest h hw hl), hc]
  rfl

theorem encL_nil {I : Bytes → Bool} {b : Bytes} (h : encL I [] = .ok b) : b = [] := by
  simp [encL] at h; exact h

theorem encL_cons {I : Bytes → Bool} {x : Val} {xs : List Val} {b : Bytes}
    (h : encL I (x :: xs) = .ok b) :
    ∃ a b', encV I x = .ok a ∧ encL I xs = .ok b' ∧ b = a ++ b' := by
  simp only [encL] at h
  split at h
  · cases h
  · rename_i a ha
    split at h
    · cases h
    · rename_i b' hb; cases h; exact ⟨a, b', ha, hb, rfl⟩

theorem encV_list {I : Bytes → Bool} {xs : List Val} {bs : Bytes} (h : encV I (.list xs) = .ok bs) :
    xs.length < 2 ^ 32 ∧ ∃ b, encL I xs = .ok b ∧ bs = w32 T_ARRAY ++ w32 xs.length ++ b := by
  simp only [encV] at h
  split at h
  · rename_i hlt
    split at h
    · rename_i b hb; cases h; exact ⟨hlt, b, hb, rfl⟩
    · cases h
  · cases h

mutual
/-- **The value roundtrip, generalised over whatever the decoder is in the
middle of.** Decoding `encV v` on top of any stack of open frames consumes
exactly those bytes and hands exactly `v` to the innermost frame. -/
theorem loop_enc (utf8 I : Bytes → Bool) : ∀ (v : Val) (bs : Bytes), encV I v = .ok bs →
    WF utf8 v → ∀ (fuel : Nat) (st : List Frame) (rest : Bytes), cnt v ≤ fuel →
    loop utf8 fuel st (bs ++ rest) = cont utf8 (deliver v st) (fuel - cnt v) rest
  | .list xs, bs, h, hw, fuel, st, rest, hf => by
    obtain ⟨hlt, b, hb, rfl⟩ := encV_list h
    have hlen := (encL_len I xs b hb).2
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp [cnt] at hf; omega⟩
    have hr := readOne_list utf8 xs b rest hlt hlen
    cases xs with
    | nil =>
      simp only [List.length_nil, ite_true] at hr ⊢
      rw [loop_value hr, encL_nil hb]
      simp [cnt, cntL]
    | cons x xs' =>
      simp only [List.length_cons, Nat.add_one_ne_zero, ite_false] at hr ⊢
      rw [loop_opened hr]
      have := loopL_enc utf8 I (x :: xs') b hb hw (by simp) [] f st rest
        (by simp only [cnt] at hf; omega)
      simp only [List.length_cons] at this
      rw [this]
      simp only [List.nil_append, cnt]
      congr 1
      omega
  | .null, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .bool _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .int _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .float _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .str _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .map _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .node _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .rel _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .path _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .vecf32 _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .point _ _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .datetime _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .date _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .time _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf
  | .duration _, bs, h, hw, fuel, st, rest, hf =>
    scalar_step utf8 I _ bs h hw (by intro _ h; cases h) fuel st rest hf

/-- The frame half: the children of a non-empty list, read on top of the frame
that is waiting for them, close that frame into exactly the list. -/
theorem loopL_enc (utf8 I : Bytes → Bool) : ∀ (xs : List Val) (bs : Bytes), encL I xs = .ok bs →
    WFL utf8 xs → xs ≠ [] → ∀ (items : List Val) (fuel : Nat) (st : List Frame) (rest : Bytes),
    cntL xs ≤ fuel →
    loop utf8 fuel (⟨items, xs.length⟩ :: st) (bs ++ rest) =
      cont utf8 (deliver (.list (items ++ xs)) st) (fuel - cntL xs) rest
  | [], _, _, _, hne, _, _, _, _, _ => absurd rfl hne
  | x :: xs', bs, h, hw, _, items, fuel, st, rest, hf => by
    obtain ⟨a, b', ha, hb', rfl⟩ := encL_cons h
    simp only [WFL] at hw
    simp only [cntL] at hf
    rw [List.append_assoc, loop_enc utf8 I x a ha hw.1 fuel _ (b' ++ rest) (by omega)]
    simp only [List.length_cons, deliver, Nat.add_one_ne_zero, ite_false, Nat.add_sub_cancel]
    cases xs' with
    | nil =>
      rw [encL_nil hb']
      simp [cntL]
    | cons y ys =>
      simp only [List.length_cons, Nat.zero_lt_succ, ite_true, cont]
      have := loopL_enc utf8 I (y :: ys) b' hb' hw.2 (by simp) (items ++ [x]) (fuel - cnt x) st
        rest (by simp only [cntL] at hf ⊢; omega)
      simp only [List.length_cons] at this
      rw [this]
      simp only [List.append_assoc, List.singleton_append, cntL]
      congr 1
      omega
end

/-- **`read(write(v)) = v`**, for every value `encV` accepts: the decoder
consumes exactly the encoding, returns exactly `v` — bit-for-bit, since every
float is its bit pattern — and leaves `rest` untouched, so the next field
parses from the right offset. -/
theorem decodeV_encV (utf8 I : Bytes → Bool) (v : Val) (bs rest : Bytes)
    (h : encV I v = .ok bs) (hw : WF utf8 v) :
    decodeV utf8 (bs ++ rest) = .ok (v, rest) := by
  have hl := encV_len I v bs h
  unfold decodeV
  rw [loop_enc utf8 I v bs h hw _ [] rest (by simp; omega)]
  simp [deliver, cont]

/-! ## Hostile input: no panic, no over-read, the fuel never runs out -/

theorem readN_consumes (m : R α) (hm : Consumes m) : ∀ n, Consumes (readN n m)
  | 0 => pure_consumes _
  | n + 1 => bind_consumes hm (fun _ => bind_consumes (readN_consumes m hm n)
      (fun _ => pure_consumes _))

theorem readN_noPanic (m : R α) (hm : NoPanic m) : ∀ n, NoPanic (readN n m)
  | 0 => pure_noPanic _
  | n + 1 => bind_noPanic hm (fun _ => bind_noPanic (readN_noPanic m hm n)
      (fun _ => pure_noPanic _))

theorem f32_consumes : Consumes f32 := bind_consumes (rdU_consumes 4) (fun _ => pure_consumes _)
theorem i64_consumes : Consumes i64 := bind_consumes (rdU_consumes 8) (fun _ => pure_consumes _)
theorem f32_noPanic : NoPanic f32 := bind_noPanic (rdU_noPanic 4) (fun _ => pure_noPanic _)
theorem i64_noPanic : NoPanic i64 := bind_noPanic (rdU_noPanic 8) (fun _ => pure_noPanic _)

/-- Everything after `read_one`'s tag, as one function of the tag. -/
def readBody (utf8 : Bytes → Bool) (t : Nat) : R Step :=
  if t = T_NULL then pure (.value .null)
  else if t = T_BOOL then do let b ← u8; pure (.value (.bool (b ≠ 0)))
  else if t = T_INT64 then do let x ← i64; pure (.value (.int x))
  else if t = T_DOUBLE then do let x ← i64; pure (.value (.float x))
  else if t = T_STRING ∨ t = T_ISTRING then do
    let s ← rString utf8; pure (.value (.str s))
  else if t = T_ARRAY then do
    let n ← u32
    let n ← guardCount n MIN_VALUE_BYTES
    if n = 0 then pure (.value (.list []))
    else pure (.opened n)
  else if t = T_MAP then fail (.badValueType T_MAP)
  else if t = T_POINT then do
    let la ← f32; let lo ← f32; pure (.value (.point la lo))
  else if t = T_VECTOR_F32 then do
    let n ← u32
    let n ← guardCount n 4
    let xs ← readN n f32
    pure (.value (.vecf32 xs))
  else if t = T_DATETIME then do let x ← i64; pure (.value (.datetime x))
  else if t = T_DATE then do let x ← i64; pure (.value (.date x))
  else if t = T_TIME then do let x ← i64; pure (.value (.time x))
  else if t = T_DURATION then do let x ← i64; pure (.value (.duration x))
  else fail (.badValueType t)

theorem readOne_eq (utf8 : Bytes → Bool) : readOne utf8 = (u32 >>= readBody utf8) := rfl

/-- The three hostile-input properties at once: only consumes a prefix, never
panics, never reports running out of fuel (the reader has no fuel of its own). -/
def Good (m : R α) : Prop :=
  Consumes m ∧ NoPanic m ∧ ∀ r, m r ≠ .error .outOfFuel

theorem good_pure (a : α) : Good (pure a : R α) := ⟨pure_consumes a, pure_noPanic a, by simp⟩
theorem good_fail {e : DErr} (h1 : e ≠ .panic) (h2 : e ≠ .outOfFuel) : Good (fail e : R α) :=
  ⟨fail_consumes e, fail_noPanic h1, by simp [h2]⟩
theorem good_bind {m : R α} {f : α → R β} (hm : Good m) (hf : ∀ a, Good (f a)) :
    Good (m >>= f) := by
  refine ⟨bind_consumes hm.1 (fun a => (hf a).1), bind_noPanic hm.2.1 (fun a => (hf a).2.1), ?_⟩
  intro r; simp only [bind_apply]; split
  · rename_i e he; intro h; cases h; exact hm.2.2 r he
  · exact (hf _).2.2 _
theorem good_ite {c : Prop} [Decidable c] {m₁ m₂ : R α} (h₁ : Good m₁) (h₂ : Good m₂) :
    Good (if c then m₁ else m₂) := by split <;> assumption
theorem good_take (n : Nat) : Good (take n) :=
  ⟨take_consumes n, take_noPanic n, by intro r; unfold take; split <;> simp⟩
theorem good_rdU (k : Nat) : Good (rdU k) := good_bind (good_take k) (fun _ => good_pure _)
theorem good_guard (c m : Nat) : Good (guardCount c m) :=
  ⟨guardCount_consumes c m, guardCount_noPanic c m, by
    intro r; unfold guardCount; split <;> simp⟩
theorem good_rString (utf8 : Bytes → Bool) : Good (rString utf8) := by
  unfold rString
  refine good_bind (good_rdU 8) (fun len => good_ite (good_fail (by decide) (by decide)) ?_)
  refine good_bind (good_take _) (fun raw => good_ite (good_fail (by decide) (by decide)) ?_)
  exact good_ite (good_pure _) (good_fail (by decide) (by decide))
theorem good_readN {m : R α} (hm : Good m) : ∀ n, Good (readN n m)
  | 0 => good_pure _
  | n + 1 => good_bind hm (fun _ => good_bind (good_readN hm n) (fun _ => good_pure _))
theorem good_f32 : Good f32 := good_bind (good_rdU 4) (fun _ => good_pure _)
theorem good_i64 : Good i64 := good_bind (good_rdU 8) (fun _ => good_pure _)

theorem good_readBody (utf8 : Bytes → Bool) (t : Nat) : Good (readBody utf8 t) := by
  have gv : ∀ {α} (m : R α) (k : α → Val), Good m → Good (m >>= fun x => pure (Step.value (k x))) :=
    fun m k hm => good_bind hm (fun _ => good_pure _)
  unfold readBody
  refine good_ite (good_pure _) (good_ite (gv _ _ (good_rdU 1)) (good_ite (gv _ _ good_i64)
    (good_ite (gv _ _ good_i64) (good_ite (gv _ _ (good_rString utf8)) (good_ite ?_
    (good_ite (good_fail (by decide) (by decide)) (good_ite ?_ (good_ite ?_
    (good_ite (gv _ _ good_i64) (good_ite (gv _ _ good_i64) (good_ite (gv _ _ good_i64)
    (good_ite (gv _ _ good_i64) (good_fail (by intro h; cases h) (by intro h; cases h))))))))))))))
  · exact good_bind (good_rdU 4) (fun _ => good_bind (good_guard _ _)
      (fun _ => good_ite (good_pure _) (good_pure _)))
  · exact good_bind good_f32 (fun _ => good_bind good_f32 (fun _ => good_pure _))
  · exact good_bind (good_rdU 4) (fun _ => good_bind (good_guard _ _)
      (fun _ => good_bind (good_readN good_f32 _) (fun _ => good_pure _)))

theorem good_readOne (utf8 : Bytes → Bool) : Good (readOne utf8) := by
  rw [readOne_eq]; exact good_bind (good_rdU 4) (good_readBody utf8)

/-- `read_one` consumes at least its 4-byte tag, and only ever a prefix. -/
theorem readOne_progress {utf8 : Bytes → Bool} {r : Bytes} {s : Step} {r' : Bytes}
    (h : readOne utf8 r = .ok (s, r')) : ∃ pre, r = pre ++ r' ∧ 4 ≤ pre.length := by
  rw [readOne_eq] at h
  simp only [bind_apply, u32, rdU, take] at h
  by_cases hl : r.length < 4
  · simp [hl] at h
  · simp only [hl, ite_false, pure_apply] at h
    obtain ⟨p, hp⟩ := (good_readBody utf8 _).1 _ _ _ h
    refine ⟨r.take 4 ++ p, ?_, ?_⟩
    · rw [List.append_assoc, ← hp, List.take_append_drop]
    · simp; omega

/-- Where `readBody` can end: a `Step.opened n` only ever comes from the array
arm, after `guard_count` has accepted `n`. -/
def OpenedOk (s : Step) (r' : Bytes) : Prop := ∀ n, s = .opened n → 1 ≤ n ∧ 4 * n ≤ r'.length

def Ends (m : R Step) : Prop := ∀ r s r', m r = .ok (s, r') → OpenedOk s r'

theorem ends_value {α} (m : R α) (k : α → Val) : Ends (m >>= fun x => pure (Step.value (k x))) := by
  intro r s r' h; simp only [bind_apply] at h; split at h
  · cases h
  · simp at h; intro n hn; rw [← h.1] at hn; cases hn
theorem ends_fail (e : DErr) : Ends (fail e) := by intro r s r' h; cases h
theorem ends_pure_value (v : Val) : Ends (pure (Step.value v)) := by
  intro r s r' h; cases h; intro n hn; cases hn
theorem ends_ite {c : Prop} [Decidable c] {m₁ m₂ : R Step} (h₁ : Ends m₁) (h₂ : Ends m₂) :
    Ends (if c then m₁ else m₂) := by split <;> assumption
theorem ends_array :
    Ends (do let n ← u32; let n ← guardCount n MIN_VALUE_BYTES
             if n = 0 then pure (Step.value (.list [])) else pure (Step.opened n)) := by
  intro r s r' h
  simp only [bind_apply] at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i c r2 hg
      obtain ⟨rfl, rfl, hb⟩ := guardCount_bound hg
      split at h
      · cases h; intro n hn; cases hn
      · cases h; intro n hn; cases hn; simp [MIN_VALUE_BYTES] at hb; constructor <;> omega

theorem ends_readBody (utf8 : Bytes → Bool) (t : Nat) : Ends (readBody utf8 t) := by
  unfold readBody
  refine ends_ite (ends_pure_value _) (ends_ite (ends_value _ _) (ends_ite (ends_value _ _)
    (ends_ite (ends_value _ _) (ends_ite (ends_value _ _) (ends_ite ends_array
    (ends_ite (ends_fail _) (ends_ite ?_ (ends_ite ?_
    (ends_ite (ends_value _ _) (ends_ite (ends_value _ _) (ends_ite (ends_value _ _)
    (ends_ite (ends_value _ _) (ends_fail _)))))))))))))
  · intro r s r' h
    simp only [bind_apply] at h
    split at h
    · cases h
    · split at h
      · cases h
      · simp at h; intro n hn; rw [← h.1] at hn; cases hn
  · intro r s r' h
    simp only [bind_apply] at h
    split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · simp at h; intro n hn; rw [← h.1] at hn; cases hn

/-- A frame is only opened for a count that is non-zero and that the bytes
still unread could hold at four bytes per child: this is the bound on each
`ThinVec::with_capacity(n)` (`value.rs:227`). -/
theorem readOne_opened_bound {utf8 : Bytes → Bool} {r : Bytes} {n : Nat} {r' : Bytes}
    (h : readOne utf8 r = .ok (.opened n, r')) : 1 ≤ n ∧ 4 * n ≤ r'.length := by
  rw [readOne_eq] at h
  simp only [bind_apply] at h
  split at h
  · cases h
  · exact ends_readBody utf8 _ _ _ _ h n rfl

/-- Every open frame is waiting on at least one child. -/
def FramesOk (st : List Frame) : Prop := ∀ f ∈ st, 1 ≤ f.left

theorem deliver_ok : ∀ (v : Val) (st : List Frame), FramesOk st →
    deliver v st ≠ .panic ∧ ∀ st', deliver v st = .more st' → FramesOk st'
  | v, [], _ => by simp [deliver]
  | v, f :: st, h => by
    have hf : 1 ≤ f.left := h f (by simp)
    have hst : FramesOk st := fun g hg => h g (by simp [hg])
    simp only [deliver]
    rw [if_neg (by omega)]
    split
    · refine ⟨by simp, ?_⟩
      intro st' he; cases he
      intro g hg; simp at hg
      rcases hg with rfl | hg
      · simp; omega
      · exact hst g hg
    · exact deliver_ok _ st hst

/-- **The decoder never panics**, on any input whatsoever — including the
`*left -= 1` of `value.rs:178`, which would underflow on a frame of zero. -/
theorem loop_noPanic (utf8 : Bytes → Bool) :
    ∀ fuel st r, FramesOk st → loop utf8 fuel st r ≠ .error .panic
  | 0, _, _, _ => by simp [loop]
  | fuel + 1, st, r, hst => by
    simp only [loop]
    split
    · rename_i e he; intro h; cases h; exact (good_readOne utf8).2.1 r he
    · rename_i n r' hr
      apply loop_noPanic utf8 fuel _ r'
      intro f hf; simp at hf
      rcases hf with rfl | hf
      · exact (readOne_opened_bound hr).1
      · exact hst f hf
    · rename_i v r' _
      obtain ⟨hp, hm⟩ := deliver_ok v st hst
      split
      · simp
      · rename_i st' hd; exact loop_noPanic utf8 fuel st' r' (hm st' hd)
      · rename_i hd; exact absurd hd hp

/-- **The fuel is never what stops the decoder**: four bytes per turn, so
`remaining / 4 + 1` turns always suffice. -/
theorem loop_fuel_enough (utf8 : Bytes → Bool) :
    ∀ fuel st r, r.length < 4 * fuel → loop utf8 fuel st r ≠ .error .outOfFuel
  | 0, _, r, h => by simp at h
  | fuel + 1, st, r, h => by
    simp only [loop]
    split
    · rename_i e he; intro hh; cases hh; exact (good_readOne utf8).2.2 r he
    · rename_i n r' hr
      obtain ⟨p, rfl, hp⟩ := readOne_progress hr
      exact loop_fuel_enough utf8 fuel _ r' (by simp at h; omega)
    · rename_i v r' hr
      obtain ⟨p, rfl, hp⟩ := readOne_progress hr
      split
      · simp
      · rename_i st' _; exact loop_fuel_enough utf8 fuel st' r' (by simp at h; omega)
      · simp

/-- `loop` only consumes. -/
theorem loop_consumes (utf8 : Bytes → Bool) :
    ∀ fuel st r v r', loop utf8 fuel st r = .ok (v, r') → ∃ pre, r = pre ++ r'
  | 0, _, _, _, _, h => by simp [loop] at h
  | fuel + 1, st, r, v, r', h => by
    simp only [loop] at h
    split at h
    · cases h
    · rename_i n r1 hr
      obtain ⟨p, rfl, _⟩ := readOne_progress hr
      obtain ⟨q, rfl⟩ := loop_consumes utf8 fuel _ _ _ _ h
      exact ⟨p ++ q, by simp⟩
    · rename_i w r1 hr
      obtain ⟨p, rfl, _⟩ := readOne_progress hr
      split at h
      · cases h; exact ⟨p, rfl⟩
      · obtain ⟨q, rfl⟩ := loop_consumes utf8 fuel _ _ _ _ h
        exact ⟨p ++ q, by simp⟩
      · cases h

/-- **`Value::decode` on hostile bytes**: whatever `r` is, it neither panics nor
over-reads, and the modelling fuel is never the reason it stops. -/
theorem decodeV_safe (utf8 : Bytes → Bool) (r : Bytes) :
    decodeV utf8 r ≠ .error .panic ∧ decodeV utf8 r ≠ .error .outOfFuel ∧
    ∀ v r', decodeV utf8 r = .ok (v, r') → ∃ pre, r = pre ++ r' :=
  ⟨loop_noPanic utf8 _ [] r (by simp [FramesOk]),
   loop_fuel_enough utf8 _ [] r (by omega),
   fun v r' h => loop_consumes utf8 _ [] r v r' h⟩

/-! ## What the primary can hand the encoder -/

-- `pending::is_valid_property` (`graph/src/runtime/pending.rs:63-81`).
mutual
def propValid (allowNull : Bool) : Val → Bool
  | .null => allowNull
  | .bool _ | .int _ | .float _ | .str _ | .point _ _ | .vecf32 _
  | .datetime _ | .date _ | .time _ | .duration _ => true
  | .list xs => propValidL xs
  | _ => false
def propValidL : List Val → Bool
  | [] => true
  | v :: vs => propValid false v && propValidL vs
end

-- Container lengths fit the `u32` counts (`value.rs:70`, `value.rs:93`).
mutual
def Fits : Val → Prop
  | .list xs => xs.length < 2 ^ 32 ∧ FitsL xs
  | .vecf32 xs => xs.length < 2 ^ 32
  | _ => True
def FitsL : List Val → Prop
  | [] => True
  | v :: vs => Fits v ∧ FitsL vs
end

mutual
/-- A property value the engine accepts never reaches the `panic!` of
`value.rs:121`: maps, nodes, edges and paths are refused where the value is
stored, which is the only reason that arm is unreachable. -/
theorem encV_total (I : Bytes → Bool) : ∀ (v : Val) (a : Bool), propValid a v = true → Fits v →
    ∃ bs, encV I v = .ok bs
  | .null, _, _, _ => ⟨_, rfl⟩
  | .bool _, _, _, _ => ⟨_, rfl⟩
  | .int _, _, _, _ => ⟨_, rfl⟩
  | .float _, _, _, _ => ⟨_, rfl⟩
  | .str _, _, _, _ => ⟨_, rfl⟩
  | .point _ _, _, _, _ => ⟨_, rfl⟩
  | .datetime _, _, _, _ => ⟨_, rfl⟩
  | .date _, _, _, _ => ⟨_, rfl⟩
  | .time _, _, _, _ => ⟨_, rfl⟩
  | .duration _, _, _, _ => ⟨_, rfl⟩
  | .vecf32 xs, _, _, hf => by simp only [Fits] at hf; simp [encV, hf]
  | .map _, _, h, _ => by simp [propValid] at h
  | .node _, _, h, _ => by simp [propValid] at h
  | .rel _, _, h, _ => by simp [propValid] at h
  | .path _, _, h, _ => by simp [propValid] at h
  | .list xs, _, h, hf => by
    simp only [propValid] at h
    simp only [Fits] at hf
    obtain ⟨b, hb⟩ := encL_total I xs h hf.2
    exact ⟨w32 T_ARRAY ++ w32 xs.length ++ b, by simp [encV, hf.1, hb]⟩
theorem encL_total (I : Bytes → Bool) : ∀ (xs : List Val), propValidL xs = true → FitsL xs →
    ∃ bs, encL I xs = .ok bs
  | [], _, _ => ⟨[], rfl⟩
  | v :: vs, h, hf => by
    simp only [propValidL, Bool.and_eq_true] at h
    simp only [FitsL] at hf
    obtain ⟨a, ha⟩ := encV_total I v false h.1 hf.1
    obtain ⟨b, hb⟩ := encL_total I vs h.2 hf.2
    exact ⟨a ++ b, by simp [encL, ha, hb]⟩
end

/-! ## Depth and allocation on hostile input -/

/-- `[[[…[null]…]]]`, `d` deep. -/
def nest : Nat → Val
  | 0 => .null
  | d + 1 => .list [nest d]

theorem encV_nest (I : Bytes → Bool) : ∀ d, encV I (nest d) =
    .ok ((List.replicate d (w32 T_ARRAY ++ w32 1)).flatten ++ w32 T_NULL)
  | 0 => rfl
  | d + 1 => by
    simp only [nest, encV, List.length_singleton, encL, encV_nest I d]
    simp [List.replicate_succ]

theorem nest_wf (utf8 : Bytes → Bool) : ∀ d, WF utf8 (nest d)
  | 0 => trivial
  | d + 1 => by simp only [nest, WF, WFL]; exact ⟨nest_wf utf8 d, trivial⟩

/-- **Depth is unbounded and costs eight bytes a level**: the decoder returns
a value `d` deep from `8d + 4` bytes, for every `d`. The decode itself is
iterative (`value.rs:137-160`), but `Value` is recursive and its `Drop` is not,
so this is the input that overflows the stack *after* a successful decode —
see `REPORT.md`, `hostile_deep_nesting_*`. -/
theorem nest_bytes_length : ∀ d,
    ((List.replicate d (w32 T_ARRAY ++ w32 1)).flatten ++ w32 T_NULL).length = 8 * d + 4
  | 0 => rfl
  | d + 1 => by
    have := nest_bytes_length d
    simp only [List.replicate_succ, List.flatten_cons, List.length_append] at this ⊢
    simp only [w32, le_length] at this ⊢
    omega

theorem nest_decodes (utf8 : Bytes → Bool) (d : Nat) :
    ∃ bs, bs.length = 8 * d + 4 ∧ decodeV utf8 bs = .ok (nest d, []) := by
  refine ⟨_, nest_bytes_length d, ?_⟩
  have := decodeV_encV utf8 (fun _ => false) (nest d) _ [] (encV_nest _ d) (nest_wf utf8 d)
  simpa using this

/-- The capacities `loop` asks for, turn by turn: `n` for every
`Frame::List` it pushes (`ThinVec::with_capacity(n)`, `value.rs:227`). A
mirror of `loop` that records instead of building, for the `#guard`s below. -/
def capTrace (utf8 : Bytes → Bool) : Nat → List Frame → Bytes → List Nat
  | 0, _, _ => []
  | fuel + 1, st, r =>
    match readOne utf8 r with
    | .error _ => []
    | .ok (.opened n, r') => n :: capTrace utf8 fuel (⟨[], n⟩ :: st) r'
    | .ok (.value v, r') =>
      match deliver v st with
      | .more st' => capTrace utf8 fuel st' r'
      | _ => []

/-- `k` nested arrays, each claiming as many children as the bytes after it
could hold at four bytes each — the most `guard_count` lets through. -/
def hostile (k : Nat) : Bytes :=
  (List.range k).flatMap fun i => w32 T_ARRAY ++ w32 (2 * (k - i - 1))

#guard (hostile 100).length = 800

def hostileCaps (k : Nat) : Nat := (capTrace (fun _ => true) (8 * k + 1) [] (hostile k)).sum

-- …so the *sum* of the capacities is quadratic in the input:
-- `k (k - 1)` elements from `8 k` bytes. At 16 bytes per `Value` (the Rust
-- `size_of`), 8 KB of input reserves ~16 MB, 1 MB reserves ~256 GB.
#guard hostileCaps 10 = 90
#guard hostileCaps 100 = 9900
#guard hostileCaps 1000 = 999000
-- The decode still fails — the claims are never met — but the reservations
-- have all been made by then.
#guard match decodeV (fun _ => true) (hostile 100) with | .error (.eof _ _) => true | _ => false

-- Bit-exact floats: NaN payloads and signed zero go through untouched, because
-- a float *is* its bit pattern on this wire.
def floatBack (bits : Nat) : Option Nat :=
  match decodeV (fun _ => true) (w32 T_DOUBLE ++ w64 bits) with
  | .ok (.float b, []) => some b.toNat
  | _ => none

#guard floatBack 0x7ff4000000000001 = some 0x7ff4000000000001  -- signalling NaN, payload 1
#guard floatBack 0xfff8000000000000 = some 0xfff8000000000000  -- negative quiet NaN
#guard floatBack 0x8000000000000000 = some 0x8000000000000000  -- -0.0

end FalkorCodec
