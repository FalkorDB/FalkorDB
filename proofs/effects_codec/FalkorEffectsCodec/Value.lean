import FalkorEffectsCodec.Prim
/-
# The `SIValue` codec: `graph/src/effects/v3/value.rs`

| here | there |
| --- | --- |
| `Val`              | `runtime::value::Value` (`graph/src/runtime/value.rs`) — floats as their bit patterns, `i64` as its two's-complement bits; a duration as `Dur::components()` |
| `T_*`              | `serialization::si_type` (`graph/src/graph/graphblas/serialization.rs`) |
| `encV` / `encL`    | `impl EffectEncode<4> for Value` (`value.rs`) |
| `daysFromCivil`    | `temporal::days_from_civil` (`graph/src/runtime/functions/temporal.rs`) |
| `durNewOk`         | the two checks of `Dur::new` (`temporal.rs`): `Ok` iff this is `true` |
| `civilFromDays`    | `runtime::value::civil_from_days` (`value.rs`, `civil_from_days_wide`) |
| `durationToLegacy` | `temporal::duration_to_legacy` (`temporal.rs`) |
| `durationFromLegacy` | `temporal::duration_from_legacy` (`temporal.rs`) |
| `Step`             | `read_one`'s `Option<Value>`: `None` = a frame was pushed |
| `readOne`          | `read_one` (`value.rs`); `lg` is `legacy_duration`, `owed` the `&mut usize` |
| `Frame`            | `enum Frame::List { items, left }` (`value.rs`) |
| `deliver`          | the inner `loop` of `decode_value` (`value.rs`) |
| `loop`             | the outer `loop` of `decode_value` (`value.rs`), with fuel |
| `owedOf`           | the invariant the `owed` comment states: the sum of `left - 1` over the stack |
| `decodeV`          | `decode_value` (`value.rs`) |
| `decodeV3` / `decodeV4` | `impl EffectDecode<EFFECTS_VERSION_V3>` (`value.rs`) / `<EFFECTS_VERSION>` (`value.rs`) |
| `propValid`        | `pending::is_valid_property` (`graph/src/runtime/pending.rs`) |

**#3610 (durations as components).** `Value::Duration` holds a `Dur`: three signed
components (months, days, seconds). v4 writes `T_DURATION` then the three `i64`s and
reads them back through `Dur::new` (`Err` → `BadDuration`); v3 (one legacy epoch
offset) is still read, through `duration_from_legacy`. `decode_value` takes the flag
`legacy_duration` (`lg` here), which only the `T_DURATION` arm reads. Proven:
`decodeV_encV` (any value, durations included, whose components `Dur::new` accepts —
every `Dur` comes out of `Dur::new` — roundtrips in v4; in v3 every duration-free value
does), `readOne_v3_duration` (v3 reads exactly 8 bytes and yields a `Dur::new`-valid
duration whose legacy offset is the stored one), `durationFromLegacy_some`, and all
the hostile-input theorems for both versions. A v4 duration is 28 bytes, still ≥
`MIN_VALUE_BYTES`, so `encV_len` and the linear-reservation bound are unchanged.

**#3267 (`eaa65eeba`).** `read_one`'s array arm now checks the claimed width against
the whole stack, `guard_count(n + owed, MIN_VALUE_BYTES)`, where `owed` counts the
children the open frames still need after this one. So `capTrace_linear`: the
capacities one decode reserves sum to at most `remaining / 4`.

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

/-- `si_type` constants (`serialization.rs`). -/
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
pairwise distinct, and each fits the 4-byte wire (`mod.rs` checks the
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
  /-- `Dur::components()`: months, days, seconds. -/
  | duration (m d s : BitVec 64)
deriving Repr, Inhabited

/-- Encode errors: `EncodeError::BlockCountTooLarge` (`error.rs`) and the
`panic!` of `value.rs`. -/
inductive EErr where
  | blockCountTooLarge
  | headerShapeMismatch
  | endpointMisaligned
  | optionsFieldTypeMismatch
  | badFieldType
  | emptyIndexSchemaList
  | idList
  | panic
  /-- `EncodeError::RowShapeMismatch` (`error.rs`). -/
  | rowShapeMismatch
  /-- `EncodeError::EmptyRecord` (`error.rs`). -/
  | emptyRecord
deriving Repr, DecidableEq

def f32s (xs : List (BitVec 32)) : Bytes := xs.flatMap (fun f => w32 f.toNat)

-- `EffectEncode<4> for Value` (`value.rs`). `interned` is
-- `string_pool::global().is_interned` (`value.rs`), which only chooses the tag.
mutual
def encV (interned : Bytes → Bool) : Val → Except EErr Bytes
  | .null => .ok (w32 T_NULL)
  | .bool b => .ok (w32 T_BOOL ++ w8 (if b then 1 else 0))
  | .int x => .ok (w32 T_INT64 ++ w64 x.toNat)
  | .float f => .ok (w32 T_DOUBLE ++ w64 f.toNat)
  | .str s => .ok (w32 (if interned s then T_ISTRING else T_STRING) ++ wString s)
  | .list xs =>
    -- `u32::try_from(items.len())` (`value.rs`)
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
  -- `let (months, days, seconds) = d.components()`, the tag, then three `i64`s (#3610)
  | .duration m d s => .ok (w32 T_DURATION ++ w64 m.toNat ++ w64 d.toNat ++ w64 s.toNat)
  -- `other => panic!("value cannot appear in an effect")` (`value.rs`)
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

/-- `for _ in 0..n { v.push(r.f32()?) }` (`value.rs`). -/
def readN : Nat → R α → R (List α)
  | 0, _ => pure []
  | n + 1, m => do let a ← m; let as ← readN n m; pure (a :: as)

/-! ## Durations (#3610): `Dur::new`, `duration_to_legacy`, `duration_from_legacy` -/

def fitsI32 (x : Int) : Bool := decide (-2 ^ 31 ≤ x ∧ x < 2 ^ 31)
def fitsI64 (x : Int) : Bool := decide (-2 ^ 63 ≤ x ∧ x < 2 ^ 63)

/-- `days_from_civil` (`temporal.rs`), in exact arithmetic. Rust computes it in
`i32`/`u32`; nothing there wraps for the years a `Dur` can reach
(`|1970 + months.div_euclid(12)| < 2^28` once `months` fits `i32`). `/` on the
signed `y` is Rust's truncating division (`Int.tdiv`); the rest is on `u32`. -/
def daysFromCivil (y : Int) (m d : Nat) : Int :=
  let y := if m ≤ 2 then y - 1 else y
  let era := (if y ≥ 0 then y else y - 399).tdiv 400
  let yoe := (y - era * 400).toNat
  let mp := if m > 2 then m - 3 else m + 9
  let doy := (153 * mp + 2) / 5 + d - 1
  let doe := yoe * 365 + yoe / 4 - yoe / 100 + doy
  era * 146097 + doe - 719468

/-- `Dur::new(months, days, seconds)` (`temporal.rs`) is `Ok` iff this holds:
`i32::try_from(months)`, then the legacy offset
`days_from_civil(1970 + months.div_euclid(12), months.rem_euclid(12) + 1, 1) * 86400
+ days * 86400 + seconds`, computed in `i128`, fits `i64`. (`div_euclid`/`rem_euclid`
by 12 are Lean's `/`/`%` on `Int`.) Which of the two representations it picks is
invisible: every reader goes through `components()`. -/
def durNewOk (m d s : BitVec 64) : Bool :=
  fitsI32 m.toInt &&
    fitsI64 (daysFromCivil (1970 + m.toInt / 12) (m.toInt % 12 + 1).toNat 1 * 86400 +
      d.toInt * 86400 + s.toInt)

/-- `x as i32` (truncating). -/
def wrapI32 (x : Int) : Int := (BitVec.ofInt 32 x).toInt
/-- `i32::saturating_add` of two `i32`s. -/
def satI32 (x : Int) : Int := if x < -2 ^ 31 then -2 ^ 31 else if x ≥ 2 ^ 31 then 2 ^ 31 - 1 else x

/-- `duration_to_legacy` (`temporal.rs`): `wrapping_mul`/`wrapping_add` in `i64`
are the exact sum reduced to 64 bits (`BitVec.ofInt`). -/
def durationToLegacy (m d s : BitVec 64) : BitVec 64 :=
  BitVec.ofInt 64 (daysFromCivil (satI32 (1970 + wrapI32 (m.toInt / 12))) (m.toInt % 12 + 1).toNat 1 * 86400 +
    d.toInt * 86400 + s.toInt)

/-- `civil_from_days` (`value.rs`) = `civil_from_days_wide` (`value.rs`) with
the year cast `as i32`. Every division is of a non-negative number by a positive one
(or `div_euclid`/`rem_euclid`), so Lean's `/`/`%` are Rust's. -/
def civilFromDays (z : Int) : Int × Int × Int :=
  let z := z + 719468
  let era := z / 146097
  let doe := z % 146097
  let yoe := (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
  let y := yoe + era * 400
  let doy := doe - (365 * yoe + yoe / 4 - yoe / 100)
  let mp := (5 * doy + 2) / 153
  let d := doy - (153 * mp + 2) / 5 + 1
  let m := if mp < 10 then mp + 3 else mp - 9
  let y := if m ≤ 2 then y + 1 else y
  (wrapI32 y, m, d)

/-- `duration_from_legacy` (`temporal.rs`), `Err` as `none`: decompose the offset
the way the C engine did; keep that only if `Dur::new` accepts it **and**
`duration_to_legacy` gives the offset back; else `Dur::new(0, 0, legacy)`, else
`Dur::new(0, days, time_of_day)` (`durationFromLegacy_some`: never reached). -/
def durationFromLegacy (legacy : BitVec 64) : Option (BitVec 64 × BitVec 64 × BitVec 64) :=
  let days := legacy.toInt / 86400
  let tod := legacy.toInt % 86400
  let (y, m, d) := civilFromDays days
  let months := BitVec.ofInt 64 ((y - 1970) * 12 + m - 1)
  let remaining := (d - 1) * 86400 + tod
  let cand := (months, BitVec.ofInt 64 (remaining.tdiv 86400), BitVec.ofInt 64 (remaining.tmod 86400))
  if durNewOk cand.1 cand.2.1 cand.2.2 && durationToLegacy cand.1 cand.2.1 cand.2.2 == legacy then some cand
  else if durNewOk 0 0 legacy then some (0, 0, legacy)
  else if durNewOk 0 (BitVec.ofInt 64 days) (BitVec.ofInt 64 tod) then
    some (0, BitVec.ofInt 64 days, BitVec.ofInt 64 tod)
  else none

theorem daysFromCivil_epoch : daysFromCivil 1970 1 1 = 0 := by decide

theorem fitsI64_toInt (x : BitVec 64) : fitsI64 x.toInt = true := by
  have h1 := BitVec.le_toInt x
  have h2 := @BitVec.toInt_lt 64 x
  simp [fitsI64] at *
  omega

theorem durNewOk_zero (x : BitVec 64) : durNewOk 0 0 x = true := by
  have h : daysFromCivil (1970 + (0 : BitVec 64).toInt / 12) ((0 : BitVec 64).toInt % 12 + 1).toNat 1 = 0 :=
    by decide
  simp only [durNewOk, h]
  have := fitsI64_toInt x
  simp_all [fitsI32]

theorem durationToLegacy_zero (x : BitVec 64) : durationToLegacy 0 0 x = x := by
  have h : daysFromCivil (satI32 (1970 + wrapI32 ((0 : BitVec 64).toInt / 12)))
      ((0 : BitVec 64).toInt % 12 + 1).toNat 1 = 0 := by decide
  simp only [durationToLegacy, h]
  simp

/-- **`duration_from_legacy` never fails, and is a right inverse of
`duration_to_legacy`**: whatever the stored offset, the duration it yields is one
`Dur::new` accepts and whose legacy offset is exactly the stored one. -/
theorem durationFromLegacy_some (x : BitVec 64) :
    ∃ m d s, durationFromLegacy x = some (m, d, s) ∧ durNewOk m d s = true ∧
      durationToLegacy m d s = x := by
  unfold durationFromLegacy
  simp only
  split
  · rename_i h
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    exact ⟨_, _, _, rfl, h.1, h.2⟩
  · simp only [durNewOk_zero x, ite_true]
    exact ⟨0, 0, x, rfl, durNewOk_zero x, durationToLegacy_zero x⟩

/-- The smallest a value can encode to (`value.rs`). -/
def MIN_VALUE_BYTES : Nat := 4

/-- `read_one` (`value.rs`), arm for arm and in the same order. `owed` is
read only by the array arm; its update `*owed += n - 1` is done by `loop` on
`.opened n`. `n + owed` cannot overflow the Rust `u64`: `n < 2^32`, and `owed` only
grows past a guard that bounds it by `remaining / 4`. -/
def readOne (utf8 : Bytes → Bool) (lg : Bool) (owed : Nat) : R Step := do
  let t ← u32
  if t = T_NULL then pure (.value .null)
  else if t = T_BOOL then do let b ← u8; pure (.value (.bool (b ≠ 0)))
  else if t = T_INT64 then do let x ← i64; pure (.value (.int x))
  else if t = T_DOUBLE then do let x ← i64; pure (.value (.float x))
  else if t = T_STRING ∨ t = T_ISTRING then do
    let s ← rString utf8; pure (.value (.str s))
  else if t = T_ARRAY then do
    let n ← u32
    -- `r.guard_count(n + *owed as u64, MIN_VALUE_BYTES)?` — the result is dropped
    let _ ← guardCount (n + owed) MIN_VALUE_BYTES
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
  -- `T_DURATION if legacy_duration`: one offset through `duration_from_legacy` (v3)
  else if t = T_DURATION ∧ lg = true then do
    let x ← i64
    match durationFromLegacy x with
    | some (m, d, s) => pure (.value (.duration m d s))
    | none => fail .badDuration
  -- `T_DURATION`: three components through `Dur::new` (v4)
  else if t = T_DURATION then do
    let m ← i64; let d ← i64; let s ← i64
    if durNewOk m d s then pure (.value (.duration m d s)) else fail .badDuration
  else fail (.badValueType t)

/-- `Frame::List { items, left }`. `items` is pushed at the end, as `ThinVec::push`. -/
structure Frame where
  items : List Val
  left : Nat

inductive Deliv where
  | done (v : Val)
  | more (st : List Frame)
  | panic

/-- The inner loop of `decode_value` (`value.rs`). `*left -= 1` on `left = 0` is an
overflow panic in debug (and a wrap in release); it is modelled as `panic`
and `loop_noPanic` proves it cannot happen. -/
def deliver : Val → List Frame → Deliv
  | v, [] => .done v
  | v, f :: st =>
    if f.left = 0 then .panic
    else if f.left - 1 > 0 then .more (⟨f.items ++ [v], f.left - 1⟩ :: st)
    else deliver (.list (f.items ++ [v])) st

/-- The outer loop of `decode_value` (`value.rs`). `owed` is its counter: `read_one`
adds `n - 1` when it opens a frame of `n`, and the inner loop subtracts one when
a frame takes a child and still waits for more (`owed -= 1`, `value.rs`) — an
underflow there would be a debug panic, modelled as `panic`, and `loop_noPanic`
proves it cannot happen. -/
def loop (utf8 : Bytes → Bool) (lg : Bool) : Nat → List Frame → Nat → R Val
  | 0, _, _ => fun _ => .error .outOfFuel
  | fuel + 1, st, owed => fun r =>
    match readOne utf8 lg owed r with
    | .error e => .error e
    | .ok (.opened n, r') => loop utf8 lg fuel (⟨[], n⟩ :: st) (owed + (n - 1)) r'
    | .ok (.value v, r') =>
      match deliver v st with
      | .done v => .ok (v, r')
      | .more st' => if owed = 0 then .error .panic else loop utf8 lg fuel st' (owed - 1) r'
      | .panic => .error .panic

/-- `decode_value(r, legacy_duration)` (`value.rs`). -/
def decodeV (utf8 : Bytes → Bool) (lg : Bool) : R Val := fun r => loop utf8 lg (r.length + 1) [] 0 r

/-- `impl EffectDecode<EFFECTS_VERSION_V3> for Value` (`value.rs`): `decode_value(r, true)`. -/
def decodeV3 (utf8 : Bytes → Bool) : R Val := decodeV utf8 true
/-- `impl EffectDecode<EFFECTS_VERSION> for Value` (`value.rs`): `decode_value(r, false)`. -/
def decodeV4 (utf8 : Bytes → Bool) : R Val := decodeV utf8 false

/-- What `owed` is meant to be: the children the open frames still need beyond
the one being read now. -/
def owedOf (st : List Frame) : Nat := (st.map (fun f => f.left - 1)).sum

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
-- `u64` (they do: they are in memory), bodies are UTF-8 (they are: they are
-- Rust `String`s), and a duration's components are ones `Dur::new` accepts (they
-- are: `Dur::new` is the only constructor of a `Dur` from components). Read as v3
-- (`lg = true`), the encoder's v4 output only round-trips without durations: v4
-- changed exactly that tag's layout.
mutual
def WF (utf8 : Bytes → Bool) (lg : Bool) : Val → Prop
  | .str s => s.length + 1 < 2 ^ 64 ∧ utf8 s = true
  | .list xs => WFL utf8 lg xs
  | .duration m d s => lg = false ∧ durNewOk m d s = true
  | _ => True
def WFL (utf8 : Bytes → Bool) (lg : Bool) : List Val → Prop
  | [] => True
  | v :: vs => WF utf8 lg v ∧ WFL utf8 lg vs
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
  | .duration m d s, bs, h => by simp [encV] at h; subst h; simp [cnt, w32]
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
    simp only
    rw [show List.flatMap (fun f => w32 f.toNat) xs = f32s xs from rfl, ih]
    rfl

/-- What the outer loop does with a finished value, by the inner loop's verdict. -/
def cont (utf8 : Bytes → Bool) (lg : Bool) : Deliv → Nat → Nat → R Val
  | .done v, _, _ => fun r => .ok (v, r)
  | .more st, fuel, owed => fun r => if owed = 0 then .error .panic else loop utf8 lg fuel st (owed - 1) r
  | .panic, _, _ => fun _ => .error .panic

theorem loop_value {utf8 lg fuel st owed r v r'} (h : readOne utf8 lg owed r = .ok (.value v, r')) :
    loop utf8 lg (fuel + 1) st owed r = cont utf8 lg (deliver v st) fuel owed r' := by
  simp only [loop, h]
  cases deliver v st <;> rfl

theorem loop_opened {utf8 lg fuel st owed r n r'} (h : readOne utf8 lg owed r = .ok (.opened n, r')) :
    loop utf8 lg (fuel + 1) st owed r = loop utf8 lg fuel (⟨[], n⟩ :: st) (owed + (n - 1)) r' := by
  simp only [loop, h]

attribute [local simp] u32_w32 u8_w8 i64_w64 f32_w32

/-- Every scalar arm of `readOne` reads back exactly what `encV` wrote. -/
theorem readOne_scalar (utf8 : Bytes → Bool) (lg : Bool) (I : Bytes → Bool) (owed : Nat) (v : Val) (bs rest : Bytes)
    (h : encV I v = .ok bs) (hw : WF utf8 lg v) (hl : ∀ xs, v ≠ .list xs) :
    readOne utf8 lg owed (bs ++ rest) = .ok (.value v, rest) := by
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
  | duration m d s =>
    simp [encV] at h; subst h
    simp only [WF] at hw
    obtain ⟨rfl, hd⟩ := hw
    simp [readOne, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_ISTRING, T_ARRAY, T_MAP,
      T_POINT, T_VECTOR_F32, T_DATETIME, T_DATE, T_TIME, T_DURATION, hd]

/-- The list arm: an empty list is a value; a non-empty one opens a frame of
exactly its length and leaves the elements unread. The guard passes because the
children's bytes (`b`) and the `owed` siblings' bytes (in `rest`) are disjoint. -/
theorem readOne_list (utf8 : Bytes → Bool) (lg : Bool) (owed : Nat) (xs : List Val) (b rest : Bytes)
    (hlt : xs.length < 2 ^ 32) (hb : 4 * xs.length ≤ b.length) (hr : 4 * owed ≤ rest.length) :
    readOne utf8 lg owed (w32 T_ARRAY ++ w32 xs.length ++ b ++ rest) =
      if xs.length = 0 then .ok (.value (.list []), b ++ rest)
      else .ok (.opened xs.length, b ++ rest) := by
  have hg : guardCount (xs.length + owed) MIN_VALUE_BYTES (b ++ rest) =
      .ok (xs.length + owed, b ++ rest) := by
    simp [guardCount, MIN_VALUE_BYTES]; omega
  simp only [List.append_assoc]
  simp [readOne, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_ISTRING, T_ARRAY, hlt, hg]
  split <;> rfl

theorem scalar_step (utf8 : Bytes → Bool) (lg : Bool) (I : Bytes → Bool) (v : Val) (bs : Bytes) (h : encV I v = .ok bs)
    (hw : WF utf8 lg v) (hl : ∀ xs, v ≠ .list xs) (fuel : Nat) (st : List Frame) (owed : Nat)
    (rest : Bytes) (hf : cnt v ≤ fuel) :
    loop utf8 lg fuel st owed (bs ++ rest) = cont utf8 lg (deliver v st) (fuel - cnt v) owed rest := by
  have hc : cnt v = 1 := by cases v <;> simp [cnt] <;> exact absurd rfl (hl _)
  obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
  rw [loop_value (readOne_scalar utf8 lg I owed v bs rest h hw hl), hc]
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
middle of.** Decoding `encV v` on top of any stack of open frames, with `owed`
siblings still to come in `rest`, consumes exactly those bytes and hands exactly
`v` to the innermost frame. -/
theorem loop_enc (utf8 : Bytes → Bool) (lg : Bool) (I : Bytes → Bool) : ∀ (v : Val) (bs : Bytes), encV I v = .ok bs →
    WF utf8 lg v → ∀ (fuel : Nat) (st : List Frame) (owed : Nat) (rest : Bytes), cnt v ≤ fuel →
    4 * owed ≤ rest.length →
    loop utf8 lg fuel st owed (bs ++ rest) = cont utf8 lg (deliver v st) (fuel - cnt v) owed rest
  | .list xs, bs, h, hw, fuel, st, owed, rest, hf, ho => by
    obtain ⟨hlt, b, hb, rfl⟩ := encV_list h
    have hlen := (encL_len I xs b hb).2
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp [cnt] at hf; omega⟩
    have hr := readOne_list utf8 lg owed xs b rest hlt hlen ho
    cases xs with
    | nil =>
      simp only [List.length_nil, ite_true] at hr ⊢
      rw [loop_value hr, encL_nil hb]
      simp [cnt, cntL]
    | cons x xs' =>
      simp only [List.length_cons, Nat.add_one_ne_zero, ite_false] at hr ⊢
      rw [loop_opened hr]
      have := loopL_enc utf8 lg I (x :: xs') b hb hw (by simp) [] f st owed rest
        (by simp only [cnt] at hf; omega) ho
      simp only [List.length_cons] at this
      rw [this]
      simp only [List.nil_append, cnt]
      congr 1
      omega
  | .null, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .bool _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .int _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .float _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .str _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .map _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .node _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .rel _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .path _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .vecf32 _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .point _ _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .datetime _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .date _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .time _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf
  | .duration _ _ _, bs, h, hw, fuel, st, owed, rest, hf, _ =>
    scalar_step utf8 lg I _ bs h hw (by intro _ h; cases h) fuel st owed rest hf

/-- The frame half: the children of a non-empty list, read on top of the frame
that is waiting for them (which owes `xs.length - 1` beyond the one being read),
close that frame into exactly the list. -/
theorem loopL_enc (utf8 : Bytes → Bool) (lg : Bool) (I : Bytes → Bool) : ∀ (xs : List Val) (bs : Bytes), encL I xs = .ok bs →
    WFL utf8 lg xs → xs ≠ [] → ∀ (items : List Val) (fuel : Nat) (st : List Frame) (owed : Nat)
    (rest : Bytes), cntL xs ≤ fuel → 4 * owed ≤ rest.length →
    loop utf8 lg fuel (⟨items, xs.length⟩ :: st) (owed + (xs.length - 1)) (bs ++ rest) =
      cont utf8 lg (deliver (.list (items ++ xs)) st) (fuel - cntL xs) owed rest
  | [], _, _, _, hne, _, _, _, _, _, _, _ => absurd rfl hne
  | x :: xs', bs, h, hw, _, items, fuel, st, owed, rest, hf, ho => by
    obtain ⟨a, b', ha, hb', rfl⟩ := encL_cons h
    simp only [WFL] at hw
    simp only [cntL] at hf
    have hb'l := (encL_len I xs' b' hb').2
    rw [List.append_assoc, List.length_cons, Nat.add_sub_cancel,
      loop_enc utf8 lg I x a ha hw.1 fuel _ (owed + xs'.length) (b' ++ rest) (by omega)
        (by simp only [List.length_append]; omega)]
    simp only [deliver, Nat.add_one_ne_zero, ite_false, Nat.add_sub_cancel]
    cases xs' with
    | nil =>
      rw [encL_nil hb']
      simp [cntL]
    | cons y ys =>
      simp only [List.length_cons, Nat.zero_lt_succ, ite_true, cont, gt_iff_lt]
      simp only [show ¬(owed + (ys.length + 1) = 0) by omega, ↓reduceIte]
      have := loopL_enc utf8 lg I (y :: ys) b' hb' hw.2 (by simp) (items ++ [x]) (fuel - cnt x) st
        owed rest (by simp only [cntL] at hf ⊢; omega) ho
      simp only [List.length_cons, Nat.add_sub_cancel] at this
      rw [show owed + (ys.length + 1) - 1 = owed + ys.length by omega, this]
      simp only [List.append_assoc, List.singleton_append, cntL]
      congr 1
      omega
end

/-- **`read(write(v)) = v`**, for every value `encV` accepts: the decoder
consumes exactly the encoding, returns exactly `v` — bit-for-bit, since every
float is its bit pattern — and leaves `rest` untouched, so the next field
parses from the right offset. Still true with the whole-stack guard: an honest
encoding always has the bytes every open frame owes. -/
theorem decodeV_encV (utf8 : Bytes → Bool) (lg : Bool) (I : Bytes → Bool) (v : Val) (bs rest : Bytes)
    (h : encV I v = .ok bs) (hw : WF utf8 lg v) :
    decodeV utf8 lg (bs ++ rest) = .ok (v, rest) := by
  have hl := encV_len I v bs h
  unfold decodeV
  rw [loop_enc utf8 lg I v bs h hw _ [] 0 rest (by simp; omega) (by simp)]
  simp [deliver, cont]

/-- The two impls: v4 (`value.rs`) is the encoder's own version, so every
value the encoder writes — a duration as its three components — reads back. -/
theorem decodeV4_encV (utf8 I : Bytes → Bool) (v : Val) (bs rest : Bytes)
    (h : encV I v = .ok bs) (hw : WF utf8 false v) :
    decodeV4 utf8 (bs ++ rest) = .ok (v, rest) :=
  decodeV_encV utf8 false I v bs rest h hw

/-- v3 (`value.rs`) reads the same bytes for every value that is not a
duration: v3 and v4 differ in that tag alone. -/
theorem decodeV3_encV (utf8 I : Bytes → Bool) (v : Val) (bs rest : Bytes)
    (h : encV I v = .ok bs) (hw : WF utf8 true v) :
    decodeV3 utf8 (bs ++ rest) = .ok (v, rest) :=
  decodeV_encV utf8 true I v bs rest h hw

/-- **A v4 duration is the tag and three components: 28 bytes.** -/
theorem encV_duration (I : Bytes → Bool) (m d s : BitVec 64) :
    encV I (.duration m d s) = .ok (w32 T_DURATION ++ w64 m.toNat ++ w64 d.toNat ++ w64 s.toNat) ∧
    (w32 T_DURATION ++ w64 m.toNat ++ w64 d.toNat ++ w64 s.toNat).length = 28 := by
  simp [encV, w32, w64]

/-- **v3's `T_DURATION`** (`value.rs`): exactly the 8 bytes of one `i64` are
read, and the duration is one `Dur::new` accepts whose `duration_to_legacy` is the
stored offset — `duration_from_legacy` never fails, so this arm never refuses. -/
theorem readOne_v3_duration (utf8 : Bytes → Bool) (owed : Nat) (x : BitVec 64) (rest : Bytes) :
    ∃ m d s, readOne utf8 true owed (w32 T_DURATION ++ w64 x.toNat ++ rest) =
        .ok (.value (.duration m d s), rest) ∧
      durNewOk m d s = true ∧ durationToLegacy m d s = x := by
  obtain ⟨m, d, s, h1, h2, h3⟩ := durationFromLegacy_some x
  refine ⟨m, d, s, ?_, h2, h3⟩
  simp [readOne, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_ISTRING, T_ARRAY, T_MAP,
    T_POINT, T_VECTOR_F32, T_DATETIME, T_DATE, T_TIME, T_DURATION, h1]

/-- The same as a whole `Value::decode` in v3. -/
theorem decodeV3_duration (utf8 : Bytes → Bool) (x : BitVec 64) (rest : Bytes) :
    ∃ m d s, decodeV3 utf8 (w32 T_DURATION ++ w64 x.toNat ++ rest) = .ok (.duration m d s, rest) ∧
      durNewOk m d s = true ∧ durationToLegacy m d s = x := by
  obtain ⟨m, d, s, h1, h2, h3⟩ := readOne_v3_duration utf8 0 x rest
  refine ⟨m, d, s, ?_, h2, h3⟩
  unfold decodeV3 decodeV
  rw [show (w32 T_DURATION ++ w64 x.toNat ++ rest).length + 1 =
    (w32 T_DURATION ++ w64 x.toNat ++ rest).length + 1 from rfl]
  generalize hL : (w32 T_DURATION ++ w64 x.toNat ++ rest).length = L
  rw [loop_value h1]
  simp [deliver, cont]

/-- **v4 refuses what `Dur::new` refuses** (`Err(msg)` → `BadDuration(msg)`), and
accepts exactly the rest. -/
theorem readOne_v4_duration (utf8 : Bytes → Bool) (owed : Nat) (m d s : BitVec 64) (rest : Bytes) :
    readOne utf8 false owed (w32 T_DURATION ++ w64 m.toNat ++ w64 d.toNat ++ w64 s.toNat ++ rest) =
      if durNewOk m d s then .ok (.value (.duration m d s), rest)
      else .error .badDuration := by
  simp only [List.append_assoc]
  by_cases hd : durNewOk m d s <;>
  simp [readOne, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_ISTRING, T_ARRAY, T_MAP,
    T_POINT, T_VECTOR_F32, T_DATETIME, T_DATE, T_TIME, T_DURATION, hd]

-- `a_v3_duration_is_the_legacy_offset` (`value.rs`): P1Y1M1DT1H1M1S as C's offset
-- decomposes into 13 months, 1 day, 3661 seconds; under v4 the same 8 bytes run out.
def legacyP1Y1M1DT1H1M1S : BitVec 64 := BitVec.ofNat 64 (31536000 + 31 * 86400 + 86400 + 3661)
#guard durationFromLegacy legacyP1Y1M1DT1H1M1S == some (13, 1, 3661)
#guard match decodeV3 (fun _ => true) (w32 T_DURATION ++ w64 legacyP1Y1M1DT1H1M1S.toNat) with
  | .ok (.duration m d s, []) => m == 13 && d == 1 && s == 3661 | _ => false
#guard match decodeV4 (fun _ => true) (w32 T_DURATION ++ w64 legacyP1Y1M1DT1H1M1S.toNat) with
  | .error (.eof _ _) => true | _ => false
-- Negative and mixed-sign components (issue 2745) survive v4, and a component
-- `Dur::new` refuses (months past `i32`) is a `BadDuration`.
#guard match decodeV4 (fun _ => true) (w32 T_DURATION ++ w64 (BitVec.ofInt 64 (-13)).toNat ++
    w64 40 ++ w64 (BitVec.ofInt 64 (-5400)).toNat) with
  | .ok (.duration m d s, []) => m.toInt == -13 && d.toInt == 40 && s.toInt == -5400 | _ => false
#guard match decodeV4 (fun _ => true) (w32 T_DURATION ++ w64 (2 ^ 31) ++ w64 0 ++ w64 0) with
  | .error .badDuration => true | _ => false
-- A legacy offset past the calendar (`duration({seconds: 9e15})`) falls back to plain seconds.
#guard durationFromLegacy (BitVec.ofNat 64 9000000000000000) == some (0, 0, BitVec.ofNat 64 9000000000000000)

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
def readBody (utf8 : Bytes → Bool) (lg : Bool) (owed t : Nat) : R Step :=
  if t = T_NULL then pure (.value .null)
  else if t = T_BOOL then do let b ← u8; pure (.value (.bool (b ≠ 0)))
  else if t = T_INT64 then do let x ← i64; pure (.value (.int x))
  else if t = T_DOUBLE then do let x ← i64; pure (.value (.float x))
  else if t = T_STRING ∨ t = T_ISTRING then do
    let s ← rString utf8; pure (.value (.str s))
  else if t = T_ARRAY then do
    let n ← u32
    let _ ← guardCount (n + owed) MIN_VALUE_BYTES
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
  -- `T_DURATION if legacy_duration`: one offset through `duration_from_legacy` (v3)
  else if t = T_DURATION ∧ lg = true then do
    let x ← i64
    match durationFromLegacy x with
    | some (m, d, s) => pure (.value (.duration m d s))
    | none => fail .badDuration
  -- `T_DURATION`: three components through `Dur::new` (v4)
  else if t = T_DURATION then do
    let m ← i64; let d ← i64; let s ← i64
    if durNewOk m d s then pure (.value (.duration m d s)) else fail .badDuration
  else fail (.badValueType t)

theorem readOne_eq (utf8 : Bytes → Bool) (lg : Bool) (owed : Nat) : readOne utf8 lg owed = (u32 >>= readBody utf8 lg owed) := rfl

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

theorem good_readBody (utf8 : Bytes → Bool) (lg : Bool) (owed t : Nat) : Good (readBody utf8 lg owed t) := by
  have gv : ∀ {α} (m : R α) (k : α → Val), Good m → Good (m >>= fun x => pure (Step.value (k x))) :=
    fun m k hm => good_bind hm (fun _ => good_pure _)
  unfold readBody
  refine good_ite (good_pure _) (good_ite (gv _ _ (good_rdU 1)) (good_ite (gv _ _ good_i64)
    (good_ite (gv _ _ good_i64) (good_ite (gv _ _ (good_rString utf8)) (good_ite ?_
    (good_ite (good_fail (by decide) (by decide)) (good_ite ?_ (good_ite ?_
    (good_ite (gv _ _ good_i64) (good_ite (gv _ _ good_i64) (good_ite (gv _ _ good_i64)
    (good_ite ?_ (good_ite ?_ (good_fail (by intro h; cases h) (by intro h; cases h)))))))))))))))
  · exact good_bind (good_rdU 4) (fun _ => good_bind (good_guard _ _)
      (fun _ => good_ite (good_pure _) (good_pure _)))
  · exact good_bind good_f32 (fun _ => good_bind good_f32 (fun _ => good_pure _))
  · exact good_bind (good_rdU 4) (fun _ => good_bind (good_guard _ _)
      (fun _ => good_bind (good_readN good_f32 _) (fun _ => good_pure _)))
  · refine good_bind good_i64 (fun x => ?_)
    split
    · exact good_pure _
    · exact good_fail (by intro h; cases h) (by intro h; cases h)
  · exact good_bind good_i64 (fun _ => good_bind good_i64 (fun _ => good_bind good_i64
      (fun _ => good_ite (good_pure _) (good_fail (by intro h; cases h) (by intro h; cases h)))))

theorem good_readOne (utf8 : Bytes → Bool) (lg : Bool) (owed : Nat) : Good (readOne utf8 lg owed) := by
  rw [readOne_eq]; exact good_bind (good_rdU 4) (good_readBody utf8 lg owed)

/-- `read_one` consumes at least its 4-byte tag, and only ever a prefix. -/
theorem readOne_progress {utf8 : Bytes → Bool} {lg : Bool} {owed : Nat} {r : Bytes} {s : Step} {r' : Bytes}
    (h : readOne utf8 lg owed r = .ok (s, r')) : ∃ pre, r = pre ++ r' ∧ 4 ≤ pre.length := by
  rw [readOne_eq] at h
  simp only [bind_apply, u32, rdU, take] at h
  by_cases hl : r.length < 4
  · simp [hl] at h
  · simp only [hl, ite_false, pure_apply] at h
    obtain ⟨p, hp⟩ := (good_readBody utf8 lg owed _).1 _ _ _ h
    refine ⟨r.take 4 ++ p, ?_, ?_⟩
    · rw [List.append_assoc, ← hp, List.take_append_drop]
    · simp; omega

/-- Where `readBody` can end: a `Step.opened n` only ever comes from the array
arm, after `guard_count` has accepted `n` together with the `owed` siblings. -/
def OpenedOk (owed : Nat) (s : Step) (r' : Bytes) : Prop :=
  ∀ n, s = .opened n → 1 ≤ n ∧ 4 * (n + owed) ≤ r'.length

def Ends (owed : Nat) (m : R Step) : Prop := ∀ r s r', m r = .ok (s, r') → OpenedOk owed s r'

theorem ends_value {α} (owed : Nat) (m : R α) (k : α → Val) :
    Ends owed (m >>= fun x => pure (Step.value (k x))) := by
  intro r s r' h; simp only [bind_apply] at h; split at h
  · cases h
  · simp at h; intro n hn; rw [← h.1] at hn; cases hn
theorem ends_fail (owed : Nat) (e : DErr) : Ends owed (fail e) := by intro r s r' h; cases h
theorem ends_pure_value (owed : Nat) (v : Val) : Ends owed (pure (Step.value v)) := by
  intro r s r' h; cases h; intro n hn; cases hn
theorem ends_ite {owed : Nat} {c : Prop} [Decidable c] {m₁ m₂ : R Step} (h₁ : Ends owed m₁)
    (h₂ : Ends owed m₂) : Ends owed (if c then m₁ else m₂) := by split <;> assumption
theorem ends_array (owed : Nat) :
    Ends owed (do let n ← u32; let _ ← guardCount (n + owed) MIN_VALUE_BYTES
                  if n = 0 then pure (Step.value (.list [])) else pure (Step.opened n)) := by
  intro r s r' h
  simp only [bind_apply] at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i c r2 hg
      obtain ⟨_, rfl, hb⟩ := guardCount_bound hg
      split at h
      · cases h; intro n hn; cases hn
      · cases h; intro n hn; cases hn; simp [MIN_VALUE_BYTES] at hb; constructor <;> omega

theorem ends_bind {α} {owed : Nat} {m : R α} {f : α → R Step} (hf : ∀ a, Ends owed (f a)) :
    Ends owed (m >>= f) := by
  intro r s r' h
  simp only [bind_apply] at h
  split at h
  · cases h
  · exact hf _ _ _ _ h

theorem ends_readBody (utf8 : Bytes → Bool) (lg : Bool) (owed t : Nat) : Ends owed (readBody utf8 lg owed t) := by
  unfold readBody
  refine ends_ite (ends_pure_value _ _) (ends_ite (ends_value _ _ _) (ends_ite (ends_value _ _ _)
    (ends_ite (ends_value _ _ _) (ends_ite (ends_value _ _ _) (ends_ite (ends_array _)
    (ends_ite (ends_fail _ _) (ends_ite ?_ (ends_ite ?_
    (ends_ite (ends_value _ _ _) (ends_ite (ends_value _ _ _) (ends_ite (ends_value _ _ _)
    (ends_ite ?_ (ends_ite ?_ (ends_fail _ _))))))))))))))
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
  · exact ends_bind fun x => by
      split
      · exact ends_pure_value _ _
      · exact ends_fail _ _
  · exact ends_bind fun _ => ends_bind fun _ => ends_bind fun _ =>
      ends_ite (ends_pure_value _ _) (ends_fail _ _)

/-- A frame is only opened for a count that is non-zero and that the bytes
still unread could hold at four bytes per child *together with* the `owed`
siblings: this is the bound on each `ThinVec::with_capacity(n)` (`value.rs`). -/
theorem readOne_opened_bound {utf8 : Bytes → Bool} {lg : Bool} {owed : Nat} {r : Bytes} {n : Nat} {r' : Bytes}
    (h : readOne utf8 lg owed r = .ok (.opened n, r')) : 1 ≤ n ∧ 4 * (n + owed) ≤ r'.length := by
  rw [readOne_eq] at h
  simp only [bind_apply] at h
  split at h
  · cases h
  · exact ends_readBody utf8 lg owed _ _ _ _ h n rfl

/-- Every open frame is waiting on at least one child. -/
def FramesOk (st : List Frame) : Prop := ∀ f ∈ st, 1 ≤ f.left

/-- The inner loop never underflows `left`, and when it hands back a stack that
still waits (`.more`), exactly one owed child has just been supplied: that is
why `owed -= 1` there never underflows. -/
theorem deliver_ok : ∀ (v : Val) (st : List Frame), FramesOk st →
    deliver v st ≠ .panic ∧ ∀ st', deliver v st = .more st' → FramesOk st' ∧ owedOf st' + 1 = owedOf st
  | v, [], _ => by simp [deliver]
  | v, f :: st, h => by
    have hf : 1 ≤ f.left := h f (by simp)
    have hst : FramesOk st := fun g hg => h g (by simp [hg])
    simp only [deliver]
    rw [ite_eq_right (by omega)]
    split
    · rename_i hgt
      refine ⟨by simp, ?_⟩
      intro st' he; cases he
      refine ⟨?_, ?_⟩
      · intro g hg; simp at hg
        rcases hg with rfl | hg
        · simp; omega
        · exact hst g hg
      · simp only [owedOf, List.map_cons, List.sum_cons]; omega
    · rename_i hle
      obtain ⟨h1, h2⟩ := deliver_ok _ st hst
      refine ⟨h1, fun st' he => ?_⟩
      obtain ⟨h3, h4⟩ := h2 st' he
      refine ⟨h3, ?_⟩
      simp only [owedOf, List.map_cons, List.sum_cons] at h4 ⊢
      omega

/-- **The decoder never panics**, on any input whatsoever — neither the
`*left -= 1` of `value.rs`, which would underflow on a frame of zero, nor
#3267's `owed -= 1` (`value.rs`): `owed` is always `owedOf` the stack. -/
theorem loop_noPanic (utf8 : Bytes → Bool) (lg : Bool) :
    ∀ fuel st owed r, FramesOk st → owed = owedOf st → loop utf8 lg fuel st owed r ≠ .error .panic
  | 0, _, _, _, _, _ => by simp [loop]
  | fuel + 1, st, owed, r, hst, ho => by
    simp only [loop]
    split
    · rename_i e he; intro h; cases h; exact (good_readOne utf8 lg owed).2.1 r he
    · rename_i n r' hr
      have hn := (readOne_opened_bound hr).1
      apply loop_noPanic utf8 lg fuel _ _ r'
      · intro f hf; simp at hf
        rcases hf with rfl | hf
        · exact hn
        · exact hst f hf
      · simp only [owedOf, List.map_cons, List.sum_cons] at ho ⊢; omega
    · rename_i v r' _
      obtain ⟨hp, hm⟩ := deliver_ok v st hst
      split
      · simp
      · rename_i st' hd
        obtain ⟨h1, h2⟩ := hm st' hd
        simp only [show ¬(owed = 0) by omega, ↓reduceIte]
        exact loop_noPanic utf8 lg fuel st' _ r' h1 (by omega)
      · rename_i hd; exact absurd hd hp

/-- **The fuel is never what stops the decoder**: four bytes per turn, so
`remaining / 4 + 1` turns always suffice. -/
theorem loop_fuel_enough (utf8 : Bytes → Bool) (lg : Bool) :
    ∀ fuel st owed r, r.length < 4 * fuel → loop utf8 lg fuel st owed r ≠ .error .outOfFuel
  | 0, _, _, r, h => by simp at h
  | fuel + 1, st, owed, r, h => by
    simp only [loop]
    split
    · rename_i e he; intro hh; cases hh; exact (good_readOne utf8 lg owed).2.2 r he
    · rename_i n r' hr
      obtain ⟨p, rfl, hp⟩ := readOne_progress hr
      exact loop_fuel_enough utf8 lg fuel _ _ r' (by simp at h; omega)
    · rename_i v r' hr
      obtain ⟨p, rfl, hp⟩ := readOne_progress hr
      split
      · simp
      · rename_i st' _
        split
        · simp
        · exact loop_fuel_enough utf8 lg fuel st' _ r' (by simp at h; omega)
      · simp

/-- `loop` only consumes. -/
theorem loop_consumes (utf8 : Bytes → Bool) (lg : Bool) :
    ∀ fuel st owed r v r', loop utf8 lg fuel st owed r = .ok (v, r') → ∃ pre, r = pre ++ r'
  | 0, _, _, _, _, _, h => by simp [loop] at h
  | fuel + 1, st, owed, r, v, r', h => by
    simp only [loop] at h
    split at h
    · cases h
    · rename_i n r1 hr
      obtain ⟨p, rfl, _⟩ := readOne_progress hr
      obtain ⟨q, rfl⟩ := loop_consumes utf8 lg fuel _ _ _ _ _ h
      exact ⟨p ++ q, by simp⟩
    · rename_i w r1 hr
      obtain ⟨p, rfl, _⟩ := readOne_progress hr
      split at h
      · cases h; exact ⟨p, rfl⟩
      · split at h
        · cases h
        · obtain ⟨q, rfl⟩ := loop_consumes utf8 lg fuel _ _ _ _ _ h
          exact ⟨p ++ q, by simp⟩
      · cases h

/-- **`Value::decode` on hostile bytes**: whatever `r` is, it neither panics nor
over-reads, and the modelling fuel is never the reason it stops. -/
theorem decodeV_safe (utf8 : Bytes → Bool) (lg : Bool) (r : Bytes) :
    decodeV utf8 lg r ≠ .error .panic ∧ decodeV utf8 lg r ≠ .error .outOfFuel ∧
    ∀ v r', decodeV utf8 lg r = .ok (v, r') → ∃ pre, r = pre ++ r' :=
  ⟨loop_noPanic utf8 lg _ [] 0 r (by simp [FramesOk]) (by simp [owedOf]),
   loop_fuel_enough utf8 lg _ [] 0 r (by omega),
   fun v r' h => loop_consumes utf8 lg _ [] 0 r v r' h⟩

/-! ## What the primary can hand the encoder -/

-- `pending::is_valid_property` (`graph/src/runtime/pending.rs`).
mutual
def propValid (allowNull : Bool) : Val → Bool
  | .null => allowNull
  | .bool _ | .int _ | .float _ | .str _ | .point _ _ | .vecf32 _
  | .datetime _ | .date _ | .time _ | .duration _ _ _ => true
  | .list xs => propValidL xs
  | _ => false
def propValidL : List Val → Bool
  | [] => true
  | v :: vs => propValid false v && propValidL vs
end

-- Container lengths fit the `u32` counts (`value.rs`, `value.rs`).
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
`value.rs`: maps, nodes, edges and paths are refused where the value is
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
  | .duration _ _ _, _, _, _ => ⟨_, rfl⟩
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

theorem nest_wf (utf8 : Bytes → Bool) (lg : Bool) : ∀ d, WF utf8 lg (nest d)
  | 0 => trivial
  | d + 1 => by simp only [nest, WF, WFL]; exact ⟨nest_wf utf8 lg d, trivial⟩

/-- **Depth is unbounded and costs eight bytes a level**: the decoder returns
a value `d` deep from `8d + 4` bytes, for every `d`. The decode itself is
iterative (`value.rs`), but `Value` is recursive and its `Drop` is not,
so this is the input that overflows the stack *after* a successful decode —
see `README.md`, `hostile_deep_nesting_*`. -/
theorem nest_bytes_length : ∀ d,
    ((List.replicate d (w32 T_ARRAY ++ w32 1)).flatten ++ w32 T_NULL).length = 8 * d + 4
  | 0 => rfl
  | d + 1 => by
    have := nest_bytes_length d
    simp only [List.replicate_succ, List.flatten_cons, List.length_append] at this ⊢
    simp only [w32, le_length] at this ⊢
    omega

theorem nest_decodes (utf8 : Bytes → Bool) (lg : Bool) (d : Nat) :
    ∃ bs, bs.length = 8 * d + 4 ∧ decodeV utf8 lg bs = .ok (nest d, []) := by
  refine ⟨_, nest_bytes_length d, ?_⟩
  have := decodeV_encV utf8 lg (fun _ => false) (nest d) _ [] (encV_nest _ d) (nest_wf utf8 lg d)
  simpa using this

/-- The capacities `loop` asks for, turn by turn: `n` for every
`Frame::List` it pushes (`ThinVec::with_capacity(n)`, `value.rs`). A
mirror of `loop` that records instead of building. -/
def capTrace (utf8 : Bytes → Bool) (lg : Bool) : Nat → List Frame → Nat → Bytes → List Nat
  | 0, _, _, _ => []
  | fuel + 1, st, owed, r =>
    match readOne utf8 lg owed r with
    | .error _ => []
    | .ok (.opened n, r') => n :: capTrace utf8 lg fuel (⟨[], n⟩ :: st) (owed + (n - 1)) r'
    | .ok (.value v, r') =>
      match deliver v st with
      | .more st' => if owed = 0 then [] else capTrace utf8 lg fuel st' (owed - 1) r'
      | _ => []

/-- **The reservations are linear in the input.** Whatever the bytes, the
capacities one decode reserves sum to at most a quarter of what is left once the
`owed` siblings are set aside: each claim is checked against the bytes no
ancestor has already claimed. -/
theorem capTrace_linear (utf8 : Bytes → Bool) (lg : Bool) :
    ∀ fuel st owed r, 4 * (capTrace utf8 lg fuel st owed r).sum ≤ r.length - 4 * owed
  | 0, _, _, _ => by simp [capTrace]
  | fuel + 1, st, owed, r => by
    simp only [capTrace]
    split
    · simp
    · rename_i n r' hr
      obtain ⟨hn, hb⟩ := readOne_opened_bound hr
      obtain ⟨p, rfl, hp⟩ := readOne_progress hr
      have ih := capTrace_linear utf8 lg fuel (⟨[], n⟩ :: st) (owed + (n - 1)) r'
      simp only [List.sum_cons, List.length_append]
      omega
    · rename_i v r' hr
      obtain ⟨p, rfl, hp⟩ := readOne_progress hr
      split
      · rename_i st' _
        split
        · simp
        · have ih := capTrace_linear utf8 lg fuel st' (owed - 1) r'
          simp only [List.length_append]
          omega
      · simp

/-- From the start of a value: at most `remaining / 4` elements reserved in all. -/
theorem decode_reserves_linear (utf8 : Bytes → Bool) (lg : Bool) (r : Bytes) :
    4 * (capTrace utf8 lg (r.length + 1) [] 0 r).sum ≤ r.length := by
  have := capTrace_linear utf8 lg (r.length + 1) [] 0 r
  omega

/-- `k` nested arrays, each claiming as many children as the bytes after it
could hold at four bytes each — the most a per-level `guard_count` lets through
(the Rust test `nested_lists_cannot_each_claim_the_whole_remaining_buffer`). -/
def hostile (k : Nat) : Bytes :=
  (List.range k).flatMap fun i => w32 T_ARRAY ++ w32 (2 * (k - i - 1))

#guard (hostile 100).length = 800

/-! ### On main: refused at the second level -/

def hostileCaps (k : Nat) : Nat := (capTrace (fun _ => true) false (8 * k + 1) [] 0 (hostile k)).sum

/-- The first level's claim passes; the second level's children and the first
level's remaining siblings cannot both fit, so it is refused up front. -/
theorem hostile_refused_second_level :
    capTrace (fun _ => true) false 81 [] 0 (hostile 10) = [18] ∧
    (match decodeV (fun _ => true) false (hostile 10) with
     | .error e => e == .implausible 33 64 | .ok _ => false) = true := by
  decide

#guard capTrace (fun _ => true) false 8001 [] 0 (hostile 1000) = [1998]
-- `ImplausibleCount` with 16 bytes consumed, as the Rust test asserts.
#guard match decodeV (fun _ => true) false (hostile 1000) with
  | .error (.implausible _ rem) => rem == 8000 - 16 | _ => false

-- Bit-exact floats: NaN payloads and signed zero go through untouched, because
-- a float *is* its bit pattern on this wire.
def floatBack (bits : Nat) : Option Nat :=
  match decodeV (fun _ => true) false (w32 T_DOUBLE ++ w64 bits) with
  | .ok (.float b, []) => some b.toNat
  | _ => none

#guard floatBack 0x7ff4000000000001 = some 0x7ff4000000000001  -- signalling NaN, payload 1
#guard floatBack 0xfff8000000000000 = some 0xfff8000000000000  -- negative quiet NaN
#guard floatBack 0x8000000000000000 = some 0x8000000000000000  -- -0.0

end FalkorCodec
