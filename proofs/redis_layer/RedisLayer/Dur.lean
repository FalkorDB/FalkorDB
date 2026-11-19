/-!
# `temporal::Dur` — the three-component duration (#3610)

`graph/src/runtime/functions/temporal.rs`. A `Value::Duration` is a `Dur`: three independent
signed components (months, days, seconds), read only through `Dur::components` (:555). The
word it is stored in (bit 0 = 1: inline, bit 0 = 0: `Arc<[i64;3]>` pointer) is canonical
(inline exactly when the components fit, :539-551) and never observed except by the attribute
store (`inline_bits` :582 / `from_inline_bits` :590) — so a `Dur` here is its components, plus
the invariant `Dur::new` (:515) checks; `inlineBits` / `ofInlineBits` model the word.

| Lean | Rust (temporal.rs) |
| --- | --- |
| `daysFromCivil`     | `days_from_civil` :470 (the module-local copy `Dur::new` and `duration_to_legacy` use) |
| `civilFromDays`     | `value::civil_from_days` (value.rs:810) over `civil_from_days_wide` (value.rs:818) |
| `legacyOffset`      | the i128 sum `Dur::new` range-checks (:530-537) |
| `DurOk`, `Dur`, `durNew` | `Dur::new` :515 (Err unless months fit i32 and the offset fits i64) |
| `inlineBits`, `decodeInline`, `ofInlineBits` | `Dur::new`'s packing :539-548, `unpack_inline` :571, `inline_bits` :582, `from_inline_bits` :590 |
| `toLegacy`          | `duration_to_legacy` :745 (wrapping ops, `saturating_add` year) |
| `fromLegacy`        | `duration_from_legacy` :719 |

* `durNew_self` — every `Dur` is what `Dur::new` returns on its own components.
* `toLegacy_eq` — for every `Dur`, the wrapping/saturating ops of `duration_to_legacy` never
  bite: it is the exact offset.
* `fromLegacy_toLegacy` — `duration_from_legacy` is total on `i64` and
  `duration_to_legacy (duration_from_legacy x) = x` (the compact reply of a duration loaded from
  an old stream is the integer that stream held).
* `inline_roundtrip` — the inline word reads back as the same `Dur`; it is odd and `< 2^64`.
-/
namespace RedisLayer.Temporal

def FitsI64 (x : Int) : Prop := -2 ^ 63 ≤ x ∧ x < 2 ^ 63
def FitsI32 (x : Int) : Prop := -2 ^ 31 ≤ x ∧ x < 2 ^ 31

instance (x : Int) : Decidable (FitsI64 x) := by unfold FitsI64; infer_instance
instance (x : Int) : Decidable (FitsI32 x) := by unfold FitsI32; infer_instance

/-- `x as i32` / `wrapping_*` on i64: two's-complement wrap. -/
def wrap32 (x : Int) : Int := (x + 2 ^ 31) % 2 ^ 32 - 2 ^ 31
def wrap64 (x : Int) : Int := (x + 2 ^ 63) % 2 ^ 64 - 2 ^ 63
/-- `i32::saturating_add` result clamp. -/
def sat32 (x : Int) : Int := max (-2 ^ 31) (min x (2 ^ 31 - 1))

/-- `days_from_civil(y, m, d)` (temporal.rs:470): `/` is Rust's truncating `i32`/`u32` division. -/
def daysFromCivil (y m d : Int) : Int :=
  let y := if m ≤ 2 then y - 1 else y
  let era := (if 0 ≤ y then y else y - 399).tdiv 400
  let yoe := y - era * 400
  let mp := if m > 2 then m - 3 else m + 9
  let doy := (153 * mp + 2) / 5 + d - 1
  let doe := yoe * 365 + yoe / 4 - yoe / 100 + doy
  era * 146097 + doe - 719468

/-- `civil_from_days(z)` (value.rs:810): `civil_from_days_wide` (:818), year cast `as i32`. -/
def civilFromDays (z : Int) : Int × Int × Int :=
  let z := z + 719468
  let era := z / 146097                     -- div_euclid
  let doe := z % 146097                     -- rem_euclid
  let yoe := (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
  let y := yoe + era * 400
  let doy := doe - (365 * yoe + yoe / 4 - yoe / 100)
  let mp := (5 * doy + 2) / 153
  let d := doy - (153 * mp + 2) / 5 + 1
  let m := if mp < 10 then mp + 3 else mp - 9
  let y := if m ≤ 2 then y + 1 else y
  (wrap32 y, m, d)

/-- The legacy epoch offset of `(months, days, seconds)`, exact (`Dur::new`'s i128 sum, :530-537;
`months.div_euclid(12) as i32` is exact once months fit i32). -/
def legacyOffset (m d s : Int) : Int :=
  daysFromCivil (1970 + m / 12) (m % 12 + 1) 1 * 86400 + (d * 86400 + s)

/-- What `Dur::new(m, d, s)` (:515) accepts: i64 arguments, months fit `i32`, offset fits `i64`. -/
def DurOk (m d s : Int) : Prop :=
  FitsI32 m ∧ FitsI64 d ∧ FitsI64 s ∧ FitsI64 (legacyOffset m d s)

instance (m d s : Int) : Decidable (DurOk m d s) := by unfold DurOk; infer_instance

/-- A `Dur`: its components (`Dur::components`, :555), built only by `Dur::new` (the word is a
private field; `from_inline_bits` only rebuilds a word `inline_bits` gave out). -/
structure Dur where
  m : Int
  d : Int
  s : Int
  ok : DurOk m d s

theorem Dur.ext' {a b : Dur} (h1 : a.m = b.m) (h2 : a.d = b.d) (h3 : a.s = b.s) : a = b := by
  cases a; cases b; simp_all

/-- `PartialEq for Dur` (:631): on components. -/
instance : DecidableEq Dur := fun a b =>
  if h : a.m = b.m ∧ a.d = b.d ∧ a.s = b.s then isTrue (Dur.ext' h.1 h.2.1 h.2.2)
  else isFalse (by intro e; subst e; exact h ⟨rfl, rfl, rfl⟩)

instance : Repr Dur := ⟨fun a _ => s!"Dur({a.m}m {a.d}d {a.s}s)"⟩

/-- `Dur::new(m, d, s)` (:515); `none` is its `Err`. -/
def durNew (m d s : Int) : Option Dur := if h : DurOk m d s then some ⟨m, d, s, h⟩ else none

theorem durNew_self (x : Dur) : durNew x.m x.d x.s = some x := by
  simp [durNew, x.ok]

theorem daysFromCivil_epoch : daysFromCivil 1970 1 1 = 0 := by decide

/-! ## `duration_to_legacy` (:745) -/

/-- `duration_to_legacy`: `1970i32.saturating_add(m.div_euclid(12) as i32)`, then wrapping
`* 86400`, `+ days.wrapping_mul(86400)`, `+ seconds` (wrapping ops compose to one wrap). -/
def toLegacy (x : Dur) : Int :=
  wrap64 (daysFromCivil (sat32 (1970 + wrap32 (x.m / 12))) (x.m % 12 + 1) 1 * 86400 +
    (x.d * 86400 + x.s))

/-- For every `Dur` the wrapping and saturation never bite: the exact offset. -/
theorem toLegacy_eq (x : Dur) : toLegacy x = legacyOffset x.m x.d x.s := by
  obtain ⟨m, d, s, ⟨hm1, hm2⟩, -, -, ho⟩ := x
  have hw : wrap32 (m / 12) = m / 12 := by unfold wrap32; omega
  have hs : sat32 (1970 + m / 12) = 1970 + m / 12 := by unfold sat32; omega
  unfold toLegacy
  simp only
  rw [hw, hs]
  unfold FitsI64 legacyOffset at ho
  unfold wrap64 legacyOffset
  omega

/-! ## `duration_from_legacy` (:719) -/

/-- The calendar decomposition `duration_from_legacy` tries first (:720-724):
`Dur::new(months, remaining / 86400, remaining % 86400)`. -/
def legacyCand (x : Int) : Option Dur :=
  let days := x / 86400                    -- div_euclid
  let tod := x % 86400                     -- rem_euclid
  let c := civilFromDays days
  let months := (c.1 - 1970) * 12 + c.2.1 - 1
  let remaining := (c.2.2 - 1) * 86400 + tod
  durNew months (remaining.tdiv 86400) (remaining.tmod 86400)

/-- `duration_from_legacy(legacy)` (:719): the candidate, kept only if it reproduces the
offset, else `Dur::new(0, 0, legacy)`, else `Dur::new(0, days, time_of_day)`; `none` is an
`Err` (never, for an `i64`: `fromLegacy_toLegacy`). -/
def fromLegacy (x : Int) : Option Dur :=
  match (legacyCand x).filter (fun c => toLegacy c == x) with
  | some c => some c
  | none =>
    match durNew 0 0 x with
    | some c => some c
    | none => durNew 0 (x / 86400) (x % 86400)

theorem legacyOffset_zero (x : Int) : legacyOffset 0 0 x = x := by
  simp [legacyOffset, daysFromCivil_epoch]

/-- **`duration_from_legacy` inverts `duration_to_legacy`**: for every `i64` offset it returns a
`Dur` whose legacy form is that offset. -/
theorem fromLegacy_toLegacy (x : Int) (hx : FitsI64 x) :
    ∃ c, fromLegacy x = some c ∧ toLegacy c = x := by
  unfold fromLegacy
  split
  · rename_i c hc
    refine ⟨c, rfl, ?_⟩
    -- the filter's own check
    cases h : legacyCand x with
    | none => simp [h, Option.filter] at hc
    | some c' =>
      rw [h] at hc
      simp only [Option.filter] at hc
      split at hc
      · rename_i hp; cases hc; simpa using hp
      · cases hc
  · have h0 : DurOk 0 0 x := by
      refine ⟨by decide, by decide, hx, ?_⟩
      rw [legacyOffset_zero]; exact hx
    have hn : durNew 0 0 x = some ⟨0, 0, x, h0⟩ := by simp [durNew, h0]
    rw [hn]
    refine ⟨_, rfl, ?_⟩
    rw [toLegacy_eq]; exact legacyOffset_zero x

/-! ## The inline word (`Dur::new` :539-548, `unpack_inline` :571) -/

/-- Inline exactly when `|months| ≤ 4095`, `|days| ≤ 131071`, `|seconds| ≤ 2^31-1` (:539-541). -/
def IsInline (x : Dur) : Prop :=
  -4095 ≤ x.m ∧ x.m ≤ 4095 ∧ -131071 ≤ x.d ∧ x.d ≤ 131071 ∧
    -(2 ^ 31 - 1) ≤ x.s ∧ x.s ≤ 2 ^ 31 - 1

instance (x : Dur) : Decidable (IsInline x) := by unfold IsInline; infer_instance

/-- `(m as u64) << 51 | (d as u64 & (2^18-1)) << 33 | (s as u64 & (2^32-1)) << 1 | 1`: the
shifted fields are disjoint, so `|` is `+`, and `<< 51` keeps the low 13 bits of `m`. -/
def packInline (m d s : Int) : Int :=
  (m % 2 ^ 13) * 2 ^ 51 + (d % 2 ^ 18) * 2 ^ 33 + (s % 2 ^ 32) * 2 + 1

/-- `Dur::inline_bits` (:582): the word when inline (`Dur::new` chose the inline form). -/
def inlineBits (x : Dur) : Option Int := if IsInline x then some (packInline x.m x.d x.s) else none

/-- Sign-extend the low `2^k`-sized field `v` (`0 ≤ v < 2^k`): an arithmetic shift of the
field moved to the top of the `i64`. -/
def sext (k v : Int) : Int := if 2 * v ≥ k then v - k else v

/-- `unpack_inline(bits)` (:571): months `b >> 51`, days `(b << 13) >> 46`, seconds
`(b << 31) >> 32` (arithmetic shifts of `bits as i64`). -/
def decodeInline (b : Int) : Int × Int × Int :=
  (sext (2 ^ 13) (b / 2 ^ 51 % 2 ^ 13), sext (2 ^ 18) (b / 2 ^ 33 % 2 ^ 18),
    sext (2 ^ 32) (b / 2 % 2 ^ 32))

theorem sext13 (v : Int) (h0 : 0 ≤ v) (h1 : v < 2 ^ 13) :
    -4096 ≤ sext (2 ^ 13) v ∧ sext (2 ^ 13) v ≤ 4095 := by
  unfold sext; split <;> omega
theorem sext18 (v : Int) (h0 : 0 ≤ v) (h1 : v < 2 ^ 18) :
    -2 ^ 17 ≤ sext (2 ^ 18) v ∧ sext (2 ^ 18) v < 2 ^ 17 := by
  unfold sext; split <;> omega
theorem sext32 (v : Int) (h0 : 0 ≤ v) (h1 : v < 2 ^ 32) :
    -2 ^ 31 ≤ sext (2 ^ 32) v ∧ sext (2 ^ 32) v < 2 ^ 31 := by
  unfold sext; split <;> omega

theorem decodeInline_range (b : Int) :
    -4096 ≤ (decodeInline b).1 ∧ (decodeInline b).1 ≤ 4095 ∧
      -2 ^ 17 ≤ (decodeInline b).2.1 ∧ (decodeInline b).2.1 < 2 ^ 17 ∧
      -2 ^ 31 ≤ (decodeInline b).2.2 ∧ (decodeInline b).2.2 < 2 ^ 31 := by
  have h1 := sext13 (b / 2 ^ 51 % 2 ^ 13) (by omega) (by omega)
  have h2 := sext18 (b / 2 ^ 33 % 2 ^ 18) (by omega) (by omega)
  have h3 := sext32 (b / 2 % 2 ^ 32) (by omega) (by omega)
  simp only [decodeInline]
  omega

/-- `days_from_civil` of a year in `1..=10^6` is a small non-negative-era day count. -/
theorem daysFromCivil_bound (y mo : Int) (hy1 : 1 ≤ y) (hy2 : y ≤ 1000000) (hm1 : 1 ≤ mo)
    (hm2 : mo ≤ 12) : -10 ^ 9 ≤ daysFromCivil y mo 1 ∧ daysFromCivil y mo 1 ≤ 10 ^ 9 := by
  unfold daysFromCivil
  have hy0 : 0 ≤ (if mo ≤ 2 then y - 1 else y) := by split <;> omega
  simp only [hy0, ite_true]
  rw [Int.tdiv_eq_ediv_of_nonneg hy0]
  split <;> split <;> omega

/-- Every inline-decoded triple is a valid `Dur` (so `from_inline_bits` always yields one). -/
theorem decodeInline_ok (b : Int) :
    DurOk (decodeInline b).1 (decodeInline b).2.1 (decodeInline b).2.2 := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := decodeInline_range b
  generalize decodeInline b = c at *
  obtain ⟨m, d, s⟩ := c
  simp only at *
  have hb := daysFromCivil_bound (1970 + m / 12) (m % 12 + 1) (by omega) (by omega) (by omega)
    (by omega)
  refine ⟨⟨by omega, by omega⟩, ⟨by omega, by omega⟩, ⟨by omega, by omega⟩, ?_⟩
  unfold legacyOffset
  constructor <;> omega

/-- `Dur::from_inline_bits(bits)` (:590): `none` is its `assert!(bits & 1 == 1)` panic. -/
def ofInlineBits (b : Int) : Option Dur :=
  if b % 2 = 1 then
    some ⟨(decodeInline b).1, (decodeInline b).2.1, (decodeInline b).2.2, decodeInline_ok b⟩
  else none

theorem sext_emod13 (m : Int) (h1 : -4096 ≤ m) (h2 : m ≤ 4095) : sext (2 ^ 13) (m % 2 ^ 13) = m := by
  unfold sext; split <;> omega
theorem sext_emod18 (m : Int) (h1 : -2 ^ 17 ≤ m) (h2 : m < 2 ^ 17) : sext (2 ^ 18) (m % 2 ^ 18) = m := by
  unfold sext; split <;> omega
theorem sext_emod32 (m : Int) (h1 : -2 ^ 31 ≤ m) (h2 : m < 2 ^ 31) : sext (2 ^ 32) (m % 2 ^ 32) = m := by
  unfold sext; split <;> omega

theorem decode_packInline (m d s : Int) (hm1 : -4095 ≤ m) (hm2 : m ≤ 4095) (hd1 : -131071 ≤ d)
    (hd2 : d ≤ 131071) (hs1 : -(2 ^ 31 - 1) ≤ s) (hs2 : s ≤ 2 ^ 31 - 1) :
    decodeInline (packInline m d s) = (m, d, s) := by
  have e1 : packInline m d s / 2 ^ 51 % 2 ^ 13 = m % 2 ^ 13 := by unfold packInline; omega
  have e2 : packInline m d s / 2 ^ 33 % 2 ^ 18 = d % 2 ^ 18 := by unfold packInline; omega
  have e3 : packInline m d s / 2 % 2 ^ 32 = s % 2 ^ 32 := by unfold packInline; omega
  simp only [decodeInline, e1, e2, e3, sext_emod13 m (by omega) hm2, sext_emod18 d (by omega) (by omega),
    sext_emod32 s (by omega) (by omega)]

/-- **Inline round trip.** The word `inline_bits` gives out is odd, fits a `u64`, and
`from_inline_bits` rebuilds the same `Dur` from it. -/
theorem inline_roundtrip (x : Dur) (b : Int) (h : inlineBits x = some b) :
    b % 2 = 1 ∧ 0 ≤ b ∧ b < 2 ^ 64 ∧ ofInlineBits b = some x := by
  unfold inlineBits at h
  split at h
  · rename_i hi
    obtain ⟨hm1, hm2, hd1, hd2, hs1, hs2⟩ := hi
    cases h
    have hodd : packInline x.m x.d x.s % 2 = 1 := by unfold packInline; omega
    refine ⟨hodd, by unfold packInline; omega, by unfold packInline; omega, ?_⟩
    simp only [ofInlineBits, hodd, ite_true, Option.some.injEq]
    have := decode_packInline x.m x.d x.s hm1 hm2 hd1 hd2 hs1 hs2
    exact Dur.ext' (show (decodeInline (packInline x.m x.d x.s)).1 = x.m by rw [this])
      (show (decodeInline (packInline x.m x.d x.s)).2.1 = x.d by rw [this])
      (show (decodeInline (packInline x.m x.d x.s)).2.2 = x.s by rw [this])
  · cases h

end RedisLayer.Temporal
