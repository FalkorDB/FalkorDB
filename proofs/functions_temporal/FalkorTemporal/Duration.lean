import FalkorTemporal.DurationPre3610
/-!
# Durations and date arithmetic (since #3610: three components)

| here | there |
| --- | --- |
| `Dur` (`Calendar.lean`) | `Dur` — `temporal.rs:504`, read through `Dur::components` `temporal.rs:555` |
| `chk`, `chkMul`, `chkAdd`, `chk32` | `i64::checked_*`, `i64::try_from(i128)`, `i32::try_from` |
| `durNew`                | `Dur::new` — `temporal.rs:515`, the range checks (its value is the components) |
| `DurWord`, `durNewW`, `packBits`, `unpackInline`, `wordComponents`, `inlineBits`, `fromInlineBits` | the `u64` word of `Dur::new` `:515`, `Dur::components` `:555`, `Dur::unpack_inline` `:571`, `Dur::inline_bits` `:582`, `Dur::from_inline_bits` `:590` |
| `constructDuration`     | `construct_duration` — `temporal.rs:663` |
| `toLegacy` / `legacyParts`, `fromLegacy` | `duration_to_legacy` — `temporal.rs:745` / `duration_from_legacy` — `temporal.rs:719` |
| `MAX_DATE_YEAR`, `checkDateRange` | `MAX_DATE_YEAR` `temporal.rs:693`, `check_date_range` — `temporal.rs:706` |
| `shiftMonths`           | the arithmetic of `shift_months` — `value.rs:725` |
| `shiftMonthsC`          | `shift_months` — `value.rs:725`, checked `i32` arithmetic (no `MAX_DATE_YEAR` guard since #3610) |
| `shiftTs`, `addDur` / `subDur` (`addDurRaw` / `subDurRaw`: the arithmetic) | `shift_timestamp` `value.rs:665`, `add_duration_to_timestamp` `value.rs:645` / `sub_duration_from_timestamp` `value.rs:653` |
| `timeAddDur` / `timeSubDur` | `(Time, Duration)` arms of `add_slow` (`value.rs:973`) and `Sub::sub` (`value.rs:1085`): the seconds component only |
| `combineDurations`      | `combine_durations` — `value.rs:701` (`Duration ± Duration`) |
| `durationCmp`           | `duration_cmp` — `value.rs:685` (`compare_value`, `sort_cmp`) |

Before #3610 a Duration was one `i64`, "the instant 1970-01-01 + years/months +
days/seconds" (that model is `DurationPre3610.lean`, `FalkorTemporal.Pre3610`). Now it
is three independent signed components, months / days / seconds; the legacy offset
survives only in the compact reply (`toLegacy`) and in values stored before
(`fromLegacy`, which replays `Pre3610.decomposeDuration`).
-/
namespace FalkorTemporal

def chk (x : Int) : Option Int := if decide (-2 ^ 63 ≤ x) && decide (x < 2 ^ 63) then some x else none
def chkMul (a b : Int) : Option Int := chk (a * b)
def chkAdd (a b : Int) : Option Int := chk (a + b)
def chk32 (x : Int) : Option Int := if decide (-2 ^ 31 ≤ x) && decide (x < 2 ^ 31) then some x else none

theorem chk_of (x : Int) (h : I64 x) : chk x = some x := by
  unfold chk; unfold I64 at h
  rw [decide_eq_true h.1, decide_eq_true h.2]; rfl
theorem chk32_of (x : Int) (h : I32 x) : chk32 x = some x := by
  unfold chk32; unfold I32 at h
  rw [decide_eq_true h.1, decide_eq_true h.2]; rfl

/-! ### `MAX_DATE_YEAR`, `check_date_range` (#2746) -/

def MAX_DATE_YEAR : Int := 2147481647
def minDateDays : Int := daysFromCivil (-MAX_DATE_YEAR) 1 1
def maxDateDays : Int := daysFromCivil MAX_DATE_YEAR 12 31

/-- `check_date_range` (`temporal.rs:706`): `Err("Date out of range")` ↦ `none`. -/
def checkDateRange (ts : Int) : Option Int :=
  if minDateDays ≤ ts / 86400 ∧ ts / 86400 ≤ maxDateDays then some ts else none

/-- `i64::try_from(i128)`/two's complement wrapping to `i64`. -/
def wrap64 (x : Int) : Int := (x + 2 ^ 63) % 2 ^ 64 - 2 ^ 63

theorem wrap64_id (x : Int) (h : I64 x) : wrap64 x = x := by
  unfold wrap64; unfold I64 at h; omega

/-! ## `Dur::new` (`temporal.rs:515`) -/

/-- `days_from_civil(1970 + months.div_euclid(12), months.rem_euclid(12) + 1, 1) * 86400`:
the calendar offset of the months (`days_from_civil` of `temporal.rs:470` is
`daysFromCivilT = daysFromCivil`; Lean's `/`, `%` by `12` are `div_euclid`, `rem_euclid`). -/
def durAnchor (m : Int) : Int := daysFromCivil (1970 + m / 12) (m % 12 + 1) 1 * 86400

/-- `Dur::new` (`temporal.rs:515`), `Err` ↦ `none`: months must fit an `i32`, and the
legacy offset (anchor + days·86400 + seconds, one `i128` sum) an `i64`. -/
def durNew (m d s : Int) : Option Dur :=
  if -2 ^ 31 ≤ m ∧ m < 2 ^ 31 ∧
      -2 ^ 63 ≤ durAnchor m + d * 86400 + s ∧ durAnchor m + d * 86400 + s < 2 ^ 63
  then some ⟨m, d, s⟩ else none

theorem durNew_iff (m d s : Int) (c : Dur) :
    durNew m d s = some c ↔ c = ⟨m, d, s⟩ ∧ I32 m ∧ I64 (durAnchor m + d * 86400 + s) := by
  unfold durNew I32 I64
  by_cases h : -2 ^ 31 ≤ m ∧ m < 2 ^ 31 ∧
      -2 ^ 63 ≤ durAnchor m + d * 86400 + s ∧ durAnchor m + d * 86400 + s < 2 ^ 63
  · rw [if_pos h]
    constructor
    · intro e; cases e; exact ⟨rfl, ⟨h.1, h.2.1⟩, ⟨h.2.2.1, h.2.2.2⟩⟩
    · rintro ⟨rfl, _, _⟩; rfl
  · rw [if_neg h]
    constructor
    · intro e; cases e
    · rintro ⟨_, ⟨a, b⟩, ⟨c, d⟩⟩; exact absurd ⟨a, b, c, d⟩ h

/-- The components of a `Dur` are what `Dur::new` was given. -/
theorem durNew_components (m d s : Int) (c : Dur) (h : durNew m d s = some c) :
    c.months = m ∧ c.days = d ∧ c.secs = s := by
  have := ((durNew_iff m d s c).1 h).1; subst this; exact ⟨rfl, rfl, rfl⟩

theorem durAnchor_zero : durAnchor 0 = 0 := by decide

/-- The zero duration and every plain-seconds `i64` duration exist. -/
theorem durNew_secs (s : Int) (h : I64 s) : durNew 0 0 s = some ⟨0, 0, s⟩ := by
  rw [durNew_iff]; refine ⟨rfl, by unfold I32; decide, ?_⟩
  rw [durAnchor_zero]; simpa using h

/-! ### The `u64` word (`Dur::new` `:515`, `components` `:555`, `unpack_inline` `:571`)

Inline exactly when `|months| ≤ 4095`, `|days| ≤ 131071`, `|seconds| ≤ 2^31 − 1`:
`bits = (months as u64) << 51 | (days & (2^18−1)) << 33 | (seconds & (2^32−1)) << 1 | 1`.
`(x as u64) << 51` keeps the low 13 bits of `x`, so the word is the sum of the three
fields (disjoint) as a natural number below `2^64`. `unpack_inline` reads `b = bits as i64`
and returns `b >> 51`, `(b << 13) >> 46`, `(b << 31) >> 32` (arithmetic shifts): the
13/18/32-bit fields sign-extended, `sx13`/`sx18`/`sx32` below. Otherwise the word is an
`Arc<[i64; 3]>` pointer (bit 0 clear, `DurWord.heap` holding the array). -/

def inlineFits (m d s : Int) : Bool :=
  decide (m.natAbs ≤ 4095) && decide (d.natAbs ≤ 131071) && decide (s.natAbs ≤ 2147483647)

def packBits (m d s : Int) : Int :=
  m % 8192 * 2251799813685248 + d % 262144 * 8589934592 + s % 4294967296 * 2 + 1

def sx13 (x : Int) : Int := if x % 8192 < 4096 then x % 8192 else x % 8192 - 8192
def sx18 (x : Int) : Int := if x % 262144 < 131072 then x % 262144 else x % 262144 - 262144
def sx32 (x : Int) : Int := if x % 4294967296 < 2147483648 then x % 4294967296 else x % 4294967296 - 4294967296

/-- `Dur::unpack_inline` (`temporal.rs:571`). -/
def unpackInline (b : Int) : Int × Int × Int :=
  (sx13 (b / 2251799813685248), sx18 (b / 8589934592), sx32 (b / 2))

inductive DurWord where
  | inline (bits : Int)
  | heap (c : Int × Int × Int)
deriving DecidableEq, Repr

/-- `Dur::new`'s word. -/
def durNewW (m d s : Int) : Option DurWord :=
  (durNew m d s).map fun _ => if inlineFits m d s then .inline (packBits m d s) else .heap (m, d, s)

/-- `Dur::components` (`temporal.rs:555`): bit 0 set ↦ `unpack_inline`, else the array. -/
def wordComponents : DurWord → Int × Int × Int
  | .inline b => unpackInline b
  | .heap c => c

/-- `Dur::inline_bits` (`temporal.rs:582`). -/
def inlineBits : DurWord → Option Int
  | .inline b => some b
  | .heap _ => none

/-- `Dur::from_inline_bits` (`temporal.rs:590`). -/
def fromInlineBits (b : Int) : DurWord := .inline b

/-- The packed word is a `u64` with the tag bit 0 set. -/
theorem packBits_word (m d s : Int) :
    0 ≤ packBits m d s ∧ packBits m d s < 2 ^ 64 ∧ packBits m d s % 2 = 1 := by
  unfold packBits; omega

/-- **Packing round-trips**: every inline-sized triple unpacks to itself. -/
theorem unpack_pack (m d s : Int) (h : inlineFits m d s = true) :
    unpackInline (packBits m d s) = (m, d, s) := by
  simp only [inlineFits, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  unfold unpackInline packBits sx13 sx18 sx32
  simp only [Prod.mk.injEq]
  refine ⟨?_, ?_, ?_⟩ <;> split <;> omega

/-- **`components ∘ new = id`**, in both forms (the unit test
`components_round_trip_in_both_forms`, for every input). -/
theorem components_new (m d s : Int) (w : DurWord) (h : durNewW m d s = some w) :
    wordComponents w = (m, d, s) := by
  unfold durNewW at h
  cases hn : durNew m d s with
  | none => simp [hn] at h
  | some c =>
    simp only [hn, Option.map_some, Option.some.injEq] at h
    subst h
    by_cases hf : inlineFits m d s = true
    · simp only [hf, ite_true, wordComponents]; exact unpack_pack m d s hf
    · simp only [hf, wordComponents]; rfl

/-- `from_inline_bits ∘ inline_bits = id`: the attribute store's round trip. -/
theorem fromInlineBits_inlineBits (w : DurWord) (b : Int) (h : inlineBits w = some b) :
    fromInlineBits b = w := by
  cases w with
  | inline b' => cases h; rfl
  | heap _ => cases h

/-- A word is inline exactly when its components fit (so equal durations have equal
forms, and `Eq`/`Hash` on components agree with the word). -/
theorem durNewW_inline_iff (m d s : Int) (w : DurWord) (h : durNewW m d s = some w) :
    (inlineBits w).isSome = inlineFits m d s := by
  unfold durNewW at h
  cases hn : durNew m d s with
  | none => simp [hn] at h
  | some c =>
    simp only [hn, Option.map_some, Option.some.injEq] at h
    subst h
    by_cases hf : inlineFits m d s = true <;> simp [hf, inlineBits]

/-- `PartialEq for Dur` (`temporal.rs:633`): equal components. -/
def durWordEq (a b : DurWord) : Bool := wordComponents a == wordComponents b

/-- **`Dur::eq` is equality of the built triples**, whichever form each word took. -/
theorem durWordEq_new (m d s m' d' s' : Int) (w w' : DurWord)
    (h : durNewW m d s = some w) (h' : durNewW m' d' s' = some w') :
    durWordEq w w' = true ↔ (m, d, s) = (m', d', s') := by
  unfold durWordEq
  rw [components_new m d s w h, components_new m' d' s' w' h']
  simp

/-! ## `construct_duration` (`temporal.rs:663`) -/

/-- `construct_duration`, `Err` ↦ `none`: years into months, weeks into days,
hours/minutes into seconds, each with `checked_*`, then `Dur::new`. -/
def constructDuration (years months weeks days hours minutes seconds : Int) : Option Dur := do
  let tm ← chkMul years 12 >>= fun y => chkAdd y months
  let td ← chkMul weeks 7 >>= fun w => chkAdd w days
  let ts ← chkMul hours 3600 >>= (fun h => chkMul minutes 60 >>= fun m => chkAdd h m)
    >>= fun s => chkAdd s seconds
  durNew tm td ts

/-- When no step overflows, the components are the normalised sums. -/
theorem constructDuration_eq (Y Mo W D H Mi S : Int) (h1 : I64 (Y * 12)) (h2 : I64 (Y * 12 + Mo))
    (h3 : I64 (W * 7)) (h4 : I64 (W * 7 + D)) (h5 : I64 (H * 3600)) (h6 : I64 (Mi * 60))
    (h7 : I64 (H * 3600 + Mi * 60)) (h8 : I64 (H * 3600 + Mi * 60 + S)) :
    constructDuration Y Mo W D H Mi S = durNew (Y * 12 + Mo) (W * 7 + D) (H * 3600 + Mi * 60 + S) := by
  unfold constructDuration chkMul chkAdd
  simp only [chk_of _ h1, chk_of _ h2, chk_of _ h3, chk_of _ h4, chk_of _ h5, chk_of _ h6,
    chk_of _ h7, chk_of _ h8, bind, Option.bind]

/-- A constructed duration keeps each unit apart: `duration({days: 31})` is not
`duration({months: 1})` (they were the same `i64` before #3610). -/
theorem construct_units_distinct :
    constructDuration 0 0 0 31 0 0 0 = some ⟨0, 31, 0⟩ ∧
    constructDuration 0 1 0 0 0 0 0 = some ⟨1, 0, 0⟩ := by decide +kernel

/-! ## The legacy epoch offset: `duration_to_legacy` / `duration_from_legacy` -/

/-- `duration_to_legacy` (`temporal.rs:745`): the anchor of the months plus days and
seconds, wrapping. `1970i32.saturating_add(months.div_euclid(12) as i32)` is exact for the
`i32` months of every `Dur` (`durNew`), hence `durAnchor`. -/
def toLegacy (c : Dur) : Int := wrap64 (durAnchor c.months + c.days * 86400 + c.secs)

/-- The components `duration_from_legacy` (`temporal.rs:719`) first tries: the legacy
decomposition (`civil_from_days` as Rust returns it, `i32` year; the year difference in
`i64`), with the remaining seconds split into days and seconds (`remaining ≥ 0`, so Rust's
`/` and `%` are Lean's). -/
def legacyParts (x : Int) : Int × Int × Int :=
  let days := x / 86400
  let tod := x % 86400
  let (y, m, d) := civilFromDays days
  let rem := (d - 1) * 86400 + tod
  ((y - 1970) * 12 + m - 1, rem / 86400, rem % 86400)

/-- `duration_from_legacy` (`temporal.rs:719`): the decomposition when `Dur::new` accepts it
and it reproduces the offset, else plain seconds (else days and seconds). -/
def fromLegacy (x : Int) : Option Dur :=
  let fallback := match durNew 0 0 x with
    | some c => some c
    | none => durNew 0 (x / 86400) (x % 86400)
  match durNew (legacyParts x).1 (legacyParts x).2.1 (legacyParts x).2.2 with
  | some c => if toLegacy c = x then some c else fallback
  | none => fallback

/-- **A stored legacy duration always loads, and keeps its length**: for every `i64`,
`duration_from_legacy` succeeds and `duration_to_legacy` gives the stored value back (so a
compact reply of a value loaded from an old RDB is what the old build replied). -/
theorem toLegacy_fromLegacy (x : Int) (hx : I64 x) :
    ∃ c, fromLegacy x = some c ∧ toLegacy c = x := by
  have hs := durNew_secs x hx
  have hl : toLegacy ⟨0, 0, x⟩ = x := by
    unfold toLegacy; simp only [durAnchor_zero, Int.zero_mul, Int.zero_add]; exact wrap64_id x hx
  unfold fromLegacy
  simp only [hs]
  split
  · rename_i c _
    by_cases h : toLegacy c = x
    · exact ⟨c, by simp [h], h⟩
    · exact ⟨⟨0, 0, x⟩, by simp [h], hl⟩
  · exact ⟨⟨0, 0, x⟩, rfl, hl⟩

/-- When the decomposition is accepted, the loaded value is it. -/
theorem fromLegacy_parts (x : Int) (c : Dur)
    (h : durNew (legacyParts x).1 (legacyParts x).2.1 (legacyParts x).2.2 = some c)
    (h2 : toLegacy c = x) : fromLegacy x = some c := by
  unfold fromLegacy; simp only [h, h2, ite_true]

/-- `toLegacy` of a valid duration does not wrap: it is the exact offset. -/
theorem toLegacy_exact (m d s : Int) (c : Dur) (h : durNew m d s = some c) :
    toLegacy c = durAnchor m + d * 86400 + s := by
  obtain ⟨rfl, _, h2⟩ := (durNew_iff m d s c).1 h
  exact wrap64_id _ h2

/-- The old encoding, by example: `duration({days: -1})` was stored as the instant
1969-12-31 (`P-1Y11M30D` after decomposition); it still loads as that, so its old date
arithmetic is preserved, and the compact reply is `-86400` as before. -/
theorem legacy_neg_day :
    toLegacy ⟨0, -1, 0⟩ = -86400 ∧ fromLegacy (-86400) = some ⟨-1, 30, 0⟩ := by decide +kernel

/-! ## Range checks (#2746, #3575): `check_date_range`

`MAX_DATE_YEAR = i32::MAX - 2000` (`temporal.rs:693`). `MIN_DATE_DAYS`/`MAX_DATE_DAYS` are
`days_from_civil(∓MAX_DATE_YEAR, 1 / 12, 1 / 31)` (`temporal.rs:698-699`). -/

theorem minDateDays_val : minDateDays = -784352284982 := by decide +kernel
theorem maxDateDays_val : maxDateDays = 784350846291 := by decide +kernel

theorem daysFromCivil_year_mono (y Y : Int) (h : y < Y) :
    daysFromCivil y 12 31 < daysFromCivil Y 1 1 := by
  unfold daysFromCivil; simp only; omega

theorem daysFromCivil_in_year (y m d : Int) (hv : ValidDate y m d) :
    daysFromCivil y 1 1 ≤ daysFromCivil y m d ∧ daysFromCivil y m d ≤ daysFromCivil y 12 31 := by
  have hdim : daysInMonth y m ≤ 31 := by
    unfold daysInMonth; split <;> (try split) <;> (try split) <;> omega
  obtain ⟨h1, h2, h3, h4⟩ := hv
  unfold daysFromCivil; simp only
  constructor <;> split <;> split <;> omega

/-- **`check_date_range` accepts exactly the timestamps whose civil year is within
`±MAX_DATE_YEAR`** (the year here is the exact `civil_from_days_wide` one, no `i32` cut). -/
theorem checkDateRange_iff (ts : Int) :
    (checkDateRange ts).isSome = true ↔
      -MAX_DATE_YEAR ≤ (civilFromDaysI (ts / 86400)).1 ∧ (civilFromDaysI (ts / 86400)).1 ≤ MAX_DATE_YEAR := by
  have hrt := dfc_civil (ts / 86400)
  have hv := civil_valid (ts / 86400)
  generalize civilFromDaysI (ts / 86400) = c at *
  obtain ⟨y, m, d⟩ := c
  simp only at hrt hv ⊢
  have l1 := daysFromCivil_in_year y m d hv
  have a1 := daysFromCivil_year_mono y (-MAX_DATE_YEAR)
  have a2 := daysFromCivil_year_mono MAX_DATE_YEAR y
  have b1 := daysFromCivil_in_year (-MAX_DATE_YEAR) 12 31 ⟨by omega, by omega, by omega, by decide⟩
  have b2 := daysFromCivil_in_year MAX_DATE_YEAR 1 1 ⟨by omega, by omega, by omega, by decide⟩
  have b3 := daysFromCivil_in_year (-MAX_DATE_YEAR) 1 1 ⟨by omega, by omega, by omega, by decide⟩
  have b4 := daysFromCivil_in_year MAX_DATE_YEAR 12 31 ⟨by omega, by omega, by omega, by decide⟩
  have c1 := daysFromCivil_year_mono (-MAX_DATE_YEAR) y
  have c2 := daysFromCivil_year_mono y MAX_DATE_YEAR
  have hM : MAX_DATE_YEAR = 2147481647 := rfl
  have e1 : minDateDays = daysFromCivil (-MAX_DATE_YEAR) 1 1 := rfl
  have e2 : maxDateDays = daysFromCivil MAX_DATE_YEAR 12 31 := rfl
  have k1 : -MAX_DATE_YEAR ≤ y → minDateDays ≤ daysFromCivil y 1 1 := by
    intro hy
    rcases Int.lt_or_eq_of_le hy with h | h
    · have := c1 h; omega
    · rw [e1, ← h]; exact Int.le_refl _
  have k2 : y ≤ MAX_DATE_YEAR → daysFromCivil y 12 31 ≤ maxDateDays := by
    intro hy
    rcases Int.lt_or_eq_of_le hy with h | h
    · have := c2 h; omega
    · rw [e2, h]; exact Int.le_refl _
  have k3 := a1
  have k4 := a2
  rw [← e1] at k3
  rw [← e2] at k4
  unfold checkDateRange
  by_cases h : minDateDays ≤ ts / 86400 ∧ ts / 86400 ≤ maxDateDays
  · simp only [h, and_self, ite_true, Option.isSome_some, true_iff]
    omega
  · simp only [h, ite_false, Option.isSome_none, Bool.false_eq_true, false_iff]
    omega

/-- The boundary: the last supported day is accepted, the next one is not. -/
theorem checkDateRange_boundary :
    (checkDateRange (maxDateDays * 86400)).isSome = true ∧
    checkDateRange ((maxDateDays + 1) * 86400) = none ∧
    (checkDateRange (minDateDays * 86400)).isSome = true ∧
    checkDateRange ((minDateDays - 1) * 86400) = none := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp [checkDateRange, minDateDays_val, maxDateDays_val] <;> omega

/-- A timestamp that passes `check_date_range` is an `i64` (so the `i64::try_from` before it
never decides anything on its own). -/
theorem checkDateRange_I64 (ts : Int) (h : (checkDateRange ts).isSome = true) : I64 ts := by
  unfold checkDateRange at h
  split at h
  · rename_i hr; rw [minDateDays_val, maxDateDays_val] at hr; unfold I64; omega
  · simp at h

/-! ## Date ± Duration -/

/-- The arithmetic of `shift_months` (`value.rs:725`): move `ts` by whole years and
months, returning (days since epoch, time of day). A day past the end of the target month
rolls over into the next month (`2020-03-31` minus one month is `2020-03-02`), as C's
`timegm` does. -/
def shiftMonths (ts years months : Int) : Int × Int :=
  let days := ts / 86400
  let tod := ts % 86400
  let (y, m, d) := civilFromDays days
  let newYear := y + years
  let nmr := m + months
  let adjYear := newYear + (nmr - 1) / 12
  let adjMonth := (nmr - 1) % 12 + 1
  let maxDay := daysInMonth adjYear adjMonth
  let (fy, fm, fd) :=
    if d > maxDay then
      let overflow := d - maxDay
      let nm := adjMonth + 1
      if nm > 12 then (adjYear + 1, 1, overflow) else (adjYear, nm, overflow)
    else (adjYear, adjMonth, d)
  (daysFromCivil fy fm fd, tod)

/-- `shift_months` depends on `years` and `months` only through the total month count
`12·years + months` (the Euclidean `div`/`rem` re-normalise it). -/
theorem shiftMonths_total (ts years months : Int) :
    shiftMonths ts years months = shiftMonths ts 0 (years * 12 + months) := by
  unfold shiftMonths
  simp only
  generalize civilFromDays (ts / 86400) = c
  obtain ⟨y, m, d⟩ := c
  simp only
  have eY : y + years + (m + months - 1) / 12 = y + 0 + (m + (years * 12 + months) - 1) / 12 := by
    omega
  have eM : (m + months - 1) % 12 + 1 = (m + (years * 12 + months) - 1) % 12 + 1 := by omega
  rw [eY, eM]

/-- `shift_months` (`value.rs:725`) since #3610, `Err` ↦ `none`: the same arithmetic with
`i32::try_from` on the years and `checked_add` on every `i32` year step, and the refusal of
a final year `i32::MIN`. There is no `MAX_DATE_YEAR` cutoff any more: the day offset that
follows may bring a far month shift back into range, and `shift_timestamp` checks the end. -/
def shiftMonthsC (ts totalMonths : Int) : Option (Int × Int) :=
  match chk32 (totalMonths / 12) with
  | none => none
  | some years =>
    let months := totalMonths % 12
    let days := ts / 86400
    let tod := ts % 86400
    match civilFromDays days with
    | (y, m, d) =>
      match chk32 (y + years) with
      | none => none
      | some newYear =>
        let nmr := m + months
        match chk32 (newYear + (nmr - 1) / 12) with
        | none => none
        | some adjYear =>
          let adjMonth := (nmr - 1) % 12 + 1
          let maxDay := daysInMonth adjYear adjMonth
          let fin : Option (Int × Int × Int) :=
            if d > maxDay then
              let overflow := d - maxDay
              let nm := adjMonth + 1
              if nm > 12 then
                (match chk32 (adjYear + 1) with
                 | none => none
                 | some ny => some (ny, 1, overflow))
              else some (adjYear, nm, overflow)
            else some (adjYear, adjMonth, d)
          match fin with
          | none => none
          | some (fy, fm, fd) => if fy = -2 ^ 31 then none else some (daysFromCivil fy fm fd, tod)

/-- The year `shift_months` reaches before the roll-over (`adj_year`). -/
def adjYear (ts totalMonths : Int) : Int :=
  (civilFromDays (ts / 86400)).1 + ((civilFromDays (ts / 86400)).2.1 + totalMonths - 1) / 12

theorem chk32_none (x : Int) (h : ¬ I32 x) : chk32 x = none := by
  unfold chk32; unfold I32 at h
  split
  · rename_i hh; simp only [Bool.and_eq_true, decide_eq_true_eq] at hh; exact absurd hh h
  · rfl

theorem toI32_range (x : Int) : I32 (toI32 x) := by unfold toI32 I32; omega

/-- **When `shift_months` answers, it is the arithmetic**, and every `i32` year it formed
(the years, `adj_year`) was exact: no wrapped year. -/
theorem shiftMonthsC_some (ts t : Int) (p : Int × Int) (h : shiftMonthsC ts t = some p) :
    p = shiftMonths ts 0 t ∧ I32 (t / 12) ∧ I32 (adjYear ts t) := by
  unfold shiftMonthsC at h
  have h1 : I32 (t / 12) := Classical.byContradiction fun hc => by
    rw [chk32_none _ hc] at h; cases h
  rw [chk32_of _ h1] at h
  simp only at h
  have hadj : adjYear ts t = (civilFromDays (ts / 86400)).1 + ((civilFromDays (ts / 86400)).2.1 + t - 1) / 12 := rfl
  obtain ⟨⟨y, m, d⟩, hc⟩ : ∃ c, civilFromDays (ts / 86400) = c := ⟨_, rfl⟩
  rw [hc] at h hadj
  simp only at h hadj
  have h2 : I32 (y + t / 12) := Classical.byContradiction fun hc => by
    rw [chk32_none _ hc] at h; cases h
  rw [chk32_of _ h2] at h
  simp only at h
  have h3 : I32 (y + t / 12 + (m + t % 12 - 1) / 12) := Classical.byContradiction fun hc => by
    rw [chk32_none _ hc] at h; cases h
  rw [chk32_of _ h3] at h
  simp only at h
  have eA : y + t / 12 + (m + t % 12 - 1) / 12 = y + 0 + (m + t - 1) / 12 := by omega
  have eM : (m + t % 12 - 1) % 12 + 1 = (m + t - 1) % 12 + 1 := by omega
  refine ⟨?_, h1, by rw [hadj]; rw [eA] at h3; simpa using h3⟩
  rw [eA, eM] at h
  unfold shiftMonths
  simp only [hc]
  generalize y + 0 + (m + t - 1) / 12 = A at h ⊢
  generalize (m + t - 1) % 12 + 1 = B at h ⊢
  by_cases hd : d > daysInMonth A B
  · rw [if_pos hd] at h; rw [if_pos hd]
    by_cases hn : B + 1 > 12
    · rw [if_pos hn] at h; rw [if_pos hn]
      have h4 : I32 (A + 1) := Classical.byContradiction fun hc => by
        rw [chk32_none _ hc] at h; cases h
      rw [chk32_of _ h4] at h
      simp only at h
      split at h
      · cases h
      · cases h; rfl
    · rw [if_neg hn] at h; rw [if_neg hn]
      simp only at h
      split at h
      · cases h
      · cases h; rfl
  · rw [if_neg hd] at h; rw [if_neg hd]
    simp only at h
    split at h
    · cases h
    · cases h; rfl

/-- **`shift_months` answers whenever the year stays in the supported range** (the
pre-#3610 guard's accepting side, `Pre3610.shiftMonthsR_eq`): with `|adj_year| ≤
MAX_DATE_YEAR + 1` and an `i32` year count every checked step succeeds. -/
theorem shiftMonthsC_eq (ts t : Int) (ht : I32 (t / 12))
    (h : -(MAX_DATE_YEAR + 1) ≤ adjYear ts t ∧ adjYear ts t ≤ MAX_DATE_YEAR + 1) :
    shiftMonthsC ts t = some (shiftMonths ts 0 t) := by
  have hcy := toI32_range (civilFromDaysI (ts / 86400)).1
  have hm := (civil_valid (ts / 86400))
  unfold adjYear at h
  unfold shiftMonthsC shiftMonths
  rw [chk32_of _ ht]
  simp only [bind, Option.bind]
  unfold civilFromDays at *
  generalize civilFromDaysI (ts / 86400) = c at *
  obtain ⟨y, m, d⟩ := c
  simp only at h hcy hm ⊢
  obtain ⟨hm1, hm2, hd1, hd2⟩ := hm
  unfold MAX_DATE_YEAR at h
  have eA : toI32 y + t / 12 + (m + t % 12 - 1) / 12 = toI32 y + 0 + (m + t - 1) / 12 := by omega
  have eM : (m + t % 12 - 1) % 12 + 1 = (m + t - 1) % 12 + 1 := by omega
  rw [chk32_of (toI32 y + t / 12) (by unfold I32 at *; omega)]
  simp only
  rw [chk32_of (toI32 y + t / 12 + (m + t % 12 - 1) / 12) (by unfold I32 at *; omega)]
  simp only
  rw [eA, eM]
  generalize hA : toI32 y + 0 + (m + t - 1) / 12 = A at *
  generalize hB : (m + t - 1) % 12 + 1 = B at *
  by_cases hd : d > daysInMonth A B
  · simp only [hd, ite_true]
    by_cases hn : B + 1 > 12
    · simp only [hn, ite_true]
      rw [chk32_of (A + 1) (by unfold I32; omega)]
      simp only
      rw [if_neg (by omega)]
    · simp only [hn, ite_false]
      rw [if_neg (by omega)]
  · simp only [hd, ite_false]
    rw [if_neg (by omega)]

/-- `shift_timestamp` (`value.rs:665`): months through `shift_months` (with the sign), then
days and seconds added in `i128`, `i64::try_from`, `check_date_range`. -/
def shiftTs (ts : Int) (c : Dur) (subtract : Bool) : Option Int :=
  let sign : Int := if subtract then -1 else 1
  match shiftMonthsC ts (sign * c.months) with
  | none => none
  | some (days, tod) => (chk ((days + sign * c.days) * 86400 + tod + sign * c.secs)).bind checkDateRange

/-- `add_duration_to_timestamp` (`value.rs:645`). -/
def addDur (ts : Int) (c : Dur) : Option Int := shiftTs ts c false
/-- `sub_duration_from_timestamp` (`value.rs:653`). -/
def subDur (ts : Int) (c : Dur) : Option Int := shiftTs ts c true

/-- The arithmetic of `add_duration_to_timestamp`, unchecked. -/
def addDurRaw (ts : Int) (c : Dur) : Int :=
  (shiftMonths ts 0 c.months).1 * 86400 + (shiftMonths ts 0 c.months).2 + c.days * 86400 + c.secs
/-- The arithmetic of `sub_duration_from_timestamp`, unchecked. -/
def subDurRaw (ts : Int) (c : Dur) : Int :=
  (shiftMonths ts 0 (-c.months)).1 * 86400 + (shiftMonths ts 0 (-c.months)).2 - c.days * 86400 - c.secs

/-- **`d - dur = d + (-dur)`, for every duration** (the checked functions, not only the
arithmetic): `shift_timestamp` negates all three components. -/
theorem subDur_eq_add_neg (ts : Int) (c : Dur) :
    subDur ts c = addDur ts ⟨-c.months, -c.days, -c.secs⟩ := by
  unfold subDur addDur shiftTs
  simp only [ite_true, Bool.false_eq_true, ite_false]
  have e1 : -1 * c.months = 1 * -c.months := by omega
  rw [e1]
  split <;> (try rfl)
  congr 2; omega

theorem shiftTs_eq_raw (ts : Int) (c : Dur) (sub : Bool) (r : Int) (h : shiftTs ts c sub = some r) :
    r = (if sub then subDurRaw ts c else addDurRaw ts c) ∧ (checkDateRange r).isSome = true := by
  unfold shiftTs at h
  simp only at h
  generalize hsg : (if sub = true then (-1 : Int) else 1) = sg at h
  cases hs : shiftMonthsC ts (sg * c.months) with
  | none => rw [hs] at h; cases h
  | some p =>
    rw [hs] at h
    obtain ⟨days, tod⟩ := p
    simp only at h
    obtain ⟨hp, _, _⟩ := shiftMonthsC_some _ _ _ hs
    cases hk : chk ((days + sg * c.days) * 86400 + tod + sg * c.secs) with
    | none => rw [hk] at h; cases h
    | some v =>
      rw [hk] at h
      simp only [Option.bind] at h
      have hv : v = (days + sg * c.days) * 86400 + tod + sg * c.secs := by
        unfold chk at hk; split at hk <;> simp_all
      have hr : r = v := by unfold checkDateRange at h; split at h <;> simp_all
      have hc : (checkDateRange r).isSome = true := by rw [hr, h]; rfl
      refine ⟨?_, hc⟩
      rw [hr, hv]
      cases sub
      · simp only [Bool.false_eq_true, ite_false] at hsg ⊢
        subst hsg
        rw [Int.one_mul] at hp
        unfold addDurRaw; rw [← hp]; simp only; omega
      · simp only [ite_true] at hsg ⊢
        subst hsg
        unfold subDurRaw
        have : -1 * c.months = -c.months := by omega
        rw [this] at hp; rw [← hp]; simp only; omega

/-- When `shift_months` answers and the result is in range, the checked function is the
arithmetic (`i64::try_from` never decides on its own: `checkDateRange_I64`). -/
theorem addDur_eq_raw (ts : Int) (c : Dur)
    (h1 : (shiftMonthsC ts c.months).isSome = true)
    (h2 : (checkDateRange (addDurRaw ts c)).isSome = true) :
    addDur ts c = some (addDurRaw ts c) := by
  unfold addDur shiftTs
  simp only [Bool.false_eq_true, ite_false, Int.one_mul]
  cases hs : shiftMonthsC ts c.months with
  | none => simp [hs] at h1
  | some p =>
    obtain ⟨hp, _, _⟩ := shiftMonthsC_some _ _ _ hs
    obtain ⟨days, tod⟩ := p
    simp only
    have e : (days + c.days) * 86400 + tod + c.secs = addDurRaw ts c := by
      unfold addDurRaw; rw [← hp]; simp only; omega
    rw [e, chk_of _ (checkDateRange_I64 _ h2)]
    simp only [Option.bind]
    unfold checkDateRange at h2 ⊢; split <;> simp_all

/-- A date or datetime produced by `+`/`-` of a duration is inside the supported range. -/
theorem addDur_in_range (ts : Int) (c : Dur) (r : Int) (h : addDur ts c = some r) :
    -MAX_DATE_YEAR ≤ (civilFromDaysI (r / 86400)).1 ∧ (civilFromDaysI (r / 86400)).1 ≤ MAX_DATE_YEAR := by
  rw [← checkDateRange_iff]; exact (shiftTs_eq_raw ts c false r h).2

theorem subDur_in_range (ts : Int) (c : Dur) (r : Int) (h : subDur ts c = some r) :
    -MAX_DATE_YEAR ≤ (civilFromDaysI (r / 86400)).1 ∧ (civilFromDaysI (r / 86400)).1 ≤ MAX_DATE_YEAR := by
  rw [← checkDateRange_iff]; exact (shiftTs_eq_raw ts c true r h).2

/-- The subtraction arithmetic is the addition's with the months negated and the days and
seconds subtracted. -/
theorem subDur_eq_shift (ts : Int) (c : Dur) :
    subDurRaw ts c =
      (shiftMonths ts 0 (-c.months)).1 * 86400 + (shiftMonths ts 0 0).2 - c.days * 86400 - c.secs := by
  unfold subDurRaw shiftMonths; simp only

theorem shiftMonths_eq_of (ts a b t : Int) (h : a * 12 + b = t) :
    shiftMonths ts 0 t = shiftMonths ts a b := by
  rw [shiftMonths_total ts a b, h]

/-- A date value (`Value::Date`) for a real calendar date. -/
def dateTs (y m d : Int) : Int := daysFromCivil y m d * 86400

/-- A small civil-year range where every intermediate `i32` in the Rust code is exact. -/
def SmallYear (y : Int) : Prop := -1000000 ≤ y ∧ y ≤ 1000000

/-- **`(date + P{t}M) - P{t}M = date`** whenever the day of month is ≤ 28 (so the
roll-over in `shiftMonths` fires on neither side). -/
theorem add_sub_months (y m d t : Int) (hv : ValidDate y m d) (hd : d ≤ 28)
    (hy : SmallYear y) (ht : -1000000 ≤ t ∧ t ≤ 1000000) :
    subDurRaw (addDurRaw (dateTs y m d) ⟨t, 0, 0⟩) ⟨t, 0, 0⟩ = dateTs y m d := by
  have hc := civil_dfc y m d hv
  obtain ⟨hm1, hm2, hd1, _⟩ := hv
  unfold addDurRaw subDurRaw dateTs
  simp only [Int.zero_mul, Int.add_zero, Int.sub_zero]
  rw [shiftMonths_eq_of _ (t / 12) (t % 12) t (by omega)]
  rw [shiftMonths_eq_of _ (-(t / 12)) (-(t % 12)) (-t) (by omega)]
  unfold shiftMonths
  have e1 : daysFromCivil y m d * 86400 / 86400 = daysFromCivil y m d := by omega
  have e2 : daysFromCivil y m d * 86400 % 86400 = 0 := by omega
  unfold civilFromDays
  have i1 : I32 y := by unfold I32 SmallYear at *; omega
  simp only [e1, e2, hc, toI32_id _ i1]
  generalize hA : y + t / 12 + (m + t % 12 - 1) / 12 = A
  generalize hB : (m + t % 12 - 1) % 12 + 1 = B
  have hB1 : 1 ≤ B ∧ B ≤ 12 := by omega
  have hdimB : 28 ≤ daysInMonth A B := by
    unfold daysInMonth; split <;> (try split) <;> (try split) <;> omega
  have nof : ¬ (d > daysInMonth A B) := by omega
  simp only [nof, ite_false]
  have hvA : ValidDate A B d := ⟨hB1.1, hB1.2, hd1, by omega⟩
  have e3 : (daysFromCivil A B d * 86400 + 0) / 86400 = daysFromCivil A B d := by omega
  have e4 : (daysFromCivil A B d * 86400 + 0) % 86400 = 0 := by omega
  have i2 : I32 A := by unfold I32 SmallYear at *; omega
  simp only [e3, e4, civil_dfc A B d hvA, toI32_id _ i2]
  have eY : A + -(t / 12) + (B + -(t % 12) - 1) / 12 = y := by omega
  have eM : (B + -(t % 12) - 1) % 12 + 1 = m := by omega
  rw [eY, eM]
  have : ¬ (d > daysInMonth y m) := by
    have : 28 ≤ daysInMonth y m := by
      unfold daysInMonth; split <;> (try split) <;> (try split) <;> omega
    omega
  simp only [this, ite_false]; omega

/-- **W2-temporal-5 stays fixed (#2984).** `ts - P{t}M = ts + P{-t}M` for every timestamp
and month count (now for every duration and the checked functions: `subDur_eq_add_neg`). -/
theorem sub_eq_add_neg_months (ts t : Int) :
    subDurRaw ts ⟨t, 0, 0⟩ = addDurRaw ts ⟨-t, 0, 0⟩ := by
  unfold subDurRaw addDurRaw; simp only; omega

/-- **The values stored before #3610 keep their date arithmetic**: a legacy offset loaded
through its decomposition (`fromLegacy_parts`) moves a date exactly as the old code moved it
by the offset (`Pre3610.addDurRaw`), whenever the old `y - 1970` did not wrap. -/
theorem legacy_add_compat (ts x : Int) (hy : I32 ((civilFromDays (x / 86400)).1 - 1970)) :
    addDurRaw ts ⟨(legacyParts x).1, (legacyParts x).2.1, (legacyParts x).2.2⟩ =
      Pre3610.addDurRaw ts x := by
  unfold addDurRaw Pre3610.addDurRaw legacyParts Pre3610.decomposeDuration
  obtain ⟨⟨y, m, d⟩, hc⟩ : ∃ c, civilFromDays (x / 86400) = c := ⟨_, rfl⟩
  simp only [hc] at hy ⊢
  rw [toI32_id _ hy, Pre3610.shiftMonths_total ts (y - 1970) (m - 1)]
  have e : (y - 1970) * 12 + (m - 1) = (y - 1970) * 12 + m - 1 := by omega
  rw [e]
  have : shiftMonths ts 0 ((y - 1970) * 12 + m - 1) = Pre3610.shiftMonths ts 0 ((y - 1970) * 12 + m - 1) := rfl
  rw [this]
  have h0 : 0 ≤ x % 86400 := Int.emod_nonneg x (by decide)
  omega

/-! ### Examples (all evaluated by the kernel) -/

/-- `duration({months: -1})` and `duration({months: 1})`. -/
def minus1M : Dur := (constructDuration 0 (-1) 0 0 0 0 0).getD default
def plus1M : Dur := (constructDuration 0 1 0 0 0 0 0).getD default

/-- The repro of W2-temporal-5 (and #2984's flow test), on the checked functions: both are
2020-03-02, and `localdatetime('2020-03-31T10:00:00') - duration({months:1})` is
`2020-03-02T10:00:00`. -/
theorem sub_vs_add_neg :
    subDur (dateTs 2020 3 31) plus1M = some (dateTs 2020 3 2) ∧
    addDur (dateTs 2020 3 31) minus1M = some (dateTs 2020 3 2) ∧
    subDur (dateTs 2020 3 31 + 36000) plus1M = some (dateTs 2020 3 2 + 36000) := by decide +kernel

/-- **The shared C/Rust divergence `Pre3610.neg_days_encoding` is fixed by #3610**:
`duration({days:-1})` is the components (0, −1, 0), and `date('2020-03-01') +
duration({days:-1})` is 2020-02-29 (openCypher; before: 2020-03-02). -/
theorem neg_days_encoding :
    constructDuration 0 0 0 (-1) 0 0 0 = some ⟨0, -1, 0⟩ ∧
    addDur (dateTs 2020 3 1) ⟨0, -1, 0⟩ = some (dateTs 2020 2 29) := by
  decide +kernel

/-- Close to the limit a chain of additions is an error and not a wrapped year (the
`Pre3610.shift_past_limit_errors` step, end to end): two more years from the last
supported day. -/
theorem shift_past_limit_errors :
    addDur (maxDateDays * 86400) ⟨24, 0, 0⟩ = none ∧
    subDur (minDateDays * 86400) ⟨24, 0, 0⟩ = none := by decide +kernel

/-- No `MAX_DATE_YEAR` cutoff inside `shift_months` (the `value.rs:733` comment): a month
shift a million years out and back by days is the identity. -/
theorem far_shift_identity :
    addDur (dateTs 2020 1 1) ⟨12000000, -365242500, 0⟩ = some (dateTs 2020 1 1) := by decide +kernel

/-! ## `Time ± Duration` (`add_slow` `value.rs:973`, `Sub::sub` `value.rs:1085`)

`(t.rem_euclid(86400) ± secs.rem_euclid(86400)).rem_euclid(86400)` on the seconds
component; months and days are whole days and do not move a time of day. -/

def timeAddDur (t : Int) (c : Dur) : Int := (t % 86400 + c.secs % 86400) % 86400
def timeSubDur (t : Int) (c : Dur) : Int := (t % 86400 - c.secs % 86400) % 86400

theorem timeAddDur_range (t : Int) (c : Dur) : 0 ≤ timeAddDur t c ∧ timeAddDur t c < 86400 := by
  unfold timeAddDur; omega
theorem timeSubDur_range (t : Int) (c : Dur) : 0 ≤ timeSubDur t c ∧ timeSubDur t c < 86400 := by
  unfold timeSubDur; omega

/-- **W2-temporal-6 stays fixed.** `Time + Duration` is the time of day of the date
arithmetic `add_duration_to_timestamp t dur`, for every `t` and duration. -/
theorem timeAddDur_eq_tod (t : Int) (c : Dur) : timeAddDur t c = addDurRaw t c % 86400 := by
  unfold timeAddDur addDurRaw shiftMonths
  generalize civilFromDays (t / 86400) = c2
  obtain ⟨y2, m2, d2⟩ := c2
  simp only
  split <;> (try split) <;> omega

theorem timeAddDur_eq (t : Int) (c : Dur) : timeAddDur t c = (t + c.secs) % 86400 := by
  unfold timeAddDur; omega
theorem timeSubDur_eq (t : Int) (c : Dur) : timeSubDur t c = (t - c.secs) % 86400 := by
  unfold timeSubDur; omega

/-- `(t + dur) - dur = t` on times of day. -/
theorem timeSub_timeAdd (t : Int) (c : Dur) (h0 : 0 ≤ t) (h1 : t < 86400) :
    timeSubDur (timeAddDur t c) c = t := by
  unfold timeSubDur timeAddDur; omega

/-- `localtime('01:00') + duration({days:1})` and `± duration({months:1})` are
`localtime('01:00')`. -/
theorem time_normalised :
    timeAddDur 3600 ((constructDuration 0 0 0 1 0 0 0).getD default) = 3600 ∧
    timeAddDur 3600 plus1M = 3600 ∧ timeSubDur 3600 plus1M = 3600 := by decide +kernel

/-- #2987: `n * 7` for `n = 2635249153387078803` is out of i64, so `checked_mul` returns
`None` and `duration('P…W')` is an error. -/
theorem week_string_checked_none : chkMul 2635249153387078803 7 = none := by decide

/-! ## `Duration ± Duration`: `combine_durations` (`value.rs:701`) -/

/-- `combine_durations`: componentwise `checked_add`/`checked_sub`, then `Dur::new`. -/
def combineDurations (a b : Dur) (subtract : Bool) : Option Dur := do
  let op := fun (x y : Int) => if subtract then chk (x - y) else chk (x + y)
  let m ← op a.months b.months
  let d ← op a.days b.days
  let s ← op a.secs b.secs
  durNew m d s

/-- Addition of durations is commutative. -/
theorem combine_comm (a b : Dur) : combineDurations a b false = combineDurations b a false := by
  unfold combineDurations
  simp only [Bool.false_eq_true, ite_false]
  rw [Int.add_comm a.months, Int.add_comm a.days, Int.add_comm a.secs]

/-- `dur - dur` is the zero duration, for every duration. -/
theorem combine_sub_self (a : Dur) : combineDurations a a true = some ⟨0, 0, 0⟩ := by
  unfold combineDurations
  simp only [ite_true, Int.sub_self]
  have : chk 0 = some 0 := by decide
  simp only [this, bind, Option.bind]
  rw [durNew_secs 0 (by unfold I64; decide)]

/-- A sum of durations is exactly the componentwise sum (when it is a duration). -/
theorem chk_some (x r : Int) (h : chk x = some r) : r = x := by
  unfold chk at h; split at h <;> simp_all

theorem combine_add_eq (a b c : Dur) (h : combineDurations a b false = some c) :
    c = ⟨a.months + b.months, a.days + b.days, a.secs + b.secs⟩ := by
  unfold combineDurations at h
  simp only [Bool.false_eq_true, ite_false] at h
  cases h1 : chk (a.months + b.months) with
  | none => simp [h1, bind, Option.bind] at h
  | some m =>
  cases h2 : chk (a.days + b.days) with
  | none => simp [h1, h2, bind, Option.bind] at h
  | some d =>
  cases h3 : chk (a.secs + b.secs) with
  | none => simp [h1, h2, h3, bind, Option.bind] at h
  | some s =>
  simp only [h1, h2, h3, bind, Option.bind] at h
  rw [((durNew_iff _ _ _ _).1 h).1, chk_some _ _ h1, chk_some _ _ h2, chk_some _ _ h3]

/-! ## Ordering: `duration_cmp` (`value.rs:685`) -/

/-- `Ord::cmp` on integers (`i128`/`i64`). -/
def cmpI (x y : Int) : Ordering := if x < y then .lt else if x = y then .eq else .gt

/-- The approximate length in seconds (a month is 2 629 746 s = 30.436875 days). -/
def durApprox (c : Dur) : Int := c.months * 2629746 + c.days * 86400 + c.secs

/-- `duration_cmp`: by approximate length, then the components lexicographically. -/
def durationCmp (a b : Dur) : Ordering :=
  (cmpI (durApprox a) (durApprox b)).then
    ((cmpI a.months b.months).then ((cmpI a.days b.days).then (cmpI a.secs b.secs)))

theorem then_eq_iff (o p : Ordering) : o.then p = .eq ↔ o = .eq ∧ p = .eq := by
  cases o <;> cases p <;> decide

theorem then_swap (o p : Ordering) : (o.then p).swap = o.swap.then p.swap := by
  cases o <;> cases p <;> decide

theorem cmpI_eq_iff (x y : Int) : cmpI x y = .eq ↔ x = y := by
  unfold cmpI
  by_cases h1 : x < y
  · rw [if_pos h1]; constructor <;> intro h <;> first | (exfalso; omega) | cases h
  · rw [if_neg h1]
    by_cases h2 : x = y
    · rw [if_pos h2]; exact ⟨fun _ => h2, fun _ => rfl⟩
    · rw [if_neg h2]; constructor <;> intro h <;> first | (exfalso; omega) | cases h

theorem cmpI_swap (x y : Int) : cmpI y x = (cmpI x y).swap := by
  unfold cmpI
  by_cases h1 : x < y
  · rw [if_pos h1, if_neg (by omega), if_neg (by omega)]; rfl
  · rw [if_neg h1]
    by_cases h2 : x = y
    · rw [if_pos h2, if_neg (by omega), if_pos h2.symm]; rfl
    · rw [if_neg h2, if_pos (by omega)]; rfl

/-- **The order agrees with equality** (`Eq` is on components, `Dur::eq` `temporal.rs:633`). -/
theorem durationCmp_eq_iff (a b : Dur) : durationCmp a b = .eq ↔ a = b := by
  obtain ⟨m1, d1, s1⟩ := a
  obtain ⟨m2, d2, s2⟩ := b
  simp only [durationCmp, then_eq_iff, cmpI_eq_iff, Dur.mk.injEq]
  constructor
  · rintro ⟨_, h1, h2, h3⟩; exact ⟨h1, h2, h3⟩
  · rintro ⟨rfl, rfl, rfl⟩; exact ⟨rfl, rfl, rfl, rfl⟩

/-- Antisymmetry: swapping the operands reverses the result. -/
theorem durationCmp_swap (a b : Dur) : durationCmp b a = (durationCmp a b).swap := by
  simp only [durationCmp, then_swap, cmpI_swap (durApprox a), cmpI_swap a.months, cmpI_swap a.days,
    cmpI_swap a.secs]

/-- `duration({months: 1})` sorts after `duration({days: 30})` and before
`duration({days: 31})`; they are three distinct values. -/
theorem durationCmp_examples :
    durationCmp ⟨1, 0, 0⟩ ⟨0, 30, 0⟩ = .gt ∧ durationCmp ⟨1, 0, 0⟩ ⟨0, 31, 0⟩ = .lt := by decide

/-- `date` order is calendar order within a month. -/
theorem dateTs_mono_day (y m d : Int) : dateTs y m d < dateTs y m (d + 1) := by
  unfold dateTs
  rw [daysFromCivil_day y m d, daysFromCivil_day y m (d + 1)]; omega

theorem temporal_order_total (a b : Int) : a ≤ b ∨ b ≤ a := Int.le_total a b
theorem temporal_order_trans (a b c : Int) (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c := Int.le_trans h1 h2
theorem temporal_order_antisymm (a b : Int) (h1 : a ≤ b) (h2 : b ≤ a) : a = b := Int.le_antisymm h1 h2

end FalkorTemporal
