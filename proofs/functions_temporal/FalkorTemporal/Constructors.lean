import FalkorTemporal.Duration
/-!
# Temporal constructors and formatting: totality and injectivity

| here | there |
| --- | --- |
| `chronoYearOk`      | chrono 0.4.45 `MIN_YEAR = -262143`, `MAX_YEAR = 262142` (`naive/date/mod.rs:2446`) |
| `tdDaysOk`          | chrono `TimeDelta::days` panics unless `days * 86400 ∈ [MIN.secs, MAX.secs]` (`time_delta.rs:86,150`) |
| `Outcome`           | `Result<NaiveDate, String>` plus the panic that kills the server |
| `weekPath`          | `date_from_components`, `week` branch — `temporal.rs:73-86` |
| `quarterPath`       | `date_from_components`, `quarter` branch — `temporal.rs:89-96` |
| `sliceOk`           | `&rest[..2]` in `parse_week_date` — `temporal.rs:248` (byte slice of a UTF-8 `str`) |
| `formatDateDigits`  | `format_date` + `write_date_into` — `value.rs:274`, `value.rs:803` |
-/
namespace FalkorTemporal

def chronoYearOk (y : Int) : Bool := decide (-262143 ≤ y) && decide (y ≤ 262142)
def tdDaysOk (n : Int) : Bool :=
  decide (-9223372036854776 ≤ n * 86400) && decide (n * 86400 ≤ 9223372036854775)

inductive Outcome where
  | panic (why : String)
  | err
  | ok (day : Int)
  deriving DecidableEq, Repr

/-- `NaiveDate + TimeDelta` (chrono `Add` impl): panics when the result leaves chrono's range. -/
def dateAddDays (day n : Int) : Outcome :=
  if !tdDaysOk n then .panic "TimeDelta::days out of bounds"
  else if chronoYearOk (civilFromDaysI (day + n)).1 then .ok (day + n)
  else .panic "`NaiveDate + TimeDelta` overflowed"

/-- `date_from_components`, week branch (`temporal.rs:73-86`) with i64 arithmetic as
`Int` (release wraps `(week - 1) * 7`; debug panics — both unmodelled beyond `tdDaysOk`). -/
def weekPath (year week dow : Int) : Outcome :=
  if ¬ (0 ≤ dow ∧ dow ≤ 6) then .err
  else if !chronoYearOk year then .err
  else
    let jan4 := daysFromCivil year 1 4
    -- weekday of jan4, Monday = 0 (1970-01-01 was a Thursday)
    let wd := (jan4 + 3) % 7
    let wk1Mon := jan4 - wd            -- `jan4 - Duration::days(wd)`: never out of range
    match dateAddDays wk1Mon ((week - 1) * 7) with
    | .ok tm => dateAddDays tm (if dow = 0 then 6 else dow - 1)
    | o => o

/-- `date_from_components`, quarter branch (`temporal.rs:89-96`): uses `checked_add_signed`
for the date but builds the `TimeDelta` with the *panicking* `Duration::days`. -/
def quarterPath (year quarter doq : Int) : Outcome :=
  let qsm := (quarter - 1) * 3 + 1
  if ¬ (1 ≤ qsm ∧ qsm ≤ 12) ∨ !chronoYearOk year then .err
  else if !tdDaysOk (doq - 1) then .panic "TimeDelta::days out of bounds"
  else
    let base := daysFromCivil year qsm 1
    if chronoYearOk (civilFromDaysI (base + (doq - 1))).1 then .ok (base + (doq - 1)) else .err

/-- **Bug (server crash, confirmed live)**: `RETURN date({week: 10000000000})` panics in
`NaiveDate + TimeDelta` (`temporal.rs:83`). C: `year must be specified` (with a year:
an error, no crash). -/
theorem week_panics :
    weekPath 1970 10000000000 1 = .panic "`NaiveDate + TimeDelta` overflowed" := by decide +kernel

/-- **Bug (server crash, confirmed live)**: a week count beyond ±1.5·10¹⁰ panics earlier,
in `TimeDelta::days` (`date({week: -9223372036854775807})` wraps to that in release). -/
theorem week_panics_delta :
    weekPath 1970 20000000000 1 = .panic "TimeDelta::days out of bounds" := by decide +kernel

/-- **Bug (server crash, confirmed live)**: `date({quarter: 2, dayOfQuarter: 9223372036854775807})`. -/
theorem quarter_panics :
    quarterPath 1970 2 9223372036854775807 = .panic "TimeDelta::days out of bounds" := by
  decide +kernel

/-- Totality on a sane domain: for `|week| ≤ 10⁶` and years well inside chrono's range the
week branch never panics — the missing guard is a range check on `week`
(C: `valid values 1 - 53`). -/
theorem weekPath_total (year week dow : Int) (hy : -200000 ≤ year ∧ year ≤ 200000)
    (hw : -1000000 ≤ week ∧ week ≤ 1000000) :
    ∀ w, weekPath year week dow ≠ .panic w := by
  intro w
  unfold weekPath dateAddDays
  -- day numbers: jan4 of `year` and ±7·10⁶ days stay within ±220000 years
  have hj := dfc_civil (daysFromCivil year 1 4)
  have bound : ∀ z : Int, -90000000 ≤ z ∧ z ≤ 90000000 →
      -262143 ≤ (civilFromDaysI z).1 ∧ (civilFromDaysI z).1 ≤ 262142 := by
    intro z hz
    have h0 := Int.emod_nonneg (z + 719468) (by decide : (146097:Int) ≠ 0)
    have h1 := Int.emod_lt_of_pos (z + 719468) (by decide : (0:Int) < 146097)
    have hy := yoe_bounds _ h0 h1
    unfold civilFromDaysI; dsimp only
    split <;> omega
  have hjan : -82000000 ≤ daysFromCivil year 1 4 ∧ daysFromCivil year 1 4 ≤ 82000000 := by
    unfold daysFromCivil; dsimp only; simp only [show ¬ ((1:Int) > 2) by decide, ite_false,
      show (1:Int) ≤ 2 by decide, ite_true]; omega
  have hwd0 := Int.emod_nonneg (daysFromCivil year 1 4 + 3) (by decide : (7:Int) ≠ 0)
  have hwd1 := Int.emod_lt_of_pos (daysFromCivil year 1 4 + 3) (by decide : (0:Int) < 7)
  generalize daysFromCivil year 1 4 = j at *
  have yOk : chronoYearOk year = true := by
    unfold chronoYearOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
  have td1 : tdDaysOk ((week - 1) * 7) = true := by
    unfold tdDaysOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
  have b1 := bound (j - (j + 3) % 7 + (week - 1) * 7) (by omega)
  have c1 : chronoYearOk (civilFromDaysI (j - (j + 3) % 7 + (week - 1) * 7)).1 = true := by
    unfold chronoYearOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
  by_cases hd : 0 ≤ dow ∧ dow ≤ 6
  · simp only [hd, not_true_eq_false, ite_false, yOk, Bool.not_true, Bool.false_eq_true, td1, c1,
      ite_true]
    have td2 : tdDaysOk (if dow = 0 then 6 else dow - 1) = true := by
      unfold tdDaysOk; split <;> simp only [Bool.and_eq_true, decide_eq_true_eq] <;> omega
    have b2 := bound (j - (j + 3) % 7 + (week - 1) * 7 + (if dow = 0 then 6 else dow - 1)) (by split <;> omega)
    have c2 : chronoYearOk (civilFromDaysI (j - (j + 3) % 7 + (week - 1) * 7 +
        (if dow = 0 then 6 else dow - 1))).1 = true := by
      unfold chronoYearOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
    simp only [td2, c2, c1, Bool.not_true, Bool.false_eq_true, ite_false, ite_true]
    intro h; cases h
  · simp only [hd, not_false_eq_true, ite_true]
    intro h; cases h

/-- `dayOfWeek` in the map constructor accepts `0..6` with `0` = Sunday
(`temporal.rs:75,84`), so ISO `dayOfWeek: 7` is rejected (C and openCypher: `1..7`,
7 = Sunday). **Divergence (confirmed live).** -/
theorem dayOfWeek_seven_rejected : weekPath 2021 1 7 = .err := by decide +kernel
theorem dayOfWeek_zero_is_sunday : weekPath 2021 1 0 = .ok (daysFromCivil 2021 1 10) := by
  decide +kernel

/-! ## `parse_week_date`: slicing a `str` at byte 2 -/

/-- UTF-8 width of a code point. -/
def utf8Len (c : Char) : Nat :=
  if c.toNat < 0x80 then 1 else if c.toNat < 0x800 then 2 else if c.toNat < 0x10000 then 3 else 4

/-- Char-boundary byte offsets of a string (as a char list). -/
def boundaries : List Char → Nat → List Nat
  | [], off => [off]
  | c :: cs, off => off :: boundaries cs (off + utf8Len c)

/-- `&rest[..2]` does not panic iff byte 2 is a char boundary. -/
def sliceOk (rest : List Char) : Bool := (boundaries rest 0).contains 2

/-- **Bug (server crash, confirmed live)**: `date('2020W1é')` — after the `W` the rest is
`"1é"`, 3 bytes, so the `rest.len() <= 2` guard sends it to `rest[..2]`, which lands inside
`é`. C: `Failed to parse date`. -/
theorem week_slice_panics : sliceOk ['1', 'é'] = false := by decide

/-- ASCII input never trips the slice. -/
theorem sliceOk_ascii (a b : Char) (rest : List Char) (ha : a.toNat < 0x80) (hb : b.toNat < 0x80) :
    sliceOk (a :: b :: rest) = true := by
  unfold sliceOk
  have e1 : utf8Len a = 1 := by simp [utf8Len, ha]
  have e2 : utf8Len b = 1 := by simp [utf8Len, hb]
  cases rest <;> simp [boundaries, e1, e2]

/-! ## `format_date` -/

/-- `write_date_into` (`value.rs:803`): four year digits of `|y|`, no sign. -/
def formatDateDigits (y m d : Int) : List Nat :=
  let yr := y.natAbs
  [yr / 1000 % 10, yr / 100 % 10, yr / 10 % 10, yr % 10, m.toNat / 10 % 10, m.toNat % 10,
    d.toNat / 10 % 10, d.toNat % 10]

/-- **Bug (Rust vs C, confirmed live)**: `toString(date({year: 12345}))` prints
`2345-01-01` (C: `12345-01-01`), and year −5 prints as year 5 (`0005-01-01`), so
`toString` on dates is not injective and misreports the value. -/
theorem formatDate_not_injective :
    formatDateDigits 12345 1 1 = formatDateDigits 2345 1 1 ∧
    formatDateDigits (-5) 1 1 = formatDateDigits 5 1 1 := by decide

/-- …but it is injective on years `0..9999` (the only range it was written for). -/
theorem formatDate_injective_4digit (y y' m m' d d' : Int)
    (hy : 0 ≤ y ∧ y ≤ 9999) (hy' : 0 ≤ y' ∧ y' ≤ 9999) (hm : 1 ≤ m ∧ m ≤ 12) (hm' : 1 ≤ m' ∧ m' ≤ 12)
    (hd : 1 ≤ d ∧ d ≤ 31) (hd' : 1 ≤ d' ∧ d' ≤ 31)
    (e : formatDateDigits y m d = formatDateDigits y' m' d') : y = y' ∧ m = m' ∧ d = d' := by
  unfold formatDateDigits at e
  simp only [List.cons.injEq] at e
  omega

end FalkorTemporal
