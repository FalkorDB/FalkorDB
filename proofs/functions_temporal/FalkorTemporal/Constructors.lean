import FalkorTemporal.Duration
/-!
# Temporal constructors and formatting: totality and injectivity

| here | there |
| --- | --- |
| `chronoYearOk`      | chrono 0.4.45 `MIN_YEAR = -262143`, `MAX_YEAR = 262142` (`naive/date/mod.rs:2446`) |
| `tdDaysOk`          | chrono `TimeDelta::days` panics unless `days * 86400 ∈ [MIN.secs, MAX.secs]` (`time_delta.rs:86,150`) |
| `Outcome`           | `Result<NaiveDate, String>` plus the panic that kills the server |
| `inRange`, `checkRange` | `check_range` — `temporal.rs:63` (`None` passes) |
| `i32Ok`, `u32Ok`    | `i32::try_from` / `u32::try_from` — `temporal.rs:92,122-125` |
| `checkedAddDays`    | `NaiveDate::checked_add_signed(Duration::days(n))` (`None` ↦ `.err`) |
| `weekPath`          | `date_from_components`, `week` branch — `temporal.rs:94-107` |
| `quarterPath`       | `date_from_components`, `quarter` branch — `temporal.rs:109-119` |
| `ymdPath`           | `date_from_components`, year/month/day tail — `temporal.rs:122-128` |
| `dateFromComponents`| `date_from_components` — `temporal.rs:82` |
| `weekSlice`         | `rest.split_at_checked(2)` in `parse_week_date` — `temporal.rs:285-287` |
| `sliceOk`           | byte 2 of `rest` is a char boundary |
| `formatDateDigits`  | `format_date` + `write_date_into` — `value.rs:307`, `value.rs:836` |
| `pre2950_*`         | the same before #2950 (737dfb5da): unchecked `NaiveDate ± TimeDelta`, `&rest[..2]` (history) |
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

/-- Every day number at most 9·10⁷ days from 1970 is inside chrono's year range. -/
theorem chronoYearOk_small (z : Int) (hz : -90000000 ≤ z ∧ z ≤ 90000000) :
    chronoYearOk (civilFromDaysI z).1 = true := by
  have h0 := Int.emod_nonneg (z + 719468) (by decide : (146097:Int) ≠ 0)
  have h1 := Int.emod_lt_of_pos (z + 719468) (by decide : (0:Int) < 146097)
  have hy := yoe_bounds _ h0 h1
  unfold chronoYearOk; simp only [Bool.and_eq_true, decide_eq_true_eq]
  unfold civilFromDaysI; dsimp only
  split <;> omega

/-- `daysFromCivil y 1 4` for a year well inside chrono's range. -/
theorem jan4_bound (year : Int) (hy : -200000 ≤ year ∧ year ≤ 200000) :
    -82000000 ≤ daysFromCivil year 1 4 ∧ daysFromCivil year 1 4 ≤ 82000000 := by
  unfold daysFromCivil; dsimp only; simp only [show ¬ ((1:Int) > 2) by decide, ite_false,
    show (1:Int) ≤ 2 by decide, ite_true]; omega

/-! ## Main (since #2950): every component is range-checked before chrono sees it -/

def inRange (v lo hi : Int) : Bool := decide (lo ≤ v) && decide (v ≤ hi)

/-- `check_range(name, value, min, max)` (`temporal.rs:63`): `None` passes. -/
def checkRange (v : Option Int) (lo hi : Int) : Bool :=
  match v with
  | some x => inRange x lo hi
  | none => true

/-- `i32::try_from` / `u32::try_from`. -/
def i32Ok (x : Int) : Bool := inRange x (-2147483648) 2147483647
def u32Ok (x : Int) : Bool := inRange x 0 4294967295

/-- `NaiveDate::checked_add_signed(chrono::Duration::days(n))`: `Duration::days` still
panics outside its bound, but the add returns `None` (↦ `.err`) instead of panicking. -/
def checkedAddDays (day n : Int) : Outcome :=
  if !tdDaysOk n then .panic "TimeDelta::days out of bounds"
  else if chronoYearOk (civilFromDaysI (day + n)).1 then .ok (day + n) else .err

/-- `date_from_components`, week branch (`temporal.rs:94-107`). `check_range("week", 1, 53)`
comes first, so `(week - 1) * 7 + day_offset - weekday_of_jan4` is in `-6..=370` and no i64
operation can overflow. Since #2988 `dayOfWeek` is ISO `1..=7` (7 = Sunday), as in the
string form and C (`check_range("dayOfWeek", day_of_week, 1, 7)`, `day_of_week - 1`). -/
def weekPath (year week dow : Int) : Outcome :=
  if !inRange week 1 53 then .err
  else if !inRange dow 1 7 then .err
  else if !chronoYearOk year then .err             -- `from_ymd_opt(year, 1, 4)`
  else
    let jan4 := daysFromCivil year 1 4
    -- weekday of jan4, Monday = 0 (1970-01-01 was a Thursday)
    let wd := (jan4 + 3) % 7
    let off := dow - 1
    checkedAddDays jan4 ((week - 1) * 7 + off - wd)

/-- The week branch before #2988 (`temporal.rs:94-109` at 5e475e6c9's parent): `dayOfWeek`
was `0..=6` with 0 = Sunday. -/
def pre2988_weekPath (year week dow : Int) : Outcome :=
  if !inRange week 1 53 then .err
  else if !inRange dow 0 6 then .err
  else if !chronoYearOk year then .err
  else
    let jan4 := daysFromCivil year 1 4
    let wd := (jan4 + 3) % 7
    let off := if dow = 0 then 6 else dow - 1
    checkedAddDays jan4 ((week - 1) * 7 + off - wd)

/-- `date_from_components`, quarter branch (`temporal.rs:109-119`). -/
def quarterPath (year quarter : Int) (doq : Option Int) : Outcome :=
  if !inRange quarter 1 4 then .err
  else if !checkRange doq 1 92 then .err
  else if !chronoYearOk year then .err             -- `from_ymd_opt(year, qsm, 1)`
  else checkedAddDays (daysFromCivil year ((quarter - 1) * 3 + 1) 1) (doq.getD 1 - 1)

/-- The year/month/day tail (`temporal.rs:122-128`): `u32::try_from` on month and day (no
more `as u32` truncation), then `NaiveDate::from_ymd_opt`. -/
def ymdPath (year month day : Int) : Outcome :=
  if u32Ok month && u32Ok day && chronoYearOk year && inRange month 1 12 &&
      inRange day 1 (daysInMonth year month) then .ok (daysFromCivil year month day)
  else .err

/-- `date_from_components` (`temporal.rs:82`): `i32::try_from(year)` (no more `as i32`
truncation), then the week, quarter or year/month/day branch. -/
def dateFromComponents (year month day week dow quarter doq : Option Int) : Outcome :=
  let y := year.getD 1970
  if !i32Ok y then .err
  else match week, quarter with
    | some w, _ => weekPath y w (dow.getD 1)
    | none, some q => quarterPath y q doq
    | none, none => ymdPath y (month.getD 1) (day.getD 1)

theorem checkedAddDays_total (day n : Int) (hn : -1000000 ≤ n ∧ n ≤ 1000000) (w : String) :
    checkedAddDays day n ≠ .panic w := by
  unfold checkedAddDays
  have : tdDaysOk n = true := by
    unfold tdDaysOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
  simp only [this, Bool.not_true, Bool.false_eq_true, ite_false]
  split <;> intro h <;> cases h

theorem weekPath_total (year week dow : Int) (w : String) : weekPath year week dow ≠ .panic w := by
  unfold weekPath
  by_cases h1 : inRange week 1 53 = true
  · by_cases h2 : inRange dow 1 7 = true
    · by_cases h3 : chronoYearOk year = true
      · simp only [h1, h2, h3, Bool.not_true, Bool.false_eq_true, ite_false]
        simp only [inRange, Bool.and_eq_true, decide_eq_true_eq] at h1 h2
        have m0 := Int.emod_nonneg (daysFromCivil year 1 4 + 3) (by decide : (7:Int) ≠ 0)
        have m1 := Int.emod_lt_of_pos (daysFromCivil year 1 4 + 3) (by decide : (0:Int) < 7)
        apply checkedAddDays_total
        omega
      · simp only [h1, h2, h3, Bool.not_true, Bool.not_false, Bool.false_eq_true, ite_false,
          ite_true]; intro h; cases h
    · simp only [h1, h2, Bool.not_true, Bool.not_false, Bool.false_eq_true, ite_false, ite_true]
      intro h; cases h
  · simp only [h1, Bool.not_false, ite_true]; intro h; cases h

theorem quarterPath_total (year quarter : Int) (doq : Option Int) (w : String) :
    quarterPath year quarter doq ≠ .panic w := by
  unfold quarterPath
  by_cases h1 : inRange quarter 1 4 = true
  · by_cases h2 : checkRange doq 1 92 = true
    · by_cases h3 : chronoYearOk year = true
      · simp only [h1, h2, h3, Bool.not_true, Bool.false_eq_true, ite_false]
        apply checkedAddDays_total
        cases doq with
        | none => simp
        | some x =>
          simp only [checkRange, inRange, Bool.and_eq_true, decide_eq_true_eq] at h2
          simp only [Option.getD_some]; omega
      · simp only [h1, h2, h3, Bool.not_true, Bool.not_false, Bool.false_eq_true, ite_false,
          ite_true]; intro h; cases h
    · simp only [h1, h2, Bool.not_true, Bool.not_false, Bool.false_eq_true, ite_false, ite_true]
      intro h; cases h
  · simp only [h1, Bool.not_false, ite_true]; intro h; cases h

theorem ymdPath_total (year month day : Int) (w : String) : ymdPath year month day ≠ .panic w := by
  unfold ymdPath; split <;> intro h <;> cases h

/-- **#2950 (W2-temporal-1, -2 fixed)**: `date_from_components` is total. For *every*
input — any `i64` (indeed any integer) in any of the seven slots, present or absent — it
returns a date or an error; no input reaches a chrono panic. `date(map)`,
`localdatetime(map)` and their slot forms (`date_struct_pure`, …) call nothing else that
can panic, so these constructors cannot take the server down. -/
theorem dateFromComponents_total (year month day week dow quarter doq : Option Int)
    (w : String) : dateFromComponents year month day week dow quarter doq ≠ .panic w := by
  unfold dateFromComponents; dsimp only
  split
  · intro h; cases h
  · split
    · exact weekPath_total _ _ _ w
    · exact quarterPath_total _ _ _ w
    · exact ymdPath_total _ _ _ w

/-- The week branch on in-range components is the ISO week date. -/
theorem weekPath_ok (year week dow : Int) (hw : 1 ≤ week ∧ week ≤ 53) (hd : 1 ≤ dow ∧ dow ≤ 7)
    (hy : -200000 ≤ year ∧ year ≤ 200000) :
    weekPath year week dow = .ok (daysFromCivil year 1 4 - (daysFromCivil year 1 4 + 3) % 7
      + (week - 1) * 7 + (dow - 1)) := by
  have hj := jan4_bound year hy
  have m0 := Int.emod_nonneg (daysFromCivil year 1 4 + 3) (by decide : (7:Int) ≠ 0)
  have m1 := Int.emod_lt_of_pos (daysFromCivil year 1 4 + 3) (by decide : (0:Int) < 7)
  unfold weekPath checkedAddDays
  have i1 : inRange week 1 53 = true := by
    simp only [inRange, Bool.and_eq_true, decide_eq_true_eq]; omega
  have i2 : inRange dow 1 7 = true := by
    simp only [inRange, Bool.and_eq_true, decide_eq_true_eq]; omega
  have i3 : chronoYearOk year = true := by
    simp only [chronoYearOk, Bool.and_eq_true, decide_eq_true_eq]; omega
  generalize daysFromCivil year 1 4 = j at *
  have e : j + ((week - 1) * 7 + (dow - 1) - (j + 3) % 7) =
      j - (j + 3) % 7 + (week - 1) * 7 + (dow - 1) := by omega
  have td : tdDaysOk ((week - 1) * 7 + (dow - 1) - (j + 3) % 7) = true := by
    unfold tdDaysOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
  have c := chronoYearOk_small (j - (j + 3) % 7 + (week - 1) * 7 + (dow - 1)) (by omega)
  simp only [i1, i2, i3, td, e, c, Bool.not_true, Bool.false_eq_true, ite_false, ite_true]

/-- The quarter branch: `dayOfQuarter` counts on from the quarter's first day (C does the
same, `DateTime_fromQuarterDate` lets `tm_mday` overflow into later months). -/
theorem quarterPath_ok (year quarter doq : Int) (hq : 1 ≤ quarter ∧ quarter ≤ 4)
    (hd : 1 ≤ doq ∧ doq ≤ 92) (hy : -200000 ≤ year ∧ year ≤ 200000) :
    quarterPath year quarter (some doq) =
      .ok (daysFromCivil year ((quarter - 1) * 3 + 1) 1 + (doq - 1)) := by
  have hb : -82000000 ≤ daysFromCivil year ((quarter - 1) * 3 + 1) 1 ∧
      daysFromCivil year ((quarter - 1) * 3 + 1) 1 ≤ 82000000 := by
    unfold daysFromCivil; dsimp only; split <;> split <;> omega
  unfold quarterPath checkedAddDays
  have i1 : inRange quarter 1 4 = true := by
    simp only [inRange, Bool.and_eq_true, decide_eq_true_eq]; omega
  have i2 : checkRange (some doq) 1 92 = true := by
    simp only [checkRange, inRange, Bool.and_eq_true, decide_eq_true_eq]; omega
  have i3 : chronoYearOk year = true := by
    simp only [chronoYearOk, Bool.and_eq_true, decide_eq_true_eq]; omega
  have td : tdDaysOk (doq - 1) = true := by
    unfold tdDaysOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
  have c := chronoYearOk_small (daysFromCivil year ((quarter - 1) * 3 + 1) 1 + (doq - 1)) (by omega)
  simp only [i1, i2, i3, td, c, Option.getD_some, Bool.not_true, Bool.false_eq_true, ite_false,
    ite_true]

/-- The fixed crashes now error (FINDINGS W2-temporal-1, -2): the live repros and the
`i64::MIN`/`i64::MAX` slots from the #2950 unit test. -/
theorem week_huge_errors : weekPath 2020 10000000000 1 = .err := by decide +kernel
theorem week_min_errors : weekPath 2020 (-9223372036854775808) 1 = .err := by decide +kernel
theorem quarter_doq_errors :
    quarterPath 2020 2 (some 9223372036854775807) = .err ∧
    quarterPath 2020 2 (some (-9223372036854775808)) = .err ∧
    quarterPath 2020 2 (some 93) = .err ∧ quarterPath 2020 2 (some 0) = .err := by decide +kernel
theorem quarter_errors :
    quarterPath 2020 0 none = .err ∧ quarterPath 2020 5 none = .err ∧
    quarterPath 2020 4294967297 none = .err := by decide +kernel

/-- The FINDINGS item 11 divergences that #2950 closed: C errors on both, and so does main
(`date({year:2021, quarter:1, dayOfQuarter:200})`, `date({year:2021, week:60, dayOfWeek:0})`). -/
theorem quarter_doq_200_errors : quarterPath 2021 1 (some 200) = .err := by decide +kernel
theorem week_60_errors : weekPath 2021 60 0 = .err := by decide +kernel

/-- `i32::try_from` / `u32::try_from` replace the truncating `as` casts: `date({year:
4294969316})` and `date({year:2020, month: 4294967297})` used to be 2020-01-01. -/
theorem ymd_no_truncation :
    dateFromComponents (some 4294969316) none none none none none none = .err ∧
    dateFromComponents (some 2020) (some 4294967297) none none none none none = .err ∧
    dateFromComponents (some 2020) (some 1) (some 4294967297) none none none none = .err := by
  decide +kernel

/-- In-range values still work: the #2950 unit test `date_in_range_components_still_work`. -/
theorem in_range_examples :
    weekPath 1984 10 1 = .ok (daysFromCivil 1984 3 5) ∧
    weekPath 1918 53 1 = .ok (daysFromCivil 1918 12 30) ∧
    quarterPath 1984 3 (some 45) = .ok (daysFromCivil 1984 8 14) ∧
    quarterPath 1984 4 (some 92) = .ok (daysFromCivil 1984 12 31) := by decide +kernel

/-- At chrono's year edges an in-range week whose date leaves chrono's range is now an
error (pre-#2950 it panicked, `pre2950_week_edge_panics`). -/
theorem week_edge_errors : weekPath (-262143) 1 1 = .err ∧ weekPath 262142 53 7 = .err := by
  decide +kernel

/-- **Fixed by #2988 (W2-temporal-9, closes #2936)**: `dayOfWeek` in the map constructor is ISO
`1..7` with `7` = Sunday, like the string form and C: `date({year:2021, week:1, dayOfWeek:7})`
is 2021-01-10, and `0` / `8` / `-1` are errors. -/
theorem dayOfWeek_seven_is_sunday : weekPath 2021 1 7 = .ok (daysFromCivil 2021 1 10) := by
  decide +kernel
theorem dayOfWeek_zero_rejected : weekPath 2021 1 0 = .err := by decide +kernel
theorem dayOfWeek_out_of_range_errors :
    weekPath 2021 1 8 = .err ∧ weekPath 2021 1 (-1) = .err := by decide +kernel

/-- Pre-#2988 (history, the divergence W2-temporal-9): `dayOfWeek` was `0..6` with `0` =
Sunday, so ISO `dayOfWeek: 7` was rejected (C and openCypher: `1..7`) and `0` accepted. -/
theorem pre2988_dayOfWeek_seven_rejected : pre2988_weekPath 2021 1 7 = .err := by decide +kernel
theorem pre2988_dayOfWeek_zero_is_sunday :
    pre2988_weekPath 2021 1 0 = .ok (daysFromCivil 2021 1 10) := by decide +kernel

/-- The fix only re-labels Sunday: on `dayOfWeek` `1..7` main is the old branch with `7`
read as the old `0`. (Monday..Saturday are unchanged.) -/
theorem weekPath_eq_pre2988 (year week dow : Int) (hd : 1 ≤ dow ∧ dow ≤ 7) :
    weekPath year week dow = pre2988_weekPath year week (if dow = 7 then 0 else dow) := by
  unfold weekPath pre2988_weekPath
  by_cases h7 : dow = 7
  · subst h7
    simp [show inRange 7 1 7 = true from by decide, show inRange 0 0 6 = true from by decide]
  · have h0 : dow ≠ 0 := by omega
    have hin : inRange dow 1 7 = true := by
      simp only [inRange, Bool.and_eq_true, decide_eq_true_eq]; omega
    have hin' : inRange dow 0 6 = true := by
      simp only [inRange, Bool.and_eq_true, decide_eq_true_eq]; omega
    simp [h7, h0, hin, hin']

/-! ## Before #2950 (history): unchecked chrono arithmetic -/

/-- `NaiveDate + TimeDelta` (chrono `Add` impl): panics when the result leaves chrono's range. -/
def pre2950_dateAddDays (day n : Int) : Outcome :=
  if !tdDaysOk n then .panic "TimeDelta::days out of bounds"
  else if chronoYearOk (civilFromDaysI (day + n)).1 then .ok (day + n)
  else .panic "`NaiveDate + TimeDelta` overflowed"

/-- Pre-#2950 week branch (`temporal.rs:73-86` at 8621212fc), i64 arithmetic as `Int`
(release wraps `(week - 1) * 7`; debug panics — both unmodelled beyond `tdDaysOk`). -/
def pre2950_weekPath (year week dow : Int) : Outcome :=
  if ¬ (0 ≤ dow ∧ dow ≤ 6) then .err
  else if !chronoYearOk year then .err
  else
    let jan4 := daysFromCivil year 1 4
    let wd := (jan4 + 3) % 7
    let wk1Mon := jan4 - wd
    match pre2950_dateAddDays wk1Mon ((week - 1) * 7) with
    | .ok tm => pre2950_dateAddDays tm (if dow = 0 then 6 else dow - 1)
    | o => o

/-- Pre-#2950 quarter branch (`temporal.rs:89-96` at 8621212fc): `checked_add_signed` for
the date but the *panicking* `Duration::days`, and no range check on `dayOfQuarter`. -/
def pre2950_quarterPath (year quarter doq : Int) : Outcome :=
  let qsm := (quarter - 1) * 3 + 1
  if ¬ (1 ≤ qsm ∧ qsm ≤ 12) ∨ !chronoYearOk year then .err
  else if !tdDaysOk (doq - 1) then .panic "TimeDelta::days out of bounds"
  else
    let base := daysFromCivil year qsm 1
    if chronoYearOk (civilFromDaysI (base + (doq - 1))).1 then .ok (base + (doq - 1)) else .err

/-- **Fixed by #2950 (W2-temporal-1, server crash)**: `date({week: 10000000000})` panicked in
`NaiveDate + TimeDelta`. Main: `week_huge_errors`. -/
theorem pre2950_week_panics :
    pre2950_weekPath 1970 10000000000 1 = .panic "`NaiveDate + TimeDelta` overflowed" := by
  decide +kernel

/-- **Fixed by #2950**: a week count beyond ±1.5·10¹⁰ panicked earlier, in `TimeDelta::days`. -/
theorem pre2950_week_panics_delta :
    pre2950_weekPath 1970 20000000000 1 = .panic "TimeDelta::days out of bounds" := by
  decide +kernel

/-- **Fixed by #2950 (W2-temporal-2, server crash)**:
`date({quarter: 2, dayOfQuarter: 9223372036854775807})`. Main: `quarter_doq_errors`. -/
theorem pre2950_quarter_panics :
    pre2950_quarterPath 1970 2 9223372036854775807 = .panic "TimeDelta::days out of bounds" := by
  decide +kernel

/-- **Fixed by #2950** (found while re-targeting; same crash class as W2-temporal-1): even
an in-range week panicked at chrono's year edges, `date({year:-262143, week:1})` (week 1
starts in -262144) and `date({year:262142, week:53, dayOfWeek:0})`. Main: `week_edge_errors`. -/
theorem pre2950_week_edge_panics :
    pre2950_weekPath (-262143) 1 1 = .panic "`NaiveDate + TimeDelta` overflowed" ∧
    pre2950_weekPath 262142 53 0 = .panic "`NaiveDate + TimeDelta` overflowed" := by
  decide +kernel

/-- Pre-#2950 totality on a sane domain (`|week| ≤ 10⁶`, `|year| ≤ 200 000`). -/
theorem pre2950_weekPath_total (year week dow : Int) (hy : -200000 ≤ year ∧ year ≤ 200000)
    (hw : -1000000 ≤ week ∧ week ≤ 1000000) :
    ∀ w, pre2950_weekPath year week dow ≠ .panic w := by
  intro w
  unfold pre2950_weekPath pre2950_dateAddDays
  have hjan := jan4_bound year hy
  have hwd0 := Int.emod_nonneg (daysFromCivil year 1 4 + 3) (by decide : (7:Int) ≠ 0)
  have hwd1 := Int.emod_lt_of_pos (daysFromCivil year 1 4 + 3) (by decide : (0:Int) < 7)
  generalize daysFromCivil year 1 4 = j at *
  have yOk : chronoYearOk year = true := by
    unfold chronoYearOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
  have td1 : tdDaysOk ((week - 1) * 7) = true := by
    unfold tdDaysOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
  have c1 := chronoYearOk_small (j - (j + 3) % 7 + (week - 1) * 7) (by omega)
  by_cases hd : 0 ≤ dow ∧ dow ≤ 6
  · simp only [hd, ite_false, yOk, Bool.not_true, Bool.false_eq_true, td1, c1,
      ite_true]
    have td2 : tdDaysOk (if dow = 0 then 6 else dow - 1) = true := by
      unfold tdDaysOk; split <;> simp only [Bool.and_eq_true, decide_eq_true_eq] <;> omega
    have c2 := chronoYearOk_small (j - (j + 3) % 7 + (week - 1) * 7 +
        (if dow = 0 then 6 else dow - 1)) (by split <;> omega)
    simp only [td2, c2, Bool.not_true, Bool.false_eq_true, ite_false, ite_true]
    intro h; cases h
  · simp only [hd, not_false_eq_true, ite_true]
    intro h; cases h

/-- #2950 only removed panics: wherever the old week branch was used as intended (week in
`1..53`, sane year), main returns the same date. -/
theorem weekPath_eq_pre2950 (year week dow : Int) (hw : 1 ≤ week ∧ week ≤ 53)
    (hd : 1 ≤ dow ∧ dow ≤ 6) (hy : -200000 ≤ year ∧ year ≤ 200000) :
    weekPath year week dow = pre2950_weekPath year week dow := by
  rw [weekPath_ok year week dow hw (by omega) hy]
  have hd0 : dow ≠ 0 := by omega
  have hd' : 0 ≤ dow ∧ dow ≤ 6 := by omega
  have hjan := jan4_bound year hy
  have hwd0 := Int.emod_nonneg (daysFromCivil year 1 4 + 3) (by decide : (7:Int) ≠ 0)
  have hwd1 := Int.emod_lt_of_pos (daysFromCivil year 1 4 + 3) (by decide : (0:Int) < 7)
  unfold pre2950_weekPath pre2950_dateAddDays
  have yOk : chronoYearOk year = true := by
    unfold chronoYearOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
  generalize daysFromCivil year 1 4 = j at *
  have td1 : tdDaysOk ((week - 1) * 7) = true := by
    unfold tdDaysOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
  have c1 := chronoYearOk_small (j - (j + 3) % 7 + (week - 1) * 7) (by omega)
  have td2 : tdDaysOk (dow - 1) = true := by
    unfold tdDaysOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
  have c2 := chronoYearOk_small (j - (j + 3) % 7 + (week - 1) * 7 + (dow - 1)) (by omega)
  simp only [hd', hd0, not_true_eq_false, ite_false, yOk, Bool.not_true, Bool.false_eq_true, td1, c1,
    td2, c2, ite_true, and_self]

/-- Likewise for the quarter branch with `dayOfQuarter` in `1..92`. -/
theorem quarterPath_eq_pre2950 (year quarter doq : Int) (hq : 1 ≤ quarter ∧ quarter ≤ 4)
    (hd : 1 ≤ doq ∧ doq ≤ 92) (hy : -200000 ≤ year ∧ year ≤ 200000) :
    quarterPath year quarter (some doq) = pre2950_quarterPath year quarter doq := by
  rw [quarterPath_ok year quarter doq hq hd hy]
  have hb : -82000000 ≤ daysFromCivil year ((quarter - 1) * 3 + 1) 1 ∧
      daysFromCivil year ((quarter - 1) * 3 + 1) 1 ≤ 82000000 := by
    unfold daysFromCivil; dsimp only; split <;> split <;> omega
  unfold pre2950_quarterPath; dsimp only
  have yOk : chronoYearOk year = true := by
    unfold chronoYearOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
  have td : tdDaysOk (doq - 1) = true := by
    unfold tdDaysOk; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega
  have c := chronoYearOk_small (daysFromCivil year ((quarter - 1) * 3 + 1) 1 + (doq - 1)) (by omega)
  have q : ¬ (¬ (1 ≤ (quarter - 1) * 3 + 1 ∧ (quarter - 1) * 3 + 1 ≤ 12) ∨ !chronoYearOk year) := by
    simp only [yOk, Bool.not_true, Bool.false_eq_true, or_false]; omega
  simp only [q, td, c, Bool.not_true, Bool.false_eq_true, ite_false, ite_true]

/-! ## `parse_week_date`: splitting a `str` at byte 2 -/

/-- UTF-8 width of a code point. -/
def utf8Len (c : Char) : Nat :=
  if c.toNat < 0x80 then 1 else if c.toNat < 0x800 then 2 else if c.toNat < 0x10000 then 3 else 4

/-- Char-boundary byte offsets of a string (as a char list). -/
def boundaries : List Char → Nat → List Nat
  | [], off => [off]
  | c :: cs, off => off :: boundaries cs (off + utf8Len c)

/-- Byte 2 is a char boundary: `&rest[..2]` does not panic, `split_at_checked(2)` is `Some`. -/
def sliceOk (rest : List Char) : Bool := (boundaries rest 0).contains 2

inductive SliceOutcome where
  | panic
  | err
  | ok
  deriving DecidableEq, Repr

/-- Main (`temporal.rs:285-287`): `rest.split_at_checked(2).ok_or_else(..)?`. -/
def weekSlice (rest : List Char) : SliceOutcome := if sliceOk rest then .ok else .err

/-- Pre-#2950: `rest[..2]` / `rest[2..]`. -/
def pre2950_weekSlice (rest : List Char) : SliceOutcome := if sliceOk rest then .ok else .panic

/-- **#2950 (W2-temporal-3 fixed)**: the week-string split never panics. -/
theorem weekSlice_total (rest : List Char) : weekSlice rest ≠ .panic := by
  unfold weekSlice; split <;> intro h <;> cases h

/-- …and agrees with the old slice wherever the old one did not panic. -/
theorem weekSlice_eq_pre2950 (rest : List Char) (h : pre2950_weekSlice rest ≠ .panic) :
    weekSlice rest = pre2950_weekSlice rest := by
  unfold weekSlice pre2950_weekSlice at *; split <;> simp_all

/-- `date('2020W1é')` is now an error ("Invalid week"), as in C ("Failed to parse date"). -/
theorem week_slice_errors : weekSlice ['1', 'é'] = .err := by decide

/-- **Fixed by #2950 (W2-temporal-3, server crash)**: `date('2020W1é')` — after the `W` the
rest is `"1é"`, 3 bytes, so the `rest.len() <= 2` guard sent it to `rest[..2]`, which lands
inside `é`. -/
theorem pre2950_week_slice_panics : pre2950_weekSlice ['1', 'é'] = .panic := by decide

/-- ASCII input never trips the slice. -/
theorem sliceOk_ascii (a b : Char) (rest : List Char) (ha : a.toNat < 0x80) (hb : b.toNat < 0x80) :
    sliceOk (a :: b :: rest) = true := by
  unfold sliceOk
  have e1 : utf8Len a = 1 := by simp [utf8Len, ha]
  have e2 : utf8Len b = 1 := by simp [utf8Len, hb]
  cases rest <;> simp [boundaries, e1, e2]

/-! ## `format_date` -/

/-- `write_date_into` (`value.rs:836`): four year digits of `|y|`, no sign. -/
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
