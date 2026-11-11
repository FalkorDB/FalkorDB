import FalkorTemporal.CalendarLaws
/-!
# Durations and date arithmetic

| here | there |
| --- | --- |
| `chk`, `chkMul`, `chkAdd` | `i64::checked_*` |
| `constructDuration`       | `construct_duration_secs` — `temporal.rs:479` (ends in `check_date_range`, #2746) |
| `decomposeDuration`       | `decompose_duration` — `temporal.rs:539` (`y - 1970` is i32: `toI32`) |
| `shiftMonths`             | `shift_months` — `value.rs:725`, arithmetic only (shared by `+` and `-` since #2984) |
| `addDur` / `addDurRaw`    | `add_duration_to_timestamp` — `value.rs:701` (checked / the unchecked arithmetic inside it) |
| `subDur` / `subDurRaw`    | `sub_duration_from_timestamp` — `value.rs:711` |
| `shiftMonthsR`            | `shift_months` — `value.rs:725`, with its `MAX_DATE_YEAR` guard (#2746) |
| `checkDateRange`          | `check_date_range` — `temporal.rs:531` |
| (not modelled)            | `Add for Value`, `(Duration, Duration)` arm — `value.rs:1048` (`add_slow`): only the `construct_decompose` round-trip it rests on is proved |
| `timeAddDur` / `timeSubDur` | `Add`/`Sub for Value`, `(Time, Duration)` arms — `add_slow` `value.rs:1069-1070`, `Sub` `:1111-1112` (since #2985) |

Every Temporal value is one `i64` of seconds; a Duration is "the instant
1970-01-01 + years/months + days/seconds", so its *components* are recovered by
decomposing that instant. Both C and Rust use this encoding.
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

/-- `check_date_range` (`temporal.rs:531`): `Err("Date out of range")` ↦ `none`. -/
def checkDateRange (ts : Int) : Option Int :=
  if minDateDays ≤ ts / 86400 ∧ ts / 86400 ≤ maxDateDays then some ts else none

/-- `construct_duration_secs` (`temporal.rs:479-518`), `Err` ↦ `none`; ends with `check_date_range`. -/
def constructDuration (years months weeks days hours minutes seconds : Int) : Option Int := do
  let tmo ← chkMul years 12 >>= fun y => chkAdd y months
  let tmo32 ← chk32 tmo
  let baseYear ← chk32 (1970 + tmo32 / 12)
  let baseMonth := tmo32 % 12 + 1
  let anchor := daysFromCivilT baseYear baseMonth 1 * 86400
  let extra ← (chkMul weeks 7 >>= fun w => chkAdd w days) >>= (fun wd => chkMul wd 86400)
    >>= (fun s => chkMul hours 3600 >>= fun h => chkAdd s h)
    >>= (fun s => chkMul minutes 60 >>= fun m => chkAdd s m)
    >>= (fun s => chkAdd s seconds)
  (chkAdd anchor extra).bind checkDateRange

/-- `construct_duration_secs` on a years/months/seconds triple, when nothing overflows. -/
theorem construct_ym_s (Y M S : Int) (h1 : I64 (Y * 12)) (h2 : I64 (Y * 12 + M))
    (h3 : I32 (Y * 12 + M)) (h4 : I32 (1970 + (Y * 12 + M) / 12)) (h5 : I64 S)
    (h6 : I64 (daysFromCivil (1970 + (Y * 12 + M) / 12) ((Y * 12 + M) % 12 + 1) 1 * 86400 + S))
    (h7 : minDateDays ≤ (daysFromCivil (1970 + (Y * 12 + M) / 12) ((Y * 12 + M) % 12 + 1) 1 * 86400 + S) / 86400 ∧
      (daysFromCivil (1970 + (Y * 12 + M) / 12) ((Y * 12 + M) % 12 + 1) 1 * 86400 + S) / 86400 ≤ maxDateDays) :
    constructDuration Y M 0 0 0 0 S =
      some (daysFromCivil (1970 + (Y * 12 + M) / 12) ((Y * 12 + M) % 12 + 1) 1 * 86400 + S) := by
  have z : I64 0 := by unfold I64; decide
  unfold constructDuration chkMul chkAdd
  simp only [Int.zero_mul, Int.zero_add, Int.mul_zero, chk_of _ h1, chk_of _ h2, chk32_of _ h3,
    chk32_of _ h4, chk_of _ z, chk_of _ h5, daysFromCivilT_eq, chk_of _ h6, bind, Option.bind, checkDateRange, h7, and_self, if_true]

/-- `decompose_duration` (`temporal.rs:539-549`) as built in release (i32 wrap). -/
def decomposeDuration (dur : Int) : Int × Int × Int :=
  let days := dur / 86400
  let tod := dur % 86400
  let (y, m, d) := civilFromDays days
  (toI32 (y - 1970), m - 1, (d - 1) * 86400 + tod)

/-! ## Law 1: `construct ∘ decompose = id` (what `Duration ± Duration` relies on) -/

theorem construct_decompose (dur : Int) (hd : I64 dur)
    (hy : -2 ^ 31 + 1970 ≤ (civilFromDaysI (dur / 86400)).1 ∧ (civilFromDaysI (dur / 86400)).1 < 2 ^ 31)
    (hmo : I32 (((civilFromDaysI (dur / 86400)).1 - 1970) * 12 + ((civilFromDaysI (dur / 86400)).2.1 - 1)))
    (hr : minDateDays ≤ dur / 86400 ∧ dur / 86400 ≤ maxDateDays) :
    constructDuration (decomposeDuration dur).1 (decomposeDuration dur).2.1 0 0 0 0
      (decomposeDuration dur).2.2 = some dur := by
  have hv := civil_valid (dur / 86400)
  have hrt := dfc_civil (dur / 86400)
  have hdiv : 86400 * (dur / 86400) + dur % 86400 = dur := Int.mul_ediv_add_emod _ _
  have ht0 := Int.emod_nonneg dur (by decide : (86400:Int) ≠ 0)
  have ht1 := Int.emod_lt_of_pos dur (by decide : (0:Int) < 86400)
  have hday := daysFromCivil_day (civilFromDaysI (dur / 86400)).1 (civilFromDaysI (dur / 86400)).2.1
    (civilFromDaysI (dur / 86400)).2.2
  have i1 : I32 (civilFromDaysI (dur / 86400)).1 := ⟨by omega, by omega⟩
  have i2 : I32 ((civilFromDaysI (dur / 86400)).1 - 1970) := ⟨by omega, by omega⟩
  unfold decomposeDuration civilFromDays
  simp only [toI32_id _ i1, toI32_id _ i2]
  generalize hc : civilFromDaysI (dur / 86400) = c at *
  obtain ⟨y, m, d⟩ := c
  simp only at hv hrt hy hmo hday ⊢
  obtain ⟨hm1, hm2, hd1, hd2⟩ := hv
  have hdim : daysInMonth y m ≤ 31 := by
    unfold daysInMonth; split <;> (try split) <;> (try split) <;> omega
  generalize dur % 86400 = tod at *
  generalize dur / 86400 = days at *
  have eY : 1970 + ((y - 1970) * 12 + (m - 1)) / 12 = y := by omega
  have eM : ((y - 1970) * 12 + (m - 1)) % 12 + 1 = m := by omega
  have c6 : daysFromCivil y m 1 * 86400 + ((d - 1) * 86400 + tod) = dur := by
    rw [← hrt] at hdiv; rw [hday] at hdiv; omega
  rw [construct_ym_s]
  · rw [eY, eM, c6]
  all_goals (unfold I64 I32 at *; (try rw [eY, eM]); omega)

/-! ## Date ± Duration -/

/-- `shift_months` (`value.rs:725`, since #2984): move `ts` by whole years and months,
returning (days since epoch, time of day). A day past the end of the target month rolls
over into the next month (`2020-03-31` minus one month is `2020-03-02`), as C's `timegm`
does. Int arithmetic: the i32 `y + years` / `-years` overflow is out of scope here (see
`COVERAGE.tsv`, "Suspected" in `FalkorTemporal.lean`). -/
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

/-- `add_duration_to_timestamp` (`value.rs:701`). -/
def addDurRaw (ts dur : Int) : Int :=
  let (years, months, rem) := decomposeDuration dur
  let (days, tod) := shiftMonths ts years months
  days * 86400 + tod + rem

/-- `sub_duration_from_timestamp` (`value.rs:711`, since #2984): the same
`shift_months` with the years and months negated, so it rolls over like `addDurRaw`. -/
def subDurRaw (ts dur : Int) : Int :=
  let (years, months, rem) := decomposeDuration dur
  let (days, tod) := shiftMonths ts (-years) (-months)
  days * 86400 + tod - rem

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

/-- A date value (`Value::Date`) for a real calendar date. -/
def dateTs (y m d : Int) : Int := daysFromCivil y m d * 86400

/-- A pure year/month duration, `duration({years: Y, months: M})`, as the value
`construct_duration_secs` stores (anchor day 1 of the target month). -/
def monthsDur (Y M : Int) : Int :=
  let t := Y * 12 + M
  daysFromCivil (1970 + t / 12) (t % 12 + 1) 1 * 86400

/-- A small civil-year range where every intermediate `i32` in the Rust code is exact. -/
def SmallYear (y : Int) : Prop := -1000000 ≤ y ∧ y ≤ 1000000

theorem decompose_monthsDur (t : Int) (ht : -1000000 ≤ t ∧ t ≤ 1000000) :
    decomposeDuration (daysFromCivil (1970 + t / 12) (t % 12 + 1) 1 * 86400) =
      (t / 12, t % 12, 0) := by
  have hv : ValidDate (1970 + t / 12) (t % 12 + 1) 1 := by
    refine ⟨by omega, by omega, by decide, ?_⟩
    unfold daysInMonth; split <;> (try split) <;> (try split) <;> omega
  have hc := civil_dfc _ _ _ hv
  unfold decomposeDuration civilFromDays
  have e1 : daysFromCivil (1970 + t / 12) (t % 12 + 1) 1 * 86400 / 86400 =
      daysFromCivil (1970 + t / 12) (t % 12 + 1) 1 := by omega
  have e2 : daysFromCivil (1970 + t / 12) (t % 12 + 1) 1 * 86400 % 86400 = 0 := by omega
  have i1 : I32 (1970 + t / 12) := ⟨by omega, by omega⟩
  have i2 : I32 (1970 + t / 12 - 1970) := ⟨by omega, by omega⟩
  simp only [e1, e2, hc, toI32_id _ i1, toI32_id _ i2]
  simp only [Prod.mk.injEq]; omega

/-- **Law 2: `(date + P{Y}Y{M}M) - P{Y}Y{M}M = date`** whenever the day of month is ≤ 28
(so the roll-over in `shiftMonths` fires on neither side). -/
theorem add_sub_months (y m d t : Int) (hv : ValidDate y m d) (hd : d ≤ 28)
    (hy : SmallYear y) (ht : -1000000 ≤ t ∧ t ≤ 1000000) :
    subDurRaw (addDurRaw (dateTs y m d) (daysFromCivil (1970 + t / 12) (t % 12 + 1) 1 * 86400))
        (daysFromCivil (1970 + t / 12) (t % 12 + 1) 1 * 86400) = dateTs y m d := by
  have hdec := decompose_monthsDur t ht
  have hc := civil_dfc y m d hv
  obtain ⟨hm1, hm2, hd1, _⟩ := hv
  unfold addDurRaw subDurRaw dateTs shiftMonths
  simp only [hdec]
  have e1 : daysFromCivil y m d * 86400 / 86400 = daysFromCivil y m d := by omega
  have e2 : daysFromCivil y m d * 86400 % 86400 = 0 := by omega
  unfold civilFromDays
  have i1 : I32 y := by unfold I32 SmallYear at *; omega
  simp only [e1, e2, hc, toI32_id _ i1]
  -- the target month after adding
  generalize hA : y + t / 12 + (m + t % 12 - 1) / 12 = A
  generalize hB : (m + t % 12 - 1) % 12 + 1 = B
  have hB1 : 1 ≤ B ∧ B ≤ 12 := by omega
  have hdimB : 28 ≤ daysInMonth A B := by
    unfold daysInMonth; split <;> (try split) <;> (try split) <;> omega
  have nof : ¬ (d > daysInMonth A B) := by omega
  simp only [nof, ite_false]
  have hvA : ValidDate A B d := ⟨hB1.1, hB1.2, hd1, by omega⟩
  have e3 : (daysFromCivil A B d * 86400 + 0 + 0) / 86400 = daysFromCivil A B d := by omega
  have e4 : (daysFromCivil A B d * 86400 + 0 + 0) % 86400 = 0 := by omega
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

/-- **W2-temporal-5 fixed (#2984).** For every timestamp `ts` and every pure year/month
duration, `ts - P{Y}Y{M}M = ts + P{-Y}Y{-M}M`: subtraction and addition share
`shift_months`, so `d - dur = d + (-dur)` (pre-#2984 this failed). -/
theorem sub_eq_add_neg_months (ts t : Int) (ht : -1000000 ≤ t ∧ t ≤ 1000000) :
    subDurRaw ts (daysFromCivil (1970 + t / 12) (t % 12 + 1) 1 * 86400) =
      addDurRaw ts (daysFromCivil (1970 + (-t) / 12) ((-t) % 12 + 1) 1 * 86400) := by
  have h1 := decompose_monthsDur t ht
  have h2 := decompose_monthsDur (-t) ⟨by omega, by omega⟩
  unfold subDurRaw addDurRaw
  simp only [h1, h2]
  rw [shiftMonths_total ts (-(t / 12)), shiftMonths_total ts ((-t) / 12)]
  have : -(t / 12) * 12 + -(t % 12) = (-t) / 12 * 12 + (-t) % 12 := by omega
  rw [this]; omega

/-- The same for any duration: `subDurRaw` is `addDurRaw` with the decomposed years and months
negated (the remaining seconds are subtracted). -/
theorem subDur_eq_shift (ts dur : Int) :
    subDurRaw ts dur =
      (shiftMonths ts 0 (-((decomposeDuration dur).1 * 12 + (decomposeDuration dur).2.1))).1 * 86400 +
        (shiftMonths ts 0 0).2 - (decomposeDuration dur).2.2 := by
  unfold subDurRaw
  generalize decomposeDuration dur = c
  obtain ⟨Y, M, R⟩ := c
  simp only
  rw [shiftMonths_total ts (-Y)]
  have e : -Y * 12 + -M = -(Y * 12 + M) := by omega
  rw [e]
  unfold shiftMonths; simp only

/-! ## Range checks (#2746, #3575): `check_date_range`, the `shift_months` guard

`MAX_DATE_YEAR = i32::MAX - 2000` (`temporal.rs:523`). `MIN_DATE_DAYS`/`MAX_DATE_DAYS` are
`days_from_civil(∓MAX_DATE_YEAR, 1 / 12, 1 / 31)` (`temporal.rs:528-529`). -/

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
  · simp only [h, and_self, if_true, Option.isSome_some, true_iff]
    omega
  · simp only [h, if_false, Option.isSome_none, Bool.false_eq_true, false_iff]
    omega

/-- The year bound keeps every `i32` step in `shift_months` and its callers exact. -/
theorem MAX_DATE_YEAR_margin : MAX_DATE_YEAR + 2000 < 2 ^ 31 := by decide

/-- The year `shift_months` computes before narrowing (`adj_year`, i64). -/
def adjYear (ts years months : Int) : Int :=
  (civilFromDays (ts / 86400)).1 + years + ((civilFromDays (ts / 86400)).2.1 + months - 1) / 12

/-- `shift_months` (`value.rs:725`, since #2746): the arithmetic above, behind the guard
`adj_year.abs() > MAX_DATE_YEAR + 1 → Err("Date out of range")`. -/
def shiftMonthsR (ts years months : Int) : Option (Int × Int) :=
  if (adjYear ts years months).natAbs > (MAX_DATE_YEAR + 1).toNat then none
  else some (shiftMonths ts years months)

theorem shiftMonthsR_eq (ts years months : Int) (h : -(MAX_DATE_YEAR + 1) ≤ adjYear ts years months ∧
    adjYear ts years months ≤ MAX_DATE_YEAR + 1) :
    shiftMonthsR ts years months = some (shiftMonths ts years months) := by
  unfold shiftMonthsR
  rw [if_neg]; unfold MAX_DATE_YEAR at *; omega

theorem shiftMonthsR_out_of_range (ts years months : Int)
    (h : MAX_DATE_YEAR + 1 < adjYear ts years months ∨ adjYear ts years months < -(MAX_DATE_YEAR + 1)) :
    shiftMonthsR ts years months = none := by
  unfold shiftMonthsR
  rw [if_pos]; unfold MAX_DATE_YEAR at *; omega

/-- No wrap: whenever `shift_months` answers, `adj_year` (and the `adj_year + 1` of the
roll-over) fits an `i32`, so the narrowing `as i32` is exact. -/
theorem shiftMonthsR_no_wrap (ts years months : Int) (r : Int × Int)
    (h : shiftMonthsR ts years months = some r) :
    I32 (adjYear ts years months) ∧ I32 (adjYear ts years months + 1) := by
  unfold shiftMonthsR at h
  split at h
  · cases h
  · unfold I32 MAX_DATE_YEAR at *; omega

/-- `add_duration_to_timestamp` (`value.rs:701`, since #2746): `shift_months`, then
`check_date_range` on the sum. `decompose_duration` is modelled as total. -/
def addDur (ts dur : Int) : Option Int :=
  let (years, months, rem) := decomposeDuration dur
  (shiftMonthsR ts years months).bind fun (days, tod) => checkDateRange (days * 86400 + tod + rem)

/-- `sub_duration_from_timestamp` (`value.rs:711`, since #2746). -/
def subDur (ts dur : Int) : Option Int :=
  let (years, months, rem) := decomposeDuration dur
  (shiftMonthsR ts (-years) (-months)).bind fun (days, tod) => checkDateRange (days * 86400 + tod - rem)

/-- When both checks pass, the checked function is the arithmetic of `addDurRaw`. -/
theorem addDur_eq_raw (ts dur : Int)
    (h1 : (shiftMonthsR ts (decomposeDuration dur).1 (decomposeDuration dur).2.1).isSome = true)
    (h2 : (checkDateRange (addDurRaw ts dur)).isSome = true) :
    addDur ts dur = some (addDurRaw ts dur) := by
  unfold addDur addDurRaw at *
  generalize decomposeDuration dur = c at *
  obtain ⟨y, m, r⟩ := c
  simp only at *
  cases hs : shiftMonthsR ts y m with
  | none => simp [hs] at h1
  | some p =>
    have e : shiftMonthsR ts y m = some (shiftMonths ts y m) := by
      unfold shiftMonthsR at hs ⊢; split at hs
      · cases hs
      · rw [if_neg ‹_›]
    rw [e] at hs; cases hs
    simp only [Option.bind]
    cases hc : checkDateRange ((shiftMonths ts y m).1 * 86400 + (shiftMonths ts y m).2 + r) with
    | none => simp [hc] at h2
    | some v => simp only [checkDateRange] at hc; split at hc <;> simp_all

/-- A date or datetime produced by `+`/`-` of a duration is inside the supported range. -/
theorem addDur_in_range (ts dur r : Int) (h : addDur ts dur = some r) :
    -MAX_DATE_YEAR ≤ (civilFromDaysI (r / 86400)).1 ∧ (civilFromDaysI (r / 86400)).1 ≤ MAX_DATE_YEAR := by
  rw [← checkDateRange_iff]
  unfold addDur at h
  generalize decomposeDuration dur = c at h
  obtain ⟨y, m, rm⟩ := c
  simp only at h
  cases hs : shiftMonthsR ts y m with
  | none => simp [hs] at h
  | some p =>
    simp only [hs, Option.bind] at h
    have : checkDateRange (p.1 * 86400 + p.2 + rm) = some r := h
    simp only [checkDateRange] at this
    split at this
    · cases this; simp [checkDateRange, *]
    · cases this

theorem subDur_in_range (ts dur r : Int) (h : subDur ts dur = some r) :
    -MAX_DATE_YEAR ≤ (civilFromDaysI (r / 86400)).1 ∧ (civilFromDaysI (r / 86400)).1 ≤ MAX_DATE_YEAR := by
  rw [← checkDateRange_iff]
  unfold subDur at h
  generalize decomposeDuration dur = c at h
  obtain ⟨y, m, rm⟩ := c
  simp only at h
  cases hs : shiftMonthsR ts (-y) (-m) with
  | none => simp [hs] at h
  | some p =>
    simp only [hs, Option.bind] at h
    have : checkDateRange (p.1 * 86400 + p.2 - rm) = some r := h
    simp only [checkDateRange] at this
    split at this
    · cases this; simp [checkDateRange, *]
    · cases this

/-- The boundary: the last supported day is accepted, the next one is not. -/
theorem checkDateRange_boundary :
    (checkDateRange (maxDateDays * 86400)).isSome = true ∧
    checkDateRange ((maxDateDays + 1) * 86400) = none ∧
    (checkDateRange (minDateDays * 86400)).isSome = true ∧
    checkDateRange ((minDateDays - 1) * 86400) = none := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp [checkDateRange, minDateDays_val, maxDateDays_val] <;> omega

/-- Close to the limit a chain of additions is an error and not a wrapped year: a month
shift past `MAX_DATE_YEAR + 1` from a date in the last supported year. -/
theorem shift_past_limit_errors :
    shiftMonthsR (maxDateDays * 86400) 2 0 = none := by decide +kernel

/-! ## Counterexamples (all evaluated by the kernel) -/

/-- `duration({months: -1})` and `duration({months: 1})`. -/
def minus1M : Int := (constructDuration 0 (-1) 0 0 0 0 0).getD 0
def plus1M : Int := (constructDuration 0 1 0 0 0 0 0).getD 0

/-- The repro of W2-temporal-5 on main (and #2984's flow test): both are 2020-03-02 (C:
2020-03-02), and `localdatetime('2020-03-31T10:00:00') - duration({months:1})` is
`2020-03-02T10:00:00`. -/
theorem sub_vs_add_neg :
    subDurRaw (dateTs 2020 3 31) plus1M = dateTs 2020 3 2 ∧
    addDurRaw (dateTs 2020 3 31) minus1M = dateTs 2020 3 2 ∧
    subDurRaw (dateTs 2020 3 31 + 36000) plus1M = dateTs 2020 3 2 + 36000 := by decide +kernel

/-- Shared C/Rust divergence from openCypher: `duration({days:-1})` is stored as the
instant 1969-12-31, which decomposes as `P-1Y11M30D`; so `date('2020-03-01') +
duration({days:-1})` is 2020-03-02 (openCypher: 2020-02-29). -/
theorem neg_days_encoding :
    decomposeDuration ((constructDuration 0 0 0 (-1) 0 0 0).getD 0) = (-1, 11, 30 * 86400) ∧
    addDurRaw (dateTs 2020 3 1) ((constructDuration 0 0 0 (-1) 0 0 0).getD 0) = dateTs 2020 3 2 := by
  decide +kernel

/-! ## `Time ± Duration` (since #2985, `add_slow` `value.rs:1069-1070`, `Sub` `:1111-1112`)

`(t.rem_euclid(86400) ± dur.rem_euclid(86400)).rem_euclid(86400)`. Lean's `%` on `Int`
with a positive divisor is `rem_euclid`. Both sides lie in `[0, 86400)` and their sum or
difference in `(-86400, 172800)`, so the i64 arithmetic cannot overflow. -/

def timeAddDur (t dur : Int) : Int := (t % 86400 + dur % 86400) % 86400
def timeSubDur (t dur : Int) : Int := (t % 86400 - dur % 86400) % 86400

/-- A `Time ± Duration` result is always a time of day. -/
theorem timeAddDur_range (t dur : Int) : 0 ≤ timeAddDur t dur ∧ timeAddDur t dur < 86400 := by
  unfold timeAddDur; omega
theorem timeSubDur_range (t dur : Int) : 0 ≤ timeSubDur t dur ∧ timeSubDur t dur < 86400 := by
  unfold timeSubDur; omega

/-- **W2-temporal-6 fixed.** Main's `Time + Duration` is the time of day of the
date-arithmetic result `add_duration_to_timestamp t dur` (every duration encoding is whole
days plus a clock offset), i.e. the openCypher result, for every `t` and `dur`. -/
theorem timeAddDur_eq_tod (t dur : Int) : timeAddDur t dur = addDurRaw t dur % 86400 := by
  unfold timeAddDur addDurRaw decomposeDuration shiftMonths
  generalize civilFromDays (dur / 86400) = c1
  obtain ⟨y1, m1, d1⟩ := c1
  generalize civilFromDays (t / 86400) = c2
  obtain ⟨y2, m2, d2⟩ := c2
  simp only
  split <;> (try split) <;> omega

/-- It is `(t + dur) mod 86400` and `(t − dur) mod 86400`. -/
theorem timeAddDur_eq (t dur : Int) : timeAddDur t dur = (t + dur) % 86400 := by
  unfold timeAddDur; omega
theorem timeSubDur_eq (t dur : Int) : timeSubDur t dur = (t - dur) % 86400 := by
  unfold timeSubDur; omega

/-- `(t + dur) - dur = t` on times of day (false before #2985, where `Time ± Duration`
went through the date arithmetic). -/
theorem timeSub_timeAdd (t dur : Int) (h0 : 0 ≤ t) (h1 : t < 86400) :
    timeSubDur (timeAddDur t dur) dur = t := by
  unfold timeSubDur timeAddDur; omega

/-- The repro of W2-temporal-6 on main: `localtime('01:00') + duration({days:1})` and
`+ duration({months:1})` are `localtime('01:00')` (C: the same). -/
theorem time_normalised :
    timeAddDur 3600 ((constructDuration 0 0 0 1 0 0 0).getD 0) = 3600 ∧
    timeAddDur 3600 plus1M = 3600 ∧ timeSubDur 3600 plus1M = 3600 := by decide +kernel

/-- #2987: `n * 7` for `n = 2635249153387078803` is out of i64 (it used to wrap to `P5D` in
release: unchecked `n * 7` at `temporal.rs:391` of 9d3869aa8^), so `checked_mul` returns
`None` and `duration('P…W')` is an error. -/
theorem week_string_checked_none : chkMul 2635249153387078803 7 = none := by decide

/-! ## Ordering: Date/Time/Duration compare as their `i64` (`compare_value`, `value.rs:1266`), a total order -/

theorem temporal_order_total (a b : Int) : a ≤ b ∨ b ≤ a := Int.le_total a b
theorem temporal_order_trans (a b c : Int) (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c := Int.le_trans h1 h2
theorem temporal_order_antisymm (a b : Int) (h1 : a ≤ b) (h2 : b ≤ a) : a = b := Int.le_antisymm h1 h2

/-- `date` order is calendar (lexicographic) order: consecutive days of a month are
consecutive, so the i64 order agrees with the calendar within a month. -/
theorem dateTs_mono_day (y m d : Int) : dateTs y m d < dateTs y m (d + 1) := by
  unfold dateTs
  rw [daysFromCivil_day y m d, daysFromCivil_day y m (d + 1)]; omega

end FalkorTemporal
