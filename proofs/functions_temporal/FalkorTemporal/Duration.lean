import FalkorTemporal.CalendarLaws
/-!
# Durations and date arithmetic

| here | there |
| --- | --- |
| `chk`, `chkMul`, `chkAdd` | `i64::checked_*` |
| `constructDuration`       | `construct_duration_secs` — `temporal.rs:482` |
| `decomposeDuration`       | `decompose_duration` — `temporal.rs:523` (`y - 1970` is i32: `toI32`) |
| `shiftMonths`             | `shift_months` — `value.rs:752` (shared by `+` and `-` since #2984) |
| `addDur`                  | `add_duration_to_timestamp` — `value.rs:728` |
| `subDur`                  | `sub_duration_from_timestamp` — `value.rs:738` (`pre2984_subDur`: the old clamping one, history) |
| `durAddDur`               | `Add for Value`, `(Duration, Duration)` arm — `value.rs:1044` |
| `timeAddDur` / `timeSubDur` | `Add`/`Sub for Value`, `(Time, Duration)` arms — `add_slow` `value.rs:1063-1064`, `Sub` `:1105-1106` (since #2985) |
| `wrapMul`                 | pre-#2987 release-mode `n * 7` in `parse_duration_string` (history; main is checked, `Parse.lean`) |

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

/-- `construct_duration_secs` (`temporal.rs:482-520`), `Err` ↦ `none`. -/
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
  chkAdd anchor extra

/-- `construct_duration_secs` on a years/months/seconds triple, when nothing overflows. -/
theorem construct_ym_s (Y M S : Int) (h1 : I64 (Y * 12)) (h2 : I64 (Y * 12 + M))
    (h3 : I32 (Y * 12 + M)) (h4 : I32 (1970 + (Y * 12 + M) / 12)) (h5 : I64 S)
    (h6 : I64 (daysFromCivil (1970 + (Y * 12 + M) / 12) ((Y * 12 + M) % 12 + 1) 1 * 86400 + S)) :
    constructDuration Y M 0 0 0 0 S =
      some (daysFromCivil (1970 + (Y * 12 + M) / 12) ((Y * 12 + M) % 12 + 1) 1 * 86400 + S) := by
  have z : I64 0 := by unfold I64; decide
  unfold constructDuration chkMul chkAdd
  simp only [Int.zero_mul, Int.zero_add, Int.mul_zero, chk_of _ h1, chk_of _ h2, chk32_of _ h3,
    chk32_of _ h4, chk_of _ z, chk_of _ h5, daysFromCivilT_eq, chk_of _ h6, bind, Option.bind]

/-- `decompose_duration` (`temporal.rs:523-533`) as built in release (i32 wrap). -/
def decomposeDuration (dur : Int) : Int × Int × Int :=
  let days := dur / 86400
  let tod := dur % 86400
  let (y, m, d) := civilFromDays days
  (toI32 (y - 1970), m - 1, (d - 1) * 86400 + tod)

/-! ## Law 1: `construct ∘ decompose = id` (what `Duration ± Duration` relies on) -/

theorem construct_decompose (dur : Int) (hd : I64 dur)
    (hy : -2 ^ 31 + 1970 ≤ (civilFromDaysI (dur / 86400)).1 ∧ (civilFromDaysI (dur / 86400)).1 < 2 ^ 31)
    (hmo : I32 (((civilFromDaysI (dur / 86400)).1 - 1970) * 12 + ((civilFromDaysI (dur / 86400)).2.1 - 1))) :
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

/-- `shift_months` (`value.rs:752-783`, since #2984): move `ts` by whole years and months,
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

/-- `add_duration_to_timestamp` (`value.rs:728-735`). -/
def addDur (ts dur : Int) : Int :=
  let (years, months, rem) := decomposeDuration dur
  let (days, tod) := shiftMonths ts years months
  days * 86400 + tod + rem

/-- `sub_duration_from_timestamp` (`value.rs:738-745`, since #2984): the same
`shift_months` with the years and months negated, so it rolls over like `addDur`. -/
def subDur (ts dur : Int) : Int :=
  let (years, months, rem) := decomposeDuration dur
  let (days, tod) := shiftMonths ts (-years) (-months)
  days * 86400 + tod - rem

/-- Pre-#2984 `sub_duration_from_timestamp` (history, W2-temporal-5): it *clamped*
(`d.min(max_day)`) where `addDur` *overflows into the next month*. -/
def pre2984_subDur (ts dur : Int) : Int :=
  let (years, months, rem) := decomposeDuration dur
  let days := ts / 86400
  let tod := ts % 86400
  let (y, m, d) := civilFromDays days
  let newYear := y - years
  let nmr := m - months
  let adjYear := newYear + (nmr - 1) / 12
  let adjMonth := (nmr - 1) % 12 + 1
  let maxDay := daysInMonth adjYear adjMonth
  let day := min d maxDay
  daysFromCivil adjYear adjMonth day * 86400 + tod - rem

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
    subDur (addDur (dateTs y m d) (daysFromCivil (1970 + t / 12) (t % 12 + 1) 1 * 86400))
        (daysFromCivil (1970 + t / 12) (t % 12 + 1) 1 * 86400) = dateTs y m d := by
  have hdec := decompose_monthsDur t ht
  have hc := civil_dfc y m d hv
  obtain ⟨hm1, hm2, hd1, _⟩ := hv
  unfold addDur subDur dateTs shiftMonths
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
`shift_months`, so `d - dur = d + (-dur)` (pre-#2984 this failed, `pre2984_sub_vs_add_neg`). -/
theorem sub_eq_add_neg_months (ts t : Int) (ht : -1000000 ≤ t ∧ t ≤ 1000000) :
    subDur ts (daysFromCivil (1970 + t / 12) (t % 12 + 1) 1 * 86400) =
      addDur ts (daysFromCivil (1970 + (-t) / 12) ((-t) % 12 + 1) 1 * 86400) := by
  have h1 := decompose_monthsDur t ht
  have h2 := decompose_monthsDur (-t) ⟨by omega, by omega⟩
  unfold subDur addDur
  simp only [h1, h2]
  rw [shiftMonths_total ts (-(t / 12)), shiftMonths_total ts ((-t) / 12)]
  have : -(t / 12) * 12 + -(t % 12) = (-t) / 12 * 12 + (-t) % 12 := by omega
  rw [this]; omega

/-- The same for any duration: `subDur` is `addDur` with the decomposed years and months
negated (the remaining seconds are subtracted). -/
theorem subDur_eq_shift (ts dur : Int) :
    subDur ts dur =
      (shiftMonths ts 0 (-((decomposeDuration dur).1 * 12 + (decomposeDuration dur).2.1))).1 * 86400 +
        (shiftMonths ts 0 0).2 - (decomposeDuration dur).2.2 := by
  unfold subDur
  generalize decomposeDuration dur = c
  obtain ⟨Y, M, R⟩ := c
  simp only
  rw [shiftMonths_total ts (-Y)]
  have e : -Y * 12 + -M = -(Y * 12 + M) := by omega
  rw [e]
  unfold shiftMonths; simp only

/-! ## Counterexamples (all evaluated by the kernel) -/

/-- `duration({months: -1})` and `duration({months: 1})`. -/
def minus1M : Int := (constructDuration 0 (-1) 0 0 0 0 0).getD 0
def plus1M : Int := (constructDuration 0 1 0 0 0 0 0).getD 0

/-- **Bug W2-temporal-5 (history, fixed by #2984 5986f4df1)**: before #2984
`date('2020-03-31') - duration({months:1})` was 2020-02-29 in Rust (clamp) but 2020-03-02
in C, and Rust's own `date + duration({months:-1})` was 2020-03-02: `d - x ≠ d + (-x)`. -/
theorem pre2984_sub_vs_add_neg :
    pre2984_subDur (dateTs 2020 3 31) plus1M = dateTs 2020 2 29 ∧
    addDur (dateTs 2020 3 31) minus1M = dateTs 2020 3 2 := by decide +kernel

/-- The repro of W2-temporal-5 on main (and #2984's flow test): both are 2020-03-02 (C:
2020-03-02), and `localdatetime('2020-03-31T10:00:00') - duration({months:1})` is
`2020-03-02T10:00:00`. -/
theorem sub_vs_add_neg :
    subDur (dateTs 2020 3 31) plus1M = dateTs 2020 3 2 ∧
    addDur (dateTs 2020 3 31) minus1M = dateTs 2020 3 2 ∧
    subDur (dateTs 2020 3 31 + 36000) plus1M = dateTs 2020 3 2 + 36000 := by decide +kernel

/-- `pre2984_subDur` and `subDur` agree whenever the clamp did not fire (day ≤ 28): the
fix only changes results that were wrong. -/
theorem subDur_eq_pre2984 (y m d t : Int) (hv : ValidDate y m d) (hd : d ≤ 28)
    (hy : SmallYear y) (ht : -1000000 ≤ t ∧ t ≤ 1000000) :
    subDur (dateTs y m d) (daysFromCivil (1970 + t / 12) (t % 12 + 1) 1 * 86400) =
      pre2984_subDur (dateTs y m d) (daysFromCivil (1970 + t / 12) (t % 12 + 1) 1 * 86400) := by
  have hdec := decompose_monthsDur t ht
  have hc := civil_dfc y m d hv
  obtain ⟨hm1, hm2, hd1, _⟩ := hv
  unfold subDur pre2984_subDur dateTs shiftMonths
  simp only [hdec]
  have e1 : daysFromCivil y m d * 86400 / 86400 = daysFromCivil y m d := by omega
  have e2 : daysFromCivil y m d * 86400 % 86400 = 0 := by omega
  unfold civilFromDays
  have i1 : I32 y := by unfold I32 SmallYear at *; omega
  simp only [e1, e2, hc, toI32_id _ i1]
  have eY : y + -(t / 12) = y - t / 12 := by omega
  have eM : m + -(t % 12) = m - t % 12 := by omega
  rw [eY, eM]
  generalize hA : y - t / 12 + (m - t % 12 - 1) / 12 = A
  generalize hB : (m - t % 12 - 1) % 12 + 1 = B
  have hB1 : 1 ≤ B ∧ B ≤ 12 := by omega
  have hdimB : 28 ≤ daysInMonth A B := by
    unfold daysInMonth; split <;> (try split) <;> (try split) <;> omega
  have nof : ¬ (d > daysInMonth A B) := by omega
  have hmin : min d (daysInMonth A B) = d := by omega
  simp only [nof, ite_false, hmin]

/-- Shared C/Rust divergence from openCypher: `duration({days:-1})` is stored as the
instant 1969-12-31, which decomposes as `P-1Y11M30D`; so `date('2020-03-01') +
duration({days:-1})` is 2020-03-02 (openCypher: 2020-02-29). -/
theorem neg_days_encoding :
    decomposeDuration ((constructDuration 0 0 0 (-1) 0 0 0).getD 0) = (-1, 11, 30 * 86400) ∧
    addDur (dateTs 2020 3 1) ((constructDuration 0 0 0 (-1) 0 0 0).getD 0) = dateTs 2020 3 2 := by
  decide +kernel

/-- **Bug W2-temporal-6 (history, fixed by #2985 289c201fe)**: before #2985
`Time + Duration` reused `add_duration_to_timestamp` and was not reduced modulo a day,
so `localtime('01:00') + duration({days:1}) = localtime('01:00')` was false (C: true),
and `+ duration({months:1})` likewise. The stored seconds differed though both printed
`01:00:00`. -/
theorem pre2985_time_not_normalised :
    addDur 3600 ((constructDuration 0 0 0 1 0 0 0).getD 0) = 3600 + 86400 ∧
    addDur 3600 plus1M = 3600 + 31 * 86400 := by decide +kernel

/-! ## `Time ± Duration` (since #2985, `add_slow` `value.rs:1063-1064`, `Sub` `:1105-1106`)

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
theorem timeAddDur_eq_tod (t dur : Int) : timeAddDur t dur = addDur t dur % 86400 := by
  unfold timeAddDur addDur decomposeDuration shiftMonths
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

/-- Pre-#2987 release-mode `n * 7` (`temporal.rs:391` at 8621212fc): two's-complement wrap. -/
def wrapMul (a b : Int) : Int := (a * b + 2 ^ 63) % 2 ^ 64 - 2 ^ 63

/-- **Bug W2-temporal-8 (history, fixed by #2987 9d3869aa8)**: before #2987
`duration('P2635249153387078803W')` was `P5D`: the week count wrapped (`n * 7` unchecked,
then `days += n`); debug builds panicked instead. Main uses `checked_mul`/`checked_add`
and errors: `Parse.lean` `dateLoop_week_overflow`, `week_string_overflow_errors`. -/
theorem pre2987_week_string_wraps : wrapMul 2635249153387078803 7 = 5 := by decide

/-- On main the same product is out of i64, so `checked_mul` returns `None`. -/
theorem week_string_checked_none : chkMul 2635249153387078803 7 = none := by decide

/-! ## Ordering: Date/Time/Duration compare as their `i64` (`value.rs:1294`), a total order -/

theorem temporal_order_total (a b : Int) : a ≤ b ∨ b ≤ a := Int.le_total a b
theorem temporal_order_trans (a b c : Int) (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c := Int.le_trans h1 h2
theorem temporal_order_antisymm (a b : Int) (h1 : a ≤ b) (h2 : b ≤ a) : a = b := Int.le_antisymm h1 h2

/-- `date` order is calendar (lexicographic) order: consecutive days of a month are
consecutive, so the i64 order agrees with the calendar within a month. -/
theorem dateTs_mono_day (y m d : Int) : dateTs y m d < dateTs y m (d + 1) := by
  unfold dateTs
  rw [daysFromCivil_day y m d, daysFromCivil_day y m (d + 1)]; omega

end FalkorTemporal
