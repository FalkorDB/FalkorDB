import FalkorTemporal.Pure
import FalkorTemporal.CalendarLaws
/-!
# Temporal accessors and formatting (value.rs:261-607)

chrono's `Utc.timestamp_opt(t, 0)` is modelled by `ChronoDT`: whether it yields
`LocalResult::Single` (`valid`), and the two derived fields we do not recompute
(weekday, ISO week). Its civil fields are *stated* to be the proleptic Gregorian calendar
of the day number (`civilFromDaysI`, the Hinnant algorithm proved correct in
CalendarLaws.lean) and the second of the day — that is chrono's documented semantics.
-/
namespace FalkorTemporal.Fns

structure ChronoDT where
  /-- `timestamp_opt(t, 0)` is `Single` (chrono's ±262 143-year range). -/
  valid : Int → Bool
  /-- `weekday().num_days_from_sunday()` of a day number. -/
  dowSun : Int → Nat
  dowSun_lt : ∀ d, dowSun d < 7
  /-- `ordinal()`, `iso_week().week()`, `iso_week().year()` of a day number. -/
  ordinal : Int → Int
  isoWeek : Int → Int
  isoYear : Int → Int

/-- Civil fields of a timestamp (`DateTime<Utc>` accessors). -/
def dayOf (t : Int) : Int := t / 86400
def yearOf (t : Int) : Int := (civilFromDaysI (dayOf t)).1
def monthOf (t : Int) : Int := (civilFromDaysI (dayOf t)).2.1
def mdayOf (t : Int) : Int := (civilFromDaysI (dayOf t)).2.2
def hourOf (t : Int) : Int := t % 86400 / 3600
def minuteOf (t : Int) : Int := t % 3600 / 60
def secondOf (t : Int) : Int := t % 60

/-- `str::eq_ignore_ascii_case` (Lean's `Char.toLower` is ASCII-only, like Rust's). -/
def eqIC (a b : String) : Bool := a.toList.map Char.toLower == b.toList.map Char.toLower

variable (K : ChronoDT)

def quarterOf (t : Int) : Int := (monthOf t - 1) / 3 + 1
/-- `(dt.date_naive() - quarter_start.date_naive()).num_days() + 1` -/
def dayOfQuarter (t : Int) : Int :=
  dayOf t - daysFromCivil (yearOf t) ((quarterOf t - 1) * 3 + 1) 1 + 1
def weekDayIso (t : Int) : Int := let w := K.dowSun (dayOf t); if w = 0 then 7 else w

/-- `get_datetime_component` (value.rs:478), the `if … else if` chain in source order. -/
def getDatetimeComponent (t : Int) (c : String) : Except String (Option Int) :=
  if !K.valid t then .ok none else
  if eqIC c "second" then .ok (some (secondOf t))
  else if eqIC c "minute" then .ok (some (minuteOf t))
  else if eqIC c "hour" then .ok (some (hourOf t))
  else if eqIC c "day" then .ok (some (mdayOf t))
  else if eqIC c "month" then .ok (some (monthOf t))
  else if eqIC c "year" then .ok (some (yearOf t))
  else if eqIC c "dayOfWeek" then .ok (some (K.dowSun (dayOf t)))
  else if eqIC c "weekDay" then .ok (some (weekDayIso K t))
  else if eqIC c "ordinalDay" then .ok (some (K.ordinal (dayOf t)))
  else if eqIC c "quarter" then .ok (some (quarterOf t))
  else if eqIC c "week" then .ok (some (K.isoWeek (dayOf t)))
  else if eqIC c "weekYear" then .ok (some (K.isoYear (dayOf t)))
  else if eqIC c "dayOfQuarter" || eqIC c "quarterDay" then .ok (some (dayOfQuarter t))
  else if eqIC c "millisecond" then .ok (some 0)
  else if eqIC c "microsecond" || eqIC c "nanosecond" then .ok (some 0)
  else .error s!"unknown datetime component {c}"

/-- `get_date_component` (value.rs:537). -/
def getDateComponent (t : Int) (c : String) : Except String (Option Int) :=
  if !K.valid t then .ok none else
  if eqIC c "day" then .ok (some (mdayOf t))
  else if eqIC c "month" then .ok (some (monthOf t))
  else if eqIC c "year" then .ok (some (yearOf t))
  else if eqIC c "dayOfWeek" then .ok (some (K.dowSun (dayOf t)))
  else if eqIC c "weekDay" then .ok (some (weekDayIso K t))
  else if eqIC c "ordinalDay" then .ok (some (K.ordinal (dayOf t)))
  else if eqIC c "quarter" then .ok (some (quarterOf t))
  else if eqIC c "week" then .ok (some (K.isoWeek (dayOf t)))
  else if eqIC c "weekYear" then .ok (some (K.isoYear (dayOf t)))
  else if eqIC c "dayOfQuarter" || eqIC c "quarterDay" then .ok (some (dayOfQuarter t))
  else .error s!"unknown date component {c}"

/-- `get_time_component` (value.rs:585). -/
def getTimeComponent (t : Int) (c : String) : Except String (Option Int) :=
  if !K.valid t then .ok none else
  if eqIC c "second" then .ok (some (secondOf t))
  else if eqIC c "minute" then .ok (some (minuteOf t))
  else if eqIC c "hour" then .ok (some (hourOf t))
  else .error s!"unknown time component {c}"

/-! ### Theorems -/

theorem components_invalid (t : Int) (c : String) (h : K.valid t = false) :
    getDatetimeComponent K t c = .ok none ∧ getDateComponent K t c = .ok none ∧
    getTimeComponent K t c = .ok none := by
  simp [getDatetimeComponent, getDateComponent, getTimeComponent, h]

/-- Clock fields are in range. -/
theorem clock_ranges (t : Int) :
    0 ≤ hourOf t ∧ hourOf t < 24 ∧ 0 ≤ minuteOf t ∧ minuteOf t < 60 ∧ 0 ≤ secondOf t ∧ secondOf t < 60 := by
  simp only [hourOf, minuteOf, secondOf]; omega

/-- Calendar fields are a real date; `quarter` ∈ 1..4; `weekDay` ∈ 1..7 (ISO, Sunday = 7). -/
theorem calendar_ranges (t : Int) :
    ValidDate (yearOf t) (monthOf t) (mdayOf t) ∧ 1 ≤ quarterOf t ∧ quarterOf t ≤ 4 ∧
    1 ≤ weekDayIso K t ∧ weekDayIso K t ≤ 7 := by
  have hv := civil_valid (dayOf t)
  have hm : 1 ≤ monthOf t ∧ monthOf t ≤ 12 := ⟨hv.1, hv.2.1⟩
  have hw := K.dowSun_lt (dayOf t)
  refine ⟨hv, ?_, ?_, ?_, ?_⟩
  · simp only [quarterOf]; omega
  · simp only [quarterOf]; omega
  · simp only [weekDayIso]; split <;> omega
  · simp only [weekDayIso]; split <;> omega

/-- The timestamp is recovered from its civil date and clock fields. -/
theorem fields_roundtrip (t : Int) :
    daysFromCivil (yearOf t) (monthOf t) (mdayOf t) * 86400 + hourOf t * 3600 + minuteOf t * 60 +
      secondOf t = t := by
  have := dfc_civil (dayOf t)
  simp only [yearOf, monthOf, mdayOf]
  rw [this]
  simp only [dayOf, hourOf, minuteOf, secondOf]
  omega

theorem time_component_cases (t : Int) (c : String) (h : K.valid t = true) :
    getTimeComponent K t "hour" = .ok (some (hourOf t)) ∧
    getTimeComponent K t "HOUR" = .ok (some (hourOf t)) ∧
    getTimeComponent K t "day" = .error "unknown time component day" := by
  simp [getTimeComponent, h, eqIC]; decide

theorem datetime_unknown (t : Int) (h : K.valid t = true) :
    getDatetimeComponent K t "epochSeconds" = .error "unknown datetime component epochSeconds" := by
  simp [getDatetimeComponent, h, eqIC]; decide

/-! ### `format_datetime` (value.rs:261), `format_date` (:281), `format_time`

Since #2986/#3575 `format_datetime` and `format_date` no longer go through chrono: they take
`div_euclid`/`rem_euclid` of the i64, `civil_from_days_wide` (year kept as i64, so every
persisted timestamp has its real year) and `push_date`. They are total: no
`<invalid timestamp>` branch. `format_time` still uses chrono. -/

/-- `{:02}` / `{:04}` of a non-negative number: left-pad with `0` (no truncation). -/
def pad (w : Nat) (n : Nat) : String :=
  let s := toString n
  String.ofList (List.replicate (w - s.length) '0') ++ s

/-- `write_two_digits` (value.rs:853): `v / 10 % 10` and `v % 10` as ASCII digits. -/
def twoDigits (v : Nat) : String :=
  String.ofList [Char.ofNat (48 + v / 10 % 10), Char.ofNat (48 + v % 10)]

/-- `push_date` (value.rs:828): years `0..=9999` through `write_two_digits` (4 digits), any
other year as `{sign}{:04}` of its magnitude, so no digit is lost and a negative year keeps
its `-`. -/
def pushDate (y : Int) (m d : Nat) : String :=
  if 0 ≤ y ∧ y ≤ 9999 then
    twoDigits (y.toNat / 100) ++ twoDigits (y.toNat % 100) ++ "-" ++ twoDigits m ++ "-" ++ twoDigits d
  else (if y < 0 then "-" else "") ++ pad 4 y.natAbs ++ "-" ++ pad 2 m ++ "-" ++ pad 2 d

/-- `format_date` (value.rs:281). `/` is `div_euclid` on a positive divisor. -/
def formatDate (t : Int) : String :=
  let (y, m, d) := civilFromDaysI (t / 86400)
  pushDate y m.toNat d.toNat

/-- `format_datetime` (value.rs:261): the date, then `T` and `write_two_digits` of the
hour, minute, second of `t.rem_euclid(86400)`. -/
def formatDatetime (t : Int) : String :=
  let (y, m, d) := civilFromDaysI (t / 86400)
  let secs := (t % 86400).toNat
  pushDate y m.toNat d.toNat ++ "T" ++ twoDigits (secs / 3600) ++ ":" ++ twoDigits (secs % 3600 / 60) ++
    ":" ++ twoDigits (secs % 60)

theorem twoDigits_eq_pad (v : Nat) (h : v < 100) : twoDigits v = pad 2 v := by
  revert v h; decide

theorem fourDigits_aux :
    ∀ a, a < 100 → ∀ b, b < 100 → twoDigits a ++ twoDigits b = pad 4 (a * 100 + b) := by
  decide +kernel

theorem fourDigits_eq_pad (y : Nat) (h : y ≤ 9999) :
    twoDigits (y / 100) ++ twoDigits (y % 100) = pad 4 y := by
  have := fourDigits_aux (y / 100) (by omega) (y % 100) (by omega)
  rw [this]; congr 1; omega

/-- `push_date` is, for every year, the sign, the magnitude padded to 4 digits (never
truncated), and `-MM-DD`, for the month and day of any real date. -/
theorem pushDate_eq (y : Int) (m d : Nat) (hm : m < 100) (hd : d < 100) :
    pushDate y m d = (if y < 0 then "-" else "") ++ pad 4 y.natAbs ++ "-" ++ pad 2 m ++ "-" ++ pad 2 d := by
  unfold pushDate
  split
  · rename_i h
    have e : y.natAbs = y.toNat := by omega
    have h1 : ¬ y < 0 := by omega
    have := fourDigits_eq_pad y.toNat (by omega)
    rw [e, if_neg h1, ← this, twoDigits_eq_pad m hm, twoDigits_eq_pad d hd]
    simp [String.append_assoc]
  · rfl

/-- Any timestamp (any `i64`) formats as its exact civil date: year all digits, sign
iff negative, then month and day as two digits. -/
theorem formatDate_spec (t : Int) :
    formatDate t = (if yearOf t < 0 then "-" else "") ++ pad 4 (yearOf t).natAbs ++ "-" ++
      pad 2 (monthOf t).toNat ++ "-" ++ pad 2 (mdayOf t).toNat := by
  have hv := civil_valid (t / 86400)
  simp only [yearOf, monthOf, mdayOf, dayOf]
  unfold formatDate
  obtain ⟨y, m, d, hc⟩ : ∃ y m d, civilFromDaysI (t / 86400) = (y, m, d) := ⟨_, _, _, rfl⟩
  simp only [hc] at hv ⊢
  have hdim : daysInMonth y m ≤ 31 := by
    unfold daysInMonth; split <;> (try split) <;> (try split) <;> omega
  obtain ⟨h1, h2, h3, h4⟩ := hv
  exact pushDate_eq y m.toNat d.toNat (by omega) (by omega)

theorem formatDatetime_spec (t : Int) :
    formatDatetime t = formatDate t ++ "T" ++ pad 2 (hourOf t).toNat ++ ":" ++ pad 2 (minuteOf t).toNat ++
      ":" ++ pad 2 (secondOf t).toNat := by
  have h0 := Int.emod_nonneg t (by decide : (86400:Int) ≠ 0)
  have h1 := Int.emod_lt_of_pos t (by decide : (0:Int) < 86400)
  unfold formatDatetime formatDate
  simp only [hourOf, minuteOf, secondOf]
  obtain ⟨y, m, d, hc⟩ : ∃ y m d, civilFromDaysI (t / 86400) = (y, m, d) := ⟨_, _, _, rfl⟩
  simp only [hc]
  have e1 : (t % 86400).toNat / 3600 = (t % 86400 / 3600).toNat := by omega
  have e2 : (t % 86400).toNat % 3600 / 60 = (t % 3600 / 60).toNat := by omega
  have e3 : (t % 86400).toNat % 60 = (t % 60).toNat := by omega
  rw [e1, e2, e3, twoDigits_eq_pad _ (by omega), twoDigits_eq_pad _ (by omega),
    twoDigits_eq_pad _ (by omega)]

/-- **#2986 / #2746 fixed.** Years past 9999 keep every digit and negative years their sign
(before: `12345-01-01` printed `2345-01-01`, `-0005` as `0005`, and `i64::MAX` seconds a year
cut to `i32`). The values of `value.rs`'s unit test `extreme_timestamps_keep_their_full_year`: -/
theorem format_extremes :
    formatDate (2 ^ 63 - 1) = "292277026596-12-04" ∧
    formatDatetime (2 ^ 63 - 1) = "292277026596-12-04T15:30:07" ∧
    formatDate (-2 ^ 63) = "-292277022657-01-27" ∧
    formatDatetime (-2 ^ 63) = "-292277022657-01-27T08:29:52" := by decide +kernel

theorem format_wide_years :
    pushDate 12345 1 1 = "12345-01-01" ∧ pushDate (-5) 1 1 = "-0005-01-01" ∧
    pushDate 5 1 1 = "0005-01-01" ∧ pushDate 9999 12 31 = "9999-12-31" ∧
    pushDate 10000 1 1 = "10000-01-01" := by decide +kernel

/-- The old defect (digits of `|y|` mod 10⁴, no sign) is gone: distinct years print apart. -/
theorem formatDate_wide_distinct :
    pushDate 12345 1 1 ≠ pushDate 2345 1 1 ∧ pushDate (-5) 1 1 ≠ pushDate 5 1 1 := by decide +kernel

/-- `format_time` (value.rs:291), still chrono's validity test. -/
def formatTime (t : Int) : String :=
  if K.valid t then pad 2 (hourOf t).toNat ++ ":" ++ pad 2 (minuteOf t).toNat ++ ":" ++
    pad 2 (secondOf t).toNat
  else s!"<invalid timestamp: {t}>"

theorem pad2_length : ∀ n, n < 100 → (pad 2 n).length = 2 := by decide

/-- `format_time` of a valid timestamp is exactly `HH:MM:SS` (8 characters). -/
theorem formatTime_length (t : Int) (h : K.valid t = true) : (formatTime K t).length = 8 := by
  have r := clock_ranges t
  simp only [formatTime, h, if_true, String.length_append]
  rw [pad2_length _ (by omega), pad2_length _ (by omega), pad2_length _ (by omega)]
  rfl

theorem format_invalid (t : Int) (h : K.valid t = false) :
    formatTime K t = s!"<invalid timestamp: {t}>" := by
  simp [formatTime, h]

end FalkorTemporal.Fns
