import FalkorTemporal.Pure
import FalkorTemporal.CalendarLaws
/-!
# Temporal accessors and formatting (value.rs:256-605), `Point` (value.rs:110-141)

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

/-- `get_datetime_component` (value.rs:473), the `if … else if` chain in source order. -/
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

/-- `get_date_component` (value.rs:532). -/
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

/-- `get_time_component` (value.rs:580). -/
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

/-! ### `format_datetime` (value.rs:256) and `format_time` (value.rs:285) -/

/-- `{:02}` / `{:04}` of a non-negative number: left-pad with `0` (no truncation). -/
def pad (w : Nat) (n : Nat) : String :=
  let s := toString n
  String.ofList (List.replicate (w - s.length) '0') ++ s

/-- `{:04}` of an `i32` year (Rust prints `-` then pads the magnitude to width-1). -/
def padYear (y : Int) : String := if y < 0 then "-" ++ pad 3 y.natAbs else pad 4 y.toNat

def formatDatetime (t : Int) : String :=
  if K.valid t then
    padYear (yearOf t) ++ "-" ++ pad 2 (monthOf t).toNat ++ "-" ++ pad 2 (mdayOf t).toNat ++ "T" ++
      pad 2 (hourOf t).toNat ++ ":" ++ pad 2 (minuteOf t).toNat ++ ":" ++ pad 2 (secondOf t).toNat
  else s!"<invalid timestamp: {t}>"

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
    formatTime K t = s!"<invalid timestamp: {t}>" ∧ formatDatetime K t = s!"<invalid timestamp: {t}>" := by
  simp [formatTime, formatDatetime, h]

end FalkorTemporal.Fns
