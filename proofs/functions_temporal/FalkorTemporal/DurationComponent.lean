import FalkorTemporal.Components
import FalkorTemporal.Constructors
/-!
# `get_duration_component` (value.rs:613): chrono-bounded, unlike every other duration path

| here | there |
| --- | --- |
| `chronoK`                 | chrono `Utc.timestamp_opt(t, 0)`: `Single` iff the year is within `MIN_YEAR = -262143 ..= MAX_YEAR = 262142` (`chronoYearOk`, Constructors.lean) |
| `getDurationComponent`    | `Value::get_duration_component` — `value.rs:613-700`, the `if … else if` chain in source order (the C `duration.c` decomposition) |
| `decomposeDuration`       | `decompose_duration` — `temporal.rs:549`, what `format_duration`, `Duration ± Duration` and `construct_duration_secs` use (Duration.lean) |

`getDurationComponent` goes through chrono, so a duration whose instant leaves chrono's year range
yields `null` for years/months/days/hours/minutes/seconds, although `check_date_range` (±`MAX_DATE_YEAR`
≈ 2.1·10⁹ years) accepts the duration and `decompose_duration` / `format_duration` handle it.
`durComp_agrees_decompose`: inside chrono's range every component equals the `decompose_duration`
one. `durComp_null_but_representable` / `bug_duration_component_null`: outside it they do not (BUG,
confirmed live: Rust `null`, C the exact component).
-/
namespace FalkorTemporal.Fns

/-- chrono's timestamp range as a `ChronoDT`; the weekday is the real one (1970-01-01 is a
Thursday), the ISO-week fields are irrelevant to durations. -/
def chronoK : ChronoDT where
  valid t := chronoYearOk (yearOf t)
  dowSun d := ((d + 4) % 7).toNat
  dowSun_lt d := by omega
  ordinal _ := 0
  isoWeek _ := 0
  isoYear _ := 0

def knownDurationComponent (c : String) : Bool :=
  eqIC c "years" || eqIC c "months" || eqIC c "weeks" || eqIC c "days" || eqIC c "hours" ||
    eqIC c "minutes" || eqIC c "seconds"

/-- `get_duration_component` (value.rs:613). The `Float` result is the integer it carries
(every component is a small integer, exact in `f64`); `none` is `Value::Null`. `month_diff < 0`
is dead (`dt.month() >= 1`) and omitted. -/
def getDurationComponent (K : ChronoDT) (t : Int) (c : String) : Except String (Option Int) :=
  if !knownDurationComponent c then .error s!"unknown duration component {c}"
  else if eqIC c "weeks" then .ok (some 0)
  else if !K.valid t then .ok none
  else
    let yd := yearOf t - 1970
    let md := monthOf t - 1
    if eqIC c "years" then .ok (some yd)
    else if eqIC c "months" then .ok (some md)
    else
      let anchor := daysFromCivil (1970 + yd) (1 + md) 1 * 86400
      if !K.valid anchor then
        .error s!"Invalid anchor date for duration decomposition (duration_secs={t})"
      else
        let rem := t - anchor
        if eqIC c "days" then .ok (some (rem / 86400))
        else if eqIC c "hours" then .ok (some (rem % 86400 / 3600))
        else if eqIC c "minutes" then .ok (some (rem % 3600 / 60))
        else .ok (some (rem % 60))

theorem durComp_unknown (K : ChronoDT) (t : Int) (c : String) (h : knownDurationComponent c = false) :
    getDurationComponent K t c = .error s!"unknown duration component {c}" := by
  simp [getDurationComponent, h]

/-- `weeks` is always `0`, for every duration (no chrono work). -/
theorem durComp_weeks (K : ChronoDT) (t : Int) : getDurationComponent K t "weeks" = .ok (some 0) := by
  simp [getDurationComponent, knownDurationComponent, eqIC]

/-- Outside chrono's range every other component is `null`. -/
theorem durComp_invalid_null (K : ChronoDT) (t : Int) (c : String) (hk : knownDurationComponent c = true)
    (hw : eqIC c "weeks" = false) (h : K.valid t = false) :
    getDurationComponent K t c = .ok none := by
  simp [getDurationComponent, hk, hw, h]

theorem chronoK_valid_iff (t : Int) : chronoK.valid t = chronoYearOk (yearOf t) := rfl

private theorem yearOk_bounds (y : Int) (h : chronoYearOk y = true) : -262143 ≤ y ∧ y ≤ 262142 := by
  unfold chronoYearOk at h; simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact h

/-- **Agreement inside chrono's range**: for a timestamp chrono accepts, each component equals the
one `decompose_duration` gives (the `(years, months, remaining_seconds)` triple split into days,
hours, minutes, seconds as `format_duration` does). -/
theorem durComp_agrees_decompose (t : Int) (hv : chronoYearOk (yearOf t) = true) :
    getDurationComponent chronoK t "years" = .ok (some (decomposeDuration t).1) ∧
    getDurationComponent chronoK t "months" = .ok (some (decomposeDuration t).2.1) ∧
    getDurationComponent chronoK t "days" = .ok (some ((decomposeDuration t).2.2 / 86400)) ∧
    getDurationComponent chronoK t "hours" = .ok (some ((decomposeDuration t).2.2 % 86400 / 3600)) ∧
    getDurationComponent chronoK t "minutes" = .ok (some ((decomposeDuration t).2.2 % 3600 / 60)) ∧
    getDurationComponent chronoK t "seconds" = .ok (some ((decomposeDuration t).2.2 % 60)) := by
  obtain ⟨y, m, d, hc⟩ : ∃ y m d, civilFromDaysI (t / 86400) = (y, m, d) := ⟨_, _, _, rfl⟩
  have hy : yearOf t = y := by simp [yearOf, dayOf, hc]
  have hm : monthOf t = m := by simp [monthOf, dayOf, hc]
  have hd : mdayOf t = d := by simp [mdayOf, dayOf, hc]
  have hv' : ValidDate y m d := by have := civil_valid (t / 86400); rw [hc] at this; exact this
  have hrt : daysFromCivil y m d = t / 86400 := by have := dfc_civil (t / 86400); rw [hc] at this; exact this
  rw [hy] at hv
  have hb := yearOk_bounds _ hv
  have hi : I32 y := ⟨by omega, by omega⟩
  have t0 := Int.emod_nonneg t (by decide : (86400:Int) ≠ 0)
  have t1 := Int.emod_lt_of_pos t (by decide : (0:Int) < 86400)
  have hdiv : 86400 * (t / 86400) + t % 86400 = t := Int.mul_ediv_add_emod _ _
  have hday := daysFromCivil_day y m d
  have i2 : I32 (y - 1970) := ⟨by omega, by omega⟩
  have hdec : decomposeDuration t = (y - 1970, m - 1, (d - 1) * 86400 + t % 86400) := by
    simp only [decomposeDuration, civilFromDays, hc, toI32_id _ hi, toI32_id _ i2]
  obtain ⟨a1, a2, a3, a4⟩ := hv'
  have hmv : ValidDate y m 1 := ⟨a1, a2, by omega, by omega⟩
  have e1 : 1970 + (y - 1970) = y := by omega
  have e2 : 1 + (m - 1) = m := by omega
  have hanch : yearOf (daysFromCivil y m 1 * 86400) = y := by
    unfold yearOf dayOf
    have : daysFromCivil y m 1 * 86400 / 86400 = daysFromCivil y m 1 := by omega
    rw [this, civil_dfc _ _ _ hmv]
  have hrem : t - daysFromCivil y m 1 * 86400 = (d - 1) * 86400 + t % 86400 := by omega
  have hval : chronoK.valid (daysFromCivil y m 1 * 86400) = true := by
    rw [chronoK_valid_iff, hanch]; exact hv
  have hvt : chronoK.valid t = true := by rw [chronoK_valid_iff, hy]; exact hv
  have hr0 : 0 ≤ (d - 1) * 86400 + t % 86400 := by omega
  rw [hdec]
  simp +decide [getDurationComponent, knownDurationComponent, eqIC, hvt, hy, hm, e1, e2, hval, hrem]

/-- Outside chrono's range the components are `null` for a duration that is nevertheless a valid
Cypher duration (`check_date_range` accepts it): the whole computation is skipped, so
`years` etc. disagree with `decompose_duration`. -/
theorem durComp_null_but_representable (t : Int) (hbad : chronoYearOk (yearOf t) = false)
    (hr : minDateDays ≤ t / 86400 ∧ t / 86400 ≤ maxDateDays) :
    (checkDateRange t).isSome = true ∧
    getDurationComponent chronoK t "years" = .ok none ∧
    getDurationComponent chronoK t "days" = .ok none := by
  have hw : ∀ c : String, knownDurationComponent c = true → eqIC c "weeks" = false →
      getDurationComponent chronoK t c = .ok none := fun c hk hw =>
    durComp_invalid_null chronoK t c hk hw (by rw [chronoK_valid_iff]; exact hbad)
  refine ⟨?_, hw "years" (by decide) (by decide), hw "days" (by decide) (by decide)⟩
  simp [checkDateRange, hr]

/-- **BUG (confirmed live)**: a duration of one million years prints as `P1000000Y3D` and adds
and subtracts, but `.years` / `.days` are `null` (C: `1000000` / `3`). The witness is the real
`construct_duration_secs(1_000_000, 0, 0, 3, 0, 0, 0)`. -/
theorem bug_duration_component_null :
    ∃ t, constructDuration 1000000 0 0 3 0 0 0 = some t ∧
      getDurationComponent chronoK t "years" = .ok none ∧
      getDurationComponent chronoK t "days" = .ok none ∧
      (decomposeDuration t).1 = 1000000 ∧ (decomposeDuration t).2.2 / 86400 = 3 := by
  refine ⟨daysFromCivil 1001970 1 1 * 86400 + 3 * 86400, ?_, ?_, ?_, ?_, ?_⟩
  · decide +kernel
  · rfl
  · rfl
  · decide +kernel
  · decide +kernel

end FalkorTemporal.Fns
