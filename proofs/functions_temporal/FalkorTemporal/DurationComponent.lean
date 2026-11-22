import FalkorTemporal.Components
import FalkorTemporal.Constructors
/-!
# `get_duration_component` (value.rs): read straight off the duration's components

Since #3610 a duration is its signed (months, days, seconds) (`Dur::components`), and
`get_duration_component` splits them the way `format_duration` prints them: years and months
from the months component, hours/minutes/seconds from the seconds component, each with Rust's
truncating `/` and `%` (`Int.tdiv`/`Int.tmod`), so every component carries the sign of its part.
No calendar or chrono step is involved, so every duration has all its components
(`durComp_total`); `durComp_years_months` / `durComp_hms` show they reassemble the stored
components exactly, and `duration_component_far` is the far-off duration (a million years)
whose components were `null` before #3610.
-/
namespace FalkorTemporal.Fns

def knownDurationComponent (c : String) : Bool :=
  eqIC c "years" || eqIC c "months" || eqIC c "weeks" || eqIC c "days" || eqIC c "hours" ||
    eqIC c "minutes" || eqIC c "seconds"

/-- `get_duration_component` (value.rs). The `Float` result is the integer it carries; a
component is at most `|i64|`, and `val as f64` is the cast of that integer. -/
def getDurationComponent (dur : Dur) (c : String) : Except String Int :=
  if eqIC c "years" then .ok (dur.months.tdiv 12)
  else if eqIC c "months" then .ok (dur.months.tmod 12)
  else if eqIC c "weeks" then .ok 0
  else if eqIC c "days" then .ok dur.days
  else if eqIC c "hours" then .ok (dur.secs.tdiv 3600)
  else if eqIC c "minutes" then .ok ((dur.secs.tmod 3600).tdiv 60)
  else if eqIC c "seconds" then .ok (dur.secs.tmod 60)
  else .error s!"unknown duration component {c}"

theorem durComp_unknown (dur : Dur) (c : String) (h : knownDurationComponent c = false) :
    getDurationComponent dur c = .error s!"unknown duration component {c}" := by
  simp only [knownDurationComponent, Bool.or_eq_false_iff] at h
  simp [getDurationComponent, h]

/-- `weeks` is always `0` (weeks are folded into days on construction). -/
theorem durComp_weeks (dur : Dur) : getDurationComponent dur "weeks" = .ok 0 := by
  simp [getDurationComponent, eqIC]

/-- **Every known component of every duration is a number** (never `null`, never an error). -/
theorem durComp_total (dur : Dur) (c : String) (h : knownDurationComponent c = true) :
    ∃ v, getDurationComponent dur c = .ok v := by
  unfold getDurationComponent
  simp only [knownDurationComponent, Bool.or_eq_true] at h
  repeat' split
  all_goals first | exact ⟨_, rfl⟩ | (exfalso; simp_all)

/-- `years * 12 + months` is the months component, sign included. -/
theorem durComp_years_months (dur : Dur) :
    ∃ y m, getDurationComponent dur "years" = .ok y ∧ getDurationComponent dur "months" = .ok m ∧
      y * 12 + m = dur.months := by
  refine ⟨dur.months.tdiv 12, dur.months.tmod 12, by simp [getDurationComponent, eqIC],
    by simp [getDurationComponent, eqIC], ?_⟩
  have := Int.mul_tdiv_add_tmod dur.months 12
  omega

/-- `hours * 3600 + minutes * 60 + seconds` is the seconds component, sign included. -/
theorem durComp_hms (dur : Dur) :
    ∃ h mi s, getDurationComponent dur "hours" = .ok h ∧
      getDurationComponent dur "minutes" = .ok mi ∧ getDurationComponent dur "seconds" = .ok s ∧
      h * 3600 + mi * 60 + s = dur.secs := by
  refine ⟨dur.secs.tdiv 3600, (dur.secs.tmod 3600).tdiv 60, dur.secs.tmod 60,
    by simp [getDurationComponent, eqIC], by simp [getDurationComponent, eqIC],
    by simp [getDurationComponent, eqIC], ?_⟩
  have h1 := Int.mul_tdiv_add_tmod dur.secs 3600
  have h2 := Int.mul_tdiv_add_tmod (dur.secs.tmod 3600) 60
  have h3 : (dur.secs.tmod 3600).tmod 60 = dur.secs.tmod 60 :=
    Int.tmod_tmod_of_dvd _ ⟨60, by decide⟩
  rw [← h3]; omega

/-- The duration of one million years and three days (`duration({years: 1000000, days: 3})`):
`.years` is `1000000` and `.days` is `3` (before #3610 both were `null`: the old reader went
through chrono's ±262,143-year range). -/
theorem duration_component_far :
    ∃ dur, constructDuration 1000000 0 0 3 0 0 0 = some dur ∧
      getDurationComponent dur "years" = .ok 1000000 ∧
      getDurationComponent dur "days" = .ok 3 := by
  refine ⟨⟨12000000, 3, 0⟩, by decide +kernel, by simp [getDurationComponent, eqIC],
    by simp [getDurationComponent, eqIC]⟩

end FalkorTemporal.Fns
