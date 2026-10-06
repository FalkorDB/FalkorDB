/-
# Proleptic-Gregorian calendar kernels (Howard Hinnant's algorithms)

Faithful `Int` models of the calendar helpers every temporal value in
FalkorDB-rs passes through:

| here | there |
| --- | --- |
| `isLeap`          | `is_leap` — `graph/src/runtime/value.rs:764` |
| `daysInMonth`     | `days_in_month` — `value.rs:752` |
| `daysFromCivil`   | `days_from_civil` — `value.rs:771` (`div_euclid`/`rem_euclid` = Lean `/`,`%` for a positive divisor) |
| `daysFromCivilT`  | `days_from_civil` — `graph/src/runtime/functions/temporal.rs:413` (a second copy, i32 arithmetic, truncating `/`) |
| `civilFromDays`   | `civil_from_days` — `value.rs:788` (the final `y as i32` is `toI32`) |

Era-local pieces (`yoeOf`, `doyOf`, `mpOf`, `dOf`, `dfcEra`) are the
sub-expressions of those functions, named so that the finite era check in
`EraCheck*.lean` can talk about them.
-/

namespace FalkorTemporal

/-- `is_leap` (`value.rs:764`). Rust `%` on i32 is truncating; divisibility tests
do not care about the sign convention, which `isLeap_tmod` records. -/
def isLeap (y : Int) : Bool := (y % 4 == 0 && y % 100 != 0) || y % 400 == 0

/-- `days_in_month` (`value.rs:752`). Only called with `1 ≤ m ≤ 12`. -/
def daysInMonth (y m : Int) : Int :=
  if m = 2 then (if isLeap y then 29 else 28)
  else if m = 4 ∨ m = 6 ∨ m = 9 ∨ m = 11 then 30 else 31

/-- A calendar date the way chrono's `NaiveDate::from_ymd_opt` accepts it (ignoring
chrono's ±262143-year limit, see `chronoYearOk`). -/
def ValidDate (y m d : Int) : Prop := 1 ≤ m ∧ m ≤ 12 ∧ 1 ≤ d ∧ d ≤ daysInMonth y m

/-- `days_from_civil` (`value.rs:771`). -/
def daysFromCivil (y m d : Int) : Int :=
  let y := if m ≤ 2 then y - 1 else y
  let era := y / 400
  let yoe := y % 400
  let doy := (153 * (if m > 2 then m - 3 else m + 9) + 2) / 5 + d - 1
  let doe := yoe * 365 + yoe / 4 - yoe / 100 + doy
  era * 146097 + doe - 719468

/-- `days_from_civil` in `temporal.rs:413`: `era = (if y >= 0 { y } else { y - 399 }) / 400`
with Rust's *truncating* i32 division, `yoe = (y - era * 400) as u32`. -/
def daysFromCivilT (y m d : Int) : Int :=
  let y := if m ≤ 2 then y - 1 else y
  let era := (if y ≥ 0 then y else y - 399).tdiv 400
  let yoe := y - era * 400
  let mp := if m > 2 then m - 3 else m + 9
  let doy := (153 * mp + 2) / 5 + d - 1
  let doe := yoe * 365 + yoe / 4 - yoe / 100 + doy
  era * 146097 + doe - 719468

/-- Era-local pieces of `civil_from_days` (`value.rs:791-797`). -/
def yoeOf (doe : Int) : Int := (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
def doyOf (doe : Int) : Int := doe - (365 * yoeOf doe + yoeOf doe / 4 - yoeOf doe / 100)
def mpOf (doe : Int) : Int := (5 * doyOf doe + 2) / 153
def dOf (doe : Int) : Int := doyOf doe - (153 * mpOf doe + 2) / 5 + 1

/-- The inverse direction inside one era: March-based year `yoe`, month `mp`. -/
def dfcEra (yoe mp d : Int) : Int := yoe * 365 + yoe / 4 - yoe / 100 + ((153 * mp + 2) / 5 + d - 1)

/-- Month length in the March-based era year (`mp = 0` is March, `mp = 11` is
February of civil year `yoe + 1`). -/
def dimMarch (yoe mp : Int) : Int :=
  if mp = 11 then (if isLeap (yoe + 1) then 29 else 28)
  else if mp = 1 ∨ mp = 3 ∨ mp = 6 ∨ mp = 8 then 30 else 31

/-- `civil_from_days` (`value.rs:788`) before the final `as i32` truncation. -/
def civilFromDaysI (z : Int) : Int × Int × Int :=
  let z := z + 719468
  let era := z / 146097
  let doe := z % 146097
  let y := yoeOf doe + era * 400
  let mp := mpOf doe
  let d := dOf doe
  let m := if mp < 10 then mp + 3 else mp - 9
  let y := if m ≤ 2 then y + 1 else y
  (y, m, d)

/-- Rust `x as i32` on an i64: two's-complement wrap. -/
def toI32 (x : Int) : Int := (x + 2 ^ 31) % 2 ^ 32 - 2 ^ 31

def I32 (x : Int) : Prop := -2 ^ 31 ≤ x ∧ x < 2 ^ 31
def I64 (x : Int) : Prop := -2 ^ 63 ≤ x ∧ x < 2 ^ 63

/-- `civil_from_days` exactly as Rust returns it: `(y as i32, m as u32, d as u32)`. -/
def civilFromDays (z : Int) : Int × Int × Int :=
  let r := civilFromDaysI z
  (toI32 r.1, r.2.1, r.2.2)

theorem toI32_id (x : Int) (h : I32 x) : toI32 x = x := by
  unfold toI32 I32 at *
  have : (x + 2 ^ 31) % 2 ^ 32 = x + 2 ^ 31 := Int.emod_eq_of_lt (by omega) (by omega)
  omega

/-! ## Algebraic lemmas (no enumeration) -/

theorem yoe_bounds (doe : Int) (h0 : 0 ≤ doe) (h1 : doe < 146097) :
    0 ≤ yoeOf doe ∧ yoeOf doe ≤ 399 := by unfold yoeOf; omega

/-- `dfcEra` undoes the era-local decomposition — pure algebra. -/
theorem dfcEra_of (doe : Int) : dfcEra (yoeOf doe) (mpOf doe) (dOf doe) = doe := by
  unfold dfcEra dOf doyOf
  generalize (153 * mpOf doe + 2) / 5 = a
  generalize yoeOf doe / 4 = b
  generalize yoeOf doe / 100 = c
  omega

/-- The two copies of `days_from_civil` agree on every month `1..12`
(the truncating-division era trick in `temporal.rs` is floor division). -/
theorem daysFromCivilT_eq (y m d : Int) :
    daysFromCivilT y m d = daysFromCivil y m d := by
  unfold daysFromCivilT daysFromCivil
  dsimp only
  generalize (if m ≤ 2 then y - 1 else y) = y'
  have hera : (if y' ≥ 0 then y' else y' - 399).tdiv 400 = y' / 400 := by
    by_cases h : y' ≥ 0
    · simp only [h, ite_true]
      rw [Int.tdiv_eq_ediv_of_nonneg h]
    · simp only [h, ite_false]
      have e : y' - 399 = -(399 - y') := by omega
      rw [e, Int.neg_tdiv, Int.tdiv_eq_ediv_of_nonneg (by omega)]
      omega
  rw [hera]
  have : y' - y' / 400 * 400 = y' % 400 := by omega
  rw [this]

/-- `days_from_civil` is affine in the day: day `d` is `d - 1` days after day 1. -/
theorem daysFromCivil_day (y m d : Int) :
    daysFromCivil y m d = daysFromCivil y m 1 + (d - 1) := by
  unfold daysFromCivil
  dsimp only
  generalize (153 * (if m > 2 then m - 3 else m + 9) + 2) / 5 = a
  omega

/-- Leap years are 400-periodic. -/
theorem isLeap_add400 (y k : Int) : isLeap (y + 400 * k) = isLeap y := by
  unfold isLeap
  have h4 : (y + 400 * k) % 4 = y % 4 := by omega
  have h100 : (y + 400 * k) % 100 = y % 100 := by omega
  have h400 : (y + 400 * k) % 400 = y % 400 := by omega
  rw [h4, h100, h400]

/-- Rust's `is_leap` uses truncating `%`; divisibility is sign-insensitive. -/
theorem isLeap_tmod (y : Int) :
    ((y.tmod 4 == 0 && y.tmod 100 != 0) || y.tmod 400 == 0) = isLeap y := by
  unfold isLeap
  have e : ∀ n : Int, n ≠ 0 → (y.tmod n = 0 ↔ y % n = 0) := by
    intro n hn
    rw [← Int.dvd_iff_tmod_eq_zero, Int.dvd_iff_emod_eq_zero]
  have e4 := e 4 (by decide)
  have e100 := e 100 (by decide)
  have e400 := e 400 (by decide)
  have b4 : (y.tmod 4 == 0) = (y % 4 == 0) := by
    by_cases h : y % 4 = 0
    · have t := e4.mpr h; simp [h, t]
    · have t : ¬ y.tmod 4 = 0 := fun t => h (e4.mp t)
      rw [beq_eq_false_iff_ne.mpr h, beq_eq_false_iff_ne.mpr t]
  have b100 : (y.tmod 100 == 0) = (y % 100 == 0) := by
    by_cases h : y % 100 = 0
    · have t := e100.mpr h; simp [h, t]
    · have t : ¬ y.tmod 100 = 0 := fun t => h (e100.mp t)
      rw [beq_eq_false_iff_ne.mpr h, beq_eq_false_iff_ne.mpr t]
  have b100' : (y.tmod 100 != 0) = (y % 100 != 0) := by
    simp only [bne, b100]
  have b400 : (y.tmod 400 == 0) = (y % 400 == 0) := by
    by_cases h : y % 400 = 0
    · have t := e400.mpr h; simp [h, t]
    · have t : ¬ y.tmod 400 = 0 := fun t => h (e400.mp t)
      rw [beq_eq_false_iff_ne.mpr h, beq_eq_false_iff_ne.mpr t]
  rw [b4, b100', b400]

end FalkorTemporal
