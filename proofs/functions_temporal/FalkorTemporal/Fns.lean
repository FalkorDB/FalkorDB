import FalkorTemporal.Calendar
/-!
# Temporal constructor functions (functions/temporal.rs:50-838)

Line-by-line models of the field extractors, the string parsers' control flow, the
`*_pure` / `*_struct_pure` constructors, the clock-reading `*_fn`s and `register`.

Assumptions are *structures*, never axioms:
* `Num F` — the abstract `f64` with Rust's `f as i64` (saturating; NaN ↦ 0).
* `Chrono` — the chrono constructors the code calls (`NaiveDate::from_ymd_opt`,
  `from_yo_opt`, `NaiveTime::from_hms_opt`), each with the law from the chrono docs
  that we rely on; `str::parse::<i32/u32/i64>` as abstract partial functions.
* the clock (`Utc::now()`, `rt.transaction_timestamp`) is an argument: theorems
  quantify over it.

Representation (as in the Rust): a `NaiveDate` is its day number since 1970-01-01,
a `NaiveTime` its second of the day, a timestamp `days * 86400 + secs`.
`date_from_components` (PROVEN elsewhere in this project, `Constructors.lean`) is a
parameter `dfc` here.
-/
namespace FalkorTemporal.Fns

/-- Abstract f64 operations used by the extractors. -/
structure Num (F : Type) where
  /-- `f as i64` -/
  toI64 : F → Int
  isNaN : F → Bool
  /-- Rust `as` saturates into the i64 range ... -/
  toI64_range : ∀ f, -2^63 ≤ toI64 f ∧ toI64 f < 2^63
  /-- ... and maps NaN to 0 (Rust reference, "Numeric cast" semantics). -/
  toI64_nan : ∀ f, isNaN f = true → toI64 f = 0

/-- The values these functions see (`Value` restricted to the variants they inspect). -/
inductive TV (F : Type) where
  | null | int (i : Int) | float (f : F) | str (s : String)
  | map (kvs : List (String × TV F)) | other (name : String)
  | date (t : Int) | time (t : Int) | datetime (t : Int) | duration (t : Int)

variable {F : Type} (N : Num F)

/-- `slot_to_int` (temporal.rs:139). -/
def slotToInt : TV F → Option Int
  | .int i => some i
  | .float f => some (N.toI64 f)
  | _ => none

/-- `OrderMap::get_str` — first binding of the key (keys are unique in an `OrderMap`). -/
def getStr (m : List (String × TV F)) (k : String) : Option (TV F) := m.lookup k

/-- `get_int_field` (temporal.rs:50). -/
def getIntField (m : List (String × TV F)) (field : String) : Option Int :=
  (getStr m field).bind fun v => match v with
    | .int i => some i
    | .float f => some (N.toI64 f)
    | _ => none

/-- `get_int_field` is `slot_to_int` of the looked-up value: the map form and the
positional-slot form read numbers identically. -/
theorem getIntField_eq_slot (m : List (String × TV F)) (k : String) :
    getIntField N m k = (getStr m k).bind (slotToInt N) := by
  unfold getIntField; cases getStr m k with
  | none => rfl
  | some v => cases v <;> rfl

theorem slotToInt_range (v : TV F) (i : Int) (h : slotToInt N v = some i)
    (hi : ∀ j, v = .int j → -2^63 ≤ j ∧ j < 2^63) : -2^63 ≤ i ∧ i < 2^63 := by
  cases v <;> simp [slotToInt] at h
  · subst h; exact hi _ rfl
  · subst h; exact N.toI64_range _

theorem slotToInt_nan (f : F) (h : N.isNaN f = true) : slotToInt N (.float f) = some 0 := by
  simp [slotToInt, N.toI64_nan f h]

/-! ## chrono -/

/-- `x as u32` for an i64 `x`: keep the low 32 bits. -/
def asU32 (x : Int) : Nat := (x % 2^32).toNat

structure Chrono where
  /-- `NaiveDate::from_ymd_opt(y, m, d)` -/
  fromYmd : Int → Nat → Nat → Option Int
  /-- `NaiveDate::from_yo_opt(y, ordinal)` -/
  fromYo : Int → Nat → Option Int
  /-- `NaiveDate::from_isoywd_opt` path, i.e. `parse_week_date` (PROVEN elsewhere) -/
  parseWeek : String → Except String Int
  /-- `str::parse::<i32>()` / `::<u32>()` / `::<i64>()` -/
  parseI32 : String → Option Int
  parseU32 : String → Option Nat
  parseI64 : String → Option Int
  /-- chrono docs: `from_ymd_opt` returns `None` on an invalid date, else that date. -/
  fromYmd_spec : ∀ y m d z, fromYmd y m d = some z → ValidDate y m d ∧ z = daysFromCivil y m d

/-- `NaiveTime::from_hms_opt(h, m, s)`: `Some` iff `h < 24 ∧ m < 60 ∧ s < 60`
(chrono docs; leap seconds only via the milli/nano constructors). -/
def fromHms (h m s : Nat) : Option Nat :=
  if h < 24 ∧ m < 60 ∧ s < 60 then some (h * 3600 + m * 60 + s) else none

theorem fromHms_lt (h m s t : Nat) (e : fromHms h m s = some t) : t < 86400 := by
  unfold fromHms at e; split at e
  · cases e; omega
  · cases e

/-- `time_from_components` (temporal.rs:117): `unwrap_or(0) as u32`, then `from_hms_opt`. -/
def timeFromComponents (hour minute second : Option Int) : Except String Nat :=
  let h := asU32 (hour.getD 0); let m := asU32 (minute.getD 0); let s := asU32 (second.getD 0)
  match fromHms h m s with
  | some t => .ok t
  | none => .error s!"Invalid time: hour={h}, minute={m}, second={s}"

theorem timeFromComponents_ok (h m s : Option Int) (t : Nat)
    (e : timeFromComponents h m s = .ok t) :
    t < 86400 ∧ t = asU32 (h.getD 0) * 3600 + asU32 (m.getD 0) * 60 + asU32 (s.getD 0) := by
  unfold timeFromComponents at e; simp only at e
  split at e
  · rename_i t' ht; cases e; refine ⟨fromHms_lt _ _ _ _ ht, ?_⟩
    unfold fromHms at ht; split at ht <;> simp_all
  · cases e

/-- The `as u32` truncation: hour `2^32 + 1` is read as hour 1 (C truncates the same way). -/
example : timeFromComponents (some (2^32 + 1)) none none = .ok 3600 := by rfl

/-- `time_from_map` (temporal.rs:129). -/
def timeFromMap (m : List (String × TV F)) : Except String Nat :=
  timeFromComponents (getIntField N m "hour") (getIntField N m "minute") (getIntField N m "second")

theorem timeFromMap_eq_slots (m : List (String × TV F)) :
    timeFromMap N m = timeFromComponents ((getStr m "hour").bind (slotToInt N))
      ((getStr m "minute").bind (slotToInt N)) ((getStr m "second").bind (slotToInt N)) := by
  simp [timeFromMap, getIntField_eq_slot]

/-- `date_from_components` signature (PROVEN in Constructors.lean); here a parameter. -/
abbrev DFC := Option Int → Option Int → Option Int → Option Int → Option Int → Option Int →
  Option Int → Except String Int

/-- `date_from_map` (temporal.rs:105). -/
def dateFromMap (dfc : DFC) (m : List (String × TV F)) : Except String Int :=
  dfc (getIntField N m "year") (getIntField N m "month") (getIntField N m "day")
    (getIntField N m "week") (getIntField N m "dayOfWeek") (getIntField N m "quarter")
    (getIntField N m "dayOfQuarter")

/-- The map form and the 7-slot form (`date_struct_pure`) build the same date when the
slots hold the map's values. -/
theorem dateFromMap_eq_slots (dfc : DFC) (m : List (String × TV F)) :
    dateFromMap N dfc m = dfc ((getStr m "year").bind (slotToInt N))
      ((getStr m "month").bind (slotToInt N)) ((getStr m "day").bind (slotToInt N))
      ((getStr m "week").bind (slotToInt N)) ((getStr m "dayOfWeek").bind (slotToInt N))
      ((getStr m "quarter").bind (slotToInt N)) ((getStr m "dayOfQuarter").bind (slotToInt N)) := by
  simp [dateFromMap, getIntField_eq_slot]

end FalkorTemporal.Fns
