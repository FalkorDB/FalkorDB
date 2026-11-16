import FalkorTemporal.Calendar
/-!
# Temporal constructor functions (functions/temporal.rs:50-1020)

Line-by-line models of the component validators (#3607: `map_components`,
`struct_components`, `int_component(s)`, `struct_ints`), the string parsers' control flow,
the `*_pure` / `*_struct_pure` constructors, the clock-reading `*_fn`s and `register`.

Assumptions are *structures*, never axioms:
* `Num F` — the abstract `f64`, with Rust's `n as f64`.
* `Chrono` — the chrono constructors the code calls (`NaiveDate::from_ymd_opt`,
  `from_yo_opt`, `NaiveTime::from_hms_opt`), each with the law from the chrono docs
  that we rely on; `str::parse::<i32/u32/i64>` as abstract partial functions.
* the clock (`Utc::now()`, `rt.transaction_timestamp`) is an argument: theorems
  quantify over it.

Representation (as in the Rust): a `NaiveDate` is its day number since 1970-01-01,
a `NaiveTime` its second of the day, a timestamp `days * 86400 + secs`. A fixed-size
array `[Option<T>; MAX_SLOTS]` is its index function `Nat → Option T` (unset = `None`).
`date_from_components` (PROVEN elsewhere in this project, `Constructors.lean`) is a
parameter `dfc` here.
-/
namespace FalkorTemporal.Fns

/-- Abstract f64 operations used by the constructors. -/
structure Num (F : Type) where
  /-- `n as f64` for an i64 `n` -/
  ofInt : Int → F

/-- The values these functions see (`Value` restricted to the variants they inspect). -/
inductive TV (F : Type) where
  | null | int (i : Int) | float (f : F) | str (s : String)
  | map (kvs : List (String × TV F)) | other (name : String)
  | date (t : Int) | time (t : Int) | datetime (t : Int) | duration (t : Int)

variable {F : Type} (N : Num F)

/-- `OrderMap::get_str` — first binding of the key (keys are unique in an `OrderMap`). -/
def getStr (m : List (String × TV F)) (k : String) : Option (TV F) := m.lookup k

/-! ## Component validation (temporal.rs:50-157, #3607) -/

/-- `str::eq_ignore_ascii_case`. -/
def ieq (a b : String) : Bool := a.toList.map Char.toLower == b.toList.map Char.toLower

theorem ieq_refl (a : String) : ieq a a = true := by simp [ieq]

/-- `keys.iter().position(p)`. -/
def position (p : String → Bool) : List String → Option Nat
  | [] => none
  | s :: ss => if p s then some 0 else (position p ss).map (· + 1)

/-- `out[i] = Some(v)` on an array. -/
def setAt {α : Type} (out : Nat → Option α) (i : Nat) (v : α) : Nat → Option α :=
  fun j => if j = i then some v else out j

/-- `UNKNOWN_KEY` (temporal.rs:54). -/
def unknownKey : String := "datetime components map contains an unknown key"

/-- `SUBSECOND_KEYS` (temporal.rs:58). -/
def subsecondKeys : List String := ["millisecond", "microsecond", "nanosecond"]

/-- `int_component` (temporal.rs:133): only an `Int` is a component. -/
def intComponent (v : TV F) (key : String) : Except String Int :=
  match v with
  | .int i => .ok i
  | _ => .error s!"{key} must be an integer value"

/-- The `for (k, v) in map.iter()` loop of `map_components` (temporal.rs:73-95):
`out` the component array, `unk` the `unknown` flag, `seen` `extra_seen`. -/
def mcLoop (keys extra : List String) :
    List (String × TV F) → (Nat → Option (TV F)) → Bool → (Nat → Bool) →
      Except String ((Nat → Option (TV F)) × Bool)
  | [], out, unk, _ => .ok (out, unk)
  | (k, v) :: kvs, out, unk, seen =>
    match position (fun s => ieq s k) keys with
    | some i =>
      match out i with
      | some _ => mcLoop keys extra kvs out true seen
      | none => mcLoop keys extra kvs (setAt out i v) unk seen
    | none =>
      match position (fun s => ieq s k) extra with
      | some i =>
        if seen i then mcLoop keys extra kvs out true seen
        else match intComponent v (extra.getD i "") with
          | .error e => .error e
          | .ok _ => mcLoop keys extra kvs out unk (fun j => j == i || seen j)
      | none => mcLoop keys extra kvs out true seen

/-- The bad-value scan before `UNKNOWN_KEY` (temporal.rs:97-110). -/
def badValues (numeric : Bool) : List String → Nat → (Nat → Option (TV F)) → Except String Unit
  | [], _, _ => .ok ()
  | key :: ks, i, out =>
    match out i with
    | none => badValues numeric ks (i + 1) out
    | some (.int _) => badValues numeric ks (i + 1) out
    | some v =>
      if numeric then
        match v with
        | .float _ => badValues numeric ks (i + 1) out
        | _ => .error s!"{key} must be a numerical value"
      else
        match intComponent v key with
        | .error e => .error e
        | .ok _ => badValues numeric ks (i + 1) out

/-- `map_components` (temporal.rs:67). -/
def mapComponents (m : List (String × TV F)) (keys extra : List String) (numeric : Bool) :
    Except String (Nat → Option (TV F)) :=
  match mcLoop keys extra m (fun _ => none) false (fun _ => false) with
  | .error e => .error e
  | .ok (out, unk) =>
    if unk then
      match badValues numeric keys 0 out with
      | .error e => .error e
      | .ok _ => .error unknownKey
    else .ok out

/-- `mask >> i & 1 == 1` for an i64 `mask` (arithmetic shift = repeated floor halving;
`& 1` = the non-negative remainder mod 2). -/
def bitAt (mask : Int) : Nat → Bool
  | 0 => mask % 2 == 1
  | i + 1 => bitAt (mask / 2) i

/-- `args.split_last()` matched against `Some((Value::Int(mask), slots))`, else `(0, [])`. -/
def splitMask (args : List (TV F)) : Int × List (TV F) :=
  match args.reverse with
  | .int mask :: rs => (mask, rs.reverse)
  | _ => (0, [])

/-- `struct_components` (temporal.rs:117): slot `i` is written iff mask bit `i` is set. -/
def structComponents (args : List (TV F)) : Nat → Option (TV F) :=
  fun i => if bitAt (splitMask args).1 i then (splitMask args).2[i]? else none

/-- The loop of `int_components` (temporal.rs:144-156), over `vals.iter().zip(keys)`. -/
def icLoop (vals : Nat → Option (TV F)) : List String → Nat → (Nat → Option Int) →
    Except String (Nat → Option Int)
  | [], _, out => .ok out
  | key :: ks, i, out =>
    match vals i with
    | none => icLoop vals ks (i + 1) out
    | some v =>
      match intComponent v key with
      | .error e => .error e
      | .ok n => icLoop vals ks (i + 1) (setAt out i n)

/-- `int_components` (temporal.rs:144). -/
def intComponents (vals : Nat → Option (TV F)) (keys : List String) :
    Except String (Nat → Option Int) :=
  icLoop vals keys 0 fun _ => none

/-- `args.last()` matched against `Some(Value::Int(mask))`, else 0. -/
def lastMask (args : List (TV F)) : Int :=
  match args.getLast? with
  | some (.int m) => m
  | _ => 0

/-- The loop of `struct_ints` (temporal.rs:241-247); `&args[i]` is `args.getD i Null`
(the binder always passes one child per slot plus the mask). -/
def siLoop (args : List (TV F)) (mask : Int) : List String → Nat → (Nat → Option Int) →
    Except String (Nat → Option Int)
  | [], _, out => .ok out
  | key :: ks, i, out =>
    if bitAt mask i then
      match intComponent (args.getD i .null) key with
      | .error e => .error e
      | .ok n => siLoop args mask ks (i + 1) (setAt out i n)
    else siLoop args mask ks (i + 1) out

/-- `struct_ints` (temporal.rs:233). -/
def structInts (args : List (TV F)) (keys : List String) : Except String (Nat → Option Int) :=
  siLoop args (lastMask args) keys 0 fun _ => none

/-- `DATE_SLOTS`, `LOCALTIME_SLOTS`, `LOCALDATETIME_SLOTS` (temporal.rs:754-779). -/
def dateSlots : List String := ["year", "month", "day", "week", "dayOfWeek", "quarter", "dayOfQuarter"]
def localtimeSlots : List String := ["hour", "minute", "second"]
def localdatetimeSlots : List String := dateSlots ++ localtimeSlots

/-! ## chrono -/

/-- `x as u32` for an i64 `x`: keep the low 32 bits (the pre-#3607 conversion). -/
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

/-- `u32::try_from(x).ok()`. -/
def u32Of (x : Int) : Option Nat := if 0 ≤ x ∧ x < 2^32 then some x.toNat else none

/-- `time_from_components` (temporal.rs:250): `unwrap_or(0)`, `u32::try_from` (since #3607;
it was `as u32`), then `from_hms_opt`. -/
def timeFromComponents (hour minute second : Option Int) : Except String Nat :=
  let h := hour.getD 0; let m := minute.getD 0; let s := second.getD 0
  match (u32Of h).bind fun h' => (u32Of m).bind fun m' => (u32Of s).bind fun s' => fromHms h' m' s' with
  | some t => .ok t
  | none => .error s!"Invalid time: hour={h}, minute={m}, second={s}"

theorem u32Of_spec (x : Int) (n : Nat) (h : u32Of x = some n) : (n : Int) = x ∧ asU32 x = n := by
  unfold u32Of at h; split at h
  · rename_i hx; cases h
    refine ⟨Int.toNat_of_nonneg hx.1, ?_⟩
    simp only [asU32]; rw [Int.emod_eq_of_lt hx.1 hx.2]
  · cases h

/-- The statement of the pre-#3607 theorem holds unchanged: a time is a second of the day,
read from the components (which are now in range, so `as u32` would not have changed them). -/
theorem timeFromComponents_ok (h m s : Option Int) (t : Nat)
    (e : timeFromComponents h m s = .ok t) :
    t < 86400 ∧ t = asU32 (h.getD 0) * 3600 + asU32 (m.getD 0) * 60 + asU32 (s.getD 0) := by
  unfold timeFromComponents at e; simp only at e
  split at e
  · rename_i t' ht; cases e
    cases h1 : u32Of (h.getD 0) with
    | none => simp [h1] at ht
    | some a =>
      cases h2 : u32Of (m.getD 0) with
      | none => simp [h1, h2] at ht
      | some b =>
        cases h3 : u32Of (s.getD 0) with
        | none => simp [h1, h2, h3] at ht
        | some c =>
          simp only [h1, h2, h3, Option.bind_some] at ht
          refine ⟨fromHms_lt _ _ _ _ ht, ?_⟩
          rw [(u32Of_spec _ _ h1).2, (u32Of_spec _ _ h2).2, (u32Of_spec _ _ h3).2]
          unfold fromHms at ht; split at ht <;> simp_all
  · cases e

/-- #3607: an hour of `2^32 + 1` is an error, no longer wrapped to hour 1. -/
theorem timeFromComponents_no_wrap :
    ∃ e, timeFromComponents (some (2^32 + 1)) none none = .error e := ⟨_, rfl⟩

/-- Historical (fixed by #3607): `time_from_components` before #3607, with `as u32`. -/
def pre3607_timeFromComponents (hour minute second : Option Int) : Except String Nat :=
  let h := asU32 (hour.getD 0); let m := asU32 (minute.getD 0); let s := asU32 (second.getD 0)
  match fromHms h m s with
  | some t => .ok t
  | none => .error s!"Invalid time: hour={h}, minute={m}, second={s}"

/-- Historical: the `as u32` truncation read hour `2^32 + 1` as hour 1. -/
theorem pre3607_u32_wrap : pre3607_timeFromComponents (some (2^32 + 1)) none none = .ok 3600 := by rfl

/-- `time_from_values` (temporal.rs:267): `vals[..3]` validated, then the time. -/
def timeFromValues (vals : Nat → Option (TV F)) : Except String Nat :=
  match intComponents vals localtimeSlots with
  | .error e => .error e
  | .ok c => timeFromComponents (c 0) (c 1) (c 2)

/-- `date_from_components` signature (PROVEN in Constructors.lean); here a parameter. -/
abbrev DFC := Option Int → Option Int → Option Int → Option Int → Option Int → Option Int →
  Option Int → Except String Int

/-- `date_from_values` (temporal.rs:226): `vals[..7]` validated, then the date. -/
def dateFromValues (dfc : DFC) (vals : Nat → Option (TV F)) : Except String Int :=
  match intComponents vals dateSlots with
  | .error e => .error e
  | .ok c => dfc (c 0) (c 1) (c 2) (c 3) (c 4) (c 5) (c 6)

end FalkorTemporal.Fns
