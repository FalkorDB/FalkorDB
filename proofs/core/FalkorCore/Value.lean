/-!
# The engine `Value` (`graph/src/runtime/value.rs:184`)

`V F` is `enum Value` with the `f64` payload abstracted to a type `F` (instantiate it with
`FalkorCore.F`, `Float`, or a record of operations); `Arc`s are transparent. `Value::Duration` carries a `Dur` (#3610). `V.order` is
`OrderedEnum::order` (value.rs:1219), `V.name` is `Value::name` (value.rs:1395).
From proofs/value_math (pilot of the shared core).
-/
namespace FalkorCore

/-- `days_from_civil` (`temporal.rs:470`) with floor `/`, `%` (equal to Rust's truncating
form: proofs/functions_temporal `daysFromCivilT_eq`). -/
def daysFromCivil (y m d : Int) : Int :=
  let y := if m ≤ 2 then y - 1 else y
  let era := y / 400
  let yoe := y % 400
  let doy := (153 * (if m > 2 then m - 3 else m + 9) + 2) / 5 + d - 1
  let doe := yoe * 365 + yoe / 4 - yoe / 100 + doy
  era * 146097 + doe - 719468

/-- The calendar offset of `months` from the epoch, as `Dur::new` computes it. -/
def durAnchor (m : Int) : Int := daysFromCivil (1970 + m / 12) (m % 12 + 1) 1 * 86400

/-- The checks of `Dur::new` (`temporal.rs:515`): months fit an `i32`, and the legacy
offset (anchor + days·86400 + seconds, one `i128` sum) fits an `i64`. Days and seconds are
`i64` fields, so they fit too. Same as proofs/functions_temporal `durNew`. -/
def durOk (m d s : Int) : Bool :=
  decide (-2 ^ 31 ≤ m) && decide (m < 2 ^ 31) && decide (-2 ^ 63 ≤ d) && decide (d < 2 ^ 63) &&
  decide (-2 ^ 63 ≤ s) && decide (s < 2 ^ 63) &&
  decide (-2 ^ 63 ≤ durAnchor m + d * 86400 + s) && decide (durAnchor m + d * 86400 + s < 2 ^ 63)

/-- `Dur` (`temporal.rs:506`, #3610): a duration's signed (months, days, seconds), as
`Dur::components` returns them (the inline/heap word is proofs/functions_temporal's
`DurWord`; `components_new` proves it transparent). Its field is private and every `Dur`
comes from `Dur::new`, so the checks are part of the type (`ok`). Before #3610 a
duration was one `i64` epoch offset. -/
structure Dur where
  months : Int
  days : Int
  secs : Int
  ok : durOk months days secs = true

/-- `Dur::new`, `Err` ↦ `none`. -/
def Dur.new? (m d s : Int) : Option Dur :=
  if h : durOk m d s = true then some ⟨m, d, s, h⟩ else none

theorem Dur.ext' {a b : Dur} (h1 : a.months = b.months) (h2 : a.days = b.days)
    (h3 : a.secs = b.secs) : a = b := by
  cases a; cases b; simp only at h1 h2 h3; subst h1 h2 h3; rfl

instance : DecidableEq Dur := fun a b =>
  if h : a.months = b.months ∧ a.days = b.days ∧ a.secs = b.secs then
    isTrue (Dur.ext' h.1 h.2.1 h.2.2)
  else isFalse (fun e => h (e ▸ ⟨rfl, rfl, rfl⟩))

instance : Inhabited Dur := ⟨⟨0, 0, 0, by decide⟩⟩

/-- `components ∘ new`: every `Dur` is rebuilt from its own components. -/
theorem Dur.new?_self (c : Dur) : Dur.new? c.months c.days c.secs = some c := by
  unfold Dur.new?; rw [dif_pos c.ok]

theorem Dur.new?_some {m d s : Int} {c : Dur} (h : Dur.new? m d s = some c) :
    c.months = m ∧ c.days = d ∧ c.secs = s := by
  unfold Dur.new? at h; split at h
  · cases h; exact ⟨rfl, rfl, rfl⟩
  · cases h

/-- `duration_cmp` (`value.rs:685`, #3610): approximate length in seconds (a month is
2 629 746 s), ties broken on the (months, days, seconds) triple. -/
def durationCmp (a b : Dur) : Ordering :=
  (compare (a.months * 2629746 + a.days * 86400 + a.secs) (b.months * 2629746 + b.days * 86400 + b.secs)).then
    ((compare a.months b.months).then ((compare a.days b.days).then (compare a.secs b.secs)))

theorem durationCmp_refl (a : Dur) : durationCmp a a = .eq := by
  simp [durationCmp, Ordering.then]

inductive V (F : Type) where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | float (f : F)
  | str (s : String)
  | list (l : List (V F))
  | map (m : List (String × V F))
  | node (id : Nat)
  | rel (id : Nat)
  | path (l : List (V F))
  | vec (v : List F)
  | point (lat lon : F)
  | datetime (t : Int)
  | date (t : Int)
  | time (t : Int)
  | duration (d : Dur)
  deriving Inhabited

variable {F : Type}

/-- `Value::name` (value.rs:1395; C's `SIType_ToString` spellings since #2954). -/
def V.name : V F → String
  | .null => "Null" | .bool _ => "Boolean" | .int _ => "Integer" | .float _ => "Float"
  | .str _ => "String" | .list _ => "List" | .map _ => "Map" | .node _ => "Node"
  | .rel _ => "Edge" | .path _ => "Path" | .vec _ => "Vectorf32" | .point .. => "Point"
  | .datetime _ => "Datetime" | .date _ => "Date" | .time _ => "Time"
  | .duration _ => "Duration"

/-- `OrderedEnum::order` (value.rs:1219). -/
def V.order : V F → Nat
  | .null => 2 ^ 15 | .bool _ => 2 ^ 12 | .int _ => 2 ^ 13 | .float _ => 2 ^ 14
  | .str _ => 2 ^ 11 | .list _ => 2 ^ 3 | .map _ => 2 ^ 0 | .node _ => 2 ^ 1
  | .rel _ => 2 ^ 2 | .path _ => 2 ^ 4 | .point .. => 2 ^ 5 | .datetime _ => 2 ^ 6
  | .date _ => 2 ^ 7 | .time _ => 2 ^ 8 | .duration _ => 2 ^ 10 | .vec _ => 2 ^ 18

def V.isNull : V F → Bool
  | .null => true
  | _ => false

def V.isNum : V F → Bool
  | .int _ | .float _ => true
  | _ => false

def V.isList : V F → Bool
  | .list _ => true
  | _ => false

end FalkorCore
