import FalkorValueMath.Conversion
/-!
# `functions/conversion.rs`: `register` (:40), `value_to_float` (:96), `value_to_string`
(:118), `to_json` (:154) — origin/main 3fec7d7c9

The bodies are `Conversion.lean`'s `toFloat`/`toStr` (f64 parsing and Display are
`ConvOps` fields, `to_json_string` is a parameter `js`). Proved here: the wrappers'
`Option`/panic structure (the `None => unreachable!()` arms are dead because arity is
exactly 1), result-type soundness against the registered `ret`, the `...OrNull` aliases
(same body, `Type::Any` argument ⇒ unsupported types give `Null` instead of a type error),
and that `register` adds distinct (lower-cased) names, so its `funcs.add` asserts hold.
Live (18920/18921): `toFloat('1.5')`, `toFloat(3)`, `toFloatOrNull(true)`, `toString(true)`
agree with C; known divergences (`toFloat(' 1')`, `toString(1.5)`, `toJSON` spacing) are
listed in `FalkorValueMath.lean`.
-/

namespace ValueMath.ConvReg

open ValueMath

variable {F : Type} (c : ConvOps F)

/-- `value_to_float` on its argument slice (`args.first()`; `None` → `Null`). -/
def toFloatArgs : List (V F) → V F
  | v :: _ => toFloat c v
  | [] => .null

/-- `value_to_string`: `None => unreachable!()` (`none` = panic). -/
def toStrArgs : List (V F) → Option (V F)
  | v :: _ => some (toStr c v)
  | [] => none

/-- `to_json` (:154): `map_or_else(|| unreachable!(), |v| String(v.to_json_string(rt)))`. -/
def toJsonArgs (js : V F → String) : List (V F) → Option (V F)
  | v :: _ => some (.str (js v))
  | [] => none

inductive Body where
  | toInteger | toFloat | toStr | toJson | isEmpty | toBoolean
  | toBooleanList | toFloatList | toIntegerList | toStringList
  deriving DecidableEq

def strL : Ty := .union [.list .any, .null]

/-- `register` (:40): `(name, args, ret, body)` for each `cypher_fn!` / `funcs.add`. -/
def convSigs : List (String × List Ty × Ty × Body) :=
  [("tointeger", [.union [.string, .bool, .int, .float, .null]], .union [.int, .null], .toInteger),
   ("toIntegerOrNull", [.any], .union [.int, .null], .toInteger),
   ("tofloat", [.union [.string, .float, .int, .null]], .union [.float, .null], .toFloat),
   ("toFloatOrNull", [.any], .union [.float, .null], .toFloat),
   ("tostring", [.union [.datetime, .date, .time, .duration, .string, .bool, .int, .float, .null, .point]],
     .union [.string, .null], .toStr),
   ("tostringornull", [.any], .union [.string, .null], .toStr),
   ("tojson", [.any], .union [.string, .null], .toJson),
   ("isEmpty", [.union [.map, .list .any, .string, .null]], .union [.bool, .null], .isEmpty),
   ("toBoolean", [.union [.string, .bool, .int, .null]], .union [.bool, .null], .toBoolean),
   ("toBooleanOrNull", [.any], .union [.bool, .null], .toBoolean),
   ("toBooleanList", [strL], strL, .toBooleanList), ("toFloatList", [strL], strL, .toFloatList),
   ("toIntegerList", [strL], strL, .toIntegerList), ("toStringList", [strL], strL, .toStringList)]

/-- Lower-cased names are distinct, so no `funcs.add` assertion fires. -/
theorem convSigs_lower_nodup :
    ["tointeger", "tointegerornull", "tofloat", "tofloatornull", "tostring", "tostringornull",
     "tojson", "isempty", "toboolean", "tobooleanornull", "tobooleanlist", "tofloatlist",
     "tointegerlist", "tostringlist"].Nodup := by decide

def bodyOf (n : String) : Option Body := (convSigs.find? (·.1 = n)).map (·.2.2.2)

/-- Each `...OrNull` alias shares its base function's body (`value_to_integer`, …). -/
theorem orNull_aliases :
    bodyOf "tointeger" = bodyOf "toIntegerOrNull" ∧ bodyOf "tofloat" = bodyOf "toFloatOrNull" ∧
    bodyOf "tostring" = bodyOf "tostringornull" ∧ bodyOf "toBoolean" = bodyOf "toBooleanOrNull" ∧
    bodyOf "tofloat" = some .toFloat ∧ bodyOf "tojson" = some .toJson := by
  decide

/-- Every argument is accepted by an alias (`Type::Any`), so the body decides: an
unsupported type yields `Null` rather than a type error. -/
theorem orNull_accepts (v : V F) : validateArgsType (.fixed [.any]) [v] = none := by
  simp [validateArgsType, validateArgsType.go, vot_any]

theorem toFloat_unsupported (b : Bool) : toFloat c (.bool b) = .null ∧ toFloat c (.list []) = .null := ⟨rfl, rfl⟩

/-- `value_to_float`: total (no panic, never `Err`); the result is a Float or Null —
exactly the registered `ret` — and an Integer is promoted with `as f64`. -/
theorem toFloat_ret (v : V F) : Accepts (toFloat c v) (.union [.float, .null]) := by
  cases v <;> simp only [toFloat, optV] <;>
  first
  | exact .union _ _ .float (by simp) (.tag _ _ rfl)
  | exact .union _ _ .null (by simp) (.tag _ _ rfl)
  | (split <;> first
      | exact .union _ _ .float (by simp) (.tag _ _ rfl)
      | exact .union _ _ .null (by simp) (.tag _ _ rfl))

theorem toFloat_arms (i : Int) (x : F) (s : String) :
    toFloat c (.int i) = .float (c.ofInt i) ∧ toFloat c (.float x) = .float x ∧
    toFloat c (.str s) = (match c.parseF s with | some f => .float f | none => .null) := by
  refine ⟨rfl, rfl, ?_⟩; simp only [toFloat, optV]; cases c.parseF s <;> rfl

/-- The wrapper never panics, and the empty-slice arm is unreachable (arity exactly 1). -/
theorem toFloatArgs_spec (v : V F) (rest : List (V F)) : toFloatArgs c (v :: rest) = toFloat c v := rfl

/-- `value_to_string`: no panic on any non-empty slice; `None` is dead since
`validate (.fixed [t]) 0 = false`; result is String or Null. -/
theorem toStr_no_panic (args : List (V F)) (h : args ≠ []) : (toStrArgs c args).isSome := by
  cases args with
  | nil => exact absurd rfl h
  | cons v _ => rfl

theorem arity_one (t : Ty) (h : t.isOpt = false) : validate (.fixed [t]) 0 = false := by
  simp [validate, h]

theorem toStr_ret (v : V F) : Accepts (toStr c v) (.union [.string, .null]) := by
  cases v <;> simp only [toStr] <;>
  first
  | exact .union _ _ .string (by simp) (.tag _ _ rfl)
  | exact .union _ _ .null (by simp) (.tag _ _ rfl)

/-- Non-listed types (List/Map/Node/…) give Null through `tostringornull` (C agrees). -/
theorem toStr_unsupported (l : List (V F)) (n : Nat) :
    toStr c (.list l) = .null ∧ toStr c (.node n) = .null := ⟨rfl, rfl⟩

/-- `to_json`: always a String for one argument (never Null, never `Err`). -/
theorem toJson_spec (js : V F → String) (v : V F) (rest : List (V F)) :
    toJsonArgs js (v :: rest) = some (.str (js v)) ∧ toJsonArgs (F := F) js [] = none := ⟨rfl, rfl⟩

theorem toJson_ret (js : V F → String) (v : V F) :
    ∃ r, toJsonArgs js [v] = some r ∧ Accepts r (.union [.string, .null]) :=
  ⟨_, rfl, .union _ _ .string (by simp) (.tag _ _ rfl)⟩

end ValueMath.ConvReg
