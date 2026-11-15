import FalkorValueMath.Value
/-!
# C FalkorDB arithmetic (reference) vs the Rust model

C (`src/value.c`, master: `SIValue_Add` / `_Multiply` / `_Divide`) runs behind the
function-argument type check of `AR_ADD`/`AR_MUL`/`AR_DIV` (`arithmetic/arithmetic_op.c`
registrations), so an operand outside the declared types is rejected *before* the null
check. Types declared (read off the live error messages):

* `*`, `/`, `%`: Integer, Float, Null;
* `+`: Map, List, Datetime, Date, Time, Duration, String, Boolean, Integer, Float, Null.

`SIValue_Add`: null → null; Int+Int → wrapped Int; any List → concat; any String →
string concat of both operands' printed forms; any Map → merge; temporal+Duration;
else `SI_GET_NUMERIC(a) + SI_GET_NUMERIC(b)` as a double (Booleans count as 0/1).
-/

namespace ValueMath
open FalkorCore

variable {F : Type}

def cMulType : V F → Bool
  | .int _ | .float _ | .null => true
  | _ => false

/-- `SI_GET_NUMERIC`: the operand as a double (Int converted, Float as is, Boolean 0/1). Only
the cases the C arithmetic reaches are defined; anything else is not numeric. -/
def cNumeric (o : FOps F) : V F → Option F
  | .int i => some (o.ofInt i)
  | .float f => some f
  | .bool b => some (if b then o.ofInt 1 else o.ofInt 0)
  | _ => none

/-- C `SIValue_Multiply` (`value.c`: both Int → `a.longval * b.longval`, wrapping; otherwise
`SI_GET_NUMERIC(a) * SI_GET_NUMERIC(b)` as a double). Written from the C source, not from the
Rust `mul`, so comparing the two is not circular. -/
def siMultiply (o : FOps F) (x y : V F) : Option (V F) :=
  match x, y with
  | .int a, .int b => some (.int (wMul a b))
  | _, _ => match cNumeric o x, cNumeric o y with
    | some u, some v => some (.float (o.mul u v))
    | _, _ => none

/-- C `a * b`: the `AR_MUL` type check (rejects before the null check), then null → null
(the wrapper's null short-circuit), then `SIValue_Multiply`. A `none` from `siMultiply` is
unreachable behind the type check (every accepted non-null type is numeric). -/
def cMul (o : FOps F) (x y : V F) : Except String (V F) :=
  if !cMulType x then .error (mulMsg x)
  else if !cMulType y then .error (mulMsg y)
  else match x, y with
    | .null, _ => .ok .null
    | _, .null => .ok .null
    | _, _ => match siMultiply o x y with
      | some v => .ok v
      | none => .error "unreachable: SIValue_Multiply on a type-checked operand"

/-- On operands C accepts, C and Rust agree on `*` (value and error): the C model above is
independent of `mul`, so this is a comparison of two implementations. -/
theorem mul_agree_on_numeric (o : FOps F) (x y : V F) (hx : cMulType x) (hy : cMulType y) :
    cMul o x y = mul o x y := by
  cases x <;> cases y <;> simp_all [cMul, cMulType, siMultiply, cNumeric, mul]

/-- `null * 'a'`: Rust returns null, C raises a type error (live-confirmed). -/
theorem null_mul_str_diverges (o : FOps F) :
    mul o .null (.str "a") = .ok .null ∧
    cMul o .null (.str "a") = .error "Type mismatch: expected Integer, Float, or Null but was String" := by
  constructor <;> rfl

/-- Printed form C's string concatenation uses (`SIValue_ToString`). -/
structure CPrint (F : Type) where
  pr : V F → String

/-- Types `AR_ADD` declares (read off the live error messages): Map, List, Datetime, Date,
Time, Duration, String, Boolean, Integer, Float, Null. -/
def cAddType : V F → Bool
  | .map _ | .list _ | .datetime _ | .date _ | .time _ | .duration _ | .str _ | .bool _
  | .int _ | .float _ | .null => true
  | _ => false

def cAddMsg (x : V F) : String :=
  s!"Type mismatch: expected Map, List, Datetime, Date, Time, Duration, String, Boolean, Integer, Float, or Null but was {x.name}"

/-- C `SIValue_Add` behind `AR_ADD`'s type check (the temporal/map cases are elided to
`other`, which the lemmas below never reach). The check runs first, so an operand outside
the declared types (Node, Edge, Path, Point, Vectorf32) is an error even next to a null. -/
def cAdd (o : FOps F) (p : CPrint F) (x y : V F)
    (other : Except String (V F)) : Except String (V F) :=
  if !cAddType x then .error (cAddMsg x)
  else if !cAddType y then .error (cAddMsg y)
  else match x, y with
  | .null, _ => .ok .null
  | _, .null => .ok .null
  | .int a, .int b => .ok (.int (wAdd a b))
  | .list a, .list b => .ok (.list (a ++ b))
  | .list a, b => .ok (.list (a ++ [b]))
  | a, .list b => .ok (.list (a :: b))
  | .str a, b => .ok (.str (a ++ p.pr b))
  | a, .str b => .ok (.str (p.pr a ++ b))
  | a, b => match cNumeric o a, cNumeric o b with
    | some u, some v => .ok (.float (o.add u v))
    | _, _ => other

/-- `'a' + date('2020-01-01')`: C concatenates, Rust raises a type error. -/
theorem str_plus_date_diverges (o : FOps F) (t : TOps) (p : CPrint F)
    (other : Except String (V F)) (d : Int) :
    add o t (.str "a") (.date d) = .error "Unexpected types for add operator (String, Date)" ∧
    cAdd o p (.str "a") (.date d) other = .ok (.str ("a" ++ p.pr (.date d))) := by
  constructor <;> rfl

/-- `true + 1`: C computes a double (`1.0 + 1.0`, printed `2`; Booleans count as 0/1 through
`SI_GET_NUMERIC`), Rust raises a type error. -/
theorem bool_plus_int_diverges (o : FOps F) (t : TOps) (p : CPrint F)
    (other : Except String (V F)) :
    add o t (.bool true) (.int 1) = .error "Unexpected types for add operator (Boolean, Integer)" ∧
    cAdd o p (.bool true) (.int 1) other = .ok (.float (o.add (o.ofInt 1) (o.ofInt 1))) := by
  refine ⟨rfl, ?_⟩
  simp [cAdd, cAddType, cNumeric]

/-- On Int/List/String-with-scalar operands C and Rust `+` agree, provided C prints
Int/Float/Bool the way Rust does (`{i}`, `{:.6}`, `true`/`false` — the live outputs
`a1.500000`, `atrue` match). -/
theorem add_agree_str_scalar (o : FOps F) (t : TOps) (p : CPrint F)
    (other : Except String (V F)) (s : String) (y : V F)
    (hy : (∃ i, y = .int i) ∨ (∃ f, y = .float f) ∨ (∃ b, y = .bool b) ∨ (∃ s', y = .str s'))
    (hpi : ∀ i, p.pr (.int i) = toString i) (hpf : ∀ f, p.pr (.float f) = o.fmt6 f)
    (hpb : ∀ b, p.pr (.bool b) = boolStr b) (hps : ∀ s', p.pr (.str s') = s') :
    add o t (.str s) y = cAdd o p (.str s) y other := by
  rcases hy with ⟨i, rfl⟩ | ⟨f, rfl⟩ | ⟨b, rfl⟩ | ⟨s', rfl⟩ <;> simp [add, addSlow, cAdd, cAddType, hpi, hpf, hpb, hps]

theorem add_agree_int_int (o : FOps F) (t : TOps) (p : CPrint F)
    (other : Except String (V F)) (a b : Int) :
    add o t (.int a) (.int b) = cAdd o p (.int a) (.int b) other := rfl

/-- Int/Float mixes: both compute the double sum. -/
theorem add_agree_float (o : FOps F) (t : TOps) (p : CPrint F)
    (other : Except String (V F)) (a b : F) (i j : Int) :
    add o t (.float a) (.float b) = cAdd o p (.float a) (.float b) other ∧
    add o t (.float a) (.int j) = cAdd o p (.float a) (.int j) other ∧
    add o t (.int i) (.float b) = cAdd o p (.int i) (.float b) other := by
  refine ⟨?_, ?_, ?_⟩ <;> simp [add, cAdd, cAddType, cNumeric]

theorem add_agree_list (o : FOps F) (t : TOps) (p : CPrint F)
    (other : Except String (V F)) (l : List (V F)) (y : V F) (hy : y ≠ .null)
    (hty : cAddType y = true) :
    add o t (.list l) y = cAdd o p (.list l) y other := by
  cases y <;> first | exact absurd rfl hy | simp_all [add, addSlow, cAdd, cAddType] | rfl

end ValueMath
