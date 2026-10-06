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

variable {F : Type}

def cMulType : V F → Bool
  | .int _ | .float _ | .null => true
  | _ => false

/-- C `a * b` (`AR_MUL` type check, then `SIValue_Multiply`; null handled by the
wrapper returning null after validation). -/
def cMul (o : FOps F) (x y : V F) : Except String (V F) :=
  if !cMulType x then .error (mulMsg x)
  else if !cMulType y then .error (mulMsg y)
  else mul o x y

/-- On operands C accepts, C and Rust agree on `*` (value and error). -/
theorem mul_agree_on_numeric (o : FOps F) (x y : V F) (hx : cMulType x) (hy : cMulType y) :
    cMul o x y = mul o x y := by
  simp [cMul, hx, hy]

/-- `null * 'a'`: Rust returns null, C raises a type error (live-confirmed). -/
theorem null_mul_str_diverges (o : FOps F) :
    mul o .null (.str "a") = .ok .null ∧
    cMul o .null (.str "a") = .error "Type mismatch: expected Integer, Float, or Null but was String" := by
  constructor <;> rfl

/-- Printed form C's string concatenation uses (`SIValue_ToString`). -/
structure CPrint (F : Type) where
  pr : V F → String

/-- C `SIValue_Add` behind `AR_ADD`'s type check (temporal/map cases elided to
`other`, which the lemmas below never reach). -/
def cAdd (o : FOps F) (p : CPrint F) (numeric : V F → Option F) (x y : V F)
    (other : Except String (V F)) : Except String (V F) :=
  match x, y with
  | .null, _ => .ok .null
  | _, .null => .ok .null
  | .int a, .int b => .ok (.int (wAdd a b))
  | .list a, .list b => .ok (.list (a ++ b))
  | .list a, b => .ok (.list (a ++ [b]))
  | a, .list b => .ok (.list (a :: b))
  | .str a, b => .ok (.str (a ++ p.pr b))
  | a, .str b => .ok (.str (p.pr a ++ b))
  | a, b => match numeric a, numeric b with
    | some u, some v => .ok (.float (o.add u v))
    | _, _ => other

/-- `'a' + date('2020-01-01')`: C concatenates, Rust raises a type error. -/
theorem str_plus_date_diverges (o : FOps F) (t : TOps) (p : CPrint F) (num : V F → Option F)
    (other : Except String (V F)) (d : Int) :
    add o t (.str "a") (.date d) = .error "Unexpected types for add operator (String, Date)" ∧
    cAdd o p num (.str "a") (.date d) other = .ok (.str ("a" ++ p.pr (.date d))) := by
  constructor <;> rfl

/-- `true + 1`: C computes a double (`2`), Rust raises a type error. -/
theorem bool_plus_int_diverges (o : FOps F) (t : TOps) (p : CPrint F) (num : V F → Option F)
    (other : Except String (V F)) (u v : F) (hu : num (.bool true) = some u)
    (hv : num (.int 1) = some v) :
    add o t (.bool true) (.int 1) = .error "Unexpected types for add operator (Boolean, Integer)" ∧
    cAdd o p num (.bool true) (.int 1) other = .ok (.float (o.add u v)) := by
  refine ⟨rfl, ?_⟩
  simp [cAdd, hu, hv]

/-- On Int/List/String-with-scalar operands C and Rust `+` agree, provided C prints
Int/Float/Bool the way Rust does (`{i}`, `{:.6}`, `true`/`false` — the live outputs
`a1.500000`, `atrue` match). -/
theorem add_agree_str_scalar (o : FOps F) (t : TOps) (p : CPrint F) (num : V F → Option F)
    (other : Except String (V F)) (s : String) (y : V F)
    (hy : (∃ i, y = .int i) ∨ (∃ f, y = .float f) ∨ (∃ b, y = .bool b) ∨ (∃ s', y = .str s'))
    (hpi : ∀ i, p.pr (.int i) = toString i) (hpf : ∀ f, p.pr (.float f) = o.fmt6 f)
    (hpb : ∀ b, p.pr (.bool b) = boolStr b) (hps : ∀ s', p.pr (.str s') = s') :
    add o t (.str s) y = cAdd o p num (.str s) y other := by
  rcases hy with ⟨i, rfl⟩ | ⟨f, rfl⟩ | ⟨b, rfl⟩ | ⟨s', rfl⟩ <;> simp [add, addSlow, cAdd, hpi, hpf, hpb, hps]

theorem add_agree_int_int (o : FOps F) (t : TOps) (p : CPrint F) (num : V F → Option F)
    (other : Except String (V F)) (a b : Int) :
    add o t (.int a) (.int b) = cAdd o p num (.int a) (.int b) other := rfl

theorem add_agree_list (o : FOps F) (t : TOps) (p : CPrint F) (num : V F → Option F)
    (other : Except String (V F)) (l : List (V F)) (y : V F) (hy : y ≠ .null) :
    add o t (.list l) y = cAdd o p num (.list l) y other := by
  cases y <;> first | exact absurd rfl hy | rfl

end ValueMath
