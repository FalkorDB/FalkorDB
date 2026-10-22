import FalkorExprSemantics.Eval
/-!
# `evaluate_param` (eval.rs:1814): parameter expressions of `CYPHER p=…`
(split out of `Eval.lean` on 2026-10-21)
-/
namespace FalkorExpr.Eval

/-! ## `evaluate_param` (eval.rs:1762) — parameter expressions of `CYPHER p=…` -/

inductive PV where
  | null | int (i : Int) | float (f : Int) | str (s : String) | list (vs : List PV)
  | map (kvs : List (String × PV))

inductive PE where
  | const (v : PV)
  | list (cs : List PE)
  | map (kvs : List (Option String × PE))  -- `none`: a non-string key node
  | neg (e : PE)
  | other

/-- i64 `checked_neg`: fails only on `i64::MIN`. -/
def checkedNeg (i : Int) : Option Int := if i = -2^63 then none else some (-i)

mutual
def evalParam : PE → Except String PV
  | .const v => .ok v
  | .list cs => do let vs ← evalParams cs; .ok (.list vs)
  | .map kvs => do let m ← evalParamMap kvs; .ok (.map m)
  | .neg e => do
    let v ← evalParam e
    match v with
    | .int i => match checkedNeg i with
      | some j => .ok (.int j)
      | none => .error "ArgumentError: integer overflow in unary minus"
    | .float f => .ok (.float (-f))    -- floats abstracted to their negation
    | _ => .ok .null
  | .other => .error "Invalid parameter expression."
def evalParams : List PE → Except String (List PV)
  | [] => .ok []
  | c :: cs => do let v ← evalParam c; let r ← evalParams cs; .ok (v :: r)
def evalParamMap : List (Option String × PE) → Except String (List (String × PV))
  | [] => .ok []
  | (some k, c) :: cs => do let v ← evalParam c; let r ← evalParamMap cs; .ok ((k, v) :: r)
  | (none, _) :: _ => .error "Map parameter key must be a string"
end

/-- Constants and lists of constants evaluate to themselves. -/
theorem evalParams_consts (vs : List PV) : evalParams (vs.map .const) = .ok vs := by
  induction vs with
  | nil => rfl
  | cons v vs ih => simp [evalParams, evalParam, ih, bind, Except.bind]

theorem evalParam_neg (i : Int) :
    evalParam (.neg (.const (.int i))) =
      if i = -2^63 then .error "ArgumentError: integer overflow in unary minus" else .ok (.int (-i)) := by
  by_cases h : i = -2^63
  · subst h; rfl
  · simp only [evalParam, checkedNeg, h, if_false, bind, Except.bind]

theorem evalParam_neg_null : evalParam (.neg (.const (.str "x"))) = .ok .null := rfl

theorem evalParam_badkey (e : PE) : evalParam (.map [(none, e)]) = .error "Map parameter key must be a string" := rfl

end FalkorExpr.Eval
