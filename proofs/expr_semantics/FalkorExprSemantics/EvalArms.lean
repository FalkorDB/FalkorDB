import FalkorExprSemantics.Frame
/-!
# The 2026-10-21 arms of `ExprEval` (eval.rs:340-1134): short-circuit and error order

Theorems about the arms `Eval.lean` gained: `AND`/`OR` stop at the first settling operand and
never evaluate the rest (`and_false_short_circuits`, `or_true_short_circuits`; live on main:
`RETURN false AND 1/0 = 1` → `false`), report the first operand error in order
(`andOr_error_first`), and are Kleene's three-valued fold when every operand is a boolean or
null (`andOr_kleene`); `XOR` is settled by a `null` (`xor_null_settles`); `NOT`
(`not_spec`); `CASE` evaluates only the matching `THEN` (`case_hit_lazy`,
`case_value_hit_lazy`) and the `ELSE` only when nothing matched (`case_miss_else`); a list
literal evaluates its elements **last to first** (`list_error_rightmost`), and otherwise
equals the left-to-right list (`evalRev_ok`); a function call checks the runtime before its
arguments (`func_constant_errors`) and validates before calling (`func_common`); `IN`
(`in_spec`); subscripts (`getElement_index`).
-/
namespace FalkorExpr.Eval

variable {V : Type} (O : Ops V) (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V))

/-! ## AND / OR -/

/-- **`false AND …` is `false`**, whatever follows (an erroring operand is never reached). -/
theorem and_false_short_circuits (a : V) (ha : O.boolOf a = some (some false)) (rest : List (E V)) :
    evalNode O ev c env (.and (.const a :: rest)) = .ok (O.bool false) := by
  simp [evalNode, andOrLoop, ha, bind, Except.bind]

/-- **`true OR …` is `true`**, whatever follows. -/
theorem or_true_short_circuits (a : V) (ha : O.boolOf a = some (some true)) (rest : List (E V)) :
    evalNode O ev c env (.or (.const a :: rest)) = .ok (O.bool true) := by
  simp [evalNode, andOrLoop, ha, bind, Except.bind]

/-- The first operand's error is reported before anything after it is looked at. -/
theorem andOr_error_first (sc n : Bool) (bad : E V) (e : String) (hbad : evalNode O ev c env bad = .error e)
    (rest : List (E V)) : andOrLoop O ev c env sc n (bad :: rest) = .error e := by
  simp [andOrLoop, hbad, bind, Except.bind]

/-- A non-boolean, non-null operand is a type error naming its type (#3294). -/
theorem andOr_mismatch (sc n : Bool) (a : V) (ha : O.boolOf a = none) (rest : List (E V)) :
    andOrLoop O ev c env sc n (.const a :: rest) = .error (boolMismatchMsg O a) := by
  simp [andOrLoop, evalNode, ha, bind, Except.bind]

/-- Kleene's fold over the operands' three-valued verdicts (`none` = null). -/
def kleeneSpec (sc n : Bool) (bs : List (Option Bool)) : Option Bool :=
  if some sc ∈ bs then some sc else if n || none ∈ bs then none else some !sc

def ofOB : Option Bool → V
  | some b => O.bool b
  | none => O.null

/-- **AND/OR is Kleene logic**: when every operand evaluates to a boolean or null, the
result is `sc` if one is `sc`, else null if one is null, else `!sc` — the three-valued
`AND` (`sc = false`) / `OR` (`sc = true`). -/
theorem andOr_kleene (sc : Bool) : ∀ (cs : List (E V)) (vs : List V) (bs : List (Option Bool)) (n : Bool),
    evalList O ev c env cs = .ok vs → vs.mapM O.boolOf = some bs →
    andOrLoop O ev c env sc n cs = .ok (ofOB O (kleeneSpec sc n bs))
  | [], vs, bs, n, h1, h2 => by
    simp [evalList] at h1; subst h1; simp at h2; subst h2
    cases n <;> simp [andOrLoop, kleeneSpec, ofOB]
  | ch :: cs, vs, bs, n, h1, h2 => by
    simp only [evalList, bind, Except.bind] at h1
    cases hv : evalNode O ev c env ch with
    | error e => simp [hv] at h1
    | ok v =>
      simp only [hv] at h1
      cases hr : evalList O ev c env cs with
      | error e => simp [hr] at h1
      | ok ws =>
        simp only [hr, Except.ok.injEq] at h1; subst h1
        simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
          Option.some.injEq] at h2
        obtain ⟨b, hb, bs', hbs', rfl⟩ := h2
        simp only [andOrLoop, hv, bind, Except.bind, hb]
        cases b with
        | none =>
          rw [andOr_kleene sc cs ws bs' true hr hbs']
          simp [kleeneSpec]
        | some b =>
          by_cases hbs : b = sc
          · subst hbs; simp [kleeneSpec, ofOB]
          · simp only [hbs, if_false]
            rw [andOr_kleene sc cs ws bs' n hr hbs']
            have hne : ¬ sc = b := fun h => hbs h.symm
            simp [kleeneSpec, hne]

/-! ## XOR, NOT -/

/-- A `null` operand settles `XOR` as `null`; nothing after it is evaluated. -/
theorem xor_null_settles (last : Option Bool) (a : V) (ha : O.boolOf a = some none) (rest : List (E V)) :
    xorLoop O ev c env last (.const a :: rest) = .ok O.null := by
  simp [xorLoop, evalNode, ha, bind, Except.bind]

/-- Two booleans: `XOR` is exclusive or. -/
theorem xor_two (a b : V) (x y : Bool) (ha : O.boolOf a = some (some x)) (hb : O.boolOf b = some (some y)) :
    evalNode O ev c env (.xor [.const a, .const b]) = .ok (O.bool (logicalXor x y)) := by
  simp [evalNode, xorLoop, ha, hb, bind, Except.bind]

theorem logicalXor_eq (x y : Bool) : logicalXor x y = (x != y) := by cases x <;> cases y <;> rfl

theorem not_spec (a : E V) (v : V) (h : evalNode O ev c env a = .ok v) :
    evalNode O ev c env (.not a) = match O.boolOf v with
      | some (some b) => .ok (O.bool !b)
      | some none => .ok O.null
      | none => .error (boolMismatchMsg O v) := by
  simp only [evalNode, h, bind, Except.bind]
  generalize O.boolOf v = x
  rcases x with _ | _ | _ <;> rfl

/-! ## CASE -/

/-- **Searched `CASE` is lazy**: the first `WHEN` that holds selects its `THEN`; later arms
and the `ELSE` are never evaluated. -/
theorem case_hit_lazy (w t : E V) (rest : List (E V)) (els : E V) (wv : V)
    (hw : evalNode O ev c env w = .ok wv) (hit : caseHit O none wv = true) :
    evalNode O ev c env (.case none (w :: t :: rest) els) = evalNode O ev c env t := by
  simp only [evalNode, evalSubj, evalArms, hw, hit, bind, Except.bind, ite_true]
  cases evalNode O ev c env t <;> rfl

/-- Value form: an arm matches when it compares `(Equal, None)` with the subject. -/
theorem case_value_hit_lazy (s w t : E V) (rest : List (E V)) (els : E V) (sv wv : V)
    (hs : evalNode O ev c env s = .ok sv) (hw : evalNode O ev c env w = .ok wv)
    (hit : O.cmp wv sv = (.eq, .none)) :
    evalNode O ev c env (.case (some s) (w :: t :: rest) els) = evalNode O ev c env t := by
  have : caseHit O (some sv) wv = true := by simp [caseHit, hit]
  simp only [evalNode, evalSubj, evalArms, hs, hw, this, bind, Except.bind, ite_true]
  cases evalNode O ev c env t <;> rfl

/-- No arm matched: the `ELSE`. -/
theorem case_miss_else (els : E V) (subj : Option (E V)) (sv : Option V)
    (hs : evalSubj O ev c env subj = .ok sv) (arms : List (E V))
    (ha : evalArms O ev c env sv arms = .ok none) :
    evalNode O ev c env (.case subj arms els) = evalNode O ev c env els := by
  simp only [evalNode, hs, ha, bind, Except.bind]

/-! ## List literals: last to first -/

/-- **A list literal reports its rightmost failing element's error** (the stack machine
evaluates the elements last to first, eval.rs:563-578). -/
theorem list_error_rightmost (a b : E V) (e : String) (hb : evalNode O ev c env b = .error e) :
    evalNode O ev c env (.list [a, b]) = .error e := by
  simp [evalNode, evalRev, hb, bind, Except.bind]

/-- When every element evaluates, the order of evaluation is invisible: the list is the
left-to-right one. -/
theorem evalRev_ok : ∀ (cs : List (E V)) (vs : List V),
    evalList O ev c env cs = .ok vs → evalRev O ev c env cs = .ok vs
  | [], vs, h => by simpa [evalList, evalRev] using h
  | ch :: cs, vs, h => by
    simp only [evalList, bind, Except.bind] at h
    cases hv : evalNode O ev c env ch with
    | error e => simp [hv] at h
    | ok v =>
      simp only [hv] at h
      cases hr : evalList O ev c env cs with
      | error e => simp [hr] at h
      | ok ws =>
        simp only [hr, Except.ok.injEq] at h; subst h
        simp [evalRev, evalRev_ok cs ws hr, hv, bind, Except.bind]

/-! ## Function calls -/

/-- The runtime is checked before any argument is evaluated: a call under the constant
evaluator errors even when an argument would. -/
theorem func_constant_errors (f : Fn V) (cs : List (E V)) :
    evalNode O Ev.constant c env (.func f cs) = .error "not a constant expression" := by
  simp [evalNode, Ev.constant, Ev.rt, bind, Except.bind]

/-- The common path: arguments left to right, `validate_args_type`, the read-only check,
the call. -/
theorem func_common (params : String → Option V) (f : Fn V) (hs : f.structFn = none) (cs : List (E V)) :
    evalNode O (Ev.fromRuntime params) c env (.func f cs) =
      (evalList O (Ev.fromRuntime params) c env cs) >>= callFn O f := by
  simp [evalNode, Ev.fromRuntime, Ev.rt, hs, bind, Except.bind]

/-! ## IN, subscripts -/

theorem in_spec (a l : E V) (v lv : V) (ha : evalNode O ev c env a = .ok v) (hl : evalNode O ev c env l = .ok lv) :
    evalNode O ev c env (.inList a l) = listContains O lv v := by
  simp [evalNode, ha, hl, bind, Except.bind]

/-- `l[i]` with a negative index counts from the end; out of range is `null`. -/
theorem getElement_index (rt : Except String Unit) (arr i : V) (vs : List V) (j : Int)
    (hl : O.asList arr = some vs) (hi : O.asInt i = some j) :
    getElem O rt arr i =
      let n := if j < 0 then (vs.length : Int) + j else j
      if 0 ≤ n ∧ n < vs.length then .ok (vs[n.toNat]?.getD O.null) else .ok O.null := by
  simp [getElem, hl, hi]

end FalkorExpr.Eval
