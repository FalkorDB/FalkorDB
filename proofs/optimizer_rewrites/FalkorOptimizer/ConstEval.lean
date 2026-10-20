import FalkorOptimizer.Basic
/-
# The plan-time constant evaluator is sound (since #2904)

`eliminate_true_filters` removes a `Filter` (or one `AND` conjunct) when
`is_constant_true` holds: `ExprEval::constant().eval(..)` — the *same* `eval_node`
as at runtime, run with no row and no runtime — returns `Ok(Bool(true))`
(eliminate_true_filters.rs:48-71). Every variable then evaluates to an error
("Variable … not found"), and so do parameters, properties of entities and
function calls (`self.rt()?`, eval.rs:278-282).

Soundness needs: whenever an operator's result is `Ok`, every child it consulted
was `Ok`, i.e. no operator swallows a child's error. Before #2904 the `Pow` arm
did (`flat_map` over `Result`), so `WHERE x ^ true` was folded to `true`
(#2902, `pre2904_pow_constant_true`). Since #2904 every modelled arm propagates
errors with `?`, and `constEval_sound` proves the constant verdict is the runtime
verdict on every row; this discharges `eliminate_whole_filter`'s and
`andLoop_drop_true`'s hypotheses (`constTrue_filter`, `andLoop_drop_constTrue`).

| here | there |
| --- | --- |
| `evalG look` | `ExprEval::eval_node` with variable lookup `look` (`resolve_var`, eval.rs:287-293) |
| `constEval`  | `ExprEval::constant().eval(.., NO_ROW, None)` — `look` always errors |
| `constTrue`  | `is_constant_true`'s fast path: `matches!(.., Ok(Value::Bool(true)))` (eliminate_true_filters.rs:52-57); the slow path re-runs it on `substitute_params`' tree (Helpers.evX_subst: same value) |
| `pre2904_powFlat` | the `Pow` arm before #2904: `flat_map(eval_node).reduce(apply_pow)` |
-/
namespace Falkor.Opt

/-- `eval` with an arbitrary variable lookup. -/
def evalG (look : Var → Except Err Val) : Expr → Except Err Val
  | .lit v => .ok v
  | .var v => look v
  | .eq a b => do let x ← evalG look a; let y ← evalG look b; pure (cmpEq x y)
  | .neq a b => do let x ← evalG look a; let y ← evalG look b; pure (notV (cmpEq x y))
  | .lt a b => do let x ← evalG look a; let y ← evalG look b; pure (cmpLt x y)
  | .gt a b => do let x ← evalG look a; let y ← evalG look b; pure (cmpLt y x)
  | .sub a b => do let x ← evalG look a; let y ← evalG look b; subV x y
  | .div a b => do let x ← evalG look a; let y ← evalG look b; divV x y
  | .and a b => andStep (evalG look a) (fun _ => evalG look b)
  | .pow a b => do let x ← evalG look a; let y ← evalG look b; powV x y

/-- The runtime evaluator is `evalG` reading the row. -/
theorem eval_eq_evalG (r : Row) (e : Expr) : eval r e = evalG (fun v => .ok (r.get v)) e := by
  induction e with
  | lit => rfl
  | var => rfl
  | eq a b iha ihb | neq a b iha ihb | lt a b iha ihb | gt a b iha ihb
  | sub a b iha ihb | div a b iha ihb | and a b iha ihb | pow a b iha ihb =>
    simp only [eval, evalG, iha, ihb]

/-- The plan-time constant evaluator: no row, every variable is an error. -/
def constEval : Expr → Except Err Val := evalG (fun _ => .error .notFound)

theorem bind_ok {α β : Type} {m : Except Err α} {f : α → Except Err β} {v : β}
    (h : (m >>= f) = .ok v) : ∃ x, m = .ok x ∧ f x = .ok v := by
  cases m with
  | error e => cases h
  | ok x => exact ⟨x, rfl, h⟩

/-- **Soundness of constant evaluation** (since #2904): if the constant evaluator
returns a value, every lookup — in particular every row — gives that value. -/
theorem constEval_sound (look : Var → Except Err Val) :
    ∀ (c : Expr) (v : Val), constEval c = .ok v → evalG look c = .ok v := by
  intro c
  induction c with
  | lit => intro v h; exact h
  | var => intro v h; cases h
  | eq a b iha ihb | neq a b iha ihb | lt a b iha ihb | gt a b iha ihb
  | sub a b iha ihb | div a b iha ihb | pow a b iha ihb =>
    intro v h
    simp only [constEval, evalG] at h ⊢
    obtain ⟨x, hx, h⟩ := bind_ok h
    obtain ⟨y, hy, h⟩ := bind_ok h
    rw [iha x hx, ihb y hy]
    exact h
  | and a b iha ihb =>
    intro v h
    simp only [constEval, evalG] at h ⊢
    unfold andStep at h ⊢
    cases ha : evalG (fun _ => .error .notFound) a with
    | error e => rw [ha] at h; cases h
    | ok x =>
      rw [ha] at h
      rw [iha x ha]
      cases x with
      | bool bx =>
        cases bx with
        | false => exact h
        | true =>
          exact ihb v h
      | null =>
        cases hb : evalG (fun _ => .error .notFound) b with
        | error e => rw [hb] at h; cases h
        | ok y => rw [hb] at h; rw [ihb y hb]; exact h
      | int => exact h
      | str => exact h

/-- Runtime form: a constant-true expression is `true` on every row. -/
theorem constEval_rows (c : Expr) (v : Val) (h : constEval c = .ok v) (r : Row) :
    eval r c = .ok v := by
  rw [eval_eq_evalG]; exact constEval_sound _ c v h

/-- `is_constant_true` (fast path). -/
def constTrue (c : Expr) : Bool :=
  match constEval c with
  | .ok (.bool true) => true
  | _ => false

theorem constTrue_eval (c : Expr) (h : constTrue c = true) (r : Row) :
    eval r c = .ok (.bool true) := by
  unfold constTrue at h
  split at h
  · rename_i heq; exact constEval_rows c _ heq r
  · cases h

/-- **`eliminate_true_filters`, whole filter** (eliminate_true_filters.rs:111-115):
removing a `Filter` whose predicate `is_constant_true` keeps every row —
`eliminate_whole_filter` with `hconst` discharged. -/
theorem constTrue_filter (c : Expr) (xs : List Row) (h : constTrue c = true) :
    filterRows c xs = .ok xs :=
  eliminate_whole_filter c xs (constTrue_eval c h)

/-- **`eliminate_true_filters`, AND conjuncts** (eliminate_true_filters.rs:118-143):
dropping the conjuncts `is_constant_true` accepts leaves the AND loop's result
unchanged on every row — `andLoop_drop_true` with `htrue` discharged. -/
theorem andLoop_drop_constTrue (r : Row) (cs : List Expr) (n : Bool) :
    andLoop r (cs.filter (fun c => !constTrue c)) n = andLoop r cs n :=
  andLoop_drop_true r (fun c => !constTrue c) cs
    (fun c _ hk => constTrue_eval c (by simpa using hk) r) n

/-- `WHERE x ^ true` is no longer constant-true: the variable's error propagates. -/
theorem pow_var_not_constTrue (x : Var) :
    constEval (.pow (.var x) (.lit (.bool true))) = .error .notFound ∧
    constTrue (.pow (.var x) (.lit (.bool true))) = false := ⟨rfl, rfl⟩

/-- …and at runtime it is the C type error (`true` is not numeric) on every row. -/
theorem pow_true_type_error (x : Var) (r : Row) :
    eval r (.pow (.var x) (.lit (.bool true))) = .error .typeMismatch := by
  simp only [eval, bind, Except.bind]
  cases r.get x <;> rfl

/-! ### History: the `Pow` arm before #2904 broke soundness (#2902) -/

/-- `apply_pow` before #2904: non-numeric operands gave `null`. -/
def pre2904_powV : Val → Val → Val
  | .int i, .int j => powNum i j
  | _, _ => .null

/-- The `Pow` arm before #2904 on two operand results: `flat_map` drops an `Err`,
and a single surviving operand is returned unpowered. -/
def pre2904_powFlat (a b : Except Err Val) : Except Err Val :=
  match a, b with
  | .ok x, .ok y => .ok (pre2904_powV x y)
  | .ok x, .error _ | .error _, .ok x => .ok x
  | .error e, .error _ => .error e

/-- **BUG (#2902, fixed by #2904)**: the constant evaluator folded `x ^ true` to
`true` (the variable's error dropped), so the Filter was removed — while at
runtime `x ^ true` was `null` for every row and the Filter kept none. -/
theorem pre2904_pow_constant_true (x : Var) (r : Row) :
    pre2904_powFlat (.error .notFound) (.ok (.bool true)) = .ok (.bool true) ∧
    pre2904_powFlat (.ok (r.get x)) (.ok (.bool true)) = .ok .null := by
  refine ⟨rfl, ?_⟩
  cases r.get x <;> rfl

end Falkor.Opt
