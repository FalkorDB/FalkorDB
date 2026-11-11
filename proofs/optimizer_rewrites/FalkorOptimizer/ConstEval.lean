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
(#2902). Since #2904 every modelled arm propagates
errors with `?`, and `constEval_sound` proves the constant verdict is the runtime
verdict on every row; this discharges `eliminate_whole_filter`'s and
`andLoop_drop_true`'s hypotheses (`constTrue_filter`, `andLoop_drop_constTrue`).

| here | there |
| --- | --- |
| `evalG look` | `ExprEval::eval_node` with variable lookup `look` (`resolve_var`, eval.rs:287-293) |
| `constEval`  | `ExprEval::constant().eval(.., NO_ROW, None)` — `look` always errors |
| `constTrue`  | `is_constant_true`'s fast path: `matches!(.., Ok(Value::Bool(true)))` (eliminate_true_filters.rs:52-57); the slow path re-runs it on `substitute_params`' tree (`substP`; `constTrue_slow`, parameters modelled as marked variable leaves) |
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

/-! ### The slow path: parameters (`substitute_params`)

`Expr` has no parameter node, so a parameter `$n` is modelled as a leaf `.var v` with
`pv v = some n`; its runtime value is `ps.lookup n` (a missing parameter is an error, as
in `evX`). `substP` is `substitute_params`: it replaces the parameters present in `ps` by
literals and leaves a missing one as the (erroring) leaf. -/

/-- `substitute_params` (eliminate_true_filters.rs:83-98). -/
def substP (pv : Var → Option String) (ps : List (String × Val)) : Expr → Expr
  | .lit v => .lit v
  | .var v =>
    match pv v with
    | some n =>
      match ps.lookup n with
      | some val => .lit val
      | none => .var v
    | none => .var v
  | .eq a b => .eq (substP pv ps a) (substP pv ps b)
  | .neq a b => .neq (substP pv ps a) (substP pv ps b)
  | .lt a b => .lt (substP pv ps a) (substP pv ps b)
  | .gt a b => .gt (substP pv ps a) (substP pv ps b)
  | .sub a b => .sub (substP pv ps a) (substP pv ps b)
  | .div a b => .div (substP pv ps a) (substP pv ps b)
  | .and a b => .and (substP pv ps a) (substP pv ps b)
  | .pow a b => .pow (substP pv ps a) (substP pv ps b)

/-- The runtime lookup of a row with parameters: a parameter reads `ps` (missing: error),
any other variable reads the row. -/
def paramLook (pv : Var → Option String) (ps : List (String × Val)) (r : Row) :
    Var → Except Err Val :=
  fun v =>
    match pv v with
    | some n =>
      match ps.lookup n with
      | some val => .ok val
      | none => .error .notFound
    | none => .ok (r.get v)

/-- Substituting parameters does not change the value under any lookup that reads them
from `ps`. -/
theorem substP_evalG (pv : Var → Option String) (ps : List (String × Val))
    (look : Var → Except Err Val)
    (h : ∀ v n val, pv v = some n → ps.lookup n = some val → look v = .ok val) :
    ∀ e, evalG look (substP pv ps e) = evalG look e := by
  intro e
  induction e with
  | lit => rfl
  | var v =>
    cases hp : pv v with
    | none => simp [substP, hp]
    | some n =>
      cases hl : ps.lookup n with
      | none => simp [substP, hp, hl]
      | some val => simp [substP, hp, hl, evalG, h v n val hp hl]
  | eq a b iha ihb | neq a b iha ihb | lt a b iha ihb | gt a b iha ihb
  | sub a b iha ihb | div a b iha ihb | and a b iha ihb | pow a b iha ihb =>
    simp only [substP, evalG, iha, ihb]

/-- **`is_constant_true`, slow path**: if the constant evaluator accepts the
parameter-substituted tree, the original expression (parameters read from `ps`) is `true`
on every row. -/
theorem constTrue_slow (pv : Var → Option String) (ps : List (String × Val)) (c : Expr)
    (h : constTrue (substP pv ps c) = true) (r : Row) :
    evalG (paramLook pv ps r) c = .ok (.bool true) := by
  have hl : ∀ v n val, pv v = some n → ps.lookup n = some val → paramLook pv ps r v = .ok val := by
    intro v n val hp hl'; simp [paramLook, hp, hl']
  rw [← substP_evalG pv ps (paramLook pv ps r) hl c]
  unfold constTrue at h
  split at h
  · rename_i heq; exact constEval_sound _ _ _ heq
  · cases h

end Falkor.Opt
