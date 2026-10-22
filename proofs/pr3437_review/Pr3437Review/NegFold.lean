import FalkorCore.Value
import FalkorCore.Float
import FalkorCore.Int64
import FalkorExprSemantics

/-!
# PR #3437: a negated numeric literal is folded at parse time (cypher.rs:2558-2572)

The unary level of `parse_expr_inner` (cypher.rs:2547-2580) sees the operand `res` it built
under one optional `-` (cypher.rs:2230-2246: a single `Dash`, so `--1` never parses and
`-(…)` keeps a `Paren` child). On main it has two special cases; the PR inserts a third
between them:

1. `Negate(Constant(Int(i64::MIN)))` → `Constant(Int(i64::MIN))` (the literal `-2^63`);
2. **PR:** `Negate(Constant(Int(i) | Float(f)))` → `Constant(Int(-i))` / `Constant(Float(-f))`;
3. a bare `Constant(Int(i64::MIN))` → `Integer overflow '9223372036854775808'`.

| Lean | Rust |
| --- | --- |
| `E` | the `ExprIR` tree under the unary level (`paren`/`other` are any other shape) |
| `level9Main`, `level9PR` | cypher.rs:2547-2580 on main / on the PR |
| `rustNegI` | `-i` on `i64` in release (`[profile.release]` has no `overflow-checks`: wraps; debug panics) |
| `negF` | `-f` on `f64` (`Neg for f64`: flips the sign bit, NaN payload kept) |
| `evalE` | `ExprEval` on `Constant` / `Negate` (eval.rs:895-907, `checkedNeg` from proofs/expr_semantics) |
| `evalParam` | `evaluate_param`'s `Negate` (eval.rs:2277-2286) |

**`fold_eq_eval`**: for every tree, the PR's level-9 result evaluates to what main's does
(`Except`-equal: same value — float bits included — or same error), under both evaluators.
**`fold_never_min`**: the folding branch never meets `i64::MIN` (case 1 is tried first), so
the wrapping `-i` is never at its overflow point and debug builds cannot panic there.
-/

namespace Pr3437.NegFold
open FalkorCore
open FalkorExpr (checkedNeg neg_eq_sub_zero)

/-- `-f` on the float model: the sign bit flips; a NaN keeps its payload. (A non-integral
value's abstract tie-order `r` is kept; only the operation's *identity* matters below.) -/
def negF : F → F
  | .nan s p => .nan (!s) p
  | .negZero => .num 0
  | .num n => if n = 0 then .negZero else .num (-n)
  | .frac lo r => .frac (-lo - 1) r
  | .negInf => .posInf
  | .posInf => .negInf

/-- `-(0.0)` is `-0.0`, which `to_bits` tells apart from `+0.0`. -/
theorem negF_zero : negF (.num 0) = .negZero ∧ bits64 (negF (.num 0)) ≠ bits64 (.num 0) := by
  refine ⟨rfl, by decide⟩

/-- `-(-0.0)` is `+0.0`; NaN keeps its payload with the sign flipped. -/
theorem negF_examples (s : Bool) (p : Nat) :
    negF .negZero = .num 0 ∧ negF (.nan s p) = .nan (!s) p := ⟨rfl, rfl⟩

/-- `-i` on `i64`, release build. -/
def rustNegI (i : Int) : Int := wSub 0 i

inductive E where
  | const (v : V F)
  | neg (e : E)
  | paren (e : E)
  | other (tag : Nat)

def overflowMsg : String := "Integer overflow '9223372036854775808'"

/-- main, cypher.rs:2547-2580 without the PR's branch. -/
def level9Main : E → Except String E
  | .neg (.const (.int i)) => if i = I64MIN then .ok (.const (.int I64MIN)) else .ok (.neg (.const (.int i)))
  | .const (.int i) => if i = I64MIN then .error overflowMsg else .ok (.const (.int i))
  | res => .ok res

/-- The PR, cypher.rs:2547-2580. -/
def level9PR : E → Except String E
  | .neg (.const (.int i)) =>
    if i = I64MIN then .ok (.const (.int I64MIN)) else .ok (.const (.int (rustNegI i)))
  | .neg (.const (.float f)) => .ok (.const (.float (negF f)))
  | .const (.int i) => if i = I64MIN then .error overflowMsg else .ok (.const (.int i))
  | res => .ok res

/-- `ExprEval` on constants and `Negate`; `evOther` evaluates every other shape. -/
def evalE (evOther : E → Except String (FalkorCore.V F)) : E → Except String (FalkorCore.V F)
  | .const v => .ok v
  | .neg e => evalE evOther e >>= fun v => match v with
    | .int i => match checkedNeg i with
      | some j => .ok (.int j)
      | none => .error "ArgumentError: integer overflow in unary minus"
    | .float f => .ok (.float (negF f))
    | .null => .ok .null
    | w => .error ("Type mismatch: expected Integer, Float, or Null but was " ++ w.name)
  | e => evOther e

/-- `evaluate_param` (eval.rs:2277-2286): a negated non-number is `Null`. -/
def evalParam (evOther : E → Except String (FalkorCore.V F)) : E → Except String (FalkorCore.V F)
  | .const v => .ok v
  | .neg e => evalParam evOther e >>= fun v => match v with
    | .int i => match checkedNeg i with
      | some j => .ok (.int j)
      | none => .error "ArgumentError: integer overflow in unary minus"
    | .float f => .ok (.float (negF f))
    | _ => .ok .null
  | e => evOther e

/-- Integer constants are `i64`s. -/
def WF : E → Prop
  | .const (.int i) => InI64 i
  | .neg e => WF e
  | .paren e => WF e
  | _ => True

theorem rustNegI_exact (i : Int) (hi : InI64 i) (hne : i ≠ I64MIN) :
    checkedNeg i = some (rustNegI i) := neg_eq_sub_zero hi hne

/-- **Claim 6.** Folding equals evaluating the `Negate`, for `ExprEval`. -/
theorem fold_eq_eval (evOther : E → Except String (FalkorCore.V F)) (res : E) (hwf : WF res) :
    (level9PR res >>= evalE evOther) = (level9Main res >>= evalE evOther) := by
  cases res with
  | neg e =>
    cases e with
    | const v =>
      cases v with
      | int i =>
        simp only [level9PR, level9Main]
        by_cases h : i = I64MIN
        · simp [h]
        · simp only [h, ite_false, bind, Except.bind, evalE]
          simp only [WF] at hwf
          rw [rustNegI_exact i hwf h]
      | float f => rfl
      | _ => rfl
    | _ => rfl
  | const v => cases v <;> rfl
  | _ => rfl

/-- … and for `evaluate_param` (parameters are parsed by the same parser). -/
theorem fold_eq_evalParam (evOther : E → Except String (FalkorCore.V F)) (res : E) (hwf : WF res) :
    (level9PR res >>= evalParam evOther) = (level9Main res >>= evalParam evOther) := by
  cases res with
  | neg e =>
    cases e with
    | const v =>
      cases v with
      | int i =>
        simp only [level9PR, level9Main]
        by_cases h : i = I64MIN
        · simp [h]
        · simp only [h, ite_false, bind, Except.bind, evalParam]
          simp only [WF] at hwf
          rw [rustNegI_exact i hwf h]
      | float f => rfl
      | _ => rfl
    | _ => rfl
  | const v => cases v <;> rfl
  | _ => rfl

/-- The fold never applies `-` to `i64::MIN`: `-9223372036854775808` takes case 1. -/
theorem fold_never_min : level9PR (.neg (.const (.int I64MIN)) : E) = .ok (.const (.int I64MIN)) ∧
    level9Main (.neg (.const (.int I64MIN)) : E) = .ok (.const (.int I64MIN)) := by
  simp [level9PR, level9Main]

/-- `-(-9223372036854775808)` keeps its `Paren`, is not folded, and raises at run time on
both branches (live: `ArgumentError: integer overflow in unary minus` on main and PR). -/
theorem paren_min_not_folded (evOther : E → Except String (FalkorCore.V F))
    (hp : evOther (.paren (.const (.int I64MIN))) = .ok (.int I64MIN)) :
    level9PR (.neg (.paren (.const (.int I64MIN))) : E) = level9Main (.neg (.paren (.const (.int I64MIN)))) ∧
    (level9PR (.neg (.paren (.const (.int I64MIN))) : E) >>= evalE evOther) =
      .error "ArgumentError: integer overflow in unary minus" := by
  refine ⟨rfl, ?_⟩
  simp [level9PR, bind, Except.bind, evalE, hp, checkedNeg]

/-- `-0.0` folds to the constant `-0.0` (bits `0x8000…`), which is what main computed. -/
theorem neg_zero_literal (evOther : E → Except String (FalkorCore.V F)) :
    level9PR (.neg (.const (.float (.num 0))) : E) = .ok (.const (.float .negZero)) ∧
    (level9Main (.neg (.const (.float (.num 0))) : E) >>= evalE evOther) = .ok (.float .negZero) :=
  ⟨rfl, rfl⟩

/-- A bare `2^63` is still refused, on both. -/
theorem bare_min_refused : level9PR (.const (.int I64MIN) : E) = .error overflowMsg ∧
    level9Main (.const (.int I64MIN) : E) = .error overflowMsg := by
  simp [level9PR, level9Main]

end Pr3437.NegFold
