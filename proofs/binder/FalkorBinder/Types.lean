/-
# Static operand checks: `validate_boolean_operands`, `expr_may_return_boolean`, `expr_may_return_entity`

| here | there |
| --- | --- |
| `TK`, `CK`, `TE` | `ExprIR` variants (bound), constant value kinds |
| `mayBool` | `expr_may_return_boolean` binder.rs |
| `mayEntity` | `expr_may_return_entity` binder.rs |
| `validate` | `validate_boolean_operands` / `_impl` binder.rs |
| `canBN`, `canEnt` | reference: can the expression *evaluate* (without error) to a Bool/Null, to an entity/Null |

`TE` is a *bound* tree, and no bound tree holds a `Paren`
(`BindExpr.bind_paren`; `BE` has no such constructor): both checks hit
`unreachable!` on one, so `TK` has no `paren` and
`Distinct` is the one transparent wrapper.

`mayBool` is used to reject a WHERE / list-comprehension predicate ("Expected
boolean predicate", binder.rs) and logical operands
(`validate_boolean_operands`); a check is *sound* if it never rejects an
expression that can produce a boolean or null.

* `mayBool_sound_except`: sound for every variant except `Negate`, `Length`,
  `ListComprehension`, `MapProjection` (all can yield `null`).
* `getElement_accepted` — `GetElement` is in the "runtime-typed → true" arm:
  `MATCH (n:N) WHERE n.flags[0] RETURN count(n)` and `WITH [true] AS l RETURN true AND l[0]`
  bind, as in C. The earlier rejection is kept as `pre3270_getElement_rejected` (historical).
* `arith_accepted`: `Add`/`Sub`/… are accepted as possibly boolean (they can
  be null), so `RETURN false AND (1+2)` returns `false` (runtime short-circuit)
  where C rejects it statically ("Type mismatch") — deviation, not a wrong answer.
* `mayEntity_sound`: `expr_may_return_entity` is sound w.r.t. `canEnt`.
* `validate_iff`: the validator accepts iff every operand of every
  AND/OR/XOR/NOT may return a boolean.
-/
namespace FalkorBinder.Types

inductive CK | null | bool | int | float | str | list | map | node | rel | path | other
  deriving DecidableEq, Repr

inductive TK
  | and | or | xor | not | distinct
  | func (retBool retEntity : Bool)
  | case_
  | list | map | mapProj | negate | length | getElement | getElements | listComp | patComp | nestedPlan
  | cmp | inOp | arith | isNode | isRel | quant | reduce
  | var (entity : Bool) | param | prop | shortestPath | pattern
  | const (c : CK)
  | regex
  deriving DecidableEq, Repr

inductive TE
  | node (k : TK) (cs : List TE)
  deriving Repr

mutual
def mayBool : TE → Bool
  | .node k cs => match k with
    | .or | .and | .xor | .not => mayBoolAll cs
    | .distinct => match cs with
      | c :: _ => mayBool c
      | [] => false
    | .func rb _ => rb
    | .case_ => true
    | .list | .map | .mapProj | .negate | .length | .getElements | .listComp
    | .patComp | .nestedPlan => false
    | .cmp | .inOp | .arith | .isNode | .isRel | .quant | .reduce | .var _ | .param | .prop
    | .getElement | .shortestPath | .pattern => true
    | .const c => c == .bool || c == .null
    | .regex => true
def mayBoolAll : List TE → Bool
  | [] => true
  | c :: cs => mayBool c && mayBoolAll cs
end

mutual
def mayEntity : TE → Bool
  | .node k cs => match k with
    | .var e => e
    | .func _ re => re
    | .case_ => true
    | .distinct => match cs with
      | c :: _ => mayEntity c
      | [] => false
    | .getElement | .prop | .param | .reduce => true
    | .const c => c == .null || c == .node || c == .rel || c == .path
    | _ => false
end

def isLogical : TK → Bool
  | .and | .or | .xor | .not => true
  | _ => false

mutual
/-- `validate_boolean_operands_impl`: `true` = `Ok(())`. -/
def validate : TE → Bool
  | .node k cs => (!isLogical k || mayBoolAll cs) && validateAll cs
def validateAll : List TE → Bool
  | [] => true
  | c :: cs => validate c && validateAll cs
end

/-! ## Reference: what an expression can evaluate to -/

mutual
/-- Can evaluate, without an error, to `true`/`false`/`null`. Function results,
variables, parameters, properties: as declared (`func`'s flag; anything for the
others). Logical operators need every operand to be able to. -/
def canBN : TE → Bool
  | .node k cs => match k with
    | .or | .and | .xor | .not => canBNAll cs
    | .distinct => match cs with
      | c :: _ => canBN c
      | [] => false
    | .func rb _ => rb
    | .case_ | .cmp | .inOp | .isNode | .isRel | .quant | .reduce | .var _ | .param | .prop
    | .shortestPath | .pattern | .regex => true
    -- `null + 1`, `-null`, `size(null)`, `[x IN null | x]`, `null{.a}` are null; `l[0]` is anything
    | .arith | .negate | .length | .getElement | .listComp | .mapProj => true
    | .list | .map | .getElements | .patComp | .nestedPlan => false
    | .const c => c == .bool || c == .null
def canBNAll : List TE → Bool
  | [] => true
  | c :: cs => canBN c && canBNAll cs
end

/-- The variants `mayBool` treats more strictly than evaluation allows (`getElement` left this
list with #3270). -/
def lossy : TK → Bool
  | .negate | .length | .listComp | .mapProj => true
  | _ => false

mutual
def noLossy : TE → Bool
  | .node k cs => !lossy k && noLossyAll cs
def noLossyAll : List TE → Bool
  | [] => true
  | c :: cs => noLossy c && noLossyAll cs
end

mutual
/-- Sound away from the four lossy variants. -/
theorem mayBool_sound_except : ∀ e, noLossy e = true → canBN e = true → mayBool e = true
  | .node k cs, hl, hc => by
    simp only [noLossy, Bool.and_eq_true, Bool.not_eq_true'] at hl
    cases k <;> simp_all [canBN, mayBool, lossy]
    case and => exact mayBoolAll_sound cs hl hc
    case or => exact mayBoolAll_sound cs hl hc
    case xor => exact mayBoolAll_sound cs hl hc
    case not => exact mayBoolAll_sound cs hl hc
    case distinct =>
      match cs, hl, hc with
      | c :: cs', hl, hc =>
        simp only [noLossyAll, Bool.and_eq_true] at hl
        exact mayBool_sound_except c hl.1 hc
theorem mayBoolAll_sound : ∀ l, noLossyAll l = true → canBNAll l = true → mayBoolAll l = true
  | [], _, _ => rfl
  | c :: cs, hl, hc => by
    simp only [noLossyAll, canBNAll, mayBoolAll, Bool.and_eq_true] at *
    exact ⟨mayBool_sound_except c hl.1 hc.1, mayBoolAll_sound cs hl.2 hc.2⟩
end

/-- #3270: the subscript `l[0]` (a list of booleans, a property list …) can be `true`, and is now
accepted as a predicate and as a logical operand (the `GetElement` arm of the "runtime-typed → true"
list, binder.rs). -/
theorem getElement_accepted (l i : TE) :
    canBN (.node .getElement [l, i]) = true ∧ mayBool (.node .getElement [l, i]) = true ∧
    mayBoolAll [.node (.const .bool) [], .node .getElement [l, i]] = true := by
  simp [canBN, mayBool, mayBoolAll]

/-- `true AND l[0]` over a literal list and index validates. -/
theorem getElement_and_validates :
    validate (.node .and [.node (.const .bool) [],
      .node .getElement [.node .list [], .node (.const .int) []]]) = true := by
  decide

/-- `expr_may_return_boolean` before #3270 (3643a301c): `GetElement` was listed with the
never-boolean kinds. Historical, root only (the inner recursion of the old function is `mayBool`
with this one arm changed). -/
def mayBoolPre3270 : TE → Bool
  | .node .getElement _ => false
  | e => mayBool e

/-- Fixed by #3270: `MATCH (n:N) WHERE n.flags[0] RETURN count(n)` was rejected ("Expected boolean
predicate") although the subscript can be `true`; C returned `1`. -/
theorem pre3270_getElement_rejected (l i : TE) :
    canBN (.node .getElement [l, i]) = true ∧ mayBoolPre3270 (.node .getElement [l, i]) = false ∧
    mayBool (.node .getElement [l, i]) = true := by
  simp [canBN, mayBoolPre3270, mayBool]

/-- `-null` and `size(null)` are null, a valid operand, but rejected; C accepts
`RETURN true AND -null`. -/
theorem negate_rejected (x : TE) : canBN (.node .negate [x]) = true ∧ mayBool (.node .negate [x]) = false :=
  ⟨rfl, rfl⟩

/-- Arithmetic is accepted (it may be null); C rejects `false AND (1+2)` statically. -/
theorem arith_accepted (a b : TE) : mayBool (.node .arith [a, b]) = true := rfl

mutual
theorem validate_iff_aux : ∀ e, validate e = true ↔
    ((match e with | .node k cs => !isLogical k || mayBoolAll cs) = true ∧
      (match e with | .node _ cs => validateAll cs) = true)
  | .node k cs => by simp [validate]
end

/-- Accepted iff every logical node's operands may return a boolean, recursively. -/
theorem validate_logical (k : TK) (cs : List TE) (hk : isLogical k = true) :
    validate (.node k cs) = (mayBoolAll cs && validateAll cs) := by
  simp [validate, hk]

theorem validate_nonlogical (k : TK) (cs : List TE) (hk : isLogical k = false) :
    validate (.node k cs) = validateAll cs := by
  simp [validate, hk]

/-! ## Entities (DELETE operands) -/

mutual
/-- Can evaluate to a node / relationship / path / null. -/
def canEnt : TE → Bool
  | .node k cs => match k with
    | .var e => e
    | .func _ re => re
    | .case_ | .getElement | .prop | .param | .reduce => true
    | .distinct => match cs with
      | c :: _ => canEnt c
      | [] => false
    | .const c => c == .null || c == .node || c == .rel || c == .path
    | _ => false
end

/-- `expr_may_return_entity` coincides with the reference. -/
theorem mayEntity_eq : ∀ e, mayEntity e = canEnt e
  | .node k cs => by
    cases k <;> (try rfl)
    case distinct => cases cs with
      | nil => rfl
      | cons c _ => exact mayEntity_eq c

end FalkorBinder.Types
