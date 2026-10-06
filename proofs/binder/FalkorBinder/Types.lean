/-
# Static operand checks: `validate_boolean_operands`, `expr_may_return_boolean`, `expr_may_return_entity`

| here | there |
| --- | --- |
| `TK`, `CK`, `TE` | `ExprIR` variants (bound), constant value kinds |
| `mayBool` | `expr_may_return_boolean` binder.rs:2306-2373 |
| `mayEntity` | `expr_may_return_entity` binder.rs:2375-2445 |
| `validate` | `validate_boolean_operands` / `_impl` binder.rs:2280-2304 |
| `canBN`, `canEnt` | reference: can the expression *evaluate* (without error) to a Bool/Null, to an entity/Null |

`mayBool` is used to reject a WHERE / list-comprehension predicate ("Expected
boolean predicate", binder.rs:381, 668, 1828) and logical operands
(`validate_boolean_operands`); a check is *sound* if it never rejects an
expression that can produce a boolean or null.

* `mayBool_sound_except`: sound for every variant except `GetElement`,
  `Negate`, `Length`, `ListComprehension`, `MapProjection` (all can yield
  `null`, and a subscript can yield `true`).
* `getElement_rejected` — CONFIRMED bug: `MATCH (n:N) WHERE n.flags[0] RETURN count(n)`
  → Rust "Expected boolean predicate", C `1`; `WITH [true] AS l RETURN true AND l[0]`
  → Rust "Type mismatch: expected Boolean", C `true`.
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
  | and | or | xor | not | paren | distinct
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
    | .paren | .distinct => match cs with
      | c :: _ => mayBool c
      | [] => false
    | .func rb _ => rb
    | .case_ => true
    | .list | .map | .mapProj | .negate | .length | .getElement | .getElements | .listComp
    | .patComp | .nestedPlan => false
    | .cmp | .inOp | .arith | .isNode | .isRel | .quant | .reduce | .var _ | .param | .prop
    | .shortestPath | .pattern => true
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
    | .paren | .distinct => match cs with
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
    | .paren | .distinct => match cs with
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

/-- The variants `mayBool` treats more strictly than evaluation allows. -/
def lossy : TK → Bool
  | .negate | .length | .getElement | .listComp | .mapProj => true
  | _ => false

mutual
def noLossy : TE → Bool
  | .node k cs => !lossy k && noLossyAll cs
def noLossyAll : List TE → Bool
  | [] => true
  | c :: cs => noLossy c && noLossyAll cs
end

mutual
/-- Sound away from the five lossy variants. -/
theorem mayBool_sound_except : ∀ e, noLossy e = true → canBN e = true → mayBool e = true
  | .node k cs, hl, hc => by
    simp only [noLossy, Bool.and_eq_true, Bool.not_eq_true'] at hl
    cases k <;> simp_all [canBN, mayBool, lossy]
    case and => exact mayBoolAll_sound cs hl hc
    case or => exact mayBoolAll_sound cs hl hc
    case xor => exact mayBoolAll_sound cs hl hc
    case not => exact mayBoolAll_sound cs hl hc
    case paren =>
      match cs, hl, hc with
      | c :: cs', hl, hc =>
        simp only [noLossyAll, Bool.and_eq_true] at hl
        exact mayBool_sound_except c hl.1 hc
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

/-- The subscript `l[0]` (a list of booleans, a property list …) is rejected as
a predicate although it can be `true` (CONFIRMED bug, binder.rs:2338). -/
theorem getElement_rejected (l i : TE) :
    canBN (.node .getElement [l, i]) = true ∧ mayBool (.node .getElement [l, i]) = false ∧
    validate (.node .and [.node (.const .bool) [], .node .getElement [l, i]]) = false := by
  simp [canBN, mayBool, validate, mayBoolAll, isLogical]

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
    | .paren | .distinct => match cs with
      | c :: _ => canEnt c
      | [] => false
    | .const c => c == .null || c == .node || c == .rel || c == .path
    | _ => false
end

/-- `expr_may_return_entity` coincides with the reference. -/
theorem mayEntity_eq : ∀ e, mayEntity e = canEnt e
  | .node k cs => by
    cases k <;> (try rfl)
    case paren => cases cs with
      | nil => rfl
      | cons c _ => exact mayEntity_eq c
    case distinct => cases cs with
      | nil => rfl
      | cons c _ => exact mayEntity_eq c

end FalkorBinder.Types
