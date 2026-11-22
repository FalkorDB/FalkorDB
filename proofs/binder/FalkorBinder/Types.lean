/-
# Static operand checks: `validate_boolean_operands`, `expr_may_return_boolean`, `expr_may_return_entity`

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
  bind, as in C.
* `arith_accepted`: `Add`/`Sub`/… are accepted by `mayBool` as possibly boolean
  (they can be null). Since #3605 the operand check also rejects arithmetic over
  constants whose result is known (`nonBool`), so `RETURN false AND (1+2)` is a
  "Type mismatch" as in C (`arith_const_rejected`).
* `mayEntity_sound`: `expr_may_return_entity` is sound w.r.t. `canEnt`.
* `validate_logical` / `validate_nonlogical`: outside a comprehension or
  quantifier predicate, a logical node accepts iff every operand passes
  `operandOk` (no statically known non-Boolean/Null type; projection: `mayBool`;
  ORDER BY: a call's declared return type), recursively; inside such a predicate
  the operands are left to the runtime (#3293, `filter_operands_deferred`).
* `nonBool_sound`, `orderBy_operand_sound`, `operandOk_sound_except`: the checks
  never reject an operand that can produce a boolean or null (projection: the
  same `lossy` exceptions as `mayBool`).
-/
namespace FalkorBinder.Types

inductive CK | null | bool | int | float | str | list | map | node | rel | path | other
  deriving DecidableEq, Repr

inductive TK
  | and | or | xor | not | distinct
  | func (retBool retEntity : Bool)
  | case_
  /-- `cv`: the kind of `constant_value` (`Negate`) / `constant_arithmetic`
  (`Add`…`Modulo`) — `some` when every leaf is a constant and evaluation
  succeeds (binder.rs); the evaluation itself is not modelled. -/
  | list | map | mapProj | negate (cv : Option CK) | length | getElement | getElements | listComp | patComp | nestedPlan
  | cmp | inOp | arith (cv : Option CK) | isNode | isRel | quant | reduce
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
    | .list | .map | .mapProj | .negate _ | .length | .getElements | .listComp
    | .patComp | .nestedPlan => false
    | .cmp | .inOp | .arith _ | .isNode | .isRel | .quant | .reduce | .var _ | .param | .prop
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

/-- `non_boolean_operand_type` (binder.rs): `true` = `Some(_)`, the operand's
type is known at bind time to be neither Boolean nor Null. -/
def nonBool : TE → Bool
  | .node k _ => match k with
    | .list | .patComp | .map => true
    | .const c | .negate (some c) | .arith (some c) => !(c == .bool || c == .null)
    | _ => false

/-- The per-operand checks of a logical node in `validate_boolean_operands_impl`
(binder.rs), in order: `non_boolean_operand_type`; without `check_shape` (ORDER BY) a
call whose declared return type `!can_return_boolean()`; with it, `expr_may_return_boolean`. -/
def operandOk (shape : Bool) (c : TE) : Bool :=
  !nonBool c &&
  (shape || match c with | .node (.func rb _) _ => rb | _ => true) &&
  (!shape || mayBool c)

def operandsOk (shape : Bool) : List TE → Bool
  | [] => true
  | c :: cs => operandOk shape c && operandsOk shape cs

/-- `ListComprehension` / `Quantifier`: child 1 is the predicate. -/
def predScope : TK → Bool
  | .listComp | .quant => true
  | _ => false

mutual
/-- `validate_boolean_operands_impl(node, in_filter, check_shape)`: `true` = `Ok(())`. -/
def validate (inF shape : Bool) : TE → Bool
  | .node k cs => (!isLogical k || inF || operandsOk shape cs) &&
      validateAll (isLogical k && inF) (predScope k) shape 0 cs
/-- The children loop: `child_in_filter = (logical && in_filter) || (predicate_scope && i == 1)`. -/
def validateAll (lf ps shape : Bool) : Nat → List TE → Bool
  | _, [] => true
  | i, c :: cs => validate (lf || (ps && i == 1)) shape c && validateAll lf ps shape (i + 1) cs
end

/-- `validate_boolean_operands` (projection items). -/
def validateProj (e : TE) : Bool := validate false true e
/-- `validate_order_by_boolean_operands` (ORDER BY, #3605). -/
def validateOrderBy (e : TE) : Bool := validate false false e

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
    | .length | .getElement | .listComp | .mapProj => true
    -- over constants the value, hence its kind, is known
    | .arith cv | .negate cv => match cv with
      | none => true
      | some c => c == .bool || c == .null
    | .list | .map | .getElements | .patComp | .nestedPlan => false
    | .const c => c == .bool || c == .null
def canBNAll : List TE → Bool
  | [] => true
  | c :: cs => canBN c && canBNAll cs
end

/-- The variants `mayBool` treats more strictly than evaluation allows (`getElement` left this
list with #3270). -/
def lossy : TK → Bool
  | .negate _ | .length | .listComp | .mapProj => true
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
    validateProj (.node .and [.node (.const .bool) [],
      .node .getElement [.node .list [], .node (.const .int) []]]) = true := by
  decide

/-- `-null` and `size(null)` are null, a valid operand, but rejected; C accepts
`RETURN true AND -null`. -/
theorem negate_rejected (cv : Option CK) (hcv : cv = none ∨ cv = some .null) (x : TE) :
    canBN (.node (.negate cv) [x]) = true ∧ mayBool (.node (.negate cv) [x]) = false := by
  rcases hcv with rfl | rfl <;> exact ⟨rfl, rfl⟩

/-- W4-plan-2 is still open: `RETURN true AND -null` is rejected in a projection. -/
theorem true_and_neg_null_rejected :
    validateProj (.node .and [.node (.const .bool) [],
      .node (.negate (some .null)) [.node (.const .null) []]]) = false := by decide

/-- Arithmetic is accepted by `mayBool` (it may be null). -/
theorem arith_accepted (cv : Option CK) (a b : TE) : mayBool (.node (.arith cv) [a, b]) = true := rfl

/-- #3605: `RETURN false AND (1+2)` — constant arithmetic with an Integer result
is rejected statically, as in C. -/
theorem arith_const_rejected :
    validateProj (.node .and [.node (.const .bool) [],
      .node (.arith (some .int)) [.node (.const .int) [], .node (.const .int) []]]) = false := by
  decide

/-- Arithmetic over a runtime value (`false AND (x+1)`) stays runtime-checked. -/
theorem arith_runtime_accepted (a b : TE) (ha : validate false true a = true)
    (hb : validate false true b = true) :
    validateProj (.node .and [.node (.const .bool) [], .node (.arith none) [a, b]]) = true := by
  simp [validateProj, validate, validateAll, isLogical, predScope, operandsOk, operandOk,
    nonBool, mayBool, ha, hb]

/-- #3293: inside a list-comprehension predicate a logical node's operands are left
to the runtime: `RETURN [x IN [] WHERE false AND 1]` binds (C returns `[]`). -/
theorem filter_operands_deferred :
    validateProj (.node .listComp [.node .list [],
      .node .and [.node (.const .bool) [], .node (.const .int) []],
      .node (.var false) []]) = true := by decide

/-- … but the projection of a list comprehension (child 2) is a value context:
`RETURN [x IN [] | false AND 1]` is still rejected. -/
theorem listComp_projection_checked :
    validateProj (.node .listComp [.node .list [], .node (.const .bool) [],
      .node .and [.node (.const .bool) [], .node (.const .int) []]]) = false := by decide

/-- A non-logical node inside a predicate is a value context again:
`[x IN [] WHERE x = (false AND 1)]` is rejected. -/
theorem filter_value_context_checked :
    validateProj (.node .listComp [.node .list [],
      .node .cmp [.node (.var false) [],
        .node .and [.node (.const .bool) [], .node (.const .int) []]],
      .node (.var false) []]) = false := by decide

/-- #3605 ORDER BY: `ORDER BY true AND 1` is rejected; `ORDER BY true AND [x IN l | x]`
(runtime-typed, may be null) is not, where the projection's shape check rejects it. -/
theorem orderBy_checks :
    validateOrderBy (.node .and [.node (.const .bool) [], .node (.const .int) []]) = false ∧
    validateOrderBy (.node .and [.node (.const .bool) [],
      .node .listComp [.node (.var false) [], .node (.const .bool) [], .node (.var false) []]]) = true ∧
    validateProj (.node .and [.node (.const .bool) [],
      .node .listComp [.node (.var false) [], .node (.const .bool) [], .node (.var false) []]]) = false := by
  decide

/-- ORDER BY rejects a call whose declared return type cannot be Boolean or Null. -/
theorem orderBy_func_rejected (re : Bool) (cs : List TE) :
    validateOrderBy (.node .and [.node (.const .bool) [], .node (.func false re) cs]) = false := by
  simp [validateOrderBy, validate, isLogical, operandsOk, operandOk, nonBool]

/-- `nonBool` is sound: an operand it flags can never be a boolean or null. -/
theorem nonBool_sound (c : TE) (h : nonBool c = true) : canBN c = false := by
  match c with
  | .node k cs =>
    cases k <;> simp_all [nonBool, canBN]
    case negate cv => cases cv <;> simp_all
    case arith cv => cases cv <;> simp_all

/-- The ORDER BY operand check never rejects an operand that can be a boolean or null. -/
theorem orderBy_operand_sound (c : TE) (h : canBN c = true) : operandOk false c = true := by
  have hn : nonBool c = false := by
    cases hb : nonBool c
    · rfl
    · rw [nonBool_sound c hb] at h; exact absurd h (by decide)
  match c, h with
  | .node k cs, h =>
    cases k <;> simp_all [operandOk, canBN]

/-- The projection operand check is sound away from the `lossy` variants. -/
theorem operandOk_sound_except (c : TE) (hl : noLossy c = true) (h : canBN c = true) :
    operandOk true c = true := by
  have hn : nonBool c = false := by
    cases hb : nonBool c
    · rfl
    · rw [nonBool_sound c hb] at h; exact absurd h (by decide)
  simp [operandOk, hn, mayBool_sound_except c hl h]

/-- Where it differs from `mayBool` alone (the pre-#3605 check), the projection operand
check only adds constant arithmetic with a known non-Boolean/Null result. -/
theorem operandOk_proj (c : TE) :
    operandOk true c = (mayBool c && !nonBool c) := by
  simp [operandOk, Bool.and_comm]

theorem nonBool_mayBool (c : TE) (h1 : nonBool c = true) (h2 : mayBool c = true) :
    ∃ k cs, c = .node (.arith (some k)) cs := by
  match c with
  | .node k cs =>
    cases k <;> simp_all [nonBool, mayBool]
    case arith cv => cases cv <;> simp_all

/-- A logical node outside a predicate: accepted iff every operand passes
`operandOk` and every child validates (none of them in a filter). -/
theorem validate_logical (k : TK) (cs : List TE) (shape : Bool) (hk : isLogical k = true) :
    validate false shape (.node k cs) =
      (operandsOk shape cs && validateAll false (predScope k) shape 0 cs) := by
  simp [validate, hk]

/-- A logical node inside a predicate: its operands are not checked, its children
stay in the filter. -/
theorem validate_logical_filter (k : TK) (cs : List TE) (shape : Bool) (hk : isLogical k = true) :
    validate true shape (.node k cs) = validateAll true (predScope k) shape 0 cs := by
  simp [validate, hk]

theorem validate_nonlogical (k : TK) (cs : List TE) (inF shape : Bool) (hk : isLogical k = false) :
    validate inF shape (.node k cs) = validateAll false (predScope k) shape 0 cs := by
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
