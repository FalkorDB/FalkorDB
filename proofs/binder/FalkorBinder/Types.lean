/-
# Static operand checks: `validate_boolean_operands`, `expr_may_return_boolean`, `expr_may_return_entity`

| here | there |
| --- | --- |
| `TK`, `CK`, `TE` | `ExprIR` variants (bound), constant value kinds |
| `mayBool` | `expr_may_return_boolean` binder.rs:2323-2389 |
| `mayEntity` | `expr_may_return_entity` binder.rs:2392-2457 |
| `validateF`, `validateCh` | `validate_boolean_operands_impl(node, in_filter, check_shape)` binder.rs:2325-2374 |
| `validate`, `validateOB` | `validate_boolean_operands` :2309 (`false, true`), `validate_order_by_boolean_operands` :2317 (`false, false`) |
| `nbType`, `constVal`/`constAll`/`mapAll`, `CEval` | `non_boolean_operand_type` :2379, `constant_value` :2432, `constant_arithmetic` :2472 (`Value` arithmetic by kind: a parameter) |
| `FT`, `FT.canBool`, `FT.typeName` | `functions::Type`, `can_return_boolean`, `type_name` :2408 |
| `canBN`, `canEnt` | reference: can the expression *evaluate* (without error) to a Bool/Null, to an entity/Null |

`TE` is a *bound* tree, and since #3335 no bound tree holds a `Paren`
(`BindExpr.bind_paren`; `BE` has no such constructor): both checks now hit
`unreachable!` on one (binder.rs:2334, 2408), so `TK` has no `paren` and
`Distinct` is the one transparent wrapper.

`mayBool` is used to reject a WHERE / list-comprehension predicate ("Expected
boolean predicate", binder.rs:352, 662, 1259, 1827) and logical operands
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
* #3293 (`validate_logical`, `validate_nonlogical`, `validate_predScope`,
  `validate_inFilter_logical`): outside a filter every operand of an
  AND/OR/XOR/NOT must pass `opOk` (no known non-boolean type — `nbType_notBN`,
  `nbType_sound` — plus `mayBool` in projections or a call's declared type in
  ORDER BY); a comprehension/quantifier predicate is a filter whose logical
  operands are left to the runtime (`filter_operands_unchecked`,
  `quant_operands_unchecked`); `arith_const_rejected`, `arith_runtime_accepted`,
  `projection_int_rejected`, `orderBy_getElement_accepted`, `orderBy_func_rejected`,
  `opOk_shape_stronger`; `typeName_not_bool`.
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

/-- `ListComprehension` / `Quantifier`: child 1 is a filter predicate. -/
def predScope : TK → Bool
  | .listComp | .quant => true
  | _ => false

/-- What `constant_value` / `constant_arithmetic` (binder.rs:2432-2493) need
from `Value`'s arithmetic, at the level of kinds. `CK` forgets the number, so
whether `i.checked_neg()` succeeds and the result of `a + b`, `a / b`, … (or
its failure: overflow, division by zero, a type error) are parameters. -/
structure CEval where
  /-- the kind of `try_fold(first, op)` over the operands' kinds; `none` = `Err` -/
  ar : List CK → Option CK
  /-- `i.checked_neg()` is `Some` -/
  negOk : Bool

variable (E : CEval)

/-- The `Negate` arm of `constant_value` (binder.rs:2457-2462). -/
def negK : CK → Option CK
  | .int => if E.negOk then some .int else none
  | .float => some .float
  | .null => some .null
  | _ => none

mutual
/-- `constant_value` (binder.rs:2432-2468), by kind; `constant_arithmetic`
(binder.rs:2472-2493) is its `arith` arm: every operand constant, at least one. -/
def constVal : TE → Option CK
  | .node k cs => match k with
    | .const c => some c
    | .list => (constAll cs).map fun _ => .list
    | .map => (mapAll cs).map fun _ => .map
    | .negate => match cs with
      | c :: _ => (constVal c).bind (negK E)
      | [] => none
    | .arith => match constAll cs with
      | some (k0 :: ks) => E.ar (k0 :: ks)
      | _ => none
    | _ => none
def constAll : List TE → Option (List CK)
  | [] => some []
  | c :: cs => match constVal c, constAll cs with
    | some k, some ks => some (k :: ks)
    | _, _ => none
/-- Map entries are `Constant(String)` key nodes holding the value as child 0. -/
def mapAll : List TE → Option Unit
  | [] => some ()
  | .node k vs :: cs => match k, vs with
    | .const .str, v :: _ => match constVal v, mapAll cs with
      | some _, some _ => some ()
      | _, _ => none
    | _, _ => none
end

def notBN (c : CK) : Bool := !(c == .bool || c == .null)

@[simp] theorem notBN_bool : notBN .bool = false := rfl
@[simp] theorem notBN_null : notBN .null = false := rfl

/-- `non_boolean_operand_type` (binder.rs:2379-2404): the kind of an operand
known at bind time to be neither Boolean nor Null. -/
def nbType : TE → Option CK
  | .node k cs => match k with
    | .list | .patComp => some .list
    | .map => some .map
    | .const c => if notBN c then some c else none
    | .negate | .arith => match constVal E (.node k cs) with
      | some c => if notBN c then some c else none
      | none => none
    | _ => none

/-- The checks on one operand of a logical node outside a filter
(binder.rs:2336-2361): no known non-boolean type; with `check_shape`
`expr_may_return_boolean`, without it the declared return type of a call. -/
def opOk (shape : Bool) (c : TE) : Bool :=
  (nbType E c).isNone &&
  (shape || match c with | .node (.func rb _) _ => rb | _ => true) &&
  (!shape || mayBool c)

mutual
/-- `validate_boolean_operands_impl(node, in_filter, check_shape)`
(binder.rs:2325-2374): `true` = `Ok(())`. -/
def validateF (inF shape : Bool) : TE → Bool
  | .node k cs =>
    (!(isLogical k && !inF) || cs.all (opOk E shape)) &&
    validateCh (isLogical k && inF) (predScope k) shape 0 cs
/-- The children loop: child `i` is in a filter iff the node is a logical node
in a filter, or `i = 1` under a comprehension/quantifier. -/
def validateCh (lf ps shape : Bool) : Nat → List TE → Bool
  | _, [] => true
  | i, c :: cs => validateF (lf || (ps && i == 1)) shape c && validateCh lf ps shape (i + 1) cs
end

/-- `validate_boolean_operands` (binder.rs:2309-2311): projections, SET values, … -/
def validate (e : TE) : Bool := validateF E false true e

/-- `validate_order_by_boolean_operands` (binder.rs:2317-2319). -/
def validateOB (e : TE) : Bool := validateF E false false e

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
a predicate although it can be `true` (CONFIRMED bug, the `GetElement` arm of the non-boolean list, binder.rs:2552). -/
theorem getElement_rejected (l i : TE) :
    canBN (.node .getElement [l, i]) = true ∧ mayBool (.node .getElement [l, i]) = false ∧
    validate E (.node .and [.node (.const .bool) [], .node .getElement [l, i]]) = false := by
  simp [canBN, mayBool, validate, validateF, isLogical, opOk, nbType]

/-- `-null` and `size(null)` are null, a valid operand, but rejected; C accepts
`RETURN true AND -null`. -/
theorem negate_rejected (x : TE) : canBN (.node .negate [x]) = true ∧ mayBool (.node .negate [x]) = false :=
  ⟨rfl, rfl⟩

/-- Arithmetic is accepted by `expr_may_return_boolean` (it may be null). -/
theorem arith_accepted (a b : TE) : mayBool (.node .arith [a, b]) = true := rfl

/-- A logical node outside a filter: every operand passes `opOk`, and every
child is validated outside a filter (binder.rs:2336-2361, 2368-2372). -/
theorem validate_logical (k : TK) (cs : List TE) (shape : Bool) (hk : isLogical k = true) :
    validateF E false shape (.node k cs) =
      (cs.all (opOk E shape) && validateCh E false false shape 0 cs) := by
  cases k <;> simp_all [validateF, isLogical, predScope]

/-- Any other node outside a comprehension/quantifier checks nothing itself and
validates its children outside a filter. -/
theorem validate_nonlogical (k : TK) (cs : List TE) (inF shape : Bool)
    (hk : isLogical k = false) (hp : predScope k = false) :
    validateF E inF shape (.node k cs) = validateCh E false false shape 0 cs := by
  simp [validateF, hk, hp]

theorem validateCh_value (shape : Bool) : ∀ (i : Nat) (cs : List TE),
    validateCh E false false shape i cs = cs.all (validateF E false shape)
  | _, [] => by simp [validateCh]
  | i, c :: cs => by simp [validateCh, validateCh_value shape (i + 1) cs]

/-- A comprehension / quantifier: child 1 (the predicate) is validated as a
filter, the list and the rest as values (binder.rs:2363-2371, #3293). -/
theorem validate_predScope (k : TK) (a p : TE) (rest : List TE) (inF shape : Bool)
    (hp : predScope k = true) :
    validateF E inF shape (.node k (a :: p :: rest)) =
      (validateF E false shape a && validateF E true shape p &&
        rest.all (validateF E false shape)) := by
  have hl : isLogical k = false := by cases k <;> simp_all [predScope, isLogical]
  have hr : validateCh E false true shape 2 rest = rest.all (validateF E false shape) := by
    clear hp hl
    suffices h : ∀ i, 2 ≤ i → validateCh E false true shape i rest = rest.all (validateF E false shape) from h 2 (Nat.le_refl _)
    induction rest with
    | nil => intro i _; simp [validateCh]
    | cons c cs ih =>
      intro i hi
      have : (i == 1) = false := by simp; omega
      simp [validateCh, this, ih (i + 1) (by omega)]
  simp [validateF, hl, hp, validateCh, hr, Bool.and_assoc]

/-- Inside a filter, a logical node's operands are left to the runtime (which
short-circuits as C's filter tree does) and stay in the filter. -/
theorem validate_inFilter_logical (k : TK) (cs : List TE) (shape : Bool) (hk : isLogical k = true) :
    validateF E true shape (.node k cs) = validateCh E true false shape 0 cs := by
  cases k <;> simp_all [validateF, isLogical, predScope]

/-- A known non-boolean type is never Boolean or Null. -/
theorem nbType_notBN (e : TE) (c : CK) (h : nbType E e = some c) : c ≠ .bool ∧ c ≠ .null := by
  obtain ⟨k, cs⟩ := e
  cases k <;> simp only [nbType] at h <;> (try split at h) <;> (try split at h) <;>
    simp_all [notBN] <;> (try (cases c <;> simp_all))

/-- Away from constant `Negate` / arithmetic (whose kind comes from `Value`'s
arithmetic), the new operand check rejects only operands that cannot
evaluate to a Boolean or Null: it is sound w.r.t. `canBN`. -/
theorem nbType_sound (k : TK) (cs : List TE) (ha : k ≠ .arith) (hn : k ≠ .negate)
    (h : canBN (.node k cs) = true) : nbType E (.node k cs) = none := by
  cases k <;> simp_all [nbType, canBN, notBN]
  rcases h with h | h <;> simp [h]

/-- #3293: `[x IN l WHERE false AND 1]` is accepted (its predicate is a filter,
`false AND 1` short-circuits at runtime as in C). Before the PR the
operand `1` was rejected ("Type mismatch: expected Boolean"). -/
theorem filter_operands_unchecked :
    validate E (.node .listComp [.node (.var false) [], .node .and
      [.node (.const .bool) [], .node (.const .int) []]]) = true := by
  simp [validate, validateF, validateCh, isLogical, predScope, nbType, opOk, mayBool]

/-- … and a quantifier predicate likewise: `any(x IN l WHERE false AND 1)`. -/
theorem quant_operands_unchecked :
    validate E (.node .quant [.node (.var false) [], .node .and
      [.node (.const .bool) [], .node (.const .int) []]]) = true := by
  simp [validate, validateF, validateCh, isLogical, predScope, nbType, opOk, mayBool]

/-- In a projection a constant integer operand is still rejected:
`RETURN false AND 1`. -/
theorem projection_int_rejected :
    validate E (.node .and [.node (.const .bool) [], .node (.const .int) []]) = false := by
  simp [validate, validateF, isLogical, nbType, opOk, notBN]

/-- #3293: constant arithmetic of known type is rejected in a projection
(`RETURN false AND (1+2)`, as C), though `expr_may_return_boolean` accepts it
(`arith_accepted`). -/
theorem arith_const_rejected (h : E.ar [.int, .int] = some .int) :
    validate E (.node .and [.node (.const .bool) [], .node .arith
      [.node (.const .int) [], .node (.const .int) []]]) = false := by
  simp [validate, validateF, isLogical, nbType, opOk, notBN, constVal, constAll, h]

/-- Arithmetic with a runtime operand (`false AND (x + 1)`) stays unchecked. -/
theorem arith_runtime_accepted :
    validate E (.node .and [.node (.const .bool) [], .node .arith
      [.node (.var false) [], .node (.const .int) []]]) = true := by
  simp [validate, validateF, validateCh, isLogical, predScope, nbType, opOk, notBN, constVal,
    constAll, mayBool]

/-- ORDER BY skips the shape check, so the subscript bug does not apply there:
`ORDER BY true AND l[0]` is accepted. -/
theorem orderBy_getElement_accepted :
    validateOB E (.node .and [.node (.const .bool) [], .node .getElement
      [.node (.var false) [], .node (.const .int) []]]) = true := by
  simp [validateOB, validateF, validateCh, isLogical, predScope, nbType, opOk]

/-- … but a call that cannot return a Boolean is rejected there
(`ORDER BY true AND toString(1)`), and so is a literal (`ORDER BY true AND 1`). -/
theorem orderBy_func_rejected (cs : List TE) :
    validateOB E (.node .and [.node (.const .bool) [], .node (.func false false) cs]) = false ∧
    validateOB E (.node .and [.node (.const .bool) [], .node (.const .int) []]) = false := by
  simp [validateOB, validateF, isLogical, nbType, opOk, notBN]

/-- In a projection the ORDER-BY checks are implied: whatever `validateOB`'s
operand check rejects, `validate`'s does too (on a call, `mayBool` is the
declared flag). -/
theorem opOk_shape_stronger (c : TE) (h : opOk E true c = true) : opOk E false c = true := by
  obtain ⟨k, cs⟩ := c
  cases k <;> simp_all [opOk, mayBool, nbType]

/-! ## `type_name` (binder.rs:2408-2428) -/

/-- `runtime::functions::Type` (functions/mod.rs:531-556). -/
inductive FT
  | null | bool | int | float | string | list (t : FT) | map | node | rel | path | vecf32 | point
  | datetime | date | time | duration | any | union (ts : List FT) | optional (t : FT)

mutual
/-- `Type::can_return_boolean` (functions/mod.rs:569-576). -/
def FT.canBool : FT → Bool
  | .bool | .null | .any => true
  | .union ts => FT.canBoolAny ts
  | .optional t => t.canBool
  | _ => false
def FT.canBoolAny : List FT → Bool
  | [] => false
  | t :: ts => t.canBool || FT.canBoolAny ts
end

/-- `type_name`: the C spelling of a single declared type. -/
def FT.typeName : FT → Option String
  | .int => some "Integer" | .float => some "Float" | .string => some "String"
  | .list _ => some "List" | .map => some "Map" | .node => some "Node" | .rel => some "Edge"
  | .path => some "Path" | .vecf32 => some "Vectorf32" | .point => some "Point"
  | .datetime => some "Datetime" | .date => some "Date" | .time => some "Time"
  | .duration => some "Duration"
  | .optional t => t.typeName
  | _ => none

/-- The error never names a type that can be Boolean (or Null): a named type
is one `can_return_boolean` rejects, so the message matches the check. -/
theorem typeName_not_bool : ∀ t : FT, t.typeName.isSome = true → t.canBool = false
  | .optional t, h => by
    simp only [FT.typeName] at h; simp only [FT.canBool]; exact typeName_not_bool t h
  | .null, h | .bool, h | .any, h | .union _, h => by simp [FT.typeName] at h
  | .int, _ | .float, _ | .string, _ | .list _, _ | .map, _ | .node, _ | .rel, _ | .path, _
  | .vecf32, _ | .point, _ | .datetime, _ | .date, _ | .time, _ | .duration, _ => rfl

/-- `Optional` is transparent; unions have no single name. -/
theorem typeName_optional (t : FT) : (FT.optional t).typeName = t.typeName := rfl
theorem typeName_union (ts : List FT) : (FT.union ts).typeName = none := rfl

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
