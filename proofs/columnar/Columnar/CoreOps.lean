import Columnar.VecEval
import Columnar.Lanes
import FalkorExprSemantics.EvalArms

/-!
# The engine's value operations as expr_semantics' `Ops`, and the shared forms agree

`coreOps rt` instantiates proofs/expr_semantics' abstract `Ops` over the engine value
`FalkorCore.V F` with the *same* models the columnar lanes use: `compare_value` is
`compareValue rt.oc` (value.rs:1255; its numeric/null arms concrete, `rt.oc` the remaining
arms), `+ - * / %` are `valueArith rt.oa` (value.rs:904-1170), `all_equals` /
`all_not_equals` are expr_semantics' `allEquals` / `allNotEquals` (eval.rs:1713, :1736), and
everything the lanes never compute (attribute reads, `get_attr`, `Contains`, map building,
…) is read from the runtime record `rt`, shared by both evaluators.

The shared forms agree (the gap item 3 of the PR #3437 review):
* `allEq_rowCmp` / `allNeq_rowCmp`: expr_semantics' `all_equals` / `all_not_equals` on two
  values = columnar's `rowCmp` (`compare_values`, vectorized.rs:239) for `=` / `<>`;
* `cmpResult_rowCmp`: the `< > <= >=` fast path (eval.rs:454-472) = `rowCmp`;
* `caseHit_caseMatch`: `eval_case`'s arm test = columnar's `caseMatch` (vector_expr.rs:513-529);
* `notRow_eq`: the per-row `NOT` arm = columnar's `notRow`;
* the arithmetic arms *are* `valueArith` (`coreOps_add`, `coreOps_arith`), the function
  `arithmetic_agree` compares the lanes against.
-/

namespace Columnar.CoreOps
open FalkorCore (V)
open FalkorExpr.Eval (E Ops Fn Compound Ev evalNode allEquals allNotEquals cmpResult caseHit)

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

/-- What both evaluators read from the runtime and the parts of `Value`'s operators no lane
inspects. -/
structure RT (F : Type) where
  params : String → Option (V F)
  oc : V F → V F → Ordering × FalkorExpr.Eval.Flag
  oa : AOp → V F → V F → Except String (V F)
  getAttr : V F → String → Except String (V F)
  nodeAttr : Nat → String → Option (V F)
  relAttr : Nat → String → Option (V F)
  matNode : List Nat → String → List (V F)
  matRel : List Nat → String → List (V F)
  labelId : String → Option Nat
  hasLabelId : Nat → Nat → Bool
  mkMap : List (String × V F) → V F
  contains : List (V F) → V F → V F
  debugPair : V F → V F → String
  nodeAttrs : Nat → List (String × V F)
  relAttrs : Nat → List (String × V F)
  distinct : List (V F) → V F
  write : Bool

/-- `Flag` (expr_semantics) ↔ `Flag` (columnar): `DisjointOrNull`. -/
def flagOf : FalkorExpr.Eval.Flag → Flag
  | .disjoint => .disjoint | .comparedNull => .comparedNull | .nan => .nan | .none => .none
def flagTo : Flag → FalkorExpr.Eval.Flag
  | .disjoint => .disjoint | .comparedNull => .comparedNull | .nan => .nan | .none => .none

theorem flagTo_flagOf (f : FalkorExpr.Eval.Flag) : flagTo (flagOf f) = f := by cases f <;> rfl

/-- `compare_value` with the remaining arms from `rt`, in columnar's flag type. -/
def ocC (rt : RT F) (a b : V F) : Ordering × Flag := ((rt.oc a b).1, flagOf (rt.oc a b).2)

/-- `compare_value` in expr_semantics' flag type. -/
def cv (rt : RT F) (a b : V F) : Ordering × FalkorExpr.Eval.Flag :=
  ((compareValue (ocC rt) a b).1, flagTo (compareValue (ocC rt) a b).2)

def asStr : V F → Option String | .str s => some s | _ => none
def asNode : V F → Option Nat | .node n => some n | _ => none
def asRel : V F → Option Nat | .rel n => some n | _ => none
def asList : V F → Option (List (V F)) | .list l => some l | _ => none
def asInt : V F → Option Int | .int i => some i | _ => none
def asMap : V F → Option (List (String × V F)) | .map m => some m | _ => none
def boolOf : V F → Option (Option Bool) | .bool b => some (some b) | .null => some none | _ => none

/-- The arithmetic node kinds of expr_semantics as columnar `AOp`s. -/
abbrev aopOf := VecEval.aopOf

/-- **The engine's operations as expr_semantics' `Ops`.** -/
def coreOps (rt : RT F) : Ops (V F) where
  null := .null
  bool := .bool
  str := .str
  isNull := V.isNull
  asStr := asStr
  asNode := asNode
  asRel := asRel
  mkMap := rt.mkMap
  add := valueArith rt.oa .add
  arith op := valueArith rt.oa (aopOf op)
  cmp := cv rt
  allEq a b := allEquals (cv rt) .null .bool [fun _ => .ok a, fun _ => .ok b]
  allNeq a b := allNotEquals (cv rt) .null .bool [fun _ => .ok a, fun _ => .ok b]
  getAttr := rt.getAttr
  nodeAttr := rt.nodeAttr
  relAttr := rt.relAttr
  name := V.name
  boolOf := boolOf
  mkList := .list
  asList := asList
  asInt := asInt
  asMap := asMap
  contains := rt.contains
  debugPair := rt.debugPair
  nodeAttrs := rt.nodeAttrs
  relAttrs := rt.relAttrs
  distinct := rt.distinct
  rtWrite := rt.write

variable (rt : RT F)

theorem flagOf_flagTo (f : Flag) : flagOf (flagTo f) = f := by cases f <;> rfl

theorem cv_flag (a b : V F) : flagOf (cv rt a b).2 = (compareValue (ocC rt) a b).2 := by
  simp [cv, flagOf_flagTo]

/-- **`all_equals` = `compare_values(.., Eq)`** on two values. -/
theorem allEq_rowCmp (a b : V F) : (coreOps rt).allEq a b = .ok (rowCmp (ocC rt) a b .eq) := by
  simp only [coreOps]
  rw [FalkorExpr.Eval.allEquals_two]
  simp only [rowCmp, compareValues, cv]
  rcases compareValue (ocC rt) a b with ⟨o, fl⟩
  cases fl <;> cases o <;> rfl

/-- **`all_not_equals` = `compare_values(.., Neq)`** on two values (#2917 semantics). -/
theorem allNeq_rowCmp (a b : V F) : (coreOps rt).allNeq a b = .ok (rowCmp (ocC rt) a b .neq) := by
  simp only [coreOps]
  rw [FalkorExpr.Eval.allNotEquals_two]
  simp only [rowCmp, compareValues, cv]
  rcases compareValue (ocC rt) a b with ⟨o, fl⟩
  cases fl <;> cases o <;> rfl

/-- **The `< > <= >=` fast path = `compare_values`** for the same operator. -/
theorem cmpResult_rowCmp (op : FalkorExpr.Eval.CmpOp) (a b : V F) :
    cmpResult (coreOps rt) op ((coreOps rt).cmp a b) = rowCmp (ocC rt) a b (VecEval.cmpOf op) := by
  simp only [coreOps, cmpResult, rowCmp, compareValues, cv]
  rcases compareValue (ocC rt) a b with ⟨o, fl⟩
  cases op <;> cases fl <;> cases o <;> rfl

/-- **`eval_case`'s arm test = the columnar one** (value form via `compare_values(.., Eq)`,
searched form "not `false` / `null`"). -/
theorem caseHit_caseMatch (sv : Option (V F)) (w : V F) :
    caseHit (coreOps rt) sv w = caseMatch (ocC rt) sv w := by
  cases sv with
  | some s =>
    rw [caseMatch_eq_compare_value]
    simp only [caseHit, coreOps, cv]
    rcases compareValue (ocC rt) w s with ⟨o, fl⟩
    cases fl <;> cases o <;> rfl
  | none =>
    simp only [caseHit, coreOps, caseMatch]
    cases w with
    | bool b => cases b <;> rfl
    | _ => rfl

/-- **The per-row `NOT` arm = columnar's `notRow`.** -/
theorem notRow_eq (v : V F) :
    (match (coreOps rt).boolOf v with
      | some (some b) => Except.ok ((coreOps rt).bool !b)
      | some none => .ok (coreOps rt).null
      | none => .error (FalkorExpr.Eval.boolMismatchMsg (coreOps rt) v)) = notRow v := by
  cases v <;> rfl

@[simp] theorem coreOps_boolOf : (coreOps rt).boolOf = boolOf := rfl
@[simp] theorem coreOps_bool : (coreOps rt).bool = V.bool := rfl
@[simp] theorem coreOps_null : (coreOps rt).null = V.null := rfl
@[simp] theorem coreOps_mkList : (coreOps rt).mkList = V.list := rfl
@[simp] theorem coreOps_str : (coreOps rt).str = V.str := rfl
@[simp] theorem coreOps_name : (coreOps rt).name = V.name := rfl
@[simp] theorem coreOps_isNull : (coreOps rt).isNull = V.isNull := rfl
@[simp] theorem coreOps_asNode : (coreOps rt).asNode = asNode := rfl
@[simp] theorem coreOps_asRel : (coreOps rt).asRel = asRel := rfl
@[simp] theorem coreOps_asStr : (coreOps rt).asStr = asStr := rfl
@[simp] theorem coreOps_nodeAttr : (coreOps rt).nodeAttr = rt.nodeAttr := rfl
@[simp] theorem coreOps_relAttr : (coreOps rt).relAttr = rt.relAttr := rfl
@[simp] theorem coreOps_rtWrite : (coreOps rt).rtWrite = rt.write := rfl

theorem coreOps_add : (coreOps rt).add = valueArith rt.oa .add := rfl
theorem coreOps_arith (op : FalkorExpr.Eval.Arith) : (coreOps rt).arith op = valueArith rt.oa (aopOf op) := rfl

end Columnar.CoreOps
