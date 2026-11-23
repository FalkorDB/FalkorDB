/-!
# `is_comparison`, `is_boolean_operator` and the `bool_valued_operand` arm of `needs_post_filter`

`graph/src/planner/optimizer/utilize_index.rs` (#2975). The term model `Index.T` has no
comparison-valued operand, so this module models the expression kinds these helpers look at
and the parent walk of `bool_valued_operand` on its own.
-/

namespace OptimizerScan.BoolOperand

/-- The `ExprIR` kinds the helpers distinguish; `other` is every remaining variant. -/
inductive EK | eq | lt | gt | le | ge | inn | and | or | paren | list | other
  deriving DecidableEq, Repr

/-- `is_comparison`. -/
def isComparison : EK → Bool
  | .eq | .lt | .gt | .le | .ge | .inn => true
  | _ => false

/-- `is_boolean_operator`. -/
def isBooleanOperator (e : EK) : Bool :=
  isComparison e || (match e with | .and | .or => true | _ => false)

def isRangeOp : EK → Bool
  | .lt | .gt | .le | .ge => true
  | _ => false

/-- `bool_valued_operand` of `needs_post_filter`: `node` is a boolean operator and its first
non-`Paren` ancestor (`ancestors`, nearest first) is a range, or, under `index-falkordb`
(`falkor`), a comparison or a `List` (an `IN` list member, since cfa509d25). -/
def boolValuedOperand (falkor : Bool) (node : EK) (ancestors : List EK) : Bool :=
  isBooleanOperator node &&
    match ancestors.find? (fun p => p != .paren) with
    | some p => isRangeOp p || (falkor && (isComparison p || p == .list))
    | none => false

theorem isComparison_iff (e : EK) :
    isComparison e = true ↔ e ∈ [.eq, .lt, .gt, .le, .ge, .inn] := by
  cases e <;> decide

theorem isBooleanOperator_iff (e : EK) :
    isBooleanOperator e = true ↔ e ∈ [.eq, .lt, .gt, .le, .ge, .inn, .and, .or] := by
  cases e <;> decide

theorem isComparison_isBooleanOperator (e : EK) (h : isComparison e = true) :
    isBooleanOperator e = true := by
  simp [isBooleanOperator, h]

/-- A boolean operator directly under a range bound always keeps the filter (no backend serves a
Bool bound), on either build; `Paren`s in between are skipped. -/
theorem range_parent_flags (falkor : Bool) (node p : EK) (parens : Nat)
    (hn : isBooleanOperator node = true) (hp : isRangeOp p = true) (rest : List EK) :
    boolValuedOperand falkor node (List.replicate parens .paren ++ p :: rest) = true := by
  have hpne : p ≠ .paren := by intro h; subst h; simp [isRangeOp] at hp
  have hfind : (List.replicate parens EK.paren ++ p :: rest).find? (fun q => q != .paren)
      = some p := by
    induction parens with
    | zero => simp [hpne]
    | succ n ih => simp [List.replicate_succ, ih]
  simp [boolValuedOperand, hn, hfind, hp]

/-- On the RediSearch build only a range parent flags a boolean operand: RediSearch keys a Bool
apart from numbers, so `n.v = (x = 1)` and `n.v IN [x = 1]` are served exactly. -/
theorem redisearch_only_range (node : EK) (ancestors : List EK) :
    boolValuedOperand false node ancestors = true →
      ∃ p, ancestors.find? (fun q => q != .paren) = some p ∧ isRangeOp p = true := by
  unfold boolValuedOperand
  cases h : ancestors.find? (fun q => q != .paren) with
  | none => simp
  | some p => intro hb; simp at hb; exact ⟨p, rfl, hb.2⟩

/-- cfa509d25: under `index-falkordb`, a boolean operator that is a member of a pushed `IN` list
keeps the filter (the native column cannot tell `true` from `1`). -/
theorem falkor_in_list_member_flags (node : EK) (hn : isBooleanOperator node = true)
    (rest : List EK) : boolValuedOperand true node (.list :: rest) = true := by
  have hfind : (EK.list :: rest).find? (fun q => q != .paren) = some .list := by
    simp
  simp [boolValuedOperand, hn, hfind]

/-- Under `index-falkordb`, a boolean operator under any comparison keeps the filter. -/
theorem falkor_comparison_parent_flags (node p : EK) (hn : isBooleanOperator node = true)
    (hp : isComparison p = true) (rest : List EK) :
    boolValuedOperand true node (p :: rest) = true := by
  have hpne : p ≠ .paren := by intro h; subst h; simp [isComparison] at hp
  have hfind : (p :: rest).find? (fun q => q != .paren) = some p := by
    simp [hpne]
  simp [boolValuedOperand, hn, hfind, hp]

/-- A non-boolean node (a property, literal, function call, …) is never flagged by this arm. -/
theorem non_boolean_not_flagged (falkor : Bool) (node : EK) (ancestors : List EK)
    (hn : isBooleanOperator node = false) : boolValuedOperand falkor node ancestors = false := by
  simp [boolValuedOperand, hn]

end OptimizerScan.BoolOperand
