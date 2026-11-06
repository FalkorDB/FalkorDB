import FalkorValueOrder.Glue

/-!
# PR #3437: `value_of_type(List, List(Any))` returns early (value.rs:1320-1321)

`vtPR` is `impl ValueTypeOf for Value::value_of_type` on the PR branch (value.rs:1315-1350):
value_order's `W.valueOfType` (main, Glue.lean) with the new first arm
`(Self::List(_), Type::List(ty)) if matches!(**ty, Type::Any) => None`, applied at every
recursion level. **`vtPR_eq`**: it returns exactly what main returns, for every value and
every type — the early arm answers what main's element walk answers, because every value
is accepted by `Any` (`valueOfType_any`).
-/

namespace Pr3437.ValueOfType
open ValueOrder.Glue

variable {F F32 : Type}

def vtPR : W F F32 → Ty → Option (Ty × Ty)
  | .list _, .list .any => none
  | .list vs, .list ty => listFirst vs ty
  | .null, .null | .bool _, .bool | .int _, .int | .float _, .float | .str _, .string
  | .point _ _, .point | .vecf32 _, .vecf32 | .map _, .map | .node _, .node | .rel _, .rel
  | .path _, .path | .datetime _, .datetime | .date _, .date | .time _, .time
  | .duration _, .duration => none
  | _, .any => none
  | v, .optional ty => vtPR v ty
  | v, .union tys => unionAll v tys tys
  | v, e => some (v.getType, e)
where
  listFirst : List (W F F32) → Ty → Option (Ty × Ty)
    | [], _ => none
    | v :: vs, ty => match vtPR v ty with
      | some r => some r
      | none => listFirst vs ty
  unionAll : W F F32 → List Ty → List Ty → Option (Ty × Ty)
    | v, [], all => some (v.getType, .union all)
    | v, t :: ts, all => match vtPR v t with
      | none => none
      | some _ => unionAll v ts all

theorem listFirst_any (vs : List (W F F32)) : W.valueOfType.listFirst vs .any = none := by
  induction vs with
  | nil => simp [W.valueOfType.listFirst]
  | cons v vs ih => simp [W.valueOfType.listFirst, valueOfType_any, ih]

mutual
/-- **Claim 5.** The early return changes no answer. -/
theorem vtPR_eq (v : W F F32) (t : Ty) : vtPR v t = v.valueOfType t := by
  cases t with
  | optional ty =>
    have ih := vtPR_eq v ty
    cases v <;> simp only [vtPR, W.valueOfType] <;> exact ih
  | union tys =>
    have ih := unionAll_eq v tys tys
    cases v <;> simp only [vtPR, W.valueOfType] <;> exact ih
  | list ty =>
    cases v with
    | list vs =>
      cases ty with
      | any => simp [vtPR, W.valueOfType, listFirst_any]
      | _ => simp only [vtPR, W.valueOfType] <;> exact listFirst_eq vs _
    | _ => simp [vtPR, W.valueOfType]
  | _ => cases v <;> simp [vtPR, W.valueOfType]
termination_by sizeOf v + sizeOf t
decreasing_by all_goals (simp_wf <;> omega)

theorem listFirst_eq (vs : List (W F F32)) (ty : Ty) :
    vtPR.listFirst vs ty = W.valueOfType.listFirst vs ty := by
  cases vs with
  | nil => simp [vtPR.listFirst, W.valueOfType.listFirst]
  | cons v vs =>
    have h1 := vtPR_eq v ty
    have h2 := listFirst_eq vs ty
    simp only [vtPR.listFirst, W.valueOfType.listFirst, h1, h2]
    cases v.valueOfType ty <;> rfl
termination_by sizeOf vs + sizeOf ty
decreasing_by all_goals (simp_wf <;> omega)

theorem unionAll_eq (v : W F F32) (ts all : List Ty) :
    vtPR.unionAll v ts all = W.valueOfType.unionAll v ts all := by
  cases ts with
  | nil => simp [vtPR.unionAll, W.valueOfType.unionAll]
  | cons t ts =>
    have h1 := vtPR_eq v t
    have h2 := unionAll_eq v ts all
    simp only [vtPR.unionAll, W.valueOfType.unionAll, h1, h2]
    cases v.valueOfType t <;> rfl
termination_by sizeOf v + sizeOf ts
decreasing_by all_goals (simp_wf <;> omega)
end

/-- In particular `size()`'s argument check `Union([List(Any), String, Null])`
(functions/list.rs:55-59) accepts every list without walking it, as main did. -/
theorem size_arg_list (vs : List (W F F32)) :
    vtPR (.list vs) (.union [.list .any, .string, .null]) = none ∧
    (W.list vs : W F F32).valueOfType (.union [.list .any, .string, .null]) = none := by
  refine ⟨?_, ?_⟩
  · simp [vtPR, vtPR.unionAll]
  · rw [← vtPR_eq]; simp [vtPR, vtPR.unionAll]

end Pr3437.ValueOfType
