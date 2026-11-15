import Columnar.FuncLane

/-!
# The registry's `hasLabels` (functions/entity.rs:62-127) satisfies the fast path's `HL`

A copy, over the engine value, of what the per-row call runs: `validate_args_type`'s `Fixed`
arm (functions/mod.rs:828-856) over `value_of_type` (value.rs:1315-1350) for the declared
arguments `[Node | Null, List(Any)]`, then `has_labels` (proofs/functions_str_list
`Entity.hasLabels` models the same body over its own value type). **`hl_hasLabelsFn`**: on a
node and a list of label names it validates and answers "every label is on the node"
(`Runtime::node_has_label`), which is what `eval_has_labels` computes from the label ids.
-/

namespace Columnar.HasLabelsFn
open FalkorCore (V)
open FalkorExpr.Eval (Fn)
open Columnar.CoreOps (RT)
open Columnar.FuncLane (hasLabelName HL)

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

/-- `Type` (functions/mod.rs:531). -/
inductive Ty where
  | null | bool | int | float | string | list (t : Ty) | map | node | rel | path | vecf32 | point
  | datetime | date | time | duration | any | union (ts : List Ty) | optional (t : Ty)

/-- `Value::get_type` (value.rs:1361). -/
def getType : V F → Ty
  | .null => .null | .bool _ => .bool | .int _ => .int | .float _ => .float | .str _ => .string
  | .list _ => .list .any | .map _ => .map | .node _ => .node | .rel _ => .rel | .path _ => .path
  | .vec _ => .vecf32 | .point .. => .point | .datetime _ => .datetime | .date _ => .date
  | .time _ => .time | .duration _ => .duration

/-- The same-type arms of `value_of_type` (value.rs:1328-1343). -/
def sameTy : V F → Ty → Bool
  | .null, .null | .bool _, .bool | .int _, .int | .float _, .float | .str _, .string
  | .point .., .point | .vec _, .vecf32 | .map _, .map | .node _, .node | .rel _, .rel
  | .path _, .path | .datetime _, .datetime | .date _, .date | .time _, .time
  | .duration _, .duration => true
  | _, .any => true
  | _, _ => false

mutual
/-- `value_of_type` (value.rs:1315): `none` when the value has the type, else
`(actual, expected)`. -/
def valueOfType : V F → Ty → Option (Ty × Ty)
  | .list vs, .list t => listCheck t vs
  | v, .optional t => if sameTy v (.optional t) then none else valueOfType v t
  | v, .union ts => if sameTy v (.union ts) then none
    else if anyMatch v ts then none else some (getType v, .union ts)
  | v, e => if sameTy v e then none else some (getType v, e)
/-- `for ty in tys { v.value_of_type(ty)?; }`: some member type matched. -/
def anyMatch (v : V F) : List Ty → Bool
  | [] => false
  | t :: ts => (valueOfType v t).isNone || anyMatch v ts
/-- `for v in vs { if let Some(res) = v.value_of_type(ty) { return Some(res) } }`. -/
def listCheck (t : Ty) : List (V F) → Option (Ty × Ty)
  | [] => none
  | v :: vs => match valueOfType v t with
    | some r => some r
    | none => listCheck t vs
end

/-- The union spelling: `A or B`, `A, B, or C` (functions/mod.rs:633-649). -/
def unionName : List String → String
  | [] => ""
  | [a] => a
  | first :: rest =>
    first ++ String.join (rest.dropLast.map fun t => ", " ++ t) ++
      (if rest.length > 1 then "," else "") ++ " or " ++ (rest.getLast?.getD "")

mutual
/-- `Display for Type` (functions/mod.rs:610-653). -/
def tyName : Ty → String
  | .null => "Null" | .bool => "Boolean" | .int => "Integer" | .float => "Float" | .string => "String"
  | .list _ => "List" | .map => "Map" | .node => "Node" | .rel => "Edge" | .path => "Path"
  | .vecf32 => "VecF32" | .point => "Point" | .datetime => "Datetime" | .date => "Date"
  | .time => "Time" | .duration => "Duration" | .any => "Any" | .optional t => tyName t
  | .union ts => unionName (tyNames ts)
def tyNames : List Ty → List String
  | [] => []
  | t :: ts => tyName t :: tyNames ts
end

def isOptional : Ty → Bool | .optional _ => true | _ => false

/-- `validate_args_type`'s `Fixed` arm (mod.rs:833-852). The `Missing argument` text prints
the type with `{:?}`; it is kept abstract (`dbg`), the call sites here never miss one. -/
def validateFixed (name : String) (dbg : Ty → String) (tys : List Ty) (args : List (V F)) : Except String Unit :=
  go 0 tys
where
  go (i : Nat) : List Ty → Except String Unit
    | [] => .ok ()
    | t :: ts =>
      if h : i < args.length then
        match valueOfType args[i] t with
        | some (actual, expected) => .error s!"Type mismatch: expected {tyName expected} but was {tyName actual}"
        | none => go (i + 1) ts
      else if isOptional t then go (i + 1) ts
      else .error s!"Missing argument {i + 1} for function '{name}', expected type {dbg t}"

/-- The label loop of `has_labels` (entity.rs:72-118): the whole list is type-checked. -/
def labelsLoop (rt : RT F) (id : Nat) : List (V F) → Bool → Except String (V F)
  | [], acc => .ok (.bool acc)
  | .str n :: ls, acc => labelsLoop rt id ls (acc && hasLabelName rt id n)
  | .int _ :: _, _ => .error "Type mismatch: expected String but was Integer"
  | .float _ :: _, _ => .error "Type mismatch: expected String but was Float"
  | .bool _ :: _, _ => .error "Type mismatch: expected String but was Boolean"
  | v :: _, _ => .error s!"Type mismatch: expected String but was {v.name}"

/-- `has_labels` (entity.rs:68): `match (iter.next(), iter.next())`. -/
def hasLabelsCall (rt : RT F) : List (V F) → Except String (V F)
  | .node id :: .list ls :: _ => labelsLoop rt id ls true
  | .null :: _ => .ok .null
  | _ :: .null :: _ => .ok .null
  | _ => .error "unreachable!() in has_labels"

/-- The registered `hasLabels` (entity.rs:62-67: `args: [Node | Null, List(Any)]`). -/
def hasLabelsFn (rt : RT F) (dbg : Ty → String) : Fn (V F) where
  name := "hasLabels"
  write := false
  nonDeterministic := false
  structFn := none
  structSlots := 0
  validate := validateFixed "hasLabels" dbg [.union [.node, .null], .list .any]
  call := hasLabelsCall rt

theorem listCheck_any : ∀ vs : List (V F), listCheck .any vs = none
  | [] => by simp [listCheck]
  | v :: vs => by
    have : valueOfType v .any = none := by cases v <;> simp [valueOfType, sameTy]
    simp [listCheck, this, listCheck_any vs]

theorem labelsLoop_strs (rt : RT F) (id : Nat) : ∀ (names : List String) (acc : Bool),
    labelsLoop rt id (names.map .str) acc = .ok (.bool (acc && names.all (hasLabelName rt id)))
  | [], acc => by simp [labelsLoop]
  | n :: ns, acc => by simp [labelsLoop, labelsLoop_strs rt id ns, Bool.and_assoc]

/-- **The registry's `hasLabels` satisfies `HL`.** -/
theorem hl_hasLabelsFn (rt : RT F) (dbg : Ty → String) : HL rt (hasLabelsFn rt dbg) := by
  intro id names
  constructor
  · simp [hasLabelsFn, validateFixed, validateFixed.go, valueOfType, sameTy, anyMatch, listCheck_any]
  · simp [hasLabelsFn, hasLabelsCall, labelsLoop_strs]

end Columnar.HasLabelsFn
