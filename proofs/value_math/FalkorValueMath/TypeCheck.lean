import FalkorValueMath.Value
/-!
# Function registry, arity and argument type checking (`graph/src/runtime/functions/mod.rs`)

| Lean | Rust |
| --- | --- |
| `Ty` | `enum Type` (mod.rs:531) |
| `valueOfType` | `ValueTypeOf::value_of_type` (value.rs:1282) |
| `getType` | `ValueGetType::get_type` (value.rs:1328) |
| `validate` | `GraphFn::validate` (mod.rs:787-823) — arity, called by the parser |
| `validateArgsType` | `GraphFn::validate_args_type` (mod.rs:828) — per call, eval.rs:955 |
| `isCompatibleWith` | `Type::is_compatible_with` (mod.rs:590) |
| `canReturnBoolean`, `canReturnEntity` | mod.rs:569, :579 |
| `tyDisplay` | `Display for Type` (mod.rs:610) |
| `Registry`, `regGet`, `regAdd`, `regIsAggregate` | `Functions::get` (:1040), `add`/`add_var_len`/`add_procedure` (:915/:971/:943), `is_aggregate` (:1071) |
| `FnTypeTag`, `fnTypeEq` | `FnType` + `PartialEq for FnType` (:514) |
-/

namespace ValueMath

inductive Ty where
  | null | bool | int | float | string
  | list (t : Ty)
  | map | node | rel | path | vecf32 | point | datetime | date | time | duration
  | any
  | union (ts : List Ty)
  | optional (t : Ty)
  deriving Inhabited

variable {F : Type}

/-- `get_type` (value.rs:1328): lists report `List(Any)`. -/
def getType : V F → Ty
  | .null => .null | .bool _ => .bool | .int _ => .int | .float _ => .float
  | .str _ => .string | .list _ => .list .any | .map _ => .map | .node _ => .node
  | .rel _ => .rel | .path _ => .path | .vec _ => .vecf32 | .point .. => .point
  | .datetime _ => .datetime | .date _ => .date | .time _ => .time | .duration _ => .duration

/-- The "same variant" arms of `value_of_type` (value.rs:1295-1310, without `Any`). -/
def tagMatch : V F → Ty → Bool
  | .null, .null | .bool _, .bool | .int _, .int | .float _, .float | .str _, .string
  | .point .., .point | .vec _, .vecf32 | .map _, .map | .node _, .node | .rel _, .rel
  | .path _, .path | .datetime _, .datetime | .date _, .date | .time _, .time
  | .duration _, .duration => true
  | _, _ => false

def Ty.isList : Ty → Bool
  | .list _ => true
  | _ => false

def Ty.isAny : Ty → Bool
  | .any => true
  | _ => false

def V.isList : V F → Bool
  | .list _ => true
  | _ => false

mutual
/-- `value_of_type` (value.rs:1282-1320), arms in source order. `none` = accepted,
`some (actual, expected)` = the mismatch reported. -/
def valueOfType : V F → Ty → Option (Ty × Ty)
  | .list vs, .list ty => listLoop vs ty
  | v, t =>
    if tagMatch v t || t.isAny then none else
    match t with
    | .optional ty => valueOfType v ty
    | .union tys => if unionLoop v tys then none else some (getType v, .union tys)
    | e => some (getType v, e)
  termination_by _ t => (sizeOf t, 0)
  decreasing_by all_goals (simp_wf; first | omega | (apply Prod.Lex.left; simp_all; omega) | skip)

/-- The `for v in vs.iter()` loop of the `List` arm: first element's mismatch. -/
def listLoop : List (V F) → Ty → Option (Ty × Ty)
  | [], _ => none
  | v :: vs, ty => match valueOfType v ty with
    | some r => some r
    | none => listLoop vs ty
  termination_by vs ty => (sizeOf ty, vs.length + 1)
  decreasing_by all_goals (simp_wf; first | (apply Prod.Lex.right; omega) | skip)

/-- The `for ty in tys { v.value_of_type(ty)?; }` loop: true iff some member accepts. -/
def unionLoop : V F → List Ty → Bool
  | _, [] => false
  | v, ty :: rest => match valueOfType v ty with
    | none => true
    | some _ => unionLoop v rest
  termination_by _ tys => (sizeOf tys, 0)
  decreasing_by all_goals (simp_wf; first | omega | (apply Prod.Lex.left; simp_all; omega) | skip)
end

#eval (valueOfType (F := Unit) (.int 1) (.union [.int, .float, .null])).isNone
#eval (valueOfType (F := Unit) .null (.optional .int)).isSome

/-- The clean specification of "argument `v` is acceptable for declared type `t`". -/
inductive Accepts : V F → Ty → Prop
  | any (v : V F) : Accepts v .any
  | tag (v : V F) (t : Ty) : tagMatch v t = true → Accepts v t
  | list (vs : List (V F)) (ty : Ty) : (∀ v ∈ vs, Accepts v ty) → Accepts (.list vs) (.list ty)
  | opt (v : V F) (ty : Ty) : Accepts v ty → Accepts v (.optional ty)
  | union (v : V F) (tys : List Ty) (ty : Ty) : ty ∈ tys → Accepts v ty → Accepts v (.union tys)

theorem listLoop_none (vs : List (V F)) (ty : Ty) :
    listLoop vs ty = none ↔ ∀ v ∈ vs, valueOfType v ty = none := by
  induction vs with
  | nil => simp [listLoop]
  | cons v rest ih =>
    rw [listLoop]
    cases h : valueOfType v ty with
    | some r => simp [h]
    | none => simp [h, ih]

theorem unionLoop_true (v : V F) (tys : List Ty) :
    unionLoop v tys = true ↔ ∃ ty ∈ tys, valueOfType v ty = none := by
  induction tys with
  | nil => simp [unionLoop]
  | cons ty rest ih =>
    rw [unionLoop]
    cases h : valueOfType v ty with
    | none => simp [h]
    | some r => simp [h, ih]

theorem tagMatch_list (v : V F) (ty : Ty) : tagMatch v (.list ty) = false := by cases v <;> rfl
theorem tagMatch_opt (v : V F) (ty : Ty) : tagMatch v (.optional ty) = false := by cases v <;> rfl
theorem tagMatch_union (v : V F) (tys : List Ty) : tagMatch v (.union tys) = false := by cases v <;> rfl
theorem tagMatch_any (v : V F) : tagMatch v .any = false := by cases v <;> rfl

theorem vot_opt (v : V F) (ty : Ty) : valueOfType v (.optional ty) = valueOfType v ty := by
  cases v <;> simp [valueOfType, tagMatch, Ty.isAny]

theorem vot_union (v : V F) (tys : List Ty) :
    valueOfType v (.union tys) = if unionLoop v tys then none else some (getType v, .union tys) := by
  cases v <;> simp [valueOfType, tagMatch, Ty.isAny]

theorem vot_any (v : V F) : valueOfType v .any = none := by
  cases v <;> simp [valueOfType, tagMatch, Ty.isAny]

theorem vot_list_list (vs : List (V F)) (ty : Ty) : valueOfType (.list vs) (.list ty) = listLoop vs ty := by
  simp [valueOfType]

theorem vot_nonlist_list (v : V F) (ty : Ty) (h : v.isList = false) :
    valueOfType v (.list ty) = some (getType v, .list ty) := by
  cases v <;> simp_all [valueOfType, tagMatch, Ty.isAny, V.isList]

/-- For a base (non-structural) type, acceptance is the tag check. -/
def Ty.isBase : Ty → Bool
  | .list _ | .any | .union _ | .optional _ => false
  | _ => true

theorem vot_base (v : V F) (t : Ty) (ht : t.isBase = true) :
    valueOfType v t = if tagMatch v t then none else some (getType v, t) := by
  cases t <;> simp [Ty.isBase] at ht <;> cases v <;> simp [valueOfType, tagMatch, Ty.isAny]

theorem accepts_base {v : V F} {t : Ty} (ht : t.isBase = true) : Accepts v t ↔ tagMatch v t = true := by
  constructor
  · intro h; cases h <;> simp_all [Ty.isBase]
  · exact Accepts.tag v t

/-- **`value_of_type` is exactly the acceptance spec**: it returns `None` iff the value
matches the declared type (`Any`; same variant; lists element-wise; `Optional` = inner;
`Union` = some member). -/
theorem valueOfType_none_iff : ∀ (t : Ty) (v : V F), valueOfType v t = none ↔ Accepts v t := by
  intro t
  induction t using Ty.rec (motive_2 := fun tys => ∀ ty ∈ tys, ∀ (v : V F),
      valueOfType v ty = none ↔ Accepts v ty)
  case list ty ih =>
    intro v
    cases v
    case list vs =>
      rw [vot_list_list, listLoop_none]
      constructor
      · intro h; exact Accepts.list vs ty (fun w hw => (ih w).1 (h w hw))
      · intro h w hw
        cases h with
        | tag _ _ h' => simp [tagMatch] at h'
        | list _ _ h' => exact (ih w).2 (h' w hw)
    all_goals
      rw [vot_nonlist_list _ _ rfl]
      simp only [reduceCtorEq, false_iff]
      intro h; cases h; simp_all [tagMatch]
  case any => intro v; simp [vot_any]; exact Accepts.any v
  case optional ty ih =>
    intro v; rw [vot_opt, ih]
    constructor
    · exact Accepts.opt v ty
    · intro h; cases h with
      | tag _ _ h' => simp [tagMatch_opt] at h'
      | opt _ _ h' => exact h'
  case union tys ih =>
    intro v; rw [vot_union]
    split
    · rename_i hu
      simp only [true_iff]
      obtain ⟨ty, hm, hn⟩ := (unionLoop_true v tys).1 hu
      exact Accepts.union v tys ty hm ((ih ty hm v).1 hn)
    · rename_i hu
      simp only [reduceCtorEq, false_iff]
      intro h
      cases h with
      | tag _ _ h' => simp [tagMatch_union] at h'
      | union _ _ ty hm h' =>
        exact hu ((unionLoop_true v tys).2 ⟨ty, hm, (ih ty hm v).2 h'⟩)
  case nil ty h v => simp at h
  case cons ty rest ih1 ih2 t ht v =>
    rcases List.mem_cons.1 ht with rfl | h
    · exact ih1 v
    · exact ih2 t h v
  all_goals
    intro v
    rw [vot_base _ _ rfl, accepts_base rfl]
    split <;> simp_all

/-! ## Arity (`GraphFn::validate`, mod.rs:787) -/

inductive FnArgs where
  | fixed (ts : List Ty)
  | varLength (t : Ty)

def Ty.isOpt : Ty → Bool
  | .optional _ => true
  | _ => false

/-- `validate` (mod.rs:787-823): `least` = non-`Optional` count, `most` = declared count;
a built-in var-length function needs ≥ 1 argument (mod.rs:815, #2990), a UDF
(`fn_type = FnType::Udf`, `isUdf`) takes any count (mod.rs:821). -/
def validate (a : FnArgs) (n : Nat) (isUdf : Bool := false) : Bool :=
  match a with
  | .fixed ts => !(n < (ts.filter (fun t => !t.isOpt)).length) && !(n > ts.length)
  | .varLength _ => if n = 0 ∧ !isUdf then false else true

theorem validate_fixed_iff (ts : List Ty) (n : Nat) (u : Bool) :
    validate (.fixed ts) n u = true ↔ (ts.filter (fun t => !t.isOpt)).length ≤ n ∧ n ≤ ts.length := by
  simp [validate] <;> omega

/-- Var-length arity: a built-in accepts exactly `n ≥ 1`, a UDF accepts every `n`. -/
theorem validate_varLength (t : Ty) (n : Nat) (u : Bool) :
    validate (.varLength t) n u = true ↔ (1 ≤ n ∨ u = true) := by
  cases u <;> simp [validate] <;> omega

/-- **#2956 fixed** (`648b41c5c`, PR #2990): `coalesce` (math.rs:314, `var_arg:
Type::Any`) with zero arguments is now rejected at arity validation, with C's text
"Received 0 arguments to function 'coalesce', expected at least 1". Historical: before
`648b41c5c` the `VarLength` arm accepted every count, so `coalesce()` returned null
(the old theorem `coalesce_zero_args_accepted`). Same for `indegree`/`outdegree`. -/
theorem coalesce_zero_args_rejected : validate (.varLength .any) 0 = false := rfl

/-- `coalesce(x)` and longer still pass. -/
theorem coalesce_some_args_ok (n : Nat) (h : 1 ≤ n) : validate (.varLength .any) n = true :=
  (validate_varLength .any n false).2 (Or.inl h)

/-- UDFs keep accepting zero arguments (mod.rs:813: "UDFs take any number"). -/
theorem udf_zero_args_ok (t : Ty) : validate (.varLength t) 0 true = true := rfl

/-! ## Per-call type check (`validate_args_type`, mod.rs:828) -/

/-- Result: `none` = ok, `some msg-kind`. -/
inductive ArgErr where
  | missing (i : Nat)
  | mismatch (actual expected : Ty)

def validateArgsType (a : FnArgs) (args : List (V F)) : Option ArgErr :=
  match a with
  | .varLength _ => none
  | .fixed ts => go ts args 0
where
  go : List Ty → List (V F) → Nat → Option ArgErr
  | [], _, _ => none
  | t :: ts, [], i => if t.isOpt then go ts [] (i + 1) else some (.missing (i + 1))
  | t :: ts, v :: vs, i => match valueOfType v t with
    | some (act, exp) => some (.mismatch act exp)
    | none => go ts vs (i + 1)

/-- Spec: supplied arguments accepted position-wise; missing ones must be `Optional`. -/
def ArgsOk : List Ty → List (V F) → Prop
  | [], _ => True
  | t :: ts, [] => t.isOpt = true ∧ ArgsOk ts []
  | t :: ts, v :: vs => Accepts v t ∧ ArgsOk ts vs

/-- **`validate_args_type` succeeds iff** every supplied argument is accepted by its
declared type and every declared-but-missing argument is `Optional`. Arguments beyond
the declared list are *not* inspected (arity is the parser's `validate`). -/
theorem validateArgsType_go_none (ts : List Ty) (args : List (V F)) (i : Nat) :
    validateArgsType.go ts args i = none ↔ ArgsOk ts args := by
  induction ts generalizing args i with
  | nil => simp [validateArgsType.go, ArgsOk]
  | cons t ts ih =>
    cases args with
    | nil =>
      simp only [validateArgsType.go, ArgsOk]
      split
      · rename_i ho; rw [ih]; simp [ho]
      · rename_i ho; simp [ho]
    | cons v vs =>
      simp only [validateArgsType.go, ArgsOk]
      cases hv : valueOfType v t with
      | some r =>
        obtain ⟨act, exp⟩ := r
        simp only [reduceCtorEq, false_iff, not_and]
        intro h; rw [← valueOfType_none_iff, hv] at h; cases h
      | none =>
        simp only
        rw [ih, ← valueOfType_none_iff, hv]; simp

/-- The math functions' argument type `Int | Float | Null` (math.rs:44 etc.). -/
def numArg : Ty := .union [.int, .float, .null]

theorem accepts_numArg (v : V F) :
    Accepts v numArg ↔ (∃ i, v = .int i) ∨ (∃ f, v = .float f) ∨ v = .null := by
  rw [← valueOfType_none_iff]
  cases v <;> simp [numArg, vot_union, unionLoop, valueOfType, tagMatch, Ty.isAny]

/-- A single-argument numeric function reaches its body only with Int, Float or Null:
the `_ => unreachable!()` arms of `abs`, `ceil`, `exp`, `floor`, `log`, `log10`,
`round`, `sign`, `sqrt` (math.rs:55-242) are dead after `validate_args_type`. -/
theorem numeric_body_cases (v : V F) (h : validateArgsType (.fixed [numArg]) [v] = none) :
    (∃ i, v = .int i) ∨ (∃ f, v = .float f) ∨ v = .null := by
  have := ((validateArgsType_go_none [numArg] [v] 0).1 h).1
  exact (accepts_numArg v).1 this

/-- `range(1, 5, null)`: an `Optional(Int)` slot rejects an explicit null (C agrees). -/
theorem optional_rejects_null : valueOfType (F := F) .null (.optional .int) = some (.null, .int) := by
  simp [vot_opt, valueOfType, tagMatch, Ty.isAny, getType]

/-! ## `is_compatible_with`, `can_return_*` (mod.rs:569-606) -/

def tyBeq : Ty → Ty → Bool
  | .null, .null | .bool, .bool | .int, .int | .float, .float | .string, .string
  | .map, .map | .node, .node | .rel, .rel | .path, .path | .vecf32, .vecf32
  | .point, .point | .datetime, .datetime | .date, .date | .time, .time
  | .duration, .duration | .any, .any => true
  | .list a, .list b => tyBeq a b
  | .optional a, .optional b => tyBeq a b
  | .union as, .union bs => listBeq as bs
  | _, _ => false
where
  listBeq : List Ty → List Ty → Bool
  | [], [] => true
  | a :: as, b :: bs => tyBeq a b && listBeq as bs
  | _, _ => false

mutual
def isCompatibleWith : Ty → Ty → Bool
  | .any, _ | .null, _ | _, .any => true
  | s, e => if tyBeq s e then true else
    match e with
    | .union tys => compatAny s tys
    | .optional inner => isCompatibleWith s inner
    | _ => false
  termination_by _ e => sizeOf e
  decreasing_by all_goals (simp_wf <;> omega)
def compatAny (s : Ty) : List Ty → Bool
  | [] => false
  | t :: ts => isCompatibleWith s t || compatAny s ts
  termination_by ts => sizeOf ts
  decreasing_by all_goals (simp_wf <;> omega)
end

theorem tyBeq_refl : ∀ t : Ty, tyBeq t t = true := by
  intro t
  induction t using Ty.rec (motive_2 := fun ts => tyBeq.listBeq ts ts = true) <;>
    simp_all [tyBeq, tyBeq.listBeq]

theorem isCompatibleWith_refl (t : Ty) : isCompatibleWith t t = true := by
  cases t <;> simp [isCompatibleWith, tyBeq_refl]

theorem isCompatibleWith_any (t : Ty) : isCompatibleWith t .any = true := by
  cases t <;> simp [isCompatibleWith]

theorem isCompatibleWith_union_mem (s t : Ty) (tys : List Ty) (h : t ∈ tys)
    (hc : isCompatibleWith s t = true) : isCompatibleWith s (.union tys) = true := by
  have hany : compatAny s tys = true := by
    induction tys with
    | nil => simp at h
    | cons t' ts ih =>
      rw [compatAny]
      rcases List.mem_cons.1 h with rfl | h'
      · simp [hc]
      · simp [ih h']
  cases s <;> simp [isCompatibleWith, hany]

mutual
def canReturnBoolean : Ty → Bool
  | .bool | .null | .any => true
  | .union ts => crbAny ts
  | .optional inner => canReturnBoolean inner
  | _ => false
def crbAny : List Ty → Bool
  | [] => false
  | t :: ts => canReturnBoolean t || crbAny ts
end

mutual
def canReturnEntity : Ty → Bool
  | .node | .rel | .path | .any => true
  | .union ts => creAny ts
  | .optional inner => canReturnEntity inner
  | _ => false
def creAny : List Ty → Bool
  | [] => false
  | t :: ts => canReturnEntity t || creAny ts
end

/-- `can_return_boolean` is sound for values: a value accepted by `t` that is a Bool or
Null implies `t` can return a boolean (so the C-mirroring filter placement check never
drops a boolean-producing expression). -/
theorem canReturnBoolean_of_accepts : ∀ (t : Ty) (v : V F), Accepts v t →
    (v.isNull ∨ ∃ b, v = .bool b) → canReturnBoolean t = true := by
  intro t
  induction t using Ty.rec (motive_2 := fun ts => ∀ ty ∈ ts, ∀ (v : V F), Accepts v ty →
      (v.isNull ∨ ∃ b, v = .bool b) → canReturnBoolean ty = true)
  case union ts ih =>
    intro v ha hv
    cases ha with
    | tag _ _ h => simp [tagMatch_union] at h
    | union _ _ ty hm h' =>
      have := ih ty hm v h' hv
      simp only [canReturnBoolean]
      clear ih h'
      induction ts with
      | nil => simp at hm
      | cons t' ts ih2 =>
        rw [crbAny]; rcases List.mem_cons.1 hm with rfl | hm'
        · simp [this]
        · simp [ih2 hm']
  case optional ty ih =>
    intro v ha hv; cases ha with
    | tag _ _ h => simp [tagMatch_opt] at h
    | opt _ _ h' => simpa [canReturnBoolean] using ih v h' hv
  case list ty _ =>
    intro v ha hv; cases ha with
    | tag _ _ h => simp [tagMatch_list] at h
    | list _ _ _ => simp [V.isNull] at hv
  case nil ty h _ _ _ => simp at h
  case cons ty rest ih1 ih2 t ht v ha hv =>
    rcases List.mem_cons.1 ht with rfl | h
    · exact ih1 v ha hv
    · exact ih2 t h v ha hv
  all_goals
    intro v ha hv
    cases ha
    all_goals first
      | rfl
      | (rcases hv with hv | ⟨b, rfl⟩ <;> (try cases v) <;> simp_all [V.isNull, tagMatch, canReturnBoolean])

/-! ## `Display for Type` (mod.rs:610): "A", "A or B", "A, B, or C" -/

def tyName : Ty → String
  | .null => "Null" | .bool => "Boolean" | .int => "Integer" | .float => "Float"
  | .string => "String" | .list _ => "List" | .map => "Map" | .node => "Node"
  | .rel => "Edge" | .path => "Path" | .vecf32 => "VecF32" | .point => "Point"
  | .datetime => "Datetime" | .date => "Date" | .time => "Time" | .duration => "Duration"
  | .any => "Any" | .union _ => "<union>" | .optional _ => "<opt>"

/-- The union arm, loop for loop: first, then `len - 2` middles with ", ", then the last
with an Oxford comma when there are more than two. -/
def unionDisplay (names : List String) : String :=
  match names with
  | [] => ""
  | first :: rest =>
    let mids := rest.take (names.length - 2)
    let lasts := rest.drop (names.length - 2)
    first ++ String.join (mids.map (", " ++ ·)) ++
      (match lasts with
       | last :: _ => (if names.length > 2 then "," else "") ++ " or " ++ last
       | [] => "")

theorem unionDisplay_one (a : String) : unionDisplay [a] = a := by
  simp [unionDisplay]

theorem unionDisplay_two (a b : String) : unionDisplay [a, b] = a ++ " or " ++ b := by
  simp [unionDisplay, String.append_assoc]

theorem unionDisplay_three (a b c : String) : unionDisplay [a, b, c] = a ++ ", " ++ b ++ ", or " ++ c := by
  simp [unionDisplay, String.append_assoc]

/-- The live message for `abs('a')` is produced by this rendering. -/
example : unionDisplay (["Integer", "Float", "Null"]) = "Integer, Float, or Null" := by decide

/-! ## Registry (`Functions`, mod.rs:907-1087) -/

inductive FnKind where
  | function | internal | procedure | aggregation | udf
  deriving DecidableEq

/-- `PartialEq for FnType` ignores payloads: it is equality of the constructor. -/
def fnTypeEq (a b : FnKind) : Bool := decide (a = b)

structure Entry where
  name : String
  kind : FnKind
  args : FnArgs

/-- Registry: association list keyed by the lowered name. `lower` is Rust's
`str::to_lowercase`, taken as an idempotent function. -/
structure Registry where
  lower : String → String
  builtins : List (String × Entry)
  udfs : List (String × Entry)

def alookup (m : List (String × Entry)) (k : String) : Option Entry :=
  (m.find? (·.1 = k)).map (·.2)

/-- `Functions::get` (mod.rs:1048): built-in of the same kind, else (for scalar/UDF
lookups) the UDF registry. -/
def regGet (r : Registry) (name : String) (k : FnKind) : Option Entry :=
  let l := r.lower name
  match alookup r.builtins l with
  | some e => if fnTypeEq e.kind k then some e else udfFallback
  | none => udfFallback
where
  udfFallback := if k = .function ∨ k = .udf then alookup r.udfs (r.lower name) else none

/-- Lookup is case-insensitive. -/
theorem regGet_lower (r : Registry) (hidem : ∀ s, r.lower (r.lower s) = r.lower s)
    (name : String) (k : FnKind) :
    regGet r (r.lower name) k = regGet r name k := by
  simp [regGet, regGet.udfFallback, hidem]

/-- `Functions::is_aggregate(name)` (mod.rs:1079) does **not** lower its argument. -/
def regIsAggregate (r : Registry) (name : String) : Bool :=
  match alookup r.builtins name with
  | some e => e.kind = .aggregation
  | none => false

/-- Latent: with ASCII lowering, `is_aggregate("COUNT")` is false although `get("COUNT",
Aggregation)` finds `count`. No caller today (grep: the only `is_aggregate` uses are the
`GraphFn` method). -/
theorem isAggregate_case_sensitive :
    -- `lower` restricted to the two names used (ASCII `to_lowercase`).
    let r : Registry := { lower := fun s => if s = "COUNT" then "count" else s,
                          builtins := [("count", ⟨"count", .aggregation, .fixed [.any]⟩)],
                          udfs := [] }
    regIsAggregate r "COUNT" = false ∧ (regGet r "COUNT" .aggregation).isSome := by
  decide

end ValueMath
