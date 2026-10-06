/-!
# `ExprEval` entry points and fast paths (runtime/eval.rs:86-330, :1098-1260, :1529-1610, :1762)

The expression tree is `E`; values are abstract (`Ops` lists every value operation the
code calls — `+ - * / %`, `compare_value`, `get_attr`, the runtime's attribute reads,
regex — as structure fields, with laws only where a theorem needs one). Compound operators
(the `eval_compound` stack machine, modelled elsewhere in this project) are a parameter
`compound`, so every theorem here holds whatever it does.
-/
namespace FalkorExpr.Eval

/-- `DisjointOrNull` -/
inductive Flag | disjoint | comparedNull | nan | none
  deriving DecidableEq

inductive CmpOp | lt | gt | le | ge
inductive Arith | sub | mul | div | rem

/-- The value operations used by `eval_node`'s fast paths. -/
structure Ops (V : Type) where
  null : V
  bool : Bool → V
  str : String → V
  isNull : V → Bool
  asStr : V → Option String
  asNode : V → Option Nat
  asRel : V → Option Nat
  mkMap : List (String × V) → V
  add : V → V → Except String V
  arith : Arith → V → V → Except String V
  cmp : V → V → Ordering × Flag
  allEq : V → V → Except String V
  allNeq : V → V → Except String V
  getAttr : V → String → Except String V
  nodeAttr : Nat → String → Option V        -- `rt.get_node_attribute`
  relAttr : Nat → String → Option V         -- `rt.get_relationship_attribute`

/-- Expression trees (`ExprIR<Variable>` restricted to the arms dispatched here; every
other arm is `compound`). -/
inductive E (V : Type) where
  | const (v : V)
  | var (id : Nat) (name : String)
  | param (name : String)
  | map (kvs : List (E V))          -- children: `Constant(String)` key nodes with one child
  | key (k : V) (child : E V)       -- a map-literal key node
  | prop (attr : String) (e : E V)
  | add (a b : E V)
  | arith (op : Arith) (a b : E V)
  | eq (a b : E V)
  | neq (a b : E V)
  | cmp (op : CmpOp) (a b : E V)
  | compound (tag : Nat) (cs : List (E V))

variable {V : Type} (O : Ops V)

/-- `ExprEval { runtime: Option<&Runtime> }`: here the runtime is its parameter map. -/
structure Ev (V : Type) where
  runtime : Option (String → Option V)

/-- `from_runtime` (eval.rs:266) and `constant` (:273). -/
def Ev.fromRuntime (params : String → Option V) : Ev V := ⟨some params⟩
def Ev.constant : Ev V := ⟨none⟩

/-- `rt()` (eval.rs:278). -/
def Ev.rt (ev : Ev V) : Except String (String → Option V) :=
  match ev.runtime with
  | some r => .ok r
  | none => .error "not a constant expression"

theorem rt_cases (p : String → Option V) :
    (Ev.fromRuntime p).rt = .ok p ∧ (Ev.constant : Ev V).rt = .error "not a constant expression" :=
  ⟨rfl, rfl⟩

/-- `resolve_var` (eval.rs:287): the row's value, else "Variable {name} not found". -/
def resolveVar (env : Option (Nat → Option V)) (id : Nat) (name : String) : Except String V :=
  match env.bind (fun e => e id) with
  | some v => .ok v
  | none => .error s!"Variable {name} not found"

theorem resolveVar_spec (env : Option (Nat → Option V)) (id : Nat) (name : String) :
    (∀ v, resolveVar env id name = .ok v ↔ ∃ e, env = some e ∧ e id = some v) ∧
    (env = none → resolveVar env id name = .error s!"Variable {name} not found") := by
  constructor
  · intro v; unfold resolveVar
    cases env with
    | none => simp
    | some e => cases h : e id <;> simp [h]
  · intro h; subst h; rfl

def cmpResult (op : CmpOp) (r : Ordering × Flag) : V :=
  match r.2 with
  | .comparedNull | .disjoint => O.null
  | .nan => O.bool false
  | .none => O.bool (match op with
    | .lt => r.1 == .lt | .gt => r.1 == .gt | .le => r.1 != .gt | .ge => r.1 != .lt)

mutual
/-- `eval_node` (eval.rs:330): the fast-path arms in source order, else `eval_compound`. -/
def evalNode (ev : Ev V) (compound : Nat → List V → Except String V)
    (env : Option (Nat → Option V)) : E V → Except String V
  | .const v => .ok v
  | .var id n => resolveVar env id n
  | .param x => do
    let rt ← ev.rt
    match rt x with
    | some v => .ok v
    | none => .error s!"Parameter {x} not found"
  | .map kvs => do
    let kv ← evalKeys ev compound env kvs
    .ok (O.mkMap kv)
  | .key _ c => evalNode ev compound env c
  | .prop attr e => do
    let obj ← evalNode ev compound env e
    match O.asNode obj, O.asRel obj with
    | some id, _ => do let _ ← ev.rt; .ok ((O.nodeAttr id attr).getD O.null)
    | none, some r => do let _ ← ev.rt; .ok ((O.relAttr r attr).getD O.null)
    | none, none => O.getAttr obj attr
  | .add a b => do
    let l ← evalNode ev compound env a
    let r ← evalNode ev compound env b
    O.add l r
  | .arith op a b => do
    let l ← evalNode ev compound env a
    let r ← evalNode ev compound env b
    O.arith op l r
  | .eq a b => do
    let l ← evalNode ev compound env a
    let r ← evalNode ev compound env b
    O.allEq l r
  | .neq a b => do
    let l ← evalNode ev compound env a
    let r ← evalNode ev compound env b
    O.allNeq l r
  | .cmp op a b => do
    let l ← evalNode ev compound env a
    let r ← evalNode ev compound env b
    .ok (cmpResult O op (O.cmp l r))
  | .compound t cs => do
    let vs ← evalList ev compound env cs
    compound t vs
/-- Map literal children: each must be a `Constant(String)` key node. -/
def evalKeys (ev : Ev V) (compound : Nat → List V → Except String V)
    (env : Option (Nat → Option V)) : List (E V) → Except String (List (String × V))
  | [] => .ok []
  | .key k c :: rest => match O.asStr k with
    | some s => do
      let v ← evalNode ev compound env c
      let r ← evalKeys ev compound env rest
      .ok ((s, v) :: r)
    | none => .error "Map key must be a string"
  | _ :: _ => .error "Map key must be a string"
def evalList (ev : Ev V) (compound : Nat → List V → Except String V)
    (env : Option (Nat → Option V)) : List (E V) → Except String (List V)
  | [] => .ok []
  | c :: cs => do
    let v ← evalNode ev compound env c
    let r ← evalList ev compound env cs
    .ok (v :: r)
end

/-- `eval_operand` (eval.rs:316): leaves inline, else `eval_node`. -/
def evalOperand (ev : Ev V) (compound : Nat → List V → Except String V)
    (env : Option (Nat → Option V)) : E V → Except String V
  | .const v => .ok v
  | .var id n => resolveVar env id n
  | e => evalNode O ev compound env e

/-- The inline leaf path is exactly the full evaluator. -/
theorem evalOperand_eq (ev : Ev V) (c : Nat → List V → Except String V) (env : Option (Nat → Option V)) (e : E V) :
    evalOperand O ev c env e = evalNode O ev c env e := by
  cases e <;> rfl

/-- `eval` (eval.rs:299): resolve the root once, then `eval_node`. Trees are given by
their root here, so `eval` is `eval_node` of the root. -/
def eval (ev : Ev V) (c : Nat → List V → Except String V) (env : Option (Nat → Option V)) (root : E V) :
    Except String V := evalNode O ev c env root

theorem eval_eq_root (ev : Ev V) (c : Nat → List V → Except String V) (env : Option (Nat → Option V)) (root : E V) : eval O ev c env root = evalNode O ev c env root := rfl

/-- The constant evaluator: parameters and entity properties need a runtime. -/
theorem constant_param (c : Nat → List V → Except String V) env (x : String) :
    evalNode O Ev.constant c env (.param x) = .error "not a constant expression" := rfl

theorem constant_prop_node (c : Nat → List V → Except String V) env (attr : String) (v : V) (id : Nat)
    (h : O.asNode v = some id) :
    evalNode O Ev.constant c env (.prop attr (.const v)) = .error "not a constant expression" := by
  simp [evalNode, h, Ev.constant, Ev.rt, bind, Except.bind]

/-- Comparison fast path: null/disjoint operands give `null`, NaN gives `false`. -/
theorem cmp_fast (ev : Ev V) (c : Nat → List V → Except String V) (env : Option (Nat → Option V)) (op : CmpOp) (a b : V) :
    evalNode O ev c env (.cmp op (.const a) (.const b)) = .ok (cmpResult O op (O.cmp a b)) ∧
    ((O.cmp a b).2 = .comparedNull → cmpResult O op (O.cmp a b) = O.null) ∧
    ((O.cmp a b).2 = .nan → cmpResult O op (O.cmp a b) = O.bool false) := by
  refine ⟨rfl, ?_, ?_⟩ <;> intro h <;> simp [cmpResult, h]

/-- `<=` is the negation of `>` on comparable operands, and `>=` of `<`. -/
theorem cmp_le_not_gt (r : Ordering × Flag) (h : r.2 = .none) :
    cmpResult O .le r = O.bool (r.1 != .gt) ∧ cmpResult O .ge r = O.bool (r.1 != .lt) := by
  simp [cmpResult, h]

/-- Map literal: a non-string key is an error. -/
theorem map_bad_key (ev : Ev V) (c : Nat → List V → Except String V) (env : Option (Nat → Option V)) (k : V) (e : E V) (h : O.asStr k = none) :
    evalNode O ev c env (.map [.key k e]) = .error "Map key must be a string" := by
  simp [evalNode, evalKeys, h, bind, Except.bind]

/-! ## `evaluate_param` (eval.rs:1762) — parameter expressions of `CYPHER p=…` -/

inductive PV where
  | null | int (i : Int) | float (f : Int) | str (s : String) | list (vs : List PV)
  | map (kvs : List (String × PV))

inductive PE where
  | const (v : PV)
  | list (cs : List PE)
  | map (kvs : List (Option String × PE))  -- `none`: a non-string key node
  | neg (e : PE)
  | other

/-- i64 `checked_neg`: fails only on `i64::MIN`. -/
def checkedNeg (i : Int) : Option Int := if i = -2^63 then none else some (-i)

mutual
def evalParam : PE → Except String PV
  | .const v => .ok v
  | .list cs => do let vs ← evalParams cs; .ok (.list vs)
  | .map kvs => do let m ← evalParamMap kvs; .ok (.map m)
  | .neg e => do
    let v ← evalParam e
    match v with
    | .int i => match checkedNeg i with
      | some j => .ok (.int j)
      | none => .error "ArgumentError: integer overflow in unary minus"
    | .float f => .ok (.float (-f))    -- floats abstracted to their negation
    | _ => .ok .null
  | .other => .error "Invalid parameter expression."
def evalParams : List PE → Except String (List PV)
  | [] => .ok []
  | c :: cs => do let v ← evalParam c; let r ← evalParams cs; .ok (v :: r)
def evalParamMap : List (Option String × PE) → Except String (List (String × PV))
  | [] => .ok []
  | (some k, c) :: cs => do let v ← evalParam c; let r ← evalParamMap cs; .ok ((k, v) :: r)
  | (none, _) :: _ => .error "Map parameter key must be a string"
end

/-- Constants and lists of constants evaluate to themselves. -/
theorem evalParams_consts (vs : List PV) : evalParams (vs.map .const) = .ok vs := by
  induction vs with
  | nil => rfl
  | cons v vs ih => simp [evalParams, evalParam, ih, bind, Except.bind]

theorem evalParam_neg (i : Int) :
    evalParam (.neg (.const (.int i))) =
      if i = -2^63 then .error "ArgumentError: integer overflow in unary minus" else .ok (.int (-i)) := by
  by_cases h : i = -2^63
  · subst h; rfl
  · simp only [evalParam, checkedNeg, h, if_false, bind, Except.bind]

theorem evalParam_neg_null : evalParam (.neg (.const (.str "x"))) = .ok .null := rfl

theorem evalParam_badkey (e : PE) : evalParam (.map [(none, e)]) = .error "Map parameter key must be a string" := rfl

end FalkorExpr.Eval
