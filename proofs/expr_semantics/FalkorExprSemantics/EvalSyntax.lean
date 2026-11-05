/-!
# `ExprEval`'s expression tree, operations and arm helpers (runtime/eval.rs:266-330, :1713-1812)

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

/-- The value operations and runtime reads `eval_node` / `eval_compound` call, as structure
fields (laws only where a theorem needs one). -/
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
  /- 2026-10-21: what the AND/OR/XOR/NOT/IN/list/subscript/projection/call arms read. -/
  name : V → String                         -- `Value::name` (value.rs:1387)
  /-- `Bool(b)` ↦ `some (some b)`, `Null` ↦ `some none`, anything else `none`. -/
  boolOf : V → Option (Option Bool)
  mkList : List V → V                       -- `Value::List`
  asList : V → Option (List V)
  asInt : V → Option Int
  asMap : V → Option (List (String × V))    -- `Value::Map`'s entries, in order
  contains : List V → V → V                 -- `Contains::contains` (value.rs:1762)
  debugPair : V → V → String                -- `format!("{v:?}")` of the `(arr, i)` pair
  nodeAttrs : Nat → List (String × V)       -- `rt.get_node_attrs`
  relAttrs : Nat → List (String × V)        -- `rt.get_relationship_attrs`
  distinct : List V → V                     -- `Distinct`'s dedup (runtime state; abstract)
  rtWrite : Bool                            -- `Runtime::write`

/-- `GraphFn` (functions/mod.rs:662) as a `FuncInvocation(Arc<GraphFn>)` node holds it, with
the fields `eval_node` and `VectorEval::eval` read: `call` is `func.func.call(rt, args)` with
the runtime fixed, `validate` is `validate_args_type` (mod.rs:828), `structFn`/`structSlots`
the positional struct-constructor form. Aggregations are not `func` nodes (see `E`). -/
structure Fn (V : Type) where
  name : String
  write : Bool
  nonDeterministic : Bool
  structFn : Option (List V → Except String V)
  structSlots : Nat
  validate : List V → Except String Unit
  call : List V → Except String V

/-- Expression trees (`ExprIR<Variable>`, parser/ast.rs:155). Every arm `eval_node` /
`eval_compound` (eval.rs:340-1134) treats specially that a list-comprehension body can reach
has its own constructor (since 2026-10-21); `compound` is every remaining arm
(`Length`, `GetElements`, `IsNode`, `IsRelationship`, `Negate`, `Pow`, `Add`…`Modulo` and
`Eq`/`Neq` with other than two children, `CompiledRegex`, `ShortestPath`, aggregate calls and
the binders `Quantifier` / `ListComprehension` / `Reduce`, `NestedPlan`, `Pattern`, …),
evaluated by a parameter that receives its children as *thunks* — it may evaluate them
lazily, in any order, or not at all — on the same row. Shapes: `case` is
`Case { has_subject }` with children `[subject?] ++ [List(arms)] ++ [else]`; `func` is a
non-aggregate `FuncInvocation` whose children are not `[Distinct, _]`; `mapProj` is
`MapProjection` with children `[base, items…]`, items `projAll` (a child `MapProjection`),
`projProp` (`Property(name)` without children) and `key` (`Constant(String)` with a child). -/
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
  | and (cs : List (E V))
  | or (cs : List (E V))
  | xor (cs : List (E V))
  | not (c : E V)
  | case (subj : Option (E V)) (arms : List (E V)) (els : E V)
  | func (f : Fn V) (cs : List (E V))
  | inList (a l : E V)
  | list (cs : List (E V))
  | getElement (a i : E V)
  | mapProj (base : E V) (items : List (E V))
  | projAll
  | projProp (attr : String)
  | distinct (cs : List (E V))

variable {V : Type} (O : Ops V)

/-- The catch-all arm's semantics: tag, then the children as thunks. -/
abbrev Compound (V : Type) := Nat → List (Unit → Except String V) → Except String V

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

/-! ## Helpers of the 2026-10-21 arms (non-recursive) -/

/-- The AND/OR/XOR/NOT operand type error (eval.rs:657-662, 680-685, 708-713, 722-726). -/
def boolMismatchMsg (v : V) : String := s!"Type mismatch: expected Boolean or Null but was {O.name v}"

/-- `logical_xor` (eval.rs:1807). -/
def logicalXor (a b : Bool) : Bool := (a && !b) || (!a && b)

/-- The `WHEN` test of `eval_case` (eval.rs:510-519): value form `compare_value == (Equal,
None)`; searched form anything but `false` / `null`. -/
def caseHit (sv : Option V) (w : V) : Bool :=
  match sv with
  | some s => O.cmp w s == (.eq, .none)
  | none => match O.boolOf w with
    | some (some false) | some none => false
    | _ => true

/-- `list_contains` (eval.rs:1760). -/
def listContains (list value : V) : Except String V :=
  match O.asList list with
  | some l => .ok (O.contains l value)
  | none => if O.isNull list then .ok O.null
    else .error s!"Type mismatch: expected List or Null but was {O.name list}"

/-- The `GetElement` match (eval.rs:583-621) on the evaluated operands; `rt` is the
`self.rt()?` of the node / relationship arms. -/
def getElem (rt : Except String Unit) (arr i : V) : Except String V :=
  match O.asList arr with
  | some vs =>
    match O.asInt i with
    | some j =>
      let len : Int := vs.length
      let n := if j < 0 then len + j else j
      if 0 ≤ n ∧ n < len then .ok (vs[n.toNat]?.getD O.null) else .ok O.null
    | none => if O.isNull i then .ok O.null
      else .error s!"Type mismatch: expected Integer but was {O.name i}"
  | none =>
    match O.asNode arr, O.asRel arr, O.asMap arr, O.asStr i with
    | some id, _, _, some key => do let _ ← rt; .ok ((O.nodeAttr id key).getD O.null)
    | none, some r, _, some key => do let _ ← rt; .ok ((O.relAttr r key).getD O.null)
    | none, none, some m, some key => .ok ((m.lookup key).getD O.null)
    | none, none, some _, none => if O.isNull i then .ok O.null
      else .error s!"Type mismatch: unexpected types {O.debugPair arr i}"
    | _, _, _, _ => if O.isNull arr then .ok O.null
      else .error s!"Type mismatch: unexpected types {O.debugPair arr i}"

/-- The common call path of the `FuncInvocation` arm (eval.rs:963-985): validate, the
read-only check, call. -/
def callFn (f : Fn V) (vs : List V) : Except String V := do
  let _ ← f.validate vs
  if !O.rtWrite && f.write then .error "graph.RO_QUERY is to be executed only on read-only queries"
  else f.call vs

/-- `Constant(Value::Null)` (the struct path skips these, eval.rs:932). -/
def isNullConst : E V → Bool
  | .const v => O.isNull v
  | _ => false

/-- A map-projection item's value of property `a` (eval.rs:1624-1640). -/
def projGet (b : V) (a : String) : V :=
  match O.asNode b, O.asRel b, O.asMap b with
  | some id, _, _ => (O.nodeAttr id a).getD O.null
  | none, some r, _ => (O.relAttr r a).getD O.null
  | none, none, some m => (m.lookup a).getD O.null
  | none, none, none => O.null

/-- The `.*` item (eval.rs:1602-1622): every attribute / entry, in order. -/
def projAllOf (b : V) : List (String × V) :=
  match O.asNode b, O.asRel b, O.asMap b with
  | some id, _, _ => O.nodeAttrs id
  | none, some r, _ => O.relAttrs r
  | none, none, some m => m
  | none, none, none => []

def projUnhandled : String := "Encountered unhandled type evaluating map projection"

end FalkorExpr.Eval
