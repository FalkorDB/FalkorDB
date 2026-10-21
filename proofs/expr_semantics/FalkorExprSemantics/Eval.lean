import FalkorExprSemantics.EvalSyntax
/-!
# `ExprEval::eval_node` (eval.rs:340-1134): the evaluator over `E`

The tree `E`, the operations `Ops`, the function record `Fn` and the non-recursive helpers of
the arms are in `EvalSyntax.lean`; this module is the mutually recursive evaluator and its
entry points.
-/
namespace FalkorExpr.Eval

variable {V : Type} (O : Ops V)

mutual
/-- `eval_node` (eval.rs:340) with `eval_compound` (:535) and `eval_case` (:495): the arms in
source order. -/
def evalNode (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) : E V → Except String V
  | .const v => .ok v
  | .var id n => resolveVar env id n
  | .param x => do
    let rt ← ev.rt
    match rt x with
    | some v => .ok v
    | none => .error s!"Parameter {x} not found"
  | .map kvs => do
    let kv ← evalKeys ev c env kvs
    .ok (O.mkMap kv)
  | .key _ ch => evalNode ev c env ch
  | .prop attr e => do
    let obj ← evalNode ev c env e
    match O.asNode obj, O.asRel obj with
    | some id, _ => do let _ ← ev.rt; .ok ((O.nodeAttr id attr).getD O.null)
    | none, some r => do let _ ← ev.rt; .ok ((O.relAttr r attr).getD O.null)
    | none, none => O.getAttr obj attr
  | .add a b => do
    let l ← evalNode ev c env a
    let r ← evalNode ev c env b
    O.add l r
  | .arith op a b => do
    let l ← evalNode ev c env a
    let r ← evalNode ev c env b
    O.arith op l r
  | .eq a b => do
    let l ← evalNode ev c env a
    let r ← evalNode ev c env b
    O.allEq l r
  | .neq a b => do
    let l ← evalNode ev c env a
    let r ← evalNode ev c env b
    O.allNeq l r
  | .cmp op a b => do
    let l ← evalNode ev c env a
    let r ← evalNode ev c env b
    .ok (cmpResult O op (O.cmp l r))
  | .compound t cs => c t (evalThunks ev c env cs)
  | .and cs => andOrLoop ev c env false false cs
  | .or cs => andOrLoop ev c env true false cs
  | .xor cs => xorLoop ev c env none cs
  | .not ch => do
    let v ← evalNode ev c env ch
    match O.boolOf v with
    | some (some b) => .ok (O.bool !b)
    | some none => .ok O.null
    | none => .error (boolMismatchMsg O v)
  | .case subj arms els => do
    let sv ← evalSubj ev c env subj
    let r ← evalArms ev c env sv arms
    match r with
    | some v => .ok v
    | none => evalNode ev c env els
  | .func f cs => do
    let _ ← ev.rt
    match f.structFn with
    | some sf =>
      if f.structSlots ≠ 0 ∧ cs.length = f.structSlots then do
        let slots ← evalSlots ev c env cs
        sf slots
      else do
        let vs ← evalList ev c env cs
        callFn O f vs
    | none => do
      let vs ← evalList ev c env cs
      callFn O f vs
  | .inList a l => do
    let v ← evalNode ev c env a
    let lv ← evalNode ev c env l
    listContains O lv v
  | .list cs => do
    let vs ← evalRev ev c env cs
    .ok (O.mkList vs)
  | .getElement a i => do
    let av ← evalNode ev c env a
    let iv ← evalNode ev c env i
    getElem O (ev.rt.map fun _ => ()) av iv
  | .mapProj base items => do
    let _ ← ev.rt
    let b ← evalNode ev c env base
    if O.isNull b then .ok O.null
    else if (O.asNode b).isSome || (O.asRel b).isSome || (O.asMap b).isSome then do
      let kvs ← projLoop ev c env b items
      .ok (O.mkMap kvs)
    else .error projUnhandled
  | .projAll | .projProp _ => .error "unreachable: a map-projection item outside a projection"
  | .distinct cs => do
    let _ ← ev.rt
    let vs ← evalList ev c env cs
    .ok (O.distinct vs)
/-- Map literal children: each must be a `Constant(String)` key node (eval.rs:359-368). -/
def evalKeys (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) :
    List (E V) → Except String (List (String × V))
  | [] => .ok []
  | .key k ch :: rest => match O.asStr k with
    | some s => do
      let v ← evalNode ev c env ch
      let r ← evalKeys ev c env rest
      .ok ((s, v) :: r)
    | none => .error "Map key must be a string"
  | _ :: _ => .error "Map key must be a string"
/-- Children left to right, first error wins (`?` per child). -/
def evalList (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) :
    List (E V) → Except String (List V)
  | [] => .ok []
  | ch :: cs => do
    let v ← evalNode ev c env ch
    let r ← evalList ev c env cs
    .ok (v :: r)
/-- The catch-all's children as thunks on the same row. -/
def evalThunks (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) :
    List (E V) → List (Unit → Except String V)
  | [] => []
  | ch :: cs => (fun _ => evalNode ev c env ch) :: evalThunks ev c env cs
/-- A list literal's elements (eval.rs:563-578): pushed in order onto the work stack, so
evaluated **last to first**; the first error met (the rightmost) wins. -/
def evalRev (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) :
    List (E V) → Except String (List V)
  | [] => .ok []
  | ch :: cs => do
    let r ← evalRev ev c env cs
    let v ← evalNode ev c env ch
    .ok (v :: r)
/-- `Or` (eval.rs:639-666, `sc = true`) / `And` (:690-717, `sc = false`): children in order,
the first `Bool(sc)` settles it (later children are not evaluated), `Null` is remembered,
anything else is a type error. -/
def andOrLoop (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) (sc seenNull : Bool) :
    List (E V) → Except String V
  | [] => .ok (if seenNull then O.null else O.bool !sc)
  | ch :: cs => do
    let v ← evalNode ev c env ch
    match O.boolOf v with
    | some (some b) => if b = sc then .ok (O.bool sc) else andOrLoop ev c env sc seenNull cs
    | some none => andOrLoop ev c env sc true cs
    | none => .error (boolMismatchMsg O v)
/-- `Xor` (eval.rs:667-689): a `Null` child settles it as `Null`. -/
def xorLoop (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) (last : Option Bool) :
    List (E V) → Except String V
  | [] => .ok (O.bool (last.getD false))
  | ch :: cs => do
    let v ← evalNode ev c env ch
    match O.boolOf v with
    | some (some b) => xorLoop ev c env (some (match last with | none => b | some l => logicalXor l b)) cs
    | some none => .ok O.null
    | none => .error (boolMismatchMsg O v)
/-- `eval_case`'s subject (eval.rs:501-505). -/
def evalSubj (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) :
    Option (E V) → Except String (Option V)
  | none => .ok none
  | some s => do let v ← evalNode ev c env s; .ok (some v)
/-- `eval_case`'s arm loop (eval.rs:508-523, `while i + 1 < num_arms`): `none` = no arm
matched (then `ELSE`); a `THEN` is evaluated only for the arm that matched. -/
def evalArms (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) (sv : Option V) :
    List (E V) → Except String (Option V)
  | w :: t :: rest => do
    let wv ← evalNode ev c env w
    if caseHit O sv wv then do let v ← evalNode ev c env t; .ok (some v)
    else evalArms ev c env sv rest
  | _ => .ok none
/-- The struct-constructor slots (eval.rs:926-939): `Constant(Null)` children are not
evaluated. -/
def evalSlots (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) :
    List (E V) → Except String (List V)
  | [] => .ok []
  | ch :: cs => do
    let v ← if isNullConst O ch then .ok O.null else evalNode ev c env ch
    let r ← evalSlots ev c env cs
    .ok (v :: r)
/-- `eval_map_projection`'s item loop (eval.rs:1600-1655), as the insert sequence. -/
def projLoop (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) (b : V) :
    List (E V) → Except String (List (String × V))
  | [] => .ok []
  | .projAll :: is => do let r ← projLoop ev c env b is; .ok (projAllOf O b ++ r)
  | .projProp a :: is => do let r ← projLoop ev c env b is; .ok ((a, projGet O b a) :: r)
  | .key k ch :: is => match O.asStr k with
    | some s => do
      let v ← evalNode ev c env ch
      let r ← projLoop ev c env b is
      .ok ((s, v) :: r)
    | none => .error projUnhandled
  | _ :: _ => .error projUnhandled
end

/-- `eval_operand` (eval.rs:316): leaves inline, else `eval_node`. -/
def evalOperand (ev : Ev V) (compound : Compound V)
    (env : Option (Nat → Option V)) : E V → Except String V
  | .const v => .ok v
  | .var id n => resolveVar env id n
  | e => evalNode O ev compound env e

/-- The inline leaf path is exactly the full evaluator. -/
theorem evalOperand_eq (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) (e : E V) :
    evalOperand O ev c env e = evalNode O ev c env e := by
  cases e <;> rfl

/-- `eval` (eval.rs:299): resolve the root once, then `eval_node`. Trees are given by
their root here, so `eval` is `eval_node` of the root. -/
def eval (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) (root : E V) :
    Except String V := evalNode O ev c env root

theorem eval_eq_root (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) (root : E V) : eval O ev c env root = evalNode O ev c env root := rfl

/-- The constant evaluator: parameters and entity properties need a runtime. -/
theorem constant_param (c : Compound V) env (x : String) :
    evalNode O Ev.constant c env (.param x) = .error "not a constant expression" := rfl

theorem constant_prop_node (c : Compound V) env (attr : String) (v : V) (id : Nat)
    (h : O.asNode v = some id) :
    evalNode O Ev.constant c env (.prop attr (.const v)) = .error "not a constant expression" := by
  simp [evalNode, h, Ev.constant, Ev.rt, bind, Except.bind]

/-- Comparison fast path: null/disjoint operands give `null`, NaN gives `false`. -/
theorem cmp_fast (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) (op : CmpOp) (a b : V) :
    evalNode O ev c env (.cmp op (.const a) (.const b)) = .ok (cmpResult O op (O.cmp a b)) ∧
    ((O.cmp a b).2 = .comparedNull → cmpResult O op (O.cmp a b) = O.null) ∧
    ((O.cmp a b).2 = .nan → cmpResult O op (O.cmp a b) = O.bool false) := by
  refine ⟨rfl, ?_, ?_⟩ <;> intro h <;> simp [cmpResult, h]

/-- `<=` is the negation of `>` on comparable operands, and `>=` of `<`. -/
theorem cmp_le_not_gt (r : Ordering × Flag) (h : r.2 = .none) :
    cmpResult O .le r = O.bool (r.1 != .gt) ∧ cmpResult O .ge r = O.bool (r.1 != .lt) := by
  simp [cmpResult, h]

/-- Map literal: a non-string key is an error. -/
theorem map_bad_key (ev : Ev V) (c : Compound V) (env : Option (Nat → Option V)) (k : V) (e : E V) (h : O.asStr k = none) :
    evalNode O ev c env (.map [.key k e]) = .error "Map key must be a string" := by
  simp [evalNode, evalKeys, h, bind, Except.bind]

end FalkorExpr.Eval
