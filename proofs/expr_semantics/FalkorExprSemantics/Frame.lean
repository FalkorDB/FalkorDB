import FalkorExprSemantics.Eval
/-!
# The frame property of `ExprEval::eval_node` (eval.rs:340-1134)

`evalNode_frame`: the value of an expression depends on the row only through the ids of the
`Variable` nodes it contains (`E.vars`) — every arm reads the row through its children, and
the only leaf that reads it is `Variable` (`resolve_var`, eval.rs:297). Moved here from
proofs/pr3437_review (`Discharge.evalNode_frame`) and extended to every arm of the
2026-10-21 model, the catch-all included (its thunks are the children on the same row).
-/
namespace FalkorExpr.Eval

variable {V : Type}

mutual
/-- The variable ids an expression reads. -/
def E.vars : E V → List Nat
  | .const _ | .param _ | .projAll | .projProp _ => []
  | .var id _ => [id]
  | .map cs | .compound _ cs | .and cs | .or cs | .xor cs | .func _ cs | .list cs
  | .distinct cs => E.varsL cs
  | .key _ ch | .prop _ ch | .not ch => E.vars ch
  | .add a b | .arith _ a b | .eq a b | .neq a b | .cmp _ a b | .inList a b
  | .getElement a b => E.vars a ++ E.vars b
  | .case s arms els => E.varsO s ++ E.varsL arms ++ E.vars els
  | .mapProj b items => E.vars b ++ E.varsL items
def E.varsL : List (E V) → List Nat
  | [] => []
  | ch :: cs => E.vars ch ++ E.varsL cs
def E.varsO : Option (E V) → List Nat
  | none => []
  | some s => E.vars s
end

variable (O : Ops V) (ev : Ev V) (c : Compound V) (e1 e2 : Nat → Option V)

theorem resolveVar_frame (id : Nat) (n : String) (h : e1 id = e2 id) :
    resolveVar (some e1) id n = resolveVar (some e2) id n := by
  simp [resolveVar, h]

/-- Agreement on a variable list. -/
def Agree (e1 e2 : Nat → Option V) (ids : List Nat) : Prop := ∀ id ∈ ids, e1 id = e2 id

theorem Agree.left {e1 e2 : Nat → Option V} {a b : List Nat} (h : Agree e1 e2 (a ++ b)) : Agree e1 e2 a :=
  fun id hi => h id (List.mem_append_left _ hi)
theorem Agree.right {e1 e2 : Nat → Option V} {a b : List Nat} (h : Agree e1 e2 (a ++ b)) : Agree e1 e2 b :=
  fun id hi => h id (List.mem_append_right _ hi)

mutual
/-- **The frame property**: `ExprEval` reads the row only through the variables the
expression names. -/
theorem evalNode_frame : ∀ e : E V, Agree e1 e2 e.vars →
    evalNode O ev c (some e1) e = evalNode O ev c (some e2) e
  | .const _, _ | .param _, _ | .projAll, _ | .projProp _, _ => rfl
  | .var id n, h => by
    simp only [evalNode]; exact resolveVar_frame e1 e2 id n (h id (by simp [E.vars]))
  | .map kvs, h => by
    simp only [evalNode]; rw [evalKeys_frame kvs (by simpa [E.vars] using h)]
  | .key _ ch, h => by
    simp only [evalNode]; exact evalNode_frame ch (by simpa [E.vars] using h)
  | .prop _ a, h => by
    simp only [evalNode]; rw [evalNode_frame a (by simpa [E.vars] using h)]
  | .not a, h => by
    simp only [evalNode]; rw [evalNode_frame a (by simpa [E.vars] using h)]
  | .add a b, h | .arith _ a b, h | .eq a b, h | .neq a b, h | .cmp _ a b, h
  | .inList a b, h | .getElement a b, h => by
    simp only [E.vars] at h
    simp only [evalNode]
    rw [evalNode_frame a h.left, evalNode_frame b h.right]
  | .compound _ cs, h => by
    simp only [evalNode]; rw [evalThunks_frame cs (by simpa [E.vars] using h)]
  | .and cs, h | .or cs, h => by
    simp only [evalNode]; rw [andOrLoop_frame cs (by simpa [E.vars] using h)]
  | .xor cs, h => by
    simp only [evalNode]; rw [xorLoop_frame cs (by simpa [E.vars] using h)]
  | .list cs, h => by
    simp only [evalNode]; rw [evalRev_frame cs (by simpa [E.vars] using h)]
  | .distinct cs, h => by
    simp only [evalNode]; rw [evalList_frame cs (by simpa [E.vars] using h)]
  | .func f cs, h => by
    simp only [E.vars] at h
    have hl := evalList_frame cs h
    have hs := evalSlots_frame cs h
    simp only [evalNode, hl, hs]
  | .case s arms els, h => by
    simp only [E.vars] at h
    have h1 := evalSubj_frame s h.left.left
    have h2 := fun sv => evalArms_frame arms h.left.right sv
    have h3 := evalNode_frame els h.right
    simp only [evalNode, h1, h2, h3]
  | .mapProj b items, h => by
    simp only [E.vars] at h
    have h1 := evalNode_frame b h.left
    have h2 := fun bv => projLoop_frame items h.right bv
    simp only [evalNode, h1, h2]
theorem evalKeys_frame : ∀ kvs : List (E V), Agree e1 e2 (E.varsL kvs) →
    evalKeys O ev c (some e1) kvs = evalKeys O ev c (some e2) kvs
  | [], _ => rfl
  | .key k ch :: rest, h => by
    simp only [E.varsL, E.vars] at h
    simp only [evalKeys]
    rw [evalNode_frame ch h.left, evalKeys_frame rest h.right]
  | .const _ :: _, _ | .var .. :: _, _ | .param _ :: _, _ | .map _ :: _, _ | .prop .. :: _, _
  | .add .. :: _, _ | .arith .. :: _, _ | .eq .. :: _, _ | .neq .. :: _, _ | .cmp .. :: _, _
  | .compound .. :: _, _ | .and _ :: _, _ | .or _ :: _, _ | .xor _ :: _, _ | .not _ :: _, _
  | .case .. :: _, _ | .func .. :: _, _ | .inList .. :: _, _ | .list _ :: _, _
  | .getElement .. :: _, _ | .mapProj .. :: _, _ | .projAll :: _, _ | .projProp _ :: _, _
  | .distinct _ :: _, _ => rfl
theorem evalList_frame : ∀ cs : List (E V), Agree e1 e2 (E.varsL cs) →
    evalList O ev c (some e1) cs = evalList O ev c (some e2) cs
  | [], _ => rfl
  | ch :: rest, h => by
    simp only [E.varsL] at h
    simp only [evalList]
    rw [evalNode_frame ch h.left, evalList_frame rest h.right]
theorem evalThunks_frame : ∀ cs : List (E V), Agree e1 e2 (E.varsL cs) →
    evalThunks O ev c (some e1) cs = evalThunks O ev c (some e2) cs
  | [], _ => rfl
  | ch :: rest, h => by
    simp only [E.varsL] at h
    simp only [evalThunks]
    rw [evalNode_frame ch h.left, evalThunks_frame rest h.right]
theorem evalRev_frame : ∀ cs : List (E V), Agree e1 e2 (E.varsL cs) →
    evalRev O ev c (some e1) cs = evalRev O ev c (some e2) cs
  | [], _ => rfl
  | ch :: rest, h => by
    simp only [E.varsL] at h
    simp only [evalRev]
    rw [evalNode_frame ch h.left, evalRev_frame rest h.right]
theorem andOrLoop_frame : ∀ cs : List (E V), Agree e1 e2 (E.varsL cs) → ∀ sc n,
    andOrLoop O ev c (some e1) sc n cs = andOrLoop O ev c (some e2) sc n cs
  | [], _, _, _ => rfl
  | ch :: rest, h, sc, n => by
    simp only [E.varsL] at h
    have hr := andOrLoop_frame rest h.right
    simp only [andOrLoop, evalNode_frame ch h.left, hr]
theorem xorLoop_frame : ∀ cs : List (E V), Agree e1 e2 (E.varsL cs) → ∀ last,
    xorLoop O ev c (some e1) last cs = xorLoop O ev c (some e2) last cs
  | [], _, _ => rfl
  | ch :: rest, h, last => by
    simp only [E.varsL] at h
    have hr := xorLoop_frame rest h.right
    simp only [xorLoop, evalNode_frame ch h.left, hr]
theorem evalSubj_frame : ∀ s : Option (E V), Agree e1 e2 (E.varsO s) →
    evalSubj O ev c (some e1) s = evalSubj O ev c (some e2) s
  | none, _ => rfl
  | some s, h => by
    simp only [E.varsO] at h
    simp only [evalSubj, evalNode_frame s h]
theorem evalArms_frame : ∀ arms : List (E V), Agree e1 e2 (E.varsL arms) → ∀ sv,
    evalArms O ev c (some e1) sv arms = evalArms O ev c (some e2) sv arms
  | [], _, _ => rfl
  | [_], _, _ => rfl
  | w :: t :: rest, h, sv => by
    simp only [E.varsL] at h
    have hr := evalArms_frame rest h.right.right
    simp only [evalArms, evalNode_frame w h.left, evalNode_frame t h.right.left, hr]
theorem evalSlots_frame : ∀ cs : List (E V), Agree e1 e2 (E.varsL cs) →
    evalSlots O ev c (some e1) cs = evalSlots O ev c (some e2) cs
  | [], _ => rfl
  | ch :: rest, h => by
    simp only [E.varsL] at h
    simp only [evalSlots, evalNode_frame ch h.left, evalSlots_frame rest h.right]
theorem projLoop_frame : ∀ items : List (E V), Agree e1 e2 (E.varsL items) → ∀ b,
    projLoop O ev c (some e1) b items = projLoop O ev c (some e2) b items
  | [], _, _ => rfl
  | .projAll :: is, h, b => by
    simp only [E.varsL, E.vars, List.nil_append] at h
    simp only [projLoop, projLoop_frame is h b]
  | .projProp _ :: is, h, b => by
    simp only [E.varsL, E.vars, List.nil_append] at h
    simp only [projLoop, projLoop_frame is h b]
  | .key k ch :: is, h, b => by
    simp only [E.varsL, E.vars] at h
    simp only [projLoop, evalNode_frame ch h.left, projLoop_frame is h.right b]
  | .const _ :: _, _, _ | .var .. :: _, _, _ | .param _ :: _, _, _ | .map _ :: _, _, _
  | .prop .. :: _, _, _ | .add .. :: _, _, _ | .arith .. :: _, _, _ | .eq .. :: _, _, _
  | .neq .. :: _, _, _ | .cmp .. :: _, _, _ | .compound .. :: _, _, _ | .and _ :: _, _, _
  | .or _ :: _, _, _ | .xor _ :: _, _, _ | .not _ :: _, _, _ | .case .. :: _, _, _
  | .func .. :: _, _, _ | .inList .. :: _, _, _ | .list _ :: _, _, _ | .getElement .. :: _, _, _
  | .mapProj .. :: _, _, _ | .distinct _ :: _, _, _ => rfl
end

end FalkorExpr.Eval
