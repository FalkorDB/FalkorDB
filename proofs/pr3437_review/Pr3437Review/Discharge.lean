import FalkorExprSemantics.Eval
import Pr3437Review.ListLoop
import Pr3437Review.Classifier
import Pr3437Review.Bridge

/-!
# PR #3437: discharging `Body.Drops` and `RowAgree` from proofs/expr_semantics' `ExprEval`

The per-element / per-row evaluator is proofs/expr_semantics' `FalkorExpr.Eval.evalNode`
(`ExprEval::eval_node`, eval.rs:506 on the PR) over its tree `E`, with the row as
`env : Nat → Option V` (`RowView::value_at`). Values are the engine's `V F`.

* **`Body.Drops` — discharged** (`newBody_drops`). `ListLoop::new` (eval.rs:323-338) drops a
  predicate that is the literal `true` and a projection that is the loop variable; on the
  owned row `env + {var: v}` the owner's evaluator gives `Bool(true)` and `v` for them.
* **Frame property — discharged for the owner's fragment** (`evalNode_frame`): `evalNode`
  depends on the row only through the ids of the `var` nodes in the tree.
* **`RowAgree` for binder-free bodies — discharged** (`rowAgree`): when `vector_plan`
  succeeds on bodies with no nested comprehension / quantifier / `reduce`, the batch row
  (`loop_var_column` at the loop id, `broadcast_column`s at the plan's ids) and the owned row
  agree on every id the bodies read, so the per-row evaluator gives the same result on
  both. No id-uniqueness premise is needed here: with no nested binder the only bound id
  is the loop variable's, which both rows hold.
* **`Col.Sound` for binder-free bodies** (`colSound_binderFree`): with columnar's
  `Dispatch.Lawful` (whose per-row reference is this `evalNode` on the batch row) the
  predicate column is main's per-element values.

**Not dischargeable from the owners' models: bodies that contain a nested
`ListComprehension` / `Quantifier` / `Reduce`.** proofs/expr_semantics models every
`eval_compound` arm as `compound : Nat → List V → Except String V`, a function of the
child *values* only, but those three arms read the row (`to_owned_row` + `insert`,
eval.rs:1176-1240), so `E` has no faithful binder arm to prove a frame property with
binders about. With binders the frame property also needs a *local* scoping fact — no id
read free in the body is bound by a nested binder elsewhere in the body (it would not be
broadcast) — which proofs/binder would have to supply as a theorem about `bindNode`'s
output; its `bindNode_covers` keeps `Env.Covers` through pattern-free expressions, but
`Covers` at the start of the expression is not a binder invariant
(`comprehension_cleanup_breaks`, `where_copy_id_reused`), so it is not provable from the
current binder model either.
-/

namespace Pr3437.Discharge
open FalkorCore FalkorExpr.Eval Pr3437.ListLoop Pr3437.Classifier

variable {F : Type}

/-! ## Variables of an `E` tree -/

mutual
def varsE : E (V F) → List Nat
  | .const _ => []
  | .var id _ => [id]
  | .param _ => []
  | .map kvs => varsL kvs
  | .key _ c => varsE c
  | .prop _ e => varsE e
  | .add a b | .arith _ a b | .eq a b | .neq a b | .cmp _ a b => varsE a ++ varsE b
  | .compound _ cs => varsL cs
def varsL : List (E (V F)) → List Nat
  | [] => []
  | c :: cs => varsE c ++ varsL cs
end

variable (O : Ops (V F)) (ev : Ev (V F)) (c : Nat → List (V F) → Except String (V F))

/-! ## The frame property -/

theorem resolveVar_frame (e1 e2 : Nat → Option (V F)) (id : Nat) (n : String) (h : e1 id = e2 id) :
    resolveVar (some e1) id n = resolveVar (some e2) id n := by
  simp [resolveVar, h]

mutual
/-- **`ExprEval` reads the row only through the variables the expression names.** -/
theorem evalNode_frame (e1 e2 : Nat → Option (V F)) :
    ∀ e : E (V F), (∀ id ∈ varsE e, e1 id = e2 id) →
      evalNode O ev c (some e1) e = evalNode O ev c (some e2) e
  | .const _, _ => rfl
  | .var id n, h => by
    simp only [evalNode]; exact resolveVar_frame e1 e2 id n (h id (by simp [varsE]))
  | .param _, _ => rfl
  | .map kvs, h => by
    simp only [evalNode]; rw [evalKeys_frame e1 e2 kvs (fun id hi => h id (by simpa [varsE] using hi))]
  | .key _ ch, h => by
    simp only [evalNode]; exact evalNode_frame e1 e2 ch (fun id hi => h id (by simpa [varsE] using hi))
  | .prop _ a, h => by
    simp only [evalNode]; rw [evalNode_frame e1 e2 a (fun id hi => h id (by simpa [varsE] using hi))]
  | .add a b, h | .arith _ a b, h | .eq a b, h | .neq a b, h | .cmp _ a b, h => by
    simp only [evalNode]
    rw [evalNode_frame e1 e2 a (fun id hi => h id (by simp [varsE, hi])),
      evalNode_frame e1 e2 b (fun id hi => h id (by simp [varsE, hi]))]
  | .compound _ cs, h => by
    simp only [evalNode]; rw [evalList_frame e1 e2 cs (fun id hi => h id (by simpa [varsE] using hi))]
theorem evalKeys_frame (e1 e2 : Nat → Option (V F)) :
    ∀ kvs : List (E (V F)), (∀ id ∈ varsL kvs, e1 id = e2 id) →
      evalKeys O ev c (some e1) kvs = evalKeys O ev c (some e2) kvs
  | [], _ => rfl
  | .key k ch :: rest, h => by
    simp only [evalKeys]
    rw [evalNode_frame e1 e2 ch (fun id hi => h id (by simp [varsL, varsE, hi])),
      evalKeys_frame e1 e2 rest (fun id hi => h id (by simp [varsL, hi]))]
  | .const _ :: _, _ | .var .. :: _, _ | .param _ :: _, _ | .map _ :: _, _ | .prop .. :: _, _
  | .add .. :: _, _ | .arith .. :: _, _ | .eq .. :: _, _ | .neq .. :: _, _ | .cmp .. :: _, _
  | .compound .. :: _, _ => rfl
theorem evalList_frame (e1 e2 : Nat → Option (V F)) :
    ∀ cs : List (E (V F)), (∀ id ∈ varsL cs, e1 id = e2 id) →
      evalList O ev c (some e1) cs = evalList O ev c (some e2) cs
  | [], _ => rfl
  | ch :: rest, h => by
    simp only [evalList]
    rw [evalNode_frame e1 e2 ch (fun id hi => h id (by simp [varsL, hi])),
      evalList_frame e1 e2 rest (fun id hi => h id (by simp [varsL, hi]))]
end

/-! ## `ListLoop::new` and `Body.Drops` -/

/-- `row.insert(var, value)` on the owned row. -/
def upd (env : Nat → Option (V F)) (vid : Nat) (v : V F) : Nat → Option (V F) :=
  fun j => if j = vid then some v else env j

def isConstTrue : E (V F) → Bool
  | .const (.bool true) => true
  | _ => false

def isLoopVar (vid : Nat) : E (V F) → Bool
  | .var id _ => id == vid
  | _ => false

/-- `ListLoop::new(var, Some(pred), Some(proj))` (eval.rs:323-338) with main's per-element
evaluators: `eval_node` of the node on `env + {var: v}`. -/
def newBody (env : Nat → Option (V F)) (vid : Nat) (pred proj : E (V F)) : Body F where
  hasPred := !isConstTrue pred
  hasProj := !isLoopVar vid proj
  pe v := evalNode O ev c (some (upd env vid v)) pred
  pj v := evalNode O ev c (some (upd env vid v)) proj

/-- **`Body.Drops` holds for the owner's evaluator.** -/
theorem newBody_drops (env : Nat → Option (V F)) (vid : Nat) (pred proj : E (V F)) :
    (newBody O ev c env vid pred proj).Drops where
  predTrue h v := by
    simp only [newBody, Bool.not_eq_false'] at h ⊢
    match pred, h with
    | .const (.bool true), _ => rfl
  projId h v := by
    simp only [newBody, Bool.not_eq_false'] at h ⊢
    match proj, h with
    | .var id n, h =>
      simp only [isLoopVar, beq_iff_eq] at h; subst h
      simp [evalNode, resolveVar, upd]

/-! ## `RowAgree` for binder-free bodies -/

-- An `E` tree as `collect_body_vars` sees it. `E`'s `compound` stands for every other
-- `ExprIR` arm; binder-free means none of them is a nested `ListComprehension` /
-- `Quantifier` / `Reduce` / excluded shape, so it is `other`.
mutual
def toBT : E (V F) → BT
  | .var id _ => .mk (.var id) []
  | .map kvs => .mk .other (toBTL kvs)
  | .key _ ch => .mk .other [toBT ch]
  | .prop _ e => .mk .other [toBT e]
  | .add a b | .arith _ a b | .eq a b | .neq a b | .cmp _ a b => .mk .other [toBT a, toBT b]
  | .compound _ cs => .mk .other (toBTL cs)
  | _ => .mk .other []
def toBTL : List (E (V F)) → List BT
  | [] => []
  | ch :: cs => toBT ch :: toBTL cs
end

mutual
theorem toBT_collect : ∀ (e : E (V F)) (r b : List Nat),
    collect (toBT e) r b = (true, r ++ varsE e, b)
  | .var id _, r, b => by simp [toBT, collect, pre, collectAll, varsE]
  | .map kvs, r, b => by simp only [toBT, collect, pre, varsE]; exact toBTL_collect kvs r b
  | .key _ ch, r, b => by
    simp only [toBT, collect, pre, collectAll, varsE]; rw [toBT_collect ch r b]; try simp [collectAll]
  | .prop _ e, r, b => by
    simp only [toBT, collect, pre, collectAll, varsE]; rw [toBT_collect e r b]; try simp [collectAll]
  | .add x y, r, b | .arith _ x y, r, b | .eq x y, r, b | .neq x y, r, b | .cmp _ x y, r, b => by
    simp only [toBT, collect, pre, collectAll, varsE]
    rw [toBT_collect x r b]; simp only; rw [toBT_collect y _ b]; try simp [collectAll]
  | .compound _ cs, r, b => by simp only [toBT, collect, pre, varsE]; exact toBTL_collect cs r b
  | .const _, r, b | .param _, r, b => by simp [toBT, collect, pre, collectAll, varsE]
theorem toBTL_collect : ∀ (cs : List (E (V F))) (r b : List Nat),
    collectAll (toBTL cs) r b = (true, r ++ varsL cs, b)
  | [], r, b => by simp [toBTL, collectAll, varsL]
  | ch :: cs, r, b => by
    simp only [toBTL, collectAll, varsL]
    rw [toBT_collect ch r b]; simp only; rw [toBTL_collect cs _ b]; simp
end

mutual
theorem varsOf_toBT : ∀ e : E (V F), varsOf (toBT e) = varsE e
  | .var id _ => by simp [toBT, varsOf, varsAll, varsE]
  | .map kvs => by simp only [toBT, varsOf, varsE, List.nil_append]; exact varsAll_toBTL kvs
  | .key _ ch => by simp [toBT, varsOf, varsAll, varsE, varsOf_toBT ch]
  | .prop _ e => by simp [toBT, varsOf, varsAll, varsE, varsOf_toBT e]
  | .add x y | .arith _ x y | .eq x y | .neq x y | .cmp _ x y => by
    simp [toBT, varsOf, varsAll, varsE, varsOf_toBT x, varsOf_toBT y]
  | .compound _ cs => by simp only [toBT, varsOf, varsE, List.nil_append]; exact varsAll_toBTL cs
  | .const _ | .param _ => by simp [toBT, varsOf, varsAll, varsE]
theorem varsAll_toBTL : ∀ cs : List (E (V F)), varsAll (toBTL cs) = varsL cs
  | [] => by simp [toBTL, varsAll, varsL]
  | ch :: cs => by simp [toBTL, varsAll, varsL, varsOf_toBT ch, varsAll_toBTL cs]
end

theorem toBT_map_eq : ∀ es : List (E (V F)), es.map toBT = toBTL es
  | [] => rfl
  | e :: es => by simp [toBTL, toBT_map_eq es]

theorem collectAll_toBT (es : List (E (V F))) (r b : List Nat) :
    collectAll (es.map toBT) r b = (true, r ++ varsL es, b) := by
  rw [toBT_map_eq]; exact toBTL_collect es r b

theorem varsAll_map (es : List (E (V F))) : varsAll (es.map toBT) = varsL es := by
  rw [toBT_map_eq]; exact varsAll_toBTL es

theorem mem_varsL {e : E (V F)} {es : List (E (V F))} (he : e ∈ es) {id : Nat} (hi : id ∈ varsE e) :
    id ∈ varsL es := by
  induction es with
  | nil => simp at he
  | cons x xs ih =>
    simp only [varsL, List.mem_append]
    rcases List.mem_cons.1 he with rfl | h
    · exact .inl hi
    · exact .inr (ih h)

theorem outer_vals {V' : Type} (valueAt : Nat → Option V') :
    ∀ (ids : List Nat) (outer : List (Nat × V')),
      ids.mapM (fun id => (valueAt id).map (id, ·)) = some outer → ∀ p ∈ outer, valueAt p.1 = some p.2
  | [], outer, h, p, hp => by simp at h; subst h; simp at hp
  | x :: xs, outer, h, p, hp => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨y, hy, ys, hys, rfl⟩ := h
    rcases List.mem_cons.1 hp with rfl | hp'
    · cases hv : valueAt x with
      | none => simp [hv] at hy
      | some v => simp only [hv, Option.map_some, Option.some.injEq] at hy; subst hy; exact hv
    · exact outer_vals valueAt xs ys hys p hp'

theorem lookup_of_mem {V' : Type} (outer : List (Nat × V')) (valueAt : Nat → Option V')
    (hall : ∀ p ∈ outer, valueAt p.1 = some p.2) (id : Nat) (v : V') (hm : (id, v) ∈ outer) :
    outer.lookup id = valueAt id := by
  induction outer with
  | nil => simp at hm
  | cons p ps ih =>
    obtain ⟨a, b⟩ := p
    simp only [List.lookup]
    by_cases h : id = a
    · subst h; simp; exact (hall (id, b) (by simp)).symm
    · have hne : (id == a) = false := by simpa using h
      simp only [hne]
      rcases List.mem_cons.1 hm with he | hm'
      · simp at he; exact absurd he.1 h
      · exact ih (fun q hq => hall q (by simp [hq])) hm'

/-- The batch row of `list_loop_columnar` (eval.rs:1563-1567) at row `r`, read through
`BatchRow::value_at`: the loop column, then the broadcast columns. -/
def batchEnv (vid : Nat) (outer : List (Nat × V F)) (chunk : List (V F)) (r : Nat) : Nat → Option (V F) :=
  fun j => if j = vid then some ((Bridge.loopVarColumn chunk).get r)
    else (outer.lookup j).map fun v => (Bridge.broadcastColumn v chunk.length).get r

/-- **`RowAgree`, binder-free bodies.** If `vector_plan` accepted the bodies (none of them
holding a nested binder), the per-row evaluator on the batch row is main's per-element
evaluator on `env + {var: chunk[r]}`, for every body and every row. -/
theorem rowAgree (env : Nat → Option (V F)) (vid : Nat) (bodies : List (E (V F)))
    (outer : List (Nat × V F)) (hplan : vectorPlan vid (bodies.map toBT) env = some outer)
    (chunk : List (V F)) (r : Nat) (hr : r < chunk.length) :
    ∀ e ∈ bodies, evalNode O ev c (some (batchEnv vid outer chunk r)) e =
      evalNode O ev c (some (upd env vid chunk[r])) e := by
  intro e he
  apply evalNode_frame
  intro id hid
  have hb := vectorPlan_broadcasts vid (bodies.map toBT) env outer hplan id
    (by rw [varsAll_map]; exact mem_varsL he hid)
  rw [collectAll_toBT] at hb
  have hall : ∀ p ∈ outer, env p.1 = some p.2 := by
    unfold vectorPlan at hplan; rw [collectAll_toBT] at hplan
    exact outer_vals env _ outer hplan
  simp only [batchEnv, upd]
  by_cases hv : id = vid
  · simp [hv, Bridge.loopVar_get chunk r hr]
  · simp only [hv, ite_false]
    rcases hb with hb | ⟨v, hval, hm⟩
    · simp at hb; exact absurd hb hv
    · rw [lookup_of_mem outer env hall id v hm, hval]
      simp [Bridge.broadcast_get v chunk.length r hr]

/-- **`Col.Sound`'s predicate half for binder-free bodies**: columnar's `eval_sound`, with
its per-row reference being `evalNode` on the batch row, gives main's per-element values. -/
theorem colSound_binderFree {C : Type} (A : Columnar.Dispatch.Arms (V F) C)
    (S : Columnar.Dispatch.RowSem (V F) C) (L : Columnar.Dispatch.Lawful A S)
    (n : Columnar.Dispatch.Node (V F)) (env : Nat → Option (V F)) (vid : Nat) (pred : E (V F))
    (outer : List (Nat × V F)) (hplan : vectorPlan vid [toBT pred] env = some outer)
    (chunk : List (V F))
    (hS : ∀ r, S.rowEval n r = evalNode O ev c (some (batchEnv vid outer chunk r)) pred)
    (col : C) (h : Columnar.Dispatch.eval A n (List.range chunk.length) = .ok col) :
    chunk.mapM (fun v => evalNode O ev c (some (upd env vid v)) pred) =
      .ok ((List.range chunk.length).map (S.get col)) := by
  apply Bridge.colSound_pred A S L n _ chunk .null _ col h
  intro r hr
  rw [hS r, rowAgree O ev c env vid [pred] outer (by simpa using hplan) chunk r hr pred (by simp)]
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hr]

end Pr3437.Discharge
