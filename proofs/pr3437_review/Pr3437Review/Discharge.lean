import FalkorExprSemantics.EvalArms
import Pr3437Review.ListLoop
import Pr3437Review.Classifier
import Pr3437Review.Bridge

/-!
# PR #3437: discharging `Body.Drops` and `RowAgree` from proofs/expr_semantics' `ExprEval`

The per-element / per-row evaluator is proofs/expr_semantics' `FalkorExpr.Eval.evalNode`
(`ExprEval::eval_node`, eval.rs:503 on the PR) over its tree `E`, with the row as
`env : Nat → Option V` (`RowView::value_at`). Values are the engine's `V F`.

* **`Body.Drops` — discharged** (`newBody_drops`). `ListLoop::new` (eval.rs:323-338) drops a
  predicate that is the literal `true` and a projection that is the loop variable; on the
  owned row `env + {var: v}` the owner's evaluator gives `Bool(true)` and `v` for them.
* **Frame property — discharged for the owner's fragment** (`evalNode_frame`): `evalNode`
  depends on the row only through the ids of the `var` nodes in the tree.
* **`RowAgree` — discharged for every body `vector_plan` accepts** (`rowAgree`, a739b990f):
  an accepted body binds nothing but the loop variable (`Classifier.vectorPlan_broadcasts`),
  so the batch row (`loop_var_column` at the loop id, `broadcast_column`s at the plan's ids)
  and the owned row agree on every id the body reads; no id-uniqueness premise. Before
  a739b990f nested binders were accepted and this needed a binder scoping theorem
  (`ClassifierPre`, `Classifier.pre_a739b990f_accepts_nested_binder`).
* **`Col.Sound`'s predicate half** (`colSound_accepted`) from columnar's `Dispatch.eval_sound`,
  **given `hS`**: columnar's abstract per-row reference `RowSem.rowEval` *is* this `evalNode`
  on the batch row.

**`hS` is not dischargeable from the owners' models** (`Dispatch.Lawful` would have to be
instantiated with `RowSem.rowEval := evalNode` and every field proved):
1. `Dispatch.Lawful` has no concrete instance anywhere: columnar proves each lane against
   its own operator-level reference (`rowCmp`, `valueArith`, `notRow`, `caseRow`,
   `foldCall`, `VExpr.colEval`) over its own value type `Columnar.V F` (seven constructors,
   abstract `FloatModel`), not over a tree evaluator, and never assembles them.
2. expr_semantics' `E` has a faithful arm only for `const`, `var`, `param`, map literals,
   `prop`, `+`, `- * / %`, `=`, `<>`, `< > <= >=`; every other arm is the eager, value-only
   `compound`. That cannot express `AND` / `OR` / `CASE`, which short-circuit
   (`compound_cannot_short_circuit`; live on main: `RETURN false AND 1/0 = 1` → `false`,
   `true OR 1/0 = 1` → `true`, `CASE WHEN true THEN 1 ELSE 1/0 END` → `1`). The classifier
   accepts all three, so they are not routed per element.
3. The comparison and arithmetic arms that both projects do model are over different value
   types (`Columnar.V F` vs expr_semantics' abstract `Ops V`), and no owner relates
   `all_equals` / `all_not_equals` (expr_semantics `evalCmp`) to columnar's
   `compareValues` / `rowCmp`, or `O.arith` to `valueArith`.
Needed: a tree-level per-row evaluator shared by both projects (or a translation
`Columnar.V F ↔ FalkorCore.V F` plus `E` arms for AND/OR/NOT/CASE/function calls with
their scalar semantics), and an instance of `Dispatch.Lawful` built from columnar's lanes.
-/

namespace Pr3437.Discharge
open FalkorCore FalkorExpr.Eval Pr3437.ListLoop Pr3437.Classifier

variable {F : Type}

variable (O : Ops (V F)) (ev : Ev (V F)) (c : Compound (V F))

/-! The variables of an `E` tree (`E.vars`) and the frame property (`evalNode_frame`) are
proofs/expr_semantics' (`FalkorExprSemantics.Frame`, every arm of the 2026-10-21 model). -/

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

/-! ## `RowAgree` for every body the classifier accepts -/

-- An `E` tree as `collect_body_vars` sees it (children in `ExprIR` order: `Case` is
-- `[subject?, List(arms), else]`, `MapProjection` is `[base, items…]`, a map key / projection key
-- is a `Constant(String)` node with one child). `E.compound t` stands for every other `ExprIR`
-- arm; `isBad t` says whether arm `t` is one `collect_body_vars` rejects (a nested binder,
-- nested plan, pattern, aggregate call): it maps to a rejected kind, everything else to
-- `other`. A `func` node is a non-aggregate call, rejected iff it writes; `distinct` is `Distinct`.
open Pr3437.ClassifierPre (BK BT varsOf varsAll) in
mutual
def toBT (isBad : Nat → Bool) : E (V F) → BT
  | .var id _ => .mk (.var id) []
  | .const _ | .param _ | .projAll | .projProp _ => .mk .other []
  | .map cs | .and cs | .or cs | .xor cs | .list cs => .mk .other (toBTL isBad cs)
  | .compound t cs => .mk (if isBad t then .distinct else .other) (toBTL isBad cs)
  | .func f cs => .mk (.func f.write false) (toBTL isBad cs)
  | .distinct cs => .mk .distinct (toBTL isBad cs)
  | .key _ ch | .prop _ ch | .not ch => .mk .other [toBT isBad ch]
  | .add a b | .arith _ a b | .eq a b | .neq a b | .cmp _ a b | .inList a b
  | .getElement a b => .mk .other [toBT isBad a, toBT isBad b]
  | .case s arms els => .mk .other (toBTO isBad s ++ [.mk .other (toBTL isBad arms), toBT isBad els])
  | .mapProj b items => .mk .other (toBT isBad b :: toBTL isBad items)
def toBTL (isBad : Nat → Bool) : List (E (V F)) → List BT
  | [] => []
  | ch :: cs => toBT isBad ch :: toBTL isBad cs
def toBTO (isBad : Nat → Bool) : Option (E (V F)) → List BT
  | none => []
  | some s => [toBT isBad s]
end

theorem varsAll_append : ∀ a b : List ClassifierPre.BT,
    ClassifierPre.varsAll (a ++ b) = ClassifierPre.varsAll a ++ ClassifierPre.varsAll b
  | [], _ => rfl
  | x :: xs, b => by simp [ClassifierPre.varsAll, varsAll_append xs b]

open Pr3437.ClassifierPre (varsOf varsAll) in
mutual
theorem varsOf_toBT (isBad : Nat → Bool) : ∀ e : E (V F), varsOf (toBT isBad e) = e.vars
  | .var id _ => by simp [toBT, varsOf, varsAll, E.vars]
  | .const _ | .param _ | .projAll | .projProp _ => by simp [toBT, varsOf, varsAll, E.vars]
  | .map cs | .and cs | .or cs | .xor cs | .list cs | .distinct cs => by
    simp only [toBT, varsOf, E.vars, List.nil_append]; exact varsAll_toBTL isBad cs
  | .func f cs => by simp only [toBT, varsOf, E.vars, List.nil_append]; exact varsAll_toBTL isBad cs
  | .compound t cs => by
    have h := varsAll_toBTL isBad cs
    cases hb : isBad t <;> simp [toBT, varsOf, E.vars, hb, h]
  | .key _ ch | .prop _ ch | .not ch => by simp [toBT, varsOf, varsAll, E.vars, varsOf_toBT isBad ch]
  | .add x y | .arith _ x y | .eq x y | .neq x y | .cmp _ x y | .inList x y | .getElement x y => by
    simp [toBT, varsOf, varsAll, E.vars, varsOf_toBT isBad x, varsOf_toBT isBad y]
  | .case s arms els => by
    simp [toBT, varsOf, varsAll, varsAll_append, E.vars, varsAll_toBTO isBad s, varsAll_toBTL isBad arms,
      varsOf_toBT isBad els]
  | .mapProj b items => by
    simp [toBT, varsOf, varsAll, E.vars, varsOf_toBT isBad b, varsAll_toBTL isBad items]
theorem varsAll_toBTL (isBad : Nat → Bool) : ∀ cs : List (E (V F)), varsAll (toBTL isBad cs) = E.varsL cs
  | [] => by simp [toBTL, varsAll, E.varsL]
  | ch :: cs => by simp [toBTL, varsAll, E.varsL, varsOf_toBT isBad ch, varsAll_toBTL isBad cs]
theorem varsAll_toBTO (isBad : Nat → Bool) : ∀ s : Option (E (V F)), varsAll (toBTO isBad s) = E.varsO s
  | none => by simp [toBTO, varsAll, E.varsO]
  | some s => by simp [toBTO, varsAll, E.varsO, varsOf_toBT isBad s]
end

theorem varsAll_map (isBad : Nat → Bool) : ∀ es : List (E (V F)),
    ClassifierPre.varsAll (es.map (toBT isBad)) = E.varsL es
  | [] => rfl
  | e :: es => by
    simp [ClassifierPre.varsAll, E.varsL, varsOf_toBT isBad e, varsAll_map isBad es]

theorem mem_varsL {e : E (V F)} {es : List (E (V F))} (he : e ∈ es) {id : Nat} (hi : id ∈ e.vars) :
    id ∈ E.varsL es := by
  induction es with
  | nil => simp at he
  | cons x xs ih =>
    simp only [E.varsL, List.mem_append]
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

/-- The batch row of `list_loop_columnar` (eval.rs:1560-1564) at row `r`, read through
`BatchRow::value_at`: the loop column, then the broadcast columns. -/
def batchEnv (vid : Nat) (outer : List (Nat × V F)) (chunk : List (V F)) (r : Nat) : Nat → Option (V F) :=
  fun j => if j = vid then some ((Bridge.loopVarColumn chunk).get r)
    else (outer.lookup j).map fun v => (Bridge.broadcastColumn v chunk.length).get r

/-- **`RowAgree`.** If `vector_plan` accepted the bodies, the per-row evaluator on the batch
row is main's per-element evaluator on `env + {var: chunk[r]}`, for every body and every row.
No premise about the bodies: since a739b990f an accepted body binds nothing but the loop
variable (`vectorPlan_broadcasts`). -/
theorem rowAgree (isBad : Nat → Bool) (env : Nat → Option (V F)) (vid : Nat) (bodies : List (E (V F)))
    (outer : List (Nat × V F)) (hplan : vectorPlan vid (bodies.map (toBT isBad)) env = some outer)
    (chunk : List (V F)) (r : Nat) (hr : r < chunk.length) :
    ∀ e ∈ bodies, evalNode O ev c (some (batchEnv vid outer chunk r)) e =
      evalNode O ev c (some (upd env vid chunk[r])) e := by
  intro e he
  apply evalNode_frame
  intro id hid
  have hb := vectorPlan_broadcasts vid (bodies.map (toBT isBad)) env outer hplan id
    (by rw [varsAll_map]; exact mem_varsL he hid)
  have hall := vectorPlan_vals vid (bodies.map (toBT isBad)) env outer hplan
  simp only [batchEnv, upd]
  by_cases hv : id = vid
  · simp [hv, Bridge.loopVar_get chunk r hr]
  · simp only [hv, ite_false]
    rcases hb with hb | ⟨v, hval, hm⟩
    · exact absurd hb hv
    · rw [lookup_of_mem outer env hall id v hm, hval]
      simp [Bridge.broadcast_get v chunk.length r hr]

/-- **`Col.Sound`'s predicate half for every accepted body**: columnar's `eval_sound`, with
its per-row reference being `evalNode` on the batch row, gives main's per-element values. -/
theorem colSound_accepted {C : Type} (A : Columnar.Dispatch.Arms (V F) C)
    (S : Columnar.Dispatch.RowSem (V F) C) (L : Columnar.Dispatch.Lawful A S)
    (n : Columnar.Dispatch.Node (V F)) (isBad : Nat → Bool) (env : Nat → Option (V F)) (vid : Nat) (pred : E (V F))
    (outer : List (Nat × V F)) (hplan : vectorPlan vid [toBT isBad pred] env = some outer)
    (chunk : List (V F))
    (hS : ∀ r, S.rowEval n r = evalNode O ev c (some (batchEnv vid outer chunk r)) pred)
    (col : C) (h : Columnar.Dispatch.eval A n (List.range chunk.length) = .ok col) :
    chunk.mapM (fun v => evalNode O ev c (some (upd env vid v)) pred) =
      .ok ((List.range chunk.length).map (S.get col)) := by
  apply Bridge.colSound_pred A S L n _ chunk .null _ col h
  intro r hr
  rw [hS r, rowAgree O ev c isBad env vid [pred] outer (by simpa using hplan) chunk r hr pred (by simp)]
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hr]

end Pr3437.Discharge

namespace Pr3437.Discharge
open FalkorCore FalkorExpr.Eval

/-- An eager catch-all: the children evaluated left to right first, then a value-only
function — the shape `compound` had before 2026-10-21. -/
def eagerC {F : Type} (g : Nat → List (V F) → Except String (V F)) : Compound (V F) :=
  fun t ths => (ths.mapM fun th => th ()) >>= g t

/-- **History (closed 2026-10-21): why the old `E` could not model `AND`/`OR`/`CASE`.**
Whatever value-only semantics an eager catch-all gives an operator, an erroring operand is
reported before the operator runs — so no choice yields Rust's `false AND <error>` =
`false`. `E` now has faithful `and` / `or` / `case` arms
(`FalkorExpr.Eval.and_false_short_circuits`, `case_hit_lazy`) and its catch-all takes thunks. -/
theorem pre20261021_compound_cannot_short_circuit {F : Type} (O : Ops (V F)) (ev : Ev (V F))
    (g : Nat → List (V F) → Except String (V F)) (env : Option (Nat → Option (V F)))
    (t : Nat) (a : V F) (bad : E (V F)) (e : String) (hbad : evalNode O ev (eagerC g) env bad = .error e) :
    evalNode O ev (eagerC g) env (.compound t [.const a, bad]) = .error e := by
  simp [evalNode, evalThunks, eagerC, hbad, bind, Except.bind, List.mapM_cons]

/-- …and the faithful arm does short-circuit (live on main: `RETURN false AND 1/0 = 1` →
`false`). -/
theorem and_short_circuits {F : Type} (O : Ops (V F)) (ev : Ev (V F)) (c : Compound (V F))
    (env : Option (Nat → Option (V F))) (a : V F) (ha : O.boolOf a = some (some false)) (bad : E (V F)) :
    evalNode O ev c env (.and [.const a, bad]) = .ok (O.bool false) :=
  and_false_short_circuits O ev c env a ha [bad]

end Pr3437.Discharge
