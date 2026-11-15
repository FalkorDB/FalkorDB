import Pr3437Review.Concrete
import Pr3437Review.Runtime

/-!
# PR #3437: the headline theorems with `hS` discharged

The review's claims 1, 2 and 4 (`Drive.comprehension_eq_main`, `quantifier_eq_main`,
`size_eq_main`, `chunkStep_eq`, `Chunked.columnar_sound`) took the vector evaluator's
contract `Col.Sound` as a premise, which in turn rested on `hS`. Here they are instantiated
with the PR's real `VectorEval` over the real chunk batch (`prCol`) and main's per-element
evaluator (`prBody`: `ListLoop::new` over expr_semantics' `evalNode` with the engine's
operations and the runtime's parameters), with the plan computed by `vector_plan` (`planOf`):

* **`pr_comprehension_eq_main`** (`eval_list_comprehension`, eval.rs:1372),
* **`pr_quantifier_eq_main`** (`ExprEval::eval_quantifier`, :1310),
* **`pr_size_eq_main`** (`size(<comprehension>)`),
* **`pr_chunkStep_eq`** (`list_loop_chunk`, :1456),
* **`pr_columnar_sound`** (`list_loop_columnar`, :1550).

The `_of` forms take the runtime record abstractly and need `PropReads`; the headline forms
take the runtime built from its attribute state (`RuntimeReads.rtOf`, `Runtime.nodeStore`), for
which `PropReads` is a theorem (`RuntimeReads.propReads`, `Runtime.propReads_withStore`). Their one premise is `hreg`: the trees'
`hasLabels` calls are the registered function (the shape of the trees the parser builds).
-/

namespace Pr3437.Final
open FalkorCore (V)
open FalkorExpr.Eval (E Ev Compound Fn evalNode)
open Columnar (FloatModel)
open Columnar.CoreOps (RT coreOps)
open Columnar.CoreBridge (PropReads)
open Pr3437.PRArms (withStore NodeStore)
open Pr3437.ListLoop (Col Body)
open Pr3437.Classifier (vectorPlan)
open Pr3437.Concrete (colOf peOf Ok Reads)

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

section
variable (rt : RT F) (c : Compound (V F)) (st : NodeStore F) (isLC isBad : Nat → Bool) (dbg : Columnar.HasLabelsFn.Ty → String)
  (envAt : Nat → Option (V F)) (vid : Nat) (pred proj : E (V F))

/-- The bodies `vector_plan` looks at: `predicate.iter().chain(projection.iter())` after
`ListLoop::new` dropped the literal `true` and the identity projection (eval.rs:323-352). -/
def bodiesOf : List (E (V F)) :=
  (if !Discharge.isConstTrue pred then [pred] else []) ++ (if !Discharge.isLoopVar vid proj then [proj] else [])

/-- `ListLoop::vector_plan` (eval.rs:344). -/
def planOf : Option (List (Nat × V F)) :=
  vectorPlan vid ((bodiesOf vid pred proj).map (Discharge.toBT isBad)) envAt

/-- `ListLoop::new` with main's per-element evaluator. -/
def prBody : Body F :=
  Discharge.newBody (coreOps (withStore rt st)) (Ev.fromRuntime rt.params) c envAt vid pred proj

/-- With no plan the columnar path is never taken (`useCol` is false); any sound `Col`
stands in. -/
def errCol : Col F := ⟨fun _ => .error "no plan", fun _ _ => .error "no plan", fun _ _ => .error "no plan"⟩

/-- The columnar half the PR runs, given its plan. -/
noncomputable def prCol : Col F :=
  match planOf isBad envAt vid pred proj with
  | some outer => colOf rt c st isLC vid outer pred proj
  | none => errCol

theorem errCol_sound (pe pj : V F → Except String (V F)) : (errCol : Col F).Sound pe pj where
  pred _ _ h := by simp [errCol] at h
  proj _ _ _ h := by simp [errCol] at h
  projV _ _ _ h := by simp [errCol] at h

/-- The trees' `hasLabels` calls are the registered `hasLabels` (functions/entity.rs:62). -/
def Registry (e : E (V F)) : Prop :=
  ∀ f ∈ e.funcs, f.name = "hasLabels" → f = Columnar.HasLabelsFn.hasLabelsFn (withStore rt st) dbg

theorem plan_ok (hp : PropReads (withStore rt st)) (e : E (V F)) (hreg : Registry rt st dbg e)
    (outer : List (Nat × V F)) (hplan : planOf isBad envAt vid pred proj = some outer)
    (he : e = pred ∨ e = proj) : Ok rt st vid outer envAt e := by
  unfold planOf at hplan
  obtain ⟨hnd, hv⟩ := Concrete.plan_ids vid _ envAt outer hplan
  refine ⟨hnd, hv, Classifier.vectorPlan_vals vid _ envAt outer hplan, ?_,
    fun f hf hn => by rw [hreg f hf hn]; exact Columnar.HasLabelsFn.hl_hasLabelsFn _ dbg, hp⟩
  have hb := Classifier.vectorPlan_broadcasts vid _ envAt outer hplan
  rw [Discharge.varsAll_map] at hb
  intro id hid
  -- a present body is in the plan's bodies; an absent one reads at most the loop variable
  by_cases hm : e ∈ bodiesOf vid pred proj
  · exact hb id (Discharge.mem_varsL hm hid)
  · rcases he with rfl | rfl
    · cases hc : Discharge.isConstTrue e with
      | false => exact absurd (by simp [bodiesOf, hc]) hm
      | true =>
        match e, hc with
        | .const (.bool true), _ => simp [E.vars] at hid
    · cases hc : Discharge.isLoopVar vid e with
      | false => exact absurd (by simp [bodiesOf, hc]) hm
      | true =>
        match e, hc with
        | .var i n, h =>
          simp only [Discharge.isLoopVar, beq_iff_eq] at h
          simp only [E.vars, List.mem_singleton] at hid
          exact .inl (hid.trans h)

/-- **`Col.Sound` for the PR — the premise `hS` discharged.** -/
theorem prCol_sound (hp : PropReads (withStore rt st)) (hP : Registry rt st dbg pred)
    (hJ : Registry rt st dbg proj) :
    (prCol rt c st isLC isBad envAt vid pred proj).Sound (prBody rt c st envAt vid pred proj).pe
      (prBody rt c st envAt vid pred proj).pj := by
  unfold prCol
  split
  · rename_i outer hplan
    exact Concrete.colOf_sound rt c st isLC vid outer pred proj envAt
      (plan_ok rt st isBad dbg envAt vid pred proj hp pred hP outer hplan (.inl rfl))
      (plan_ok rt st isBad dbg envAt vid pred proj hp proj hJ outer hplan (.inr rfl))
  · exact errCol_sound _ _

theorem prBody_drops : (prBody rt c st envAt vid pred proj).Drops :=
  Discharge.newBody_drops _ _ _ envAt vid pred proj

variable (useCol : List (V F) → Bool) (hp : PropReads (withStore rt st))
  (hP : Registry rt st dbg pred) (hJ : Registry rt st dbg proj)
include hp hP hJ

/-- **Claims 1/2, comprehension (`eval_list_comprehension`)** — no `hS`. -/
theorem pr_comprehension_eq_main_of (src : ListLoop.Source F) :
    ListLoop.prListComp (prBody rt c st envAt vid pred proj) (prCol rt c st isLC isBad envAt vid pred proj)
        useCol false src =
      ListLoop.mainListComp (prBody rt c st envAt vid pred proj).pe (prBody rt c st envAt vid pred proj).pj src :=
  ListLoop.comprehension_eq_main _ _ useCol (prBody_drops rt c st envAt vid pred proj)
    (prCol_sound rt c st isLC isBad dbg envAt vid pred proj hp hP hJ) src

/-- **Claim 4 (`size(<comprehension>)`)** — no `hS`. -/
theorem pr_size_eq_main_of (src : ListLoop.Source F) :
    ListLoop.prListComp (prBody rt c st envAt vid pred proj) (prCol rt c st isLC isBad envAt vid pred proj)
        useCol true src =
      ListLoop.mainSize (ListLoop.mainListComp (prBody rt c st envAt vid pred proj).pe
        (prBody rt c st envAt vid pred proj).pj src) :=
  ListLoop.size_eq_main _ _ useCol (prBody_drops rt c st envAt vid pred proj)
    (prCol_sound rt c st isLC isBad dbg envAt vid pred proj hp hP hJ) src

/-- **Claims 1/2, quantifiers (`ExprEval::eval_quantifier`)** — no `hS`. -/
theorem pr_quantifier_eq_main_of (q : FalkorStrList.QType) (r : Except String (V F)) :
    ListLoop.prQuant (prBody rt c st envAt vid pred proj) (prCol rt c st isLC isBad envAt vid pred proj)
        useCol q r =
      ListLoop.mainQuant (prBody rt c st envAt vid pred proj).pe q r :=
  ListLoop.quantifier_eq_main _ _ useCol (prBody_drops rt c st envAt vid pred proj)
    (prCol_sound rt c st isLC isBad dbg envAt vid pred proj hp hP hJ) q r

/-- **`list_loop_chunk`** = main's loop on the chunk — no `hS`. -/
theorem pr_chunkStep_eq_of (s : ListLoop.Sink F) (chunk : List (V F)) :
    ListLoop.chunkStep (prBody rt c st envAt vid pred proj) (prCol rt c st isLC isBad envAt vid pred proj)
        useCol s chunk =
      ListLoop.ref (prBody rt c st envAt vid pred proj).pe (prBody rt c st envAt vid pred proj).pj s chunk :=
  ListLoop.chunkStep_eq _ _ useCol (prBody_drops rt c st envAt vid pred proj)
    (prCol_sound rt c st isLC isBad dbg envAt vid pred proj hp hP hJ) s chunk

/-- **`list_loop_columnar`**: a chunk it finishes gives main's loop's result — no `hS`. -/
theorem pr_columnar_sound_of (s : ListLoop.Sink F) (chunk : List (V F)) (s' : ListLoop.Sink F)
    (h : ListLoop.columnar (prBody rt c st envAt vid pred proj) (prCol rt c st isLC isBad envAt vid pred proj)
      s chunk = .ok s') :
    ListLoop.ref (prBody rt c st envAt vid pred proj).pe (prBody rt c st envAt vid pred proj).pj s chunk = .ok s' :=
  ListLoop.columnar_sound _ _ (prBody_drops rt c st envAt vid pred proj)
    (prCol_sound rt c st isLC isBad dbg envAt vid pred proj hp hP hJ) s chunk s' h
end

/-! ## Over the runtime's attribute state: no `PropReads` premise -/

section
variable (s : Runtime.PSt F) (rt0 : RT F) (c : Compound (V F)) (isLC isBad : Nat → Bool)
  (dbg : Columnar.HasLabelsFn.Ty → String) (envAt : Nat → Option (V F)) (vid : Nat) (pred proj : E (V F))
  (useCol : List (V F) → Bool)

/-- The runtime of the evaluation: its attribute reads are the state's. -/
abbrev rtS : RT F := Columnar.RuntimeReads.rtOf s.toSt rt0
abbrev stS : NodeStore F := Runtime.nodeStore s

theorem propReads_S : PropReads (withStore (rtS s rt0) (stS s)) := Runtime.propReads_withStore s rt0

variable (hP : Registry (rtS s rt0) (stS s) dbg pred) (hJ : Registry (rtS s rt0) (stS s) dbg proj)
include hP hJ

/-- **Claims 1/2, comprehension (`ExprEval::eval_list_comprehension`).** The PR's list
comprehension, columnar chunks included, is main's: same list, same order, or the same error. -/
theorem pr_comprehension_eq_main (src : ListLoop.Source F) :
    ListLoop.prListComp (prBody (rtS s rt0) c (stS s) envAt vid pred proj)
        (prCol (rtS s rt0) c (stS s) isLC isBad envAt vid pred proj) useCol false src =
      ListLoop.mainListComp (prBody (rtS s rt0) c (stS s) envAt vid pred proj).pe
        (prBody (rtS s rt0) c (stS s) envAt vid pred proj).pj src :=
  pr_comprehension_eq_main_of _ c _ isLC isBad dbg envAt vid pred proj useCol (propReads_S s rt0) hP hJ src

/-- **Claim 4 (`size(<comprehension>)`).** -/
theorem pr_size_eq_main (src : ListLoop.Source F) :
    ListLoop.prListComp (prBody (rtS s rt0) c (stS s) envAt vid pred proj)
        (prCol (rtS s rt0) c (stS s) isLC isBad envAt vid pred proj) useCol true src =
      ListLoop.mainSize (ListLoop.mainListComp (prBody (rtS s rt0) c (stS s) envAt vid pred proj).pe
        (prBody (rtS s rt0) c (stS s) envAt vid pred proj).pj src) :=
  pr_size_eq_main_of _ c _ isLC isBad dbg envAt vid pred proj useCol (propReads_S s rt0) hP hJ src

/-- **Claims 1/2, quantifiers (`ExprEval::eval_quantifier`).** -/
theorem pr_quantifier_eq_main (q : FalkorStrList.QType) (r : Except String (V F)) :
    ListLoop.prQuant (prBody (rtS s rt0) c (stS s) envAt vid pred proj)
        (prCol (rtS s rt0) c (stS s) isLC isBad envAt vid pred proj) useCol q r =
      ListLoop.mainQuant (prBody (rtS s rt0) c (stS s) envAt vid pred proj).pe q r :=
  pr_quantifier_eq_main_of _ c _ isLC isBad dbg envAt vid pred proj useCol (propReads_S s rt0) hP hJ q r

/-- **`ExprEval::list_loop_chunk`**: a chunk, columnar or per element, is main's loop on it. -/
theorem pr_chunkStep_eq (sk : ListLoop.Sink F) (chunk : List (V F)) :
    ListLoop.chunkStep (prBody (rtS s rt0) c (stS s) envAt vid pred proj)
        (prCol (rtS s rt0) c (stS s) isLC isBad envAt vid pred proj) useCol sk chunk =
      ListLoop.ref (prBody (rtS s rt0) c (stS s) envAt vid pred proj).pe
        (prBody (rtS s rt0) c (stS s) envAt vid pred proj).pj sk chunk :=
  pr_chunkStep_eq_of _ c _ isLC isBad dbg envAt vid pred proj useCol (propReads_S s rt0) hP hJ sk chunk

/-- **`ExprEval::list_loop_columnar`**: a chunk it finishes gives main's loop's result. -/
theorem pr_columnar_sound (sk : ListLoop.Sink F) (chunk : List (V F)) (sk' : ListLoop.Sink F)
    (h : ListLoop.columnar (prBody (rtS s rt0) c (stS s) envAt vid pred proj)
      (prCol (rtS s rt0) c (stS s) isLC isBad envAt vid pred proj) sk chunk = .ok sk') :
    ListLoop.ref (prBody (rtS s rt0) c (stS s) envAt vid pred proj).pe
      (prBody (rtS s rt0) c (stS s) envAt vid pred proj).pj sk chunk = .ok sk' :=
  pr_columnar_sound_of _ c _ isLC isBad dbg envAt vid pred proj (propReads_S s rt0) hP hJ sk chunk sk' h
end

end Pr3437.Final
