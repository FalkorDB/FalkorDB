/-
# Claim 2 premise and claim 5: what is re-used from the owning projects

* **"A read runtime runs no `borrow_mut` site."** Every `borrow_mut` on `runtime.g`
  (audit in `Protocol`) sits in a SET/CREATE/REMOVE/DELETE/MERGE operator, in `CommitOp`
  (refuses `!write`, `ops/commit.rs:50`) or in index DDL (refuses `!write`,
  `index_ddl.rs:38,95`); write procedures are refused on a read runtime
  (`eval.rs:959,979`, `ops/procedure_call.rs:67`). A read runtime is built only for a plan
  with no `Commit`/`CreateIndex`/`DropIndex` (`graph_core.rs:609-633`, `:687-698`,
  `record.rs:67-83`). `redis_layer`'s `RedisLayer.Ro` models the parser's `write` flag and
  the planner's `Commit` insertion (`cypher.rs:780-926`, `planner/mod.rs:2522-2524`,
  `:2853-2856`); here we prove from it that such a plan contains no updating operator
  (`read_plan_no_update`), discharging the premise.
* **Claim 5.** runtime_core's accessor models (`RuntimeCore.Accessors`) are functions of
  the world `W` the read sees (deleted snapshot, pending, *committed graph*) and never of
  how the graph was borrowed. `Cell.graph_same_result` shows `graph()` and `g.borrow()`
  yield the same graph, so every accessor theorem there transfers unchanged
  (`accessor_unchanged`, `materialize_unchanged`).
-/
import Pr3436Review.Cell
import RedisLayer.RoQuery
import RuntimeCore.Accessors

namespace Pr3436.Reuse
open RedisLayer.Ro

mutual
/-- Every updating operator in a plan comes from an updating clause. -/
theorem plan_update : ∀ (q : List Clause) (w : Bool),
    (plan q w).any (· == .update) = true → mutatesP q = true
  | [], w, h => by cases w <;> simp [plan] at h
  | c :: cs, w, h => by
    cases c with
    | update => simp [mutatesP, mutatesPC]
    | createIndex => simp [mutatesP, mutatesPC]
    | dropIndex => simp [mutatesP, mutatesPC]
    | ftDrop => simp [mutatesP, mutatesPC]
    | read =>
      simp only [plan, List.any_cons] at h
      simp only [mutatesP, mutatesPC, Bool.false_or]
      exact plan_update cs w (by simpa using h)
    | call p =>
      simp only [plan, List.any_cons] at h
      simp only [mutatesP, mutatesPC, Bool.false_or]
      exact plan_update cs w (by cases p <;> simpa using h)
    | with_ =>
      simp only [plan, List.any_append] at h
      simp only [mutatesP, mutatesPC, Bool.false_or]
      refine plan_update cs false ?_
      cases w <;> simpa using h
    | ret =>
      simp only [plan, List.any_append] at h
      simp only [mutatesP, mutatesPC, Bool.false_or]
      refine plan_update cs false ?_
      cases w <;> simpa using h
    | sub q =>
      simp only [plan, List.any_cons, List.any_append] at h
      simp only [mutatesP, mutatesPC, Bool.or_eq_true]
      have h' : (planSub q).any (· == .update) = true ∨ (plan cs w).any (· == .update) = true := by
        simpa using h
      rcases h' with h' | h'
      · exact Or.inl (plan_update q false (by simpa [planSub] using h'))
      · exact Or.inr (plan_update cs w h')
end

/-- **Premise discharged.** A plan that `execute_query` / `execute_profile` / `record`
classify as a read (no `Commit`, `CreateIndex`, `DropIndex`) contains no updating
operator, so a read runtime never reaches a `borrow_mut` of its graph. -/
theorem read_plan_no_update (q : List Clause) (h : isWritePlan (plan q false) = false) :
    (plan q false).any (· == .update) = false := by
  cases hu : (plan q false).any (· == .update) with
  | false => rfl
  | true =>
    have := plan_write q false (plan_update q false hu)
    rw [h] at this; cases this

/-! ### Claim 5: runtime_core's accessor theorems are about the same world -/

open Pr3436.Cell RuntimeCore.Accessors

/-- `Runtime::get_node_attribute` (runtime_core `Accessors.getNodeAttribute`) answers the
same through `graph()` as through `g.borrow()`. -/
theorem accessor_unchanged (h : Heap W) (rt : Rt) (hr : rt.ok) (r₁ r₂ : GraphRef) (h₁ h₂ : Heap W)
    (hpr : graph h rt = .ok (r₁, h₁)) (hmain : borrowMain h rt = .ok (r₂, h₂)) (id : Nat) (name : String) :
    getNodeAttribute (deref h₁ r₁) id name = getNodeAttribute (deref h₂ r₂) id name :=
  graph_same_result (fun w => getNodeAttribute w id name) h rt hr r₁ r₂ h₁ h₂ hpr hmain

/-- Same for the batched `materialize_node_property_values` and for every other accessor:
any `f : W → α`. -/
theorem materialize_unchanged (h : Heap W) (rt : Rt) (hr : rt.ok) (r₁ r₂ : GraphRef) (h₁ h₂ : Heap W)
    (hpr : graph h rt = .ok (r₁, h₁)) (hmain : borrowMain h rt = .ok (r₂, h₂))
    (bg : List Nat → Nat → List Val) (ids : List Nat) (name : String) :
    materialize (deref h₁ r₁) bg ids name = materialize (deref h₂ r₂) bg ids name :=
  graph_same_result (fun w => materialize w bg ids name) h rt hr r₁ r₂ h₁ h₂ hpr hmain

end Pr3436.Reuse
