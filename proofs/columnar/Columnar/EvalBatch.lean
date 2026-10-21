import Columnar.RuntimeReads
import Columnar.BatchMisc
import FalkorExprSemantics.EvalMore

/-!
# `ExprEval::eval_batch` (eval.rs:1150-1206): proofs/expr_semantics' `BatchLaws`, discharged

expr_semantics proves `evalBatch_eq_perRow` (the bulk `var.attr` path of `eval_batch` gives what
per-row evaluation gives) under `BatchLaws`: (1) `materialize_node_property_values` is
`get_node_attribute` per id, and (2) a node column holds the rows' nodes. Here both are proved
for the real pieces: the batch is columnar's `Batch` with the `BatchRow` views of the `active`
rows (`batchM`; `extract_node_ids` is `Batch.extractNodeIds`, batch.rs:1371), and the runtime
is `RuntimeReads.rtOf` of its attribute state (`matNode_eq`). **`laws`** is the instance;
**`evalBatch_sound`**: `eval_batch` = `eval_batch_per_row` with no premise.
-/

namespace Columnar.EvalBatch
open FalkorCore (V)
open FalkorExpr.Eval (BatchM BatchLaws colMatches evalBatch evalBatchPerRow JV E Ev Compound)
open Columnar (FloatModel Batch viewAt)
open Columnar.CoreOps (RT coreOps)
open Columnar.RuntimeReads (St rtOf matNode matNode_eq)

variable {F W : Type} [FloatModel F]

set_option linter.unusedSectionVars false

/-- What `eval_batch(tree, batch, active)` sees: the `BatchRow` view of every active row, and
`extract_node_ids(var)` read at the active rows (`active.iter().map(|&i| node_ids[i])`). -/
def batchM (b : Batch F) (active : List Nat) : BatchM (V F) where
  rows := active.map fun r => fun id => viewAt b r id
  nodeCol v := (b.extractNodeIds v).map fun ids => active.map fun r => ids.getD r 0

theorem colMatches_of (b : Batch F) (hwf : b.WF) (rt : RT F) (s : St F) (v : Nat) (ids : List Nat)
    (hcol : b.column v = .nodeIds ids) :
    ∀ active : List Nat, (∀ r ∈ active, r < b.len) →
      colMatches (coreOps (rtOf s rt)) v (active.map fun r => ids.getD r 0) (active.map fun r => fun id => viewAt b r id)
  | [], _ => trivial
  | r :: rs, h => by
    refine ⟨⟨.node (ids.getD r 0), ?_, rfl, rfl⟩, colMatches_of b hwf rt s v ids hcol rs (fun x hx => h x (by simp [hx]))⟩
    exact CoreBridge.column_node b (hwf := hwf) hcol (h r (by simp))

/-- **`BatchLaws` for the real batch and runtime.** -/
def laws (b : Batch F) (hwf : b.WF) (active : List Nat) (hact : ∀ r ∈ active, r < b.len) (s : St F) (rt : RT F)
    (toJV : V F → JV W) : BatchLaws (batchM b active) (coreOps (rtOf s rt)) toJV where
  materialize := matNode s
  materialize_eq ids a := matNode_eq s ids a
  col_rows v ids h := by
    simp only [batchM, Batch.extractNodeIds, Option.map_eq_some_iff] at h
    obtain ⟨ids', hids, rfl⟩ := h
    split at hids
    · rename_i v' hcol
      simp only [Option.some.injEq] at hids; subst hids
      exact colMatches_of b hwf rt s v _ hcol active hact
    · simp at hids

/-- **`eval_batch` = `eval_batch_per_row`** for every tree, batch and runtime state. -/
theorem evalBatch_sound (b : Batch F) (hwf : b.WF) (active : List Nat) (hact : ∀ r ∈ active, r < b.len)
    (s : St F) (rt : RT F) (toJV : V F → JV W) (c : Compound (V F)) (root : E (V F)) :
    evalBatch (coreOps (rtOf s rt)) toJV (Ev.fromRuntime rt.params) c (laws b hwf active hact s rt toJV) root =
      evalBatchPerRow (coreOps (rtOf s rt)) toJV (Ev.fromRuntime rt.params) c (batchM b active) root :=
  FalkorExpr.Eval.evalBatch_eq_perRow _ toJV _ rt.params c _ root

end Columnar.EvalBatch
