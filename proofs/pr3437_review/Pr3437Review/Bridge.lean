import Columnar.Dispatch
import FalkorCore.Value

/-!
# PR #3437: the vector evaluator over a chunk batch, against columnar's `eval_sound`

`Col.Sound` (Chunked.lean) is what the chunk loop needs from `VectorEval`. Here it is
derived from proofs/columnar's `Dispatch.eval_sound` — *whenever `VectorEval::eval` returns a
column, entry `i` is the per-row evaluator on row `rows[i]`* — plus one premise about how
the chunk batch is built:

* **`RowAgree`** — the per-row evaluator on batch row `r` (a `BatchRow` over the batch
  `list_loop_columnar` builds, eval.rs:1563-1567) equals main's per-element evaluator on the
  owned row `env + {var: chunk[r]}`. It holds when (a) the batch's columns read back the
  chunk element and the broadcast outer values (`loopVar_get`, `broadcast_get` below;
  `set_column`'s `classify_stored_column` is lossless: columnar `Column.classifyStored_*`),
  (b) the body reads the row only through `ExprIR::Variable` children, every one of which
  is the loop variable, a broadcast outer variable or one the body binds itself
  (`Classifier.vectorPlan_broadcasts`), and (c) `ExprEval::eval_node`'s result depends on the
  row only through the variables the body reads (a frame property of the unmodelled
  `ExprEval`), with binder ids unique within the expression (`Binder::fresh_var`,
  binder.rs:2255). (c) is the undischarged part.

The PR's one change to `VectorEval::eval` — `!counts_comprehension(node)` on the function
arm (vector_expr.rs:255-257) — sends `size(<comprehension>)` to `eval_per_row`; `armsPR`
instantiates columnar's `Arms` with that guard and `lawful_pr` shows `eval_sound` survives it.
-/

namespace Pr3437.Bridge
open Columnar.Dispatch

variable {Val C : Type}

theorem mapM_of_pointwise {α β : Type} (f : α → Except String β) :
    ∀ (ys : List α) (g : Nat → β), (∀ i (h : i < ys.length), f ys[i] = .ok (g i)) →
      ys.mapM f = .ok ((List.range ys.length).map g)
  | [], g, _ => rfl
  | y :: ys, g, h => by
    have h0 := h 0 (by simp)
    have ih := mapM_of_pointwise f ys (fun i => g (i + 1)) (fun i hi => by
      have := h (i + 1) (by simp; omega)
      simpa only [List.getElem_cons_succ] using this)
    simp only [List.mapM_cons, List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map]
    simp only [List.getElem_cons_zero] at h0
    rw [h0, ih]; rfl

/-- **Bridge.** From columnar's `eval_sound`: a column `VectorEval::eval` returns over the
batch rows `rows` (each `< chunk.length`) holds main's per-element values of the chunk
elements at those rows. -/
theorem eval_rows_mapM (A : Arms Val C) (S : RowSem Val C) (L : Lawful A S) (n : Node Val)
    (pe : Val → Except String Val) (xs : List Val) (d : Val)
    (hrow : ∀ r, r < xs.length → S.rowEval n r = pe (xs.getD r d))
    (rows : List Nat) (hr : ∀ r ∈ rows, r < xs.length) (col : C) (h : eval A n rows = .ok col) :
    (rows.map (xs.getD · d)).mapM pe = .ok ((List.range rows.length).map (S.get col)) := by
  have hs := eval_sound A S L n rows col h
  have := mapM_of_pointwise pe (rows.map (xs.getD · d)) (S.get col) (fun i hi => by
    simp only [List.length_map] at hi
    simp only [List.getElem_map]
    rw [← hrow _ (hr _ (List.getElem_mem hi))]
    exact hs i hi)
  simpa using this

/-- The predicate over the whole chunk (`all = (0..len)`, eval.rs:1568): `Col.Sound.pred`. -/
theorem colSound_pred (A : Arms Val C) (S : RowSem Val C) (L : Lawful A S) (n : Node Val)
    (pe : Val → Except String Val) (xs : List Val) (d : Val)
    (hrow : ∀ r, r < xs.length → S.rowEval n r = pe (xs.getD r d))
    (col : C) (h : eval A n (List.range xs.length) = .ok col) :
    xs.mapM pe = .ok ((List.range xs.length).map (S.get col)) := by
  have := eval_rows_mapM A S L n pe xs d hrow (List.range xs.length) (by simp) col h
  have hx : (List.range xs.length).map (xs.getD · d) = xs := by
    apply List.ext_getElem
    · simp
    · intro i h1 h2; simp at h1; simp [List.getElem?_eq_getElem h1]
  rw [hx, List.length_range] at this; exact this

/-! ## The PR's `VectorEval::eval` guard (vector_expr.rs:253-269) -/

/-- `counts_comprehension(node)` (eval.rs:315-320); `isLC` recognises
`ExprIR::ListComprehension` (an `other` kind in columnar's `Kind`). -/
def countsComprehension (isLC : Kind Val → Bool) : Node Val → Bool
  | .mk (.func name _ _ _) [.mk k _] => isLC k && name == "size"
  | _ => false

/-- The PR's arms: a counted comprehension leaves the function arm (and so `hasLabels`,
whose name it does not carry) for `eval_per_row`. -/
def armsPR (A : Arms Val C) (isLC : Kind Val → Bool) : Arms Val C :=
  { A with
    evalFunction := fun sub n rows =>
      if countsComprehension isLC n then A.perRow n rows else A.evalFunction sub n rows }

theorem lawful_pr (A : Arms Val C) (S : RowSem Val C) (L : Lawful A S) (isLC : Kind Val → Bool) :
    Lawful (armsPR A isLC) S :=
  { L with
    func := fun ev n rows col hsub h => by
      simp only [armsPR] at h
      split at h
      · exact L.perRow n rows col h
      · exact L.func ev n rows col hsub h }

/-- **`eval_sound` holds for the PR's `VectorEval::eval`.** -/
theorem eval_sound_pr (A : Arms Val C) (S : RowSem Val C) (L : Lawful A S) (isLC : Kind Val → Bool)
    (n : Node Val) (rows : List Nat) (col : C) (h : eval (armsPR A isLC) n rows = .ok col) :
    Sound S n rows col :=
  eval_sound (armsPR A isLC) S (lawful_pr A S L isLC) n rows col h

/-- `size(<comprehension>)` is evaluated per row by the PR's `VectorEval`. -/
theorem counted_goes_per_row (A : Arms Val C) (isLC : Kind Val → Bool) (k : Kind Val)
    (cs : List (Node Val)) (rows : List Nat) (hk : isLC k = true) :
    eval (armsPR A isLC) (.mk (.func "size" false false false) [.mk k cs]) rows =
      A.perRow (.mk (.func "size" false false false) [.mk k cs]) rows := by
  unfold eval
  simp [armsPR, countsComprehension, hk, isDistinct]

/-! ## The chunk batch's columns (`loop_var_column`, `broadcast_column`) -/

open FalkorCore

/-- `Column` (batch.rs:321) at the shapes these two build. -/
inductive Column (F : Type) where
  | nodeIds (ids : List Nat) | relIds (ids : List Nat) | ints (xs : List Int)
  | floats (xs : List F) | values (vs : List (V F))

def Column.get {F : Type} : Column F → Nat → V F
  | .nodeIds ids, i => .node (ids.getD i 0)
  | .relIds ids, i => .rel (ids.getD i 0)
  | .ints xs, i => .int (xs.getD i 0)
  | .floats xs, i => match xs[i]? with | some f => .float f | none => .null
  | .values vs, i => vs.getD i .null

def allNodes {F : Type} : List (V F) → Option (List Nat)
  | [] => some []
  | .node id :: vs => (allNodes vs).map (id :: ·)
  | _ :: _ => none

/-- `loop_var_column` (eval.rs:408-417). -/
def loopVarColumn {F : Type} (chunk : List (V F)) : Column F :=
  match allNodes chunk with
  | some ids => .nodeIds ids
  | none => .values chunk

/-- `broadcast_column` (eval.rs:420-431). -/
def broadcastColumn {F : Type} (v : V F) (len : Nat) : Column F :=
  match v with
  | .node id => .nodeIds (List.replicate len id)
  | .rel id => .relIds (List.replicate len id)
  | .int i => .ints (List.replicate len i)
  | .float f => .floats (List.replicate len f)
  | other => .values (List.replicate len other)

theorem allNodes_get {F : Type} : ∀ (chunk : List (V F)) (ids : List Nat), allNodes chunk = some ids →
    ∀ i (h : i < chunk.length), V.node (ids.getD i 0) = chunk[i]
  | [], _, _, i, h => by simp at h
  | .node id :: vs, ids, h, i, hi => by
    simp only [allNodes] at h
    cases hv : allNodes vs with
    | none => simp [hv] at h
    | some ids' =>
      simp only [hv, Option.map_some, Option.some.injEq] at h; subst h
      cases i with
      | zero => rfl
      | succ i => simpa using allNodes_get vs ids' hv i (by simpa using hi)
  | .null :: _, _, h, _, _ | .bool _ :: _, _, h, _, _ | .int _ :: _, _, h, _, _
  | .float _ :: _, _, h, _, _ | .str _ :: _, _, h, _, _ | .list _ :: _, _, h, _, _
  | .map _ :: _, _, h, _, _ | .rel _ :: _, _, h, _, _ | .path _ :: _, _, h, _, _
  | .vec _ :: _, _, h, _, _ | .point .. :: _, _, h, _, _ | .datetime _ :: _, _, h, _, _
  | .date _ :: _, _, h, _, _ | .time _ :: _, _, h, _, _ | .duration _ :: _, _, h, _, _ => by
    simp [allNodes] at h

/-- The loop-variable column reads back the chunk element at every row. -/
theorem loopVar_get {F : Type} (chunk : List (V F)) (i : Nat) (h : i < chunk.length) :
    (loopVarColumn chunk).get i = chunk[i] := by
  unfold loopVarColumn
  cases ha : allNodes chunk with
  | some ids => exact allNodes_get chunk ids ha i h
  | none => simp [Column.get, h]

/-- A broadcast column reads back the outer value at every row (`-0.0`/NaN bits included:
the `f64` is copied, never re-derived). -/
theorem broadcast_get {F : Type} (v : V F) (len i : Nat) (h : i < len) :
    (broadcastColumn v len).get i = v := by
  cases v <;> simp [broadcastColumn, Column.get, h]

end Pr3437.Bridge
