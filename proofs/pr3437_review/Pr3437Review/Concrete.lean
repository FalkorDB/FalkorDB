import Pr3437Review.PRArms
import Pr3437Review.ChunkBatch
import Pr3437Review.Discharge
import Pr3437Review.Drive
import Columnar.HasLabelsFn

/-!
# PR #3437: `Col.Sound` for the PR's real `VectorEval` over the real chunk batch

`colOf` is what `list_loop_columnar` (eval.rs:1550-1652) calls: the PR's `VectorEval`
(`PRArms.armsPR`) over `ChunkBatch.chunkBatch` — `ve.eval(predicate, &batch, &all)`,
`ve.eval(projection, &batch, &passed)`, `ve.eval_values(projection, &batch, &passed)` — with the
columns read back as the loop reads them (`toChunkEC`). The per-element evaluator is
expr_semantics' `evalNode` over the engine's operations (`CoreOps.coreOps`) with the
runtime's parameters, on the owned row `env + {var: v}` (`Discharge.newBody`).

**`colOf_sound`** discharges the review's last premise `hS` (and so `Col.Sound`): for every
body `vector_plan` accepts, a column the PR's `VectorEval` returns holds main's
per-element values. It is `VecEval.eval_sound` / `evalValues_sound` with the `Lawful`
instance `PRArms.lawfulPR`, plus `RowAgree` on the real batch (`rowAgree`: the batch row
`BatchRow::value_at` agrees with the owned row on every variable an accepted body reads,
`ChunkBatch.chunkBatch_loop` / `chunkBatch_outer` and `Classifier.vectorPlan_broadcasts`, then
expr_semantics' frame property `evalNode_frame`).

What it rests on, stated as premises of the final theorems:
* **`PropReads`** — the runtime's bulk property fetch returns, id for id, the per-row
  attribute read (proofs/runtime_core `materialize_eq`'s `hbg` / `hName`, CONDITIONAL there:
  runtime.rs `materialize_{node,relationship}_property_values` vs `get_*_attribute`).
* `hreg` — the tree's `hasLabels` calls are the registry's function
  (`HasLabelsFn.hasLabelsFn`, functions/entity.rs:62); not a premise about Rust behaviour but
  the shape of the trees the parser builds (`get_functions().get("hasLabels")`).
-/

namespace Pr3437.Concrete
open FalkorCore (V)
open FalkorExpr.Eval (E Ev Compound Fn evalNode Agree evalNode_frame)
open Columnar (Batch Column EC FloatModel viewAt)
open Columnar.CoreOps (RT coreOps)
open Columnar.CoreBridge (rowSem PropReads env propertyValues)
open Pr3437.PRArms (armsPR withStore NodeStore lawfulPR)
open Pr3437.ChunkBatch (chunkBatch)
open Pr3437.ListLoop (Col Body)
open Pr3437.Classifier (vectorPlan)

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

/-- A column as `list_loop_columnar` reads it: a `Bools(mask, nulls)` cell by cell
(`mask.iter().enumerate()`, `nulls.is_null(i)`), any other lane through `ExprColumn::get`. -/
def toChunkEC (len : Nat) : EC F → ListLoop.EC F
  | .bools mask nulls => .bools ((List.range mask.length).map fun i => (mask.getD i false, nulls.isNull i))
  | col => .vals ((List.range len).map col.get)

theorem toChunkEC_vals (len : Nat) (col : EC F) (hw : col.WF len) :
    (toChunkEC len col).vals' = (List.range len).map col.get := by
  cases col with
  | bools mask nulls =>
    simp only [EC.WF] at hw
    simp only [toChunkEC, ListLoop.EC.vals', List.map_map, hw.1]
    apply List.map_congr_left
    intro i _
    simp [ListLoop.cell, EC.get, List.getD_eq_getElem?_getD]
  | _ => rfl

/-! ## The outer bindings of a plan -/

theorem nodup_eraseDups : ∀ l : List Nat, l.eraseDups.Nodup
  | [] => by simp
  | a :: as => by
    rw [List.eraseDups_cons, List.nodup_cons]
    refine ⟨fun h => ?_, nodup_eraseDups _⟩
    have := (List.mem_eraseDups.1 h)
    simp at this
termination_by l => l.length
decreasing_by simp_wf; have := List.length_filter_le (fun b => !b == a) as; omega

theorem mapM_pairs_fst {V' : Type} (valueAt : Nat → Option V') : ∀ (ids : List Nat) (out : List (Nat × V')),
    ids.mapM (fun id => (valueAt id).map (id, ·)) = some out → out.map Prod.fst = ids
  | [], out, h => by simp at h; subst h; rfl
  | x :: xs, out, h => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨y, hy, ys, hys, rfl⟩ := h
    cases hv : valueAt x with
    | none => simp [hv] at hy
    | some v =>
      simp only [hv, Option.map_some, Option.some.injEq] at hy; subst hy
      simp [mapM_pairs_fst valueAt xs ys hys]

/-- The plan's outer ids are distinct and are not the loop variable (`sort_unstable` +
`dedup`, `filter(id != var.id)`, eval.rs:356-362). -/
theorem plan_ids (vid : Nat) (bodies : List ClassifierPre.BT) (envAt : Nat → Option (V F))
    (outer : List (Nat × V F)) (h : vectorPlan vid bodies envAt = some outer) :
    (outer.map Prod.fst).Nodup ∧ vid ∉ outer.map Prod.fst := by
  unfold vectorPlan at h
  split at h
  · rename_i reads _
    rw [mapM_pairs_fst envAt _ outer h]
    refine ⟨(nodup_eraseDups reads).sublist List.filter_sublist, fun hm => ?_⟩
    simp at hm
  · simp at h

/-! ## `RowAgree` on the real batch -/

section
variable (vid : Nat) (outer : List (Nat × V F)) (chunk : List (V F)) (envAt : Nat → Option (V F))

/-- The variables an accepted body reads: the loop variable, or an outer binding. -/
def Reads (ids : List Nat) : Prop := ∀ id ∈ ids, id = vid ∨ ∃ v, envAt id = some v ∧ (id, v) ∈ outer

/-- **`RowAgree`**: the batch row and the owned row `env + {var: chunk[r]}` agree on every
variable an accepted body reads. -/
theorem rowAgree (hnd : (outer.map Prod.fst).Nodup) (hv : vid ∉ outer.map Prod.fst)
    (hvals : ∀ p ∈ outer, envAt p.1 = some p.2) (ids : List Nat) (hreads : Reads vid outer envAt ids)
    (r : Nat) (hr : r < chunk.length) :
    Agree (fun id => viewAt (chunkBatch vid outer chunk) r id) (Discharge.upd envAt vid chunk[r]) ids := by
  intro id hid
  rcases hreads id hid with rfl | ⟨v, henv, hm⟩
  · show viewAt (chunkBatch id outer chunk) r id = _
    rw [ChunkBatch.chunkBatch_loop id outer chunk hv r hr]; simp [Discharge.upd]
  · show viewAt (chunkBatch vid outer chunk) r id = _
    rw [ChunkBatch.chunkBatch_outer vid outer chunk hnd id v hm r hr]
    have hne : id ≠ vid := fun h => hv (h ▸ List.mem_map.2 ⟨(id, v), hm, rfl⟩)
    simp [Discharge.upd, hne, henv]

theorem reads_scope (ids : List Nat) (hreads : Reads vid outer envAt ids) :
    ∀ id ∈ ids, id < (chunkBatch vid outer chunk).cols.length := by
  intro id hid
  obtain ⟨h1, h2⟩ := ChunkBatch.chunkBatch_space vid outer chunk
  rcases hreads id hid with rfl | ⟨v, _, hm⟩
  · exact h1
  · exact h2 (id, v) hm
end

/-! ## The PR's `VectorEval` calls and their soundness -/

section
variable (rt : RT F) (c : Compound (V F)) (st : NodeStore F) (isLC : Nat → Bool) (vid : Nat)
  (outer : List (Nat × V F)) (pred proj : E (V F))

/-- What `list_loop_columnar` calls (eval.rs:1569, :1630, :1643) over the chunk batch. -/
noncomputable def colOf : Col F where
  colP chunk := (Columnar.VecEval.eval (armsPR rt c (chunkBatch vid outer chunk) st isLC) pred
    (List.range chunk.length)).map (toChunkEC chunk.length)
  colJ chunk rows := (Columnar.VecEval.eval (armsPR rt c (chunkBatch vid outer chunk) st isLC) proj rows).map
    (toChunkEC rows.length)
  colJv chunk rows := Columnar.VecEval.evalValues (armsPR rt c (chunkBatch vid outer chunk) st isLC)
    (fun col len => col.intoValues len) (propertyValues (withStore rt st) (chunkBatch vid outer chunk).column)
    (fun id rows => rows.map fun r => ((chunkBatch vid outer chunk).valueAt id r).getD .null) proj rows

/-- The per-element evaluator: `self.eval_node(body, Some(&row), ..)` on `env + {var: v}`. -/
def peOf (envAt : Nat → Option (V F)) (e : E (V F)) (v : V F) : Except String (V F) :=
  evalNode (coreOps (withStore rt st)) (Ev.fromRuntime rt.params) c (some (Discharge.upd envAt vid v)) e

/-- The hypotheses the soundness of a body's columns rests on. -/
structure Ok (envAt : Nat → Option (V F)) (e : E (V F)) : Prop where
  nodup : (outer.map Prod.fst).Nodup
  notLoop : vid ∉ outer.map Prod.fst
  vals : ∀ p ∈ outer, envAt p.1 = some p.2
  reads : Reads vid outer envAt e.vars
  hl : ∀ f ∈ e.funcs, f.name = "hasLabels" → Columnar.FuncLane.HL (withStore rt st) f
  props : PropReads (withStore rt st)

/-- **A body's columns are its per-element values**, on every set of chunk rows. -/
theorem body_sound (envAt : Nat → Option (V F)) (e : E (V F)) (ok : Ok rt st vid outer envAt e)
    (chunk : List (V F)) (rows : List Nat) (hrows : ∀ r ∈ rows, r < chunk.length) (col : EC F)
    (h : Columnar.VecEval.eval (armsPR rt c (chunkBatch vid outer chunk) st isLC) e rows = .ok col) :
    col.WF rows.length ∧ ∀ i (hi : i < rows.length),
      peOf rt c st vid envAt e (chunk.getD rows[i] .null) = .ok (col.get i) := by
  obtain ⟨hwf, _, hact⟩ := ChunkBatch.chunkBatch_wf vid outer chunk
  have L := lawfulPR rt c (chunkBatch vid outer chunk) st isLC hwf ok.props
  have hs := Columnar.VecEval.eval_sound L e rows col
    ⟨reads_scope vid outer chunk envAt e.vars ok.reads, ok.hl⟩
    (fun r hr => by rw [hact]; exact List.mem_range.2 (hrows r hr)) h
  refine ⟨hs.1, fun i hi => ?_⟩
  have h2 := hs.2 i hi
  have hr := hrows _ (List.getElem_mem hi)
  simp only [rowSem, Columnar.CoreBridge.rowEval, env] at h2
  rw [evalNode_frame _ _ _ _ _ e (rowAgree vid outer chunk envAt ok.nodup ok.notLoop ok.vals e.vars ok.reads _ hr)] at h2
  simp only [peOf, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hr, Option.getD_some]
  exact h2

/-- The bulk property values `eval_values` hands back are the per-element reads. -/
theorem propVals_sound (hp : PropReads (withStore rt st)) (chunk : List (V F))
    (attr : String) (ch : E (V F)) (rows : List Nat) (vs : List (V F)) (hrows : ∀ r ∈ rows, r < chunk.length)
    (h : propertyValues (withStore rt st) (chunkBatch vid outer chunk).column attr ch rows = some vs) :
    vs.length = rows.length ∧ ∀ i (hi : i < rows.length),
      Columnar.CoreBridge.rowEval (withStore rt st) c (chunkBatch vid outer chunk) (.prop attr ch) rows[i] =
        .ok (vs.getD i .null) := by
  obtain ⟨hwf, hlen, _⟩ := ChunkBatch.chunkBatch_wf vid outer chunk
  match ch, h with
  | .var id n, h =>
    simp only [propertyValues] at h
    split at h
    · rename_i ids hcol
      simp only [Option.some.injEq] at h; subst h
      rw [hp.matNode]
      refine ⟨by simp, fun i hi => ?_⟩
      have hr : rows[i] < (chunkBatch vid outer chunk).len := by rw [hlen]; exact hrows _ (List.getElem_mem hi)
      simp [Columnar.CoreBridge.rowEval, evalNode, FalkorExpr.Eval.resolveVar, env,
        Columnar.CoreBridge.column_node _ (hwf := hwf) hcol hr, Columnar.CoreOps.asNode, Ev.fromRuntime, Ev.rt,
        bind, Except.bind, List.getElem?_eq_getElem hi]
    · rename_i ids hcol
      simp only [Option.some.injEq] at h; subst h
      rw [hp.matRel]
      refine ⟨by simp, fun i hi => ?_⟩
      have hr : rows[i] < (chunkBatch vid outer chunk).len := by rw [hlen]; exact hrows _ (List.getElem_mem hi)
      simp [Columnar.CoreBridge.rowEval, evalNode, FalkorExpr.Eval.resolveVar, env,
        Columnar.CoreBridge.column_rel _ (hwf := hwf) hcol hr, Columnar.CoreOps.asNode, Columnar.CoreOps.asRel,
        Ev.fromRuntime, Ev.rt, bind, Except.bind, List.getElem?_eq_getElem hi]
    · simp at h

theorem viewAt_scope (b : Batch F) (id r : Nat) (h : id < b.cols.length) :
    viewAt b r id = some ((b.valueAt id r).getD .null) := by
  simp only [viewAt]; cases b.valueAt id r <;> simp [h]

/-- **`Col.Sound` for the PR's `VectorEval` over the chunk batch** — the review's premise
`hS`, discharged for every body whose plan `vector_plan` produced (`Ok`). -/
theorem colOf_sound (envAt : Nat → Option (V F)) (okP : Ok rt st vid outer envAt pred)
    (okJ : Ok rt st vid outer envAt proj) :
    (colOf rt c st isLC vid outer pred proj).Sound (peOf rt c st vid envAt pred) (peOf rt c st vid envAt proj) where
  pred xs ecol h := by
    simp only [colOf, Except.map] at h
    split at h
    · simp at h
    · rename_i col hcol
      simp only [Except.ok.injEq] at h; subst h
      obtain ⟨hw, hv⟩ := body_sound rt c st isLC vid outer envAt pred okP xs (List.range xs.length)
        (fun r hr => List.mem_range.1 hr) col hcol
      rw [toChunkEC_vals _ _ (by simpa using hw)]
      apply Bridge.mapM_of_pointwise
      intro i hi
      have := hv i (by simpa using hi)
      simpa [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi] using this
  proj xs rows ecol h hrows := by
    simp only [colOf, Except.map] at h
    split at h
    · simp at h
    · rename_i col hcol
      obtain ⟨_, hv⟩ := body_sound rt c st isLC vid outer envAt proj okJ xs rows hrows col hcol
      refine ⟨_, Bridge.mapM_of_pointwise _ (rows.map (xs.getD · .null)) col.get (fun i hi => ?_)⟩
      simp only [List.getElem_map]
      exact hv i (by simpa using hi)
  projV xs rows ys h hrows := by
    obtain ⟨hwf, _, hact⟩ := ChunkBatch.chunkBatch_wf vid outer xs
    have L := lawfulPR rt c (chunkBatch vid outer xs) st isLC hwf okJ.props
    have hsc : Columnar.VecEval.InScope (fun id => id < (chunkBatch vid outer xs).cols.length)
        (Columnar.FuncLane.HL (withStore rt st)) proj :=
      ⟨reads_scope vid outer xs envAt proj.vars okJ.reads, okJ.hl⟩
    have hrows' : ∀ r ∈ rows, r ∈ (chunkBatch vid outer xs).active := fun r hr => by
      rw [hact]; exact List.mem_range.2 (hrows r hr)
    have hs := Columnar.VecEval.evalValues_sound L (fun col len => col.intoValues len)
      (propertyValues (withStore rt st) (chunkBatch vid outer xs).column)
      (fun id rows => rows.map fun r => ((chunkBatch vid outer xs).valueAt id r).getD .null) .null
      (fun col rows e hs i hi => by
        rw [hs.2 i hi]; simp [List.getD_eq_getElem?_getD, EC.intoValues_get col rows.length hs.1 i hi]; rfl)
      (fun id n rows hid _ i hi => by
        simp [rowSem, Columnar.CoreBridge.rowEval, env, evalNode, FalkorExpr.Eval.resolveVar,
          viewAt_scope _ id _ hid, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi])
      (fun attr ch rows vs _ hr hp i hi =>
        (propVals_sound rt c st vid outer okJ.props xs attr ch rows vs
          (fun r h' => by have := hr r h'; rw [hact] at this; exact List.mem_range.1 this) hp).2 i hi)
      proj rows ys hsc hrows' h
    -- the length of what `eval_values` hands back
    have hlen : ys.length = rows.length := by
      simp only [colOf, Columnar.VecEval.evalValues] at h
      have fb : ∀ (ys : List (V F)),
          ((Columnar.VecEval.eval (armsPR rt c (chunkBatch vid outer xs) st isLC) proj rows).map
            fun col => col.intoValues rows.length) = .ok ys → ys.length = rows.length := by
        intro ys h'
        cases he : Columnar.VecEval.eval (armsPR rt c (chunkBatch vid outer xs) st isLC) proj rows with
        | error e => simp [he, Except.map] at h'
        | ok col =>
          simp only [he, Except.map, Except.ok.injEq] at h'; subst h'
          exact EC.intoValues_length col rows.length (Columnar.VecEval.eval_sound L proj rows col hsc hrows' he).1
      split at h
      · rename_i attr ch
        split at h
        · rename_i hp
          simp only [Except.ok.injEq] at h; subst h
          exact (propVals_sound rt c st vid outer okJ.props xs attr ch rows _ hrows hp).1
        · exact fb ys h
      · simp only [Except.ok.injEq] at h; subst h; simp
      · exact fb ys h
    have hpt := Bridge.mapM_of_pointwise (peOf rt c st vid envAt proj) (rows.map (xs.getD · .null))
      (fun i => ys.getD i .null) (fun i hi => by
        simp only [List.length_map] at hi
        have h2 := hs i hi
        have hr := hrows _ (List.getElem_mem hi)
        simp only [rowSem, Columnar.CoreBridge.rowEval, env] at h2
        rw [evalNode_frame _ _ _ _ _ proj
          (rowAgree vid outer xs envAt okJ.nodup okJ.notLoop okJ.vals proj.vars okJ.reads _ hr)] at h2
        simp only [List.getElem_map, peOf, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hr,
          Option.getD_some]
        exact h2)
    rw [hpt]
    congr 1
    apply List.ext_getElem
    · simp [hlen]
    · intro i h1 h2; simp at h1; simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2]

end

end Pr3437.Concrete
