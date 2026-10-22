import Columnar.HasLabelsFn
import Columnar.LaneWF
import Columnar.Dispatch

/-!
# The `Dispatch.Lawful` instance: `VectorEval` over a batch is sound against `ExprEval`

The arms of `VecEval.eval` instantiated with the lanes modelled in this project, over one
batch `b` and the runtime record `rt` (`armsOf`), and the per-row reference *being*
proofs/expr_semantics' `evalNode` on the batch row `BatchRow::value_at` (`rowSem`; this is
what `eval_per_row` runs, vector_expr.rs:708-725). **`lawful`** proves every field of
`VecEval.Lawful` from the lane theorems: `evalVariable_agree`, `propertyValues` +
`classifyExact_exact`, `evalAndOr_sound`, `negate_agree` (with `notRow_eq`),
`arithmetic_agree`, `evalCase_sound`, `compareColumns_agree` (with `allEq_rowCmp`,
`allNeq_rowCmp`, `cmpResult_rowCmp`), `hasLabelsLane_sound`, `evalFunction_sound`.
**`vec_sound`**: whenever `VectorEval::eval` returns a column over valid rows, entry `i` is
`ExprEval::eval_node` of the tree on row `rows[i]`.

One hypothesis remains, about runtime code this project does not model: **`PropReads`** —
the bulk property fetch `materialize_{node,relationship}_property_values` returns, id for
id, what the per-row `get_{node,relationship}_attribute` returns (or `Null`). That is
proofs/runtime_core's `materialize_eq` premise (`hbg`, and `hName` for relationships), CONDITIONAL
there. `HL` (the registry's `hasLabels`) is discharged by `HasLabelsFn.hl_hasLabelsFn`.
-/

namespace Columnar.CoreBridge
open FalkorCore (V)
open FalkorExpr.Eval (E Ops Compound Ev Fn evalNode)
open Columnar.CoreOps (RT coreOps)
open Columnar.VecEval (Arms RowSem Lawful Sound SoundEv)

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

/-! ## `eval_property` (vector_expr.rs:333-354) and `property_values` (:356-389) -/

/-- `property_values`: a bound node / relationship column, gathered at `rows`, one bulk fetch. -/
def propertyValues (rt : RT F) (col : Nat → Column F) (attr : String) : E (V F) → List Nat → Option (List (V F))
  | .var id _, rows =>
    match col id with
    | .nodeIds ids => some (rt.matNode (rows.map fun r => ids.getD r 0) attr)
    | .relIds ids => some (rt.matRel (rows.map fun r => ids.getD r 0) attr)
    | _ => none
  | _, _ => none

/-- The `match col` of `eval_property` on `classify_exact_column`'s result. -/
def toEC (len : Nat) : Column F × Bits → EC F
  | (.ints d, n) => .ints d n
  | (.floats d, n) => .floats d n
  | (.values d, _) => .values d
  | (other, _) => .values ((List.range len).map fun i => (other.get? i).getD .null)

/-- `eval_property`. -/
def evalProperty (rt : RT F) (col : Nat → Column F) (attr : String) (ch : E (V F)) (rows : List Nat) :
    Option (EC F) :=
  (propertyValues rt col attr ch rows).map fun vs => toEC rows.length (classifyExact vs)

/-- The runtime contract the property fast path rests on (proofs/runtime_core `materialize_eq`). -/
structure PropReads (rt : RT F) : Prop where
  matNode : ∀ ids a, rt.matNode ids a = ids.map fun id => (rt.nodeAttr id a).getD .null
  matRel : ∀ ids a, rt.matRel ids a = ids.map fun id => (rt.relAttr id a).getD .null

theorem intsOf_length : ∀ {vs : List (V F)} {is : List Int}, intsOf true vs = some is → is.length = vs.length
  | [], is, h => by simp [intsOf] at h; subst h; rfl
  | v :: vs, is, h => by
    cases v <;> simp [intsOf] at h
    all_goals obtain ⟨is', h', rfl⟩ := h; simp [intsOf_length h']

theorem floatsOf_length : ∀ {vs : List (V F)} {fs : List F}, floatsOf true .pure vs = some fs → fs.length = vs.length
  | [], fs, h => by simp [floatsOf] at h; subst h; rfl
  | v :: vs, fs, h => by
    cases v <;> simp [floatsOf] at h
    all_goals obtain ⟨fs', h', rfl⟩ := h; simp [floatsOf_length h']

/-- **The property column reads back the fetched values** (`classify_exact_column` is lossless). -/
theorem toEC_exact' (vs : List (V F)) :
    (toEC vs.length (classifyExact vs)).WF vs.length ∧
      ∀ i (hi : i < vs.length), (toEC vs.length (classifyExact vs)).get i = vs[i] := by
  have hx := classifyExact_exact vs
  simp only [classifyExact, classifyNumeric] at hx ⊢
  cases hi : intsOf true vs with
  | some is =>
    simp only [hi] at hx ⊢
    refine ⟨⟨intsOf_length hi, by simp [Bits.ofValues]⟩, fun i hlt => ?_⟩
    have := hx i
    simp only [readN, Column.get?, List.getElem?_eq_getElem hlt] at this
    simp only [toEC, EC.get]
    split
    · rename_i hn; simp only [hn, ite_true, Option.some.injEq] at this; exact this
    · rename_i hn
      simp only [hn, Bool.false_eq_true, ite_false, Option.map_eq_some_iff] at this
      obtain ⟨x, hx', hv⟩ := this
      simp [hx', hv]
  | none =>
    simp only [hi] at hx ⊢
    cases hf : floatsOf true .pure vs with
    | some fs =>
      simp only [ne_eq, reduceCtorEq, not_false_eq_true, ite_true, hf] at hx ⊢
      refine ⟨⟨floatsOf_length hf, by simp [Bits.ofValues]⟩, fun i hlt => ?_⟩
      have := hx i
      simp only [readN, Column.get?, List.getElem?_eq_getElem hlt] at this
      simp only [toEC, EC.get]
      split
      · rename_i hn; simp only [hn, ite_true, Option.some.injEq] at this; exact this
      · rename_i hn
        simp only [hn, Bool.false_eq_true, ite_false, Option.map_eq_some_iff] at this
        obtain ⟨x, hx', hv⟩ := this
        simp [hx', hv]
    | none =>
      simp only [ne_eq, reduceCtorEq, not_false_eq_true, ite_true, hf]
      refine ⟨rfl, fun i hlt => ?_⟩
      simp [toEC, EC.get, List.getElem?_eq_getElem hlt]

theorem toEC_exact (vs : List (V F)) (len : Nat) (hl : vs.length = len) :
    (toEC len (classifyExact vs)).WF len ∧
      ∀ i (hi : i < len), (toEC len (classifyExact vs)).get i = vs[i]?.getD .null := by
  subst hl
  obtain ⟨hw, hg⟩ := toEC_exact' vs
  exact ⟨hw, fun i hi => by rw [hg i hi, List.getElem?_eq_getElem hi]; rfl⟩

/-! ## The arms and the per-row reference -/

section
variable (rt : RT F) (c : Compound (V F)) (b : Batch F)

/-- The batch row as `ExprEval`'s environment (`BatchRow::new(batch, row)`). -/
def env (r : Nat) : Option (Nat → Option (V F)) := some fun id => viewAt b r id

/-- `ExprEval::from_runtime(rt).eval_node(node, Some(&BatchRow::new(batch, row)), None)`. -/
def rowEval (e : E (V F)) (r : Nat) : Except String (V F) :=
  evalNode (coreOps rt) (Ev.fromRuntime rt.params) c (env b r) e

def rowSem : RowSem (V F) (EC F) where
  get := EC.get
  wf := fun col len => col.WF len
  rowEval := rowEval rt c b

/-- `eval_per_row` (vector_expr.rs:708-725). -/
def perRow (e : E (V F)) (rows : List Nat) : Except String (EC F) :=
  (rows.mapM (rowEval rt c b e)).map EC.values

/-- `VectorEval` over batch `b` (vector_expr.rs:211-283). -/
def armsOf : Arms (V F) (EC F) where
  scalar := .scalar
  params := rt.params
  evalVariable id rows := (evalVariable b id rows).getD (.values [])
  evalProperty := evalProperty rt b.column
  perRow := perRow rt c b
  andOr := AndOrLane.evalAndOr
  negate := negate
  arith op x y len := arithmetic rt.oa op x y len
  case ev s arms els rows := CaseLane.evalCase rt ev rows s arms els
  compare x y op len := compareColumns (CoreOps.ocC rt) x y op len
  hasLabels := FuncLane.hasLabelsLane rt b.column
  evalFunction := FuncLane.evalFunction
end

/-! ## `Lawful` -/

section
variable (rt : RT F) (c : Compound (V F)) (b : Batch F)

theorem soundEv_kid (Rows : Nat → Prop) (ev : E (V F) → List Nat → Except String (EC F)) (ch : E (V F))
    (h : SoundEv (rowSem rt c b) Rows ev ch) :
    AndOrLane.KidOK rt (Ev.fromRuntime rt.params) c (env b) ev Rows ch :=
  fun rs col hrs hcol i hi => (h rs col hrs hcol).2 i hi

theorem column_node {hwf : b.WF} {id : Nat} {ids : List Nat} (hcol : b.column id = .nodeIds ids)
    {r : Nat} (hr : r < b.len) : viewAt b r id = some (.node (ids.getD r 0)) := by
  have hlen : ids.length = b.len := by
    have := Batch.column_len hwf (i := id) (by rw [hcol]; rfl); rw [hcol] at this; exact this
  simp [viewAt, Batch.valueAt, hcol, Column.isUnbound, Column.get?,
    List.getElem?_eq_getElem (show r < ids.length by omega)]

theorem column_rel {hwf : b.WF} {id : Nat} {ids : List Nat} (hcol : b.column id = .relIds ids)
    {r : Nat} (hr : r < b.len) : viewAt b r id = some (.rel (ids.getD r 0)) := by
  have hlen : ids.length = b.len := by
    have := Batch.column_len hwf (i := id) (by rw [hcol]; rfl); rw [hcol] at this; exact this
  simp [viewAt, Batch.valueAt, hcol, Column.isUnbound, Column.get?,
    List.getElem?_eq_getElem (show r < ids.length by omega)]

/-- The binary operand pattern: both operands sound, the per-row value is `op` of theirs. -/
theorem binary_row (a b' : E (V F)) (rows : List Nat) (x y : EC F)
    (hx : Sound (rowSem rt c b) a rows x) (hy : Sound (rowSem rt c b) b' rows y) (i : Nat) (hi : i < rows.length) :
    rowEval rt c b a rows[i] = .ok (x.get i) ∧ rowEval rt c b b' rows[i] = .ok (y.get i) :=
  ⟨hx.2 i hi, hy.2 i hi⟩

/-- **The `Lawful` instance**: every arm of `VectorEval::eval` over batch `b` is sound against
`ExprEval` on the batch row, for variables in the batch's column space and rows of the batch,
given the runtime's `PropReads`. -/
theorem lawful (hwf : b.WF) (hp : PropReads rt) :
    Lawful (armsOf rt c b) (rowSem rt c b) (fun id => id < b.cols.length) (fun r => r ∈ b.active)
      (FuncLane.HL rt) where
  scalar v e rows h := ⟨trivial, fun i hi => h _ (List.getElem_mem hi)⟩
  const v r := rfl
  param name v r h := by
    simp only [armsOf] at h
    simp [rowSem, rowEval, evalNode, Ev.fromRuntime, Ev.rt, h, bind, Except.bind]
  var id n rows hs hr := by
    obtain ⟨col, hcol, hget⟩ := evalVariable_agree hwf id rows hr hs
    refine ⟨?_, fun i hi => ?_⟩
    · simp only [armsOf, rowSem, hcol, Option.getD_some]; exact evalVariable_wf b id rows col hcol
    · simp [rowSem, rowEval, evalNode, FalkorExpr.Eval.resolveVar, env, armsOf, hcol, ← hget i hi]
  prop attr ch rows col hs hr h := by
    simp only [armsOf, evalProperty, Option.map_eq_some_iff] at h
    obtain ⟨vs, hvs, rfl⟩ := h
    match ch, hvs with
    | .var id n, hvs =>
      simp only [propertyValues] at hvs
      have hrl : ∀ i (hi : i < rows.length), rows[i] < b.len :=
        fun i hi => Batch.active_lt hwf _ (hr _ (List.getElem_mem hi))
      split at hvs
      · rename_i ids hcol
        simp only [Option.some.injEq] at hvs; subst hvs
        rw [hp.matNode]
        obtain ⟨hw, hg⟩ := toEC_exact (List.map (fun id => (rt.nodeAttr id attr).getD V.null)
          (List.map (fun r => ids.getD r 0) rows)) rows.length (by simp)
        refine ⟨hw, fun i hi => ?_⟩
        simp only [rowSem]; rw [hg i hi]
        simp [rowEval, evalNode, FalkorExpr.Eval.resolveVar, env, column_node b (hwf := hwf) hcol (hrl i hi),
          CoreOps.asNode, Ev.fromRuntime, Ev.rt, bind, Except.bind, List.getElem?_eq_getElem hi]
      · rename_i ids hcol
        simp only [Option.some.injEq] at hvs; subst hvs
        rw [hp.matRel]
        obtain ⟨hw, hg⟩ := toEC_exact (List.map (fun id => (rt.relAttr id attr).getD V.null)
          (List.map (fun r => ids.getD r 0) rows)) rows.length (by simp)
        refine ⟨hw, fun i hi => ?_⟩
        simp only [rowSem]; rw [hg i hi]
        simp [rowEval, evalNode, FalkorExpr.Eval.resolveVar, env, column_rel b (hwf := hwf) hcol (hrl i hi),
          CoreOps.asNode, CoreOps.asRel, Ev.fromRuntime, Ev.rt, bind, Except.bind, List.getElem?_eq_getElem hi]
      · simp at hvs
  perRow e rows col h := by
    simp only [armsOf, perRow, Except.map] at h
    split at h
    · simp at h
    · rename_i vs hm
      simp only [Except.ok.injEq] at h; subst h
      obtain ⟨hl, hv⟩ := Dispatch.perRow_spec (rowEval rt c b e) rows vs hm
      refine ⟨by simp [rowSem, EC.WF, hl], fun i hi => ?_⟩
      obtain ⟨v, hvi, hev⟩ := hv i hi
      simp [rowSem, EC.get, hvi, hev]
  andOr ev cs rows sc col hk hr h := by
    have := AndOrLane.evalAndOr_sound rt (Ev.fromRuntime rt.params) c (env b) ev sc rows _ cs col
      (fun ch hc => soundEv_kid rt c b _ ev ch (hk ch hc)) hr h
    refine ⟨this.1, fun i hi => ?_⟩
    cases sc <;> simpa [rowSem, rowEval, evalNode] using this.2 i hi
  negate ch rows x col hx h := by
    have hag := negate_agree x rows.length
    simp only [armsOf] at h
    rw [h] at hag
    simp only [Except.map] at hag
    refine ⟨negate_wf x rows.length col h, fun i hi => ?_⟩
    have := FuncLane.mapM_map_ok _ col.get _ hag.symm i (List.mem_range.2 hi)
    have hxi : evalNode (coreOps rt) (Ev.fromRuntime rt.params) c (env b rows[i]) ch = .ok (x.get i) := hx.2 i hi
    simp only [rowSem, rowEval, evalNode, bind, Except.bind]
    rw [hxi]
    exact (CoreOps.notRow_eq rt (x.get i)).trans this
  add a b' rows x y col hx hy h := by
    have hag := arithmetic_agree rt.oa .add x y rows.length hx.1 hy.1
    simp only [armsOf] at h
    rw [h] at hag
    simp only [Except.map] at hag
    refine ⟨arithmetic_wf rt.oa .add x y rows.length hx.1 hy.1 col h, fun i hi => ?_⟩
    have := FuncLane.mapM_map_ok _ col.get _ hag.symm i (List.mem_range.2 hi)
    have hxi : evalNode (coreOps rt) (Ev.fromRuntime rt.params) c (env b rows[i]) a = .ok (x.get i) := hx.2 i hi
    have hyi : evalNode (coreOps rt) (Ev.fromRuntime rt.params) c (env b rows[i]) b' = .ok (y.get i) := hy.2 i hi
    simp only [rowSem, rowEval, evalNode, bind, Except.bind]
    rw [hxi, hyi]
    exact this
  arith op a b' rows x y col hx hy h := by
    have hag := arithmetic_agree rt.oa (VecEval.aopOf op) x y rows.length hx.1 hy.1
    simp only [armsOf] at h
    rw [h] at hag
    simp only [Except.map] at hag
    refine ⟨arithmetic_wf rt.oa _ x y rows.length hx.1 hy.1 col h, fun i hi => ?_⟩
    have := FuncLane.mapM_map_ok _ col.get _ hag.symm i (List.mem_range.2 hi)
    have hxi : evalNode (coreOps rt) (Ev.fromRuntime rt.params) c (env b rows[i]) a = .ok (x.get i) := hx.2 i hi
    have hyi : evalNode (coreOps rt) (Ev.fromRuntime rt.params) c (env b rows[i]) b' = .ok (y.get i) := hy.2 i hi
    simp only [rowSem, rowEval, evalNode, bind, Except.bind]
    rw [hxi, hyi]
    exact this
  case ev s arms els rows col hk hr h := by
    have := CaseLane.evalCase_sound rt (Ev.fromRuntime rt.params) c (env b) ev rows _ hr s arms els col
      (fun ch hc => soundEv_kid rt c b _ ev ch (hk ch hc)) h
    exact ⟨this.1, this.2⟩
  eq a b' rows x y hx hy := by
    refine ⟨compareColumns_wf _ x y .eq rows.length hx.1 hy.1, fun i hi => ?_⟩
    have hxi : evalNode (coreOps rt) (Ev.fromRuntime rt.params) c (env b rows[i]) a = .ok (x.get i) := hx.2 i hi
    have hyi : evalNode (coreOps rt) (Ev.fromRuntime rt.params) c (env b rows[i]) b' = .ok (y.get i) := hy.2 i hi
    simp only [rowSem, rowEval, evalNode, bind, Except.bind, armsOf]
    rw [hxi, hyi]
    simp only
    rw [CoreOps.allEq_rowCmp, compareColumns_agree _ x y .eq rows.length hx.1 hy.1 i hi]
  neq a b' rows x y hx hy := by
    refine ⟨compareColumns_wf _ x y .neq rows.length hx.1 hy.1, fun i hi => ?_⟩
    have hxi : evalNode (coreOps rt) (Ev.fromRuntime rt.params) c (env b rows[i]) a = .ok (x.get i) := hx.2 i hi
    have hyi : evalNode (coreOps rt) (Ev.fromRuntime rt.params) c (env b rows[i]) b' = .ok (y.get i) := hy.2 i hi
    simp only [rowSem, rowEval, evalNode, bind, Except.bind, armsOf]
    rw [hxi, hyi]
    simp only
    rw [CoreOps.allNeq_rowCmp, compareColumns_agree _ x y .neq rows.length hx.1 hy.1 i hi]
  cmp op a b' rows x y hx hy := by
    refine ⟨compareColumns_wf _ x y _ rows.length hx.1 hy.1, fun i hi => ?_⟩
    have hxi : evalNode (coreOps rt) (Ev.fromRuntime rt.params) c (env b rows[i]) a = .ok (x.get i) := hx.2 i hi
    have hyi : evalNode (coreOps rt) (Ev.fromRuntime rt.params) c (env b rows[i]) b' = .ok (y.get i) := hy.2 i hi
    simp only [rowSem, rowEval, evalNode, bind, Except.bind, armsOf]
    rw [hxi, hyi]
    simp only
    rw [CoreOps.cmpResult_rowCmp, compareColumns_agree _ x y _ rows.length hx.1 hy.1 i hi]
  hasLabels f cs rows col hg hl hs hr h := by
    simp only [VecEval.fnGuard, Bool.and_eq_true, Option.isNone_iff_eq_none, Bool.not_eq_true'] at hg
    have := FuncLane.hasLabelsLane_sound rt c b hwf f cs rows col hl hg.1.1 hg.1.2
      (fun r hr' => Batch.active_lt hwf r (hr r hr')) h
    exact ⟨this.1, this.2⟩
  func ev f cs rows col hg hk hr h := by
    simp only [VecEval.fnGuard, Bool.and_eq_true, Option.isNone_iff_eq_none, Bool.not_eq_true'] at hg
    have := FuncLane.evalFunction_sound rt c (env b) ev rows _ f cs col hg.1.1 hg.1.2
      (fun ch hc => soundEv_kid rt c b _ ev ch (hk ch hc)) hr h
    exact ⟨this.1, this.2⟩

/-- **`VectorEval::eval` is `ExprEval` row by row**: whenever it returns a column over rows of
the batch, for a tree whose variables are in the batch's column space and whose `hasLabels`
calls satisfy `HL`, entry `i` is `eval_node` of the tree on the batch row `rows[i]`. -/
theorem vec_sound (hwf : b.WF) (hp : PropReads rt) (e : E (V F)) (rows : List Nat) (col : EC F)
    (hsc : VecEval.InScope (fun id => id < b.cols.length) (FuncLane.HL rt) e)
    (hrows : ∀ r ∈ rows, r ∈ b.active) (h : VecEval.eval (armsOf rt c b) e rows = .ok col) :
    col.WF rows.length ∧ ∀ i (hi : i < rows.length),
      evalNode (coreOps rt) (Ev.fromRuntime rt.params) c (env b rows[i]) e = .ok (col.get i) :=
  VecEval.eval_sound (lawful rt c b hwf hp) e rows col hsc hrows h
end

end Columnar.CoreBridge
