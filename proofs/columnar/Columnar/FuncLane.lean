import Columnar.CaseLane

/-!
# `VectorEval::eval_function` (vector_expr.rs:557-642) and `eval_has_labels` (:644-706)

| here | there |
| --- | --- |
| `evalFunction` | `eval_function`: every argument column first (`?` in order), then the all-`Scalar` fold or the per-row calls, each `validate_args_type` then `func.func.call` (`Lanes.foldCall`) |
| `hasLabelsLane` | `eval_has_labels`: `[Variable over NodeIds, List of Constant(String)…]`, names resolved once, unregistered → all `false` |
| `hasLabelName` | `Runtime::node_has_label` (runtime.rs:1690-1703): `label_id` then `node_has_label_id` |
| `Ty`, `valueOfType`, `validateFixed` | `Type` (functions/mod.rs:531), `value_of_type` (value.rs:1315), `validate_args_type`'s `Fixed` arm (mod.rs:828-856) |
| `hasLabelsCall`, `hasLabelsFn` | the registered `hasLabels` (functions/entity.rs:62-127; proofs/functions_str_list `Entity.hasLabels`) |

**`evalFunction_sound`**, **`hasLabelsLane_sound`** (under `HL f`: `f` validates and computes
`hasLabels` on a node and a list of names), and **`hl_hasLabelsFn`**: the registry's
`hasLabels` satisfies `HL`.
-/

namespace Columnar.FuncLane
open FalkorCore (V)
open FalkorExpr.Eval (E Ops Compound Ev Fn evalNode evalList evalRev callFn)
open Columnar.CoreOps (RT coreOps)

variable {F : Type} [FloatModel F]

set_option linter.unusedSectionVars false

/-- The fold-or-loop body: `validate_args_type(&args)?; func.func.call(rt, &args)`. -/
def callOf (f : Fn (V F)) (args : List (V F)) : Except String (V F) := do
  let _ ← f.validate args
  f.call args

/-- `eval_function` (vector_expr.rs:557). -/
def evalFunction (ev : E (V F) → List Nat → Except String (EC F)) (f : Fn (V F)) (cs : List (E (V F)))
    (rows : List Nat) : Except String (EC F) := do
  let columns ← cs.mapM (fun ch => ev ch rows)
  foldCall (callOf f) (!f.nonDeterministic) columns rows.length

theorem mapM_map_ok {α β : Type} (g : α → Except String β) (k : α → β) : ∀ (l : List α),
    l.mapM g = .ok (l.map k) → ∀ x ∈ l, g x = .ok (k x)
  | [], _, x, hx => by simp at hx
  | a :: l, h, x, hx => by
    simp only [List.mapM_cons, List.map_cons] at h
    obtain ⟨y, hy, h⟩ := VecEval.bind_ok h
    obtain ⟨ys, hys, h⟩ := VecEval.bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq, List.cons.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    rcases List.mem_cons.1 hx with rfl | hx
    · exact hy
    · exact mapM_map_ok g k l hys x hx

theorem foldCall_wf (g : List (V F) → Except String (V F)) (det : Bool) (args : List (EC F)) (len : Nat)
    (col : EC F) (h : foldCall g det args len = .ok col) : col.WF len := by
  unfold foldCall at h
  split at h
  · cases hg : g (args.map (·.get 0)) with
    | error e => simp [hg, Except.map] at h
    | ok v => simp only [hg, Except.map, Except.ok.injEq] at h; subst h; trivial
  · cases hm : (List.range len).mapM (fun i => g (args.map (·.get i))) with
    | error e => simp [hm, Except.map] at h
    | ok vs =>
      simp only [hm, Except.map, Except.ok.injEq] at h; subst h
      have := mapM_length hm; simpa [EC.WF] using this

theorem foldCall_rows (g : List (V F) → Except String (V F)) (det : Bool) (args : List (EC F)) (len : Nat)
    (col : EC F) (h : foldCall g det args len = .ok col) :
    ∀ i < len, g (args.map (·.get i)) = .ok (col.get i) := by
  have := foldCall_agree g det args len
  rw [h] at this
  simp only [Except.map] at this
  intro i hi
  exact mapM_map_ok _ col.get (List.range len) this.symm i (List.mem_range.2 hi)

section
variable (rt : RT F) (c : Compound (V F)) (envOf : Nat → Option (Nat → Option (V F)))
  (ev : E (V F) → List Nat → Except String (EC F)) (rows : List Nat) (Rows : Nat → Prop)

abbrev ev0 (rt : RT F) : Ev (V F) := Ev.fromRuntime rt.params
abbrev KidOK := Columnar.AndOrLane.KidOK rt (ev0 rt) c envOf ev Rows

theorem evalList_cols (i : Nat) (hi : i < rows.length) : ∀ (cs : List (E (V F))) (cols : List (EC F)),
    (∀ ch ∈ cs, KidOK rt c envOf ev Rows ch) → (∀ r ∈ rows, Rows r) → cs.mapM (fun ch => ev ch rows) = .ok cols →
    evalList (coreOps rt) (ev0 rt) c (envOf rows[i]) cs = .ok (cols.map (·.get i))
  | [], cols, _, _, h => by simp [pure, Except.pure] at h; subst h; rfl
  | ch :: cs, cols, hk, hrows, h => by
    simp only [List.mapM_cons] at h
    obtain ⟨col, hcol, h⟩ := VecEval.bind_ok h
    obtain ⟨cols', hcols, h⟩ := VecEval.bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
    have h1 := hk ch (by simp) rows col hrows hcol i hi
    have h2 := evalList_cols i hi cs cols' (fun x hx => hk x (by simp [hx])) hrows hcols
    simp [evalList, h1, h2, bind, Except.bind]

/-- **`eval_function` = the per-row call, at every row** (children sound; the guard of
vector_expr.rs:253-258: no struct form, not a write). -/
theorem evalFunction_sound (f : Fn (V F)) (cs : List (E (V F))) (col : EC F)
    (hs : f.structFn = none) (hw : f.write = false)
    (hk : ∀ ch ∈ cs, KidOK rt c envOf ev Rows ch) (hrows : ∀ r ∈ rows, Rows r)
    (h : evalFunction ev f cs rows = .ok col) :
    col.WF rows.length ∧ ∀ i (hi : i < rows.length),
      evalNode (coreOps rt) (ev0 rt) c (envOf rows[i]) (.func f cs) = .ok (col.get i) := by
  unfold evalFunction at h
  obtain ⟨cols, hcols, h⟩ := VecEval.bind_ok h
  refine ⟨foldCall_wf _ _ _ _ col h, fun i hi => ?_⟩
  have hg := foldCall_rows _ _ _ _ col h i hi
  have hl := evalList_cols rt c envOf ev rows Rows i hi cs cols hk hrows hcols
  have : evalNode (coreOps rt) (ev0 rt) c (envOf rows[i]) (.func f cs) =
      (evalList (coreOps rt) (ev0 rt) c (envOf rows[i]) cs) >>= callFn (coreOps rt) f := by
    simp [evalNode, Ev.fromRuntime, Ev.rt, hs, bind, Except.bind]
  rw [this, hl]
  simpa [callOf, callFn, hw, bind, Except.bind] using hg
end

/-! ## `eval_has_labels` -/

/-- `Runtime::node_has_label` (runtime.rs:1690-1703). -/
def hasLabelName (rt : RT F) (id : Nat) (name : String) : Bool :=
  match rt.labelId name with
  | some l => rt.hasLabelId id l
  | none => false

def strConst : E (V F) → Option String
  | .const (.str s) => some s
  | _ => none

/-- `eval_has_labels` (vector_expr.rs:644): `col` is `batch.column`. -/
def hasLabelsLane (rt : RT F) (col : Nat → Column F) (cs : List (E (V F))) (rows : List Nat) : Option (EC F) :=
  match cs with
  | [.var id _, .list (e :: es)] =>
    match col id with
    | .nodeIds ids =>
      match (e :: es).mapM strConst with
      | none => none
      | some names =>
        match names.mapM rt.labelId with
        | none => some (.bools (rows.map fun _ => false) (Bits.none rows.length))
        | some ls => some (.bools (rows.map fun r => ls.all fun l => rt.hasLabelId (ids.getD r 0) l)
            (Bits.none rows.length))
    | _ => none
  | _ => none

/-- What the fast path needs of the function named `hasLabels`. -/
def HL (rt : RT F) (f : Fn (V F)) : Prop :=
  ∀ id (names : List String), f.validate [.node id, .list (names.map .str)] = .ok () ∧
    f.call [.node id, .list (names.map .str)] = .ok (.bool (names.all (hasLabelName rt id)))

theorem all_labels (rt : RT F) (x : Nat) : ∀ (names : List String) (ls : List Nat),
    names.mapM rt.labelId = some ls →
    (ls.all fun l => rt.hasLabelId x l) = (names.all fun nm => hasLabelName rt x nm)
  | [], ls, h => by simp [pure] at h; subst h; rfl
  | a :: as, ls, h => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨l, hl, ls', hls', rfl⟩ := h
    simp [hasLabelName, hl, all_labels rt x as ls' hls']

theorem all_labels_none (rt : RT F) (x : Nat) : ∀ (names : List String),
    names.mapM rt.labelId = none → (names.all fun nm => hasLabelName rt x nm) = false
  | [], h => by simp [pure] at h
  | a :: as, h => by
    simp only [List.mapM_cons, Option.bind_eq_bind] at h
    cases hl : rt.labelId a with
    | none => simp [hasLabelName, hl]
    | some l =>
      simp only [hl, Option.bind_some] at h
      cases hr : as.mapM rt.labelId with
      | none => simp [all_labels_none rt x as hr]
      | some r => simp [hr, pure] at h

theorem strConst_eval (rt : RT F) (c : Compound (V F)) (env : Option (Nat → Option (V F))) :
    ∀ (es : List (E (V F))) (names : List String), es.mapM strConst = some names →
      evalRev (coreOps rt) (ev0 rt) c env es = .ok (names.map .str)
  | [], names, h => by simp [pure] at h; subst h; rfl
  | e :: es, names, h => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨n, hn, ns, hns, rfl⟩ := h
    match e, hn with
    | .const (.str s), hn =>
      simp only [strConst, Option.some.injEq] at hn; subst hn
      simp [evalRev, strConst_eval rt c env es ns hns, evalNode, bind, Except.bind]

/-- **The `hasLabels` column kernel = the per-row call**, given `HL` (proved for the registry's
function in `hl_hasLabelsFn`). The per-row environment is the batch row `BatchRow::value_at`. -/
theorem hasLabelsLane_sound (rt : RT F) (c : Compound (V F)) (b : Batch F) (hwf : b.WF)
    (f : Fn (V F)) (cs : List (E (V F))) (rows : List Nat) (col : EC F)
    (hf : HL rt f) (hsf : f.structFn = none) (hwr : f.write = false)
    (hrows : ∀ r ∈ rows, r < b.len) (h : hasLabelsLane rt b.column cs rows = some col) :
    col.WF rows.length ∧ ∀ i (hi : i < rows.length),
      evalNode (coreOps rt) (ev0 rt) c (some (fun id => viewAt b rows[i] id)) (.func f cs) = .ok (col.get i) := by
  unfold hasLabelsLane at h
  split at h
  · rename_i id n e es
    split at h
    · rename_i ids hcol
      split at h
      · simp at h
      · rename_i names hnames
        have hlen : ids.length = b.len := by
          have := Batch.column_len hwf (i := id) (by rw [hcol]; rfl); rw [hcol] at this; exact this
        have hrow : ∀ i (hi : i < rows.length), viewAt b rows[i] id = some (.node (ids.getD rows[i] 0)) := by
          intro i hi
          have hr := hrows _ (List.getElem_mem hi)
          simp [viewAt, Batch.valueAt, hcol, Column.isUnbound, Column.get?,
            List.getElem?_eq_getElem (show rows[i] < ids.length by omega)]
        have hper : ∀ i (hi : i < rows.length),
            evalNode (coreOps rt) (ev0 rt) c (some (fun id => viewAt b rows[i] id)) (.func f [.var id n, .list (e :: es)]) =
              .ok (.bool (names.all (hasLabelName rt (ids.getD rows[i] 0)))) := by
          intro i hi
          have hv := (hf (ids.getD rows[i] 0) names).1
          have hc := (hf (ids.getD rows[i] 0) names).2
          have hl := strConst_eval rt c (some (fun id => viewAt b rows[i] id)) (e :: es) names hnames
          have h1 : evalNode (coreOps rt) (ev0 rt) c (some (fun id => viewAt b rows[i] id))
              (.func f [.var id n, .list (e :: es)]) =
              (evalList (coreOps rt) (ev0 rt) c (some (fun id => viewAt b rows[i] id)) [.var id n, .list (e :: es)])
                >>= callFn (coreOps rt) f := by
            simp [evalNode, Ev.fromRuntime, Ev.rt, hsf, bind, Except.bind]
          have hvar : evalNode (coreOps rt) (ev0 rt) c (some (fun id => viewAt b rows[i] id)) (.var id n) =
              .ok (.node (ids.getD rows[i] 0)) := by
            simp [evalNode, FalkorExpr.Eval.resolveVar, hrow i hi]
          have hlist : evalNode (coreOps rt) (ev0 rt) c (some (fun id => viewAt b rows[i] id)) (.list (e :: es)) =
              .ok (.list (names.map .str)) := by
            simp only [evalNode]; rw [hl]; rfl
          rw [h1]
          simp only [evalList, hvar, hlist, bind, Except.bind]
          generalize ids.getD rows[i] 0 = x at hv hc ⊢
          simp only [callFn, hv, hc, hwr, Bool.and_false]
          rfl
        split at h
        · rename_i hnone
          simp only [Option.some.injEq] at h; subst h
          refine ⟨by simp [EC.WF, Bits.none], fun i hi => ?_⟩
          rw [hper i hi, all_labels_none rt _ names hnone]
          simp [EC.get, Bits.isNull, Bits.none, List.getElem?_replicate, hi]
        · rename_i ls hls
          simp only [Option.some.injEq] at h; subst h
          refine ⟨by simp [EC.WF, Bits.none], fun i hi => ?_⟩
          rw [hper i hi, ← all_labels rt _ names ls hls]
          simp [EC.get, Bits.isNull, Bits.none, List.getElem?_replicate, hi]
    · simp at h
  · simp at h

end Columnar.FuncLane
